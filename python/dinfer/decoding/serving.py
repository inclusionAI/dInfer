import os
import logging
import multiprocessing as mp
import traceback
from queue import Empty
import numpy as np
import torch
import torch.distributed as dist
from vllm import distributed as vllm_dist
from transformers import AutoConfig
_original_from_pretrained = AutoConfig.from_pretrained
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
import socket
from typing import Any, Optional, List, Optional, Tuple
import random
import time
from contextlib import closing


from .parallel_strategy import ThresholdParallelDecoder, CreditThresholdParallelDecoder, HierarchyDecoder
from .utils import KVCacheFactory, BlockIteratorFactory
from .generate_uniform import IterSmoothWithVicinityCacheDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, BlockWiseDiffusionLLM, BlockDiffusionLLM
from ..model.modeling_fused_olmoe import FusedOlmoeForCausalLM
from ..model.modeling_llada import LLaDAModelLM
import time
import json
import sys
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

def is_port_available(port: int) -> bool:
        """
        Check if a single port is available.

        :param port: The port number to check.
        :return: Whether the port is available.
        """
        sock_type = socket.SOCK_STREAM
        try:
            with closing(socket.socket(socket.AF_INET, sock_type)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False

def find_continuous_ports(
        num_ports: int,
        start_port: int = 30000,
        end_port: int = 65535,
        protocol: str = 'tcp',
        max_attempts: int = 300,
        retry_delay: float = 0.1
    ) -> Optional[Tuple[int, List[int]]]:
    """
    Find a sequence of consecutive free ports within a specified range.

    :param num_ports: Number of consecutive ports required.
    :param start_port: Starting port number (default: 1024).
    :param end_port: Ending port number (default: 65535).
    :param protocol: Protocol type ('tcp' only).
    :param max_attempts: Maximum number of retry attempts.
    :param retry_delay: Delay between retries (in seconds).
    :return: A tuple of (start_port, list_of_ports) if found, otherwise raise ValueError.
    """
    # Validate input parameters
    if num_ports <= 0:
        raise ValueError("Number of ports must be greater than 0")
    if start_port < 0 or end_port > 65535:
        raise ValueError("Port range must be between 0 and 65535")
    if start_port > end_port:
        raise ValueError("Start port cannot be greater than end port")
    if (end_port - start_port + 1) < num_ports:
        raise ValueError(f"Port range ({start_port}-{end_port}) is too small to accommodate {num_ports} consecutive ports")

    # calculate the max start port
    max_start = end_port - num_ports + 1

    for attempt in range(max_attempts):
        # Seeding with a combination of process ID and high-resolution timestamp
        pid = os.getpid()
        seed = pid + int(time.time() * 1_000_000)
        random.seed(seed)

        current = random.randint(start_port, max_start-1)
        while current <= max_start:
            # check current port
            if not is_port_available(current):
                current += 1
                continue

            # check following n+1 port
            all_available = True
            for offset in range(1, num_ports):
                port_to_check = current + offset
                if not is_port_available(port_to_check):
                    all_available = False
                    current = port_to_check + 1  
                    break

            # find available port and then lock these port
            if all_available:
                # check port again
                final_check = True
                for port in range(current, current + num_ports):
                    if not is_port_available(port):
                        final_check = False
                        current = port + 1
                        break

                if final_check:
                    ports = list(range(current, current + num_ports))
                    return current, ports

        # No port found in this attempt; sleep and retry.
        time.sleep(retry_delay)
    
    # No port found in all attempts; raise valueerror
    raise ValueError(f"No available ports found after {max_attempts} attempts. Range: {start_port}-{end_port}, Number of ports: {num_ports}")

def load_local_config(model_dir):
    # load config.json from model_path
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("auto_map", None)

    # dtype rename
    if "torch_dtype" in data and "dtype" not in data:
        td = data["torch_dtype"]
        if isinstance(td, str):
            m = {
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            data["dtype"] = m.get(td.lower(), td)
        else:
            data["dtype"] = td

    # return with PretrainedConfig format.
    cfg = PretrainedConfig.from_dict(data)
    cfg.name_or_path = model_dir
    return cfg

# When multiple workers run under DP/TP, concurrent AutoConfig calls can cause race conditions during config loading due to compilation overhead. 
# To avoid this, we override AutoConfig to directly load and convert config.json from the model path.
def _patched(cls, name_or_path, *args, **kwargs):
    subfolder = kwargs.get("subfolder")
    local_dir = os.path.join(name_or_path, subfolder) if subfolder else name_or_path
    assert os.path.isdir(local_dir), f"Model path does not exist: {local_dir}"
    cfg_file = os.path.join(local_dir, "config.json")
    assert os.path.isfile(cfg_file), f"Config.json file does not exist: {cfg_file}"
    
    try:
        return load_local_config(local_dir)
    except Exception as e:
        # Fall back to original loader on local config failure.
        logger.warning(f"Failed to load config from local path '{local_dir}': {e}")
        logger.warning("Falling back to Hugging Face's original loader...")
    
    return _original_from_pretrained(name_or_path, *args, **kwargs)

class SamplingParams:
    """ The parameters used for sampling a sequence.

    Parameters
    ----------
    threshold : float
        The threshold used for threshold-based parallel decoding algorithm.
    cache : str
        The kv-cache type. Valid values include 'prefix', 'dual' and ''.
    temperature : float
        The temperature used for decoding tokens.
    early_stop : bool
        Whether to stop generating tokens after encountering an EOS.
    cont_weight : float
        This is used by IterSmooth algorithm.
    prefix_look : int
        This is used by vicinity KV-cache refresh algorithm.
        This determines the number of tokens before the decoding block that should recompute key and value states in every diffusion iteration.
    after_look : int
        This is used by vicinity KV-cache refresh algorithm.
        This determines the number of tokens after the decoding block that should recompute key and value states in every diffusion iteration.
    warmup_steps : int
        This is used by vicinity KV-cache refresh algorithm.
        This determines the number of steps at the beginning that we need to refresh key and value states of the entire sequence.
    enable_torch_compile : bool
        Whether to use torch compile for the model code.
    mask_id : int
        The mask ID
    eos_id : int
        The EOS ID
    """
    def __init__(self, threshold=0.9, low_threshold=0.6, cache='dual', temperature=0., early_stop=True, cont_weight=0.3,
            prefix_look=16, after_look=16, warmup_steps=4, enable_torch_compile=True, mask_id=156895, eos_id=156892, 
            parallel_decoding='threshold', use_credit=False, use_bd=True, max_length=4096, ep_size=1, prefilling_limit=256,
            mini_batch_size=1, batch_size=1, use_naive_batching=False):
        self.threshold = threshold
        self.low_threshold = low_threshold
        self.cache = cache
        self.temperature = temperature
        self.early_stop = early_stop
        self.cont_weight = cont_weight
        self.prefix_look = prefix_look
        self.after_look = after_look
        self.warmup_steps = warmup_steps
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.enable_torch_compile = enable_torch_compile
        self.parallel_decoding = parallel_decoding
        self.use_credit = use_credit
        self.use_bd = use_bd
        self.max_length = max_length
        self.ep_size = ep_size
        self.prefilling_limit = prefilling_limit
        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_size
        self.use_naive_batching = use_naive_batching

def init_generator(model, sample_params, backend='vllm', max_length=4096):
    if sample_params.parallel_decoding == 'threshold':
        if sample_params.use_credit:
            decoder = CreditThresholdParallelDecoder(temperature=sample_params.temperature, threshold=sample_params.threshold,
                    mask_id=sample_params.mask_id, eos_id=sample_params.eos_id)
        else:
            decoder = ThresholdParallelDecoder(temperature=sample_params.temperature, threshold=sample_params.threshold,
                    mask_id=sample_params.mask_id, eos_id=sample_params.eos_id)
    else:
        decoder = HierarchyDecoder(temperature=sample_params.temperature, threshold=sample_params.threshold, low_threshold=sample_params.low_threshold,
                    mask_id=sample_params.mask_id, eos_id=sample_params.eos_id)
        


    if sample_params.cache == 'prefix' or sample_params.cache == 'dual':
        cache_factory = KVCacheFactory(sample_params.cache, is_bd_model=sample_params.use_bd, backend=backend, max_length=max_length)
    else:
        cache_factory = None

    if not sample_params.use_bd:
        if cache_factory is not None and sample_params.cont_weight > 0:
            dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=cache_factory,
                    early_stop=sample_params.early_stop, cont_weight=sample_params.cont_weight, prefix_look=sample_params.prefix_look,
                    after_look=sample_params.after_look, warmup_steps=sample_params.warmup_steps)
        elif cache_factory is not None:
            dllm = VicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=cache_factory,
                    early_stop=sample_params.early_stop, prefix_look=sample_params.prefix_look,
                    after_look=sample_params.after_look, warmup_steps=sample_params.warmup_steps)
        elif sample_params.cont_weight > 0:
            dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=None,
                    early_stop=sample_params.early_stop, cont_weight=sample_params.cont_weight)
        else:
            dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=None,
                    early_stop=sample_params.early_stop)
    else:
        dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True), 
            cache_factory=cache_factory, early_stop=sample_params.early_stop, maximum_unroll=1, expected_tpf=15, backend=backend, 
            prefilling_limit=sample_params.prefilling_limit, mini_batch_size=sample_params.mini_batch_size, use_naive_batching=sample_params.use_naive_batching)

    return dllm

def generate(dllm, device, req_q, res_q):
    while True:
        data = req_q.get()
        if isinstance(data, str):
            assert data == 'stop'
            break
        else:
            input_ids, gen_len, block_len, chat_uuid = data
        input_ids = input_ids.to(device)
        out = dllm.generate(input_ids, gen_length=gen_len, block_length=block_len, chat_uuid=chat_uuid)
        num_forwards = dllm.num_forwards
        if res_q is not None:
            res_q.put((out, num_forwards))
def sglang_llada2_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port, error_q=None):
    try:
        _sglang_llada2_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port)
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"[SERVER PROCESS {rank} ERROR]: {error_msg}")
        if error_q is not None:
            try:
                error_q.put((e, error_msg))
            except:
                logger.error(f"[ERROR_Q exception in {rank}].")

def _sglang_llada2_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)
    logger.info(f'start v2 server. server port: {master_port}')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    from sglang.srt import distributed
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(world_size, sample_params.ep_size, 1, backend='nccl')
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.moe import initialize_moe_config
    from ..model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from .diffusion_runner import ModelRunner
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    # override AutoConfig to directly load and convert config.json from the model path
    AutoConfig.from_pretrained = classmethod(_patched)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # ServerArgs call autoconfig in sglang/srt/utils/hf_transformers_utils.py. We use overrided autoconbfig here.
    server_args = ServerArgs(model_path=model_path, enable_dp_attention=True, trust_remote_code=True, tp_size=world_size, dp_size = 1, pp_size = 1,
                            port=master_port+1, dist_init_addr="127.0.0.1:{}".format(master_port+2))

    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
    except ImportError:
        pass
    else:
        set_global_server_args_for_scheduler(server_args)
    initialize_dp_attention(
        server_args=server_args,
        model_config=model_config,
    )
    initialize_moe_config(server_args)
    model = LLaDA2SGLangLM(config=model_config, expert_map_path='.').eval()
    torch.set_default_dtype(torch.bfloat16)
    model.load_weights(model_path, device=device)
    initialize_moe_config(server_args)
    
    
    model = model.to(device)
    max_length = sample_params.max_length
    prefill_lengths=[32*i for i in range(1, sample_params.prefilling_limit//32+1)]
    supported_batch_sizes = [2**i for i in range(int(np.log2(sample_params.mini_batch_size))+1)]
    model = ModelRunner(model, device, server_args=server_args, max_length=max_length, prefill_lengths=prefill_lengths, 
                        enable_compile=sample_params.enable_torch_compile, supported_batch_sizes=supported_batch_sizes, 
                        use_cross_block=sample_params.batch_size==1)

    dllm = init_generator(model, sample_params, backend='sglang', max_length=max_length)
    generate(dllm, model.device, req_q=q, res_q=res_q)

def moe_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port, error_q=None):
    try:
        _moe_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port)
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"[SERVER PROCESS {rank} ERROR]: {error_msg}")
        if error_q is not None:
            try:
                error_q.put((e, error_msg))
            except:
                logger.error(f"[ERROR_Q exception in {rank}].")

def _moe_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)
    logger.info(f'start MOE server. server port: {master_port}')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    vllm_dist.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    vllm_dist.initialize_model_parallel(world_size, backend='nccl')
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = FusedOlmoeForCausalLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16)
        if world_size > 1:
            model.tensor_parallel(world_size)
        if sample_params.enable_torch_compile:
            model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
        model = model.to(device)

        dllm = init_generator(model, sample_params)
        generate(dllm, model.device, req_q=q, res_q=res_q)

    # TODO(zhengda) we should destroy the distributed environment. However, the function hangs if TP/EP is turned on.
    #vllm_dist.destroy_distributed_environment()

def server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port, error_q=None):
    try:
        _server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port)
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"[SERVER PROCESS {rank} ERROR]: {error_msg}")
        if error_q is not None:
            try:
                error_q.put((e, error_msg))
            except:
                logger.error(f"[ERROR_Q exception in {rank}].")

def _server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
    device = torch.device(gpu_id)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    if world_size > 1:
        model.tensor_parallel(rank, world_size)
    if sample_params.enable_torch_compile:
        model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
    model = model.to(device)

    dllm = init_generator(model, sample_params)
    generate(dllm, model.device, req_q=q, res_q=res_q)

    dist.destroy_process_group()

class ServerGroup:
    def __init__(self):
        self.procs = []
        self.req_qs = []
        self.res_q = None
        self.error_qs = [] 

    def add_request(self, req):
        assert len(self.req_qs) != 0 and len(self.req_qs) == len(self.procs)
        for q in self.req_qs:
            q.put(req)

    def get_response(self, timeout=None):
        for i, eq in enumerate(self.error_qs):
            if not eq.empty():
                (e, error_msg) = eq.get_nowait()
                print(f"Worker {i} failed: {error_msg}")
                raise e
        try:
            return self.res_q.get(timeout=timeout)
        except Empty:
            print(f"[FATAL ERROR] Response timeout - possible hang or crash.")
            raise RuntimeError("Inference timed out or worker crashed.")

    # def _handle_error(self, msg):
    #     logger.error(f"[FATAL ERROR] {msg}")
    #     self.stop_running()
    #     raise RuntimeError(msg)

    def start_server(self, model_path, model_type, sample_params, server_port, gpus, backend):
        ctx = mp.get_context('spawn')
        assert len(self.procs) == 0, 'The server is already running.'
        procs = []
        req_qs = []
        error_qs = [] 
        for i, gpu in enumerate(gpus):
            if i == 0:
                res_q = ctx.Queue()
                self.res_q = res_q
            else:
                res_q = None
            q = ctx.Queue()
            req_qs.append(q)
            error_q = ctx.Queue()  
            error_qs.append(error_q)
            if backend=='sglang':
                assert model_type.startswith('llada2')
                p = ctx.Process(target=sglang_llada2_server_process, args=(model_path, sample_params, len(gpus), i, gpu, q, res_q, server_port, error_q))
            elif model_type=='llada-moe':
                p = ctx.Process(target=moe_server_process, args=(model_path, sample_params, len(gpus), i, gpu, q, res_q, server_port, error_q))
            else:
                p = ctx.Process(target=server_process, args=(model_path, sample_params, len(gpus), i, gpu, q, res_q, server_port, error_q))
            p.daemon = True
            procs.append(p)
            p.start()
        self.procs = procs
        self.req_qs = req_qs
        self.error_qs = error_qs  

    def is_running(self):
        return len(self.procs) != 0

    def stop_running(self):
        for p in self.procs:
            p.kill()
            p.join()
        self.procs = []
        self.req_qs = []
        self.error_qs = [] 
        self.res_q = None

class ServerHandle:
    def __init__(self):
        self.groups = []
        self.need_response = 0

    def add_requests(self, reqs, chat_uuid=None):
        prompts, gen_length, block_length = reqs
        self.need_response = 0
        # assert len(self.groups) == prompts.shape[0], 'We cannot only use DP to support batch size > 1.'
        if len(self.groups) == prompts.shape[0]:
            for i, prompt in enumerate(prompts):
                self.groups[i].add_request((prompt.unsqueeze(0), gen_length, block_length, chat_uuid))
            self.need_response = len(prompts)
        else:
            partial_data = torch.chunk(prompts, len(self.groups), dim=0)
            for i in range(len(partial_data)):
                self.groups[i].add_request((partial_data[i], gen_length, block_length, chat_uuid))
            self.need_response = len(partial_data)


    def get_responses(self, timeout=None):
        res = []
        for i in range(self.need_response):
            group = self.groups[i]
            res.append(group.get_response(timeout=timeout))
        return res

    def start_server(self, model_path, model_type, sample_params, server_port, num_gpus, dp_size, tpep_size, backend):
        gpu = 0
        assert num_gpus >= dp_size * tpep_size
        for i in range(dp_size):
            self.groups.append(ServerGroup())
            gpus = [gpu + i for i in range(tpep_size)]
            logger.info(f'start server group on GPU {gpus}, server port: {server_port[i]}')
            self.groups[-1].start_server(model_path, model_type, sample_params, server_port[i], gpus, backend=backend)
            gpu = gpus[-1] + 1
            

    def is_running(self):
        return len(self.groups) != 0

    def stop_running(self):
        for group in self.groups:
            group.stop_running()
        self.groups = []

handle = ServerHandle()

class DiffusionLLMServing:
    """ Serving dLLM inference.

    This is an experimental feature to enable serving in dInfer.
    This class creates multiple processes to enable dLLM inference in the background. A new request is sent to the background processes
    for model inference and the result is sent back to the main process.

    Parameters
    ----------
    model : str
        The model path
    is_moe : bool
        Whether this is a MOE model. This leads to using different model code and inference code.
    sample_params : SamplingParams
        The parameters used in sampling.
    server_port: int
        The port for communication between the background process.
    num_gpus : int
        The number of GPUs used for parallel computation.
    max_retries : int
        The maximum number of retry attempts per DP group when searching for free ports.
    start_port : int
        The starting port number (inclusive) for the port search range.
    end_port : int
        The ending port number (inclusive) for the port search range.
    """
    def __init__(self, model, model_type='llada2', sample_params=None, server_port=None, num_gpus=None, dp_size=None, tpep_size=None, backend='sglang', timeout = None,
                max_retries=3, start_port=30000, end_port=60000
                ):
        if sample_params is None:
            sample_params = SamplingParams()
        self.sample_params = sample_params
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        if dp_size is None:
            dp_size = 1
        if tpep_size is None:
            tpep_size = num_gpus // dp_size
        assert dp_size * tpep_size <= num_gpus
        if server_port == None:
            # find a free port for each group
            server_port = self.port_selection(dp_size, start_port, end_port, max_retries=max_retries)
        else:
            # Keep compatibility with the existing implementation: assign ports starting from the current port + 1.
            server_port = [server_port+i for i in range(dp_size)]
        logger.info(f"find server port:{server_port}")

        if not handle.is_running():
            handle.start_server(model, model_type, sample_params, server_port, num_gpus, dp_size, tpep_size, backend)
        self.num_forwards = 0
        self.timeout = timeout

    def generate(self, prompts, gen_length=128, block_length=128, chat_uuid=None):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompts: Torch.Tensor
            A tensor of shape (b, L) that contains the input prompts.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: A tensor of shape (b, L') that contains the prompt tokens and the generated tokens.
            The generation results of different lengths are padded with EOS.
        '''
        prompts = prompts.cpu()
        handle.add_requests((prompts, gen_length, block_length), chat_uuid=chat_uuid)
        rets = handle.get_responses(timeout=self.timeout)
        max_len = max([tensor.shape[1] for (tensor, _) in rets])
        total_batch_size = sum([tensor.shape[0] for (tensor, _) in rets])
        res = torch.zeros(total_batch_size, max_len, dtype=rets[0][0].dtype)
        res[:] = self.sample_params.eos_id
        sum_num_forwards = 0
        p = 0
        for i, (tensor, num_forwards) in enumerate(rets):
            sum_num_forwards = max(sum_num_forwards, num_forwards)
            out_len = int(tensor.shape[1])
            res[p:p+len(tensor), :out_len] = tensor
        self.num_forwards = sum_num_forwards
        return res

    def stop_serving(self):
        """ Stop model serving.
        """
        handle.stop_running()
        return None

    def port_selection(self, dp_size, start_port, end_port, max_retries=3):
        # select continuous port for each dp group ranging from start_port to end_port
        ports=[] 
        for dp in range(dp_size):
            for retry in range(max_retries):
                try:
                    # start with 7 ports: 1 nccl port; 6 continus port for server_args: 1 server_args default port + 5 extra ports if dp_attention is available
                    # for simplity, we select 7 continuous_ports for each group
                    port, dp_ports_list = find_continuous_ports(num_ports=7, start_port=start_port, end_port=end_port) 
                    ports.append(port) # record the first port for this group
                    logger.info("Allocated port range [%d-%d] (%d ports) for DP group %d", dp_ports_list[0], dp_ports_list[-1], len(dp_ports_list), dp)
                    break
                except ValueError as e:
                    if retry < max_retries - 1:
                        logger.warning("DP rank %d, attempt %d failed: %s. Retrying...",dp, retry + 1, e)
                        time.sleep(1)
                    else:
                        raise RuntimeError(
                            f"Failed to allocate ports for DP rank {dp} after {max_retries} attempts"
                        ) from e
        return ports
    

