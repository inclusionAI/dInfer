import os
import logging
import multiprocessing as mp

import torch
import torch.distributed as dist
from vllm import distributed as vllm_dist
from transformers import AutoConfig
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config

from .parallel_strategy import ThresholdParallelDecoder, CreditThresholdParallelDecoder, HierarchyDecoder
from .utils import KVCacheFactory, BlockIteratorFactory
from .generate_uniform import IterSmoothWithVicinityCacheDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, BlockWiseDiffusionLLM, BlockDiffusionLLM
from ..model.modeling_fused_olmoe import FusedOlmoeForCausalLM
from ..model.modeling_llada_origin import LLaDAModelLM

logger = logging.getLogger(__name__)

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
            parallel_decoding='threshold', use_credit=False, use_bd=True, max_length=4096):
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
            cache_factory=cache_factory, early_stop=sample_params.early_stop, maximum_unroll=4, expected_tpf=4, backend=backend)

    return dllm

def generate(dllm, device, req_q, res_q):
    while True:
        data = req_q.get()
        if isinstance(data, str):
            assert data == 'stop'
            break
        else:
            input_ids, gen_len, block_len = data
        out = dllm.generate(input_ids, gen_length=gen_len, block_length=block_len)
        num_forwards = dllm.num_forwards
        if res_q is not None:
            res_q.put((out, num_forwards))


def sglang_llada2_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)
    logger.info(f'start v2 server. server port: {master_port}')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    from sglang.srt import distributed
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(world_size, world_size, 1, backend='nccl')
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.moe import initialize_moe_config
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from dinfer.decoding.diffusion_runner import ModelRunner
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    server_args = ServerArgs(model_path=model_path, enable_dp_attention=True, trust_remote_code=True, tp_size=world_size, dp_size = 1, pp_size = 1)
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
    model = ModelRunner(model, device, server_args=server_args, max_length=max_length, enable_compile=sample_params.enable_torch_compile)

    dllm = init_generator(model, sample_params, backend='sglang', max_length=max_length)
    generate(dllm, model.device, req_q=q, res_q=res_q)

def moe_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
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

def server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
    torch.cuda.set_device(gpu_id)
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

    def add_request(self, req):
        assert len(self.req_qs) != 0 and len(self.req_qs) == len(self.procs)
        for q in self.req_qs:
            q.put(req)

    def get_response(self):
        return self.res_q.get()

    def start_server(self, model_path, model_type, sample_params, server_port, gpus, backend):
        ctx = mp.get_context('spawn')
        assert len(self.procs) == 0, 'The server is already running.'
        procs = []
        req_qs = []
        for i, gpu in enumerate(gpus):
            if i == 0:
                res_q = ctx.Queue()
                self.res_q = res_q
            else:
                res_q = None
            q = ctx.Queue()
            req_qs.append(q)
            if backend=='sglang':
                assert model_type.startswith('llada2')
                p = ctx.Process(target=sglang_llada2_server_process, args=(model_path, sample_params, len(gpus), i, gpu, q, res_q, server_port))
            elif model_type=='llada-moe':
                p = ctx.Process(target=moe_server_process, args=(model_path, sample_params, len(gpus), i, gpu, q, res_q, server_port))
            else:
                p = ctx.Process(target=server_process, args=(model_path, sample_params, len(gpus), i, gpu, q, res_q, server_port))
            p.daemon = True
            procs.append(p)
            p.start()
        self.procs = procs
        self.req_qs = req_qs

    def is_running(self):
        return len(self.procs) != 0

    def stop_running(self):
        for q in self.req_qs:
            q.put('stop')
        for p in self.procs:
            p.join()

        self.procs = []
        self.req_qs = []
        self.req_q = None

class ServerHandle:
    def __init__(self):
        self.groups = []

    def add_requests(self, reqs):
        prompts, gen_length, block_length = reqs
        assert len(self.groups) == prompts.shape[0], 'We cannot only use DP to support batch size > 1.'
        for i, prompt in enumerate(prompts):
            self.groups[i].add_request((prompt.unsqueeze(0), gen_length, block_length))

    def get_responses(self):
        res = []
        for group in self.groups:
            res.append(group.get_response())
        return res

    def start_server(self, model_path, model_type, sample_params, server_port, num_gpus, dp_size, tpep_size, backend):
        gpu = 0
        assert num_gpus >= dp_size * tpep_size
        for i in range(dp_size):
            self.groups.append(ServerGroup())
            server_port += 1
            gpus = [gpu + i for i in range(tpep_size)]
            logger.info(f'start server group on GPU {gpus}, server port: {server_port}')
            self.groups[-1].start_server(model_path, model_type, sample_params, server_port, gpus, backend=backend)
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
    server_port : int
        The port for communication between the background process.
    num_gpus : int
        The number of GPUs used for parallel computation.
    """
    def __init__(self, model, model_type='llada2', sample_params=None, server_port=12345, num_gpus=None, dp_size=None, tpep_size=None, backend='sglang'):
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
        if not handle.is_running():
            handle.start_server(model, model_type, sample_params, server_port, num_gpus, dp_size, tpep_size, backend)
        self.num_forwards = 0

    def generate(self, prompts, gen_length=128, block_length=128):
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
        handle.add_requests((prompts, gen_length, block_length))
        rets = handle.get_responses()
        max_len = max([tensor.shape[1] for (tensor, _) in rets])
        res = torch.zeros(len(rets), max_len, dtype=rets[0][0].dtype)
        res[:] = self.sample_params.eos_id
        sum_num_forwards = 0
        for i, (tensor, num_forwards) in enumerate(rets):
            sum_num_forwards = max(sum_num_forwards, num_forwards)
            out_len = int(tensor.shape[1])
            res[i, :out_len] = tensor[0]
        self.num_forwards = sum_num_forwards
        return res

    def stop_serving(self):
        """ Stop model serving.
        """
        handle.stop_running()
