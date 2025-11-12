import torch
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Callable
import tqdm
import time
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_bool_env_var,
    is_hip,
)
import bisect
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    set_custom_all_reduce,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator, graph_capture
import logging
from .utils import KVCache
import os
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
)
from sglang.srt.custom_op import CustomOp
from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode
_is_hip = is_hip()

logger = logging.getLogger(__name__)
# 假设的上下文管理器，用于在捕获期间冻结垃圾回收
@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()

    

def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "default"
                ),
                dynamic=_is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE"),
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm
            


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class ModelRunner:
    def __init__(self, model: torch.nn.Module, device: str = "cuda", enable_cuda_graph: bool = True, supported_batch_sizes: Optional[list] = None, enable_compile:bool=True, server_args=None, max_length=2048, block_length=32):
        self.model = model.to(device)
        device = str(device)
        self.server_args = server_args
        if device.startswith("cuda:"):
            self.device = "cuda"
            self.gpu_id = int(device.split(":")[1])
        else:
            self.device = device
            self.gpu_id = torch.cuda.current_device()
        self.enable_compile = enable_compile
        self.enable_cuda_graph = enable_cuda_graph and (device != "cpu") # CPU 模式下禁用
        self.supported_batch_sizes = supported_batch_sizes or [1, ] # 默认支持的 batch sizes
        self.max_length = max_length
        self.block_length = block_length
        # 设置模型为评估模式
        x = torch.arange(block_length, dtype=torch.long, device=device).unsqueeze(0)
        
        self.model.eval()
        self.tp_group = get_tp_group()
        # self.cuda_graph_runners = {}
        set_custom_all_reduce(True)

        # _to_torch(self.model, reverse=True, num_tokens=1024)
        self.forward_normal(x, use_cache=True)
        # self.tp_group.ca_comm = backup_ca_comm
        self.init_device_graphs()
        

    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.graph_runner = CudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def forward_normal(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[torch.Tensor] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        backup_ca_comm = self.tp_group.ca_comm
        _to_torch(self.model, reverse=False, num_tokens=input_ids.numel())
        ret = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            past_key_values=past_key_values,
            replace_position=replace_position,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        self.tp_group.ca_comm = backup_ca_comm
        
        return ret
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[torch.Tensor] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if isinstance(past_key_values, KVCache):
            past_key_values = past_key_values._data
        
        # 简化判断：如果 input_ids 的 seq_len 为 block_length，则认为是 decode 阶段
        is_decode_phase = input_ids is not None and input_ids.shape[1] == self.block_length and use_cache and past_key_values is not None
        can_run_graph = bool(
            is_decode_phase
            and self.graph_runner
            and self.graph_runner.can_run(input_ids, position_ids, past_key_values)
        )
        if can_run_graph and self.enable_cuda_graph:
            # print('run cuda graph')
            ret = self.graph_runner.replay(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
            return ret

        # print('run normal')
        ret = self.forward_normal(input_ids, position_ids, inputs_embeds, pp_proxy_tensors, past_key_values, replace_position, use_cache, attention_mask)
        # if ret.past_key_values is None:
        # else:
        #     print('run normal', len(ret.past_key_values))
        # 默认路径：标准 PyTorch 执行
        return ret
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
        
        
        
class CudaGraphRunner:
    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.capture_bs, self.compile_bs = model_runner.supported_batch_sizes, model_runner.supported_batch_sizes
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}
        self.disable_padding = True
        self.enable_compile = model_runner.enable_compile

        self.max_bs = max(self.capture_bs)
        self.seq_len_fill_value = 0
        self.num_tokens_per_bs = self.model_runner.block_length
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.tp_size = get_attention_tp_size()
        with torch.device(self.device):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.position_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            num_layers = self.model_runner.model.config.num_hidden_layers
            num_kv_heads = self.model_runner.model.config.num_key_value_heads
            num_heads = self.model_runner.model.config.num_attention_heads
            head_dim = self.model_runner.model.config.hidden_size // num_heads
            self.past_key_values = torch.zeros((num_layers, 2, self.max_bs, num_kv_heads//self.tp_size, self.model_runner.max_length, head_dim), dtype=torch.bfloat16)
            
        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
            )
            
    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with self.device_module.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _create_device_graph(self):
        return torch.cuda.CUDAGraph()
    
    def capture(self) -> None:
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(
            self.model_runner.server_args.enable_cudagraph_gc
        ), graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            avail_mem = get_available_gpu_memory(
                self.model_runner.device,
                self.model_runner.gpu_id,
                empty_cache=False,
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            for i, bs in enumerate(capture_range):
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                    )

                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs and self.enable_compile,
                    num_tokens=bs * self.num_tokens_per_bs,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = output_buffers

                # Save gemlite cache after each capture
                save_gemlite_cache()


    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = self._create_device_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens].view(bs, self.num_tokens_per_bs)
        position_ids = self.position_ids[:num_tokens].view(bs, self.num_tokens_per_bs)
        past_key_values = self.past_key_values[:, :, :bs]

        # Run and capture
        def run_once():
            # print('run once', input_ids.shape, position_ids.shape, past_key_values.shape)
            logits_output = forward(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=None,
                pp_proxy_tensors=None,
                past_key_values=past_key_values,
                replace_position=(0, 0),
                use_cache=True,
                attention_mask=None,
            )
            return logits_output

        for _ in range(2):
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        return graph, out

    def can_run(self, input_ids, position_ids, past_key_values):
        cuda_graph_bs = input_ids.shape[0]
        # print('can run?', cuda_graph_bs, self.graphs.keys(), self.max_bs)
        is_bs_supported = (
            cuda_graph_bs in self.graphs  # 不 padding 模式 → 必须 exact match
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs  # padding 模式 → 可填充至最近更大的 graph
        )
        return is_bs_supported


    def replay_prepare(self, input_ids, position_ids, past_key_values):
        raw_bs = input_ids.shape[0]
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # 查找最接近的支持的 bs
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]

        # 拷贝真实数据到静态 buffer
        self.input_ids[:raw_num_token].copy_(input_ids.flatten())
        self.position_ids[:raw_num_token].copy_(position_ids.flatten())
        minimal_length = min(self.past_key_values.shape[4], past_key_values.shape[4])
        self.past_key_values[:, :, :bs, :, minimal_length:].fill_(0)
        self.past_key_values[:, :, :bs, :, :minimal_length].copy_(past_key_values[:, :, :, :, :minimal_length])
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs


    def replay(
        self, input_ids, position_ids, past_key_values,
    ):
        self.replay_prepare(input_ids, position_ids, past_key_values)

        # Replay
        self.graphs[self.bs].replay()

        output = self.output_buffers[self.bs]
        return output