import torch
import numpy as np
from torch._C import dtype
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import torch.distributed as dist
import time
import tqdm
from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.moe import initialize_moe_config
from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer.decoding.diffusion_runner import ModelRunner
from dinfer import BlockIteratorFactory, KVCacheFactory, BlockDiffusionLLM
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM

import json
from multiprocessing import Process
from pathlib import Path
import pytest

from dinfer.model import BailingMoeV2ForCausalLM, LLaDA2MoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory, SamplingParams, DiffusionLLMServing
from dinfer import ThresholdParallelDecoder, BlockDiffusionLLMAttnmask, BlockDiffusionLLM
import difflib

#model_path = '/mnt/dllm/luxiaocheng/moe-mini-v2-e256-1009-fp8-ml4-grouprouter-20T-mdmcpt-block-diffusion-bl32-4k-noshift-100B'
model_path = '/mnt/infra/dulun.dl/models/dllm-mini/block-diffusion-sft-2k-v2-full-bd/LLaDA2-mini-preview-ep4-v0'
#model_path = '/mnt/infra/dulun.dl/models/dllm-mini/block-diffusion-sft-2k-v2-full-bd/LLaDA2-mini-preview-ep4-v0'
dataset_path = '/ossfs/workspace/dumped_prompts'
dataset='openai_humaneval'

FILE_PATH = Path(__file__).resolve()
sample_path = FILE_PATH.with_name(f"{FILE_PATH.stem[:-8]}_sample.json")

model = None
gpu_id = 0
device = torch.device(gpu_id)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=156895, eos_id=156892) 

def init_sglang_dist():
  torch.cuda.set_device(gpu_id)

  from sglang.srt import distributed
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '40399'
  distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
  distributed.initialize_model_parallel(1, 1, 1, backend='nccl')
  print("[Loading model]")

  from sglang.srt.layers.dp_attention import initialize_dp_attention
  model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
  server_args = ServerArgs(model_path=model_path, enable_dp_attention=True, trust_remote_code=True, tp_size=1, dp_size = 1, pp_size = 1)
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
  max_length = 2048
  model = ModelRunner(model, device, server_args=server_args, max_length=max_length)
  return model
model = init_sglang_dist()


def run_bd(use_kvcache):
  with open(sample_path, "r") as f:
    samples = json.load(f)

    ans = []
    for sample in samples:
      prompt = [sample['question']]
      prompt[0] = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt[0]+'<|role_end|><role>ASSISTANT</role>' 
      
      input_ids = tokenizer(prompt)['input_ids']
      input_ids = torch.tensor(input_ids).to(device)

      if not use_kvcache:
        dllm = BlockDiffusionLLMAttnmask(model, decoder, BlockIteratorFactory(use_block_diffusion=True), early_stop=True)
      else:
        dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True), 
          cache_factory=KVCacheFactory('prefix', is_bd_model=True, max_length=2048), early_stop=True, 
          maximum_unroll=4, expected_tpf=4, backend='sglang')

      out = dllm.generate(input_ids, gen_length=256, block_length=32)
      new_ans = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
      
      #assert(new_ans == sample['answer'])
      ans.append(new_ans)
    
    return ans


def run_bd_serving(use_kvcache):
  with open(sample_path, "r") as f:
    samples = json.load(f)

    sample_params = SamplingParams(threshold=0.9, cache='prefix', temperature=0., early_stop=True, cont_weight=0, prefix_look=0, 
            after_look=0, warmup_steps=0, enable_torch_compile=True, mask_id=156895, eos_id=156892, parallel_decoding='threshold', 
            use_credit=False, use_bd=True, max_length=2048)
    dllm_server = DiffusionLLMServing(model_path, model_type='llada2-mini', sample_params=sample_params, server_port=40567, num_gpus=1, dp_size=1, tpep_size=1, backend='sglang')

    ans = []
    for sample in samples:
      prompt = [sample['question']]
      prompt[0] = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt[0]+'<|role_end|><role>ASSISTANT</role>' 
      
      input_ids = tokenizer(prompt)['input_ids']
      input_ids = torch.tensor(input_ids).to(device)

      out = dllm_server.generate(input_ids, gen_length=256, block_length=32)
      new_ans = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
      
      #assert(new_ans == sample['answer'])
      ans.append(new_ans)
    
    return ans

def test_bd():
  ans_cache = run_bd(use_kvcache=True)
  ans_serving = run_bd_serving(use_kvcache=True)

  for i in range(len(ans_cache)):
    assert(ans_cache[i] == ans_serving[i])
  
