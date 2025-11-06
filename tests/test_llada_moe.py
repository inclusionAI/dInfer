import os
import logging
from multiprocessing import Process
import random
import pytest
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel, AutoConfig
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config

from dinfer.model import FusedOlmoeForCausalLM, LLaDAModelLM
from dinfer import BlockWiseDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM, BlockWiseDiffusionLLMWithSP
from dinfer import ThresholdParallelDecoder, HierarchyDecoder
from dinfer import DiffusionLLMServing, SamplingParams

from dinfer.model.modeling_llada_fastdllm import LLaDAModelLM as LLaDAModelLM_fastdllm
from dinfer.decoding.generate_fastdllm import generate, generate_with_prefix_cache, generate_with_dual_cache
from dinfer.decoding.generate_dist import generate as generate_sp
from dinfer.decoding.generate_uniform import BaseDiffusionIteration
from dinfer.decoding.generate_hierarchy import generate_hierarchy
from dinfer.decoding.utils import TokenArray, DistAlignedTokenArray, BlockIterator, BlockIteratorFactory, KVCacheFactory, gather_sequence_block, BlockLoc
from dinfer.decoding.utils import DiffusionKVCacheManager
from dinfer.decoding.generate_merge import generate_merge
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from test_generate import IterSmoothDiffusionLLM as IterSmoothDiffusionLLM_test
from test_generate import IterSmoothWithVicinityCacheDiffusionLLM as IterSmoothWithVicinityCacheDiffusionLLM_test

moe_model_path = '/mnt/dllm/fengling/moe/workdir/7bA1b_anneal_15t_0827_500B_further_8k_enneal_train_4k_ep3_v7_1e-5/step45567_converted_hf_fusemoe'
# moe_model_path = '/data/dulun/models/llada-moe-sft/llada-moe-sft-model/7bA1b_anneal_19t_500B_further_8k_anneal_train_4k_ep3_v8p5/step45567_converted_hf_fusemoe/'

def get_prompts(tokenizer, mask_id, device, num=1):
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids1 = torch.tensor(tokenizer(prompt)['input_ids']).to(device).unsqueeze(0)
    len1 = input_ids1.shape[1]

    if num == 2:
        prompt = "Lily can run 12 kilometers per hour for 4 hours. How many kilometers can she run in 4 hours? "
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids2 = torch.tensor(tokenizer(prompt)['input_ids']).to(device).unsqueeze(0)
        len2 = input_ids2.shape[1]
        ret = torch.zeros(2, max(len1, len2), dtype=input_ids1.dtype)
        ret[0, 0:len1] = input_ids1
        ret[1, 0:len2] = input_ids2
    else:
        ret = input_ids1

    return ret

gpu_id = 1
device = torch.device(gpu_id)
decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892, use_float64=True)
h_decoder = HierarchyDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892, low_threshold=0.4)
tokenizer = AutoTokenizer.from_pretrained(moe_model_path, trust_remote_code=True)
input_ids = get_prompts(tokenizer, mask_id=156895, device=device)
model = None

@pytest.fixture(scope="session", autouse=True)
def init_vllm_dist(worker_id):
  torch.cuda.set_device(gpu_id)
  from vllm import distributed
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '37977'
  distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
  distributed.initialize_model_parallel(1, backend='nccl')
  print("[Loading model]")
  # setup EP
  parallel_config = ParallelConfig(enable_expert_parallel = True)
  with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
      model_config = AutoConfig.from_pretrained(moe_model_path, trust_remote_code=True)
      global model
      model = FusedOlmoeForCausalLM(config=model_config).eval()
      model.load_weights(moe_model_path, torch_dtype=torch.bfloat16)
      model = model.to(device)
  yield

  distributed.destroy_model_parallel()
  distributed.destroy_distributed_environment()


def test_llada_moe_hierarchy():
  # Test block-wise hierarchical diffusion MOE-LLM without KV-cache
  print('Test block-wise hierarchical diffusion MOE-LLM without KV-cache')
  dllm = BlockWiseDiffusionLLM(model, h_decoder, BlockIteratorFactory(), early_stop=True)
  res = dllm.generate(input_ids, gen_length=128, block_length=32)
  res1, nfe = generate_hierarchy(model, input_ids, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892,decoding='hierarchy_fast_v2',
                                        low_threshold=0.4, remask_threshold=0.4)
  res1 = res1[res1 != 156892]
  assert res.shape[1] == len(res1)
  res1 = res1.to(res.device)
  assert torch.all(res == res1)


def test_llada_moe_blockwise():
  # Test generation without cache.
  print('Test block-wise diffusion MOE-LLM without KV-cache')
  dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
  res = dllm.generate(input_ids, gen_length=128, block_length=32)
  res1, nfe = generate(model, input_ids, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892)
  res2, nfe = generate_merge(model, input_ids, None, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892, parallel_decoding='threshold', early_stop=False,)
  res1 = res1[res1 != 156892]
  res2 = res2[res2 != 156892]
  assert res.shape[1] == len(res1)
  assert res.shape[1] == len(res2)
  res1 = res1.to(res.device)
  res2 = res2.to(res.device)
  assert torch.all(res == res1)
  assert torch.all(res == res2)


def test_llada_moe_batching():
  # Test generation without cache with batch size == 2.
  dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
  print('Test block-wise diffusion MOE-LLM without KV-cache and batch size == 2')
  input_ids2 = get_prompts(tokenizer, mask_id=156895, device=device, num=2)
  res2 = dllm.generate(input_ids2, gen_length=128, block_length=32)
  assert res2.shape[0] == 2

def test_llada_moe_itersmooth():
  # Test generation with iteration smooth without kv-cache.
  print('Test block-wise diffusion MOE-LLM with iteration smooth without kv-cache')
  dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
  dllm1 = IterSmoothDiffusionLLM_test(model, decoder, BlockIteratorFactory(), early_stop=True)
  res = dllm.generate(input_ids, gen_length=128, block_length=32)
  res1 = dllm1.generate(input_ids, gen_length=128, block_length=32)
  assert dllm.num_forwards == dllm1.num_forwards
  assert dllm.cache_updates == 0
  assert res.shape[1] == res1.shape[1]
  res1 = res1.to(res.device)
  assert torch.all(res == res1)

def test_llada_moe_dual_cache():
  # Test generation with dual cache
  print('Test block-wise diffusion MOE-LLM with dual KV-cache')
  dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
  res = dllm.generate(input_ids, gen_length=256, block_length=32)
  res1, nfe = generate_with_dual_cache(model, input_ids, gen_length=256, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892)
  res1 = res1[res1 != 156892]
  assert res.shape[1] == len(res1)
  res1 = res1.to(res.device)
  assert torch.all(res == res1)

def test_llada_moe_dual_cache_batching():
  # Test generation with dual cache with batch size == 2
  print('Test block-wise diffusion MOE-LLM with dual KV-cache and batch size == 2')
  dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
  input_ids2 = get_prompts(tokenizer, mask_id=156895, device=device, num=2)
  res2 = dllm.generate(input_ids2, gen_length=256, block_length=32)
  assert res2.shape[0] == 2

def test_llada_moe_itersmooth_cache():
  # Test generation with iteration smooth with kv-cache.
  print('Test block-wise diffusion MOE-LLM with iteration smooth with kv-cache')
  dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
  dllm1 = IterSmoothDiffusionLLM_test(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
  res = dllm.generate(input_ids, gen_length=128, block_length=32)
  res1 = dllm1.generate(input_ids, gen_length=128, block_length=32)
  assert dllm.num_forwards == dllm1.num_forwards
  assert dllm.cache_updates > 0
  assert dllm.cache_updates == dllm1.cache_updates
  assert res.shape[1] == res1.shape[1]
  res1 = res1.to(res.device)
  assert torch.all(res == res1)

def test_llada_moe_itersmooth_vicinity_cache():
  # Test generation with iteration smooth and vicinity cache update.
  print('Test block-wise diffusion MOE-LLM with iteration smooth with vicinity cache update')
  dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
  dllm1 = IterSmoothWithVicinityCacheDiffusionLLM_test(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
  res = dllm.generate(input_ids, gen_length=128, block_length=32)
  res1 = dllm1.generate(input_ids, gen_length=128, block_length=32)
  assert dllm.num_forwards == dllm1.num_forwards
  assert dllm.cache_updates > 0
  assert dllm.cache_updates == dllm1.cache_updates
  assert res.shape[1] == res1.shape[1]
  res1 = res1.to(res.device)
  assert torch.all(res == res1)

        

        

    