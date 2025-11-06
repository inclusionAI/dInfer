import os
import logging
from multiprocessing import Process
import random

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

model_path = "/mnt/dllm/model_hub/LLaDA-1.5/"
# model_path = "/data/myx/llm/vllm/model/LLaDA-1_5/"

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

class SimulateBlockIterator:
    """ This class simulates the block iterator in VicinityCacheDiffusionLLM.
    """
    def __init__(self, x, block_length, mask_id):
        self.x = x
        self.iter = 0
        self.block_length = block_length
        self.mask_id = mask_id

    def __iter__(self):
        self.iter = 0
        return self

    def move_next(self):
        current_block_start = self.x.prompt.shape[1] + self.iter * self.block_length
        current_block_end = current_block_start + self.block_length
        current_block_end = min(current_block_end, self.x.total_length)
        # If all tokens have been decoded, move to the next block.
        if torch.all(self.x[:, current_block_start:current_block_end] != self.mask_id):
            self.iter += 1

    def __next__(self):
        self.move_next()
        current_block_start = self.x.prompt.shape[1] + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = current_block_start + self.block_length
        current_block_end = min(current_block_end, self.x.total_length)
        return BlockLoc(current_block_start, current_block_end), self.x[:, current_block_start:current_block_end]

class SimulateBlockIteratorFactory:
    def create(self, x, block_length):
        return SimulateBlockIterator(x, block_length, 126336)

torch.cuda.set_device(0)
device = torch.device(0)
config = AutoConfig.from_pretrained(model_path)
config.flash_attention = True
model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
model = model.to(device)
fastdllm_model = LLaDAModelLM_fastdllm.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
fastdllm_model = fastdllm_model.to(device)
decoder = ThresholdParallelDecoder(0, threshold=0.9, use_float64=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
input_ids = get_prompts(tokenizer, mask_id=126336, device=device)
batch_size = 1
input_ids = torch.tensor(input_ids).to(device).repeat(batch_size, 1)


def test_sw_dual_cache():
  print('Test sliding-window diffusion LLM with dual KV-cache')
  dllm = VicinityCacheDiffusionLLM(model, decoder, SimulateBlockIteratorFactory(), KVCacheFactory('dual'))
  res = dllm.generate(input_ids, gen_length=128, block_length=32)
  res1, nfe = generate_with_dual_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
  res1 = res1[res1 != 126081]
  assert res.shape[1] == len(res1)
  res1 = res1.to(res.device)
  assert torch.all(res == res1)

def test_prefix_cache():
  print('Test block-wise diffusion LLM with prefix KV-cache')
  dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('prefix'))
  res = dllm.generate(input_ids, gen_length=128, block_length=32)
  res1, nfe = generate_with_prefix_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
  res1 = res1[res1 != 126081]
  assert res.shape[1] == len(res1)
  res1 = res1.to(res.device)
  assert torch.all(res == res1)

def test_dual_cache():
  print('Test block-wise diffusion LLM with dual KV-cache')
  dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=KVCacheFactory('dual'), early_stop=True)
  res = dllm.generate(input_ids, gen_length=128, block_length=32)
  res1, nfe = generate_with_dual_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
  res1 = res1[res1 != 126081]
  assert res.shape[1] == len(res1)
  res1 = res1.to(res.device)
  assert torch.all(res == res1)
    

if __name__ == '__main__':
    print("Start test sliding window with dual cach...")
    test_sw_dual_cache()
    print("Start test prefix cache...")
    test_prefix_cache()
    print("Start test dual cache...")
    test_dual_cache()