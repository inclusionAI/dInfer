<div align="center">

# dInfer: An Efficient Inference Framework for Diffusion Language Models

</div>

<h4 align="center">

[![License: MIT](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![HuggingFace: Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
[![Technical Report: Arxiv](https://img.shields.io/badge/Technical%20Report-Arxiv-red)](https://arxiv.org/abs/2510.08666)

<!-- [![arXiv][arxiv-image]][arxiv-url] -->

</h4>

dInfer is an efficient and extensible inference framework for dLLMs. It modularizes inference into four components:
model, diffusion iteration manager, decoding strategy and KV-cache management, and provides well-designed APIs for
flexible combinations of algorithms in each component.

<p align="center">
  <img src="https://raw.githubusercontent.com/inclusionAI/dInfer/refs/heads/master/assets/Framework2.png" alt="dInfer v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: Overall Architecture of dInfer
</p>

dInfer supports multiple dLLM variants, including LLaDA and LLaDA-MoE. It introduces multiple algorithms in each of
the components to improve the decoding quality and inference speed. This includes a soft diffusion iteration algorithm
for smoother denoising, hierarchical and credit decoding for enhanced parallel decoding, and a vicinity refresh strategy
for KV-cache management to mitigate cache staleness.
Beyond algorithmic improvements, it integrates several system-level optimizations. It supports both tensor parallelism
(TP) and expert parallelism (EP) to maximize GPU utilization even at batch size 1. It leverages PyTorch compilation and
NVIDIA CUDA Graphs for efficient kernel execution, and introduces a loop unrolling mechanism to eliminate CUDA stream
bubbles across diffusion iterations.

## Benchmark results

<p align="center">
  <img src="https://raw.githubusercontent.com/inclusionAI/dInfer/refs/heads/master/assets/dinfer_tps.png" alt="dInfer v0.1 speedup" width="600">
  <br>
  <b>Figure</b>: Benchmark results
</p>

On HumanEval, dInfer achieves over 1,100 TPS at batch size 1, and averages more than 800 TPS across six benchmarks on
a single node with $8\times$ H800 GPUs. Compared to Fast-dLLM, dInfer delivers more than a $10\times$ speedup while
maintaining accuracy; on LLaDA-MoE it provides a $2-3\times$ speedup over QWen2.5-3B on vLLM with comparable quality.

## Get started

Please follow the instruction below to install dInfer.

```
git clone https://github.com/inclusionAI/dInfer.git
cd dInfer
pip install .
```

### Run dInfer with LLaDA-MoE downloaded from HuggingFace

This project supports using LLaDA(-MoE) checkpoints from HuggingFace. After downloading a model, run the CPU conversion script to fuse MoE experts into FusedMoE format that can be loaded locally.

Step 1: Download checkpoints

```bash
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Example: Instruct checkpoint
hf download inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --repo-type model \
  --local-dir /path/to/LLaDA-MoE-7B-A1B-Instruct
```

Step 2: Convert to FusedMoE format

We need to convert the model weight format to support FusedMoE.
Use the conversion tool to fuse the experts in the MoE layer.

```bash
# From repo root
python tools/transfer.py \
  --input  /path/to/LLaDA-MoE-7B-A1B-Instruct \
  --output /path/to/LLaDA-MoE-7B-A1B-Instruct-fused
```

Step 3: Use the model in dInfer

```python
import os
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config

from dinfer.model import FusedOlmoeForCausalLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM

m = "/path/to/LLaDA-MoE-7B-A1B-Instruct-fused"
tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)

device = torch.device(0)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12346'
distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
distributed.initialize_model_parallel(1, backend='nccl')
parallel_config = ParallelConfig(enable_expert_parallel = True)
with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
    model_config = AutoConfig.from_pretrained(m, trust_remote_code=True)
    model = FusedOlmoeForCausalLM(config=model_config).eval()
    model.load_weights(m, torch_dtype=torch.bfloat16)
    model = model.to(device)

decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892)
dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(True), cache_factory=KVCacheFactory('dual'))

prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?"
m = [{"role": "user", "content": prompt}, ]
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(prompt)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
res = dllm.generate(input_ids, gen_length=1024, block_length=64)
print(tokenizer.decode(res[0, input_ids.shape[1]:], skip_special_tokens=False))
```

### Evaluate Dinfer performance on different benchmarks 
We provide an integration with lm‑eval‑harness, which allows you to evaluate dInfer‑based models on a variety of standard benchmarks.
To simplify usage, we also include a ready‑to‑run bash script that contains all the necessary environment variables and evaluation parameters.
You only need to modify the configuration values in the script (e.g., model_path, parallel_decoding, threshold, tp_size, etc.),
and then execute it to launch the full evaluation process.

```python
cd evaluations
bash eval_gsm8k.sh
```


## Cite

```
@article{dinfer,
    title={dInfer: An Efficient Inference Framework for Diffusion Language Models},
    author={Yuxin Ma, Lun Du, Lanning Wei, Kun Chen, Qian Xu, Kangyu Wang, Guofeng Feng, Guoshan Lu, Lin Liu, Xiaojing Qi, Xinyuan Zhang, Zhen Tao, Haibo Feng, Ziyun Jiang, Ying Xu, Zenan Huang, Yihong Zhuang, Haokai Xu, Jiaqi Hu, Zhenzhong Lan, Junbo Zhao, Jianguo Li, Da Zheng},
    year={2025},
    journal={arXiv preprint arXiv:2510.08666}
}
```
