<div align="center">
  <img src="assets/logo.svg" width="40%" alt="dInfer" />
</div>

[![License: MIT](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![HuggingFace: Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
[![Technical Report: Arxiv](https://img.shields.io/badge/Technical%20Report-Arxiv-red)](https://arxiv.org/abs/2510.08666)


## Introduction
dInfer is an efficient and extensible inference framework for dLLMs. As illustrated in the following architecture, it modularizes inference into four components:
*model*, *diffusion iteration manager*, *decoder* and *KV-cache manager*. It provides well-designed APIs for
flexible algorithms combinations in each component. Now supports batched inference for improved throughput.

<p align="center">
  <img src="assets/Framework2.png" alt="dInfer v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: Overall Architecture of dInfer
</p>

dInfer supports multiple dLLM variants, including LLaDA and LLaDA-MoE.

**Algorithmic improvements:**
- Soft diffusion iteration for smoother denoising
- Hierarchical and credit decoding for enhanced parallel decoding
- Vicinity refresh strategy for KV-cache management to mitigate cache staleness

**System-level optimizations:**
- Tensor Parallelism (TP) and Expert Parallelism (EP) to maximize GPU utilization across batch sizes
- Dynamic batching support for improved throughput on multi-request workloads
- PyTorch compilation and NVIDIA CUDA Graphs for efficient kernel execution
- Loop unrolling mechanism to eliminate CUDA stream bubbles across diffusion iterations

## Contents
- [Supported Models](#supported-models)
- [Benchmark Results](#benchmark-results)
- [Getting Started](#getting-started)

## Supported Models

dInfer supports multiple diffusion language model variants with different architectures and sizes. Below are the HuggingFace model links and their corresponding implementation files:

### LLaDA2.0
**Implementation**: [modeling_llada2_moe.py](python/dinfer/model/modeling_llada2_moe.py)

| Model | Size | HuggingFace Link | Description |
|-------|------|------------------|-------------|
| LLaDA2.0-mini-preview | 16B | [inclusionAI/LLaDA2.0-mini-preview](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview) | MoE dLLM focused on efficient reasoning and tool use |
| LLaDA2.0-flash-preview | 100B | [inclusionAI/LLaDA2.0-flash-preview](https://huggingface.co/inclusionAI/LLaDA2.0-flash-preview) | Large MoE dLLM targeting advanced code/math reasoning |

**Features**:
- Trained using Block Diffusion to improve throughput and stability
- Supports tool calling and complex agent-based task execution
- Excels at complex mathematical reasoning and code generation
- Supports both Expert Parallelism (EP) and Tensor Parallelism (TP)
- **Decoding algorithms**: Hierarchical, Credit, Threshold

### LLaDA-MoE Models (Mixture-of-Experts)

**Implementation**: [modeling_fused_olmoe.py](python/dinfer/model/modeling_fused_olmoe.py)

| Model | Size | HuggingFace Link | Description |
|-------|------|------------------|-------------|
| LLaDA-MoE-7B-A1B-Base | 7B | [inclusionAI/LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base) | Pretrained MoE dLLM |
| LLaDA-MoE-7B-A1B-Instruct | 7B | [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) | Instruction-tuned MoE variant |

**Features**:
- Sparse Mixture-of-Experts with 64 experts
- FusedMoE optimization for efficient inference
- Support both Expert Parallelism (EP) and Tensor Parallelism (TP)
- **Decoding algorithms**: Hierarchical, Credit, Threshold

### LLaDA Models (Dense)

**Implementation**: [modeling_llada.py](python/dinfer/model/modeling_llada.py)

| Model | Size | HuggingFace Link | Description |
|-------|------|------------------|-------------|
| LLaDA-8B-Base | 8B | [GSAI-ML/LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) | Pretrained dense dLLM |
| LLaDA-8B-Instruct | 8B | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | SFT instruction-following variant |
| LLaDA-1.5 | 8B | [GSAI-ML/LLaDA-1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5) | LLaDA-8B aligned with VRPO |

**Features**:
- Dense transformer architecture
- Optimized for single-GPU and multi-GPU inference
- **Decoding algorithms**: Hierarchical, Credit, Threshold


## Benchmark Results

<p align="center">
  <img src="assets/dinfer_tps.png" alt="dInfer v0.1 speedup" width="600">
  <br>
  <b>Figure</b>: Benchmark results
</p>

**Performance on HumanEval:**
- Over 1,100 TPS at batch size 1
- Averages 800+ TPS across six benchmarks on a single node with 8× H800 GPUs

**Speedup comparisons:**
- 10× faster than Fast-dLLM while maintaining accuracy
- 2-3× faster than Qwen2.5-3B on vLLM (LLaDA-MoE) with comparable quality

## Getting Started

Please follow the instruction below to install dInfer.

```
git clone https://github.com/inclusionAI/dInfer.git
cd dInfer
pip install .
```

### Download from HuggingFace and Convert to FusedMoE Format

This project supports using LLaDA and LLaDA-MoE checkpoints from HuggingFace. After downloading a model, run the conversion script to fuse MoE experts into FusedMoE format for local loading.

#### 1) Download checkpoints

```bash
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Example: Instruct checkpoint
hf download inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --repo-type model \
  --local-dir /path/to/LLaDA-MoE-7B-A1B-Instruct
```

#### 2) Convert to FusedMoE format

Use the conversion tool to fuse MoE experts.

```bash
# From repo root
python tools/transfer.py \
  --input  /path/to/LLaDA-MoE-7B-A1B-Instruct \
  --output /path/to/LLaDA-MoE-7B-A1B-Instruct-fused
```

**After conversion, the output directory contains:**
- `modeling_fused_olmoe.py`
- `config.json` with:
  - `architectures: [FusedOlmoeForCausalLM]`
  - `auto_map.AutoModelForCausalLM: modeling_fused_olmoe.FusedOlmoeForCausalLM`

#### 3) Load the model

- **Load via Auto classes:**
```python
from dinfer.model import AutoModelForCausalLM
from transformers import AutoTokenizer
m = "/path/to/LLaDA-MoE-7B-A1B-Instruct-fused"
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True, torch_dtype="bfloat16")
```

### Tutorial

- **Benchmark-only (speed)** — scripts in `benchmarks/`
  - Measure throughput (TPS) only; predictions are saved under `--output_dir`; no automatic scoring.
  - Example 1 (LLaDA-MoE, threshold decoder, TP across 4 GPUs):

```bash
python benchmarks/benchmark_dataset.py \
  --model_name inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --model_type llada_moe \
  --dataset dataset_path \
  --gen_len 1024 \
  --block_length 64 \
  --gpu 0,1,2,3 \
  --output_dir runs/llada_moe_threshold \
  --use_tp \
  --parallel_decoding threshold \
  --threshold 0.8 \
  --cache dual \
  --prefix_look 16 \
  --after_look 16 \
  --warmup_times 4 \
  --cont_weight 0.3
```

  - Example 2 (threshold decoder, TP across 4 GPUs, LLaDA2-mini):

  ```
  python benchmarks/benchmark_dataset.py \
    --model_name inclusionAI/LLaDA2.0-mini-preview \
    --model_type llada2 \
    --dataset dataset_path \
    --gen_len 2048 \
    --block_length 32 \
    --gpu 0,1,2,3 \
    --output_dir runs/llada2_mini \
    --use_tp \
    --parallel_decoding threshold \
    --threshold 0.9 \
    --cache prefix \
    --use_bd
  ```

  - Example 3 (threshold decoder, TP across 4 GPUs, LLaDA2-flash):
  ```
  python benchmarks/benchmark_dataset.py \
    --model_name inclusionAI/LLaDA2.0-flash-preview \
    --model_type llada2 \
    --dataset dataset_path \
    --gen_len 2048 \
    --block_length 32 \
    --gpu 0,1,2,3 \
    --output_dir runs/llada2_mini \
    --use_tp \
    --parallel_decoding threshold \
    --threshold 0.9 \
    --cache prefix \
    --use_bd
  ```
 
  - Other entry points:
    - `benchmark.py` — Single-sample profiling.


- **End-to-end evaluation (speed + accuracy)** — scripts in `evaluations/`
  - Built on HuggingFace `lm-eval-harness`; computes both TPS and benchmark scores.
  - Tasks provided:
    - `gsm8k_llada`: math reasoning.
    - `mbpp_sanitized_llada`: sanitized Python code generation.
  - For more examples and comprehensive instructions, see [our quickstart guide](evaluations/eval_guide.md).
  - Currently, the evaluation configuration is only aligned with LLaDA-MoE; lm-eval evaluations for llada2-mini/flash and other models will be updated later.


## Citation
```
@article{dinfer,
    title={dInfer: An Efficient Inference Framework for Diffusion Language Models},
    author={Yuxin Ma, Lun Du, Lanning Wei, Kun Chen, Qian Xu, Kangyu Wang, Guofeng Feng, Guoshan Lu, Lin Liu, Xiaojing Qi, Xinyuan Zhang, Zhen Tao, Haibo Feng, Ziyun Jiang, Ying Xu, Zenan Huang, Yihong Zhuang, Haokai Xu, Jiaqi Hu, Zhenzhong Lan, Junbo Zhao, Jianguo Li, Da Zheng},
    year={2025},
    journal={}
}
```
