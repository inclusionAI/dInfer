## Evaluate Dinfer performance on different benchmarks 
We provide an evaluation framework based on dInfer integrated with the 🤗 HuggingFace lm‑eval‑harness.
It supports Tensor Parallel (TP) and Data Parallel (DP) inference for easy evaluation of large‑scale dLLMs.

For the llada‑moe model, we have adapted two benchmark tasks already integrated in this framework:

* mbpp_sanitized_llada: A sanitized Python code‑generation benchmark derived from MBPP;
* gsm8k_llada: A math reasoning benchmark adapted from GSM8K.

### 1️⃣ Install Dependencies

```bash
pip install -U accelerate evaluate datasets lm_eval hf_transfer
```

### 2️⃣ Set Environment Variables

Before running evaluation, set these variables:

```bash
# Allow model code evaluation
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
# Select GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

### 3️⃣ Define Hyperparameters

```bash
length=1024              # generation length
block_length=64          # block size for diffusion LLM
model_path='your_model_path'
output_path='your_output_folder'

# Cache & diffusion config
cache='dual'             # 'dual' for dual cache/ 'prefix' for prefix cache / '' for no cache
prefix_look=16
after_look=16
warmup_times=4
cont_weight=0.3
use_credit=False         # use credit for credit-based decoding
use_compile=True
use_cudagraph=True

# Parallelism config
gpus='0,1,2,3'
parallel='tp'            # 'tp' for tensor parallel, 'dp' for accelerate DP

# Evaluation task
# for llada 1.5 use tasks gsm8k_llada1.5 mbpp_sanitized_llada1.5
# for llada moe use tasks gsm8k_llada_moe mbpp_sanitized_llada_moe
task=mbpp_sanitized_llada_moe # or gsm8k_llada_moe
```
### ⚙️ Run with Tensor Parallel (TP)

Run evaluation with **multi‑GPU tensor parallelism** (default):

```bash
parallel_decoding='threshold'  # or "hierarchy"
threshold=0.8
low_threshold=0.5

python eval_dinfer.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args \
  model_path=${model_path},\
  gen_length=${length},\
  block_length=${block_length},\
  threshold=${threshold},\
  low_threshold=${low_threshold},\
  show_speed=True,\
  save_dir=${output_path},\
  parallel_decoding=${parallel_decoding},\
  prefix_look=${prefix_look},\
  after_look=${after_look},\
  cache=${cache},\
  warmup_times=${warmup_times},\
  use_compile=${use_compile},\
  parallel=${parallel},\
  cont_weight=${cont_weight},\
  use_credit=${use_credit},\
  gpus=${gpus} \
  --output_path ${output_path} \
  --include_path ./tasks \
  --apply_chat_template
```

💡 *Internally, this launches multiple GPU processes and automatically initializes NCCL and tensor‑parallel communication.*


### 🧩 Run with Accelerate (Data Parallel, DP)

If you prefer **data‑parallel** evaluation (each GPU handles separate requests):

```bash
parallel='dp'

accelerate launch eval_dinfer.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args \
  model_path=${model_path},\
  gen_length=${length},\
  block_length=${block_length},\
  threshold=${threshold},\
  low_threshold=${low_threshold},\
  show_speed=True,\
  save_dir=${output_path},\
  parallel_decoding=${parallel_decoding},\
  prefix_look=${prefix_look},\
  after_look=${after_look},\
  cache=${cache},\
  warmup_times=${warmup_times},\
  use_compile=${use_compile},\
  parallel=${parallel},\
  cont_weight=${cont_weight},\
  use_credit=${use_credit},\
  gpus=${gpus} \
  --output_path ${output_path} \
  --include_path ./tasks \
  --apply_chat_template
```

✅ `accelerate` automatically sets multi‑GPU ranks, ports, and distributed environments.

---

### 🧮 Use Hierarchy Parallel Decoding

Enable hierarchical decoding for improved quality:

```bash
parallel_decoding='hierarchy'
threshold=0.92
low_threshold=0.62

python eval_dinfer.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args \
  model_path=${model_path},\
  gen_length=${length},\
  block_length=${block_length},\
  threshold=${threshold},\
  low_threshold=${low_threshold},\
  save_dir=${output_path},\
  parallel_decoding=${parallel_decoding},\
  prefix_look=${prefix_look},\
  after_look=${after_look},\
  cache=${cache},\
  warmup_times=${warmup_times},\
  cont_weight=${cont_weight} \
  --output_path ${output_path} \
  --include_path ./tasks \
  --apply_chat_template \
  --log_samples
```

---

### 💿 Use Credit for Threshold Parallel Decoding

```bash
parallel_decoding='threshold'
threshold=0.8
use_credit=True

python eval_dinfer.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args \
  model_path=${model_path},\
  gen_length=${length},\
  block_length=${block_length},\
  threshold=${threshold},\
  low_threshold=${low_threshold},\
  save_dir=${output_path},\
  parallel_decoding=${parallel_decoding},\
  prefix_look=${prefix_look},\
  after_look=${after_look},\
  cache=${cache},\
  warmup_times=${warmup_times},\
  cont_weight=${cont_weight} \
  --output_path ${output_path} \
  --include_path ./tasks \
  --apply_chat_template \
  --log_samples
```