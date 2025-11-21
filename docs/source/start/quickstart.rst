.. _quickstart:

==========
Quickstart
==========

Last updated: 2025-11-21

This page provides **minimal runnable examples** for different dInfer model
families on a single GPU:

- Dense LLaDA models (LLaDA-Base, LLaDA-Instruct, and LLaDA1.5)
- LLaDA-MoE models
- LLaDA2 block-diffusion models (mini/flash)

For advanced decoding strategies and performance tuning, see
:ref:`quickstart_next`.

.. contents::
   :local:
   :depth: 2

-----------------------------------
1. Dense LLaDA (Single-GPU)
-----------------------------------

This is the simplest way to get started with dInfer using a dense LLaDA model.

.. code-block:: python

   import torch
   from transformers import AutoTokenizer, AutoConfig
   from dinfer.model import LLaDAModelLM
   from dinfer import BlockIteratorFactory
   from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM

   # 1. Device
   device = torch.device("cuda:0")

   # 2. Local model path
   model_name = "/path/to/local/LLaDA-8B-Instruct"

   # 3. Load tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
   model = LLaDAModelLM.from_pretrained(
       model_name,
       torch_dtype=torch.bfloat16,
       init_device=str(device),
   ).eval().to(device)

   # 4. Configure decoder (threshold-based parallel decoding)
   mask_id = 126336  # LLaDA mask token ID
   eos_id  = 126081  # LLaDA EOS token ID

   decoder = ThresholdParallelDecoder(
       temperature=0.0,   # Greedy decoding
       threshold=0.9,     # Confidence threshold for token acceptance
       mask_id=mask_id,
       eos_id=eos_id,
   )

   # 5. Initialize diffusion LLM
   dllm = BlockWiseDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(start_block_align=True),
       cache_factory=None,   # No KV cache for basic usage
       early_stop=True,      # Stop when EOS is generated
   )

   # 6. Prepare input
   prompt = "What is the capital of France?"
   inputs = tokenizer(prompt).to(device)

   # 7. Generate
   with torch.no_grad():
       output_ids = dllm.generate(
           inputs["input_ids"],
           gen_length=256,     # Maximum new tokens
           block_length=64,    # Block size for parallel decoding
       )

   # 8. Decode output
   generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
   print(f"Generated: {generated_text}")

-----------------------------------
2. LLaDA-MoE (Single-GPU, fused)
-----------------------------------

LLaDA-MoE models use a fused MoE implementation in dInfer, with vLLM providing
the distributed/expert-parallel runtime. Even on **one GPU**, vLLM's
distributed environment must be initialized.

.. code-block:: python

   import os
   import torch
   from transformers import AutoTokenizer, AutoConfig
   from vllm import distributed
   from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
   from dinfer.model import FusedOlmoeForCausalLM
   from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM, BlockIteratorFactory

   # 1. Device
   gpu_id = 0
   torch.cuda.set_device(gpu_id)
   device = torch.device(f"cuda:{gpu_id}")

   # 2. Local model path
   model_name = "/path/to/local/LLaDA-MoE-7B-A1B-Instruct"

   # 3. Load tokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

   # 4. Initialize vLLM distributed environment (even for single GPU)
   world_size = 1
   rank = 0
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "45601"

   distributed.init_distributed_environment(
       world_size=world_size,
       rank=rank,
       init_method="env://",
       local_rank=rank,
       backend="nccl",
   )
   distributed.initialize_model_parallel(world_size, backend="nccl")

   # 5. Enable expert parallelism (EP)
   parallel_config = ParallelConfig(enable_expert_parallel=True)
   with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
       print("Loading LLaDA-MoE model with EP enabled...")

       model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
       model = FusedOlmoeForCausalLM(config=model_config).eval()
       model.load_weights(model_name, torch_dtype=torch.bfloat16)
       model = model.to(device)

       # MoE-specific special token IDs
       mask_id = 156895
       eos_id  = 156892

       decoder = ThresholdParallelDecoder(
           temperature=0.0,
           threshold=0.9,
           mask_id=mask_id,
           eos_id=eos_id,
       )

       dllm = BlockWiseDiffusionLLM(
           model=model,
           decoder=decoder,
           iterator_factory=BlockIteratorFactory(start_block_align=True),
           early_stop=True,
       )

       # 6. Generate
       prompt = "Explain quantum computing in simple terms."
       input_ids = tokenizer(prompt)["input_ids"]
       input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

       with torch.no_grad():
           output = dllm.generate(input_ids, gen_length=512, block_length=64)

       generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
       print(generated_text)

.. important::

   - vLLM distributed environment must be initialized even for **single-GPU**
     MoE inference.
   - ``ParallelConfig(enable_expert_parallel=True)`` is required for MoE.
   - Mask/EOS token IDs for MoE models differ from dense LLaDA.

-----------------------------------
3. LLaDA2 (Block Diffusion)
-----------------------------------

LLaDA2.x models are trained with **block diffusion** and should be wrapped with
``BlockDiffusionLLM`` in dInfer.

.. code-block:: python

   import torch
   from transformers import AutoTokenizer, AutoConfig
   from dinfer.model import LLaDA2MoeModelLM
   from dinfer import (
       BlockDiffusionLLM,
       BlockIteratorFactory,
       KVCacheFactory,
       ThresholdParallelDecoder,
   )

   # 1. Device
   device = torch.device("cuda:0")

   # 2. Local model path
   model_name = "/path/to/local/LLaDA2.0-mini-preview"

   # 3. Load tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
   model = LLaDA2MoeModelLM(config=model_config).eval()
   model.load_weights(model_name, torch_dtype=torch.bfloat16, device=device)
   model = model.to(device)

   # 4. Special tokens for LLaDA2
   mask_id = 156895
   eos_id  = 156892

   decoder = ThresholdParallelDecoder(
       temperature=0.0,
       threshold=0.9,
       mask_id=mask_id,
       eos_id=eos_id,
   )

   # LLaDA2 typically uses prefix cache with block-diffusion flag
   cache_factory = KVCacheFactory(cache_type="prefix", is_bd_model=True)

   dllm = BlockDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(
           start_block_align=True,
           use_block_diffusion=True,  # Enable block diffusion mode
       ),
       cache_factory=cache_factory,
       early_stop=True,
   )

   prompt = "Summarize the key ideas of diffusion language models."
   inputs = tokenizer(prompt, return_tensors="pt").to(device)

   with torch.no_grad():
       output_ids = dllm.generate(
           inputs["input_ids"],
           gen_length=2048,
           block_length=32,
       )

   print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

----------------------
4. Key Parameters
----------------------

Across all models, the following parameters are the most important knobs:

- ``gen_length``  
  Maximum number of **new tokens** to generate.
  Typical range: ``256–2048``.

- ``block_length``  
  Number of tokens decoded in parallel per diffusion iteration.

  - Larger values → more parallelism, higher speed, potentially lower quality.
  - Typical range: ``32–128`` to start with.

- ``threshold`` (decoder)  
  Confidence threshold (0.0–1.0) for accepting decoded tokens.

  - Higher → more conservative, higher quality, more iterations.
  - Lower → more aggressive, faster but potentially lower quality.

.. tip::

   - A good starting point is: ``block_length = 64`` and ``threshold = 0.9``. Lower the threshold for speed; increase it for quality.
   - For LLaDA2, the `block_length`` should ideally be set to 32, as this is consistent with the training setup and yields the best performance.

.. _quickstart_next:

----------------
5. Next Steps
----------------

Once you can run the basic examples above, you can explore:

- :doc:`Advanced decoding algorithms <advanced_decoding>`  
  (hierarchical decoding, credit-based decoding, iterative smoothing,
  block diffusion details, KV cache strategies)

- :doc:`Performance tuning <performance>`  
  (CUDA Graphs, prompt bucketing, tensor parallelism,
  multi-GPU benchmarks)
