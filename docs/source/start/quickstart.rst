.. _quickstart:

==========
Quickstart
==========

Last updated: 2025-11-13

This page shows minimal single-GPU inference, MoE usage notes, advanced decoding
variants, and performance tips for **dInfer**.

.. note::

   The examples below assume you have installed dInfer (see :ref:`installation`)
   and have a CUDA-enabled environment. Some snippets call third-party libraries
   (e.g., ``transformers``, ``vllm``). Install them as needed for your setup.

-----------------------------
1. Basic Single-GPU Inference
-----------------------------

A minimal example to get started with dInfer:

.. code-block:: python

   import torch
   from transformers import AutoTokenizer, AutoConfig
   from dinfer.model import LLaDAModelLM
   from dinfer import BlockIteratorFactory, KVCacheFactory
   from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM

   # Set device
   device = torch.device("cuda:0")

   # Load from local path
   model_name = "/path/to/local/LLaDA-8B-Instruct"

   # Load tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
   model = LLaDAModelLM.from_pretrained(
       model_name,
       torch_dtype=torch.bfloat16,
       init_device=str(device)
   ).eval().to(device)

   # Configure decoder with threshold-based parallel decoding
   mask_id = 126336  # LLaDA mask token ID
   eos_id  = 126081  # LLaDA EOS token ID
   decoder = ThresholdParallelDecoder(
       temperature=0.0,   # Greedy decoding
       threshold=0.9,     # Confidence threshold for token acceptance
       mask_id=mask_id,
       eos_id=eos_id
   )

   # Initialize diffusion LLM with block iterator
   dllm = BlockWiseDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(start_block_align=True),
       cache_factory=None,   # No KV cache for basic usage
       early_stop=True       # Stop when EOS is generated
   )

   # Prepare input
   prompt = "What is the capital of France?"
   input_ids = tokenizer(prompt)['input_ids']
   input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

   # Generate
   with torch.no_grad():
       output = dllm.generate(
           input_ids,
           gen_length=256,      # Maximum generation length
           block_length=64      # Block size for parallel decoding
       )

   # Decode output
   generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
   print(f"Generated: {generated_text}")

--------------------------------
1.1 Basic LLaDA-MoE Inference
--------------------------------

For Mixture-of-Experts models, use the fused implementation with proper vLLM configuration:

.. code-block:: python

   import os
   import torch
   from transformers import AutoTokenizer, AutoConfig
   from vllm import distributed
   from vllm.config import ParallelConfig, VllmConfig
   from vllm.config import set_current_vllm_config
   from dinfer.model import FusedOlmoeForCausalLM
   from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM, BlockIteratorFactory

   # Setup device
   gpu_id = 0
   torch.cuda.set_device(gpu_id)
   device = torch.device(f"cuda:{gpu_id}")

   # Load model from local path
   model_name = "/path/to/local/LLaDA-MoE-7B-A1B-Instruct"

   # Load tokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

   # Initialize vLLM distributed environment (even for single GPU)
   world_size = 1
   rank = 0
   os.environ['MASTER_ADDR'] = 'localhost'
   os.environ['MASTER_PORT'] = '45601'
   distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
   distributed.initialize_model_parallel(world_size, backend='nccl')

   # Expert Parallelism (required for MoE models)
   parallel_config = ParallelConfig(enable_expert_parallel=True)
   with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
       print("Loading model with EP enabled...")

       model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
       model = FusedOlmoeForCausalLM(config=model_config).eval()
       model.load_weights(model_name, torch_dtype=torch.bfloat16)
       model = model.to(device)

       # MoE uses different special token IDs
       mask_id = 156895
       eos_id  = 156892

       decoder = ThresholdParallelDecoder(
           temperature=0.0,
           threshold=0.9,
           mask_id=mask_id,
           eos_id=eos_id
       )

       dllm = BlockWiseDiffusionLLM(
           model=model,
           decoder=decoder,
           iterator_factory=BlockIteratorFactory(start_block_align=True),
           early_stop=True
       )

       # Generate
       prompt = "Explain quantum computing in simple terms."
       input_ids = tokenizer(prompt)['input_ids']
       input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

       with torch.no_grad():
           output = dllm.generate(input_ids, gen_length=512, block_length=64)

       generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
       print(generated_text)

.. important::

   - Initialize the vLLM distributed environment **even on single GPU**.
   - Use ``ParallelConfig(enable_expert_parallel=True)`` for MoE models.
   - Keep all model operations inside ``set_current_vllm_config(...)`` context.

------------------------------
1.2 Understanding Parameters
------------------------------

- **``gen_length``**: Maximum number of tokens to generate.
- **``block_length``**: Number of tokens decoded in parallel per diffusion iteration.
  
  - Larger values → more parallelism but potentially lower quality.
  - Typical range: **32–128**.

- **``threshold``**: Confidence threshold (0.0–1.0) for accepting decoded tokens.
  
  - Higher → more conservative, higher quality, more iterations.
  - Lower → more aggressive, faster but potentially lower quality.

-------------------------------------
2. Advanced Decoding Algorithms (DL)
-------------------------------------

2.1 Hierarchical Decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dinfer import HierarchyDecoder

   decoder = HierarchyDecoder(
       temperature=0.0,
       threshold=0.9,      # High confidence threshold
       low_threshold=0.3,  # Low confidence threshold
       mask_id=mask_id,
       eos_id=eos_id
   )

   dllm = BlockWiseDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(start_block_align=True),
       early_stop=True
   )

   output = dllm.generate(input_ids, gen_length=512, block_length=64)

**How it works:**

- Tokens with confidence > ``threshold`` are immediately accepted.
- Tokens with confidence < ``low_threshold`` remain masked.
- Intermediate-confidence tokens are accepted if they are local maxima in masked regions.

2.2 Credit-Based Threshold Decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dinfer import CreditThresholdParallelDecoder

   decoder = CreditThresholdParallelDecoder(
       temperature=0.0,
       threshold=0.9,
       mask_id=mask_id,
       eos_id=eos_id
   )

   dllm = BlockWiseDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(start_block_align=True),
       early_stop=True
   )
   output = dllm.generate(input_ids, gen_length=512, block_length=64)

**Benefits:**

- Accumulates "credits" for consistently high-confidence tokens.
- Reduces premature acceptance in difficult regions.
- Stabilizes convergence on challenging prompts.

2.3 Iterative Smoothing with Vicinity Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dinfer import IterSmoothWithVicinityCacheDiffusionLLM, KVCacheFactory

   cache_factory = KVCacheFactory(
       cache_type='dual',     # prefix + suffix refresh
       is_bd_model=False
   )

   dllm = IterSmoothWithVicinityCacheDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(start_block_align=True),
       cache_factory=cache_factory,
       early_stop=True,
       cont_weight=0.3,       # Continuity regularization
       prefix_look=16,        # Look-back context size
       after_look=16,         # Look-ahead context size
       warmup_steps=4
   )
   output = dllm.generate(input_ids, gen_length=512, block_length=64)

**Key parameters:**

- ``cont_weight`` (0.0–1.0): larger → smoother token transitions.
- ``prefix_look``: tokens to look back for context.
- ``after_look``: tokens to look ahead for context.
- ``warmup_steps``: iterations with full diffusion at the start.

2.4 Block Diffusion (LLaDA2.0 Models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from transformers import AutoConfig
   from dinfer.model import LLaDA2MoeModelLM
   from dinfer import BlockDiffusionLLM, KVCacheFactory, BlockIteratorFactory, ThresholdParallelDecoder

   device = torch.device("cuda:0")
   model_name = "/path/to/local/LLaDA2.0-mini-preview"

   # Load LLaDA2 model
   model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
   model = LLaDA2MoeModelLM(config=model_config).eval()
   model.load_weights(model_name, torch_dtype=torch.bfloat16, device=device)
   model = model.to(device)

   mask_id = 156895
   eos_id  = 156892

   decoder = ThresholdParallelDecoder(
       temperature=0.0,
       threshold=0.9,
       mask_id=mask_id,
       eos_id=eos_id
   )

   cache_factory = KVCacheFactory(cache_type='prefix', is_bd_model=True)

   dllm = BlockDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(
           start_block_align=True,
           use_block_diffusion=True  # Enable block diffusion mode
       ),
       cache_factory=cache_factory,
       early_stop=True
   )

   output = dllm.generate(input_ids, gen_length=2048, block_length=32)

2.5 KV Cache Strategies
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dinfer import KVCacheFactory, BlockWiseDiffusionLLM, BlockIteratorFactory

   # Option 1: Prefix caching only (common for causal LMs)
   cache_factory = KVCacheFactory(cache_type='prefix', is_bd_model=False)

   # Option 2: Dual caching (prefix + suffix refresh)
   cache_factory = KVCacheFactory(cache_type='dual', is_bd_model=False)

   # Option 3: No caching (simplest, but slower)
   cache_factory = None

   dllm = BlockWiseDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(start_block_align=True),
       cache_factory=cache_factory,
       early_stop=True
   )

**Cache Type Comparison**

- ``prefix``: caches prompt/fixed prefix; best for single-turn, low memory.
- ``dual``: caches prefix + refreshes vicinity tokens; best for complex/multi-turn, medium memory.
- ``None``: recompute everything; fine for very short sequences.

--------------------------------------------
3. Performance Optimization: Practical Tips
--------------------------------------------

3.1 ``torch.compile()`` (Fusion & CUDA Graphs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch

   # Enable torch.compile on model forward pass
   model.forward = torch.compile(
       model.forward,
       mode='reduce-overhead',  # Enables CUDA Graph capture on stable regions
       fullgraph=False,
       dynamic=True
   )

**Shape signatures & warmup**

- CUDA Graphs are compiled per input signature (shape/stride/dtype/device).
- The first iteration for a *new* signature incurs extra compile/capture time.
- Warm up before timing:

.. code-block:: python

   # Warmup to populate CUDA Graphs
   for _ in range(2):
       _ = dllm.generate(input_ids, gen_length=256, block_length=64)

   # Measure steady-state
   output = dllm.generate(input_ids, gen_length=512, block_length=64)

**Guidance**

- Bucket/pad prompts to a small set of lengths (multiples of 16/32).
- Keep ``block_length`` fixed across runs when comparing speed.
- If shapes vary wildly or runs are short, use ``mode='default'`` (fusion only).

3.2 Tensor Parallelism (Multi-GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skeleton for TP with vLLM:

.. code-block:: python

   import os
   import torch
   from transformers import AutoTokenizer, AutoConfig
   from vllm import distributed
   from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
   from dinfer.model import FusedOlmoeForCausalLM
   from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM, BlockIteratorFactory

   def setup_model_with_tp(world_size, rank, gpu_id, model_name):
       """Setup model with tensor parallelism"""

       torch.cuda.set_device(gpu_id)
       device = torch.device(f"cuda:{gpu_id}")

       tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

       os.environ['MASTER_ADDR'] = 'localhost'
       os.environ['MASTER_PORT'] = str(45601)

       distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
       distributed.initialize_model_parallel(world_size, backend='nccl')

       print(f"[Rank {rank}] Loading model with TP={world_size}")

       parallel_config = ParallelConfig(enable_expert_parallel=True)
       with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
           model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

           model = FusedOlmoeForCausalLM(config=model_config).eval()
           model.load_weights(model_name, torch_dtype=torch.bfloat16)

           if world_size > 1:
               print(f"[Rank {rank}] Enabling TP")
               model.tensor_parallel(tp_size=world_size)

           model = model.to(device)

           # Optional: compile for performance
           model.forward = torch.compile(
               model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True
           )

           # Typical decoder setup (IDs depend on model)
           mask_id, eos_id = 156895, 156892
           decoder = ThresholdParallelDecoder(
               temperature=0.0, threshold=0.9, mask_id=mask_id, eos_id=eos_id
           )

           dllm = BlockWiseDiffusionLLM(
               model=model,
               decoder=decoder,
               iterator_factory=BlockIteratorFactory(start_block_align=True),
               early_stop=True
           )
           return dllm, tokenizer, device

**Key points**

- Initialize vLLM distributed env **before** loading the model.
- ``initialize_model_parallel(world_size)`` sets up TP groups.
- Call ``model.tensor_parallel(tp_size)`` after loading weights.

-----------------
4. What’s Next?
-----------------

- See :ref:`installation` for environment/setup details.
- Add your first dataset or evaluation and try the advanced decoders above.
- When you add more guides, include them in the :doc:`index </index>` to appear in the sidebar.
