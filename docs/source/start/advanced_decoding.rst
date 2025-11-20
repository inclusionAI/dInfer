.. _advanced_decoding:

=========================
Advanced Decoding Methods
=========================

Last updated: 2025-11-20

This page introduces several advanced decoding strategies supported by dInfer,
building on the basic setup shown in :doc:`quickstart`.

.. note::

   In all code snippets below, we assume you have already:

   - Loaded a tokenizer and model.
   - Defined ``mask_id`` and ``eos_id``.
   - Created a prompt and corresponding ``input_ids`` tensor on the correct device.

--------------------------------
1. Hierarchical Decoding
--------------------------------

Hierarchical decoding uses two thresholds to balance quality and speed.

.. code-block:: python

   from dinfer import HierarchyDecoder, BlockWiseDiffusionLLM, BlockIteratorFactory

   decoder = HierarchyDecoder(
       temperature=0.0,
       threshold=0.9,      # High confidence threshold
       low_threshold=0.3,  # Low confidence threshold
       mask_id=mask_id,
       eos_id=eos_id,
   )

   dllm = BlockWiseDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(start_block_align=True),
       early_stop=True,
   )

   output = dllm.generate(input_ids, gen_length=512, block_length=64)

**How it works:**

- Tokens with confidence > ``threshold`` are accepted immediately.
- Tokens with confidence < ``low_threshold`` remain masked.
- Tokens with intermediate confidence are accepted **only if** they are
  local maxima within masked regions.

This creates a hierarchy:

1. High-confidence tokens.
2. Medium-confidence tokens in promising regions.
3. Remaining low-confidence tokens.

----------------------------------------------
2. Credit-Based Threshold Decoding
----------------------------------------------

Credit-based decoding tracks decoding history to make better decisions.

.. code-block:: python

   from dinfer import CreditThresholdParallelDecoder
   from dinfer import BlockWiseDiffusionLLM, BlockIteratorFactory

   decoder = CreditThresholdParallelDecoder(
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

   output = dllm.generate(input_ids, gen_length=512, block_length=64)

**Benefits:**

- Accumulates "credits" for tokens that repeatedly have high confidence.
- Helps prevent premature acceptance in difficult regions.
- Leads to more stable convergence in challenging generation scenarios.

-----------------------------------------------------
3. Iterative Smoothing with Vicinity-Aware KV Cache
-----------------------------------------------------

To improve coherence, you can use iterative smoothing together with a
vicinity-aware KV cache.

.. code-block:: python

   from dinfer import IterSmoothWithVicinityCacheDiffusionLLM, KVCacheFactory
   from dinfer import BlockIteratorFactory

   cache_factory = KVCacheFactory(
       cache_type='dual',     # Use both prefix and suffix caching
       is_bd_model=False,
   )

   dllm = IterSmoothWithVicinityCacheDiffusionLLM(
       model=model,
       decoder=decoder,
       iterator_factory=BlockIteratorFactory(start_block_align=True),
       cache_factory=cache_factory,
       early_stop=True,
       cont_weight=0.3,       # Continuity weight for smoothing
       prefix_look=16,        # Look-back context size
       after_look=16,         # Look-ahead context size
       warmup_steps=4,        # Number of warmup iterations
   )

   output = dllm.generate(input_ids, gen_length=512, block_length=64)

**Key parameters:**

- ``cont_weight`` (0.0–1.0):  
  Controls the strength of continuity regularization.  
  Higher → smoother transitions; lower → more independent predictions.

- ``prefix_look``:  
  Number of tokens to look back for context.

- ``after_look``:  
  Number of tokens to look ahead for context.

- ``warmup_steps``:  
  Number of initial iterations with full diffusion before enabling smoothing.

---------------------------------------------
4. Block Diffusion (LLaDA2.0 Models)
---------------------------------------------

LLaDA2.0 models are trained with block diffusion and require special handling.

.. code-block:: python

   import torch
   from transformers import AutoConfig
   from dinfer.model import LLaDA2MoeModelLM
   from dinfer import BlockDiffusionLLM, KVCacheFactory, BlockIteratorFactory
   from dinfer import ThresholdParallelDecoder

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
       eos_id=eos_id,
   )

   cache_factory = KVCacheFactory(cache_type='prefix', is_bd_model=True)

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

   output = dllm.generate(input_ids, gen_length=2048, block_length=32)

-------------------------------------------------
5. KV Cache Strategies in dInfer
-------------------------------------------------

dInfer supports multiple KV cache strategies for efficiency:

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
       early_stop=True,
   )

**Cache type comparison:**

- ``prefix``:
  - Caches only the prompt and fixed prefix context.
  - Best for: Single-turn generation, simple prompts.
  - Memory usage: Low.

- ``dual``:
  - Caches both prefix and dynamically refreshes vicinity tokens.
  - Best for: Multi-turn generation, complex reasoning tasks.
  - Memory usage: Medium.

- ``None``:
  - No caching; recomputes everything.
  - Best for: Very short sequences, debugging scenarios.
  - Memory usage: Lowest.
