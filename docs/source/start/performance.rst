.. _performance_tuning:

===========================
Performance Tuning in dInfer
===========================

Last updated: 2025-11-20

This page covers practical performance optimizations for dInfer, focusing on:

- ``torch.compile`` (kernel fusion + CUDA Graphs)
- Prompt/shape management
- Tensor parallelism (multi-GPU)
- A complete multi-GPU benchmark script

---------------------------------------------
1. ``torch.compile``: Fusion & CUDA Graphs
---------------------------------------------

PyTorch 2.0+ supports both **kernel fusion** and **CUDA Graph capture**
under ``torch.compile``. For dLLM workloads, these are very important.

.. code-block:: python

   import torch

   # Enable torch.compile on model forward pass
   model.forward = torch.compile(
       model.forward,
       mode="reduce-overhead",  # Enables CUDA Graph capture on stable regions
       fullgraph=False,         # Allow graph breaks if needed
       dynamic=True,            # Multiple shapes allowed (with care)
   )

Key points:

- Kernel fusion reduces kernel launch overhead by fusing small ops.
- ``mode="reduce-overhead"`` enables CUDA Graph capture on stabilized regions.
- Each **input signature** (shape/stride/dtype/device/guards) may trigger a new
  graph capture.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1.1 Warmup & Shape Signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- CUDA Graphs are compiled **per signature**.
- The **first iteration** for a new signature incurs extra compile/capture time.
- In dLLMs, changing **batch size**, **prompt length**, or ``block_length``
  often leads to new signatures.

Always warm up before timing:

.. code-block:: python

   # Warmup to populate CUDA Graphs (important for peak speed)
   for _ in range(2):
       _ = dllm.generate(input_ids, gen_length=256, block_length=64)

   # Now measure steady-state performance
   output = dllm.generate(input_ids, gen_length=512, block_length=64)

Practical guidance:

- **Bucket and pad** prompts to a small set of length classes
  (e.g., multiples of 16 or 32).
- Keep ``block_length`` fixed when measuring speed.
- Report both **cold-start** (first run) and **warm** (steady-state) metrics.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1.2 When to Avoid ``reduce-overhead``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- With highly dynamic shapes (ad-hoc batch sizes / prompt lengths), frequent
  recaptures may hurt performance.
- In such cases, use:

  .. code-block:: python

     model.forward = torch.compile(
         model.forward,
         mode="default",   # Fusion only, no CUDA Graphs
         fullgraph=False,
         dynamic=True,
     )

- For very short one-off runs or extremely memory-constrained GPUs, the
  overhead of capture may outweigh its benefits.

--------------------------------------
2. Tensor Parallelism (Multi-GPU)
--------------------------------------

This section shows how to set up **tensor parallelism** (TP) with vLLM.

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

       # Set device
       torch.cuda.set_device(gpu_id)
       device = torch.device(f"cuda:{gpu_id}")

       # Load tokenizer
       tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

       # Initialize vLLM distributed environment
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

       print(f"[Rank {rank}] Loading model with TP={world_size}")

       # Expert Parallelism (for MoE models)
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
               model.forward,
               mode="reduce-overhead",
               fullgraph=False,
               dynamic=True,
           )

           # Typical MoE token IDs
           mask_id, eos_id = 156895, 156892
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

           return dllm, tokenizer, device

Key points:

- Initialize vLLM distributed env **before** loading the model.
- ``distributed.initialize_model_parallel(world_size)`` sets up TP groups.
- Call ``model.tensor_parallel(tp_size)`` after loading weights.

---------------------------------------------------
3. Complete Multi-GPU Script with All Optimizations
---------------------------------------------------

Below is a **production-style benchmark script** that combines:

- vLLM distributed setup
- MoE / LLaDA / LLaDA2 models
- TP (optional)
- KV cache strategies
- Iterative smoothing
- Block diffusion
- ``torch.compile`` + CUDA Graphs
- Simple throughput metrics (FPS/TPF/TPS)

.. note::

   This script is intentionally long and intended as a reference. For real
   projects, you may want to factor pieces into modules and configuration
   files.

.. code-block:: python

   import torch
   import os
   import time
   from transformers import AutoTokenizer, AutoConfig
   from vllm import distributed
   from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
   from dinfer.model import (
       FusedOlmoeForCausalLM,
       LLaDA2MoeModelLM,
       LLaDAModelLM,
   )
   from dinfer import (
       BlockIteratorFactory,
       KVCacheFactory,
       ThresholdParallelDecoder,
       CreditThresholdParallelDecoder,
       HierarchyDecoder,
       BlockWiseDiffusionLLM,
       IterSmoothWithVicinityCacheDiffusionLLM,
       BlockDiffusionLLM,
   )


   @torch.no_grad()
   def main(world_size, rank, gpu_id, args):
       """Main inference function with all optimizations"""

       print(f"[Rank {rank}] Started with world_size={world_size}, gpu_id={gpu_id}")

       # Set device
       torch.cuda.set_device(gpu_id)
       device = torch.device(f"cuda:{gpu_id}")

       # Load tokenizer
       tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

       # Initialize vLLM distributed environment
       os.environ["MASTER_ADDR"] = "localhost"
       os.environ["MASTER_PORT"] = str(45601 + args.port_offset)

       distributed.init_distributed_environment(
           world_size=world_size,
           rank=rank,
           init_method="env://",
           local_rank=rank,
           backend="nccl",
       )
       distributed.initialize_model_parallel(args.tp_size, backend="nccl")

       print(f"[Rank {rank}] Loading model...")

       # Setup Expert Parallelism
       parallel_config = ParallelConfig(enable_expert_parallel=True)
       with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
           # Load model config
           model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

           # Load model based on type
           if args.model_type == "llada_moe":
               model = FusedOlmoeForCausalLM(config=model_config).eval()
               model.load_weights(args.model_name, torch_dtype=torch.bfloat16)
               mask_id = 156895
               eos_id = 156892
           elif args.model_type == "llada2":
               model = LLaDA2MoeModelLM(config=model_config).eval()
               model.load_weights(args.model_name, torch_dtype=torch.bfloat16, device=device)
               mask_id = 156895
               eos_id = 156892
           elif args.model_type == "llada":
               model = LLaDAModelLM.from_pretrained(
                   args.model_name,
                   torch_dtype=torch.bfloat16,
                   init_device=str(device),
               ).eval()
               mask_id = 126336
               eos_id = 126081
           else:
               raise ValueError(f"Model type {args.model_type} not supported")

           # Enable tensor parallelism
           if args.tp_size > 1 and args.use_tp:
               print(f"[Rank {rank}] Enabling TP with tp_size={args.tp_size}")
               model.tensor_parallel(args.tp_size)

           model = model.to(device)

           # Compile model for better performance
           model.forward = torch.compile(
               model.forward,
               mode="reduce-overhead",
               fullgraph=False,
               dynamic=True,
           )

           # Setup decoder
           if args.parallel_decoding == "threshold":
               if args.use_credit:
                   decoder = CreditThresholdParallelDecoder(
                       temperature=0.0,
                       threshold=args.threshold,
                       mask_id=mask_id,
                       eos_id=eos_id,
                   )
               else:
                   decoder = ThresholdParallelDecoder(
                       temperature=0.0,
                       threshold=args.threshold,
                       mask_id=mask_id,
                       eos_id=eos_id,
                   )
           else:  # hierarchy
               decoder = HierarchyDecoder(
                   temperature=0.0,
                   threshold=args.threshold,
                   low_threshold=args.low_threshold,
                   mask_id=mask_id,
                   eos_id=eos_id,
               )

           # Setup KV cache
           use_sw = (
               args.prefix_look > 0
               or args.after_look > 0
               or args.warmup_times > 0
           )

           if args.cache in ["prefix", "dual"]:
               cache_factory = KVCacheFactory(args.cache, is_bd_model=args.use_bd)
           else:
               cache_factory = None

           # Setup diffusion LLM based on configuration
           if not args.use_bd and args.cont_weight > 0 and use_sw:
               dllm = IterSmoothWithVicinityCacheDiffusionLLM(
                   model,
                   decoder,
                   BlockIteratorFactory(start_block_align=True),
                   cache_factory=cache_factory,
                   early_stop=True,
                   cont_weight=args.cont_weight,
                   prefix_look=args.prefix_look,
                   after_look=args.after_look,
                   warmup_steps=args.warmup_times,
               )
           elif not args.use_bd and args.cont_weight == 0 and not use_sw:
               dllm = BlockWiseDiffusionLLM(
                   model,
                   decoder,
                   BlockIteratorFactory(start_block_align=True),
                   cache_factory=cache_factory,
                   early_stop=True,
               )
           elif args.use_bd:
               dllm = BlockDiffusionLLM(
                   model,
                   decoder,
                   BlockIteratorFactory(
                       start_block_align=True,
                       use_block_diffusion=True,
                   ),
                   cache_factory=cache_factory,
                   early_stop=True,
               )
           else:
               raise ValueError("Invalid configuration")

           # Warmup for CUDA graph compilation
           prompt = (
               "Lily can run 12 kilometers per hour for 4 hours. "
               "After that, she can run 6 kilometers per hour. "
               "How many kilometers can she run in 8 hours?"
           )
           input_ids = tokenizer(prompt)["input_ids"]
           input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

           print(f"[Rank {rank}] Warming up...")
           for _ in range(2):
               _ = dllm.generate(
                   input_ids,
                   gen_length=args.gen_len,
                   block_length=args.block_length,
               )

           # Actual inference with timing
           print(f"[Rank {rank}] Running inference...")
           prev_forwards = dllm.num_forwards

           torch.cuda.synchronize()
           inner_start = time.time()

           output = dllm.generate(
               input_ids,
               gen_length=args.gen_len,
               block_length=args.block_length,
           )

           torch.cuda.synchronize()
           inner_stop = time.time()

           sample_time = inner_stop - inner_start
           nfe = dllm.num_forwards - prev_forwards

           token_number = int((output != eos_id).sum() - input_ids.shape[1])
           tpf = token_number / nfe
           tps = token_number / sample_time
           fps = nfe / sample_time

           if rank == 0:
               print(f"\n{'=' * 60}")
               print("Performance Metrics:")
               print(f"  NFE (iterations):   {nfe:4d}")
               print(f"  Tokens generated:   {token_number:4d}")
               print(f"  FPS:                {fps:4.2f}")
               print(f"  TPF:                {tpf:2.2f}")
               print(f"  TPS:                {tps:4.2f}")
               print(f"{'=' * 60}\n")
               print(
                   "Generated text:\n"
                   f"{tokenizer.decode(output[0], skip_special_tokens=True)}"
               )


   def process_args(args):
       """Process and validate arguments"""
       import warnings

       gpus = [int(gpu) for gpu in args.gpu.split(",")]

       if len(gpus) > 1 and not args.use_tp:
           warnings.warn(
               "Using multiple GPUs without tensor parallelism. "
               "TP will be enabled."
           )
           args.use_tp = True
       elif len(gpus) == 1 and args.use_tp:
           warnings.warn(
               "Using tensor parallelism with only one GPU. "
               "TP will be disabled."
           )
           args.use_tp = False

       if args.model_type == "llada2" and not args.use_bd:
           warnings.warn("Using llada2 without block diffusion is not recommended.")

       if args.model_type == "llada2" and args.cache == "":
           warnings.warn(
               "Using llada2 without kvcache. Cache will be set to prefix."
           )
           args.cache = "prefix"

       args.tp_size = len(gpus)
       args.port_offset = gpus[0]

       return args


   if __name__ == "__main__":
       from multiprocessing import Process
       import argparse

       torch.multiprocessing.set_start_method("spawn")

       parser = argparse.ArgumentParser(description="dInfer Benchmark")
       parser.add_argument("--model_name", type=str, required=True,
                           help="Path or name of the model")
       parser.add_argument("--model_type", type=str, default="llada_moe",
                           choices=["llada", "llada_moe", "llada2"],
                           help="Type of model to use")
       parser.add_argument("--gpu", type=str, default="0",
                           help='Comma-separated GPU IDs (e.g., "0,1,2,3")')
       parser.add_argument("--gen_len", type=int, default=1024,
                           help="Maximum generation length")
       parser.add_argument("--block_length", type=int, default=64,
                           help="Block length for parallel decoding")
       parser.add_argument("--threshold", type=float, default=0.9,
                           help="Confidence threshold for token acceptance")
       parser.add_argument("--low_threshold", type=float, default=0.3,
                           help="Low confidence threshold (for hierarchy decoder)")
       parser.add_argument("--parallel_decoding", type=str, default="threshold",
                           choices=["threshold", "hierarchy"],
                           help="Parallel decoding strategy")
       parser.add_argument("--use_credit", action="store_true",
                           help="Use credit-based threshold decoder")
       parser.add_argument("--cache", type=str, default="",
                           choices=["", "prefix", "dual"],
                           help="KV cache strategy")
       parser.add_argument("--use_tp", action="store_true",
                           help="Enable tensor parallelism")
       parser.add_argument("--use_bd", action="store_true",
                           help="Use block diffusion (for LLaDA2)")
       parser.add_argument("--cont_weight", type=float, default=0.0,
                           help="Continuity weight for iterative smoothing")
       parser.add_argument("--prefix_look", type=int, default=0,
                           help="Prefix look-back size")
       parser.add_argument("--after_look", type=int, default=0,
                           help="After look-ahead size")
       parser.add_argument("--warmup_times", type=int, default=0,
                           help="Number of warmup steps")
       parser.add_argument("--use_shift", action="store_true",
                           help="Use shift in block-wise diffusion")

       args = parser.parse_args()
       args = process_args(args)

       print(f"Configuration: {args}")

       gpus = [int(gpu) for gpu in args.gpu.split(",")]

       if len(gpus) == 1:
           # Single GPU
           main(1, 0, gpus[0], args)
       else:
           # Multi-GPU
           processes = []
           for i, gpu_id in enumerate(gpus):
               p = Process(target=main, args=(len(gpus), i, gpu_id, args))
               p.daemon = True
               processes.append(p)
               p.start()

           for p in processes:
               p.join()
