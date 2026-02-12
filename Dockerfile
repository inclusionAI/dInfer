# dInfer - Inference framework for diffusion LLMs
# https://github.com/inclusionAI/dInfer
#
# Build:
#   docker build -t dinfer .
#
# Run benchmark with sample prompt (single GPU):
#   docker run --gpus '"device=0"' -v /path/to/models:/models dinfer \
#     python benchmarks/benchmark.py --model_name /models/LLaDA-8B-Instruct \
#     --model_type llada --gpu 0
#
# Run benchmark (multi-GPU with tensor parallelism):
#   docker run --gpus all --ipc=host -v /path/to/models:/models dinfer \
#     python benchmarks/benchmark.py --model_name /models/LLaDA-MoE-7B-A1B-Instruct \
#     --model_type llada_moe --gpu 0,1,2,3 --use_tp
#
# Convert MoE model to FusedMoE format:
#   docker run --gpus '"device=0"' -v /path/to/models:/models dinfer \
#     python -m tools.transfer --input /models/LLaDA-MoE-7B-A1B-Instruct \
#     --output /models/LLaDA-MoE-7B-A1B-Instruct-fused
#
# Interactive Python session:
#   docker run -it --gpus all --ipc=host -v /path/to/models:/models dinfer python
#
# Note: OpenAI-compatible API serving is not currently supported.
# Use the Python API (dinfer.DiffusionLLMServing) for programmatic inference.

FROM vllm/vllm-openai:nightly
# Or pin to a version, e.g.: v0.12.0

LABEL org.opencontainers.image.source="https://github.com/inclusionAI/dInfer"
LABEL org.opencontainers.image.description="dInfer - Inference framework for diffusion LLMs"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Copy project files
COPY setup.py .
COPY python/ python/
COPY tools/ tools/
COPY benchmarks/ benchmarks/
COPY evaluations/ evaluations/

# Install dInfer without pinned dependencies (base image provides vllm)
# Then install compatible versions of remaining dependencies
RUN pip install --no-cache-dir --no-deps . && \
    pip install --no-cache-dir scipy tqdm hf_transfer sglang

# For running evaluations, install additional dependencies:
#   pip install accelerate evaluate datasets lm_eval

# Enable HuggingFace transfer for faster model downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Disable tokenizers parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Default to running a benchmark help command
CMD ["python", "benchmarks/benchmark.py", "--help"]
