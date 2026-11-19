# Environment for the DISI HPC cluster (giano.cs.unibo.it). Source it before working:
#   source hpc/env.sh
# Home is only 400 MB, so everything large (environment, caches, data, models, results) lives on scratch.
# Scratch files not accessed for 40 days are deleted: copy your results to your laptop after each job.

export SCRATCH=/scratch.hpc/$USER
export VENV=$SCRATCH/venv-claudette
export MODELS=$SCRATCH/models
export LOGS=$SCRATCH/logs

# caches of pip, uv, Hugging Face, vLLM, Triton and PyTorch: on scratch, not in the home folder
export XDG_CACHE_HOME=$SCRATCH/cache
export HF_HOME=$SCRATCH/cache/huggingface
export UV_CACHE_DIR=$SCRATCH/cache/uv
export UV_PYTHON_INSTALL_DIR=$SCRATCH/python
export PIP_CACHE_DIR=$SCRATCH/cache/pip
export VLLM_CACHE_ROOT=$SCRATCH/cache/vllm
export TRITON_CACHE_DIR=$SCRATCH/cache/triton
export PATH=$SCRATCH/bin:$PATH

# The open-weight model served by vLLM. To replicate with another model, change these two lines
# (e.g. Qwen/Qwen3.5-27B-FP8 if the installed vLLM does not support Qwen3.6).
export LLM_MODEL=Qwen/Qwen3.6-27B-FP8
export LLM_DIR=$MODELS/Qwen3.6-27B-FP8

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$MODELS" "$LOGS" "$SCRATCH/bin"
