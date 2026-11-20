#!/bin/bash
# One-time setup on giano, the login node (it has internet; the GPU nodes may not).
# Run it from the project folder on scratch:
#   cd /scratch.hpc/$USER/Claudette-RAG && bash hpc/setup.sh
# It takes 30-60 minutes, mostly to download the model (about 30 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh

echo "== 1/4 Python environment"
if ! command -v uv >/dev/null; then          # uv: a fast Python and package installer, installed on scratch
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCRATCH/bin" UV_NO_MODIFY_PATH=1 sh
fi
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"

echo "== 2/4 vLLM (with PyTorch for CUDA) and the project's dependencies"
uv pip install vllm
uv pip install -r requirements.txt

echo "== 3/4 Data, knowledge base and multilingual evaluation set"
python scripts/download_data.py
python scripts/download_multilingual.py
python scripts/prepare_multilingual_eval.py
python scripts/build_knowledge_base.py

echo "== 4/4 Models: $LLM_MODEL, the retrieval embedder BAAI/bge-m3 and Legal-BERT"
hf download "$LLM_MODEL" --local-dir "$LLM_DIR"
hf download nlpaueb/legal-bert-base-uncased
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', device='cpu')"

python -m pytest -q
echo "Setup done. Next: sbatch hpc/run.sbatch pilot"
