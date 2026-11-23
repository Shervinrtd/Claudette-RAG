#!/bin/bash
#SBATCH --job-name=claudette-rag
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch.hpc/%u/logs/%x-%j.out
#
# Runs the experiments with an open-weight model served by vLLM on one L40 GPU (48 GB).
# Submit from the project folder on giano:
#   sbatch hpc/run.sbatch pilot          # 40 sentences, plain LLM + RAG: checks speed and answers (minutes)
#   sbatch hpc/run.sbatch english        # validation + ablations + gate + test + faithfulness
#   sbatch hpc/run.sbatch multilingual   # plain LLM + RAG + gate on en/de/it/pl (needs the english stage first)
#   sbatch hpc/run.sbatch baselines      # free baselines on en/de/it/pl: TF-IDF, kNN, Legal-BERT (no LLM)
#   sbatch hpc/run.sbatch all            # baselines, english, then multilingual
# Every run resumes from its log: if the job stops (time limit, error), submit it again.
# A shorter --time on the command line helps the job start sooner (Slurm can fit it into a gap):
#   sbatch --time=01:00:00 hpc/run.sbatch pilot
# The baselines need no LLM and also run on the smaller GPUs:
#   sbatch --partition=rtx2080 --mem=32G --time=03:00:00 hpc/run.sbatch baselines
set -uo pipefail
STAGE=${1:-pilot}
cd "$SLURM_SUBMIT_DIR" || exit 1
source hpc/env.sh
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # GPU nodes may have no internet: use the downloaded files
echo "Job $SLURM_JOB_ID on $(hostname), stage $STAGE, model $LLM_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv

step() { echo; echo "== $(date +%H:%M) $*"; "$@" || echo "!! FAILED: $*"; }

# retrieval index (built before vLLM takes the GPU)
[ -f data/processed/index/bge-m3/meta.json ] || step python scripts/build_index.py

baselines() {                                    # GPU work that must run before vLLM takes the GPU
    step python scripts/baselines_multilingual.py
    step python scripts/train_bert.py --tag hpc --multilingual
}
case "$STAGE" in
    baselines) baselines; echo; echo "== $(date +%H:%M) stage baselines finished."; exit 0 ;;
    all) baselines ;;
esac

# start the model server; thinking is switched off per request (--no-thinking), the reasoning
# parser only keeps any stray thinking text out of the answer. --max-num-seqs: Qwen3.5/3.6 keep a
# per-request state (Mamba cache) and on one L40 there is room for about 177 of them, fewer than
# vLLM's default of 256; our scripts send at most 32 requests at a time.
vllm serve "$LLM_DIR" --served-model-name "$LLM_MODEL" --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching --max-num-seqs 64 \
    --language-model-only --reasoning-parser qwen3 > "$LOGS/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
ready() { python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=5)" 2>/dev/null; }
for _ in $(seq 1 120); do                        # wait up to 30 minutes (first start compiles kernels)
    ready && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM stopped: see $LOGS/vllm-$SLURM_JOB_ID.log"; exit 1; }
    sleep 15
done
ready || { echo "vLLM did not start within 30 minutes: see $LOGS/vllm-$SLURM_JOB_ID.log"; exit 1; }
echo "vLLM is ready"

LLM="--model $LLM_MODEL --base-url http://localhost:8000/v1 --no-thinking --workers 32"
M=$(python -c "from claudette_rag.llm import file_safe; print(file_safe('$LLM_MODEL'))")

pilot() {
    step python scripts/run_llm.py $LLM --split validation --sample 40
    step python scripts/run_rag.py $LLM --sample 40
}
english() {
    step python scripts/run_llm.py $LLM --split validation
    step python scripts/run_rag.py $LLM
    step python scripts/run_rag.py $LLM --k-rationales 0 --max-provisions 0
    step python scripts/run_rag.py $LLM --k-examples 0
    step python scripts/gate_rag.py "rag_${M}_e8r5p6_validation"
    step python scripts/gate_rag.py "rag_${M}_e8r0p0_validation"
    step python scripts/run_llm.py $LLM --split test
    step python scripts/run_rag.py $LLM --split test
    step python scripts/run_rag.py $LLM --split test --k-rationales 0 --max-provisions 0
    step python scripts/gate_rag.py "rag_${M}_e8r5p6_test"
    step python scripts/gate_rag.py "rag_${M}_e8r0p0_test"
    step python scripts/faithfulness.py --run "rag_${M}_e8r5p6_test" --workers 32
}
multilingual() {
    for lang in en de it pl; do
        step python scripts/run_llm.py $LLM --dataset multilingual --language $lang
        step python scripts/run_rag.py $LLM --dataset multilingual --language $lang
        step python scripts/gate_rag.py "rag_${M}_e8r5p6_multilingual_$lang"
    done
    step python scripts/multilingual_fairness.py --model "$LLM_MODEL"
}
case "$STAGE" in
    pilot) pilot ;;
    english) english ;;
    multilingual) multilingual ;;
    all) english; multilingual ;;
    *) echo "unknown stage '$STAGE' (pilot, english, multilingual, baselines, all)"; exit 1 ;;
esac
echo; echo "== $(date +%H:%M) stage $STAGE finished. Copy results/ to your laptop (see hpc/README.md)."
