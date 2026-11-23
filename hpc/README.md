# Running the experiments on the DISI HPC cluster

The open-weight replication and the multilingual experiment run on the DISI HPC Cluster for Education: login node `giano.cs.unibo.it`, Slurm, queue `l40` (one NVIDIA L40 with 48 GB per node, 3-day job limit, about 3 running jobs per user).

The model is **Qwen3.6-27B in FP8** (about 30 GB, fits one L40), served by [vLLM](https://docs.vllm.ai) through an OpenAI-compatible API. Our scripts talk to it with `--base-url`, with the same prompts, answer schema, temperature 0 and seed 42 as gpt-4.1-mini, and with the model's thinking mode switched off (`--no-thinking`). To use another model, change `LLM_MODEL` and `LLM_DIR` in `hpc/env.sh`.

**Storage rules.** The home folder has only 400 MB, so everything goes to `/scratch.hpc/$USER` (environment, caches, data, models, results; `hpc/env.sh` sets this up). Scratch files not accessed for 40 days are deleted and there are no backups: copy the results to your laptop after every job.

| File | What it does |
|---|---|
| `hpc/env.sh` | paths and caches on scratch, model name |
| `hpc/setup.sh` | one-time setup on giano: Python environment, vLLM, data, knowledge base, models |
| `hpc/run.sbatch` | Slurm job: builds the retrieval index, starts vLLM, waits until it is ready, runs a stage, stops vLLM |

## 1. Once: set up the project on giano

From your laptop, push the latest code to GitHub first. Then, in PowerShell:

```bash
ssh alireza.shahidiani@giano.cs.unibo.it
```

On giano:

```bash
cd /scratch.hpc/$USER
git clone https://github.com/Shervinrtd/Claudette-RAG.git
cd Claudette-RAG
mkdir -p /scratch.hpc/$USER/logs
nohup bash hpc/setup.sh > /scratch.hpc/$USER/logs/setup.log 2>&1 &
tail -f /scratch.hpc/$USER/logs/setup.log          # Ctrl+C stops watching, not the setup
```

The setup takes 30-60 minutes (the model is about 30 GB) and keeps running if you close the SSH window. It ends with the tests and `Setup done`.

## 2. Pilot job (a few minutes after the model has loaded)

```bash
cd /scratch.hpc/$USER/Claudette-RAG
sbatch --time=01:00:00 hpc/run.sbatch pilot
squeue -u $USER                                      # PD = waiting for a GPU, R = running
tail -f /scratch.hpc/$USER/logs/claudette-rag-<job id>.out
```

The first start of vLLM takes about 10 minutes (loading the model and compiling GPU kernels, cached for later jobs). The pilot classifies 40 validation sentences with the plain LLM and with RAG: check that there are no parse errors and no truncation warning, and look at the seconds per request.

## 3. Full run

```bash
sbatch hpc/run.sbatch all            # baselines, english stage, then multilingual stage
```

| Stage | What runs |
|---|---|
| `english` | plain LLM, RAG and the two ablations on validation; the evidence gate tuned on validation; plain LLM, RAG and examples-only on test, with the gate; the faithfulness and stability tests |
| `multilingual` | plain LLM, RAG and the gate (with the setting chosen on validation) on the English, German, Italian and Polish evaluation sets, then the fairness analysis by language (`scripts/multilingual_fairness.py`) |
| `baselines` | free baselines on the four languages, without the LLM: TF-IDF + SVM and the kNN vote (`scripts/baselines_multilingual.py`), and Legal-BERT retrained with 3 seeds (`train_bert.py --tag hpc --multilingual`) |

Every run resumes from its log, so if a job stops (time limit, node failure), submit the same command again. If something fails, the job log shows `!! FAILED` and the next step still runs; the vLLM log is in `/scratch.hpc/$USER/logs/vllm-<job id>.log`.

## 4. Copy the results to your laptop

In PowerShell, in the project folder on your laptop:

```powershell
scp -r alireza.shahidiani@giano.cs.unibo.it:/scratch.hpc/alireza.shahidiani/Claudette-RAG/results .
```

This merges the new result files (named after the model, e.g. `rag_Qwen-Qwen3.6-27B-FP8_e8r5p6_test.json`) into your local `results` folder; then commit them from the laptop. To update the code on giano later: `git pull` in the project folder.

## Troubleshooting

- **Find the error of a failed vLLM start**: the job log only says `vLLM stopped`; the cause is the last line of the error block in the vLLM log: `grep "core.py" /scratch.hpc/$USER/logs/vllm-<job id>.log | tail -12`.
- **`max_num_seqs (256) exceeds available Mamba cache blocks`**: fixed in `run.sbatch` with `--max-num-seqs 64`. Qwen3.5/3.6 keep a state per running request and one L40 has room for about 177 of them, fewer than vLLM's default of 256.
- **"Your GPU does not have native support for FP8 computation"** (warning, not an error): this vLLM build stores the weights in FP8 but computes in 16-bit on the L40 (Marlin kernel). The answers are the same; it is only somewhat slower.
- **vLLM does not start**: read the vLLM log. If the model is not supported by the installed vLLM, set `LLM_MODEL=Qwen/Qwen3.5-27B-FP8` and `LLM_DIR=$MODELS/Qwen3.5-27B-FP8` in `hpc/env.sh`, download it on giano (`source hpc/env.sh && source $VENV/bin/activate && hf download $LLM_MODEL --local-dir $LLM_DIR`) and submit again.
- **The job waits for a long time** (`PD` with reason `Priority` or `Resources`): the L40 queue is shared with other students.
  - `sinfo -p l40` shows the state of the 4 L40 nodes, `squeue -p l40` who is using them and for how long, and `squeue -j <job id> --start` Slurm's estimate of when your job will start.
  - Ask for less time: Slurm can start a short job in a gap before a long one (backfill), so give each job only the time it needs, e.g. `sbatch --time=01:00:00 hpc/run.sbatch pilot`, `--time=12:00:00` for `english` or `multilingual`. A job that hits its time limit resumes when you submit it again.
  - The `baselines` stage needs no LLM and fits the smaller GPUs: `sbatch --partition=rtx2080 --mem=32G --time=03:00:00 hpc/run.sbatch baselines`.
- **"Disk quota exceeded" in the home folder**: something wrote a cache outside scratch; check with `du -sh ~/.cache ~/.local` and delete it.
- **The environment or the model disappeared**: scratch deletes files unused for 40 days; run `bash hpc/setup.sh` again.
