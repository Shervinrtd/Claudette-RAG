# Claudette-RAG

**Explainable and fair detection of unfair clauses in Terms of Service with Retrieval-Augmented LLMs**

Project for the *Ethics in Artificial Intelligence* course, University of Bologna (A.Y. 2025/2026).

- Alireza Shahidiani (alireza.shahidiani@studio.unibo.it)
- Kiana Alian (kiana.alian@studio.unibo.it)

## Goal

Online Terms of Service (ToS) often contain clauses that are potentially unfair to consumers under EU consumer law (Directive 93/13/EEC). This project builds a Retrieval-Augmented Generation (RAG) system that detects such clauses **and explains why** they are unfair, by retrieving legal rationales and annotated examples from a structured knowledge base. The system is compared with a fine-tuned BERT classifier and with an LLM without retrieval, and evaluated for explainability, robustness and fairness across languages.

## Project structure

```
.
├── data/
│   ├── raw/              # downloaded dataset (not tracked by git)
│   └── processed/        # derived data (not tracked by git)
├── knowledge_base/       # legal knowledge base (YAML, written by hand)
├── notebooks/            # exploration and analysis notebooks
├── scripts/              # entry-point scripts (download, training, evaluation)
├── src/claudette_rag/    # project source code
├── report/               # LaTeX report (initial and final) and figures
├── results/              # predictions and evaluation reports of every system
├── tests/                # unit tests (python -m pytest)
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

Requires Python 3.10 or newer. Create a virtual environment in the project folder and install the dependencies:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Optionally activate it (`.\.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on Linux/macOS) so that `python` points to the environment. In VS Code or Jupyter, select `.venv` as the interpreter/kernel.

### GPU (for the BERT baseline)

PyTorch must be installed with CUDA support from the PyTorch index (the default `pip install torch` on Windows is CPU-only). Check the CUDA version of your NVIDIA driver with `nvidia-smi` (top right, "CUDA Version") and choose a build that is **equal or lower**:

```powershell
nvidia-smi
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130   # driver CUDA >= 13.0
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126   # driver CUDA 12.6-12.9
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data

We use the **UNFAIR-ToS** task of the LexGLUE benchmark (Chalkidis et al., 2022), which contains sentences from 50 online Terms of Service annotated by legal experts in the Claudette project (Lippi et al., 2019). Each sentence has zero or more labels among 8 categories of potentially unfair clauses; sentences without labels are considered fair.

Download it into `data/raw/` with:

```powershell
.\.venv\Scripts\python scripts\download_data.py   # Windows
```

```bash
.venv/bin/python scripts/download_data.py         # Linux / macOS
```

For the multilingual fairness experiment we also use the **multilingual Claudette corpus** (Drawzeski et al., 2021): 25 Terms of Service annotated by legal experts in English, German, Italian and Polish, with an unfairness level per sentence (clearly fair / potentially unfair / clearly unfair). License: CC BY-NC 2.5 (non-commercial use). Download it and build the evaluation set (companies that overlap with UNFAIR-ToS are excluded to avoid data leakage):

```powershell
.\.venv\Scripts\python scripts\download_multilingual.py
.\.venv\Scripts\python scripts\prepare_multilingual_eval.py
```

## Knowledge base

The legal knowledge base used by the RAG system is in `knowledge_base/` (see its README): legal provisions, one card per unfairness category, 89 legal rationales (adapted from Ruggeri et al., 2022, and written by us) and the training sentences as examples. Validate it and export it as documents with:

```powershell
.\.venv\Scripts\python scripts\build_knowledge_base.py
```

## Evaluation

Every system writes its predictions in the same format, one JSON object per line (`{"id": "test-17", "labels": [1, 3]}`, an empty list meaning *fair*), and is scored with the same code (`src/claudette_rag/evaluation.py`, tested in `tests/`):

- **LexGLUE micro-F1 and macro-F1** ("8+1": an extra *fair* label is added, as in Chalkidis et al. 2022), so our numbers are comparable with published ones;
- **macro-F1 over the 8 unfair categories**, and precision/recall/F1 per category with the number of gold examples;
- **unfair-clause detection precision and recall** (any category vs fair): for consumer protection, a missed unfair clause (low recall) is worse than a false alarm;
- **95% bootstrap confidence intervals**, because some categories have very few test examples.

```powershell
.\.venv\Scripts\python -m pytest                                   # run the tests
.\.venv\Scripts\python scripts\baseline_tfidf.py                    # always-fair and TF-IDF+SVM baselines
.\.venv\Scripts\python scripts\evaluate.py results\predictions\tfidf_svm_test.jsonl
```

## Results (UNFAIR-ToS test set)

| System | LexGLUE micro-F1 | LexGLUE macro-F1 | Unfair precision | Unfair recall |
|---|---|---|---|---|
| Always "fair" | 88.9 | 10.5 | 0.0 | 0.0 |
| TF-IDF + linear SVM (ours) | 95.4 | 78.6 | 91.5 | 69.2 |
| Plain LLM: gpt-4.1-mini, zero-shot, no retrieval (ours) | 84.0 | 64.2 | 41.3 | 93.6 |
| Legal-BERT fine-tuned, mean ± std over 3 seeds (ours) | 95.9 ± 0.4 | 82.7 ± 1.1 | 85.7 ± 4.4 | 82.0 ± 3.5 |
| TF-IDF + SVM, as reported by LexGLUE | 94.7 | 75.0 | – | – |
| Legal-BERT, as reported by LexGLUE | 96.0 | 83.0 | – | – |

A system that never flags anything already reaches 88.9 micro-F1, because 89% of the sentences are fair: micro-F1 alone is misleading, so we always report macro-F1 and the recall of unfair clauses.

The plain LLM catches almost every unfair clause (recall 93.6) but raises many false alarms (precision 41.3: 390 sentences flagged for 172 truly unfair). Most false alarms are sentences on the *topic* of an unfair clause, e.g. procedural sentences inside long arbitration sections (72 of the 77 arbitration false alarms are within ~50 sentences of a gold arbitration clause), some of which even protect the consumer. Cost of the full test run: about $0.87.

Fine-tuned Legal-BERT reproduces the published result (95.9 / 82.7 vs 96.0 / 83.0) and has the best balance between precision and recall, but it still misses about one unfair clause in five and gives no reason for its decisions. Each seed trains in 7-11 minutes on a laptop RTX 4050 (6 GB). The three baselines fail in different ways: TF-IDF misses many unfair clauses, the plain LLM raises many false alarms, and Legal-BERT is accurate but opaque. The RAG system (Phase 4) aims at LLM-level recall with fewer false alarms, and an explanation for every decision.

```powershell
.\.venv\Scripts\python scripts\run_llm.py --dry-run      # prompt and cost estimate, no API call
.\.venv\Scripts\python scripts\run_llm.py                # plain LLM baseline on the test set
```

The fine-tuned encoder baseline (Legal-BERT by default, 3 seeds, early stopping on validation macro-F1) runs on the GPU:

```powershell
.\.venv\Scripts\python scripts\train_bert.py --limit 300 --epochs 1 --seeds 1   # 1-minute smoke test
.\.venv\Scripts\python scripts\train_bert.py                                    # Legal-BERT, seeds 1 2 3
.\.venv\Scripts\python scripts\train_bert.py --model bert-base-uncased          # plain BERT
```

## Notebooks

| Notebook | Content |
|---|---|
| `notebooks/01_data_exploration.ipynb` | UNFAIR-ToS: class imbalance, categories, multi-label, lengths, leakage |
| `notebooks/02_multilingual_exploration.ipynb` | Multilingual corpus: annotation per language, unfairness levels, overlap with UNFAIR-ToS, evaluation set |

## Status

- [x] Phase 0: proposal and initial report
- [x] Phase 1: data download and exploration (`notebooks/01_data_exploration.ipynb`)
- [x] Phase 2: legal knowledge base (`knowledge_base/`, `scripts/build_knowledge_base.py`)
- [x] Phase 3: baselines (always fair, TF-IDF+SVM, plain LLM, Legal-BERT) and shared evaluation
- [ ] Phase 4: RAG pipeline
- [ ] Phase 5: evaluation (robustness, explainability)
- [ ] Phase 6: multilingual fairness (extension)
- [ ] Phase 7: demo (extension)
