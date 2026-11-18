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
├── notebooks/            # exploration and analysis notebooks
├── scripts/              # entry-point scripts (download, training, evaluation)
├── src/claudette_rag/    # project source code
├── report/               # LaTeX report (initial and final) and figures
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

## Notebooks

| Notebook | Content |
|---|---|
| `notebooks/01_data_exploration.ipynb` | UNFAIR-ToS: class imbalance, categories, multi-label, lengths, leakage |
| `notebooks/02_multilingual_exploration.ipynb` | Multilingual corpus: annotation per language, unfairness levels, overlap with UNFAIR-ToS, evaluation set |

## Status

- [x] Phase 0: proposal and initial report
- [x] Phase 1: data download and exploration (`notebooks/01_data_exploration.ipynb`)
- [ ] Phase 2: legal knowledge base
- [ ] Phase 3: baselines (BERT, LLM without retrieval)
- [ ] Phase 4: RAG pipeline
- [ ] Phase 5: evaluation (robustness, explainability)
- [ ] Phase 6: multilingual fairness (extension)
- [ ] Phase 7: demo (extension)
