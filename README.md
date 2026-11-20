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
├── hpc/                  # scripts for the DISI GPU cluster (open-weight model with vLLM)
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

### Open-weight models: your machine or a GPU cluster (optional)

Every LLM script (`run_llm.py`, `run_rag.py`, `faithfulness.py`) can also use an open-weight model behind any OpenAI-compatible server, with `--base-url`: the prompts, answer schema, temperature and seed stay the same, only the model changes, and there is no API cost. Runs are saved under the model's name (e.g. `rag_Qwen-Qwen3.5-27B_e8r5p6_test.json`) and record the time per request instead of a cost.

On a GPU cluster, serve the model with [vLLM](https://docs.vllm.ai) (fast, many requests in parallel) and switch off the model's thinking mode, so that it answers directly like gpt-4.1-mini. For the DISI cluster, `hpc/` contains the complete setup and Slurm job (see `hpc/README.md`); in general:

```bash
vllm serve <model> --port 8000 --max-model-len 8192 &
python scripts/run_rag.py --model <model> --base-url http://localhost:8000/v1 --no-thinking --workers 32
```

On a laptop, [Ollama](https://ollama.com) works the same way. Our prompts are about 2,600 tokens, so raise Ollama's context length first, otherwise it may silently cut the prompts (the scripts print a warning if this happens):

```powershell
setx OLLAMA_CONTEXT_LENGTH 8192          # then quit Ollama from the tray icon and start it again
ollama pull qwen3:8b
.\.venv\Scripts\python scripts\run_rag.py --model qwen3:8b --base-url http://localhost:11434/v1 --sample 40
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
| kNN vote over retrieved training sentences, bge-m3, k=3, t=2 (ours) | 94.8 | 77.8 | 78.0 | 82.6 |
| Plain LLM: gpt-4.1-mini, zero-shot, no retrieval (ours) | 84.0 | 64.2 | 41.3 | 93.6 |
| Legal-BERT fine-tuned, mean ± std over 3 seeds (ours) | 95.9 ± 0.4 | 82.7 ± 1.1 | 85.7 ± 4.4 | 82.0 ± 3.5 |
| RAG: gpt-4.1-mini + examples + legal knowledge, no gate (ours) | 90.2 | 71.3 | 55.0 | 93.0 |
| RAG examples only + evidence gate (ours, secondary) | 95.6 | 78.9 | 83.7 | 80.8 |
| **RAG + evidence gate (ours, main system)** | **95.6** | **79.4** | **82.3** | **83.7** |
| TF-IDF + SVM, as reported by LexGLUE | 94.7 | 75.0 | – | – |
| Legal-BERT, as reported by LexGLUE | 96.0 | 83.0 | – | – |

A system that never flags anything already reaches 88.9 micro-F1, because 89% of the sentences are fair: micro-F1 alone is misleading, so we always report macro-F1 and the recall of unfair clauses.

The plain LLM catches almost every unfair clause (recall 93.6) but raises many false alarms (precision 41.3: 390 sentences flagged for 172 truly unfair). Most false alarms are sentences on the *topic* of an unfair clause, e.g. procedural sentences inside long arbitration sections (72 of the 77 arbitration false alarms are within ~50 sentences of a gold arbitration clause), some of which even protect the consumer. Cost of the full test run: about $0.87.

Fine-tuned Legal-BERT reproduces the published result (95.9 / 82.7 vs 96.0 / 83.0) and has the best balance between precision and recall, but it still misses about one unfair clause in five and gives no reason for its decisions. Each seed trains in 7-11 minutes on a laptop RTX 4050 (6 GB). The three baselines fail in different ways: TF-IDF misses many unfair clauses, the plain LLM raises many false alarms, and Legal-BERT is accurate but opaque. The RAG system (Phase 4) aims at LLM-level recall with fewer false alarms, and an explanation for every decision.

**Main system (Phase 4).** The RAG system with the evidence gate, with every choice made on validation and the test set used once, reaches 79.4 macro-F1 with 82.3 precision and 83.7 recall, without training the LLM. To know which differences are real, `scripts/compare_systems.py` runs a paired bootstrap (1,000 resamples of the same test sentences):

| Main system minus... | Macro-F1 | Unfair precision | Unfair recall |
|---|---|---|---|
| Legal-BERT (seeds 1 / 2 / 3) | −2.5 / −4.6 / −2.8, none significant | +0.4 / −8.3* / −2.4 | −0.6 / +5.8 / 0.0 |
| TF-IDF + SVM | +0.8 | −9.3* | **+14.5*** |
| kNN vote | +1.6 | +4.3 | +1.2 |
| Plain LLM | **+15.2*** | **+41.0*** | −9.9* |
| RAG without the gate | **+8.1*** | **+27.3*** | −9.3* |
| RAG examples only + gate | +0.5 | −1.4 | +2.9* |

\* significant at 5% (the 95% interval of the difference does not contain 0).

In accuracy, the main system is statistically on par with fine-tuned Legal-BERT, TF-IDF and the kNN vote: Legal-BERT keeps a lead of 2.5-4.6 macro-F1 points, which this test set (7 to 38 examples per category) cannot confirm. It is clearly better than the plain LLM and than RAG without the gate, and it finds 14.5 points more unfair clauses than TF-IDF. What it adds over all of them is the explanation: every flagged clause comes with a reason, 95.4% cite a specific legal rationale or provision (65.7% a specific article, Annex point or court case, such as Annex 1(j)-(k) of Directive 93/13 for unilateral changes or CJEU Océano for jurisdiction clauses), 77.7% cite the expert-labelled precedents they resemble, and no cited id was invented. The legal knowledge also adds recall over the examples-only system (+2.9, significant). Cost of the test run: $1.64 (main) and $1.29 (examples only).

```powershell
.\.venv\Scripts\python scripts\compare_systems.py      # paired bootstrap: main system vs every other system
```

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

## Retrieval (Phase 4)

For each sentence to classify, the retriever (`src/claudette_rag/retrieval.py`) finds in the knowledge base:

- the **8 most similar training sentences** with their gold labels (fair and unfair), as worked examples;
- the **5 most similar legal rationales** (why a kind of clause is or is not unfair);
- the **legal provisions** those rationales are based on (at most 6), so every explanation can cite the law.

Sentences and documents are embedded with the multilingual model **BAAI/bge-m3** (the same index serves the German, Italian and Polish experiment of Phase 6) and searched with exact cosine similarity, which is fast enough for 5,584 documents without a vector database. Build the index once (downloads about 2.3 GB the first time; a few minutes on the GPU), then measure retrieval quality on the validation set:

```powershell
.\.venv\Scripts\python scripts\build_index.py        # -> data/processed/index/bge-m3/
.\.venv\Scripts\python scripts\eval_retrieval.py     # hit@k, purity, kNN baseline
```

`eval_retrieval.py` reports, for k = 1 to 10, how often at least one retrieved example (or rationale) has the category of an unfair sentence, and how clean the examples retrieved for fair sentences are. It also scores a **kNN baseline** that needs no LLM: a category is predicted when at least t of the k nearest training sentences have it, with k and t chosen on validation and scored once on test.

Retrieval quality with bge-m3 on the validation set (230 unfair and 2,045 fair sentences):

| k | Example hit@k | Rationale hit@k | Fair purity@k |
|---|---|---|---|
| 1 | 72.6 | 78.7 | 95.9 |
| 3 | 90.9 | 94.3 | 94.2 |
| 5 | 94.8 | 96.5 | 93.7 |
| 8 | 97.4 | 97.8 | 92.7 |

With 8 examples and 5 rationales, 97% of the unfair sentences receive at least one example and one rationale of their own category, so the evidence the LLM needs is almost always in its context. For fair sentences about 93% of the retrieved examples are fair, which gives the LLM evidence against false alarms. The retrieved examples alone already carry a strong signal: the kNN vote (k=3, t=2), with no training and no LLM, scores 77.8 macro-F1 on test, close to TF-IDF+SVM, with the recall of Legal-BERT (82.6). Its weak point is arbitration (precision 46.7), the same category where the plain LLM raises most false alarms: sentences of an arbitration section look alike, whether they are unfair or not.

## RAG system (Phase 4)

The RAG system (`src/claudette_rag/rag.py`, `scripts/run_rag.py`) uses the same model, settings, task description and category cards as the plain LLM baseline; the only difference is the retrieved knowledge in the prompt: before each sentence, the 8 most similar training sentences with their expert labels, 5 legal rationales and the linked legal provisions, each with an id. The answer must list in `cited_ids` the retrieved items the decision relies on, and the citations are checked against what was actually retrieved. Every answer is saved with its explanation, citations and retrieved ids in `results/explanations/`.

All choices are made on the validation set; the test set is used once, at the end.

```powershell
.\.venv\Scripts\python scripts\run_rag.py --dry-run         # one real prompt and the cost estimate, no API call
.\.venv\Scripts\python scripts\run_rag.py --sample 40       # 20 unfair + 20 fair validation sentences
.\.venv\Scripts\python scripts\run_rag.py                   # full validation set (about $2.30, 30 minutes)
.\.venv\Scripts\python scripts\gate_rag.py rag_gpt-4.1-mini-2025-04-14_e8r5p6_validation   # evidence gate, free
```

Results on the **validation** set (2,275 sentences, 230 unfair), gpt-4.1-mini:

| System | LexGLUE micro-F1 | LexGLUE macro-F1 | Unfair precision | Unfair recall | Sentences flagged |
|---|---|---|---|---|---|
| Plain LLM (no retrieval) | 83.6 | 60.8 | 38.5 | 92.2 | 550 |
| kNN vote (k=3, t=2), no LLM | 94.7 | 76.0 | 76.3 | 77.0 | 232 |
| RAG | 88.9 | 65.7 | 50.2 | 92.2 | 422 |
| RAG + evidence gate (k=8, m=3) | 94.9 | 77.5 | 75.7 | 82.6 | 251 |

Retrieval helps the LLM: with the same recall, it removes 128 of the 338 false alarms of the plain LLM (arbitration precision rises from 9.0 to 20.5), and 99.8% of its unfair verdicts cite a retrieved rationale or provision, with no invented ids. But the LLM still flags many sentences on their topic and ignores the evidence: of the 210 fair sentences it flags, 132 have at least 6 of their 8 most similar sentences labelled fair by the experts, and 44 have all 8.

The **evidence gate** turns this observation into a transparent rule: a category proposed by the LLM is kept only if at least m of the k most similar expert-labelled sentences have it, so that every flagged clause has both a legal explanation and precedents. (k, m) was chosen on validation by a rule fixed in advance: the best macro-F1 among the settings with unfair recall of at least 80%, since for consumer protection a missed unfair clause is worse than a false alarm. The gate un-flags 171 sentences: 149 false alarms, but also 22 truly unfair clauses (recall 92.2 to 82.6), a trade-off we discuss in the ethical analysis. Validation numbers of the gated system are optimistic because (k, m) was chosen on them; the test set gives the unbiased comparison.

### Ablation: which knowledge helps? (validation)

| Knowledge in the prompt | Macro-F1 | Unfair precision | Unfair recall | Flagged | False alarms | With evidence gate (k=8, m=3): macro-F1 / P / R |
|---|---|---|---|---|---|---|
| None (plain LLM) | 60.8 | 38.5 | 92.2 | 550 | 338 | – |
| Legal only: rationales + provisions | 52.9 | 31.1 | 97.0 | 716 | 493 | – |
| Examples only | 71.5 | 61.1 | 87.4 | 329 | 128 | 78.4 / 78.5 / 80.9 |
| Examples + legal (full RAG) | 65.7 | 50.2 | 92.2 | 422 | 210 | 77.5 / 75.7 / 82.6 |

```powershell
.\.venv\Scripts\python scripts\run_rag.py --k-rationales 0 --max-provisions 0   # examples only
.\.venv\Scripts\python scripts\run_rag.py --k-examples 0                        # legal knowledge only
```

The useful signal comes from the expert-labelled examples. Legal knowledge on its own makes the LLM flag even more sentences than with no retrieval (493 false alarms instead of 338, mostly limitation of liability and arbitration): reading descriptions of unfair clauses primes the model to find them everywhere. Added to the examples it still costs precision (210 false alarms instead of 128). But the legal knowledge is what grounds the explanations in the law: in the full RAG, 95.5% of the unfair verdicts cite a specific rationale or provision (not only the general test of Article 3(1)), while without it an explanation can only point to similar sentences. After the evidence gate the two systems are within one point of macro-F1, well inside the confidence intervals. We therefore keep **full RAG + evidence gate** as the main system, declared before the test run because its explanations are grounded in the law, and report examples only + gate as a secondary system.

Citations that do not match a retrieved item are rare: 0 in the full RAG, 2 with examples only (a category code written as an id, and one example id that was not retrieved), 3 with legal knowledge only (category codes written as ids).

## Evaluation of the explanations (Phase 5)

All analyses use the main system on the test set (175 flagged clauses, 187 flagged categories).

```powershell
.\.venv\Scripts\python scripts\explanation_quality.py    # are the citations consistent with the verdict? (free)
.\.venv\Scripts\python scripts\error_analysis.py         # error types, with neighbouring sentences (free)
.\.venv\Scripts\python scripts\faithfulness.py           # evidence removal and stability tests (about $0.80)
```

### Grounding of the explanations (automatic)

| For each flagged category, the explanation... | Correct category (145) | Wrong category (42) |
|---|---|---|
| cites a legal rationale or provision | 100% | 100% |
| ... other than the general test of Article 3(1) | 96% | 95% |
| cites legal evidence consistent with the category (a rationale of that category or a provision linked to it) | 95% | 81% |
| cites an expert-labelled example with that category | 79% | 62% |
| cites a "clearly fair" rationale to support an unfair verdict | 1% | 0% |
| cites an id that was not retrieved | 0% | 0% |

The explanations are grounded: every flag cites legal sources, almost always consistent with the category and never invented. But **grounding is not correctness**: wrong flags are justified almost as convincingly as correct ones. Without the gate the gap is even smaller: 87% of the wrong flags of the plain RAG cite legal evidence consistent with the category they claim, against 95% of the correct ones. What separates right from wrong is the expert precedents (76% vs 35% of flags cite a matching expert example), which is exactly what the evidence gate uses. A convincing legal explanation is therefore not a guarantee that the verdict is right: a risk of automation bias that we discuss in the ethical analysis.

### Error analysis

`error_analysis.py` classifies the 74 test sentences where the main system disagrees with the experts and writes them, with two sentences of context on each side, to `results/error_analysis_<system>.csv`. A first reading of all 74 errors gives these patterns (the `manual_type` column of the CSV is for checking them):

- **False alarms (31).**
  - 6 are near misses: the system flagged a sentence next to the expert-labelled clause of the same category (for example the list of reasons that follows a termination clause).
  - 7 are summaries, procedural details or waivers inside an arbitration section ("please note that these terms contain a mandatory arbitration provision...").
  - 4 treat a reference to other policies ("you agree to our commerce terms") as *contract by using*.
  - 3 miss a legal condition: termination for the consumer's own non-payment, or arbitration limited to business users.
  - 2 are trivial mentions of a change.
  - 9 look unfair under the knowledge-base criteria even though the experts labelled them fair, for example "content that violates any of ebay's policies may be deleted at ebay's discretion". These are possible annotation inconsistencies.
- **Missed clauses (28).**
  - 14 were found by the LLM but removed by the gate because they had few expert precedents. 5 of them exclude liability for third-party sites or content, and 2 bind the user to the privacy policy by mere use.
  - 2 were given the wrong category and then removed by the gate.
  - 12 were judged fair by the LLM: 6 are clear errors, for example a one-year limit for bringing claims judged "common and generally accepted". 3 are notification sentences that belong to a unilateral-change clause. 3 are labelled unfair by the experts although the clause is mutual or favours the consumer: "you may terminate your duolingo account at any time", or Brazilian law for residents of Brazil.
- **Wrong category (15).** Mostly *unilateral termination* versus *unilateral change* on sentences that let the provider "limit", "modify or discontinue" a service, where the two categories overlap.

UNFAIR-ToS also contains duplicate sentences in the test set (e.g. test-612/616 and test-613/617).

### Faithfulness and stability

`faithfulness.py` sends again the prompt of every sentence the RAG judged unfair, without the items its explanation cites, and, as a control, without the same number of retrieved items of the same types that were not cited. If the citations are the real reasons of the decision, removing them must change the verdict much more often than the control. It also repeats 300 unchanged prompts to measure how stable the answers are.

| Sentences judged unfair by the RAG (with citations) | Remove the cited items: verdict flips to fair | Remove as many uncited items (control) | Exact McNemar test |
|---|---|---|---|
| All (291) | 27.8% | 8.2% | p = 1e-12 |
| Control with exactly the same mix of types (225) | 30.2% | 9.3% | p = 6e-11 |
| Correct category (157) | 10.8% | 2.5% | p = 0.001 |
| Wrong flag (134) | 47.8% | 14.9% | p = 1e-9 |

- **The citations are faithful.** Removing the cited items changes the verdict three times more often than removing the same number of uncited items of the same types: the explanation points to evidence the decision really depends on.
- **The citations are not the whole reason.** In 64% of the cases the labels survive the removal of their citations, because the model also reads the sentence itself.
- **Correct flags are robust.** Only 10.8% flip without their citations: the model recognises these clauses largely on its own.
- **Wrong flags depend on the evidence.** Half of them (64 of 134) disappear when their cited items are removed, and all 64 are fair for the experts. In these cases the retrieved evidence talked the model into a false alarm, the same priming effect seen in the legal-only ablation.
- **The answers are stable.** The same prompt gives the same labels in 98.3% of the 300 repeats. The 5 differences are all borderline fair/unfair decisions, which shows that temperature 0 and a fixed seed do not make the API fully deterministic.
- **Cost:** about $0.80.

### Limits of this evaluation

The automatic checks show that the explanations are grounded (they cite retrieved legal sources consistent with the verdict, and never invent one) and faithful (the cited evidence really drives the decision). They do not show that the legal reasoning is correct: that would need a review of the explanations by legal experts, which is outside the scope of this project. Grounding and faithfulness are necessary for a trustworthy explanation, but not sufficient, as the convincingly cited false alarms above show.

## Multilingual fairness (Phase 6)

Does the system protect consumers equally in every language? The groups are the languages of the multilingual evaluation set: the Terms of Service of the same 10 companies in English, German, Italian and Polish, annotated by legal experts. English, the language of the training data and of the knowledge base, is the reference group.

Classifying about 10,000 sentences per system with gpt-4.1-mini was beyond our budget, so the whole pipeline is replicated with an open-weight model, Qwen3.6-27B (FP8), on one GPU of the DISI cluster (see `hpc/README.md`), and all cross-language comparisons use this single model. The gpt-4.1-mini results above remain our main results on the English benchmark; the replication also shows which findings hold across models.

`scripts/multilingual_fairness.py` reports, for each system and language, the criteria of group fairness seen in the course: recall of unfair clauses (equality of opportunity), false-alarm rate (equal false positive rates; together: equalized odds), precision (conditional use error), share of sentences flagged (statistical parity) and macro-F1. Each language is compared with English through the ratio of the rates, with the AEQUITAS convention that a ratio between 0.8 and 1.25 counts as parity, and through the difference, with a 95% bootstrap interval.

```bash
sbatch hpc/run.sbatch all                                            # on the cluster: everything
python scripts/baselines_multilingual.py                             # TF-IDF and kNN on the four languages
python scripts/train_bert.py --tag hpc --multilingual                # Legal-BERT, 3 seeds, four languages
python scripts/multilingual_fairness.py --model Qwen/Qwen3.6-27B-FP8 # fairness by language
```

Results: to be added.

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
- [x] Phase 4: RAG pipeline (retriever, kNN baseline, RAG with cited explanations, evidence gate, ablations, final test run, significance tests)
- [x] Phase 5: evaluation of the explanations (grounding, error analysis, faithfulness, stability)
- [ ] Phase 6: multilingual fairness (extension; in progress: open-weight replication on the DISI cluster)
- [ ] Phase 7: demo (extension)
