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
├── report/               # LaTeX report (initial and final) and figures
├── .gitignore
└── README.md
```

Code, data scripts and notebooks will be added phase by phase.

## Status

- [x] Phase 0: proposal and initial report
- [ ] Phase 1: data download and exploration
- [ ] Phase 2: legal knowledge base
- [ ] Phase 3: baselines (BERT, LLM without retrieval)
- [ ] Phase 4: RAG pipeline
- [ ] Phase 5: evaluation (robustness, explainability)
- [ ] Phase 6: multilingual fairness (extension)
- [ ] Phase 7: demo (extension)
