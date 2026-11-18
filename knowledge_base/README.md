# Legal knowledge base

This folder is the "library" that the RAG system searches before deciding whether a Terms of Service clause is potentially unfair. It is written by hand (YAML files) and assembled by `scripts/build_knowledge_base.py`, which also checks it for errors.

## Files

| File | What it contains | Written by |
|---|---|---|
| `sources.yaml` | Bibliography: every source the knowledge base relies on | us |
| `legal_provisions.yaml` | The legal rules (Directive 93/13/EEC, Brussels Ia, Rome I, CJEU judgments), summarised in our own words with the exact reference and an official link | us |
| `categories.yaml` | One card per unfairness category: definition, when it is clearly fair / potentially unfair / clearly unfair, why it can harm the consumer, legal basis | us, based on Lippi et al. (2019) and the legal sources |
| `rationales.yaml` | Legal rationales: the specific practices that make a clause potentially or clearly unfair (and, for some categories, what makes it fair) | 5 categories adapted from Ruggeri et al. (2022); 3 categories written by us |

The **examples** (annotated sentences) are not written by hand: the build script takes them from the UNFAIR-ToS **training set only**, and removes any training sentence that also appears in the validation or test set, so the retriever can never find a test sentence with its answer.

## Four kinds of documents

When the build script runs, every entry becomes a document in `data/processed/kb_documents.jsonl`:

```
provision  ->  "Annex 1(g): the provider may not end an open-ended contract without reasonable notice..."
category   ->  "Unilateral termination: the provider's right to suspend or terminate... clearly unfair if..."
rationale  ->  "ter.no_notice: termination without notice -> clearly unfair, because..."
example    ->  "we may terminate your account at any time ."  labels: [Unilateral termination]
```

The retriever searches all of them; the LLM receives the clause plus the retrieved documents and must cite the ids it relied on (e.g. `ter.no_notice`, `dir93.annex.1g`), which makes every explanation checkable.

## Unfairness levels

Following the Claudette annotation scheme (Lippi et al., 2019):

- **clearly fair**: the clause belongs to a category but protects the consumer (e.g. disputes can be brought where the consumer lives);
- **potentially unfair**: the clause may cause a significant imbalance, depending on the circumstances;
- **clearly unfair**: the clause is unfair under the criteria of the legal sources (e.g. liability excluded for personal injury).

Note: UNFAIR-ToS only says *whether* a sentence is potentially unfair and in which category; the levels are used here to write better explanations.

## How to edit

- Every entry needs a unique `id`. Rationale ids start with the category code (`ltd.`, `ter.`, `ch.`, `cr.`, `use.`, `law.`, `j.`, `a.`).
- `legal_basis` and `source` must point to ids that exist in `legal_provisions.yaml` and `sources.yaml`.
- Run `python scripts/build_knowledge_base.py` after every change: it stops with a clear message if something is wrong.
