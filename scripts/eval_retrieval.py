"""How good is the retriever? (no LLM involved, no cost)

On the validation set it measures, for k = 1, 3, 5, 8, 10:
- example hit@k:   for an unfair sentence, at least one of the k retrieved training sentences
                   has one of its gold categories;
- rationale hit@k: for an unfair sentence, at least one of the k retrieved rationales
                   belongs to one of its gold categories;
- fair purity@k:   for a fair sentence, the share of retrieved examples that are fair
                   (high = the examples will not push the LLM towards a false alarm).

It also scores a retrieval-only baseline, kNN: a category is predicted when at least t of
the k nearest training sentences have it. k and t are chosen on validation (LexGLUE
macro-F1) and the chosen setting is then scored once on the test set.

Usage (from the project root, after scripts/build_index.py):
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --model intfloat/multilingual-e5-base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from claudette_rag.evaluation import evaluate, format_report, load_gold
from claudette_rag.knowledge_base import load_knowledge_base
from claudette_rag.retrieval import DEFAULT_EMBEDDER, KBIndex, SentenceTransformerEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
K_VALUES = [1, 3, 5, 8, 10]
KNN_K = [1, 3, 5, 8, 10, 15]


def retrieval_quality(index: KBIndex, vectors: np.ndarray, gold: list[dict], code_of: dict[int, str]) -> dict:
    kmax = max(K_VALUES)
    examples = index.search(vectors, "example", kmax)
    rationales = index.search(vectors, "rationale", kmax)
    table = {}
    unfair = [i for i, g in enumerate(gold) if g["labels"]]
    fair = [i for i, g in enumerate(gold) if not g["labels"]]
    for k in K_VALUES:
        ex_hits = rat_hits = 0
        for i in unfair:
            gold_codes = {code_of[label] for label in gold[i]["labels"]}
            ex_hits += any(gold_codes & set(doc["categories"]) for doc, _ in examples[i][:k])
            rat_hits += any(gold_codes & set(doc["categories"]) for doc, _ in rationales[i][:k])
        purity = np.mean([np.mean([not doc["labels"] for doc, _ in examples[i][:k]]) for i in fair])
        table[k] = {"example_hit": ex_hits / len(unfair), "rationale_hit": rat_hits / len(unfair),
                    "fair_purity": float(purity)}
    return table


def knn_predict(neighbour_labels: list[list[list[int]]], k: int, t: int, n_labels: int) -> list[list[int]]:
    preds = []
    for neighbours in neighbour_labels:
        counts = np.zeros(n_labels, dtype=int)
        for labels in neighbours[:k]:
            counts[labels] += 1
        preds.append([int(c) for c in np.flatnonzero(counts >= t)])
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_EMBEDDER)
    args = parser.parse_args()
    short = args.model.rstrip("/").split("/")[-1]

    index = KBIndex.load(PROJECT_ROOT / "data" / "processed" / "index" / short)
    embedder = SentenceTransformerEmbedder(index.model_name)
    categories = sorted(load_knowledge_base(PROJECT_ROOT / "knowledge_base")["categories"],
                        key=lambda c: c["label_index"])
    code_of = {c["label_index"]: c["id"] for c in categories}
    label_names = [c["name"] for c in categories]

    gold = {s: load_gold(PROJECT_ROOT, "unfair_tos", s) for s in ("validation", "test")}
    vectors = {s: embedder.encode([g["text"].strip() for g in gold[s]]) for s in gold}

    quality = retrieval_quality(index, vectors["validation"], gold["validation"], code_of)
    print(f"Retrieval quality on validation ({index.model_name})")
    print(f"{'k':>3} {'example hit@k':>14} {'rationale hit@k':>16} {'fair purity@k':>14}")
    for k, row in quality.items():
        print(f"{k:>3} {100 * row['example_hit']:14.1f} {100 * row['rationale_hit']:16.1f} "
              f"{100 * row['fair_purity']:14.1f}")

    # kNN vote baseline: choose (k, t) on validation, score once on test
    neighbours = {s: [[doc["labels"] for doc, _ in row]
                      for row in index.search(vectors[s], "example", max(KNN_K))] for s in gold}
    best = None
    for k in KNN_K:
        for t in range(1, k + 1):
            pred = knn_predict(neighbours["validation"], k, t, len(label_names))
            score = evaluate([g["labels"] for g in gold["validation"]], pred, label_names,
                             n_bootstrap=0)["headline"]["lexglue_macro_f1"]
            if best is None or score > best[0]:
                best = (score, k, t)
    _, k, t = best
    print(f"\nkNN baseline: best on validation k={k}, t={t} (macro-F1 {100 * best[0]:.1f})")

    run = f"knn_{short}"
    for split in ("validation", "test"):
        pred = knn_predict(neighbours[split], k, t, len(label_names))
        pred_path = RESULTS / "predictions" / f"{run}_{split}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for g, labels in zip(gold[split], pred):
                f.write(json.dumps({"id": g["id"], "labels": labels}) + "\n")
        report = evaluate([g["labels"] for g in gold[split]], pred, label_names)
        report["run"] = {"name": f"{run}_{split}", "system": "kNN vote over retrieved training sentences",
                         "embedder": index.model_name, "k": k, "t": t, "dataset": "unfair_tos",
                         "split": split, "predictions": str(pred_path.relative_to(PROJECT_ROOT))}
        (RESULTS / f"{run}_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_report(f"{run}_test (k={k}, t={t})", report))

    out = RESULTS / f"retrieval_{short}_validation.json"
    out.write_text(json.dumps({"embedder": index.model_name, "split": "validation",
                               "quality_at_k": quality, "knn_choice": {"k": k, "t": t,
                               "validation_macro_f1": best[0]}}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
