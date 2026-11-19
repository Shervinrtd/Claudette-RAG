"""Is system A really better or worse than system B? Paired bootstrap on the same sentences.

Compares a reference system (default: our main system, full RAG + evidence gate) with every
other system on the UNFAIR-ToS test set: difference of LexGLUE macro-F1, unfair precision and
unfair recall, with a 95% interval and a p-value from 1,000 paired resamples of the test
sentences. A difference whose interval contains 0 (p >= 0.05) is not significant: with only
7 to 38 test examples per category, a few sentences can move macro-F1 by several points.

Usage (from the project root):
    python scripts/compare_systems.py
    python scripts/compare_systems.py --reference bert_legal-bert-base-uncased_seed1_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from claudette_rag.evaluation import (align, headline_metrics, load_gold, paired_bootstrap, read_jsonl,
                                      to_matrix)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = PROJECT_ROOT / "results" / "predictions"
REFERENCE = "gated_k8m3_rag_gpt-4.1-mini-2025-04-14_e8r5p6_test"
SYSTEMS = {
    "Legal-BERT seed 1": "bert_legal-bert-base-uncased_seed1_test",
    "Legal-BERT seed 2": "bert_legal-bert-base-uncased_seed2_test",
    "Legal-BERT seed 3": "bert_legal-bert-base-uncased_seed3_test",
    "TF-IDF + SVM": "tfidf_svm_test",
    "kNN vote (bge-m3)": "knn_bge-m3_test",
    "Plain LLM": "llm_zeroshot_gpt-4.1-mini-2025-04-14_test",
    "RAG full, no gate": "rag_gpt-4.1-mini-2025-04-14_e8r5p6_test",
    "RAG examples only + gate": "gated_k8m3_rag_gpt-4.1-mini-2025-04-14_e8r0p0_test",
}
METRICS = {"lexglue_macro_f1": "macro-F1", "binary_precision": "precision", "binary_recall": "recall"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--reference", default=REFERENCE, help="prediction file name (without .jsonl)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    gold = load_gold(PROJECT_ROOT, "unfair_tos", "test")
    n_labels = 8

    def matrix(name: str):
        _, pred = align(gold, read_jsonl(PREDICTIONS / f"{name}.jsonl"))
        return to_matrix(pred, n_labels)

    y_true = to_matrix([g["labels"] for g in gold], n_labels)
    ref = matrix(args.reference)
    ref_scores = headline_metrics(y_true, ref)
    print(f"Reference: {args.reference}\n  " + "  ".join(
        f"{label} {100 * ref_scores[key]:.1f}" for key, label in METRICS.items()))
    print(f"\nDifference reference - system, 95% paired-bootstrap interval, p-value ({args.n_bootstrap} resamples)")
    results = {}
    for label, name in SYSTEMS.items():
        if name == args.reference or not (PREDICTIONS / f"{name}.jsonl").exists():
            continue
        comparison = paired_bootstrap(y_true, ref, matrix(name), args.n_bootstrap)
        results[label] = {"predictions": name, **{key: comparison[key] for key in METRICS}}
        cells = []
        for key, metric in METRICS.items():
            c = comparison[key]
            mark = "*" if c["p_value"] < 0.05 else " "
            cells.append(f"{metric} {100 * c['difference']:+5.1f} [{100 * c['low']:+5.1f},{100 * c['high']:+5.1f}] "
                         f"p={c['p_value']:.3f}{mark}")
        print(f"  vs {label:26s} " + " | ".join(cells))
    print("\n* = significant at 5%: the interval does not contain 0")

    out = PROJECT_ROOT / "results" / f"comparison_{args.reference}.json"
    out.write_text(json.dumps({"reference": args.reference, "n_bootstrap": args.n_bootstrap,
                               "reference_scores": ref_scores, "comparisons": results}, indent=2),
                   encoding="utf-8")
    print(f"Saved -> {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
