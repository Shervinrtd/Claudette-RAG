"""Simple reference baselines on UNFAIR-ToS.

1. "Always fair": predicts that no sentence is unfair. It shows why accuracy-like
   metrics are misleading on this imbalanced dataset.
2. TF-IDF + linear SVM (one classifier per category), the classical baseline of the
   LexGLUE paper (reported test scores: 94.7 micro-F1 / 75.0 macro-F1). Getting close
   to those numbers is also a check that our evaluation code follows the LexGLUE convention.
   The regularisation strength C is chosen on the validation set (LexGLUE macro-F1).

Usage (from the project root):
    python scripts/baseline_tfidf.py

Output: results/predictions/{always_fair,tfidf_svm}_{validation,test}.jsonl
        results/{always_fair,tfidf_svm}_{validation,test}.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from claudette_rag.evaluation import evaluate, format_report, load_gold, to_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
C_GRID = [0.1, 1.0, 10.0, 100.0]


def save(name: str, split: str, gold: list[dict], pred_labels: list[list[int]],
         label_names: list[str], bootstrap: int = 1000) -> dict:
    run = f"{name}_{split}"
    pred_path = RESULTS / "predictions" / f"{run}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for g, labels in zip(gold, pred_labels):
            f.write(json.dumps({"id": g["id"], "labels": labels}) + "\n")
    report = evaluate([g["labels"] for g in gold], pred_labels, label_names, n_bootstrap=bootstrap)
    report["run"] = {"name": run, "predictions": str(pred_path.relative_to(PROJECT_ROOT)),
                     "dataset": "unfair_tos", "split": split}
    (RESULTS / f"{run}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    label_names = json.loads((PROJECT_ROOT / "data/raw/metadata.json").read_text(encoding="utf-8"))["label_names"]
    k = len(label_names)
    data = {s: load_gold(PROJECT_ROOT, "unfair_tos", s) for s in ("train", "validation", "test")}

    # 1. Always fair
    for split in ("validation", "test"):
        report = save("always_fair", split, data[split], [[] for _ in data[split]], label_names)
        if split == "test":
            print(format_report("always_fair_test", report), "\n")

    # 2. TF-IDF + linear SVM
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform([r["text"] for r in data["train"]])
    y_train = to_matrix([r["labels"] for r in data["train"]], k).astype(int)

    def predict(model, split):
        y = model.predict(vectorizer.transform([r["text"] for r in data[split]]))
        return [list(map(int, np.flatnonzero(row))) for row in y]

    best = None
    for c in C_GRID:
        model = OneVsRestClassifier(LinearSVC(C=c, max_iter=10000, random_state=0)).fit(x_train, y_train)
        val_labels = predict(model, "validation")
        score = evaluate([r["labels"] for r in data["validation"]], val_labels, label_names,
                         n_bootstrap=0)["headline"]["lexglue_macro_f1"]
        print(f"C={c:<5} validation LexGLUE macro-F1 = {100 * score:.1f}")
        if best is None or score > best[0]:
            best = (score, c, model)
    _, c, model = best
    print(f"-> selected C={c}\n")

    save("tfidf_svm", "validation", data["validation"], predict(model, "validation"), label_names)
    report = save("tfidf_svm", "test", data["test"], predict(model, "test"), label_names)
    print(format_report(f"tfidf_svm_test (C={c})", report))


if __name__ == "__main__":
    main()
