"""Free baselines on the multilingual evaluation set (English, German, Italian, Polish).

- TF-IDF + linear SVM, trained on the English UNFAIR-ToS training set exactly as in
  baseline_tfidf.py (C chosen on validation). Its vocabulary is English, so it is expected to
  miss almost everything in the other languages: a monolingual model is not fair to them.
- kNN vote over the training sentences retrieved with the multilingual embedder bge-m3, with
  k and t as chosen on validation by eval_retrieval.py, and the same queries as the RAG system
  (the original sentences). It shows how much of the signal survives across languages
  without any LLM.

Legal-BERT (English-only) is evaluated with: python scripts/train_bert.py --tag hpc --multilingual

Usage (from the project root, after scripts/build_index.py and scripts/eval_retrieval.py):
    python scripts/baselines_multilingual.py

Output: results/predictions/{tfidf_svm,knn_bge-m3}_multilingual_<lang>.jsonl and results/<run>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from claudette_rag.evaluation import evaluate, load_gold, to_matrix
from claudette_rag.retrieval import KBIndex, SentenceTransformerEmbedder, knn_vote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
LANGUAGES = ["en", "de", "it", "pl"]
C_GRID = [0.1, 1.0, 10.0, 100.0]


def save(run: str, records: list[dict], pred: list[list[int]], label_names: list[str], info: dict) -> dict:
    pred_path = RESULTS / "predictions" / f"{run}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for r, labels in zip(records, pred):
            f.write(json.dumps({"id": r["id"], "labels": labels}) + "\n")
    report = evaluate([r["labels"] for r in records], pred, label_names)
    report["run"] = {"name": run, "dataset": "multilingual", **info,
                     "predictions": pred_path.relative_to(PROJECT_ROOT).as_posix()}
    (RESULTS / f"{run}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    label_names = json.loads((PROJECT_ROOT / "data/raw/metadata.json").read_text(encoding="utf-8"))["label_names"]
    n_labels = len(label_names)
    data = {s: load_gold(PROJECT_ROOT, "unfair_tos", s) for s in ("train", "validation")}
    multilingual = {lang: load_gold(PROJECT_ROOT, "multilingual", language=lang) for lang in LANGUAGES}
    summary = {}

    # ---- TF-IDF + linear SVM (as in baseline_tfidf.py)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform([r["text"] for r in data["train"]])
    y_train = to_matrix([r["labels"] for r in data["train"]], n_labels).astype(int)

    def predict(model, records):
        y = model.predict(vectorizer.transform([r["text"] for r in records]))
        return [list(map(int, np.flatnonzero(row))) for row in y]

    best = None
    for c in C_GRID:
        model = OneVsRestClassifier(LinearSVC(C=c, max_iter=10000, random_state=0)).fit(x_train, y_train)
        score = evaluate([r["labels"] for r in data["validation"]], predict(model, data["validation"]),
                         label_names, n_bootstrap=0)["headline"]["lexglue_macro_f1"]
        if best is None or score > best[0]:
            best = (score, c, model)
    _, c, model = best
    print(f"TF-IDF + SVM: C={c} chosen on validation")
    for lang, records in multilingual.items():
        report = save(f"tfidf_svm_multilingual_{lang}", records, predict(model, records), label_names,
                      {"system": "TF-IDF + linear SVM (trained on English)", "language": lang, "C": c})
        summary[("TF-IDF + SVM", lang)] = report["headline"]

    # ---- kNN vote with the multilingual embedder
    choice = json.loads((RESULTS / "retrieval_bge-m3_validation.json").read_text(encoding="utf-8"))["knn_choice"]
    k, t = choice["k"], choice["t"]
    index = KBIndex.load(PROJECT_ROOT / "data" / "processed" / "index" / "bge-m3")
    embedder = SentenceTransformerEmbedder(index.model_name)
    print(f"kNN vote: k={k}, t={t} chosen on validation, embedder {index.model_name}")
    for lang, records in multilingual.items():
        vectors = embedder.encode([(r.get("text_raw") or r["text"]).strip() for r in records])
        neighbours = [[doc["labels"] for doc, _ in row] for row in index.search(vectors, "example", k)]
        report = save(f"knn_bge-m3_multilingual_{lang}", records, knn_vote(neighbours, k, t, n_labels),
                      label_names, {"system": "kNN vote over retrieved training sentences",
                                    "embedder": index.model_name, "k": k, "t": t, "language": lang})
        summary[("kNN (bge-m3)", lang)] = report["headline"]

    print(f"\n{'system':16s} {'lang':5s} {'macro-F1':>9s} {'recall':>7s} {'precision':>10s}")
    for (system, lang), h in summary.items():
        print(f"{system:16s} {lang:5s} {100 * h['lexglue_macro_f1']:9.1f} {100 * h['binary_recall']:7.1f} "
              f"{100 * h['binary_precision']:10.1f}")
    print("\nSaved -> results/{tfidf_svm,knn_bge-m3}_multilingual_<lang>.json")


if __name__ == "__main__":
    main()
