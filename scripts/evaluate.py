"""Score a prediction file with the shared metrics.

Usage (from the project root):
    python scripts/evaluate.py results/predictions/tfidf_svm_test.jsonl
    python scripts/evaluate.py results/predictions/rag_it.jsonl --dataset multilingual --language it

The report is printed and saved as results/<name>.json (name = prediction file name).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from claudette_rag.evaluation import align, evaluate, format_report, load_gold, read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("predictions", type=Path, help="jsonl file with {'id', 'labels'} per line")
    parser.add_argument("--dataset", choices=["unfair_tos", "multilingual"], default="unfair_tos")
    parser.add_argument("--split", default="test", help="UNFAIR-ToS split (default: test)")
    parser.add_argument("--language", choices=["en", "de", "it", "pl"], help="multilingual only")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap samples (0 = off)")
    parser.add_argument("--name", help="name of the run (default: prediction file name)")
    args = parser.parse_args()

    name = args.name or args.predictions.stem
    label_names = json.loads((PROJECT_ROOT / "data/raw/metadata.json").read_text(encoding="utf-8"))["label_names"]
    gold = load_gold(PROJECT_ROOT, args.dataset, args.split, args.language)
    gold_labels, pred_labels = align(gold, read_jsonl(args.predictions))

    report = evaluate(gold_labels, pred_labels, label_names, n_bootstrap=args.bootstrap)
    report["run"] = {"name": name, "predictions": str(args.predictions), "dataset": args.dataset,
                     "split": args.split if args.dataset == "unfair_tos" else None,
                     "language": args.language}
    print(format_report(name, report))

    out = PROJECT_ROOT / "results" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
