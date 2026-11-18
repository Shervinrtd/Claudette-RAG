"""BERT baseline: fine-tune a BERT-style model on UNFAIR-ToS with several seeds.

Usage (from the project root):
    python scripts/train_bert.py                                        # Legal-BERT, seeds 1 2 3
    python scripts/train_bert.py --model bert-base-uncased --seeds 1    # plain BERT, one seed
    python scripts/train_bert.py --limit 300 --epochs 1 --seeds 1       # quick smoke test

For each seed: results/predictions/bert_<model>_seed<s>_{validation,test}.jsonl and
results/bert_<model>_seed<s>_{validation,test}.json. Over all seeds:
results/bert_<model>_summary.json (mean and standard deviation of the test metrics).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from claudette_rag.bert import make_loader, predict_probs, probs_to_labels, train
from claudette_rag.evaluation import evaluate, format_report, load_gold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def write_run(run: str, split: str, records: list[dict], pred: list[list[int]],
              label_names: list[str], info: dict) -> dict:
    pred_path = RESULTS / "predictions" / f"{run}_{split}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for r, labels in zip(records, pred):
            f.write(json.dumps({"id": r["id"], "labels": labels}) + "\n")
    report = evaluate([r["labels"] for r in records], pred, label_names)
    report["run"] = {"name": f"{run}_{split}", "dataset": "unfair_tos", "split": split,
                     "predictions": str(pred_path.relative_to(PROJECT_ROOT)), **info}
    (RESULTS / f"{run}_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="use only the first N training sentences (debug)")
    parser.add_argument("--save-model", action="store_true", help="save the best model of each seed in models/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no GPU found, training on CPU will be very slow. "
              "Check the PyTorch installation (torch.cuda.is_available()).")

    label_names = json.loads((PROJECT_ROOT / "data/raw/metadata.json").read_text(encoding="utf-8"))["label_names"]
    data = {s: load_gold(PROJECT_ROOT, "unfair_tos", s) for s in ("train", "validation", "test")}
    if args.limit:
        data["train"] = data["train"][:args.limit]
    short = args.model.rstrip("/").split("/")[-1]
    base = f"bert_{short}" + (f"_limit{args.limit}" if args.limit else "")

    test_reports = []
    for seed in args.seeds:
        run = f"{base}_seed{seed}"
        print(f"\n=== {run}: training on {len(data['train'])} sentences")
        start = time.time()
        model, tokenizer, history = train(
            args.model, data["train"], data["validation"], label_names, seed=seed, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size, max_length=args.max_length,
            patience=args.patience, threshold=args.threshold, device=device)
        info = {"system": "fine-tuned encoder", "model": args.model, "seed": seed,
                "training_minutes": round((time.time() - start) / 60, 1),
                "hyperparameters": {k: getattr(args, k) for k in
                                    ("epochs", "patience", "lr", "batch_size", "max_length", "threshold")},
                "history": history}
        for split in ("validation", "test"):
            loader = make_loader(data[split], tokenizer, len(label_names), args.max_length, 64, False)
            pred = probs_to_labels(predict_probs(model, loader, device), args.threshold)
            report = write_run(run, split, data[split], pred, label_names, info)
        test_reports.append(report)
        print(format_report(f"{run}_test", report))
        if args.save_model:
            out = PROJECT_ROOT / "models" / run
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print(f"Model saved -> {out.relative_to(PROJECT_ROOT)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    keys = list(test_reports[0]["headline"])
    summary = {"model": args.model, "seeds": args.seeds, "test": {}}
    for key in keys:
        values = [r["headline"][key] for r in test_reports]
        summary["test"][key] = {"mean": statistics.mean(values),
                                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                                "per_seed": values}
    (RESULTS / f"{base}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== {base}: test results over seeds {args.seeds} (mean ± std)")
    for key, v in summary["test"].items():
        print(f"{key:<28} {100 * v['mean']:5.1f} ± {100 * v['std']:.1f}")


if __name__ == "__main__":
    main()
