"""Download the UNFAIR-ToS dataset (LexGLUE benchmark) into data/raw/.

UNFAIR-ToS contains sentences from 50 online Terms of Service annotated in the
Claudette project. Each sentence has zero or more labels among 8 categories of
potentially unfair clauses; a sentence with no label is considered fair.

Usage (from the project root):
    python scripts/download_data.py

Output:
    data/raw/unfair_tos_train.jsonl
    data/raw/unfair_tos_validation.jsonl
    data/raw/unfair_tos_test.jsonl
    data/raw/metadata.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import datasets
from datasets import DatasetDict, load_dataset

DATASET = "coastalcph/lex_glue"
CONFIG = "unfair_tos"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw"

# Used only if the label names cannot be read from the dataset itself.
FALLBACK_LABEL_NAMES = [
    "Limitation of liability",
    "Unilateral termination",
    "Unilateral change",
    "Content removal",
    "Contract by using",
    "Choice of law",
    "Jurisdiction",
    "Arbitration",
]


def load_unfair_tos() -> DatasetDict:
    """Load the dataset from the Hugging Face Hub, with a Parquet fallback."""
    try:
        return load_dataset(DATASET, CONFIG)
    except Exception as error:  # e.g. script-based loading not supported
        print(f"Standard loading failed ({type(error).__name__}: {error})")
        print("Retrying with the Parquet version of the dataset...")
        return load_dataset(DATASET, revision="refs/convert/parquet", data_dir=CONFIG)


def get_label_names(dsd: DatasetDict) -> tuple[list[str], str]:
    """Return the label names and where they came from."""
    first_split = next(iter(dsd.values()))
    feature = first_split.features.get("labels")
    names = getattr(getattr(feature, "feature", None), "names", None)
    if names:
        return list(names), "dataset features"
    print("WARNING: label names not found in the dataset features, using the fallback list.")
    return FALLBACK_LABEL_NAMES, "fallback list (check the order!)"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dsd = load_unfair_tos()
    label_names, names_source = get_label_names(dsd)

    split_sizes = {}
    for split, ds in dsd.items():
        path = OUT_DIR / f"unfair_tos_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, example in enumerate(ds):
                labels = list(example["labels"])
                record = {
                    "id": f"{split}-{i}",
                    "text": example["text"],
                    "labels": labels,
                    "label_names": [label_names[label] for label in labels],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        split_sizes[split] = len(ds)
        print(f"Saved {len(ds):>5} sentences -> {path.relative_to(PROJECT_ROOT)}")

    metadata = {
        "source": f"https://huggingface.co/datasets/{DATASET}",
        "config": CONFIG,
        "label_names": label_names,
        "label_names_source": names_source,
        "split_sizes": split_sizes,
        "datasets_version": datasets.__version__,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata -> {meta_path.relative_to(PROJECT_ROOT)}")
    print("Labels:", ", ".join(f"{i}={name}" for i, name in enumerate(label_names)))


if __name__ == "__main__":
    main()
