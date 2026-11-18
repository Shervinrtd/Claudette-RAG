"""Download the multilingual Claudette corpus into data/raw/multilingual/.

"A Corpus for Multilingual Analysis of Online Terms of Service"
(Drawzeski et al., NLLP 2021): 25 Terms of Service, each annotated by legal
experts in English, German, Italian and Polish (100 contracts in total).
Each sentence has an unfairness level (clearly_fair / potentially_unfair /
clearly_unfair / untagged) and boolean tags for 9 categories
(a, ch, cr, j, law, ltd, ter, use, pinc).

We use the Hugging Face version prepared by Joel Niklaus, split by company.
License: CC BY-NC 2.5 (non-commercial use only).

Usage (from the project root):
    python scripts/download_multilingual.py

Output:
    data/raw/multilingual/tos_multilingual_{train,validation,test}.jsonl
    data/raw/multilingual/metadata.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "joelniklaus/online_terms_of_service"
DATA_SUFFIXES = (".jsonl", ".jsonl.xz", ".jsonl.gz", ".json", ".json.xz", ".parquet")
SPLIT_ALIASES = {"train": "train", "validation": "validation", "val": "validation",
                 "dev": "validation", "test": "test"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "multilingual"


def split_of(filename: str) -> str | None:
    """Guess the split from a file name such as 'data/train.jsonl.xz'."""
    name = Path(filename).name.lower()
    for alias, split in SPLIT_ALIASES.items():
        if name.startswith(alias) or f"_{alias}" in name or f"-{alias}" in name:
            return split
    return None


def read_file(path: str) -> pd.DataFrame:
    """Read a jsonl/json/parquet file (compression is inferred from the suffix)."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    repo_files = list_repo_files(REPO_ID, repo_type="dataset")
    data_files = [f for f in repo_files if f.lower().endswith(DATA_SUFFIXES)]
    by_split: dict[str, list[str]] = {}
    for f in data_files:
        split = split_of(f)
        if split:
            by_split.setdefault(split, []).append(f)

    if not by_split:
        raise SystemExit(f"No data files recognised in {REPO_ID}. Files in the repo:\n  "
                         + "\n  ".join(repo_files))

    summary = {}
    for split in ("train", "validation", "test"):
        files = sorted(by_split.get(split, []))
        if not files:
            print(f"WARNING: no file found for split '{split}'")
            continue
        df = pd.concat([read_file(hf_hub_download(REPO_ID, f, repo_type="dataset")) for f in files],
                       ignore_index=True)
        out_path = OUT_DIR / f"tos_multilingual_{split}.jsonl"
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)

        summary[split] = {
            "source_files": files,
            "sentences": len(df),
            "languages": df["language"].value_counts().to_dict() if "language" in df else None,
            "companies": sorted(df["company"].unique().tolist()) if "company" in df else None,
        }
        print(f"Saved {len(df):>6} sentences -> {out_path.relative_to(PROJECT_ROOT)} "
              f"(from {', '.join(files)})")

    columns = list(df.columns)
    metadata = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "paper": "Drawzeski et al. (2021), A Corpus for Multilingual Analysis of Online Terms of Service, NLLP Workshop",
        "license": "CC BY-NC 2.5 (non-commercial use only)",
        "columns": columns,
        "splits": summary,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata -> {meta_path.relative_to(PROJECT_ROOT)}")
    print("Columns:", ", ".join(columns))


if __name__ == "__main__":
    main()
