"""Build the multilingual evaluation set used for the cross-language fairness experiment.

The multilingual Claudette corpus contains newer (2021) versions of many contracts
that are also in UNFAIR-ToS. To avoid testing the system on sentences it was
trained on (or that are in its knowledge base), we:

1. compare every English sentence (>= 6 words) with all UNFAIR-ToS sentences and
   mark it as a near-duplicate if the similarity is >= 90 (ignoring spaces,
   punctuation and casing);
2. keep only companies whose English version has < 10% near-duplicates;
3. drop companies whose language versions are not parallel (different documents);
4. map the annotations to the 8 UNFAIR-ToS categories: a sentence is unfair if its
   level is potentially_unfair or clearly_unfair, and its labels are its category
   tags. "Privacy included" (pinc) is not in UNFAIR-ToS, so sentences unfair only
   because of pinc are flagged.

Usage (from the project root, after both download scripts):
    python scripts/prepare_multilingual_eval.py

Output:
    data/processed/multilingual_eval.jsonl
    data/processed/multilingual_eval_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from claudette_rag.text import normalize_like_unfair_tos, squash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "validation", "test"]
NEAR_DUPLICATE_SCORE = 90     # rapidfuzz ratio (0-100) on squashed text
MAX_COMPANY_OVERLAP = 10.0    # % of English sentences that are near-duplicates
MIN_WORDS = 6                 # shorter lines are mostly headers, dates, URLs
NON_PARALLEL = {"Terravision": "English version is a different, ~3x longer document"}

# Multilingual tags -> UNFAIR-ToS label indices (same order as data/raw/metadata.json)
TAG_TO_LABEL = {"ltd": 0, "ter": 1, "ch": 2, "cr": 3, "use": 4, "law": 5, "j": 6, "a": 7}
UNFAIR_LEVELS = {"potentially_unfair", "clearly_unfair"}


def load_jsonl(paths) -> pd.DataFrame:
    return pd.concat([pd.read_json(p, lines=True).assign(source_split=s) for s, p in paths],
                     ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tos = load_jsonl([(s, RAW / f"unfair_tos_{s}.jsonl") for s in SPLITS])
    ml = load_jsonl([(s, RAW / "multilingual" / f"tos_multilingual_{s}.jsonl") for s in SPLITS])
    label_names = json.loads((RAW / "metadata.json").read_text(encoding="utf-8"))["label_names"]

    # 1. Near-duplicates of UNFAIR-ToS among English sentences
    en = ml[(ml["language"] == "en") & (ml["sentence"].str.split().str.len() >= MIN_WORDS)].copy()
    scores = process.cdist(en["sentence"].map(squash).tolist(), tos["text"].map(squash).tolist(),
                           scorer=fuzz.ratio, workers=-1, dtype=np.uint8)
    en["best_similarity"] = scores.max(axis=1)
    en["near_duplicate"] = en["best_similarity"] >= NEAR_DUPLICATE_SCORE

    # 2-3. Company selection
    overlap = (en.groupby("company")["near_duplicate"].mean() * 100).round(1)
    companies = {}
    for company, pct in overlap.sort_values().items():
        if company in NON_PARALLEL:
            status = f"excluded: {NON_PARALLEL[company]}"
        elif pct >= MAX_COMPANY_OVERLAP:
            status = "excluded: overlaps with UNFAIR-ToS"
        else:
            status = "kept"
        companies[company] = {"overlap_pct": float(pct), "status": status}
    kept = [c for c, info in companies.items() if info["status"] == "kept"]

    # 4. Build the evaluation records
    ev = ml[ml["company"].isin(kept)].copy()
    # Only English sentences can be matched to UNFAIR-ToS (other languages are
    # translations), so the duplicate flag is set for English rows only.
    dup_keys = set(map(tuple, en.loc[en["near_duplicate"], ["company", "line_number"]].values))

    records = []
    for i, row in enumerate(ev.itertuples(index=False)):
        is_unfair = row.unfairness_level in UNFAIR_LEVELS
        labels = sorted(TAG_TO_LABEL[t] for t in TAG_TO_LABEL if is_unfair and getattr(row, t))
        records.append({
            "id": f"ml-{row.language}-{row.company}-{row.line_number}",
            "language": row.language,
            "company": row.company,
            "line_number": int(row.line_number),
            "text_raw": row.sentence,
            "text": normalize_like_unfair_tos(row.sentence),
            "labels": labels,
            "label_names": [label_names[j] for j in labels],
            "unfairness_level": row.unfairness_level,
            "pinc": bool(row.pinc),
            "pinc_only": bool(is_unfair and row.pinc and not labels),
            "in_unfair_tos": row.language == "en" and (row.company, row.line_number) in dup_keys,
        })
    out_path = OUT / "multilingual_eval.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(records)
    df["unfair"] = df["labels"].str.len() > 0
    per_language = df.groupby("language").agg(sentences=("id", "size"), unfair=("unfair", "sum"),
                                              pinc_only=("pinc_only", "sum")).to_dict(orient="index")
    summary = {
        "settings": {"near_duplicate_score": NEAR_DUPLICATE_SCORE,
                     "max_company_overlap_pct": MAX_COMPANY_OVERLAP, "min_words": MIN_WORDS},
        "english_sentences_checked": int(len(en)),
        "english_near_duplicates_pct": round(float(en["near_duplicate"].mean() * 100), 1),
        "companies": companies,
        "kept_companies": kept,
        "per_language": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in per_language.items()},
    }
    (OUT / "multilingual_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"English sentences near-duplicate of UNFAIR-ToS: {summary['english_near_duplicates_pct']}%")
    print(f"Kept {len(kept)} companies: {', '.join(kept)}")
    for lang, s in summary["per_language"].items():
        print(f"  {lang}: {s['sentences']:>5} sentences, {s['unfair']:>3} unfair "
              f"(+{s['pinc_only']} unfair only for 'privacy included')")
    print(f"Saved -> {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
