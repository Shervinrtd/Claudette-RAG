"""Validate the legal knowledge base and export it as retrievable documents.

Usage (from the project root, after scripts/download_data.py):
    python scripts/build_knowledge_base.py

Output:
    data/processed/kb_documents.jsonl   one document per provision, category, rationale, example
    data/processed/kb_summary.json      counts, for the report
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from claudette_rag.knowledge_base import (load_examples, load_knowledge_base, to_documents,
                                          validate)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "knowledge_base"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    label_names = json.loads((RAW_DIR / "metadata.json").read_text(encoding="utf-8"))["label_names"]
    kb = load_knowledge_base(KB_DIR)

    errors = validate(kb, label_names)
    if errors:
        print(f"The knowledge base has {len(errors)} problem(s):")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("Knowledge base is valid.")

    examples, removed = load_examples(RAW_DIR)
    docs = to_documents(kb, examples, label_names)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "kb_documents.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    rationales = kb["rationales"]
    summary = {
        "documents_by_type": dict(Counter(d["type"] for d in docs)),
        "rationales_by_category": dict(Counter(r["category"] for r in rationales)),
        "rationales_by_level": dict(Counter(r["level"] for r in rationales)),
        "rationales_by_source": dict(Counter(r["source"] for r in rationales)),
        "examples": {
            "total": len(examples),
            "unfair": sum(bool(e["labels"]) for e in examples),
            "fair": sum(not e["labels"] for e in examples),
            "removed_because_in_validation_or_test": removed,
        },
    }
    (OUT_DIR / "kb_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {len(docs)} documents -> {out_path.relative_to(PROJECT_ROOT)}")
    for doc_type, n in summary["documents_by_type"].items():
        print(f"  {doc_type:<10} {n:>5}")
    print("Rationales per category:", summary["rationales_by_category"])
    print("Rationales per level:   ", summary["rationales_by_level"])
    print(f"Examples: {summary['examples']['unfair']} unfair + {summary['examples']['fair']} fair "
          f"({removed} training sentences removed because they also occur in validation/test)")


if __name__ == "__main__":
    main()
