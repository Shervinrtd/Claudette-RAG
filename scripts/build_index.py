"""Embed the knowledge-base documents and save the search index.

Usage (from the project root, after scripts/build_knowledge_base.py):
    python scripts/build_index.py                         # BAAI/bge-m3 (multilingual)
    python scripts/build_index.py --model intfloat/multilingual-e5-base

Output: data/processed/index/<model>/{embeddings.npy, documents.jsonl, meta.json}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from claudette_rag.retrieval import DEFAULT_EMBEDDER, KBIndex, SentenceTransformerEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def index_dir(model_name: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / "index" / model_name.rstrip("/").split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_EMBEDDER)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with (PROJECT_ROOT / "data" / "processed" / "kb_documents.jsonl").open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    print(f"Embedding {len(docs)} documents with {args.model} ...")
    start = time.time()
    embedder = SentenceTransformerEmbedder(args.model, batch_size=args.batch_size)
    index = KBIndex.build(docs, embedder)
    out = index_dir(args.model)
    index.save(out)
    print(f"Done in {time.time() - start:.0f}s: {index.embeddings.shape[0]} vectors of dimension "
          f"{index.embeddings.shape[1]} -> {out.relative_to(PROJECT_ROOT)}")
    print("Documents per type:", {t: len(i) for t, i in index.by_type.items()})


if __name__ == "__main__":
    main()
