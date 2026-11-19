"""Dense retrieval over the legal knowledge base.

For a clause, the retriever returns
- the k most similar **examples** (annotated training sentences, fair and unfair);
- the k most similar **rationales** (practices that make a clause unfair, or clearly fair);
- the **legal provisions** cited by those rationales (followed through their legal_basis,
  not retrieved by similarity), so the LLM sees the legal reason behind each rationale.

Documents are embedded once with a multilingual sentence-embedding model (default
BAAI/bge-m3), so a clause in Italian, German or Polish can retrieve English documents.
The knowledge base is small (~5,600 documents), so exact cosine search with numpy is fast
and no approximate vector database is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDER = "BAAI/bge-m3"


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model; returns L2-normalised float32 vectors."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER, device: str | None = None,
                 batch_size: int = 32, max_seq_length: int = 256):
        import torch
        from sentence_transformers import SentenceTransformer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        if device == "cuda":
            self.model.half()                      # fp16: half the memory, faster on GPU
        self.model.max_seq_length = max_seq_length
        self.name = model_name
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=len(texts) > 500)
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class KBIndex:
    """Embedded knowledge-base documents, searchable by document type."""

    docs: list[dict]
    embeddings: np.ndarray
    model_name: str
    by_type: dict[str, np.ndarray] = field(init=False, repr=False)
    by_id: dict[str, dict] = field(init=False, repr=False)

    def __post_init__(self):
        if len(self.docs) != len(self.embeddings):
            raise ValueError("documents and embeddings have different lengths")
        types = sorted({d["type"] for d in self.docs})
        self.by_type = {t: np.array([i for i, d in enumerate(self.docs) if d["type"] == t]) for t in types}
        self.by_id = {d["id"]: d for d in self.docs}

    @classmethod
    def build(cls, docs: list[dict], embedder) -> "KBIndex":
        return cls(docs, embedder.encode([d["text"] for d in docs]), embedder.name)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with (directory / "documents.jsonl").open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        meta = {"model_name": self.model_name, "n_documents": len(self.docs),
                "dimension": int(self.embeddings.shape[1]),
                "documents_by_type": {t: int(len(i)) for t, i in self.by_type.items()}}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "KBIndex":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        with (directory / "documents.jsonl").open(encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        return cls(docs, np.load(directory / "embeddings.npy"), meta["model_name"])

    def search(self, query_vectors: np.ndarray, doc_type: str, k: int) -> list[list[tuple[dict, float]]]:
        """Top-k documents of one type for each query vector (cosine similarity)."""
        if k <= 0 or doc_type not in self.by_type:
            return [[] for _ in range(len(query_vectors))]
        idx = self.by_type[doc_type]
        sims = np.asarray(query_vectors, dtype=np.float32) @ self.embeddings[idx].T
        k = min(k, len(idx))
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        results = []
        for row, candidates in zip(sims, top):
            order = candidates[np.argsort(-row[candidates])]
            results.append([(self.docs[idx[j]], float(row[j])) for j in order])
        return results


@dataclass
class RetrievalConfig:
    k_examples: int = 8
    k_rationales: int = 5
    max_provisions: int = 6


def retrieve(index: KBIndex, query_vectors: np.ndarray, config: RetrievalConfig) -> list[dict]:
    """Examples, rationales and linked legal provisions for each query vector."""
    examples = index.search(query_vectors, "example", config.k_examples)
    rationales = index.search(query_vectors, "rationale", config.k_rationales)
    results = []
    for ex, rat in zip(examples, rationales):
        provision_ids: list[str] = []
        for doc, _ in rat:
            for ref in doc.get("legal_basis", []):
                if ref not in provision_ids and ref in index.by_id:
                    provision_ids.append(ref)
        provisions = [index.by_id[p] for p in provision_ids[:config.max_provisions]]
        results.append({"examples": ex, "rationales": rat, "provisions": provisions})
    return results
