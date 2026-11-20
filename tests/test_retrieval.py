"""Tests for claudette_rag.retrieval with a fake word-overlap embedder (no model download)."""

import re

import numpy as np

from claudette_rag.retrieval import KBIndex, RetrievalConfig, retrieve


class WordEmbedder:
    """Bag-of-words vectors over a fixed vocabulary, L2-normalised."""

    name = "fake-words"

    def __init__(self, vocabulary):
        self.vocab = {w: i for i, w in enumerate(vocabulary)}

    def encode(self, texts):
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                if word in self.vocab:
                    out[row, self.vocab[word]] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


DOCS = [
    {"id": "dir93.annex.1g", "type": "provision", "categories": [], "text": "termination without notice"},
    {"id": "dir93.annex.1q", "type": "provision", "categories": [], "text": "arbitration hinders legal action"},
    {"id": "ter.no_notice", "type": "rationale", "categories": ["ter"], "level": "clearly_unfair",
     "text": "terminate account without notice", "legal_basis": ["dir93.annex.1g"]},
    {"id": "a.arb_obligatory", "type": "rationale", "categories": ["a"], "level": "potentially_unfair",
     "text": "disputes resolved by arbitration", "legal_basis": ["dir93.annex.1q", "missing.ref"]},
    {"id": "ex.1", "type": "example", "categories": ["ter"], "labels": [1],
     "text": "we may terminate your account at any time without notice"},
    {"id": "ex.2", "type": "example", "categories": [], "labels": [], "text": "contact us by email"},
    {"id": "ex.3", "type": "example", "categories": ["a"], "labels": [7],
     "text": "all disputes will be resolved by binding arbitration"},
]
VOCAB = sorted({w for d in DOCS for w in re.findall(r"[a-z]+", d["text"])})


def test_search_returns_best_match_of_the_requested_type():
    emb = WordEmbedder(VOCAB)
    index = KBIndex.build(DOCS, emb)
    q = emb.encode(["can we terminate the account without notice"])
    ex = index.search(q, "example", 2)[0]
    assert ex[0][0]["id"] == "ex.1" and len(ex) == 2
    assert ex[0][1] >= ex[1][1]                        # sorted by similarity
    rat = index.search(q, "rationale", 5)[0]
    assert rat[0][0]["id"] == "ter.no_notice" and len(rat) == 2   # k capped at available docs
    assert index.search(q, "unknown-type", 3) == [[]]


def test_retrieve_follows_legal_basis_and_skips_unknown_references():
    emb = WordEmbedder(VOCAB)
    index = KBIndex.build(DOCS, emb)
    q = emb.encode(["disputes are resolved by arbitration", "we terminate without notice"])
    out = retrieve(index, q, RetrievalConfig(k_examples=1, k_rationales=1, max_provisions=5))
    assert out[0]["examples"][0][0]["id"] == "ex.3"
    assert [p["id"] for p in out[0]["provisions"]] == ["dir93.annex.1q"]      # missing.ref skipped
    assert [p["id"] for p in out[1]["provisions"]] == ["dir93.annex.1g"]


def test_save_and_load_roundtrip(tmp_path):
    emb = WordEmbedder(VOCAB)
    index = KBIndex.build(DOCS, emb)
    index.save(tmp_path / "idx")
    loaded = KBIndex.load(tmp_path / "idx")
    assert loaded.model_name == "fake-words" and len(loaded.docs) == len(DOCS)
    np.testing.assert_allclose(loaded.embeddings, index.embeddings)
    q = emb.encode(["binding arbitration"])
    assert loaded.search(q, "example", 1)[0][0][0]["id"] == "ex.3"


def test_knn_vote():
    from claudette_rag.retrieval import knn_vote
    neighbours = [[[1], [1, 2], [], [2]], [[], [], [0]]]
    assert knn_vote(neighbours, k=3, t=2, n_labels=3) == [[1], []]
    assert knn_vote(neighbours, k=4, t=2, n_labels=3) == [[1, 2], []]
    assert knn_vote(neighbours, k=3, t=1, n_labels=3) == [[1, 2], [0]]
