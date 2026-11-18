"""Tests for claudette_rag.bert with a tiny randomly initialised BERT (CPU, no download)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tokenizers import BertWordPieceTokenizer  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from claudette_rag.bert import make_loader, predict_probs, probs_to_labels, train  # noqa: E402

NAMES = ["A", "B", "C"]
RECORDS = [
    {"id": f"s{i}", "text": text, "labels": labels} for i, (text, labels) in enumerate([
        ("we may terminate your account at any time .", [0]),
        ("we are not liable for any damages .", [1]),
        ("disputes are resolved by binding arbitration .", [2]),
        ("we may terminate the service and we are not liable .", [0, 1]),
        ("welcome to our service .", []),
        ("contact us by email .", []),
    ])
]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tinybert")
    wordpiece = BertWordPieceTokenizer(lowercase=True)
    wordpiece.train_from_iterator([r["text"] for r in RECORDS] * 5, vocab_size=200)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=wordpiece._tokenizer, unk_token="[UNK]",
                                        pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
                                        mask_token="[MASK]")
    tokenizer.save_pretrained(path)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    BertModel(config).save_pretrained(path)
    return str(path)


def test_probs_to_labels():
    probs = np.array([[0.9, 0.2, 0.5], [0.1, 0.1, 0.1]])
    assert probs_to_labels(probs) == [[0, 2], []]
    assert probs_to_labels(probs, threshold=0.95) == [[], []]


def test_loader_keeps_labels_aligned(tiny_model_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    loader = make_loader(RECORDS, tokenizer, len(NAMES), max_length=32, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["labels"].tolist()[:4] == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]


def test_train_and_predict(tiny_model_dir):
    lines = []
    model, tokenizer, history = train(tiny_model_dir, RECORDS, RECORDS, NAMES, seed=1, epochs=2,
                                      lr=1e-3, batch_size=2, max_length=32, patience=2,
                                      device=torch.device("cpu"), log=lines.append)
    assert len(history["epochs"]) == 2 and 1 <= history["best_epoch"] <= 2
    loader = make_loader(RECORDS, tokenizer, len(NAMES), 32, 8, False)
    probs = predict_probs(model, loader, torch.device("cpu"))
    assert probs.shape == (len(RECORDS), len(NAMES))
    assert ((probs >= 0) & (probs <= 1)).all()
