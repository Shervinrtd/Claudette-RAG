"""Tests for claudette_rag.evaluation (run with: python -m pytest)."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from claudette_rag.evaluation import align, evaluate, headline_metrics, to_matrix

NAMES = ["A", "B", "C"]


def test_hand_computed_example():
    # 4 sentences, 3 categories. Gold: [A], [], [B], [A, C]; predicted: [A], [B], [], [A]
    gold = [[0], [], [1], [0, 2]]
    pred = [[0], [1], [], [0]]
    report = evaluate(gold, pred, NAMES, n_bootstrap=0)
    per = report["per_label"]

    # A: tp=2 -> P=R=F1=1
    assert per["A"]["f1"] == pytest.approx(1.0)
    # B: tp=0, fp=1 (sentence 2), fn=1 (sentence 3) -> F1=0
    assert per["B"]["f1"] == pytest.approx(0.0)
    # C: missed once -> F1=0, support 1
    assert per["C"]["f1"] == 0.0 and per["C"]["support"] == 1
    # Fair label: gold fair = sentence 2; predicted fair = sentence 3 -> tp=0 -> F1=0
    assert report["per_label"]["Fair (no unfair category)"]["support"] == 1

    # Micro over 4 labels: tp=2, fp=2 (B on s2, fair on s3), fn=3 (fair s2, B s3, C s4)
    p, r = 2 / 4, 2 / 5
    assert report["headline"]["lexglue_micro_f1"] == pytest.approx(2 * p * r / (p + r))
    # Binary: gold unfair = s1, s3, s4; predicted unfair = s1, s2, s4 -> tp=2, fp=1, fn=1
    assert report["headline"]["binary_precision"] == pytest.approx(2 / 3)
    assert report["headline"]["binary_recall"] == pytest.approx(2 / 3)
    assert report["binary_confusion"] == {"true_positive": 2, "false_positive": 1,
                                          "false_negative": 1, "true_negative": 0}


def test_matches_scikit_learn_lexglue_convention():
    rng = np.random.default_rng(0)
    n, k = 500, 8
    y_true = rng.random((n, k)) < 0.05
    y_pred = rng.random((n, k)) < 0.05
    gold = [list(np.flatnonzero(row)) for row in y_true]
    pred = [list(np.flatnonzero(row)) for row in y_pred]

    # LexGLUE: append a column that is 1 when a sentence has no label
    t9 = np.hstack([y_true, ~y_true.any(1, keepdims=True)])
    p9 = np.hstack([y_pred, ~y_pred.any(1, keepdims=True)])
    h = headline_metrics(to_matrix(gold, k), to_matrix(pred, k))
    assert h["lexglue_micro_f1"] == pytest.approx(f1_score(t9, p9, average="micro", zero_division=0))
    assert h["lexglue_macro_f1"] == pytest.approx(f1_score(t9, p9, average="macro", zero_division=0))
    assert h["macro_f1_unfair_categories"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0))


def test_perfect_and_always_fair():
    gold = [[0], [], [1, 2], []]
    perfect = evaluate(gold, gold, NAMES, n_bootstrap=0)["headline"]
    assert perfect["lexglue_micro_f1"] == pytest.approx(1.0)
    assert perfect["binary_recall"] == pytest.approx(1.0)

    always_fair = evaluate(gold, [[] for _ in gold], NAMES, n_bootstrap=0)["headline"]
    assert always_fair["binary_recall"] == 0.0
    assert always_fair["macro_f1_unfair_categories"] == 0.0


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    gold = [[int(rng.integers(0, 3))] if rng.random() < 0.3 else [] for _ in range(300)]
    pred = [g if rng.random() < 0.8 else [] for g in gold]
    report = evaluate(gold, pred, NAMES, n_bootstrap=200)
    low, high = report["confidence_95"]["binary_recall"]
    assert low <= report["headline"]["binary_recall"] <= high


def test_align_detects_problems():
    gold = [{"id": "a", "labels": [0]}, {"id": "b", "labels": []}]
    assert align(gold, [{"id": "b", "labels": []}, {"id": "a", "labels": [0]}]) == ([[0], []], [[0], []])
    with pytest.raises(ValueError, match="no prediction"):
        align(gold, [{"id": "a", "labels": [0]}])
    with pytest.raises(ValueError, match="duplicate"):
        align(gold, [{"id": "a", "labels": []}, {"id": "a", "labels": []}, {"id": "b", "labels": []}])
    with pytest.raises(ValueError, match="do not match"):
        align(gold, [{"id": "a", "labels": []}, {"id": "b", "labels": []}, {"id": "z", "labels": []}])
    with pytest.raises(ValueError, match="out of range"):
        to_matrix([[5]], 3)


def test_paired_bootstrap():
    import numpy as np
    from claudette_rag.evaluation import paired_bootstrap
    rng = np.random.default_rng(0)
    y = rng.random((300, 8)) < 0.05
    same = paired_bootstrap(y, y.copy(), y.copy(), n_samples=200)
    assert same["lexglue_macro_f1"]["difference"] == 0 and same["lexglue_macro_f1"]["p_value"] == 1.0
    never = np.zeros_like(y)                                  # a system that never flags anything
    better = paired_bootstrap(y, y.copy(), never, n_samples=200)["binary_recall"]
    assert better["difference"] == 1.0 and better["low"] > 0 and better["p_value"] < 0.05


def test_mcnemar_exact():
    from claudette_rag.evaluation import mcnemar_exact
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)                  # 10 discordant pairs, all one way
    assert mcnemar_exact(52, 5) < 1e-9
