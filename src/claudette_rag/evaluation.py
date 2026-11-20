"""Evaluation shared by every system (TF-IDF, BERT, plain LLM, RAG).

All systems write their predictions in the same format (one JSON object per line):

    {"id": "test-17", "labels": [1, 3]}        # indices of the predicted unfair categories
    {"id": "test-18", "labels": []}            # predicted fair

and are scored with the same functions, so their results are directly comparable.

Metrics
-------
- LexGLUE micro-F1 and macro-F1 ("8+1"): as in Chalkidis et al. (2022), an extra
  label "fair" (= no unfair category) is added before computing F1 over the 9 labels.
  This makes our numbers comparable with the published ones.
- Macro-F1 over the 8 unfair categories only (without the "fair" label).
- Per category: precision, recall, F1 and support (number of gold sentences).
- Binary detection (any unfair category vs fair): precision, recall, F1. For consumer
  protection, recall matters most: a missed unfair clause is worse than a false alarm.
- Bootstrap confidence intervals: the test set has few examples of some categories,
  so differences between systems must be read together with their uncertainty.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FAIR = "Fair (no unfair category)"


# ----------------------------------------------------------------------------- loading
def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gold(project_root: Path, dataset: str = "unfair_tos", split: str = "test",
              language: str | None = None, exclude_pinc_only: bool = True,
              exclude_in_unfair_tos: bool = True) -> list[dict]:
    """Gold records ({"id", "text", "labels", ...}) for UNFAIR-ToS or the multilingual set."""
    root = Path(project_root)
    if dataset == "unfair_tos":
        return read_jsonl(root / "data" / "raw" / f"unfair_tos_{split}.jsonl")
    if dataset == "multilingual":
        records = read_jsonl(root / "data" / "processed" / "multilingual_eval.jsonl")
        if language:
            records = [r for r in records if r["language"] == language]
        if exclude_pinc_only:
            records = [r for r in records if not r["pinc_only"]]
        if exclude_in_unfair_tos:
            records = [r for r in records if not r["in_unfair_tos"]]
        return records
    raise ValueError(f"unknown dataset '{dataset}'")


def align(gold: list[dict], predictions: list[dict]) -> tuple[list[list[int]], list[list[int]]]:
    """Match predictions to gold records by id; fail loudly on missing or extra ids."""
    pred_by_id: dict[str, list[int]] = {}
    for p in predictions:
        if p["id"] in pred_by_id:
            raise ValueError(f"duplicate prediction for id '{p['id']}'")
        pred_by_id[p["id"]] = list(p["labels"])
    gold_ids = [g["id"] for g in gold]
    missing = [i for i in gold_ids if i not in pred_by_id]
    extra = set(pred_by_id) - set(gold_ids)
    if missing:
        raise ValueError(f"{len(missing)} gold sentences have no prediction (e.g. '{missing[0]}')")
    if extra:
        raise ValueError(f"{len(extra)} predictions do not match any gold sentence "
                         f"(e.g. '{next(iter(extra))}')")
    return [list(g["labels"]) for g in gold], [pred_by_id[i] for i in gold_ids]


# ----------------------------------------------------------------------------- metrics
def to_matrix(label_lists: list[list[int]], n_labels: int) -> np.ndarray:
    """List of label-index lists -> boolean matrix (n_sentences, n_labels)."""
    y = np.zeros((len(label_lists), n_labels), dtype=bool)
    for i, labels in enumerate(label_lists):
        for label in labels:
            if not 0 <= label < n_labels:
                raise ValueError(f"label {label} out of range 0..{n_labels - 1}")
            y[i, label] = True
    return y


def _prf(tp, fp, fn):
    """Precision, recall, F1 (0 when undefined, as in scikit-learn with zero_division=0)."""
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1


def _counts(y_true: np.ndarray, y_pred: np.ndarray):
    tp = (y_true & y_pred).sum(axis=0)
    fp = (~y_true & y_pred).sum(axis=0)
    fn = (y_true & ~y_pred).sum(axis=0)
    return tp, fp, fn


def _with_fair_column(y: np.ndarray) -> np.ndarray:
    return np.hstack([y, ~y.any(axis=1, keepdims=True)])


def headline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """The summary numbers (fast: used inside the bootstrap)."""
    tp, fp, fn = _counts(_with_fair_column(y_true), _with_fair_column(y_pred))
    _, _, f1_9 = _prf(tp, fp, fn)
    _, _, micro = _prf(tp.sum(), fp.sum(), fn.sum())
    _, _, f1_8 = _prf(tp[:-1], fp[:-1], fn[:-1])
    b_true, b_pred = y_true.any(axis=1), y_pred.any(axis=1)
    bp, br, bf = _prf((b_true & b_pred).sum(), (~b_true & b_pred).sum(), (b_true & ~b_pred).sum())
    return {
        "lexglue_micro_f1": float(micro),
        "lexglue_macro_f1": float(f1_9.mean()),
        "macro_f1_unfair_categories": float(f1_8.mean()),
        "binary_precision": float(bp),
        "binary_recall": float(br),
        "binary_f1": float(bf),
    }


def bootstrap(y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 1000,
              seed: int = 42) -> dict[str, list[float]]:
    """95% bootstrap confidence intervals of the headline metrics (resampling sentences)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws: dict[str, list[float]] = {}
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        for name, value in headline_metrics(y_true[idx], y_pred[idx]).items():
            draws.setdefault(name, []).append(value)
    return {name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for name, v in draws.items()}


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     n_samples: int = 1000, seed: int = 42) -> dict[str, dict[str, float]]:
    """Difference A - B of the headline metrics on the same resampled sentences.

    For each metric: the observed difference, its 95% interval, and a two-sided p-value
    (twice the share of resamples where the difference has the opposite sign). Because
    both systems are scored on the same resample, the sentences that are hard for both
    cancel out, which makes this test sharper than comparing two separate intervals.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed_a, observed_b = headline_metrics(y_true, pred_a), headline_metrics(y_true, pred_b)
    draws: dict[str, list[float]] = {name: [] for name in observed_a}
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        a, b = headline_metrics(y_true[idx], pred_a[idx]), headline_metrics(y_true[idx], pred_b[idx])
        for name in draws:
            draws[name].append(a[name] - b[name])
    out = {}
    for name, values in draws.items():
        values = np.asarray(values)
        diff = observed_a[name] - observed_b[name]
        opposite = (values <= 0).mean() if diff > 0 else (values >= 0).mean()
        out[name] = {"difference": float(diff), "low": float(np.percentile(values, 2.5)),
                     "high": float(np.percentile(values, 97.5)), "p_value": float(min(1.0, 2 * opposite))}
    return out


def evaluate(gold_labels: list[list[int]], pred_labels: list[list[int]],
             label_names: list[str], n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """Full evaluation report as a JSON-serialisable dict."""
    n_labels = len(label_names)
    y_true, y_pred = to_matrix(gold_labels, n_labels), to_matrix(pred_labels, n_labels)

    tp, fp, fn = _counts(_with_fair_column(y_true), _with_fair_column(y_pred))
    precision, recall, f1 = _prf(tp, fp, fn)
    names = list(label_names) + [FAIR]
    per_label = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
               "support": int(tp[i] + fn[i]), "predicted": int(tp[i] + fp[i])}
        for i, name in enumerate(names)
    }

    b_true, b_pred = y_true.any(axis=1), y_pred.any(axis=1)
    report = {
        "n_sentences": int(len(y_true)),
        "n_gold_unfair": int(b_true.sum()),
        "n_predicted_unfair": int(b_pred.sum()),
        "headline": headline_metrics(y_true, y_pred),
        "binary_confusion": {
            "true_positive": int((b_true & b_pred).sum()), "false_positive": int((~b_true & b_pred).sum()),
            "false_negative": int((b_true & ~b_pred).sum()), "true_negative": int((~b_true & ~b_pred).sum()),
        },
        "per_label": per_label,
    }
    if n_bootstrap:
        report["confidence_95"] = bootstrap(y_true, y_pred, n_bootstrap, seed)
    return report


def format_report(name: str, report: dict) -> str:
    """Human-readable summary for the console."""
    h, ci = report["headline"], report.get("confidence_95", {})

    def fmt(key):
        value = f"{100 * h[key]:5.1f}"
        if key in ci:
            low, high = ci[key]
            value += f"  [{100 * low:4.1f}-{100 * high:4.1f}]"
        return value

    lines = [
        f"=== {name}: {report['n_sentences']} sentences, {report['n_gold_unfair']} unfair "
        f"(predicted unfair: {report['n_predicted_unfair']})",
        f"LexGLUE micro-F1 (8+1)      {fmt('lexglue_micro_f1')}",
        f"LexGLUE macro-F1 (8+1)      {fmt('lexglue_macro_f1')}",
        f"Macro-F1, 8 categories      {fmt('macro_f1_unfair_categories')}",
        f"Unfair detection precision  {fmt('binary_precision')}",
        f"Unfair detection recall     {fmt('binary_recall')}",
        f"Unfair detection F1         {fmt('binary_f1')}",
        "",
        f"{'category':<27}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}",
    ]
    for label, m in report["per_label"].items():
        lines.append(f"{label:<27}{100 * m['precision']:7.1f}{100 * m['recall']:7.1f}"
                     f"{100 * m['f1']:7.1f}{m['support']:9d}")
    return "\n".join(lines)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test for two paired conditions on the same items.

    b: items where only the first condition gives the outcome, c: only the second. Under the null
    hypothesis both are equally likely, so min(b, c) follows a binomial(b + c, 1/2).
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ----------------------------------------------------------------------------- group fairness
def group_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Rates used by the group-fairness criteria (an unfair clause is the "positive" case).

    recall: unfair sentences flagged (equality of opportunity compares it across groups);
    false_alarm_rate: fair sentences flagged (equal false positive rates; with recall: equalized odds);
    precision: flags that are right (1 - precision is the conditional use error of positive predictions);
    flagged: share of all sentences flagged (statistical parity); macro_f1: LexGLUE macro-F1.
    """
    h = headline_metrics(y_true, y_pred)
    b_true, b_pred = y_true.any(axis=1), y_pred.any(axis=1)
    fair = ~b_true
    return {"recall": h["binary_recall"], "precision": h["binary_precision"],
            "false_alarm_rate": float((b_pred & fair).sum() / max(1, fair.sum())),
            "flagged": float(b_pred.mean()), "macro_f1": h["lexglue_macro_f1"]}


def bootstrap_gap(y_ref: np.ndarray, preds_ref: list[np.ndarray], y_grp: np.ndarray, preds_grp: list[np.ndarray],
                  n_samples: int = 1000, seed: int = 42) -> dict[str, dict[str, float]]:
    """Difference group - reference of every rate, with a 95% interval and a two-sided p-value.

    The two groups are different sentences, so they are resampled independently. With several
    prediction sets per group (e.g. seeds of a fine-tuned model) the rate is their mean.
    """
    def mean_rates(y, preds, idx):
        rates = [group_rates(y[idx], p[idx]) for p in preds]
        return {k: float(np.mean([r[k] for r in rates])) for k in rates[0]}

    rng = np.random.default_rng(seed)
    observed = {k: v - mean_rates(y_ref, preds_ref, np.arange(len(y_ref)))[k]
                for k, v in mean_rates(y_grp, preds_grp, np.arange(len(y_grp))).items()}
    draws: dict[str, list[float]] = {k: [] for k in observed}
    for _ in range(n_samples):
        r = mean_rates(y_ref, preds_ref, rng.integers(0, len(y_ref), len(y_ref)))
        g = mean_rates(y_grp, preds_grp, rng.integers(0, len(y_grp), len(y_grp)))
        for k in draws:
            draws[k].append(g[k] - r[k])
    out = {}
    for k, values in draws.items():
        values = np.asarray(values)
        diff = observed[k]
        opposite = (values <= 0).mean() if diff > 0 else (values >= 0).mean()
        out[k] = {"difference": diff, "low": float(np.percentile(values, 2.5)),
                  "high": float(np.percentile(values, 97.5)), "p_value": float(min(1.0, 2 * opposite))}
    return out
