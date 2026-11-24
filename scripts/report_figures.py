"""Figures of the final report, drawn from the saved results (no model call).

Usage (from the project root, after the experiments):
    python scripts/report_figures.py      # -> report/figures/{test_macro_f1,gate_tradeoff,recall_by_language}.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "report" / "figures"

# validated categorical slots (light mode), text and grid tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT_2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def test_macro_f1() -> None:
    """Macro-F1 on the test set with 95% bootstrap intervals."""
    rows = [  # (label, result file, main system?)
        ("Plain LLM (gpt-4.1-mini)", "llm_zeroshot_gpt-4.1-mini-2025-04-14_test", False),
        ("RAG, no gate", "rag_gpt-4.1-mini-2025-04-14_e8r5p6_test", False),
        ("kNN vote (bge-m3)", "knn_bge-m3_test", False),
        ("TF-IDF + SVM", "tfidf_svm_test", False),
        ("RAG examples only + gate", "gated_k8m3_rag_gpt-4.1-mini-2025-04-14_e8r0p0_test", False),
        ("RAG + gate, Qwen3.6-27B", "gated_k8m3_rag_Qwen-Qwen3.6-27B-FP8_e8r5p6_test", False),
        ("RAG + gate (main system)", "gated_k8m3_rag_gpt-4.1-mini-2025-04-14_e8r5p6_test", True),
        ("Legal-BERT (seeds 1, 2, 3)", [f"bert_legal-bert-base-uncased_seed{s}_test" for s in (1, 2, 3)], False),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for y, (label, names, main) in enumerate(rows):
        names = names if isinstance(names, list) else [names]
        offsets = [0.0] if len(names) == 1 else [-0.22, 0.0, 0.22]
        for off, name in zip(offsets, names):
            report = load(name)
            value = 100 * report["headline"]["lexglue_macro_f1"]
            low, high = (100 * v for v in report["confidence_95"]["lexglue_macro_f1"])
            ax.plot([low, high], [y + off] * 2, color=BLUE, linewidth=1.6 if main else 1.0, solid_capstyle="round")
            ax.plot(value, y + off, "o", color=BLUE, markersize=6 if main else 4.5,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row[2]:
            tick.set_fontweight("bold")
    ax.set_xlabel("LexGLUE macro-F1 on the test set, with 95% bootstrap interval")
    ax.set_xlim(55, 90)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "test_macro_f1.pdf")
    plt.close(fig)


# label positions (offset in points, alignment) for settings whose points are close together
LABEL_OFFSETS = {(8, 3): (10, 7, "left"), (5, 2): (8, -9, "left"), (3, 1): (7, 2, "left"),
                 (8, 2): (-7, -9, "right"), (5, 3): (-7, 2, "right"), (3, 2): (7, -9, "left")}


def gate_tradeoff() -> None:
    """Precision and recall on validation of the evidence-gate settings and of the systems without it."""
    choice = load("gate_choice_gpt-4.1-mini-2025-04-14_e8r5p6")
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.axvline(80, color=NEUTRAL, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(80.4, 95.5, "recall 80%: minimum of the selection rule", color=TEXT_2, fontsize=7.5, va="top")
    for g in choice["grid"]:
        r, p = 100 * g["binary_recall"], 100 * g["binary_precision"]
        chosen = (g["k"], g["m"]) == (choice["k"], choice["m"])
        ax.plot(r, p, "o", color=BLUE, markersize=7 if chosen else 5, markeredgecolor="white", markeredgewidth=0.8)
        if chosen:
            ax.plot(r, p, "o", markersize=13, markerfacecolor="none", markeredgecolor=BLUE, markeredgewidth=1.2)
        dx, dy, ha = LABEL_OFFSETS.get((g["k"], g["m"]), (6, -3, "left"))
        ax.annotate(f"k={g['k']}, m={g['m']}" + (" (chosen)" if chosen else ""), (r, p), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color=TEXT if chosen else TEXT_2, ha=ha,
                    fontweight="bold" if chosen else "normal")
    for name, label in (("rag_gpt-4.1-mini-2025-04-14_e8r5p6_validation", "RAG, no gate"),
                        ("llm_zeroshot_gpt-4.1-mini-2025-04-14_validation", "plain LLM"),
                        ("knn_bge-m3_validation", "kNN vote")):
        h = load(name)["headline"]
        r, p = 100 * h["binary_recall"], 100 * h["binary_precision"]
        ax.plot(r, p, "s", color=NEUTRAL, markersize=5, markeredgecolor="white", markeredgewidth=0.8)
        dx, ha = (-7, "right") if label == "kNN vote" else (7, "left")
        ax.annotate(label, (r, p), xytext=(dx, -3), textcoords="offset points", fontsize=7.5, color=TEXT_2, ha=ha)
    ax.set_xlabel("Recall of unfair sentences (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_xlim(45, 100)
    ax.set_ylim(30, 96)
    fig.tight_layout()
    fig.savefig(OUT / "gate_tradeoff.pdf")
    plt.close(fig)


def recall_by_language() -> None:
    """Recall of unfair clauses per language and system (multilingual set, Qwen3.6-27B for the LLM systems)."""
    data = load("multilingual_fairness_Qwen-Qwen3.6-27B-FP8")["systems"]
    rows = [("Legal-BERT (English only)", "Legal-BERT"), ("TF-IDF + SVM (English only)", "TF-IDF"),
            ("kNN vote (bge-m3)", "kNN vote"), ("Plain LLM", "Plain LLM"), ("RAG, no gate", "RAG ("),
            ("RAG + gate", "RAG + gate")]
    languages = [("en", "English (reference)", NEUTRAL, "o"), ("de", "German", BLUE, "s"),
                 ("it", "Italian", ORANGE, "^"), ("pl", "Polish", AQUA, "D")]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    for y, (label, prefix) in enumerate(rows):
        key = next(k for k in data if k.startswith(prefix))
        langs = data[key]["languages"]
        for i, (code, name, color, marker) in enumerate(languages):
            ax.plot(100 * langs[code]["recall"], y + (i - 1.5) * 0.12, marker, color=color,
                    markersize=6 if marker == "^" else 5,
                    markeredgecolor="white", markeredgewidth=0.6, label=name if y == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-3, 100)
    ax.set_xlabel("Recall of unfair sentences (%)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=4, fontsize=8, handletextpad=0.3,
              columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "recall_by_language.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_macro_f1()
    gate_tradeoff()
    recall_by_language()
    print(f"Figures saved in {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
