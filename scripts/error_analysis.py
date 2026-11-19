"""What kind of mistakes does the main system make? (automatic first pass, no API call)

Every test sentence where the main system (RAG + evidence gate) disagrees with the experts is
classified, with its neighbouring sentences for context:

False alarms (the experts say fair, the system flags it):
- near miss: an expert-labelled clause of the same category is within 2 sentences, so the
  system found the right clause but flagged a neighbouring sentence of it (annotation
  granularity: the experts label one sentence, the unfair term may span several);
- other false alarm: no such clause nearby (topic false alarm or a disagreement with the experts).
Missed clauses (the experts say unfair, the system says fair):
- removed by the gate: the LLM found the right category but fewer than m of the k nearest
  expert-labelled sentences have it;
- missed by the LLM: the LLM judged the sentence fair;
- wrong category: the LLM flagged a different category, which the gate also removed.
Wrong category: unfair for both, but with different categories (confusion pairs are counted).

Sentences that look like list items (ending with ";" or ":", or following a sentence that ends
with ":") are marked, because a list item read alone loses its context.

UNFAIR-ToS keeps the sentences of each document in order, but it has no document ids: near the
border between two documents the "neighbours" may come from another document.

The table results/error_analysis_<system>.csv (one row per error, with context, explanation and
citations, and an empty "manual_type" column) is the input of the manual error analysis.

Usage (from the project root):
    python scripts/error_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from claudette_rag.evaluation import load_gold
from claudette_rag.knowledge_base import load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
WINDOW = 2


def is_list_item(text: str, previous: str) -> bool:
    return text.rstrip().endswith((";", ":")) or previous.rstrip().endswith(":")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run", default="rag_gpt-4.1-mini-2025-04-14_e8r5p6_test", help="RAG run (LLM answers)")
    parser.add_argument("--final", default="gated_k8m3_rag_gpt-4.1-mini-2025-04-14_e8r5p6_test",
                        help="predictions of the system analysed")
    parser.add_argument("--k", type=int, default=8, help="gate: number of nearest examples (for the support count)")
    args = parser.parse_args()

    categories = sorted(load_knowledge_base(PROJECT_ROOT / "knowledge_base")["categories"],
                        key=lambda c: c["label_index"])
    names = [c["name"] for c in categories]
    index_of_name = {n: i for i, n in enumerate(names)}
    gold_records = sorted(load_gold(PROJECT_ROOT, "unfair_tos", "test"), key=lambda g: int(g["id"].split("-")[1]))
    position = {g["id"]: i for i, g in enumerate(gold_records)}
    with (RESULTS / "explanations" / f"{args.run}.jsonl").open(encoding="utf-8") as f:
        expl = {r["id"]: r for r in map(json.loads, f)}
    with (RESULTS / "predictions" / f"{args.final}.jsonl").open(encoding="utf-8") as f:
        final = {r["id"]: set(r["labels"]) for r in map(json.loads, f)}
    with (PROJECT_ROOT / "data" / "processed" / "kb_documents.jsonl").open(encoding="utf-8") as f:
        example_labels = {d["id"]: d["labels"] for d in map(json.loads, f) if d["type"] == "example"}

    def text_at(i: int) -> str:
        return gold_records[i]["text"].strip() if 0 <= i < len(gold_records) else ""

    rows, outcome_counts, type_counts, confusion = [], Counter(), Counter(), Counter()
    list_rate = Counter()
    for g in gold_records:
        sid, i = g["id"], position[g["id"]]
        gold, pred = set(g["labels"]), final[sid]
        llm = {index_of_name[n] for n in expl[sid]["predicted"]}
        list_item = is_list_item(text_at(i), text_at(i - 1))
        list_rate["all"] += 1
        list_rate["all_list"] += list_item
        if gold == pred:
            outcome_counts["correct" if not gold else "correct unfair"] += 1
            continue
        if not gold:
            outcome = "false alarm"
            near = any(set(gold_records[j]["labels"]) & pred
                       for j in range(max(0, i - WINDOW), min(len(gold_records), i + WINDOW + 1)) if j != i)
            auto = "near miss (clause nearby)" if near else "other false alarm"
        elif not pred:
            outcome = "missed clause"
            if gold & llm:
                auto = "removed by the gate"
            elif not llm:
                auto = "missed by the LLM"
            else:
                auto = "LLM wrong category, removed by the gate"
        else:
            outcome = "wrong category"
            auto = "partly right" if gold & pred else "different category"
            for a in gold - pred:
                for b in pred - gold:
                    confusion[(names[a], names[b])] += 1
        outcome_counts[outcome] += 1
        type_counts[(outcome, auto)] += 1
        list_rate[f"{outcome}_n"] += 1
        list_rate[f"{outcome}_list"] += list_item
        neighbours = [example_labels[e] for e in expl[sid]["retrieved"]["examples"][:args.k]]
        rows.append({
            "id": sid, "outcome": outcome, "auto_type": auto, "manual_type": "", "list_item": list_item,
            "gold": "; ".join(names[x] for x in sorted(gold)) or "fair",
            "llm": "; ".join(names[x] for x in sorted(llm)) or "fair",
            "final": "; ".join(names[x] for x in sorted(pred)) or "fair",
            "support_of_llm_categories": "; ".join(f"{names[c]} {sum(c in n for n in neighbours)}/{len(neighbours)}"
                                                   for c in sorted(llm)),
            "previous_2": text_at(i - 2), "previous_1": text_at(i - 1), "sentence": text_at(i),
            "next_1": text_at(i + 1), "next_2": text_at(i + 2),
            "explanation": expl[sid]["explanation"], "cited": " ".join(expl[sid]["cited"]),
        })

    print(f"Errors of {args.final} on the test set ({len(gold_records)} sentences)\n")
    for outcome in ("false alarm", "missed clause", "wrong category"):
        n = outcome_counts[outcome]
        print(f"{outcome}: {n}  (list items: {list_rate[f'{outcome}_list']})")
        for (o, auto), c in sorted(type_counts.items(), key=lambda x: -x[1]):
            if o == outcome:
                print(f"    {auto:42s} {c:3d}")
    print(f"\nList items among all test sentences: {list_rate['all_list']} of {list_rate['all']} "
          f"({100 * list_rate['all_list'] / list_rate['all']:.1f}%)")
    if confusion:
        print("Category confusions (gold -> predicted):",
              ", ".join(f"{a} -> {b} ({c})" for (a, b), c in confusion.most_common()))

    out_csv = RESULTS / f"error_analysis_{args.final}.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: opens cleanly in Excel
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"system": args.final, "outcomes": dict(outcome_counts),
               "types": {f"{o} | {a}": c for (o, a), c in type_counts.items()},
               "list_items": dict(list_rate),
               "confusions": {f"{a} -> {b}": c for (a, b), c in confusion.items()}}
    out_json = RESULTS / f"error_analysis_{args.final}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_csv.relative_to(PROJECT_ROOT).as_posix()} (one row per error, for the manual analysis)")


if __name__ == "__main__":
    main()
