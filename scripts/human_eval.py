"""Human evaluation of the explanations: rating forms for two annotators, then scoring.

`make` draws a fixed sample of 60 answers of the main system on the test set and writes one Excel
rating form per annotator (same items, same order) and a hidden key:
- 30 clauses the system flagged with a correct category,
- 20 false alarms (flagged, but fair for the experts),
- 5 sentences the LLM correctly judged fair, and 5 unfair clauses it judged fair.
The forms do not show the expert labels or the group of each item, and the items are shuffled.
Each annotator first gives their own judgment of the clause (Q1), then rates the system's
explanation (Q2-Q5). The two annotators must work independently.

`score` reads the two filled forms and reports: agreement between the annotators (Cohen's kappa,
linear weights for ordered answers), the rated quality by group, and how the annotators' own
judgment compares with the expert labels, for example whether they consider the false alarms
unfair.

Usage (from the project root):
    python scripts/human_eval.py make                     # -> results/human_eval/form_alireza.xlsx, form_kiana.xlsx
    python scripts/human_eval.py score                    # after both forms are filled in
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from claudette_rag.evaluation import load_gold
from claudette_rag.knowledge_base import load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUT = RESULTS / "human_eval"
RUN = "rag_gpt-4.1-mini-2025-04-14_e8r5p6_test"
FINAL = "gated_k8m3_rag_gpt-4.1-mini-2025-04-14_e8r5p6_test"
GROUPS = {"correct flag": 30, "false alarm": 20, "correct fair": 5, "missed by the LLM": 5}
QUESTIONS = {
    "Q1_is_unfair": ("Q1. Your own judgment, BEFORE reading the system's answer: is this sentence a potentially "
                     "unfair clause?", ["yes", "no", "unsure"]),
    "Q2_describes_clause": ("Q2. Does the explanation describe correctly what the sentence says?",
                            ["yes", "partly", "no"]),
    "Q3_legal_reasoning": ("Q3. Is the legal reasoning correct (does the reason given support the verdict under "
                           "EU consumer law and the category definitions)?", ["yes", "partly", "no"]),
    "Q4_sources_relevant": ("Q4. Are the cited sources relevant to the reasoning?", ["all", "some", "none"]),
    "Q5_helpful": ("Q5. Would this explanation help a consumer understand the issue?", ["yes", "somewhat", "no"]),
}
ORDINAL = {"yes": 1.0, "partly": 0.5, "somewhat": 0.5, "no": 0.0, "all": 1.0, "some": 0.5, "none": 0.0}
COLUMNS = ["item", "previous sentence", "SENTENCE", "next sentence", "Q1_is_unfair", "system verdict",
           "explanation", "cited sources", "Q2_describes_clause", "Q3_legal_reasoning", "Q4_sources_relevant",
           "Q5_helpful", "comment"]


def load_inputs():
    categories = sorted(load_knowledge_base(PROJECT_ROOT / "knowledge_base")["categories"],
                        key=lambda c: c["label_index"])
    names = [c["name"] for c in categories]
    gold_list = sorted(load_gold(PROJECT_ROOT, "unfair_tos", "test"), key=lambda g: int(g["id"].split("-")[1]))
    with (RESULTS / "explanations" / f"{RUN}.jsonl").open(encoding="utf-8") as f:
        expl = {r["id"]: r for r in map(json.loads, f)}
    with (RESULTS / "predictions" / f"{FINAL}.jsonl").open(encoding="utf-8") as f:
        final = {r["id"]: set(r["labels"]) for r in map(json.loads, f)}
    with (PROJECT_ROOT / "data" / "processed" / "kb_documents.jsonl").open(encoding="utf-8") as f:
        docs = {d["id"]: d for d in map(json.loads, f)}
    return names, gold_list, expl, final, docs


def make(annotators: list[str]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    names, gold_list, expl, final, docs = load_inputs()
    position = {g["id"]: i for i, g in enumerate(gold_list)}
    groups = defaultdict(list)
    for g in gold_list:
        sid, gold, pred = g["id"], set(g["labels"]), final[g["id"]]
        llm_fair = not expl[sid]["predicted"]
        if pred and gold & pred:
            groups["correct flag"].append(sid)
        elif pred and not gold:
            groups["false alarm"].append(sid)
        elif llm_fair and not gold:
            groups["correct fair"].append(sid)
        elif llm_fair and gold:
            groups["missed by the LLM"].append(sid)
    rng = random.Random(42)
    sample = [(group, sid) for group, n in GROUPS.items() for sid in rng.sample(groups[group], n)]
    rng.shuffle(sample)

    def text_at(i: int) -> str:
        return gold_list[i]["text"].strip() if 0 <= i < len(gold_list) else ""

    def source(doc_id: str) -> str:
        d = docs[doc_id]
        if d["type"] == "example":
            label = "unfair: " + ", ".join(d["categories"]) if d["categories"] else "fair"
            return f'[{doc_id}] similar sentence, expert label {label}: "{d["text"]}"'
        return f"[{doc_id}] {d['text']}"

    rows, key = [], []
    for item, (group, sid) in enumerate(sample, start=1):
        i, r = position[sid], expl[sid]
        verdict = ("UNFAIR: " + ", ".join(names[x] for x in sorted(final[sid]))) if final[sid] else "FAIR"
        rows.append([item, text_at(i - 1), text_at(i), text_at(i + 1), None, verdict, r["explanation"],
                     "\n".join(source(c) for c in r["cited"]) or "(none)", None, None, None, None, None])
        key.append({"item": item, "id": sid, "group": group,
                    "gold": [names[x] for x in gold_list[i]["labels"]],
                    "system": [names[x] for x in sorted(final[sid])]})

    OUT.mkdir(parents=True, exist_ok=True)
    instructions = [
        "Human evaluation of the explanations of Claudette-RAG",
        "",
        "Each row is one sentence from the Terms of Service of an online platform, with the verdict and the "
        "explanation produced by our system. The previous and next sentences are given for context only.",
        "Work alone: do not discuss the items with the other annotator until both forms are complete.",
        "For each row: read the SENTENCE and answer Q1 first, before reading the system's verdict and explanation. "
        "Then read the verdict, the explanation and the cited sources, and answer Q2-Q5. Choose the answers from "
        "the drop-down lists; use the comment column for anything worth noting (e.g. a wrong claim in the explanation).",
        "",
    ] + [text for text, _ in QUESTIONS.values()] + [
        "",
        "Category definitions: knowledge_base/categories.yaml (or the category cards in the README). "
        "A clause is potentially unfair if, contrary to good faith, it causes a significant imbalance in the "
        "parties' rights to the detriment of the consumer (Directive 93/13/EEC, Article 3(1)).",
    ]
    for annotator in annotators:
        wb = Workbook()
        ws_info = wb.active
        ws_info.title = "Instructions"
        for line in instructions:
            ws_info.append([line])
        ws_info.column_dimensions["A"].width = 140
        for cell in ws_info["A"]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws_info["A1"].font = Font(bold=True, size=13)

        ws = wb.create_sheet("Items")
        ws.append(COLUMNS)
        for row in rows:
            ws.append(row)
        widths = {"A": 6, "B": 30, "C": 50, "D": 30, "E": 12, "F": 22, "G": 50, "H": 60,
                  "I": 12, "J": 12, "K": 12, "L": 12, "M": 30}
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        header_fill = PatternFill("solid", fgColor="DDDDDD")
        answer_fill = PatternFill("solid", fgColor="FFF6D5")
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            row[2].font = Font(bold=True)
        for column_name, (_, options) in QUESTIONS.items():
            letter = chr(ord("A") + COLUMNS.index(column_name))
            validation = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True)
            ws.add_data_validation(validation)
            validation.add(f"{letter}2:{letter}{len(rows) + 1}")
            for r in range(2, len(rows) + 2):
                ws[f"{letter}{r}"].fill = answer_fill
        ws.freeze_panes = "E2"
        path = OUT / f"form_{annotator}.xlsx"
        wb.save(path)
        print(f"Saved -> {path.relative_to(PROJECT_ROOT).as_posix()}")
    key_path = OUT / "key.json"
    key_path.write_text(json.dumps({"run": RUN, "system": FINAL, "seed": 42, "groups": GROUPS, "items": key},
                                   indent=2), encoding="utf-8")
    print(f"Saved -> {key_path.relative_to(PROJECT_ROOT).as_posix()} (do not open it before rating)")


def read_form(path: Path) -> dict[int, dict]:
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True)["Items"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    answers = {}
    for row in rows[1:]:
        record = dict(zip(header, row))
        answers[int(record["item"])] = {q: (str(record[q]).strip().lower() if record[q] else None) for q in QUESTIONS}
    return answers


def score(annotators: list[str]) -> None:
    from sklearn.metrics import cohen_kappa_score

    key = {k["item"]: k for k in json.loads((OUT / "key.json").read_text(encoding="utf-8"))["items"]}
    forms = {a: read_form(OUT / f"form_{a}.xlsx") for a in annotators}
    missing = {a: [i for i, ans in f.items() for q, v in ans.items() if v is None] for a, f in forms.items()}
    for a, m in missing.items():
        if m:
            print(f"Warning: {a} left {len(m)} answers empty (items {sorted(set(m))[:10]}...); they are skipped.")
    a, b = annotators
    report = {"annotators": annotators, "agreement": {}, "quality_by_group": {}, "q1_vs_experts": {}}

    print("Agreement between the annotators")
    for q, (_, options) in QUESTIONS.items():
        pairs = [(forms[a][i][q], forms[b][i][q]) for i in key if forms[a][i][q] and forms[b][i][q]]
        x, y = zip(*pairs)
        ordered = q != "Q1_is_unfair"
        kappa = cohen_kappa_score(x, y, labels=options, weights="linear" if ordered else None)
        raw = sum(u == v for u, v in pairs) / len(pairs)
        report["agreement"][q] = {"n": len(pairs), "raw_agreement": raw, "cohen_kappa": kappa,
                                  "weights": "linear" if ordered else "none"}
        print(f"  {q:22s} raw agreement {100 * raw:5.1f}%   kappa {kappa:5.2f}{' (linear weights)' if ordered else ''}")

    print("\nRated quality by group (mean of both annotators: yes/all = 1, partly/some/somewhat = 0.5, no/none = 0)")
    for group in GROUPS:
        items = [i for i, k in key.items() if k["group"] == group]
        cells = {}
        for q in list(QUESTIONS)[1:]:
            values = [ORDINAL[forms[x][i][q]] for x in annotators for i in items if forms[x][i][q] in ORDINAL]
            cells[q] = sum(values) / len(values) if values else float("nan")
        report["quality_by_group"][group] = {"n": len(items), **cells}
        print(f"  {group:20s} (n={len(items):2d}) " + "  ".join(f"{q[:2]} {v:.2f}" for q, v in cells.items()))

    print("\nThe annotators' own judgment (Q1) compared with the expert labels")
    for group in GROUPS:
        items = [i for i, k in key.items() if k["group"] == group]
        c = Counter(forms[x][i]["Q1_is_unfair"] for x in annotators for i in items)
        report["q1_vs_experts"][group] = dict(c)
        experts = "unfair" if group in ("correct flag", "missed by the LLM") else "fair"
        print(f"  {group:20s} experts say {experts:6s} -> annotators: "
              + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    out = OUT / "human_eval_scores.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out.relative_to(PROJECT_ROOT).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["make", "score"])
    parser.add_argument("--annotators", default="alireza,kiana", help="two names, comma-separated")
    args = parser.parse_args()
    annotators = [a.strip().lower() for a in args.annotators.split(",")]
    if len(annotators) != 2:
        raise SystemExit("Give exactly two annotators.")
    make(annotators) if args.command == "make" else score(annotators)


if __name__ == "__main__":
    main()
