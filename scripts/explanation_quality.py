"""Are the explanations grounded in the right evidence? (automatic checks, no API call)

For every clause flagged by a system (by default our main system: RAG + evidence gate on the
test set) and every category it assigns, the script checks what the explanation cites:

- grounded: at least one retrieved item is cited;
- legal: a legal rationale or provision is cited; specific: other than the general test of
  Article 3(1) of Directive 93/13;
- category-consistent: the citations support the category that was assigned, i.e. a rationale
  of that category, a provision linked to that category in the knowledge base (Article 3(1)
  excluded), or an expert-labelled example with that category;
- contradiction: a "clearly fair" rationale is cited to support an unfair verdict;
- invented: a cited id that was not retrieved (hallucinated source).

The checks are split by whether the category is correct according to the experts, to see whether
wrong flags come with weaker evidence. They measure grounding, not the correctness of the legal
reasoning, which would need a review by legal experts.

Usage (from the project root):
    python scripts/explanation_quality.py
    python scripts/explanation_quality.py --run rag_gpt-4.1-mini-2025-04-14_e8r5p6_test --final rag_gpt-4.1-mini-2025-04-14_e8r5p6_test
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from claudette_rag.knowledge_base import load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
GENERIC = {"dir93.art3.1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run", default="rag_gpt-4.1-mini-2025-04-14_e8r5p6_test",
                        help="RAG run whose explanations are analysed")
    parser.add_argument("--final", default="gated_k8m3_rag_gpt-4.1-mini-2025-04-14_e8r5p6_test",
                        help="predictions of the system whose flags are analysed (e.g. after the gate)")
    args = parser.parse_args()

    categories = sorted(load_knowledge_base(PROJECT_ROOT / "knowledge_base")["categories"],
                        key=lambda c: c["label_index"])
    code_of = {c["label_index"]: c["id"] for c in categories}
    index_of_name = {c["name"]: c["label_index"] for c in categories}
    with (PROJECT_ROOT / "data" / "processed" / "kb_documents.jsonl").open(encoding="utf-8") as f:
        docs = {d["id"]: d for d in map(json.loads, f)}
    provision_categories = defaultdict(set)          # provision -> categories of the rationales citing it
    for d in docs.values():
        if d["type"] == "rationale":
            for p in d.get("legal_basis", []):
                provision_categories[p].update(d["categories"])

    with (RESULTS / "explanations" / f"{args.run}.jsonl").open(encoding="utf-8") as f:
        explanations = {r["id"]: r for r in map(json.loads, f)}
    with (RESULTS / "predictions" / f"{args.final}.jsonl").open(encoding="utf-8") as f:
        final = {r["id"]: r["labels"] for r in map(json.loads, f)}

    def supports(doc_id: str, code: str) -> str | None:
        """Which kind of citation supports category `code`, if any."""
        d = docs.get(doc_id)
        if d is None:
            return None
        if d["type"] == "rationale" and code in d["categories"]:
            return "rationale"
        if d["type"] == "provision" and doc_id not in GENERIC and code in provision_categories[doc_id]:
            return "provision"
        if d["type"] == "example" and code in d["categories"]:
            return "example"
        return None

    counts = {"correct": defaultdict(int), "wrong": defaultdict(int)}
    examples_of_problems = []
    for sid, labels in final.items():
        r = explanations[sid]
        gold = {index_of_name[n] for n in r["gold"]}
        cited = r["cited"]
        legal = [c for c in cited if docs[c]["type"] in ("rationale", "provision")]
        for label in labels:
            code = code_of[label]
            kinds = {supports(c, code) for c in cited} - {None}
            group = counts["correct" if label in gold else "wrong"]
            group["flags"] += 1
            group["grounded"] += bool(cited)
            group["legal"] += bool(legal)
            group["specific_legal"] += any(c not in GENERIC for c in legal)
            group["consistent_any"] += bool(kinds)
            group["consistent_legal"] += bool(kinds & {"rationale", "provision"})
            for kind in ("rationale", "provision", "example"):
                group[f"consistent_{kind}"] += kind in kinds
            contradiction = any(docs[c]["type"] == "rationale" and docs[c]["level"] == "clearly_fair" for c in cited)
            group["contradiction"] += contradiction
            group["invented"] += bool(r["invalid_cited"])
            if not kinds and len(examples_of_problems) < 5:
                examples_of_problems.append({"id": sid, "category": code, "gold": r["gold"],
                                             "text": r["text"][:160], "cited": cited})

    rows = [("flags (sentence, category)", "flags"), ("cite a retrieved item", "grounded"),
            ("cite a rationale or provision", "legal"), ("... other than Article 3(1)", "specific_legal"),
            ("evidence consistent with the category", "consistent_any"),
            ("  via a rationale of that category", "consistent_rationale"),
            ("  via a provision linked to that category", "consistent_provision"),
            ("  via an example with that category", "consistent_example"),
            ("legal evidence consistent with the category", "consistent_legal"),
            ("cite a 'clearly fair' rationale", "contradiction"), ("cite an invented id", "invented")]
    total = {k: counts["correct"][k] + counts["wrong"][k] for _, k in rows}
    print(f"Explanation grounding of {args.final}\n(explanations from {args.run})\n")
    print(f"{'':45s} {'all':>12s} {'correct cat.':>14s} {'wrong cat.':>12s}")
    summary = {}
    for title, key in rows:
        cells = []
        for group in (total, counts["correct"], counts["wrong"]):
            n = group["flags"]
            cells.append(f"{group[key]:>5d}" if key == "flags" else f"{group[key]:>4d} ({100 * group[key] / max(1, n):3.0f}%)")
        print(f"{title:45s} {cells[0]:>12s} {cells[1]:>14s} {cells[2]:>12s}")
        summary[key] = {"all": total[key], "correct": counts["correct"][key], "wrong": counts["wrong"][key]}

    print("\nFlags whose citations do not support the category (first 5):")
    for p in examples_of_problems:
        print(f"  {p['id']} [{p['category']}] gold={p['gold']} cited={p['cited']}\n    {p['text']}")
    out = RESULTS / f"explanation_quality_{args.final}.json"
    out.write_text(json.dumps({"explanations": args.run, "predictions": args.final, "counts": summary,
                               "unsupported_examples": examples_of_problems}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
