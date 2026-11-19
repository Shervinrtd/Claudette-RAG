"""Evidence gate: keep an LLM category only when expert-labelled precedents support it.

The validation answers of the RAG system show that the LLM still flags many sentences on
their topic alone, even when most of the retrieved sentences were labelled fair by the
experts. The gate is a transparent rule applied after the LLM: a category predicted by the
LLM is kept only if at least m of the k most similar training sentences have that category.
A flagged clause then has both a legal explanation with citations and expert precedents.

No API call: the script reads the explanation file of a RAG run (retrieved ids) and the
knowledge base, so it can be re-run for free.

Choice of (k, m), fixed in advance: on the validation set, the highest LexGLUE macro-F1
among the settings with unfair recall >= 80%, because for consumer protection a missed
unfair clause is worse than a false alarm (if no setting reaches 80%: the highest recall). The choice is saved and reused, unchanged, on
the test set and on the multilingual data.

Usage (from the project root):
    python scripts/gate_rag.py rag_gpt-4.1-mini-2025-04-14_e8r5p6_validation   # grid + choice on validation
    python scripts/gate_rag.py rag_gpt-4.1-mini-2025-04-14_e8r5p6_test         # apply the validation choice
    python scripts/gate_rag.py <run> --k 5 --m 2                               # a fixed setting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from claudette_rag.evaluation import evaluate, format_report, load_gold
from claudette_rag.knowledge_base import load_knowledge_base
from claudette_rag.llm import file_safe
from claudette_rag.rag import choose_gate, gate_by_neighbours

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
GRID = [(k, m) for k in (3, 5, 8) for m in (1, 2, 3)]
MIN_RECALL = 0.80


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("run", help="name of a RAG run (results/<run>.json)")
    parser.add_argument("--k", type=int, help="number of nearest training sentences")
    parser.add_argument("--m", type=int, help="how many of them must have the category")
    args = parser.parse_args()

    info = json.loads((RESULTS / f"{args.run}.json").read_text(encoding="utf-8"))["run"]
    categories = sorted(load_knowledge_base(PROJECT_ROOT / "knowledge_base")["categories"],
                        key=lambda c: c["label_index"])
    label_names = [c["name"] for c in categories]
    index_of = {name: i for i, name in enumerate(label_names)}

    with (PROJECT_ROOT / info["explanations"].replace("\\", "/")).open(encoding="utf-8") as f:   # Windows paths
        rows = [json.loads(line) for line in f]
    with (PROJECT_ROOT / "data" / "processed" / "kb_documents.jsonl").open(encoding="utf-8") as f:
        example_labels = {d["id"]: d["labels"] for d in map(json.loads, f) if d["type"] == "example"}
    if not any(r["retrieved"]["examples"] for r in rows):
        raise SystemExit("The gate needs retrieved examples: this run has none (--k-examples 0).")
    gold_by_id = {g["id"]: g["labels"] for g in
                  load_gold(PROJECT_ROOT, info["dataset"], info["split"], info.get("language"))}
    gold = [gold_by_id[r["id"]] for r in rows]
    llm = [sorted(index_of[n] for n in r["predicted"]) for r in rows]
    neighbours = [[example_labels[i] for i in r["retrieved"]["examples"]] for r in rows]

    def gated(k: int, m: int) -> list[list[int]]:
        return [gate_by_neighbours(p, n, k, m) for p, n in zip(llm, neighbours)]

    choice_path = RESULTS / f"gate_choice_{file_safe(info['model'])}_e{info['retrieval']['k_examples']}" \
                            f"r{info['retrieval']['k_rationales']}p{info['retrieval']['max_provisions']}.json"
    if args.k and args.m:
        k, m, how = args.k, args.m, "fixed on the command line"
    elif info["split"] == "validation" and info["dataset"] == "unfair_tos" and not info.get("sample"):
        grid = []
        print(f"Evidence gate on {args.run} ({len(rows)} sentences)\n"
              f"{'k':>3} {'m':>3} {'macro-F1':>9} {'precision':>10} {'recall':>7}")
        for k, m in GRID:
            if m > k or k > max(len(n) for n in neighbours):
                continue
            h = evaluate(gold, gated(k, m), label_names, n_bootstrap=0)["headline"]
            grid.append({"k": k, "m": m, **{key: h[key] for key in
                         ("lexglue_macro_f1", "binary_precision", "binary_recall")}})
            print(f"{k:>3} {m:>3} {100 * h['lexglue_macro_f1']:9.1f} {100 * h['binary_precision']:10.1f} "
                  f"{100 * h['binary_recall']:7.1f}")
        best, how = choose_gate(grid, MIN_RECALL)
        k, m = best["k"], best["m"]
        choice_path.write_text(json.dumps({"source_run": args.run, "rule": how, "k": k, "m": m,
                                           "grid": grid}, indent=2), encoding="utf-8")
        print(f"\nChosen: k={k}, m={m} ({how}) -> {choice_path.relative_to(PROJECT_ROOT)}")
    else:
        if not choice_path.exists():
            raise SystemExit(f"No validation choice in {choice_path.name}: run the gate on the full "
                             "validation run first, or give --k and --m.")
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        k, m, how = choice["k"], choice["m"], f"chosen on validation ({choice['source_run']})"

    pred = gated(k, m)
    name = f"gated_k{k}m{m}_{args.run}"
    pred_path = RESULTS / "predictions" / f"{name}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r, labels in zip(rows, pred):
            f.write(json.dumps({"id": r["id"], "labels": labels}) + "\n")
    report = evaluate(gold, pred, label_names)
    report["run"] = {**info, "name": name, "system": "RAG + evidence gate", "source_run": args.run,
                     "gate": {"k": k, "m": m, "choice": how,
                              "categories_removed": sum(len(a) - len(b) for a, b in zip(llm, pred)),
                              "sentences_unflagged": sum(bool(a) and not b for a, b in zip(llm, pred))},
                     "predictions": pred_path.relative_to(PROJECT_ROOT).as_posix()}
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    gate = report["run"]["gate"]
    print("\n" + format_report(f"{name} (k={k}, m={m})", report))
    print(f"\nThe gate removed {gate['categories_removed']} predicted categories; "
          f"{gate['sentences_unflagged']} sentences flagged by the LLM are now fair.")
    print(f"Saved -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
