"""Do the explanations tell the real reasons of the decisions? Evidence-removal and stability tests.

Faithfulness: for every sentence the RAG system judged unfair (and that cites retrieved items),
the same prompt is sent again in two versions:
- without the cited items: the examples, rationales and provisions the explanation cites are
  removed from the retrieved knowledge;
- control: the same number of retrieved items that were NOT cited are removed instead, chosen at
  random with the same mix of types (examples / rationales / provisions).
If the citations show the real reasons, removing them should change the verdict much more often
than removing items of the same kind that the explanation did not use.

Stability: the unchanged prompts of a random sample of sentences are sent again, to measure how
often the same input gives the same answer (temperature 0 and a fixed seed do not guarantee
identical outputs with the OpenAI API).

The retrieved knowledge is rebuilt from the ids saved with each answer, so the prompts are
identical to the original run except for the removed items.

Usage (from the project root):
    python scripts/faithfulness.py --dry-run          # number of requests and estimated cost
    python scripts/faithfulness.py                    # both tests on the test-set RAG run
    python scripts/faithfulness.py --test stability --stability-sample 300
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from claudette_rag.evaluation import load_gold, mcnemar_exact
from claudette_rag.knowledge_base import load_knowledge_base
from claudette_rag.llm import estimate_cost, make_client, parse_answer, response_format, run_all
from claudette_rag.rag import build_rag_system_prompt, context_from_ids, matched_control, rag_messages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
def sentence_text(record: dict) -> str:
    return (record.get("text_raw") or record["text"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run", default="rag_gpt-4.1-mini-2025-04-14_e8r5p6_test")
    parser.add_argument("--test", choices=["faithfulness", "stability", "all"], default="all")
    parser.add_argument("--stability-sample", type=int, default=300)
    parser.add_argument("--workers", type=int, help="parallel requests (default 4, or 1 for a local model)")
    parser.add_argument("--base-url", help="local OpenAI-compatible server (default: the one of the analysed run)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    info = json.loads((RESULTS / f"{args.run}.json").read_text(encoding="utf-8"))["run"]
    model = info["model"]
    base_url = args.base_url or info.get("base_url")
    workers = args.workers or (1 if base_url else 4)
    options = info.get("server_options")          # the same server parameters as the analysed run
    categories = sorted(load_knowledge_base(PROJECT_ROOT / "knowledge_base")["categories"],
                        key=lambda c: c["label_index"])
    code_to_label = {c["id"]: c["label_index"] for c in categories}
    index_of_name = {c["name"]: c["label_index"] for c in categories}
    system_prompt = build_rag_system_prompt(categories)
    fmt = response_format([c["id"] for c in categories], with_citations=True)
    with (PROJECT_ROOT / "data" / "processed" / "kb_documents.jsonl").open(encoding="utf-8") as f:
        docs = {d["id"]: d for d in map(json.loads, f)}
    with (RESULTS / "explanations" / f"{args.run}.jsonl").open(encoding="utf-8") as f:
        expl = {r["id"]: r for r in map(json.loads, f)}
    gold = {g["id"]: g for g in load_gold(PROJECT_ROOT, info["dataset"], info["split"], info.get("language"))}
    original = {sid: sorted(index_of_name[n] for n in r["predicted"]) for sid, r in expl.items()}

    # ---- the requests of each condition: (condition, record, context)
    jobs: dict[str, list[tuple[dict, dict]]] = {}
    unmatched = 0
    matched_ids: set[str] = set()
    if args.test in ("faithfulness", "all"):
        flagged = [sid for sid, r in expl.items() if r["predicted"] and r["cited"]]
        jobs["remove_cited"], jobs["remove_uncited"] = [], []
        for sid in flagged:
            r = expl[sid]
            cited = set(r["cited"])
            drop, matched = matched_control(r["retrieved"], cited, random.Random(f"control-{sid}"))
            unmatched += not matched
            if matched:
                matched_ids.add(sid)
            jobs["remove_cited"].append((gold[sid], context_from_ids(r["retrieved"], docs, cited)))
            jobs["remove_uncited"].append((gold[sid], context_from_ids(r["retrieved"], docs, drop)))
    if args.test in ("stability", "all"):
        ids = sorted(expl, key=lambda s: int(s.split("-")[-1]))
        sample = random.Random(42).sample(ids, min(args.stability_sample, len(ids)))
        jobs["repeat"] = [(gold[sid], context_from_ids(expl[sid]["retrieved"], docs)) for sid in sample]

    contexts = {cond: {rec["id"]: ctx for rec, ctx in items} for cond, items in jobs.items()}

    def maker(cond: str):
        return lambda rec: rag_messages(system_prompt, sentence_text(rec), contexts[cond][rec["id"]])

    if args.dry_run:
        total = 0.0
        for cond, items in jobs.items():
            make = maker(cond)
            rows = [{"prompt_tokens": sum(len(m["content"]) for m in make(rec)) / 4,
                     "cached_tokens": len(system_prompt) / 4, "completion_tokens": 130} for rec, _ in items]
            cost = 0.0 if base_url else estimate_cost(rows, model)
            total += cost
            print(f"{cond:15s} {len(items):4d} requests, about ${cost:.2f}")
        if unmatched:
            print(f"({unmatched} sentences cite more items of some type than were left uncited: "
                  "their control removes fewer items)")
        print(f"Total: about ${total:.2f} with {model} (with the system prompt cached; up to ~40% more without)")
        return

    client = make_client(base_url, PROJECT_ROOT)

    report = {"run": args.run, "model": model, "conditions": {}}
    answers_by_condition = {}
    for cond, items in jobs.items():
        log = RESULTS / "llm_logs" / f"{cond}_{args.run}.jsonl"
        print(f"{cond}: {len(items)} requests")
        answers = run_all([rec for rec, _ in items], maker(cond), client, model, fmt, log, workers=workers,
                          extra_body=options)
        answers_by_condition[cond] = {sid: parse_answer(answers[sid]["content"], code_to_label)["labels"]
                                      for sid in (rec["id"] for rec, _ in items)}
        rows = [answers[rec["id"]] for rec, _ in items]
        report["conditions"][cond] = {"requests": len(items),
                                      "estimated_cost_usd": 0.0 if base_url else round(estimate_cost(rows, model), 4)}

    def compare(cond: str, subset=None) -> dict:
        new = answers_by_condition[cond]
        ids = [s for s in new if subset is None or s in subset]
        c = Counter()
        for s in ids:
            before, after = original[s], new[s]
            c["n"] += 1
            c["same_labels"] += before == after
            c["same_verdict"] += bool(before) == bool(after)
            c["unfair_to_fair"] += bool(before) and not after
        n = max(1, c["n"])
        return {"n": c["n"], "same_labels": c["same_labels"] / n, "same_verdict": c["same_verdict"] / n,
                "unfair_to_fair": c["unfair_to_fair"] / n}

    print()
    if "remove_cited" in answers_by_condition:
        correct = {s for s in answers_by_condition["remove_cited"] if set(original[s]) & set(gold[s]["labels"])}
        wrong = set(answers_by_condition["remove_cited"]) - correct
        print("Faithfulness: sentences judged unfair, with citations "
              f"({len(correct)} with a correct category, {len(wrong)} without)")
        print(f"{'':52s} {'verdict flips to fair':>22s} {'labels unchanged':>17s}")
        for cond, title in (("remove_cited", "remove the cited items"), ("remove_uncited", "remove as many uncited items")):
            for subset_name, subset in (("all", None), ("matched control", matched_ids),
                                        ("correct", correct), ("wrong", wrong)):
                r = compare(cond, subset)
                report["conditions"][cond][subset_name] = r
                label = f"{title} ({subset_name})"
                print(f"  {label:50s} {100 * r['unfair_to_fair']:9.1f}% {100 * r['same_labels']:16.1f}%   (n={r['n']})")
        cited_ans, control_ans = answers_by_condition["remove_cited"], answers_by_condition["remove_uncited"]
        print("\n  Paired test on the same sentences (exact McNemar): does removing the cited items flip the verdict"
              " more often than the control?")
        for subset_name, subset in (("all", None), ("matched control", matched_ids), ("correct", correct), ("wrong", wrong)):
            ids = [s for s in cited_ans if subset is None or s in subset]
            b = sum(not cited_ans[s] and bool(control_ans[s]) for s in ids)
            c = sum(bool(cited_ans[s]) and not control_ans[s] for s in ids)
            p = mcnemar_exact(b, c)
            report["conditions"]["remove_cited"][subset_name]["mcnemar"] = {
                "only_cited_removal_flips": b, "only_control_flips": c, "p_value": p}
            print(f"    {subset_name:16s} only the cited removal flips {b:3d}, only the control flips {c:3d}, p = {p:.2g}")
        fixed = [s for s in wrong if not cited_ans[s]]
        report["wrong_flags_flipped_by_removing_citations"] = {
            "flipped": len(fixed), "of_which_fair_for_the_experts": sum(not gold[s]["labels"] for s in fixed)}
        print(f"  Wrong flags that become fair without their cited items: {len(fixed)}, of which "
              f"{sum(not gold[s]['labels'] for s in fixed)} are fair for the experts (the citations caused the error)")
        report["control_unmatched_sentences"] = unmatched
        print(f"  ('matched control': the {len(matched_ids)} sentences whose control removes exactly as many items "
              "of each type as were cited)")
    if "repeat" in answers_by_condition:
        r = compare("repeat")
        report["conditions"]["repeat"]["agreement"] = r
        print(f"\nStability: the same {r['n']} prompts sent again -> same labels {100 * r['same_labels']:.1f}%, "
              f"same verdict {100 * r['same_verdict']:.1f}%")
    total = sum(c["estimated_cost_usd"] for c in report["conditions"].values())
    print(f"\nCost: ~${total:.2f}")
    out = RESULTS / f"faithfulness_{args.run}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved -> {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
