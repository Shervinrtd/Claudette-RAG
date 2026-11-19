"""Plain LLM baseline: classify sentences with an OpenAI model, without retrieval.

The model receives the task description and the definition of the 8 categories
(with their fairness criteria), but no legal rationales and no examples: that extra
knowledge is what the RAG system adds.

Usage (from the project root):
    python scripts/run_llm.py --dry-run                     # show the prompt and the estimated cost, no API call
    python scripts/run_llm.py --sample 40                   # quick test: 20 unfair + 20 fair test sentences
    python scripts/run_llm.py                               # full UNFAIR-ToS test set
    python scripts/run_llm.py --dataset multilingual --language it
    python scripts/run_llm.py --model qwen3:8b --base-url http://localhost:11434/v1   # free local model (Ollama)

Output (run = llm_zeroshot_<model>_<data>):
    results/llm_logs/<run>.jsonl         raw answers, explanations, token usage (resumable)
    results/predictions/<run>.jsonl      {"id", "labels"} for the evaluation
    results/<run>.json                   evaluation report (+ cost and consistency checks)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from claudette_rag.evaluation import evaluate, format_report, load_gold
from claudette_rag.knowledge_base import load_knowledge_base
from claudette_rag.llm import (DEFAULT_MODEL, OLLAMA_URL, build_system_prompt, estimate_cost, file_safe,
                               make_client, parse_answer, response_format, run_all, server_options,
                               truncation_warning, usage_summary, zero_shot_messages)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sentence_text(record: dict) -> str:
    """Original text when available (multilingual set), otherwise the UNFAIR-ToS text."""
    return (record.get("text_raw") or record["text"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", choices=["unfair_tos", "multilingual"], default="unfair_tos")
    parser.add_argument("--split", default="test")
    parser.add_argument("--language", choices=["en", "de", "it", "pl"])
    parser.add_argument("--sample", type=int, help="balanced random sample (half unfair) for quick tests")
    parser.add_argument("--workers", type=int, help="parallel requests (default 4, or 1 for a local model)")
    parser.add_argument("--base-url", help=f"OpenAI-compatible server of a local model, e.g. {OLLAMA_URL} (Ollama)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="local server (vLLM): switch off the model's thinking mode, like gpt-4.1-mini")
    parser.add_argument("--extra-body", help="local server: extra request parameters as JSON text")
    parser.add_argument("--dry-run", action="store_true", help="print prompt and cost estimate only")
    args = parser.parse_args()
    workers = args.workers or (1 if args.base_url else 4)
    options = server_options(args.extra_body, args.no_thinking)

    kb = load_knowledge_base(PROJECT_ROOT / "knowledge_base")
    categories = sorted(kb["categories"], key=lambda c: c["label_index"])
    code_to_label = {c["id"]: c["label_index"] for c in categories}
    label_names = [c["name"] for c in categories]
    system_prompt = build_system_prompt(categories)
    fmt = response_format([c["id"] for c in categories])

    gold = load_gold(PROJECT_ROOT, args.dataset, args.split, args.language)
    data_name = args.split if args.dataset == "unfair_tos" else f"multilingual_{args.language or 'all'}"
    if args.sample:
        rng = random.Random(42)
        unfair = [g for g in gold if g["labels"]]
        fair = [g for g in gold if not g["labels"]]
        half = args.sample // 2
        gold = rng.sample(unfair, min(half, len(unfair))) + rng.sample(fair, min(args.sample - half, len(fair)))
        data_name += f"_sample{args.sample}"
    run = f"llm_zeroshot_{file_safe(args.model)}_{data_name}"

    if args.dry_run:
        example = zero_shot_messages(system_prompt, sentence_text(gold[0]))
        print("=== SYSTEM PROMPT ===\n" + system_prompt + "\n\n=== USER MESSAGE ===\n" + example[1]["content"])
        prompt_tokens = (len(system_prompt) + 60) / 4 + sum(len(sentence_text(g)) for g in gold) / 4 / len(gold)
        n = len(gold)
        est = estimate_cost([{"prompt_tokens": prompt_tokens, "completion_tokens": 90}] * n, args.model)
        price = "no API cost (local model)" if args.base_url else f"about ${est:.2f} with {args.model} (before prompt-caching discounts)"
        print(f"\n{n} sentences, ~{prompt_tokens:.0f} prompt tokens each -> {price}")
        return

    client = make_client(args.base_url, PROJECT_ROOT)

    print(f"Run {run}: {len(gold)} sentences with {args.model}")
    log_path = PROJECT_ROOT / "results" / "llm_logs" / f"{run}.jsonl"
    answers = run_all(gold, lambda r: zero_shot_messages(system_prompt, sentence_text(r)),
                      client, args.model, fmt, log_path, workers=workers,
                      extra_body=options)

    rows = [answers[g["id"]] for g in gold]
    if args.base_url and (warning := truncation_warning(rows)):
        print(warning)
    parsed = [parse_answer(r["content"], code_to_label) for r in rows]
    pred_path = PROJECT_ROOT / "results" / "predictions" / f"{run}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for g, p in zip(gold, parsed):
            f.write(json.dumps({"id": g["id"], "labels": p["labels"]}) + "\n")

    report = evaluate([g["labels"] for g in gold], [p["labels"] for p in parsed], label_names)
    report["run"] = {"name": run, "system": "plain LLM (zero-shot, no retrieval)", "model": args.model,
                     "base_url": args.base_url, "server_options": options,
                     "dataset": args.dataset, "split": args.split, "language": args.language,
                     "sample": args.sample, "predictions": str(pred_path.relative_to(PROJECT_ROOT)),
                     "log": str(log_path.relative_to(PROJECT_ROOT))}
    report["llm"] = {
        **usage_summary(rows, args.model, local=bool(args.base_url)),
        "inconsistent_answers": sum(not p["consistent"] for p in parsed),
        "parse_errors": sum(p["parse_error"] for p in parsed),
    }
    out = PROJECT_ROOT / "results" / f"{run}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + format_report(run, report))
    llm = report["llm"]
    print(f"\nCost so far for this run: ~${llm['estimated_cost_usd']:.3f} | "
          f"{llm['mean_seconds_per_request']} s per request | "
          f"inconsistent answers: {llm['inconsistent_answers']} | parse errors: {llm['parse_errors']}")
    print(f"Saved -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
