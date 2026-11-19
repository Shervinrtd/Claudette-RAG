"""RAG system: classify sentences with an OpenAI model and knowledge retrieved from the knowledge base.

Same model, settings, task description and category cards as the plain LLM baseline
(scripts/run_llm.py). In addition, for each sentence the model receives the most similar
training sentences with their expert labels, legal rationales and legal provisions, and it
must cite the ids of the items its decision relies on.

The default split is VALIDATION: choices (k values, prompt) are made on validation, and the
test set is used once, for the final run (--split test).

Usage (from the project root, after scripts/build_index.py):
    python scripts/run_rag.py --dry-run                    # prompt of one sentence and estimated cost, no API call
    python scripts/run_rag.py --sample 40                  # quick look: 20 unfair + 20 fair validation sentences
    python scripts/run_rag.py                              # full validation set
    python scripts/run_rag.py --k-rationales 0 --max-provisions 0     # ablation: examples only
    python scripts/run_rag.py --split test                 # final run
    python scripts/run_rag.py --model qwen3:8b --base-url http://localhost:11434/v1   # free local model (Ollama)

Output (run = rag_<model>_e<k examples>r<k rationales>p<max provisions>_<data>):
    results/llm_logs/<run>.jsonl         raw answers and token usage (resumable)
    results/predictions/<run>.jsonl      {"id", "labels"} for the evaluation
    results/explanations/<run>.jsonl     sentence, gold, prediction, explanation, citations, retrieved ids
    results/<run>.json                   evaluation report (+ cost and citation checks)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from claudette_rag.evaluation import evaluate, format_report, load_gold
from claudette_rag.knowledge_base import load_knowledge_base
from claudette_rag.llm import (DEFAULT_MODEL, OLLAMA_URL, estimate_cost, file_safe, make_client, parse_answer,
                               response_format, run_all, server_options, truncation_warning, usage_summary)
from claudette_rag.rag import build_rag_system_prompt, check_citations, context_ids, rag_messages
from claudette_rag.retrieval import KBIndex, RetrievalConfig, SentenceTransformerEmbedder, retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sentence_text(record: dict) -> str:
    """Original text when available (multilingual set), otherwise the UNFAIR-ToS text."""
    return (record.get("text_raw") or record["text"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", choices=["unfair_tos", "multilingual"], default="unfair_tos")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--language", choices=["en", "de", "it", "pl"])
    parser.add_argument("--sample", type=int, help="balanced random sample (half unfair) for quick tests")
    parser.add_argument("--workers", type=int, help="parallel requests (default 4, or 1 for a local model)")
    parser.add_argument("--base-url", help=f"OpenAI-compatible server of a local model, e.g. {OLLAMA_URL} (Ollama)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="local server (vLLM): switch off the model's thinking mode, like gpt-4.1-mini")
    parser.add_argument("--extra-body", help="local server: extra request parameters as JSON text")
    parser.add_argument("--dry-run", action="store_true", help="print one prompt and the cost estimate only")
    parser.add_argument("--index", default="bge-m3", help="folder in data/processed/index/")
    parser.add_argument("--k-examples", type=int, default=8)
    parser.add_argument("--k-rationales", type=int, default=5)
    parser.add_argument("--max-provisions", type=int, default=6)
    args = parser.parse_args()
    workers = args.workers or (1 if args.base_url else 4)
    options = server_options(args.extra_body, args.no_thinking)

    kb = load_knowledge_base(PROJECT_ROOT / "knowledge_base")
    categories = sorted(kb["categories"], key=lambda c: c["label_index"])
    code_to_label = {c["id"]: c["label_index"] for c in categories}
    label_names = [c["name"] for c in categories]
    system_prompt = build_rag_system_prompt(categories)
    fmt = response_format([c["id"] for c in categories], with_citations=True)
    config = RetrievalConfig(args.k_examples, args.k_rationales, args.max_provisions)

    gold = load_gold(PROJECT_ROOT, args.dataset, args.split, args.language)
    data_name = args.split if args.dataset == "unfair_tos" else f"multilingual_{args.language or 'all'}"
    if args.sample:
        rng = random.Random(42)
        unfair = [g for g in gold if g["labels"]]
        fair = [g for g in gold if not g["labels"]]
        half = args.sample // 2
        gold = rng.sample(unfair, min(half, len(unfair))) + rng.sample(fair, min(args.sample - half, len(fair)))
        data_name += f"_sample{args.sample}"
    setting = f"e{config.k_examples}r{config.k_rationales}p{config.max_provisions}"
    run = f"rag_{file_safe(args.model)}_{setting}_{data_name}"

    # ---- retrieval (local, free)
    index_dir = PROJECT_ROOT / "data" / "processed" / "index" / args.index
    if not (index_dir / "meta.json").exists():
        sys.exit(f"No index in {index_dir.relative_to(PROJECT_ROOT)}: run scripts/build_index.py first.")
    index = KBIndex.load(index_dir)
    print(f"Retrieving knowledge for {len(gold)} sentences with {index.model_name} ({setting}) ...")
    vectors = SentenceTransformerEmbedder(index.model_name).encode([sentence_text(g) for g in gold])
    contexts = dict(zip([g["id"] for g in gold], retrieve(index, vectors, config)))

    def make_messages(record: dict) -> list[dict]:
        return rag_messages(system_prompt, sentence_text(record), contexts[record["id"]])

    if args.dry_run:
        example = next((g for g in gold if g["labels"]), gold[0])
        messages = make_messages(example)
        print("=== SYSTEM PROMPT ===\n" + system_prompt + "\n\n=== USER MESSAGE ===\n" + messages[1]["content"])
        print(f"\n(gold labels of this sentence: {example.get('label_names') or 'fair'})")
        system_tokens = len(system_prompt) / 4
        rows = [{"prompt_tokens": system_tokens + len(make_messages(g)[1]["content"]) / 4,
                 "completion_tokens": 130} for g in gold]
        full = estimate_cost(rows, args.model)
        cached = estimate_cost([{**r, "cached_tokens": system_tokens} for r in rows], args.model)
        mean_prompt = sum(r["prompt_tokens"] for r in rows) / len(rows)
        price = ("no API cost (local model)" if args.base_url else
                 f"about ${full:.2f} with {args.model}, about ${cached:.2f} when the system prompt comes from the prompt cache")
        print(f"\n{len(gold)} sentences, ~{mean_prompt:.0f} prompt tokens each -> {price}")
        return

    # ---- LLM
    client = make_client(args.base_url, PROJECT_ROOT)

    print(f"Run {run}: {len(gold)} sentences with {args.model}")
    log_path = PROJECT_ROOT / "results" / "llm_logs" / f"{run}.jsonl"
    answers = run_all(gold, make_messages, client, args.model, fmt, log_path, workers=workers,
                      extra_body=options)

    rows = [answers[g["id"]] for g in gold]
    if args.base_url and (warning := truncation_warning(rows)):
        print(warning)
    parsed = [parse_answer(r["content"], code_to_label) for r in rows]
    cites = [check_citations(p["cited_ids"], contexts[g["id"]]) for g, p in zip(gold, parsed)]

    pred_path = PROJECT_ROOT / "results" / "predictions" / f"{run}.jsonl"
    expl_path = PROJECT_ROOT / "results" / "explanations" / f"{run}.jsonl"
    for path in (pred_path, expl_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as fp, expl_path.open("w", encoding="utf-8") as fe:
        for g, p, c in zip(gold, parsed, cites):
            fp.write(json.dumps({"id": g["id"], "labels": p["labels"]}) + "\n")
            fe.write(json.dumps({
                "id": g["id"], "text": sentence_text(g),
                "gold": [label_names[i] for i in g["labels"]],
                "predicted": [label_names[i] for i in p["labels"]],
                "explanation": p["explanation"], "cited": c["valid"], "invalid_cited": c["invalid"],
                "retrieved": context_ids(contexts[g["id"]]),
            }, ensure_ascii=False) + "\n")

    report = evaluate([g["labels"] for g in gold], [p["labels"] for p in parsed], label_names)
    report["run"] = {"name": run, "system": "RAG (retrieved examples, rationales and provisions)",
                     "model": args.model, "base_url": args.base_url, "server_options": options,
                     "embedder": index.model_name,
                     "retrieval": asdict(config),
                     "dataset": args.dataset, "split": args.split, "language": args.language,
                     "sample": args.sample, "predictions": str(pred_path.relative_to(PROJECT_ROOT)),
                     "explanations": str(expl_path.relative_to(PROJECT_ROOT)),
                     "log": str(log_path.relative_to(PROJECT_ROOT))}
    report["llm"] = {
        **usage_summary(rows, args.model, local=bool(args.base_url)),
        "inconsistent_answers": sum(not p["consistent"] for p in parsed),
        "parse_errors": sum(p["parse_error"] for p in parsed),
    }
    unfair_verdicts = [c for p, c in zip(parsed, cites) if p["labels"]]
    report["citations"] = {
        "answers_citing_retrieved_items": round(sum(bool(c["valid"]) for c in cites) / len(cites), 4),
        "unfair_verdicts": len(unfair_verdicts),
        "unfair_verdicts_citing_rationale_or_provision":
            round(sum(bool(c["legal"]) for c in unfair_verdicts) / max(1, len(unfair_verdicts)), 4),
        "invalid_citations": sum(len(c["invalid"]) for c in cites),
        "answers_with_invalid_citation": sum(bool(c["invalid"]) for c in cites),
    }
    out = PROJECT_ROOT / "results" / f"{run}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + format_report(run, report))
    llm, cit = report["llm"], report["citations"]
    print(f"\nCost of this run: ~${llm['estimated_cost_usd']:.3f} | {llm['mean_seconds_per_request']} s per request | "
          f"inconsistent answers: "
          f"{llm['inconsistent_answers']} | parse errors: {llm['parse_errors']}")
    print(f"Citations: {100 * cit['answers_citing_retrieved_items']:.0f}% of answers cite retrieved items; "
          f"{100 * cit['unfair_verdicts_citing_rationale_or_provision']:.0f}% of the {cit['unfair_verdicts']} "
          f"unfair verdicts cite a rationale or provision; invented ids: {cit['invalid_citations']}")
    print(f"Saved -> {out.relative_to(PROJECT_ROOT)}  (explanations: {expl_path.relative_to(PROJECT_ROOT)})")


if __name__ == "__main__":
    main()
