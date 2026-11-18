"""Classify Terms of Service sentences with an LLM.

Shared by the plain LLM baseline (category definitions only) and the RAG system
(category definitions + retrieved knowledge), so that the only difference between
the two is the retrieved context.

Main design choices
- The model answers with a strict JSON schema: an explanation first, then the verdict
  and the categories ("reason, then decide").
- Deterministic settings where the model allows them (temperature 0, fixed seed) and a
  dated model snapshot, for reproducibility.
- Every answer is appended to a log file as soon as it arrives: an interrupted run can be
  resumed without paying twice for the same sentences.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"

# USD per 1M tokens: (input, cached input, output). Checked on 2026-11-18; used only to
# estimate costs in the logs, not for billing.
PRICES = {
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
}

LEVEL_TEXT = {"potentially_unfair": "Potentially unfair if", "clearly_unfair": "Clearly unfair if",
              "clearly_fair": "NOT unfair (clearly fair) if"}


# ----------------------------------------------------------------------------- prompts
def build_system_prompt(categories: list[dict]) -> str:
    """Task description + one block per category (definition and unfairness criteria)."""
    blocks = []
    for c in categories:
        lines = [f"[{c['id']}] {c['name']}: {c['definition']}"]
        for level in ("potentially_unfair", "clearly_unfair", "clearly_fair"):
            if c["levels"].get(level):
                lines.append(f"  - {LEVEL_TEXT[level]}: {c['levels'][level]}")
        blocks.append("\n".join(lines))
    codes = ", ".join(c["id"] for c in categories)
    return f"""You are an expert in EU consumer protection law. You review single sentences taken from online Terms of Service and decide whether each sentence contains a potentially unfair clause under Council Directive 93/13/EEC: a standard term that, contrary to the requirement of good faith, causes a significant imbalance in the parties' rights and obligations to the detriment of the consumer.

There are {len(categories)} categories of potentially unfair clauses:

{chr(10).join(blocks)}

Rules:
- A sentence can belong to several categories at the same time, or to none.
- A sentence is unfair if, for at least one category, it is potentially unfair or clearly unfair. A sentence about one of these topics that is clearly fair (for example, it lets consumers go to the courts of their own country) is NOT unfair.
- Most sentences of a contract are fair: definitions, descriptions of the service, contact details, obligations that protect users. Do not flag a sentence only because it mentions a topic.
- The sentence may be lowercased and tokenized (for example "terms ." or "you 're").

Answer with a JSON object:
- "explanation": one or two sentences explaining your decision;
- "unfair": true or false;
- "categories": the list of category codes that apply (choose from {codes}); empty if the sentence is fair."""


def zero_shot_messages(system_prompt: str, sentence: str) -> list[dict]:
    """Plain LLM baseline: no retrieved knowledge."""
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sentence: {sentence}"}]


def response_format(category_codes: list[str]) -> dict:
    """Strict JSON schema for the answer (OpenAI structured outputs)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "unfair_clause_verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "unfair": {"type": "boolean"},
                    "categories": {"type": "array", "items": {"type": "string", "enum": category_codes}},
                },
                "required": ["explanation", "unfair", "categories"],
                "additionalProperties": False,
            },
        },
    }


# ----------------------------------------------------------------------------- parsing
def parse_answer(content: str, code_to_label: dict[str, int]) -> dict:
    """JSON answer -> {"labels", "explanation", "consistent"}.

    The verdict decides: if "unfair" is false the sentence is fair, whatever the categories;
    if "unfair" is true but no category is given, the sentence is counted as fair (we cannot
    score an unfair verdict without a category). Both cases are flagged as inconsistent.
    """
    try:
        answer = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"labels": [], "explanation": None, "consistent": False, "parse_error": True}
    codes = [c for c in answer.get("categories", []) if c in code_to_label]
    unfair = bool(answer.get("unfair"))
    labels = sorted({code_to_label[c] for c in codes}) if unfair else []
    return {"labels": labels, "explanation": answer.get("explanation"),
            "consistent": unfair == bool(codes), "parse_error": False}


# ----------------------------------------------------------------------------- calling
def _is_retryable(error: Exception) -> bool:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    return isinstance(error, (openai.RateLimitError, openai.APIConnectionError,
                              openai.APITimeoutError, openai.InternalServerError))


def call_model(client, model: str, messages: list[dict], fmt: dict, max_retries: int = 6) -> dict:
    """One chat completion, retried with exponential backoff on transient errors."""
    kwargs = {"model": model, "messages": messages, "response_format": fmt,
              "max_completion_tokens": 400, "seed": 42}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["reasoning_effort"] = "low"     # reasoning models: no temperature setting
    else:
        kwargs["temperature"] = 0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            usage = response.usage
            details = getattr(usage, "prompt_tokens_details", None)
            return {
                "content": response.choices[0].message.content,
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
                "completion_tokens": usage.completion_tokens,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            }
        except Exception as error:  # noqa: BLE001
            if attempt == max_retries - 1 or not _is_retryable(error):
                raise
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("unreachable")


def run_all(records: list[dict], make_messages, client, model: str, fmt: dict, log_path: Path,
            workers: int = 4, progress_every: int = 50) -> dict[str, dict]:
    """Classify every record; answers are appended to log_path and reused on a re-run."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["id"]] = row
    todo = [r for r in records if r["id"] not in done]
    if done:
        print(f"Resuming: {len(done)} answers already in {log_path.name}, {len(todo)} to go")

    lock = threading.Lock()
    with log_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_model, client, model, make_messages(r), fmt): r for r in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            row = {"id": record["id"], "model": model, **future.result()}
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done[record["id"]] = row
            if n % progress_every == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} answered")
    return done


def estimate_cost(rows: list[dict], model: str) -> float:
    """Approximate cost in USD of the logged calls."""
    prices = next((p for prefix, p in PRICES.items() if model.startswith(prefix)), None)
    if prices is None:
        return float("nan")
    p_in, p_cached, p_out = prices
    total = 0.0
    for r in rows:
        cached = r.get("cached_tokens", 0)
        total += ((r["prompt_tokens"] - cached) * p_in + cached * p_cached
                  + r["completion_tokens"] * p_out) / 1e6
    return total
