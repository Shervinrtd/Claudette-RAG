"""Tests for claudette_rag.llm with a fake OpenAI client (no API calls, no cost)."""

import json
import time
from types import SimpleNamespace

import pytest

from claudette_rag.llm import (build_system_prompt, call_model, estimate_cost, parse_answer,
                               response_format, run_all, zero_shot_messages)

CATEGORIES = [
    {"id": "ter", "name": "Unilateral termination", "definition": "Provider may terminate.",
     "levels": {"potentially_unfair": "reasons given", "clearly_unfair": "any time, no notice",
                "clearly_fair": None}},
    {"id": "j", "name": "Jurisdiction", "definition": "Which courts decide.",
     "levels": {"potentially_unfair": None, "clearly_unfair": "far from consumer",
                "clearly_fair": "consumer's own courts"}},
]
CODE_TO_LABEL = {"ter": 1, "j": 6}


class FakeCompletions:
    def __init__(self, answer_fn):
        self.answer_fn = answer_fn
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=50,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=200))
        message = SimpleNamespace(content=self.answer_fn(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage,
                               system_fingerprint="fp_test")


def fake_client(answer_fn):
    completions = FakeCompletions(answer_fn)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def keyword_answer(kwargs):
    sentence = kwargs["messages"][-1]["content"]
    unfair = "terminate" in sentence
    return json.dumps({"explanation": "test", "unfair": unfair, "categories": ["ter"] if unfair else []})


def test_system_prompt_contains_every_category_and_level():
    prompt = build_system_prompt(CATEGORIES)
    assert "[ter] Unilateral termination" in prompt and "[j] Jurisdiction" in prompt
    assert "NOT unfair (clearly fair) if: consumer's own courts" in prompt
    assert "Potentially unfair if: reasons given" in prompt
    assert "choose from ter, j" in prompt


def test_response_format_is_strict():
    fmt = response_format(["ter", "j"])
    schema = fmt["json_schema"]["schema"]
    assert fmt["json_schema"]["strict"] is True
    assert schema["required"] == ["explanation", "unfair", "categories"]
    assert schema["properties"]["categories"]["items"]["enum"] == ["ter", "j"]


@pytest.mark.parametrize("answer, labels, consistent", [
    ({"explanation": "x", "unfair": True, "categories": ["j", "ter"]}, [1, 6], True),
    ({"explanation": "x", "unfair": False, "categories": []}, [], True),
    ({"explanation": "x", "unfair": False, "categories": ["ter"]}, [], False),  # verdict decides
    ({"explanation": "x", "unfair": True, "categories": []}, [], False),
])
def test_parse_answer(answer, labels, consistent):
    parsed = parse_answer(json.dumps(answer), CODE_TO_LABEL)
    assert parsed["labels"] == labels and parsed["consistent"] == consistent


def test_parse_answer_handles_broken_json():
    parsed = parse_answer("not json", CODE_TO_LABEL)
    assert parsed["labels"] == [] and parsed["parse_error"]


def test_call_model_uses_deterministic_settings():
    client, completions = fake_client(keyword_answer)
    msgs = zero_shot_messages("system", "we may terminate your account")
    result = call_model(client, "gpt-4.1-mini-2025-04-14", msgs, response_format(["ter"]))
    assert completions.calls[-1]["temperature"] == 0 and completions.calls[-1]["seed"] == 42
    assert result["cached_tokens"] == 200
    call_model(client, "gpt-5.4-mini", msgs, response_format(["ter"]))
    assert "temperature" not in completions.calls[-1]
    assert completions.calls[-1]["reasoning_effort"] == "low"


def test_run_all_saves_and_resumes(tmp_path):
    records = [{"id": f"s{i}", "text": t} for i, t in
               enumerate(["we may terminate your account", "welcome to our service", "contact us"])]
    make = lambda r: zero_shot_messages("system", r["text"])  # noqa: E731
    fmt = response_format(["ter", "j"])
    log = tmp_path / "log.jsonl"

    client, completions = fake_client(keyword_answer)
    answers = run_all(records[:2], make, client, "gpt-4.1-mini", fmt, log, workers=2)
    assert len(completions.calls) == 2 and set(answers) == {"s0", "s1"}

    # second run: only the new sentence is sent to the API
    client2, completions2 = fake_client(keyword_answer)
    answers = run_all(records, make, client2, "gpt-4.1-mini", fmt, log, workers=2)
    assert len(completions2.calls) == 1 and set(answers) == {"s0", "s1", "s2"}
    assert parse_answer(answers["s0"]["content"], CODE_TO_LABEL)["labels"] == [1]
    assert len(log.read_text().splitlines()) == 3


def test_estimate_cost():
    rows = [{"prompt_tokens": 1_000_000, "cached_tokens": 0, "completion_tokens": 1_000_000}]
    assert estimate_cost(rows, "gpt-4.1-mini-2025-04-14") == pytest.approx(0.40 + 1.60)
    assert estimate_cost(rows, "unknown-model") != estimate_cost(rows, "unknown-model")  # NaN


def test_run_all_stops_cleanly_and_keeps_received_answers(tmp_path):
    records = [{"id": f"s{i}", "text": f"sentence {i}"} for i in range(6)]
    make = lambda r: zero_shot_messages("system", r["text"])  # noqa: E731

    def failing_answer(kwargs):
        time.sleep(0.05)                                          # a real request takes time
        if kwargs["messages"][-1]["content"].endswith("sentence 2"):
            raise ValueError("no credits remaining")             # not retryable: fails at once
        return json.dumps({"explanation": "x", "unfair": False, "categories": []})

    client, completions = fake_client(failing_answer)
    log = tmp_path / "log.jsonl"
    with pytest.raises(ValueError):
        run_all(records, make, client, "gpt-4.1-mini", response_format(["ter"]), log, workers=1)
    saved = [json.loads(line)["id"] for line in log.read_text().splitlines()]
    assert len(completions.calls) < len(records)                  # waiting requests were cancelled
    assert len(saved) == len(completions.calls) - 1 and "s2" not in saved   # every answer received is kept


def test_out_of_credits_is_not_retried():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    no_credits = openai.RateLimitError("no credits", response=response, body={"code": "insufficient_quota"})
    busy = openai.RateLimitError("slow down", response=response, body={"code": "rate_limit_exceeded"})

    def raise_no_credits(kwargs):
        raise no_credits

    client, completions = fake_client(raise_no_credits)
    with pytest.raises(openai.RateLimitError):
        call_model(client, "gpt-4.1-mini", zero_shot_messages("s", "x"), response_format(["ter"]))
    assert len(completions.calls) == 1                             # no waiting, no retries
    from claudette_rag.llm import _is_retryable
    assert _is_retryable(busy) and not _is_retryable(no_credits)


def test_local_server_gets_max_tokens_and_file_safe_names():
    from claudette_rag.llm import file_safe, is_local, usage_summary
    client, completions = fake_client(keyword_answer)
    client.base_url = "http://localhost:11434/v1/"                    # e.g. Ollama
    result = call_model(client, "qwen3:8b", zero_shot_messages("s", "terminate"), response_format(["ter"]))
    sent = completions.calls[-1]
    assert is_local(client) and sent["max_tokens"] == 400 and "max_completion_tokens" not in sent
    assert sent["temperature"] == 0 and sent["seed"] == 42 and result["latency_s"] >= 0
    openai_client, openai_calls = fake_client(keyword_answer)          # no base_url: the OpenAI API
    call_model(openai_client, "gpt-4.1-mini", zero_shot_messages("s", "x"), response_format(["ter"]))
    assert "max_completion_tokens" in openai_calls.calls[-1] and not is_local(openai_client)
    assert file_safe("qwen3:8b") == "qwen3-8b" and file_safe("gpt-4.1-mini-2025-04-14") == "gpt-4.1-mini-2025-04-14"
    summary = usage_summary([result], "qwen3:8b", local=True)
    assert summary["estimated_cost_usd"] == 0.0 and summary["mean_seconds_per_request"] is not None


def test_truncation_warning():
    from claudette_rag.llm import truncation_warning
    assert truncation_warning([{"prompt_tokens": 1500 + i} for i in range(50)]) is None
    capped = [{"prompt_tokens": min(2048, 1800 + 20 * i)} for i in range(50)]      # many prompts cut at 2048
    assert "truncated" in truncation_warning(capped)


def test_server_options_and_extra_body():
    from claudette_rag.llm import server_options
    assert server_options(None, False) is None
    assert server_options(None, True) == {"chat_template_kwargs": {"enable_thinking": False}}
    merged = server_options('{"top_k": 1, "chat_template_kwargs": {"x": 1}}', True)
    assert merged == {"top_k": 1, "chat_template_kwargs": {"x": 1, "enable_thinking": False}}
    client, completions = fake_client(keyword_answer)
    client.base_url = "http://gpu-node:8000/v1"
    call_model(client, "Qwen/Qwen3.5-27B", zero_shot_messages("s", "x"), response_format(["ter"]),
               extra_body=server_options(None, True))
    assert completions.calls[-1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
