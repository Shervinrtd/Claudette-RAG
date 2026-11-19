"""Tests for claudette_rag.rag (prompt, context formatting, citation checks); no API calls."""

import json

from claudette_rag.llm import build_system_prompt, parse_answer, response_format
from claudette_rag.rag import (build_rag_system_prompt, check_citations, context_ids, format_context,
                               gate_by_neighbours, rag_messages)

CATEGORIES = [
    {"id": "ter", "name": "Unilateral termination", "definition": "Provider may terminate.",
     "levels": {"potentially_unfair": "reasons given", "clearly_unfair": "any time, no notice",
                "clearly_fair": None}},
    {"id": "j", "name": "Jurisdiction", "definition": "Which courts decide.",
     "levels": {"potentially_unfair": None, "clearly_unfair": "far from consumer",
                "clearly_fair": "consumer's own courts"}},
]
CONTEXT = {
    "examples": [({"id": "ex.train-1", "categories": ["ter"], "text": "we may terminate your account ."}, 0.9),
                 ({"id": "ex.train-2", "categories": [], "text": "x" * 700}, 0.8)],
    "rationales": [({"id": "ter.no_notice", "text": "Unilateral termination - clearly unfair: no notice."}, 0.7)],
    "provisions": [{"id": "dir93.annex.1g", "text": "Directive 93/13/EEC, Annex 1(g) - termination."}],
}
EMPTY = {"examples": [], "rationales": [], "provisions": []}


def test_rag_prompt_extends_the_baseline_prompt():
    base = build_system_prompt(CATEGORIES)
    rag = build_rag_system_prompt(CATEGORIES)
    assert rag.startswith(base.split("\n\nAnswer with a JSON object:\n")[0])   # same task and categories
    assert "cited_ids" in rag and "Retrieved knowledge" in rag
    assert "cited_ids" not in base and "Retrieved knowledge" not in base       # baseline unchanged
    assert "choose from ter, j" in rag


def test_format_context():
    text = format_context(CONTEXT)
    assert '[ex.train-1] (unfair: ter) "we may terminate your account ."' in text
    assert "[ex.train-2] (fair)" in text and " [...]" in text and "x" * 601 not in text   # long example clipped
    assert "[ter.no_notice] Unilateral termination" in text and "[dir93.annex.1g] Directive" in text
    assert text.index("Similar sentences") < text.index("Legal rationales") < text.index("Legal provisions")
    assert "Legal rationales" not in format_context({**CONTEXT, "rationales": []})
    assert format_context(EMPTY) == ""


def test_rag_messages_keep_the_system_prompt_fixed_and_the_sentence_last():
    with_context = rag_messages("SYSTEM", "we may terminate", CONTEXT)
    without = rag_messages("SYSTEM", "contact us", EMPTY)
    assert with_context[0] == without[0] == {"role": "system", "content": "SYSTEM"}
    assert with_context[1]["content"].startswith("Retrieved knowledge:")
    assert with_context[1]["content"].endswith("Sentence: we may terminate")
    assert without[1]["content"] == "Sentence: contact us"


def test_response_format_with_citations():
    schema = response_format(["ter", "j"], with_citations=True)["json_schema"]["schema"]
    assert list(schema["properties"]) == ["explanation", "cited_ids", "unfair", "categories"]
    assert schema["required"] == ["explanation", "cited_ids", "unfair", "categories"]
    assert "cited_ids" not in response_format(["ter", "j"])["json_schema"]["schema"]["properties"]


def test_parse_and_check_citations():
    content = json.dumps({"explanation": "no notice", "unfair": True, "categories": ["ter"],
                          "cited_ids": ["ter.no_notice", "[ex.train-1]", "dir93.art99", " ter.no_notice", ""]})
    parsed = parse_answer(content, {"ter": 1, "j": 6})
    assert parsed["labels"] == [1]
    assert parsed["cited_ids"] == ["ter.no_notice", "ex.train-1", "dir93.art99"]  # brackets, blanks, duplicates
    assert check_citations(parsed["cited_ids"], CONTEXT) == {
        "valid": ["ter.no_notice", "ex.train-1"], "invalid": ["dir93.art99"], "legal": ["ter.no_notice"]}
    baseline = parse_answer(json.dumps({"explanation": "x", "unfair": False, "categories": []}), {"ter": 1})
    assert baseline["cited_ids"] == []


def test_context_ids():
    assert context_ids(CONTEXT) == {"examples": ["ex.train-1", "ex.train-2"],
                                    "rationales": ["ter.no_notice"], "provisions": ["dir93.annex.1g"]}


def test_gate_keeps_only_categories_with_enough_precedents():
    neighbours = [[1], [1, 2], [], [2], [1]]          # labels of the 5 nearest training sentences
    assert gate_by_neighbours([1, 2, 7], neighbours, k=5, m=2) == [1, 2]
    assert gate_by_neighbours([1, 2, 7], neighbours, k=3, m=2) == [1]     # only the 3 nearest count
    assert gate_by_neighbours([1, 2], neighbours, k=5, m=4) == []
    assert gate_by_neighbours([], neighbours, k=5, m=1) == []
