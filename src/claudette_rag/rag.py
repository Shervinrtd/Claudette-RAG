"""The RAG prompt: retrieved knowledge in, a cited verdict out.

The RAG system uses the same task description, category cards, model and settings as the
plain LLM baseline (llm.py). What it adds:
- a short section of the system prompt on how to use the retrieved knowledge;
- in the user message, before the sentence: similar training sentences with their expert
  labels, legal rationales and the legal provisions they rely on (retrieval.retrieve);
- a "cited_ids" answer field, checked afterwards against what was actually retrieved, so
  every explanation can be traced to its sources and invented sources are counted.

The system prompt is identical for every sentence and comes first, so that the API can
reuse it from its prompt cache; everything that changes per sentence is in the user message.
"""

from __future__ import annotations

import random

from .llm import build_system_prompt

ANSWER_HEADER = "\n\nAnswer with a JSON object:\n"

RETRIEVAL_SECTION = """

Retrieved knowledge:
Before each sentence you receive knowledge retrieved from a legal knowledge base, selected because its wording is similar to the sentence:
- sentences from other Terms of Service, with the labels given by legal experts ("fair", or the codes of the unfair categories);
- legal rationales: when a kind of clause is or is not unfair;
- the legal provisions those rationales are based on.
Every item has an id in square brackets. Use the knowledge as evidence, but judge the sentence itself: a retrieved item can be similar in wording and different in meaning, and some items are not relevant. Similar sentences that experts labelled fair are evidence that the sentence may be fair too."""

RAG_ANSWER_FIELDS = """- "explanation": one or two sentences explaining your decision;
- "cited_ids": the ids of the retrieved items that support your decision (for an unfair sentence, include the rationales or provisions that apply); only ids that appear in the retrieved knowledge; empty if none is relevant;
- "unfair": true or false;
- "categories": the list of category codes that apply (choose from {codes}); empty if the sentence is fair."""


def build_rag_system_prompt(categories: list[dict]) -> str:
    """Baseline system prompt + how to use the retrieved knowledge + the citation field."""
    base = build_system_prompt(categories)
    head, _ = base.split(ANSWER_HEADER)
    codes = ", ".join(c["id"] for c in categories)
    return head + RETRIEVAL_SECTION + ANSWER_HEADER + RAG_ANSWER_FIELDS.format(codes=codes)


def _clip(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " [...]"


def format_context(context: dict, max_example_chars: int = 600) -> str:
    """Retrieved knowledge of one sentence -> text block; empty sections are left out."""
    sections = []
    if context.get("examples"):
        lines = ["Similar sentences from other Terms of Service, labelled by legal experts:"]
        for doc, _score in context["examples"]:
            label = "unfair: " + ", ".join(doc["categories"]) if doc["categories"] else "fair"
            lines.append(f'[{doc["id"]}] ({label}) "{_clip(doc["text"], max_example_chars)}"')
        sections.append("\n".join(lines))
    if context.get("rationales"):
        sections.append("\n".join(["Legal rationales:"]
                                  + [f"[{doc['id']}] {doc['text']}" for doc, _ in context["rationales"]]))
    if context.get("provisions"):
        sections.append("\n".join(["Legal provisions:"]
                                  + [f"[{doc['id']}] {doc['text']}" for doc in context["provisions"]]))
    return "\n\n".join(sections)


def rag_messages(system_prompt: str, sentence: str, context: dict) -> list[dict]:
    knowledge = format_context(context)
    user = f"Sentence: {sentence}"
    if knowledge:
        user = f"Retrieved knowledge:\n\n{knowledge}\n\n{user}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def check_citations(cited_ids: list[str], context: dict) -> dict:
    """Split the cited ids into retrieved ones (valid) and invented ones (invalid).

    "legal" are the valid citations of a rationale or provision, i.e. the explanation
    points to a legal reason and not only to similar sentences.
    """
    legal_ids = {doc["id"] for doc, _ in context.get("rationales", [])}
    legal_ids |= {doc["id"] for doc in context.get("provisions", [])}
    retrieved = legal_ids | {doc["id"] for doc, _ in context.get("examples", [])}
    return {"valid": [c for c in cited_ids if c in retrieved],
            "invalid": [c for c in cited_ids if c not in retrieved],
            "legal": [c for c in cited_ids if c in legal_ids]}


def gate_by_neighbours(labels: list[int], neighbour_labels: list[list[int]], k: int, m: int) -> list[int]:
    """Evidence gate: keep a predicted category only if at least m of the k most similar
    training sentences (expert-labelled precedents, most similar first) have it."""
    top = neighbour_labels[:k]
    return [label for label in labels if sum(label in n for n in top) >= m]


def context_ids(context: dict) -> dict:
    """Ids of the retrieved items, for the explanation log."""
    return {"examples": [doc["id"] for doc, _ in context.get("examples", [])],
            "rationales": [doc["id"] for doc, _ in context.get("rationales", [])],
            "provisions": [doc["id"] for doc in context.get("provisions", [])]}


TYPES = ("examples", "rationales", "provisions")


def context_from_ids(retrieved: dict, docs: dict, drop=frozenset()) -> dict:
    """Saved retrieved ids -> context in the format of retrieval.retrieve, without the ids in `drop`.

    Rebuilds exactly the knowledge shown to the LLM (the similarity scores are not used in the prompt).
    """
    return {"examples": [(docs[i], 0.0) for i in retrieved["examples"] if i not in drop],
            "rationales": [(docs[i], 0.0) for i in retrieved["rationales"] if i not in drop],
            "provisions": [docs[i] for i in retrieved["provisions"] if i not in drop]}


def matched_control(retrieved: dict, cited: set[str], rng: random.Random) -> tuple[set[str], bool]:
    """Uncited items to remove for the faithfulness control: for each type, as many as were cited.

    Returns the ids and whether the match was exact (False when too few uncited items of a type exist).
    """
    drop, matched = set(), True
    for kind in TYPES:
        n_cited = sum(i in cited for i in retrieved[kind])
        uncited = [i for i in retrieved[kind] if i not in cited]
        if n_cited > len(uncited):
            matched = False
        drop |= set(rng.sample(uncited, min(n_cited, len(uncited))))
    return drop, matched
