"""Load, validate and export the legal knowledge base (knowledge_base/*.yaml)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from claudette_rag.text import squash

LEVELS = ("clearly_fair", "potentially_unfair", "clearly_unfair")
LEVEL_TEXT = {
    "clearly_fair": "clearly fair",
    "potentially_unfair": "potentially unfair",
    "clearly_unfair": "clearly unfair",
}
FILES = ("sources", "legal_provisions", "categories", "rationales")


def load_knowledge_base(kb_dir: Path) -> dict[str, list[dict]]:
    """Read the four YAML files of the knowledge base."""
    kb = {}
    for name in FILES:
        with (kb_dir / f"{name}.yaml").open(encoding="utf-8") as f:
            kb[name] = yaml.safe_load(f) or []
    return kb


def validate(kb: dict[str, list[dict]], label_names: list[str]) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errors: list[str] = []

    all_ids = [entry.get("id") for name in FILES for entry in kb[name]]
    for entry_id, n in Counter(all_ids).items():
        if entry_id is None:
            errors.append("an entry has no 'id'")
        elif n > 1:
            errors.append(f"id '{entry_id}' is used {n} times")

    sources = {s["id"] for s in kb["sources"]}
    provisions = {p["id"] for p in kb["legal_provisions"]}
    categories = {c["id"]: c for c in kb["categories"]}

    for p in kb["legal_provisions"]:
        for field in ("source", "reference", "title", "summary"):
            if not p.get(field):
                errors.append(f"provision '{p['id']}' has no '{field}'")
        if p.get("source") not in sources:
            errors.append(f"provision '{p['id']}' cites unknown source '{p.get('source')}'")

    if len(categories) != len(label_names):
        errors.append(f"{len(categories)} categories, but the dataset has {len(label_names)} labels")
    for c in kb["categories"]:
        idx = c.get("label_index")
        if not isinstance(idx, int) or not 0 <= idx < len(label_names):
            errors.append(f"category '{c['id']}' has an invalid label_index {idx}")
        elif c.get("name") != label_names[idx]:
            errors.append(f"category '{c['id']}' is called '{c.get('name')}' but label {idx} "
                          f"is '{label_names[idx]}'")
        for field in ("definition", "why_it_matters", "levels", "legal_basis", "source"):
            if not c.get(field):
                errors.append(f"category '{c['id']}' has no '{field}'")
        if not any((c.get("levels") or {}).get(level) for level in LEVELS):
            errors.append(f"category '{c['id']}' defines no unfairness level")
        for ref in c.get("legal_basis") or []:
            if ref not in provisions:
                errors.append(f"category '{c['id']}' cites unknown provision '{ref}'")
        if c.get("source") not in sources:
            errors.append(f"category '{c['id']}' cites unknown source '{c.get('source')}'")

    per_category = Counter()
    for r in kb["rationales"]:
        rid, cat = r["id"], r.get("category")
        if cat not in categories:
            errors.append(f"rationale '{rid}' has unknown category '{cat}'")
            continue
        per_category[cat] += 1
        if not rid.startswith(f"{cat}."):
            errors.append(f"rationale '{rid}' should start with '{cat}.'")
        level = r.get("level")
        if level not in LEVELS:
            errors.append(f"rationale '{rid}' has invalid level '{level}'")
        elif not categories[cat]["levels"].get(level):
            errors.append(f"rationale '{rid}' is '{level}', a level not used for category '{cat}'")
        if not r.get("practice"):
            errors.append(f"rationale '{rid}' has no 'practice'")
        if r.get("source") not in sources:
            errors.append(f"rationale '{rid}' cites unknown source '{r.get('source')}'")
        for ref in r.get("legal_basis") or []:
            if ref not in provisions:
                errors.append(f"rationale '{rid}' cites unknown provision '{ref}'")
    for cat in categories:
        if per_category[cat] == 0:
            errors.append(f"category '{cat}' has no rationale")

    return errors


def load_examples(raw_dir: Path) -> tuple[list[dict], int]:
    """Training sentences used as examples, minus those that also occur in validation/test.

    Returns the examples and the number of training sentences removed.
    """
    def read(split):
        with (raw_dir / f"unfair_tos_{split}.jsonl").open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    held_out = {squash(r["text"]) for split in ("validation", "test") for r in read(split)}
    train = read("train")
    examples = [r for r in train if squash(r["text"]) not in held_out]
    return examples, len(train) - len(examples)


def to_documents(kb: dict[str, list[dict]], examples: list[dict],
                 label_names: list[str]) -> list[dict]:
    """Turn every knowledge-base entry and example into a retrievable document."""
    categories = {c["id"]: c for c in kb["categories"]}
    code_of_label = {c["label_index"]: c["id"] for c in kb["categories"]}
    docs = []

    for p in kb["legal_provisions"]:
        docs.append({
            "id": p["id"], "type": "provision", "categories": [], "level": None,
            "text": f"{p['reference']} - {p['title']}: {p['summary']}",
            "source": p["source"],
        })

    for c in kb["categories"]:
        level_lines = " ".join(f"{LEVEL_TEXT[level].capitalize()} if: {c['levels'][level]}"
                               for level in LEVELS if c["levels"].get(level))
        docs.append({
            "id": f"cat.{c['id']}", "type": "category", "categories": [c["id"]], "level": None,
            "text": (f"{c['name']}. {c['definition']} {level_lines} "
                     f"Why it matters for the consumer: {c['why_it_matters']}"),
            "legal_basis": c["legal_basis"], "source": c["source"],
        })

    for r in kb["rationales"]:
        c = categories[r["category"]]
        docs.append({
            "id": r["id"], "type": "rationale", "categories": [r["category"]], "level": r["level"],
            "text": f"{c['name']} - {LEVEL_TEXT[r['level']]}: {r['practice']}",
            "legal_basis": list(dict.fromkeys(c["legal_basis"] + (r.get("legal_basis") or []))),
            "source": r["source"],
        })

    for ex in examples:
        codes = [code_of_label[label] for label in ex["labels"]]
        docs.append({
            "id": f"ex.{ex['id']}", "type": "example", "categories": codes,
            "level": "potentially_unfair" if codes else "fair",
            "text": ex["text"].strip(),
            "labels": ex["labels"], "label_names": [label_names[i] for i in ex["labels"]],
            "source": "chalkidis2022",
        })
    return docs
