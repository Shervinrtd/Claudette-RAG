"""Tests for the demo pipeline and its rendering, with a fake embedder and a fake LLM (no download, no API)."""

import csv
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from claudette_rag.demo import (EXAMPLE_DOCUMENT, model_card, render_clause, render_document, short_citation,
                                write_csv)
from claudette_rag.pipeline import (EVIDENCE_ONLY, RAG_GATE, ClauseAnalyzer, load_gate, looks_like_heading,
                                    split_document, split_sentences)
from claudette_rag.retrieval import KBIndex, RetrievalConfig
from claudette_rag.text import readable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    {"id": "dir93.annex.1q", "type": "provision", "categories": [], "source": "dir93_13",
     "text": "Directive 93/13/EEC, Annex 1(q) - arbitration hinders legal action"},
    {"id": "dir93.annex.1g", "type": "provision", "categories": [], "source": "dir93_13",
     "text": "Directive 93/13/EEC, Annex 1(g) - terminate without reasonable notice"},
    {"id": "a.arb_obligatory", "type": "rationale", "categories": ["a"], "source": "ruggeri2022",
     "text": "Arbitration - potentially unfair: disputes resolved by an arbitrator",
     "legal_basis": ["dir93.annex.1q"]},
    {"id": "ter.no_notice", "type": "rationale", "categories": ["ter"], "source": "ruggeri2022",
     "text": "Unilateral termination - terminate the account without notice", "legal_basis": ["dir93.annex.1g"]},
]
ARBITRATION = ["all disputes will be resolved by binding arbitration", "disputes are settled by arbitration",
               "any dispute goes to binding arbitration in delaware"]
TERMINATION = ["we may terminate your account without notice", "we can terminate the account at any time"]
FAIR = ["contact us by email for support", "send an email to our support team", "our support team answers email",
        "you can delete your account in settings", "the service is provided in english"]
for n, text in enumerate(ARBITRATION):
    DOCS.append({"id": f"ex.a{n}", "type": "example", "categories": ["a"], "labels": [7], "text": text})
for n, text in enumerate(TERMINATION):
    DOCS.append({"id": f"ex.t{n}", "type": "example", "categories": ["ter"], "labels": [1], "text": text})
for n, text in enumerate(FAIR):
    DOCS.append({"id": f"ex.f{n}", "type": "example", "categories": [], "labels": [], "text": text})


class WordEmbedder:
    name = "fake-words"

    def __init__(self):
        words = sorted({w for d in DOCS for w in re.findall(r"[a-z]+", d["text"].lower())})
        self.vocab = {w: i for i, w in enumerate(words)}

    def encode(self, texts):
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                if word in self.vocab:
                    out[row, self.vocab[word]] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeCompletions:
    """Answers like an LLM: arbitration -> a (and an unsupported ch), terminate -> ter, otherwise fair."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def create(self, **kwargs):
        if self.fail:
            raise RuntimeError("server unreachable")
        sentence = kwargs["messages"][-1]["content"].rsplit("Sentence: ", 1)[1].lower()
        self.calls.append(sentence)
        if "arbitration" in sentence:
            answer = {"explanation": "Mandatory arbitration <b>removes</b> the courts [a.arb_obligatory].",
                      "cited_ids": ["[a.arb_obligatory]", "dir93.annex.1q", "made.up"], "unfair": True,
                      "categories": ["a", "ch"]}
        elif "terminate" in sentence:
            answer = {"explanation": "Termination without notice.", "cited_ids": ["ter.no_notice"],
                      "unfair": True, "categories": ["ter"]}
        else:
            answer = {"explanation": "A contact detail.", "cited_ids": [], "unfair": False, "categories": []}
        usage = SimpleNamespace(prompt_tokens=900, completion_tokens=60, prompt_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))],
                               usage=usage, system_fingerprint=None)


def fake_client(fail=False):
    completions = FakeCompletions(fail)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions), base_url="http://localhost:9/v1"), completions


@pytest.fixture()
def project(tmp_path):
    shutil.copytree(PROJECT_ROOT / "knowledge_base", tmp_path / "knowledge_base")
    KBIndex.build(DOCS, WordEmbedder()).save(tmp_path / "data" / "processed" / "index" / "bge-m3")
    (tmp_path / "results").mkdir()
    # gate: a category needs 2 of the 3 nearest precedents (the real choice is k=8, m=3)
    (tmp_path / "results" / "gate_choice_fake-model_e8r5p6.json").write_text(json.dumps({"k": 3, "m": 2}))
    return tmp_path


def analyzer_for(project, fail=False, model="fake-model"):
    client, completions = fake_client(fail)
    analyzer = ClauseAnalyzer(project, model=model, client=client, embedder=WordEmbedder(), workers=2)
    return analyzer, completions


# ----------------------------------------------------------------------------- text
def test_split_sentences_keeps_abbreviations_and_splits_real_ends():
    assert split_sentences("We may act (e.g. for fraud) under Art. 5 of U.S. law. You agree. \"Quoted\" too!") == [
        "We may act (e.g. for fraud) under Art. 5 of U.S. law.", "You agree.", "\"Quoted\" too!"]
    assert split_sentences("Per es. il Foro di Milano. Np. tak. Z.B. so.") == [
        "Per es. il Foro di Milano.", "Np. tak.", "Z.B. so."]
    assert split_sentences("version 2.0 is out. done") == ["version 2.0 is out. done"]   # no capital: same sentence


def test_split_document_joins_wrapped_lines_and_finds_headings():
    paragraphs = split_document("1. Terms\nWe may change\nthese terms at any time.\n\nYou agree.\r\nnew line")
    assert paragraphs == [["1. Terms"], ["We may change these terms at any time."], ["You agree."], ["new line"]]
    assert looks_like_heading(["1. Terms"]) and looks_like_heading(["§ 3 Governing law"])
    assert not looks_like_heading(["We may terminate your account at any time"])     # 8 words: analysed
    assert not looks_like_heading(["We may ban you."])                               # final period: a sentence
    assert not looks_like_heading(["One.", "Two"])


def test_readable_undoes_the_unfair_tos_tokenization():
    assert readable("by using the app ( `` stacks '' ) , you agree ; we wo n't charge $ 5 .") == \
        'By using the app ("stacks"), you agree; we won\'t charge $5.'


def test_short_citation():
    assert short_citation("Ruggeri, F., Lagioia, F., Lippi, M., Torroni, P. (2022). Detecting ...") == "Ruggeri et al. 2022"
    assert short_citation("Lagioia, F. (2026). Text analytics ...") == "Lagioia 2026"
    assert short_citation("Council Directive 93/13/EEC of 5 April 1993 on unfair terms") == "Directive 93/13/EEC"
    assert short_citation("CJEU, judgment of 28 July 2016, Amazon EU, Case C-191/15, EU:C:2016:612") == "CJEU C-191/15"


# ----------------------------------------------------------------------------- pipeline
def test_gate_choice_is_read_for_the_model_and_defaults_otherwise(project):
    config = RetrievalConfig()
    assert load_gate(project / "results", "fake-model", config)[0] == {"k": 3, "m": 2}
    gate, why = load_gate(project / "results", "other-model", config)
    assert gate == {"k": 8, "m": 3} and "not tuned" in why


def test_clause_is_flagged_and_the_gate_removes_an_unsupported_category(project):
    analyzer, completions = analyzer_for(project)
    a = analyzer.analyze_clause("All disputes shall be resolved by binding arbitration.")
    assert a.status == "flagged" and a.asked_model and len(completions.calls) == 1
    assert a.model_categories == ["ch", "a"] or a.model_categories == ["a", "ch"]
    assert a.categories == ["a"] and a.removed_by_gate == ["ch"]       # ch has no precedent
    assert a.votes["a"] >= 2 and a.votes["ch"] == 0
    assert a.cited == ["a.arb_obligatory", "dir93.annex.1q"] and a.invalid_cited == ["made.up"]
    assert a.usage["cost_usd"] == 0.0 and a.usage["prompt_tokens"] == 900   # local server: no cost


def test_one_clause_always_asks_the_model(project):
    analyzer, completions = analyzer_for(project)
    a = analyzer.analyze_clause("Contact us by email for support.")
    assert max(a.votes.values()) < analyzer.gate["m"]      # no precedent support...
    assert a.asked_model and a.status == "not_flagged"     # ...but the model's reasoning is still shown


def test_document_shortcut_skips_the_model_only_where_the_gate_cannot_flag(project):
    analyzer, completions = analyzer_for(project)
    text = ("1. Disputes\nAll disputes go to binding arbitration. Send an email to our support team.\n"
            "We may terminate your account without notice. The service is provided in English.")
    segments = analyzer.analyze_document(text)
    analysed = [s for s in segments if s.analysis]
    assert [s.text for s in segments if not s.analysis] == ["1. Disputes"]         # heading, not analysed
    asked = [s for s in analysed if s.analysis.asked_model]
    assert len(completions.calls) == len(asked) == 2
    assert all(max(s.analysis.votes.values()) < analyzer.gate["m"] for s in analysed if not s.analysis.asked_model)
    # exactly the verdicts of the full system, which asks the model about every sentence
    full = analyzer.analyze_sentences([s.text for s in analysed], always_ask=True)
    assert [s.analysis.categories for s in analysed] == [f.categories for f in full]
    assert [s.analysis.categories for s in analysed] == [["a"], [], ["ter"], []]


def test_evidence_only_needs_no_model(project):
    analyzer = ClauseAnalyzer(project, embedder=WordEmbedder())
    assert analyzer.modes == [EVIDENCE_ONLY]
    a = analyzer.analyze_clause("Disputes are settled by binding arbitration.", EVIDENCE_ONLY)
    assert a.categories == ["a"] and not a.asked_model and a.gate == {"k": 3, "t": 2}
    with pytest.raises(ValueError):
        analyzer.analyze_clause("Disputes are settled by binding arbitration.", RAG_GATE)


def test_a_failed_model_call_gives_no_verdict(project):
    analyzer, _ = analyzer_for(project, fail=True)
    a = analyzer.analyze_clause("All disputes shall be resolved by binding arbitration.")
    assert a.status == "error" and a.categories == [] and "server unreachable" in a.error
    page = render_clause(a, analyzer)
    assert "No verdict" in page and "No potentially unfair clause found" not in page


# ----------------------------------------------------------------------------- rendering
def test_rendering_escapes_text_from_the_user_and_the_model(project):
    analyzer, _ = analyzer_for(project)
    a = analyzer.analyze_clause('<script>alert("x")</script> disputes go to arbitration')
    page = render_clause(a, analyzer)
    assert "<script>" not in page
    assert "<b>removes</b>" not in page and "&lt;b&gt;removes&lt;/b&gt;" in page     # the model's text
    document = render_document(analyzer.analyze_document('<img src=x onerror="alert(1)"> We may terminate '
                                                         'your account without notice.'), analyzer)
    assert "<img" not in document and "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in document
    assert "Potentially unfair" in page and "Arbitration" in page and "made.up" in page
    assert "eur-lex.europa.eu" in page and "[Directive 93/13/EEC]" in page   # provision -> official text


def test_document_page_and_csv(project, tmp_path):
    analyzer, _ = analyzer_for(project)
    segments = analyzer.analyze_document(EXAMPLE_DOCUMENT)
    page = render_document(segments, analyzer)
    flagged = [s for s in segments if s.analysis and s.analysis.status == "flagged"]
    assert f"Flagged clauses ({len(flagged)})" in page and flagged
    path = write_csv(segments, analyzer, tmp_path / "out.csv")
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(segments)
    assert {r["status"] for r in rows} <= {"flagged", "not flagged", "error", "not analysed"}


def test_model_card_reports_the_measured_results(project):
    analyzer, _ = analyzer_for(project)
    assert "not legal advice" in model_card(project, analyzer)
    report = {"headline": {"lexglue_macro_f1": 0.794, "binary_precision": 0.823, "binary_recall": 0.837}}
    (project / "results" / "gated_k3m2_rag_fake-model_e8r5p6_test.json").write_text(json.dumps(report))
    card = model_card(project, analyzer)
    assert "| RAG + evidence gate (fake-model) | 79.4 | 82.3 | 83.7 |" in card


def test_the_app_builds(project):
    pytest.importorskip("gradio")
    from claudette_rag.demo import build_app
    analyzer, _ = analyzer_for(project)
    assert build_app(analyzer, project) is not None
