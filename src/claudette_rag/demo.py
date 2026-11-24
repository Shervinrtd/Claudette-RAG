"""Web demo: paste a clause or a whole Terms of Service and see the verdict, the explanation and the evidence.

The rendering functions return HTML strings and do not need Gradio, so they can be tested on their own;
build_app() assembles the Gradio interface (launched by scripts/demo.py). Every piece of text that comes
from the user, the model or the knowledge base is HTML-escaped before it is shown.
"""

from __future__ import annotations

import csv
import html
import json
import re
import tempfile
from pathlib import Path

from .llm import file_safe
from .pipeline import (EVIDENCE_ONLY, MIN_WORDS, RAG_GATE, Analysis, ClauseAnalyzer, Segment, split_sentences,
                       word_count)
from .text import readable

MAX_CLAUSE_CHARS = 3000
MAX_DOCUMENT_CHARS = 200_000

# Our own example clauses (not taken from the datasets), in the four languages of the multilingual evaluation.
EXAMPLE_CLAUSES = [
    "We reserve the right to modify these Terms at any time without notice; your continued use of the Service "
    "after any change constitutes acceptance of the new Terms.",
    "Any dispute arising out of or relating to these Terms shall be resolved exclusively through final and binding "
    "arbitration, and you waive your right to a trial by jury.",
    "To the fullest extent permitted by law, we shall not be liable for any loss of data, loss of profits or "
    "interruption of the Service, however caused.",
    "You can delete your account at any time from the Settings page, and we will refund any prepaid fees for the "
    "remaining period.",
    "Wir können Ihr Konto jederzeit und ohne Angabe von Gründen sperren oder kündigen.",
    "Per qualsiasi controversia relativa ai presenti Termini sarà competente in via esclusiva il Foro di Milano.",
    "Zastrzegamy sobie prawo do usunięcia dowolnych treści przesłanych przez użytkownika, według własnego uznania "
    "i bez uprzedzenia.",
]

# A short Terms of Service of an invented company, written for the demo: fair and unfair clauses mixed.
EXAMPLE_DOCUMENT = """Nimbus Notes - Terms of Service
Last updated: 1 March 2026

1. Acceptance of the Terms
By creating an account or by using Nimbus Notes, you agree to be bound by these Terms. If you do not agree, please do not use the service.

2. Your account
You must be at least 16 years old to use Nimbus Notes. You are responsible for keeping your password confidential. You can delete your account at any time from the Settings page.

3. Changes to the service and to these Terms
We may modify or discontinue any part of the service, and we may amend these Terms at any time at our sole discretion. Your continued use of the service after the changes take effect means that you accept the new Terms.

4. Your content
You keep all rights to the notes you create. We may remove any content that we consider inappropriate, without prior notice and without giving reasons.

5. Termination
We may suspend or terminate your account at any time, for any reason or no reason, without liability to you. If we terminate your account because we discontinue the service, we will refund any prepaid fees for the remaining period.

6. Liability
To the fullest extent permitted by law, Nimbus Notes shall not be liable for any loss of data, loss of profits or any indirect damages arising from your use of the service. Nothing in these Terms limits our liability for death or personal injury caused by our negligence.

7. Disputes
These Terms are governed by the laws of the State of Delaware, without regard to its conflict of law rules. Any dispute arising from these Terms shall be settled exclusively by binding arbitration in Wilmington, Delaware, and you waive any right to take part in a class action.

8. Contact
If you have questions about these Terms, you can write to us at support@nimbusnotes.example.
"""

CSS = """
.cr { font-size: 15px; line-height: 1.5; color: var(--body-text-color); }
.cr h4 { margin: 18px 0 6px; font-size: 15px; font-weight: 600; }
.cr p { margin: 4px 0; }
.cr-verdict { border-left: 4px solid; border-radius: 6px; padding: 10px 14px; margin-bottom: 6px; }
.cr-verdict-title { font-size: 18px; font-weight: 600; }
.cr-verdict-sub { margin-top: 2px; color: var(--body-text-color-subdued); }
.cr-flagged { border-color: #d97706; background: rgba(217, 119, 6, 0.12); }
.cr-clear { border-color: #16a34a; background: rgba(22, 163, 74, 0.10); }
.cr-overruled { border-color: #64748b; background: rgba(100, 116, 139, 0.12); }
.cr-error { border-color: #dc2626; background: rgba(220, 38, 38, 0.10); }
.cr-chip { display: inline-block; border: 1px solid var(--border-color-primary); border-radius: 999px;
           padding: 0 8px; margin: 1px 3px 1px 0; font-size: 13px; white-space: nowrap; }
.cr-id { font-family: var(--font-mono, monospace); font-size: 12px; color: var(--body-text-color-subdued); }
.cr-muted { color: var(--body-text-color-subdued); font-size: 13px; }
.cr-note { border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 6px 10px; margin: 8px 0; }
.cr ul { margin: 4px 0 4px 18px; padding: 0; }
.cr li { margin: 3px 0; }
.cr table { border-collapse: collapse; width: 100%; font-size: 14px; border: none !important; }
.cr th, .cr td { border: none !important; border-bottom: 1px solid var(--border-color-primary) !important;
                 padding: 5px 6px; text-align: left; vertical-align: top; }
.cr th { font-weight: 600; color: var(--body-text-color-subdued); font-size: 13px; }
.cr-votes { display: grid; grid-template-columns: max-content minmax(120px, 260px) max-content max-content;
            gap: 4px 10px; align-items: center; margin: 6px 0; }
.cr-track { position: relative; height: 8px; border-radius: 4px; background: var(--border-color-primary); }
.cr-bar { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: #4f46e5; }
.cr-bar.cr-below { opacity: 0.45; }
.cr-threshold { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--body-text-color); }
.cr-steps { margin-left: 20px; }
.cr-meta { margin-top: 14px; }
.cr-summary { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 8px; }
.cr-stat { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 8px 14px; min-width: 120px; }
.cr-stat b { display: block; font-size: 22px; }
.cr details { border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 6px 10px; margin: 6px 0; }
.cr summary { cursor: pointer; }
.cr-doc p { margin: 0 0 10px; }
.cr-doc mark.cr-hl { background: rgba(217, 119, 6, 0.25); color: inherit; border-radius: 3px; padding: 0 2px; }
.cr-doc .cr-borderline { text-decoration: underline dotted #64748b; text-underline-offset: 3px; }
.cr-doc .cr-err { text-decoration: underline wavy #dc2626; }
.cr-doc .cr-heading { font-weight: 600; }
.cr-doc sup { font-size: 11px; color: var(--body-text-color-subdued); }
.cr-header h1 { margin: 0; font-size: 26px; }
.cr-header p { margin: 4px 0 0; }
"""


def esc(text) -> str:
    return html.escape("" if text is None else str(text), quote=True)


# ----------------------------------------------------------------------------- pieces
def _name(analyzer: ClauseAnalyzer, code: str) -> str:
    return analyzer.category[code]["name"]


def _chips(analyzer: ClauseAnalyzer, codes: list[str]) -> str:
    return "".join(f'<span class="cr-chip">{esc(_name(analyzer, c))}</span>' for c in codes)


def _names(analyzer: ClauseAnalyzer, codes: list[str]) -> str:
    return ", ".join(_name(analyzer, c) for c in codes)


def _verdict(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    k = a.gate["k"]
    if a.status == "error":
        css, title = "cr-error", "&#10006; No verdict: the model could not answer"
        sub = (f"{esc(a.error)}<br>Try again, or choose <i>Evidence only</i>, which needs no language model. "
               "This is not a sign that the clause is fair.")
    elif a.status == "flagged":
        css = "cr-flagged"
        title = f"&#9888; Potentially unfair: {_chips(analyzer, a.categories)}"
        support = "; ".join(f"{esc(_name(analyzer, c))}: {a.votes[c]} of {k}" for c in a.categories)
        if a.mode == RAG_GATE:
            sub = ("Flagged for a closer look. The model's verdict is backed by similar clauses that legal "
                   f"experts labelled with the same category ({support}).")
            if a.removed_by_gate:
                removed = "; ".join(f"{esc(_name(analyzer, c))}: {a.votes[c]} of {k}" for c in a.removed_by_gate)
                sub += (" The model also suggested a category that too few of them have, so it was removed "
                        f"({removed}; at least {a.gate['m']} needed).")
        else:
            sub = (f"Evidence only: similar clauses labelled by legal experts have this category ({support}; "
                   f"at least {a.gate['t']} needed). No language model was used, so there is no written explanation.")
    elif a.removed_by_gate:
        css, title = "cr-overruled", "&#9673; Not flagged: the evidence gate overruled the model"
        detail = "; ".join(f"{esc(_name(analyzer, c))}: {a.votes[c]} of {k}" for c in a.removed_by_gate)
        sub = (f"The model suggested {esc(_names(analyzer, a.removed_by_gate))}, but too few similar clauses "
               f"labelled by legal experts have this category ({detail}; at least {a.gate['m']} needed). "
               "A borderline case: worth reading if in doubt.")
    else:
        css, title = "cr-clear", "&#10004; No potentially unfair clause found"
        sub = ("This is not a guarantee that the clause is fair: the system misses some unfair clauses "
               "(see the About tab).")
    return (f'<div class="cr-verdict {css}"><div class="cr-verdict-title">{title}</div>'
            f'<div class="cr-verdict-sub">{sub}</div></div>')


def _why(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    parts = []
    if a.explanation:
        parts.append(f"<h4>The model's explanation</h4><p>{esc(a.explanation)}</p>")
        if not a.consistent:
            parts.append('<p class="cr-muted">The model\'s answer was inconsistent (its yes/no verdict and its '
                         'categories disagree); as in the evaluation, it is read as "not unfair".</p>')
    shown = a.categories or a.removed_by_gate
    if shown:
        parts.append("<h4>Why this kind of clause matters</h4><ul>" + "".join(
            f"<li><b>{esc(_name(analyzer, c))}:</b> {esc(analyzer.category[c]['why_it_matters'])}</li>"
            for c in shown) + "</ul>")
    return "".join(parts)


def short_citation(citation: str) -> str:
    """Bibliography entry -> short label: "Lippi et al. 2019", "Directive 93/13/EEC", "CJEU C-191/15"."""
    if citation.startswith("CJEU"):
        case = re.search(r"Cases? (C-[\d/]+)", citation)
        return f"CJEU {case.group(1)}" if case else "CJEU"
    author, year = re.match(r"([^,(]+), [A-Z]\.", citation), re.search(r"\((\d{4})\)", citation)
    if author and year:
        several = citation.split("(")[0].count(".,") >= 1
        return f"{author.group(1).strip()}{' et al.' if several else ''} {year.group(1)}"
    return citation.split(" of ")[0].replace("Council ", "").strip()[:60]


def _legal_item(doc: dict, analyzer: ClauseAnalyzer) -> str:
    source = analyzer.source_of(doc)
    if doc["id"] in analyzer.provisions:
        p = analyzer.provisions[doc["id"]]
        text = f"<b>{esc(p['reference'])}</b> - {esc(p['title'])}: {esc(p['summary'])}"
    else:
        text = esc(doc["text"])
    link = ""
    if source:
        label = esc(short_citation(source["citation"])) if source.get("citation") else ""
        url = source.get("url")
        link = (f' <a class="cr-muted" href="{esc(url)}" target="_blank" rel="noopener">[{label or "source"}]</a>'
                if url else f' <span class="cr-muted">[{label}]</span>')
    return f'<li>{text}{link} <span class="cr-id">{esc(doc["id"])}</span></li>'


def _legal(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    if a.mode != RAG_GATE:
        return ""
    legal = [doc for doc, _ in a.context.get("rationales", [])] + list(a.context.get("provisions", []))
    cited = [d for d in legal if d["id"] in a.cited]
    other = [d for d in legal if d["id"] not in a.cited]
    parts = []
    if cited:
        parts.append("<h4>Legal knowledge cited by the model</h4><ul>"
                     + "".join(_legal_item(d, analyzer) for d in cited) + "</ul>")
    elif a.asked_model:
        parts.append('<h4>Legal knowledge cited by the model</h4><p class="cr-muted">None: the model cited no '
                     'legal rationale or provision.</p>')
    if a.invalid_cited:
        parts.append(f'<p class="cr-muted">The model also cited {len(a.invalid_cited)} id(s) that were not among the '
                     f'retrieved items; they are ignored: {esc(", ".join(a.invalid_cited))}</p>')
    if other:
        title = "Other retrieved legal knowledge, not cited" if a.asked_model else "Retrieved legal knowledge"
        parts.append(f"<details><summary>{title} ({len(other)})</summary><ul>"
                     + "".join(_legal_item(d, analyzer) for d in other) + "</ul></details>")
    return "".join(parts)


def _vote_bars(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    k = a.gate["k"]
    need = a.gate["m"] if a.mode == RAG_GATE else a.gate["t"]
    codes = [c for c in analyzer.codes if a.votes[c] or c in a.model_categories or c in a.categories]
    if not codes:
        return (f'<p>None of the {k} most similar clauses was labelled unfair by the experts, so no category '
                f'can reach the {need} needed.</p>')
    rows = []
    for c in codes:
        v = a.votes[c]
        ok = v >= need
        rows.append(
            f'<span>{esc(_name(analyzer, c))}</span>'
            f'<span class="cr-track" role="img" aria-label="{v} of {k}">'
            f'<span class="cr-bar{"" if ok else " cr-below"}" style="width:{100 * v / k:.0f}%"></span>'
            f'<span class="cr-threshold" style="left:calc({100 * need / k:.0f}% - 1px)"></span></span>'
            f'<span>{v} of {k}</span>'
            f'<span class="cr-muted">{"&#10004; enough evidence" if ok else "below the threshold"}</span>')
    return (f'<p class="cr-muted">For each category: how many of the {k} most similar clauses the experts labelled '
            f'with it. The vertical mark is the threshold ({need} of {k}).</p>'
            f'<div class="cr-votes">{"".join(rows)}</div>')


def _precedents(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    k = a.gate["k"]
    rows = []
    for rank, (doc, score) in enumerate(a.precedents, start=1):
        label = _chips(analyzer, doc["categories"]) if doc["categories"] else '<span class="cr-muted">fair</span>'
        cited = ' <span class="cr-chip">cited</span>' if doc["id"] in a.cited else ""
        used = "" if rank <= k else ' <span class="cr-muted">(not used by the threshold)</span>'
        rows.append(f"<tr><td>{rank}</td><td>{score:.2f}</td><td>{label}{cited}{used}</td>"
                    f'<td>{esc(readable(doc["text"]))} <span class="cr-id">{esc(doc["id"])}</span></td></tr>')
    return ("<h4>Precedents: the most similar clauses, labelled by legal experts</h4>"
            + _vote_bars(a, analyzer)
            + "<details><summary>Show the precedents</summary>"
              "<table><tr><th>#</th><th>Similarity</th><th>Expert label</th><th>Clause</th></tr>"
            + "".join(rows) + "</table></details>")


def _steps(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    ctx = a.context
    n_unfair = sum(bool(doc["categories"]) for doc, _ in ctx.get("examples", []))
    retrieved = (f"Retrieved the {len(ctx.get('examples', []))} most similar clauses labelled by legal experts "
                 f"({n_unfair} of them unfair)")
    if a.mode == EVIDENCE_ONLY:
        steps = [retrieved + ".",
                 f"A category is flagged when at least {a.gate['t']} of the {a.gate['k']} most similar clauses have it "
                 "(the kNN baseline; no language model).",
                 "Verdict: " + (esc(_names(analyzer, a.categories)) if a.categories else "not flagged") + "."]
        return "<h4>How this verdict was reached</h4><ol class='cr-steps'>" + "".join(
            f"<li>{s}</li>" for s in steps) + "</ol>"
    steps = [retrieved + f", {len(ctx.get('rationales', []))} legal rationales and "
             f"{len(ctx.get('provisions', []))} legal provisions."]
    if a.error:
        steps.append(f"The language model ({esc(analyzer.model)}) could not answer: {esc(a.error)}.")
    elif not a.asked_model:
        steps.append(f"The model was not asked: no category has {a.gate['m']} of {a.gate['k']} precedents, so the "
                     "evidence gate could not keep any category, whatever the model answered.")
    else:
        proposal = esc(_names(analyzer, a.model_categories)) if a.model_categories else "not unfair"
        steps.append(f"The language model ({esc(analyzer.model)}) read the clause and the retrieved knowledge "
                     f"and answered: {proposal}.")
        n_cited = len(a.cited) + len(a.invalid_cited)
        steps.append(f"Citation check: {len(a.cited)} of the {n_cited} cited ids were retrieved."
                     if n_cited else "Citation check: the model cited no retrieved item.")
        if a.model_categories:
            checks = "; ".join(
                f"{esc(_name(analyzer, c))}: {a.votes[c]} of {a.gate['k']} "
                + ("&#10004; kept" if c in a.categories else "&#10006; removed") for c in a.model_categories)
            steps.append(f"Evidence gate (a category needs {a.gate['m']} of the {a.gate['k']} precedents): {checks}.")
    if not a.error:
        steps.append("Verdict: " + (esc(_names(analyzer, a.categories)) if a.categories else "not flagged") + ".")
    return "<h4>How this verdict was reached</h4><ol class='cr-steps'>" + "".join(
        f"<li>{s}</li>" for s in steps) + "</ol>"


def _meta(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    if a.mode == EVIDENCE_ONLY:
        return f'<p class="cr-meta cr-muted">Evidence only (kNN over {esc(analyzer.index.model_name)} embeddings).</p>'
    if not a.usage:
        return ""
    u = a.usage
    cost = "no API cost (local model)" if analyzer.local else f"about ${u['cost_usd']:.4f}"
    return (f'<p class="cr-meta cr-muted">{esc(analyzer.model)} &middot; {u["latency_s"]:.1f} s &middot; '
            f'{u["prompt_tokens"] + u["completion_tokens"]:,} tokens &middot; {cost}</p>')


def render_details(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    """Explanation and evidence of one analysed sentence (without the verdict box)."""
    return _why(a, analyzer) + _legal(a, analyzer) + _precedents(a, analyzer) + _steps(a, analyzer) + _meta(a, analyzer)


def render_clause(a: Analysis, analyzer: ClauseAnalyzer) -> str:
    note = ""
    if len(split_sentences(a.text)) > 1:
        note = ('<p class="cr-note">This text has more than one sentence. The system was built and evaluated on '
                'single sentences: for longer text, the <b>Check a document</b> tab analyses each sentence '
                'separately.</p>')
    return f'<div class="cr">{note}{_verdict(a, analyzer)}{render_details(a, analyzer)}</div>'


def notice(text: str) -> str:
    return f'<div class="cr"><p class="cr-note">{esc(text)}</p></div>'


# ----------------------------------------------------------------------------- documents
def _is_borderline(a: Analysis | None) -> bool:
    return bool(a and a.status == "not_flagged" and a.removed_by_gate)


def render_document(segments: list[Segment], analyzer: ClauseAnalyzer) -> str:
    analysed = [s for s in segments if s.analysis]
    if not analysed:
        return notice(f"No sentence to analyse: the demo analyses sentences of at least {MIN_WORDS} words.")
    number = {id(s): n for n, s in enumerate(analysed, start=1)}
    flagged = [s for s in analysed if s.analysis.status == "flagged"]
    borderline = [s for s in analysed if _is_borderline(s.analysis)]
    errors = [s for s in analysed if s.analysis.status == "error"]
    mode = analysed[0].analysis.mode

    stats = [(len(analysed), "sentences analysed"), (len(flagged), "flagged as potentially unfair")]
    if mode == RAG_GATE:
        asked = sum(s.analysis.asked_model or bool(s.analysis.error) for s in analysed)
        cost = sum((s.analysis.usage or {}).get("cost_usd", 0.0) for s in analysed)
        stats.append((asked, "sent to the language model"))
        stats.append(("free" if analyzer.local else f"${cost:.3f}", "estimated cost"))
    parts = ['<div class="cr"><div class="cr-summary">' + "".join(
        f'<div class="cr-stat"><b>{esc(v)}</b>{esc(label)}</div>' for v, label in stats) + "</div>"]

    counts = {c: sum(c in s.analysis.categories for s in flagged) for c in analyzer.codes}
    if flagged:
        parts.append("<p>Categories found: " + " ".join(
            f'<span class="cr-chip">{esc(_name(analyzer, c))}: {n}</span>' for c, n in counts.items() if n) + "</p>")
    if mode == RAG_GATE:
        skipped = len(analysed) - sum(s.analysis.asked_model or bool(s.analysis.error) for s in analysed)
        parts.append(f'<p class="cr-muted">{skipped} sentences were not sent to the model: none of their '
                     f'{analyzer.gate["k"]} most similar expert-labelled clauses shares a category often enough '
                     f'(at least {analyzer.gate["m"]} times), so the evidence gate could not flag them whatever the '
                     'model answered. This saves cost and changes no verdict.</p>')
    if errors:
        parts.append(f'<div class="cr-verdict cr-error"><div class="cr-verdict-title">&#10006; {len(errors)} '
                     'sentences have no verdict</div><div class="cr-verdict-sub">The model could not answer: '
                     f'{esc(errors[0].analysis.error)}. They are marked with a wavy line below; do not read them '
                     'as fair.</div></div>')

    def item(s: Segment) -> str:
        a = s.analysis
        chips = _chips(analyzer, a.categories or a.removed_by_gate)
        return (f'<details><summary><span class="cr-muted">#{number[id(s)]}</span> {esc(s.text)} {chips}</summary>'
                f'{_verdict(a, analyzer)}{render_details(a, analyzer)}</details>')

    parts.append(f"<h4>Flagged clauses ({len(flagged)})</h4>")
    parts.append("".join(item(s) for s in flagged) if flagged else
                 '<p>None. This is not a guarantee that the document is fair: see the About tab for how often '
                 'the system misses unfair clauses.</p>')
    if borderline:
        parts.append(f"<h4>Borderline: suggested by the model, not confirmed by precedents ({len(borderline)})</h4>"
                     + "".join(item(s) for s in borderline))

    parts.append('<h4>Full text</h4><p class="cr-muted"><mark class="cr-hl" style="background:rgba(217,119,6,.25)">'
                 'highlighted</mark> = flagged; <span style="text-decoration:underline dotted #64748b">dotted</span> = '
                 'borderline; <span style="text-decoration:underline wavy #dc2626">wavy</span> = no verdict; '
                 '<b>bold</b> = heading or short piece, not analysed.</p><div class="cr-doc">')
    paragraph, line = 0, []
    for s in segments + [Segment("", -1)]:
        if s.paragraph != paragraph:
            parts.append("<p>" + " ".join(line) + "</p>")
            paragraph, line = s.paragraph, []
        if s.paragraph < 0:
            break
        a = s.analysis
        if a is None:
            line.append(f'<span class="cr-heading">{esc(s.text)}</span>')
        elif a.status == "flagged":
            names = esc(_names(analyzer, a.categories))
            line.append(f'<mark class="cr-hl" title="{names}">{esc(s.text)}</mark><sup>#{number[id(s)]} {names}</sup>')
        elif _is_borderline(a):
            line.append(f'<span class="cr-borderline" title="suggested: {esc(_names(analyzer, a.removed_by_gate))}">'
                        f'{esc(s.text)}</span><sup>#{number[id(s)]}</sup>')
        elif a.status == "error":
            line.append(f'<span class="cr-err" title="no verdict">{esc(s.text)}</span>')
        else:
            line.append(esc(s.text))
    parts.append("</div></div>")
    return "".join(parts)


def write_csv(segments: list[Segment], analyzer: ClauseAnalyzer, path: Path | None = None) -> Path:
    """One row per piece of the document, for reading in a spreadsheet."""
    if path is None:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", prefix="claudette_rag_", delete=False)
        handle.close()
        path = Path(handle.name)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "sentence", "status", "categories", "suggested_by_model_not_confirmed",
                         "explanation", "cited_ids", "precedent_votes", "sent_to_model", "error"])
        n = 0
        for s in segments:
            a = s.analysis
            if a is None:
                writer.writerow(["", s.text, "not analysed", "", "", "", "", "", "", ""])
                continue
            n += 1
            writer.writerow([n, s.text, a.status.replace("_", " "), _names(analyzer, a.categories),
                             _names(analyzer, a.removed_by_gate), a.explanation or "", " ".join(a.cited),
                             json.dumps({c: v for c, v in a.votes.items() if v}), "yes" if a.asked_model else "no",
                             a.error or ""])
    return Path(path)


# ----------------------------------------------------------------------------- about / model card
def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def measured_performance(project_root: Path, analyzer: ClauseAnalyzer) -> dict:
    """Test-set and per-language results of the systems the demo runs, read from results/."""
    results = Path(project_root) / "results"
    gate = analyzer.gate
    candidates = ([analyzer.model] if analyzer.model else []) + ["gpt-4.1-mini-2025-04-14"]
    test, measured_with = None, None
    for model in candidates:
        test = _load(results / f"gated_k{gate['k']}m{gate['m']}_rag_{file_safe(model)}_e8r5p6_test.json")
        if test:
            measured_with = model
            break
    # per-language results of the running model if it was evaluated, otherwise of the evaluated open-weight model
    own = results / f"multilingual_fairness_{file_safe(analyzer.model)}.json" if analyzer.model else None
    files = sorted(results.glob("multilingual_fairness_*.json"))
    fairness = _load(own) if own and own.exists() else (_load(files[0]) if files else None)
    return {"test": test, "test_model": measured_with, "knn_test": _load(results / "knn_bge-m3_test.json"),
            "fairness": fairness, "fairness_model": fairness["model"] if fairness else None}


def model_card(project_root: Path, analyzer: ClauseAnalyzer) -> str:
    perf = measured_performance(project_root, analyzer)
    running = (f"**{analyzer.model}**" + (" (local server)" if analyzer.local else " (OpenAI API)")
               if analyzer.model else "no language model: only *Evidence only* is available")
    lines = [
        "### What this is",
        "A research prototype from the course project *Claudette-RAG* (Ethics in AI, University of Bologna). It reads "
        "sentences of online Terms of Service and flags those that may be unfair to consumers under EU consumer law "
        "(Council Directive 93/13/EEC), in eight categories: limitation of liability, unilateral termination, "
        "unilateral change, content removal, contract by using, choice of law, jurisdiction and arbitration.",
        "",
        "**It is not legal advice.** It points a human reader to the clauses that deserve a closer look and shows "
        "the evidence behind each verdict, so that the reader can judge it. You are interacting with an AI system.",
        "",
        f"Language model running now: {running}. Retrieval: {esc(analyzer.index.model_name)} embeddings. "
        f"Evidence gate: k = {analyzer.gate['k']}, m = {analyzer.gate['m']} ({esc(analyzer.gate_source)}).",
        "",
        "### How it decides (RAG + evidence gate)",
        "1. **Retrieval.** The 8 most similar sentences from Terms of Service labelled by legal experts (the training "
        "data of UNFAIR-ToS), 5 legal rationales and the legal provisions they rely on.",
        "2. **Language model.** It reads the sentence and this knowledge, explains its decision, cites the items it "
        "relies on and chooses the categories.",
        "3. **Citation check.** Cited ids that were not retrieved are reported and ignored.",
        f"4. **Evidence gate.** A category is kept only if at least {analyzer.gate['m']} of the {analyzer.gate['k']} "
        "most similar expert-labelled sentences have it. On the English test set (gpt-4.1-mini), this cut the "
        "model's false alarms from 131 to 31, at the cost of 16 unfair sentences no longer found (160 to 144 of 172).",
        "",
        "*Evidence only* skips the language model: a category is flagged when at least "
        f"{analyzer.knn['t']} of the {analyzer.knn['k']} most similar expert-labelled sentences have it. It is free "
        "and runs offline, but gives no written explanation.",
        "",
        "### How well it works (measured on data it never saw)",
    ]
    rows = []
    if perf["test"]:
        h = perf["test"]["headline"]
        rows.append(f"| RAG + evidence gate ({esc(perf['test_model'])}) | {_pct(h['lexglue_macro_f1'])} | "
                    f"{_pct(h['binary_precision'])} | {_pct(h['binary_recall'])} |")
    if perf["knn_test"]:
        h = perf["knn_test"]["headline"]
        rows.append(f"| Evidence only (kNN) | {_pct(h['lexglue_macro_f1'])} | {_pct(h['binary_precision'])} | "
                    f"{_pct(h['binary_recall'])} |")
    if rows:
        lines += ["English test set of UNFAIR-ToS (1,607 sentences, 172 unfair):", "",
                  "| System | Macro-F1 | Precision | Recall |", "|---|---|---|---|", *rows, "",
                  "*Precision*: of the sentences the system flags, the share that legal experts also labelled unfair. "
                  "*Recall*: of the unfair sentences, the share the system flags.", ""]
        if perf["test_model"] and analyzer.model and perf["test_model"] != analyzer.model:
            lines += [f"The model running now was not evaluated; the numbers above are for {esc(perf['test_model'])}.", ""]
    if perf["fairness"]:
        systems = perf["fairness"]["systems"]
        table = []
        for name, label in (("RAG + gate", "RAG + evidence gate"), ("kNN vote", "Evidence only (kNN)")):
            key = next((s for s in systems if s.startswith(name)), None)
            if key:
                langs = systems[key]["languages"]
                table.append(f"| {label} | " + " | ".join(
                    f"{_pct(langs[l]['recall'])} / {_pct(langs[l]['false_alarm_rate'])}" for l in ("en", "de", "it", "pl"))
                             + " |")
        if table:
            lines += ["### By language",
                      f"Recall / false-alarm rate (%) on the multilingual Claudette set: Terms of Service of the same "
                      f"10 companies in four languages (measured with {esc(perf['fairness_model'])}):", "",
                      "| System | English | German | Italian | Polish |", "|---|---|---|---|---|", *table, "",
                      "For both systems, no difference from English is statistically significant, but German is "
                      "always the lowest. English-only models (Legal-BERT, TF-IDF) flag almost nothing in the "
                      "other languages.",
                      ""]
    lines += [
        "### Limits",
        "- **It misses unfair clauses**: about 1 in 6 on the English test set, about 1 in 4 on the multilingual set. "
        "\"Not flagged\" does not mean \"fair\".",
        "- **A convincing explanation is not proof.** In our evaluation, wrong flags cited legal sources almost as "
        "often as correct ones. The expert-labelled precedents are the better check: look at them.",
        "- **One sentence at a time.** Under the Directive (Article 4) a term is judged in the context of the whole "
        "contract; the system does not see that context.",
        "- **Its knowledge is the training data**: Terms of Service of large online platforms. Unusual services, "
        "newer kinds of clauses, or other languages than English, German, Italian and Polish were not evaluated.",
        "",
        "### Your text",
        "With the OpenAI model, the text you check is sent to the OpenAI API; with a local model, it stays on this "
        "computer. The demo stores nothing (the CSV of a document is a temporary file for you to download). Do not "
        "paste personal data.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the app
MODE_LABELS = {RAG_GATE: "RAG + evidence gate (language model)", EVIDENCE_ONLY: "Evidence only (no language model, free)"}


def build_app(analyzer: ClauseAnalyzer, project_root: Path):
    import gradio as gr

    choices = [(MODE_LABELS[m], m) for m in analyzer.modes]
    model_line = (f"Language model: {esc(analyzer.model)}" if analyzer.model
                  else "No language model configured: evidence only")
    header = (
        '<div class="cr cr-header"><h1>Claudette-RAG</h1>'
        "<p>Finds potentially unfair clauses in Terms of Service, and shows the legal reasons and the "
        "expert-labelled precedents behind each verdict.</p>"
        f'<p class="cr-muted">AI system &middot; research prototype, not legal advice &middot; {model_line}</p></div>')

    def check_clause(text: str, mode: str) -> str:
        text = (text or "").strip()
        if word_count(text) < MIN_WORDS:
            return notice("Paste a clause (a sentence of a Terms of Service) first.")
        if len(text) > MAX_CLAUSE_CHARS:
            return notice(f"The clause is longer than {MAX_CLAUSE_CHARS} characters: use the Check a document tab.")
        return render_clause(analyzer.analyze_clause(text, mode), analyzer)

    def check_document(text: str, mode: str, progress=gr.Progress()):
        text = (text or "").strip()
        if not text:
            return notice("Paste the text of a Terms of Service first."), None
        if len(text) > MAX_DOCUMENT_CHARS:
            return notice(f"The text is longer than {MAX_DOCUMENT_CHARS:,} characters."), None
        try:
            segments = analyzer.analyze_document(text, mode, progress=lambda f, d: progress(f, desc=d))
        except ValueError as error:
            return notice(str(error)), None
        if not any(s.analysis for s in segments):
            return render_document(segments, analyzer), None
        return render_document(segments, analyzer), str(write_csv(segments, analyzer))

    with gr.Blocks(title="Claudette-RAG: unfair clauses in Terms of Service") as app:
        gr.HTML(header)
        with gr.Tabs():
            with gr.Tab("Check a clause"):
                clause = gr.Textbox(lines=4, label="Clause",
                                    placeholder="Paste one sentence of a Terms of Service (English, German, "
                                                "Italian or Polish)")
                with gr.Row():
                    clause_mode = gr.Radio(choices, value=analyzer.modes[0], label="System", scale=3)
                    clause_button = gr.Button("Check the clause", variant="primary", scale=1)
                gr.Examples([[c] for c in EXAMPLE_CLAUSES], inputs=[clause], label="Examples (our own wording)")
                clause_out = gr.HTML()
                clause_button.click(check_clause, [clause, clause_mode], clause_out)
                clause.submit(check_clause, [clause, clause_mode], clause_out)
            with gr.Tab("Check a document"):
                document = gr.Textbox(lines=14, max_lines=30, label="Terms of Service",
                                      placeholder="Paste the whole text; every sentence is checked")
                with gr.Row():
                    document_mode = gr.Radio(choices, value=analyzer.modes[0], label="System", scale=3)
                    document_button = gr.Button("Check the document", variant="primary", scale=1)
                gr.Examples([[EXAMPLE_DOCUMENT]], inputs=[document],
                            label="Example: the Terms of an invented company")
                document_out = gr.HTML()
                document_csv = gr.File(label="Results as a table (CSV)", interactive=False)
                document_button.click(check_document, [document, document_mode], [document_out, document_csv])
            with gr.Tab("About"):
                gr.Markdown(model_card(project_root, analyzer))
    return app


def launch(app, **kwargs):
    """Gradio 6 takes the theme and the CSS at launch."""
    import gradio as gr
    return app.launch(theme=gr.themes.Soft(), css=CSS, **kwargs)

