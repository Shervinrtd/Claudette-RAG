"""The complete system for one clause or a whole document, as used by the demo (scripts/demo.py).

For each sentence it runs exactly the system evaluated on the test set, "RAG + evidence gate":
1. retrieval (retrieval.retrieve): the 8 most similar expert-labelled training sentences
   ("precedents"), 5 legal rationales and the legal provisions they rely on;
2. precedent votes: for each category, how many of the k nearest precedents the experts labelled with it;
3. the LLM reads the sentence and the retrieved knowledge and answers with an explanation, the ids of
   the items it relies on and the categories (rag.rag_messages, llm.call_model, llm.parse_answer);
4. the cited ids are checked against what was retrieved (rag.check_citations);
5. evidence gate (rag.gate_by_neighbours): a category is kept only if at least m of the k precedents have it.

Shortcut for documents: when no category has m precedent votes, the gate removes whatever the LLM
answers, so the LLM is not asked. No verdict changes: on the test set and on the four languages of the
multilingual set, none of the sentences skipped this way was flagged by the evaluated system, and about
86% of the calls are saved.

Evidence-only mode (no LLM, no cost): the kNN baseline, a category is flagged when at least t of the
k most similar precedents have it (k=3, t=2 chosen on validation, results/retrieval_bge-m3_validation.json).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .knowledge_base import load_knowledge_base
from .llm import call_model, estimate_cost, file_safe, is_local, parse_answer, response_format
from .rag import build_rag_system_prompt, check_citations, gate_by_neighbours, rag_messages
from .retrieval import KBIndex, RetrievalConfig, SentenceTransformerEmbedder, knn_vote, retrieve

RAG_GATE, EVIDENCE_ONLY = "rag_gate", "evidence_only"
DEFAULT_GATE = {"k": 8, "m": 3}     # chosen on validation for gpt-4.1-mini and for Qwen3.6-27B
DEFAULT_KNN = {"k": 3, "t": 2}      # chosen on validation for the kNN baseline
MIN_WORDS = 3                       # shorter pieces (headings, numbers) are shown but not analysed
MAX_SENTENCES = 1000


# ----------------------------------------------------------------------------- sentence splitting
_ABBREVIATIONS = {
    # English
    "e.g", "i.e", "etc", "inc", "ltd", "co", "corp", "llc", "no", "nos", "art", "arts", "sec", "para",
    "vs", "v", "mr", "mrs", "ms", "dr", "st", "jr", "approx", "incl", "fig", "p", "pp", "cf", "ca",
    # German, Italian, Polish
    "z.b", "d.h", "bzw", "ggf", "vgl", "abs", "nr", "ziff", "s", "ecc", "es", "pag", "cfr", "sig",
    "n", "np", "tj", "tzn", "ust", "pkt", "zł", "ul", "itd", "itp", "m.in", "dz.u",
}
_BOUNDARY = re.compile(r"[.!?]+[\"'”’»)\]]*\s+")
_SENTENCE_END = re.compile(r"[.!?:;][\"'”’»)\]]*\s*$")


def _is_abbreviation(before: str) -> bool:
    """True if the text before a period ends with an abbreviation or an initial (e.g., Art., U.S.)."""
    match = re.search(r"(\S+)$", before)
    if not match:
        return False
    token = match.group(1).lstrip("(\"'“‘«").lower()
    return token in _ABBREVIATIONS or len(token) == 1 or "." in token


def split_sentences(paragraph: str) -> list[str]:
    """One paragraph -> sentences: a sentence ends at . ! or ? followed by a capital letter, a digit,
    an opening quote or a bracket, unless the period ends an abbreviation (Art. 5, e.g. Spotify, U.S. law)."""
    sentences, start = [], 0
    for match in _BOUNDARY.finditer(paragraph):
        nxt = paragraph[match.end():match.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘«(["):
            continue
        if match.group(0).startswith(".") and not match.group(0).startswith("..") \
                and _is_abbreviation(paragraph[start:match.start()]):
            continue
        sentences.append(paragraph[start:match.end()].strip())
        start = match.end()
    if paragraph[start:].strip():
        sentences.append(paragraph[start:].strip())
    return sentences


def split_document(text: str) -> list[list[str]]:
    """Document -> paragraphs -> sentences.

    Every non-empty line is a paragraph, except that a line starting with a lowercase letter continues
    the previous line when that one does not end a sentence (text copied from a PDF, wrapped mid-sentence).
    """
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    paragraphs: list[str] = []
    previous_blank = True
    for line in lines:
        if not line:
            previous_blank = True
            continue
        if paragraphs and not previous_blank and line[0].islower() and not _SENTENCE_END.search(paragraphs[-1]):
            paragraphs[-1] += " " + line
        else:
            paragraphs.append(line)
        previous_blank = False
    return [split_sentences(p) for p in paragraphs]


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


_NUMBERING = re.compile(r"^(§\s*)?\d+(\.\d+)*[.)]?\s+")


def looks_like_heading(paragraph: list[str]) -> bool:
    """A one-line paragraph without final punctuation and at most 5 words (after a section number such as
    "7." or "§ 3"), e.g. "1. Acceptance of the Terms": shown, not analysed. Kept strict on purpose: skipping
    a real clause would be worse than analysing a heading."""
    if len(paragraph) != 1 or _SENTENCE_END.search(paragraph[0]):
        return False
    return word_count(_NUMBERING.sub("", paragraph[0])) <= 5


# ----------------------------------------------------------------------------- results
@dataclass
class Analysis:
    """Everything the system did for one sentence (the demo shows all of it)."""

    text: str
    mode: str
    context: dict = field(repr=False)          # retrieved knowledge, format of retrieval.retrieve
    votes: dict[str, int]                      # category code -> precedents (of the k nearest) with it
    gate: dict                                 # {"k", "m"} in rag_gate mode, {"k", "t"} in evidence_only mode
    asked_model: bool = False
    model_categories: list[str] = field(default_factory=list)   # categories proposed by the LLM
    explanation: str | None = None
    cited: list[str] = field(default_factory=list)              # cited ids that were retrieved
    invalid_cited: list[str] = field(default_factory=list)      # cited ids that were not retrieved
    consistent: bool = True                    # "unfair" and "categories" of the answer agree
    categories: list[str] = field(default_factory=list)         # final verdict (empty = not flagged)
    removed_by_gate: list[str] = field(default_factory=list)
    error: str | None = None                   # the model could not be reached: no verdict
    usage: dict | None = None                  # latency, tokens and estimated cost of the model call

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        return "flagged" if self.categories else "not_flagged"

    @property
    def precedents(self) -> list[tuple[dict, float]]:
        return self.context.get("examples", [])


@dataclass
class Segment:
    """A piece of a document: a sentence that was analysed, or a short piece (heading) that was not."""

    text: str
    paragraph: int
    analysis: Analysis | None = None


# ----------------------------------------------------------------------------- the system
def load_gate(results_dir: Path, model: str | None, config: RetrievalConfig) -> tuple[dict, str]:
    """Gate setting chosen on validation for this model (scripts/gate_rag.py), or the default."""
    if model:
        path = Path(results_dir) / (f"gate_choice_{file_safe(model)}_e{config.k_examples}"
                                    f"r{config.k_rationales}p{config.max_provisions}.json")
        if path.exists():
            choice = json.loads(path.read_text(encoding="utf-8"))
            return {"k": choice["k"], "m": choice["m"]}, f"chosen on validation for {model}"
    return dict(DEFAULT_GATE), ("not tuned for this model: k=8, m=3 is the setting chosen on validation "
                                "for gpt-4.1-mini and for Qwen3.6-27B")


def load_knn(results_dir: Path) -> dict:
    path = Path(results_dir) / "retrieval_bge-m3_validation.json"
    if path.exists():
        choice = json.loads(path.read_text(encoding="utf-8"))["knn_choice"]
        return {"k": choice["k"], "t": choice["t"]}
    return dict(DEFAULT_KNN)


class ClauseAnalyzer:
    """RAG + evidence gate (with an LLM) or evidence only (kNN, no LLM) for clauses and documents."""

    def __init__(self, project_root: Path, model: str | None = None, client=None, extra_body: dict | None = None,
                 index_name: str = "bge-m3", embedder=None, retrieval: RetrievalConfig | None = None,
                 workers: int = 4, embed_device: str | None = None):
        project_root = Path(project_root)
        kb = load_knowledge_base(project_root / "knowledge_base")
        self.categories = sorted(kb["categories"], key=lambda c: c["label_index"])
        self.codes = [c["id"] for c in self.categories]
        self.category = {c["id"]: c for c in self.categories}
        self.code_to_label = {c["id"]: c["label_index"] for c in self.categories}
        self.label_to_code = {i: code for code, i in self.code_to_label.items()}
        self.sources = {s["id"]: s for s in kb["sources"]}
        self.provisions = {p["id"]: p for p in kb["legal_provisions"]}

        index_dir = project_root / "data" / "processed" / "index" / index_name
        if not (index_dir / "meta.json").exists():
            raise SystemExit(f"No index in {index_dir}: run scripts/build_knowledge_base.py and "
                             "scripts/build_index.py first.")
        self.index = KBIndex.load(index_dir)
        self.embedder = embedder or SentenceTransformerEmbedder(self.index.model_name, device=embed_device)
        self.retrieval = retrieval or RetrievalConfig()

        self.model, self.client, self.extra_body, self.workers = model, client, extra_body, workers
        if model and client is None:
            raise ValueError("a model needs a client (llm.make_client)")
        self.local = bool(client) and is_local(client)
        results = project_root / "results"
        self.gate, self.gate_source = load_gate(results, model, self.retrieval)
        if self.gate["k"] > self.retrieval.k_examples:
            raise ValueError("the gate looks at more precedents than are retrieved")
        self.knn = load_knn(results)
        self.system_prompt = build_rag_system_prompt(self.categories)
        self.fmt = response_format(self.codes, with_citations=True)

    @property
    def modes(self) -> list[str]:
        return [RAG_GATE, EVIDENCE_ONLY] if self.model else [EVIDENCE_ONLY]

    # ---- one clause: the model is always asked, so that its reasoning is shown even when the gate overrules it
    def analyze_clause(self, text: str, mode: str = RAG_GATE) -> Analysis:
        return self.analyze_sentences([text.strip()], mode, always_ask=True)[0]

    # ---- a document: the model is asked only where the gate could keep a category
    def analyze_document(self, text: str, mode: str = RAG_GATE, progress=None) -> list[Segment]:
        segments, todo = [], []
        for p, sentences in enumerate(split_document(text)):
            heading = looks_like_heading(sentences)
            for sentence in sentences:
                segments.append(Segment(sentence, p))
                if not heading and word_count(sentence) >= MIN_WORDS:
                    todo.append(segments[-1])
        if len(todo) > MAX_SENTENCES:
            raise ValueError(f"The document has {len(todo)} sentences; the demo analyses at most {MAX_SENTENCES}.")
        for segment, analysis in zip(todo, self.analyze_sentences([s.text for s in todo], mode, progress=progress)):
            segment.analysis = analysis
        return segments

    def analyze_sentences(self, sentences: list[str], mode: str = RAG_GATE, always_ask: bool = False,
                          progress=None) -> list[Analysis]:
        if mode not in self.modes:
            raise ValueError(f"mode {mode!r} is not available (available: {self.modes})")
        if not sentences:
            return []
        if progress:
            progress(0.0, "Retrieving similar clauses and legal knowledge")
        contexts = retrieve(self.index, self.embedder.encode(sentences), self.retrieval)
        if mode == EVIDENCE_ONLY:
            return [self._evidence_only(s, c) for s, c in zip(sentences, contexts)]

        analyses = [self._with_votes(s, c, RAG_GATE) for s, c in zip(sentences, contexts)]
        to_ask = [a for a in analyses if always_ask or max(a.votes.values()) >= self.gate["m"]]
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            for n, _ in enumerate(pool.map(self._ask_model, to_ask), start=1):
                if progress:
                    progress(n / len(to_ask), f"Asked the model about {n} of {len(to_ask)} sentences")
        return analyses

    # ---- steps
    def _with_votes(self, text: str, context: dict, mode: str) -> Analysis:
        k = self.gate["k"] if mode == RAG_GATE else self.knn["k"]
        nearest = [doc["categories"] for doc, _ in context["examples"][:k]]
        votes = {code: sum(code in n for n in nearest) for code in self.codes}
        gate = dict(self.gate) if mode == RAG_GATE else dict(self.knn)
        return Analysis(text=text, mode=mode, context=context, votes=votes, gate=gate)

    def _evidence_only(self, text: str, context: dict) -> Analysis:
        analysis = self._with_votes(text, context, EVIDENCE_ONLY)
        labels = [doc["labels"] for doc, _ in context["examples"]]
        predicted = knn_vote([labels], self.knn["k"], self.knn["t"], len(self.codes))[0]
        analysis.categories = [self.label_to_code[i] for i in predicted]
        return analysis

    def _ask_model(self, analysis: Analysis) -> None:
        messages = rag_messages(self.system_prompt, analysis.text, analysis.context)
        try:
            row = call_model(self.client, self.model, messages, self.fmt, max_retries=3, extra_body=self.extra_body)
        except Exception as error:  # noqa: BLE001 - shown to the user instead of a verdict
            analysis.error = f"{type(error).__name__}: {error}"
            return
        parsed = parse_answer(row["content"], self.code_to_label)
        cites = check_citations(parsed["cited_ids"], analysis.context)
        analysis.asked_model = True
        analysis.model_categories = [self.label_to_code[i] for i in parsed["labels"]]
        analysis.explanation = parsed["explanation"]
        analysis.cited, analysis.invalid_cited = cites["valid"], cites["invalid"]
        analysis.consistent = parsed["consistent"]
        if parsed["parse_error"]:
            analysis.error = "the model's answer could not be read (not valid JSON)"
            return
        nearest = [doc["categories"] for doc, _ in analysis.context["examples"]]
        analysis.categories = gate_by_neighbours(analysis.model_categories, nearest, self.gate["k"], self.gate["m"])
        analysis.removed_by_gate = [c for c in analysis.model_categories if c not in analysis.categories]
        analysis.usage = {"latency_s": row.get("latency_s"), "prompt_tokens": row["prompt_tokens"],
                          "completion_tokens": row["completion_tokens"],
                          "cost_usd": 0.0 if self.local else estimate_cost([row], self.model)}

    # ---- helpers for the display
    def source_of(self, item: dict) -> dict | None:
        """Bibliography entry of a knowledge-base item (provision or rationale)."""
        source_id = self.provisions.get(item["id"], {}).get("source") or item.get("source")
        return self.sources.get(source_id)
