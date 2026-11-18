"""Text utilities shared by data preparation, models and the demo."""

from __future__ import annotations

import re

# UNFAIR-ToS sentences are lowercased and tokenized (e.g. "people 's", "terms ."),
# so text from other sources is normalised the same way before being compared or
# classified. This is an approximation of the original tokenization.
_PUNCTUATION = re.compile(r"([.,;:!?()\[\]{}\"“”‘’«»„])")
_CLITICS = re.compile(r"(\w)('s|'re|'ve|'ll|'d|'m|n't)\b", flags=re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def normalize_like_unfair_tos(text: str) -> str:
    """Lowercase and split off punctuation, imitating the UNFAIR-ToS text format."""
    text = text.lower().replace("’", "'")
    text = _CLITICS.sub(r"\1 \2", text)
    text = _PUNCTUATION.sub(r" \1 ", text)
    return _SPACES.sub(" ", text).strip()


def squash(text: str) -> str:
    """Keep only lowercase letters and digits: used to find near-duplicate sentences
    regardless of tokenization differences (e.g. "won't" vs "wo n't")."""
    return re.sub(r"[^a-z0-9]", "", text.lower())
