"""Shared text utilities for retrieval and answer generation.

Centralises the small helpers that were previously copy-pasted between
``src/retrieval/simple_retriever.py`` and ``src/generation/template_answerer.py``.
Kept dependency-free (regex only) so it can be imported from any layer.
"""

from __future__ import annotations

import re
from typing import FrozenSet

# Stopwords used to filter query tokens before retrieval / sentence matching.
STOPWORDS: FrozenSet[str] = frozenset(
    {
        "is", "what", "the", "a", "an", "and", "or", "in", "on", "at",
        "for", "with", "about", "to", "of", "how", "why", "where", "when",
        "who", "whom", "whose", "which", "are", "do", "does", "did", "can",
        "could", "should", "would", "you", "your", "my", "our", "their",
    }
)

# Signals that a candidate "sentence" is actually code, markup, or a URL,
# i.e. something we never want to surface as a natural-language answer.
_BAD_SENTENCE_SIGNALS = ("shields.io", "github.com", "http", "www.", ".py", ".html", "img.")

_RE_MARKDOWN_BOLD_ITALIC = re.compile(r"\*+")
_RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_RE_FIRST_ALPHA = re.compile(r"[a-zA-Z]")


def is_good_sentence(s: str) -> bool:
    """Return True if ``s`` looks like a clean, capitalised English sentence.

    Filters out fragments shorter than 15 chars, lines that start with a
    lowercase letter (usually code or wrapped continuations), and obvious
    URL / markup signatures.
    """
    s_clean = s.strip()
    if len(s_clean) < 15:
        return False
    first_alpha = _RE_FIRST_ALPHA.search(s_clean)
    if first_alpha and first_alpha.group(0).islower():
        return False
    lowered = s_clean.lower()
    if any(bad in lowered for bad in _BAD_SENTENCE_SIGNALS):
        return False
    return True


def clean_for_matching(text: str) -> str:
    """Lowercase and strip Markdown noise so substring matches behave naturally."""
    t = _RE_MARKDOWN_BOLD_ITALIC.sub("", text)
    t = _RE_MARKDOWN_LINK.sub(r"\1", t)
    return t.lower()
