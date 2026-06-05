"""Shared text utilities for retrieval and answer generation.

Centralises the small helpers that were previously copy-pasted between
``src/retrieval/simple_retriever.py`` and ``src/generation/template_answerer.py``.
Kept dependency-free (regex only) so it can be imported from any layer.
"""

from __future__ import annotations

import re
import unicodedata
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

# Self-RAG / RETRO control tokens and other markers we never want to surface
# as part of a natural-language answer. These slip past ``is_good_sentence``
# because the sentence around them still starts with a capital letter.
_RE_CONTROL_TOKEN = re.compile(
    r"\[IS[A-Z]+|\[Retrieve|\[Yes\]|\[No\]|=Relevant|=Fully|=Partially|\bbpb\b|>\s*Repeat|#\$|Output hey"
)
# Three or more decimal numbers in a row signals a table row leaking into prose.
_RE_NUMBER_TABLE = re.compile(r"(?:\d+\.\d+\s+){3,}")
_OFF_TOPIC_SAMPLE_MARKERS = (
    "bohemian rhapsody",
    "alan george rad",
    "rad cliffe",
    "para-badminton",
    "paralympic",
    "st. patrick",
    "tout ce qui",
    "defendu par la loi",
    "despotisme",
    "decimals of",
    "retro[off]",
)


_HYPHEN_LINEBREAK_RE = re.compile(r"\b([A-Za-z]{2,})-\s+([A-Za-z]{2,})\b")
_HYPHEN_KEEP_PREFIXES = frozenset({
    "fine", "general", "large", "open", "pre",
    "question", "state", "task",
})


def repair_pdf_hyphenation(text: str) -> str:
    """Glue PDF line-wrap hyphenations back together.

    PDFs often break long words across lines as "compet- itive". This restores
    them to "competitive" while keeping legitimate compound hyphens
    ("fine-tuned", "open-source").
    """
    def replace(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if left.lower() in _HYPHEN_KEEP_PREFIXES:
            return f"{left}-{right}"
        return f"{left}{right}"

    return _HYPHEN_LINEBREAK_RE.sub(replace, text)


# Common Vietnamese typing shortcuts seen in chat. Mapped to canonical
# diacritic-stripped form so meta/chitchat phrase matching catches them all
# without enumerating every variant.
_VN_SHORTCUTS = {
    "dc": "duoc", "ddc": "duoc", "đc": "duoc",
    "k": "khong", "ko": "khong", "kg": "khong", "khg": "khong",
    "j": "gi", "ji": "gi",
    "m": "may", "mn": "moi nguoi",
    "e": "em", "a": "anh",
    "t": "tao",
    "bik": "biet", "biek": "biet", "bít": "biet", "bek": "biet",
    "cau": "ban", "cậu": "ban",
    "cx": "cung", "cũng": "cung",
    "ji": "gi",
    "vs": "voi", "vợi": "voi",
    "z": "vay", "zậy": "vay",
    "iu": "yeu",
    "thik": "thich",
    "đg": "dang",
}


def normalize_vn(text: str) -> str:
    """Normalise casual Vietnamese for phrase matching.

    Strips diacritics, lowercases, replaces 'đ'→'d', and expands the most
    common chat shortcuts (đc/dc/ddc → duoc, k/ko → khong, m → may, j → gi).
    Lets meta/chitchat phrase lists stay short while still catching variants
    like "m làm ddc gì" / "m bik j" / "bot giúp dc cái gì".
    """
    if not text:
        return ""
    # Lowercase + strip combining diacritics. NFD splits "ấ" into "a"+◌́, then
    # we drop the combining mark; "đ" is a separate codepoint so handled below.
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    # Use explicit codepoint escapes so matching stays stable across Windows
    # console encodings and source viewers.
    s = s.replace("\u0111", "d").replace("\u0110", "d")
    # Token-level shortcut expansion.
    tokens = re.findall(r"[a-z0-9]+|[^a-z0-9\s]", s)
    expanded = [_VN_SHORTCUTS.get(tok, tok) for tok in tokens]
    return re.sub(r"\s+", " ", " ".join(expanded)).strip()


def is_noise_fragment(s: str) -> bool:
    """Return True for sentence fragments that are tables, control tokens,
    or OCR garbage that should never appear in a chat answer."""
    s_clean = s.strip()
    if not s_clean:
        return True
    if _RE_CONTROL_TOKEN.search(s_clean):
        return True
    if _RE_NUMBER_TABLE.search(s_clean):
        return True
    lowered = s_clean.lower()
    if any(marker in lowered for marker in _OFF_TOPIC_SAMPLE_MARKERS):
        return True
    non_space = [c for c in s_clean if not c.isspace()]
    if non_space:
        letters = sum(1 for c in non_space if c.isalpha())
        if letters / len(non_space) < 0.6:
            return True
    return False


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
