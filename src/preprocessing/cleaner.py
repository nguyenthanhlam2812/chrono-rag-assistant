"""
Source-type-aware text cleaner for ChronoRAG.

Centralises cleaning so downstream chunking, labeling, retrieval, and
event detection do not receive obvious markup/script/code noise, while
preserving timeline signals (years, dates, versions, citations).

Design rules
------------
* ``clean_text(text, source_type)`` is the single public entry point.
* Fenced code blocks are **never** removed globally.
  - ``paper`` / ``survey``: light LaTeX/PDF artifact normalisation.
  - ``docs`` / ``github`` / ``blog`` / ``wiki`` / everything else:
    preserve code blocks, version strings, dates, headings, release notes.
* HTML noise (``<script>``, ``<style>``, ``<svg>``, ``<nav>``, etc.) is
  stripped when found in raw text regardless of source type.
* Markdown badge lines (``[![...](https://img.shields.io/...)]...``) are removed.
* HTML comments ``<!-- ... -->`` are removed.
* Whitespace / blank lines / control characters are normalised.
* Year, date, version, and citation strings are **never** destroyed.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str, source_type: Optional[str] = None) -> str:
    """Clean *text* with rules appropriate for *source_type*.

    Parameters
    ----------
    text : str
        Raw document text (already extracted by the parser layer).
    source_type : str or None
        One of ``paper``, ``survey``, ``blog``, ``docs``, ``github``,
        ``wiki``, or ``None``.  ``None`` falls through to the generic
        (docs-like) pipeline.

    Returns
    -------
    str
        Cleaned text with timeline signals preserved.
    """
    if not text:
        return ""

    st = (source_type or "").strip().lower()

    if st in ("paper", "survey"):
        text = _clean_paper(text)
    else:
        # docs / github / blog / wiki / unknown
        text = _clean_general(text)

    # Common final normalisation (all source types)
    text = _normalise_whitespace(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Source-type pipelines
# ---------------------------------------------------------------------------

def _clean_paper(text: str) -> str:
    """Light cleaning for academic papers / surveys.

    Focuses on LaTeX / PDF extraction artefacts while keeping dates,
    version strings, and citation markers intact.
    """
    text = _strip_html_noise(text)
    text = _strip_html_comments(text)
    text = _strip_markdown_badges(text)
    text = _normalise_latex_artefacts(text)
    text = _strip_control_chars(text)
    return text


def _clean_general(text: str) -> str:
    """Cleaning for docs / github / blog / wiki and any other source type.

    Preserves fenced code blocks, headings, version strings, dates.
    """
    text = _strip_html_noise(text)
    text = _strip_html_comments(text)
    text = _strip_markdown_badges(text)
    text = _strip_control_chars(text)
    return text


# ---------------------------------------------------------------------------
# Individual cleaning helpers
# ---------------------------------------------------------------------------

# ---- HTML noise -----------------------------------------------------------

# Regex for inline HTML tags that carry no useful prose:
#   <script>...</script>, <style>...</style>, <svg>...</svg>,
#   <nav>...</nav>, <footer>...</footer>, <header>...</header>
_RE_HTML_BLOCKS = re.compile(
    r"<\s*(script|style|svg|nav|footer|header)\b[^>]*>.*?</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Stray HTML tags (e.g. <div>, <span>, </p>, <br/>, <img .../>)
_RE_HTML_TAGS = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^>]*?/?>")

# HTML entities  (&amp;  &nbsp;  &#160;  &ensp; etc.)
_RE_HTML_ENTITIES = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")


def _strip_html_noise(text: str) -> str:
    """Remove script/style/svg/nav/footer/header blocks, stray tags, entities."""
    text = _replace_html_entities(text)
    text = _RE_HTML_BLOCKS.sub("", text)
    text = _RE_HTML_TAGS.sub("", text)
    return text


def _replace_html_entities(text: str) -> str:
    """Replace common HTML entities with their plain-text equivalents."""
    # Named entities we care about
    _MAP = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&nbsp;": " ",
        "&ensp;": " ",
        "&emsp;": " ",
        "&ndash;": "-",
        "&mdash;": "--",
    }
    for entity, replacement in _MAP.items():
        text = text.replace(entity, replacement)
    # Remaining numeric / unknown entities -> empty
    text = _RE_HTML_ENTITIES.sub("", text)
    return text


# ---- HTML comments --------------------------------------------------------

_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_html_comments(text: str) -> str:
    return _RE_HTML_COMMENT.sub("", text)


# ---- Markdown badge lines -------------------------------------------------

# Lines like: [![Badge Text](https://img.shields.io/...)](...)
_RE_MD_BADGE_LINE = re.compile(
    r"^\[?!\[[^\]]*\]\([^)]*img\.shields\.io[^)]*\)[^\n]*$",
    re.MULTILINE,
)

# Markdown image references that point to shields.io (inline, not full-line)
_RE_MD_BADGE_INLINE = re.compile(
    r"\[?!\[[^\]]*\]\([^)]*img\.shields\.io[^)]*\)\]?(?:\([^)]*\))?",
)


def _strip_markdown_badges(text: str) -> str:
    text = _RE_MD_BADGE_LINE.sub("", text)
    text = _RE_MD_BADGE_INLINE.sub("", text)
    return text


# ---- LaTeX / PDF artefacts ------------------------------------------------

# Common LaTeX inline commands that survive PDF->text: \textbf{}, \textit{}, etc.
_RE_LATEX_INLINE = re.compile(r"\\(?:textbf|textit|emph|texttt)\{([^}]*)\}")
# \cite{...}, \ref{...}, \label{...} -- keep the content inside braces
_RE_LATEX_REF = re.compile(r"\\(?:cite|ref|label|eqref)\{([^}]*)\}")
# Stray single backslash commands (e.g. \noindent, \vspace{...})
_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+")
# Ligature artefacts: ff/fi/fl/ffi/ffl compatibility characters
_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def _normalise_latex_artefacts(text: str) -> str:
    # Unwrap simple formatting commands
    text = _RE_LATEX_INLINE.sub(r"\1", text)
    text = _RE_LATEX_REF.sub(r"\1", text)
    # Remove remaining stray commands
    text = _RE_LATEX_CMD.sub("", text)
    # Fix ligatures
    for lig, repl in _LIGATURE_MAP.items():
        text = text.replace(lig, repl)
    return text


# ---- Control characters ---------------------------------------------------

# Control chars except \n, \r, \t
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_control_chars(text: str) -> str:
    return _RE_CTRL.sub("", text)


# ---- Whitespace normalisation ---------------------------------------------

def _normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and limit consecutive blank lines to 2."""
    # Collapse horizontal whitespace (preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Strip trailing spaces on each line
    text = re.sub(r" +\n", "\n", text)
    # Collapse 3+ consecutive newlines -> 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
