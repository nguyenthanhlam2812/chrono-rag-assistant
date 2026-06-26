from __future__ import annotations

from typing import Iterable, List

from src.utils.text import normalize_vn


HINT_MARKER = "retrieval hints:"


TOPIC_HINTS = {
    "rag": (
        "retrieval augmented generation",
        "knowledge intensive NLP",
        "question answering",
        "retrieval corpus documents",
        "survey recent trends",
        "applications",
    ),
    "ai_agent": (
        "LLM agents",
        "autonomous agents",
        "reasoning acting planning",
        "tool use",
        "multi agent framework",
        "survey recent trends",
        "applications",
    ),
    "knowledge_distillation": (
        "knowledge distillation",
        "teacher student model",
        "model compression",
        "DistilBERT TinyBERT MiniLM MobileBERT",
        "survey recent trends",
        "applications",
    ),
}

TOPIC_SIGNALS = {
    "rag": (
        "rag",
        "retrieval augmented",
        "retrieval-augmented",
        "self rag",
        "self-rag",
        "realm",
        "dpr",
        "retro",
        "atlas",
    ),
    "ai_agent": (
        "ai agent",
        "agent",
        "agents",
        "autogpt",
        "autogen",
        "react",
        "toolformer",
        "langgraph",
        "multi agent",
    ),
    "knowledge_distillation": (
        "knowledge distillation",
        "distillation",
        "distilbert",
        "tinybert",
        "minilm",
        "mobilebert",
        "teacher student",
    ),
}

TREND_SIGNALS = (
    "xu huong",
    "hien nay",
    "gan day",
    "moi nhat",
    "phat trien",
    "tuong lai",
    "ung dung",
    "ap dung",
    "trend",
    "trends",
    "current",
    "recent",
    "latest",
    "future",
    "application",
    "applications",
)

TEMPORAL_SIGNALS = (
    "khi nao",
    "nam nao",
    "ra doi",
    "xuat hien",
    "introduced",
    "proposed",
    "released",
    "when",
    "year",
)

COMPARISON_SIGNALS = (
    "so sanh",
    "khac gi",
    "khac nhau",
    "difference",
    "compare",
    "versus",
    " vs ",
)


def expand_query_for_retrieval(query: str, topic: str | None = None) -> str:
    """Append lightweight retrieval hints for broad/VI questions.

    The user's original question is preserved verbatim. Extra hints are only
    for lexical retrieval, not for display. This is intentionally modest:
    it helps BM25/vector search with Vietnamese phrasing such as
    "xu huong hien nay cua AI agent" without routing every casual chat
    message into the corpus.
    """

    raw = (query or "").strip()
    if not raw:
        return query
    if HINT_MARKER in raw.lower():
        return raw

    norm = normalize_vn(raw)
    topic_key = _normalise_topic(topic) or _topic_from_text(norm)
    if not topic_key:
        return raw

    has_topic_signal = _contains_any(norm, TOPIC_SIGNALS[topic_key])
    has_retrieval_intent = (
        has_topic_signal
        or _contains_any(norm, TREND_SIGNALS)
        or _contains_any(norm, TEMPORAL_SIGNALS)
        or _contains_any(norm, COMPARISON_SIGNALS)
    )
    if not has_retrieval_intent:
        return raw

    hints: list[str] = []
    if has_topic_signal or _contains_any(norm, TREND_SIGNALS):
        hints.extend(TOPIC_HINTS[topic_key])
    if _contains_any(norm, TEMPORAL_SIGNALS):
        hints.extend(("introduced proposed released year timeline milestone",))
    if _contains_any(norm, COMPARISON_SIGNALS):
        hints.extend(("compare difference benchmark results outperform",))

    if not hints:
        return raw
    return f"{raw}\n{HINT_MARKER} {' '.join(_dedupe_words(hints))}"


def generate_retrieval_queries(query: str, topic: str | None = None) -> List[str]:
    """Generate a small set of lexical retrieval queries.

    This is a deterministic, local version of query rewrite/multi-query:
    - query 1 is always the user's original wording;
    - query 2 is the expanded retrieval-hint version;
    - query 3 is a compact topic/intent query for broad trend/comparison asks.

    The caller fuses rankings across these queries, so a loose Vietnamese
    question can still benefit from English paper vocabulary without changing
    what the LLM/user sees.
    """

    raw = (query or "").strip()
    if not raw:
        return []

    norm = normalize_vn(raw)
    explicit_topic = _topic_from_text(norm)
    topic_key = _normalise_topic(topic) or explicit_topic
    queries = [raw]

    expanded = expand_query_for_retrieval(raw, topic_key)
    if expanded != raw:
        queries.append(expanded)

    if topic_key and (expanded != raw or explicit_topic):
        compact = _compact_topic_query(norm, topic_key)
        if compact:
            queries.append(compact)

    deduped: list[str] = []
    seen = set()
    for item in queries:
        key = " ".join(item.lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:3]


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _topic_from_text(text: str) -> str:
    for topic_key, signals in TOPIC_SIGNALS.items():
        if _contains_any(text, signals):
            return topic_key
    return ""


def _compact_topic_query(norm: str, topic_key: str) -> str:
    hints = list(TOPIC_HINTS.get(topic_key, ()))
    if not hints:
        return ""
    parts = [" ".join(hints)]
    if _contains_any(norm, TREND_SIGNALS):
        parts.append("current recent trends survey applications")
    if _contains_any(norm, TEMPORAL_SIGNALS):
        parts.append("introduced proposed released timeline milestone year")
    if _contains_any(norm, COMPARISON_SIGNALS):
        parts.append("compare difference benchmark results outperform")
    return " ".join(parts)


def _normalise_topic(topic: str | None) -> str:
    value = normalize_vn(topic or "").replace("-", "_").replace(" ", "_")
    if not value:
        return ""
    if "distill" in value or value in {"kd", "knowledge_distillation"}:
        return "knowledge_distillation"
    if "agent" in value:
        return "ai_agent"
    if "rag" in value:
        return "rag"
    return ""


def _dedupe_words(phrases: Iterable[str]) -> list[str]:
    words: list[str] = []
    seen = set()
    for phrase in phrases:
        for word in phrase.split():
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(word)
    return words
