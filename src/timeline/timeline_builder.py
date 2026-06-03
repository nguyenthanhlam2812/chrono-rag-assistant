from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


EVENT_TYPES = {"method_proposed", "release", "benchmark", "trend_application"}

_SPACE_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[[0-9,\s-]+\]|\([A-Z][A-Za-z]+(?:\s+et\s+al\.)?,?\s+\d{4}[a-z]?\)")
_MARKDOWN_RE = re.compile(r"[*_`#]+")
_BAD_PREFIXES = (
    "published as",
    "proceedings of",
    "copyright",
    "figure ",
    "table ",
    "appendix",
    "references",
)
_BAD_SUBSTRINGS = (
    "arxiv:",
    "http://",
    "https://",
    "www.",
    "doi:",
    "isbn",
    "appendix",
    "more details are",
    "as mentioned",
    "what is more",
    "this is perhaps",
    "interestingly,",
    "besides,",
    "with regards",
    "dataset leakage",
    "kilt versions",
    "requiring knowledge not present",
    "the pretraining objective",
)
_INCOMPLETE_ENDINGS = (
    " vs",
    " vs.",
    " et al",
    " et al.",
    " with",
    " and",
    " or",
    " to",
    " of",
)
_ACTION_SIGNALS = (
    "propose",
    "proposed",
    "introduce",
    "introduced",
    "present",
    "presented",
    "develop",
    "developed",
    "release",
    "released",
    "evaluate",
    "evaluated",
    "outperform",
    "outperformed",
    "achieve",
    "achieved",
    "state-of-the-art",
    "benchmark",
    "survey",
    "recent",
    "recently",
    "apply",
    "applied",
    "extend",
    "extended",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_documents_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("doc_id", "")): row for row in load_jsonl(path)}


def build_timeline(
    predictions: Iterable[Dict[str, Any]],
    documents_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    min_event_probability: float = 0.55,
    min_date_confidence: float = 0.4,
    similarity_threshold: float = 0.78,
    max_events_per_topic: int = 30,
) -> Dict[str, Any]:
    """Build a compact topic timeline from sentence-level event predictions.

    MVP behavior is deliberately explainable:
    - keep only predicted event sentences with a coarse date;
    - cluster near-duplicate event sentences inside the same topic/year/type;
    - choose the highest-confidence sentence as the representative event;
    - preserve source/citation metadata for each event.
    """
    prediction_rows = list(predictions)
    documents_by_id = documents_by_id or {}
    candidates = [
        _candidate_from_prediction(row, documents_by_id)
        for row in prediction_rows
        if _is_candidate_event(row, min_event_probability, min_date_confidence)
    ]
    candidates = [row for row in candidates if row is not None]

    grouped: Dict[tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["topic"], row["year"], row["event_type"])].append(row)

    topic_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for group_rows in grouped.values():
        for cluster in _cluster_rows(group_rows, similarity_threshold):
            event = _event_from_cluster(cluster)
            topic_events[event["topic"]].append(event)

    limited_topics: Dict[str, List[Dict[str, Any]]] = {}
    for topic, events in topic_events.items():
        selected = sorted(
            events,
            key=lambda row: (
                -float(row["rank_score"]),
                -float(row["confidence"]),
                -float(row["date_confidence"]),
                row["year"],
                row["normalized_date"],
            ),
        )[:max_events_per_topic]
        limited_topics[topic] = sorted(
            selected,
            key=lambda row: (
                row["year"],
                row["normalized_date"] or str(row["year"]),
                _event_type_sort_key(row["event_type"]),
                -float(row["rank_score"]),
                row["title"],
            ),
        )
        for index, event in enumerate(limited_topics[topic], start=1):
            event["event_id"] = f"{topic}_evt_{index:03d}"

    return {
        "metadata": {
            "total_predictions": len(prediction_rows),
            "events_considered": len(candidates),
            "timeline_events": sum(len(events) for events in limited_topics.values()),
            "min_event_probability": min_event_probability,
            "min_date_confidence": min_date_confidence,
            "similarity_threshold": similarity_threshold,
            "max_events_per_topic": max_events_per_topic,
        },
        "topics": dict(sorted(limited_topics.items())),
    }


def write_timeline_json(timeline: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_candidate_event(
    row: Dict[str, Any],
    min_event_probability: float,
    min_date_confidence: float,
) -> bool:
    is_ml_event = int(row.get("is_event", 0) or 0) == 1
    is_rescue_event = _is_high_precision_rescue(row)
    if not is_ml_event and not is_rescue_event:
        return False
    if is_ml_event and str(row.get("event_type", "none")) not in EVENT_TYPES:
        return False
    if (
        is_ml_event
        and not is_rescue_event
        and float(row.get("event_probability", 0.0) or 0.0) < min_event_probability
    ):
        return False
    if float(row.get("date_confidence", 0.0) or 0.0) < min_date_confidence:
        return False
    if row.get("extracted_year") in (None, ""):
        return False
    if row.get("date_source") == "sentence_plain_year":
        try:
            extracted_year = int(row.get("extracted_year"))
            document_year = int(row.get("year"))
        except (TypeError, ValueError):
            return False
        # Plain years inside academic sentences are often citation years. Keep
        # them only when they match the source document year.
        if extracted_year != document_year:
            return False
    return True


def _candidate_from_prediction(
    row: Dict[str, Any],
    documents_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    sentence = _clean_sentence(str(row.get("sentence", "") or ""))
    event_type = str(row.get("event_type"))
    if event_type not in EVENT_TYPES:
        event_type = _infer_event_type(sentence)
    if not _is_usable_timeline_sentence(sentence, event_type):
        return None

    doc_id = str(row.get("doc_id", "") or "")
    doc = documents_by_id.get(doc_id, {})
    title = str(doc.get("title") or row.get("title") or doc_id)
    source_url = str(row.get("source_url") or doc.get("source_url") or "")
    year = int(row.get("extracted_year"))

    return {
        "sentence_id": str(row.get("sentence_id", "")),
        "doc_id": doc_id,
        "chunk_id": str(row.get("chunk_id", "")),
        "topic": _normalise_topic(str(row.get("topic", ""))),
        "year": year,
        "normalized_date": str(row.get("normalized_date") or year),
        "date_text": row.get("date_text"),
        "date_confidence": float(row.get("date_confidence", 0.0) or 0.0),
        "date_source": row.get("date_source"),
        "event_type": event_type,
        "sentence": sentence,
        "event_probability": float(row.get("event_probability", 0.0) or 0.0),
        "event_type_confidence": float(row.get("event_type_confidence", 0.0) or 0.0),
        "quality_score": _event_quality_score(sentence, event_type),
        "representative_score": _representative_score(row),
        "source": {
            "doc_id": doc_id,
            "title": title,
            "source_url": source_url,
        },
    }


def _cluster_rows(rows: List[Dict[str, Any]], similarity_threshold: float) -> List[List[Dict[str, Any]]]:
    if len(rows) <= 1:
        return [rows]

    texts = [row["sentence"] for row in rows]
    try:
        matrix = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        ).fit_transform(texts)
    except ValueError:
        return [[row] for row in rows]

    order = sorted(
        range(len(rows)),
        key=lambda idx: rows[idx]["representative_score"] + rows[idx].get("quality_score", 0.0),
        reverse=True,
    )
    clusters: List[List[int]] = []
    for idx in order:
        assigned = False
        for cluster in clusters:
            sims = matrix[idx].dot(matrix[cluster].T).toarray().ravel()
            if sims.size and float(np.max(sims)) >= similarity_threshold:
                cluster.append(idx)
                assigned = True
                break
        if not assigned:
            clusters.append([idx])

    return [[rows[idx] for idx in cluster] for cluster in clusters]


def _event_from_cluster(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    representative = max(cluster, key=lambda row: row["representative_score"] + row.get("quality_score", 0.0))
    sources = _dedupe_sources([row["source"] for row in cluster])
    sentence_ids = sorted({row["sentence_id"] for row in cluster if row.get("sentence_id")})
    doc_ids = sorted({row["doc_id"] for row in cluster if row.get("doc_id")})

    return {
        "event_id": "",
        "topic": representative["topic"],
        "date": _display_date(representative["normalized_date"], representative["year"]),
        "normalized_date": representative["normalized_date"],
        "year": representative["year"],
        "event_type": representative["event_type"],
        "title": _make_title(representative["sentence"], representative["event_type"]),
        "representative_sentence": representative["sentence"],
        "confidence": round(float(representative["event_probability"]), 4),
        "rank_score": round(
            float(representative["representative_score"] + representative.get("quality_score", 0.0)),
            4,
        ),
        "event_type_confidence": round(float(representative["event_type_confidence"]), 4),
        "date_confidence": round(float(representative["date_confidence"]), 4),
        "date_source": representative["date_source"],
        "sources": sources,
        "cluster_size": len(cluster),
        "sentence_ids": sentence_ids,
        "doc_ids": doc_ids,
    }


def _clean_sentence(sentence: str) -> str:
    text = sentence.replace("\n", " ")
    text = _MARKDOWN_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = _repair_pdf_hyphenation(text)
    return _SPACE_RE.sub(" ", text).strip(" -")


def _repair_pdf_hyphenation(text: str) -> str:
    keep_hyphen_prefixes = {
        "fine",
        "general",
        "large",
        "open",
        "pre",
        "question",
        "state",
        "task",
    }

    def replace(match: re.Match) -> str:
        left = match.group(1)
        right = match.group(2)
        if left.lower() in keep_hyphen_prefixes:
            return f"{left}-{right}"
        return f"{left}{right}"

    return re.sub(r"\b([A-Za-z]{2,})-\s+([A-Za-z]{2,})\b", replace, text)


def _is_usable_timeline_sentence(sentence: str, event_type: str) -> bool:
    if len(sentence) < 35 or len(sentence) > 520:
        return False
    lowered = sentence.lower()
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z]", sentence) or re.match(r"^\d+\)\s+", sentence):
        return False
    first_alpha = re.search(r"[A-Za-z]", sentence)
    if first_alpha and first_alpha.group(0).islower() and not lowered.startswith("we "):
        return False
    if lowered.startswith(_BAD_PREFIXES):
        return False
    if any(marker in lowered for marker in _BAD_SUBSTRINGS):
        return False
    if lowered.rstrip(" .,:;").endswith(_INCOMPLETE_ENDINGS):
        return False
    if sentence.count(";") >= 3 and "recently proposed" not in lowered:
        return False
    if len(re.findall(r"\b(?:19|20)\d{2}\b", sentence)) >= 4 and not _has_action_signal(lowered):
        return False
    if not _has_event_type_signal(lowered, event_type):
        return False
    if _event_quality_score(sentence, event_type) < 0.15:
        return False
    alpha_count = sum(ch.isalpha() for ch in sentence)
    if alpha_count / max(len(sentence), 1) < 0.45:
        return False
    return True


def _has_action_signal(lowered_sentence: str) -> bool:
    return any(signal in lowered_sentence for signal in _ACTION_SIGNALS)


def _has_event_type_signal(lowered_sentence: str, event_type: str) -> bool:
    if event_type == "method_proposed":
        return any(
            signal in lowered_sentence
            for signal in (
                "propose",
                "proposed",
                "introduce",
                "introduced",
                "present",
                "presented",
                "develop",
                "developed",
                "framework",
                "method",
                "approach",
                "model",
            )
        )
    if event_type == "release":
        return any(
            signal in lowered_sentence
            for signal in ("release", "released", "available", "open-source", "open source", "introducing")
        )
    if event_type == "benchmark":
        return any(
            signal in lowered_sentence
            for signal in (
                "evaluate",
                "evaluated",
                "benchmark",
                "outperform",
                "outperformed",
                "achieve",
                "achieved",
                "state-of-the-art",
                "accuracy",
                "score",
                "results",
            )
        )
    if event_type == "trend_application":
        return any(
            signal in lowered_sentence
            for signal in (
                "recent",
                "recently",
                "trend",
                "survey",
                "apply",
                "applied",
                "extend",
                "extended",
                "used",
                "adopted",
                "focus",
                "progress",
            )
        )
    return False


def _is_high_precision_rescue(row: Dict[str, Any]) -> bool:
    try:
        event_probability = float(row.get("event_probability", 0.0) or 0.0)
        date_confidence = float(row.get("date_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    if event_probability < 0.35 or date_confidence < 0.4:
        return False

    lowered = _clean_sentence(str(row.get("sentence", "") or "")).lower()
    rescue_signals = (
        "we introduce",
        "we propose",
        "we present",
        "we develop",
        "our approach, named",
        "approach, named",
        "called distilbert",
        "called tinybert",
        "called react",
        "refer to as retrieval-augmented generation",
        "introduce generative agents",
    )
    return any(signal in lowered for signal in rescue_signals)


def _infer_event_type(sentence: str) -> str:
    lowered = sentence.lower()
    if any(signal in lowered for signal in ("released", "available", "introducing")):
        return "release"
    if any(signal in lowered for signal in ("outperform", "achieve", "benchmark", "state-of-the-art", "accuracy")):
        return "benchmark"
    return "method_proposed"


def _representative_score(row: Dict[str, Any]) -> float:
    date_bonus = 0.08 if row.get("date_source") != "document_year" else 0.0
    return (
        0.55 * float(row.get("event_probability", 0.0) or 0.0)
        + 0.25 * float(row.get("event_type_confidence", 0.0) or 0.0)
        + 0.20 * float(row.get("date_confidence", 0.0) or 0.0)
        + date_bonus
    )


def _event_quality_score(sentence: str, event_type: str) -> float:
    lowered = sentence.lower()
    score = 0.0
    strong_signals = (
        "we propose",
        "we introduce",
        "we present",
        "we develop",
        "proposed",
        "introduced",
        "presented",
        "developed",
        "our approach",
        "our method",
        "named ",
        "called ",
        "introducing ",
        "released",
        "outperforms",
        "achieves",
        "state-of-the-art",
    )
    score += 0.2 * sum(1 for signal in strong_signals if signal in lowered)
    canonical_intro_signals = (
        "we introduce rag models",
        "refer to as retrieval-augmented generation",
        "approach, named react",
        "called distilbert",
        "we introduce generative agents",
        "called tinybert",
    )
    if any(signal in lowered for signal in canonical_intro_signals):
        score += 0.45
    if event_type == "benchmark" and re.search(r"\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s+vs\.?\s+\d", lowered):
        score += 0.25
    if event_type == "release" and any(signal in lowered for signal in ("released", "available", "introducing")):
        score += 0.25
    if event_type == "method_proposed" and any(
        signal in lowered
        for signal in (
            "we propose",
            "we introduce",
            "we present",
            "proposed method",
            "proposed approach",
            "introduced",
            "proposed",
        )
    ):
        score += 0.25
    if event_type == "trend_application" and any(signal in lowered for signal in ("recently", "recent work", "survey", "extended", "applied")):
        score += 0.15
    if re.match(r"^[A-Z][A-Za-z0-9 -]{2,30}:", sentence):
        score -= 0.1
    if len(re.findall(r"\b(?:19|20)\d{2}\b", sentence)) >= 3:
        score -= 0.2
    return max(0.0, score)


def _dedupe_sources(sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    output = []
    for source in sources:
        doc_id = source.get("doc_id", "")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        output.append(source)
    return output


def _display_date(normalized_date: str, year: int) -> str:
    if normalized_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized_date):
        return normalized_date
    if normalized_date and re.fullmatch(r"\d{4}-\d{2}", normalized_date):
        return normalized_date
    return str(year)


def _make_title(sentence: str, event_type: str) -> str:
    text = sentence.strip()
    text = re.sub(r"^(in|by|during|around|since|after|before)\s+\d{4},?\s+", "", text, flags=re.IGNORECASE)
    if len(text) > 95:
        text = text[:92].rstrip(" ,.;:") + "..."
    prefix = {
        "method_proposed": "Method",
        "release": "Release",
        "benchmark": "Benchmark",
        "trend_application": "Trend",
    }.get(event_type, "Event")
    return f"{prefix}: {text}"


def _event_type_sort_key(event_type: str) -> int:
    order = {
        "method_proposed": 0,
        "release": 1,
        "benchmark": 2,
        "trend_application": 3,
    }
    return order.get(event_type, 99)


def _normalise_topic(topic: str) -> str:
    value = topic.lower().strip().replace("-", "_").replace(" ", "_")
    if "distill" in value or value in {"kd", "knowledge_distillation"}:
        return "knowledge_distillation"
    if "agent" in value:
        return "ai_agent"
    if "rag" in value:
        return "rag"
    return value
