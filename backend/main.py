from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflows.online_pipeline import (  # noqa: E402
    available_ml_models,
    get_local_qa_answer,
    load_evaluation_metrics,
)
from src.generation.llm_answerer import llm_runtime_status  # noqa: E402
from src.utils.text import repair_pdf_hyphenation  # noqa: E402

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "event_predictions.jsonl"
TIMELINE_PATH = PROJECT_ROOT / "data" / "processed" / "timeline.json"
METRICS_PATH = PROJECT_ROOT / "data" / "eval" / "ml_baseline_metrics.json"
RETRIEVAL_METRICS_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_metrics.json"
HF_BENCHMARK_PATH = PROJECT_ROOT / "data" / "eval" / "huggingface_benchmark.json"
VECTOR_SUMMARY_PATH = PROJECT_ROOT / "data" / "vector_db" / "index_summary.json"

TOPICS = [
    {"id": "rag", "label": "RAG"},
    {"id": "ai_agent", "label": "AI Agent"},
    {"id": "knowledge_distillation", "label": "Knowledge Distillation"},
]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    topic: str
    question: str
    history: Optional[List[ChatMessage]] = None


app = FastAPI(
    title="ChronoRAG API",
    version="0.1.0",
    description="Backend API for ChronoRAG timeline-aware research assistant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    docs = _documents()
    predictions = _predictions()
    timeline = _timeline()
    return {
        "status": "healthy",
        "documents": len(docs),
        "predictions": len(predictions),
        "timelineEvents": sum(len(events) for events in timeline.get("topics", {}).values()),
        "models": available_ml_models(),
        "generation": llm_runtime_status(),
    }


@app.get("/api/topics")
def topics() -> Dict[str, Any]:
    return {"topics": TOPICS}


@app.get("/api/overview")
def overview(topic: str = Query("rag")) -> Dict[str, Any]:
    topic_key = _normalise_topic(topic)
    docs = _topic_documents(topic_key)
    predictions = _topic_predictions(topic_key)
    event_predictions = [row for row in predictions if int(row.get("is_event", 0) or 0) == 1]
    timeline_events = _topic_timeline(topic_key)

    years = [
        int(row["year"])
        for row in timeline_events
        if isinstance(row.get("year"), int) or str(row.get("year", "")).isdigit()
    ]
    if not years:
        years = [
            int(row.get("year"))
            for row in docs
            if str(row.get("year", "")).isdigit()
        ]
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"

    avg_confidence = (
        round(mean(float(row.get("event_probability", 0.0) or 0.0) for row in event_predictions), 3)
        if event_predictions
        else 0.0
    )
    docs_total = len(_documents())
    processed_topics = sorted({_normalise_topic(str(row.get("topic", ""))) for row in _documents()})

    return {
        "topic": topic_key,
        "summary": {
            "eventsDetected": len(event_predictions),
            "timelineEvents": len(timeline_events),
            "docsIngested": len(docs),
            "totalDocs": docs_total,
            "avgConfidence": avg_confidence,
            "yearRange": year_range,
            "corpusIngestedPercent": 100.0 if docs_total else 0.0,
        },
        "topicBreakdown": [
            {
                "topic": item["id"],
                "label": item["label"],
                "documents": len(_topic_documents(item["id"])),
                "events": len([row for row in _topic_predictions(item["id"]) if int(row.get("is_event", 0) or 0) == 1]),
                "timelineEvents": len(_topic_timeline(item["id"])),
            }
            for item in TOPICS
        ],
        "modelStatus": {
            "eventDetector": "TF-IDF + LinearSVM",
            "eventTypeModel": "TF-IDF + SGD Logistic",
            "retrieval": _retrieval_status(),
            "timeline": "Rule-filtered ML events + TF-IDF clustering",
            "processedTopics": processed_topics,
        },
        "timelinePreview": timeline_events[:6],
    }


@app.get("/api/timeline")
def timeline(topic: str = Query("rag"), limit: int = Query(30, ge=1, le=100)) -> Dict[str, Any]:
    topic_key = _normalise_topic(topic)
    events = _topic_timeline(topic_key)[:limit]
    metadata = _timeline().get("metadata", {})
    return {
        "topic": topic_key,
        "source": _display_path(TIMELINE_PATH),
        "summary": {
            "total": len(_topic_timeline(topic_key)),
            **metadata,
        },
        "events": events,
    }


@app.get("/api/events")
def events(
    topic: str = Query("rag"),
    limit: int = Query(30, ge=1, le=200),
    event_only: bool = True,
) -> Dict[str, Any]:
    topic_key = _normalise_topic(topic)
    rows = _topic_predictions(topic_key)
    if event_only:
        rows = [row for row in rows if int(row.get("is_event", 0) or 0) == 1]
        rows.sort(
            key=lambda row: (
                float(row.get("event_probability", 0.0) or 0.0),
                float(row.get("event_type_confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
    else:
        rows.sort(key=lambda row: float(row.get("event_probability", 0.0) or 0.0), reverse=True)

    return {
        "topic": topic_key,
        "summary": {
            "totalSentences": len(_topic_predictions(topic_key)),
            "predictedEvents": len([row for row in _topic_predictions(topic_key) if int(row.get("is_event", 0) or 0) == 1]),
        },
        "events": [_prediction_for_api(row) for row in rows[:limit]],
    }


@app.get("/api/sources")
def sources(topic: Optional[str] = Query(None)) -> Dict[str, Any]:
    topic_key = _normalise_topic(topic) if topic else None
    docs = _documents()
    if topic_key:
        docs = [row for row in docs if _normalise_topic(str(row.get("topic", ""))) == topic_key]

    return {
        "topic": topic_key,
        "sources": [_document_for_api(row) for row in docs],
    }


@app.get("/api/evaluation")
def evaluation(model: str = Query("sgd_log")) -> Dict[str, Any]:
    metrics = _metrics()
    model_names = sorted(metrics.get("models", {}).keys()) if metrics else available_ml_models()
    if model not in model_names and model_names:
        model = model_names[0]

    shaped = load_evaluation_metrics(model)
    model_metrics = metrics.get("models", {}).get(model, {}) if metrics else {}
    binary_test = model_metrics.get("binary", {}).get("test", {})
    event_type_test = model_metrics.get("event_type", {}).get("test", {})

    comparison = []
    for name, row in sorted(metrics.get("models", {}).items()) if metrics else []:
        comparison.append(
            {
                "model": name,
                "binaryF1": _metric(row, "binary", "f1_macro"),
                "eventTypeF1": _metric(row, "event_type", "f1_macro"),
                "binaryAccuracy": _metric(row, "binary", "accuracy"),
                "eventTypeAccuracy": _metric(row, "event_type", "accuracy"),
            }
        )

    return {
        "model": model,
        "models": model_names,
        "summary": shaped["summary"],
        "binary": {
            "accuracy": binary_test.get("accuracy"),
            "f1Macro": binary_test.get("f1_macro"),
            "precisionMacro": binary_test.get("precision_macro"),
            "recallMacro": binary_test.get("recall_macro"),
            "confusionMatrix": binary_test.get("confusion_matrix", []),
            "labels": binary_test.get("labels", []),
        },
        "eventType": {
            "accuracy": event_type_test.get("accuracy"),
            "f1Macro": event_type_test.get("f1_macro"),
            "precisionMacro": event_type_test.get("precision_macro"),
            "recallMacro": event_type_test.get("recall_macro"),
            "confusionMatrix": event_type_test.get("confusion_matrix", []),
            "labels": event_type_test.get("labels", []),
        },
        "comparison": comparison,
    }


@app.get("/api/huggingface_eval")
def huggingface_eval() -> Dict[str, Any]:
    """Standalone retrieval benchmark on a public HuggingFace dataset.

    Produced offline by ``scripts/18_eval_huggingface_beir.py`` against the
    BEIR/SciFact dataset. Lets us report numbers comparable to the IR
    literature instead of only the in-corpus eval, which is custom-built.
    """
    data = _huggingface_benchmark()
    if not data:
        return {
            "available": False,
            "source": _display_path(HF_BENCHMARK_PATH),
            "note": "Run `python scripts/18_eval_huggingface_beir.py` to populate.",
        }
    return {
        "available": True,
        "source": _display_path(HF_BENCHMARK_PATH),
        "summary": data.get("summary", {}),
        "config": data.get("config", {}),
    }


@app.get("/api/retrieval_eval")
def retrieval_eval() -> Dict[str, Any]:
    """Retrieval benchmark numbers (Recall@k, MRR, per-topic).

    Produced offline by ``scripts/17_eval_retrieval.py`` against the
    hand-curated set at ``data/eval/retrieval_eval.jsonl``. Returns an empty
    object when the benchmark has not been run yet so the UI can fall back
    gracefully.
    """
    data = _retrieval_metrics()
    if not data:
        return {
            "available": False,
            "source": _display_path(RETRIEVAL_METRICS_PATH),
            "note": "Run `python scripts/17_eval_retrieval.py` to populate.",
        }
    return {
        "available": True,
        "source": _display_path(RETRIEVAL_METRICS_PATH),
        "summary": data.get("summary", {}),
        "perTopic": data.get("per_topic", {}),
        "perIntent": data.get("per_intent", {}),
        "config": data.get("config", {}),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    answer = get_local_qa_answer(request.topic, request.question, history=history)
    return {
        "topic": _normalise_topic(request.topic),
        "question": request.question,
        "answer": answer.get("answer", ""),
        "citations": answer.get("citations", []),
        "mode": answer.get("mode", "local"),
        "provider": answer.get("provider", "template"),
        "model": answer.get("model", "template-answerer"),
    }


@lru_cache(maxsize=1)
def _documents() -> List[Dict[str, Any]]:
    return _load_jsonl(DOCUMENTS_PATH)


@lru_cache(maxsize=1)
def _predictions() -> List[Dict[str, Any]]:
    return _load_jsonl(PREDICTIONS_PATH)


@lru_cache(maxsize=1)
def _timeline() -> Dict[str, Any]:
    if not TIMELINE_PATH.exists():
        return {"metadata": {}, "topics": {}}
    try:
        data = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"metadata": {}, "topics": {}}
    return data if isinstance(data, dict) else {"metadata": {}, "topics": {}}


@lru_cache(maxsize=1)
def _metrics() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@lru_cache(maxsize=1)
def _vector_summary() -> Dict[str, Any]:
    if not VECTOR_SUMMARY_PATH.exists():
        return {}
    try:
        return json.loads(VECTOR_SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@lru_cache(maxsize=1)
def _retrieval_metrics() -> Dict[str, Any]:
    if not RETRIEVAL_METRICS_PATH.exists():
        return {}
    try:
        return json.loads(RETRIEVAL_METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@lru_cache(maxsize=1)
def _huggingface_benchmark() -> Dict[str, Any]:
    if not HF_BENCHMARK_PATH.exists():
        return {}
    try:
        return json.loads(HF_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _topic_documents(topic: str) -> List[Dict[str, Any]]:
    return [row for row in _documents() if _normalise_topic(str(row.get("topic", ""))) == topic]


def _topic_predictions(topic: str) -> List[Dict[str, Any]]:
    return [row for row in _predictions() if _normalise_topic(str(row.get("topic", ""))) == topic]


def _topic_timeline(topic: str) -> List[Dict[str, Any]]:
    return list(_timeline().get("topics", {}).get(topic, []))


def _prediction_for_api(row: Dict[str, Any]) -> Dict[str, Any]:
    sentence = repair_pdf_hyphenation(str(row.get("sentence", "") or ""))
    return {
        "sentenceId": row.get("sentence_id"),
        "docId": row.get("doc_id"),
        "chunkId": row.get("chunk_id"),
        "topic": _normalise_topic(str(row.get("topic", ""))),
        "year": row.get("year"),
        "sourceUrl": row.get("source_url"),
        "sentence": sentence,
        "isEvent": int(row.get("is_event", 0) or 0),
        "probability": float(row.get("event_probability", 0.0) or 0.0),
        "eventType": row.get("event_type"),
        "eventTypeConfidence": float(row.get("event_type_confidence", 0.0) or 0.0),
        "normalizedDate": row.get("normalized_date"),
        "dateConfidence": float(row.get("date_confidence", 0.0) or 0.0),
        "dateSource": row.get("date_source"),
    }


_PDF_HEADER_RE = re.compile(
    r"^(?:arXiv:\S+(?:\s+\[[^\]]+\])?(?:\s+\d{1,2}\s+\w+\s+\d{4})?\s+|\d{1,2}\s+(?=[A-Z]))"
)


def _document_for_api(row: Dict[str, Any]) -> Dict[str, Any]:
    text = str(row.get("text", "") or "")
    title = str(row.get("title", "") or "")
    # Normalise whitespace first so multi-line headers/titles get detected and
    # stripped cleanly (raw text often has "Title\nSubtitle\nAuthor...").
    preview = re.sub(r"\s+", " ", text).strip()
    preview = repair_pdf_hyphenation(preview)
    # Strip PDF preamble like "arXiv:2002.08909v1 [cs.CL] 10 Feb 2020" or a
    # bare leading section number "1 Title".
    preview = _PDF_HEADER_RE.sub("", preview)
    # If the document text simply restates its own title at the top, drop that
    # duplicate so the preview adds new information.
    if title:
        normalised_title = re.sub(r"\s+", " ", title).strip()
        if preview.lower().startswith(normalised_title.lower()):
            preview = preview[len(normalised_title):].lstrip(" :-.,")
    return {
        "docId": row.get("doc_id"),
        "title": row.get("title"),
        "topic": _normalise_topic(str(row.get("topic", ""))),
        "sourceType": row.get("source_type"),
        "sourceUrl": row.get("source_url"),
        "year": row.get("year"),
        "authors": row.get("authors"),
        "wordCount": len(text.split()),
        "preview": preview[:280],
    }


def _normalise_topic(topic: Optional[str]) -> str:
    if not topic:
        return "rag"
    value = topic.lower().strip().replace("-", "_").replace(" ", "_")
    if "distill" in value or value in {"kd", "knowledge_distillation"}:
        return "knowledge_distillation"
    if "agent" in value:
        return "ai_agent"
    if "rag" in value:
        return "rag"
    return value


def _retrieval_status() -> str:
    summary = _vector_summary()
    chunks = summary.get("bm25", {}).get("chunks_indexed")
    if chunks:
        return f"BM25 indexed ({chunks} chunks)"
    return "BM25 local fallback"


def _metric(row: Dict[str, Any], family: str, metric: str) -> Optional[float]:
    value = row.get(family, {}).get("test", {}).get(metric)
    return round(float(value), 4) if value is not None else None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
