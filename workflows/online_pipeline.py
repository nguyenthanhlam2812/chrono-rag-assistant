"""Backend functions for the Streamlit demo app.

This module is named ``online_pipeline.py`` for historical reasons but it is
NOT an online inference pipeline. It is the data layer for the Streamlit UI
(``app/streamlit_app.py``) and has three responsibilities:

1. Demo content for the three MVP topics (timeline, sentence predictions,
   fallback Q&A) -- used when the corpus is not yet processed.
2. ``load_evaluation_metrics`` -- reads the **real** Sprint 4 metrics JSON
   produced by ``scripts/04_train_ml_classifier.py`` and shapes it for the
   Evaluation tab. Falls back to a clearly-labelled placeholder if training
   hasn't been run yet.
3. ``get_local_qa_answer`` -- wires the Chatbot tab to ``SimpleRetriever`` +
   ``TemplateAnswerer`` when ``data/processed/chunks.jsonl`` exists.
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ML_METRICS_PATH = PROJECT_ROOT / "data" / "eval" / "ml_baseline_metrics.json"
EVENT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "event_predictions.jsonl"
TIMELINE_PATH = PROJECT_ROOT / "data" / "processed" / "timeline.json"

EVENT_TYPE_LABELS = ["method_proposed", "release", "benchmark", "trend_application"]


def get_demo_timeline(topic: str) -> List[Dict[str, Any]]:
    """Sprint 0 demo timeline used by the UI when no precomputed timeline exists."""
    topic_clean = topic.lower().strip()

    if "rag" in topic_clean:
        return [
            {
                "event_id": "rag_evt_001",
                "date": "May 2020",
                "year": 2020,
                "event_type": "method_proposed",
                "title": "Retrieval-Augmented Generation (RAG) proposed",
                "representative_sentence": "In 2020, Lewis et al. proposed Retrieval-Augmented Generation (RAG) to combine parametric and non-parametric memory.",
                "confidence": 0.95,
                "sources": [
                    {
                        "doc_id": "rag_001",
                        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                        "source_url": "https://arxiv.org/abs/2005.11401",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "rag_evt_002",
                "date": "Late 2022",
                "year": 2022,
                "event_type": "release",
                "title": "Open-source RAG orchestrators",
                "representative_sentence": "In 2022, open-source frameworks like LangChain and LlamaIndex were released, making RAG implementation accessible.",
                "confidence": 0.88,
                "sources": [
                    {
                        "doc_id": "rag_001",
                        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                        "source_url": "https://arxiv.org/abs/2005.11401",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "rag_evt_003",
                "date": "Early 2024",
                "year": 2024,
                "event_type": "method_proposed",
                "title": "Microsoft introduces GraphRAG",
                "representative_sentence": "In early 2024, Microsoft introduced GraphRAG, which leverages Knowledge Graphs rather than simple vector similarity for retrieval.",
                "confidence": 0.91,
                "sources": [
                    {
                        "doc_id": "rag_001",
                        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                        "source_url": "https://arxiv.org/abs/2005.11401",
                    }
                ],
                "cluster_size": 1,
            },
        ]

    if "agent" in topic_clean:
        return [
            {
                "event_id": "agent_evt_001",
                "date": "March 2023",
                "year": 2023,
                "event_type": "release",
                "title": "AutoGPT release",
                "representative_sentence": "In March 2023, Toran Bruce Richards released AutoGPT, demonstrating self-directed task loops that execute complex multi-step objectives.",
                "confidence": 0.94,
                "sources": [
                    {
                        "doc_id": "agent_001",
                        "title": "LLM Powered Autonomous Agents",
                        "source_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "agent_evt_002",
                "date": "June 2023",
                "year": 2023,
                "event_type": "trend_application",
                "title": "LLM-powered autonomous agent framework",
                "representative_sentence": "In June 2023, Lilian Weng published a seminal blog post outlining LLM-powered autonomous agents.",
                "confidence": 0.96,
                "sources": [
                    {
                        "doc_id": "agent_001",
                        "title": "LLM Powered Autonomous Agents",
                        "source_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "agent_evt_003",
                "date": "2024",
                "year": 2024,
                "event_type": "release",
                "title": "Multi-agent frameworks gain popularity",
                "representative_sentence": "By 2024, multi-agent orchestrations like CrewAI and Microsoft's AutoGen emerged.",
                "confidence": 0.85,
                "sources": [
                    {
                        "doc_id": "agent_001",
                        "title": "LLM Powered Autonomous Agents",
                        "source_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    }
                ],
                "cluster_size": 1,
            },
        ]

    if "distill" in topic_clean:
        return [
            {
                "event_id": "kd_evt_001",
                "date": "2015",
                "year": 2015,
                "event_type": "method_proposed",
                "title": "Knowledge Distillation popularized by Hinton et al.",
                "representative_sentence": "In 2015, Geoffrey Hinton, Oriol Vinyals, and Jeff Dean popularized the concept of Knowledge Distillation.",
                "confidence": 0.97,
                "sources": [
                    {
                        "doc_id": "kd_001",
                        "title": "Distilling the Knowledge in a Neural Network",
                        "source_url": "https://arxiv.org/abs/1503.02531",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "kd_evt_002",
                "date": "2019",
                "year": 2019,
                "event_type": "release",
                "title": "DistilBERT release",
                "representative_sentence": "In 2019, Victor Sanh et al. released DistilBERT, compressing BERT by 40% while retaining 97% of its performance.",
                "confidence": 0.90,
                "sources": [
                    {
                        "doc_id": "kd_001",
                        "title": "Distilling the Knowledge in a Neural Network",
                        "source_url": "https://arxiv.org/abs/1503.02531",
                    }
                ],
                "cluster_size": 1,
            },
            {
                "event_id": "kd_evt_003",
                "date": "2020",
                "year": 2020,
                "event_type": "method_proposed",
                "title": "TinyBERT architecture introduced",
                "representative_sentence": "In 2020, Jiao et al. introduced TinyBERT, performing transformer-layer distillation.",
                "confidence": 0.89,
                "sources": [
                    {
                        "doc_id": "kd_001",
                        "title": "Distilling the Knowledge in a Neural Network",
                        "source_url": "https://arxiv.org/abs/1503.02531",
                    }
                ],
                "cluster_size": 1,
            },
        ]

    return []


def get_timeline_events(topic: str, limit: int = 30) -> Dict[str, Any]:
    """Load the real generated timeline for a topic, with a demo fallback."""
    timeline = _load_timeline_json(TIMELINE_PATH)
    topic_key = _normalise_topic(topic)
    if timeline:
        events = timeline.get("topics", {}).get(topic_key, [])
        if events:
            return {
                "is_real": True,
                "source": _display_path(TIMELINE_PATH),
                "summary": {
                    "total": len(events),
                    "topic": topic_key,
                    **timeline.get("metadata", {}),
                },
                "events": events[:limit],
            }

    fallback = get_demo_timeline(topic)[:limit]
    return {
        "is_real": False,
        "source": "demo fallback",
        "summary": {"total": len(fallback), "topic": topic_key},
        "events": fallback,
    }


def get_demo_sentence_predictions(topic: str) -> List[Dict[str, Any]]:
    """Sprint 0 demo predictions for the Event Detection tab."""
    return [
        {
            "sentence": "Retrieval-Augmented Generation (RAG) has become a key methodology in modern natural language processing.",
            "is_event": 0,
            "prob": 0.12,
            "type": "none",
        },
        {
            "sentence": "In 2020, Lewis et al. proposed RAG to combine parametric memory and non-parametric memory.",
            "is_event": 1,
            "prob": 0.95,
            "type": "method_proposed",
        },
        {
            "sentence": "By 2021, various researchers adapted RAG for question answering and open domain dialogues.",
            "is_event": 1,
            "prob": 0.74,
            "type": "trend_application",
        },
        {
            "sentence": "In 2022, open-source frameworks like LangChain and LlamaIndex were released.",
            "is_event": 1,
            "prob": 0.88,
            "type": "release",
        },
        {
            "sentence": "During 2023, Vector Databases like Pinecone, Milvus, and Qdrant saw massive adoption.",
            "is_event": 0,
            "prob": 0.45,
            "type": "none",
        },
        {
            "sentence": "In early 2024, Microsoft introduced GraphRAG, which leverages Knowledge Graphs for retrieval.",
            "is_event": 1,
            "prob": 0.91,
            "type": "method_proposed",
        },
    ]


def get_event_predictions(topic: str, limit: int = 10) -> Dict[str, Any]:
    """Load real precomputed event predictions for the UI, with demo fallback."""
    records = _load_predictions_jsonl(EVENT_PREDICTIONS_PATH)
    if not records:
        return {
            "is_real": False,
            "source": "demo fallback",
            "summary": {"total": len(get_demo_sentence_predictions(topic)), "events": 0},
            "rows": get_demo_sentence_predictions(topic)[:limit],
        }

    topic_key = _normalise_topic(topic)
    topic_records = [row for row in records if _normalise_topic(str(row.get("topic", ""))) == topic_key]
    if not topic_records:
        return {
            "is_real": False,
            "source": "demo fallback",
            "summary": {"total": len(get_demo_sentence_predictions(topic)), "events": 0},
            "rows": get_demo_sentence_predictions(topic)[:limit],
        }

    event_rows = [
        row for row in topic_records
        if int(row.get("is_event", 0)) == 1
    ]
    non_event_rows = [
        row for row in topic_records
        if int(row.get("is_event", 0)) == 0
    ]

    event_rows = sorted(
        event_rows,
        key=lambda row: (
            float(row.get("event_probability", 0.0)),
            float(row.get("event_type_confidence", 0.0)),
        ),
        reverse=True,
    )
    non_event_rows = sorted(
        non_event_rows,
        key=lambda row: float(row.get("event_probability", 0.0)),
    )

    event_target = min(max(limit - 2, 1), len(event_rows))
    selected = event_rows[:event_target]
    selected.extend(non_event_rows[: max(0, limit - len(selected))])

    rows = [_prediction_row_for_ui(row) for row in selected]
    return {
        "is_real": True,
        "source": _display_path(EVENT_PREDICTIONS_PATH),
        "summary": {
            "total": len(topic_records),
            "events": len(event_rows),
            "non_events": len(non_event_rows),
        },
        "rows": rows,
    }


def _load_predictions_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _load_timeline_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("topics"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _prediction_row_for_ui(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sentence": row.get("sentence", ""),
        "is_event": int(row.get("is_event", 0)),
        "prob": float(row.get("event_probability", 0.0)),
        "type": row.get("event_type", "none"),
        "event_type_confidence": float(row.get("event_type_confidence", 0.0)),
        "doc_id": row.get("doc_id", ""),
        "year": row.get("year"),
        "source_url": row.get("source_url", ""),
        "binary_model": row.get("binary_model", ""),
        "event_type_model": row.get("event_type_model", ""),
    }


def _normalise_topic(topic: str) -> str:
    value = topic.lower().strip().replace("-", "_").replace(" ", "_")
    if "distill" in value or value in {"kd", "knowledge_distillation"}:
        return "knowledge_distillation"
    if "agent" in value:
        return "ai_agent"
    if "rag" in value:
        return "rag"
    return value


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_fallback_answer(topic: str, question: str) -> Dict[str, Any]:
    """Hand-crafted Sprint 0 answer used when no processed corpus exists yet."""
    topic_clean = topic.lower().strip()

    if "rag" in topic_clean:
        return {
            "answer": "Retrieval-Augmented Generation (RAG) was introduced in 2020 by Lewis et al. [rag_001] as a hybrid approach that joins parametric memory with external source documents [rag_001]. Major framework packages like LangChain were released in 2022 [rag_001] to speed up deployment, and Microsoft's GraphRAG launched in 2024 [rag_001] to improve retrieval context using Knowledge Graphs.",
            "citations": [
                {
                    "doc_id": "rag_001",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "source_url": "https://arxiv.org/abs/2005.11401",
                }
            ],
        }

    if "agent" in topic_clean:
        return {
            "answer": "LLM powered autonomous agents are designed using planning, memory, and tool usage architectures [agent_001]. AutoGPT was released in March 2023 [agent_001] followed by Weng's foundational agent post in June 2023 [agent_001]. Multi-agent orchestrations like CrewAI and AutoGen gained popularity in 2024 [agent_001].",
            "citations": [
                {
                    "doc_id": "agent_001",
                    "title": "LLM Powered Autonomous Agents",
                    "source_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                }
            ],
        }

    return {
        "answer": "Knowledge Distillation transfers dark knowledge from a big teacher model to a smaller student [kd_001]. Popularized by Hinton in 2015 [kd_001], it led to compact models like DistilBERT in 2019 [kd_001] and TinyBERT in 2020 [kd_001].",
        "citations": [
            {
                "doc_id": "kd_001",
                "title": "Distilling the Knowledge in a Neural Network",
                "source_url": "https://arxiv.org/abs/1503.02531",
            }
        ],
    }


def _load_metrics_json(metrics_path: Path) -> Optional[Dict[str, Any]]:
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _confusion_matrix_to_markdown(
    matrix: List[List[int]], labels: List[str], title: str
) -> str:
    if not matrix or not labels:
        return f"_No confusion matrix available for {title}._"
    header = "| True \\\\ Predicted | " + " | ".join(labels) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(labels)) + " |"
    rows = []
    for label, row in zip(labels, matrix):
        cells = [f"**{v}**" if labels[idx] == label else str(v) for idx, v in enumerate(row)]
        rows.append(f"| **{label}** | " + " | ".join(cells) + " |")
    return "\n".join([f"**{title}**", "", header, sep, *rows])


def available_ml_models() -> List[str]:
    """Return the list of ML models that have a real metrics entry in the JSON."""
    data = _load_metrics_json(ML_METRICS_PATH)
    if not data or "models" not in data:
        return []
    return sorted(data["models"].keys())


def load_evaluation_metrics(model_name: str = "logreg") -> Dict[str, Any]:
    """Read real Sprint 4 metrics if available, otherwise return a labelled placeholder.

    The returned dict has the shape consumed by the Streamlit Evaluation tab:
    ``summary`` (4 KPI strings), ``confusion_matrix_markdown``,
    ``experiment_markdown``, ``is_real`` (bool), ``source`` (str).
    """
    data = _load_metrics_json(ML_METRICS_PATH)

    if not data or "models" not in data or model_name not in data["models"]:
        return {
            "is_real": False,
            "source": "placeholder (run scripts/04_train_ml_classifier.py to populate)",
            "summary": {
                "event_detection_f1": "N/A",
                "event_type_macro_f1": "N/A",
                "retrieval_recall_at_5": "Sprint 5",
                "timeline_date_accuracy": _timeline_status(),
            },
            "confusion_matrix_markdown": (
                "_No real metrics found at `data/eval/ml_baseline_metrics.json`. "
                "Run `python scripts/04_train_ml_classifier.py` first._"
            ),
            "experiment_markdown": (
                "_Train the ML baselines to populate this comparison table._"
            ),
        }

    model_metrics = data["models"][model_name]
    binary_test = model_metrics["binary"]["test"]
    event_type_test = model_metrics["event_type"]["test"]

    binary_f1 = _format_pct(binary_test.get("f1_macro", 0.0))
    event_type_f1 = (
        _format_pct(event_type_test["f1_macro"])
        if isinstance(event_type_test, dict) and "f1_macro" in event_type_test
        else "N/A"
    )

    # Confusion matrix from the event-type test split (4-class).
    confusion_md = ""
    if isinstance(event_type_test, dict) and "confusion_matrix" in event_type_test:
        confusion_md = _confusion_matrix_to_markdown(
            event_type_test["confusion_matrix"],
            event_type_test.get("labels", EVENT_TYPE_LABELS),
            f"Event-type confusion matrix -- {model_name} (test split)",
        )
    else:
        confusion_md = "_Event-type test split has no rows; see val split instead._"

    # Experiment comparison: build from whatever models are present in JSON.
    rows = [
        "| Model | Binary F1 (test) | Event-type F1 macro (test) | Notes |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for name, m in sorted(data["models"].items()):
        b_test = m.get("binary", {}).get("test", {})
        e_test = m.get("event_type", {}).get("test", {})
        b_f1 = _format_pct(b_test.get("f1_macro", 0.0)) if "f1_macro" in b_test else "N/A"
        e_f1 = _format_pct(e_test["f1_macro"]) if isinstance(e_test, dict) and "f1_macro" in e_test else "N/A"
        marker = " <- selected" if name == model_name else ""
        rows.append(f"| **{name}**{marker} | {b_f1} | {e_f1} | TF-IDF baseline |")
    rows.append("| BiLSTM | N/A | N/A | Sprint 4 DL (optional) -- not implemented |")
    experiment_md = "\n".join(rows)

    return {
        "is_real": True,
        "source": str(ML_METRICS_PATH.relative_to(PROJECT_ROOT)),
        "summary": {
            "event_detection_f1": binary_f1,
            "event_type_macro_f1": event_type_f1,
            "retrieval_recall_at_5": "Sprint 5",
            "timeline_date_accuracy": _timeline_status(),
        },
        "confusion_matrix_markdown": confusion_md,
        "experiment_markdown": experiment_md,
    }


def get_local_qa_answer(topic: str, question: str) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate a template-based answer with real citations."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.simple_retriever import SimpleRetriever
    from src.generation.template_answerer import TemplateAnswerer

    chunks_path = PROJECT_ROOT / 'data' / 'processed' / 'chunks.jsonl'

    if not chunks_path.exists():
        return get_fallback_answer(topic, question)

    try:
        retriever = HybridRetriever(
            chunks_path=chunks_path,
            index_dir=PROJECT_ROOT / "data" / "vector_db",
            use_vector=True,
        )
        chunks = retriever.retrieve(question, topic=topic, top_k=3)
    except Exception:
        retriever = SimpleRetriever(chunks_path)
        chunks = retriever.retrieve(question, topic=topic, top_k=3) if retriever.chunks else []

    answerer = TemplateAnswerer()
    if not chunks:
        return answerer.generate_answer([], query=question)
    return answerer.generate_answer(chunks, query=question)


def _timeline_status() -> str:
    timeline = _load_timeline_json(TIMELINE_PATH)
    if not timeline:
        return "Not built"
    total = timeline.get("metadata", {}).get("timeline_events")
    if isinstance(total, int):
        return f"{total} events"
    return "Built"
