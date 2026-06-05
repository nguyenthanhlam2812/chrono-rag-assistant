"""Backend functions for the ChronoRAG dashboard.

This module is named ``online_pipeline.py`` for historical reasons but it is
NOT an online inference pipeline. It is the data layer behind the FastAPI
backend (`backend/main.py`) that serves the React dashboard. It has three
responsibilities:

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
import re
from functools import lru_cache
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

    The returned dict has the shape consumed by the Evaluation tab:
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


@lru_cache(maxsize=4)
def _get_retriever(chunks_path: Path):
    """Cache one retriever per chunks file so we don't reload bm25.pkl
    (~3MB) and chunks.jsonl on every chat request. Falls back to a simple
    BM25-only retriever if the hybrid index can't be built."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.simple_retriever import SimpleRetriever

    try:
        return HybridRetriever(
            chunks_path=chunks_path,
            index_dir=chunks_path.parent.parent / "vector_db",
            use_vector=True,
        )
    except Exception:
        return SimpleRetriever(chunks_path)


# Identity / help phrases that should never be routed through the RAG
# pipeline -- the corpus has no answer for "who are you" and retrieving on
# the bare token "ai" (or "you") used to surface a confident-sounding but
# unrelated paragraph about AI deployment.
#
# Patterns are stored already normalised (diacritic-stripped, casual VN
# shortcuts expanded) -- see ``normalize_vn``. A single canonical pattern
# like "may lam duoc gi" then matches "m làm dc gì", "m làm đc gì",
# "m làm ddc gì", "mày lam dc j", etc.
_RAW_META_PATTERNS = (
    # --- Identity (VI) ---
    "ban la ai", "ban la gi", "ban ten gi", "ten ban la gi",
    "may la ai", "may la gi", "may ten gi",
    "em la ai", "anh la ai", "tao la ai", "cau la ai",
    "bot la ai", "bot la gi", "bot ten gi",
    "ai tao ra may", "ai tao may", "ai lam ra may", "ai lam ra ban",
    "ai tao ra ban", "ai tao ban",
    "gioi thieu ban than", "gioi thieu ve ban", "gioi thieu chinh ban",
    # --- Capability (VI) ---
    "may lam duoc gi", "may lam gi duoc", "may lam gi",
    "may co the lam gi", "may co the lam duoc gi", "may co the giup gi",
    "may giup duoc gi", "may giup gi duoc", "may giup gi", "giup duoc gi",
    "may biet gi", "may biet lam gi", "may biet nhung gi",
    "may de lam gi", "may dung lam gi",
    "ban lam duoc gi", "ban lam gi duoc", "ban co the lam gi",
    "ban co the lam duoc gi", "ban giup duoc gi", "ban giup gi",
    "ban biet gi", "ban biet lam gi",
    "bot lam duoc gi", "bot giup duoc gi", "bot giup gi", "bot biet gi",
    # --- Usage / help (VI) ---
    "dung sao", "dung the nao", "su dung the nao",
    "huong dan", "huong dan dung", "huong dan su dung",
    "huong dan dung may", "lam sao de dung",
    # --- Identity / capability (EN) ---
    "who are you", "what are you", "what can you do", "what do you know",
    "what are your capabilities", "introduce yourself", "tell me about yourself",
    "who made you", "what's your name", "whats your name", "your name",
    "how do i use you", "how does this work",
    # --- Help shortcuts ---
    "help me", "help please", "/help", "help",
)

_META_QUESTION_PHRASES = tuple(
    __import__("src.utils.text", fromlist=["normalize_vn"]).normalize_vn(p)
    for p in _RAW_META_PATTERNS
)

_META_RESPONSE_VI = (
    "Mình là ChronoRAG, trợ lý nghiên cứu timeline cho 3 chủ đề: "
    "RAG, AI Agent, và Knowledge Distillation. Mình trả lời dựa trên "
    "corpus paper local kèm citation, không bịa.\n\n"
    "Thử hỏi: \"What is RAG?\", \"How does ReAct work?\", "
    "\"DistilBERT vs TinyBERT — what is the difference?\""
)

TEMPORAL_ENTITY_ANSWERS: Dict[str, Dict[str, Any]] = {
    "rag": {
        "answer_vi": (
            "RAG ra đời/được giới thiệu vào năm 2020 trong paper "
            "\"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks\" của Lewis et al. [rag_001]"
        ),
        "answer_en": (
            "RAG was introduced in 2020 in Lewis et al.'s paper "
            "\"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks\". [rag_001]"
        ),
        "citation": {
            "doc_id": "rag_001",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "source_url": "https://arxiv.org/abs/2005.11401",
        },
    },
    "self-rag": {
        "answer_vi": "Self-RAG được giới thiệu vào năm 2023 trong paper về Self-Reflective Retrieval-Augmented Generation. [rag_007]",
        "answer_en": "Self-RAG was introduced in 2023 in the Self-Reflective Retrieval-Augmented Generation paper. [rag_007]",
        "citation": {
            "doc_id": "rag_007",
            "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            "source_url": "https://arxiv.org/abs/2310.11511",
        },
    },
    "autogpt": {
        "answer_vi": "AutoGPT xuất hiện/phát hành vào năm 2023 dưới dạng dự án GitHub open-source. [agent_005]",
        "answer_en": "AutoGPT appeared as an open-source GitHub project in 2023. [agent_005]",
        "citation": {
            "doc_id": "agent_005",
            "title": "AutoGPT GitHub repository README",
            "source_url": "https://github.com/Significant-Gravitas/AutoGPT",
        },
    },
    "autogen": {
        "answer_vi": "AutoGen được Microsoft giới thiệu/phát hành trong giai đoạn 2023 như một framework open-source cho multi-agent AI. [agent_006]",
        "answer_en": "AutoGen was introduced by Microsoft around 2023 as an open-source framework for multi-agent AI. [agent_006]",
        "citation": {
            "doc_id": "agent_006",
            "title": "Microsoft AutoGen documentation",
            "source_url": "https://microsoft.github.io/autogen",
        },
    },
    "knowledge_distillation": {
        "answer_vi": "Knowledge Distillation trở thành mốc nền tảng vào năm 2015 với paper \"Distilling the Knowledge in a Neural Network\" của Hinton et al. [kd_001]",
        "answer_en": "Knowledge Distillation became a foundational technique in 2015 with Hinton et al.'s \"Distilling the Knowledge in a Neural Network\". [kd_001]",
        "citation": {
            "doc_id": "kd_001",
            "title": "Distilling the Knowledge in a Neural Network",
            "source_url": "https://arxiv.org/abs/1503.02531",
        },
    },
    "distilbert": {
        "answer_vi": "DistilBERT được giới thiệu vào năm 2019 như một phiên bản BERT nhỏ hơn dùng knowledge distillation. [kd_002]",
        "answer_en": "DistilBERT was introduced in 2019 as a smaller BERT model trained with knowledge distillation. [kd_002]",
        "citation": {
            "doc_id": "kd_002",
            "title": "DistilBERT, a distilled version of BERT",
            "source_url": "https://arxiv.org/abs/1910.01108",
        },
    },
}


CONCEPT_ENTITY_ANSWERS: Dict[str, Dict[str, Any]] = {
    "multi_agent_frameworks": {
        "answer_vi": (
            "Multi-agent frameworks là nhóm framework cho phép nhiều AI agent phối hợp qua hội thoại, vai trò hoặc luồng hành động để giải quyết nhiệm vụ. "
            "Trong corpus ChronoRAG, AutoGen là ví dụ framework multi-agent open-source, còn CAMEL mô tả cách các communicative agents phối hợp qua role-playing/task-solving. [agent_006] [agent_018]"
        ),
        "answer_en": (
            "Multi-agent frameworks let multiple AI agents coordinate through conversations, roles, or action workflows to solve tasks. "
            "In the ChronoRAG corpus, AutoGen is an open-source multi-agent framework, while CAMEL studies communicative agents that cooperate through role-playing/task-solving. [agent_006] [agent_018]"
        ),
        "citations": [
            {
                "doc_id": "agent_006",
                "title": "Microsoft AutoGen documentation",
                "source_url": "https://microsoft.github.io/autogen",
            },
            {
                "doc_id": "agent_018",
                "title": "CAMEL: Communicative Agents for \"Mind\" Exploration of Large Language Model Society",
                "source_url": "http://arxiv.org/abs/2303.17760",
            },
        ],
    }
}


def _looks_like_meta_question(question: str) -> bool:
    from src.utils.text import normalize_vn

    raw = (question or "").strip()
    # Bare "?" / "??" looks like the user fumbling for help. Other punctuation
    # bursts ("!!!", "...") are just noise -- let abstain handle them.
    if raw and all(ch == "?" for ch in raw):
        return True
    if not raw:
        return False
    normalised = normalize_vn(raw).rstrip(" ?!.,")
    if not normalised:
        return False
    if normalised in {"ban", "may", "you", "help", "bot"}:
        return True
    return any(phrase in normalised for phrase in _META_QUESTION_PHRASES)


def _try_temporal_entity_answer(question: str) -> Optional[Dict[str, Any]]:
    """Answer simple year/date questions from stable timeline facts.

    These questions are where pure lexical retrieval is weakest: "RAG ra đời
    năm bao nhiêu" shares the token "RAG" with many later papers, so BM25 can
    retrieve CRAG/FLARE instead of the founding RAG paper. For demo quality we
    route explicit temporal asks to curated corpus-backed milestones.
    """
    from src.utils.text import normalize_vn

    q = normalize_vn(question or "")
    if not q:
        return None
    temporal_signals = (
        "ra doi", "ra mat", "xuat hien", "duoc gioi thieu", "duoc de xuat",
        "phat hanh", "nam nao", "nam bao nhieu", "khi nao", "tu nam nao",
        "introduced", "proposed", "released", "when was", "what year",
    )
    if not any(signal in q for signal in temporal_signals):
        return None

    entity = _detect_temporal_entity(q)
    if not entity:
        return None
    row = TEMPORAL_ENTITY_ANSWERS.get(entity)
    if not row:
        return None
    is_english = bool(re.search(r"\b(when|what year|introduced|released|proposed)\b", q))
    return {
        "answer": row["answer_en"] if is_english else row["answer_vi"],
        "citations": [row["citation"]],
    }


def _try_concept_entity_answer(question: str) -> Optional[Dict[str, Any]]:
    """Answer short stable concept queries that often hit PDF fragments."""
    from src.utils.text import normalize_vn

    raw = question or ""
    q = normalize_vn(raw)
    if not q:
        return None
    compact = f" {q} "
    asks_multi_agent_framework = (
        ("multi agent" in compact or "multi-agent" in raw.lower())
        and ("framework" in compact or "frameworks" in compact)
    )
    if not asks_multi_agent_framework:
        return None
    row = CONCEPT_ENTITY_ANSWERS["multi_agent_frameworks"]
    is_english = bool(re.search(r"\b(what|explain|framework|multi-agent)\b", raw.lower()))
    return {
        "answer": row["answer_en"] if is_english else row["answer_vi"],
        "citations": row["citations"],
    }


def _detect_temporal_entity(normalized_question: str) -> Optional[str]:
    q = normalized_question
    if "self rag" in q or "self-rag" in q or "self - rag" in q or "selfrag" in q:
        return "self-rag"
    if "knowledge distillation" in q or ("knowledge" in q and "distillation" in q):
        return "knowledge_distillation"
    for entity in ("distilbert", "autogen", "autogpt", "rag"):
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(entity)}(?:[^a-z0-9]|$)", q):
            return entity
    return None


def _with_generation_meta(
    payload: Dict[str, Any],
    *,
    mode: str = "local",
    provider: str = "template",
    model: str = "template-answerer",
) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched.setdefault("mode", mode)
    enriched.setdefault("provider", provider)
    enriched.setdefault("model", model)
    return enriched


# Each topic carries a small dictionary of strong indicator phrases. If the
# question mentions one, we trust the question over the dropdown -- this fixes
# the "AI Agent có từ năm nào" while topic=rag failure mode.
#
# Items are matched as case-insensitive WORD-BOUNDARY regex tokens so that
# "rag" matches "rag", "RAG", "rag?", "RAG là gì" -- but NOT "fragment",
# "drag", "ragged". Multi-word phrases like "ai agent" match as-is.
_TOPIC_INTENT_HINTS: Dict[str, tuple] = {
    "ai_agent": (
        "ai agent", "ai agents",
        "autogen", "autogpt", "babyagi", "metagpt", "crewai", "agentverse",
        "langgraph", "toolformer",
        "react", "react agent", "react framework",
        "multi-agent", "multi agent",
        "agent loop", "autonomous agent", "autonomous agents", "agentic",
        "llm agent", "llm-powered agent",
    ),
    "knowledge_distillation": (
        "knowledge distillation", "distillation", "distil", "distilled",
        "distilbert", "tinybert", "mobilebert", "minilm", "albert",
        "teacher-student", "teacher student",
        "model compression", "kd",
    ),
    "rag": (
        "rag", "retrieval-augmented", "retrieval augmented",
        "self-rag", "self rag", "graphrag",
        "realm", "atlas",
        "dense passage retrieval", "dense passage retriever", "dpr",
        "retro",  # word boundary stops "retrofit"
        "flare", "crag",
        "active retrieval", "corrective retrieval",
    ),
}


def _detect_topic_intent(question: str) -> Optional[str]:
    """Return the topic the question most clearly asks about, or None.

    Used to override the UI topic dropdown when the user clearly types about
    a different topic ("AI Agent có từ năm nào" while RAG is selected).
    """
    q = (question or "").lower().strip()
    if not q:
        return None
    for topic_key, hints in _TOPIC_INTENT_HINTS.items():
        for hint in hints:
            if " " in hint:
                if hint in q:
                    return topic_key
            else:
                # Word-boundary so "rag" doesn't fire on "fragment"
                if re.search(rf"(?:^|[^a-z0-9]){re.escape(hint)}(?:[^a-z0-9]|$)", q):
                    return topic_key
    return None


# Short follow-up cues that almost certainly point back at the previous turn.
# Used to enrich retrieval for questions like "nó làm được gì" — the bare
# tokens have no BM25 signal, so we prepend the previous user message.
_FOLLOWUP_CUES = (
    "nó", "no ", "vậy", "thế",
    "it ", "they ", "that ", "this ", "those ", "these ",
)


def _augment_short_followup(
    question: str, history: Optional[List[Dict[str, str]]]
) -> str:
    """Prepend the previous user message when the current question looks like
    a short pronoun follow-up. Leaves longer / self-contained questions alone."""
    if not history:
        return question
    q = (question or "").strip()
    if not q or len(q) > 40:
        return question
    lowered = f" {q.lower()} "
    if not any(cue in lowered for cue in _FOLLOWUP_CUES):
        return question
    last_user = next(
        (str(h.get("content", "")).strip() for h in reversed(history) if h.get("role") == "user" and h.get("content")),
        "",
    )
    if not last_user or last_user == q:
        return question
    return f"{last_user} -- {q}"


def _looks_like_non_corpus_question(question: str) -> bool:
    """Catch obvious off-topic / adversarial prompts before retrieval.

    A small lexical RAG system is vulnerable to accidental token overlap: SQL
    prompt junk can match "drop" in result text, and Vietnamese daily-life
    questions can match random OCR fragments. If the question has no ChronoRAG
    entity and clearly asks outside the MVP domain, abstain early.
    """
    from src.utils.text import normalize_vn

    raw = (question or "").strip()
    if not raw:
        return False

    if _detect_topic_intent(raw) or _detect_temporal_entity(normalize_vn(raw)):
        return False

    lowered = raw.lower()
    if re.search(
        r"(drop\s+table|select\s+\*|insert\s+into|delete\s+from|<script|</script|--|;\s*--|ignore\s+previous|jailbreak)",
        lowered,
    ):
        return True

    q = normalize_vn(raw)
    off_topic_signals = (
        "lich su viet nam",
        "toi nen an",
        "nen an gi",
        "an gi toi nay",
        "an gi hom nay",
        "mon an",
        "bong da",
        "anime",
        "stock price",
        "gia co phieu",
        "thoi tiet",
        "weather",
    )
    return any(signal in q for signal in off_topic_signals)


def get_local_qa_answer(
    topic: str,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate a template-based answer with real citations.

    ``history`` is an optional list of ``{"role": "user"|"assistant", "content": str}``
    messages (most recent last). It is only used by the LLM augmentation step
    so that follow-up questions like "nó làm được gì" can be resolved against
    the previous turn -- the local template answerer is intentionally stateless.
    """
    from src.generation.query_router import maybe_answer_direct
    from src.generation.llm_answerer import (
        maybe_generate_llm_answer,
        maybe_generate_project_llm_answer,
    )
    from src.generation.project_answerer import generate_project_answer
    from src.generation.scope_guard import (
        SCOPE_BORDERLINE,
        SCOPE_OUT,
        borderline_answer,
        classify_chat_scope,
        is_project_guidance_question,
    )
    from src.generation.template_answerer import NO_ANSWER_MESSAGE, TemplateAnswerer

    scope_decision = classify_chat_scope(question)
    if scope_decision.label == SCOPE_OUT:
        return _with_generation_meta({"answer": NO_ANSWER_MESSAGE, "citations": []})
    if scope_decision.label == SCOPE_BORDERLINE:
        return _with_generation_meta(
            {"answer": borderline_answer(), "citations": []},
            mode="router",
            provider="scope_guard",
            model="scope-guard:borderline",
        )

    direct_answer = maybe_answer_direct(question, history=history)
    if direct_answer is not None:
        return _with_generation_meta(direct_answer)

    if _looks_like_meta_question(question):
        return _with_generation_meta({"answer": _META_RESPONSE_VI, "citations": []})

    # Catch greetings / thanks / time / common-term definitions BEFORE the RAG
    # pipeline. Otherwise "transformer là gì" abstains (not in corpus) and
    # "chào bạn" makes BM25 retrieve unrelated chunks.
    from src.generation.conversational_handler import handle_conversational
    chitchat = handle_conversational(question)
    if chitchat is not None:
        return _with_generation_meta(chitchat)

    if is_project_guidance_question(question):
        project_llm_answer = maybe_generate_project_llm_answer(question, history=history)
        if project_llm_answer is not None:
            return project_llm_answer
        return generate_project_answer(question)

    if _looks_like_non_corpus_question(question):
        return _with_generation_meta({"answer": NO_ANSWER_MESSAGE, "citations": []})

    temporal_answer = _try_temporal_entity_answer(question)
    if temporal_answer is not None:
        return _with_generation_meta(temporal_answer)

    concept_answer = _try_concept_entity_answer(question)
    if concept_answer is not None:
        return _with_generation_meta(concept_answer)

    chunks_path = PROJECT_ROOT / 'data' / 'processed' / 'chunks.jsonl'

    if not chunks_path.exists():
        return _with_generation_meta(get_fallback_answer(topic, question))

    # Short follow-ups like "nó làm được gì" have no BM25 signal on their own;
    # the retrieval query reuses the previous user message so the right chunks
    # come back. The LLM still receives the *original* question + history so
    # the answer addresses what was actually asked.
    retrieval_query = _augment_short_followup(question, history)

    # Trust an explicit topic the question mentions over the UI dropdown so the
    # retriever doesn't pull from the wrong corpus slice.
    resolved_topic = (
        _detect_topic_intent(retrieval_query) or _detect_topic_intent(question) or topic
    )

    retriever = _get_retriever(chunks_path)
    chunks = (
        retriever.retrieve(retrieval_query, topic=resolved_topic, top_k=3)
        if getattr(retriever, "chunks", None)
        else []
    )

    answerer = TemplateAnswerer()
    if not chunks:
        return _with_generation_meta(answerer.generate_answer([], query=question))
    # Use the augmented query for the local answerer too -- otherwise short
    # follow-ups like "nó làm đc gì" have no content tokens and the relevance
    # gate trips before the LLM ever sees the history.
    local_answer = answerer.generate_answer(chunks, query=retrieval_query)
    llm_answer = maybe_generate_llm_answer(question, chunks, local_answer, history=history)
    return llm_answer or _with_generation_meta(local_answer)


def _timeline_status() -> str:
    timeline = _load_timeline_json(TIMELINE_PATH)
    if not timeline:
        return "Not built"
    total = timeline.get("metadata", {}).get("timeline_events")
    if isinstance(total, int):
        return f"{total} events"
    return "Built"
