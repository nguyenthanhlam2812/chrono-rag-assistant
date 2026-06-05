"""Query routing for the ChronoRAG chat layer.

The RAG retriever is good at corpus questions, but it should not be asked to
answer product/help/metrics questions such as "what can I ask?" or "how
accurate is the model?". This module catches those cases before retrieval.

LangChain support is intentionally optional. When ``CHAT_ROUTER=langchain`` and
``langchain-openai`` is installed, an LLM can classify ambiguous intent through
an OpenAI-compatible endpoint such as LM Studio. Deterministic rules still run
first so the demo remains stable without LangChain.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.utils.text import normalize_vn


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_PATH = PROJECT_ROOT / "data" / "eval" / "ml_baseline_metrics.json"

INTENT_CAPABILITY = "capability"
INTENT_EVALUATION = "evaluation"
INTENT_SCOPE = "scope"
INTENT_LLM_STATUS = "llm_status"
INTENT_NONE = "none"


@dataclass(frozen=True)
class QueryRoute:
    intent: str
    confidence: float
    reason: str = ""


def route_query(question: str) -> QueryRoute:
    """Classify high-level chat intent before corpus retrieval.

    Returns ``INTENT_NONE`` for normal RAG questions.
    """
    q = normalize_vn(question or "")
    if not q:
        return QueryRoute(INTENT_NONE, 0.0, "empty")

    rule_route = _route_with_rules(q)
    if rule_route.intent != INTENT_NONE:
        return rule_route

    llm_route = _route_with_langchain(question)
    if llm_route and llm_route.intent != INTENT_NONE:
        return llm_route

    return QueryRoute(INTENT_NONE, 0.0, "fallthrough")


def maybe_answer_direct(question: str) -> Optional[Dict[str, Any]]:
    """Return a direct non-retrieval answer when the query is about the app.

    These answers deliberately carry no citations because they are not claims
    from the research corpus; they describe project capability or evaluation
    artifacts.
    """
    route = route_query(question)
    if route.intent == INTENT_NONE:
        return None

    q = normalize_vn(question or "")
    wants_capability = route.intent == INTENT_CAPABILITY or _has_capability_signal(q)
    wants_eval = route.intent == INTENT_EVALUATION or _has_evaluation_signal(q)
    wants_scope = route.intent == INTENT_SCOPE or _has_scope_signal(q)
    wants_llm = route.intent == INTENT_LLM_STATUS

    parts = []
    if wants_capability or wants_scope:
        parts.append(_capability_answer())
    if wants_eval:
        parts.append(_evaluation_answer())
    if wants_llm:
        parts.append(_llm_status_answer())

    if not parts:
        return None

    return {
        "answer": "\n\n".join(parts),
        "citations": [],
        "mode": "local",
        "provider": "router",
        "model": f"query-router:{route.intent}",
    }


def _route_with_rules(q: str) -> QueryRoute:
    has_capability = _has_capability_signal(q)
    has_eval = _has_evaluation_signal(q)

    if has_capability and has_eval:
        return QueryRoute(INTENT_CAPABILITY, 0.98, "capability+evaluation")
    if has_capability:
        return QueryRoute(INTENT_CAPABILITY, 0.96, "capability")
    if has_eval:
        return QueryRoute(INTENT_EVALUATION, 0.96, "evaluation")
    if _has_scope_signal(q):
        return QueryRoute(INTENT_SCOPE, 0.92, "scope")
    if _has_llm_signal(q):
        return QueryRoute(INTENT_LLM_STATUS, 0.9, "llm_status")
    return QueryRoute(INTENT_NONE, 0.0, "no_rule")


def _has_capability_signal(q: str) -> bool:
    patterns = (
        r"\b(?:tao|toi|t|minh)\s+co\s+the\s+hoi\b",
        r"\bhoi\s+(?:may|ban|bot|m)\s+(?:duoc\s+)?(?:nhung\s+)?(?:cai\s+)?gi\b",
        r"\bhoi\s+(?:duoc\s+)?(?:nhung\s+)?(?:cai\s+)?gi\b",
        r"\b(?:may|ban|bot|m)\s+(?:co\s+the\s+)?(?:lam|giup|tra\s+loi)\s+(?:duoc\s+)?gi\b",
        r"\b(?:may|ban|bot|m)\s+biet\s+(?:nhung\s+)?gi\b",
        r"\bwhat\s+can\s+(?:i\s+ask|you\s+do)\b",
        r"\bwhat\s+questions\s+can\s+i\s+ask\b",
        r"\bcapabilit(?:y|ies)\b",
        r"\bhelp\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)


def _has_evaluation_signal(q: str) -> bool:
    if _is_metric_definition_question(q):
        return False
    patterns = (
        "do chinh xac",
        "chinh xac bao nhieu",
        "dung bao nhieu",
        "model dung",
        "ket qua train",
        "ket qua danh gia",
        "diem model",
        "accuracy",
        "macro f1",
        "f1",
        "precision",
        "recall",
        "confusion matrix",
        "evaluation",
        "metric",
        "metrics",
    )
    return any(pattern in q for pattern in patterns)


def _is_metric_definition_question(q: str) -> bool:
    metric_terms = r"(?:precision|recall|f1|f1 score|accuracy|metric|confusion matrix)"
    definition_cues = (
        rf"\b{metric_terms}\s+(?:la\s+)?gi\b",
        rf"\bwhat\s+is\s+{metric_terms}\b",
        rf"\bdefine\s+{metric_terms}\b",
        rf"\bexplain\s+{metric_terms}\b",
    )
    return any(re.search(pattern, q) for pattern in definition_cues)


def _has_scope_signal(q: str) -> bool:
    patterns = (
        "data nay tap trung",
        "corpus nay tap trung",
        "du lieu nay tap trung",
        "dang tap trung ve",
        "co nhung chu de nao",
        "gom nhung chu de nao",
        "pham vi corpus",
        "scope",
        "topics",
    )
    return any(pattern in q for pattern in patterns)


def _has_llm_signal(q: str) -> bool:
    patterns = (
        "dang dung llm nao",
        "dang dung model nao",
        "model chat",
        "lm studio",
        "openai",
        "openrouter",
        "llm dang bat",
        "llm bat chua",
    )
    return any(pattern in q for pattern in patterns)


def _capability_answer() -> str:
    return (
        "Đại ka có thể hỏi ChronoRAG trong phạm vi 3 chủ đề MVP: RAG, AI Agent, "
        "và Knowledge Distillation. Các kiểu câu hỏi hợp nhất là: định nghĩa "
        "khái niệm, mốc thời gian/paper nào đề xuất, so sánh phương pháp, "
        "benchmark/kết quả, nguồn citation, và timeline theo từng topic. "
        "Nếu hỏi ngoài corpus, hệ thống sẽ từ chối thay vì bịa."
    )


def _evaluation_answer() -> str:
    best = _best_metrics()
    if not best:
        return (
            "Chưa tìm thấy file metrics thật. Chạy `python scripts/04_train_ml_classifier.py` "
            "để sinh `data/eval/ml_baseline_metrics.json`."
        )

    return (
        f"Metrics hiện tại trên test split: model tốt nhất cho event detection là "
        f"{best['model']}. Binary F1 = {best['binary_f1']}, Binary Accuracy = "
        f"{best['binary_accuracy']}, Event-type Macro-F1 = {best['type_f1']}, "
        f"Event-type Accuracy = {best['type_accuracy']}. Đây là độ đo cho ML "
        "event classifier/timeline, không phải cam kết chatbot đúng tuyệt đối."
    )


def _llm_status_answer() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower() or "mock"
    if provider == "lmstudio":
        model = os.getenv("LMSTUDIO_MODEL", "local-model")
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        return (
            f"Chat đang cấu hình LM Studio với model `{model}` tại `{base_url}`. "
            "Nếu LM Studio server hoặc model chưa bật thì hệ thống sẽ rơi về Local RAG."
        )
    if provider in {"openai", "openrouter"}:
        model = os.getenv("OPENAI_MODEL" if provider == "openai" else "OPENROUTER_MODEL", "")
        return f"Chat đang cấu hình provider `{provider}` với model `{model or 'default'}`."
    return "Chat đang chạy Local RAG/template answerer, chưa bật LLM provider."


def _best_metrics() -> Optional[Dict[str, str]]:
    if not METRICS_PATH.exists():
        return None
    try:
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    models = data.get("models", {})
    best_name = None
    best_row: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for name, row in models.items():
        binary_test = row.get("binary", {}).get("test", {})
        score = float(binary_test.get("f1_macro", 0.0) or 0.0)
        if score > best_score:
            best_name = name
            best_row = row
            best_score = score
    if not best_name or not best_row:
        return None

    binary_test = best_row.get("binary", {}).get("test", {})
    type_test = best_row.get("event_type", {}).get("test", {})
    return {
        "model": best_name,
        "binary_f1": _pct(binary_test.get("f1_macro")),
        "binary_accuracy": _pct(binary_test.get("accuracy")),
        "type_f1": _pct(type_test.get("f1_macro")),
        "type_accuracy": _pct(type_test.get("accuracy")),
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _route_with_langchain(question: str) -> Optional[QueryRoute]:
    """Optional LangChain classifier for ambiguous router cases.

    This is off by default. It is useful when ``CHAT_ROUTER=langchain`` and
    LM Studio/OpenAI-compatible chat is available, but tests and demos should
    not depend on it.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if os.getenv("CHAT_ROUTER", "rules").strip().lower() != "langchain":
        return None

    try:
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider == "lmstudio":
        api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        model = os.getenv("LMSTUDIO_MODEL", "local-model")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    else:
        return None
    if not api_key.strip():
        return None

    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user's ChronoRAG chat intent. Return only JSON "
                "with keys intent and confidence. Valid intents: capability, "
                "evaluation, scope, llm_status, none. Use none for normal "
                "corpus questions about RAG, AI Agents, or Knowledge Distillation.",
            ),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0,
        max_tokens=80,
        timeout=20,
    )

    try:
        data = (prompt | llm | parser).invoke({"question": question})
    except Exception:
        return None

    intent = str(data.get("intent", "")).strip().lower()
    confidence = _safe_float(data.get("confidence"), 0.0)
    if intent not in {INTENT_CAPABILITY, INTENT_EVALUATION, INTENT_SCOPE, INTENT_LLM_STATUS, INTENT_NONE}:
        return None
    if confidence < 0.65:
        return None
    return QueryRoute(intent, confidence, "langchain")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
