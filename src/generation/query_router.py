"""Fallback query routing for the ChronoRAG chat layer.

The primary chat path is LLM-first hybrid RAG. These deterministic routes are
kept as a safety net for cases where the LLM is disabled, times out, or needs a
small high-precision answer such as "who am I?" or "what can I ask?".

LangChain support for this old router is intentionally optional and disabled
unless ``CHAT_ROUTER=langchain`` is set explicitly. Normal demo configuration
should use ``CHAT_ANSWERER=requests`` for the answer-generation layer instead.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.utils.text import normalize_vn


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_PATH = PROJECT_ROOT / "data" / "eval" / "ml_baseline_metrics.json"

INTENT_CAPABILITY = "capability"
INTENT_EVALUATION = "evaluation"
INTENT_SCOPE = "scope"
INTENT_LLM_STATUS = "llm_status"
INTENT_USER_IDENTITY = "user_identity"
INTENT_UNCLEAR_FOLLOWUP = "unclear_followup"
INTENT_USER_CORRECTION = "user_correction"
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


def maybe_answer_direct(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a direct non-retrieval answer when the query is about the app.

    These answers deliberately carry no citations because they are not claims
    from the research corpus; they describe project capability or evaluation
    artifacts.
    """
    route = route_query(question)
    if route.intent == INTENT_NONE:
        return None

    q = normalize_vn(question or "")
    if route.intent == INTENT_USER_IDENTITY:
        return _direct_payload(_user_identity_answer(), route.intent)
    if route.intent == INTENT_UNCLEAR_FOLLOWUP:
        return _direct_payload(_unclear_followup_answer(history), route.intent)
    if route.intent == INTENT_USER_CORRECTION:
        return _direct_payload(_user_correction_answer(), route.intent)

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

    return _direct_payload("\n\n".join(parts), route.intent)


def _route_with_rules(q: str) -> QueryRoute:
    if _has_user_identity_signal(q):
        return QueryRoute(INTENT_USER_IDENTITY, 0.98, "user_identity")
    if _has_user_correction_signal(q):
        return QueryRoute(INTENT_USER_CORRECTION, 0.95, "user_correction")
    if _has_unclear_followup_signal(q):
        return QueryRoute(INTENT_UNCLEAR_FOLLOWUP, 0.9, "unclear_followup")

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


def _has_user_identity_signal(q: str) -> bool:
    patterns = (
        r"\b(?:may|ban|bot|m)\s+(?:co\s+)?biet\s+(?:tao|toi|t|minh)\s+la\s+ai\s*(?:khong|ko)?\b",
        r"\b(?:may|ban|bot|m)\s+(?:co\s+)?nho\s+(?:tao|toi|t|minh)\s+la\s+ai\s*(?:khong|ko)?\b",
        r"\b(?:may|ban|bot|m)\s+co\s+biet\s+(?:gi\s+)?ve\s+(?:tao|toi|t|minh)\b",
        r"\b(?:tao|toi|t|minh)\s+hoi\s+(?:ve\s+)?(?:tao|toi|t|minh)\s*(?:ay|do)?\b",
        r"\bwho\s+am\s+i\b",
        r"\bdo\s+you\s+know\s+who\s+i\s+am\b",
        r"\bdo\s+you\s+remember\s+me\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)


def _has_user_correction_signal(q: str) -> bool:
    patterns = (
        r"\b(?:tao|toi|t|minh)\s+(?:khong|ko|k)\s+co\s+hoi\s+(?:may|ban|bot|m)\b",
        r"\b(?:tao|toi|t|minh)\s+(?:khong|ko|k)\s+hoi\s+(?:may|ban|bot|m)\b",
        r"\b(?:khong|ko|k)\s+phai\s+y\s+(?:do|day|ay)\b",
        r"\b(?:sai|sai\s+roi|nham|nham\s+roi)\b",
        r"\bnot\s+what\s+i\s+asked\b",
        r"\byou\s+misunderstood\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)


def _has_unclear_followup_signal(q: str) -> bool:
    compact = re.sub(r"\s+", " ", q).strip()
    patterns = (
        r"^(?:tao|toi|t|minh)\s+hoi\s+(?:ay|do|cai\s+do|cau\s+do|cau\s+ay)$",
        r"^y\s+(?:tao|toi|t|minh)\s+la\s+(?:cau\s+)?(?:do|ay)$",
        r"^(?:cau\s+)?(?:do|ay)\s+(?:thi\s+)?sao$",
        r"^(?:tra\s+loi\s+)?(?:cai\s+)?(?:do|ay)\s+di$",
        r"^(?:that|it|that one)\s+(?:one\s+)?(?:then|please)?$",
    )
    return any(re.search(pattern, compact) for pattern in patterns)


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


def _user_identity_answer() -> str:
    return (
        "Mình không biết danh tính thật của đại ka nếu đại ka chưa nói rõ trong phiên chat. "
        "Trong demo này mình chỉ thấy nội dung đại ka nhập, topic đang chọn và corpus ChronoRAG; "
        "mình không có hồ sơ cá nhân hay bộ nhớ lâu dài về người dùng. Nếu UI đang hiện tên "
        "`Tlam` thì đó là nhãn demo frontend, không phải thông tin mình tự suy luận."
    )


def _user_correction_answer() -> str:
    return (
        "Đúng rồi đại ka, câu vừa rồi mình hiểu chưa đúng ý. Đại ka gửi lại câu hỏi cụ thể một câu thôi nhé; "
        "nếu hỏi về RAG, AI Agent hoặc Knowledge Distillation thì mình sẽ bám corpus và gắn citation."
    )


def _unclear_followup_answer(history: Optional[List[Dict[str, str]]]) -> str:
    last_user = _last_user_question(history)
    if last_user and _has_user_identity_signal(normalize_vn(last_user)):
        return "Nếu đại ka đang hỏi câu trước thì câu trả lời là: " + _user_identity_answer()
    if last_user:
        return (
            f"Đại ka đang nhắc tới câu trước: \"{last_user[:120]}\" đúng không? "
            "Câu hiện tại hơi thiếu ngữ cảnh, đại ka hỏi lại rõ hơn một chút để mình trả lời đúng nguồn."
        )
    return (
        "Câu này hơi thiếu ngữ cảnh nên mình chưa biết đại ka đang chỉ tới ý nào. "
        "Đại ka hỏi lại rõ hơn một chút nhé, ví dụ: `RAG ra đời năm nào?` hoặc `so sánh DistilBERT và TinyBERT`."
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


def _direct_payload(answer: str, intent: str) -> Dict[str, Any]:
    return {
        "answer": answer,
        "citations": [],
        "mode": "router",
        "provider": "router",
        "model": f"query-router:{intent}",
    }


def _last_user_question(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    for turn in reversed(history):
        if turn.get("role") == "user":
            content = str(turn.get("content", "")).strip()
            if content:
                return content
    return ""


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

    This is off by default and kept only for backwards-compatible experiments.
    Tests and demos should not depend on it.
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
                "evaluation, scope, llm_status, user_identity, unclear_followup, "
                "user_correction, none. Use none for normal corpus questions about RAG, AI Agents, "
                "or Knowledge Distillation.",
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
    if intent not in {
        INTENT_CAPABILITY,
        INTENT_EVALUATION,
        INTENT_SCOPE,
        INTENT_LLM_STATUS,
        INTENT_USER_IDENTITY,
        INTENT_UNCLEAR_FOLLOWUP,
        INTENT_USER_CORRECTION,
        INTENT_NONE,
    }:
        return None
    if confidence < 0.65:
        return None
    return QueryRoute(intent, confidence, "langchain")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
