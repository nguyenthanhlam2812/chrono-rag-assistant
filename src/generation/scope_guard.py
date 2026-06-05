"""Scope classification for the ChronoRAG chat layer.

The research corpus is intentionally narrow, but the demo chatbot also needs
to handle project-facing questions such as "write a test plan" or "what should
the assistant do when retrieval confidence is low".  This module separates
three cases before retrieval:

* in_scope: ChronoRAG/RAG/agent/KD/evaluation/project design questions.
* borderline: adjacent questions that need a clarification or narrowed answer.
* out_of_scope: unrelated, unsafe, or explicitly scope-breaking requests.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.utils.text import normalize_vn


SCOPE_IN = "in_scope"
SCOPE_OUT = "out_of_scope"
SCOPE_BORDERLINE = "borderline"


@dataclass(frozen=True)
class ScopeDecision:
    label: str
    confidence: float
    reason: str


_PROJECT_TERMS = (
    "chrono rag",
    "chronorag",
    "chrono-rag",
    "chrono-rag-assistant",
    "rag",
    "retrieval",
    "retrieve",
    "retriever",
    "reranker",
    "bm25",
    "vector",
    "embedding",
    "citation",
    "source attribution",
    "grounding",
    "hallucination",
    "faithfulness",
    "answer relevance",
    "source freshness",
    "temporal",
    "timeline",
    "metadata",
    "schema",
    "corpus",
    "document ingestion",
    "ingestion",
    "chunk",
    "query rewriting",
    "prompt template",
    "langchain",
    "llm",
    "lm studio",
    "ai agent",
    "agent",
    "tool routing",
    "tool-call",
    "tool call",
    "agent workflow",
    "safe tool fallback",
    "retriever-agent",
    "reranker service",
    "action confirmation policy",
    "agent memory policy",
    "agent observability",
    "handoff",
    "reflection",
    "memory retrieval",
    "conversation memory retrieval",
    "multi-step",
    "noisy labels",
    "sequence-level distillation",
    "knowledge distillation",
    "distillation",
    "teacher",
    "student",
    "teacher-student",
    "kd",
    "distilbert",
    "tinybert",
    "mobilebert",
    "minilm",
    "evaluation",
    "metric",
    "metrics",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "confusion matrix",
    "regression test",
    "test case",
    "acceptance criteria",
    "quality gate",
    "guardrail",
    "prompt injection",
)

_PROJECT_ACTION_TERMS = (
    "thiet ke",
    "tao tieu chi",
    "de xuat",
    "giai thich",
    "kiem tra logic",
    "nen xu ly",
    "nen tra loi",
    "should the assistant",
    "how should the assistant",
    "how should the system",
    "when should the assistant",
    "when should chronorag",
    "ask clarification",
    "ask for clarification",
    "ask a clarification",
    "ask instead of answering",
    "abstain instead of answering",
    "create a regression test",
    "write an expected behavior",
    "expected behavior when the user asks",
    "expected behavior when user asks",
    "user asks",
    "should answer",
    "should refuse",
    "should redirect",
    "what metrics should be used",
    "design an agent workflow",
    "safe tool fallback",
    "agent memory policy",
    "conversation memory retrieval",
    "action confirmation policy",
    "agent observability",
    "explain how to test",
    "for chrono",
    "trong repo",
    "trong pipeline",
)

_BORDERLINE_TERMS = (
    "borderline",
    "ask for clarification",
    "hoi ro",
    "hoi ro them",
    "can hoi",
    "can hoi lai",
    "phan loai scope",
    "classify this request",
    "assistant nen xu ly la in-scope hay can hoi lai",
    "should the assistant answer or ask",
    "agent nen tra loi luon hay hoi ro",
)

_UNSAFE_TERMS = (
    "lay api key nguoi khac",
    "steal api key",
    "leak api key",
    "hack",
    "malware",
    "phishing",
    "bypass",
    "jailbreak",
)

_OFF_TOPIC_TERMS = (
    "mua laptop",
    "dat do an",
    "dia diem an choi",
    "mon an",
    "an gi",
    "nau pho",
    "visa du lich",
    "visa",
    "lich trinh",
    "da lat",
    "du lich",
    "tour",
    "tour nhat ban",
    "khach san",
    "ve may bay",
    "vay ngan hang",
    "stock price",
    "gia co phieu",
    "co phieu",
    "ty gia",
    "ty gia hom nay",
    "dau tu",
    "bitcoin",
    "bao hiem",
    "xet nghiem mau",
    "chan doan benh",
    "thuoc khang sinh",
    "dau nguc",
    "dau bung",
    "giam can",
    "medical",
    "iphone",
    "giay chay bo",
    "may loc khong khi",
    "mua domain",
    "nhac tiktok",
    "anime",
    "movie",
    "phim",
    "lich chieu phim",
    "review game",
    "mount everest",
    "everest",
    "how tall",
    "height of",
    "bong da",
    "politics",
    "bau cu",
    "tong thong",
    "chinh phu",
    "tin chinh tri",
    "xung dot quoc te",
    "luat moi",
    "weather",
    "thoi tiet",
    "nhan tin xin loi",
    "personal life",
    "chia tay",
    "tan crush",
    "caption facebook",
    "cai nhau",
    "mua qua",
    "qua sinh nhat",
    "celebrity gossip",
    "drama k-pop",
    "k-pop",
    "lua dao qua dien thoai",
    "sql join homework",
    "lam bai sql",
    "sql homework",
    "giai phuong trinh",
    "giai bai xac suat",
    "bai xac suat",
    "bai van nghi luan",
    "dich doan van",
    "excel macro",
    "arduino",
    "unity game",
    "css responsive",
    "java swing",
    "lich su viet nam",
)

_SCOPE_BREAKERS = (
    "ignore the repo scope",
    "do not mention rag",
    "khong can lien quan den rag",
    "pretend this is about ai",
    "dung tu choi",
    "dont refuse",
    "don't refuse",
)


def classify_chat_scope(question: str) -> ScopeDecision:
    """Classify a user query into in/out/borderline project scope."""

    raw = (question or "").strip()
    if not raw:
        return ScopeDecision(SCOPE_OUT, 0.8, "empty")

    q = normalize_vn(raw)
    q_en = raw.lower()
    combined = f"{q} {q_en}"

    if any(term in combined for term in _BORDERLINE_TERMS):
        return ScopeDecision(SCOPE_BORDERLINE, 0.92, "borderline_phrase")

    has_project = _has_any(combined, _PROJECT_TERMS) or _has_any(combined, _PROJECT_ACTION_TERMS)
    has_scope_breaker = _has_any(combined, _SCOPE_BREAKERS)
    has_off_topic = _has_any(combined, _OFF_TOPIC_TERMS)
    has_unsafe = _has_any(combined, _UNSAFE_TERMS)

    if has_unsafe and not _has_any(combined, ("guardrail", "safety", "safe", "refusal", "prompt injection")):
        return ScopeDecision(SCOPE_OUT, 0.98, "unsafe_unrelated")

    if has_scope_breaker and (has_off_topic or not has_project):
        return ScopeDecision(SCOPE_OUT, 0.98, "scope_breaking_instruction")

    if has_off_topic and not has_project:
        return ScopeDecision(SCOPE_OUT, 0.94, "off_topic")

    if has_project:
        return ScopeDecision(SCOPE_IN, 0.86, "project_or_domain_terms")

    # Short greetings and common terms are handled elsewhere. Unknown prose
    # falls through to retrieval first; the answerer can still abstain.
    return ScopeDecision(SCOPE_IN, 0.45, "unknown_try_retrieval")


def is_project_guidance_question(question: str) -> bool:
    """Return True for in-scope questions that are not necessarily paper RAG."""

    raw = (question or "").strip()
    if not raw:
        return False
    q = normalize_vn(raw)
    combined = f"{q} {raw.lower()}"
    return _has_any(combined, _PROJECT_ACTION_TERMS) or _has_any(
        combined,
        (
            "acceptance criteria",
            "test plan",
            "test case",
            "regression test",
            "quality gate",
            "ask clarification",
            "ask for clarification",
            "ask a clarification",
            "ask instead of answering",
            "abstain instead of answering",
            "answer policy",
            "expected behavior when the user asks",
            "expected behavior when user asks",
            "user asks",
            "should answer",
            "should refuse",
            "should redirect",
            "when should chronorag",
            "implementation guidance",
            "debug repo",
            "config",
            "metadata schema",
            "source freshness",
            "retrieval confidence",
            "no relevant document",
            "without hallucinating",
            "should trust",
            "nen chon",
            "nen luu",
            "nen dung metadata",
        ),
    )


def out_of_scope_answer() -> str:
    return (
        "Không tìm thấy thông tin đủ liên quan trong corpus hiện tại. "
        "Vui lòng hỏi về RAG, AI Agent hoặc Knowledge Distillation."
    )


def borderline_answer() -> str:
    return (
        "Câu này nằm sát ranh giới scope. Đại ka nên làm rõ câu hỏi đang muốn "
        "xử lý phần nào của ChronoRAG: RAG/retrieval, AI Agent workflow, "
        "Knowledge Distillation, evaluation hay implementation. Nếu chỉ là vấn "
        "đề ngoài dự án, hệ thống nên từ chối hoặc chuyển hướng; nếu có liên "
        "quan tới pipeline ChronoRAG thì trả lời phần đó và nêu rõ giới hạn."
    )


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_term_matches(text, term) for term in terms)


def _term_matches(text: str, term: str) -> bool:
    return bool(re.search(rf"(?:^|[^a-z0-9]){re.escape(term)}(?:[^a-z0-9]|$)", text))
