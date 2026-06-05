from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from src.generation.template_answerer import NO_ANSWER_MESSAGE


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TIMEOUT_SECONDS = 20


def llm_runtime_status() -> Dict[str, Any]:
    """Return sanitized LLM status for API/UI display without exposing keys."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider in {"", "mock", "local", "template"}:
        return {
            "mode": "local",
            "provider": "mock",
            "model": "template-answerer",
            "configured": False,
        }
    if provider == "openai":
        configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
        return {
            "mode": "llm" if configured else "local",
            "provider": "openai",
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "configured": configured,
        }
    if provider == "openrouter":
        configured = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
        return {
            "mode": "llm" if configured else "local",
            "provider": "openrouter",
            "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "configured": configured,
        }
    if provider == "lmstudio":
        return {
            "mode": "llm",
            "provider": "lmstudio",
            "model": os.getenv("LMSTUDIO_MODEL", "local-model"),
            "configured": True,
        }
    return {
        "mode": "local",
        "provider": provider or "unknown",
        "model": "template-answerer",
        "configured": False,
    }


def maybe_generate_llm_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    local_answer: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Optionally synthesize a nicer answer from retrieved context.

    This is deliberately conservative: it only runs when LLM_PROVIDER is set
    and the local answerer already found relevant cited context. If the LLM is
    not configured, fails, or refuses to cite, the caller keeps the local answer.

    ``history`` is an optional list of recent ``{"role", "content"}`` chat
    turns (most recent last). It lets the LLM resolve follow-up references
    like "nó làm được gì" against the previous turn. We cap it at the last
    few turns to keep the prompt small and predictable.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider in {"", "mock", "local", "template"}:
        return None
    if not chunks or not local_answer.get("citations"):
        return None
    if str(local_answer.get("answer", "")).strip() == NO_ANSWER_MESSAGE:
        return None

    context = _format_context(chunks)
    if not context:
        return None

    trimmed_history = _trim_history(history)

    model_used = ""
    if provider == "openai":
        model_used = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        answer = _call_chat_completion(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=model_used,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            question=question,
            context=context,
            history=trimmed_history,
        )
    elif provider == "openrouter":
        model_used = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        answer = _call_chat_completion(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            model=model_used,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            question=question,
            context=context,
            history=trimmed_history,
            extra_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "ChronoRAG"),
            },
        )
    elif provider == "lmstudio":
        model_used = os.getenv("LMSTUDIO_MODEL", "local-model")
        answer = _call_chat_completion(
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            model=model_used,
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            question=question,
            context=context,
            history=trimmed_history,
            max_tokens=900,
            timeout_seconds=90,
            disable_thinking=True,
        )
    else:
        return None

    if not answer:
        return None
    if answer.strip() == NO_ANSWER_MESSAGE:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "citations": [],
            "mode": "llm",
            "provider": provider,
            "model": model_used,
        }

    citations = _citations_from_chunks(chunks)
    if citations and not any(f"[{cite['doc_id']}]" in answer for cite in citations):
        answer = f"{answer.rstrip()} [{citations[0]['doc_id']}]"

    return {
        "answer": answer.strip(),
        "citations": citations,
        "mode": "llm",
        "provider": provider,
        "model": model_used,
    }


def maybe_generate_project_llm_answer(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate an uncited project-level answer for valid ChronoRAG questions.

    This is the "hybrid" part of Hybrid RAG + LLM: when a question is clearly
    about ChronoRAG design/testing/guardrails but the paper corpus does not
    contain enough evidence, the chatbot can still answer as project guidance
    without faking citations.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider in {"", "mock", "local", "template"}:
        return None

    trimmed_history = _trim_history(history)
    model_used = ""
    if provider == "openai":
        model_used = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        answer = _call_project_completion(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=model_used,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            question=question,
            history=trimmed_history,
        )
    elif provider == "openrouter":
        model_used = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        answer = _call_project_completion(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            model=model_used,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            question=question,
            history=trimmed_history,
            extra_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "ChronoRAG"),
            },
        )
    elif provider == "lmstudio":
        model_used = os.getenv("LMSTUDIO_MODEL", "local-model")
        answer = _call_project_completion(
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            model=model_used,
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            question=question,
            history=trimmed_history,
            max_tokens=760,
            timeout_seconds=90,
            disable_thinking=True,
        )
    else:
        return None

    if not answer:
        return None
    return {
        "answer": answer.strip(),
        "citations": [],
        "mode": "llm",
        "provider": provider,
        "model": model_used,
    }


_SYSTEM_PROMPT = (
    "You are ChronoRAG, a timeline-aware AI/ML research assistant for three "
    "topics: RAG, AI Agents, and Knowledge Distillation.\n\n"
    "Rules:\n"
    "1. Answer ONLY from the supplied Context. Never invent facts, dates, or "
    "papers that the Context does not explicitly state.\n"
    "2. You may reason across multiple context snippets: compare methods, infer "
    "differences, summarize trade-offs, and connect timeline facts, but every "
    "factual claim must be grounded in the Context and cited.\n"
    "3. If the question asks for a specific year, date, author, or numeric "
    "comparison that the Context does not directly state, do NOT guess. "
    f"Reply with exactly this and nothing else: {NO_ANSWER_MESSAGE}\n"
    "4. If the Context is about a different topic than what the user asked "
    f"about, reply with exactly: {NO_ANSWER_MESSAGE}\n"
    "5. Use the prior chat turns ONLY to resolve pronoun references "
    "(\"nó\", \"it\", \"that one\"). Do not let them invent new facts.\n"
    "6. Keep answers concise (2-5 sentences). For comparison questions, use "
    "short bullets when clearer. Use inline citations like "
    "[rag_001] right after factual claims.\n"
    "7. Language is mandatory: if the user asks in Vietnamese, answer in "
    "Vietnamese. Keep only paper titles, model names, and technical terms in "
    "English. Do not switch the whole answer to English just because the "
    "Context is English."
)

_PROJECT_SYSTEM_PROMPT = (
    "You are ChronoRAG's project-level AI assistant. ChronoRAG is an AI/NLP "
    "course project and demo for a timeline-aware research assistant. It covers "
    "three MVP domains: RAG, AI Agents, and Knowledge Distillation. The system "
    "has ingestion, preprocessing, BM25/vector retrieval, event detection, "
    "timeline building, evaluation, and a React/FastAPI chat UI.\n\n"
    "Rules:\n"
    "1. Answer project-scope questions about implementation, testing, evaluation, "
    "guardrails, source freshness, metadata, retrieval confidence, and demo "
    "behavior.\n"
    "2. Do not invent paper citations. If you are giving project guidance rather "
    "than a corpus-grounded claim, say it as guidance and do not add fake doc ids.\n"
    "3. If the user asks for unrelated shopping, medical, travel, finance, "
    "homework, entertainment, or unsafe instructions, say it is outside scope and "
    "redirect to RAG/AI Agent/KD.\n"
    "4. Be practical and concise. Prefer short bullets or a compact checklist.\n"
    "5. If the user asks in Vietnamese, answer in Vietnamese. Keep technical terms "
    "like RAG, BM25, FAISS, LangChain, KD, F1 in English."
)


def _call_chat_completion(
    *,
    api_key: str,
    model: str,
    base_url: str,
    question: str,
    context: str,
    history: Optional[List[Dict[str, str]]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    max_tokens: int = 520,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    disable_thinking: bool = False,
) -> Optional[str]:
    if not api_key.strip():
        return None

    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:600]})
    language_instruction = (
        "Answer language: Vietnamese. TRẢ LỜI BẰNG TIẾNG VIỆT, chỉ giữ thuật ngữ kỹ thuật/citation bằng tiếng Anh."
        if _looks_vietnamese(question)
        else "Answer language: English."
    )
    user_content = f"{language_instruction}\n\nQuestion:\n{question}\n\nContext:\n{context}"
    if disable_thinking:
        user_content = "/no_think\n" + user_content
    messages.append({
        "role": "user",
        "content": user_content,
    })
    return _send_chat_completion(
        api_key=api_key,
        model=model,
        base_url=base_url,
        messages=messages,
        extra_headers=extra_headers,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def _call_project_completion(
    *,
    api_key: str,
    model: str,
    base_url: str,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    max_tokens: int = 620,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    disable_thinking: bool = False,
) -> Optional[str]:
    if not api_key.strip():
        return None
    messages: List[Dict[str, str]] = [{"role": "system", "content": _PROJECT_SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:500]})
    language_instruction = (
        "Answer language: Vietnamese. TRẢ LỜI BẰNG TIẾNG VIỆT."
        if _looks_vietnamese(question)
        else "Answer language: English."
    )
    user_content = f"{language_instruction}\n\nQuestion:\n{question}"
    if disable_thinking:
        user_content = "/no_think\n" + user_content
    messages.append({"role": "user", "content": user_content})
    return _send_chat_completion(
        api_key=api_key,
        model=model,
        base_url=base_url,
        messages=messages,
        extra_headers=extra_headers,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def _send_chat_completion(
    *,
    api_key: str,
    model: str,
    base_url: str,
    messages: List[Dict[str, str]],
    extra_headers: Optional[Dict[str, str]] = None,
    max_tokens: int = 520,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    if os.getenv("CHAT_ANSWERER", "langchain").strip().lower() == "langchain":
        answer = _send_langchain_chat_completion(
            api_key=api_key,
            model=model,
            base_url=base_url,
            messages=messages,
            extra_headers=extra_headers,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        if answer:
            return answer

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        **(extra_headers or {}),
    }
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return _clean_llm_output(str(data["choices"][0]["message"]["content"]).strip())
    except Exception:
        return None


def _send_langchain_chat_completion(
    *,
    api_key: str,
    model: str,
    base_url: str,
    messages: List[Dict[str, str]],
    extra_headers: Optional[Dict[str, str]] = None,
    max_tokens: int = 520,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    lc_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    try:
        llm = ChatOpenAI(
            api_key=api_key.strip(),
            base_url=base_url,
            model=model,
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=timeout_seconds,
            default_headers=extra_headers or None,
        )
        result = llm.invoke(lc_messages)
        return _clean_llm_output(str(result.content).strip())
    except Exception:
        return None


def _clean_llm_output(text: str) -> str:
    replacements = {
        "检索": " truy xuất ",
        "生成": " sinh ",
        "模型": " mô hình ",
        "知识": " tri thức ",
        "蒸馏": " distillation ",
    }
    replacements["vector밀 độ"] = "chỉ mục vector dày đặc"
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _trim_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Keep at most the last 3 user/assistant pairs (~6 turns). Strip the
    canned welcome and abstain messages so the LLM doesn't echo them."""
    if not history:
        return []
    keep: List[Dict[str, str]] = []
    for turn in history[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if content == NO_ANSWER_MESSAGE:
            continue
        keep.append({"role": role, "content": content})
    return keep


def _looks_vietnamese(text: str) -> bool:
    value = text.lower()
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", value):
        return True
    return bool(re.search(r"\b(là|gì|so sánh|khác|năm|bao nhiêu|vì sao|như thế nào|điểm nào)\b", value))


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    seen = set()
    for chunk in chunks[:4]:
        doc_id = str(chunk.get("doc_id", "") or "").strip()
        if not doc_id:
            continue
        seen.add(doc_id)
        title = str(chunk.get("title", "") or doc_id).strip()
        text = _clean_context_text(str(chunk.get("text", "") or ""))
        if not text:
            continue
        blocks.append(f"[{doc_id}] {title}\n{text[:1400]}")
    return "\n\n".join(blocks)


def _clean_context_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _citations_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for chunk in chunks[:4]:
        doc_id = str(chunk.get("doc_id", "") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(
            {
                "doc_id": doc_id,
                "title": str(chunk.get("title", "") or doc_id),
                "source_url": str(chunk.get("source_url", "") or ""),
            }
        )
    return out
