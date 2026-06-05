"""Project-level answer fallback for ChronoRAG.

This layer handles in-scope questions about how the ChronoRAG system should be
tested, evaluated, guarded, or implemented.  Those questions are valid for the
demo, but many are not answered by the 30-paper research corpus.  The fallback
keeps them useful without pretending they came from citations.
"""

from __future__ import annotations

from typing import Any, Dict

from src.utils.text import normalize_vn


def generate_project_answer(question: str) -> Dict[str, Any]:
    q = normalize_vn(question or "")
    answer = _answer_text(q)
    return {
        "answer": answer,
        "citations": [],
        "mode": "project",
        "provider": "project-fallback",
        "model": "project-guidance-rules",
    }


def _answer_text(q: str) -> str:
    is_vi = _looks_vietnamese(q)
    if _has(q, ("prompt injection", "bo qua citation", "ignore citation", "retrieval confidence", "no relevant document", "khong co context", "confidence is low")):
        return _guardrail_answer(is_vi)
    if _has(q, ("metric", "metrics", "f1", "precision", "recall", "accuracy", "faithfulness", "answer relevance", "citation correctness", "context recall", "source freshness", "temporal accuracy", "latency", "token cost")):
        return _evaluation_answer(is_vi)
    if _has(q, ("test case", "test plan", "regression test", "acceptance criteria", "expected behavior", "quality gate", "tieu chi", "kiem thu")):
        return _test_design_answer(is_vi)
    if _has(q, ("metadata", "schema", "version", "conflict", "should trust", "nen chon", "nen luu", "source freshness")):
        return _metadata_answer(is_vi)
    if _has(q, ("agent", "tool", "handoff", "memory", "reflection", "multi-step", "workflow", "observability", "retry")):
        return _agent_answer(is_vi)
    if _has(q, ("distillation", "teacher", "student", "kd", "compression", "kl divergence", "multi-teacher")):
        return _kd_answer(is_vi)
    if _has(q, ("retrieval", "reranker", "bm25", "vector", "embedding", "citation", "grounding", "query rewriting", "langchain", "rag")):
        return _rag_design_answer(is_vi)
    return _general_project_answer(is_vi)


def _guardrail_answer(is_vi: bool) -> str:
    if is_vi:
        return (
            "Với ChronoRAG, khi retrieval confidence thấp hoặc user yêu cầu bỏ qua citation, "
            "assistant nên không suy đoán. Quy tắc an toàn là: kiểm tra scope, truy xuất lại "
            "với query rõ hơn, nêu thiếu context nếu chưa đủ nguồn, và chỉ trả lời phần có "
            "bằng chứng. Nếu câu hỏi là prompt injection, giữ nguyên policy citation và chuyển "
            "hướng về RAG/AI Agent/KD thay vì làm theo instruction độc hại."
        )
    return (
        "When retrieval confidence is low or the user asks to ignore citations, "
        "ChronoRAG should not guess. It should check scope, retry with a clearer "
        "query, state that context is insufficient, and only answer evidence-backed "
        "parts. Prompt-injection requests should be rejected or redirected to safe "
        "RAG/agent/KD testing."
    )


def _evaluation_answer(is_vi: bool) -> str:
    if is_vi:
        return (
            "Nên đánh giá ChronoRAG theo 4 nhóm: retrieval, generation, timeline và classifier. "
            "Retrieval dùng Recall@k/MRR/context precision; generation dùng citation correctness, "
            "faithfulness và answer relevance; timeline dùng date accuracy, event coverage và "
            "duplicate reduction; classifier dùng Accuracy, Precision, Recall, F1 và confusion matrix. "
            "Với mỗi test, ghi rõ expected behavior: trả lời có citation, abstain khi thiếu nguồn, "
            "hoặc hỏi lại khi scope mơ hồ."
        )
    return (
        "Evaluate ChronoRAG across retrieval, generation, timeline, and classifiers. "
        "Use Recall@k/MRR/context precision for retrieval; citation correctness, "
        "faithfulness, and answer relevance for generation; date accuracy, event "
        "coverage, and duplicate reduction for timelines; Accuracy/Precision/Recall/F1 "
        "and confusion matrices for classifiers."
    )


def _test_design_answer(is_vi: bool) -> str:
    if is_vi:
        return (
            "Một test case tốt cho ChronoRAG nên có: input query, topic đang chọn, expected scope "
            "decision, expected retrieval/citation, expected answer behavior và negative checks. "
            "Acceptance criteria: câu đúng domain phải trả lời dựa trên corpus hoặc project policy; "
            "câu ngoài domain phải từ chối/chuyển hướng; câu thiếu context phải hỏi lại hoặc abstain; "
            "mọi factual claim từ corpus phải có citation."
        )
    return (
        "A strong ChronoRAG test case should include input query, selected topic, "
        "expected scope decision, expected retrieval/citation behavior, expected answer "
        "behavior, and negative checks. In-domain questions should answer from corpus "
        "or project policy; out-of-domain questions should refuse/redirect; missing "
        "context should trigger clarification or abstain; corpus facts need citations."
    )


def _metadata_answer(is_vi: bool) -> str:
    if is_vi:
        return (
            "Quy tắc thiết kế: metadata lưu thuộc tính ổn định ở mức document/chunk như doc_id, "
            "topic, source_url, version, retrieved_at, year và published_date. Nội dung chunk chỉ "
            "nên chứa text dùng cho retrieval. Khi hai nguồn mâu thuẫn, ưu tiên nguồn approved mới "
            "hơn, có version/retrieved_at rõ hơn, và phải nói rõ uncertainty nếu metadata không đủ."
        )
    return (
        "Design rule: metadata stores stable document/chunk attributes such as doc_id, "
        "topic, source_url, version, retrieved_at, year, and published_date. Chunk text "
        "should stay focused on retrieval content. When sources conflict, prefer the "
        "newer approved source with clearer version/retrieved_at, and state uncertainty "
        "when metadata is insufficient."
    )


def _agent_answer(is_vi: bool) -> str:
    if is_vi:
        return (
            "Với AI Agent trong ChronoRAG, workflow nên tách rõ: plan, retrieve, validate context, "
            "answer, cite và fallback. Tool routing phải có guardrail: không gọi tool khi scope sai, "
            "không lộ chain-of-thought, log lỗi tool, retry có giới hạn, và hỏi lại nếu action thiếu "
            "thông tin. Test nên đo tool-call accuracy, recovery rate và citation consistency."
        )
    return (
        "For AI-agent workflows in ChronoRAG, separate planning, retrieval, context "
        "validation, answering, citation, and fallback. Tool routing needs guardrails: "
        "do not call tools outside scope, do not expose hidden reasoning, log tool "
        "errors, limit retries, and ask for clarification when required. Test tool-call "
        "accuracy, recovery rate, and citation consistency."
    )


def _kd_answer(is_vi: bool) -> str:
    if is_vi:
        return (
            "Với Knowledge Distillation, câu trả lời nên phân biệt teacher, student, loss và "
            "evaluation. Nếu thiết kế test, nên kiểm KL divergence/cross-entropy, compression ratio, "
            "latency, accuracy/F1 giữ lại, và so sánh student với teacher/baseline. Nếu thiếu nguồn "
            "cụ thể trong corpus, nói rõ đó là guidance kỹ thuật chứ không phải citation paper."
        )
    return (
        "For Knowledge Distillation, answers should distinguish teacher, student, loss, "
        "and evaluation. Tests should cover KL divergence/cross-entropy, compression "
        "ratio, latency, retained accuracy/F1, and student-vs-teacher/baseline comparisons. "
        "If corpus evidence is missing, label the response as technical guidance rather "
        "than a paper-cited claim."
    )


def _rag_design_answer(is_vi: bool) -> str:
    if is_vi:
        return (
            "Với RAG trong ChronoRAG, pipeline nên gồm scope guard, query rewrite/history, hybrid "
            "retrieval BM25+dense, context quality gate, LLM answer synthesis và citation check. "
            "Nếu context không đủ, assistant nên abstain hoặc hỏi lại; nếu có context, mọi claim "
            "quan trọng phải gắn doc_id. Reranker và LangChain nên nằm ở generation/retrieval layer, "
            "không trộn vào UI."
        )
    return (
        "For ChronoRAG's RAG pipeline, use scope guard, query rewrite/history, hybrid "
        "BM25+dense retrieval, context quality gating, LLM answer synthesis, and citation "
        "checking. If context is insufficient, abstain or ask a clarification; if context "
        "is enough, attach doc_id citations to key claims. Reranker/LangChain belong in "
        "retrieval/generation layers, not UI code."
    )


def _general_project_answer(is_vi: bool) -> str:
    if is_vi:
        return (
            "Với phạm vi ChronoRAG, nên trả lời theo hướng: xác định scope, kiểm tra có nguồn "
            "trong corpus hay không, nếu có thì cite, nếu không thì nêu rõ đây là project guidance "
            "không citation. Tránh bịa paper/date/source và ưu tiên câu trả lời ngắn, có tiêu chí kiểm thử."
        )
    return (
        "Within ChronoRAG scope, answer by checking scope first, then checking whether "
        "the corpus supports the claim. Cite when evidence exists; otherwise clearly mark "
        "the response as uncited project guidance. Avoid inventing papers, dates, or sources, "
        "and prefer concise answers with testable criteria."
    )


def _has(q: str, terms: tuple[str, ...]) -> bool:
    return any(term in q for term in terms)


def _looks_vietnamese(q: str) -> bool:
    return any(term in q for term in ("dai ka", "nen", "neu", "hoi", "tra loi", "thiet ke", "trong repo", "khong", "co ")) or not re_is_ascii(q)


def re_is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

