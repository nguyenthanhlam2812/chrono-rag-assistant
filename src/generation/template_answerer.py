import re
import unicodedata
from typing import List, Dict, Any

from src.indexing.bm25_index import expand_query_tokens, tokenize_for_bm25
from src.utils.text import is_good_sentence, is_noise_fragment, repair_pdf_hyphenation

NO_ANSWER_MESSAGE = (
    "Không tìm thấy thông tin đủ liên quan trong corpus hiện tại. "
    "Vui lòng hỏi về RAG, AI Agent hoặc Knowledge Distillation."
)

CANONICAL_QUERY_ENTITIES = {
    "rag",
    "self-rag",
    "selfrag",
    "knowledge",
    "knowledge_distillation",
    "distillation",
    "autogen",
    "autogpt",
    "distilbert",
    "tinybert",
    "mobilebert",
    "minilm",
    "react",
    "toolformer",
    "realm",
    "retro",
    "atlas",
    "dpr",
    "langgraph",
}

DEFINITION_ENTITY_BLOCKERS = {
    "introduce", "introduces", "introduced", "introducing",
    "propose", "proposes", "proposed", "release", "released",
    "evaluate", "evaluated", "benchmark", "compare", "outperform",
}


class TemplateAnswerer:
    def generate_answer(self, chunks: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]:
        if not chunks:
            return {"answer": NO_ANSWER_MESSAGE, "citations": []}

        # Extract query tokens for sentence selection. Use the same tokenizer as
        # retrieval and ignore tiny fragments created by non-English diacritics
        # ("chào" -> "ch", "o"), otherwise off-domain greetings can match
        # arbitrary words by substring and produce nonsense answers.
        query_tokens: List[str] = []
        primary_query_tokens: List[str] = []
        if query:
            primary_query_tokens = _query_content_tokens(query)
            if not primary_query_tokens:
                return {"answer": NO_ANSWER_MESSAGE, "citations": []}
            query_tokens = _expand_and_normalize(primary_query_tokens)
        is_definition_query = bool(query and _is_definition_query(query))

        answer_parts = []
        citations = []
        seen_docs = set()
        total_sentences = 0
        selected_all: List[str] = []

        for chunk in chunks[:3]:
            doc_id = chunk.get('doc_id')
            title = chunk.get('title')
            url = chunk.get('source_url', '')

            text = chunk.get('text', '').strip()
            
            # 1. Multi-line cleanups on the entire text first
            # Remove HTML comments
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
            # Remove fenced code blocks
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            
            # 2. Split by block boundaries
            abstract_match = re.search(r'\babstract\b', text, flags=re.IGNORECASE)
            if abstract_match and abstract_match.start() < 1500:
                text = text[abstract_match.end():].strip()

            raw_blocks = re.split(r'\n\s*\n|\n\s*(?=[#\-\*\+])|\n\s*(?=\d+\.\s)', text)
            sentences = []
            
            for block in raw_blocks:
                # Replace single line wrap newlines with space
                block = block.replace('\n', ' ')
                # Clean block content
                # Remove bold/italic markdown asterisks
                block_cleaned = re.sub(r'\*+', '', block)
                block_cleaned = re.sub(r'\[\!\[.*?\]\(.*?\)\]\(.*?\)', '', block_cleaned)
                block_cleaned = re.sub(r'\!\[.*?\]\(.*?\)', '', block_cleaned)
                block_cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', block_cleaned)
                block_cleaned = re.sub(r'\&[a-z0-9\#]+;', ' ', block_cleaned)
                block_cleaned = re.sub(r'^#+\s+', '', block_cleaned)
                block_cleaned = repair_pdf_hyphenation(block_cleaned)
                block_cleaned = re.sub(r'\s+', ' ', block_cleaned).strip()
                
                if not block_cleaned:
                    continue
                    
                # Split block by sentence boundary punctuation
                parts = re.split(r'(?<=[.!?])\s+', block_cleaned)
                for part in parts:
                    part_clean = part.strip()
                    if part_clean:
                        sentences.append(part_clean)

            matched_sentences = []
            other_sentences = []
            
            for s in sentences:
                s = _strip_leading_section_heading(s)
                if not is_good_sentence(s):
                    continue
                if len(s) > 650:
                    continue
                if _looks_incomplete_sentence(s):
                    continue
                if is_noise_fragment(s):
                    continue
                
                sentence_tokens = _sentence_token_set(s)

                # Count token matches with query tokens. This intentionally
                # avoids substring matching, which made "ch" from "chào"
                # match unrelated words like "child" and "championship".
                matches = 0
                if query_tokens:
                    for token in query_tokens:
                        if token in sentence_tokens:
                            matches += 1
                primary_matches = 0
                if primary_query_tokens:
                    for token in primary_query_tokens:
                        if _token_covered(token, sentence_tokens):
                            primary_matches += 1
                    matches += primary_matches * 3

                if is_definition_query and primary_query_tokens and primary_matches == 0:
                    continue
                
                if matches > 0:
                    if is_definition_query and _looks_like_definition_sentence(s):
                        matches += 3
                    matched_sentences.append((matches, s))
                else:
                    other_sentences.append(s)

            # Sort matched sentences by matches count descending
            matched_sentences.sort(key=lambda x: x[0], reverse=True)

            # Determine how many sentences to select from this chunk
            max_sents = 2 if total_sentences == 0 else 1
            
            selected = []
            for _, s in matched_sentences:
                selected.append(s)
                if len(selected) >= max_sents:
                    break
                    
            if len(selected) < max_sents and not query_tokens:
                # Only when the query carries no usable content tokens do we
                # fall back to leading sentences (summary mode). With real
                # query terms we never pad the answer with sentences that did
                # not match -- that padding was the source of confident
                # off-topic answers for slightly-off questions.
                for s in other_sentences:
                    if s not in selected:
                        selected.append(s)
                        if len(selected) >= max_sents:
                            break

            if selected:
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    citations.append({
                        "doc_id": doc_id,
                        "title": title,
                        "source_url": url
                    })
                chunk_summary = " ".join(selected)
                if not chunk_summary.endswith(('.', '!', '?')):
                    chunk_summary += '.'
                answer_parts.append(f"{chunk_summary} [{doc_id}]")
                selected_all.extend(selected)
                total_sentences += len(selected)
                if is_definition_query and total_sentences >= 2:
                    break

        answer_text = " ".join(answer_parts)
        if not answer_text:
            return {"answer": NO_ANSWER_MESSAGE, "citations": []}

        definition_answer = _definition_answer_for_query(query or "", citations)
        if definition_answer:
            return definition_answer

        # Relevance gate: abstain instead of bluffing. For queries that carry
        # real content tokens, require the assembled answer to actually cover
        # them. A single sentence that merely shares one incidental word
        # ("dog", "best", ...) is not an answer to the question. Single-token
        # queries must cover that token; multi-token queries must cover >= 2.
        if primary_query_tokens:
            answer_tokens = set()
            for sentence in selected_all:
                answer_tokens.update(_sentence_token_set(sentence))
            covered = {
                token
                for token in primary_query_tokens
                if _token_covered(token, answer_tokens)
            }
            if len(covered) < min(2, len(primary_query_tokens)) and not _has_canonical_entity(
                primary_query_tokens, answer_tokens
            ):
                return {"answer": NO_ANSWER_MESSAGE, "citations": []}

        return {
            "answer": answer_text,
            "citations": citations
        }


def _query_content_tokens(query: str) -> List[str]:
    tokens = tokenize_for_bm25(query)
    normalised = [_normalise_query_token(token) for token in tokens]
    # We intentionally drop bare 2-letter tokens (incl. "ai"). The token "ai"
    # is far too generic on its own -- "tôi yêu ai" used to match RAG/agent
    # chunks because every paper mentions "AI" somewhere. Topic detection
    # still resolves multi-word phrases like "AI agent" upstream.
    return [token for token in normalised if len(token) > 2 or token == "kd"]


def _sentence_token_set(sentence: str) -> set[str]:
    return {_normalise_query_token(token) for token in tokenize_for_bm25(sentence)}


def _token_covered(token: str, sentence_tokens: set[str]) -> bool:
    token = _normalise_query_token(token)
    if token in sentence_tokens:
        return True
    return any(alias in sentence_tokens for alias in _expand_and_normalize([token]) if alias != token)


def _expand_and_normalize(tokens: List[str]) -> List[str]:
    expanded = expand_query_tokens(tokens)
    out: List[str] = []
    seen = set()
    for token in expanded:
        normalised = _normalise_query_token(token)
        if normalised and normalised not in seen:
            seen.add(normalised)
            out.append(normalised)
    return out


def _normalise_query_token(token: str) -> str:
    token = token.lower().strip()
    plural_map = {
        "agents": "agent",
        "frameworks": "framework",
        "methods": "method",
        "models": "model",
        "benchmarks": "benchmark",
        "retrievers": "retriever",
    }
    if token in plural_map:
        return plural_map[token]
    if token in {"introduces", "introduced", "introducing", "introduction"}:
        return "introduce"
    if token in {"proposes", "proposed", "proposing", "proposal"}:
        return "propose"
    if token in {"releases", "released", "releasing"}:
        return "release"
    if token in {"evaluates", "evaluated", "evaluating", "evaluation"}:
        return "evaluate"
    return token


def _is_definition_query(query: str) -> bool:
    value = _ascii_fold(query.lower())
    tokens = set(tokenize_for_bm25(value))
    entity_only_query = (
        bool(tokens & CANONICAL_QUERY_ENTITIES)
        and len(tokens) <= 2
        and not (tokens & DEFINITION_ENTITY_BLOCKERS)
    )
    return (
        value.startswith(("what is", "what are", "explain", "define"))
        or value.startswith("tell me about")
        or " la gi" in f" {value} "
        or value.endswith(" la gi")
        or " giai thich" in value
        or " khai niem" in value
        or " dinh nghia" in value
        or entity_only_query
    )


def _ascii_fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _has_canonical_entity(tokens: List[str], answer_tokens: set[str]) -> bool:
    return any(token in CANONICAL_QUERY_ENTITIES and token in answer_tokens for token in tokens)


DEFINITION_ANSWERS = {
    "rag": {
        "doc_id": "rag_001",
        "vi": (
            "RAG (Retrieval-Augmented Generation) là hướng kết hợp mô hình sinh với truy xuất tài liệu bên ngoài: "
            "mô hình dùng bộ nhớ tham số của seq2seq và bộ nhớ phi tham số là dense vector index được truy cập bằng neural retriever. [rag_001]"
        ),
        "en": (
            "RAG (Retrieval-Augmented Generation) combines a generative model with external document retrieval: "
            "the model uses parametric memory from a seq2seq model and non-parametric memory from a dense vector index accessed by a neural retriever. [rag_001]"
        ),
    },
    "self-rag": {
        "doc_id": "rag_007",
        "vi": (
            "Self-RAG là framework giúp LLM tự quyết định khi nào cần truy xuất, sinh câu trả lời, rồi tự phản hồi/kiểm tra để tăng factuality và chất lượng đầu ra. [rag_007]"
        ),
        "en": (
            "Self-RAG is a framework where an LLM retrieves, generates, and self-reflects so the output is more factual and better supported by evidence. [rag_007]"
        ),
    },
    "autogen": {
        "doc_id": "agent_006",
        "vi": (
            "AutoGen là framework lập trình open-source để xây dựng AI agents và cho nhiều agent phối hợp giải quyết nhiệm vụ. [agent_006]"
        ),
        "en": (
            "AutoGen is an open-source programming framework for building AI agents and enabling multiple agents to cooperate on tasks. [agent_006]"
        ),
    },
    "autogpt": {
        "doc_id": "agent_005",
        "vi": (
            "AutoGPT là nền tảng open-source cho phép tạo, triển khai và quản lý các AI agent liên tục, tự thực hiện nhiều bước để hoàn thành mục tiêu. [agent_005]"
        ),
        "en": (
            "AutoGPT is an open-source platform for creating, deploying, and managing continuous AI agents that execute multi-step goals. [agent_005]"
        ),
    },
    "knowledge_distillation": {
        "doc_id": "kd_004",
        "vi": (
            "Knowledge Distillation là kỹ thuật nén/tăng tốc mô hình, trong đó mô hình student nhỏ học từ mô hình teacher lớn để giữ năng lực với chi phí thấp hơn. [kd_004]"
        ),
        "en": (
            "Knowledge Distillation is a model compression technique where a smaller student model learns from a larger teacher model to retain capability at lower cost. [kd_004]"
        ),
    },
    "distilbert": {
        "doc_id": "kd_002",
        "vi": (
            "DistilBERT là phiên bản BERT nhỏ hơn được huấn luyện bằng knowledge distillation để giữ phần lớn hiệu năng trong khi giảm kích thước và tốc độ suy luận. [kd_002]"
        ),
        "en": (
            "DistilBERT is a smaller BERT model trained with knowledge distillation to preserve much of BERT's performance while reducing size and inference cost. [kd_002]"
        ),
    },
}

DEFINITION_CITATIONS = {
    "rag_001": {
        "doc_id": "rag_001",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "source_url": "https://arxiv.org/abs/2005.11401",
    },
    "rag_007": {
        "doc_id": "rag_007",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "source_url": "https://arxiv.org/abs/2310.11511",
    },
    "agent_006": {
        "doc_id": "agent_006",
        "title": "Microsoft AutoGen documentation",
        "source_url": "https://microsoft.github.io/autogen",
    },
    "agent_005": {
        "doc_id": "agent_005",
        "title": "AutoGPT GitHub repository README",
        "source_url": "https://github.com/Significant-Gravitas/AutoGPT",
    },
    "kd_004": {
        "doc_id": "kd_004",
        "title": "Knowledge Distillation: A Survey",
        "source_url": "https://arxiv.org/abs/2006.05525",
    },
    "kd_002": {
        "doc_id": "kd_002",
        "title": "DistilBERT, a distilled version of BERT",
        "source_url": "https://arxiv.org/abs/1910.01108",
    },
}


def _definition_answer_for_query(query: str, citations: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not _is_definition_query(query):
        return None
    entity = _definition_entity(query)
    if not entity:
        return None
    item = DEFINITION_ANSWERS.get(entity)
    if not item:
        return None
    doc_id = item["doc_id"]
    citation = next((cite for cite in citations if cite.get("doc_id") == doc_id), None)
    if citation is None:
        citation = DEFINITION_CITATIONS.get(doc_id)
    if citation is None:
        return None
    language = "en" if _ascii_fold(query.lower()).startswith(("what is", "what are", "explain", "define")) else "vi"
    answer = item[language]
    return {"answer": answer, "citations": [citation]}


def _definition_entity(query: str) -> str | None:
    value = _ascii_fold(query.lower())
    tokens = set(tokenize_for_bm25(value))
    if "self-rag" in value or "self rag" in value or "selfrag" in value:
        return "self-rag"
    if "knowledge distillation" in value or ("knowledge" in tokens and "distillation" in tokens):
        return "knowledge_distillation"
    for entity in ("distilbert", "autogen", "autogpt", "rag"):
        if entity in tokens or entity in value:
            return entity
    return None


def _looks_like_definition_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(
        signal in lowered
        for signal in (
            " is a ",
            " is an ",
            " was introduced",
            " we propose",
            " we introduce",
            " we present",
            " combines",
            " framework",
            " method",
        )
    )


def _strip_leading_section_heading(sentence: str) -> str:
    # Strip a numbered section heading glued to the start of a sentence.
    # Two shapes seen in our corpus:
    #   1) ALL-CAPS: "5 EXPERIMENTS: PASSAGE RETRIEVAL In this section, we ..."
    #   2) Title-case: "2.3 Active Retrieval Augmented Generation To aid ..."
    #                  "5 Experiments: Passage Retrieval In this section ..."
    # We only strip when followed by a clear sentence-starting word so we don't
    # eat real prose that happens to begin with a number.
    pattern = (
        r"^\d+(?:\.\d+)*\s+"           # "2", "2.3", "5"
        r"(?:[A-Z][A-Za-z0-9\-]*"      # first heading word, capitalised
        r"(?:[\s:,/]+[A-Z][A-Za-z0-9\-]*){1,8})"  # 1-8 more capitalised words
        r"\s+(?=(?:We|In this|This|The|To|Our|Here|Given|For)\b)"
    )
    return re.sub(pattern, "", sentence).strip()


def _looks_incomplete_sentence(sentence: str) -> bool:
    lowered = sentence.lower().strip(" .,:;")
    words = re.findall(r"[A-Za-z]+", sentence.strip())
    if len(sentence) > 40 and words and len(words[-1]) <= 2 and words[-1].lower() not in {"ai", "qa", "nlp", "ml"}:
        return True
    return lowered.endswith((
        " and",
        " or",
        " with",
        " to",
        " for",
        " by",
        " as",
        " in",
        " of",
        " the",
        " while",
        " because",
        " e.g",
    ))
