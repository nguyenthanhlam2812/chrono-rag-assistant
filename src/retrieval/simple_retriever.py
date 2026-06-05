import re
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

from src.indexing.bm25_index import tokenize_for_bm25
from src.utils.text import STOPWORDS, clean_for_matching, is_good_sentence

# Field-level term-frequency weights. Title hits matter most because doc titles
# are the strongest topical signal; topic hits are secondary; body text is the
# baseline.
WEIGHT_TF_TEXT = 1.0
WEIGHT_TF_TITLE = 5.0
WEIGHT_TF_TOPIC = 2.0

# Boost given for each query-token match that lands inside a clean,
# capitalised sentence (passes ``is_good_sentence``). Heavily favours chunks
# whose matches are in real prose rather than code or tables.
WEIGHT_NATURAL_SENTENCE_MATCH = 10.0

# Boost when the chunk contains a "definition-like" pattern such as
# "<query_token> is/was/enables/provides". Tuned to surface intro paragraphs
# of papers/blogs over passing mentions, but capped (one boost per pattern,
# per token) so it cannot drown out other signals across many tokens.
WEIGHT_DEFINITION_PATTERN = 50.0

# Penalty per code/config signal when the user is NOT asking a code question.
# Prevents Sprint 1.5 Q&A from surfacing setup snippets for prose queries.
PENALTY_CODE_SIGNAL = 15.0

_DEFINITION_VERBS = ("is", "was", "enables", "provides")
_CODE_SIGNALS = (
    "import ", "os.environ", "api_key", "pip install", "docker",
    "code_execution", "llm_config", "assistant =", "user_proxy =",
    "```python", "``` python", "={",
)
_CODE_QUERY_KEYWORDS = frozenset({
    "code", "config", "install", "api_key", "import", "key",
    "example", "pip", "setup", "docker", "requirements",
})

CANONICAL_ENTITY_DOCS = {
    "rag": {"rag_001"},
    "self-rag": {"rag_007"},
    "selfrag": {"rag_007"},
    "autogpt": {"agent_005"},
    "autogen": {"agent_006"},
    "knowledge_distillation": {"kd_001", "kd_004"},
    "distillation": {"kd_001", "kd_004"},
    "distilbert": {"kd_002", "kd_011"},
}

DEFINITION_ENTITY_BLOCKERS = {
    "introduce", "introduces", "introduced", "introducing",
    "propose", "proposes", "proposed", "release", "released",
    "evaluate", "evaluated", "benchmark", "compare", "outperform",
}


class SimpleRetriever:
    def __init__(self, chunks_path: Path = None):
        if chunks_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            chunks_path = repo_root / 'data' / 'processed' / 'chunks.jsonl'
        self.chunks_path = Path(chunks_path)
        self.chunks = []
        self.load_chunks()

    def load_chunks(self):
        if not self.chunks_path.exists():
            return
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))

    def retrieve(self, query: str, topic: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        # Tokenize query with the same rules as BM25. This avoids producing
        # one-letter junk tokens from Vietnamese diacritics in queries like
        # "RAG là gì".
        tokens = tokenize_for_bm25(query)
        query_tokens = [_normalise_query_token(t) for t in tokens if t not in STOPWORDS]
        
        # If no tokens left after stopword filtering, use all tokens
        if not query_tokens:
            query_tokens = tokens
            
        if not query_tokens:
            return []

        # Check if query asks for code/configs
        is_code_query = any(k in query.lower() for k in _CODE_QUERY_KEYWORDS)
        is_definition_query = _is_definition_query(query)
        query_entities = _query_entities(query)

        scored_chunks = []
        # Normalize topic mapping
        topic_normalized = None
        if topic:
            topic_clean = topic.lower().strip()
            if "rag" in topic_clean:
                topic_normalized = "rag"
            elif "agent" in topic_clean:
                topic_normalized = "ai_agent"
            elif "distill" in topic_clean:
                topic_normalized = "knowledge_distillation"

        for chunk in self.chunks:
            # Check topic match if specified
            if topic_normalized and chunk.get('topic') != topic_normalized:
                continue

            chunk_text = chunk.get('text', '').lower()
            chunk_title = chunk.get('title', '').lower()
            chunk_topic = chunk.get('topic', '').lower()

            # Clean and tokenize chunk fields
            text_tokens = re.sub(r'[^a-zA-Z0-9]', ' ', chunk_text).split()
            title_tokens = re.sub(r'[^a-zA-Z0-9]', ' ', chunk_title).split()
            topic_tokens = re.sub(r'[^a-zA-Z0-9]', ' ', chunk_topic).split()

            score = 0.0
            for token in query_tokens:
                tf_text = text_tokens.count(token)
                tf_title = title_tokens.count(token)
                tf_topic = topic_tokens.count(token)

                score += tf_text * WEIGHT_TF_TEXT
                score += tf_title * WEIGHT_TF_TITLE
                score += tf_topic * WEIGHT_TF_TOPIC

            # 1. Prefer chunks whose query terms land inside natural sentences.
            sentences = re.split(r'(?<=[.!?])\s+|\n+', chunk.get('text', ''))
            natural_sentence_matches = 0
            for s in sentences:
                if is_good_sentence(s):
                    for token in query_tokens:
                        if token in s.lower():
                            natural_sentence_matches += 1
            score += natural_sentence_matches * WEIGHT_NATURAL_SENTENCE_MATCH

            # 2. Boost definition patterns (e.g. "RAG is", "X provides").
            chunk_text_clean = clean_for_matching(chunk.get('text', ''))
            for token in query_tokens:
                for verb in _DEFINITION_VERBS:
                    pat = f"{token} {verb}"
                    if re.search(r'\b' + re.escape(pat) + r'\b', chunk_text_clean):
                        score += WEIGHT_DEFINITION_PATTERN

            # 3. Penalise code/config-heavy chunks when the query is prose.
            if not is_code_query:
                code_matches = sum(chunk_text.count(sig) for sig in _CODE_SIGNALS)
                if code_matches > 0:
                    score = max(0.0, score - code_matches * PENALTY_CODE_SIGNAL)

            # 4. Definition questions need stable intro/foundational sources,
            # not arbitrary benchmark/table chunks that happen to mention RAG.
            if is_definition_query:
                doc_id = str(chunk.get("doc_id", "") or "").lower()
                start_char = int(chunk.get("start_char", 0) or 0)
                for entity, doc_ids in CANONICAL_ENTITY_DOCS.items():
                    if entity in query_entities and doc_id in doc_ids:
                        score += 120.0
                if start_char < 1200:
                    score += 55.0
                elif start_char < 5000:
                    score += 20.0
                elif start_char > 12000:
                    score -= 30.0
                if any(signal in chunk_text[:900] for signal in ("we propose", "we introduce", "is a", "is an", "framework", "method")):
                    score += 35.0

            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by score descending, then start_char ascending (tie-breaker)
        scored_chunks.sort(key=lambda x: (-x[0], x[1].get('start_char', 0)))

        return [chunk for _, chunk in scored_chunks[:top_k]]


def _is_definition_query(query: str) -> bool:
    value = _ascii_fold(query.lower())
    tokens = set(tokenize_for_bm25(value))
    entity_only_query = (
        bool(_query_entities(value) & set(CANONICAL_ENTITY_DOCS))
        and len(tokens) <= 2
        and not (tokens & DEFINITION_ENTITY_BLOCKERS)
    )
    return (
        value.startswith(("what is", "what are", "explain", "define"))
        or " la gi" in f" {value} "
        or value.endswith(" la gi")
        or " giai thich" in value
        or " khai niem" in value
        or " dinh nghia" in value
        or entity_only_query
    )


def _query_entities(query: str) -> set[str]:
    value = _ascii_fold(query.lower())
    tokens = set(tokenize_for_bm25(value))
    entities = set(tokens)
    if "self-rag" in value or "self rag" in value or "selfrag" in value:
        entities.update({"self-rag", "selfrag"})
    if "knowledge distillation" in value or ("knowledge" in tokens and "distillation" in tokens):
        entities.add("knowledge_distillation")
    return entities


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
    return plural_map.get(token, token)


def _ascii_fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))
