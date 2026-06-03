import re
import json
from pathlib import Path
from typing import List, Dict, Any

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

        # Tokenize query
        query_cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', query.lower())
        tokens = [t for t in query_cleaned.split() if t]

        query_tokens = [t for t in tokens if t not in STOPWORDS]
        
        # If no tokens left after stopword filtering, use all tokens
        if not query_tokens:
            query_tokens = tokens
            
        if not query_tokens:
            return []

        # Check if query asks for code/configs
        is_code_query = any(k in query.lower() for k in _CODE_QUERY_KEYWORDS)

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

            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by score descending, then start_char ascending (tie-breaker)
        scored_chunks.sort(key=lambda x: (-x[0], x[1].get('start_char', 0)))

        return [chunk for _, chunk in scored_chunks[:top_k]]
