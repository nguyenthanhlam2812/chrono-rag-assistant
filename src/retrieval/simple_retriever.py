import re
import json
from pathlib import Path
from typing import List, Dict, Any

def clean_for_matching(text: str) -> str:
    # Remove markdown bold/italic asterisks
    t = re.sub(r'\*+', '', text)
    # Remove markdown link markup keeping anchor text
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', t)
    return t.lower()

def is_good_sentence(s: str) -> bool:
    s_clean = s.strip()
    if len(s_clean) < 15:
        return False
    # Check if first alphabetical character is lowercase
    first_alpha = re.search(r'[a-zA-Z]', s_clean)
    if first_alpha and first_alpha.group(0).islower():
        return False
    # Exclude code or remnant signatures
    bad_signals = ['shields.io', 'github.com', 'http', 'www.', '.py', '.html', 'img.']
    if any(bad in s_clean.lower() for bad in bad_signals):
        return False
    return True

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
        
        # Stopwords list
        stopwords = {
            "is", "what", "the", "a", "an", "and", "or", "in", "on", "at", 
            "for", "with", "about", "to", "of", "how", "why", "where", "when", 
            "who", "whom", "whose", "which", "are", "do", "does", "did", "can",
            "could", "should", "would", "you", "your", "my", "our", "their"
        }
        query_tokens = [t for t in tokens if t not in stopwords]
        
        # If no tokens left after stopword filtering, use all tokens
        if not query_tokens:
            query_tokens = tokens
            
        if not query_tokens:
            return []

        # Check if query asks for code/configs
        code_query_keywords = {"code", "config", "install", "api_key", "import", "key", "example", "pip", "setup", "docker", "requirements", "joke"}
        is_code_query = any(k in query.lower() for k in code_query_keywords)

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
                # Term frequency weighted by field importance
                tf_text = text_tokens.count(token)
                tf_title = title_tokens.count(token)
                tf_topic = topic_tokens.count(token)

                score += tf_text * 1.0
                score += tf_title * 5.0
                score += tf_topic * 2.0

            # 1. Prefer chunks with query terms in natural sentences
            # Split text by sentence-like boundaries and count natural matches
            sentences = re.split(r'(?<=[.!?])\s+|\n+', chunk.get('text', ''))
            natural_sentence_matches = 0
            for s in sentences:
                if is_good_sentence(s):
                    for token in query_tokens:
                        if token in s.lower():
                            natural_sentence_matches += 1
            score += natural_sentence_matches * 10.0

            # 2. Boost definition patterns (e.g., "X is", "X provides", etc.)
            chunk_text_clean = clean_for_matching(chunk.get('text', ''))
            for token in query_tokens:
                patterns = [
                    f"{token} is",
                    f"{token} was",
                    f"{token} enables",
                    f"{token} provides"
                ]
                for pat in patterns:
                    if re.search(r'\b' + re.escape(pat) + r'\b', chunk_text_clean):
                        score += 50.0

            # 3. Penalize chunks with lots of code / config if query is not code-related
            if not is_code_query:
                code_signals = ["import ", "os.environ", "api_key", "pip install", "docker", "code_execution", "llm_config", "assistant =", "user_proxy =", "```python", "``` python", "={"]
                code_matches = sum(chunk_text.count(sig) for sig in code_signals)
                if code_matches > 0:
                    score = max(0.0, score - code_matches * 15.0)

            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by score descending, then start_char ascending (tie-breaker)
        scored_chunks.sort(key=lambda x: (-x[0], x[1].get('start_char', 0)))

        return [chunk for _, chunk in scored_chunks[:top_k]]
