from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import STOPWORDS

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]+|\b(?:19|20)\d{2}\b")

QUERY_ALIASES = {
    "rag": ["retrieval", "augmented", "generation"],
    "react": ["reasoning", "acting", "actions"],
    "dpr": ["dense", "passage", "retrieval"],
    "realm": ["retrieval", "augmented", "language", "model"],
    "retro": ["retrieval", "enhanced", "transformer"],
    "atlas": ["retrieval", "language", "model"],
    "distilbert": ["distilled", "bert", "distillation"],
    "tinybert": ["transformer", "bert", "distillation"],
    "minilm": ["language", "model", "distillation"],
    "mobilebert": ["bert", "compression", "mobile"],
    "autogpt": ["autonomous", "agent"],
    "autogen": ["multi", "agent", "framework"],
    "langgraph": ["agent", "graph", "framework"],
    "toolformer": ["tools", "language", "model"],
}


def tokenize_for_bm25(text: str) -> List[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text or "")]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def build_bm25_index(chunks_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Build and persist a BM25 index from processed chunks."""
    from rank_bm25 import BM25Okapi

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found at {chunks_path}")

    tokenized = [tokenize_for_bm25(_chunk_text(chunk)) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    (output_dir / "bm25_chunk_ids.json").write_text(
        json.dumps([chunk.get("chunk_id", "") for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "bm25_chunk_metadata.jsonl", [_chunk_metadata(chunk) for chunk in chunks])

    return {
        "chunks_indexed": len(chunks),
        "bm25_path": str(output_dir / "bm25.pkl"),
        "chunk_ids_path": str(output_dir / "bm25_chunk_ids.json"),
        "metadata_path": str(output_dir / "bm25_chunk_metadata.jsonl"),
    }


def score_bm25(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    bm25: Any | None = None,
) -> List[float]:
    """Score chunks with BM25, building an in-memory index when needed."""
    from rank_bm25 import BM25Okapi

    query_tokens = expand_query_tokens(tokenize_for_bm25(query))
    if not query_tokens or not chunks:
        return [0.0 for _ in chunks]
    if bm25 is None:
        tokenized = [tokenize_for_bm25(_chunk_text(chunk)) for chunk in chunks]
        bm25 = BM25Okapi(tokenized)
    else:
        tokenized = [tokenize_for_bm25(_chunk_text(chunk)) for chunk in chunks]

    raw_scores = [float(score) for score in bm25.get_scores(query_tokens)]
    lexical_scores = [_lexical_overlap_score(query_tokens, doc_tokens) for doc_tokens in tokenized]
    return [raw + lexical for raw, lexical in zip(raw_scores, lexical_scores)]


def _lexical_overlap_score(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_token_set = set(doc_tokens)
    return float(sum(1 for token in query_tokens if token in doc_token_set))


def expand_query_tokens(tokens: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(QUERY_ALIASES.get(token.lower(), []))
    seen = set()
    deduped: List[str] = []
    for token in expanded:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def load_bm25_index(index_dir: Path) -> Any | None:
    path = index_dir / "bm25.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _chunk_text(chunk: Dict[str, Any]) -> str:
    return " ".join(
        str(chunk.get(field, "") or "")
        for field in ("title", "topic", "text")
    )


def _chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "topic": chunk.get("topic", ""),
        "title": chunk.get("title", ""),
        "source_url": chunk.get("source_url", ""),
        "year": chunk.get("year"),
    }
