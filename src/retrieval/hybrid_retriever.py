from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.indexing.bm25_index import load_bm25_index, score_bm25, tokenize_for_bm25
from src.retrieval.query_expansion import generate_retrieval_queries


class HybridRetriever:
    """Retrieve chunks with BM25 and optional FAISS dense search.

    The class is intentionally robust for local demos:
    - if a persisted BM25 index exists, it uses it;
    - otherwise it builds BM25 in memory from ``chunks.jsonl``;
    - if FAISS files are absent, it silently uses BM25 only.
    """

    def __init__(
        self,
        chunks_path: Path | None = None,
        index_dir: Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        hybrid_alpha: float = 0.4,
        fusion_strategy: str = "rrf",
        rrf_k: int = 60,
        use_vector: bool = True,
        use_reranker: bool | None = None,
        reranker_model: str | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.chunks_path = chunks_path or repo_root / "data" / "processed" / "chunks.jsonl"
        self.index_dir = index_dir or repo_root / "data" / "vector_db"
        self.embedding_model = embedding_model
        self.hybrid_alpha = max(0.0, min(1.0, hybrid_alpha))
        self.fusion_strategy = (fusion_strategy or "rrf").lower().strip()
        self.rrf_k = max(1, int(rrf_k))
        self.use_vector = use_vector
        self.use_reranker = (
            _env_enabled("RETRIEVAL_RERANKER") if use_reranker is None else use_reranker
        )
        self.reranker_model = reranker_model or os.getenv(
            "RETRIEVAL_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self._reranker = None
        self.chunks = _load_jsonl(self.chunks_path)
        self._chunk_by_id = {str(chunk.get("chunk_id", "")): chunk for chunk in self.chunks}
        self._bm25 = None
        self._bm25_chunk_ids: List[str] = []
        self._faiss_index = None
        self._faiss_chunk_ids: List[str] = []
        self._encoder = None
        self._load_indexes()

    def retrieve(self, query: str, topic: str | None = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip() or not self.chunks or top_k <= 0:
            return []

        candidate_indices = self._candidate_indices(topic)
        if not candidate_indices:
            return []

        search_queries = generate_retrieval_queries(query, topic)
        scored = self._fused_scores(search_queries, candidate_indices)
        if not scored:
            return []
        scored = self._rerank(query, scored, top_n=max(top_k * 8, 24))
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].get("start_char", 0)))
        results: List[Dict[str, Any]] = []
        selected = _mmr_select(scored, self.chunks, top_k=top_k)
        for score, idx in selected:
            row = dict(self.chunks[idx])
            row["_retrieval_score"] = round(float(score), 6)
            row["_retrieval_mode"] = self._retrieval_mode(search_queries)
            results.append(row)
        return results

    def _fused_scores(
        self, search_queries: List[str], candidate_indices: List[int]
    ) -> List[tuple[float, int]]:
        if not search_queries:
            return []

        if self.fusion_strategy == "weighted":
            return self._weighted_scores(search_queries[0], candidate_indices)

        chunk_id_to_idx = {
            str(self.chunks[idx].get("chunk_id", "")): idx for idx in candidate_indices
        }
        fused: Dict[str, float] = {}
        for search_query in search_queries:
            bm25_scores = self._bm25_scores(search_query, candidate_indices)
            self._add_rrf_scores(fused, bm25_scores, weight=1.0)

            vector_scores = (
                self._vector_scores(search_query, candidate_indices) if self.use_vector else {}
            )
            if vector_scores:
                self._add_rrf_scores(fused, vector_scores, weight=self.hybrid_alpha)

            for idx in candidate_indices:
                chunk_id = str(self.chunks[idx].get("chunk_id", ""))
                if chunk_id in fused:
                    fused[chunk_id] += _query_intent_boost(search_query, self.chunks[idx])

        scored = [
            (score, chunk_id_to_idx[chunk_id])
            for chunk_id, score in fused.items()
            if score > 0 and chunk_id in chunk_id_to_idx
        ]
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].get("start_char", 0)))
        return scored

    def _weighted_scores(self, query: str, candidate_indices: List[int]) -> List[tuple[float, int]]:
        bm25_scores = self._bm25_scores(query, candidate_indices)
        vector_scores = self._vector_scores(query, candidate_indices) if self.use_vector else {}

        bm25_norm = _normalise_scores(bm25_scores)
        vector_norm = _normalise_scores(vector_scores)
        scored: List[tuple[float, int]] = []
        for idx in candidate_indices:
            chunk_id = str(self.chunks[idx].get("chunk_id", ""))
            sparse_score = bm25_norm.get(chunk_id, 0.0)
            dense_score = vector_norm.get(chunk_id, 0.0)
            if dense_score > 0:
                score = (1.0 - self.hybrid_alpha) * sparse_score + self.hybrid_alpha * dense_score
            else:
                score = sparse_score
            score += _query_intent_boost(query, self.chunks[idx])
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].get("start_char", 0)))
        return scored

    def _add_rrf_scores(
        self,
        fused: Dict[str, float],
        scores: Dict[str, float],
        weight: float = 1.0,
    ) -> None:
        ranked = [
            item for item in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if item[1] > 0
        ]
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (self.rrf_k + rank)

    def _rerank(
        self,
        query: str,
        scored: List[tuple[float, int]],
        top_n: int,
    ) -> List[tuple[float, int]]:
        if not self.use_reranker or not scored:
            return scored
        reranker = self._get_reranker()
        if reranker is None:
            return scored

        head = scored[:top_n]
        tail = scored[top_n:]
        pairs = [(query, _reranker_text(self.chunks[idx])) for _, idx in head]
        try:
            rerank_scores = reranker.predict(pairs)
        except Exception:
            return scored

        reranked = []
        for original_rank, ((base_score, idx), rerank_score) in enumerate(zip(head, rerank_scores)):
            # Keep a tiny part of the fused score as a stable tie-breaker.
            combined = float(rerank_score) + (base_score * 0.01) - (original_rank * 1e-6)
            reranked.append((combined, idx))
        reranked.sort(key=lambda item: (-item[0], self.chunks[item[1]].get("start_char", 0)))
        return reranked + tail

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model)
        except Exception:
            self._reranker = None
        return self._reranker

    def _retrieval_mode(self, search_queries: List[str]) -> str:
        bits = ["rrf" if self.fusion_strategy == "rrf" else "weighted"]
        bits.append("hybrid" if self.use_vector and self._faiss_index is not None else "bm25")
        if len(search_queries) > 1:
            bits.append("multi_query")
        if self.use_reranker and self._reranker is not None:
            bits.append("rerank")
        return "+".join(bits)

    def _load_indexes(self) -> None:
        try:
            self._bm25 = load_bm25_index(self.index_dir)
            ids_path = self.index_dir / "bm25_chunk_ids.json"
            if self._bm25 is not None and ids_path.exists():
                self._bm25_chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        except Exception:
            self._bm25 = None
            self._bm25_chunk_ids = []

        if not self.use_vector:
            return
        faiss_path = self.index_dir / "faiss.index"
        ids_path = self.index_dir / "faiss_chunk_ids.json"
        if not faiss_path.exists() or not ids_path.exists():
            return
        try:
            import faiss

            self._faiss_index = faiss.read_index(str(faiss_path))
            self._faiss_chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        except Exception:
            self._faiss_index = None
            self._faiss_chunk_ids = []

    def _candidate_indices(self, topic: str | None) -> List[int]:
        topic_key = _normalise_topic(topic)
        if not topic_key:
            return list(range(len(self.chunks)))
        return [
            idx
            for idx, chunk in enumerate(self.chunks)
            if _normalise_topic(str(chunk.get("topic", ""))) == topic_key
        ]

    def _bm25_scores(self, query: str, candidate_indices: List[int]) -> Dict[str, float]:
        if (
            self._bm25 is not None
            and self._bm25_chunk_ids
            and len(self._bm25_chunk_ids) == len(self.chunks)
        ):
            scores = score_bm25(query, self.chunks, bm25=self._bm25)
            candidate_set = set(candidate_indices)
            return {
                str(self.chunks[idx].get("chunk_id", "")): scores[idx]
                for idx in candidate_set
            }

        candidate_chunks = [self.chunks[idx] for idx in candidate_indices]
        scores = score_bm25(query, candidate_chunks, bm25=None)
        return {
            str(chunk.get("chunk_id", "")): score
            for chunk, score in zip(candidate_chunks, scores)
        }

    def _vector_scores(self, query: str, candidate_indices: List[int]) -> Dict[str, float]:
        if self._faiss_index is None or not self._faiss_chunk_ids:
            return {}
        try:
            query_vec = self._encode_query(query)
            limit = min(max(len(candidate_indices), 10), len(self._faiss_chunk_ids))
            scores, indices = self._faiss_index.search(query_vec, limit)
        except Exception:
            return {}

        allowed = {str(self.chunks[idx].get("chunk_id", "")) for idx in candidate_indices}
        out: Dict[str, float] = {}
        for score, faiss_idx in zip(scores[0], indices[0]):
            if faiss_idx < 0 or faiss_idx >= len(self._faiss_chunk_ids):
                continue
            chunk_id = str(self._faiss_chunk_ids[int(faiss_idx)])
            if chunk_id in allowed:
                out[chunk_id] = max(float(score), out.get(chunk_id, 0.0))
        return out

    def _encode_query(self, query: str) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        embedding = self._encoder.encode([query], convert_to_numpy=True)
        embedding = np.asarray(embedding, dtype="float32")
        import faiss

        faiss.normalize_L2(embedding)
        return embedding


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _reranker_text(chunk: Dict[str, Any]) -> str:
    title = str(chunk.get("title", "") or "").strip()
    text = str(chunk.get("text", "") or "").strip()
    return f"{title}\n{text[:1400]}".strip()


def _normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {}
    return {key: max(0.0, value / max_score) for key, value in scores.items()}


def _normalise_topic(topic: str | None) -> str:
    if not topic:
        return ""
    value = topic.lower().strip().replace("-", "_").replace(" ", "_")
    if "distill" in value or value in {"kd", "knowledge_distillation"}:
        return "knowledge_distillation"
    if "agent" in value:
        return "ai_agent"
    if "rag" in value:
        return "rag"
    return value


def _mmr_select(
    scored: List[tuple[float, int]],
    chunks: List[Dict[str, Any]],
    top_k: int,
    diversity: float = 0.18,
) -> List[tuple[float, int]]:
    """Small redundancy filter inspired by MMR.

    It keeps the top lexical/dense hit first, then favours candidates that add
    different text. This helps broad questions return a mix of survey/method
    chunks instead of near-duplicate neighboring chunks from the same paper.
    """

    if top_k <= 1 or len(scored) <= top_k:
        return scored[:top_k]

    pool = scored[: max(top_k * 6, 12)]
    selected: List[tuple[float, int]] = [pool[0]]
    remaining = pool[1:]
    selected_tokens = [_chunk_tokens(chunks[pool[0][1]])]

    while remaining and len(selected) < top_k:
        best_pos = 0
        best_score = float("-inf")
        for pos, (score, idx) in enumerate(remaining):
            candidate_tokens = _chunk_tokens(chunks[idx])
            max_overlap = max(
                (_jaccard(candidate_tokens, existing) for existing in selected_tokens),
                default=0.0,
            )
            adjusted = score - diversity * max_overlap
            if adjusted > best_score:
                best_pos = pos
                best_score = adjusted
        chosen = remaining.pop(best_pos)
        selected.append(chosen)
        selected_tokens.append(_chunk_tokens(chunks[chosen[1]]))
    return selected


def _chunk_tokens(chunk: Dict[str, Any]) -> set[str]:
    text = " ".join(
        str(chunk.get(field, "") or "") for field in ("title", "topic", "text")
    )
    return set(tokenize_for_bm25(text))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


CANONICAL_ENTITY_DOCS = {
    "rag": {"rag_001"},
    "self-rag": {"rag_007"},
    "selfrag": {"rag_007"},
    "react": {"agent_002"},
    "toolformer": {"agent_003"},
    "knowledge_distillation": {"kd_001", "kd_004"},
    "distillation": {"kd_001", "kd_004"},
    "distilbert": {"kd_002", "kd_011"},
    "tinybert": {"kd_003"},
    "mobilebert": {"kd_009"},
    "minilm": {"kd_008"},
    "autogpt": {"agent_005"},
    "autogen": {"agent_006"},
    "langgraph": {"agent_007"},
}

DEFINITION_ENTITY_BLOCKERS = {
    "introduce", "introduces", "introduced", "introducing",
    "propose", "proposes", "proposed", "release", "released",
    "evaluate", "evaluated", "benchmark", "compare", "outperform",
}


def _query_intent_boost(query: str, chunk: Dict[str, Any]) -> float:
    query_lower = query.lower().strip()
    query_tokens = [token for token in tokenize_for_bm25(query) if len(token) > 2]
    if not query_tokens:
        return 0.0
    query_entities = _query_entities(query)

    title = str(chunk.get("title", "") or "").lower()
    text = str(chunk.get("text", "") or "").lower()
    first_block = text[:900]
    boost = 0.0

    is_definition_query = _is_definition_query(query)
    if is_definition_query:
        doc_id = str(chunk.get("doc_id", "") or "").lower()
        start_char = int(chunk.get("start_char", 0) or 0)
        for entity, doc_ids in CANONICAL_ENTITY_DOCS.items():
            if entity in query_entities and doc_id in doc_ids:
                boost += 2.75

        if start_char < 1000:
            boost += 1.10
        elif start_char < 2000:
            boost += 0.65
        elif start_char < 6000:
            boost += 0.25
        elif start_char > 12000:
            boost -= 0.55
        for token in query_tokens:
            if token in title:
                boost += 0.35
            if token in first_block:
                boost += 0.18
        intro_signals = (
            "we propose", "we introduce", "we present", "we develop",
            "is a", "is an", "combine", "combines", "framework", "method",
        )
        if any(signal in first_block for signal in intro_signals):
            boost += 0.35
        benchmark_signals = (
            "outperform", "accuracy", "f1", "score", "table", "benchmark",
            "results", "ablation",
        )
        if any(signal in first_block for signal in benchmark_signals):
            boost -= 0.75

    return boost


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


def _ascii_fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))
