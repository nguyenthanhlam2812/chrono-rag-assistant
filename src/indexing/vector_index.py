from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.utils.io import read_jsonl, write_jsonl


def build_faiss_index(
    chunks_path: Path,
    output_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    encoder: Any | None = None,
) -> Dict[str, Any]:
    """Build a FAISS cosine-similarity index from processed chunks.

    ``encoder`` is injectable for tests; production uses SentenceTransformer.
    """
    import faiss

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found at {chunks_path}")

    texts = [str(chunk.get("text", "") or "") for chunk in chunks]
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name)

    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    (output_dir / "faiss_chunk_ids.json").write_text(
        json.dumps([chunk.get("chunk_id", "") for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "faiss_chunk_metadata.jsonl", [_chunk_metadata(chunk) for chunk in chunks])

    return {
        "chunks_indexed": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_path": str(output_dir / "faiss.index"),
        "chunk_ids_path": str(output_dir / "faiss_chunk_ids.json"),
        "metadata_path": str(output_dir / "faiss_chunk_metadata.jsonl"),
    }


def _chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "topic": chunk.get("topic", ""),
        "title": chunk.get("title", ""),
        "source_url": chunk.get("source_url", ""),
        "year": chunk.get("year"),
    }
