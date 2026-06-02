#!/usr/bin/env python
"""
06_build_vector_index.py
========================
CLI script that builds both the FAISS dense vector index and the BM25
sparse index from the chunked documents produced in earlier pipeline
stages.

Usage::

    python scripts/06_build_vector_index.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.indexing.build_vector_index import build_faiss_index
from src.indexing.build_bm25 import build_bm25_index

logger = setup_logger("build_vector_index")


def main() -> None:
    """Load config, resolve paths, and build FAISS + BM25 indices."""
    logger.info("=== Sprint 5: Building FAISS & BM25 indices ===")

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config = load_config()

    processed_dir = Path(config["paths"]["processed_data_dir"])
    vector_db_dir = Path(config["paths"]["vector_db_dir"])
    embedding_model: str = config["indexing"]["embedding_model"]

    chunks_path = processed_dir / "chunks.jsonl"

    logger.info("Chunks file   : %s", chunks_path)
    logger.info("Output dir    : %s", vector_db_dir)
    logger.info("Embedding model: %s", embedding_model)

    # ------------------------------------------------------------------
    # 2. Build FAISS index
    # ------------------------------------------------------------------
    logger.info("--- Building FAISS index ---")
    n_faiss = build_faiss_index(
        chunks_path=chunks_path,
        output_dir=vector_db_dir,
        model_name=embedding_model,
    )
    logger.info("FAISS indexing complete – %d chunks indexed.", n_faiss)

    # ------------------------------------------------------------------
    # 3. Build BM25 index
    # ------------------------------------------------------------------
    logger.info("--- Building BM25 index ---")
    n_bm25 = build_bm25_index(
        chunks_path=chunks_path,
        output_dir=vector_db_dir,
    )
    logger.info("BM25 indexing complete – %d chunks indexed.", n_bm25)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    logger.info("=== All indices built successfully ===")


if __name__ == "__main__":
    main()
