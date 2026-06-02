"""
Build a FAISS vector index from chunked documents.

Reads chunks.jsonl, encodes texts with a SentenceTransformer model,
builds a FAISS IndexFlatIP (inner product on L2-normalised vectors),
and saves the index alongside ordered chunk metadata.
"""

import sys
from pathlib import Path
from typing import Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure project root is on sys.path for sibling imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_jsonl, write_jsonl
from src.utils.logger import setup_logger

logger = setup_logger("build_vector_index")

# Metadata fields carried over from each chunk record
_META_FIELDS = ("chunk_id", "doc_id", "topic", "title", "source_url", "year")


def build_faiss_index(
    chunks_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "all-MiniLM-L6-v2",
) -> int:
    """Build a FAISS inner-product index from *chunks.jsonl*.

    Parameters
    ----------
    chunks_path:
        Path to the ``chunks.jsonl`` file produced by the chunking step.
    output_dir:
        Directory where ``faiss.index`` and ``chunk_metadata.jsonl`` will
        be written.  Created automatically if it does not exist.
    model_name:
        HuggingFace SentenceTransformer model identifier used to produce
        dense embeddings.

    Returns
    -------
    int
        Number of chunks that were indexed.
    """
    chunks_path = Path(chunks_path)
    output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # 1. Load chunks
    # ------------------------------------------------------------------
    if not chunks_path.exists():
        logger.warning("chunks.jsonl not found at %s – skipping FAISS build.", chunks_path)
        return 0

    chunks = read_jsonl(chunks_path)
    if not chunks:
        logger.warning("chunks.jsonl is empty – skipping FAISS build.")
        return 0

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # ------------------------------------------------------------------
    # 2. Encode texts
    # ------------------------------------------------------------------
    texts = [chunk.get("text", "") for chunk in chunks]

    logger.info("Loading SentenceTransformer model '%s' …", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Encoding %d chunks …", len(texts))
    embeddings: np.ndarray = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True
    )

    # Normalise so that inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    # ------------------------------------------------------------------
    # 3. Build FAISS index (Inner Product on unit vectors)
    # ------------------------------------------------------------------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built – %d vectors of dimension %d.", index.ntotal, dim)

    # ------------------------------------------------------------------
    # 4. Persist artefacts
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved to %s", index_path)

    metadata = [
        {field: chunk.get(field) for field in _META_FIELDS}
        for chunk in chunks
    ]
    meta_path = output_dir / "chunk_metadata.jsonl"
    write_jsonl(meta_path, metadata)
    logger.info("Chunk metadata (%d records) saved to %s", len(metadata), meta_path)

    return len(chunks)
