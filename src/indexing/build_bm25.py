"""
Build a BM25 sparse-retrieval index from chunked documents.

Reads chunks.jsonl, tokenises each chunk text (lowercased whitespace
split), constructs a ``rank_bm25.BM25Okapi`` index, and persists the
index and corpus ID list as pickle files.
"""

import pickle
import sys
from pathlib import Path
from typing import List, Union

from rank_bm25 import BM25Okapi

# Ensure project root is on sys.path for sibling imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_jsonl
from src.utils.logger import setup_logger

logger = setup_logger("build_bm25")


def _tokenize(text: str) -> List[str]:
    """Lowercase and split on whitespace – intentionally simple."""
    return text.lower().split()


def build_bm25_index(
    chunks_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> int:
    """Build a BM25Okapi index from *chunks.jsonl*.

    Parameters
    ----------
    chunks_path:
        Path to the ``chunks.jsonl`` file produced by the chunking step.
    output_dir:
        Directory where ``bm25.pkl`` and ``bm25_corpus.pkl`` will be
        written.  Created automatically if it does not exist.

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
        logger.warning("chunks.jsonl not found at %s – skipping BM25 build.", chunks_path)
        return 0

    chunks = read_jsonl(chunks_path)
    if not chunks:
        logger.warning("chunks.jsonl is empty – skipping BM25 build.")
        return 0

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # ------------------------------------------------------------------
    # 2. Tokenise
    # ------------------------------------------------------------------
    tokenized_corpus: List[List[str]] = [
        _tokenize(chunk.get("text", "")) for chunk in chunks
    ]

    # ------------------------------------------------------------------
    # 3. Build BM25 index
    # ------------------------------------------------------------------
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25Okapi index built over %d documents.", len(tokenized_corpus))

    # ------------------------------------------------------------------
    # 4. Persist artefacts
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = output_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("BM25 index saved to %s", bm25_path)

    chunk_ids: List[str] = [chunk.get("chunk_id", "") for chunk in chunks]
    corpus_path = output_dir / "bm25_corpus.pkl"
    with open(corpus_path, "wb") as f:
        pickle.dump(chunk_ids, f)
    logger.info("BM25 corpus IDs (%d entries) saved to %s", len(chunk_ids), corpus_path)

    return len(chunks)
