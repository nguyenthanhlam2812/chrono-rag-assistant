import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.bm25_index import build_bm25_index
from src.indexing.vector_index import build_faiss_index
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("build_vector_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ChronoRAG BM25 and FAISS indexes from processed chunks."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Input chunks.jsonl path. Defaults to config paths.processed_data_dir/chunks.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output vector DB directory. Defaults to config paths.vector_db_dir.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="SentenceTransformer model for FAISS. Defaults to config indexing.embedding_model.",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Only build BM25. Useful on weak machines or before downloading embedding models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    processed_dir = PROJECT_ROOT / config["paths"]["processed_data_dir"]
    chunks_path = args.chunks or (processed_dir / "chunks.jsonl")
    output_dir = args.output_dir or (PROJECT_ROOT / config["paths"]["vector_db_dir"])
    embedding_model = args.embedding_model or config["indexing"]["embedding_model"]

    logger.info("Building retrieval indexes")
    logger.info("Chunks: %s", chunks_path)
    logger.info("Output dir: %s", output_dir)

    summary = {
        "chunks_path": str(chunks_path),
        "output_dir": str(output_dir),
        "bm25": build_bm25_index(chunks_path, output_dir),
        "faiss": None,
    }
    logger.info("BM25 indexed %d chunks", summary["bm25"]["chunks_indexed"])

    if args.skip_faiss:
        logger.info("Skipping FAISS build by request")
    else:
        logger.info("Building FAISS index with embedding model %s", embedding_model)
        summary["faiss"] = build_faiss_index(
            chunks_path=chunks_path,
            output_dir=output_dir,
            model_name=embedding_model,
        )
        logger.info("FAISS indexed %d chunks", summary["faiss"]["chunks_indexed"])

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "index_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Index summary saved to %s", summary_path)

if __name__ == "__main__":
    main()
