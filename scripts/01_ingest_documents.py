from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import write_jsonl
from src.utils.logger import setup_logger
from src.ingest.document_loader import load_documents

logger = setup_logger("ingest_documents_cli")

def main() -> None:
    config = load_config()
    raw_dir = Path(config["paths"]["raw_data_dir"])
    metadata_csv_path = raw_dir / "metadata.csv"
    processed_dir = Path(config["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting document ingestion via CLI...")
    documents = load_documents(metadata_csv_path, raw_dir)
    
    if documents:
        doc_jsonl_path = processed_dir / "documents.jsonl"
        write_jsonl(doc_jsonl_path, documents)
        logger.info(f"Ingestion completed. Saved {len(documents)} documents to {doc_jsonl_path}")
    else:
        logger.error("No documents loaded.")

if __name__ == "__main__":
    main()
