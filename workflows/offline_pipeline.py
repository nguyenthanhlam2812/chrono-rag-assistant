from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import write_jsonl
from src.utils.logger import setup_logger
from src.ingest.document_loader import load_documents
from src.preprocessing.cleaner import clean_text
from src.preprocessing.chunker import chunk_document
from src.preprocessing.sentence_splitter import split_chunks_into_sentences

logger = setup_logger("offline_pipeline")

def run_offline_pipeline() -> None:
    """
    Run the end-to-end offline pipeline:
    1. Ingest raw documents from raw_data_dir using metadata.csv
    2. Preprocess: clean, chunk, and split into sentences
    3. Export processed datasets to processed_data_dir
    """
    logger.info("Starting Offline Pipeline Setup...")
    
    # 1. Load config
    config = load_config()
    raw_dir = Path(config["paths"]["raw_data_dir"])
    metadata_csv_path = raw_dir / "metadata.csv"
    processed_dir = Path(config["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocessing configs
    chunk_size = config["preprocessing"].get("chunk_size", 1000)
    chunk_overlap = config["preprocessing"].get("chunk_overlap", 200)
    min_sent_len = config["preprocessing"].get("min_sentence_len", 15)
    
    # 2. Ingest
    logger.info("--- Step 1: Ingesting Documents ---")
    documents = load_documents(metadata_csv_path, raw_dir)
    
    if not documents:
        logger.error("No documents loaded. Pipeline aborted.")
        return
        
    doc_jsonl_path = processed_dir / "documents.jsonl"
    logger.info(f"Saving ingested documents to {doc_jsonl_path}")
    write_jsonl(doc_jsonl_path, documents)
    
    # 3. Clean and Chunk
    logger.info("--- Step 2: Cleaning and Chunking Documents ---")
    all_chunks = []
    for doc in documents:
        # Apply cleaner
        doc_copy = doc.copy()
        doc_copy["text"] = clean_text(doc["text"], source_type=doc.get("source_type"))
        
        # Apply chunker
        chunks = chunk_document(doc_copy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
        
    chunks_jsonl_path = processed_dir / "chunks.jsonl"
    logger.info(f"Saving generated chunks ({len(all_chunks)}) to {chunks_jsonl_path}")
    write_jsonl(chunks_jsonl_path, all_chunks)
    
    # 4. Split sentences
    logger.info("--- Step 3: Splitting Chunks into Sentences ---")
    all_sentences = split_chunks_into_sentences(all_chunks, min_sentence_len=min_sent_len)
    
    sentences_jsonl_path = processed_dir / "sentences.jsonl"
    logger.info(f"Saving sentences ({len(all_sentences)}) to {sentences_jsonl_path}")
    write_jsonl(sentences_jsonl_path, all_sentences)
    
    logger.info("Offline Pipeline finished successfully!")

if __name__ == "__main__":
    run_offline_pipeline()
