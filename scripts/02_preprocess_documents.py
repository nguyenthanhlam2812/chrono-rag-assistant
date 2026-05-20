from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logger import setup_logger
from src.preprocessing.cleaner import clean_text
from src.preprocessing.chunker import chunk_document
from src.preprocessing.sentence_splitter import split_chunks_into_sentences

logger = setup_logger("preprocess_documents_cli")

def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_data_dir"])
    doc_jsonl_path = processed_dir / "documents.jsonl"
    
    if not doc_jsonl_path.exists():
        logger.error(f"Documents file not found at {doc_jsonl_path}. Please run ingestion first.")
        return
        
    logger.info(f"Loading documents from {doc_jsonl_path}...")
    documents = read_jsonl(doc_jsonl_path)
    
    chunk_size = config["preprocessing"].get("chunk_size", 1000)
    chunk_overlap = config["preprocessing"].get("chunk_overlap", 200)
    min_sent_len = config["preprocessing"].get("min_sentence_len", 15)
    
    # Clean and Chunk
    logger.info("Cleaning and chunking documents...")
    all_chunks = []
    for doc in documents:
        doc_copy = doc.copy()
        doc_copy["text"] = clean_text(doc["text"])
        chunks = chunk_document(doc_copy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
        
    chunks_jsonl_path = processed_dir / "chunks.jsonl"
    write_jsonl(chunks_jsonl_path, all_chunks)
    logger.info(f"Saved {len(all_chunks)} chunks to {chunks_jsonl_path}")
    
    # Sentence splitting
    logger.info("Splitting chunks into sentences...")
    all_sentences = split_chunks_into_sentences(all_chunks, min_sentence_len=min_sent_len)
    
    sentences_jsonl_path = processed_dir / "sentences.jsonl"
    write_jsonl(sentences_jsonl_path, all_sentences)
    logger.info(f"Saved {len(all_sentences)} sentences to {sentences_jsonl_path}")
    
    logger.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
