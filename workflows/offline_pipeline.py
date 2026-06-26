from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import write_jsonl
from src.utils.logger import setup_logger
from src.utils.pipeline_cache import PipelineCache, chained_fingerprint, raw_corpus_fingerprint
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
    extract_tables = config["preprocessing"].get("extract_pdf_tables", False)
    ocr = config["preprocessing"].get("ocr_pdf_images", False)
    pdf_backend = config["preprocessing"].get("pdf_backend", "pymupdf")
    cache_enabled = config["preprocessing"].get("pipeline_cache", True)
    cache = PipelineCache(processed_dir / ".cache", enabled=cache_enabled)
    parser_fingerprint = raw_corpus_fingerprint(
        metadata_csv_path,
        raw_dir,
        {
            "extract_tables": extract_tables,
            "ocr": ocr,
            "pdf_backend": pdf_backend,
        },
    )
    documents_fingerprint = chained_fingerprint(parser_fingerprint, {"stage": "documents_v2"})
    chunks_fingerprint = chained_fingerprint(
        documents_fingerprint,
        {"stage": "chunks_v2", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
    )
    sentences_fingerprint = chained_fingerprint(
        chunks_fingerprint,
        {"stage": "sentences_v2", "min_sentence_len": min_sent_len},
    )
    doc_jsonl_path = processed_dir / "documents.jsonl"
    chunks_jsonl_path = processed_dir / "chunks.jsonl"
    sentences_jsonl_path = processed_dir / "sentences.jsonl"

    # 2. Ingest
    logger.info("--- Step 1: Ingesting Documents ---")
    cleaned_docs = cache.load_jsonl_if_valid("documents", documents_fingerprint, doc_jsonl_path)
    if cleaned_docs is not None:
        logger.info(f"Using cached processed documents ({len(cleaned_docs)})")
    else:
        documents = load_documents(
            metadata_csv_path, raw_dir,
            extract_tables=extract_tables, ocr=ocr, pdf_backend=pdf_backend,
        )

        if not documents:
            logger.error("No documents loaded. Pipeline aborted.")
            return

        # Apply cleaner to documents before saving to documents.jsonl
        cleaned_docs = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_copy["text"] = clean_text(doc["text"], source_type=doc.get("source_type"))
            cleaned_docs.append(doc_copy)

        logger.info(f"Saving processed documents to {doc_jsonl_path}")
        write_jsonl(doc_jsonl_path, cleaned_docs)
        cache.mark_valid("documents", documents_fingerprint, doc_jsonl_path, len(cleaned_docs))
    
    # 3. Clean and Chunk
    logger.info("--- Step 2: Cleaning and Chunking Documents ---")
    all_chunks = cache.load_jsonl_if_valid("chunks", chunks_fingerprint, chunks_jsonl_path)
    if all_chunks is not None:
        logger.info(f"Using cached chunks ({len(all_chunks)})")
    else:
        all_chunks = []
        for doc in cleaned_docs:
            # Apply chunker to the already-cleaned processed document text.
            chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)

        logger.info(f"Saving generated chunks ({len(all_chunks)}) to {chunks_jsonl_path}")
        write_jsonl(chunks_jsonl_path, all_chunks)
        cache.mark_valid("chunks", chunks_fingerprint, chunks_jsonl_path, len(all_chunks))
    
    # 4. Split sentences
    logger.info("--- Step 3: Splitting Chunks into Sentences ---")
    all_sentences = cache.load_jsonl_if_valid("sentences", sentences_fingerprint, sentences_jsonl_path)
    if all_sentences is not None:
        logger.info(f"Using cached sentences ({len(all_sentences)})")
    else:
        all_sentences = split_chunks_into_sentences(all_chunks, min_sentence_len=min_sent_len)

        logger.info(f"Saving sentences ({len(all_sentences)}) to {sentences_jsonl_path}")
        write_jsonl(sentences_jsonl_path, all_sentences)
        cache.mark_valid("sentences", sentences_fingerprint, sentences_jsonl_path, len(all_sentences))
    
    logger.info("Offline Pipeline finished successfully!")

if __name__ == "__main__":
    run_offline_pipeline()
