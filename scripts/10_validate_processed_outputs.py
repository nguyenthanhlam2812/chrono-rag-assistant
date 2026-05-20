from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.schema_validation import validate_processed_data

logger = setup_logger("validate_processed_outputs_cli")

def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_data_dir"])
    
    documents_path = processed_dir / "documents.jsonl"
    chunks_path = processed_dir / "chunks.jsonl"
    sentences_path = processed_dir / "sentences.jsonl"
    
    min_sent_len = config.get("preprocessing", {}).get("min_sentence_len", 15)
    
    logger.info("Starting schema and quality validation of processed outputs...")
    logger.info(f"Documents: {documents_path}")
    logger.info(f"Chunks: {chunks_path}")
    logger.info(f"Sentences: {sentences_path}")
    
    report = validate_processed_data(
        documents_path=documents_path,
        chunks_path=chunks_path,
        sentences_path=sentences_path,
        min_sentence_len=min_sent_len
    )
    
    # Save JSON report
    # The requirement specifies: data/eval/processed_validation_report.json
    eval_dir = PROJECT_ROOT / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "processed_validation_report.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Validation report saved to {report_path}")
    
    # Print validation summary to console
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total Documents: {report['total_documents']}")
    print(f"Total Chunks:    {report['total_chunks']}")
    print(f"Total Sentences: {report['total_sentences']}")
    print(f"Errors Found:    {len(report['errors'])}")
    print(f"Warnings Found:  {len(report['warnings'])}")
    print("-" * 50)
    
    if report["per_document_stats"]:
        print("Per-Document Status:")
        for stat in report["per_document_stats"]:
            print(f"  - {stat['doc_id']} ({stat['topic']}): {stat['status']} "
                  f"(Words: {stat['word_count']}, Chunks: {stat['chunk_count']}, Sentences: {stat['sentence_count']})")
    
    if report["errors"]:
        print("\nErrors:")
        for err in report["errors"]:
            print(f"  [ERROR] {err}")
            
    if report["warnings"]:
        print("\nWarnings:")
        for warn in report["warnings"]:
            print(f"  [WARNING] {warn}")
            
    print("=" * 50)
    
    if report["errors"]:
        logger.error("Validation FAILED due to errors.")
        sys.exit(1)
    else:
        logger.info("Validation PASSED successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
