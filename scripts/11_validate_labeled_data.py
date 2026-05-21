import sys
import argparse
import logging
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.label_validation import validate_labeled_data

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="ChronoRAG Labeled Dataset Validator CLI")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the labeled or pre-labeled CSV file to validate."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prelabel",
        choices=["prelabel", "labeled"],
        help="Validation mode: 'prelabel' (default) or 'labeled'."
    )
    parser.add_argument(
        "--sentences-jsonl",
        type=str,
        default=str(PROJECT_ROOT / "data/processed/sentences.jsonl"),
        help="Path to the processed sentences.jsonl file for reference checks."
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.input)
    sentences_path = Path(args.sentences_jsonl)
    
    logging.info(f"Validating file: {csv_path}")
    logging.info(f"Validation mode: {args.mode}")
    logging.info(f"Using sentences database: {sentences_path}")
    
    report = validate_labeled_data(csv_path, sentences_path, mode=args.mode)
    
    # Print statistics summary
    stats = report["stats"]
    print("\n" + "="*50)
    print(f" DATASET STATS SUMMARY ({args.mode} mode)")
    print("="*50)
    print(f"Total Rows: {stats['total_rows']}")
    
    print("\nTopic Distribution:")
    for topic, count in sorted(stats["topic_counts"].items()):
        ratio = count / stats["total_rows"]
        print(f"  - {topic}: {count} ({ratio:.2%})")
        
    print("\nTop 5 Document Contributions:")
    sorted_docs = sorted(stats["doc_counts"].items(), key=lambda x: x[1], reverse=True)
    for doc_id, count in sorted_docs[:5]:
        ratio = count / stats["total_rows"]
        print(f"  - {doc_id}: {count} ({ratio:.2%})")
    if len(sorted_docs) > 5:
        print(f"  - ... and {len(sorted_docs) - 5} other documents")

    print("\nis_event Distribution:")
    for val, count in sorted(stats["is_event_counts"].items()):
        ratio = count / stats["total_rows"]
        print(f"  - '{val}': {count} ({ratio:.2%})")
        
    print("\nevent_type Distribution:")
    for val, count in sorted(stats["event_type_counts"].items()):
        ratio = count / stats["total_rows"]
        print(f"  - '{val}': {count} ({ratio:.2%})")
        
    print("\nlabel_method Distribution:")
    for val, count in sorted(stats["label_method_counts"].items()):
        ratio = count / stats["total_rows"]
        print(f"  - '{val}': {count} ({ratio:.2%})")
        
    print("="*50)

    # Print Warnings
    if report["warnings"]:
        print(f"\nWARNINGS ({len(report['warnings'])} found):")
        for warning in report["warnings"]:
            print(f"  [WARN] {warning}")
    else:
        print("\nNo warnings found.")

    # Print Errors
    if report["errors"]:
        print(f"\nERRORS ({len(report['errors'])} found):")
        for error in report["errors"]:
            print(f"  [ERROR] {error}")
        print("\nValidation failed!")
        sys.exit(1)
    else:
        print("\nValidation successful (0 errors)!")
        sys.exit(0)

if __name__ == "__main__":
    main()
