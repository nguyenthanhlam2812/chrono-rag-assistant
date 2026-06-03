import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.timeline.timeline_builder import (
    build_timeline,
    load_documents_by_id,
    load_jsonl,
    write_timeline_json,
)
from src.utils.logger import setup_logger

logger = setup_logger("build_timeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build timeline.json from event predictions.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "event_predictions.jsonl",
        help="Path to sentence-level event predictions JSONL.",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "documents.jsonl",
        help="Path to processed documents JSONL for title/source metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "timeline.json",
        help="Output timeline JSON path.",
    )
    parser.add_argument("--min-event-probability", type=float, default=0.55)
    parser.add_argument("--min-date-confidence", type=float, default=0.4)
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--max-events-per-topic", type=int, default=30)
    args = parser.parse_args()

    logger.info("Building timeline from %s", args.predictions)
    predictions = load_jsonl(args.predictions)
    documents_by_id = load_documents_by_id(args.documents)
    timeline = build_timeline(
        predictions,
        documents_by_id,
        min_event_probability=args.min_event_probability,
        min_date_confidence=args.min_date_confidence,
        similarity_threshold=args.similarity_threshold,
        max_events_per_topic=args.max_events_per_topic,
    )
    write_timeline_json(timeline, args.output)
    logger.info("Timeline saved to %s", args.output)
    logger.info("Timeline metadata: %s", timeline["metadata"])

if __name__ == "__main__":
    main()
