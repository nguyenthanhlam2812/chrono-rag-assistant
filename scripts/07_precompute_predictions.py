import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("precompute_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute ChronoRAG event predictions for processed sentences."
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sentences.jsonl",
        help="Input processed sentences JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/event_predictions.jsonl",
        help="Output predictions JSONL.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "saved_models",
        help="Directory containing Sprint 4 ML baseline .pkl files.",
    )
    parser.add_argument(
        "--binary-model-name",
        default="linearsvm",
        choices=["logreg", "linearsvm", "sgd_log"],
        help="Binary event detector model name.",
    )
    parser.add_argument(
        "--event-type-model-name",
        default="sgd_log",
        choices=["logreg", "linearsvm", "sgd_log"],
        help="Event type classifier model name.",
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary event probability/score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.models.inference import (
        load_sentence_records,
        predict_event_sentences,
        summarize_predictions,
        write_predictions_jsonl,
    )

    binary_model_path = (
        args.models_dir / f"ml_{args.binary_model_name}_event_binary.pkl"
    )
    event_type_model_path = (
        args.models_dir / f"ml_{args.event_type_model_name}_event_type.pkl"
    )

    logger.info("Loading sentences from %s", args.sentences)
    sentence_records = load_sentence_records(args.sentences)
    logger.info("Loaded %d sentences", len(sentence_records))
    logger.info("Binary model: %s", binary_model_path)
    logger.info("Event type model: %s", event_type_model_path)

    predictions = predict_event_sentences(
        sentence_records=sentence_records,
        binary_model_path=binary_model_path,
        event_type_model_path=event_type_model_path,
        binary_model_name=args.binary_model_name,
        event_type_model_name=args.event_type_model_name,
        event_threshold=args.event_threshold,
    )
    write_predictions_jsonl(predictions, args.output)
    summary = summarize_predictions(predictions)

    logger.info("Saved predictions to %s", args.output)
    logger.info("Prediction summary: %s", summary)

if __name__ == "__main__":
    main()
