"""
Script 07: Pre-compute event predictions for all sentences.

Loads sentences.jsonl, runs the trained ML models (event detection +
event type classification), extracts dates, and writes predictions.jsonl.

Usage:
    python scripts/07_precompute_predictions.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logger import setup_logger
from src.models.predict_events import load_event_models, predict_sentences_batch
from src.timeline.date_extractor import extract_date

logger = setup_logger("precompute_predictions")


def main() -> None:
    config = load_config()

    processed_dir = Path(config["paths"]["processed_data_dir"])
    saved_models_dir = Path(config["paths"]["saved_models_dir"])
    sentences_path = processed_dir / "sentences.jsonl"
    output_path = processed_dir / "predictions.jsonl"

    # --- validate inputs -------------------------------------------------
    if not sentences_path.exists():
        logger.error(f"sentences.jsonl not found at {sentences_path}")
        logger.error("Run the preprocessing pipeline first (scripts 01 & 02).")
        return

    if not (saved_models_dir / "event_detector.pkl").exists():
        logger.error(f"Trained models not found in {saved_models_dir}")
        logger.error("Run scripts/04_train_ml_classifier.py first.")
        return

    # --- load data -------------------------------------------------------
    logger.info(f"Loading sentences from {sentences_path} ...")
    sentences = read_jsonl(sentences_path)
    logger.info(f"Loaded {len(sentences)} sentences.")

    if not sentences:
        logger.warning("No sentences to predict. Exiting.")
        return

    # --- load models -----------------------------------------------------
    logger.info(f"Loading models from {saved_models_dir} ...")
    models = load_event_models(saved_models_dir)
    logger.info("Models loaded successfully.")

    # --- batch predict ---------------------------------------------------
    texts = [s.get("text", "") for s in sentences]
    BATCH_SIZE = 500
    all_predictions: list[dict] = []

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_preds = predict_sentences_batch(batch_texts, models)

        for i, pred in enumerate(batch_preds):
            idx = batch_start + i
            sent = sentences[idx]

            # Extract date for this sentence
            doc_year = sent.get("year")
            if doc_year is not None:
                try:
                    doc_year = int(doc_year)
                except (ValueError, TypeError):
                    doc_year = None

            date_info = extract_date(sent.get("text", ""), document_year=doc_year)

            record = {
                "sentence_id": sent.get("sentence_id", ""),
                "doc_id": sent.get("doc_id", ""),
                "chunk_id": sent.get("chunk_id", ""),
                "topic": sent.get("topic", ""),
                "text": sent.get("text", ""),
                "is_event": pred["is_event"],
                "event_prob": round(pred["event_prob"], 4),
                "event_type": pred["event_type"],
                "type_confidence": round(pred["type_confidence"], 4),
                "date_text": date_info["date_text"],
                "normalized_date": date_info["normalized_date"],
                "extracted_year": date_info["extracted_year"],
                "date_confidence": date_info["date_confidence"],
                "date_source": date_info["date_source"],
                "source_url": sent.get("source_url", ""),
                "source_year": doc_year,
            }
            all_predictions.append(record)

        logger.info(
            f"  Predicted batch {batch_start+1}-{batch_end} / {len(texts)}"
        )

    # --- summary stats ---------------------------------------------------
    n_events = sum(1 for p in all_predictions if p["is_event"] == 1)
    n_with_date = sum(
        1 for p in all_predictions if p["is_event"] == 1 and p["extracted_year"] is not None
    )
    logger.info(f"Prediction summary: {n_events} events detected out of {len(all_predictions)} sentences.")
    logger.info(f"  Events with extracted date: {n_with_date}")

    # --- write output ----------------------------------------------------
    write_jsonl(output_path, all_predictions)
    logger.info(f"Predictions saved to {output_path}")

    # --- also save a compact metrics summary -----------------------------
    eval_dir = Path(config["paths"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_sentences": len(all_predictions),
        "total_events": n_events,
        "events_with_date": n_with_date,
        "event_rate": round(n_events / max(len(all_predictions), 1), 4),
    }
    summary_path = eval_dir / "precompute_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Pre-compute summary saved to {summary_path}")


if __name__ == "__main__":
    main()
