"""
Script 08: Build timeline from pre-computed predictions.

Loads predictions.jsonl, clusters duplicate events, and generates
timeline.json files (per-topic and combined).

Usage:
    python scripts/08_build_timeline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.timeline.timeline_builder import build_all_timelines

logger = setup_logger("build_timeline")


def main() -> None:
    config = load_config()

    processed_dir = Path(config["paths"]["processed_data_dir"])
    predictions_path = processed_dir / "predictions.jsonl"

    if not predictions_path.exists():
        logger.error(f"predictions.jsonl not found at {predictions_path}")
        logger.error("Run scripts/07_precompute_predictions.py first.")
        return

    # Timeline config
    sim_threshold = config.get("timeline", {}).get("similarity_threshold", 0.78)
    year_diff = config.get("timeline", {}).get("year_diff_threshold", 1)
    embedding_model = config.get("indexing", {}).get("embedding_model", "all-MiniLM-L6-v2")

    output_dir = processed_dir

    logger.info("=" * 60)
    logger.info("Building timelines from predictions")
    logger.info(f"  Predictions: {predictions_path}")
    logger.info(f"  Similarity threshold: {sim_threshold}")
    logger.info(f"  Year diff threshold: {year_diff}")
    logger.info(f"  Embedding model: {embedding_model}")
    logger.info("=" * 60)

    combined = build_all_timelines(
        predictions_path=predictions_path,
        output_dir=output_dir,
        similarity_threshold=sim_threshold,
        year_diff_threshold=year_diff,
        model_name=embedding_model,
    )

    total_events = sum(
        t.get("total_events", 0) for t in combined.get("timelines", {}).values()
    )

    logger.info("=" * 60)
    logger.info(f"Timeline build complete. Total events across all topics: {total_events}")
    for topic, tl in combined.get("timelines", {}).items():
        logger.info(f"  [{topic}] {tl.get('total_events', 0)} events")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
