import sys
import logging
from pathlib import Path

# Add src/ to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.labeling_exporter import LabelingExporter

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    config_path = PROJECT_ROOT / "configs/labeling_config.yaml"
    sentences_path = PROJECT_ROOT / "data/processed/sentences.jsonl"
    metadata_path = PROJECT_ROOT / "data/raw/metadata.csv"
    output_dir = PROJECT_ROOT / "data/labeled"

    logging.info("Starting labeling dataset exporter...")
    logging.info(f"Config path: {config_path}")
    logging.info(f"Sentences path: {sentences_path}")
    logging.info(f"Metadata path: {metadata_path}")
    logging.info(f"Output directory: {output_dir}")

    exporter = LabelingExporter(
        config_path=config_path,
        sentences_path=sentences_path,
        metadata_path=metadata_path
    )

    n_candidates, n_sample = exporter.export(output_dir)
    logging.info(f"Labeling data export completed successfully!")
    logging.info(f"Generated {n_candidates} rows in labeling_candidates.csv")
    logging.info(f"Generated {n_sample} rows in labeling_candidates_sample.csv")

if __name__ == "__main__":
    main()
