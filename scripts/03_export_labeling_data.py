import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("export_labeling_data")

def main() -> None:
    logger.info("Sprint 2 component: Exporting data for annotation (Placeholder)")
    logger.info("This feature will be fully implemented in Sprint 2.")

if __name__ == "__main__":
    main()
