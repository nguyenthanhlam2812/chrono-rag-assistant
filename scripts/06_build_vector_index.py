import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("build_vector_index")

def main() -> None:
    logger.info("Sprint 5 component: Building FAISS and BM25 index (Placeholder)")
    logger.info("This feature will be fully implemented in Sprint 5.")

if __name__ == "__main__":
    main()
