import sys
import unittest
from pathlib import Path

# Add the project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from workflows.offline_pipeline import run_offline_pipeline

class TestSprint0(unittest.TestCase):
    def test_offline_pipeline(self):
        """Run offline pipeline and assert that output files exist and are not empty."""
        print("Running offline pipeline...")
        run_offline_pipeline()
        
        config = load_config()
        processed_dir = Path(config["paths"]["processed_data_dir"])
        
        docs_file = processed_dir / "documents.jsonl"
        chunks_file = processed_dir / "chunks.jsonl"
        sents_file = processed_dir / "sentences.jsonl"
        
        print("Checking output files existence...")
        self.assertTrue(docs_file.exists(), f"Expected document file does not exist: {docs_file}")
        self.assertTrue(chunks_file.exists(), f"Expected chunks file does not exist: {chunks_file}")
        self.assertTrue(sents_file.exists(), f"Expected sentences file does not exist: {sents_file}")
        
        # Verify non-empty
        self.greater_than_zero(docs_file)
        self.greater_than_zero(chunks_file)
        self.greater_than_zero(sents_file)
        print("Sprint 0 offline pipeline validation passed!")
        
    def greater_than_zero(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0, f"File {path} is empty")

if __name__ == "__main__":
    unittest.main()
