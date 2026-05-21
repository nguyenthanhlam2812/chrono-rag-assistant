import sys
import unittest
import tempfile
import json
import subprocess
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.schema_validation import validate_processed_data
from src.utils.config import load_config

class TestProcessedOutputsValidation(unittest.TestCase):
    """Unit tests for processed schema & quality validation."""

    def test_current_processed_outputs_pass_validation(self):
        """1. Real current processed outputs should pass validation (with at most warnings, no errors)."""
        config = load_config()
        processed_dir = PROJECT_ROOT / Path(config["paths"]["processed_data_dir"])
        
        doc_path = processed_dir / "documents.jsonl"
        chunk_path = processed_dir / "chunks.jsonl"
        sent_path = processed_dir / "sentences.jsonl"
        
        # In case the files do not exist yet, skip the test
        if not doc_path.exists() or not chunk_path.exists() or not sent_path.exists():
            self.skipTest("Real processed outputs do not exist yet.")
            
        report = validate_processed_data(doc_path, chunk_path, sent_path)
        self.assertEqual(len(report["errors"]), 0, f"Expected no errors, got: {report['errors']}")
        self.assertGreater(report["total_documents"], 0)
        self.assertGreater(report["total_chunks"], 0)
        self.assertGreater(report["total_sentences"], 0)

    def test_missing_required_field_fails(self):
        """2. Document missing a required field should fail validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            doc_path = tmp_path / "documents.jsonl"
            chunk_path = tmp_path / "chunks.jsonl"
            sent_path = tmp_path / "sentences.jsonl"
            
            # Missing "title"
            doc = {
                "doc_id": "doc_1",
                "topic": "rag",
                "source_type": "paper",
                "source_url": "http://example.com",
                "year": 2024,
                "authors": ["Author"],
                "text": "This is a document with enough words to pass the length checks. " * 20,
                "local_path": "path.txt",
                "retrieved_at": "2024-01-01"
            }
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
                
            chunk_path.touch()
            sent_path.touch()
            
            report = validate_processed_data(doc_path, chunk_path, sent_path)
            self.assertGreater(len(report["errors"]), 0)
            self.assertTrue(any("missing required field" in err.lower() for err in report["errors"]))

    def test_duplicate_doc_id_fails(self):
        """3. Duplicate doc_id should fail validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            doc_path = tmp_path / "documents.jsonl"
            chunk_path = tmp_path / "chunks.jsonl"
            sent_path = tmp_path / "sentences.jsonl"
            
            doc1 = {
                "doc_id": "doc_1", "title": "Title 1", "topic": "rag", "source_type": "paper",
                "source_url": "http://example.com", "year": 2024, "authors": ["Author"],
                "text": "This is a document with enough words to pass the length checks. " * 20,
                "local_path": "path.txt", "retrieved_at": "2024-01-01"
            }
            doc2 = doc1.copy()
            doc2["title"] = "Title 2"
            
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc1) + "\n")
                f.write(json.dumps(doc2) + "\n")
                
            chunk_path.touch()
            sent_path.touch()
            
            report = validate_processed_data(doc_path, chunk_path, sent_path)
            self.assertGreater(len(report["errors"]), 0)
            self.assertTrue(any("duplicate doc_id" in err.lower() for err in report["errors"]))

    def test_empty_document_text_fails(self):
        """4. Document with empty text should fail validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            doc_path = tmp_path / "documents.jsonl"
            chunk_path = tmp_path / "chunks.jsonl"
            sent_path = tmp_path / "sentences.jsonl"
            
            doc = {
                "doc_id": "doc_1", "title": "Title 1", "topic": "rag", "source_type": "paper",
                "source_url": "http://example.com", "year": 2024, "authors": ["Author"],
                "text": "", # Empty
                "local_path": "path.txt", "retrieved_at": "2024-01-01"
            }
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
                
            chunk_path.touch()
            sent_path.touch()
            
            report = validate_processed_data(doc_path, chunk_path, sent_path)
            self.assertGreater(len(report["errors"]), 0)
            self.assertTrue(any("empty" in err.lower() or "none" in err.lower() for err in report["errors"]))

    def test_short_document_text_fails(self):
        """5. Document text with < 100 words should be reported as error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            doc_path = tmp_path / "documents.jsonl"
            chunk_path = tmp_path / "chunks.jsonl"
            sent_path = tmp_path / "sentences.jsonl"
            
            doc = {
                "doc_id": "doc_1", "title": "Title 1", "topic": "rag", "source_type": "paper",
                "source_url": "http://example.com", "year": 2024, "authors": ["Author"],
                "text": "Short text.", # less than 100 words
                "local_path": "path.txt", "retrieved_at": "2024-01-01"
            }
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
                
            chunk_path.touch()
            sent_path.touch()
            
            report = validate_processed_data(doc_path, chunk_path, sent_path)
            self.assertGreater(len(report["errors"]), 0)
            self.assertTrue(any("word count is too low" in err.lower() for err in report["errors"]))

    def test_chunk_unknown_doc_id_fails(self):
        """6. Chunk referencing an unknown doc_id should fail validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            doc_path = tmp_path / "documents.jsonl"
            chunk_path = tmp_path / "chunks.jsonl"
            sent_path = tmp_path / "sentences.jsonl"
            
            doc = {
                "doc_id": "doc_1", "title": "Title 1", "topic": "rag", "source_type": "paper",
                "source_url": "http://example.com", "year": 2024, "authors": ["Author"],
                "text": "This is a document with enough words to pass the length checks. " * 20,
                "local_path": "path.txt", "retrieved_at": "2024-01-01"
            }
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
                
            # Wrong doc_id "doc_2"
            chunk = {
                "chunk_id": "chunk_1", "doc_id": "doc_2", "topic": "rag", "title": "Title 1",
                "chunk_index": 1, "text": "Some text content here.", "start_char": 0, "end_char": 20,
                "source_url": "http://example.com", "year": 2024
            }
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(chunk) + "\n")
                
            sent_path.touch()
            
            report = validate_processed_data(doc_path, chunk_path, sent_path)
            self.assertGreater(len(report["errors"]), 0)
            self.assertTrue(any("unknown doc_id" in err.lower() for err in report["errors"]))

    def test_sentence_unknown_chunk_id_fails(self):
        """7. Sentence referencing an unknown chunk_id should fail validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            doc_path = tmp_path / "documents.jsonl"
            chunk_path = tmp_path / "chunks.jsonl"
            sent_path = tmp_path / "sentences.jsonl"
            
            doc = {
                "doc_id": "doc_1", "title": "Title 1", "topic": "rag", "source_type": "paper",
                "source_url": "http://example.com", "year": 2024, "authors": ["Author"],
                "text": "This is a document with enough words to pass the length checks. " * 20,
                "local_path": "path.txt", "retrieved_at": "2024-01-01"
            }
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
                
            chunk = {
                "chunk_id": "chunk_1", "doc_id": "doc_1", "topic": "rag", "title": "Title 1",
                "chunk_index": 1, "text": "Some text content here.", "start_char": 0, "end_char": 20,
                "source_url": "http://example.com", "year": 2024
            }
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(chunk) + "\n")
                
            # Sentence referencing "chunk_2"
            sent = {
                "sentence_id": "sent_1", "doc_id": "doc_1", "chunk_id": "chunk_2",
                "topic": "rag", "sentence_index": 1, "text": "This is a sentence.",
                "source_url": "http://example.com", "year": 2024
            }
            with open(sent_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(sent) + "\n")
                
            report = validate_processed_data(doc_path, chunk_path, sent_path)
            self.assertGreater(len(report["errors"]), 0)
            self.assertTrue(any("unknown chunk_id" in err.lower() for err in report["errors"]))

    def test_html_script_noise_is_detected(self):
        """8. Obvious script/style/comment noise should be detected case-insensitively."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            doc_path = tmp_path / "documents.jsonl"
            chunk_path = tmp_path / "chunks.jsonl"
            sent_path = tmp_path / "sentences.jsonl"
            
            # Text contains an uppercase script tag and a comment.
            doc = {
                "doc_id": "doc_1", "title": "Title 1", "topic": "rag", "source_type": "paper",
                "source_url": "http://example.com", "year": 2024, "authors": ["Author"],
                "text": "This text contains <SCRIPT>alert('x')</SCRIPT> tag and <!-- comment -->",
                "local_path": "path.txt", "retrieved_at": "2024-01-01"
            }
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
                
            chunk_path.touch()
            sent_path.touch()
            
            report = validate_processed_data(doc_path, chunk_path, sent_path)
            self.assertGreater(len(report["errors"]), 0)
            self.assertTrue(any("contains '<script'" in err.lower() for err in report["errors"]))

    def test_document_badge_noise_is_detected(self):
        """Document-level badge noise should also be detected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            doc_path = tmp_path / "documents.jsonl"
            chunk_path = tmp_path / "chunks.jsonl"
            sent_path = tmp_path / "sentences.jsonl"

            doc = {
                "doc_id": "doc_1", "title": "Title 1", "topic": "rag", "source_type": "github",
                "source_url": "http://example.com", "year": 2024, "authors": ["Author"],
                "text": ("This is a document with enough words to pass the length checks. " * 20)
                        + " https://img.shields.io/badge/build-passing-green",
                "local_path": "path.md", "retrieved_at": "2024-01-01"
            }
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")

            chunk_path.touch()
            sent_path.touch()

            report = validate_processed_data(doc_path, chunk_path, sent_path)
            self.assertGreater(len(report["errors"]), 0)
            self.assertTrue(any("img.shields.io" in err.lower() for err in report["errors"]))

    def test_mojibake_markers_are_detected(self):
        """Document text containing mojibake markers should fail validation."""
        mojibake_examples = ["â", "Â", "Ã", "Å", "Ä", "Ō", "Ć", "┬", "\uFFFD", "ðŸ"]
        for marker in mojibake_examples:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                doc_path = tmp_path / "documents.jsonl"
                chunk_path = tmp_path / "chunks.jsonl"
                sent_path = tmp_path / "sentences.jsonl"

                doc = {
                    "doc_id": "doc_1", "title": "Title 1", "topic": "rag", "source_type": "paper",
                    "source_url": "http://example.com", "year": 2024, "authors": ["Author"],
                    "text": "This is a document with enough words to pass the length checks. " * 20 + f" Error marker: {marker}",
                    "local_path": "path.txt", "retrieved_at": "2024-01-01"
                }
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(doc) + "\n")

                chunk_path.touch()
                sent_path.touch()

                report = validate_processed_data(doc_path, chunk_path, sent_path)
                self.assertGreater(
                    len(report["errors"]), 
                    0, 
                    f"Expected validation error for mojibake marker '{marker}' in text, but none found."
                )

    def test_cli_script_runs_and_writes_report(self):
        """9. CLI script should execute successfully and write JSON report to data/eval/processed_validation_report.json."""
        cmd = [sys.executable, "scripts/10_validate_processed_outputs.py"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        # Verify the command returns 0 exit code if real outputs are correct
        self.assertEqual(res.returncode, 0, f"CLI script failed: {res.stderr}\nStdout: {res.stdout}")
        
        # Verify validation report file was written
        report_path = PROJECT_ROOT / "data" / "eval" / "processed_validation_report.json"
        self.assertTrue(report_path.exists(), "Validation report file was not created.")
        
        # Verify it is valid JSON with expected fields
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.assertIn("total_documents", data)
        self.assertIn("total_chunks", data)
        self.assertIn("total_sentences", data)
        self.assertIn("errors", data)
        self.assertIn("warnings", data)
        self.assertIn("per_document_stats", data)

if __name__ == "__main__":
    unittest.main()
