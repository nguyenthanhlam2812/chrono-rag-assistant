import unittest
import tempfile
import csv
import json
from pathlib import Path

# Add project root to sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.label_validation import validate_labeled_data

class TestLabelValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a mock sentences.jsonl file
        self.sentences_jsonl_path = self.temp_path / "sentences.jsonl"
        self.mock_sentence_ids = [f"doc1_s{i}" for i in range(10)] + [f"doc2_s{i}" for i in range(10)]
        with open(self.sentences_jsonl_path, "w", encoding="utf-8") as f:
            for s_id in self.mock_sentence_ids:
                f.write(json.dumps({"sentence_id": s_id}) + "\n")
                
        # Define headers
        self.headers = [
            "sentence_id", "doc_id", "chunk_id", "topic", "title", "source_url",
            "year", "sentence", "is_event", "event_type", "annotator", "label_method", "notes"
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, filename, rows):
        path = self.temp_path / filename
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            for row in rows:
                writer.writerow(row)
        return path

    def test_prelabel_mode_valid(self):
        """Pre-labeled CSV with empty label fields should pass validation with 0 errors."""
        rows = [
            ["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""],
            ["doc1_s1", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "Another fine sentence that is valid and long enough.", "", "", "", "", ""]
        ]
        csv_path = self.write_csv("valid_prelabel.csv", rows)
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="prelabel")
        self.assertEqual(len(report["errors"]), 0, f"Expected no errors, got: {report['errors']}")

    def test_labeled_mode_valid(self):
        """Labeled CSV with valid combinations should pass validation with 0 errors."""
        rows = [
            ["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "1", "method_proposed", "Alice", "human", ""],
            ["doc1_s1", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "Another fine sentence that is valid and long enough.", "0", "none", "Bob", "human", ""]
        ]
        csv_path = self.write_csv("valid_labeled.csv", rows)
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="labeled")
        self.assertEqual(len(report["errors"]), 0, f"Expected no errors, got: {report['errors']}")

    def test_missing_column(self):
        """CSV with missing required columns should trigger errors."""
        csv_path = self.temp_path / "bad_cols.csv"
        # Missing 'notes' and 'sentence_id'
        bad_headers = [h for h in self.headers if h not in ["notes", "sentence_id"]]
        with open(csv_path, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(bad_headers)
            writer.writerow(["doc1", "chunk1", "rag", "Title", "http://url", "2024", "Sentence text", "0", "none", "Alice", "human"])
            
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report["errors"]), 0)
        self.assertTrue(any("missing required columns" in err for err in report["errors"]))

    def test_duplicate_sentence_id(self):
        """CSV with duplicate sentence_id should trigger error."""
        rows = [
            ["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""],
            ["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "Another fine sentence that is valid and long enough.", "", "", "", "", ""]
        ]
        csv_path = self.write_csv("dup_id.csv", rows)
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report["errors"]), 0)
        self.assertTrue(any("duplicate sentence_id" in err.lower() for err in report["errors"]))

    def test_nonexistent_sentence_id(self):
        """CSV with sentence_id not in the sentences database should fail validation."""
        rows = [
            ["nonexistent_id", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""]
        ]
        csv_path = self.write_csv("nonexistent.csv", rows)
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report["errors"]), 0)
        self.assertTrue(any("does not exist in processed sentences.jsonl" in err for err in report["errors"]))

    def test_empty_sentence_text(self):
        """CSV with empty sentence text should trigger error."""
        rows = [
            ["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "", "", "", "", "", ""]
        ]
        csv_path = self.write_csv("empty_text.csv", rows)
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report["errors"]), 0)
        self.assertTrue(any("empty sentence text" in err.lower() for err in report["errors"]))

    def test_warning_sentence_length(self):
        """Sentence lengths outside 20-500 character range should emit warnings."""
        rows = [
            ["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "Short.", "", "", "", "", ""],
            ["doc1_s1", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "A" * 501, "", "", "", "", ""]
        ]
        csv_path = self.write_csv("warnings_len.csv", rows)
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="prelabel")
        self.assertEqual(len(report["errors"]), 0)
        self.assertGreater(len(report["warnings"]), 0)
        self.assertTrue(any("length" in warn.lower() for warn in report["warnings"]))

    def test_warning_doc_dominance(self):
        """A single document contributing > 8% of the total dataset should emit a warning."""
        # We write 20 rows, one doc contributing 2 rows (10% of total rows)
        # Wait, the list has 20 sentence_ids in self.mock_sentence_ids. Let's make sure they map correctly.
        rows = []
        for i in range(18):
            rows.append([f"doc2_s{i % 10}", "doc2", "chunk1", "rag", "Title 2", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""])
        # Now add 2 from doc1 (total 20 rows, doc1 = 10%, doc2 = 90%)
        rows.append(["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""])
        rows.append(["doc1_s1", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""])
        
        # We need to make sure all sentence IDs in the CSV are unique to avoid duplicate error
        for idx, row in enumerate(rows):
            row[0] = f"doc1_s{idx % 10}" if idx % 2 == 0 else f"doc2_s{idx % 10}"
        
        # Re-set mock sentence IDs in setUp logic to match whatever we generate
        with open(self.sentences_jsonl_path, "w", encoding="utf-8") as f:
            for idx in range(len(rows)):
                s_id = f"s_{idx}"
                rows[idx][0] = s_id
                f.write(json.dumps({"sentence_id": s_id}) + "\n")
                
        csv_path = self.write_csv("warnings_dominance.csv", rows)
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="prelabel")
        self.assertEqual(len(report["errors"]), 0)
        self.assertGreater(len(report["warnings"]), 0)
        self.assertTrue(any("dominance" in warn.lower() or "contributes" in warn.lower() for warn in report["warnings"]))

    def test_warning_topic_imbalance(self):
        """Topic distribution outside 30.0% - 36.6% should emit a warning."""
        rows = [
            ["s_0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""],
            ["s_1", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""],
            ["s_2", "doc1", "chunk1", "ai_agent", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "", ""]
        ]
        with open(self.sentences_jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"sentence_id": row[0]}) + "\n")
                
        csv_path = self.write_csv("warnings_topic.csv", rows)
        report = validate_labeled_data(csv_path, self.sentences_jsonl_path, mode="prelabel")
        self.assertEqual(len(report["errors"]), 0)
        self.assertGreater(len(report["warnings"]), 0)
        self.assertTrue(any("topic" in warn.lower() or "distribution" in warn.lower() for warn in report["warnings"]))

    def test_labeled_mode_errors(self):
        """Strict labeled mode should flag illegal labels and annotations."""
        # 1. Invalid is_event
        rows1 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "yes", "none", "Alice", "human", ""]]
        csv_path1 = self.write_csv("bad_is_event.csv", rows1)
        report1 = validate_labeled_data(csv_path1, self.sentences_jsonl_path, mode="labeled")
        self.assertGreater(len(report1["errors"]), 0)
        self.assertTrue(any("is_event must be '0' or '1'" in err for err in report1["errors"]))
        
        # 2. Invalid event_type
        rows2 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "1", "invalid_class", "Alice", "human", ""]]
        csv_path2 = self.write_csv("bad_event_type.csv", rows2)
        report2 = validate_labeled_data(csv_path2, self.sentences_jsonl_path, mode="labeled")
        self.assertGreater(len(report2["errors"]), 0)
        self.assertTrue(any("event_type must be one of" in err for err in report2["errors"]))
        
        # 3. is_event = 0, event_type = method_proposed (mismatch)
        rows3 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "0", "method_proposed", "Alice", "human", ""]]
        csv_path3 = self.write_csv("mismatch_0.csv", rows3)
        report3 = validate_labeled_data(csv_path3, self.sentences_jsonl_path, mode="labeled")
        self.assertGreater(len(report3["errors"]), 0)
        self.assertTrue(any("event_type must be 'none' when is_event is '0'" in err for err in report3["errors"]))
        
        # 4. is_event = 1, event_type = none (mismatch)
        rows4 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "1", "none", "Alice", "human", ""]]
        csv_path4 = self.write_csv("mismatch_1.csv", rows4)
        report4 = validate_labeled_data(csv_path4, self.sentences_jsonl_path, mode="labeled")
        self.assertGreater(len(report4["errors"]), 0)
        self.assertTrue(any("event_type must not be 'none' when is_event is '1'" in err for err in report4["errors"]))
        
        # 5. Missing annotator
        rows5 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "0", "none", "", "human", ""]]
        csv_path5 = self.write_csv("empty_annotator.csv", rows5)
        report5 = validate_labeled_data(csv_path5, self.sentences_jsonl_path, mode="labeled")
        self.assertGreater(len(report5["errors"]), 0)
        self.assertTrue(any("annotator is blank" in err for err in report5["errors"]))
        
        # 6. Invalid label_method
        rows6 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "0", "none", "Alice", "invalid_method", ""]]
        csv_path6 = self.write_csv("bad_method.csv", rows6)
        report6 = validate_labeled_data(csv_path6, self.sentences_jsonl_path, mode="labeled")
        self.assertGreater(len(report6["errors"]), 0)
        self.assertTrue(any("label_method must be one of" in err for err in report6["errors"]))

    def test_prelabel_mode_invalid_cases(self):
        """Pre-label mode should reject invalid non-blank values and semantic mismatches."""
        # 1. Invalid is_event
        rows1 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "invalid_val", "", "", "", ""]]
        csv_path1 = self.write_csv("pre_bad_is_event.csv", rows1)
        report1 = validate_labeled_data(csv_path1, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report1["errors"]), 0)
        self.assertTrue(any("is_event must be blank, '0', or '1'" in err for err in report1["errors"]))
        
        # 2. Invalid event_type
        rows2 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "bad_event", "", "", ""]]
        csv_path2 = self.write_csv("pre_bad_event_type.csv", rows2)
        report2 = validate_labeled_data(csv_path2, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report2["errors"]), 0)
        self.assertTrue(any("event_type must be blank or one of" in err for err in report2["errors"]))
        
        # 3. Invalid label_method
        rows3 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "", "", "", "bad_method", ""]]
        csv_path3 = self.write_csv("pre_bad_label_method.csv", rows3)
        report3 = validate_labeled_data(csv_path3, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report3["errors"]), 0)
        self.assertTrue(any("label_method must be blank or one of" in err for err in report3["errors"]))
        
        # 4. Semantic mismatch (is_event = "0" and event_type = "method_proposed")
        rows4 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "0", "method_proposed", "", "", ""]]
        csv_path4 = self.write_csv("pre_mismatch_0.csv", rows4)
        report4 = validate_labeled_data(csv_path4, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report4["errors"]), 0)
        self.assertTrue(any("event_type must be 'none' when is_event is '0'" in err for err in report4["errors"]))

        # 5. Semantic mismatch (is_event = "1" and event_type = "none")
        rows5 = [["doc1_s0", "doc1", "chunk1", "rag", "Title 1", "http://url", "2024", "This is a valid sentence of normal length.", "1", "none", "", "", ""]]
        csv_path5 = self.write_csv("pre_mismatch_1.csv", rows5)
        report5 = validate_labeled_data(csv_path5, self.sentences_jsonl_path, mode="prelabel")
        self.assertGreater(len(report5["errors"]), 0)
        self.assertTrue(any("event_type must not be 'none' when is_event is '1'" in err for err in report5["errors"]))

if __name__ == "__main__":
    unittest.main()
