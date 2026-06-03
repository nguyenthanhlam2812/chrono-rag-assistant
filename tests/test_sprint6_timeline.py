from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.timeline.timeline_builder import build_timeline, load_jsonl, write_timeline_json


class TestTimelineBuilder(unittest.TestCase):
    def test_builds_deduplicated_topic_timeline(self):
        predictions = [
            {
                "sentence_id": "s1",
                "doc_id": "rag_001",
                "chunk_id": "c1",
                "topic": "rag",
                "sentence": "In 2020, Lewis et al. introduced Retrieval-Augmented Generation for knowledge-intensive NLP tasks.",
                "is_event": 1,
                "event_probability": 0.92,
                "event_type": "method_proposed",
                "event_type_confidence": 0.81,
                "normalized_date": "2020",
                "extracted_year": 2020,
                "date_text": "2020",
                "date_confidence": 0.85,
                "date_source": "sentence_context_year",
                "source_url": "https://example.com/rag",
            },
            {
                "sentence_id": "s2",
                "doc_id": "rag_002",
                "chunk_id": "c2",
                "topic": "rag",
                "sentence": "In 2020, Retrieval-Augmented Generation was introduced for knowledge-intensive NLP tasks.",
                "is_event": 1,
                "event_probability": 0.84,
                "event_type": "method_proposed",
                "event_type_confidence": 0.77,
                "normalized_date": "2020",
                "extracted_year": 2020,
                "date_text": "2020",
                "date_confidence": 0.85,
                "date_source": "sentence_context_year",
                "source_url": "https://example.com/rag2",
            },
            {
                "sentence_id": "s3",
                "doc_id": "rag_003",
                "chunk_id": "c3",
                "topic": "rag",
                "sentence": "This sentence describes retrieval background but is not a chronological event.",
                "is_event": 0,
                "event_probability": 0.1,
                "event_type": "none",
                "event_type_confidence": 0.0,
                "normalized_date": "2020",
                "extracted_year": 2020,
                "date_confidence": 0.85,
            },
        ]
        documents = {
            "rag_001": {
                "doc_id": "rag_001",
                "title": "RAG Paper",
                "source_url": "https://example.com/rag",
            },
            "rag_002": {
                "doc_id": "rag_002",
                "title": "RAG Survey",
                "source_url": "https://example.com/rag2",
            },
        }

        timeline = build_timeline(predictions, documents, similarity_threshold=0.45)
        events = timeline["topics"]["rag"]

        self.assertEqual(timeline["metadata"]["events_considered"], 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["year"], 2020)
        self.assertEqual(events[0]["event_type"], "method_proposed")
        self.assertEqual(events[0]["cluster_size"], 2)
        self.assertEqual(events[0]["sources"][0]["title"], "RAG Paper")

    def test_writes_and_loads_timeline_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "timeline.json"
            timeline = {"metadata": {"timeline_events": 0}, "topics": {}}
            write_timeline_json(timeline, output)

            rows_path = Path(tmp) / "rows.jsonl"
            rows_path.write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")

            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), timeline)
            self.assertEqual(load_jsonl(rows_path), [{"a": 1}])


if __name__ == "__main__":
    unittest.main()
