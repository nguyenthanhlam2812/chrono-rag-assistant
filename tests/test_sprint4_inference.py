"""Sprint 4B inference tests for precomputed event predictions."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.version_info >= (3, 13):
    raise unittest.SkipTest(
        "Sprint 4 inference tests require Python 3.10-3.12 with scikit-learn."
    )

try:
    import joblib  # noqa: E402
except Exception as exc:  # pragma: no cover - environment guard
    raise unittest.SkipTest(f"joblib/sklearn stack is not available: {exc}") from exc

from src.models.inference import (  # noqa: E402
    EVENT_TYPES,
    PREDICTION_FIELDS,
    load_sentence_records,
    predict_event_sentences,
    summarize_predictions,
    write_predictions_jsonl,
)
from src.models.ml_baseline import build_model  # noqa: E402


def _sentence_record(sentence_id: str, text: str, topic: str = "rag") -> dict:
    return {
        "sentence_id": sentence_id,
        "doc_id": f"{topic}_001",
        "chunk_id": f"{topic}_001_c0001",
        "topic": topic,
        "year": 2023,
        "source_url": "https://example.com/source",
        "text": text,
    }


class TestSprint4Inference(unittest.TestCase):
    def _train_temp_models(self, tmp_path: Path) -> tuple[Path, Path]:
        train_texts = [
            "In 2020, Lewis et al. proposed a retrieval augmented generation method.",
            "In 2022, LangChain was released as an open source framework.",
            "The model achieved 78 F1 on the NaturalQuestions benchmark.",
            "Recently, retrieval augmented methods have been applied in industry.",
            "This paragraph describes background notation and implementation details.",
            "The appendix contains examples and additional qualitative analysis.",
            "We use standard preprocessing before running the experiment.",
            "The section introduces the notation used throughout the paper.",
        ]
        binary_labels = [1, 1, 1, 1, 0, 0, 0, 0]

        binary_model = build_model("linearsvm", max_features=5000)
        binary_model.fit(train_texts, binary_labels)
        binary_path = tmp_path / "ml_linearsvm_event_binary.pkl"
        joblib.dump(binary_model, binary_path)

        type_model = build_model("sgd_log", max_features=5000)
        type_model.fit(
            train_texts[:4],
            ["method_proposed", "release", "benchmark", "trend_application"],
        )
        type_path = tmp_path / "ml_sgd_log_event_type.pkl"
        joblib.dump(type_model, type_path)
        return binary_path, type_path

    def test_predict_event_sentences_schema_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            binary_path, type_path = self._train_temp_models(tmp_path)
            records = [
                _sentence_record(
                    "rag_001_s0001",
                    "In 2020, Lewis et al. proposed retrieval augmented generation.",
                ),
                _sentence_record(
                    "rag_001_s0002",
                    "This sentence is background context without a chronological event.",
                ),
            ]

            predictions = predict_event_sentences(
                sentence_records=records,
                binary_model_path=binary_path,
                event_type_model_path=type_path,
                binary_model_name="linearsvm",
                event_type_model_name="sgd_log",
                event_threshold=0.5,
            )

            self.assertEqual(len(predictions), 2)
            for row in predictions:
                self.assertTrue(PREDICTION_FIELDS.issubset(row.keys()))
                self.assertIn(row["is_event"], {0, 1})
                self.assertGreaterEqual(row["event_probability"], 0.0)
                self.assertLessEqual(row["event_probability"], 1.0)
                if row["is_event"] == 0:
                    self.assertEqual(row["event_type"], "none")
                else:
                    self.assertIn(row["event_type"], EVENT_TYPES)

            summary = summarize_predictions(predictions)
            self.assertEqual(summary["total_sentences"], 2)
            self.assertEqual(
                summary["predicted_events"] + summary["predicted_non_events"],
                2,
            )

    def test_write_and_load_predictions_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "event_predictions.jsonl"
            predictions = [
                {
                    "sentence_id": "rag_001_s0001",
                    "doc_id": "rag_001",
                    "chunk_id": "rag_001_c0001",
                    "topic": "rag",
                    "year": 2020,
                    "source_url": "https://example.com",
                    "sentence": "In 2020, RAG was proposed.",
                    "is_event": 1,
                    "event_probability": 0.9,
                    "event_type": "method_proposed",
                    "event_type_confidence": 0.8,
                    "date_text": "2020",
                    "normalized_date": "2020",
                    "extracted_year": 2020,
                    "date_confidence": 0.7,
                    "date_source": "sentence_plain_year",
                    "binary_model": "linearsvm",
                    "event_type_model": "sgd_log",
                }
            ]
            write_predictions_jsonl(predictions, output_path)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["sentence_id"], "rag_001_s0001")

    def test_load_sentence_records_requires_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sentences.jsonl"
            path.write_text('{"sentence_id": "s1"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_sentence_records(path)


class TestOnlinePipelinePredictionLoader(unittest.TestCase):
    def test_get_event_predictions_reads_jsonl_by_topic(self):
        import workflows.online_pipeline as online_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "event_predictions.jsonl"
            row = {
                "sentence_id": "agent_001_s0001",
                "doc_id": "agent_001",
                "chunk_id": "agent_001_c0001",
                "topic": "ai_agent",
                "year": 2023,
                "source_url": "https://example.com",
                "sentence": "In 2023, AutoGPT was released.",
                "is_event": 1,
                "event_probability": 0.9,
                "event_type": "release",
                "event_type_confidence": 0.7,
                "binary_model": "linearsvm",
                "event_type_model": "sgd_log",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            old_path = online_pipeline.EVENT_PREDICTIONS_PATH
            try:
                online_pipeline.EVENT_PREDICTIONS_PATH = path
                bundle = online_pipeline.get_event_predictions("AI Agent", limit=3)
            finally:
                online_pipeline.EVENT_PREDICTIONS_PATH = old_path

            self.assertTrue(bundle["is_real"])
            self.assertEqual(bundle["summary"]["events"], 1)
            self.assertEqual(bundle["rows"][0]["doc_id"], "agent_001")


if __name__ == "__main__":
    unittest.main()
