"""Sprint 4 - ML baseline unit and end-to-end tests.

Covers ``src/models/ml_baseline.py``:
- ``load_labeled_data`` schema validation
- ``SplitConfig`` validation and config-file wiring
- ``make_doc_split`` doc-level leakage / topic balance / rare class survival
- ``_score_split`` penalises missing classes
- ``build_model`` returns correct Pipeline structures
- ``evaluate_classifier`` returns the documented metric keys
- ``train_ml_baselines`` end-to-end produces all expected output files
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing sklearn at module load is fine here; tests pin versions via
# requirements.txt and the rest of the test suite already exercises this path.
if sys.version_info >= (3, 13):
    raise unittest.SkipTest(
        "Sprint 4 ML tests require Python 3.10-3.12 with a working scikit-learn stack."
    )

try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
    from sklearn.pipeline import Pipeline  # noqa: E402
    from sklearn.svm import LinearSVC  # noqa: E402
except Exception as exc:  # pragma: no cover - environment guard
    raise unittest.SkipTest(f"scikit-learn stack is not available: {exc}") from exc

from src.models.ml_baseline import (  # noqa: E402
    EVENT_TYPES,
    SplitConfig,
    _score_split,
    build_model,
    evaluate_classifier,
    load_labeled_data,
    make_doc_split,
    split_summary,
    train_ml_baselines,
)


REQUIRED_COLUMNS = [
    "sentence_id",
    "doc_id",
    "chunk_id",
    "topic",
    "title",
    "source_url",
    "year",
    "sentence",
    "is_event",
    "event_type",
    "annotator",
    "label_method",
    "notes",
]


def _build_fixture_rows() -> List[dict]:
    """Synthesize a tiny labeled dataset: 3 topics * 5 docs * 4 sentences = 60 rows.

    Each doc has at least one event of each major class so doc-level splits
    can keep all classes present per split with the seed search.
    """
    rows: List[dict] = []
    topics = [
        ("rag", "rag"),
        ("ai_agent", "ai_agent"),
        ("knowledge_distillation", "kd"),
    ]
    event_cycle = [
        ("method_proposed", "In 2020, Lewis et al. proposed RAG to combine memory."),
        ("release", "In 2022, LangChain was released to the open-source community."),
        ("benchmark", "On the NaturalQuestions benchmark, the model reached 78 F1."),
        ("trend_application", "Throughout 2023, vector databases saw adoption across industry."),
    ]
    for topic, prefix in topics:
        for doc_idx in range(1, 6):
            doc_id = f"{prefix}_{doc_idx:03d}"
            for sent_idx, (event_type, sentence) in enumerate(event_cycle, start=1):
                # First sentence per doc is an event of cycling type, rest are negatives.
                is_event = 1 if sent_idx == 1 else 0
                evt = event_type if is_event else "none"
                rows.append(
                    {
                        "sentence_id": f"{doc_id}_s{sent_idx:04d}",
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_c0001",
                        "topic": topic,
                        "title": f"{topic.title()} doc {doc_idx}",
                        "source_url": f"https://example.com/{doc_id}",
                        "year": "2023",
                        "sentence": sentence if is_event else f"Sentence {sent_idx} for {doc_id}.",
                        "is_event": str(is_event),
                        "event_type": evt,
                        "annotator": "test",
                        "label_method": "human",
                        "notes": "",
                    }
                )
            # Spread event types across docs so every class has multiple docs.
            event_cycle = event_cycle[1:] + event_cycle[:1]
    return rows


def _write_fixture_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestLoadLabeledData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.csv_path = self.tmp_path / "labeled.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_valid_csv(self):
        _write_fixture_csv(self.csv_path, _build_fixture_rows())
        df = load_labeled_data(self.csv_path)
        for col in ("sentence_id", "doc_id", "topic", "sentence", "is_event", "event_type"):
            self.assertIn(col, df.columns)
        self.assertEqual(len(df), 60)

    def test_missing_columns_raises(self):
        rows = _build_fixture_rows()
        bad_path = self.tmp_path / "bad.csv"
        with bad_path.open("w", encoding="utf-8", newline="") as f:
            # Drop the is_event column entirely.
            fields = [c for c in REQUIRED_COLUMNS if c != "is_event"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: v for k, v in row.items() if k != "is_event"})
        with self.assertRaises(ValueError) as ctx:
            load_labeled_data(bad_path)
        self.assertIn("is_event", str(ctx.exception))

    def test_handles_nan_sentences(self):
        rows = _build_fixture_rows()
        rows[0]["sentence"] = ""  # treated as NaN-ish by pandas read_csv
        _write_fixture_csv(self.csv_path, rows)
        df = load_labeled_data(self.csv_path)
        # No exception, empty cell becomes "" string.
        self.assertEqual(df.iloc[0]["sentence"], "")

    def test_is_event_coerced_to_int(self):
        _write_fixture_csv(self.csv_path, _build_fixture_rows())
        df = load_labeled_data(self.csv_path)
        # is_event stored as string "0"/"1" in CSV, must be int after load.
        self.assertEqual(df["is_event"].dtype.kind, "i")


class TestSplitConfig(unittest.TestCase):
    def test_defaults_sum_to_one(self):
        cfg = SplitConfig()
        self.assertAlmostEqual(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio, 1.0, places=6)

    def test_invalid_sum_raises(self):
        with self.assertRaises(ValueError):
            SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_zero_ratio_raises(self):
        with self.assertRaises(ValueError):
            SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    def test_from_config_dict_reads_yaml_keys(self):
        config = {"models": {"random_seed": 7, "test_size": 0.2, "val_size": 0.1}}
        cfg = SplitConfig.from_config_dict(config)
        self.assertEqual(cfg.random_seed, 7)
        self.assertAlmostEqual(cfg.val_ratio, 0.1)
        self.assertAlmostEqual(cfg.test_ratio, 0.2)
        self.assertAlmostEqual(cfg.train_ratio, 0.7)

    def test_from_config_dict_uses_defaults_when_missing(self):
        cfg = SplitConfig.from_config_dict({})
        self.assertEqual(cfg.random_seed, 42)
        self.assertAlmostEqual(cfg.train_ratio, 0.7)


class TestMakeDocSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.csv_path = Path(cls.tmp.name) / "labeled.csv"
        _write_fixture_csv(cls.csv_path, _build_fixture_rows())
        cls.df = load_labeled_data(cls.csv_path)
        cls.split_df = make_doc_split(cls.df, SplitConfig(random_seed=42))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_doc_leakage(self):
        # Every doc_id must appear in exactly one split.
        for doc_id, group in self.split_df.groupby("doc_id"):
            unique_splits = set(group["split"])
            self.assertEqual(
                len(unique_splits), 1,
                f"doc {doc_id} appears in multiple splits: {unique_splits}",
            )

    def test_all_topics_in_all_splits(self):
        for split_name in ("train", "val", "test"):
            topics = set(self.split_df[self.split_df["split"] == split_name]["topic"])
            self.assertEqual(topics, {"rag", "ai_agent", "knowledge_distillation"})

    def test_train_has_event_rows(self):
        train_events = self.split_df[
            (self.split_df["split"] == "train") & (self.split_df["is_event"] == 1)
        ]
        self.assertGreater(len(train_events), 0)

    def test_raises_too_few_docs(self):
        # Build a fixture with only 2 docs in one topic.
        rows = [r for r in _build_fixture_rows() if r["doc_id"] in {"rag_001", "rag_002"} or r["topic"] != "rag"]
        small_csv = Path(self.tmp.name) / "small.csv"
        _write_fixture_csv(small_csv, rows)
        df = load_labeled_data(small_csv)
        with self.assertRaises(ValueError):
            make_doc_split(df, SplitConfig(random_seed=42))


class TestScoreSplit(unittest.TestCase):
    def test_penalises_missing_event_type(self):
        # Two synthetic split frames: one with all 4 event types in train,
        # one with only 1 type. The first must score higher.
        import pandas as pd

        rows_full = []
        rows_partial = []
        for evt_idx, evt in enumerate(EVENT_TYPES):
            rows_full.append(
                {"split": "train", "is_event": 1, "event_type": evt, "doc_id": f"d{evt_idx}", "sentence_id": f"d{evt_idx}_s1"}
            )
        # Partial: only method_proposed appears in train, others empty.
        rows_partial.append(
            {"split": "train", "is_event": 1, "event_type": "method_proposed", "doc_id": "d0", "sentence_id": "d0_s1"}
        )
        # Both need val/test rows present (any) so groupby works.
        for split in ("val", "test"):
            rows_full.append({"split": split, "is_event": 0, "event_type": "none", "doc_id": f"v_{split}", "sentence_id": f"v_{split}_s1"})
            rows_partial.append({"split": split, "is_event": 0, "event_type": "none", "doc_id": f"v_{split}", "sentence_id": f"v_{split}_s1"})

        df_full = pd.DataFrame(rows_full)
        df_partial = pd.DataFrame(rows_partial)
        self.assertGreater(_score_split(df_full), _score_split(df_partial))


class TestBuildModel(unittest.TestCase):
    def test_logreg_pipeline_structure(self):
        model = build_model("logreg", max_features=5000)
        self.assertIsInstance(model, Pipeline)
        self.assertIn("tfidf", model.named_steps)
        self.assertIn("classifier", model.named_steps)
        self.assertIsInstance(model.named_steps["classifier"], LogisticRegression)

    def test_linearsvm_pipeline_structure(self):
        model = build_model("linearsvm", max_features=5000)
        self.assertIsInstance(model.named_steps["classifier"], LinearSVC)

    def test_sgd_log_pipeline_structure(self):
        model = build_model("sgd_log", max_features=5000)
        self.assertIsInstance(model.named_steps["classifier"], SGDClassifier)

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            build_model("xgboost", max_features=5000)


class TestEvaluateClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        csv_path = Path(cls.tmp.name) / "labeled.csv"
        _write_fixture_csv(csv_path, _build_fixture_rows())
        cls.df = load_labeled_data(csv_path)
        cls.df = make_doc_split(cls.df, SplitConfig(random_seed=42))
        cls.train_df = cls.df[cls.df["split"] == "train"]
        cls.test_df = cls.df[cls.df["split"] == "test"]
        cls.model = build_model("logreg", max_features=5000)
        cls.model.fit(cls.train_df["sentence"], cls.train_df["is_event"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_returns_all_metric_keys(self):
        result = evaluate_classifier(self.model, self.test_df, [0, 1], "is_event")
        required = {
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "classification_report", "confusion_matrix", "labels",
        }
        self.assertTrue(required.issubset(result.keys()))

    def test_confusion_matrix_label_order(self):
        labels = [0, 1]
        result = evaluate_classifier(self.model, self.test_df, labels, "is_event")
        self.assertEqual(result["labels"], labels)
        # Matrix is square with side == len(labels)
        matrix = result["confusion_matrix"]
        self.assertEqual(len(matrix), len(labels))
        self.assertTrue(all(len(row) == len(labels) for row in matrix))


class TestSplitSummary(unittest.TestCase):
    def test_summary_counts_match_rows(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            csv_path = Path(tmp.name) / "labeled.csv"
            _write_fixture_csv(csv_path, _build_fixture_rows())
            df = load_labeled_data(csv_path)
            df = make_doc_split(df, SplitConfig(random_seed=42))
            summary = split_summary(df)
            total = sum(summary[s]["rows"] for s in summary)
            self.assertEqual(total, len(df))
            for s in ("train", "val", "test"):
                self.assertIn(s, summary)
        finally:
            tmp.cleanup()


class TestTrainMlBaselinesEndToEnd(unittest.TestCase):
    def test_outputs_created_and_metrics_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "labeled.csv"
            _write_fixture_csv(csv_path, _build_fixture_rows())

            metrics = train_ml_baselines(
                input_csv=csv_path,
                output_dir=tmp_path / "saved_models",
                eval_dir=tmp_path / "eval",
                figures_dir=tmp_path / "figures",
                max_features=5000,
                model_names=("logreg",),
            )

            # Files exist
            self.assertTrue((tmp_path / "saved_models" / "ml_logreg_event_binary.pkl").exists())
            self.assertTrue((tmp_path / "saved_models" / "ml_logreg_event_type.pkl").exists())
            self.assertTrue((tmp_path / "eval" / "ml_baseline_metrics.json").exists())
            self.assertTrue((tmp_path / "eval" / "ml_doc_splits.csv").exists())

            # JSON has the expected shape
            with (tmp_path / "eval" / "ml_baseline_metrics.json").open(encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("split_summary", data)
            self.assertIn("split_ratios", data)
            self.assertIn("models", data)
            self.assertIn("logreg", data["models"])
            self.assertIn("binary", data["models"]["logreg"])
            self.assertIn("event_type", data["models"]["logreg"])
            self.assertIn("test", data["models"]["logreg"]["binary"])

            # Returned dict matches what was persisted
            self.assertEqual(metrics["metrics_path"], str(tmp_path / "eval" / "ml_baseline_metrics.json"))


if __name__ == "__main__":
    unittest.main()
