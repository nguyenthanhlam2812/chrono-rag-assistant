import unittest
import tempfile
import pandas as pd
import json
import pickle
from pathlib import Path

# Add project root to sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train_event_baseline import (
    split_dataset,
    train_binary_classifier,
    train_multiclass_classifier
)
from src.models.predict_events import (
    load_event_models,
    predict_sentences_batch,
    predict_sentence_event
)

class TestSprint4Training(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a mock labeled dataset CSV
        # 10 documents total, 3 topics
        # docs: doc1..doc10
        self.csv_path = self.temp_path / "mock_labeled.csv"
        
        rows = []
        # Let's create some sentences
        # doc1..doc4: topic 'rag'
        # doc5..doc7: topic 'ai_agent'
        # doc8..doc10: topic 'kd'
        doc_topics = {
            "doc1": "rag", "doc2": "rag", "doc3": "rag", "doc4": "rag",
            "doc5": "ai_agent", "doc6": "ai_agent", "doc7": "ai_agent",
            "doc8": "kd", "doc9": "kd", "doc10": "kd"
        }
        
        sentence_idx = 0
        for doc_id, topic in doc_topics.items():
            # each doc has 5 sentences
            for i in range(5):
                sentence_id = f"{doc_id}_s{i}"
                sentence = f"Sentence {i} in document {doc_id} about {topic}."
                # make some events
                is_event = 1 if i == 0 else 0
                event_type = "method_proposed" if i == 0 else "none"
                
                rows.append({
                    "sentence_id": sentence_id,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_c0",
                    "topic": topic,
                    "title": f"Title of {doc_id}",
                    "source_url": f"http://{doc_id}",
                    "year": 2020 + i,
                    "sentence": sentence,
                    "is_event": is_event,
                    "event_type": event_type,
                    "annotator": "Alice",
                    "label_method": "human",
                    "notes": ""
                })
                
        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
        
        self.splits_dir = self.temp_path / "splits"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_dataset(self):
        """Test split_dataset groups by doc_id, stratifies by topic, and output correct ratios."""
        train_df, val_df, test_df = split_dataset(
            self.csv_path,
            self.splits_dir,
            seed=42,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Verify split files are written
        self.assertTrue((self.splits_dir / "train.csv").exists())
        self.assertTrue((self.splits_dir / "val.csv").exists())
        self.assertTrue((self.splits_dir / "test.csv").exists())
        
        # Group by doc_id and check they do not overlap
        train_docs = set(train_df["doc_id"].unique())
        val_docs = set(val_df["doc_id"].unique())
        test_docs = set(test_df["doc_id"].unique())
        
        self.assertTrue(train_docs.isdisjoint(val_docs))
        self.assertTrue(train_docs.isdisjoint(test_docs))
        self.assertTrue(val_docs.isdisjoint(test_docs))
        
        # Topic stratification verification
        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            topics = df["topic"].unique()
            self.assertEqual(len(topics), 3, f"Expected all 3 topics in {name} split")

    def test_training_and_inference_pipeline(self):
        """Test end-to-end training and prediction logic."""
        train_df, val_df, test_df = split_dataset(
            self.csv_path,
            self.splits_dir,
            seed=42,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Train binary event detector
        bin_vect, bin_model, bin_f1 = train_binary_classifier(train_df, val_df, seed=42)
        self.assertIsNotNone(bin_vect)
        self.assertIsNotNone(bin_model)
        
        # Train multiclass event type classifier
        mc_vect, mc_model = train_multiclass_classifier(train_df, val_df, seed=42)
        self.assertIsNotNone(mc_vect)
        self.assertIsNotNone(mc_model)
        
        # Save models to temp directory
        models_dir = self.temp_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        with open(models_dir / "event_detector.pkl", "wb") as f:
            pickle.dump({"vectorizer": bin_vect, "model": bin_model}, f)
        with open(models_dir / "type_classifier.pkl", "wb") as f:
            pickle.dump({"vectorizer": mc_vect, "model": mc_model}, f)
            
        # Load and run predictions
        models = load_event_models(models_dir)
        self.assertIn("binary_vectorizer", models)
        self.assertIn("binary_model", models)
        self.assertIn("multiclass_vectorizer", models)
        self.assertIn("multiclass_model", models)
        
        # Predict on sentences
        test_sentences = [
            "This is a sentence proposed by some method in 2023.",
            "Normal text without events."
        ]
        
        batch_results = predict_sentences_batch(test_sentences, models)
        self.assertEqual(len(batch_results), 2)
        
        for res in batch_results:
            self.assertIn("is_event", res)
            self.assertIn("event_prob", res)
            self.assertIn("event_type", res)
            self.assertIn("type_confidence", res)
            self.assertTrue(isinstance(res["is_event"], int))
            self.assertTrue(isinstance(res["event_prob"], float))
            self.assertTrue(isinstance(res["event_type"], str))
            self.assertTrue(isinstance(res["type_confidence"], float))
            
            # Semantic consistency check
            if res["is_event"] == 0:
                self.assertEqual(res["event_type"], "none")
                
        # Predict single sentence
        single_res = predict_sentence_event(test_sentences[0], models)
        self.assertEqual(single_res["is_event"], batch_results[0]["is_event"])

if __name__ == "__main__":
    unittest.main()
