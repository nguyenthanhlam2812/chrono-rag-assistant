import os
import sys
import json
import argparse
import pickle
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.models.train_event_baseline import (
    split_dataset,
    train_binary_classifier,
    train_multiclass_classifier
)

logger = setup_logger("train_ml_classifier_cli")

def main() -> None:
    parser = argparse.ArgumentParser(description="ChronoRAG ML Baseline Classifier Training CLI")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the labeled sentences CSV file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save trained model files."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports and plots."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and training."
    )
    
    args = parser.parse_args()
    config = load_config()
    
    # Resolve paths
    input_path = Path(args.input) if args.input else Path(config["paths"]["labeled_data_dir"]) / "labeled_sentences.csv"
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["paths"]["saved_models_dir"])
    eval_dir = Path(args.eval_dir) if args.eval_dir else Path(config["paths"]["eval_dir"])
    
    splits_dir = Path(config["paths"]["labeled_data_dir"]) / "splits"
    
    logger.info("Starting Sprint 4 Training Pipeline...")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output models directory: {output_dir}")
    logger.info(f"Evaluation directory: {eval_dir}")
    
    # 1. Split Dataset
    train_df, val_df, test_df = split_dataset(input_path, splits_dir, seed=args.seed)
    
    # 2. Train Event sentence detector (binary)
    logger.info("Training binary event detector...")
    bin_vectorizer, bin_model, best_val_f1 = train_binary_classifier(train_df, val_df, seed=args.seed)
    
    # 3. Train Event type classifier (multiclass)
    logger.info("Training multiclass event type classifier...")
    mc_vectorizer, mc_model = train_multiclass_classifier(train_df, val_df, seed=args.seed)
    
    # 4. Save model files
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_model_path = output_dir / "event_detector.pkl"
    multiclass_model_path = output_dir / "type_classifier.pkl"
    
    logger.info(f"Saving binary event detector to {binary_model_path}")
    with open(binary_model_path, "wb") as f:
        pickle.dump({"vectorizer": bin_vectorizer, "model": bin_model}, f)
        
    logger.info(f"Saving multiclass event type classifier to {multiclass_model_path}")
    with open(multiclass_model_path, "wb") as f:
        pickle.dump({"vectorizer": mc_vectorizer, "model": mc_model}, f)
        
    # 5. Evaluate on Test Set
    logger.info("Evaluating models on Test Set...")
    X_test_raw = test_df["sentence"].astype(str).tolist()
    y_test_bin = test_df["is_event"].tolist()
    y_test_mc = test_df["event_type"].tolist()
    
    # Transform test set features
    X_test_bin = bin_vectorizer.transform(X_test_raw)
    X_test_mc = mc_vectorizer.transform(X_test_raw)
    
    # Evaluate binary
    bin_preds = bin_model.predict(X_test_bin)
    bin_acc = accuracy_score(y_test_bin, bin_preds)
    bin_precision, bin_recall, bin_f1, _ = precision_recall_fscore_support(
        y_test_bin, bin_preds, average="binary", zero_division=0
    )
    
    logger.info(f"Test Binary Metrics -> Accuracy: {bin_acc:.4f}, Precision: {bin_precision:.4f}, Recall: {bin_recall:.4f}, F1: {bin_f1:.4f}")
    
    # Evaluate multiclass
    mc_preds = mc_model.predict(X_test_mc)
    mc_acc = accuracy_score(y_test_mc, mc_preds)
    
    # Compute macro metrics
    mc_precision, mc_recall, mc_macro_f1, _ = precision_recall_fscore_support(
        y_test_mc, mc_preds, average="macro", zero_division=0
    )
    
    # Detailed classification report
    mc_report = classification_report(y_test_mc, mc_preds, output_dict=True, zero_division=0)
    
    logger.info(f"Test Multiclass Metrics -> Accuracy: {mc_acc:.4f}, Macro-F1: {mc_macro_f1:.4f}")
    
    # Generate Confusion Matrix
    cm_labels = sorted(list(set(y_test_mc) | set(mc_preds)))
    cm = confusion_matrix(y_test_mc, mc_preds, labels=cm_labels)
    
    # 6. Save metrics and plots
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "metrics.json"
    
    metrics = {
        "event_detector": {
            "accuracy": float(bin_acc),
            "precision": float(bin_precision),
            "recall": float(bin_recall),
            "f1": float(bin_f1)
        },
        "type_classifier": {
            "accuracy": float(mc_acc),
            "macro_f1": float(mc_macro_f1),
            "precision": float(mc_precision),
            "recall": float(mc_recall),
            "report": mc_report
        }
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved test set metrics to {metrics_path}")
    
    # Plot confusion matrix
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_plot_path = figures_dir / "confusion_matrix.png"
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title("Confusion Matrix - Event Type Classifier")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()
    
    logger.info(f"Saved confusion matrix plot to {cm_plot_path}")
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
