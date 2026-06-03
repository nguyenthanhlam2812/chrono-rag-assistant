from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if sys.version_info >= (3, 13):
    raise RuntimeError(
        "Sprint 4 ML baseline requires Python 3.10-3.12 because the local "
        "scientific stack is not stable on Python 3.13. Use the conda env "
        "'ml_webapp' or run training on Kaggle."
    )

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


EVENT_TYPES = ["method_proposed", "release", "benchmark", "trend_application"]


@dataclass(frozen=True)
class SplitConfig:
    random_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got "
                f"train={self.train_ratio} + val={self.val_ratio} + "
                f"test={self.test_ratio} = {total}"
            )
        for name, value in (
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")

    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> "SplitConfig":
        """Build SplitConfig from a loaded configs/config.yaml dict.

        Reads ``models.random_seed``, ``models.test_size``, ``models.val_size``.
        ``train_ratio`` is derived as ``1 - test_size - val_size``.
        Missing keys fall back to dataclass defaults.
        """
        models_cfg = (config or {}).get("models", {}) or {}
        test_ratio = float(models_cfg.get("test_size", 0.15))
        val_ratio = float(models_cfg.get("val_size", 0.15))
        train_ratio = round(1.0 - test_ratio - val_ratio, 6)
        random_seed = int(models_cfg.get("random_seed", 42))
        return cls(
            random_seed=random_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )


def load_labeled_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {
        "sentence_id",
        "doc_id",
        "topic",
        "sentence",
        "is_event",
        "event_type",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["sentence"] = df["sentence"].fillna("").astype(str)
    df["is_event"] = df["is_event"].astype(int)
    df["event_type"] = df["event_type"].astype(str)
    return df


def make_doc_split(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Assign train/val/test by doc_id, balanced by topic where possible."""
    best_df: pd.DataFrame | None = None
    best_score: Tuple[int, int, int, int] | None = None

    # Search a few deterministic seeds. With scarce classes such as `release`,
    # doc-level splitting can accidentally hide a class from train/val/test.
    # The search keeps topic balance fixed and chooses the healthiest split.
    for offset in range(250):
        candidate = _make_doc_split_once(df, config, seed=config.random_seed + offset)
        score = _score_split(candidate)
        if best_score is None or score > best_score:
            best_df = candidate
            best_score = score

    if best_df is None:
        raise ValueError("Could not create a valid document-level split")
    return best_df


def _make_doc_split_once(df: pd.DataFrame, config: SplitConfig, seed: int) -> pd.DataFrame:
    doc_meta = (
        df.groupby("doc_id", as_index=False)
        .agg(topic=("topic", "first"), row_count=("sentence_id", "count"))
        .sort_values(["topic", "doc_id"])
    )

    assignments: Dict[str, str] = {}
    for topic, topic_docs in doc_meta.groupby("topic", sort=True):
        shuffled = topic_docs.sample(frac=1.0, random_state=seed)
        doc_ids = list(shuffled["doc_id"])
        total = len(doc_ids)
        if total < 3:
            raise ValueError(f"Need at least 3 documents for topic '{topic}'")

        n_train = max(1, round(total * config.train_ratio))
        n_val = max(1, round(total * config.val_ratio))
        if n_train + n_val >= total:
            n_train = max(1, total - 2)
            n_val = 1
        n_test = total - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = max(1, n_train - 1)

        for doc_id in doc_ids[:n_train]:
            assignments[doc_id] = "train"
        for doc_id in doc_ids[n_train : n_train + n_val]:
            assignments[doc_id] = "val"
        for doc_id in doc_ids[n_train + n_val :]:
            assignments[doc_id] = "test"

    split_df = df.copy()
    split_df["split"] = split_df["doc_id"].map(assignments)
    if split_df["split"].isna().any():
        missing_docs = sorted(split_df.loc[split_df["split"].isna(), "doc_id"].unique())
        raise ValueError(f"Missing split assignments for docs: {missing_docs}")
    return split_df


def _score_split(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    score = 0
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        event_df = split_df[split_df["is_event"] == 1]
        present_types = set(event_df["event_type"])
        missing = set(EVENT_TYPES) - present_types
        if split_name == "train":
            score -= 1000 * len(missing)
            score += 20 * len(present_types)
        else:
            score -= 80 * len(missing)
            score += 10 * len(present_types)
        if event_df.empty:
            score -= 500
        score += min(len(event_df), 50)

    # Prefer more even split sizes after the required class coverage checks.
    split_sizes = df["split"].value_counts().to_dict()
    size_spread = max(split_sizes.values()) - min(split_sizes.values())
    return (score, -size_spread, split_sizes.get("test", 0), split_sizes.get("val", 0))


def build_model(model_name: str, max_features: int) -> Pipeline:
    word_features = max(1000, int(max_features * 0.7))
    char_features = max(1000, max_features - word_features)
    vectorizer = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    ngram_range=(1, 3),
                    min_df=1,
                    max_features=word_features,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=char_features,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    if model_name == "logreg":
        # `lbfgs` supports both binary and multiclass natively. Older sklearn
        # versions allowed `liblinear` for multiclass via one-vs-rest, but
        # sklearn >=1.7 raises ValueError, so we standardise on lbfgs.
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
    elif model_name == "linearsvm":
        classifier = LinearSVC(class_weight="balanced", random_state=42)
    elif model_name == "sgd_log":
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.15,
            max_iter=3000,
            tol=1e-4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def evaluate_classifier(
    model: Pipeline,
    df: pd.DataFrame,
    labels: Sequence[Any],
    target_col: str,
) -> Dict[str, Any]:
    y_true = list(df[target_col])
    y_pred = list(model.predict(df["sentence"]))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[Any],
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def split_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for split_name, split_df in df.groupby("split"):
        summary[split_name] = {
            "rows": int(len(split_df)),
            "docs": int(split_df["doc_id"].nunique()),
            "topics": split_df["topic"].value_counts().sort_index().to_dict(),
            "is_event": split_df["is_event"].value_counts().sort_index().to_dict(),
            "event_type": split_df["event_type"].value_counts().sort_index().to_dict(),
        }
    return summary


def train_ml_baselines(
    input_csv: Path,
    output_dir: Path,
    eval_dir: Path,
    figures_dir: Path,
    random_seed: int = 42,
    max_features: int = 30000,
    model_names: Iterable[str] = ("logreg", "linearsvm"),
    split_config: SplitConfig | None = None,
) -> Dict[str, Any]:
    df = load_labeled_data(input_csv)
    if split_config is None:
        split_config = SplitConfig(random_seed=random_seed)
    df = make_doc_split(df, split_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    split_path = eval_dir / "ml_doc_splits.csv"
    df[["sentence_id", "doc_id", "topic", "is_event", "event_type", "split"]].to_csv(
        split_path, index=False, encoding="utf-8"
    )

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    metrics: Dict[str, Any] = {
        "input_csv": str(input_csv),
        "random_seed": split_config.random_seed,
        "max_features": max_features,
        "split_ratios": {
            "train": split_config.train_ratio,
            "val": split_config.val_ratio,
            "test": split_config.test_ratio,
        },
        "split_summary": split_summary(df),
        "models": {},
        "notes": [
            "Splits are assigned by doc_id to reduce document-level leakage.",
            "Class weights are balanced for both Logistic Regression and Linear SVM.",
            "The release class is low-resource and should be discussed separately in the report.",
        ],
    }

    binary_labels = [0, 1]
    type_labels = EVENT_TYPES
    train_events = train_df[train_df["is_event"] == 1]
    val_events = val_df[val_df["is_event"] == 1]
    test_events = test_df[test_df["is_event"] == 1]

    for model_name in model_names:
        binary_model = build_model(model_name, max_features=max_features)
        binary_model.fit(train_df["sentence"], train_df["is_event"])
        binary_model_path = output_dir / f"ml_{model_name}_event_binary.pkl"
        joblib.dump(binary_model, binary_model_path)

        type_model = build_model(model_name, max_features=max_features)
        type_model.fit(train_events["sentence"], train_events["event_type"])
        type_model_path = output_dir / f"ml_{model_name}_event_type.pkl"
        joblib.dump(type_model, type_model_path)

        model_metrics: Dict[str, Any] = {
            "binary_model_path": str(binary_model_path),
            "event_type_model_path": str(type_model_path),
            "binary": {},
            "event_type": {},
        }

        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            result = evaluate_classifier(binary_model, split_df, binary_labels, "is_event")
            model_metrics["binary"][split_name] = result
            plot_confusion_matrix(
                result["confusion_matrix"],
                binary_labels,
                f"{model_name} binary {split_name}",
                figures_dir / f"ml_{model_name}_binary_{split_name}_confusion.png",
            )

        for split_name, split_df in [("val", val_events), ("test", test_events)]:
            if split_df.empty:
                model_metrics["event_type"][split_name] = {
                    "warning": "No event rows available in this split."
                }
                continue
            result = evaluate_classifier(type_model, split_df, type_labels, "event_type")
            model_metrics["event_type"][split_name] = result
            plot_confusion_matrix(
                result["confusion_matrix"],
                type_labels,
                f"{model_name} event type {split_name}",
                figures_dir / f"ml_{model_name}_event_type_{split_name}_confusion.png",
            )

        metrics["models"][model_name] = model_metrics

    metrics_path = eval_dir / "ml_baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    metrics["metrics_path"] = str(metrics_path)
    metrics["split_path"] = str(split_path)
    return metrics
