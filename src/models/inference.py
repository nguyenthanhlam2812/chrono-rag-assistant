from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if sys.version_info >= (3, 13):
    raise RuntimeError(
        "Sprint 4 inference requires Python 3.10-3.12 with a working "
        "scikit-learn stack. Use the verified Python 3.11 environment."
    )

import joblib
import numpy as np

from src.timeline.date_extractor import extract_date


EVENT_TYPES = ["method_proposed", "release", "benchmark", "trend_application"]

PREDICTION_FIELDS = {
    "sentence_id",
    "doc_id",
    "chunk_id",
    "topic",
    "year",
    "source_url",
    "sentence",
    "is_event",
    "event_probability",
    "event_type",
    "event_type_confidence",
    "date_text",
    "normalized_date",
    "extracted_year",
    "date_confidence",
    "date_source",
    "binary_model",
    "event_type_model",
}


def load_sentence_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if "sentence_id" not in record:
                raise ValueError(f"Missing sentence_id at {path}:{line_no}")
            if "text" not in record and "sentence" not in record:
                raise ValueError(f"Missing text/sentence at {path}:{line_no}")
            records.append(record)
    return records


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run scripts/04_train_ml_classifier.py first."
        )
    return joblib.load(path)


def predict_event_sentences(
    sentence_records: Sequence[Dict[str, Any]],
    binary_model_path: Path,
    event_type_model_path: Path,
    binary_model_name: str,
    event_type_model_name: str,
    event_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    if not 0.0 <= event_threshold <= 1.0:
        raise ValueError(f"event_threshold must be in [0, 1], got {event_threshold}")

    binary_model = load_model(binary_model_path)
    event_type_model = load_model(event_type_model_path)
    sentences = [str(r.get("text") or r.get("sentence") or "") for r in sentence_records]

    binary_probs = _positive_class_probabilities(binary_model, sentences, positive_label=1)
    is_event = [1 if prob >= event_threshold else 0 for prob in binary_probs]

    event_indices = [idx for idx, flag in enumerate(is_event) if flag == 1]
    event_type_by_idx: Dict[int, Tuple[str, float]] = {}
    if event_indices:
        event_sentences = [sentences[idx] for idx in event_indices]
        type_labels, type_confidences = _predicted_labels_with_confidence(
            event_type_model, event_sentences
        )
        for idx, label, confidence in zip(event_indices, type_labels, type_confidences):
            label_text = str(label)
            if label_text not in EVENT_TYPES:
                label_text = "trend_application"
            event_type_by_idx[idx] = (label_text, confidence)

    predictions: List[Dict[str, Any]] = []
    for idx, record in enumerate(sentence_records):
        event_type, type_confidence = event_type_by_idx.get(idx, ("none", 0.0))
        date_info = extract_date(
            sentences[idx],
            document_year=_coerce_year(record.get("year")),
        )
        predictions.append(
            {
                "sentence_id": record.get("sentence_id", ""),
                "doc_id": record.get("doc_id", ""),
                "chunk_id": record.get("chunk_id", ""),
                "topic": record.get("topic", ""),
                "year": record.get("year"),
                "source_url": record.get("source_url", ""),
                "sentence": sentences[idx],
                "is_event": is_event[idx],
                "event_probability": round(float(binary_probs[idx]), 6),
                "event_type": event_type,
                "event_type_confidence": round(float(type_confidence), 6),
                "date_text": date_info["date_text"],
                "normalized_date": date_info["normalized_date"],
                "extracted_year": date_info["extracted_year"],
                "date_confidence": date_info["date_confidence"],
                "date_source": date_info["date_source"],
                "binary_model": binary_model_name,
                "event_type_model": event_type_model_name,
            }
        )
    return predictions


def write_predictions_jsonl(predictions: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for prediction in predictions:
            missing = PREDICTION_FIELDS - set(prediction)
            if missing:
                raise ValueError(f"Prediction missing required fields: {sorted(missing)}")
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def summarize_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    event_count = sum(1 for row in predictions if int(row.get("is_event", 0)) == 1)
    type_counts: Dict[str, int] = {}
    topic_counts: Dict[str, Dict[str, int]] = {}
    for row in predictions:
        event_type = str(row.get("event_type", "none"))
        topic = str(row.get("topic", "unknown"))
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        if topic not in topic_counts:
            topic_counts[topic] = {"total": 0, "events": 0}
        topic_counts[topic]["total"] += 1
        topic_counts[topic]["events"] += int(row.get("is_event", 0))
    return {
        "total_sentences": len(predictions),
        "predicted_events": event_count,
        "predicted_non_events": len(predictions) - event_count,
        "events_with_date": sum(
            1
            for row in predictions
            if int(row.get("is_event", 0)) == 1 and row.get("extracted_year") is not None
        ),
        "event_type_counts": dict(sorted(type_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
    }


def _coerce_year(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_class_probabilities(model: Any, texts: Sequence[str], positive_label: Any) -> List[float]:
    labels, confidences = _class_probability_matrix(model, texts)
    try:
        positive_idx = list(labels).index(positive_label)
    except ValueError:
        try:
            positive_idx = list(labels).index(str(positive_label))
        except ValueError as exc:
            raise ValueError(f"Model classes do not contain positive label {positive_label!r}") from exc
    return [float(row[positive_idx]) for row in confidences]


def _predicted_labels_with_confidence(model: Any, texts: Sequence[str]) -> Tuple[List[Any], List[float]]:
    predictions = list(model.predict(texts))
    labels, confidences = _class_probability_matrix(model, texts)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    pred_confidences: List[float] = []
    for pred, row in zip(predictions, confidences):
        idx = label_to_idx.get(pred)
        pred_confidences.append(float(max(row)) if idx is None else float(row[idx]))
    return predictions, pred_confidences


def _class_probability_matrix(model: Any, texts: Sequence[str]) -> Tuple[List[Any], np.ndarray]:
    labels = _model_classes(model)
    if not texts:
        return labels, np.empty((0, len(labels)))

    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(texts), dtype=float)
        return labels, probs

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(texts), dtype=float)
        if scores.ndim == 1:
            positive = np.asarray([_sigmoid(float(score)) for score in scores])
            probs = np.column_stack([1.0 - positive, positive])
        else:
            probs = np.vstack([_softmax(row) for row in scores])
        return labels, probs

    predictions = list(model.predict(texts))
    probs = np.zeros((len(texts), len(labels)), dtype=float)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for row_idx, pred in enumerate(predictions):
        probs[row_idx, label_to_idx[pred]] = 1.0
    return labels, probs


def _model_classes(model: Any) -> List[Any]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise ValueError("Model does not expose classes_")
    return list(classes)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
