import os
import random
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report

from src.utils.logger import setup_logger

logger = setup_logger("train_event_baseline")

def split_dataset(
    csv_path: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the labeled dataset into train, validation, and test sets.
    The split is performed at the document level (doc_id) and stratified by topic.
    """
    logger.info(f"Loading labeled sentences from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter rows that are actually labeled (is_event is not null/empty)
    # is_event can be float/int or string in the CSV
    df = df[df["is_event"].notna()].copy()
    df["is_event"] = df["is_event"].astype(int)
    
    # Fill empty notes or event_types
    df["event_type"] = df["event_type"].fillna("none")
    # Clean string spaces if any
    if df["event_type"].dtype == object:
        df["event_type"] = df["event_type"].astype(str).str.strip()
    
    # Get unique docs and their topic
    doc_topics = df.groupby("doc_id")["topic"].first().reset_index()
    
    train_docs, val_docs, test_docs = [], [], []
    
    # Stratified split by topic
    topics = doc_topics["topic"].unique()
    for topic in topics:
        topic_docs = doc_topics[doc_topics["topic"] == topic]["doc_id"].tolist()
        
        # Sort to ensure deterministic behavior before shuffle
        topic_docs.sort()
        
        # Shuffle with a seeded random state
        rng = random.Random(seed)
        rng.shuffle(topic_docs)
        
        n = len(topic_docs)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test
        
        train_docs.extend(topic_docs[:n_train])
        val_docs.extend(topic_docs[n_train:n_train + n_val])
        test_docs.extend(topic_docs[n_train + n_val:])
        
    train_df = df[df["doc_id"].isin(train_docs)].copy()
    val_df = df[df["doc_id"].isin(val_docs)].copy()
    test_df = df[df["doc_id"].isin(test_docs)].copy()
    
    # Ensure splits dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    
    logger.info(f"Split completed. Unique docs: train={len(train_docs)}, val={len(val_docs)}, test={len(test_docs)}")
    logger.info(f"Split row counts: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df

def train_binary_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int = 42
) -> Tuple[TfidfVectorizer, Any, float]:
    """
    Train event sentence detector (binary classifier).
    Fits TF-IDF on train, compares LogReg vs Calibrated SVM, and returns the best model.
    """
    X_train_raw = train_df["sentence"].astype(str).tolist()
    y_train = train_df["is_event"].tolist()
    
    X_val_raw = val_df["sentence"].astype(str).tolist()
    y_val = val_df["is_event"].tolist()
    
    logger.info("Extracting TF-IDF features for binary classifier...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    
    # 1. Train Logistic Regression
    lr = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_val_preds = lr.predict(X_val)
    lr_f1 = f1_score(y_val, lr_val_preds)
    
    # 2. Train Calibrated Linear SVM
    base_svm = LinearSVC(class_weight="balanced", random_state=seed, max_iter=2000, dual=False)
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train, y_train)
    svm_val_preds = svm.predict(X_val)
    svm_f1 = f1_score(y_val, svm_val_preds)
    
    logger.info(f"Validation F1-score: LogisticRegression={lr_f1:.4f}, Calibrated Linear SVM={svm_f1:.4f}")
    
    if lr_f1 >= svm_f1:
        logger.info("Selecting Logistic Regression as the best binary model.")
        return vectorizer, lr, lr_f1
    else:
        logger.info("Selecting Calibrated Linear SVM as the best binary model.")
        return vectorizer, svm, svm_f1

def train_multiclass_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int = 42
) -> Tuple[TfidfVectorizer, Any]:
    """
    Train event type classifier (5-class classifier).
    Fits TF-IDF on train, trains Calibrated Linear SVM on all sentences.
    """
    X_train_raw = train_df["sentence"].astype(str).tolist()
    y_train = train_df["event_type"].tolist()
    
    X_val_raw = val_df["sentence"].astype(str).tolist()
    y_val = val_df["event_type"].tolist()
    
    logger.info("Extracting TF-IDF features for multiclass classifier...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    
    # Train Calibrated Linear SVM
    base_svm = LinearSVC(class_weight="balanced", random_state=seed, max_iter=2000, dual=False)
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train, y_train)
    
    val_preds = svm.predict(X_val)
    logger.info("Multiclass classification report on Validation set:")
    logger.info("\n" + classification_report(y_val, val_preds, zero_division=0))
    
    return vectorizer, svm
