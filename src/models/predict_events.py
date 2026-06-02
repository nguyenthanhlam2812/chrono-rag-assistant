import pickle
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
def load_event_models(model_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load saved classification models and TF-IDF vectorizers.
    """
    model_dir = Path(model_dir)
    binary_path = model_dir / "event_detector.pkl"
    multiclass_path = model_dir / "type_classifier.pkl"
    
    if not binary_path.exists():
        raise FileNotFoundError(f"Binary model file not found at {binary_path}")
    if not multiclass_path.exists():
        raise FileNotFoundError(f"Multiclass model file not found at {multiclass_path}")
        
    with open(binary_path, "rb") as f:
        binary_data = pickle.load(f)
        
    with open(multiclass_path, "rb") as f:
        multiclass_data = pickle.load(f)
        
    return {
        "binary_vectorizer": binary_data["vectorizer"],
        "binary_model": binary_data["model"],
        "multiclass_vectorizer": multiclass_data["vectorizer"],
        "multiclass_model": multiclass_data["model"]
    }

def predict_sentences_batch(sentences: List[str], models: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Predict event probability, labels, and event types for a batch of sentences.
    """
    if not sentences:
        return []
        
    # Extract binary features and run predictions
    bin_vect = models["binary_vectorizer"]
    bin_model = models["binary_model"]
    
    X_bin = bin_vect.transform(sentences)
    is_event_labels = bin_model.predict(X_bin)
    
    # Check if model supports predict_proba
    if hasattr(bin_model, "predict_proba"):
        is_event_probs = bin_model.predict_proba(X_bin)[:, 1]
    else:
        # Fallback to decision function or mock probabilities if not calibrated
        if hasattr(bin_model, "decision_function"):
            dec = bin_model.decision_function(X_bin)
            is_event_probs = 1 / (1 + np.exp(-dec)) # sigmoid
        else:
            is_event_probs = [float(lbl) for lbl in is_event_labels]
            
    # Extract multiclass features and run predictions
    mc_vect = models["multiclass_vectorizer"]
    mc_model = models["multiclass_model"]
    
    X_mc = mc_vect.transform(sentences)
    event_types = mc_model.predict(X_mc)
    
    if hasattr(mc_model, "predict_proba"):
        probs = mc_model.predict_proba(X_mc)
        confidences = []
        for idx, pred_class in enumerate(event_types):
            # Map class label to its index in mc_model.classes_
            class_idx = list(mc_model.classes_).index(pred_class)
            confidences.append(float(probs[idx, class_idx]))
    else:
        confidences = [1.0] * len(sentences)
        
    results = []
    for idx in range(len(sentences)):
        is_ev = int(is_event_labels[idx])
        ev_type = str(event_types[idx])
        
        # Consistent mapping: if is_event is 0, event_type should be none
        if is_ev == 0:
            ev_type = "none"
            
        results.append({
            "is_event": is_ev,
            "event_prob": float(is_event_probs[idx]),
            "event_type": ev_type,
            "type_confidence": float(confidences[idx])
        })
        
    return results

def predict_sentence_event(sentence: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict event probability, label, and event type for a single sentence.
    """
    results = predict_sentences_batch([sentence], models)
    return results[0]
