import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Any, Optional

def calculate_metrics(y_true: List[str], y_pred: List[str], y_probs: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None, top_k: int = 5) -> Dict[str, float]:
    """
    Calculates literature-backed metrics for classification.
    """
    if len(y_true) == 0:
        return {}
        
    metrics = {}
    
    # Instance Accuracy
    metrics['instance_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Average Class Accuracy (Balanced Accuracy)
    metrics['average_class_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Top-K Accuracy
    if y_probs is not None and classes is not None and top_k > 1 and len(classes) >= top_k:
        # Sort probabilities and get top K indices
        top_k_preds = np.argsort(y_probs, axis=1)[:, -top_k:]
        
        correct = 0
        for i, true_label in enumerate(y_true):
            true_idx = np.where(classes == true_label)[0]
            if len(true_idx) > 0 and true_idx[0] in top_k_preds[i]:
                correct += 1
        metrics[f'top_{top_k}_accuracy'] = correct / len(y_true)
        
    return metrics

def generate_hierarchical_confusion_matrices(y_true_dicts: List[Dict[str, str]], y_pred_dicts: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Generates confusion matrices at different taxonomic levels.
    """
    levels = ['family', 'genus', 'species']
    matrices = {}
    
    for level in levels:
        y_true_level = [d.get(level, 'unknown') for d in y_true_dicts]
        y_pred_level = [d.get(level, 'unknown') for d in y_pred_dicts]
        
        labels = sorted(list(set(y_true_level + y_pred_level)))
        cm = confusion_matrix(y_true_level, y_pred_level, labels=labels)
        matrices[level] = {
            'matrix': cm,
            'labels': labels
        }
        
    return matrices
