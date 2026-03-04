import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

def get_classifier_models() -> Dict[str, Any]:
    """Returns a dictionary mapping classifier names to instantiated MultiOutput models."""
    return {
        'KNN': MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
        'SVM': MultiOutputClassifier(SVC(probability=True, random_state=42)),
        'RandomForest': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
        'MLP': MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
        'LogisticRegression': MultiOutputClassifier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)))
    }

def train_classifiers(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Trains a suite of classifiers and returns the fitted models along with training times."""
    models = get_classifier_models()
    trained_models = {}
    training_times = {}
    
    if len(y_train) < 2:
        raise ValueError("Training data must contain multiple samples.")
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_times[name] = time.time() - start_time
        trained_models[name] = model
        
    return trained_models, training_times

def predict_classifiers(models: Dict[str, Any], X_test: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Runs prediction on a suite of trained models, returning predictions, probabilities and inference times."""
    predictions = {}
    probabilities = {}
    inference_times = {}
    
    for name, model in models.items():
        start_time = time.time()
        preds = model.predict(X_test)
        inference_times[name] = time.time() - start_time
        predictions[name] = preds
        
        # Determine probability if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)
        else:
            probs = None
        probabilities[name] = probs
            
    return predictions, probabilities, inference_times
