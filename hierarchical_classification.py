import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple, Any

class HierarchicalMultiTaskClassifier(BaseEstimator, ClassifierMixin):
    """
    Simultaneously predicts multiple targets (e.g., Species and Spinal Position)
    and incorporates a hierarchy-aware penalty during evaluation or ensemble weighting.
    """
    def __init__(self, base_estimator=None):
        if base_estimator is None:
            # Random Forest natively supports multi-output, but we'll use the wrapper
            # to ensure consistent interface across estimator types.
            self.base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.base_estimator = base_estimator
            
        self.model = MultiOutputClassifier(self.base_estimator)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the multi-output model.
        y should be shape (n_samples, n_targets), e.g. [species, position]
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the multiple targets.
        Returns shape (n_samples, n_targets)
        """
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Returns a list of probability arrays, one for each target.
        """
        return self.model.predict_proba(X)
        
def compute_hierarchical_loss(y_true_species: List[str], y_pred_species: List[str], 
                             taxonomy_dicts: List[Dict[str, str]]) -> float:
    """
    Computes a penalty score based on taxonomic distance.
    Distance scaling: Correct=0, Genus match=1, Family match=2, Total mismatch=3
    """
    from taxonomy_utils import TaxonomyTree
    
    total_loss = 0.0
    for i, (pred, true) in enumerate(zip(y_pred_species, y_true_species)):
        # In a real scenario we'd look up the full taxonomy dict for the predicated and true species.
        # Here we mock it by filtering the provided dictionaries.
        
        # Find true dict
        true_dict = next((d for d in taxonomy_dicts if d.get('species') == true), {'species': true})
        # Find pred dict
        pred_dict = next((d for d in taxonomy_dicts if d.get('species') == pred), {'species': pred})
        
        loss = TaxonomyTree.get_taxonomic_distance(pred_dict, true_dict)
        total_loss += loss
        
    return total_loss / len(y_true_species) if len(y_true_species) > 0 else 0.0
