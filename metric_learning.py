import numpy as np
from metric_learn import LMNN, NCA
from typing import Optional, List, Union

class LatentMetricLearner:
    """
    Supervised metric learning wrapper to fine-tune the encoded latents 
    to cluster shapes by a target variable (e.g., Species or Spinal Position).
    """
    def __init__(self, method: str = 'LMNN', k: int = 3, max_iter: int = 100):
        """
        Args:
            method: Either 'LMNN' (Large Margin Nearest Neighbor) or 'NCA' (Neighborhood Components Analysis)
            k: number of neighbors (used by LMNN)
            max_iter: maximum optimization iterations
        """
        self.method_name = method
        self.k = k
        self.max_iter = max_iter
        self.transformer = None
        
        if self.method_name == 'LMNN':
            self.transformer = LMNN(k=self.k, max_iter=self.max_iter, random_state=42)
        elif self.method_name == 'NCA':
            self.transformer = NCA(max_iter=self.max_iter, random_state=42)
        else:
            raise ValueError(f"Unsupported metric learning method: {method}. Choose 'LMNN' or 'NCA'.")
            
    def fit(self, X: np.ndarray, y: Union[List[str], np.ndarray]):
        """
        Fits the metric learning transformation on the input latents X 
        using labels y.
        """
        if len(np.unique(y)) < 2:
            raise ValueError("Training data must contain multiple classes to perform metric learning.")
            
        self.transformer.fit(X, y)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the learned transformation to the input latents X.
        """
        if self.transformer is None or not hasattr(self.transformer, 'components_'):
            raise RuntimeError("Transformer must be fitted before calling transform.")
            
        return self.transformer.transform(X)
        
    def fit_transform(self, X: np.ndarray, y: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Fits the transformer and returns the transformed X.
        """
        return self.fit(X, y).transform(X)
