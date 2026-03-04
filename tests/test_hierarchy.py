import pytest
import numpy as np
from hierarchical_classification import HierarchicalMultiTaskClassifier, compute_hierarchical_loss

def test_hierarchical_multitask_classifier():
    np.random.seed(42)
    X = np.random.rand(50, 10)
    
    # Target 1: Species, Target 2: Position
    y_species = np.array(['A'] * 25 + ['B'] * 25)
    y_pos = np.array(['Cervical', 'Thoracic'] * 25)
    
    # Stack columns to make (n_samples, 2)
    y = np.column_stack((y_species, y_pos))
    
    clf = HierarchicalMultiTaskClassifier()
    clf.fit(X, y)
    
    X_test = np.random.rand(5, 10)
    preds = clf.predict(X_test)
    
    assert preds.shape == (5, 2)
    
    probs = clf.predict_proba(X_test)
    assert len(probs) == 2 # One array per target
    assert probs[0].shape == (5, 2) # Species probs (A, B)
    assert probs[1].shape == (5, 2) # Pos probs (Cervical, Thoracic)

def test_compute_hierarchical_loss():
    taxonomy_dicts = [
        {'family': 'F1', 'genus': 'G1', 'species': 'S1'},
        {'family': 'F1', 'genus': 'G1', 'species': 'S2'},
        {'family': 'F1', 'genus': 'G2', 'species': 'S3'},
        {'family': 'F2', 'genus': 'G3', 'species': 'S4'},
    ]
    
    # Perfect predictions - Loss 0
    l_perf = compute_hierarchical_loss(['S1', 'S2'], ['S1', 'S2'], taxonomy_dicts)
    assert l_perf == 0.0
    
    # Genus match, species mismatch (Distance 1) -> S1 vs S2
    l_genus = compute_hierarchical_loss(['S1'], ['S2'], taxonomy_dicts)
    assert l_genus == 1.0
    
    # Family match, genus mismatch (Distance 2) -> S1 vs S3
    l_fam = compute_hierarchical_loss(['S1'], ['S3'], taxonomy_dicts)
    assert l_fam == 2.0
    
    # Total mismatch (Distance 3) -> S1 vs S4
    l_miss = compute_hierarchical_loss(['S1'], ['S4'], taxonomy_dicts)
    assert l_miss == 3.0
    
    # Average across batch
    l_batch = compute_hierarchical_loss(['S1', 'S1'], ['S2', 'S4'], taxonomy_dicts)
    # (1.0 + 3.0) / 2 = 2.0
    assert l_batch == 2.0
