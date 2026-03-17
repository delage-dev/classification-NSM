import numpy as np
from supervised_classifiers import get_classifier_models, train_classifiers, predict_classifiers

def test_classifiers_training_and_prediction():
    # Mock data equivalent to latents
    np.random.seed(42)
    X_train = np.random.rand(100, 512)
    y_train = ['species_A'] * 50 + ['species_B'] * 50
    
    models, train_times = train_classifiers(X_train, y_train)
    
    # Check that we trained 5 models
    assert len(models) == 5
    assert len(train_times) == 5
    
    # Check predictions
    X_test = np.random.rand(10, 512)
    preds, probs, inf_times = predict_classifiers(models, X_test)
    
    assert 'KNN' in preds
    assert 'SVM' in preds
    assert 'MLP' in preds
    assert 'RandomForest' in preds
    assert 'LogisticRegression' in preds
    
    # Check output shapes
    for name in models.keys():
        assert len(preds[name]) == 10
        if probs[name] is not None:
            assert probs[name].shape == (10, 2)
            # Probabilities should sum to 1
            np.testing.assert_allclose(np.sum(probs[name], axis=1), 1.0, rtol=1e-5)
    
    # Check inference time recording
    for name in models.keys():
        assert name in inf_times
        assert isinstance(inf_times[name], float)
