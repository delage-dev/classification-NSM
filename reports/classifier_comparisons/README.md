# Classifier Comparisons

### Overview
This directory contains comparative analytics generated from the output logs of `classify_vertebrae.py` over an entire validation set. 

By employing multiple supervised algorithms (`KNN`, `SVM`, `Random Forest`, `Multilayer Perceptron (MLP)`, and `Logistic Regression`) on the latent embeddings passed through `metric-learn` NCA transformation, we can quantitatively assess which architecture scales best to our 3D dataset.

### Visualizations

#### 1. Performance Comparison Bar Chart (`performance_comparison_bar.png`)
* **What it measures:** The categorical `Instance Accuracy` (total correct / total samples) and the `Macro F1 Score` (harmonic mean of Precision and Recall, calculated independantly for each class and then averaged) representing overall robustness regardless of class imbalance.
* **Why it matters:** Accuracy alone is deceiving if 90% of the fossils belong to one species. F1 gives us confidence that the classification boundaries learned on the latent space are resilient and not over-fitting to majority genera. 

#### 2. Inference Time Box Plot (`inference_time_boxplot.png`)
* **What it measures:** The distribution of prediction latencies (in fractions of a second) required for an algorithm to label a newly encoded novel `$512$-dim` latent mesh vector. 
* **Why it matters:** While all these algorithms are relatively lightweight, `Random Forest` tree transversals may take objectively longer to execute at inference time than a `Logistic Regression` matrix multiplication. Understanding latency dictates which model powers the `interactive_classification.ipynb` widget efficiently.
