# Hierarchical Confusion Matrices

### Overview
This directory maps the geometric relationships of the model's failed predictions along the target phylogenetic tree. Because traditional `Accuracy` simply grades a prediction as $0$ or $1$, it fails to reward a model for "near misses". 

By mapping confusion matrices at the **Species**, **Genus**, and **Family** levels, we can evaluate the successful execution of our *Hierarchy-Aware Loss penalties*.

### Visualizations

#### 1. Species-Level Confusion Matrix (`confusion_matrix_species.png`)
* **What it measures:** The baseline truth vs predicted spread exactly mapping the lowest leaf node of the taxonomy. 
* **Why it matters:** The model's raw capability to distinguish highly intricate morphological nuances between closely related vertebrae. (e.g., distinguishing between `Tribolonotus novaeguineae` and `Tribolonotus gracilis`).

#### 2. Genus-Level Confusion Matrix (`confusion_matrix_genus.png`)
* **What it measures:** Rolling up the species labels into their parent Genus grouping. 
* **Why it matters:** If the matrix shows that misclassified `Tribolonotus` samples are primarily predicted as another `Tribolonotus` species rather than a drastically different Genus like `Corucia`, the latent space has successfully maintained phylogenetic clustering. The mistake severity is low.

#### 3. Family-Level Confusion Matrix (`confusion_matrix_family.png`)
* **What it measures:** Rolling up the Genus labels into the highest taxonomy tier (e.g. `Scincidae` vs `Cordylidae`).
* **Why it matters:** This diagonal should be near perfect. Misclassifying an entire Family of lizards implies a fundamental breakdown in the DeepSDF encoding boundaries. By grading hierarchy, we gain qualitative assurance in the model's geometrical rationale.
