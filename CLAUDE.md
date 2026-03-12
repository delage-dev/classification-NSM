# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural Shape Models (NSM) for fossil vertebrae classification. Uses deep learning to learn signed distance function (SDF) representations of 3D lizard/squamate vertebrae, then classifies them using the learned latent space. Forked from [gattia/NSM](https://github.com/gattia/NSM) under GNU Affero GPL 3.0.

## Setup

```bash
conda create -n NSM python=3.10
conda activate NSM
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge -c defaults
pip install -r requirements.txt
pip install .
```

Environment spec: `environment.yaml`. Package installed via `setup.py` (exposes the `NSM` package).

## Key Commands

```bash
# Train the NSM model (edit train_model.py config vars first)
python train_model.py

# Train with hierarchy-aware loss (contrastive + classification heads)
python train_model_hierarchy.py

# Run full classification pipeline (PCA+KNN, supervised classifiers, metric learning)
# Predicts species, genus, and spinal position
python classify_vertebrae_v3.py

# Run tests
pytest tests/
```

## Architecture

### Core Pipeline

1. **Mesh loading**: 3D vertebrae meshes (VTK format) from `vertebrae_meshes/`
2. **SDF sampling**: `NSM/datasets/sdf_dataset.py` computes signed distance functions, cached in `cache/`
3. **Model training**: `NSM/train/train_deep_sdf.py` trains a TriplanarDecoder that maps latent codes → SDF values
4. **Latent extraction**: Each vertebra gets a learned latent vector (dim 512)
5. **Classification**: Supervised classifiers and metric learning on top of latent features
6. **Evaluation**: Multi-level taxonomy metrics (species/genus/family/position)

### NSM Package (`NSM/`)

- `models/triplanar.py` — **TriplanarDecoder**: main architecture using tri-plane feature projection
- `models/deep_sdf.py` — Alternative DeepSDF decoder
- `datasets/sdf_dataset.py` — SDF sampling dataset with caching
- `mesh/main.py` — Mesh generation from latent codes via marching cubes
- `train/train_deep_sdf.py` — Training loop
- `optimization.py` — PCA initialization, nearest neighbor search in latent space
- `helper_funcs.py` — Mesh I/O, config loading, ICP alignment, monkey-patches for pymskt
- `losses.py` — Chamfer, ASSD, EMD loss functions
- `reconstruct/` — Shape reconstruction/completion from partial meshes

### Hierarchy-Aware Training (`train_model_hierarchy.py`)

- `hierarchy_loss.py` — Loss components: `TaxonomyLabelEncoder` (parses labels from filenames), `HierarchyContrastiveLoss` (taxonomy-weighted contrastive loss on latent codes), `TaxonomyClassificationHeads` (MLP heads predicting species/genus/family/position)
- `train_model_hierarchy.py` — Training script that adds hierarchy contrastive loss + classification heads to the SDF reconstruction training. Saves classification heads and taxonomy mappings alongside model checkpoints.
- Config keys: `hierarchy_contrastive_weight`, `classification_head_weight`, `hierarchy_contrastive_warmup`, `classification_head_warmup`, `classification_level_weights`

### Classification Layer (top-level scripts)

- `classify_vertebrae_v3.py` — Full pipeline: loads trained model, processes meshes, runs all classifiers predicting species/genus/position, outputs results to timestamped `results/classify_YYYYMMDD_HHMMSS/` directories
- `supervised_classifiers_v2.py` — Trains KNN, SVM, RandomForest, MLP, LogisticRegression
- `metric_learning.py` — LMNN and NCA metric learning via `metric_learn` library
- `evaluation_metrics.py` — Accuracy, precision, recall, F1, confusion matrices at each taxonomy level
- `taxonomy_utils.py` — Parses filenames like `{family}_{genus}_{species}_{specimen_id}-{c|t|l}{num}.vtk` to extract taxonomy and vertebral position (Cervical/Thoracic/Lumbar)
- `run_utils.py` — Creates timestamped run directories with README manifests

### Data Conventions

- Mesh filenames encode taxonomy: `family_genus_species_specimenid-c1.vtk` (c=Cervical, t=Thoracic, l=Lumbar)
- Trained model checkpoints live in experiment dirs (e.g., `run_v56/`): `model/{epoch}.pth`, `latent_codes/{epoch}.pth`, `model_params_config.json`
- Config: `vertebrae_config.json` controls all training hyperparameters

### Important Patterns

- **pymskt monkey-patching**: Both `train_model.py` and `classify_vertebrae_v3.py` monkey-patch `pymskt` for float64 SDF computation and mesh scalar loading. This is required for compatibility.
- **Device handling**: Scripts auto-detect CUDA → MPS → CPU. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for macOS.
- **Classification uses `os.chdir(TRAIN_DIR)`**: The classify script changes working directory into the experiment dir (e.g., `run_v56/`) early in execution.

### Important Notes
- Perserving data from training runs is paramount. Visibility is key. No output files should be overwritten. Instead, any script that generates output files should first check if the output directory already exists. If it does, it should create a new directory with a timestamp and save the output files to that directory. If it does not, it should create the directory and save the output files to that directory.
- Explainability of output results is key. Every directory for output should contain a README.md file that explains what the output files are and how to use them. It should also include some key metrics, such as summary statistics, relevant number of training/testing files, and any other information that would be useful for understanding the output files. 
- Comparison of results between runs is key. Any script that generates output files should also generate a comparison file that compares the results of the current run to the results of previous runs. This comparison file should be saved in the same directory as the output files.
- Research papers for further expansion of the project can be found in `literature/`. Not everthing in these papers will be implemented, but they should be used as a guide for potential improvements and extensions of the project.