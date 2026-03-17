# classify_vertebrae_v4.py
# ========================
# Version 4: Ablation framework for systematic comparison of classification
# approaches. Runs the same validation set through multiple configurations
# (toggling metric learning and supervised classifiers on/off) and produces
# cross-configuration comparison tables, charts, and confusion matrices.
#
# Configurations:
#   A) Baseline           — raw latent codes, distance-based retrieval only
#   B) + Metric Learning  — NCA-transformed latents, distance-based retrieval
#   C) + Classifiers      — raw latents, supervised classifiers (no metric learning)
#   D) + ML + Classifiers — NCA-transformed latents + supervised classifiers (full pipeline)
#
# Output structure:
#   results/ablation_YYYYMMDD_HHMMSS/
#   ├── README.md
#   ├── summary_statistics.md
#   ├── comparison_metrics.csv        # All configs side-by-side
#   ├── comparison_charts/
#   │   ├── ablation_accuracy_comparison.png
#   │   ├── ablation_f1_comparison.png
#   │   └── ablation_distance_comparison.png
#   ├── A_baseline/
#   │   ├── README.md
#   │   ├── predictions.csv
#   │   ├── detailed_metrics.csv
#   │   ├── distance_metrics.csv
#   │   └── confusion_matrices/
#   ├── B_metric_learning/
#   │   └── ...
#   ├── C_classifiers/
#   │   └── ...
#   └── D_full_pipeline/
#       └── ...
# ========================

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys
import torch
import numpy as np
import pandas as pd
import json
import datetime
import time as time_module
import pyvista as pv
import pymskt.mesh.meshes as meshes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.mesh import get_sdfs, create_mesh
import vtk
import re
import random
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional

from NSM.helper_funcs import (
    NumpyTransform, load_config, load_model_and_latents,
    convert_ply_to_vtk, get_sdfs, fixed_point_coords,
    safe_load_mesh_scalars, extract_species_prefix,
)
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, find_similar, find_similar_cos
from supervised_classifiers_v2 import train_classifiers, predict_classifiers
from metric_learning import LatentMetricLearner
from taxonomy_utils import parse_taxonomy_from_filename
from evaluation_metrics import (
    calculate_metrics,
    metrics_to_dataframe,
    generate_hierarchical_confusion_matrices,
    generate_position_confusion_matrix,
)
from run_utils import create_run_directory, write_run_manifest

# Monkey-patch pymskt
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)


# ======================================================================
# Ablation Configuration Definitions
# ======================================================================

ABLATION_CONFIGS = OrderedDict([
    ("A_baseline", {
        "label": "Baseline (Distance Only)",
        "description": "Raw latent codes from NSM, distance-based retrieval only (cosine + euclidean). No metric learning, no supervised classifiers.",
        "use_metric_learning": False,
        "use_classifiers": False,
    }),
    ("B_metric_learning", {
        "label": "Baseline + Metric Learning",
        "description": "NCA-transformed latent codes, distance-based retrieval only. Metric learning reshapes the latent space to cluster by species before distance matching.",
        "use_metric_learning": True,
        "use_classifiers": False,
    }),
    ("C_classifiers", {
        "label": "Baseline + Classifiers",
        "description": "Raw latent codes with supervised classifiers (KNN, SVM, RF, MLP, LR) trained on top. No metric learning transformation.",
        "use_metric_learning": False,
        "use_classifiers": True,
    }),
    ("D_full_pipeline", {
        "label": "Baseline + ML + Classifiers",
        "description": "Full pipeline: NCA metric learning followed by supervised classifiers. This is equivalent to the v3 approach.",
        "use_metric_learning": True,
        "use_classifiers": True,
    }),
])


# ======================================================================
# Configuration
# ======================================================================
TRAIN_DIR = "run_v56"
os.chdir(TRAIN_DIR)
CKPT = '2000'
LC_PATH = f'latent_codes/{CKPT}.pth'
MODEL_PATH = f'model/{CKPT}.pth'

config = load_config(config_path='model_params_config.json')

# Determine device
if torch.cuda.is_available():
    device = "cuda:0"
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple MPS (M1/M2/M3 GPU)")
else:
    device = "cpu"
    print("Using CPU")

config['device'] = device

train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# ======================================================================
# Locate mesh directory
# ======================================================================
potential_dirs = ["vertebrae_meshes", "../vertebrae_meshes",
                  "../../vertebrae_meshes", "data/vertebrae_meshes"]
mesh_dir = None
for d in potential_dirs:
    if os.path.isdir(d):
        mesh_dir = d
        break
if mesh_dir is None:
    print("Warning: Could not locate 'vertebrae_meshes' directory.")
    mesh_dir = "."

test_paths = config['test_paths']
mesh_list_raw = test_paths  # Use the larger test set (351) instead of val set (119)
mesh_list = [os.path.join(mesh_dir, os.path.basename(p)) for p in mesh_list_raw]

# ======================================================================
# Create top-level ablation run directory
# ======================================================================
run_dir = create_run_directory(base_dir="results", prefix="ablation")
print(f"\n{'='*60}")
print(f"Ablation run directory: {run_dir}")
print(f"{'='*60}\n")


# ======================================================================
# Helper: optimize latent vector for a novel mesh
# ======================================================================
def optimize_latent(decoder, points, sdf_vals, latent_size, mean_latent,
                    latent_codes, top_k_reg, device, iters=1000, lr=1e-3):
    init_latent_torch = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg)
    latent = init_latent_torch.clone().detach().float().to(device).requires_grad_()
    optimizer = torch.optim.Adam([latent], lr=lr)
    sdf_vals = sdf_vals.float().to(device)
    decoder = decoder.to(device)
    points = points.float().to(device)
    for i in range(iters):
        optimizer.zero_grad()
        pred_sdf = get_sdfs(decoder, points, latent, device=device)
        loss = F.l1_loss(pred_sdf.squeeze(), sdf_vals)
        loss.backward()
        optimizer.step()
        if i % 200 == 0 or i == iters - 1:
            print(f"[{i}/{iters}] Loss: {loss.item():.6f}")
    return latent.detach().to(device)


# ======================================================================
# Load model and latent codes
# ======================================================================
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)
_, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)


# ======================================================================
# Prepare training data & labels
# ======================================================================
X_train_raw = latent_codes.cpu().numpy()

y_train_species = []
y_train_genera = []
y_train_positions = []
valid_indices = []

for f_idx, f in enumerate(all_vtk_files):
    parsed = parse_taxonomy_from_filename(f)
    if parsed and parsed.get('species') and parsed.get('position'):
        y_train_species.append(parsed['species'])
        y_train_genera.append(parsed['genus'])
        y_train_positions.append(parsed['position'])
        valid_indices.append(f_idx)
    else:
        print(f"Skipping training file with invalid label format: {f}")

X_train_valid = X_train_raw[valid_indices]
y_train = np.column_stack((y_train_species, y_train_genera, y_train_positions))


# ======================================================================
# Pre-compute metric learning transformation (used by configs B and D)
# ======================================================================
unique_species = np.unique(y_train_species)
metric_learner = None
X_train_ml = None

if len(y_train_species) > 0 and len(unique_species) > 1:
    print("\nPre-computing NCA Metric Learning transformation...")
    ml_start = time_module.time()
    metric_learner = LatentMetricLearner(method='NCA', max_iter=100)
    X_train_ml = metric_learner.fit_transform(X_train_valid, np.array(y_train_species))
    ml_time = time_module.time() - ml_start
    print(f"Metric learning fitted in {ml_time:.3f}s")
else:
    ml_time = 0.0
    print("Warning: Not enough species for metric learning.")


# ======================================================================
# Pre-train classifiers for each applicable config
# ======================================================================
# Config C: classifiers on raw latents
# Config D: classifiers on NCA-transformed latents
print("\nTraining classifiers for each configuration...")

classifiers_raw = None
classifiers_raw_times = {}
classifiers_ml = None
classifiers_ml_times = {}

if len(y_train) >= 2:
    # Config C: raw latent classifiers
    print("  Training classifiers on raw latent codes (Config C)...")
    classifiers_raw, classifiers_raw_times = train_classifiers(X_train_valid, y_train)
    for name, t in classifiers_raw_times.items():
        print(f"    {name}: {t:.3f}s")

    # Config D: metric-learning transformed classifiers
    if X_train_ml is not None:
        print("  Training classifiers on NCA-transformed latent codes (Config D)...")
        classifiers_ml, classifiers_ml_times = train_classifiers(X_train_ml, y_train)
        for name, t in classifiers_ml_times.items():
            print(f"    {name}: {t:.3f}s")
else:
    raise ValueError("Not enough valid training labels!")


# ======================================================================
# Pre-compute training tensors on device for distance computation
# (avoids redundant CPU->GPU transfers inside the per-mesh loop)
# ======================================================================
print("\nPre-computing training tensors on device for distance computation...")
train_tensor_raw = torch.tensor(X_train_raw, dtype=torch.float32).to(device)
if X_train_ml is not None:
    X_train_ml_all = metric_learner.transform(X_train_raw)  # transform full training set once
    train_tensor_ml = torch.tensor(X_train_ml_all, dtype=torch.float32).to(device)
else:
    X_train_ml_all = None
    train_tensor_ml = None


# ======================================================================
# Process each test mesh — optimize latent once, then evaluate
# across all ablation configurations
# ======================================================================
print(f"\nProcessing {len(mesh_list)} test meshes...")

# Per-config accumulators
config_logs = {cfg_name: [] for cfg_name in ABLATION_CONFIGS}

for mesh_idx, vert_fname in enumerate(mesh_list):
    mesh_basename = os.path.basename(vert_fname)
    mesh_stem = os.path.splitext(mesh_basename)[0]
    print(f"\033[32m\n=== Processing {mesh_basename} ({mesh_idx + 1}/{len(mesh_list)}) ===\033[0m")

    # --- Parse ground truth ---
    parsed_truth = parse_taxonomy_from_filename(mesh_basename)
    gt_species = parsed_truth.get('species') if parsed_truth else extract_species_prefix(mesh_basename)
    gt_genus = parsed_truth.get('genus') if parsed_truth else None
    gt_family = parsed_truth.get('family') if parsed_truth else None
    gt_position = parsed_truth.get('position') if parsed_truth else None

    # --- Set up inference dataset ---
    actual_fname = vert_fname
    if '.ply' in vert_fname:
        _, actual_fname = convert_ply_to_vtk(vert_fname)

    sdf_dataset = SDFSamples(
        list_mesh_paths=[actual_fname],
        multiprocessing=False,
        subsample=config["samples_per_object_per_batch"],
        print_filename=True,
        n_pts=config["n_pts_per_object"],
        p_near_surface=config['percent_near_surface'],
        p_further_from_surface=config['percent_further_from_surface'],
        sigma_near=config['sigma_near'],
        sigma_far=config['sigma_far'],
        rand_function=config['random_function'],
        center_pts=config['center_pts'],
        norm_pts=config['normalize_pts'],
        reference_mesh=None,
        verbose=config['verbose'],
        save_cache=config['cache'],
        equal_pos_neg=config['equal_pos_neg'],
        fix_mesh=config['fix_mesh'],
    )

    print("Setting up dataset")
    sdf_sample = sdf_dataset[0]
    sample_dict, _ = sdf_sample
    points = sample_dict['xyz'].float().to(device)
    sdf_vals = sample_dict['gt_sdf'].float()

    # --- Optimize latent (done once per mesh, shared across all configs) ---
    print("Optimizing latent vector...")
    latent_novel = optimize_latent(model, points, sdf_vals, config['latent_size'],
                                   mean_latent, latent_codes, top_k_reg, device=device)
    novel_vec_raw = latent_novel.cpu().detach().float().numpy()
    print("Latent optimization complete.")

    # Pre-compute NCA-transformed novel vector
    novel_vec_ml = metric_learner.transform(novel_vec_raw) if metric_learner is not None else None

    # --- Evaluate each ablation config ---
    for cfg_name, cfg in ABLATION_CONFIGS.items():
        entry = {
            "mesh": mesh_basename,
            "ground_truth_species": gt_species,
            "ground_truth_genus": gt_genus,
            "ground_truth_family": gt_family,
            "ground_truth_position": gt_position,
        }

        # Choose pre-computed training tensor and novel vector for distance
        if cfg["use_metric_learning"] and train_tensor_ml is not None:
            cur_train_tensor = train_tensor_ml
            novel_for_distance = novel_vec_ml
        else:
            cur_train_tensor = train_tensor_raw
            novel_for_distance = novel_vec_raw

        # --- Distance-based classification (all configs) ---
        with torch.no_grad():
            novel_tensor = torch.tensor(novel_for_distance, dtype=torch.float32).to(device)

            # Cosine similarity top-5
            cos_sims = F.cosine_similarity(novel_tensor, cur_train_tensor)
            cos_top5_indices = torch.argsort(cos_sims, descending=True)[:5].cpu().numpy()
            cos_top5_distances = (1 - cos_sims[cos_top5_indices]).cpu().numpy()

            # Euclidean distance top-5
            euc_dists = torch.norm(cur_train_tensor - novel_tensor, dim=1)
            euc_top5_indices = torch.argsort(euc_dists)[:5].cpu().numpy()
            euc_top5_distances = euc_dists[euc_top5_indices].cpu().numpy()

        # Extract species predictions from distance methods
        cos_top1_species = extract_species_prefix(all_vtk_files[cos_top5_indices[0]])
        cos_top5_species = [extract_species_prefix(all_vtk_files[idx]) for idx in cos_top5_indices]
        euc_top1_species = extract_species_prefix(all_vtk_files[euc_top5_indices[0]])
        euc_top5_species = [extract_species_prefix(all_vtk_files[idx]) for idx in euc_top5_indices]

        # Parse genus from top-1 distance matches
        cos_top1_parsed = parse_taxonomy_from_filename(all_vtk_files[cos_top5_indices[0]])
        euc_top1_parsed = parse_taxonomy_from_filename(all_vtk_files[euc_top5_indices[0]])

        entry["cos_top1_species"] = cos_top1_species
        entry["cos_top1_genus"] = cos_top1_parsed.get('genus') if cos_top1_parsed else None
        entry["cos_top1_position"] = cos_top1_parsed.get('position') if cos_top1_parsed else None
        entry["cos_top1_match"] = "yes" if gt_species and gt_species == cos_top1_species else "no"
        entry["cos_top5_match"] = "yes" if gt_species and gt_species in cos_top5_species else "no"
        entry["euc_top1_species"] = euc_top1_species
        entry["euc_top1_genus"] = euc_top1_parsed.get('genus') if euc_top1_parsed else None
        entry["euc_top1_position"] = euc_top1_parsed.get('position') if euc_top1_parsed else None
        entry["euc_top1_match"] = "yes" if gt_species and gt_species == euc_top1_species else "no"
        entry["euc_top5_match"] = "yes" if gt_species and gt_species in euc_top5_species else "no"

        for rank in range(5):
            entry[f"cos_similar_{rank+1}_name"] = all_vtk_files[cos_top5_indices[rank]]
            entry[f"cos_similar_{rank+1}_distance"] = float(cos_top5_distances[rank])
            entry[f"euc_similar_{rank+1}_name"] = all_vtk_files[euc_top5_indices[rank]]
            entry[f"euc_similar_{rank+1}_distance"] = float(euc_top5_distances[rank])

        # --- Supervised classifier predictions (configs C and D only) ---
        if cfg["use_classifiers"]:
            if cfg["use_metric_learning"] and classifiers_ml is not None:
                trained = classifiers_ml
                test_vec = novel_vec_ml
            else:
                trained = classifiers_raw
                test_vec = novel_vec_raw

            predictions, probabilities, inference_times = predict_classifiers(trained, test_vec)

            for clf_name in trained.keys():
                preds = predictions[clf_name][0]
                entry[f"{clf_name}_predicted_species"] = preds[0]
                entry[f"{clf_name}_predicted_genus"] = preds[1]
                entry[f"{clf_name}_predicted_position"] = preds[2]
                entry[f"{clf_name}_match_species"] = "yes" if gt_species and gt_species == preds[0] else "no"
                entry[f"{clf_name}_match_genus"] = "yes" if gt_genus and gt_genus == preds[1] else "no"
                entry[f"{clf_name}_match_position"] = "yes" if gt_position and str(gt_position) == str(preds[2]) else "no"
                entry[f"{clf_name}_inf_time"] = inference_times[clf_name]

                if probabilities[clf_name] is not None:
                    entry[f"{clf_name}_species_confidence"] = f"{max(probabilities[clf_name][0][0]):.2%}"
                    entry[f"{clf_name}_genus_confidence"] = f"{max(probabilities[clf_name][1][0]):.2%}"
                    entry[f"{clf_name}_position_confidence"] = f"{max(probabilities[clf_name][2][0]):.2%}"

        config_logs[cfg_name].append(entry)


# ======================================================================
# Compute metrics and save per-config outputs
# ======================================================================
print(f"\n{'='*60}")
print("Computing metrics for each ablation configuration...")
print(f"{'='*60}")

CLASSIFIER_NAMES = list(classifiers_raw.keys()) if classifiers_raw else []

all_config_summaries = []  # For cross-config comparison

for cfg_name, cfg in ABLATION_CONFIGS.items():
    print(f"\n--- {cfg['label']} ({cfg_name}) ---")

    cfg_dir = os.path.join(run_dir, cfg_name)
    os.makedirs(cfg_dir, exist_ok=True)
    cm_dir = os.path.join(cfg_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    df = pd.DataFrame(config_logs[cfg_name])
    df.to_csv(os.path.join(cfg_dir, "predictions.csv"), index=False)

    # --- Distance-based metrics (computed for all configs) ---
    distance_rows = []
    for method, label in [("cos", "Cosine Similarity"), ("euc", "Euclidean Distance")]:
        t1_col = f"{method}_top1_match"
        t5_col = f"{method}_top5_match"
        if t1_col in df.columns:
            top1_acc = (df[t1_col] == "yes").mean()
            distance_rows.append({"method": label, "metric": "top_1_accuracy", "value": top1_acc})
        if t5_col in df.columns:
            top5_acc = (df[t5_col] == "yes").mean()
            distance_rows.append({"method": label, "metric": "top_5_accuracy", "value": top5_acc})

        # Genus and position accuracy from distance top-1
        genus_col = f"{method}_top1_genus"
        pos_col = f"{method}_top1_position"
        if genus_col in df.columns and "ground_truth_genus" in df.columns:
            mask = df["ground_truth_genus"].notna() & df[genus_col].notna()
            if mask.sum() > 0:
                genus_acc = (df.loc[mask, "ground_truth_genus"] == df.loc[mask, genus_col]).mean()
                distance_rows.append({"method": label, "metric": "genus_top_1_accuracy", "value": genus_acc})
        if pos_col in df.columns and "ground_truth_position" in df.columns:
            mask = df["ground_truth_position"].notna() & df[pos_col].notna()
            if mask.sum() > 0:
                pos_acc = (df.loc[mask, "ground_truth_position"] == df.loc[mask, pos_col]).mean()
                distance_rows.append({"method": label, "metric": "position_top_1_accuracy", "value": pos_acc})

    distance_df = pd.DataFrame(distance_rows)
    if not distance_df.empty:
        distance_df.to_csv(os.path.join(cfg_dir, "distance_metrics.csv"), index=False)
        for _, row in distance_df.iterrows():
            print(f"  {row['method']} — {row['metric']}: {row['value']:.4f}")

    # Build summary entry for this config
    config_summary = {"config": cfg_name, "label": cfg["label"]}
    for _, row in distance_df.iterrows():
        key = f"{row['method'].split()[0].lower()}_{row['metric']}"
        config_summary[key] = row['value']

    # --- Supervised classifier metrics (configs C and D only) ---
    all_metrics_rows = []
    if cfg["use_classifiers"]:
        for clf_name in CLASSIFIER_NAMES:
            for target, gt_col, pred_col in [
                ("species", "ground_truth_species", f"{clf_name}_predicted_species"),
                ("genus", "ground_truth_genus", f"{clf_name}_predicted_genus"),
                ("position", "ground_truth_position", f"{clf_name}_predicted_position"),
            ]:
                if pred_col not in df.columns or gt_col not in df.columns:
                    continue
                mask = df[gt_col].notna() & df[pred_col].notna()
                y_true = df.loc[mask, gt_col].tolist()
                y_pred = df.loc[mask, pred_col].tolist()
                if not y_true:
                    continue
                m = calculate_metrics(y_true, y_pred)
                m_df = metrics_to_dataframe(m, classifier_name=clf_name)
                m_df["target"] = target
                all_metrics_rows.append(m_df)

                if target == "species":
                    config_summary[f"{clf_name}_species_accuracy"] = m["instance_accuracy"]
                    config_summary[f"{clf_name}_species_f1_macro"] = m["f1_macro"]
                    config_summary[f"{clf_name}_species_precision_macro"] = m["precision_macro"]
                    config_summary[f"{clf_name}_species_recall_macro"] = m["recall_macro"]
                elif target == "position":
                    config_summary[f"{clf_name}_position_accuracy"] = m["instance_accuracy"]
                    config_summary[f"{clf_name}_position_f1_macro"] = m["f1_macro"]
                    config_summary[f"{clf_name}_position_precision_macro"] = m["precision_macro"]
                    config_summary[f"{clf_name}_position_recall_macro"] = m["recall_macro"]

    if all_metrics_rows:
        metrics_df = pd.concat(all_metrics_rows, ignore_index=True)
        metrics_df.to_csv(os.path.join(cfg_dir, "detailed_metrics.csv"), index=False)

        summary_rows = metrics_df[metrics_df["level"] == "summary"]
        for _, row in summary_rows.iterrows():
            target = row.get("target", "?")
            clf = row.get("classifier", "?")
            acc = row.get("instance_accuracy", "N/A")
            f1 = row.get("f1_macro", "N/A")
            if isinstance(acc, float):
                print(f"  [{clf}] {target}: Acc={acc:.4f}, F1={f1:.4f}")

    all_config_summaries.append(config_summary)

    # --- Confusion matrices ---
    if len(df) >= 2:
        # Determine best approach for CMs
        best_approach = None
        best_acc = 0

        # Check distance-based species accuracy
        for method in ["cos", "euc"]:
            col = f"{method}_top1_match"
            if col in df.columns:
                acc = (df[col] == "yes").mean()
                if acc > best_acc:
                    best_acc = acc
                    best_approach = ("distance", method)

        # Check classifier accuracy
        if cfg["use_classifiers"]:
            for clf_name in CLASSIFIER_NAMES:
                col = f"{clf_name}_predicted_species"
                if col in df.columns:
                    acc = (df["ground_truth_species"] == df[col]).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_approach = ("classifier", clf_name)

        # Generate CMs using the best approach
        if best_approach is not None:
            approach_type, approach_name = best_approach

            if approach_type == "classifier":
                y_pred_species = df[f"{approach_name}_predicted_species"].tolist()
                cm_title_suffix = approach_name
            else:
                y_pred_species = df[f"{approach_name}_top1_species"].tolist()
                cm_title_suffix = f"{'Cosine' if approach_name == 'cos' else 'Euclidean'} Top-1"

            print(f"  Generating confusion matrices using: {cm_title_suffix} ({best_acc:.1%})")

            # Hierarchical CMs
            true_dicts, pred_dicts = [], []
            species_taxonomy_map = {}
            for filename in df["mesh"]:
                parsed = parse_taxonomy_from_filename(filename)
                if parsed:
                    true_dicts.append(parsed)
                    species_taxonomy_map[parsed["species"]] = parsed
                else:
                    true_dicts.append({"family": "unknown", "genus": "unknown", "species": "unknown"})

            for pred_sp in y_pred_species:
                if pred_sp in species_taxonomy_map:
                    pred_dicts.append(species_taxonomy_map[pred_sp])
                else:
                    pred_dicts.append({"family": "unknown", "genus": "unknown", "species": pred_sp})

            cm_results = generate_hierarchical_confusion_matrices(true_dicts, pred_dicts)
            for level, data in cm_results.items():
                matrix, labels = data["matrix"], data["labels"]
                plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) * 0.7)))
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels, yticklabels=labels)
                plt.title(f"CM — {level.capitalize()} [{cfg['label']}] ({cm_title_suffix})")
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(cm_dir, f"confusion_matrix_{level}.png"), dpi=300)
                plt.close()

            # Position CM
            if approach_type == "classifier":
                pos_pred_col = f"{approach_name}_predicted_position"
            else:
                pos_pred_col = f"{approach_name}_top1_position"

            if pos_pred_col in df.columns and "ground_truth_position" in df.columns:
                mask = df["ground_truth_position"].notna() & df[pos_pred_col].notna()
                y_true_pos = df.loc[mask, "ground_truth_position"].tolist()
                y_pred_pos = df.loc[mask, pos_pred_col].tolist()
                if y_true_pos:
                    pos_cm = generate_position_confusion_matrix(y_true_pos, y_pred_pos)
                    labels = pos_cm["labels"]
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(pos_cm["matrix"], annot=True, fmt="d", cmap="Oranges",
                                xticklabels=labels, yticklabels=labels)
                    plt.title(f"CM — Position [{cfg['label']}] ({cm_title_suffix})")
                    plt.xlabel("Predicted Position")
                    plt.ylabel("True Position")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(os.path.join(cm_dir, "confusion_matrix_position.png"), dpi=300)
                    plt.close()

    # --- Per-config README ---
    cfg_files = {
        "predictions.csv": "Per-mesh predictions and ground truth",
        "distance_metrics.csv": "Top-1/Top-5 accuracy for distance-based retrieval",
    }
    if cfg["use_classifiers"]:
        cfg_files["detailed_metrics.csv"] = "Full per-classifier, per-class metrics"
    cfg_files["confusion_matrices/"] = "Confusion matrices at family/genus/species/position levels"

    write_run_manifest(
        cfg_dir,
        description=cfg["description"],
        approach=cfg["label"],
        script_path=os.path.abspath(__file__),
        test_data_paths=[os.path.basename(p) for p in mesh_list],
        checkpoint=CKPT,
        metric_learning_method="NCA" if cfg["use_metric_learning"] else None,
        classifier_names=CLASSIFIER_NAMES if cfg["use_classifiers"] else None,
        extra_files=cfg_files,
    )


# ======================================================================
# Cross-configuration comparison
# ======================================================================
print(f"\n{'='*60}")
print("Generating cross-configuration comparison...")
print(f"{'='*60}")

comparison_df = pd.DataFrame(all_config_summaries)
comparison_csv_path = os.path.join(run_dir, "comparison_metrics.csv")
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"Comparison CSV saved to: {comparison_csv_path}")

# --- Comparison charts ---
charts_dir = os.path.join(run_dir, "comparison_charts")
os.makedirs(charts_dir, exist_ok=True)


# ---- Helper: build a unified "all approaches" dataframe for a target ----
def _build_unified_rows(comparison_df, target="species"):
    """
    Builds a flat list of dicts with columns:
        Configuration, Approach, Accuracy, F1, Precision, Recall
    covering distance methods AND classifiers for every config, so they can
    all appear on the same bar chart.
    """
    rows = []
    acc_suffix = "top_1_accuracy" if target == "species" else "position_top_1_accuracy"
    for _, row in comparison_df.iterrows():
        label = row["label"]
        # Distance-based approaches
        for method, pretty in [("cosine", "Cosine Top-1"), ("euclidean", "Euclidean Top-1")]:
            acc_key = f"{method}_{acc_suffix}"
            val = row.get(acc_key)
            if val is not None and pd.notna(val):
                rows.append({"Configuration": label, "Approach": pretty,
                             "Accuracy": val, "F1": None, "Precision": None, "Recall": None})
        # Top-5 for species only
        if target == "species":
            for method, pretty in [("cosine", "Cosine Top-5"), ("euclidean", "Euclidean Top-5")]:
                val = row.get(f"{method}_top_5_accuracy")
                if val is not None and pd.notna(val):
                    rows.append({"Configuration": label, "Approach": pretty,
                                 "Accuracy": val, "F1": None, "Precision": None, "Recall": None})
        # Supervised classifiers
        for clf_name in CLASSIFIER_NAMES:
            acc_key = f"{clf_name}_{target}_accuracy"
            f1_key = f"{clf_name}_{target}_f1_macro"
            prec_key = f"{clf_name}_{target}_precision_macro"
            rec_key = f"{clf_name}_{target}_recall_macro"
            acc_val = row.get(acc_key)
            if acc_val is not None and pd.notna(acc_val):
                rows.append({
                    "Configuration": label,
                    "Approach": clf_name,
                    "Accuracy": acc_val,
                    "F1": row.get(f1_key) if pd.notna(row.get(f1_key, None)) else None,
                    "Precision": row.get(prec_key) if pd.notna(row.get(prec_key, None)) else None,
                    "Recall": row.get(rec_key) if pd.notna(row.get(rec_key, None)) else None,
                })
    return rows


# ======================================================================
# SPECIES charts
# ======================================================================
species_rows = _build_unified_rows(comparison_df, target="species")

if species_rows:
    sp_df = pd.DataFrame(species_rows)

    # Chart S1: Unified accuracy — all approaches x all configs on one chart
    plt.figure(figsize=(18, 8))
    sns.barplot(data=sp_df, x="Approach", y="Accuracy", hue="Configuration", palette="Set2")
    plt.title("Species — Instance Accuracy: All Approaches Across Configurations")
    plt.ylim(0, 1.05)
    plt.ylabel("Instance Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "species_accuracy_all_approaches.png"), dpi=300)
    plt.close()

    # Chart S2: Unified F1 — classifiers only (distance has no F1)
    f1_rows = [r for r in species_rows if r["F1"] is not None]
    if f1_rows:
        f1_df = pd.DataFrame(f1_rows)
        plt.figure(figsize=(14, 7))
        sns.barplot(data=f1_df, x="Approach", y="F1", hue="Configuration", palette="Set1")
        plt.title("Species — F1 (Macro): Classifiers Across Configurations")
        plt.ylim(0, 1.05)
        plt.ylabel("F1 Score (macro)")
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "species_f1_all_approaches.png"), dpi=300)
        plt.close()

    # Chart S3: Precision & Recall side-by-side for classifiers
    prec_rec_rows = [r for r in species_rows if r["Precision"] is not None]
    if prec_rec_rows:
        pr_df = pd.DataFrame(prec_rec_rows)
        pr_melted = pd.melt(pr_df, id_vars=["Configuration", "Approach"],
                            value_vars=["Precision", "Recall"],
                            var_name="Metric", value_name="Score")
        plt.figure(figsize=(16, 7))
        sns.barplot(data=pr_melted, x="Approach", y="Score", hue="Metric",
                    palette="coolwarm",
                    # Group by config via faceting below
                    )
        plt.title("Species — Precision & Recall (Macro): Classifiers")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "species_precision_recall.png"), dpi=300)
        plt.close()

    # Chart S4: Distance-based retrieval comparison across configs
    dist_only = [r for r in species_rows if "Top-" in r["Approach"]]
    if dist_only:
        dist_df = pd.DataFrame(dist_only)
        plt.figure(figsize=(14, 7))
        sns.barplot(data=dist_df, x="Configuration", y="Accuracy", hue="Approach", palette="viridis")
        plt.title("Species — Distance-Based Retrieval Accuracy Across Configurations")
        plt.ylim(0, 1.05)
        plt.ylabel("Accuracy")
        plt.xticks(rotation=15, ha="right")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "species_distance_comparison.png"), dpi=300)
        plt.close()


# ======================================================================
# POSITION (Spinal Region) charts
# ======================================================================
position_rows = _build_unified_rows(comparison_df, target="position")

if position_rows:
    pos_df = pd.DataFrame(position_rows)

    # Chart P1: Unified accuracy — all approaches x all configs
    plt.figure(figsize=(18, 8))
    sns.barplot(data=pos_df, x="Approach", y="Accuracy", hue="Configuration", palette="Set2")
    plt.title("Spinal Position — Instance Accuracy: All Approaches Across Configurations")
    plt.ylim(0, 1.05)
    plt.ylabel("Instance Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "position_accuracy_all_approaches.png"), dpi=300)
    plt.close()

    # Chart P2: Unified F1 — classifiers only
    f1_rows_pos = [r for r in position_rows if r["F1"] is not None]
    if f1_rows_pos:
        f1_df_pos = pd.DataFrame(f1_rows_pos)
        plt.figure(figsize=(14, 7))
        sns.barplot(data=f1_df_pos, x="Approach", y="F1", hue="Configuration", palette="Set1")
        plt.title("Spinal Position — F1 (Macro): Classifiers Across Configurations")
        plt.ylim(0, 1.05)
        plt.ylabel("F1 Score (macro)")
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Configuration", fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "position_f1_all_approaches.png"), dpi=300)
        plt.close()

    # Chart P3: Precision & Recall side-by-side for classifiers
    prec_rec_pos = [r for r in position_rows if r["Precision"] is not None]
    if prec_rec_pos:
        pr_df_pos = pd.DataFrame(prec_rec_pos)
        pr_melted_pos = pd.melt(pr_df_pos, id_vars=["Configuration", "Approach"],
                                value_vars=["Precision", "Recall"],
                                var_name="Metric", value_name="Score")
        plt.figure(figsize=(16, 7))
        sns.barplot(data=pr_melted_pos, x="Approach", y="Score", hue="Metric", palette="coolwarm")
        plt.title("Spinal Position — Precision & Recall (Macro): Classifiers")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "position_precision_recall.png"), dpi=300)
        plt.close()

    # Chart P4: Distance-based position retrieval
    dist_only_pos = [r for r in position_rows if "Top-1" in r["Approach"]]
    if dist_only_pos:
        dist_df_pos = pd.DataFrame(dist_only_pos)
        plt.figure(figsize=(14, 7))
        sns.barplot(data=dist_df_pos, x="Configuration", y="Accuracy", hue="Approach", palette="viridis")
        plt.title("Spinal Position — Distance-Based Retrieval Accuracy Across Configurations")
        plt.ylim(0, 1.05)
        plt.ylabel("Accuracy")
        plt.xticks(rotation=15, ha="right")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "position_distance_comparison.png"), dpi=300)
        plt.close()


# ======================================================================
# Combined best-of-each-approach chart (species + position side-by-side)
# ======================================================================
combined_chart_data = []
for _, row in comparison_df.iterrows():
    label = row["label"]
    for target_name, target_key, dist_key in [
        ("Species", "species", "top_1_accuracy"),
        ("Position", "position", "position_top_1_accuracy"),
    ]:
        # Best distance accuracy
        best_dist = max(
            row.get(f"cosine_{dist_key}", 0) or 0,
            row.get(f"euclidean_{dist_key}", 0) or 0,
        )
        if best_dist > 0:
            combined_chart_data.append({"Configuration": label, "Target": target_name,
                                        "Approach": "Best Distance (Top-1)", "Accuracy": best_dist})
        # Best classifier accuracy
        best_clf_acc = 0
        for clf_name in CLASSIFIER_NAMES:
            val = row.get(f"{clf_name}_{target_key}_accuracy", 0) or 0
            if val > best_clf_acc:
                best_clf_acc = val
        if best_clf_acc > 0:
            combined_chart_data.append({"Configuration": label, "Target": target_name,
                                        "Approach": "Best Classifier", "Accuracy": best_clf_acc})

if combined_chart_data:
    combined_df = pd.DataFrame(combined_chart_data)
    # Faceted by target
    g = sns.catplot(data=combined_df, x="Configuration", y="Accuracy", hue="Approach",
                    col="Target", kind="bar", palette="Dark2", height=6, aspect=1.3)
    g.set_titles("{col_name} Prediction")
    g.set_xticklabels(rotation=20, ha="right")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1.05)
    g.fig.suptitle("Best Approach per Configuration — Species vs Position", y=1.02)
    g.tight_layout()
    g.savefig(os.path.join(charts_dir, "best_approach_species_vs_position.png"), dpi=300)
    plt.close()


# ======================================================================
# Summary statistics markdown
# ======================================================================
summary_md_path = os.path.join(run_dir, "summary_statistics.md")
with open(summary_md_path, "w", encoding="utf-8") as f:
    f.write("# Ablation Study: Summary Statistics\n\n")
    f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    f.write(f"**Checkpoint:** {CKPT}  \n")
    f.write(f"**Test Meshes:** {len(mesh_list)}  \n")
    f.write(f"**Training Samples:** {len(X_train_valid)} ({len(X_train_raw) - len(X_train_valid)} skipped)  \n\n")

    f.write("## Configurations\n\n")
    f.write("| ID | Configuration | Metric Learning | Classifiers |\n")
    f.write("|----|---------------|-----------------|-------------|\n")
    for cfg_name, cfg in ABLATION_CONFIGS.items():
        ml = "NCA" if cfg["use_metric_learning"] else "None"
        clf = "KNN, SVM, RF, MLP, LR" if cfg["use_classifiers"] else "None"
        f.write(f"| {cfg_name} | {cfg['label']} | {ml} | {clf} |\n")
    f.write("\n")

    # ---- Helper for formatting metric values ----
    def _fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else str(val if val is not None else "N/A")

    # ================================================================
    # SPECIES PREDICTION
    # ================================================================
    f.write("---\n\n# Species Prediction\n\n")

    # Distance-based species table
    f.write("## Distance-Based Species Retrieval\n\n")
    f.write("| Configuration | Cosine Top-1 | Cosine Top-5 | Euclidean Top-1 | Euclidean Top-5 |\n")
    f.write("|---------------|-------------|-------------|----------------|----------------|\n")
    for _, row in comparison_df.iterrows():
        f.write(f"| {row['label']} "
                f"| {_fmt(row.get('cosine_top_1_accuracy'))} "
                f"| {_fmt(row.get('cosine_top_5_accuracy'))} "
                f"| {_fmt(row.get('euclidean_top_1_accuracy'))} "
                f"| {_fmt(row.get('euclidean_top_5_accuracy'))} |\n")
    f.write("\n")

    # Classifier species table
    clf_configs = [cfg_name for cfg_name, cfg in ABLATION_CONFIGS.items() if cfg["use_classifiers"]]
    if clf_configs and CLASSIFIER_NAMES:
        f.write("## Supervised Classifier — Species\n\n")
        header = "| Classifier |"
        sep = "|------------|"
        for cfg_name in clf_configs:
            lbl = ABLATION_CONFIGS[cfg_name]['label']
            header += f" {lbl} Acc | {lbl} F1 | {lbl} Prec | {lbl} Rec |"
            sep += "------|------|------|------|"
        f.write(header + "\n" + sep + "\n")

        for clf_name in CLASSIFIER_NAMES:
            row_str = f"| {clf_name} |"
            for cfg_name in clf_configs:
                cfg_row = comparison_df[comparison_df["config"] == cfg_name].iloc[0]
                row_str += (f" {_fmt(cfg_row.get(f'{clf_name}_species_accuracy'))} |"
                            f" {_fmt(cfg_row.get(f'{clf_name}_species_f1_macro'))} |"
                            f" {_fmt(cfg_row.get(f'{clf_name}_species_precision_macro'))} |"
                            f" {_fmt(cfg_row.get(f'{clf_name}_species_recall_macro'))} |")
            f.write(row_str + "\n")
        f.write("\n")

    # ================================================================
    # SPINAL POSITION PREDICTION
    # ================================================================
    f.write("---\n\n# Spinal Position Prediction (Cervical / Thoracic / Lumbar)\n\n")

    # Distance-based position table
    f.write("## Distance-Based Position Retrieval\n\n")
    f.write("| Configuration | Cosine Top-1 | Euclidean Top-1 |\n")
    f.write("|---------------|-------------|----------------|\n")
    for _, row in comparison_df.iterrows():
        f.write(f"| {row['label']} "
                f"| {_fmt(row.get('cosine_position_top_1_accuracy'))} "
                f"| {_fmt(row.get('euclidean_position_top_1_accuracy'))} |\n")
    f.write("\n")

    # Classifier position table
    if clf_configs and CLASSIFIER_NAMES:
        f.write("## Supervised Classifier — Position\n\n")
        header = "| Classifier |"
        sep = "|------------|"
        for cfg_name in clf_configs:
            lbl = ABLATION_CONFIGS[cfg_name]['label']
            header += f" {lbl} Acc | {lbl} F1 | {lbl} Prec | {lbl} Rec |"
            sep += "------|------|------|------|"
        f.write(header + "\n" + sep + "\n")

        for clf_name in CLASSIFIER_NAMES:
            row_str = f"| {clf_name} |"
            for cfg_name in clf_configs:
                cfg_row = comparison_df[comparison_df["config"] == cfg_name].iloc[0]
                row_str += (f" {_fmt(cfg_row.get(f'{clf_name}_position_accuracy'))} |"
                            f" {_fmt(cfg_row.get(f'{clf_name}_position_f1_macro'))} |"
                            f" {_fmt(cfg_row.get(f'{clf_name}_position_precision_macro'))} |"
                            f" {_fmt(cfg_row.get(f'{clf_name}_position_recall_macro'))} |")
            f.write(row_str + "\n")
        f.write("\n")

    # ================================================================
    # IMPACT ANALYSIS
    # ================================================================
    f.write("---\n\n# Impact Analysis\n\n")

    # --- Impact of Metric Learning ---
    f.write("## Impact of Metric Learning\n\n")
    f.write("Comparing A vs B (distance only) and C vs D (with classifiers) "
            "isolates the effect of NCA metric learning.\n\n")

    for pair_label, without_cfg, with_cfg in [
        ("Distance-Based", "A_baseline", "B_metric_learning"),
        ("With Classifiers", "C_classifiers", "D_full_pipeline"),
    ]:
        row_without = comparison_df[comparison_df["config"] == without_cfg]
        row_with = comparison_df[comparison_df["config"] == with_cfg]
        if row_without.empty or row_with.empty:
            continue
        row_without = row_without.iloc[0]
        row_with = row_with.iloc[0]

        f.write(f"### {pair_label}\n\n")

        # Species deltas
        f.write("**Species:**\n")
        for metric_label, key in [("Cosine Top-1", "cosine_top_1_accuracy"),
                                   ("Euclidean Top-1", "euclidean_top_1_accuracy")]:
            before = row_without.get(key)
            after = row_with.get(key)
            if isinstance(before, float) and isinstance(after, float):
                delta = after - before
                sign = "+" if delta >= 0 else ""
                f.write(f"- {metric_label}: {before:.4f} -> {after:.4f} ({sign}{delta:.4f})\n")

        # Position deltas
        f.write("\n**Spinal Position:**\n")
        for metric_label, key in [("Cosine Top-1", "cosine_position_top_1_accuracy"),
                                   ("Euclidean Top-1", "euclidean_position_top_1_accuracy")]:
            before = row_without.get(key)
            after = row_with.get(key)
            if isinstance(before, float) and isinstance(after, float):
                delta = after - before
                sign = "+" if delta >= 0 else ""
                f.write(f"- {metric_label}: {before:.4f} -> {after:.4f} ({sign}{delta:.4f})\n")

        # Classifier deltas (C vs D)
        if pair_label == "With Classifiers" and CLASSIFIER_NAMES:
            f.write("\n**Classifier Species Accuracy (without ML -> with ML):**\n")
            for clf_name in CLASSIFIER_NAMES:
                before = row_without.get(f"{clf_name}_species_accuracy")
                after = row_with.get(f"{clf_name}_species_accuracy")
                if isinstance(before, float) and isinstance(after, float):
                    delta = after - before
                    sign = "+" if delta >= 0 else ""
                    f.write(f"- {clf_name}: {before:.4f} -> {after:.4f} ({sign}{delta:.4f})\n")
            f.write("\n**Classifier Position Accuracy (without ML -> with ML):**\n")
            for clf_name in CLASSIFIER_NAMES:
                before = row_without.get(f"{clf_name}_position_accuracy")
                after = row_with.get(f"{clf_name}_position_accuracy")
                if isinstance(before, float) and isinstance(after, float):
                    delta = after - before
                    sign = "+" if delta >= 0 else ""
                    f.write(f"- {clf_name}: {before:.4f} -> {after:.4f} ({sign}{delta:.4f})\n")

        f.write("\n")

    # --- Impact of Classifiers ---
    f.write("## Impact of Supervised Classifiers\n\n")
    f.write("Comparing A vs C (without ML) and B vs D (with ML) "
            "isolates the effect of adding classifiers.\n\n")

    for pair_label, without_cfg, with_cfg in [
        ("Without Metric Learning", "A_baseline", "C_classifiers"),
        ("With Metric Learning", "B_metric_learning", "D_full_pipeline"),
    ]:
        row_without = comparison_df[comparison_df["config"] == without_cfg]
        row_with = comparison_df[comparison_df["config"] == with_cfg]
        if row_without.empty or row_with.empty:
            continue
        row_without = row_without.iloc[0]
        row_with = row_with.iloc[0]

        f.write(f"### {pair_label}\n\n")

        for target_name, dist_key, clf_target_key in [
            ("Species", "top_1_accuracy", "species_accuracy"),
            ("Spinal Position", "position_top_1_accuracy", "position_accuracy"),
        ]:
            cos_val = row_without.get(f"cosine_{dist_key}", 0) or 0
            euc_val = row_without.get(f"euclidean_{dist_key}", 0) or 0
            best_dist = max(cos_val, euc_val)

            best_clf = 0
            best_clf_name = "N/A"
            for clf_name in CLASSIFIER_NAMES:
                val = row_with.get(f"{clf_name}_{clf_target_key}", 0) or 0
                if val > best_clf:
                    best_clf = val
                    best_clf_name = clf_name

            if best_dist > 0 or best_clf > 0:
                f.write(f"**{target_name}:**\n")
                f.write(f"- Best Distance Top-1: {best_dist:.4f}\n")
                if best_clf > 0:
                    delta = best_clf - best_dist
                    sign = "+" if delta >= 0 else ""
                    f.write(f"- Best Classifier ({best_clf_name}): {best_clf:.4f} ({sign}{delta:.4f})\n")
                f.write("\n")

        f.write("\n")

    # Training times
    f.write("## Training Times\n\n")
    f.write("| Component | Time (s) |\n")
    f.write("|-----------|----------|\n")
    f.write(f"| NCA Metric Learning | {ml_time:.4f} |\n")
    for name, t in classifiers_raw_times.items():
        f.write(f"| {name} (raw latents) | {t:.4f} |\n")
    if classifiers_ml_times:
        for name, t in classifiers_ml_times.items():
            f.write(f"| {name} (NCA-transformed) | {t:.4f} |\n")
    f.write("\n")

print(f"Summary statistics written to: {summary_md_path}")


# ======================================================================
# Save training times
# ======================================================================
all_training_times = {
    "metric_learning_NCA": ml_time,
    "classifiers_raw": classifiers_raw_times,
    "classifiers_ml": classifiers_ml_times,
}
with open(os.path.join(run_dir, "training_times.json"), "w", encoding="utf-8") as f:
    json.dump(all_training_times, f, indent=2)


# ======================================================================
# Write top-level run manifest
# ======================================================================
produced_files = {
    "README.md": "Top-level run manifest for this ablation study",
    "summary_statistics.md": "Human-readable summary with comparison tables and impact analysis",
    "comparison_metrics.csv": "All configurations side-by-side in a single CSV",
    "training_times.json": "Training times for all components",
    "comparison_charts/": "Cross-configuration comparison visualizations",
    "comparison_charts/species_accuracy_all_approaches.png": "All approaches x all configs — species accuracy",
    "comparison_charts/species_f1_all_approaches.png": "Classifier F1 across configs — species",
    "comparison_charts/species_precision_recall.png": "Precision & recall across configs — species",
    "comparison_charts/species_distance_comparison.png": "Distance retrieval across configs — species",
    "comparison_charts/position_accuracy_all_approaches.png": "All approaches x all configs — position accuracy",
    "comparison_charts/position_f1_all_approaches.png": "Classifier F1 across configs — position",
    "comparison_charts/position_precision_recall.png": "Precision & recall across configs — position",
    "comparison_charts/position_distance_comparison.png": "Distance retrieval across configs — position",
    "comparison_charts/best_approach_species_vs_position.png": "Best approach per config, species vs position faceted",
}
for cfg_name, cfg in ABLATION_CONFIGS.items():
    produced_files[f"{cfg_name}/"] = cfg["description"]

write_run_manifest(
    run_dir,
    description=(
        f"Ablation study comparing {len(ABLATION_CONFIGS)} classification configurations "
        f"on {len(mesh_list)} validation meshes from checkpoint {CKPT}. "
        f"Isolates the contribution of NCA metric learning and supervised classifiers."
    ),
    approach="Ablation: Baseline / +Metric Learning / +Classifiers / +Both",
    script_path=os.path.abspath(__file__),
    train_data_paths=train_paths,
    test_data_paths=[os.path.basename(p) for p in mesh_list],
    config=config,
    classifier_names=CLASSIFIER_NAMES,
    metric_learning_method="NCA (toggled per config)",
    checkpoint=CKPT,
    notes=(
        "Ablation Configurations:\n"
        "  A) Baseline — raw latent codes, distance-based retrieval only\n"
        "  B) + Metric Learning — NCA-transformed latents, distance-based retrieval\n"
        "  C) + Classifiers — raw latents, supervised classifiers (KNN/SVM/RF/MLP/LR)\n"
        "  D) + ML + Classifiers — NCA-transformed latents + classifiers (full v3 pipeline)\n\n"
        "Metrics computed per configuration:\n"
        "- Instance Accuracy, Average Class Accuracy\n"
        "- Precision (macro, weighted, per-class)\n"
        "- Recall (macro, weighted, per-class)\n"
        "- F1 (macro, weighted, per-class)\n"
        "- Top-1 / Top-5 Accuracy (distance methods)\n"
        "- Confusion Matrices (species, genus, family, spinal position)\n"
        "- Training and inference times\n\n"
        "Cross-configuration outputs:\n"
        "- Side-by-side comparison CSV and markdown tables\n"
        "- Impact analysis: isolated effect of metric learning and classifiers\n"
        "- Comparison bar charts"
    ),
    extra_files=produced_files,
)

print(f"\n{'='*60}")
print(f"Ablation study complete! All results saved to:")
print(f"  {run_dir}")
print(f"{'='*60}")
