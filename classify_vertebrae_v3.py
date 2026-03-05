# classify_vertebrae_v3.py
# ========================
# Version 3: Adds dynamic, non-overwriting run directories, comprehensive
# evaluation metrics (per Project Proposal V2), and README manifests.
#
# Key changes vs v2:
#   - Results go into a timestamped directory under results/
#   - Every run directory gets a README.md manifest
#   - After all meshes are classified, aggregate metrics are computed for
#     every approach (each supervised classifier + cosine + euclidean)
#   - All metrics from the proposal are exported:
#       Instance Accuracy, Avg Class Accuracy, Precision, Recall, F1
#       (macro + weighted + per-class), Top-5 Accuracy, confusion matrices
#       at species / genus / family / position levels
#   - Training times are logged to JSON
#   - A summary_statistics.md is written
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import torch.nn.functional as F
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.mesh import get_sdfs, create_mesh
import vtk
import re
import random

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
from run_utils import create_run_directory, write_run_manifest, update_manifest_files_table

# Monkey-patch pymskt
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

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

val_paths = config['val_paths']
mesh_list_raw = val_paths  # Evaluates on all validation paths instead of just 1
mesh_list = [os.path.join(mesh_dir, os.path.basename(p)) for p in mesh_list_raw]

# ======================================================================
# Create dynamic run directory
# ======================================================================
run_dir = create_run_directory(base_dir="results", prefix="classify")
predictions_dir = os.path.join(run_dir, "predictions")
os.makedirs(predictions_dir, exist_ok=True)
print(f"\n{'='*60}")
print(f"Run directory: {run_dir}")
print(f"{'='*60}\n")

# ======================================================================
# Define functions
# ======================================================================
def optimize_latent(decoder, points, sdf_vals, latent_size, iters=1000, lr=1e-3):
    init_latent_torch = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg)
    latent = init_latent_torch.clone().detach().requires_grad_()
    optimizer = torch.optim.Adam([latent], lr=lr)
    sdf_vals = sdf_vals.to(device)
    decoder = decoder.to(device)
    points = points.to(device)
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
X_train_all = latent_codes.cpu().numpy()

y_train_species = []
y_train_positions = []
valid_indices = []

for f_idx, f in enumerate(all_vtk_files):
    parsed = parse_taxonomy_from_filename(f)
    if parsed and parsed.get('species') and parsed.get('position'):
        y_train_species.append(parsed['species'])
        y_train_positions.append(parsed['position'])
        valid_indices.append(f_idx)
    else:
        print(f"Skipping training file with invalid label format: {f}")

X_train = X_train_all[valid_indices]
y_train = np.column_stack((y_train_species, y_train_positions))

# ======================================================================
# Metric Learning
# ======================================================================
unique_species = np.unique(y_train_species)
if len(y_train_species) > 0 and len(unique_species) > 1:
    print("Applying NCA Metric Learning transformation to latent space (via species)...")
    metric_learner = LatentMetricLearner(method='NCA', max_iter=100)
    X_train = metric_learner.fit_transform(X_train, np.array(y_train_species))
    print("Latent space transformed successfully.")
else:
    metric_learner = None

# ======================================================================
# Train Classifiers
# ======================================================================
if len(y_train) == 0:
    raise ValueError("No valid training labels found!")

print("Training suite of multi-output classifiers...")
trained_models, training_times = train_classifiers(X_train, y_train)
for name, t in training_times.items():
    print(f"  {name} trained in {t:.3f}s")
print(f"Classifiers trained on {len(X_train)} samples ({len(X_train_all) - len(X_train)} skipped).")

# Save training times
training_times_path = os.path.join(run_dir, "training_times.json")
with open(training_times_path, "w") as f:
    json.dump(training_times, f, indent=2)

# ======================================================================
# Classify meshes
# ======================================================================
summary_log = []

for mesh_idx, vert_fname in enumerate(mesh_list):
    mesh_basename = os.path.basename(vert_fname)
    mesh_stem = os.path.splitext(mesh_basename)[0]
    print(f"\033[32m\n=== Processing {mesh_basename} ({mesh_idx + 1}/{len(mesh_list)}) ===\033[0m")

    # Per-mesh output directory
    outfpath = os.path.join(predictions_dir, mesh_stem)
    os.makedirs(outfpath, exist_ok=True)

    # --- Set up inference dataset ---
    if '.ply' in vert_fname:
        _, vert_fname = convert_ply_to_vtk(vert_fname)

    sdf_dataset = SDFSamples(
        list_mesh_paths=[vert_fname],
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
    points = sample_dict['xyz'].to(device)
    sdf_vals = sample_dict['gt_sdf']

    # Optimize latents
    print("Optimizing latents")
    latent_novel = optimize_latent(model, points, sdf_vals, config['latent_size'])
    print("Translated novel mesh into latent space!")

    # --- Predict Class ---
    novel_vec_raw = latent_novel.cpu().detach().numpy()
    novel_vec = metric_learner.transform(novel_vec_raw) if metric_learner is not None else novel_vec_raw

    predictions, probabilities, inference_times = predict_classifiers(trained_models, novel_vec)

    knn_preds = predictions['KNN'][0]
    print(f"Predicted Species (KNN): {knn_preds[0]}")
    print(f"Predicted Position (KNN): {knn_preds[1]}")

    # --- Distance-based classification ---
    similar_ids_cos, distances_cos = find_similar_cos(latent_novel, latent_codes, top_k=5, n_std=2, device=device)
    similar_ids_euc, distances_euc = find_similar(latent_novel, latent_codes, top_k=5, n_std=2, device=device)

    # --- Save detailed classification results ---
    results_fpath = os.path.join(outfpath, 'classification_results.txt')
    from sklearn.multioutput import MultiOutputClassifier
    with open(results_fpath, "w") as f:
        f.write(f"Classification Results for: {mesh_basename}\n")
        f.write("=" * 60 + "\n\n")

        # 1. Supervised Classifiers
        f.write("--- SUPERVISED CLASSIFIERS ---\n")
        for clf_name, clf_model in trained_models.items():
            preds = predictions[clf_name][0]
            f.write(f"[{clf_name}]\n")
            f.write(f"  Predicted Species: {preds[0]}\n")
            f.write(f"  Predicted Position: {preds[1]}\n")
            f.write(f"  Inference Time: {inference_times[clf_name]:.4f}s\n")
            if probabilities[clf_name] is not None:
                probs = probabilities[clf_name]
                if isinstance(clf_model, MultiOutputClassifier):
                    estimators = clf_model.estimators_
                    sp_classes = estimators[0].classes_
                    sp_probs = probs[0][0]
                    f.write("  Species Probabilities (>1%):\n")
                    for cls, prob in zip(sp_classes, sp_probs):
                        if prob > 0.01:
                            f.write(f"    {cls}: {prob:.2%}\n")
                    pos_classes = estimators[1].classes_
                    pos_probs = probs[1][0]
                    f.write("  Position Probabilities (>1%):\n")
                    for cls, prob in zip(pos_classes, pos_probs):
                        if prob > 0.01:
                            f.write(f"    {cls}: {prob:.2%}\n")
            f.write("\n")

        # 2. Cosine Similarity
        f.write("--- COSINE SIMILARITY (Top-5) ---\n")
        for rank, (idx, d) in enumerate(zip(similar_ids_cos, distances_cos), 1):
            species = extract_species_prefix(all_vtk_files[idx])
            f.write(f"  {rank}. {all_vtk_files[idx]} (distance: {d:.4f}, species: {species})\n")
        f.write("\n")

        # 3. Euclidean Distance
        f.write("--- EUCLIDEAN DISTANCE (Top-5) ---\n")
        for rank, (idx, d) in enumerate(zip(similar_ids_euc, distances_euc), 1):
            species = extract_species_prefix(all_vtk_files[idx])
            f.write(f"  {rank}. {all_vtk_files[idx]} (distance: {d:.4f}, species: {species})\n")
        f.write("\n")

    print(f"Classification results saved to: {results_fpath}")

    # --- Visualization: PCA ---
    latents = latent_codes.cpu().numpy()
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(latents)
    novel_coord = pca.transform(latent_novel.cpu().numpy())[0]
    similar_coords = coords_2d[similar_ids_cos]

    plt.figure(figsize=(8, 6))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], color='gray', alpha=0.3, label='Training Meshes')
    plt.scatter(similar_coords[0, 0], similar_coords[0, 1], color='hotpink', s=80, label='Most Similar')
    if len(similar_coords) > 1:
        plt.scatter(similar_coords[1:, 0], similar_coords[1:, 1], color='blue', s=60, label='Other Top-5 Similar')
    plt.scatter(*novel_coord, color='red', s=80, label='Novel Mesh')
    for idx_c, (x, y) in zip(similar_ids_cos, similar_coords):
        plt.text(x, y, all_vtk_files[idx_c].split('.')[0], fontsize=6, color='black')
    plt.title("Latent Space Visualization (PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outfpath, "latent_space_pca.png"), dpi=300)
    plt.close()

    # --- Visualization: t-SNE ---
    latent_novel_np = latent_novel.detach().cpu().numpy()
    latents_with_novel = np.vstack([latents, latent_novel_np])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    coords_with_novel = tsne.fit_transform(latents_with_novel)
    train_coords = coords_with_novel[:-1]
    novel_coord_tsne = coords_with_novel[-1]
    similar_coords_tsne = train_coords[similar_ids_cos]

    plt.figure(figsize=(8, 6))
    plt.scatter(train_coords[:, 0], train_coords[:, 1], color='grey', alpha=0.1, label='Training Meshes')
    plt.scatter(similar_coords_tsne[0, 0], similar_coords_tsne[0, 1], color='hotpink', alpha=0.5, label='Most Similar')
    if len(similar_coords_tsne) > 1:
        plt.scatter(similar_coords_tsne[1:, 0], similar_coords_tsne[1:, 1], color='blue', alpha=0.5, label='Other Top-5 Similar')
    plt.scatter(*novel_coord_tsne, color='red', alpha=0.5, label='Novel Mesh')
    for idx_c, (x, y) in zip(similar_ids_cos, similar_coords_tsne):
        plt.text(x, y, all_vtk_files[idx_c].split('.')[0], fontsize=6, color='black')
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outfpath, "latent_space_tsne.png"), dpi=300)
    plt.close()

    # --- Reconstruct mesh ---
    recon_grid_origin = 1.0
    n_pts_per_axis = 256
    voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
    voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
    offset = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    icp_transform = NumpyTransform(np.eye(4))
    objects = 1

    mesh_out = create_mesh(
        decoder=model, latent_vector=latent_novel,
        n_pts_per_axis=n_pts_per_axis, voxel_origin=voxel_origin,
        voxel_size=voxel_size, path_original_mesh=None,
        offset=offset, scale=scale, icp_transform=icp_transform,
        objects=objects, verbose=True, device=device,
    )

    if isinstance(mesh_out, list):
        mesh_out = mesh_out[0]
    mesh_pv = mesh_out.extract_geometry() if not isinstance(mesh_out, pv.PolyData) else mesh_out

    output_mesh_path = os.path.join(outfpath, f"{mesh_stem}_decoded_novel.vtk")
    mesh_pv.save(output_mesh_path)
    print(f"Novel mesh saved to: {output_mesh_path}")

    # --- Build summary log entry ---
    parsed_truth = parse_taxonomy_from_filename(mesh_basename)
    mesh_species = parsed_truth.get('species') if parsed_truth else extract_species_prefix(mesh_basename)
    mesh_position = parsed_truth.get('position') if parsed_truth else None

    cos_top1_species = extract_species_prefix(all_vtk_files[similar_ids_cos[0]]) if similar_ids_cos else None
    cos_top1_match = "yes" if mesh_species and mesh_species == cos_top1_species else "no"
    cos_top5_match = "yes" if any(extract_species_prefix(all_vtk_files[idx]) == mesh_species for idx in similar_ids_cos) else "no"

    euc_top1_species = extract_species_prefix(all_vtk_files[similar_ids_euc[0]]) if similar_ids_euc else None
    euc_top1_match = "yes" if mesh_species and mesh_species == euc_top1_species else "no"
    euc_top5_match = "yes" if any(extract_species_prefix(all_vtk_files[idx]) == mesh_species for idx in similar_ids_euc) else "no"

    top_k_summary = {
        "mesh": mesh_basename,
        "ground_truth_species": mesh_species,
        "ground_truth_position": mesh_position,
        "output_mesh": output_mesh_path,
    }

    for clf_name in trained_models.keys():
        pred_sp = predictions[clf_name][0][0]
        pred_pos = predictions[clf_name][0][1]

        match_sp = "yes" if mesh_species and mesh_species == pred_sp else "no"
        match_pos = "yes" if mesh_position and str(mesh_position) == str(pred_pos) else "no"

        if probabilities[clf_name] is not None:
            sp_conf = max(probabilities[clf_name][0][0])
            pos_conf = max(probabilities[clf_name][1][0])
            top_k_summary[f"{clf_name}_species_confidence"] = f"{sp_conf:.2%}"
            top_k_summary[f"{clf_name}_position_confidence"] = f"{pos_conf:.2%}"
        else:
            top_k_summary[f"{clf_name}_species_confidence"] = "N/A"
            top_k_summary[f"{clf_name}_position_confidence"] = "N/A"

        top_k_summary[f"{clf_name}_predicted_species"] = pred_sp
        top_k_summary[f"{clf_name}_predicted_position"] = pred_pos
        top_k_summary[f"{clf_name}_match_species"] = match_sp
        top_k_summary[f"{clf_name}_match_position"] = match_pos
        top_k_summary[f"{clf_name}_inf_time"] = inference_times[clf_name]

    top_k_summary.update({
        "cos_top1_species": cos_top1_species,
        "cos_top1_match": cos_top1_match,
        "cos_top5_match": cos_top5_match,
        "euc_top1_species": euc_top1_species,
        "euc_top1_match": euc_top1_match,
        "euc_top5_match": euc_top5_match,
    })
    for rank, (idx, dist) in enumerate(zip(similar_ids_cos, distances_cos), 1):
        top_k_summary[f"cos_similar_{rank}_name"] = all_vtk_files[idx]
        top_k_summary[f"cos_similar_{rank}_distance"] = dist
    for rank, (idx, dist) in enumerate(zip(similar_ids_euc, distances_euc), 1):
        top_k_summary[f"euc_similar_{rank}_name"] = all_vtk_files[idx]
        top_k_summary[f"euc_similar_{rank}_distance"] = dist

    summary_log.append(top_k_summary)

    # Write per-mesh README
    per_mesh_files = {
        "classification_results.txt": "Detailed classification results from all approaches",
        "latent_space_pca.png": "PCA visualization of the latent space with novel mesh",
        "latent_space_tsne.png": "t-SNE visualization of the latent space with novel mesh",
        f"{mesh_stem}_decoded_novel.vtk": "Reconstructed mesh from the optimized latent vector",
    }
    write_run_manifest(
        outfpath,
        description=f"Classification results for {mesh_basename}",
        approach="Supervised (KNN/SVM/RF/MLP/LR) + Cosine/Euclidean Distance",
        script_path=os.path.abspath(__file__),
        test_data_paths=[vert_fname],
        checkpoint=CKPT,
        metric_learning_method="NCA" if metric_learner else None,
        extra_files=per_mesh_files,
    )


# ======================================================================
# Export summary CSV
# ======================================================================
summary_csv_path = os.path.join(run_dir, "summary_predictions.csv")
df = pd.DataFrame(summary_log)
df.to_csv(summary_csv_path, index=False)
print(f"\nSummary CSV saved to: {summary_csv_path}")


# ======================================================================
# Compute aggregate metrics (across all evaluated meshes)
# ======================================================================
print("\n" + "=" * 60)
print("Computing aggregate classification metrics...")
print("=" * 60)

CLASSIFIER_NAMES = list(trained_models.keys())
all_metrics_rows = []

for clf_name in CLASSIFIER_NAMES:
    sp_col = f"{clf_name}_predicted_species"
    pos_col = f"{clf_name}_predicted_position"

    # --- Species metrics ---
    if sp_col in df.columns:
        y_true_sp = df["ground_truth_species"].tolist()
        y_pred_sp = df[sp_col].tolist()
        sp_metrics = calculate_metrics(y_true_sp, y_pred_sp)
        sp_df = metrics_to_dataframe(sp_metrics, classifier_name=clf_name)
        sp_df["target"] = "species"
        all_metrics_rows.append(sp_df)

    # --- Position metrics ---
    if pos_col in df.columns and "ground_truth_position" in df.columns:
        mask = df["ground_truth_position"].notna() & df[pos_col].notna()
        y_true_pos = df.loc[mask, "ground_truth_position"].tolist()
        y_pred_pos = df.loc[mask, pos_col].tolist()
        if y_true_pos:
            pos_metrics = calculate_metrics(y_true_pos, y_pred_pos)
            pos_df = metrics_to_dataframe(pos_metrics, classifier_name=clf_name)
            pos_df["target"] = "position"
            all_metrics_rows.append(pos_df)

if all_metrics_rows:
    metrics_df = pd.concat(all_metrics_rows, ignore_index=True)
    metrics_csv_path = os.path.join(run_dir, "detailed_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Detailed metrics CSV saved to: {metrics_csv_path}")

    # Print summary to console
    summary_rows = metrics_df[metrics_df["level"] == "summary"]
    for _, row in summary_rows.iterrows():
        target = row.get("target", "?")
        clf = row.get("classifier", "?")
        acc = row.get("instance_accuracy", "N/A")
        avg_acc = row.get("average_class_accuracy", "N/A")
        f1 = row.get("f1_macro", "N/A")
        print(f"  [{clf}] {target}: Acc={acc:.4f}, AvgClassAcc={avg_acc:.4f}, F1={f1:.4f}"
              if isinstance(acc, float) else f"  [{clf}] {target}: {acc}")

# Distance-based metrics
distance_rows = []
for method in ["cos", "euc"]:
    label = "Cosine Similarity" if method == "cos" else "Euclidean Distance"
    t1 = f"{method}_top1_match"
    t5 = f"{method}_top5_match"
    if t1 in df.columns:
        distance_rows.append({"method": label, "metric": "top_1_accuracy",
                              "value": (df[t1] == "yes").mean()})
    if t5 in df.columns:
        distance_rows.append({"method": label, "metric": "top_5_accuracy",
                              "value": (df[t5] == "yes").mean()})

distance_df = pd.DataFrame(distance_rows)
if not distance_df.empty:
    distance_csv = os.path.join(run_dir, "distance_metrics.csv")
    distance_df.to_csv(distance_csv, index=False)
    print(f"Distance metrics saved to: {distance_csv}")


# ======================================================================
# Write summary statistics markdown
# ======================================================================
summary_md_path = os.path.join(run_dir, "summary_statistics.md")
with open(summary_md_path, "w") as f:
    f.write("# Summary Statistics\n\n")
    f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    f.write(f"**Checkpoint:** {CKPT}  \n")
    f.write(f"**Metric Learning:** {'NCA' if metric_learner else 'None'}  \n")
    f.write(f"**Test Meshes:** {len(mesh_list)}  \n\n")

    if all_metrics_rows:
        sp_summary = metrics_df[
            (metrics_df["target"] == "species") & (metrics_df["level"] == "summary")
        ]
        if not sp_summary.empty:
            f.write("## Species Prediction\n\n")
            cols = ["classifier", "instance_accuracy", "average_class_accuracy",
                    "precision_macro", "recall_macro", "f1_macro",
                    "precision_weighted", "recall_weighted", "f1_weighted", "n_samples"]
            existing = [c for c in cols if c in sp_summary.columns]
            f.write(sp_summary[existing].to_markdown(index=False))
            f.write("\n\n")

        pos_summary = metrics_df[
            (metrics_df["target"] == "position") & (metrics_df["level"] == "summary")
        ]
        if not pos_summary.empty:
            f.write("## Spinal Position Prediction\n\n")
            cols = ["classifier", "instance_accuracy", "average_class_accuracy",
                    "precision_macro", "recall_macro", "f1_macro", "n_samples"]
            existing = [c for c in cols if c in pos_summary.columns]
            f.write(pos_summary[existing].to_markdown(index=False))
            f.write("\n\n")

    if not distance_df.empty:
        f.write("## Distance-Based Retrieval\n\n")
        f.write(distance_df.to_markdown(index=False))
        f.write("\n\n")

    f.write("## Training Times\n\n")
    f.write("| Classifier | Training Time (s) |\n")
    f.write("|------------|-------------------|\n")
    for name, t in training_times.items():
        f.write(f"| {name} | {t:.4f} |\n")
    f.write("\n")

print(f"Summary statistics written to: {summary_md_path}")


# ======================================================================
# Generate confusion matrices (if enough data)
# ======================================================================
cm_dir = os.path.join(run_dir, "confusion_matrices")
os.makedirs(cm_dir, exist_ok=True)

if len(df) >= 2:
    # Pick best classifier
    best_clf, best_acc = "KNN", 0
    for clf_name in CLASSIFIER_NAMES:
        col = f"{clf_name}_predicted_species"
        if col in df.columns:
            acc = (df["ground_truth_species"] == df[col]).mean()
            if acc > best_acc:
                best_acc, best_clf = acc, clf_name

    print(f"\nGenerating confusion matrices using best classifier: {best_clf} ({best_acc:.1%})")

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

    for pred_sp in df[f"{best_clf}_predicted_species"]:
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
        plt.title(f"Confusion Matrix — {level.capitalize()} Level ({best_clf})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, f"confusion_matrix_{level}.png"), dpi=300)
        plt.close()

    # Position CM
    pos_col = f"{best_clf}_predicted_position"
    if pos_col in df.columns and "ground_truth_position" in df.columns:
        mask = df["ground_truth_position"].notna() & df[pos_col].notna()
        y_true_pos = df.loc[mask, "ground_truth_position"].tolist()
        y_pred_pos = df.loc[mask, pos_col].tolist()
        if y_true_pos:
            pos_cm = generate_position_confusion_matrix(y_true_pos, y_pred_pos)
            labels = pos_cm["labels"]
            plt.figure(figsize=(8, 6))
            sns.heatmap(pos_cm["matrix"], annot=True, fmt="d", cmap="Oranges",
                        xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix — Spinal Position ({best_clf})")
            plt.xlabel("Predicted Position")
            plt.ylabel("True Position")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(cm_dir, "confusion_matrix_position.png"), dpi=300)
            plt.close()

    print(f"Confusion matrices saved to: {cm_dir}")
else:
    print("Not enough samples for confusion matrices (need >= 2).")


# ======================================================================
# Write top-level run manifest
# ======================================================================
produced_files = {
    "README.md": "Run manifest documenting this experiment",
    "summary_predictions.csv": "Per-mesh predictions from all approaches",
    "detailed_metrics.csv": "Full classification metrics (per-class and summary)",
    "distance_metrics.csv": "Top-1/Top-5 accuracy for distance-based approaches",
    "training_times.json": "Time taken to train each classifier",
    "summary_statistics.md": "Human-readable summary of all classification metrics",
    "predictions/": "Directory containing per-mesh results (plots, reconstructions, etc.)",
    "confusion_matrices/": "Confusion matrices at family/genus/species/position levels",
}

write_run_manifest(
    run_dir,
    description=(
        f"Classification of {len(mesh_list)} vertebra mesh(es) using supervised classifiers "
        f"(KNN, SVM, RF, MLP, LR) + cosine/euclidean distance retrieval, "
        f"with NCA metric learning on latent codes from checkpoint {CKPT}."
    ),
    approach="Supervised Multi-Output Classifiers + Distance-Based Retrieval + NCA Metric Learning",
    script_path=os.path.abspath(__file__),
    train_data_paths=train_paths,
    test_data_paths=[os.path.basename(p) for p in mesh_list],
    config=config,
    classifier_names=list(trained_models.keys()),
    metric_learning_method="NCA" if metric_learner else None,
    checkpoint=CKPT,
    notes=(
        "Metrics computed per Project Proposal V2 requirements:\n"
        "- Instance Accuracy\n"
        "- Average Class Accuracy\n"
        "- Precision (macro, weighted, per-class)\n"
        "- Recall (macro, weighted, per-class)\n"
        "- F1 (macro, weighted, per-class)\n"
        "- Top-5 Accuracy (distance methods)\n"
        "- Confusion Matrices (species, genus, family, spinal position)\n"
        "- Training times for each classifier\n"
        "- Inference times per prediction"
    ),
    extra_files=produced_files,
)

print(f"\n{'='*60}")
print(f"Run complete! All results saved to:")
print(f"  {run_dir}")
print(f"{'='*60}")
