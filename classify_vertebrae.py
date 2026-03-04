# Identify novel meshes from latent space
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import numpy as np
import pandas as pd
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.mesh import get_sdfs  
import torch.nn.functional as F
import json
import pyvista as pv
import pymskt.mesh.meshes as meshes
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from NSM.mesh import create_mesh
import vtk
import re
import random
from NSM.helper_funcs import NumpyTransform, load_config, load_model_and_latents, convert_ply_to_vtk, get_sdfs, fixed_point_coords, safe_load_mesh_scalars, extract_species_prefix
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, find_similar, find_similar_cos
from supervised_classifiers import train_classifiers, predict_classifiers
from metric_learning import LatentMetricLearner
# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

# Define PC index and model checkpoint to use for analysis of novel mdeshes
TRAIN_DIR = "run_v56" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '2000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'

# Load config
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
    
# Override config device if it was hardcoded
config['device'] = device

train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# List of meshes to be classified
# Randomly select test paths but FIX THE PATHS to be local
remote_prefix = "/home/k.wolcott/NSM/nsm/"
local_prefix = os.getcwd() + "/" # running in run_v56, but data might be in .. so let's check
# Actually, the data seems to be flat in run_v56 based on list_dir? No, list_dir showed subdirs.
# Let's assume the files are in `vertebrae_meshes` relative to the root, OR checking run_v56 structure again.
# Wait, list_dir of run_v56 showed `classify_vertebrae`, `latent_codes`, `model`. It did NOT show `vertebrae_meshes`.
# The user's prompt implied they have "everything needed inside run_v56".
# If the meshes aren't there, we might need to look elsewhere.
# But for now, let's look at where `all_vtk_files` come from.
# The config lists absolute paths.
# If the user has the data locally, it's likely in `../vertebrae_meshes` or similar.
# Let's try to find where the .vtk files are. 
# But to be safe, let's try to just use the filename if we can find it.
# Actually, the user's previous context showed `sample.txt` with filenames.
# Let's try to find `vertebrae_meshes` in the parent or current dir.

# Attempt to locate the local mesh directory
potential_dirs = [
    "vertebrae_meshes",
    "../vertebrae_meshes", 
    "../../vertebrae_meshes",
    "data/vertebrae_meshes"
]
mesh_dir = None
for d in potential_dirs:
    if os.path.isdir(d):
        mesh_dir = d
        break
        
if mesh_dir is None:
    print("Warning: Could not locate 'vertebrae_meshes' directory. Testing probably will fail if meshes aren't found.")
    mesh_dir = "." # Fallback

val_paths = config['val_paths']
mesh_list_raw = random.sample(val_paths, min(100, len(val_paths)))
mesh_list = []
for p in mesh_list_raw:
    fname = os.path.basename(p)
    # Construct local path
    local_p = os.path.join(mesh_dir, fname)
    mesh_list.append(local_p)


# Define functions

# Optimie latent vector for inference (since DeepSDF has no encoder, this is how you run novel data through for inference)
def optimize_latent(decoder, points, sdf_vals, latent_size, iters=1000, lr=1e-3):
    init_latent_torch = pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg) # initialize near mean using PCAs for regularization
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

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)
_, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)

# --- Learned Classifier ---
# 1. Prepare Features (X)
X_train_all = latent_codes.cpu().numpy()

# 2. Prepare Labels (Y)
# extract_species_prefix is already imported from NSM.helper_funcs
y_train_all = [extract_species_prefix(f) for f in all_vtk_files]

# Filter out samples where labels could not be extracted (None)
X_train = []
y_train = []
for x, y, f in zip(X_train_all, y_train_all, all_vtk_files):
    if y is not None:
        X_train.append(x)
        y_train.append(y)
    else:
        print(f"Skipping training file with invalid label format: {f}")

X_train = np.array(X_train)
# y_train is a list of strings

# 2.5 Apply Metric Learning to cluster by species
if len(y_train) > 0 and len(np.unique(y_train)) > 1:
    print("Applying NCA Metric Learning transformation to latent space...")
    metric_learner = LatentMetricLearner(method='NCA', max_iter=100)
    X_train = metric_learner.fit_transform(X_train, y_train)
    print("Latent space transformed successfully.")
else:
    metric_learner = None

# 3. Train Classifiers (KNN, SVM, RF, MLP, LR)
if len(y_train) == 0:
    raise ValueError("No valid training labels found! Check regex in extract_species_prefix.")

print("Training suite of classifiers...")
trained_models, training_times = train_classifiers(X_train, y_train)
for name, t in training_times.items():
    print(f"  {name} trained in {t:.3f}s")
print(f"Classifiers trained on {len(X_train)} samples ({len(X_train_all) - len(X_train)} skipped).")
# --------------------------

# Loop through meshes
summary_log = []
for i, vert_fname in enumerate(mesh_list):    
    print(f"\033[32m\n=== Processing {os.path.basename(vert_fname)} ===\033[0m")
    print(f"\033[32m\n=== Mesh {i} / {len(mesh_list)} ===\033[0m")
    # Make a new dir to save predictions
    outfpath = 'classify_vertebrae/predictions/' + os.path.splitext(os.path.basename(vert_fname))[0]
    print("Making a new directory to save model predictions and outputs at: ", outfpath)
    os.makedirs(outfpath, exist_ok=True)

    # --- Set up inference dataset ---

    # Convert plys to vtks
    if '.ply' in vert_fname:
        ply_fname = vert_fname
        _, vert_fname = convert_ply_to_vtk(ply_fname)

    # Setup your dataset with just one mesh
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
        fix_mesh=config['fix_mesh']
        )

    # Get the point/SDF data
    print("Setting up dataset")
    sdf_sample = sdf_dataset[0]  # returns a dict
    sample_dict, _ = sdf_sample
    points = sample_dict['xyz'].to(device) # shape: [N, 3]
    sdf_vals = sample_dict['gt_sdf']  # shape: [N, 1]

    # Optimize latents (DeepSDF has no encoder, so must use optimization to encode novel data)
    print("Optimizing latents")
    latent_novel = optimize_latent(model, points, sdf_vals, config['latent_size'])
    print("Translated novel mesh into latent space!")

    # --- Predict Class ---
    # Convert novel latent to numpy (1, 512)
    novel_vec_raw = latent_novel.cpu().detach().numpy()
    
    # Apply Metric Learning Transform if available
    if metric_learner is not None:
        novel_vec = metric_learner.transform(novel_vec_raw)
    else:
        novel_vec = novel_vec_raw
    
    # Predict with all classifiers
    predictions, probabilities, inference_times = predict_classifiers(trained_models, novel_vec)
    
    print(f"Predicted Species (KNN): {predictions['KNN'][0]}")
    # ---------------------

    # --- Classify vertebra ---

    # Find most similar latents using COSINE SIMILARITY
    similar_ids_cos, distances_cos = find_similar_cos(latent_novel, latent_codes, top_k=5, n_std=2, device=device)

    # Find most similar latents using EUCLIDEAN DISTANCE
    similar_ids_euc, distances_euc = find_similar(latent_novel, latent_codes, top_k=5, n_std=2, device=device)

    # Write most similar meshes to txt file (legacy: cosine only)
    sim_mesh_fpath = outfpath + '/' + 'similar_meshes_pca_regularized_95pct_cos.txt'
    with open(sim_mesh_fpath, "w") as f:
        print(f"Most similar mesh indices to file: {os.path.basename(vert_fname)}\n")
        f.write(f"Most similar mesh indices to file: {os.path.basename(vert_fname)}:\n")
        for i, d in zip(similar_ids_cos, distances_cos):
            line = f"Name: {all_vtk_files[i]}, Index: {i}, Distance: {d:.4f}"
            print(line)
            f.write(line + "\n")

    # --- Save classification results from ALL three methods ---
    results_fpath = outfpath + '/classification_results.txt'
    with open(results_fpath, "w") as f:
        f.write(f"Classification Results for: {os.path.basename(vert_fname)}\n")
        f.write("=" * 60 + "\n\n")
        
        # 1. Supervised Classifiers
        f.write("--- SUPERVISED CLASSIFIERS ---\n")
        for clf_name, clf_model in trained_models.items():
            f.write(f"[{clf_name}]\n")
            f.write(f"  Predicted Species: {predictions[clf_name][0]}\n")
            f.write(f"  Inference Time: {inference_times[clf_name]:.4f}s\n")
            if probabilities[clf_name] is not None:
                classes = clf_model.classes_
                probs = probabilities[clf_name][0]
                for cls, prob in zip(classes, probs):
                    if prob > 0.01:
                        f.write(f"    {cls}: {prob:.2%}\n")
            f.write("\n")
        
        # 2. Cosine Similarity
        f.write("--- COSINE SIMILARITY (Top-5) ---\n")
        for rank, (i, d) in enumerate(zip(similar_ids_cos, distances_cos), 1):
            species = extract_species_prefix(all_vtk_files[i])
            f.write(f"  {rank}. {all_vtk_files[i]} (distance: {d:.4f}, species: {species})\n")
        f.write("\n")
        
        # 3. Euclidean Distance
        f.write("--- EUCLIDEAN DISTANCE (Top-5) ---\n")
        for rank, (i, d) in enumerate(zip(similar_ids_euc, distances_euc), 1):
            species = extract_species_prefix(all_vtk_files[i])
            f.write(f"  {rank}. {all_vtk_files[i]} (distance: {d:.4f}, species: {species})\n")
        f.write("\n")
    
    print(f"Classification results saved to: {results_fpath}")

    # --- Inspect novel latent using clustering analysis ---

    # PCA Plot
    # Data loading
    latents = latent_codes.cpu().numpy()
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(latents)
    novel_coord = pca.transform(latent_novel.cpu().numpy())[0]
    similar_coords = coords_2d[similar_ids_cos]  # Use cosine similarity for plots
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], color='gray', alpha=0.3, label='Training Meshes')
    # Plot most similar (1st one) in pink
    plt.scatter(similar_coords[0, 0], similar_coords[0, 1], color='hotpink', s=80, label='Most Similar')
    # Plot next 4 similar in blue
    if len(similar_coords) > 1:
        plt.scatter(similar_coords[1:, 0], similar_coords[1:, 1], color='blue', s=60, label='Other Top-5 Similar')
    # Plot novel mesh in red
    plt.scatter(*novel_coord, color='red', s=80, label='Novel Mesh')
    # Annotate each of the top-5 similar meshes
    for idx, (x, y) in zip(similar_ids_cos, similar_coords):
        plt.text(x, y, all_vtk_files[idx].split('.')[0], fontsize=6, color='black')
    plt.title("Latent Space Visualization (PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfpath + "/latent_space_pca_pca_regularized_95pct_cos.png", dpi=300)
    plt.close()

    # t-SNE Plot
    # Data loading
    latent_novel_np = latent_novel.detach().cpu().numpy()
    latents_with_novel = np.vstack([latents, latent_novel_np])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    coords_with_novel = tsne.fit_transform(latents_with_novel)
    train_coords = coords_with_novel[:-1]
    novel_coord = coords_with_novel[-1]
    similar_coords = train_coords[similar_ids_cos]  # Use cosine similarity for plots
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(train_coords[:, 0], train_coords[:, 1], color='grey', alpha=0.1, label='Training Meshes')
    # Plot most similar (1st one) in pink
    plt.scatter(similar_coords[0, 0], similar_coords[0, 1], color='hotpink', alpha=0.5, label='Most Similar')
    # Plot next 4 similar in blue
    if len(similar_coords) > 1:
        plt.scatter(similar_coords[1:, 0], similar_coords[1:, 1], color='blue', alpha=0.5, label='Other Top-5 Similar')
    # Plot novel mesh in red
    plt.scatter(*novel_coord, color='red', alpha=0.5, label='Novel Mesh')
    # Annotate each of the top-5 similar meshes
    for idx, (x, y) in zip(similar_ids_cos, similar_coords):
        plt.text(x, y, all_vtk_files[idx].split('.')[0], fontsize=6, color='black')
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfpath + "/latent_space_tsne_pca_regularized_95pct_cos.png", dpi=300)
    plt.close()

    # --- Reconstruct optimized latent into mesh to confirm it looks normal ---
    
    # Reconstruction parameters
    recon_grid_origin = 1.0
    n_pts_per_axis = 256
    voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
    voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
    offset = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    icp_transform = NumpyTransform(np.eye(4))
    objects = 1

    # Reconstruct the novel mesh 
    mesh_out = create_mesh(
        decoder=model,
        latent_vector=latent_novel,
        n_pts_per_axis=n_pts_per_axis,
        voxel_origin=voxel_origin,
        voxel_size=voxel_size,
        path_original_mesh=None,
        offset=offset,
        scale=scale,
        icp_transform=icp_transform,
        objects=objects,
        verbose=True,
        device=device,
        )

    # Ensure it's PyVista PolyData
    if isinstance(mesh_out, list):
        mesh_out = mesh_out[0]
    if not isinstance(mesh_out, pv.PolyData):
        mesh_pv = mesh_out.extract_geometry()
    else:
        mesh_pv = mesh_out

    # Save mesh
    output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_decoded_novel_pca_regularized_95pct_cos.vtk"
    mesh_pv.save(output_path)
    print(f"Novel mesh saved to: {output_path}")

    # Save results to summary log
    # Get species prefix
    mesh_species = extract_species_prefix(os.path.basename(vert_fname))
    
    # --- Cosine Similarity Matches ---
    cos_top1_species = extract_species_prefix(all_vtk_files[similar_ids_cos[0]]) if similar_ids_cos else None
    cos_top1_match = "yes" if mesh_species and mesh_species == cos_top1_species else "no"
    cos_top5_match = "yes" if any(extract_species_prefix(all_vtk_files[i]) == mesh_species for i in similar_ids_cos) else "no"
    
    # --- Euclidean Distance Matches ---
    euc_top1_species = extract_species_prefix(all_vtk_files[similar_ids_euc[0]]) if similar_ids_euc else None
    euc_top1_match = "yes" if mesh_species and mesh_species == euc_top1_species else "no"
    euc_top5_match = "yes" if any(extract_species_prefix(all_vtk_files[i]) == mesh_species for i in similar_ids_euc) else "no"
    
    # Prepare summary log with all methods
    top_k_summary = {
        "mesh": os.path.basename(vert_fname),
        "ground_truth_species": mesh_species,
        "output_mesh": output_path,
    }
    
    # Add supervised classifier results
    for clf_name in trained_models.keys():
        pred = predictions[clf_name][0]
        match = "yes" if mesh_species and mesh_species == pred else "no"
        if probabilities[clf_name] is not None:
            conf = max(probabilities[clf_name][0])
            top_k_summary[f"{clf_name}_confidence"] = f"{conf:.2%}"
        else:
            top_k_summary[f"{clf_name}_confidence"] = "N/A"
            
        top_k_summary[f"{clf_name}_predicted_species"] = pred
        top_k_summary[f"{clf_name}_match"] = match
        top_k_summary[f"{clf_name}_inf_time"] = inference_times[clf_name]

    top_k_summary.update({
        # Cosine similarity results
        "cos_top1_species": cos_top1_species,
        "cos_top1_match": cos_top1_match,
        "cos_top5_match": cos_top5_match,
        # Euclidean distance results
        "euc_top1_species": euc_top1_species,
        "euc_top1_match": euc_top1_match,
        "euc_top5_match": euc_top5_match,
    })
    # Add cosine top-5 similar mesh names and distances
    for rank, (i, dist) in enumerate(zip(similar_ids_cos, distances_cos), 1):
        top_k_summary[f"cos_similar_{rank}_name"] = all_vtk_files[i]
        top_k_summary[f"cos_similar_{rank}_distance"] = dist
    # Add euclidean top-5 similar mesh names and distances
    for rank, (i, dist) in enumerate(zip(similar_ids_euc, distances_euc), 1):
        top_k_summary[f"euc_similar_{rank}_name"] = all_vtk_files[i]
        top_k_summary[f"euc_similar_{rank}_distance"] = dist
    summary_log.append(top_k_summary)

# Export results to summary log
df = pd.DataFrame(summary_log)
df.to_csv("summary_matches_95pct_cos.csv", index=False)
print("Summary saved to summary_matches_95pct_cos.csv")