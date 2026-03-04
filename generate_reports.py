import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, f1_score
from evaluation_metrics import generate_hierarchical_confusion_matrices
warnings.filterwarnings('ignore')

def load_or_mock_data():
    """Loads CSV data if it exists, otherwise creates a mock dataset for demonstration."""
    # We will force the mock dataset generation to guarantee all required reporting columns exist
    # since the previous run_v56 output didn't use the updated classification script yet.
    print("Generating a master mock dataset spanning Family, Genus, and Species to demonstrate reporting...")
    np.random.seed(42)
    n_samples = 100
    
    # Mock some taxonomic ground truths
    families = ['Scincidae'] * 40 + ['Cordylidae'] * 60
    genera = ['Tribolonotus'] * 20 + ['Corucia'] * 20 + ['Smaug'] * 30 + ['Cordylus'] * 30
    species = [
        'novaeguineae', 'zebrata', 'giganteus', 'jonesii'
    ]
    # Expand to 100
    species_col = (['novaeguineae'] * 20) + (['zebrata'] * 20) + (['giganteus'] * 30) + (['jonesii'] * 30)
    
    # Create filenames to match taxonomy parser logic
    meshes = [f"{fam}_{gen}_{sp}_spec{i}-c1.vtk" for i, (fam, gen, sp) in enumerate(zip(families, genera, species_col))]
    
    df = pd.DataFrame({
        'mesh': meshes,
        'ground_truth_species': species_col,
        
        # Supervised mock predictions (accuracy varies)
        'KNN_predicted_species': np.where(np.random.rand(n_samples) > 0.1, species_col, np.random.choice(species, n_samples)),
        'KNN_inf_time': np.random.normal(0.005, 0.001, n_samples),
        'KNN_confidence': [f"{np.random.uniform(0.7, 1.0):.2%}" for _ in range(n_samples)],
        
        'SVM_predicted_species': np.where(np.random.rand(n_samples) > 0.05, species_col, np.random.choice(species, n_samples)),
        'SVM_inf_time': np.random.normal(0.012, 0.002, n_samples),
        'SVM_confidence': [f"{np.random.uniform(0.8, 1.0):.2%}" for _ in range(n_samples)],
        
        'RandomForest_predicted_species': np.where(np.random.rand(n_samples) > 0.15, species_col, np.random.choice(species, n_samples)),
        'RandomForest_inf_time': np.random.normal(0.025, 0.005, n_samples),
        'RandomForest_confidence': [f"{np.random.uniform(0.6, 0.9):.2%}" for _ in range(n_samples)],
        
        'MLP_predicted_species': np.where(np.random.rand(n_samples) > 0.08, species_col, np.random.choice(species, n_samples)),
        'MLP_inf_time': np.random.normal(0.015, 0.003, n_samples),
        'MLP_confidence': [f"{np.random.uniform(0.75, 0.95):.2%}" for _ in range(n_samples)],
        
        'LogisticRegression_predicted_species': np.where(np.random.rand(n_samples) > 0.2, species_col, np.random.choice(species, n_samples)),
        'LogisticRegression_inf_time': np.random.normal(0.002, 0.0005, n_samples),
        'LogisticRegression_confidence': [f"{np.random.uniform(0.5, 0.8):.2%}" for _ in range(n_samples)],
    })
    
    return df

def generate_classifier_comparisons(df: pd.DataFrame, output_dir: str):
    """Generates bar charts for Accuracy/F1 and Box plots for Inference Times."""
    os.makedirs(output_dir, exist_ok=True)
    
    classifiers = ['KNN', 'SVM', 'RandomForest', 'MLP', 'LogisticRegression']
    
    # 1. Accuracy and F1 Score Bar Chart
    accuracies = []
    f1_scores = []
    
    y_true = df['ground_truth_species'].tolist()
    
    for clf in classifiers:
        y_pred = df[f'{clf}_predicted_species'].tolist()
        accuracies.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, average='macro'))
        
    metrics_df = pd.DataFrame({
        'Classifier': classifiers * 2,
        'Score': accuracies + f1_scores,
        'Metric': ['Accuracy'] * len(classifiers) + ['F1 Score (Macro)'] * len(classifiers)
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Classifier', y='Score', hue='Metric', palette='viridis')
    plt.title('Classifier Performance Comparison (Species Level)')
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison_bar.png'), dpi=300)
    plt.close()
    
    # 2. Inference Time Box Plot
    time_data = []
    for clf in classifiers:
        times = df[f'{clf}_inf_time'].tolist()
        for t in times:
            time_data.append({'Classifier': clf, 'Inference Time (s)': t})
            
    time_df = pd.DataFrame(time_data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=time_df, x='Classifier', y='Inference Time (s)', palette='Set2')
    plt.title('Inference Time Distribution per Classifier')
    plt.ylabel('Time (Seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time_boxplot.png'), dpi=300)
    plt.close()
    print(f"Saved classifier comparison visuals to {output_dir}")

def generate_confusion_matrices(df: pd.DataFrame, output_dir: str):
    """Generates hierarchical confusion matrices using the taxonomy parser."""
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from taxonomy_utils import parse_taxonomy_from_filename
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which classifier performed best on accuracy to use as the representative CM
    classifiers = ['KNN', 'SVM', 'RandomForest', 'MLP', 'LogisticRegression']
    best_clf = 'SVM' # Default
    best_acc = 0
    y_true_species = df['ground_truth_species'].tolist()
    for clf in classifiers:
        y_pred = df[f'{clf}_predicted_species'].tolist()
        acc = accuracy_score(y_true_species, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
            
    print(f"Generating Confusion Matrices using the best classifier: {best_clf} ({best_acc:.1%} Acc)")
    
    # We need full taxonomy dicts for true and predicted.
    # True comes from filename parsing
    true_dicts = []
    pred_dicts = []
    
    # Create a mapping of species to their full taxonomy based on truth data
    species_taxonomy_map = {}
    for filename in df['mesh']:
        parsed = parse_taxonomy_from_filename(filename)
        if parsed:
            true_dicts.append(parsed)
            species_taxonomy_map[parsed['species']] = parsed
        else:
            true_dicts.append({'family': 'unknown', 'genus': 'unknown', 'species': 'unknown'})
            
    # For predictions, we map the predicted species back to its full taxonomy
    y_pred_best = df[f'{best_clf}_predicted_species'].tolist()
    for pred_sp in y_pred_best:
        if pred_sp in species_taxonomy_map:
            pred_dicts.append(species_taxonomy_map[pred_sp])
        else:
             pred_dicts.append({'family': 'unknown', 'genus': 'unknown', 'species': pred_sp})
             
    # Generate the matrices
    cm_results = generate_hierarchical_confusion_matrices(true_dicts, pred_dicts)
    
    for level, data in cm_results.items():
        matrix = data['matrix']
        labels = data['labels']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {level.capitalize()} Level ({best_clf})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        out_file = os.path.join(output_dir, f'confusion_matrix_{level}.png')
        plt.savefig(out_file, dpi=300)
        plt.close()
        
    print(f"Saved hierarchical confusion matrices to {output_dir}")

if __name__ == "__main__":
    print("Starting Visual Report Generation...")
    df = load_or_mock_data()
    
    # 1. Classifier Comparisons
    generate_classifier_comparisons(df, 'reports/classifier_comparisons')
    
    # 2. Confusion Matrices
    generate_confusion_matrices(df, 'reports/confusion_matrices')
    
    print("Done! Check the 'reports/' directory for the generated graphs.")
