import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
from pysimlr.deep import lend_simr
import os

# Set backend to Agg for headless environments
import matplotlib
matplotlib.use('Agg')

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

def gini(x):
    """Compute Gini coefficient of a 1D array."""
    if np.sum(x) == 0: return 0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def generate_v_matrices_plot():
    # ---------------------------------------------------------
    # 1. Load Heart Disease Dataset
    # ---------------------------------------------------------
    url_heart = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols_heart = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df_heart = pd.read_csv(url_heart, names=cols_heart).replace('?', np.nan).dropna()
    
    # Split features into two views
    heart_v1_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg']
    heart_v2_cols = ['thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    X1_h = StandardScaler().fit_transform(df_heart[heart_v1_cols].values.astype(float))
    X2_h = StandardScaler().fit_transform(df_heart[heart_v2_cols].values.astype(float))

    # Train LEND with SPARSITY (matches v21 gold standard)
    res_heart = lend_simr(
        [torch.from_numpy(X1_h).float(), torch.from_numpy(X2_h).float()], 
        k=2, epochs=150, 
        energy_type='nc', mixing_algorithm='newton', 
        positivity='positive', sparseness_quantile=0.5, verbose=False, nsa_iterations=3
    )
    v1_h = res_heart['v'][0].detach().numpy()
    v2_h = res_heart['v'][1].detach().numpy()

    # ---------------------------------------------------------
    # 2. Load Diabetes Dataset
    # ---------------------------------------------------------
    data_diab = load_diabetes()
    diab_cols = data_diab.feature_names
    df_diab = pd.DataFrame(data_diab.data, columns=diab_cols)
    
    # Split features into two views
    diab_v1_cols = ['age', 'sex', 'bmi', 'bp']
    diab_v2_cols = ['s1', 's2', 's3', 's4', 's5', 's6']
    
    X1_d = StandardScaler().fit_transform(df_diab[diab_v1_cols].values.astype(float))
    X2_d = StandardScaler().fit_transform(df_diab[diab_v2_cols].values.astype(float))

    # Train LEND with SPARSITY
    res_diab = lend_simr(
        [torch.from_numpy(X1_d).float(), torch.from_numpy(X2_d).float()], 
        k=2, epochs=150, 
        energy_type='logcosh', mixing_algorithm='newton', 
        positivity='positive', sparseness_quantile=0.5, verbose=False, nsa_iterations=3
    )
    v1_d = res_diab['v'][0].detach().numpy()
    v2_d = res_diab['v'][1].detach().numpy()

    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------
    cmap = sns.color_palette("viridis", as_cmap=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.suptitle("Hierarchical Interpretation of Learned NSA Layer Weights (V matrices)", fontsize=18, fontweight='bold', y=1.02)

    def plot_heatmap(ax, v_matrix, row_labels, title):
        # Calculate Gini for each latent dimension
        ginis = [gini(np.abs(v_matrix[:, i])) for i in range(v_matrix.shape[1])]
        
        # Dataframe for clustering
        df_v = pd.DataFrame(v_matrix, index=row_labels, columns=[f'L{i+1}\nG={ginis[i]:.2f}' for i in range(len(ginis))])
        
        # Simple clustering by row sum for visual grouping without full clustermap complexity in subplot
        row_order = df_v.abs().sum(axis=1).sort_values(ascending=False).index
        df_v = df_v.loc[row_order]

        sns.heatmap(df_v, annot=True, fmt=".2f", cmap=cmap, vmin=0, 
                    cbar=True, ax=ax, yticklabels=True, 
                    linewidths=1, linecolor='white')
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.tick_params(axis='y', rotation=0)

    # Row 1: Heart
    plot_heatmap(axes[0, 0], v1_h, heart_v1_cols, "Heart Disease: View 1 (Clinical)")
    plot_heatmap(axes[0, 1], v2_h, heart_v2_cols, "Heart Disease: View 2 (Physiological)")

    # Row 2: Diabetes
    plot_heatmap(axes[1, 0], v1_d, diab_v1_cols, "Diabetes: View 1 (Vitals)")
    plot_heatmap(axes[1, 1], v2_d, diab_v2_cols, "Diabetes: View 2 (Biomarkers)")

    plt.figtext(0.5, 0.01, "NSA Layer acts as a 'hard' feature selector. Gini coefficient (G) measures weight sparsity/concentration.", 
                ha="center", fontsize=12, style='italic')

    plt.tight_layout()
    
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/nsa_v_interpretability.png", dpi=300, bbox_inches='tight')
    print("Saved updated clustered/annotated figure to paper/figures/nsa_v_interpretability.png")

if __name__ == "__main__":
    generate_v_matrices_plot()
