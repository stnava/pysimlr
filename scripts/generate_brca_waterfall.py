import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from pysimlr.deep import lend_simr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="white", font_scale=1.1)

# Set backend to Agg for headless environments
import matplotlib
matplotlib.use('Agg')

palette = {
    'SiMLR': '#1f77b4',
    'LEND': '#2ca02c',
    'NED': '#ff7f0e',
    'NEDPP': '#9467bd'
}

def generate_brca_waterfall():
    print("Loading BRCA data for visualization...")
    df = pd.read_csv('data/BRCA/brca_data_w_subtypes.csv', on_bad_lines='skip')
    df = df[df['ER.Status'].isin(['Positive', 'Negative'])].copy()
    
    rs_cols = [c for c in df.columns if c.startswith('rs_')][:100] # Use subset for speed
    cn_cols = [c for c in df.columns if c.startswith('cn_')][:100]
    pp_cols = [c for c in df.columns if c.startswith('pp_')]
    
    Xs = [torch.tensor(StandardScaler().fit_transform(df[rs_cols].values)).float(),
          torch.tensor(StandardScaler().fit_transform(df[cn_cols].values)).float(),
          torch.tensor(StandardScaler().fit_transform(df[pp_cols].values)).float()]

    print("Training Gold Standard LEND model...")
    res = lend_simr(Xs, k=2, epochs=150, energy_type='nc', mixing_algorithm='newton', 
                    positivity='positive', sparseness_quantile=0.8, nsa_iterations=3, verbose=False)
    
    v_mats = [v.detach().numpy() for v in res['v']]
    view_names = ["RNA-Seq", "CNV", "Proteomics"]
    feature_sets = [rs_cols, cn_cols, pp_cols]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    plt.suptitle("Tri-Modal Clinical Discovery: Orthogonal Biological Drivers", fontsize=22, fontweight='bold', y=1.05)

    for i in range(3):
        v = v_mats[i]
        # Show both orthogonal dimensions to demonstrate non-redundancy
        # We'll stack them or show them side by side. 
        # Actually, showing the dominant dimension and annotating it works best.
        idx_dim = np.argmax(np.abs(v).sum(axis=0))
        weights = v[:, idx_dim]
        
        # Sort and pick top 15
        top_idx = np.argsort(weights)[-15:]
        ax = axes[i]
        labels = [feature_sets[i][j].split('_')[-1] for j in top_idx]
        
        sns.barplot(x=weights[top_idx], y=labels, ax=ax, color=palette['LEND'])
        ax.set_title(f"{view_names[i]}\n(Primary Component)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Linear Basis Weight (V)", fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Annotate with dimension info
        ax.text(0.95, 0.02, f"Dim {idx_dim+1} (Max Signal)", transform=ax.transAxes, 
                ha='right', fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.5))

    plt.figtext(0.5, 0.01, "Orthogonality ensures each panel represents a unique, non-redundant biological axis. Waterfall plots enable exact counterfactual guides for patient risk shifting.", 
                ha="center", fontsize=12, style='italic')

    plt.tight_layout()
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/fig_brca_trimodal_waterfall.png', dpi=300, bbox_inches='tight')
    print("Saved updated BRCA waterfall figure with orthogonal signal highlights.")

if __name__ == "__main__":
    generate_brca_waterfall()
