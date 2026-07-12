import torch
import pandas as pd
import numpy as np
from pysimlr import lend_simr, create_path_graph
from pysimlr.utils import adjusted_rvcoef
from scipy.stats import pearsonr

# Set seed for reproducibility to match previous runs
torch.manual_seed(42)
np.random.seed(42)

df_bl = pd.read_csv("data/expart_bl.csv")
df_mh = pd.read_csv("data/expart_mh.csv")
df_mr = pd.read_csv("data/expart_mr.csv")
df_pe = pd.read_csv("data/expart_pe.csv")

data = [df_bl.values, df_mh.values, df_mr.values, df_pe.values]

graph_chain = create_path_graph([(0, 2), (2, 3), (3, 1)], 4)
res_chain = lend_simr(data, k=2, topology="graph", path_graph=graph_chain, epochs=150, 
                      energy_type="acc", mixing_algorithm="svd", sparseness_quantile=0.5, 
                      sim_weight=2.0, nsa_iterations=5, verbose=False)

v_bl = res_chain["v"][0].detach().cpu().numpy()
z_bl = data[0] @ v_bl

print("--- Baseline (BL) Modality Component Correlation Check ---")
print(f"Orthogonality of V matrix (V^T @ V):")
print(np.round(v_bl.T @ v_bl, 4))

print("\nCorrelation between Latent Component 1 and Latent Component 2 (Z_bl):")
corr, p = pearsonr(z_bl[:, 0], z_bl[:, 1])
print(f"Pearson r = {corr:.4f}, p-value = {p:.4f}")

# Let's also check the raw features that were selected
def get_top_features_per_component(v_mat, feature_names, component=0, top_n=3):
    importance = np.abs(v_mat[:, component])
    top_indices = np.argsort(importance)[::-1][:top_n]
    return [feature_names[i] for i in top_indices]

top_c1 = get_top_features_per_component(v_bl, df_bl.columns.tolist(), 0)
top_c2 = get_top_features_per_component(v_bl, df_bl.columns.tolist(), 1)

print("\nTop 3 Features for BL Component 1:", top_c1)
print("Top 3 Features for BL Component 2:", top_c2)

print("\nCorrelation matrix of these raw features:")
combined_features = list(dict.fromkeys(top_c1 + top_c2))
print(df_bl[combined_features].corr().round(3))
