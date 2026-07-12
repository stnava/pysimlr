import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pysimlr import lend_simr, create_path_graph

def get_top_features_per_component(v_mat, feature_names, component=0, top_n=10):
    importance = np.abs(v_mat[:, component])
    top_indices = np.argsort(importance)[::-1][:top_n]
    return [feature_names[i] for i in top_indices], importance[top_indices]

def generate_interpretability_assets():
    out_dir = "expart_report_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Data and Feature Names
    df_bl = pd.read_csv("data/expart_bl.csv")
    df_mh = pd.read_csv("data/expart_mh.csv")
    df_mr = pd.read_csv("data/expart_mr.csv")
    df_pe = pd.read_csv("data/expart_pe.csv")
    
    data = [df_bl.values, df_mh.values, df_mr.values, df_pe.values]
    feature_names = [df_bl.columns.tolist(), df_mh.columns.tolist(), df_mr.columns.tolist(), df_pe.columns.tolist()]
    modality_names = ["BL (Biological)", "MH (Mental Health)", "MR (Imaging)", "PE (Performance)"]
    
    # Fit the Sequential Chain Model with Final Winner Parameters
    print("Fitting Sequential Chain model (Final Winner Parameters) for interpretability...")
    graph_chain = create_path_graph([(0, 2), (2, 3), (3, 1)], 4)
    res_chain = lend_simr(data, k=2, topology="graph", path_graph=graph_chain, epochs=150, 
                          energy_type="acc", mixing_algorithm="newton", sparseness_quantile=0.5, 
                          sim_weight=2.0, verbose=False, optimizer_type="lars", dynamic_weights=False)
    
    # Extract V matrices
    v_mats = [v.detach().cpu().numpy() for v in res_chain["v"]]
    
    # 1. Feature Importance Bar Plots
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    for i in range(4):
        for comp in range(2):
            top_feats, top_imps = get_top_features_per_component(v_mats[i], feature_names[i], component=comp, top_n=10)
            display_names = [name.replace("npsy_", "").replace("_Total", "") for name in top_feats]
            sns.barplot(x=top_imps, y=display_names, hue=display_names, ax=axes[i, comp], palette="viridis", legend=False)
            axes[i, comp].set_title(f"{modality_names[i]} - Latent Feature {comp+1}")
            axes[i, comp].set_xlabel("Absolute Weight")
            axes[i, comp].set_ylabel("")
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance_all.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Cross-Modality Correlation Matrix
    selected_dfs = []
    for i, df in enumerate([df_bl, df_mh, df_mr, df_pe]):
        top_c1, _ = get_top_features_per_component(v_mats[i], feature_names[i], component=0, top_n=3)
        top_c2, _ = get_top_features_per_component(v_mats[i], feature_names[i], component=1, top_n=3)
        top_all = list(dict.fromkeys(top_c1 + top_c2))
        selected_dfs.append(df[top_all])
        
    df_selected = pd.concat(selected_dfs, axis=1)
    clean_cols = [c.replace("npsy_", "").replace("_Total", "").replace("_Raw", "") for c in df_selected.columns]
    df_selected.columns = clean_cols
    corr_matrix = df_selected.corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax_corr, annot_kws={"size": 8})
    ax_corr.set_title("Cross-Modality Correlation (Final Winner Parameters)", fontsize=16)
    plt.savefig(os.path.join(out_dir, "top_features_correlation.png"), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Updated interpretability assets generated in {out_dir}/")

if __name__ == "__main__":
    generate_interpretability_assets()
