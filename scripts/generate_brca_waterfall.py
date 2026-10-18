import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local src is imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))
from pysimlr.deep import lend_simr

sns.set_theme(style="white", font_scale=1.1)

palette = {
    'SiMLR': '#1f77b4',
    'LEND': '#2ca02c',
    'NED': '#ff7f0e',
    'NEDPP': '#9467bd'
}


def generate_brca_waterfall():
    print("Loading cached TCGA-BRCA multi-omics data (N=822, p=[200, 150, 150])...")
    cache_path = os.path.join(repo_root, "data", "tcga", "brca_er_cached.npz")
    d = np.load(cache_path, allow_pickle=True)
    y = d["y"]
    assays = ["rna", "cnv", "rppa"]
    Xs = [torch.tensor(StandardScaler().fit_transform(d[t])).float() for t in assays]
    feature_sets = [list(d[f"{t}_cols"]) for t in assays]

    print("Training LEND model with recommended benchmark defaults (pca init, consolidate=False)...")
    torch.manual_seed(42)
    np.random.seed(42)
    res = lend_simr(
        Xs,
        k=3,
        epochs=45,
        energy_type="acc",
        mixing_algorithm="svd",
        positivity="either",
        nsa_w=0.5,
        nsa_iterations=2,
        consolidate=False,
        verbose=False,
    )

    v_mats = [v.detach().numpy() for v in res["v"]]
    view_names = ["RNA-Seq (mRNA, p=200)", "Copy Number (CNV, p=150)", "Proteomics (RPPA, p=150)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 10.5))
    plt.suptitle(
        "Tri-Modal Clinical Discovery (TCGA-BRCA, N=822): Orthogonal ER-Status Drivers",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    for i in range(3):
        v = v_mats[i]
        z_i = Xs[i].numpy() @ v
        corrs = [abs(np.corrcoef(z_i[:, c], y)[0, 1]) for c in range(v.shape[1])]
        idx_dim = int(np.argmax(corrs))
        weights = np.abs(v[:, idx_dim])

        # Sort ascending so largest bar is at bottom or descending so largest is at top
        top_idx = np.argsort(weights)[-15:][::-1]
        ax = axes[i]
        labels = [str(feature_sets[i][j])[:20] for j in top_idx]

        sns.barplot(x=weights[top_idx], y=labels, ax=ax, color=palette["LEND"])
        ax.set_title(f"{view_names[i]}\n(ER-Aligned Component #{idx_dim+1}, |r|={corrs[idx_dim]:.2f})",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Linear Basis Weight (|V|)", fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        ax.text(
            0.96,
            0.04,
            f"Comp #{idx_dim+1} (|r|={corrs[idx_dim]:.2f})",
            transform=ax.transAxes,
            ha="right",
            fontsize=10,
            style="italic",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.figtext(
        0.5,
        0.01,
        "Stiefel-constrained OLEE basis V_m isolates hormone-receptor transcripts (PGR, PIP, GRPR), 8p11-p12/17q23 amplicons (ZNF703, ERLIN2, BRF2, RPS6KB1), and RPPA proteins (ER-alpha, GATA3, HER2, PR).",
        ha="center",
        fontsize=11.5,
        style="italic",
    )

    out_dir = os.path.join(repo_root, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "fig_brca_trimodal_waterfall.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved updated BRCA waterfall figure to {out_file}")


if __name__ == "__main__":
    generate_brca_waterfall()
