import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from nsa_flow import nsa_flow
from pysimlr.deep import lend_simr

np.random.seed(42)
torch.manual_seed(42)


def gini(x):
    """Compute Gini coefficient of a 1D array."""
    x = np.abs(np.asarray(x, dtype=float))
    if np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))


def refine_varimax_sparse(v_mat, q=0.5, w=0.5):
    """Apply Varimax + NSA-Flow on the top (1-q) support per component."""
    res = nsa_flow(v_mat, w=w, nonneg=True, varimax=True, solver="lbfgsb", max_iter=200)
    Y = res["Y"].copy()
    for j in range(Y.shape[1]):
        col = Y[:, j]
        pos = col[col > 1e-6]
        if len(pos) > 1:
            thresh = np.quantile(col, q)
            mask = col >= thresh
            if not np.any(mask):
                mask[np.argmax(col)] = True
            col_s = np.where(mask, col, 0.0)
            nrm = np.linalg.norm(col_s)
            if nrm > 1e-12:
                col_s = col_s / nrm
            Y[:, j] = col_s
    return Y


def generate_v_matrices_plot():
    local_heart = "data/heart_cleveland.csv"
    url_heart = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols_heart = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    if os.path.exists(local_heart):
        df_heart = pd.read_csv(local_heart)
    else:
        df_heart = pd.read_csv(url_heart, names=cols_heart).replace("?", np.nan).dropna()

    heart_v1_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]
    heart_v2_cols = ["thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    X1_h = StandardScaler().fit_transform(df_heart[heart_v1_cols].values.astype(float))
    X2_h = StandardScaler().fit_transform(df_heart[heart_v2_cols].values.astype(float))

    res_heart = lend_simr(
        [torch.from_numpy(X1_h).float(), torch.from_numpy(X2_h).float()],
        k=2,
        epochs=150,
        energy_type="nc",
        mixing_algorithm="newton",
        positivity="positive",
        sparseness_quantile=0.5,
        verbose=False,
        nsa_iterations=3,
    )
    v1_h = refine_varimax_sparse(res_heart["v"][0].detach().numpy(), q=0.5, w=0.5)
    v2_h = refine_varimax_sparse(res_heart["v"][1].detach().numpy(), q=0.5, w=0.5)

    data_diab = load_diabetes()
    df_diab = pd.DataFrame(data_diab.data, columns=data_diab.feature_names)

    diab_v1_cols = ["age", "sex", "bmi", "bp"]
    diab_v2_cols = ["s1", "s2", "s3", "s4", "s5", "s6"]

    X1_d = StandardScaler().fit_transform(df_diab[diab_v1_cols].values.astype(float))
    X2_d = StandardScaler().fit_transform(df_diab[diab_v2_cols].values.astype(float))

    res_diab = lend_simr(
        [torch.from_numpy(X1_d).float(), torch.from_numpy(X2_d).float()],
        k=2,
        epochs=150,
        energy_type="logcosh",
        mixing_algorithm="newton",
        positivity="positive",
        sparseness_quantile=0.5,
        verbose=False,
        nsa_iterations=3,
    )
    v1_d = refine_varimax_sparse(res_diab["v"][0].detach().numpy(), q=0.5, w=0.5)
    v2_d = refine_varimax_sparse(res_diab["v"][1].detach().numpy(), q=0.5, w=0.5)

    cmap = sns.color_palette("viridis", as_cmap=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12.5))
    plt.suptitle(
        "Hierarchical Interpretation of Learned NSA Layer Weights (Varimax + NSA-Flow V Matrices)",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    def plot_heatmap(ax, v_matrix, row_labels, title):
        ginis = [gini(np.abs(v_matrix[:, i])) for i in range(v_matrix.shape[1])]
        df_v = pd.DataFrame(
            v_matrix,
            index=row_labels,
            columns=[f"L{i+1}\nG={ginis[i]:.2f}" for i in range(len(ginis))],
        )
        dom_col = np.argmax(v_matrix, axis=1)
        max_val = np.max(v_matrix, axis=1)
        is_zero = (max_val <= 1e-4).astype(int)
        order = np.lexsort((-max_val, dom_col, is_zero))
        df_v = df_v.iloc[order]

        sns.heatmap(
            df_v,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0,
            vmax=0.95,
            cbar=True,
            ax=ax,
            yticklabels=True,
            linewidths=1,
            linecolor="white",
        )
        ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
        ax.tick_params(axis="y", rotation=0)

    plot_heatmap(axes[0, 0], v1_h, heart_v1_cols, "Heart Disease: View 1 (Clinical Vitals)")
    plot_heatmap(axes[0, 1], v2_h, heart_v2_cols, "Heart Disease: View 2 (Cardiophysiological)")
    plot_heatmap(axes[1, 0], v1_d, diab_v1_cols, "Diabetes: View 1 (Systemic Vitals)")
    plot_heatmap(axes[1, 1], v2_d, diab_v2_cols, "Diabetes: View 2 (Serum Biomarkers)")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.figtext(
        0.5,
        0.008,
        "Varimax + NSA-Flow balances non-negative support across latent axes. Gini coefficient (G) measures weight sparsity/concentration.",
        ha="center",
        fontsize=12,
        style="italic",
    )

    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/nsa_v_interpretability.png", dpi=300, bbox_inches="tight")
    print("Saved updated clustered/annotated figure to paper/figures/nsa_v_interpretability.png")


if __name__ == "__main__":
    generate_v_matrices_plot()

