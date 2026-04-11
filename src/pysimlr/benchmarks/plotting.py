
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any


def plot_pareto_recovery_vs_r2(df: pd.DataFrame, title: str = "Performance Tradeoff: Recovery vs. Predictive Power"):
    """Scatter plot of Recovery (Adjusted RV) vs Test R2."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="recovery",
        y="test_r2",
        hue="model",
        size="sparsity",
        sizes=(50, 200),
        alpha=0.7,
    )
    plt.title(title)
    plt.xlabel("Latent Recovery (Adjusted RV)")
    plt.ylabel("Outcome Prediction (Test R2)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return plt.gcf()


def plot_sparsity_sensitivity(df: pd.DataFrame):
    """Line plots showing sensitivity of key metrics to sparsity."""
    metrics = ["recovery", "test_r2", "recon_error"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        sns.lineplot(data=df, x="sparsity", y=metric, hue="model", marker="o", ax=axes[i])
        axes[i].set_title(f"{metric.replace('_', ' ').title()} vs Sparsity")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_stability_diagnostics(df: pd.DataFrame):
    """Panel of plots showing stability and collapse metrics."""
    metrics = ["u_std_mean", "u_norm_sd", "collapsed_dims"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="sparsity", y=metric, hue="model", ax=axes[i])
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")

    plt.tight_layout()
    return fig


def plot_sparsity_vs_orthogonality(df: pd.DataFrame, title: str = "Orthogonality Defect vs. Sparsity"):
    """Line plot showing how orthogonality degrades as sparsity increases."""
    if "orthogonality_defect" not in df.columns:
        return None
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="sparsity", y="orthogonality_defect", hue="model", marker="o")
    plt.title(title)
    plt.ylabel("Invariant Orthogonality Defect (lower is better)")
    plt.xlabel("Sparsity Quantile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_reconstruction_tradeoff(df: pd.DataFrame):
    """Plot latent recovery vs reconstruction error."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="recon_error",
        y="recovery",
        hue="model",
        style="sparsity",
        s=100,
    )
    plt.title("Reconstruction-Recovery Tradeoff")
    plt.xlabel("Normalized Reconstruction Error (lower is better)")
    plt.ylabel("Latent Recovery (higher is better)")
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def _resolve_first_layer_v(result: Dict[str, Any], modality_idx: int):
    first_layer = result.get("first_layer")
    if isinstance(first_layer, dict):
        v_list = first_layer.get("v")
        if isinstance(v_list, list) and modality_idx < len(v_list):
            return v_list[modality_idx]
    v_list = result.get("v")
    if isinstance(v_list, list) and modality_idx < len(v_list):
        return v_list[modality_idx]
    return None


def plot_v_heatmaps(results_dict: Dict[str, Any], modality_idx: int = 0):
    """Plot first-layer basis heatmaps for multiple models side-by-side."""
    models = list(results_dict.keys())
    if not models:
        return None

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for i, m_name in enumerate(models):
        res = results_dict[m_name]
        v = _resolve_first_layer_v(res, modality_idx)
        if v is None:
            axes[i].set_visible(False)
            continue
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()

        sns.heatmap(v, cmap="coolwarm", center=0, ax=axes[i], cbar=False)
        axes[i].set_title(f"{m_name}\n(First-Layer Basis)")

    plt.tight_layout()
    return fig


def plot_latent_correlation(u_pred: torch.Tensor, u_true: torch.Tensor, title: str = "Learned vs True Latent Correlation"):
    """Plot correlation matrix between learned and true latent dimensions."""
    u_pred_np = u_pred.detach().cpu().numpy()
    u_true_np = u_true.detach().cpu().numpy()

    corr = np.zeros((u_pred_np.shape[1], u_true_np.shape[1]))
    for i in range(u_pred_np.shape[1]):
        for j in range(u_true_np.shape[1]):
            corr[i, j] = np.abs(np.corrcoef(u_pred_np[:, i], u_true_np[:, j])[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="viridis", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("True Latent Dimension")
    plt.ylabel("Learned Dimension")
    return plt.gcf()


def plot_first_layer_alignment_heatmap(result: Dict[str, Any], modality_idx: int = 0):
    """Visualize deep-layer versus first-layer component alignment for one modality."""
    interpretability = result.get("interpretability", {})
    alignment = interpretability.get("deep_layer_alignment", {})
    modalities = alignment.get("modalities", [])
    if modality_idx >= len(modalities):
        return None
    corr = modalities[modality_idx].get("component_correlation")
    if not isinstance(corr, torch.Tensor):
        return None
    corr_np = corr.detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_np, annot=False, cmap="vlag", center=0)
    plt.title(f"Modality {modality_idx}: Deep vs First-Layer Alignment")
    plt.xlabel("Deep Component")
    plt.ylabel("First-Layer Component")
    plt.tight_layout()
    return plt.gcf()


def plot_first_layer_feature_importance(result: Dict[str, Any], modality_idx: int = 0, top_k: int = 10):
    """Bar plot of the strongest first-layer feature importances for one modality."""
    interpretability = result.get("interpretability", {})
    shared_report = interpretability.get("shared_to_first_layer", {})
    combined = shared_report.get("combined", {})
    feature_importance = combined.get("feature_importance", [])
    if modality_idx >= len(feature_importance):
        return None
    feat = feature_importance[modality_idx]
    if not isinstance(feat, torch.Tensor) or feat.numel() == 0:
        return None
    feat = torch.abs(feat).detach().cpu().flatten()
    top_k = max(1, min(int(top_k), feat.numel()))
    values, indices = torch.topk(feat, k=top_k)
    labels = [f"f{idx}" for idx in indices.tolist()]
    plt.figure(figsize=(8, 5))
    positions = np.arange(top_k)
    plt.barh(positions, values.numpy())
    plt.yticks(positions, labels)
    plt.title(f"Modality {modality_idx}: Top First-Layer Features")
    plt.xlabel("Absolute Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    return plt.gcf()


def plot_interpretability_tradeoff(df: pd.DataFrame):
    """Scatter plot of predictive power versus first-layer preservation."""
    required = {"test_r2", "first_layer_prediction_preservation"}
    if not required.issubset(df.columns):
        return None
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="first_layer_prediction_preservation",
        y="test_r2",
        hue="model",
        size="sparsity" if "sparsity" in df.columns else None,
        sizes=(50, 200),
        alpha=0.8,
    )
    plt.title("Interpretability-Preservation vs Predictive Power")
    plt.xlabel("First-Layer Prediction Preservation")
    plt.ylabel("Outcome Prediction (Test R2)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()
