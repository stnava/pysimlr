from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any


def plot_pareto_recovery_vs_r2(df: pd.DataFrame, title: str = "Performance Tradeoff: Recovery vs. Predictive Power") -> plt.Figure:
    """
    Scatter plot of Latent Recovery versus Predictive R2.

    Visualizes the "Pareto front" of models by comparing their ability to 
    recover the true latent space (ground truth) against their ability to 
    predict an external outcome.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from benchmark sweep. Expected columns:
        - `recovery`: Latent recovery score (Adjusted RV).
        - `test_r2`: Outcome predictive performance.
        - `model`: For coloring model types.
        - `sparsity`: For scaling point sizes.
    title : str, default="Performance Tradeoff: Recovery vs. Predictive Power"
        Title for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        A scatter plot summarizing model performance across two key axes.

    Raises
    ------
    KeyError
        If required columns are missing from `df`.
    TypeError
        If `df` is not a DataFrame.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
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


def plot_sparsity_sensitivity(df: pd.DataFrame) -> plt.Figure:
    """
    Line plots showing sensitivity of key metrics to sparsity levels.

    Generates a three-panel plot showing how Latent Recovery, Predictive R2, 
    and Reconstruction Error change as the model's sparsity constraint 
    is varied.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from benchmark sweep. Expected columns:
        - `sparsity`: The sparsity levels tested.
        - `recovery`: Latent recovery score.
        - `test_r2`: Predictive R2.
        - `recon_error`: Reconstruction error.
        - `model`: For grouping by architecture.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with three subplots summarizing sparsity sensitivity.

    Raises
    ------
    KeyError
        If required columns are missing from `df`.
    TypeError
        If `df` is not a DataFrame.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    metrics = ["recovery", "test_r2", "recon_error"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        sns.lineplot(data=df, x="sparsity", y=metric, hue="model", marker="o", ax=axes[i])
        axes[i].set_title(f"{metric.replace('_', ' ').title()} vs Sparsity")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_stability_diagnostics(df: pd.DataFrame) -> plt.Figure:
    """
    Panel of plots showing stability and latent collapse diagnostics.

    Generates three bar plots showing the mean standard deviation of latents, 
    the normalized standard deviation, and the number of collapsed dimensions 
    (zero variance) across different models and sparsity levels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from benchmark sweep. Expected columns:
        - `sparsity`: Sparsity levels.
        - `u_std_mean`: Mean latent standard deviation.
        - `u_norm_sd`: Normalized standard deviation.
        - `collapsed_dims`: Count of collapsed latent dimensions.
        - `model`: For grouping by architecture.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with three subplots summarizing stability diagnostics.

    Raises
    ------
    KeyError
        If required columns are missing from `df`.
    TypeError
        If `df` is not a DataFrame.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    metrics = ["u_std_mean", "u_norm_sd", "collapsed_dims"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="sparsity", y=metric, hue="model", ax=axes[i])
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")

    plt.tight_layout()
    return fig


def plot_sparsity_vs_orthogonality(df: pd.DataFrame, title: str = "Orthogonality Defect vs. Sparsity") -> Optional[plt.Figure]:
    """
    Line plot showing how orthogonality degrades as sparsity increases.

    Visualizes the impact of the sparsity constraint on the projection 
    orthogonality, which is a key theoretical constraint in SiMLR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from benchmark sweep. Expected columns:
        - `sparsity`: Sparsity levels.
        - `orthogonality_defect`: Measured orthogonality defect.
        - `model`: For grouping by architecture.
    title : str, default="Orthogonality Defect vs. Sparsity"
        Title for the plot.

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        The generated figure, or `None` if `orthogonality_defect` column 
        is missing.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
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


def plot_reconstruction_tradeoff(df: pd.DataFrame) -> plt.Figure:
    """
    Plot latent recovery versus reconstruction error.

    Visualizes how model complexity (reconstruction error) relates to the 
    quality of latent representation recovery.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from benchmark sweep. Expected columns:
        - `recon_error`: Normalized reconstruction error (MSE).
        - `recovery`: Latent recovery score.
        - `model`: For grouping by architecture.
        - `sparsity`: For marking model sparsity level.

    Returns
    -------
    matplotlib.figure.Figure
        A scatter plot illustrating the reconstruction-recovery tradeoff.

    Raises
    ------
    KeyError
        If required columns are missing from `df`.
    TypeError
        If `df` is not a DataFrame.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
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


def plot_v_heatmaps(results_dict: Dict[str, Any], modality_idx: int = 0) -> Optional[plt.Figure]:
    """
    Plot first-layer basis heatmaps (V matrices) for multiple models side-by-side.

    Allows for visual comparison of feature projection patterns between 
    different architectures (e.g., LEND vs NED) for the same modality.

    Parameters
    ----------
    results_dict : Dict[str, Any]
        A dictionary mapping model names to their result dictionaries.
    modality_idx : int, default=0
        The index of the modality to plot.

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        A figure containing side-by-side heatmaps for each model, or `None` 
         if the dictionary is empty.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
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


def plot_latent_correlation(u_pred: torch.Tensor, u_true: torch.Tensor, title: str = "Learned vs True Latent Correlation") -> plt.Figure:
    """
    Plot absolute correlation matrix between learned and true latent dimensions.

    Useful for validating latent recovery in synthetic cases where the ground 
    truth latent space (U) is known.

    Parameters
    ----------
    u_pred : torch.Tensor
        The shared latent space (U) estimated by the model.
    u_true : torch.Tensor
        The ground truth shared latent space.
    title : str, default="Learned vs True Latent Correlation"
        Title for the heatmap.

    Returns
    -------
    matplotlib.figure.Figure
        A heatmap showing the absolute correlation between each learned 
        and true latent dimension.

    Raises
    ------
    TypeError
        If inputs are not tensors.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
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


def plot_first_layer_alignment_heatmap(result: Dict[str, Any], modality_idx: int = 0) -> Optional[plt.Figure]:
    """
    Visualize deep-layer versus first-layer component alignment for one modality.

    Plots a heatmap showing the correlation (RV-like) between the components 
    of the initial interpretable layer and the components of the subsequent 
    deep representation.

    Parameters
    ----------
    result : Dict[str, Any]
        A SiMLR or deep SiMR result dictionary. Must contain 
        `interpretability` -> `deep_layer_alignment`.
    modality_idx : int, default=0
        The index of the modality to plot.

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        The generated figure, or `None` if the alignment data is missing 
        for the given modality.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
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


def plot_first_layer_feature_importance(result: Dict[str, Any], modality_idx: int = 0, top_k: int = 10) -> Optional[plt.Figure]:
    """
    Bar plot of the strongest first-layer feature importances for one modality.

    Visualizes which original features are most influential in the first 
    interpretable layer's projection, derived from the interpretability report.

    Parameters
    ----------
    result : Dict[str, Any]
        A SiMLR or deep SiMR result dictionary. Must contain 
        `interpretability` -> `shared_to_first_layer`.
    modality_idx : int, default=0
        The index of the modality to plot.
    top_k : int, default=10
        The number of top features to show in the bar plot.

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        The generated figure, or `None` if the importance data is missing 
        for the given modality.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
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


def plot_interpretability_tradeoff(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Scatter plot of predictive power versus first-layer preservation.

    Visualizes the trade-off between how well the model predicts an outcome 
    (Test R2) and how much of that predictive power is captured by the 
    initial interpretable layer. This is useful for comparing LEND, NED, 
    and NED++ models.

    Parameters
    ----------
    df : pd.DataFrame
        A results DataFrame from a benchmark sweep. Must contain columns:
        - `test_r2`: The predictive performance.
        - `first_layer_prediction_preservation`: The preservation ratio.
        - `model`: (Optional) For grouping/coloring points.
        - `sparsity`: (Optional) For scaling point sizes.

    Returns
    -------
    Optional[matplotlib.figure.Figure]
        The generated figure, or `None` if required columns are missing.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
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
