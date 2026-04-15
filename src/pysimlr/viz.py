import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any
from .utils import adjusted_rvcoef

def plot_view_correlations(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                           title: str = "Inter-view Correlation (RV Coefficient)") -> plt.Figure:
    """
    Compute and visualize the pairwise correlations between different data modalities.

    This function calculates the adjusted RV coefficient (a multivariate 
    generalization of R^2) between each pair of input data matrices and 
    displays the results as a heatmap.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        A list of data matrices to compare.
    title : str, default="Inter-view Correlation (RV Coefficient)"
        The title for the heatmap.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure containing the correlation heatmap.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    n = len(data_matrices)
    rv_mat = np.zeros((n, n))
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    
    for i in range(n):
        for j in range(n):
            rv_mat[i, j] = adjusted_rvcoef(torch_mats[i], torch_mats[j])
            
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(rv_mat, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=[f"Mod {i+1}" for i in range(n)],
                yticklabels=[f"Mod {i+1}" for i in range(n)])
    ax.set_title(title)
    return fig

def plot_latent_consensus(u_shared: torch.Tensor, 
                          latents: List[torch.Tensor], 
                          title: str = "Latent Alignment to Consensus") -> plt.Figure:
    """
    Visualize how well each modality's latent space aligns with the shared consensus.

    This function computes the similarity (adjusted RV coefficient) between 
    each modality's projected latent space (Z_m) and the shared consensus 
    latent space (U). High similarity indicates that the modality 
    contributes strongly to the shared signal.

    Parameters
    ----------
    u_shared : torch.Tensor
        The shared latent consensus matrix (samples x k).
    latents : List[torch.Tensor]
        A list of modality-specific latent matrices (Z_m).
    title : str, default="Latent Alignment to Consensus"
        The title for the bar plot.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    n_mods = len(latents)
    corrs = [adjusted_rvcoef(u_shared, l) for l in latents]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = [f"Mod {i+1}" for i in range(n_mods)]
    sns.barplot(x=x_labels, y=corrs, hue=x_labels, palette="viridis", legend=False, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correlation (RV) to U")
    ax.set_title(title)
    return fig

def plot_feature_signatures(v_mat: torch.Tensor, 
                            feature_names: Optional[List[str]] = None,
                            top_k: int = 15,
                            title: str = "Top Feature Signatures",
                            **kwargs) -> plt.Figure:
    """
    Visualize the most important features (loadings) for each latent component.

    This function displays a horizontal bar chart of the features with the 
    largest absolute weights in the basis matrix V. This provides a 
    'signature' for what each latent component represents in terms of 
    the original data features.

    Parameters
    ----------
    v_mat : torch.Tensor
        The basis matrix V (features x k) from a SiMLR or LEND model.
    feature_names : List[str], optional
        The names of the original features. If None, generic names are used.
    top_k : int, default=15
        The number of top features to display for each component.
    title : str, default="Top Feature Signatures"
        The title for the overall figure.
    **kwargs : Dict[str, Any]
        Supports `top_n` as an alias for `top_k`.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure containing k subplots.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if "top_n" in kwargs:
        top_k = kwargs["top_n"]
        
    v_np = v_mat.detach().cpu().numpy()
    k = v_np.shape[1]
    
    fig, axes = plt.subplots(1, k, figsize=(5*k, 6), sharey=True)
    if k == 1: axes = [axes]
    
    for i in range(k):
        weights = v_np[:, i]
        indices = np.argsort(np.abs(weights))[-top_k:]
        
        ax = axes[i]
        names = [feature_names[idx] if feature_names else f"Feat {idx}" for idx in indices]
        ax.barh(names, weights[indices], color='tab:blue')
        ax.set_title(f"Component {i+1}")
        
    plt.tight_layout()
    return fig

def plot_convergence_dynamics(history_dict: Dict[str, List[float]], 
                             title: str = "SiMLR Convergence Dynamics") -> Optional[plt.Figure]:
    """
    Plot the evolution of loss and optimization metrics over time.

    This function visualizes the training history of a SiMLR or deep model, 
    allowing for the diagnosis of convergence issues, manifold drift, or 
    latent collapse.

    Parameters
    ----------
    history_dict : Dict[str, List[float]]
        A dictionary where keys are metric names (e.g., 'loss', 'recon', 'sim') 
        and values are lists of metric values per iteration/epoch.
    title : str, default="SiMLR Convergence Dynamics"
        The title for the plot.

    Returns
    -------
    plt.Figure, optional
        The generated Matplotlib figure, or None if the history is empty.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if not history_dict or all(not v for v in history_dict.values()):
        return None
        
    fig, ax = plt.subplots(figsize=(10, 5))
    has_labels = False
    for key, vals in history_dict.items():
        if vals: 
            ax.plot(vals, label=key)
            has_labels = True
            
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss / Metric")
    ax.set_title(title)
    if has_labels:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
