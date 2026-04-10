"""
Multi-View Visualization API for pysimlr.
Provides high-level tools for analyzing discovery and unmixing performance.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
from .utils import adjusted_rvcoef

def plot_view_correlations(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                           names: Optional[List[str]] = None,
                           title: str = "Cross-View Similarity (Adjusted RV)"):
    """Heatmap showing pairwise similarity between modalities."""
    n = len(data_matrices)
    if names is None:
        names = [f"View {i+1}" for i in range(n)]
        
    rv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            m1 = torch.as_tensor(data_matrices[i]).float()
            m2 = torch.as_tensor(data_matrices[j]).float()
            rv_matrix[i, j] = adjusted_rvcoef(m1, m2)
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(rv_matrix, annot=True, xticklabels=names, yticklabels=names, cmap='viridis', vmin=0, vmax=1)
    plt.title(title)
    return plt.gcf()

def plot_latent_consensus(u_shared: torch.Tensor, 
                          modality_latents: List[torch.Tensor],
                          modality_names: Optional[List[str]] = None,
                          dim_idx: int = 0):
    """Plot distribution of a specific latent dimension across all views vs. consensus."""
    n_views = len(modality_latents)
    if modality_names is None:
        modality_names = [f"View {i+1}" for i in range(n_views)]
        
    plt.figure(figsize=(10, 6))
    
    # Plot consensus as a baseline
    sns.kdeplot(u_shared[:, dim_idx].detach().cpu().numpy(), label="Consensus (U)", lw=3, color='black', ls='--')
    
    for i in range(n_views):
        sns.kdeplot(modality_latents[i][:, dim_idx].detach().cpu().numpy(), label=modality_names[i], alpha=0.6)
        
    plt.title(f"Latent Dimension {dim_idx} Distribution: View-wise vs. Consensus")
    plt.xlabel("Latent Value")
    plt.legend()
    return plt.gcf()

def plot_feature_signatures(v_mat: torch.Tensor, 
                            feature_names: Optional[List[str]] = None,
                            top_n: int = 20,
                            title: str = "Feature Signatures (V Matrix Weights)"):
    """Plot the highest-weight features for the first few latent components."""
    v_np = v_mat.detach().cpu().numpy()
    k = v_np.shape[1]
    n_plots = min(k, 3)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]
        
    for i in range(n_plots):
        weights = v_np[:, i]
        indices = np.argsort(np.abs(weights))[-top_n:]
        
        y_labels = [feature_names[idx] if feature_names else f"Feat {idx}" for idx in indices]
        
        axes[i].barh(range(top_n), weights[indices], color='tab:blue')
        axes[i].set_yticks(range(top_n))
        axes[i].set_yticklabels(y_labels)
        axes[i].set_title(f"Component {i}")
        axes[i].grid(True, alpha=0.3)
        
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_convergence_dynamics(history_dict: Dict[str, List[float]], 
                              title: str = "Training Dynamics"):
    """Multi-panel plot showing loss, reconstruction, and similarity over time."""
    keys = [k for k in history_dict.keys() if "history" in k]
    n = len(keys)
    if n == 0: return None
    
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]
    
    for i, key in enumerate(keys):
        axes[i].plot(history_dict[key])
        axes[i].set_title(key.replace('_', ' ').title())
        axes[i].set_xlabel("Epoch / Iteration")
        axes[i].grid(True, alpha=0.3)
        
    plt.suptitle(title)
    plt.tight_layout()
    return fig
