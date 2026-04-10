import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from typing import Optional, List, Union, Dict, Any

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
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel("Latent Recovery (Adjusted RV)")
    plt.ylabel("Outcome Prediction (Test R2)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()

def plot_sparsity_sensitivity(df: pd.DataFrame):
    """Line plots showing sensitivity of key metrics to sparsity."""
    metrics = ["recovery", "test_r2", "recon_error"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        sns.lineplot(
            data=df, 
            x="sparsity", 
            y=metric, 
            hue="model", 
            marker='o', 
            ax=axes[i]
        )
        axes[i].set_title(f"{metric.replace('_', ' ').title()} vs Sparsity")
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    return fig

def plot_stability_diagnostics(df: pd.DataFrame):
    """Panel of plots showing stability and collapse metrics."""
    metrics = ["u_std_mean", "u_norm_sd", "collapsed_dims"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        sns.barplot(
            data=df, 
            x="sparsity", 
            y=metric, 
            hue="model", 
            ax=axes[i]
        )
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")
        
    plt.tight_layout()
    return fig

def plot_reconstruction_tradeoff(df: pd.DataFrame):
    """Plot Latent Recovery vs Reconstruction Error."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, 
        x="recon_error", 
        y="recovery", 
        hue="model", 
        style="sparsity",
        s=100
    )
    plt.title("Reconstruction-Recovery Tradeoff")
    plt.xlabel("Normalized Reconstruction Error (lower is better)")
    plt.ylabel("Latent Recovery (higher is better)")
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def plot_v_heatmaps(results_dict: Dict[str, Any], modality_idx: int = 0):
    """Plot V matrix heatmaps for multiple models side-by-side."""
    models = list(results_dict.keys())
    if not models:
        return None
        
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
        
    for i, m_name in enumerate(models):
        res = results_dict[m_name]
        v = res["v"][modality_idx]
        if isinstance(v, torch.Tensor):
            v = v.numpy()
            
        sns.heatmap(v, cmap='coolwarm', center=0, ax=axes[i], cbar=False)
        axes[i].set_title(f"{m_name}\n(V Matrix)")
        
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
    sns.heatmap(corr, annot=True, cmap='viridis', vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("True Latent Dimension")
    plt.ylabel("Learned Dimension")
    return plt.gcf()
