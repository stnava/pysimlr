import torch
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from pysimlr.benchmarks.plotting import (
    plot_pareto_recovery_vs_r2,
    plot_sparsity_sensitivity,
    plot_stability_diagnostics,
    plot_sparsity_vs_orthogonality,
    plot_reconstruction_tradeoff,
    plot_v_heatmaps,
    plot_latent_correlation
)

@pytest.fixture
def sample_df():
    data = {
        "model": ["linear", "linear", "ned", "ned"],
        "sparsity": [0.0, 0.5, 0.0, 0.5],
        "recovery": [0.8, 0.7, 0.9, 0.85],
        "test_r2": [0.6, 0.5, 0.7, 0.8],
        "recon_error": [0.1, 0.15, 0.05, 0.08],
        "u_std_mean": [1.0, 0.9, 1.1, 1.0],
        "u_norm_sd": [0.2, 0.3, 0.1, 0.15],
        "collapsed_dims": [0, 0, 0, 0],
        "orthogonality_defect": [0.01, 0.05, 0.02, 0.06]
    }
    return pd.DataFrame(data)

def test_plot_pareto_recovery_vs_r2(sample_df):
    fig = plot_pareto_recovery_vs_r2(sample_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_sparsity_sensitivity(sample_df):
    fig = plot_sparsity_sensitivity(sample_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_stability_diagnostics(sample_df):
    fig = plot_stability_diagnostics(sample_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_sparsity_vs_orthogonality(sample_df):
    fig = plot_sparsity_vs_orthogonality(sample_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Missing column
    df_missing = sample_df.drop(columns=["orthogonality_defect"])
    fig_none = plot_sparsity_vs_orthogonality(df_missing)
    assert fig_none is None

def test_plot_reconstruction_tradeoff(sample_df):
    fig = plot_reconstruction_tradeoff(sample_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_v_heatmaps():
    results = {
        "linear": {"v": [torch.randn(10, 2), torch.randn(10, 2)]},
        "ned": {"v": [torch.randn(10, 2), torch.randn(10, 2)]}
    }
    fig = plot_v_heatmaps(results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Empty
    assert plot_v_heatmaps({}) is None

def test_plot_latent_correlation():
    u_pred = torch.randn(50, 2)
    u_true = torch.randn(50, 2)
    fig = plot_latent_correlation(u_pred, u_true)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
