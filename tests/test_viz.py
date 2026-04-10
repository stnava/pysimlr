import torch
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pysimlr.viz import (
    plot_view_correlations,
    plot_latent_consensus,
    plot_feature_signatures,
    plot_convergence_dynamics
)

def test_plot_view_correlations():
    data = [torch.randn(50, 10), torch.randn(50, 10)]
    fig = plot_view_correlations(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_latent_consensus():
    u_shared = torch.randn(50, 2)
    mod_latents = [torch.randn(50, 2), torch.randn(50, 2)]
    fig = plot_latent_consensus(u_shared, mod_latents)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_feature_signatures():
    v_mat = torch.randn(100, 5)
    fig = plot_feature_signatures(v_mat)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # With feature names
    names = [f"F{i}" for i in range(100)]
    fig2 = plot_feature_signatures(v_mat, feature_names=names, top_n=10)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

def test_plot_convergence_dynamics():
    history = {
        "loss_history": [10, 5, 2, 1, 0.5],
        "recon_history": [20, 15, 10, 5, 2]
    }
    fig = plot_convergence_dynamics(history)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Empty history
    fig_empty = plot_convergence_dynamics({})
    assert fig_empty is None
