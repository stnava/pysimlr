import torch
import numpy as np
import pytest
from pysimlr.benchmarks.metrics import (
    latent_recovery_score,
    outcome_r2_score,
    reconstruction_mse,
    latent_variance_diagnostics,
    shared_private_diagnostics,
    calculate_v_orthogonality,
    calculate_all_metrics
)

def test_latent_recovery_score():
    u_true = torch.randn(100, 5)
    u_pred = u_true + 0.1 * torch.randn(100, 5)
    score = latent_recovery_score(u_pred, u_true)
    assert isinstance(score, float)
    assert score > 0.5

def test_outcome_r2_score():
    u_pred = torch.randn(100, 5)
    weights = np.random.randn(5, 1)
    y_true = (u_pred.numpy() @ weights).flatten() + 0.1 * np.random.randn(100)
    score = outcome_r2_score(u_pred, y_true)
    assert isinstance(score, float)
    assert score > 0.8

def test_reconstruction_mse():
    data = [torch.randn(100, 10), torch.randn(100, 20)]
    recons = [d + 0.05 * torch.randn(*d.shape) for d in data]
    mse = reconstruction_mse(data, recons)
    assert isinstance(mse, float)
    assert mse < 0.1

def test_latent_variance_diagnostics():
    u = torch.randn(100, 5)
    # Force one dimension to collapse
    u[:, 0] = 0.0
    diag = latent_variance_diagnostics(u)
    assert "u_std_mean" in diag
    assert "collapsed_dims" in diag
    assert diag["collapsed_dims"] >= 1

def test_shared_private_diagnostics():
    shared = [torch.randn(100, 5), torch.randn(100, 5)]
    private = [torch.randn(100, 3), torch.randn(100, 3)]
    diag = shared_private_diagnostics(shared, private)
    assert "mod0_shared_var" in diag
    assert "mod1_cross_cov" in diag

def test_calculate_v_orthogonality():
    v1 = torch.eye(10, 5)
    v2 = torch.eye(10, 5)
    defect = calculate_v_orthogonality([v1, v2])
    assert isinstance(defect, float)
    assert defect < 1e-5

def test_calculate_all_metrics():
    u_true = torch.randn(100, 5)
    u_pred = u_true + 0.01 * torch.randn(100, 5)
    y_true = np.random.randn(100)
    data = [torch.randn(100, 10)]
    recons = [data[0] + 0.01 * torch.randn(100, 10)]
    
    metrics = calculate_all_metrics(u_pred, u_true, y_true, data, recons)
    assert "recovery" in metrics
    assert "test_r2" in metrics
    assert "recon_error" in metrics
    
    shared = [torch.randn(100, 2)]
    private = [torch.randn(100, 2)]
    v_mats = [torch.eye(10, 2)]
    
    metrics_full = calculate_all_metrics(
        u_pred, u_true, y_true, data, recons,
        shared_latents=shared, private_latents=private, v_mats=v_mats
    )
    assert "mod0_shared_var" in metrics_full
    assert "orthogonality_defect" in metrics_full
