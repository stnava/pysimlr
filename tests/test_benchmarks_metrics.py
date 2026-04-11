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
    calculate_all_metrics,
    first_layer_sparsity_metrics,
    alignment_metrics_from_report,
    prediction_preservation_metrics_from_report,
    shared_attribution_metrics_from_report,
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


def test_pr4_first_layer_metric_extractors():
    first_layer = {
        "orthogonality_defect": [0.1, 0.2],
        "sparsity_summary": [
            {"component_density": [0.2, 0.4], "component_l0": [2, 4]},
            {"component_density": [0.1, 0.3], "component_l0": [1, 3]},
        ],
    }
    metrics = first_layer_sparsity_metrics(first_layer)
    assert metrics["first_layer_density_mean"] > 0.0
    assert metrics["first_layer_l0_mean"] > 0.0
    assert metrics["first_layer_orthogonality_mean"] == pytest.approx(0.15)


def test_pr4_alignment_and_attribution_metric_extractors():
    alignment = {
        "modalities": [
            {
                "global_r2": 0.8,
                "component_correlation": torch.tensor([[1.0, 0.2], [0.1, 0.7]]),
                "feature_importance": torch.tensor([0.4, 0.2, 0.1]),
            }
        ]
    }
    shared = {
        "per_modality": [{"global_r2": 0.75}],
        "combined": {"component_importance": torch.tensor([0.8, 0.2])},
    }
    pred = {
        "per_modality": [{"global_r2": 0.5}, {"global_r2": 0.25}],
        "shared_latent_baseline": {"global_r2": 0.8},
    }

    align_metrics = alignment_metrics_from_report(alignment)
    shared_metrics = shared_attribution_metrics_from_report(shared)
    pred_metrics = prediction_preservation_metrics_from_report(pred)

    assert align_metrics["first_layer_alignment_r2_mean"] == pytest.approx(0.8)
    assert align_metrics["first_layer_alignment_corr_mean"] > 0.0
    assert shared_metrics["shared_to_first_layer_r2_mean"] == pytest.approx(0.75)
    assert shared_metrics["shared_component_concentration"] > 0.0
    assert pred_metrics["first_layer_prediction_r2_mean"] == pytest.approx(0.375)
    assert pred_metrics["first_layer_prediction_preservation"] > 0.0
