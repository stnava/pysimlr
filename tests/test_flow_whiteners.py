import pytest
import torch
import numpy as np
import pandas as pd

from antstorch.lamnr_flows import lamnr_flows_whitener, apply_lamnr_flows_whitener
from antstorch import (
    lamnr_flows_whitener as top_lamnr_flows_whitener,
    apply_lamnr_flows_whitener as top_apply_lamnr_flows_whitener
)
from pysimlr import flow_whiten_matrix, FlowWhitener, flow_whitener


def test_antstorch_whitener_exports():
    """Verify clean accessibility of flow whitener functions from both ANTsTorch entry points."""
    assert callable(lamnr_flows_whitener)
    assert callable(apply_lamnr_flows_whitener)
    assert top_lamnr_flows_whitener is lamnr_flows_whitener
    assert top_apply_lamnr_flows_whitener is apply_lamnr_flows_whitener


def test_antstorch_lamnr_flows_whitener_multiview_train():
    """Verify ANTsTorch lamnr_flows_whitener multi-view training pipeline."""
    np.random.seed(42)
    n = 50
    p1, p2 = 4, 5
    df1 = pd.DataFrame(np.random.randn(n, p1).copy(), columns=[f"v1_{i}" for i in range(p1)])
    df2 = pd.DataFrame(np.random.randn(n, p2).copy(), columns=[f"v2_{i}" for i in range(p2)])

    trainer_out = lamnr_flows_whitener(
        [df1, df2],
        K=4,
        max_iter=20,
        batch_size=25,
        val_batch_size=25,
        val_interval=10,
        base_distribution="GaussianPCA",
        pca_latent_dimension=3,
        early_stop_enabled=False
    )

    assert isinstance(trainer_out, dict)
    assert "models" in trainer_out
    assert "metrics" in trainer_out
    assert "val_history" in trainer_out
    assert "dataset_normalizers" in trainer_out

    assert len(trainer_out["models"]) == 2
    assert trainer_out["metrics"]["best_step"] > 0


def test_antstorch_whitener_forward_and_reconstruction():
    """Verify exact bijective invertibility of trained normalizing flow models in latent space."""
    np.random.seed(123)
    n, p = 30, 4
    x = np.random.randn(n, p).copy()
    df = pd.DataFrame(x, columns=[f"feat_{i}" for i in range(p)])

    trainer_out = lamnr_flows_whitener(
        [df],
        K=4,
        max_iter=15,
        batch_size=15,
        val_batch_size=15,
        val_interval=10,
        base_distribution="GaussianPCA",
        pca_latent_dimension=4,
        early_stop_enabled=False
    )

    # Forward pass to latent z space
    z_df = apply_lamnr_flows_whitener(trainer_out, [df], direction="forward", output_space="z")
    assert isinstance(z_df, list)
    assert z_df[0].shape == (n, p)
    assert np.all(np.isfinite(z_df[0].values))

    # Inverse pass from latent z space back to data space
    rec_df = apply_lamnr_flows_whitener(trainer_out, z_df, direction="inverse", input_space="z")
    assert rec_df[0].shape == (n, p)
    assert np.all(np.isfinite(rec_df[0].values))

    # Invertibility to high precision
    max_recon_diff = np.max(np.abs(rec_df[0].values - df.values))
    assert max_recon_diff < 1e-5, f"Reconstruction difference too large: {max_recon_diff}"


def test_antstorch_whitener_output_spaces():
    """Verify forward application with different output spaces (whitened, z, whitened_full)."""
    np.random.seed(99)
    n, p = 40, 4
    pca_dim = 3
    df = pd.DataFrame(np.random.randn(n, p).copy(), columns=[f"x_{i}" for i in range(p)])

    trainer_out = lamnr_flows_whitener(
        [df],
        K=4,
        max_iter=10,
        batch_size=20,
        val_batch_size=20,
        base_distribution="GaussianPCA",
        pca_latent_dimension=pca_dim,
        early_stop_enabled=False
    )

    # Output space 'z' -> dimension p
    z_res = apply_lamnr_flows_whitener(trainer_out, [df], direction="forward", output_space="z")
    assert z_res[0].shape == (n, p)

    # Output space 'whitened' -> dimension pca_latent_dimension
    whitened_res = apply_lamnr_flows_whitener(trainer_out, [df], direction="forward", output_space="whitened")
    assert whitened_res[0].shape == (n, pca_dim)
    assert np.all(np.isfinite(whitened_res[0].values))

    # Output space 'whitened_full' -> dimension p
    whitened_full = apply_lamnr_flows_whitener(trainer_out, [df], direction="forward", output_space="whitened_full")
    assert whitened_full[0].shape == (n, p)
    assert np.all(np.isfinite(whitened_full[0].values))


def test_antstorch_whitener_diag_gaussian():
    """Verify training with DiagGaussian base distribution."""
    np.random.seed(7)
    n, p = 30, 4
    df = pd.DataFrame(np.random.randn(n, p).copy(), columns=[f"g_{i}" for i in range(p)])

    trainer_out = lamnr_flows_whitener(
        [df],
        K=4,
        max_iter=10,
        batch_size=15,
        val_batch_size=15,
        base_distribution="DiagGaussian",
        early_stop_enabled=False
    )

    assert "models" in trainer_out
    z_res = apply_lamnr_flows_whitener(trainer_out, [df], direction="forward", output_space="z")
    assert z_res[0].shape == (n, p)


def test_pysimlr_flow_whiten_matrix_single_tensor():
    """Verify pysimlr flow_whiten_matrix function on a single PyTorch tensor."""
    torch.manual_seed(42)
    n, p = 35, 4
    x = torch.randn(n, p)

    res = flow_whiten_matrix(
        x,
        K=4,
        max_iter=15,
        batch_size=20,
        base_distribution="GaussianPCA",
        pca_latent_dimension=3,
        output_space="whitened"
    )

    assert isinstance(res, dict)
    assert "whitened_matrix" in res
    assert "whitened_views" in res
    assert "whitened_dfs" in res
    assert "models" in res
    assert "trainer_output" in res
    assert "whitener" in res

    # Check tensor type preservation
    assert isinstance(res["whitened_matrix"], torch.Tensor)
    assert res["whitened_matrix"].shape == (n, 3)
    assert res["whitened_matrix"].device == x.device
    assert not torch.isnan(res["whitened_matrix"]).any()


def test_pysimlr_flow_whiten_matrix_multiview_tensors():
    """Verify pysimlr flow_whiten_matrix function on multi-view input."""
    torch.manual_seed(42)
    n = 40
    x1 = torch.randn(n, 4)
    x2 = torch.randn(n, 6)

    res = flow_whiten_matrix(
        [x1, x2],
        K=4,
        max_iter=15,
        batch_size=20,
        base_distribution="GaussianPCA",
        pca_latent_dimension=3,
        output_space="whitened"
    )

    assert isinstance(res["whitened_views"], list)
    assert len(res["whitened_views"]) == 2
    assert res["whitened_views"][0].shape == (n, 3)
    assert res["whitened_views"][1].shape == (n, 3)
    assert isinstance(res["whitened_views"][0], torch.Tensor)
    assert isinstance(res["whitened_views"][1], torch.Tensor)


def test_pysimlr_flow_whitener_class_fit_transform():
    """Verify FlowWhitener object-oriented interface (fit, transform, inverse_transform)."""
    torch.manual_seed(42)
    n, p = 30, 4
    x = torch.randn(n, p)

    whitener = FlowWhitener(
        K=4,
        max_iter=15,
        batch_size=15,
        base_distribution="GaussianPCA",
        pca_latent_dimension=p,
        output_space="z"
    )

    whitener.fit(x)
    assert whitener.trainer_output is not None
    assert whitener.models is not None

    # Transform to z space
    z = whitener.transform(x, output_space="z")
    assert isinstance(z, torch.Tensor)
    assert z.shape == (n, p)

    # Invert back to original data space
    x_rec = whitener.inverse_transform(z, input_space="z")
    assert isinstance(x_rec, torch.Tensor)
    assert x_rec.shape == (n, p)
    assert torch.allclose(x, x_rec, atol=1e-5)

    # Test fit_transform
    z_fit = whitener.fit_transform(x, output_space="z")
    assert z_fit.shape == (n, p)


def test_pysimlr_flow_whiten_matrix_numpy_and_dataframe():
    """Verify flow_whiten_matrix accepts and handles NumPy arrays and pandas DataFrames."""
    np.random.seed(42)
    n, p = 25, 4
    arr = np.random.randn(n, p)
    df = pd.DataFrame(arr.copy(), columns=[f"col_{i}" for i in range(p)])

    # Test with NumPy array
    res_np = flow_whiten_matrix(arr, K=4, max_iter=10, batch_size=15, output_space="z")
    assert isinstance(res_np["whitened_matrix"], np.ndarray)
    assert res_np["whitened_matrix"].shape == (n, p)

    # Test with DataFrame
    res_df = flow_whiten_matrix(df, K=4, max_iter=10, batch_size=15, output_space="z")
    assert isinstance(res_df["whitened_matrix"], pd.DataFrame)
    assert res_df["whitened_matrix"].shape == (n, p)


def test_pysimlr_flow_whitener_alias():
    """Verify flow_whitener alias function behaves identically to flow_whiten_matrix."""
    assert flow_whitener is flow_whiten_matrix
    torch.manual_seed(42)
    x = torch.randn(20, 3)
    res = flow_whitener(x, K=4, max_iter=10, batch_size=10, output_space="z")
    assert "whitened_matrix" in res
