"""Whitening must produce an identity covariance.

`whiten_matrix` used to return `u * (1 / s)`, but `safe_pca` already returns
the orthonormal left singular vectors -- i.e. the scores with `s` divided out
once. Dividing again gave column variances falling off as 1/s**2 instead of
being constant, so the function did not whiten at all.
"""
import numpy as np
import pytest
import torch

from pysimlr.svd import safe_pca, whiten_matrix


def _cov(w):
    wc = w - w.mean(dim=0, keepdim=True)
    return wc.t() @ wc / (w.shape[0] - 1)


@pytest.mark.parametrize("spectrum", [
    [10.0, 5.0, 1.0, 0.5, 0.1],
    [1.0, 1.0, 1.0],
    [100.0, 0.01],
])
def test_whitened_covariance_is_identity(spectrum):
    torch.manual_seed(0)
    k = len(spectrum)
    x = torch.randn(300, k) @ torch.diag(torch.tensor(spectrum))
    res = whiten_matrix(x)
    w = res["whitened_matrix"]
    assert res["rank"] == k
    cov = _cov(w)
    assert torch.allclose(cov, torch.eye(k), atol=1e-4), (
        f"max deviation {float((cov - torch.eye(k)).abs().max()):.3e}"
    )


def test_whitened_columns_are_zero_mean_unit_variance():
    torch.manual_seed(1)
    x = torch.randn(150, 4) @ torch.randn(4, 6)
    w = whiten_matrix(x, nc=4)["whitened_matrix"]
    assert torch.allclose(w.mean(0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(w.std(0), torch.ones(4), atol=1e-3)


def test_whitening_is_invariant_to_input_scaling():
    """Whitening removes scale, so scaling the input must not change the result
    (up to per-component sign)."""
    torch.manual_seed(2)
    x = torch.randn(120, 5)
    a = whiten_matrix(x, nc=5)["whitened_matrix"]
    b = whiten_matrix(x * 1000.0, nc=5)["whitened_matrix"]
    for j in range(5):
        c = abs(float(np.corrcoef(a[:, j].numpy(), b[:, j].numpy())[0, 1]))
        assert c > 0.999, f"component {j} changed under input rescaling (|r|={c:.4f})"


def test_rank_deficient_input_reports_rank_and_zeroes_null_directions():
    torch.manual_seed(3)
    x = torch.randn(60, 6)
    x[:, 1] = x[:, 0] * 3.5           # exact collinearity -> rank 5
    res = whiten_matrix(x, nc=6)
    w = res["whitened_matrix"]
    assert res["rank"] == 5
    assert torch.isfinite(w).all()
    cov = _cov(w)
    assert torch.allclose(cov[:5, :5], torch.eye(5), atol=1e-4)
    # the null direction is returned as exact zeros, not noise blown up by 1/~0
    assert torch.equal(w[:, 5], torch.zeros(60))


def test_nc_larger_than_rank_does_not_crash():
    """safe_pca used to raise a shape-mismatch RuntimeError here."""
    res = whiten_matrix(torch.randn(5, 100), nc=10)
    assert res["whitened_matrix"].shape == (5, 10)
    assert torch.isfinite(res["whitened_matrix"]).all()


@pytest.mark.parametrize("shape,nc", [((5, 100), 10), ((100, 5), 10), ((3, 3), 8), ((1, 10), 4)])
def test_safe_pca_shapes_are_always_nc_wide(shape, nc):
    r = safe_pca(torch.randn(*shape), nc=nc)
    assert r["u"].shape == (shape[0], nc)
    assert r["v"].shape == (shape[1], nc)
    assert r["s"].shape == (nc,)


def test_safe_pca_constant_features_get_zero_loadings():
    torch.manual_seed(4)
    x = torch.randn(30, 5)
    x[:, 2] = 1.0
    r = safe_pca(x, nc=3)
    assert torch.equal(r["v"][2], torch.zeros(3))


def test_all_constant_input_is_degenerate_but_finite():
    r = whiten_matrix(torch.ones(10, 4), nc=3)
    assert r["rank"] == 0
    assert torch.equal(r["whitened_matrix"], torch.zeros(10, 3))
