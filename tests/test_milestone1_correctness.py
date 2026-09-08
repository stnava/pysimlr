"""Comprehensive Unit Verification Suite for Milestone 1 Defects.

Verifies:
1. Bug 1: multiscale_svd rank slicing (actual_nev = min(nev, len(s))) and full locn neighborhood aggregation.
2. Bug 2: smooth_regression properly fits target Y, computes regularized projection, regression coefficients, and y_pred.
3. Bug 3: sparse.py sparse_dist / sparse_distance_matrix guard against zero division when sigma=0.0.
4. Bug 4: consensus.py Bessel correction guard in compute_shared_consensus preventing NaN on single sample (N=1) or zero variance.
5. Bug 5: utils.py preprocess_data (centerAndScale) N=1 NaN protection; and exact adjusted_rvcoef formula from canonical ANTsR::adjusted_rvcoef.
6. Bug 6: optimizers.py ArmijoGradient line search sign alignment for cross-view correlation ascent.
7. Bug 7: paths.py simlr_path and permutation_test list consensus handling (_to_consensus_tensor).
8. Bug 8: simlr.py strict fast-failure input validation in simlr() raising ValueError on sample count mismatch (Ni != Nj) or 1D arrays.
"""

import pytest
import torch
import numpy as np

from pysimlr import (
    simlr,
    predict_simlr,
    predict_shared_latent,
    multiscale_svd,
    adjusted_rvcoef,
    rvcoef,
    smooth_matrix_prediction,
    smooth_regression,
    sparse_distance_matrix,
    sparse_distance_matrix_xy,
    compute_shared_consensus,
)
from pysimlr.sparse import sparse_dist
from pysimlr.paths import simlr_path, permutation_test
from pysimlr.utils import preprocess_data
from pysimlr.optimizers import ArmijoGradient, BidirectionalArmijoGradient


# ============================================================================
# Bug 1: multiscale_svd rank slicing and locn neighborhood aggregation
# ============================================================================

def test_m1_multiscale_svd_rank_slicing_nev_exceeds_dimension():
    """Verify multiscale_svd handles nev > min(n_samples_knn, n_features) without RuntimeError."""
    torch.manual_seed(42)
    x = torch.randn(20, 3)
    r = torch.tensor([1.0, 2.0])
    # nev=5 exceeds p=3 and knn=4: must return sliced tensor of size (len(r), min(nev, max_rank))
    res = multiscale_svd(x, r, locn=2, nev=5, knn=4)
    assert isinstance(res, dict)
    assert "evals_vs_scale" in res
    evals = res["evals_vs_scale"]
    assert evals.shape == (2, 3)  # min(5, min(4, 3)) = 3
    assert not torch.isnan(evals).any()


def test_m1_multiscale_svd_neighborhood_aggregation():
    """Verify multiscale_svd aggregates across all sampled locn points in KNN mode."""
    torch.manual_seed(42)
    x = torch.randn(30, 4)
    r = torch.tensor([1.0, 3.0])
    
    # Run with locn=5 and knn=6
    res = multiscale_svd(x, r, locn=5, nev=3, knn=6)
    evals = res["evals_vs_scale"]
    assert evals.shape == (2, 3)
    assert not torch.isnan(evals).any()
    assert (evals >= 0.0).all()

    # Radius-only mode (knn=0)
    res_rad = multiscale_svd(x, r, locn=[0, 1, 2, 3], nev=3, knn=0)
    assert res_rad["evals_vs_scale"].shape == (2, 3)
    assert not torch.isnan(res_rad["evals_vs_scale"]).any()


# ============================================================================
# Bug 2: smooth_regression target Y fitting, coefficients, and y_pred
# ============================================================================

def test_m1_smooth_regression_fits_target_y():
    """Verify smooth_regression fits target Y and returns distinct models for different targets."""
    torch.manual_seed(42)
    x = torch.randn(50, 10)
    w1 = torch.randn(10, 2)
    y1 = x @ w1 + torch.randn(50, 2) * 0.05
    w2 = torch.randn(10, 2)
    y2 = x @ w2 + torch.randn(50, 2) * 0.05

    res1 = smooth_regression(x, y1, nv=3)
    res2 = smooth_regression(x, y2, nv=3)

    # Must contain regression coefficients, y_pred, and projection
    assert "coef" in res1
    assert "y_pred" in res1
    assert "v" in res1
    assert "projection" in res1

    assert res1["y_pred"].shape == (50, 2)
    assert res1["coef"].shape == (10, 2)

    # Different targets must yield different coefficients and predictions
    assert not torch.allclose(res1["coef"], res2["coef"], atol=1e-2)
    assert not torch.allclose(res1["y_pred"], res2["y_pred"], atol=1e-2)


def test_m1_smooth_regression_reconstruction_accuracy():
    """Verify smooth_regression accurately recovers linear mapping on principal subspace."""
    torch.manual_seed(123)
    n, p = 60, 5
    x = torch.randn(n, p)
    true_w = torch.tensor([2.0, -1.5, 1.0, 0.5, -0.5])
    y = x @ true_w + torch.randn(n) * 0.01

    res = smooth_regression(x, y, nv=p, alpha=1e-6)
    assert res["y_pred"].shape == (n,)
    assert res["coef"].shape == (p,)

    # Correlation between true y and predicted y must be > 0.99
    corr = torch.corrcoef(torch.stack([y, res["y_pred"]]))[0, 1]
    assert corr.item() > 0.99


def test_m1_smooth_regression_sample_mismatch():
    """Verify smooth_regression and smooth_matrix_prediction fast-fail on sample count mismatch."""
    x = torch.randn(20, 5)
    y = torch.randn(15, 2)
    with pytest.raises(ValueError, match=r"Sample count mismatch"):
        smooth_regression(x, y)
    with pytest.raises(ValueError, match=r"Sample count mismatch"):
        smooth_matrix_prediction(x, y)


# ============================================================================
# Bug 3: sparse.py zero division guard when sigma=0.0
# ============================================================================

def test_m1_sparse_dist_zero_sigma_exception():
    """Verify sparse distance functions raise ValueError on sigma <= 0.0."""
    x = torch.randn(10, 3)
    y = torch.randn(8, 3)

    # sparse_distance_matrix
    with pytest.raises(ValueError, match=r"sigma must be strictly positive"):
        sparse_distance_matrix(x, k=3, sigma=0.0)
    with pytest.raises(ValueError, match=r"sigma must be strictly positive"):
        sparse_distance_matrix(x, k=3, sigma=-0.5)

    # sparse_dist alias
    with pytest.raises(ValueError, match=r"sigma must be strictly positive"):
        sparse_dist(x, k=3, sigma=0.0)

    # sparse_distance_matrix_xy
    with pytest.raises(ValueError, match=r"sigma must be strictly positive"):
        sparse_distance_matrix_xy(x, y, k=3, sigma=0.0)


def test_m1_sparse_dist_positive_sigma_affinity():
    """Verify positive sigma computes valid Gaussian affinities without NaNs or Infs."""
    torch.manual_seed(42)
    x = torch.randn(15, 4)
    smat = sparse_dist(x, k=4, sigma=1.5)
    assert smat.shape == (15, 15)
    assert not torch.isnan(smat).any()
    assert not torch.isinf(smat).any()
    assert (smat >= 0.0).all()
    assert (smat <= 1.0).all()


# ============================================================================
# Bug 4: consensus.py Bessel correction guard on N=1
# ============================================================================

def test_m1_consensus_single_sample_n1_no_nan():
    """Verify compute_shared_consensus on N=1 does not produce NaNs or zero-division."""
    torch.manual_seed(42)
    x1 = torch.randn(1, 6)
    x2 = torch.randn(1, 6)
    u = compute_shared_consensus([x1, x2], k=2, topology="star", training=False)
    assert not torch.isnan(u).any(), "Consensus U contains NaNs on N=1!"
    assert u.shape == (1, 2)


def test_m1_simlr_end_to_end_single_sample_n1():
    """Verify full SiMLR model fits cleanly on single sample N=1 with zero NaNs."""
    torch.manual_seed(42)
    x1 = torch.randn(1, 8)
    x2 = torch.randn(1, 6)
    res = simlr([x1, x2], k=2, iterations=5, scale_list=["centerAndScale", "np"])
    assert not torch.isnan(res["u"]).any(), "SiMLR consensus U contains NaNs on N=1!"
    assert not torch.isnan(res["v"][0]).any(), "Basis V[0] contains NaNs on N=1!"
    assert not torch.isnan(res["v"][1]).any(), "Basis V[1] contains NaNs on N=1!"


# ============================================================================
# Bug 5: utils.py preprocess_data N=1 protection and adjusted_rvcoef formula
# ============================================================================

def test_m1_preprocess_center_and_scale_n1():
    """Verify preprocess_data centerAndScale on N=1 does not produce NaNs in data or provenance."""
    x = torch.randn(1, 10)
    x_scaled, prov = preprocess_data(x, ["centerAndScale"])
    assert not torch.isnan(x_scaled).any()
    assert not torch.isnan(prov["cas_std"]).any()
    assert torch.allclose(prov["cas_std"], torch.ones_like(prov["cas_std"]))


def test_m1_adjusted_rvcoef_exact_formula():
    """Verify adjusted_rvcoef matches canonical ANTsR formula (1.0 on identity, near 0 on noise)."""
    torch.manual_seed(42)
    x = torch.randn(60, 20)
    y_noise = torch.randn(60, 20)

    # Self-correlation must equal 1.0
    rv_self = adjusted_rvcoef(x, x)
    assert np.isclose(rv_self, 1.0, atol=1e-4)

    # High-dimensional noise: raw RV is inflated, adjusted RV is centered near zero
    raw_rv = rvcoef(x, y_noise)
    adj_rv = adjusted_rvcoef(x, y_noise)
    assert raw_rv > 0.05, "Raw RV should be positively biased"
    assert abs(adj_rv) < 0.05, f"Adjusted RV should be near zero for noise, got {adj_rv}"

    # Edge cases
    assert adjusted_rvcoef(x[:1], y_noise[:1]) == 0.0
    assert adjusted_rvcoef(torch.zeros(10, 5), torch.zeros(10, 5)) == 0.0


# ============================================================================
# Bug 6: optimizers.py ArmijoGradient line search sign alignment
# ============================================================================

def test_m1_armijo_gradient_energy_monotonic_descent():
    """Verify ArmijoGradient steps reduce energy and align with correlation ascent."""
    torch.manual_seed(42)
    P, K = 10, 2
    V = torch.randn(P, K)
    A = torch.randn(P, K)

    # Objective energy E(V) = -sum(A * V), min E <=> max correlation
    def energy_fn(v):
        return -torch.sum(A * v)

    descent_grad = A.clone()  # -grad(E) = +A
    opt = ArmijoGradient("armijo_gradient", [V], learning_rate=0.1)

    # Test step with energy function
    v_next = opt.step(0, V, descent_grad, energy_fn)
    assert energy_fn(v_next) < energy_fn(V), "ArmijoGradient step increased energy!"

    # Test fallback without energy function
    opt_fb = ArmijoGradient("armijo_gradient", [V], learning_rate=0.1)
    v_fb = opt_fb.step(0, V, descent_grad, None)
    assert energy_fn(v_fb) < energy_fn(V), "Armijo fallback increased energy!"


def test_m1_bidirectional_armijo_gradient_descent():
    """Verify BidirectionalArmijoGradient steps reduce energy."""
    torch.manual_seed(42)
    P, K = 8, 2
    V = torch.randn(P, K)
    A = torch.randn(P, K)

    def energy_fn(v):
        return -torch.sum(A * v)

    descent_grad = A.clone()
    opt = BidirectionalArmijoGradient("bidirectional_armijo", [V], learning_rate=0.1)
    v_next = opt.step(0, V, descent_grad, energy_fn)
    assert energy_fn(v_next) < energy_fn(V), "BidirectionalArmijo step increased energy!"


def test_m1_simlr_armijo_gradient_execution():
    """Verify SiMLR executes reliably with optimizer_type='armijo_gradient'."""
    torch.manual_seed(42)
    z = torch.randn(35, 2)
    x1 = z @ torch.randn(2, 6) + torch.randn(35, 6) * 0.1
    x2 = z @ torch.randn(2, 5) + torch.randn(35, 5) * 0.1

    res = simlr([x1, x2], k=2, iterations=10, optimizer_type="armijo_gradient")
    assert "u" in res
    assert "v" in res
    assert not torch.isnan(res["u"]).any()
    assert not torch.isnan(res["v"][0]).any()


# ============================================================================
# Bug 7: paths.py simlr_path and permutation_test list consensus handling
# ============================================================================

def test_m1_simlr_path_loo_topology():
    """Verify simlr_path computes consensus correlations cleanly under topology='loo'."""
    torch.manual_seed(42)
    x1 = torch.randn(25, 8)
    x2 = torch.randn(25, 6)
    x3 = torch.randn(25, 7)
    path_model = [[0, 1], [0, 1, 2]]

    res = simlr_path([x1, x2, x3], k=2, path_model=path_model, iterations=3, topology="loo")
    assert "path_results" in res
    assert "consensus_correlations" in res
    assert len(res["consensus_correlations"]) == 2
    assert np.isclose(res["consensus_correlations"][-1], 1.0, atol=1e-3)
    assert not np.isnan(res["consensus_correlations"][0])


def test_m1_simlr_path_graph_topology():
    """Verify simlr_path computes consensus correlations cleanly under topology='graph'."""
    torch.manual_seed(42)
    x1 = torch.randn(25, 8)
    x2 = torch.randn(25, 6)
    x3 = torch.randn(25, 7)
    graph = {0: [1], 1: [0, 2], 2: [1]}
    path_model = [[0, 1], [0, 1, 2]]

    res = simlr_path([x1, x2, x3], k=2, path_model=path_model, iterations=3, topology="graph", path_graph=graph)
    assert len(res["consensus_correlations"]) == 2
    assert np.isclose(res["consensus_correlations"][-1], 1.0, atol=1e-3)


def test_m1_permutation_test_loo_topology():
    """Verify permutation_test evaluates view-specific consensus under topology='loo'."""
    torch.manual_seed(42)
    x1 = torch.randn(20, 5)
    x2 = torch.randn(20, 5)
    x3 = torch.randn(20, 5)
    perm_res = permutation_test([x1, x2, x3], k=2, n_permutations=2, iterations=2, topology="loo")
    assert "observed_similarity" in perm_res
    assert "p_value" in perm_res
    assert not np.isnan(perm_res["observed_similarity"])
    assert 0.0 <= perm_res["p_value"] <= 1.0


# ============================================================================
# Bug 8: simlr.py strict fast-failure input contract validation
# ============================================================================

def test_m1_input_contract_mismatched_samples():
    """Verify fast failure with descriptive ValueError on mismatched view sample counts."""
    x1 = torch.randn(20, 5)
    x2 = torch.randn(30, 5)
    with pytest.raises(ValueError, match=r"identical number of samples|mismatched sample counts"):
        simlr([x1, x2], k=2)


def test_m1_input_contract_1d_array():
    """Verify fast failure with descriptive ValueError on 1D arrays."""
    x1 = torch.randn(20)
    x2 = torch.randn(20, 5)
    with pytest.raises(ValueError, match=r"2D"):
        simlr([x1, x2], k=2)


def test_m1_input_contract_empty_views():
    """Verify fast failure on empty list."""
    with pytest.raises(ValueError, match=r"non-empty"):
        simlr([], k=2)


def test_m1_input_contract_invalid_k():
    """Verify fast failure on non-positive k."""
    x1 = torch.randn(20, 5)
    x2 = torch.randn(20, 5)
    with pytest.raises(ValueError, match=r"positive integer"):
        simlr([x1, x2], k=0)
    with pytest.raises(ValueError, match=r"positive integer"):
        simlr([x1, x2], k=-2)


def test_m1_predict_simlr_contracts():
    """Verify predict_simlr and predict_shared_latent validate input contracts."""
    x1 = torch.randn(20, 5)
    x2 = torch.randn(20, 5)
    res = simlr([x1, x2], k=2, iterations=3)

    # View count mismatch
    with pytest.raises(ValueError, match=r"Expected 2 data matrices"):
        predict_simlr([x1], res)
    with pytest.raises(ValueError, match=r"Expected 2 data matrices"):
        predict_shared_latent([x1], res)

    # Sample count mismatch at predict time
    with pytest.raises(ValueError, match=r"identical number of samples|mismatched sample counts"):
        predict_simlr([torch.randn(15, 5), torch.randn(25, 5)], res)
    with pytest.raises(ValueError, match=r"identical number of samples|mismatched sample counts"):
        predict_shared_latent([torch.randn(15, 5), torch.randn(25, 5)], res)

    # 1D array at predict time
    with pytest.raises(ValueError, match=r"2D"):
        predict_simlr([torch.randn(20), torch.randn(20, 5)], res)
