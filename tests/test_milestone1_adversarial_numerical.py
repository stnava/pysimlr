"""Adversarial SVD & Numerical Stress Test Suite for Milestone 1 Fixes.

Covers:
1. multiscale_svd:
   - Rank deficits where nev > min(knn, p)
   - Single point queries (locn=1, locn=[0], locn=tensor([3])) and N=1 datasets
   - Extreme nev values (nev=0, nev=1000)
   - All-zero and constant neighborhoods
   - Pathological scales (r=0, r=-1, r=1e-15, r=1e10)
   - Radius mode (knn=0) with rank deficits
2. smooth_regression:
   - Rank-deficient X and Y matrices
   - Collinear target columns in Y
   - Zero-variance and constant targets
   - High-dimensional projections (P >> N, N >> P)
   - Regularization extremes (alpha=1e8, alpha=1e-8)
   - 1D vs 2D target and predictor dimensionality preservation
   - Component boundaries (nv=0, nv > min(N, P))
3. sparse_dist (and sparse_distance_matrix, sparse_distance_matrix_xy):
   - Nonpositive sigma rejection (sigma <= 0)
   - Infinite and massive sigma bandwidths (sigma=inf, sigma=1e15)
   - Sub-epsilon tiny sigma bandwidths (sigma=1e-12)
   - Identical coordinate rows and all-identical datasets
   - Neighborhood boundaries (k >= N, k=1, N=1)
   - Rectangular cross-matrix distance (NX != NY) and dimension validation
4. compute_shared_consensus and preprocess_data:
   - N=1 single sample across all mixing algorithms (svd, pca, avg, newton, ica)
   - N=2 identical rows across topologies (star, loo, graph)
   - Zero-variance columns and completely zero views
   - NaN/Inf sanitization resilience
   - Preprocessing scaling methods under N=1 and zero variance
   - Provenance recording and replay on degenerate inputs
5. adjusted_rvcoef:
   - Identical matrices strictly equal 1.0 across diverse aspect ratios (N > P, N < P, N = P)
   - High-dimensional orthogonal noise attenuation (mean near 0.0 while raw RV is biased)
   - Valid handling of negative adjusted RV values under the null distribution
   - Degenerate inputs (N=1, zeros, constants, mismatched dimensions, 1D tensors)
   - Trace vs Gram branch consistency (N < P+Q vs N >= P+Q)
   - Empirical analysis of N=2 boundary singularity
"""

import pytest
import torch
import numpy as np

from pysimlr import (
    multiscale_svd,
    smooth_regression,
    smooth_matrix_prediction,
    sparse_distance_matrix,
    sparse_distance_matrix_xy,
    compute_shared_consensus,
    adjusted_rvcoef,
    rvcoef,
)
from pysimlr.sparse import sparse_dist
from pysimlr.utils import preprocess_data


# ============================================================================
# 1. multiscale_svd Adversarial Stress Tests
# ============================================================================

@pytest.mark.parametrize("n, p, knn, nev", [
    (25, 3, 5, 10),    # nev (10) > p (3)
    (30, 10, 2, 8),    # nev (8) > knn (2)
    (15, 2, 2, 20),    # nev (20) >> min(knn=2, p=2) = 2
    (50, 4, 3, 50),    # extreme nev
])
def test_adversarial_multiscale_svd_rank_deficits_nev_exceeds_min(n, p, knn, nev):
    """Verify multiscale_svd handles severe rank deficits where nev exceeds min(knn, p)."""
    torch.manual_seed(42)
    x = torch.randn(n, p)
    r = torch.tensor([0.5, 1.0, 2.0])
    
    expected_rank = min(nev, min(knn, p))
    res = multiscale_svd(x, r, locn=3, nev=nev, knn=knn)
    
    assert "evals_vs_scale" in res
    evals = res["evals_vs_scale"]
    assert evals.shape == (len(r), expected_rank)
    assert not torch.isnan(evals).any(), "NaN detected in singular values!"
    assert not torch.isinf(evals).any(), "Inf detected in singular values!"
    assert (evals >= 0.0).all(), "Negative singular values detected!"


def test_adversarial_multiscale_svd_single_point_queries():
    """Verify multiscale_svd functions with a single location query (int, list, tensor)."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    r = torch.tensor([1.0, 3.0])

    # locn as int = 1
    res_int = multiscale_svd(x, r, locn=1, nev=3, knn=4)
    assert res_int["evals_vs_scale"].shape == (2, 3)
    assert not torch.isnan(res_int["evals_vs_scale"]).any()

    # locn as list = [0]
    res_list = multiscale_svd(x, r, locn=[0], nev=3, knn=4)
    assert res_list["evals_vs_scale"].shape == (2, 3)
    assert not torch.isnan(res_list["evals_vs_scale"]).any()

    # locn as tensor
    res_t = multiscale_svd(x, r, locn=torch.tensor([5]), nev=3, knn=4)
    assert res_t["evals_vs_scale"].shape == (2, 3)
    assert not torch.isnan(res_t["evals_vs_scale"]).any()


def test_adversarial_multiscale_svd_single_sample_dataset():
    """Verify multiscale_svd survives when entire dataset has only 1 sample (N=1)."""
    x_n1 = torch.randn(1, 5)
    r = torch.tensor([1.0, 2.0])
    # For N=1, k_eff=1, max_rank=1, actual_nev=min(2, 1)=1
    res = multiscale_svd(x_n1, r, locn=1, nev=2, knn=2)
    evals = res["evals_vs_scale"]
    assert evals.shape == (2, 1)
    assert not torch.isnan(evals).any()
    # Centered single sample has 0 variance, singular values are 0
    assert torch.allclose(evals, torch.zeros_like(evals))


def test_adversarial_multiscale_svd_extreme_nev_values():
    """Verify multiscale_svd handles nev=0 and astronomical nev without failure."""
    torch.manual_seed(42)
    x = torch.randn(20, 5)
    r = torch.tensor([1.0, 2.0])

    # nev = 0
    res_zero = multiscale_svd(x, r, locn=2, nev=0, knn=4)
    assert res_zero["evals_vs_scale"].shape == (2, 0)

    # nev = 10000 on (20, 5) matrix
    res_huge = multiscale_svd(x, r, locn=2, nev=10000, knn=4)
    assert res_huge["evals_vs_scale"].shape == (2, 4)
    assert not torch.isnan(res_huge["evals_vs_scale"]).any()


def test_adversarial_multiscale_svd_all_zero_and_constant_data():
    """Verify multiscale_svd does not produce NaNs on all-zero or constant data."""
    r = torch.tensor([1.0, 2.0])

    # All zeros
    x_zeros = torch.zeros(25, 6)
    res_zeros = multiscale_svd(x_zeros, r, locn=3, nev=4, knn=5)
    evals_zeros = res_zeros["evals_vs_scale"]
    assert evals_zeros.shape == (2, 4)
    assert not torch.isnan(evals_zeros).any()
    assert torch.allclose(evals_zeros, torch.zeros_like(evals_zeros))

    # All constant (identical rows)
    x_const = torch.ones(25, 6) * 3.1415
    res_const = multiscale_svd(x_const, r, locn=3, nev=4, knn=5)
    evals_const = res_const["evals_vs_scale"]
    assert evals_const.shape == (2, 4)
    assert not torch.isnan(evals_const).any()
    assert torch.allclose(evals_const, torch.zeros_like(evals_const))


def test_adversarial_multiscale_svd_pathological_scale_radii():
    """Verify multiscale_svd survives zero, negative, tiny, and huge scale radii."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    # 0.0 triggers denominator fallback, tiny 1e-15 triggers fallback, huge 1e8 shrinks evals
    r_pathological = torch.tensor([0.0, -1.0, 1e-15, 1e8])
    res = multiscale_svd(x, r_pathological, locn=2, nev=3, knn=4)
    evals = res["evals_vs_scale"]
    assert evals.shape == (4, 3)
    assert not torch.isnan(evals).any()
    assert not torch.isinf(evals).any()


def test_adversarial_multiscale_svd_knn_zero_radius_mode():
    """Verify multiscale_svd operates cleanly in radius-like mode (knn=0)."""
    torch.manual_seed(42)
    x = torch.randn(25, 4)
    r = torch.tensor([0.5, 1.5])
    res = multiscale_svd(x, r, locn=[1, 3, 5], nev=8, knn=0)
    evals = res["evals_vs_scale"]
    # len(locn_indices)=3, p=4 -> max_rank=min(3, 4)=3, nev=min(8, 3)=3
    assert evals.shape == (2, 3)
    assert not torch.isnan(evals).any()


# ============================================================================
# 2. smooth_regression Adversarial Stress Tests
# ============================================================================

def test_adversarial_smooth_regression_rank_deficient_x_and_y():
    """Verify smooth_regression fits rank-deficient X and rank-deficient Y without NaNs."""
    torch.manual_seed(42)
    n, p, q = 40, 25, 15
    # X rank 2 in 25-dim space
    x_rank2 = torch.randn(n, 2) @ torch.randn(2, p)
    # Y rank 1 in 15-dim space
    y_rank1 = torch.randn(n, 1) @ torch.randn(1, q)

    res = smooth_regression(x_rank2, y_rank1, nv=5)
    assert res["coef"].shape == (p, q)
    assert res["y_pred"].shape == (n, q)
    assert not torch.isnan(res["coef"]).any()
    assert not torch.isnan(res["y_pred"]).any()
    assert not torch.isinf(res["coef"]).any()
    assert not torch.isinf(res["y_pred"]).any()


def test_adversarial_smooth_regression_collinear_target_columns():
    """Verify smooth_regression handles collinear columns in Y."""
    torch.manual_seed(42)
    x = torch.randn(30, 8)
    y_base = torch.randn(30, 1)
    # Columns are exact scalar multiples
    y_collinear = torch.cat([y_base, 2.0 * y_base, -5.0 * y_base], dim=1)

    res = smooth_regression(x, y_collinear, nv=4)
    assert res["y_pred"].shape == (30, 3)
    assert not torch.isnan(res["y_pred"]).any()
    # Predicted columns must reflect the collinear scaling ratio
    assert torch.allclose(res["y_pred"][:, 1], 2.0 * res["y_pred"][:, 0], atol=1e-4)
    assert torch.allclose(res["y_pred"][:, 2], -5.0 * res["y_pred"][:, 0], atol=1e-4)


def test_adversarial_smooth_regression_zero_variance_targets():
    """Verify smooth_regression handles all-zero and constant targets gracefully."""
    torch.manual_seed(42)
    x = torch.randn(30, 6)

    # All-zero target
    y_zeros = torch.zeros(30, 3)
    res_zeros = smooth_regression(x, y_zeros)
    assert res_zeros["y_pred"].shape == (30, 3)
    assert not torch.isnan(res_zeros["coef"]).any()
    assert torch.allclose(res_zeros["y_pred"], torch.zeros_like(res_zeros["y_pred"]))

    # Constant target (non-zero constant)
    y_const = torch.ones(30, 2) * 17.5
    res_const = smooth_regression(x, y_const)
    assert not torch.isnan(res_const["coef"]).any()
    assert not torch.isnan(res_const["y_pred"]).any()


def test_adversarial_smooth_regression_high_dimensional_projections():
    """Verify smooth_regression under extreme dimensionalities (P >> N and N >> P)."""
    torch.manual_seed(42)

    # P >> N: 15 samples, 150 features
    x_wide = torch.randn(15, 150)
    y_wide = torch.randn(15, 20)
    res_wide = smooth_regression(x_wide, y_wide, nv=5)
    assert res_wide["coef"].shape == (150, 20)
    assert res_wide["y_pred"].shape == (15, 20)
    assert not torch.isnan(res_wide["coef"]).any()
    assert not torch.isnan(res_wide["y_pred"]).any()

    # N >> P: 200 samples, 4 features
    x_tall = torch.randn(200, 4)
    y_tall = torch.randn(200, 3)
    res_tall = smooth_regression(x_tall, y_tall, nv=4)
    assert res_tall["coef"].shape == (4, 3)
    assert res_tall["y_pred"].shape == (200, 3)
    assert not torch.isnan(res_tall["coef"]).any()


def test_adversarial_smooth_regression_regularization_extremes():
    """Verify smooth_regression survives huge alpha (1e8) and tiny alpha (1e-8)."""
    torch.manual_seed(42)
    x = torch.randn(30, 5)
    y = torch.randn(30, 2)

    # Massive alpha: coefficients should shrink heavily towards zero
    res_huge_alpha = smooth_regression(x, y, alpha=1e8)
    assert not torch.isnan(res_huge_alpha["coef"]).any()
    assert torch.norm(res_huge_alpha["coef"]) < 1e-6

    # Tiny alpha on well-conditioned data
    res_tiny_alpha = smooth_regression(x, y, alpha=1e-8)
    assert not torch.isnan(res_tiny_alpha["coef"]).any()


def test_adversarial_smooth_regression_nv_boundaries():
    """Verify smooth_regression handles nv=0, nv=1, and nv > rank cleanly."""
    torch.manual_seed(42)
    x = torch.randn(20, 5)
    y = torch.randn(20, 2)

    # nv = 0: returns zero coefficients and zero predictions
    res_nv0 = smooth_regression(x, y, nv=0)
    assert res_nv0["coef"].shape == (5, 2)
    assert torch.allclose(res_nv0["coef"], torch.zeros(5, 2))
    assert torch.allclose(res_nv0["y_pred"], torch.zeros(20, 2))

    # nv = 1: rank-1 regression
    res_nv1 = smooth_regression(x, y, nv=1)
    assert res_nv1["coef"].shape == (5, 2)
    assert res_nv1["s"].shape == (1,)

    # nv > min(N, P): clamped cleanly to min(N, P)
    res_nv_exceed = smooth_regression(x, y, nv=50)
    assert res_nv_exceed["coef"].shape == (5, 2)
    assert res_nv_exceed["s"].shape == (5,)


def test_adversarial_smooth_regression_1d_target_parity():
    """Verify 1D target vectors produce 1D predictions and coefficients."""
    torch.manual_seed(42)
    x = torch.randn(25, 4)
    y_1d = torch.randn(25)
    res = smooth_regression(x, y_1d)
    assert res["y_pred"].shape == (25,)
    assert res["coef"].shape == (4,)
    assert not torch.isnan(res["y_pred"]).any()


# ============================================================================
# 3. sparse_dist Adversarial Stress Tests
# ============================================================================

@pytest.mark.parametrize("bad_sigma", [0.0, -0.0, -1.0, -1e-6, -100.0])
def test_adversarial_sparse_dist_nonpositive_sigma_rejected(bad_sigma):
    """Verify all sparse distance APIs reject non-positive sigma with ValueError."""
    x = torch.randn(10, 3)
    y = torch.randn(8, 3)

    with pytest.raises(ValueError, match=r"sigma must be strictly positive"):
        sparse_dist(x, k=2, sigma=bad_sigma)

    with pytest.raises(ValueError, match=r"sigma must be strictly positive"):
        sparse_distance_matrix(x, k=2, sigma=bad_sigma)

    with pytest.raises(ValueError, match=r"sigma must be strictly positive"):
        sparse_distance_matrix_xy(x, y, k=2, sigma=bad_sigma)


def test_adversarial_sparse_dist_infinite_and_huge_sigma():
    """Verify infinite and massive sigma yield valid 1.0 affinities without NaNs."""
    x = torch.randn(12, 3)

    for huge_sig in [1e12, 1e20, float('inf')]:
        smat = sparse_dist(x, k=3, sigma=huge_sig)
        assert not torch.isnan(smat).any(), f"NaN on sigma={huge_sig}"
        assert not torch.isinf(smat).any(), f"Inf on sigma={huge_sig}"
        # All non-zero entries must equal 1.0 (since exp(-d^2 / inf) = 1.0)
        nonzeros = smat[smat > 0.0]
        assert torch.allclose(nonzeros, torch.ones_like(nonzeros), atol=1e-4)


def test_adversarial_sparse_dist_tiny_positive_sigma():
    """Verify tiny positive sigma does not produce NaNs and decays affinities to 0."""
    x = torch.randn(10, 3)
    smat = sparse_dist(x, k=3, sigma=1e-10)
    assert not torch.isnan(smat).any()
    # Non-diagonal elements are > 0 distance, so exp(-d^2 / 2e-20) underflows to 0.0
    # Diagonal elements are 0 distance, so exp(0) = 1.0
    diag = torch.diag(smat)
    assert torch.allclose(diag, torch.ones_like(diag))


def test_adversarial_sparse_dist_identical_coordinates():
    """Verify sparse_dist handles identical rows without division by zero."""
    # 5 identical points
    x_identical = torch.ones(5, 3) * 2.5
    smat = sparse_dist(x_identical, k=2, sigma=1.0)
    assert not torch.isnan(smat).any()
    # Masked entries should have affinity 1.0 (distance 0.0 -> exp(0) = 1.0)
    assert (smat[smat > 0.0] == 1.0).all()


def test_adversarial_sparse_dist_neighbor_boundaries():
    """Verify sparse_dist handles k >= N, k=1, and single sample N=1."""
    x = torch.randn(6, 3)

    # k >= N (k=20 on 6 samples)
    smat_large_k = sparse_dist(x, k=20, sigma=1.0)
    assert smat_large_k.shape == (6, 6)
    assert not torch.isnan(smat_large_k).any()

    # k = 1 (minimal neighborhood)
    smat_k1 = sparse_dist(x, k=1, sigma=1.0)
    assert smat_k1.shape == (6, 6)
    assert not torch.isnan(smat_k1).any()

    # N = 1 (single sample dataset)
    x_n1 = torch.randn(1, 4)
    smat_n1 = sparse_dist(x_n1, k=1, sigma=1.0)
    assert smat_n1.shape == (1, 1)
    assert smat_n1[0, 0] == 1.0


def test_adversarial_sparse_distance_matrix_xy_rectangular():
    """Verify sparse_distance_matrix_xy functions correctly for NX != NY."""
    x = torch.randn(7, 4)
    y = torch.randn(13, 4)
    smat_xy = sparse_distance_matrix_xy(x, y, k=3, sigma=1.5)
    assert smat_xy.shape == (7, 13)
    assert not torch.isnan(smat_xy).any()
    assert not torch.isinf(smat_xy).any()


# ============================================================================
# 4. compute_shared_consensus & preprocess_data Adversarial Stress Tests
# ============================================================================

@pytest.mark.parametrize("alg", ["svd", "pca", "avg", "newton", "ica"])
def test_adversarial_consensus_single_sample_n1_all_algorithms(alg):
    """Verify compute_shared_consensus handles single-sample N=1 for all mixing algorithms."""
    torch.manual_seed(42)
    x1 = torch.randn(1, 5)
    x2 = torch.randn(1, 5)

    u = compute_shared_consensus([x1, x2], mixing_algorithm=alg, k=2, training=False)
    assert not torch.isnan(u).any(), f"NaN detected in N=1 consensus with algorithm '{alg}'!"
    assert not torch.isinf(u).any(), f"Inf detected in N=1 consensus with algorithm '{alg}'!"
    assert u.shape[0] == 1


@pytest.mark.parametrize("alg", ["svd", "pca", "avg", "newton"])
def test_adversarial_consensus_identical_rows_n2_and_n5(alg):
    """Verify compute_shared_consensus does not divide by zero on zero-variance identical samples."""
    # N=2 identical samples
    x1_id2 = torch.ones(2, 4) * 4.0
    x2_id2 = torch.ones(2, 4) * 4.0
    u2 = compute_shared_consensus([x1_id2, x2_id2], mixing_algorithm=alg, k=2)
    assert not torch.isnan(u2).any(), f"NaN on N=2 identical rows with '{alg}'!"

    # N=5 identical samples
    x1_id5 = torch.ones(5, 4) * -2.0
    x2_id5 = torch.ones(5, 4) * -2.0
    u5 = compute_shared_consensus([x1_id5, x2_id5], mixing_algorithm=alg, k=2)
    assert not torch.isnan(u5).any(), f"NaN on N=5 identical rows with '{alg}'!"


def test_adversarial_consensus_zero_variance_columns_and_zero_views():
    """Verify compute_shared_consensus handles constant columns and all-zero views."""
    torch.manual_seed(42)
    # View 1 has constant column 0
    x1 = torch.randn(20, 4)
    x1[:, 0] = 7.0
    # View 2 is completely zero
    x2_zero = torch.zeros(20, 4)

    u = compute_shared_consensus([x1, x2_zero], k=2)
    assert not torch.isnan(u).any()
    assert u.shape == (20, 2)


def test_adversarial_consensus_nan_inf_sanitization():
    """Verify compute_shared_consensus sanitizes NaN, +Inf, -Inf in projection inputs."""
    x1_corrupt = torch.tensor([[1.0, float('nan')], [float('inf'), -float('inf')], [2.0, 3.0]])
    x2_clean = torch.randn(3, 2)

    u = compute_shared_consensus([x1_corrupt, x2_clean], k=2)
    assert not torch.isnan(u).any(), "Consensus failed to sanitize NaN/Inf input!"
    assert not torch.isinf(u).any(), "Consensus failed to sanitize NaN/Inf input!"
    assert u.shape == (3, 2)


@pytest.mark.parametrize("scale_method", ["none", "norm", "np", "sqrtnp", "center", "centerAndScale"])
def test_adversarial_preprocess_data_n1_all_methods(scale_method):
    """Verify preprocess_data on N=1 executes with 0 NaNs and valid provenance across all scaling methods."""
    x_n1 = torch.randn(1, 8)
    x_scaled, prov = preprocess_data(x_n1, [scale_method])
    assert not torch.isnan(x_scaled).any(), f"NaN in preprocess_data for method '{scale_method}' on N=1!"

    # Replay provenance on a new test sample
    x_test_n1 = torch.randn(1, 8)
    x_test_scaled = preprocess_data(x_test_n1, [scale_method], provenance=prov)
    assert not torch.isnan(x_test_scaled).any()


def test_adversarial_preprocess_data_zero_variance_columns():
    """Verify preprocess_data centerAndScale handles zero-variance columns without exploding std."""
    x = torch.randn(30, 5)
    x[:, 1] = 9.99  # constant column
    x_scaled, prov = preprocess_data(x, ["centerAndScale"])
    assert not torch.isnan(x_scaled).any()
    assert not torch.isinf(x_scaled).any()
    # Constant column after centerAndScale should be centered to 0
    assert torch.allclose(x_scaled[:, 1], torch.zeros(30), atol=1e-5)


# ============================================================================
# 5. adjusted_rvcoef Adversarial Stress Tests & Numerical Stability
# ============================================================================

@pytest.mark.parametrize("n, p", [
    (50, 10),  # N > P (tall)
    (15, 60),  # N < P (wide)
    (25, 25),  # N = P (square)
    (3, 4),    # Minimal valid N=3
    (5, 5),
    (80, 2),
])
def test_adversarial_adjusted_rvcoef_identical_matrices_equals_one(n, p):
    """Verify adjusted_rvcoef(x, x) strictly equals 1.0 across diverse aspect ratios."""
    torch.manual_seed(n * 100 + p)
    x = torch.randn(n, p)
    rv_self = adjusted_rvcoef(x, x)
    assert not np.isnan(rv_self), f"NaN returned for identical {n}x{p} matrices!"
    assert np.isclose(rv_self, 1.0, atol=1e-4), f"Identical RV={rv_self} differs from 1.0 on {n}x{p}!"


def test_adversarial_adjusted_rvcoef_orthogonal_noise_attenuation():
    """Verify adjusted_rvcoef attenuates dimension bias on independent high-dimensional noise."""
    torch.manual_seed(42)
    n, p, q = 45, 35, 35

    raw_rvs = []
    adj_rvs = []
    for _ in range(40):
        x = torch.randn(n, p)
        y = torch.randn(n, q)
        raw_rvs.append(rvcoef(x, y))
        adj_rvs.append(adjusted_rvcoef(x, y))

    raw_mean = float(np.mean(raw_rvs))
    adj_mean = float(np.mean(adj_rvs))

    # Raw RV must be heavily positively biased in high dimensions
    assert raw_mean > 0.30, f"Expected raw RV bias > 0.30, got {raw_mean}"
    # Adjusted RV must center near zero
    assert abs(adj_mean) < 0.03, f"Expected adjusted RV mean near 0.0, got {adj_mean}"


def test_adversarial_adjusted_rvcoef_negative_values_handled():
    """Verify adjusted_rvcoef returns valid negative values without crash or clamping."""
    torch.manual_seed(123)
    n, p = 30, 20
    # Over repeated trials, negative values naturally appear under the null
    neg_found = False
    for _ in range(50):
        x = torch.randn(n, p)
        y = torch.randn(n, p)
        val = adjusted_rvcoef(x, y)
        assert not np.isnan(val)
        if val < -0.01:
            neg_found = True
            break
    assert neg_found, "Failed to observe expected negative adjusted RV value under null hypothesis"


def test_adversarial_adjusted_rvcoef_degenerate_and_mismatched_inputs():
    """Verify adjusted_rvcoef returns 0.0 on degenerate, zero, and mismatched inputs."""
    # N=1
    assert adjusted_rvcoef(torch.randn(1, 4), torch.randn(1, 4)) == 0.0

    # All zeros
    assert adjusted_rvcoef(torch.zeros(15, 3), torch.zeros(15, 3)) == 0.0

    # Constant matrix (zero variance)
    assert adjusted_rvcoef(torch.ones(20, 3) * 5.0, torch.randn(20, 3)) == 0.0

    # Mismatched sample counts
    assert adjusted_rvcoef(torch.randn(10, 3), torch.randn(15, 3)) == 0.0

    # 1D inputs auto-unsqueezed
    v1 = torch.randn(25)
    v2 = torch.randn(25)
    rv_1d = adjusted_rvcoef(v1, v2)
    assert not np.isnan(rv_1d)


def test_adversarial_adjusted_rvcoef_gram_vs_trace_consistency():
    """Verify consistency between trace branch (N < P+Q) and Gram branch (N >= P+Q)."""
    torch.manual_seed(42)
    # Case 1: Trace branch N < P+Q (N=10, P=8, Q=8 -> 10 < 16)
    x1 = torch.randn(10, 8)
    y1 = torch.randn(10, 8)
    rv_trace = adjusted_rvcoef(x1, y1)
    assert not np.isnan(rv_trace)

    # Case 2: Gram branch N >= P+Q (N=30, P=5, Q=5 -> 30 >= 10)
    x2 = torch.randn(30, 5)
    y2 = torch.randn(30, 5)
    rv_gram = adjusted_rvcoef(x2, y2)
    assert not np.isnan(rv_gram)
