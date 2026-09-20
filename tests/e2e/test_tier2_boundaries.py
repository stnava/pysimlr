"""Tier 2: Boundary and Corner Case Stress Tests for pysimlr public APIs.

Exercises:
- Empty inputs and zero-length modality collections.
- Mismatched sample sizes and 1D array contracts.
- Single sample N=1 numerical stability (Bessel correction guards, PCA, SVD).
- Rank-deficient data, singular matrices, repeated columns, constant zero-variance features.
- Collinear feature configurations.
- Gaussian kernel sigma=0.0 boundary behavior.
- Extreme dimensions (P >> N and N >> P).
- Latent dimension boundary limits (k=1 and k=min(N, P)).
- NaN and Inf data sanitization across SVD, PCA, and consensus.
- Graph topology extremes (disconnected graphs, cyclic graphs).
- Sparseness quantile extremes (0.0 dense to 0.85 sparse).
- Neural / flow hyperparameter boundaries (batch_size=N, dropout=0.0).
"""

import pytest
import torch
import numpy as np

from pysimlr import (
    simlr,
    lend_simr,
    flow_simr,
    compute_shared_consensus,
    ba_svd,
    safe_pca,
    whiten_matrix,
    multiscale_svd,
    sparse_distance_matrix,
    adjusted_rvcoef,
)


def test_boundary_empty_data_matrices_list():
    """Verify passing an empty list of data matrices raises a clear ValueError or error."""
    with pytest.raises((ValueError, IndexError)):
        simlr([], k=2)

    with pytest.raises((ValueError, IndexError)):
        lend_simr([], k=2, epochs=2)


def test_boundary_empty_consensus_tensor():
    """Verify compute_shared_consensus handles empty projection list gracefully."""
    res = compute_shared_consensus([])
    assert isinstance(res, torch.Tensor)
    assert res.numel() == 0


def test_boundary_mismatched_sample_counts_fast_fail():
    """Verify fast failure when modalities have differing sample sizes (N1 != N2)."""
    x1 = torch.randn(30, 8)
    x2 = torch.randn(45, 10)

    with pytest.raises((ValueError, RuntimeError)):
        simlr([x1, x2], k=2, iterations=5)


def test_boundary_1d_input_array_contract():
    """Verify fast failure when a 1D vector is provided instead of 2D matrix."""
    x1 = torch.randn(40)  # 1D array
    x2 = torch.randn(40, 6)

    with pytest.raises((ValueError, IndexError)):
        simlr([x1, x2], k=2, iterations=5)


def test_boundary_minimal_sample_n2_consensus_stability():
    """Verify minimal sample size N=2 does not crash or produce NaNs in consensus."""
    p1 = torch.randn(2, 4)
    p2 = torch.randn(2, 4)

    u = compute_shared_consensus([p1, p2], mixing_algorithm="svd", k=2, topology="star", training=False)
    assert u.shape == (2, 2)
    assert not torch.isnan(u).any(), "N=2 consensus produced NaNs"
    assert not torch.isinf(u).any(), "N=2 consensus produced Infs"


def test_boundary_single_sample_n1_ba_svd():
    """Verify ba_svd operates stably on N=1 single-sample row matrix."""
    x = torch.randn(1, 8)
    u, s, v = ba_svd(x, nu=1, nv=1)

    assert u.shape == (1, 1)
    assert s.shape == (1,)
    assert v.shape == (8, 1)
    assert not torch.isnan(u).any()
    assert not torch.isnan(s).any()


def test_boundary_single_sample_n1_safe_pca():
    """Verify safe_pca handles single-sample N=1 without crashing."""
    x = torch.randn(1, 6)
    pca = safe_pca(x, nc=1)

    assert pca["u"].shape == (1, 1)
    assert pca["v"].shape == (6, 1)
    assert not torch.isnan(pca["u"]).any()


def test_boundary_rank_deficient_all_zeros_view():
    """Verify safe_pca and ba_svd handle an all-zeros rank-0 matrix without crashing."""
    x_zero = torch.zeros(35, 10)
    pca = safe_pca(x_zero, nc=2)

    assert pca["u"].shape == (35, 2)
    assert pca["v"].shape == (10, 2)
    assert torch.all(pca["s"] == 0.0)


def test_boundary_rank_deficient_repeated_columns():
    """Verify safe_pca and ba_svd handle a rank-1 matrix with duplicate columns."""
    torch.manual_seed(42)
    col = torch.randn(40, 1)
    x_rank1 = col.repeat(1, 6)  # 6 identical columns

    pca = safe_pca(x_rank1, nc=3)
    assert pca["u"].shape == (40, 3)
    assert not torch.isnan(pca["u"]).any()
    # Only 1 positive singular value, rest zero
    assert pca["s"][0] > 0.0
    assert torch.all(pca["s"][1:] < 1e-4)


def test_boundary_collinear_features_handling():
    """Verify whiten_matrix handles collinear features without singular matrix crashes."""
    torch.manual_seed(42)
    x = torch.randn(50, 6)
    # Make column 1 an exact multiple of column 0
    x[:, 1] = x[:, 0] * 3.5

    res = whiten_matrix(x, nc=4)
    whitened = res["whitened_matrix"]
    assert whitened.shape == (50, 4)
    assert not torch.isnan(whitened).any()


def test_boundary_constant_feature_zero_variance():
    """Verify safe_pca masks zero-variance constant features smoothly."""
    torch.manual_seed(42)
    x = torch.randn(40, 5)
    # Column 2 has constant value 42.0 (zero variance)
    x[:, 2] = 42.0

    pca = safe_pca(x, nc=2)
    assert pca["u"].shape == (40, 2)
    assert not torch.isnan(pca["u"]).any()


def test_boundary_sparse_distance_matrix_sigma_zero():
    """Verify sparse_distance_matrix with sigma=0.0 is either safely bounded or raises."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)

    try:
        smat = sparse_distance_matrix(x, k=3, sigma=0.0)
        assert smat.shape == (20, 20)
    except (ValueError, ZeroDivisionError):
        # Graceful rejection of zero sigma is also acceptable
        pass


def test_boundary_extreme_high_dimensional_p_much_greater_than_n():
    """Verify SiMLR executes reliably in high-dimensional regime where P >> N (P=120, N=12)."""
    torch.manual_seed(42)
    n, p1, p2, k = 12, 120, 100, 2
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)

    res = simlr([x1, x2], k=k, iterations=8, optimizer_type="hybrid_adam")
    assert res["u"].shape == (n, k)
    assert res["v"][0].shape == (p1, k)
    assert res["v"][1].shape == (p2, k)
    assert not torch.isnan(res["u"]).any()


def test_boundary_extreme_tall_matrix_n_much_greater_than_p():
    """Verify SiMLR executes reliably in tall matrix regime where N >> P (N=200, P=4)."""
    torch.manual_seed(42)
    n, p1, p2, k = 200, 4, 5, 2
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)

    res = simlr([x1, x2], k=k, iterations=8, optimizer_type="hybrid_adam")
    assert res["u"].shape == (n, k)
    assert res["v"][0].shape == (p1, k)
    assert res["v"][1].shape == (p2, k)
    assert not torch.isnan(res["u"]).any()


def test_boundary_latent_dim_k_equals_1_univariate():
    """Verify univariate latent space k=1 across SiMLR and Flow-SiMR."""
    torch.manual_seed(42)
    n, p1, p2 = 40, 8, 6
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)

    res_simlr = simlr([x1, x2], k=1, iterations=6)
    assert res_simlr["u"].shape == (n, 1)
    assert res_simlr["v"][0].shape == (p1, 1)

    res_flow = flow_simr([x1, x2], k=1, epochs=4, batch_size=20, verbose=False)
    assert res_flow["latents"][0].shape == (n, 1)


def test_boundary_latent_dim_k_equals_min_dim():
    """Verify maximum rank latent dimension k=min(N, P)."""
    torch.manual_seed(42)
    n, p1, p2 = 30, 5, 6
    k = 5  # min(p1, p2)
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)

    res = simlr([x1, x2], k=k, iterations=6)
    assert res["u"].shape == (n, k)
    assert res["v"][0].shape == (p1, k)


def test_boundary_nan_input_sanitization_ba_svd():
    """Verify ba_svd sanitizes NaN entries using internal safe conversion."""
    torch.manual_seed(42)
    x = torch.randn(40, 8)
    x[0, 0] = float('nan')
    x[5, 3] = float('nan')

    u, s, v = ba_svd(x, nu=3, nv=3)
    assert not torch.isnan(u).any(), "ba_svd leaked NaNs into U"
    assert not torch.isnan(s).any(), "ba_svd leaked NaNs into S"
    assert not torch.isnan(v).any(), "ba_svd leaked NaNs into V"


def test_boundary_inf_input_sanitization_safe_pca():
    """Verify safe_pca handles infinite values without producing NaNs/Infs."""
    torch.manual_seed(42)
    x = torch.randn(40, 8)
    x[2, 1] = float('inf')
    x[4, 2] = float('-inf')

    pca = safe_pca(x, nc=3)
    assert not torch.isnan(pca["u"]).any()
    assert not torch.isinf(pca["u"]).any()


def test_boundary_nan_projections_consensus_sanitization():
    """Verify compute_shared_consensus replaces NaN projections without failure."""
    torch.manual_seed(42)
    p1 = torch.randn(30, 3)
    p2 = torch.randn(30, 3)
    p1[0, 0] = float('nan')

    u = compute_shared_consensus([p1, p2], k=3, topology="star", training=False)
    assert u.shape == (30, 3)
    assert not torch.isnan(u).any()


def test_boundary_unconnected_graph_topology():
    """Verify graph consensus with empty neighbor sets returns self-projections."""
    torch.manual_seed(42)
    p1 = torch.randn(25, 2)
    p2 = torch.randn(25, 2)
    empty_graph = {0: [], 1: []}

    u_list = compute_shared_consensus([p1, p2], topology="graph", path_graph=empty_graph, training=False)
    assert len(u_list) == 2
    assert u_list[0].shape == (25, 2)
    assert u_list[1].shape == (25, 2)


def test_boundary_cyclic_path_graph_topology():
    """Verify cyclic path graph (0 -> 1 -> 2 -> 0) computes consensus smoothly."""
    torch.manual_seed(42)
    projections = [torch.randn(30, 2) for _ in range(3)]
    cycle_graph = {0: [1], 1: [2], 2: [0]}

    u_list = compute_shared_consensus(projections, topology="graph", path_graph=cycle_graph, training=False)
    assert len(u_list) == 3
    for u in u_list:
        assert u.shape == (30, 2)
        assert not torch.isnan(u).any()


def test_boundary_sparseness_quantile_dense_zero():
    """Verify sparseness_quantile=0.0 runs dense optimization without thresholding."""
    torch.manual_seed(42)
    x1 = torch.randn(30, 6)
    x2 = torch.randn(30, 8)

    res = simlr([x1, x2], k=2, sparseness_quantile=0.0, iterations=6)
    assert res["u"] is not None
    # Weights should be predominantly non-zero
    assert torch.all(torch.abs(res["v"][0]) > 0.0)


def test_boundary_high_w_gives_high_sparsity():
    """A large NSA-Flow weight aggressively sparsifies the basis.

    Sparsity used to be a quantile threshold applied after the projection;
    it is now a property of the set the projection solves onto, so `w` is the
    knob. w -> 1 drives disjoint supports.
    """
    torch.manual_seed(42)
    x1 = torch.randn(40, 15)
    x2 = torch.randn(40, 12)

    res = simlr([x1, x2], k=2, constraint="orthox0.9", positivity="positive",
                iterations=10)
    v = res["v"][0]
    frac_zero = float((v.abs() < 1e-9).float().mean())
    assert frac_zero > 0.30, f"w=0.9 gave only {frac_zero:.2f} zeros"


def test_boundary_flow_batch_size_equals_sample_size():
    """Verify flow_simr with full-batch training (batch_size=N)."""
    torch.manual_seed(42)
    n = 30
    x1 = torch.randn(n, 6)
    x2 = torch.randn(n, 6)

    res = flow_simr([x1, x2], k=2, epochs=4, batch_size=n, verbose=False)
    assert res["latents"][0].shape == (n, 2)
    assert len(res["loss_history"]) == 4


def test_boundary_deep_zero_dropout():
    """Verify lend_simr functions properly with dropout=0.0 (deterministic graph)."""
    torch.manual_seed(42)
    x1 = torch.randn(35, 8)
    x2 = torch.randn(35, 6)

    res = lend_simr([x1, x2], k=2, epochs=4, batch_size=15, dropout=0.0, verbose=False)
    assert res["latents"][0].shape == (35, 2)
    assert not np.isnan(res["loss_history"][-1])


def test_boundary_all_discordant_projections_pruning_fallback():
    """Verify when prune_threshold is unrealistically strict (0.999), consensus does not collapse."""
    torch.manual_seed(42)
    # Two independent noise projections
    p1 = torch.randn(30, 2)
    p2 = torch.randn(30, 2)

    u = compute_shared_consensus([p1, p2], prune_threshold=0.999, training=False)
    assert u.shape == (30, 2)
    assert not torch.isnan(u).any()
