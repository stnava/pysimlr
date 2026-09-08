"""
Unit tests for pysimlr.consensus: compute_shared_consensus with topology="graph".
Addresses Feature 16 in PROJECT.md (Milestone 3).
"""
import pytest
import torch
import numpy as np
from pysimlr.consensus import compute_shared_consensus


def test_consensus_graph_missing_path_graph_contract():
    """Verify compute_shared_consensus raises descriptive ValueError if path_graph is None with topology='graph'."""
    projections = [torch.randn(20, 2), torch.randn(20, 2)]
    with pytest.raises(ValueError, match="path_graph must be provided for topology='graph'"):
        compute_shared_consensus(projections, topology="graph", path_graph=None)


def test_consensus_graph_mathematical_neighbor_alignment():
    """Verify each modality aligns specifically to its specified graph neighbors under 'avg' mixing."""
    torch.manual_seed(42)
    n, k = 30, 2
    p0 = torch.randn(n, k)
    p1 = torch.randn(n, k)
    p2 = torch.randn(n, k)
    projections = [p0, p1, p2]

    # Graph: 0 <-> 1, node 2 is isolated
    path_graph = {0: [1], 1: [0], 2: []}

    u_list = compute_shared_consensus(
        projections,
        topology="graph",
        path_graph=path_graph,
        mixing_algorithm="avg",
        training=False,
    )

    assert len(u_list) == 3

    # Helper to compute expected standardized projection
    def std_proj(p):
        p_safe = torch.nan_to_num(p, nan=0.0)
        p_rms = torch.norm(p_safe, p="fro") / np.sqrt(p_safe.shape[0])
        p_norm = p_safe / (p_rms + 1e-8)
        p_c = p_norm - p_norm.mean(0, keepdim=True)
        u_std = torch.std(p_c, dim=0, keepdim=True)
        u_std = torch.where(torch.isnan(u_std) | (u_std < 1e-6), torch.ones_like(u_std), u_std)
        return p_c / u_std

    # Modality 0 aligns to neighbor 1
    expected_u0 = std_proj(p1)
    # Modality 1 aligns to neighbor 0
    expected_u1 = std_proj(p0)
    # Modality 2 has no neighbors -> falls back to orig_norm_projs[2]
    expected_u2 = torch.nan_to_num(p2, nan=0.0) / (torch.norm(p2, p="fro") / np.sqrt(n) + 1e-8)

    torch.testing.assert_close(u_list[0], expected_u0, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(u_list[1], expected_u1, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(u_list[2], expected_u2, atol=1e-4, rtol=1e-4)


def test_consensus_graph_multi_neighbor_average():
    """Verify a node connected to multiple neighbors computes the consensus of all its neighbors."""
    torch.manual_seed(42)
    n, k = 25, 2
    projections = [torch.randn(n, k) for _ in range(3)]

    # Star graph with center 0: 0 connected to 1 and 2; 1 and 2 only connected to 0
    path_graph = {0: [1, 2], 1: [0], 2: [0]}

    u_list = compute_shared_consensus(
        projections,
        topology="graph",
        path_graph=path_graph,
        mixing_algorithm="avg",
        training=False,
    )

    assert len(u_list) == 3
    for u in u_list:
        assert u.shape == (n, k)
        assert not torch.isnan(u).any()


@pytest.mark.parametrize("algorithm", ["svd", "pca", "avg", "newton", "ica"])
def test_consensus_graph_all_mixing_algorithms(algorithm):
    """Verify all 5 mixing algorithms operate cleanly under topology='graph'."""
    torch.manual_seed(42)
    n, k = 35, 2
    projections = [torch.randn(n, k) for _ in range(3)]
    path_graph = {0: [1], 1: [0, 2], 2: [1]}

    u_list = compute_shared_consensus(
        projections,
        topology="graph",
        path_graph=path_graph,
        mixing_algorithm=algorithm,
        training=False,
    )

    assert len(u_list) == 3
    for u in u_list:
        assert u.shape == (n, k)
        assert not torch.isnan(u).any()
        assert not torch.isinf(u).any()


def test_consensus_graph_training_anchor_generation():
    """Verify training=True under graph topology returns both consensus latents and anchor matrix."""
    torch.manual_seed(42)
    n, k = 30, 2
    projections = [torch.randn(n, 3), torch.randn(n, 4)]
    path_graph = {0: [1], 1: [0]}

    res = compute_shared_consensus(
        projections,
        topology="graph",
        path_graph=path_graph,
        k=k,
        training=True,
    )

    assert isinstance(res, tuple)
    u_list, anchor = res
    assert isinstance(u_list, list)
    assert len(u_list) == 2
    assert anchor is not None
    # Anchor shape should be (total_dim, k) -> (3 + 4, 2) = (7, 2)
    assert anchor.shape == (7, 2)


def test_consensus_graph_with_prune_threshold():
    """Verify prune_threshold filters invalid/low-correlation neighbors under graph topology."""
    torch.manual_seed(42)
    n, k = 40, 2
    p0 = torch.randn(n, k)
    p1 = torch.randn(n, k)
    # p2 has extreme scale / unrelated noise
    p2 = torch.randn(n, k) * 50.0
    projections = [p0, p1, p2]

    # Graph where 0 connects to 1 and 2
    path_graph = {0: [1, 2], 1: [0], 2: [0]}

    u_list = compute_shared_consensus(
        projections,
        topology="graph",
        path_graph=path_graph,
        prune_threshold=0.3,
        training=False,
    )

    assert len(u_list) == 3
    for u in u_list:
        assert u.shape == (n, k)
        assert not torch.isnan(u).any()


def test_consensus_graph_with_modality_weights():
    """Verify modality_weights properly weight neighbors under graph topology."""
    torch.manual_seed(42)
    n, k = 30, 2
    projections = [torch.randn(n, k) for _ in range(3)]
    path_graph = {0: [1, 2], 1: [0], 2: [0]}
    weights = torch.tensor([1.0, 3.0, 0.5])

    u_list = compute_shared_consensus(
        projections,
        topology="graph",
        path_graph=path_graph,
        modality_weights=weights,
        training=False,
    )

    assert len(u_list) == 3
    for u in u_list:
        assert u.shape == (n, k)
        assert not torch.isnan(u).any()


def test_consensus_graph_asymmetric_directed():
    """Verify asymmetric path graph aligns nodes according to directed neighborhood."""
    torch.manual_seed(42)
    n, k = 20, 2
    projections = [torch.randn(n, k), torch.randn(n, k)]
    # Node 0 aligns to 1, but Node 1 has no neighbors
    path_graph = {0: [1], 1: []}

    u_list = compute_shared_consensus(
        projections,
        topology="graph",
        path_graph=path_graph,
        training=False,
    )

    assert len(u_list) == 2
    # Node 1 should retain its own projection
    expected_u1 = torch.nan_to_num(projections[1], nan=0.0) / (torch.norm(projections[1], p="fro") / np.sqrt(n) + 1e-8)
    torch.testing.assert_close(u_list[1], expected_u1, atol=1e-4, rtol=1e-4)
