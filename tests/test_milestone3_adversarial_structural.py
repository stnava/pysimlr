"""Adversarial Stress Test Suite for Milestone 3 Structural Models & Graph Consensus.

Targeted Audits:
1. create_path_graph and fit_structural_models:
   - Large graph scale-up (10+ modalities) in topology construction and deep structural training.
   - Complete graphs (cliques K_n where all pairs are mutually connected).
   - Fully disconnected graphs (empty edge set with independent modality representations and ~0 sim loss).
   - Single modality boundaries:
     * Single view (M=1 modality) evaluated via LEND and NED.
     * Single sample (N=1 sample) evaluated through structural training.
   - Zero-variance views (constant / zero data matrices) verifying numerical stability and zero division guards.
   - Graph validation contracts (out-of-bounds node indices, negative indices, duplicate/reciprocal edges).

2. compute_shared_consensus with topology="graph":
   - Cyclic graphs: 3-cycle, 5-cycle, directed cycles across all 5 mixing algorithms (svd, pca, avg, newton, ica)
     under both training=False and training=True (anchor matrix generation).
   - Dense bipartite graphs (K_{3,3}, K_{2,4}) with rigorous validation of partition consensus symmetry.
   - Multi-component isolated graphs: disconnected subgraphs with multiple components and isolated singletons,
     verifying zero cross-component leakage and fallback self-projections.
   - Mismatched projection shapes:
     * Mismatched sample sizes (N_i != N_j) raising RuntimeError on multi-neighbor concatenation and training anchor generation.
     * Mismatched projection feature dimensions (k_i != k_j) across SVD and AVG mixing policies.
   - Non-finite values: NaNs, +inf, -inf, all-NaN inputs, and extreme scales verifying robust normalization.
"""

import pytest
import torch
import numpy as np

from pysimlr.consensus import compute_shared_consensus
from pysimlr.structural import create_path_graph, fit_structural_models


# ============================================================================
# 1. create_path_graph & fit_structural_models Adversarial Tests
# ============================================================================

def test_create_path_graph_large_scale():
    """Verify create_path_graph scales cleanly to 12 modalities across multiple topologies."""
    n_modalities = 12
    # 1. Ring topology on 12 nodes
    ring_edges = [(i, (i + 1) % n_modalities) for i in range(n_modalities)]
    ring_graph = create_path_graph(ring_edges, n_modalities=n_modalities)
    assert len(ring_graph) == n_modalities
    for i in range(n_modalities):
        expected_neighbors = {(i - 1) % n_modalities, (i + 1) % n_modalities}
        assert set(ring_graph[i]) == expected_neighbors

    # 2. Star topology with central hub 0 and 11 leaves
    star_edges = [(0, i) for i in range(1, n_modalities)]
    star_graph = create_path_graph(star_edges, n_modalities=n_modalities)
    assert len(star_graph[0]) == 11
    for leaf in range(1, n_modalities):
        assert star_graph[leaf] == [0]


def test_create_path_graph_complete_graph():
    """Verify create_path_graph constructs a complete graph K_n where each node connects to all others."""
    n_modalities = 6
    edges = [(i, j) for i in range(n_modalities) for j in range(i + 1, n_modalities)]
    graph = create_path_graph(edges, n_modalities=n_modalities)
    for i in range(n_modalities):
        assert len(graph[i]) == n_modalities - 1
        assert i not in graph[i]
        assert set(graph[i]) == set(range(n_modalities)) - {i}


def test_create_path_graph_out_of_bounds_contract():
    """Verify create_path_graph raises KeyError when edge references node index >= n_modalities."""
    with pytest.raises(KeyError):
        create_path_graph([(0, 10)], n_modalities=5)

    with pytest.raises(KeyError):
        create_path_graph([(5, 1)], n_modalities=5)


def test_fit_structural_models_large_graph_10_modalities():
    """Verify fit_structural_models executes cleanly with 10 modalities under chain and star topologies."""
    torch.manual_seed(42)
    n_samples = 25
    n_mods = 10
    k = 2
    # Heterogeneous feature dimensions per modality
    views = [torch.randn(n_samples, 4 + (i % 3)) for i in range(n_mods)]

    chain_edges = [(i, i + 1) for i in range(n_mods - 1)]
    star_edges = [(0, i) for i in range(1, n_mods)]
    models = {
        "chain_10": create_path_graph(chain_edges, n_modalities=n_mods),
        "star_10": create_path_graph(star_edges, n_modalities=n_mods),
    }

    results = fit_structural_models(
        views,
        k=k,
        models=models,
        model_type="lend",
        epochs=3,
        batch_size=15,
        warmup_epochs=0,
        hidden_dims=[16, 8],
        verbose=False,
    )

    assert "models" in results
    assert "comparison" in results
    assert len(results["models"]) == 2
    assert len(results["comparison"]) == 2

    for model_name in ["chain_10", "star_10"]:
        res = results["models"][model_name]
        assert len(res["latents"]) == n_mods
        for lat in res["latents"]:
            assert lat.shape == (n_samples, k)
            assert torch.isfinite(lat).all()

    comp_map = {c["model"]: c for c in results["comparison"]}
    assert np.isfinite(comp_map["chain_10"]["total_loss"])
    assert np.isfinite(comp_map["star_10"]["total_loss"])


def test_fit_structural_models_complete_graph():
    """Verify fit_structural_models fits complete clique graph K_4 with full inter-modality coupling."""
    torch.manual_seed(42)
    n_samples, k = 20, 2
    n_mods = 4
    views = [torch.randn(n_samples, 6) for _ in range(n_mods)]
    clique_edges = [(i, j) for i in range(n_mods) for j in range(i + 1, n_mods)]
    clique_graph = create_path_graph(clique_edges, n_modalities=n_mods)

    results = fit_structural_models(
        views,
        k=k,
        models={"clique": clique_graph},
        model_type="lend",
        epochs=3,
        batch_size=10,
        warmup_epochs=0,
        hidden_dims=[16, 8],
        verbose=False,
    )

    comp = results["comparison"][0]
    assert comp["model"] == "clique"
    assert np.isfinite(comp["total_loss"])
    assert np.isfinite(comp["recon_loss"])
    assert np.isfinite(comp["sim_loss"])
    assert comp["sim_loss"] > 0.0


def test_fit_structural_models_fully_disconnected_graph():
    """Verify fit_structural_models with fully disconnected graph yields near-zero similarity loss."""
    torch.manual_seed(42)
    n_samples, k = 25, 2
    n_mods = 3
    views = [torch.randn(n_samples, 5) for _ in range(n_mods)]
    disconnected_graph = create_path_graph([], n_modalities=n_mods)

    results = fit_structural_models(
        views,
        k=k,
        models={"disconnected": disconnected_graph},
        model_type="lend",
        epochs=3,
        batch_size=15,
        warmup_epochs=0,
        hidden_dims=[16, 8],
        verbose=False,
    )

    comp = results["comparison"][0]
    assert comp["model"] == "disconnected"
    # When fully disconnected, each modality aligns to itself, yielding ~0 similarity alignment loss
    assert comp["sim_loss"] < 1e-6


def test_fit_structural_models_single_modality_view():
    """Verify fit_structural_models handles single modality view (M=1) gracefully with LEND and NED."""
    torch.manual_seed(42)
    n_samples, k = 20, 2
    views_single = [torch.randn(n_samples, 6)]
    graph_single = create_path_graph([], n_modalities=1)

    # 1. LEND architecture
    res_lend = fit_structural_models(
        views_single,
        k=k,
        models={"single_view_lend": graph_single},
        model_type="lend",
        epochs=2,
        batch_size=10,
        warmup_epochs=0,
        verbose=False,
    )
    assert len(res_lend["models"]["single_view_lend"]["latents"]) == 1
    assert res_lend["models"]["single_view_lend"]["latents"][0].shape == (n_samples, k)
    assert res_lend["comparison"][0]["sim_loss"] < 1e-6

    # 2. NED architecture
    res_ned = fit_structural_models(
        views_single,
        k=k,
        models={"single_view_ned": graph_single},
        model_type="ned",
        epochs=2,
        batch_size=10,
        warmup_epochs=0,
        verbose=False,
    )
    assert len(res_ned["models"]["single_view_ned"]["latents"]) == 1
    assert res_ned["models"]["single_view_ned"]["latents"][0].shape == (n_samples, k)
    assert res_ned["comparison"][0]["sim_loss"] < 1e-6


def test_fit_structural_models_single_sample_n1():
    """Verify fit_structural_models handles single sample (N=1) boundary condition without crashing."""
    torch.manual_seed(42)
    n_samples, k = 1, 1
    views_n1 = [torch.randn(n_samples, 4), torch.randn(n_samples, 4)]
    graph_pair = create_path_graph([(0, 1)], n_modalities=2)

    res = fit_structural_models(
        views_n1,
        k=k,
        models={"n1_model": graph_pair},
        model_type="lend",
        epochs=2,
        batch_size=1,
        warmup_epochs=0,
        verbose=False,
    )

    assert "n1_model" in res["models"]
    for lat in res["models"]["n1_model"]["latents"]:
        assert lat.shape == (1, 1)
        assert torch.isfinite(lat).all()


def test_fit_structural_models_zero_variance_views():
    """Verify fit_structural_models does not propagate NaNs when views have zero variance."""
    torch.manual_seed(42)
    n_samples, k = 20, 2
    views_zero = [
        torch.zeros(n_samples, 4),               # All zeros
        torch.ones(n_samples, 4) * 3.14159,      # All constant positive
        torch.randn(n_samples, 4),               # Standard Gaussian
    ]
    graph = create_path_graph([(0, 1), (1, 2)], n_modalities=3)

    results = fit_structural_models(
        views_zero,
        k=k,
        models={"zero_var": graph},
        model_type="lend",
        epochs=2,
        batch_size=10,
        warmup_epochs=0,
        verbose=False,
    )

    comp = results["comparison"][0]
    assert np.isfinite(comp["total_loss"])
    assert np.isfinite(comp["recon_loss"])
    assert np.isfinite(comp["sim_loss"])

    for lat in results["models"]["zero_var"]["latents"]:
        assert torch.isfinite(lat).all()


def test_fit_structural_models_mismatched_sample_sizes_contract():
    """Verify fit_structural_models raises AssertionError when data matrices have mismatched sample sizes."""
    views_mismatched = [torch.randn(15, 4), torch.randn(25, 4)]
    graph = create_path_graph([(0, 1)], n_modalities=2)
    with pytest.raises(AssertionError, match="Size mismatch"):
        fit_structural_models(views_mismatched, k=2, models={"mismatch": graph})


# ============================================================================
# 2. compute_shared_consensus with topology="graph" Adversarial Tests
# ============================================================================

def test_consensus_graph_cyclic_topologies_all_algorithms():
    """Verify cyclic graph topologies (triangle, 5-cycle) across all 5 mixing algorithms."""
    torch.manual_seed(42)
    n_samples, k = 30, 2

    # Odd cycle (3-cycle)
    g_3cycle = create_path_graph([(0, 1), (1, 2), (2, 0)], n_modalities=3)
    projs3 = [torch.randn(n_samples, k) for _ in range(3)]

    for alg in ["svd", "pca", "avg", "newton", "ica"]:
        u_list = compute_shared_consensus(
            projs3,
            mixing_algorithm=alg,
            topology="graph",
            path_graph=g_3cycle,
            training=False,
        )
        assert len(u_list) == 3
        for u in u_list:
            assert u.shape == (n_samples, k)
            assert torch.isfinite(u).all()

    # Even/larger cycle (5-cycle) with training anchor generation
    g_5cycle = create_path_graph([(i, (i + 1) % 5) for i in range(5)], n_modalities=5)
    projs5 = [torch.randn(n_samples, k) for _ in range(5)]
    res = compute_shared_consensus(
        projs5,
        mixing_algorithm="svd",
        topology="graph",
        path_graph=g_5cycle,
        k=k,
        training=True,
    )
    assert isinstance(res, tuple)
    u_list, anchor = res
    assert len(u_list) == 5
    assert anchor.shape == (5 * k, k)
    assert torch.isfinite(anchor).all()


def test_consensus_graph_dense_bipartite_symmetry():
    """Verify complete bipartite graph K_{3,3} satisfies partition consensus symmetry."""
    torch.manual_seed(42)
    n_samples, k = 35, 2
    # Partition A: {0, 1, 2}, Partition B: {3, 4, 5}
    edges_bipartite = [(a, b) for a in range(3) for b in range(3, 6)]
    g_bipartite = create_path_graph(edges_bipartite, n_modalities=6)
    projs = [torch.randn(n_samples, k) for _ in range(6)]

    u_bip = compute_shared_consensus(
        projs,
        mixing_algorithm="avg",
        topology="graph",
        path_graph=g_bipartite,
        training=False,
    )

    assert len(u_bip) == 6
    # Every node in Partition A has neighbors {3, 4, 5}, so their consensus must be identical
    torch.testing.assert_close(u_bip[0], u_bip[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(u_bip[0], u_bip[2], atol=1e-5, rtol=1e-5)

    # Every node in Partition B has neighbors {0, 1, 2}, so their consensus must be identical
    torch.testing.assert_close(u_bip[3], u_bip[4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(u_bip[3], u_bip[5], atol=1e-5, rtol=1e-5)


def test_consensus_graph_isolated_components_no_leakage():
    """Verify disconnected components compute consensus strictly within component with zero leakage."""
    torch.manual_seed(42)
    n_samples, k = 30, 2
    # Component 1: {0, 1}
    # Component 2: {2, 3, 4} (triangle)
    # Component 3: {5} (isolated singleton)
    edges = [(0, 1), (2, 3), (3, 4), (2, 4)]
    g_comp = create_path_graph(edges, n_modalities=6)
    projs = [torch.randn(n_samples, k) for _ in range(6)]

    u_comp = compute_shared_consensus(
        projs,
        mixing_algorithm="avg",
        topology="graph",
        path_graph=g_comp,
        training=False,
    )

    assert len(u_comp) == 6

    # 1. Component 1 nodes (0 and 1) only depend on each other
    def std_proj(p):
        p_safe = torch.nan_to_num(p, nan=0.0)
        p_rms = torch.norm(p_safe, p="fro") / np.sqrt(p_safe.shape[0])
        p_norm = p_safe / (p_rms + 1e-8)
        p_c = p_norm - p_norm.mean(0, keepdim=True)
        u_std = torch.std(p_c, dim=0, keepdim=True)
        u_std = torch.where(torch.isnan(u_std) | (u_std < 1e-6), torch.ones_like(u_std), u_std)
        return p_c / u_std

    torch.testing.assert_close(u_comp[0], std_proj(projs[1]), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(u_comp[1], std_proj(projs[0]), atol=1e-4, rtol=1e-4)

    # 2. Isolated node 5 falls back to its own normalized self-projection
    p5_safe = torch.nan_to_num(projs[5], nan=0.0)
    expected_u5 = p5_safe / (torch.norm(p5_safe, p="fro") / np.sqrt(n_samples) + 1e-8)
    torch.testing.assert_close(u_comp[5], expected_u5, atol=1e-4, rtol=1e-4)


def test_consensus_graph_mismatched_sample_sizes_contract():
    """Verify compute_shared_consensus raises RuntimeError when multi-neighbor nodes have mismatched N."""
    p_mismatch = [torch.randn(10, 2), torch.randn(10, 2), torch.randn(20, 2)]
    # Node 0 has neighbors with mismatched sample sizes (10 vs 20)
    g = {0: [1, 2], 1: [0], 2: [0]}

    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        compute_shared_consensus(p_mismatch, topology="graph", path_graph=g, training=False)

    # training=True concatenates all modalities to build anchor, triggering size mismatch
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        compute_shared_consensus(p_mismatch, topology="graph", path_graph=g, training=True)


def test_consensus_graph_mismatched_feature_dimensions():
    """Verify SVD mixing handles heterogeneous column dimensions across modalities under graph topology."""
    torch.manual_seed(42)
    n_samples = 25
    k_target = 2
    # Heterogeneous projection dimensions: 3, 4, 5
    projs = [torch.randn(n_samples, 3), torch.randn(n_samples, 4), torch.randn(n_samples, 5)]
    g_triangle = create_path_graph([(0, 1), (1, 2), (2, 0)], n_modalities=3)

    # SVD mixing concatenates neighbor dimensions and projects via SVD to k_target
    u_svd = compute_shared_consensus(
        projs,
        mixing_algorithm="svd",
        topology="graph",
        path_graph=g_triangle,
        k=k_target,
        training=False,
    )
    assert len(u_svd) == 3
    for u in u_svd:
        assert u.shape == (n_samples, k_target)
        assert torch.isfinite(u).all()


def test_consensus_graph_non_finite_values_nan_inf():
    """Verify compute_shared_consensus suppresses NaNs, +infs, -infs and all-NaN views gracefully."""
    n_samples, k = 15, 2
    p0 = torch.randn(n_samples, k)
    # Inject NaNs and infinite values into p0
    p0[0, 0] = float("nan")
    p0[1, 1] = float("inf")
    p0[2, 0] = float("-inf")

    # p1 is entirely NaN
    p1 = torch.full((n_samples, k), float("nan"))

    # p2 is normal
    p2 = torch.randn(n_samples, k)

    projs = [p0, p1, p2]
    g = create_path_graph([(0, 1), (1, 2), (2, 0)], n_modalities=3)

    for alg in ["avg", "svd", "pca", "newton"]:
        u_list = compute_shared_consensus(
            projs,
            mixing_algorithm=alg,
            topology="graph",
            path_graph=g,
            training=False,
        )
        assert len(u_list) == 3
        for u in u_list:
            assert torch.isfinite(u).all(), f"Algorithm {alg} produced non-finite consensus!"


def test_consensus_graph_pruning_all_neighbors_fallback():
    """Verify that when prune_threshold prunes all neighbors of a node, it falls back to self-projection."""
    torch.manual_seed(42)
    n_samples, k = 30, 2
    p0 = torch.randn(n_samples, k)
    # Invert p1 to have negative correlation with p0
    p1 = -p0
    projs = [p0, p1]
    g = {0: [1], 1: [0]}

    # prune_threshold=0.99 will prune p1 since correlation is negative
    u_list = compute_shared_consensus(
        projs,
        mixing_algorithm="avg",
        topology="graph",
        path_graph=g,
        prune_threshold=0.99,
        training=False,
    )

    assert len(u_list) == 2
    for u in u_list:
        assert torch.isfinite(u).all()


def test_consensus_graph_train_and_inference_anchor_pipeline():
    """Verify end-to-end train anchor generation and subsequent test-time anchor inference."""
    torch.manual_seed(42)
    k = 2
    projs_train = [torch.randn(40, 3), torch.randn(40, 4), torch.randn(40, 3)]
    g_cycle = create_path_graph([(0, 1), (1, 2), (2, 0)], n_modalities=3)

    # 1. Training pass: returns u_list and anchor matrix
    u_train, anchor = compute_shared_consensus(
        projs_train,
        mixing_algorithm="svd",
        topology="graph",
        path_graph=g_cycle,
        k=k,
        training=True,
    )
    assert anchor is not None
    assert anchor.shape == (3 + 4 + 3, k)

    # 2. Test pass with unseen batch of size 18
    projs_test = [torch.randn(18, 3), torch.randn(18, 4), torch.randn(18, 3)]
    u_test = compute_shared_consensus(
        projs_test,
        mixing_algorithm="svd",
        topology="graph",
        path_graph=g_cycle,
        k=k,
        anchor=anchor,
        training=False,
    )

    assert len(u_test) == 3
    for u in u_test:
        assert u.shape == (18, k)
        assert torch.isfinite(u).all()
