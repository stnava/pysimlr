"""
Unit tests for pysimlr.structural: create_path_graph and fit_structural_models.
Addresses Feature 14 in PROJECT.md (Milestone 3).
"""
import pytest
import torch
import numpy as np
from pysimlr.structural import create_path_graph, fit_structural_models


def test_create_path_graph_topologies():
    """Verify create_path_graph generates correct undirected adjacency lists for various topologies."""
    # 1. Chain topology: 0 <-> 1 <-> 2
    chain = create_path_graph([(0, 1), (1, 2)], n_modalities=3)
    assert chain == {0: [1], 1: [0, 2], 2: [1]}

    # 2. Star topology: 0 center, 1, 2, 3 leaves
    star = create_path_graph([(0, 1), (0, 2), (0, 3)], n_modalities=4)
    assert star == {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}

    # 3. Cycle topology: 0 <-> 1 <-> 2 <-> 0
    cycle = create_path_graph([(0, 1), (1, 2), (2, 0)], n_modalities=3)
    assert set(cycle[0]) == {1, 2}
    assert set(cycle[1]) == {0, 2}
    assert set(cycle[2]) == {0, 1}

    # 4. Isolated nodes: 4 modalities, edge only between 0 and 1
    isolated = create_path_graph([(0, 1)], n_modalities=4)
    assert isolated == {0: [1], 1: [0], 2: [], 3: []}

    # 5. Empty edge list
    empty = create_path_graph([], n_modalities=3)
    assert empty == {0: [], 1: [], 2: []}


def test_create_path_graph_duplicates_and_self_loops():
    """Verify create_path_graph deduplicates identical and reversed edges, and handles self-loops."""
    # Duplicate and reciprocal edges
    graph = create_path_graph([(0, 1), (0, 1), (1, 0)], n_modalities=2)
    assert graph == {0: [1], 1: [0]}
    assert len(graph[0]) == 1
    assert len(graph[1]) == 1

    # Self-loop
    graph_self = create_path_graph([(0, 0)], n_modalities=2)
    assert graph_self == {0: [0], 1: []}


def test_fit_structural_models_lend():
    """Verify fit_structural_models evaluates and compares structural path hypotheses using LEND."""
    torch.manual_seed(42)
    n, k = 30, 2
    views = [torch.randn(n, 8), torch.randn(n, 10), torch.randn(n, 6)]

    models = {
        "chain": create_path_graph([(0, 1), (1, 2)], n_modalities=3),
        "star": create_path_graph([(0, 1), (0, 2)], n_modalities=3),
        "independent": create_path_graph([], n_modalities=3),
    }

    results = fit_structural_models(
        views,
        k=k,
        models=models,
        model_type="lend",
        epochs=4,
        batch_size=15,
        warmup_epochs=0,
        hidden_dims=[32, 16],
        verbose=False,
    )

    assert "models" in results
    assert "comparison" in results
    assert len(results["models"]) == 3
    assert len(results["comparison"]) == 3

    comp_by_name = {c["model"]: c for c in results["comparison"]}
    for name in ["chain", "star", "independent"]:
        assert name in comp_by_name
        model_res = results["models"][name]
        assert model_res["model_type"] == "lend_simr"
        assert len(model_res["latents"]) == 3
        for lat in model_res["latents"]:
            assert lat.shape == (n, k)

        comp = comp_by_name[name]
        assert "total_loss" in comp
        assert "recon_loss" in comp
        assert "sim_loss" in comp
        assert "converged_iter" in comp
        # Check that comparison metrics match the mean of the last 5 epochs of the histories
        expected_recon = np.mean(model_res["recon_history"][-5:])
        expected_sim = np.mean(model_res["sim_history"][-5:])
        expected_total = np.mean(model_res["loss_history"][-5:])
        np.testing.assert_allclose(comp["recon_loss"], expected_recon, rtol=1e-5)
        np.testing.assert_allclose(comp["sim_loss"], expected_sim, rtol=1e-5)
        np.testing.assert_allclose(comp["total_loss"], expected_total, rtol=1e-5)


def test_fit_structural_models_ned():
    """Verify fit_structural_models evaluates structural path hypotheses using NED."""
    torch.manual_seed(42)
    n, k = 25, 2
    views = [torch.randn(n, 6), torch.randn(n, 6)]

    models = {
        "connected": create_path_graph([(0, 1)], n_modalities=2),
        "disconnected": create_path_graph([], n_modalities=2),
    }

    results = fit_structural_models(
        views,
        k=k,
        models=models,
        model_type="ned",
        epochs=3,
        batch_size=15,
        warmup_epochs=0,
        hidden_dims=[16, 8],
        verbose=False,
    )

    assert len(results["comparison"]) == 2
    for name in ["connected", "disconnected"]:
        res = results["models"][name]
        assert res["model_type"] == "ned_simr"
        assert len(res["latents"]) == 2


def test_fit_structural_models_numpy_inputs():
    """Verify fit_structural_models accepts NumPy arrays and handles few epochs (< 5) cleanly."""
    np.random.seed(42)
    n, k = 25, 2
    views_np = [np.random.randn(n, 5), np.random.randn(n, 7)]

    models = {
        "m1": create_path_graph([(0, 1)], n_modalities=2)
    }

    results = fit_structural_models(
        views_np,
        k=k,
        models=models,
        model_type="lend",
        epochs=2,
        batch_size=15,
        warmup_epochs=0,
        verbose=False,
    )

    assert "m1" in results["models"]
    comp = results["comparison"][0]
    assert np.isfinite(comp["total_loss"])
    assert np.isfinite(comp["recon_loss"])
    assert np.isfinite(comp["sim_loss"])


def test_fit_structural_models_empty_dict():
    """Verify fit_structural_models handles empty models dictionary without error."""
    views = [torch.randn(10, 4), torch.randn(10, 4)]
    results = fit_structural_models(views, k=2, models={})
    assert results == {"models": {}, "comparison": []}
