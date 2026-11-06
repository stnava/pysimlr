"""
Unit tests for pysimlr.structural.create_path_graph and the graph comparison that replaced
fit_structural_models: `pysimlr.compare_path_models` (plan 2026-11-06, section 4.2).
"""
import numpy as np
import pytest
import torch

from pysimlr.sem import compare_path_models
from pysimlr.structural import create_path_graph


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


_DEEP = {"iterations": 4, "batch_size": 15, "warmup_epochs": 0, "hidden_dims": [32, 16], "verbose": False}


def _chain_views(n=120, seed=0):
    from pysimlr.benchmarks.synthetic_cases import build_path_graph_case
    return build_path_graph_case([(0, 1), (1, 2)], n, 2, (8, 10, 6), 0.5, random_state=seed)["views"]


@pytest.mark.parametrize("method", ["lend", "ned"])
def test_compare_path_models_refits_each_topology(method):
    """Each graph is refitted (topology='graph', path_graph=its edges) and scored on held-out rows."""
    graphs = {"chain": create_path_graph([(0, 1), (1, 2)], 3), "star": [(0, 1), (0, 2)], "independent": []}
    out = compare_path_models(method=method, k=2, views=_chain_views(), graphs=graphs, n_splits=2,
                              fit_params=_DEEP, n_permutations=19)
    sp = out["splits"]
    assert len(sp) == 6 and list(sp.graph[:3]) == ["chain", "star", "independent"]
    assert list(sp.topology[:3]) == ["graph", "graph", "star"]
    assert list(out["summary"].graph) == ["chain", "star", "independent"]
    assert "recon_loss" not in sp and "total_loss" not in sp      # training losses are not criteria
    assert np.isfinite(sp.heldout_loglik_own.to_numpy(float)).all()


def test_compare_path_models_accepts_numpy_and_confounds():
    g = np.random.default_rng(42)
    views = [g.standard_normal((90, 5)), g.standard_normal((90, 7))]
    C = g.standard_normal((90, 2))
    out = compare_path_models(method="simlr", k=2, views=views, graphs={"m1": [(0, 1)]}, n_splits=1,
                              fit_params={"iterations": 5, "confounds": C}, covariates=C, n_permutations=19)
    assert np.isfinite(out["splits"].oos_r2.iloc[0])


def test_compare_path_models_argument_errors():
    views = [torch.randn(30, 4), torch.randn(30, 4)]
    with pytest.raises(ValueError, match="graphs"):
        compare_path_models(method="simlr", k=2, views=views, graphs={})
    with pytest.raises(ValueError, match="method, k and views"):
        compare_path_models(graphs={"a": []})
    with pytest.raises(ValueError, match="topology"):
        compare_path_models(method="simlr", k=2, views=views, graphs={"a": []}, fit_params={"topology": "loo"})
