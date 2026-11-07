"""Tests for score-level path model plotting functions in pysimlr.plotting.

Verifies Figure return types, single-panel ax parameter honoring, solid edge
counts matching significant edges, sign coloring in component mode, graph-fit
rejection status line, moderation indicators, and marker dodging in comparison.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pysimlr
from pysimlr import plotting as P
from pysimlr.benchmarks.synthetic_cases import build_path_graph_case


@pytest.fixture(scope="module")
def quick_sem_case():
    """Fast fitted path model and model comparison for visualization tests."""
    case = build_path_graph_case([(0, 1), (1, 2)], 120, 2, (8, 10, 8),
                                 0.5, covariate_effect=0.8, random_state=0)
    views, C, y = case["views"], case["C"], case["outcome"]
    n_tr = 80
    tr = [v[:n_tr] for v in views]
    te = [v[n_tr:] for v in views]
    res = pysimlr.fit("simlr", tr, 2, random_state=0, confounds=C[:n_tr], iterations=10)
    graphs = {
        "chain + outcome": [(0, 1), (1, 2), (2, 3)],
        "star on view0": [(0, 1), (0, 2), (2, 3)],
        "empty": [],
    }
    pm = pysimlr.path_model(res, te, graph=graphs["chain + outcome"], covariates=C[n_tr:],
                            outcome=y[n_tr:], conditioning="parents", n_permutations=30,
                            random_state=0)
    cmp = pysimlr.compare_path_models(res, tr, te, graphs, covariates_train=C[:n_tr],
                                      covariates_test=C[n_tr:], outcome_train=y[:n_tr],
                                      outcome_test=y[n_tr:], n_bootstrap=40,
                                      n_permutations=20, random_state=0)
    return pm, cmp


def test_plot_path_model_returns_figure_and_honors_ax(quick_sem_case):
    pm, _ = quick_sem_case
    fig = P.plot_path_model(pm)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    fig_cust, ax_cust = plt.subplots(figsize=(6, 5))
    fig_ret = P.plot_path_model(pm, ax=ax_cust)
    assert fig_ret is fig_cust

    # Edge counts drawn match significant edges in the table
    alpha = 0.05
    sig_count = sum(bool(e["p_maxt"] is not None and e["p_maxt"] < alpha)
                    for e in pm["edges"].to_dict("records"))
    hyp_nonsig_count = sum(bool(e["present"] and not (e["p_maxt"] is not None and e["p_maxt"] < alpha))
                           for e in pm["edges"].to_dict("records"))

    solid_lines = [line for line in ax_cust.lines if line.get_linestyle() == "-"]
    dashed_lines = [line for line in ax_cust.lines if line.get_linestyle() != "-"]
    assert len(solid_lines) == sig_count
    assert len(dashed_lines) == hyp_nonsig_count
    plt.close(fig_cust)


def test_plot_path_model_component_coloring(quick_sem_case):
    pm, _ = quick_sem_case
    fig = P.plot_path_model(pm, component=0)
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]

    # Lines should use sign color: _BLUE[6] for positive, _RED[4] for negative
    from matplotlib.colors import to_hex
    for line in ax.lines:
        hex_c = to_hex(line.get_color())
        assert hex_c.lower() in (P._BLUE[6].lower(), P._RED[4].lower(), P._WARN.lower(), P._INK2.lower())
    plt.close(fig)


def test_plot_path_model_status_line_on_rejected_fit(quick_sem_case):
    pm, _ = quick_sem_case
    pm_rej = dict(pm)
    pm_rej["fit"] = dict(pm["fit"], rejected=True, max_abs_p=0.003, max_absent_pillai=0.45)

    fig = P.plot_path_model(pm_rej)
    all_text = " ".join(t.get_text() for t in fig.texts)
    assert "Graph-fit test rejected" in all_text
    plt.close(fig)


def test_plot_path_model_moderation_markers(quick_sem_case):
    pm, _ = quick_sem_case
    pm_mod = dict(pm)
    pm_mod["moderation"] = pd.DataFrame([
        {"a": "view0", "b": "view1", "component": "sum", "F": 4.5, "df1": 2, "df2": None,
         "p_perm": 0.001, "p_maxt": 0.002, "p_holm": 0.004, "q_bh": 0.004}
    ])

    fig = P.plot_path_model(pm_mod)
    ax = fig.axes[0]
    # Moderation marker line should be present
    markers = [line.get_marker() for line in ax.lines]
    assert "^" in markers
    # Legend handle should mention moderated edge
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert any("moderated edge" in t for t in legend_texts)
    plt.close(fig)


def test_plot_edge_matrix(quick_sem_case):
    pm, _ = quick_sem_case
    fig_r = P.plot_edge_matrix(pm, what="r")
    assert isinstance(fig_r, matplotlib.figure.Figure)
    plt.close(fig_r)

    fig_beta = P.plot_edge_matrix(pm, what="beta")
    assert isinstance(fig_beta, matplotlib.figure.Figure)
    plt.close(fig_beta)


def test_plot_path_comparison_and_dodge(quick_sem_case):
    _, cmp = quick_sem_case
    fig = P.plot_path_comparison(cmp)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    # Test that equal present_pass and absent_pass are dodged vertically
    cmp_equal = {
        "best": "graph_a",
        "comparison": [
            {"graph": "graph_a", "heldout_loglik": -10.0, "delta_vs_top": 0.0,
             "ci_low": -0.5, "ci_high": 0.5, "best_fraction": 0.8,
             "present_pass": 1.0, "absent_pass": 1.0, "n_edges": 3},
            {"graph": "graph_b", "heldout_loglik": -20.0, "delta_vs_top": -10.0,
             "ci_low": -12.0, "ci_high": -8.0, "best_fraction": 0.2,
             "present_pass": 0.5, "absent_pass": 0.2, "n_edges": 2},
        ]
    }
    fig_dodge = P.plot_path_comparison(cmp_equal)
    ax2 = fig_dodge.axes[1]
    # ax2 has 2 scatter collections: present (circles), absent (diamonds)
    pres_coll = ax2.collections[0]
    abs_coll = ax2.collections[1]
    pres_offsets = pres_coll.get_offsets()
    abs_offsets = abs_coll.get_offsets()

    # For graph_a (row 1, since sorted by heldout_loglik, -20 then -10),
    # graph_a has present_pass == absent_pass == 1.0. Its y coordinates must be dodged:
    # pres_y != abs_y
    idx_a = 1  # graph_a is at index 1
    assert pres_offsets[idx_a, 1] != abs_offsets[idx_a, 1]
    assert np.isclose(pres_offsets[idx_a, 1] - abs_offsets[idx_a, 1], 0.24, atol=1e-3)

    # For graph_b (row 0), present_pass != absent_pass, so y coordinates are not dodged
    idx_b = 0
    assert np.isclose(pres_offsets[idx_b, 1], 0.0)
    assert np.isclose(abs_offsets[idx_b, 1], 0.0)
    plt.close(fig_dodge)


def test_plot_edge_null_returns_figure_and_honors_ax(quick_sem_case):
    pm, _ = quick_sem_case
    fig = P.plot_edge_null(pm)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Check y-axis label is Permutation count
    assert fig.axes[0].get_ylabel() == "Permutation count"
    plt.close(fig)

    fig_cust, ax_cust = plt.subplots()
    fig_ret = P.plot_edge_null(pm, edges=[("view0", "view1")], ax=ax_cust)
    assert fig_ret is fig_cust
    assert ax_cust.get_ylabel() == "Permutation count"
    plt.close(fig_cust)


def test_plot_edge_features_returns_figure_and_honors_ax(quick_sem_case):
    pm, _ = quick_sem_case
    fig = P.plot_edge_features(pm, "view0 - view1")
    assert isinstance(fig, matplotlib.figure.Figure)
    # Check that stem colors use signed diverging colors (_BLUE[6] and _RED[4])
    from matplotlib.colors import to_hex
    ax0 = fig.axes[0]
    line_coll = next(c for c in ax0.collections if isinstance(c, matplotlib.collections.LineCollection))
    stem_colors = [to_hex(c).lower() for c in line_coll.get_colors()]
    assert len(stem_colors) > 0
    assert all(c in (P._BLUE[6].lower(), P._RED[4].lower()) for c in stem_colors)
    # Check endpoints view0 and view1 appear in figure header / xlabel
    header_text = " ".join(t.get_text() for t in fig.texts) or ax0.title.get_text()
    assert "view0" in header_text and "view1" in header_text
    assert "view0" in ax0.get_xlabel() and "view1" in ax0.get_xlabel()
    plt.close(fig)

    fig_cust, ax_cust = plt.subplots()
    fig_ret = P.plot_edge_features(pm, "view0 - view1", component=0, ax=ax_cust)
    assert fig_ret is fig_cust
    assert len(ax_cust.get_yticklabels()) <= 15
    plt.close(fig_cust)

    # Test tuple specification and custom feature names
    fn = [f"custom_gene_{i}" for i in range(20)]
    fig_names = P.plot_edge_features(pm, ("view0", "view1"), top_n=5,
                                    feature_names=fn)
    assert isinstance(fig_names, matplotlib.figure.Figure)
    labels = [t.get_text() for t in fig_names.axes[0].get_yticklabels()]
    assert any("custom_gene_" in l for l in labels)
    plt.close(fig_names)

    # Nonexistent edge raises ValueError
    with pytest.raises(ValueError, match="not found in feature_contributions"):
        P.plot_edge_features(pm, "nonexistent - edge")


def test_plot_edge_brain_map_fake(quick_sem_case, monkeypatch):
    import sys
    import types
    known = {"left_entorhinal": 1006, "right_entorhinal": 2006, "left_superior_frontal": 1028}
    mod = types.ModuleType("antsxviz")

    def dkt_id(name):
        if name not in known:
            raise KeyError(name)
        return known[name]

    def rows_from_encoding(enc, cmap=0, notes=None):
        return {e["label"]: {"values": dict(zip(e["names"], map(float, e["values"]))),
                             "selected": [], "vmax": e["vmax"], "cmap": cmap}
                for e in enc.values()}

    def resolve_clim(vals, clim, scale):
        return -1.0, 1.0, True

    def load_fsaverage(density="41k", surface="inflated"):
        return {"lh": None, "rh": None}

    def surface_figure(rows, meshes, title=None, subtitle=None):
        fig, _ = plt.subplots()
        return fig

    scene_mod = types.ModuleType("antsxviz.scene")
    scene_mod.value_cmap = lambda div, view: "coolwarm"

    mod.dkt_id = dkt_id
    mod.rows_from_encoding = rows_from_encoding
    mod.resolve_clim = resolve_clim
    mod.load_fsaverage = load_fsaverage
    mod.surface_figure = surface_figure
    mod.scene = scene_mod

    monkeypatch.setitem(sys.modules, "antsxviz", mod)
    monkeypatch.setitem(sys.modules, "antsxviz.scene", scene_mod)

    pm, _ = quick_sem_case
    fn = ["left_entorhinaldktcortex", "right_entorhinaldktcortex", "left_superior_frontaldktcortex",
          "csftissues", "feat4", "feat5", "feat6", "feat7"]
    fig = P.plot_edge_brain_map(pm, "view0 - view1", component=0, feature_names=fn)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    # When no features map to DKT, raises ValueError
    with pytest.raises(ValueError, match="is a DKT cortical region"):
        P.plot_edge_brain_map(pm, "view0 - view1", component=0,
                              feature_names=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"])


def test_plot_edge_brain_map_real(quick_sem_case):
    pytest.importorskip("antsxviz")
    pm, _ = quick_sem_case
    fn = ["left_entorhinaldktcortex", "right_entorhinaldktcortex", "left_superior_frontaldktcortex",
          "csftissues", "feat4", "feat5", "feat6", "feat7"]
    fig = P.plot_edge_brain_map(pm, "view0 - view1", component=0, feature_names=fn)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

