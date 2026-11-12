"""Unit tests for adaptive contributor passthrough in pysimlr.plotting.

Covers:
- Parameter signatures for plot_brain_map, plot_edge_brain_map, plot_feature_chord
- Atlas validation and dispatch for "deep", "systems_v2", "connectome"
- Safe mock defense when downstream surface_figure has legacy/mock signatures
- _derive_edge_active auto-derivation across edge_stats, edge_components, and edges
- plot_feature_chord delegation to antsxviz.contributor_chord_figure vs legacy fallback
"""
from __future__ import annotations

import inspect
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pysimlr
from pysimlr import plotting as pplt
from pysimlr.plotting import _derive_edge_active, plot_brain_map, plot_edge_brain_map, plot_feature_chord


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------
def _make_dummy_simlr_res(n_views: int = 2, n_features: int = 4, k: int = 2) -> dict[str, Any]:
    """Generates a minimal valid fit result compatible with feature_map_values and feature_table."""
    rng = np.random.default_rng(42)
    scores = [rng.standard_normal((30, k)) for _ in range(n_views)]
    weights = [rng.standard_normal((n_features, k)) for _ in range(n_views)]
    # structure is correlation between features and component scores
    structure = [rng.uniform(-0.8, 0.8, size=(n_features, k)) for _ in range(n_views)]
    vnames = [f"view_{i}" for i in range(n_views)]
    fnames = [[f"feat_{i}_{j}" for j in range(n_features)] for i in range(n_views)]
    return {
        "scores": scores,
        "latents": scores,
        "u": scores[0],
        "weights": weights,
        "loadings": weights,
        "structure": structure,
        "loading_rotation": np.eye(k),
        "relative_weight": [np.abs(s) / (np.sum(np.abs(s), axis=0, keepdims=True) + 1e-12) for s in structure],
        "view_names": vnames,
        "feature_names": fnames,
        "k": k,
        "n_views": n_views,
    }


def _make_dummy_pm(edge: tuple[str, str] = ("view_0", "view_1"), component: int = 0) -> dict[str, Any]:
    """Generates a minimal path model result with feature_contributions."""
    n_features = 4
    feats = list(range(n_features))
    contribs = [0.4, -0.2, 0.1, -0.05]
    fc_df = pd.DataFrame({
        "edge": [f"{edge[0]} - {edge[1]}"] * n_features,
        "a": [edge[0]] * n_features,
        "b": [edge[1]] * n_features,
        "component": [component] * n_features,
        "feature": feats,
        "contribution": contribs,
    })
    return {
        "result": "mock_fit",
        "nodes": list(edge),
        "feature_contributions": fc_df,
    }


# ---------------------------------------------------------------------------
# 1. Signatures
# ---------------------------------------------------------------------------
def test_plot_brain_map_signature():
    """plot_brain_map includes max_k, active, null_bound, and ranked_twin in its signature."""
    sig = inspect.signature(plot_brain_map)
    for p in ("max_k", "active", "null_bound", "ranked_twin"):
        assert p in sig.parameters
    assert sig.parameters["max_k"].default is None
    assert sig.parameters["active"].default is True
    assert sig.parameters["null_bound"].default is None
    assert sig.parameters["ranked_twin"].default is False


def test_plot_edge_brain_map_signature():
    """plot_edge_brain_map includes max_k, active, null_bound, ranked_twin, and alpha."""
    sig = inspect.signature(plot_edge_brain_map)
    for p in ("max_k", "active", "null_bound", "ranked_twin", "alpha"):
        assert p in sig.parameters
    assert sig.parameters["max_k"].default is None
    assert sig.parameters["active"].default is None  # None triggers auto-derivation
    assert sig.parameters["null_bound"].default is None
    assert sig.parameters["ranked_twin"].default is False
    assert sig.parameters["alpha"].default == 0.05


def test_plot_feature_chord_signature():
    """plot_feature_chord includes max_k, active, null_bound, target_colors, adaptive."""
    sig = inspect.signature(plot_feature_chord)
    for p in ("max_k", "active", "null_bound", "target_colors", "title", "subtitle", "adaptive"):
        assert p in sig.parameters
    assert sig.parameters["max_k"].default is None
    assert sig.parameters["active"].default is True
    assert sig.parameters["adaptive"].default is None


# ---------------------------------------------------------------------------
# 2. Atlas Validation
# ---------------------------------------------------------------------------
def test_plot_brain_map_atlas_validation():
    """Accepts 'deep' and 'systems_v2', rejects unrecognized atlases."""
    res = _make_dummy_simlr_res()
    # Unrecognized atlas raises ValueError with descriptive message
    with pytest.raises(ValueError, match="atlas must be"):
        plot_brain_map(res, atlas="unrecognized_atlas")


def test_plot_edge_brain_map_atlas_validation():
    """Accepts 'deep', 'systems_v2', and 'connectome', rejects unrecognized atlases."""
    pm = _make_dummy_pm()
    with pytest.raises(ValueError, match="atlas must be"):
        plot_edge_brain_map(pm, edge=("view_0", "view_1"), atlas="unrecognized_atlas")


# ---------------------------------------------------------------------------
# 3. Safe Mock Defense
# ---------------------------------------------------------------------------
def test_plot_brain_map_safe_mock_defense(monkeypatch):
    """When antsxviz.surface_figure has a legacy/mock signature, unexpected kwargs are withheld."""
    res = _make_dummy_simlr_res()
    seen = {}

    def mock_surface_figure(rows, meshes, title=None, subtitle=None):
        seen["called"] = True
        seen["title"] = title
        return "mock_figure"

    fake_av = types.ModuleType("antsxviz")
    fake_av.surface_figure = mock_surface_figure
    fake_av.load_fsaverage = lambda **kw: {"lh": None, "rh": None}
    fake_av.rows_from_encoding = lambda enc, **kw: {k: {"values": {1006: 0.1}} for k in enc}
    fake_av.dkt_id = lambda name: 1006
    fake_av.resolve_clim = lambda vals, clim, scale: (-1.0, 1.0, True)
    fake_scene = types.ModuleType("antsxviz.scene")
    fake_scene.value_cmap = lambda div, view: "coolwarm"
    fake_av.scene = fake_scene
    monkeypatch.setattr(pplt, "_require_antsxviz", lambda: fake_av)

    # Calling with max_k and active does not raise TypeError
    out = plot_brain_map(
        res,
        atlas="dkt",
        kinds=("structure",),
        max_k=5,
        active=False,
        null_bound=0.2,
        ranked_twin=True,
        region_names={"feat_0_0": "left_entorhinal"},
    )
    assert out == "mock_figure"
    assert seen["called"] is True


def test_plot_edge_brain_map_safe_mock_defense(monkeypatch):
    """plot_edge_brain_map does not raise TypeError with fixed mock surface_figure signature."""
    pm = _make_dummy_pm()
    seen = {}

    def mock_surface_figure(rows, meshes, title=None, subtitle=None):
        seen["called"] = True
        return "mock_edge_figure"

    fake_av = types.ModuleType("antsxviz")
    fake_av.surface_figure = mock_surface_figure
    fake_av.load_fsaverage = lambda **kw: {"lh": None, "rh": None}
    fake_av.rows_from_encoding = lambda enc, **kw: {k: {"values": {1006: 0.1}} for k in enc}
    fake_av.dkt_id = lambda name: 1006
    fake_av.resolve_clim = lambda vals, clim, scale: (-1.0, 1.0, True)
    fake_scene = types.ModuleType("antsxviz.scene")
    fake_scene.value_cmap = lambda div, view: "coolwarm"
    fake_av.scene = fake_scene
    monkeypatch.setattr(pplt, "_require_antsxviz", lambda: fake_av)

    out = plot_edge_brain_map(
        pm,
        edge=("view_0", "view_1"),
        atlas="dkt",
        max_k=3,
        active=False,
        region_names={0: "left_entorhinal"},
    )
    assert out == "mock_edge_figure"
    assert seen["called"] is True


# ---------------------------------------------------------------------------
# 4. Atlas Routing in plot_brain_map
# ---------------------------------------------------------------------------
def test_plot_brain_map_deep_atlas_routing(monkeypatch):
    """atlas='deep' routes to deep_brain_figure (single kind) or multimodal_deep_figure (multiple kinds)."""
    res = _make_dummy_simlr_res()
    seen = {}

    def mock_deep(single_map, **kwargs):
        seen["fn"] = "deep_brain_figure"
        seen["map"] = single_map
        seen["kwargs"] = kwargs
        return "deep_fig"

    def mock_multimodal(maps, **kwargs):
        seen["fn"] = "multimodal_deep_figure"
        seen["maps"] = maps
        seen["kwargs"] = kwargs
        return "multimodal_fig"

    fake_av = types.ModuleType("antsxviz")
    fake_av.deep_brain_figure = mock_deep
    fake_av.multimodal_deep_figure = mock_multimodal
    monkeypatch.setattr(pplt, "_require_antsxviz", lambda: fake_av)

    # 1. Single kind -> deep_brain_figure
    out1 = plot_brain_map(res, atlas="deep", kinds=("structure",), max_k=4, active=False, null_bound=0.1)
    assert out1 == "deep_fig"
    assert seen["fn"] == "deep_brain_figure"
    assert seen["kwargs"]["max_k"] == 4
    assert seen["kwargs"]["active"] is False
    assert seen["kwargs"]["null_bound"] == 0.1

    # 2. Multiple kinds -> multimodal_deep_figure
    out2 = plot_brain_map(
        res,
        atlas="deep",
        kinds=("structure", "relative_weight"),
        maps={"relative_weight": res["relative_weight"]},
        max_k=5,
        active=True,
    )
    assert out2 == "multimodal_fig"
    assert seen["fn"] == "multimodal_deep_figure"
    assert seen["kwargs"]["max_k"] == 5
    assert seen["kwargs"]["active_modalities"] is True


def test_plot_brain_map_systems_v2_atlas_routing(monkeypatch):
    """atlas='systems_v2' routes to systems_brain_figure."""
    res = _make_dummy_simlr_res()
    seen = {}

    def mock_systems(single_map, **kwargs):
        seen["fn"] = "systems_brain_figure"
        seen["kwargs"] = kwargs
        return "systems_fig"

    fake_av = types.ModuleType("antsxviz")
    fake_av.systems_brain_figure = mock_systems
    monkeypatch.setattr(pplt, "_require_antsxviz", lambda: fake_av)

    out = plot_brain_map(res, atlas="systems_v2", kinds=("structure",), max_k=6, active=True, null_bound=0.25)
    assert out == "systems_fig"
    assert seen["fn"] == "systems_brain_figure"
    assert seen["kwargs"]["max_k"] == 6
    assert seen["kwargs"]["active"] is True
    assert seen["kwargs"]["null_bound"] == 0.25


# ---------------------------------------------------------------------------
# 5. _derive_edge_active Logic
# ---------------------------------------------------------------------------
def test_derive_edge_active_from_edge_stats():
    """_derive_edge_active evaluates edge_stats dictionary."""
    # Significant edge: abs(beta) > 0 and q < alpha
    pm_sig = {"edge_stats": {("A", "B"): {"beta": 0.45, "q": 0.012}}}
    assert _derive_edge_active(pm_sig, ("A", "B"), alpha=0.05) is True

    # Insignificant edge: q >= alpha -> active=False
    pm_ns = {"edge_stats": {("A", "B"): {"beta": 0.45, "q": 0.25}}}
    assert _derive_edge_active(pm_ns, ("A", "B"), alpha=0.05) is False

    # Zero beta: abs(beta) == 0 -> active=False even if q < alpha
    pm_zero = {"edge_stats": {("A", "B"): {"beta": 0.0, "q": 0.001}}}
    assert _derive_edge_active(pm_zero, ("A", "B"), alpha=0.05) is False

    # String format key "A - B"
    pm_str = {"edge_stats": {"A - B": {"beta": 0.3, "q": 0.02}}}
    assert _derive_edge_active(pm_str, "A - B", alpha=0.05) is True


def test_derive_edge_active_from_edge_components():
    """_derive_edge_active evaluates edge_components DataFrame per component."""
    df = pd.DataFrame([
        {"a": "X", "b": "Y", "component": 0, "beta": 0.35, "q_bh": 0.01},
        {"a": "X", "b": "Y", "component": 1, "beta": 0.10, "q_bh": 0.40},
    ])
    pm = {"edge_components": df, "nodes": ["X", "Y"]}
    assert _derive_edge_active(pm, ("X", "Y"), component=0, alpha=0.05) is True
    assert _derive_edge_active(pm, ("X", "Y"), component=1, alpha=0.05) is False


def test_derive_edge_active_from_edges_df():
    """_derive_edge_active evaluates edges summary DataFrame."""
    df = pd.DataFrame([
        {"a": "U", "b": "V", "beta": 0.25, "q_bh": 0.03},
    ])
    pm = {"edges": df, "nodes": ["U", "V"]}
    assert _derive_edge_active(pm, ("U", "V"), alpha=0.05) is True
    assert _derive_edge_active(pm, ("U", "V"), alpha=0.01) is False


def test_derive_edge_active_fallback_default():
    """_derive_edge_active defaults to True when no stats table is provided."""
    pm_empty = {"nodes": ["M", "N"]}
    assert _derive_edge_active(pm_empty, ("M", "N")) is True


# ---------------------------------------------------------------------------
# 6. plot_edge_brain_map Auto-Derivation and Atlas Routing
# ---------------------------------------------------------------------------
def test_plot_edge_brain_map_auto_derives_active(monkeypatch):
    """plot_edge_brain_map auto-derives active=False when path model edge is insignificant."""
    pm = _make_dummy_pm(edge=("view_0", "view_1"))
    pm["edge_stats"] = {("view_0", "view_1"): {"beta": 0.3, "q": 0.30}}  # q >= 0.05

    seen = {}

    def mock_region_scene(maps, atlas, **kwargs):
        seen["kwargs"] = kwargs
        return "region_fig"

    fake_av = types.ModuleType("antsxviz")
    fake_av.region_scene_figure = mock_region_scene
    monkeypatch.setattr(pplt, "_require_antsxviz", lambda: fake_av)

    plot_edge_brain_map(pm, edge=("view_0", "view_1"), atlas="hoa", active=None)
    assert seen["kwargs"]["active"] is False


def test_plot_edge_brain_map_connectome_atlas_routing(monkeypatch):
    """plot_edge_brain_map supports atlas='connectome' delegating to antsxviz.connectome_figure."""
    pm = _make_dummy_pm(edge=("view_0", "view_1"))
    seen = {}

    def mock_connectome(edges, **kwargs):
        seen["called"] = True
        seen["kwargs"] = kwargs
        return "connectome_fig"

    fake_av = types.ModuleType("antsxviz")
    fake_av.connectome_figure = mock_connectome
    monkeypatch.setattr(pplt, "_require_antsxviz", lambda: fake_av)

    out = plot_edge_brain_map(pm, edge=("view_0", "view_1"), atlas="connectome", max_k=4, active=True)
    assert out == "connectome_fig"
    assert seen["called"] is True
    assert seen["kwargs"]["max_k"] == 4
    assert seen["kwargs"]["adaptive"] is True


# ---------------------------------------------------------------------------
# 7. plot_feature_chord Delegation vs Legacy Fallback
# ---------------------------------------------------------------------------
def test_plot_feature_chord_adaptive_delegation(monkeypatch):
    """When adaptive options are provided, plot_feature_chord delegates to contributor_chord_figure."""
    res = _make_dummy_simlr_res()
    seen = {}

    def mock_chord(modality_to_targets, **kwargs):
        seen["called"] = True
        seen["data"] = modality_to_targets
        seen["kwargs"] = kwargs
        return "chord_fig"

    fake_av = types.ModuleType("antsxviz")
    fake_av.contributor_chord_figure = mock_chord
    monkeypatch.setattr(pplt, "_require_antsxviz", lambda: fake_av)

    out = plot_feature_chord(
        res,
        view_names=["view_0", "view_1"],
        max_k=4,
        active=False,
        null_bound=0.15,
        target_colors={"C1": "#336699"},
    )
    assert out == "chord_fig"
    assert seen["called"] is True
    assert seen["kwargs"]["max_k"] == 4
    assert seen["kwargs"]["target_colors"] == {"C1": "#336699"}
    assert "view_0" in seen["data"]
    assert "C1" in seen["data"]["view_0"]


def test_plot_feature_chord_legacy_fallback():
    """When no adaptive parameters are provided, executes legacy chord code maintaining _pysimlr_values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = _make_dummy_simlr_res()
    fig = plot_feature_chord(res, top_n=3)
    assert hasattr(fig, "_pysimlr_values")
    assert "features" in fig._pysimlr_values
    assert "components" in fig._pysimlr_values
    plt.close(fig)
