"""`pysimlr.plot_brain_map` and its adapter to antsxviz (the ANTsX house renderer)."""
import sys
import types

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.plotting import brain_rows, cortical_region_name


def test_cortical_region_name():
    assert cortical_region_name("left_entorhinal") == "left_entorhinal"
    assert cortical_region_name("right_superior_frontaldktcortex") == "right_superior_frontal"
    assert cortical_region_name("left_entorhinaldktregions") == "left_entorhinal"
    for not_cortex in ("csftissues", "temporal_ldktlobes", "gmtissues", "v0.3"):
        assert cortical_region_name(not_cortex) is None


def _fake_antsxviz(monkeypatch):
    """A stand-in exposing only what brain_rows uses, so the adapter is tested without the renderer."""
    known = {"left_entorhinal": 1006, "right_entorhinal": 2006, "left_superior_frontal": 1028}
    mod = types.ModuleType("antsxviz")

    def dkt_id(name):
        if name not in known:
            raise KeyError(name)
        return known[name]

    def rows_from_encoding(enc, cmap=0, notes=None):
        return {e["label"]: {"values": dict(zip(e["names"], map(float, e["values"]))),
                             "selected": [n for n, s in zip(e["names"], e["selected"]) if s]
                             if e["selected"] is not None else [], "vmax": e["vmax"], "cmap": cmap}
                for e in enc.values()}

    mod.dkt_id, mod.rows_from_encoding = dkt_id, rows_from_encoding
    monkeypatch.setitem(sys.modules, "antsxviz", mod)


def test_brain_rows_keeps_cortex_reports_the_rest(monkeypatch):
    _fake_antsxviz(monkeypatch)
    names = ["left_entorhinaldktregions", "left_entorhinaldktcortex", "right_entorhinaldktcortex",
             "csftissues", "left_superior_frontaldktcortex", "left_bogusdktcortex"]
    enc = {"relative_weight": {"names": names, "values": np.array([1.0, 2.0, 3.0, 9.0, 4.0, 5.0]), "vmax": 9.0,
                               "selected": np.array([False, True, False, True, False, False]),
                               "stability": 0.9, "units": "%", "label": "relative weight", "view_name": "v"}}
    rows, dropped = brain_rows(enc)
    vals = rows["relative weight"]["values"]
    # the DKT cortex label wins over the parcellation variant for the same region
    assert vals == {"left_entorhinal": 2.0, "right_entorhinal": 3.0, "left_superior_frontal": 4.0}
    assert rows["relative weight"]["selected"] == ["left_entorhinal"]
    assert set(dropped) == {"left_entorhinaldktregions", "csftissues", "left_bogusdktcortex"}


def test_missing_antsxviz_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "antsxviz", None)          # import antsxviz -> ImportError
    with pytest.raises(ImportError, match=r"pysimlr\[viz\]"):
        brain_rows({"structure": {"names": ["left_entorhinal"], "values": [1.0], "vmax": 1.0,
                                  "selected": None, "label": "structure"}})


def test_plot_brain_map_end_to_end():
    av = pytest.importorskip("antsxviz")
    try:
        meshes = av.load_fsaverage()
    except (FileNotFoundError, OSError) as err:          # assets not staged on this machine
        pytest.skip(f"fsaverage assets unavailable: {err}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.random.default_rng(0); n = 160
    z = g.standard_normal((n, 2))
    regions = ["left_entorhinal", "right_entorhinal", "left_superior_frontal", "right_superior_frontal",
               "left_precuneus", "right_precuneus", "left_insula", "right_insula"]
    names0 = regions + ["csftissues"]
    xs = [torch.tensor(z @ g.standard_normal((2, p)) + 0.5 * g.standard_normal((n, p))).float() for p in (9, 7)]
    res = pysimlr.fit("simlr", xs, k=2, iterations=30, random_state=0)
    fig = pysimlr.plot_brain_map(res, view=0, feature_names=[names0, [f"b{i}" for i in range(7)]],
                                 kinds=("structure",), meshes=meshes)
    assert len(fig.axes) >= 5                                   # five views plus a colorbar
    texts = " ".join(t.get_text() for t in fig.texts)
    assert "csftissues" in texts                                # the non-cortical feature is named, not hidden
    plt.close(fig)


def test_palette_matches_antsxviz():
    """pysimlr keeps its own palette (core install has no antsxviz); it must not drift from
    the house renderer's, or a table and a brain from the same fit would differ in colour."""
    av = pytest.importorskip("antsxviz")
    import antsxviz.style as st
    from pysimlr.plotting import VIEW_COLORS, map_colormap
    assert list(VIEW_COLORS) == list(st.VIEW_COLORS)
    x = np.linspace(0, 1, 21)
    for v in range(len(VIEW_COLORS)):
        assert np.allclose([map_colormap(v)(t) for t in x], [av.map_colormap(v)(t) for t in x], atol=0)


def _fake_scene_antsxviz(monkeypatch, seen):
    mod = types.ModuleType("antsxviz")
    scene = types.ModuleType("antsxviz.scene")

    def record(name):
        def f(maps, *a, **kw):
            seen.update(fn=name, maps=maps, args=a, **kw)
            return "fig"
        return f

    mod.region_scene_figure = record("region")
    mod.tract_scene_figure = record("tract")
    mod.connectome_figure = record("connectome")
    monkeypatch.setitem(sys.modules, "antsxviz", mod)
    monkeypatch.setitem(sys.modules, "antsxviz.scene", scene)


def _small_fit(names0, p1=4):
    g = np.random.default_rng(0)
    z = g.standard_normal((80, 2))
    xs = [torch.tensor(z @ g.standard_normal((2, p)) + 0.5 * g.standard_normal((80, p))).float()
          for p in (len(names0), p1)]
    return pysimlr.fit("simlr", xs, k=2, iterations=10, random_state=0), [names0, [f"b{i}" for i in range(p1)]]


def test_plot_brain_map_volumetric_atlas_dispatch(monkeypatch):
    """Volumetric atlases go to the antsxviz scene functions with rows, renaming, scale and pass-through kwargs."""
    seen = {}
    _fake_scene_antsxviz(monkeypatch, seen)
    names0 = ["FA_HOA_Thalamus_Left", "dti_mean_fa", "my_put"]
    res, fn = _small_fit(names0)
    out = pysimlr.plot_brain_map(res, view=0, kinds=("structure",), atlas="hoa", feature_names=fn,
                                 region_names={"my_put": "FA_HOA_Putamen_Left"}, scale="zero_max",
                                 brain_mask="MASK", views=("left",))
    assert out == "fig" and seen["fn"] == "region" and seen["args"] == ("hoa",)
    assert seen["scale"] == "zero_max" and seen["brain_mask"] == "MASK" and seen["views"] == ("left",)
    (row,) = seen["maps"].values()
    assert list(row) == ["FA_HOA_Thalamus_Left", "dti_mean_fa", "FA_HOA_Putamen_Left"]
    pysimlr.plot_brain_map(res, view=0, kinds=("structure",), atlas="tract_sheets", feature_names=fn)
    assert seen["fn"] == "tract" and seen["style"] == "sheets" and seen["scale"] == "minmax"
    with pytest.raises(ValueError, match="DKT surface only"):
        pysimlr.plot_brain_map(res, atlas="hoa", tiers={"alpha": 0.05})
    with pytest.raises(ValueError, match="atlas must be"):
        pysimlr.plot_brain_map(res, atlas="slices", feature_names=fn)


def test_plot_connectome_dispatch(monkeypatch):
    seen = {}
    _fake_scene_antsxviz(monkeypatch, seen)
    names0 = ["rsf_yeo_netconn_DefaultA_2_ContB", "rsf_yeo_netconn_VisCent_2_VisCent", "rsf_yeo_netconn_nparcels_DefaultA"]
    res, fn = _small_fit(names0)
    assert pysimlr.plot_connectome(res, view=0, feature_names=fn, top_k=5) == "fig"
    assert seen["fn"] == "connectome" and set(seen["maps"]) == set(names0) and seen["top_k"] == 5


def test_dkt_scale_sets_row_limits():
    av = pytest.importorskip("antsxviz")
    from pysimlr.plotting import _apply_scale
    rows = {"m": {"values": {"left_entorhinal": 0.2, "right_entorhinal": 0.5}, "vmax": 0.5}}
    _apply_scale(rows, "minmax", None, 0, av)
    assert (rows["m"]["vmin"], rows["m"]["vmax"]) == (0.2, 0.5)
    _apply_scale(rows, "zero_max", None, 0, av)
    assert (rows["m"]["vmin"], rows["m"]["vmax"]) == (0.0, 0.5)
    _apply_scale(rows, "minmax", {"m": (0.0, 1.0)}, 0, av)
    assert (rows["m"]["vmin"], rows["m"]["vmax"]) == (0.0, 1.0)
