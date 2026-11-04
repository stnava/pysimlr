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
