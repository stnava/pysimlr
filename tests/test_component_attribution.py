"""`pysimlr.component_attribution`: held-out attribution of components, view spectra, and refit
stability of the predictive support; and its figure."""
import json
import warnings

import numpy as np
import pytest
import torch

import pysimlr


def _design(seed=0, n=400):
    """Views 0 and 1 share a signal s that drives the outcome; view 2 is a global-factor view."""
    g = np.random.default_rng(seed)
    s = g.standard_normal(n)
    gf = g.standard_normal(n)
    w0 = np.zeros(20); w0[:4] = 1.0
    x0 = np.outer(s, w0) + g.standard_normal((n, 20))
    x1 = np.outer(s, g.uniform(0.5, 1.0, 12)) + g.standard_normal((n, 12))
    x2 = np.outer(gf, g.uniform(1.0, 2.0, 15)) + 0.6 * np.outer(s, np.r_[np.ones(3), np.zeros(12)]) + g.standard_normal((n, 15))
    y = s + 0.5 * g.standard_normal(n)
    z = lambda a, ref: (a - ref.mean(0)) / ref.std(0)
    tr, te = slice(0, 280), slice(280, n)
    xs = [x0, x1, x2]
    return [z(x[tr], x[tr]) for x in xs], [z(x[te], x[tr]) for x in xs], y[tr], y[te]


@pytest.fixture(scope="module")
def att():
    xtr, xte, ytr, yte = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit("simlr", [torch.tensor(x).float() for x in xtr], 2, positivity="positive", random_state=0)
        out = pysimlr.component_attribution(res, xtr, xte, ytr, yte, n_refits=3, n_jobs=1, random_state=1,
                                            variants={"cw0": {"collapse_weight": 0.0}},
                                            feature_names=[[f"a{i}" for i in range(20)], [f"b{i}" for i in range(12)],
                                                           [f"c{i}" for i in range(15)]])
    return out, res, xtr, xte, ytr, yte


def test_schema_and_json(att):
    a = att[0]
    json.dumps(a)                                              # JSON-ready as is
    assert a["views"] == [0, 1, 2] and a["k"] == 2 and a["n_refits"] == 3
    assert len(a["components"]) == 6 and set(a["refits"]) == {"default", "cw0"}
    for m in ("0", "1", "2"):
        sp = a["spectra"][m]
        assert np.isclose(sum(sp["eigen_share"]), 1.0) and len(sp["consensus_vs_mean"]) == 2
        assert len(a["full_support"][m]) == len(a["feature_names"][m])
        for h in a["refits"]["default"]:
            assert len(h[m]["support"]) == len(a["feature_names"][m]) and h[m]["component"] in (1, 2)


def test_r2_alone_and_unique_match_a_direct_computation(att):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    a, res, xtr, xte, ytr, yte = att
    V = [np.asarray(v.detach(), float) for v in res["v"]]
    Z = np.hstack([x @ v for x, v in zip(xtr, V)]); Zt = np.hstack([x @ v for x, v in zip(xte, V)])
    r2 = lambda c: r2_score(yte, LinearRegression().fit(Z[:, c], ytr).predict(Zt[:, c]))
    full = r2(list(range(6)))
    assert np.isclose(a["full_r2"], full)
    for i, c in enumerate(a["components"]):
        assert np.isclose(c["alone"], r2([i]))
        assert np.isclose(c["unique"], full - r2([j for j in range(6) if j != i]))


def test_global_factor_view_has_a_dominant_spectrum(att):
    a = att[0]
    assert a["spectra"]["2"]["eigen_share"][0] > 0.5 > a["spectra"]["1"]["eigen_share"][1]
    assert a["spectra"]["2"]["mean_corr"] > 0.4


def test_signal_views_carry_the_prediction(att):
    a = att[0]
    best = {m: max(c["alone"] for c in a["components"] if c["view"] == m) for m in (0, 1)}
    assert min(best.values()) > 0.2


def test_refit_rows_are_honoured_and_reproducible():
    xtr, xte, ytr, yte = _design(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit("simlr", [torch.tensor(x).float() for x in xtr], 2, random_state=0)
        rows = [np.arange(0, 280, 2), np.arange(1, 280, 2)]
        a = pysimlr.component_attribution(res, xtr, xte, ytr, yte, refit_rows=rows, n_jobs=1)
        b = pysimlr.component_attribution(res, xtr, xte, ytr, yte, refit_rows=rows, n_jobs=1)
    assert a["n_refits"] == 2 and np.isclose(a["refit_fraction"], 0.5)
    assert json.dumps(a["refits"], sort_keys=True) == json.dumps(b["refits"], sort_keys=True)


def test_missing_test_view_is_skipped_and_errors():
    xtr, xte, ytr, yte = _design(seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit("simlr", [torch.tensor(x).float() for x in xtr], 2, random_state=0)
        a = pysimlr.component_attribution(res, xtr, [xte[0], xte[1], None], ytr, yte, n_refits=1, n_jobs=1)
    assert a["views"] == [0, 1] and set(a["spectra"]) == {"0", "1"}
    with pytest.raises(ValueError, match="entries"):
        pysimlr.component_attribution(res, xtr[:2], xte, ytr, yte, n_refits=1, n_jobs=1)
    with pytest.raises(ValueError, match="no view"):
        pysimlr.component_attribution(res, xtr, [None, None, None], ytr, yte, n_refits=1, n_jobs=1)
    with pytest.raises(ValueError, match="length"):
        pysimlr.component_attribution(res, xtr, xte, ytr[:10], yte, n_refits=1, n_jobs=1)


def test_plot_component_attribution(att):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pysimlr.plotting import plot_component_attribution
    a = att[0]
    fig = plot_component_attribution(a, selection_views=[0, 2], compare={2: "cw0"}, feature_word="region",
                                     variant_labels={"cw0": "collapse_weight = 0"}, titles={"A": "custom A"})
    assert len(fig.axes) == 4
    assert fig.axes[0].get_title("left") == "custom A"          # the theme left-aligns titles
    assert any("collapse_weight = 0" in t.get_text() for t in fig.axes[3].get_legend().get_texts())
    fig2 = plot_component_attribution(a)                              # defaults: spectrum + best two views
    assert len(fig2.axes) == 4
    plt.close("all")
