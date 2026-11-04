"""`pysimlr.feature_maps`: alternatives to structure coefficients, in the fit's frame."""
import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.loadings import (_component_partial_map, _corr_map, _global_partial_map, _partial_map,
                              _relative_weight_map, _zscore)


def _views(seed=0, n=240, ps=(18, 14, 10)):
    g = np.random.default_rng(seed)
    z = g.standard_normal((n, 2))
    return [torch.tensor(z @ g.standard_normal((2, p)) + 0.6 * g.standard_normal((n, p))).float() for p in ps]


def _global_factor_view(seed=0, n=300, p=40, n_support=6, rho=0.7, direct=0.3, g_max=3.0):
    """The ADNI regime: every feature loads a global factor g (loadings 0.3-3; PC1
    ~63% of variance, DiReCT thickness has 58%) that correlates rho with the shared
    signal s; only the first ``n_support`` features carry s directly, and they load
    g weakly (0.5). Heavily-global non-drivers then correlate with s more than the
    drivers do."""
    r = np.random.default_rng(seed)
    s = r.standard_normal(n)
    g = rho * s + np.sqrt(1 - rho ** 2) * r.standard_normal(n)
    b = r.uniform(0.3, g_max, p); b[:n_support] = 0.5
    x = g[:, None] * b + r.standard_normal((n, p))
    x[:, :n_support] += direct * s[:, None]
    return x, s, set(range(n_support))


def _auc(score, support):
    from sklearn.metrics import roc_auc_score
    y = np.zeros(len(score)); y[list(support)] = 1
    return roc_auc_score(y, np.abs(score))


def test_partial_matches_textbook_partial_correlation():
    r = np.random.default_rng(1)
    x = r.standard_normal((500, 6)); t = x[:, :2].sum(1, keepdims=True) + r.standard_normal((500, 1))
    xz, tz = _zscore(x), _zscore(t)
    got = _partial_map(xz, tz, 0.0)[:, 0]
    prec = np.linalg.inv(np.corrcoef(np.column_stack([x, t]).T))
    want = -prec[:-1, -1] / np.sqrt(np.diag(prec)[:-1] * prec[-1, -1])
    assert np.allclose(got, want, atol=1e-10)


def test_component_partial_equals_structure_for_uncorrelated_targets():
    r = np.random.default_rng(2)
    x = r.standard_normal((300, 8))
    t, _ = np.linalg.qr(_zscore(r.standard_normal((300, 3))))     # exactly uncorrelated columns
    xz, tz = _zscore(x), _zscore(t)
    assert np.allclose(_component_partial_map(xz, tz), _corr_map(xz, tz), atol=1e-10)


def test_relative_weights_sum_to_r2_and_are_nonnegative():
    r = np.random.default_rng(3)
    x = r.standard_normal((400, 7)); x[:, 1] += 0.8 * x[:, 0]
    t = x @ r.standard_normal((7, 2)) + r.standard_normal((400, 2))
    xz, tz = _zscore(x), _zscore(t)
    w = _relative_weight_map(xz, tz)
    beta, *_ = np.linalg.lstsq(xz, tz, rcond=None)
    r2 = 1 - ((tz - xz @ beta) ** 2).sum(0) / (tz ** 2).sum(0)
    assert (w >= 0).all()
    assert np.allclose(w.sum(0), r2, atol=1e-10)


def test_relative_weight_undefined_when_p_exceeds_n():
    r = np.random.default_rng(4)
    with pytest.raises(ValueError, match="fewer features than rows"):
        _relative_weight_map(_zscore(r.standard_normal((20, 30))), _zscore(r.standard_normal((20, 1))))
    # partial is still defined there, through shrinkage
    m = _partial_map(_zscore(r.standard_normal((20, 30))), _zscore(r.standard_normal((20, 1))), "ledoit_wolf")
    assert np.isfinite(m).all()


def test_global_partial_and_partial_beat_structure_on_a_global_factor_view():
    """The ADNI failure mode: every feature shares a factor correlated with the signal."""
    x, s, sup = _global_factor_view()
    xz, tz = _zscore(x), _zscore(s[:, None])
    a_struct = _auc(_corr_map(xz, tz)[:, 0], sup)
    a_part = _auc(_partial_map(xz, tz, "ledoit_wolf")[:, 0], sup)
    a_glob = _auc(_global_partial_map(xz, tz, "pc1")[:, 0], sup)
    a_rw = _auc(_relative_weight_map(xz, tz)[:, 0], sup)
    assert a_part >= a_struct + 0.05 and a_glob >= a_struct + 0.05 and a_rw >= a_struct + 0.05, \
        (a_struct, a_part, a_glob, a_rw)


def test_feature_maps_shapes_frame_and_other_views_target():
    xs = _views()
    res = pysimlr.fit("simlr", xs, k=2, iterations=40, random_state=0)
    out = pysimlr.feature_maps(res, xs)
    assert set(out) == set(pysimlr.FEATURE_MAP_KINDS)
    for kind, mats in out.items():
        assert [tuple(m.shape) for m in mats] == [(x.shape[1], 2) for x in xs], kind
        assert all(torch.isfinite(m).all() for m in mats), kind
    # The default (other-views) target differs from the consensus target.
    cons = pysimlr.feature_maps(res, xs, kinds=("structure",), target="consensus")["structure"]
    assert not torch.allclose(cons[0], out["structure"][0])
    # structure with the consensus target is the fit's own structure (same frame).
    ref = pysimlr.loadings(res, xs, u=pysimlr.predict(res, xs)["u"])["structure"]
    assert torch.allclose(cons[0], ref[0], atol=1e-5)


def test_other_views_target_excludes_own_view():
    xs = _views(seed=5)
    res = pysimlr.fit("simlr", xs, k=2, iterations=40, random_state=0)
    a = pysimlr.feature_maps(res, xs, kinds=("structure",))["structure"][0]
    # Recompute by hand: mean of the standardized, signed scores of views 1 and 2.
    from pysimlr.api import _preprocess_views
    xp, _ = _preprocess_views(xs, res["scale_list"], res["provenance_list"])
    rot = res["loading_rotation"].double().numpy()
    pred = pysimlr.predict(res, xs)
    u = pred["u"].double().numpy() @ rot
    zs = []
    for z in pred["latents"]:
        z = z.double().numpy() @ rot
        sg = np.where(((z - z.mean(0)) * (u - u.mean(0))).sum(0) >= 0, 1.0, -1.0)
        zs.append(_zscore(z * sg))
    t = _zscore(zs[1] + zs[2])
    want = _corr_map(_zscore(xp[0].double().numpy()), t)
    assert np.allclose(a.numpy(), want, atol=1e-5)


def test_feature_maps_errors_and_per_view_control():
    xs = _views()
    res = pysimlr.fit("simlr", xs, k=2, iterations=30, random_state=0)
    with pytest.raises(ValueError, match="unknown feature map"):
        pysimlr.feature_maps(res, xs, kinds=("weights",))
    with pytest.raises(ValueError, match="target"):
        pysimlr.feature_maps(res, xs, target="self")
    with pytest.raises(ValueError, match="control has 2 entries"):
        pysimlr.feature_maps(res, xs, kinds=("global_partial",), control=["pc1", None])
    out = pysimlr.feature_maps(res, xs, kinds=("global_partial",), control=["pc1", None, "mean"])
    assert torch.isfinite(out["global_partial"][0]).all()
    assert torch.isnan(out["global_partial"][1]).all()


def test_relative_weight_nan_with_warning_on_wide_view():
    g = np.random.default_rng(6); z = g.standard_normal((40, 2))
    xs = [torch.tensor(z @ g.standard_normal((2, p)) + 0.5 * g.standard_normal((40, p))).float() for p in (60, 8)]
    res = pysimlr.fit("simlr", xs, k=2, iterations=30, random_state=0)
    with pytest.warns(RuntimeWarning, match="relative_weight is NaN for view 0"):
        out = pysimlr.feature_maps(res, xs, kinds=("relative_weight", "partial"))
    assert torch.isnan(out["relative_weight"][0]).all()
    assert torch.isfinite(out["relative_weight"][1]).all()
    assert torch.isfinite(out["partial"][0]).all()


def test_feature_maps_on_a_deep_result():
    xs = _views(n=160)
    res = pysimlr.fit("lend", xs, k=2, iterations=3, random_state=0)
    out = pysimlr.feature_maps(res, xs, kinds=("structure", "partial"))
    assert all(torch.isfinite(m).all() for m in out["partial"])


def test_loading_inference_reports_alternative_maps():
    xs = _views(n=200)
    inf = pysimlr.loading_inference("simlr", xs, 2, n_subsamples=4, iterations=30,
                                    maps=("partial", "relative_weight"))
    for kind in ("partial", "relative_weight"):
        e = inf["maps"][kind]
        assert [tuple(m.shape) for m in e["map"]] == [(x.shape[1], 2) for x in xs]
        assert len(e["component_stability"]) == len(xs) and len(e["component_stability"][0]) == 2
        assert all((lo <= hi + 1e-6).all() for lo, hi in zip(e["ci_low"], e["ci_high"]))
        assert all(((f >= 0) & (f <= 1)).all() for f in e["selection_frequency"])
    # relative weights stay non-negative after alignment (signs are not applied to them)
    assert all((lo >= -1e-9).all() for lo in inf["maps"]["relative_weight"]["ci_low"])
    with pytest.raises(ValueError, match="unknown feature map"):
        pysimlr.loading_inference("simlr", xs, 2, n_subsamples=2, iterations=10, maps=("nope",))


def test_feature_table_and_plots_take_alternative_maps():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = _views(n=200)
    inf = pysimlr.loading_inference("simlr", xs, 2, n_subsamples=4, iterations=30, maps=("partial",))
    res = inf["result"]
    extra = pysimlr.feature_maps(res, xs, kinds=("relative_weight", "global_partial"))
    tab = pysimlr.feature_table(res, inf, maps=extra)
    for col in ("partial", "partial_ci_low", "partial_stability", "relative_weight", "global_partial"):
        assert col in tab.columns, col
    fig = pysimlr.plot_top_features(res, inf, kind="partial")
    assert "stability" in fig.axes[0].get_title(loc="left")
    plt.close(fig)
    fig = pysimlr.plot_top_features(res, maps=extra, kind="relative_weight")
    assert fig.axes[0].get_xlim()[0] == 0
    plt.close(fig)
    with pytest.raises(ValueError, match="not available"):
        pysimlr.plot_top_features(res, kind="partial")
    fig = pysimlr.plot_feature_map_comparison(res, maps=extra, inf=inf, view=1)
    assert len(fig.axes[0].get_xticklabels()) == 4          # structure, partial, relative_weight, global_partial
    plt.close(fig)
    with pytest.raises(ValueError, match="at least two"):
        pysimlr.plot_feature_map_comparison(res)


def test_grid_shift_and_shared_encoding():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = _views(n=200)
    inf = pysimlr.loading_inference("simlr", xs, 2, n_subsamples=4, iterations=30, maps=("relative_weight",))
    res = inf["result"]
    names = [[f"{'left' if i % 2 == 0 else 'right'}_r{i // 2}" for i in range(x.shape[1])] for x in xs]
    enc = pysimlr.feature_map_values(res, inf=inf, view=0, feature_names=names)
    rw = enc["relative_weight"]
    assert abs(np.nansum(rw["values"]) - 100.0) < 1e-6           # percent of explained variance
    assert rw["vmax"] == np.nanmax(rw["values"]) and rw["selected"].dtype == bool
    assert enc["structure"]["units"] == "|corr(x, u)|"
    fig = pysimlr.plot_feature_map_shift(res, inf=inf, view=0, feature_names=names)
    plt.close(fig)
    cm = pysimlr.map_colormap(2)
    assert cm(0.0)[:3] != cm(1.0)[:3]


def test_feature_map_grid_via_antsxviz():
    """The grid is drawn by antsxviz.region_table_figure (needs the viz extra)."""
    pytest.importorskip("antsxviz")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = _views(n=200)
    inf = pysimlr.loading_inference("simlr", xs, 2, n_subsamples=4, iterations=30, maps=("relative_weight",))
    names = [[f"{'left' if i % 2 == 0 else 'right'}_r{i // 2}" for i in range(x.shape[1])] for x in xs]
    split = lambda n: (n.split("_", 1)[1], "L" if n.startswith("left") else "R")
    groups = {f"r{i}": ("a" if i < 4 else "b") for i in range(9)}
    fig = pysimlr.plot_feature_map_grid(inf["result"], inf=inf, view=0, feature_names=names, split=split,
                                        groups=groups)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["L", "R"]
    assert len(ax.get_yticklabels()) == 9                         # 18 features -> 9 paired rows
    plt.close(fig)


def test_feature_map_grid_without_antsxviz_names_the_extra(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "antsxviz", None)
    xs = _views(n=120)
    res = pysimlr.fit("simlr", xs, k=2, iterations=20, random_state=0)
    with pytest.raises(ImportError, match=r"pysimlr\[viz\]"):
        pysimlr.plot_feature_map_grid(res, view=0)
