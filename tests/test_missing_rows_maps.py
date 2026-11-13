"""`feature_maps`, `loading_inference` and `loadings` with ``missing_rows="mask"`` and unobserved rows.

Regression for v0.31.1: with unobserved (all-NaN) rows, ``feature_maps(..., target="other_views")``
returned all-zero maps for every view (the other-views score was NaN on a row, its z-score became 0
through ``_zscore``'s ``sd > 0`` test) and ``loading_inference(..., maps=...)`` inherited them, while
its ``oos_coupling`` read exactly 0.0 (``np.where(den > 0, ..., 0.0)`` on NaN). Every quantity is now computed on
the rows where its inputs exist; what cannot be computed is NaN with the reason and the effective n recorded.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import MissingRowsWarning
from pysimlr.loadings import FEATURE_MAP_KINDS, _masked_coupling, structure_coefficients

WIDTHS = (12, 10, 14, 8)
REF_FILE = Path(__file__).parent / "data" / "missing_rows_main_reference.npz"


def make_views(n=120, k=1, seed=0, signal=1.5, widths=WIDTHS):
    """Views sharing a planted latent: feature block ``[3j, 3j + 4)`` of every view loads on component ``j``."""
    r = np.random.default_rng(seed)
    z = r.standard_normal((n, k))
    pats, vs = [], []
    for p in widths:
        w = np.zeros((k, p))
        for j in range(k):
            w[j, 3 * j:3 * j + 4] = 1.0
        pats.append(w)
        vs.append(torch.tensor(z @ w * signal + r.standard_normal((n, p)), dtype=torch.float32))
    return vs, pats


def with_missing(vs, block=30, single=50):
    """View 1 unobserved for the first ``block`` rows; view 2 unobserved for row ``single``."""
    out = [v.clone() for v in vs]
    out[1][:block] = float("nan")
    out[2][single] = float("nan")
    return out


def fit_mask(vs, k, seed=0):
    return pysimlr.fit("simlr", vs, k, missing_rows="mask", random_state=seed)


def span_r2(col, pats, view):
    """R^2 of a map column on the planted patterns of ``view`` (sign- and rotation-free)."""
    a = np.column_stack([np.ones(len(col)), pats[view].T])
    beta, *_ = np.linalg.lstsq(a, col, rcond=None)
    res = col - a @ beta
    return 1.0 - res.var() / col.var()


@pytest.mark.parametrize("k", [1, 2])
def test_reproducer_maps_are_nonzero_and_track_the_planted_pattern(k):
    vs, pats = make_views(k=k)
    vm = with_missing(vs)
    res = fit_mask(vm, k)
    maps = pysimlr.feature_maps(res, vm, target="other_views")
    for kind in FEATURE_MAP_KINDS:
        for m, t in enumerate(maps[kind]):
            assert t.shape == (WIDTHS[m], k)
            assert torch.isfinite(t).all(), (kind, m)
            assert float(t.abs().max()) > 0.05, f"{kind} view {m} is all zeros"
    for m, t in enumerate(maps["structure"]):
        r2 = [span_r2(t[:, j].numpy(), pats, m) for j in range(k)]
        assert min(r2) > 0.5, (m, r2)


@pytest.mark.parametrize("target", ["other_views", "consensus"])
@pytest.mark.parametrize("kind", FEATURE_MAP_KINDS)
def test_every_kind_and_target_under_missing_rows(kind, target):
    vs, _ = make_views(k=2)
    vm = with_missing(vs)
    res = fit_mask(vm, 2)
    maps, info = pysimlr.feature_maps(res, vm, kinds=[kind], target=target, return_info=True)
    n_obs = [int((~torch.isnan(v).all(1)).sum()) for v in vm]
    assert info["partial"] and info["n_eff"] == n_obs        # every observed row has another view observed
    for m, t in enumerate(maps[kind]):
        assert torch.isfinite(t).all() and float(t.abs().max()) > 0.05, (kind, target, m)
        assert (info["n_eff_feature"][m] == n_obs[m]).all()
        assert info["reasons"][kind][m] is None


def assert_maps_close(a, b, atol=1e-5):
    assert a.keys() == b.keys()
    for kind in a:
        for x, y in zip(a[kind], b[kind]):
            np.testing.assert_allclose(x.numpy(), y.numpy(), atol=atol, rtol=1e-4, err_msg=kind)


def test_equals_complete_case_for_two_views():
    vs, _ = make_views(k=2, widths=(12, 10))
    vm = [vs[0].clone(), vs[1].clone()]
    vm[1][:30] = float("nan")
    res = fit_mask(vm, 2)
    masked = pysimlr.feature_maps(res, vm)
    complete = pysimlr.feature_maps(res, [v[30:] for v in vm])
    assert_maps_close(masked, complete)
    # the other-views score of the observed view is NaN on the 30 rows too, so both views use the same 90 rows
    _, info = pysimlr.feature_maps(res, vm, return_info=True)
    assert info["n_eff"] == [90, 90]


def test_equals_complete_case_when_rows_are_absent_from_every_view():
    """Rows unobserved in all views are dropped; four views, every kind, `loadings` too."""
    vs, _ = make_views(k=2)
    res = fit_mask(vs, 2)                                    # mask mode on complete rows
    drop = np.arange(0, 20)
    keep = np.setdiff1d(np.arange(120), drop)
    blank = [v.clone() for v in vs]
    for v in blank:
        v[drop] = float("nan")
    assert_maps_close(pysimlr.feature_maps(res, blank), pysimlr.feature_maps(res, [v[keep] for v in vs]))
    assert_maps_close(pysimlr.feature_maps(res, blank, target="consensus", u=res["u"]),
                      pysimlr.feature_maps(res, [v[keep] for v in vs], target="consensus", u=res["u"][keep]))
    a = pysimlr.loadings(res, blank, u=res["u"])
    b = pysimlr.loadings(res, [v[keep] for v in vs], u=res["u"][keep])
    for x, y in zip(a["structure"], b["structure"]):
        np.testing.assert_allclose(x.numpy(), y.numpy(), atol=1e-5)
    for x, y in zip(a["loadings"], b["loadings"]):
        np.testing.assert_allclose(x.numpy(), y.numpy(), atol=1e-5)


def test_row_observed_in_one_view_only_and_row_observed_nowhere():
    vs, _ = make_views(k=1)
    res = fit_mask(vs, 1)
    one = [v.clone() for v in vs]
    for v in one[1:]:
        v[70] = float("nan")                                  # row 70: view 0 only
    for v in one:
        v[71] = float("nan")                                  # row 71: nowhere
    maps, info = pysimlr.feature_maps(res, one, return_info=True)
    # row 70 enters no map: view 0 has no other-views target there, views 1-3 are unobserved there; row 71 nowhere
    assert info["n_eff"] == [118, 118, 118, 118]
    for kind in FEATURE_MAP_KINDS:
        for t in maps[kind]:
            assert torch.isfinite(t).all() and float(t.abs().max()) > 0.05
    drop = [v[np.setdiff1d(np.arange(120), [70, 71])] for v in vs]
    # scores are standardized on each view's observed rows (view 0 keeps row 70), so close but not identical
    ref = pysimlr.feature_maps(res, drop)
    for kind in FEATURE_MAP_KINDS:
        for x, y in zip(maps[kind], ref[kind]):
            assert float((x - y).abs().max()) < 0.05, kind


def test_min_rows_guard_and_constant_columns_are_nan_with_reason_not_zero():
    vs, _ = make_views(k=1)
    res = fit_mask(vs, 1)
    few = [v.clone() for v in vs]
    few[1][5:] = float("nan")                                 # view 1 observed on 5 rows only
    with pytest.warns(MissingRowsWarning, match="only 5 rows"):
        maps, info = pysimlr.feature_maps(res, few, return_info=True)
    assert info["n_eff"][1] == 5
    for kind in FEATURE_MAP_KINDS:
        assert torch.isnan(maps[kind][1]).all()
        assert "min_rows=10" in info["reasons"][kind][1]
        assert torch.isfinite(maps[kind][0]).all() and info["reasons"][kind][0] is None
    maps2, info2 = pysimlr.feature_maps(res, few, min_rows=5, kinds=["structure"], return_info=True)
    assert torch.isfinite(maps2["structure"][1]).all() and info2["reasons"]["structure"][1] is None
    # a feature that is constant on the observed rows: its row is NaN (not 0), the others are computed
    const = [v.clone() for v in vs]
    const[1][:30] = float("nan")
    const[1][30:, 3] = 7.0
    with pytest.warns(MissingRowsWarning, match="constant"):
        maps3, info3 = pysimlr.feature_maps(res, const, kinds=["structure", "partial"], return_info=True)
    for kind in ("structure", "partial"):
        t = maps3[kind][1]
        assert torch.isnan(t[3]).all() and np.isfinite(np.delete(t.numpy(), 3, 0)).all()
    assert info3["n_constant_features"][1] == 1 and info3["n_eff_feature"][1][3] == 0
    with pytest.raises(ValueError, match="min_rows"):
        pysimlr.feature_maps(res, vs, min_rows=2)


def test_structure_coefficients_refuse_a_view_with_fewer_than_three_observed_rows():
    u = torch.randn(10, 1)
    x = torch.randn(10, 3)
    with pytest.raises(ValueError, match="at least 3"):
        structure_coefficients([x], u, obs_masks=[torch.tensor([True, True] + [False] * 8)])


def test_masked_coupling_matches_the_formula_and_is_nan_not_zero_when_undefined():
    r = np.random.default_rng(0)
    base = r.standard_normal((40, 2))
    z = [(base + 0.7 * r.standard_normal((40, 2))) for _ in range(3)]
    z = [(a - a.mean(0)) / a.std(0) for a in z]
    coup, n_used = _masked_coupling(z)
    assert n_used == [40, 40, 40]
    tot = sum(z)
    ref = np.mean([[np.corrcoef(zm[:, j], (tot - zm)[:, j])[0, 1] for j in range(2)] for zm in z], axis=0)
    np.testing.assert_allclose(coup, ref, atol=1e-10)
    z_nan = [a.copy() for a in z]
    z_nan[0][:30] = np.nan
    z_nan[1][30:] = np.nan                                    # views 0 and 1 share no row; view 2 sees both
    coup, n_used = _masked_coupling(z_nan)
    assert n_used == [10, 30, 40] and np.isfinite(coup).all()
    z_only = [z[0].copy(), z[1].copy()]
    z_only[0][:] = np.nan
    coup, n_used = _masked_coupling(z_only)
    assert np.isnan(coup).all() and n_used == [0, 0]


def test_loading_inference_with_missing_rows_on_signal_and_on_noise():
    vs, _ = make_views(n=120, k=1, seed=3, signal=1.6)
    vm = with_missing(vs)
    kw = dict(n_subsamples=6, random_state=0, missing_rows="mask", iterations=60,
              maps=("structure", "partial"), n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sig = pysimlr.loading_inference("simlr", vm, 1, **kw)
    assert sig["oos_coupling"] is not None and np.isfinite(sig["oos_coupling"]).all()
    assert sig["oos_coupling"][0] > 0.4
    assert len(sig["oos_coupling_n"]) == 4 and min(sig["oos_coupling_n"]) >= 3
    for kind in ("structure", "partial"):
        e = sig["maps"][kind]
        assert e["n_eff"] == [120, 90, 119, 120]
        for mp in e["map"]:
            assert torch.isfinite(mp).all() and float(mp.abs().max()) > 0.05
    # noise: independent views
    r = np.random.default_rng(4)
    noise = [torch.tensor(r.standard_normal((120, p)), dtype=torch.float32) for p in WIDTHS]
    noise = with_missing(noise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nul = pysimlr.loading_inference("simlr", noise, 1, **kw)
    assert np.isfinite(nul["oos_coupling"]).all()
    assert abs(nul["oos_coupling"][0]) < 0.25 < sig["oos_coupling"][0]


@pytest.mark.parametrize("fn", ["feature_relevance", "component_attribution"])
def test_complete_row_functions_refuse_unobserved_rows_instead_of_returning_zeros(fn):
    vs, _ = make_views(k=1)
    vm = with_missing(vs)
    res = fit_mask(vm, 1)
    y = np.random.default_rng(1).standard_normal(120)
    with pytest.raises(ValueError, match="unobserved"):
        if fn == "feature_relevance":
            pysimlr.feature_relevance(res, vm, None, outcome=y, n_permutations=10, n_refits=0, n_jobs=1)
        else:
            pysimlr.component_attribution(res, vm, [v[:40] for v in vm], y, y[:40], n_refits=2, n_jobs=1)


def _reference_quantities():
    """Values the complete-data code paths produce; the stored copy was generated on `main` (v0.31.0)."""
    vs, _ = make_views(n=100, k=2, seed=7)
    out = {}
    for tag, extra in (("plain", {}), ("mask", {"missing_rows": "mask"})):
        res = pysimlr.fit("simlr", vs, 2, random_state=0, **extra)
        for target in ("other_views", "consensus"):
            maps = pysimlr.feature_maps(res, vs, target=target)
            for kind, lst in maps.items():
                for m, t in enumerate(lst):
                    out[f"{tag}/{target}/{kind}/{m}"] = t.numpy()
        for m, c in enumerate(res["structure"]):
            out[f"{tag}/structure/{m}"] = c.numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inf = pysimlr.loading_inference("simlr", vs, 2, n_subsamples=4, random_state=0, iterations=40,
                                        maps=("structure", "partial"), n_jobs=1)
    out["inf/oos_coupling"] = np.asarray(inf["oos_coupling"], float)
    out["inf/oos_coupling_splits"] = np.asarray(inf["oos_coupling_splits"], float)
    out["inf/oos_r2"] = np.asarray(inf["oos_r2"], float)
    for kind in ("structure", "partial"):
        for m, t in enumerate(inf["maps"][kind]["map"]):
            out[f"inf/{kind}/{m}"] = t.numpy()
        for m, t in enumerate(inf["maps"][kind]["ci_low"]):
            out[f"inf/{kind}_ci_low/{m}"] = t.numpy()
    return out


def test_no_change_without_missing_rows_against_values_computed_on_main():
    ref = np.load(REF_FILE)
    now = _reference_quantities()
    assert set(ref.files) == set(now)
    for key in ref.files:
        np.testing.assert_allclose(now[key], ref[key], atol=1e-5, rtol=1e-4, err_msg=key)
