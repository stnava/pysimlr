"""attribute_target, the view_coupling_signs fix and the small interpretation-tool gaps (synthetic data only)."""
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.loadings import _result_view_settings, view_coupling_signs
from pysimlr.stats import permutation_correlations
from pysimlr.utils import MissingRowsWarning

warnings.filterwarnings("ignore", category=UserWarning)


def _planted(n=160, seed=0, nuisance=False, noise=0.8):
    """Three views; the target rides on 4 features of view 1 (+), 3 of view 2 (-, a negatively coupled view)."""
    r = np.random.default_rng(seed)
    z = r.standard_normal(n)
    v0 = 0.9 * r.standard_normal((n, 18))
    if nuisance:
        g = r.standard_normal(n)
        v0 = v0 + 1.5 * g[:, None] * r.uniform(0.8, 1.2, 18)
    v1 = 0.9 * r.standard_normal((n, 20)) + z[:, None] * np.r_[np.ones(4), np.zeros(16)]
    v2 = 0.9 * r.standard_normal((n, 15)) - z[:, None] * np.r_[np.ones(3), np.zeros(12)]
    y = z + noise * r.standard_normal(n)
    return [v0, v1, v2], y


@pytest.fixture(scope="module")
def planted_fit():
    views, y = _planted()
    return views, y, pysimlr.attribute_target(views, y, n_null=3, random_state=0)


# ------------------------------------------------------------------------------- view_coupling_signs
def _two_view_negative(seed=0, n=300):
    r = np.random.default_rng(seed)
    z = r.standard_normal(n)
    a = z[:, None] * np.r_[np.ones(8), np.zeros(22)] + 0.7 * r.standard_normal((n, 30))
    b = -z[:, None] * np.r_[np.ones(8), np.zeros(22)] + 0.7 * r.standard_normal((n, 30))
    t = z + 0.3 * r.standard_normal(n)
    return [torch.tensor(x, dtype=torch.float32) for x in (a, b, t[:, None])]


def test_negatively_coupled_view_gets_minus_one_through_per_view_params():
    """Regression: per_view_params settings were ignored, so every view returned +1."""
    v = _two_view_negative()
    pv = {i: dict(positivity="conical", sparseness=0.8, energy_type="nc_abs") for i in range(3)}
    pv[2]["sparseness"] = None
    res = pysimlr.fit("simlr", v, 1, random_state=0, per_view_params=pv, initialization_type="pca", iterations=150)
    u = res["u"].double()[:, 0].numpy()
    coup = []
    for m in range(2):
        x = v[m].double()
        s = ((x - x.mean(0)) / x.std(0)) @ res["v"][m].double()
        coup.append(np.corrcoef(s[:, 0].numpy(), u)[0, 1])
    assert coup[0] * coup[1] < 0, "fixture must contain one negatively coupled view"
    signs = [float(s) for s in pysimlr.loadings(res, v)["coupling_signs"]]
    expected = [1.0 if c > 0 else -1.0 for c in coup]
    assert signs[:2] == expected
    assert [float(s) for s in res["coupling_signs"]][:2] == expected
    pos, et = _result_view_settings(res, 3)
    assert set(pos) == {"positive"} and set(et) == {"nc_abs"}


def test_global_kwargs_path_unchanged_and_signed_energy_stays_plus_one():
    v = _two_view_negative()
    r1 = pysimlr.fit("simlr", v, 1, random_state=0, positivity="conical", energy_type="nc_abs", sparseness=0.8,
                     initialization_type="pca", iterations=100)
    assert float(pysimlr.loadings(r1, v)["coupling_signs"][1]) == -1.0
    r2 = pysimlr.fit("simlr", v, 1, random_state=0, positivity="either", energy_type="nc", sparseness=0.8,
                     initialization_type="pca", iterations=100)
    assert all(float(s) == 1.0 for s in pysimlr.loadings(r2, v)["coupling_signs"])


def test_view_coupling_signs_accepts_per_view_settings():
    c = [torch.tensor([[-0.8], [-0.7], [0.1]]), torch.tensor([[-0.8], [-0.7], [0.1]])]
    s = view_coupling_signs(c, None, positivity=["positive", "either"], energy_type=["nc_abs", "nc_abs"])
    assert [float(x) for x in s] == [-1.0, 1.0]
    with pytest.raises(ValueError):
        view_coupling_signs(c, None, positivity=["positive"], energy_type="nc_abs")


# ----------------------------------------------------------------------------------- attribute_target
def test_planted_support_is_recovered_and_signed_per_view(planted_fit):
    views, y, out = planted_fit
    m0, m1, m2 = out["maps"]
    assert out["mode"] == "fit" and len(out["maps"]) == 3 and out["map"].shape[0] == 18 + 20 + 15
    assert (m1[:4] > 0.3).all() and np.abs(m1[4:]).max() < np.abs(m1[:4]).min()
    assert (m2[:3] < -0.3).all()          # positive = rises with the target: the negatively built view is negative
    assert np.abs(m0).max() < 0.35
    assert out["view_signs"][2] == -1.0 and out["view_signs"][1] == 1.0
    for v in range(1, 3):
        truth = np.array([np.corrcoef(views[v][:, j], y)[0, 1] for j in range(views[v].shape[1])])
        assert np.corrcoef(out["maps"][v], truth)[0, 1] > 0.9


def test_score_flag_and_keys(planted_fit):
    _, _, out = planted_fit
    sc = out["score"]
    assert sc["kind"] == "out_of_fold" and sc["predictive"] and sc["r"] > sc["bar"] + 0.2
    assert np.isfinite(sc["oof_scores"]).all()
    for k in ("maps", "map", "feature_names", "weights", "view_signs", "target_sign", "score", "generic", "uncertainty",
              "inference", "result", "frame", "rows_used", "mode", "method", "settings"):
        assert k in out


def test_null_target_is_flagged_not_predictive():
    views, _ = _planted(seed=3)
    y = np.random.default_rng(9).standard_normal(len(views[0]))
    with pytest.warns(UserWarning, match="does not predict"):
        out = pysimlr.attribute_target(views, y, random_state=0)
    assert not out["score"]["predictive"]


def test_generic_variance_diagnostic_with_pc1_aligned_nuisance():
    """A view dominated by a global factor: with no signal in the target, the map follows that factor and the
    diagnostic says so; the null is estimated from permuted-target refits."""
    r = np.random.default_rng(4)
    n = 200
    g = r.standard_normal(n)
    views = [g[:, None] * r.uniform(0.2, 1.5, 25) + 0.5 * r.standard_normal((n, 25)),
             r.standard_normal((n, 12)) + 0.2 * g[:, None]]
    y = 0.5 * g + r.standard_normal(n)      # weak, PC1-aligned association
    out = pysimlr.attribute_target(views, y, n_null=3, random_state=0)
    gen = out["generic"]
    assert gen["n_null"] == 3
    v0 = gen["views"][0]
    assert v0["map_pc1_absr"] > 0.8 and v0["latent_pc1_absr"] > 0.5
    assert v0["null_map_pc1_absr"] is not None and np.isfinite(v0["map_excess"])
    no_null = pysimlr.attribute_target(views, y, random_state=0)["generic"]["views"][0]
    assert no_null["null_map_pc1_absr"] is None and no_null["generic"] is None
    # planted off-PC1 signal: not generic
    pv, py = _planted(nuisance=True)
    og = pysimlr.attribute_target(pv, py, n_null=3, random_state=0)["generic"]["views"]
    assert og[0]["map_pc1_absr"] < og[1]["map_pc1_absr"] or not og[1]["generic"]
    assert og[1]["generic"] is False


def test_groups_remove_group_mean_differences():
    r = np.random.default_rng(5)
    n = 160
    grp = np.repeat(["a", "b"], n // 2)
    shift = np.where(grp == "a", 0.0, 3.0)
    z = r.standard_normal(n)
    v0 = r.standard_normal((n, 14)) + shift[:, None] * np.r_[np.ones(7), np.zeros(7)]     # confounded by group only
    v1 = r.standard_normal((n, 14)) + z[:, None] * np.r_[np.ones(4), np.zeros(10)]
    y = z + 0.6 * r.standard_normal(n) + shift
    pooled = pysimlr.attribute_target([v0, v1], y, random_state=0)
    within = pysimlr.attribute_target([v0, v1], y, groups=grp, n_null=2, random_state=0)
    assert np.abs(pooled["maps"][0][:7]).mean() > 0.4          # the group difference alone looks like an association
    assert np.abs(within["maps"][0][:7]).mean() < 0.5 * np.abs(pooled["maps"][0][:7]).mean()
    assert (within["maps"][1][:4] > 0.3).all()
    uni = pysimlr.attribute_target([v0, v1], y, groups=grp, method="univariate", n_null=50, random_state=0)
    assert uni["inference"][1]["p_perm"][:4].max() < 0.1 and uni["inference"][0]["p_perm"].min() > 0.01


def test_missing_rows_target_and_unobserved_view_rows():
    views, y = _planted(n=150, seed=7)
    y = y.copy()
    y[:10] = np.nan
    views = [v.copy() for v in views]
    views[0][20:35] = np.nan
    with pytest.warns(MissingRowsWarning, match="dropped 10"):
        out = pysimlr.attribute_target(views, y, random_state=0)
    assert out["n_used"] == 140 and not out["rows_used"][:10].any()
    assert np.isfinite(out["map"]).all() and (out["maps"][1][:4] > 0.25).all()
    bad = [v.copy() for v in views]
    bad[1][3, 2] = np.nan
    with pytest.raises(ValueError, match="partially missing"):
        pysimlr.attribute_target(bad, np.nan_to_num(y), random_state=0)


def test_covariates_are_adjusted_out():
    r = np.random.default_rng(8)
    n = 160
    age = r.standard_normal(n)
    z = r.standard_normal(n)
    v0 = r.standard_normal((n, 12)) + age[:, None] * np.r_[np.ones(6), np.zeros(6)]       # follows age only
    v1 = r.standard_normal((n, 12)) + z[:, None] * np.r_[np.ones(4), np.zeros(8)]
    y = z + age + 0.5 * r.standard_normal(n)
    raw = pysimlr.attribute_target([v0, v1], y, method="univariate")
    adj = pysimlr.attribute_target([v0, v1], y, covariates=age, method="univariate")
    assert raw["maps"][0][:6].mean() > 0.3 and abs(adj["maps"][0][:6].mean()) < 0.15
    assert adj["maps"][1][:4].mean() > 0.3


def test_frozen_reference_projects_small_sample(planted_fit):
    views, y, ref = planted_fit
    small = slice(0, 30)
    out = pysimlr.attribute_target([v[small] for v in views], y[small], reference=ref)
    assert out["mode"] == "reference" and out["score"]["kind"] == "frozen_reference" and out["score"]["predictive"]
    assert (out["maps"][1][:4] > 0.2).all() and (out["maps"][2][:3] < -0.2).all()
    lo, hi = out["uncertainty"]["ci_low"][1], out["uncertainty"]["ci_high"][1]
    assert (lo <= out["maps"][1]).all() and (out["maps"][1] <= hi).all()
    # the same small sample refitted alone is a different (noisier) object; the frozen one does not refit
    raw = pysimlr.attribute_target([v[small] for v in views], y[small], reference=ref["result"])
    assert raw["mode"] == "reference" and np.corrcoef(raw["map"], out["map"])[0, 1] > 0.95
    with pytest.raises(ValueError, match="n_bag"):
        pysimlr.attribute_target([v[small] for v in views], y[small], reference=ref, n_bag=3)
    with pytest.raises(ValueError, match="covariate"):
        pysimlr.attribute_target([v[small] for v in views], y[small], reference=ref["result"], covariates=y[small])


def test_bagged_uncertainty_with_calibrated_selection():
    views, y = _planted(n=120, seed=2)
    out = pysimlr.attribute_target(views, y, n_bag=6, n_null=2, random_state=0)
    u = out["uncertainty"]
    assert u["kind"] == "delete_d_rescaled" and u["delete_d"] == 120 - u["subsample_rows"]
    assert len(u["mean"]) == 3 and all((lo <= hi).all() for lo, hi in zip(u["ci_low"], u["ci_high"]))
    assert u["n_selected"] is not None and u["expected_false_selections"] <= u["fdr"] * max(u["n_selected"], 1) + 1e-9
    assert sum(int(s.sum()) for s in u["selected"]) == u["n_selected"]
    off = pysimlr.attribute_target(views, y, random_state=0)
    assert off["uncertainty"] is None


def test_deterministic_given_seed():
    views, y = _planted(n=100, seed=11)
    a = pysimlr.attribute_target(views, y, random_state=3)
    b = pysimlr.attribute_target(views, y, random_state=3)
    np.testing.assert_array_equal(a["map"], b["map"])


def test_argument_validation():
    views, y = _planted(n=60)
    with pytest.raises(ValueError, match="method"):
        pysimlr.attribute_target(views, y, method="weights")
    with pytest.raises(ValueError, match="one value per row"):
        pysimlr.attribute_target(views, y[:10])
    with pytest.raises(ValueError, match="n_bag"):
        pysimlr.attribute_target(views, y, method="univariate", n_bag=2)


# ------------------------------------------------------------------------------------------ small gaps
def test_feature_maps_return_target():
    views, y = _planted(n=100, seed=1)
    t = [torch.tensor(v, dtype=torch.float32) for v in views]
    res = pysimlr.fit("simlr", t, 1, random_state=0, iterations=60)
    base = pysimlr.feature_maps(res, t, kinds=("structure",))
    maps, extra = pysimlr.feature_maps(res, t, kinds=("structure",), return_target=True)
    for a, b in zip(base["structure"], maps["structure"]):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert len(extra["target"]) == 3 and extra["target"][0].shape == (100, 1)
    # the target of view 0 is the mean of the other views' scores, so z-scored it reproduces the map
    z = extra["target"][0][:, 0]
    x0 = np.asarray(torch.as_tensor(views[0]))
    x0 = (x0 - x0.mean(0)) / x0.std(0)
    c = (x0 - x0.mean(0)).T @ (z - z.mean()) / (len(z) * x0.std(0) * z.std())
    np.testing.assert_allclose(np.abs(c), np.abs(maps["structure"][0].numpy()[:, 0]), atol=1e-5)
    both, ex2 = pysimlr.feature_maps(res, t, kinds=("structure",), return_target=True, return_info=True)
    assert "target" in ex2 and "n_eff" in ex2


def test_feature_maps_return_target_masked_rows():
    views, y = _planted(n=100, seed=1)
    t = [torch.tensor(v, dtype=torch.float32) for v in views]
    t[1][:12] = float("nan")
    res = pysimlr.fit("simlr", t, 1, random_state=0, iterations=60, missing_rows="mask")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, extra = pysimlr.feature_maps(res, t, kinds=("structure",), return_target=True)
    assert np.isnan(extra["target"][1][:12]).all()
    assert extra["target"][1].shape == (100, 1) and np.isfinite(extra["target"][0][50:]).all()


def test_loading_inference_separate_draw_seed():
    views, y = _planted(n=60, seed=1)
    t = [torch.tensor(v, dtype=torch.float32) for v in views]
    kw = dict(n_subsamples=3, iterations=30, return_refits=("rows",))
    base = pysimlr.loading_inference("simlr", t, 1, random_state=1, **kw)
    same = pysimlr.loading_inference("simlr", t, 1, random_state=1, draw_random_state=1, **kw)
    diff = pysimlr.loading_inference("simlr", t, 1, random_state=1, draw_random_state=2, **kw)
    r = lambda o: np.concatenate([np.asarray(x) for x in o["refits"]["rows"]])
    np.testing.assert_array_equal(r(base), r(same))
    assert not np.array_equal(r(base), r(diff))
    np.testing.assert_allclose(np.asarray(base["structure"][0]), np.asarray(diff["structure"][0]))   # reference fit unchanged


def test_permutation_correlations_drops_masked_rows_with_warning():
    r = np.random.default_rng(0)
    x = r.standard_normal((40, 5))
    y = x[:, 0] + r.standard_normal(40)
    ref = permutation_correlations(x[5:], y[5:], n_perms=30, seed=1)
    xm, ym = x.copy(), y.copy()
    xm[:3] = np.nan
    ym[3:5] = np.nan
    with pytest.warns(MissingRowsWarning, match="dropped 5"):
        got = permutation_correlations(xm, ym, n_perms=30, seed=1)
    np.testing.assert_allclose(got["r"], ref["r"])
    np.testing.assert_allclose(got["p_perm"], ref["p_perm"])
    grp = np.repeat([0, 1], 20)
    with pytest.warns(MissingRowsWarning):
        g = permutation_correlations(xm, ym, n_perms=20, seed=1, groups=grp)
    assert g["r"].shape == (5,)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        permutation_correlations(x, y, n_perms=10, seed=1)       # no masked rows: no warning
    with pytest.raises(ValueError):
        permutation_correlations(x, np.full(40, np.nan), n_perms=10, seed=1)
