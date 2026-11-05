"""The confound sink (`pysimlr.fit(method, views, k, confounds=...)`), for every method family.

Every claim is checked where it can fail: held-out decorrelation and
biological recovery come from `pysimlr.predict` on rows the fit never saw,
the permutation null is checked for calibration, and the downstream tools are
run so that a confound matrix that does not follow its rows raises.
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

import pysimlr
from pysimlr.api import fit, predict
from pysimlr.attribution import component_attribution
from pysimlr.confounds import (apply_view_sink, confound_decorrelation, fit_confound_sink, max_abs_corr,
                               permute_view_confounds)
from pysimlr.loadings import loading_inference, select_rank
from pysimlr.relevance import feature_relevance
from pysimlr.utils import heldout_coupling

DIMS = (24, 30, 18)
#: method -> budget keywords small enough for the suite.
BUDGET = {
    "simlr": dict(iterations=30),
    "lend": dict(iterations=25),
    "ned": dict(iterations=100),     # 25 epochs leave NED unconverged (held-out R ~0.5-0.7 even without confounds)
    "ned_shared_private": dict(iterations=25),
    "flow_v": dict(iterations=25),
    "pca": {},
    "nsa_pipeline": {},
}


def _data(n_train=160, n_test=120, conf_amp=3.0, rho=0.0, seed=0):
    """Two shared biological factors ``z`` and two confounds ``c`` ``conf_amp`` times stronger.

    ``rho`` is the correlation between ``z[:, 0]`` and ``c[:, 0]`` (0: orthogonal).
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    z = rng.standard_normal((n, 2))
    c = rng.standard_normal((n, 2))
    c[:, 0] = rho * z[:, 0] + np.sqrt(1.0 - rho ** 2) * c[:, 0]
    views = []
    for p in DIMS:
        b = np.abs(rng.standard_normal((2, p)))
        a = rng.standard_normal((2, p))
        views.append((z @ b + conf_amp * c @ a + 0.5 * rng.standard_normal((n, p))).astype(np.float32))
    tr, te = slice(0, n_train), slice(n_train, n)
    return ([torch.tensor(v[tr]) for v in views], [torch.tensor(v[te]) for v in views],
            z[tr], z[te], c[tr], c[te])


@contextlib.contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def data():
    return _data()


# ----------------------------------------------------------------------------- contract
def test_no_confounds_leaves_the_contract_unchanged(data):
    xtr, xte, *_ = data
    res = fit("pca", xtr, 2)
    assert res["confound_sink"] is None
    pred = predict(res, xte)
    assert "u_sink" not in pred and "latents_sink" not in pred
    assert "confounds" not in res["parameters"]["used"]
    assert res["parameters"]["defaults"]["confounds"] is None


def test_every_method_accepts_the_sink_parameters():
    for name in pysimlr.method_names():
        params = pysimlr.describe_method(name)["parameters"]
        assert {"confounds", "confound_strength", "confound_rank"} <= set(params), name


@pytest.mark.parametrize("bad, match", [
    (dict(confound_strength=1.5), "confound_strength"),
    (dict(confound_strength=-0.1), "confound_strength"),
    (dict(confound_strength=True), "confound_strength"),
    (dict(confound_rank=0), "confound_rank"),
    (dict(confound_rank=1.5), "confound_rank"),
])
def test_invalid_sink_parameters_raise(data, bad, match):
    xtr, _, _, _, ctr, _ = data
    with pytest.raises(ValueError, match=match):
        fit("pca", xtr, 2, confounds=ctr, **bad)


def test_sink_parameters_without_confounds_raise(data):
    with pytest.raises(ValueError, match="confounds is None"):
        fit("pca", data[0], 2, confound_strength=0.5)


@pytest.mark.parametrize("conf, match", [
    (np.zeros((5, 2)), "rows"),
    ([np.zeros((160, 1))], "entries"),
    ({7: np.zeros((160, 1))}, "unknown view key"),
    ([None, None, None], "no confound matrix"),
    (np.ones((160, 3)), "constant or missing"),
    ("site", "must be an array"),
])
def test_malformed_confounds_raise(data, conf, match):
    with pytest.raises(ValueError, match=match):
        fit("pca", data[0], 2, confounds=conf)


def test_formats_nan_and_constant_columns_are_recorded(data):
    xtr, _, _, _, ctr, _ = data
    df = pd.DataFrame({"a": ctr[:, 0], "b": ctr[:, 1], "const": 1.0})
    df.loc[[0, 3, 7], "a"] = np.nan
    res = fit("pca", xtr, 2, confounds=df)
    info = res["confound_sink"]["info"][0]
    assert info["n_imputed"] == 3
    assert [d["reason"] for d in info["dropped_columns"]] == ["constant"]
    assert res["confound_sink"]["confound_rank"] == [2, 2, 2]
    one_d = fit("pca", xtr, 2, confounds=pd.Series(ctr[:, 0]))
    assert one_d["confound_sink"]["confound_rank"] == [1, 1, 1]


def test_per_view_list_and_dict_agree_and_skip_unconfounded_views(data):
    xtr, xte, _, _, ctr, _ = data
    as_list = fit("pca", xtr, 2, confounds=[ctr, None, ctr[:, :1]])
    as_dict = fit("pca", xtr, 2, confounds={0: ctr, "view2": ctr[:, :1]})
    assert torch.allclose(as_list["u"], as_dict["u"], atol=1e-6)
    assert as_list["confound_sink"]["confound_rank"] == [2, None, 1]
    assert as_list["confound_sink"]["views"][1] is None
    pred = predict(as_list, xte)
    assert pred["latents_sink"][1] is None and pred["latents_sink"][2].shape == (xte[0].shape[0], 1)
    assert pred["u_sink"] is None                       # per-view confounds have no shared sink score


def test_rank_truncation_warns_and_is_recorded(data):
    xtr, _, _, _, ctr, _ = data
    with pytest.warns(RuntimeWarning, match="truncated"):
        res = fit("pca", xtr, 2, confounds=ctr, confound_rank=1)
    info = res["confound_sink"]["info"][0]
    assert info["rank_truncated"] and info["rank"] == 1 and info["numerical_rank"] == 2
    assert {"lambda_rel", "df", "df_cap", "grid_extended"} <= set(info)
    assert info["df"] <= info["df_cap"]


# ----------------------------------------------------------------------------- every family
@pytest.mark.parametrize("method", list(BUDGET))
def test_every_family_cleans_held_out_rows_without_test_confounds(data, method):
    xtr, xte, ztr, zte, ctr, cte = data
    with _quiet():
        plain = fit(method, xtr, 2, random_state=0, **BUDGET[method])
        sink = fit(method, xtr, 2, random_state=0, confounds=ctr, **BUDGET[method])
    before = confound_decorrelation(plain, xte, cte)["max"]
    after = confound_decorrelation(sink, xte, cte)["max"]
    assert before > 0.6, f"{method}: fixture confound should dominate the unadjusted fit ({before:.3f})"
    assert after < 0.25, f"{method}: held-out max |r(u, C)| {after:.3f} after the sink"
    # Planted biology recovered on held-out rows: multiple R of each factor on predict(u).
    u_te = predict(sink, xte)["u"].double().numpy()
    design = np.column_stack([np.ones(len(u_te)), u_te])
    r = [np.corrcoef(design @ np.linalg.lstsq(design, zte[:, j], rcond=None)[0], zte[:, j])[0, 1]
         for j in range(2)]
    assert min(r) > 0.8, f"{method}: held-out biological multiple R {np.round(r, 3)}"
    rec = sink["confound_sink"]
    # result["u"] is the deployed estimator on the training rows.
    assert rec["u_predict_max_abs_diff"] < 1e-4
    assert torch.allclose(predict(sink, xtr)["u"], sink["u"], atol=1e-4)
    assert rec["train_max_abs_corr"] < 0.25
    used = sink["parameters"]["used"]
    assert used["confounds"] is ctr and used["confound_strength"] == 1.0 and used["confound_rank"] is None


@pytest.mark.parametrize("method", ["simlr", "pca", "lend"])
def test_single_subject_prediction_matches_the_batch(data, method):
    """Holds by construction (a fixed affine map per view); kept as a regression check, not evidence."""
    xtr, xte, _, _, ctr, _ = data
    with _quiet():
        res = fit(method, xtr, 2, random_state=0, confounds=ctr, **BUDGET[method])
    batch = predict(res, xte)
    for i in (0, 5, 17):
        one = predict(res, [x[i:i + 1] for x in xte])
        assert torch.allclose(one["u"], batch["u"][i:i + 1], atol=1e-5)
        assert torch.allclose(one["u_sink"], batch["u_sink"][i:i + 1], atol=1e-5)


def test_soft_strength_interpolates_held_out_decorrelation(data):
    xtr, xte, _, _, ctr, cte = data
    out = []
    for g in (0.0, 0.5, 1.0):
        res = fit("pca", xtr, 2, confounds=ctr, confound_strength=g)
        out.append(confound_decorrelation(res, xte, cte)["max"])
    assert out[0] > out[1] > out[2]
    assert out[0] > 0.6 and out[2] < 0.2


def test_sink_scores_estimate_held_out_confounds(data):
    xtr, xte, _, _, ctr, cte = data
    res = fit("pca", xtr, 2, confounds=ctr)
    s = predict(res, xte)["u_sink"].double().numpy()
    design = np.column_stack([np.ones(len(s)), s])
    r = [np.corrcoef(design @ np.linalg.lstsq(design, cte[:, j], rcond=None)[0], cte[:, j])[0, 1] for j in range(2)]
    assert min(r) > 0.9


def test_covarying_biology_is_kept_partly_at_soft_strength():
    """With biology correlated 0.6 with a confound, gamma < 1 keeps more of it than gamma = 1."""
    xtr, xte, _, zte, ctr, _ = _data(rho=0.6, seed=3)
    rec = {}
    for g in (0.5, 1.0):
        res = fit("pca", xtr, 2, confounds=ctr, confound_strength=g)
        u = predict(res, xte)["u"].double().numpy()
        rec[g] = max(abs(np.corrcoef(u[:, j], zte[:, 0])[0, 1]) for j in range(2))
    assert rec[0.5] > rec[1.0]


# ----------------------------------------------------------------------------- missing views and extension
def test_missing_view_prediction_is_cleaned(data):
    xtr, xte, _, _, ctr, cte = data
    res = fit("simlr", xtr, 2, random_state=0, confounds=ctr, iterations=30)
    with _quiet():
        sub = predict(res, {0: xte[0], 1: xte[1]})
        plain = fit("simlr", xtr, 2, random_state=0, iterations=30)
        sub_plain = predict(plain, {0: xte[0], 1: xte[1]})
    assert set(sub["latents_sink"]) == {"view0", "view1"} and sub["u_sink"].shape == (xte[0].shape[0], 2)
    assert max_abs_corr(sub["u"], cte)["max"] < 0.3 < max_abs_corr(sub_plain["u"], cte)["max"]


def test_extend_requires_and_uses_confounds(data):
    xtr, xte, _, _, ctr, cte = data
    rng = np.random.default_rng(9)
    n_tr = xtr[0].shape[0]
    full_c = np.vstack([ctr, cte])
    new = torch.tensor(np.vstack([xtr[0].numpy(), xte[0].numpy()]) @ np.abs(rng.standard_normal((24, 12)))
                       + 3.0 * full_c @ rng.standard_normal((2, 12))).float()
    res = fit("simlr", xtr, 2, random_state=0, confounds=ctr, iterations=30)
    with pytest.raises(ValueError, match="pass confounds"):
        pysimlr.extend(res, xtr, {"new": new[:n_tr]}, iterations=20, warn=False)
    with _quiet():
        ext = pysimlr.extend(res, xtr, {"new": new[:n_tr]}, iterations=20, warn=False, confounds=ctr)
        alone = predict(ext, {"new": new[n_tr:]})
    assert "confound_sink" in ext["extensions"]["new"]["provenance"]
    assert max_abs_corr(alone["u"], cte)["max"] < 0.3
    with _quiet():
        plain = fit("simlr", xtr, 2, random_state=0, iterations=20)
    with pytest.raises(ValueError, match="no confound sink"):
        pysimlr.extend(plain, xtr, {"new": new[:n_tr]}, confounds=ctr, warn=False)
    with pytest.raises(ValueError, match="pass confounds"):
        pysimlr.extend(res, xtr, {"new": new[:n_tr]}, how="reembed", k_new=2)


def test_probe_views_are_refused_with_a_sink(data):
    xtr, _, _, _, ctr, _ = data
    res = fit("simlr", xtr, 2, random_state=0, confounds=ctr, iterations=20)
    with pytest.raises(ValueError, match="probe_views are not cleaned"):
        pysimlr.feature_maps(res, xtr, kinds=("partial",), probe_views=[xtr[0], None, None])


# ----------------------------------------------------------------------------- permutation nulls
def test_operator_commutes_with_row_permutation(data):
    xtr, _, _, _, ctr, _ = data
    perm = np.random.default_rng(4).permutation(xtr[0].shape[0])
    clean, rec = fit_confound_sink(xtr, ctr)
    clean_p, rec_p = fit_confound_sink([x[perm] for x in xtr], ctr[perm])
    for a, b in zip(clean, clean_p):
        assert torch.allclose(a[perm], b, atol=1e-4)
    assert rec["confound_rank"] == rec_p["confound_rank"]


def test_null_refit_uses_the_observed_confound_rank(data):
    """A shared matrix permuted per view keeps each view's basis rank (the old pooled basis tripled it)."""
    xtr, _, _, _, ctr, _ = data
    n = xtr[0].shape[0]
    rng = np.random.default_rng(1)
    perms = [None, rng.permutation(n), rng.permutation(n)]
    nulls = [x if p is None else x[torch.as_tensor(p)] for x, p in zip(xtr, perms)]
    _, obs = fit_confound_sink(xtr, ctr)
    _, null = fit_confound_sink(nulls, permute_view_confounds(ctr, perms))
    assert obs["confound_rank"] == null["confound_rank"] == [2, 2, 2]
    clean_obs, _ = fit_confound_sink(xtr, ctr)
    clean_null, _ = fit_confound_sink(nulls, permute_view_confounds(ctr, perms))
    for m in (1, 2):
        assert torch.allclose(clean_obs[m][torch.as_tensor(perms[m])], clean_null[m], atol=1e-4)


def test_permutation_null_is_calibrated_with_confounds():
    """No shared biology, a shared confound in every view: the permutation p-value is ~uniform.

    Same refit path as `loading_inference` (per-view confound permutation);
    statistic: largest cross-view canonical-style correlation of the views'
    first-layer scores. 120 datasets x 19 permutations of the PCA baseline.
    """
    n_rep, n_perm, n = 120, 19, 120
    ps = []
    for rep in range(n_rep):
        rng = np.random.default_rng(1000 + rep)
        c = rng.standard_normal((n, 2))
        views = [torch.tensor(3.0 * c @ rng.standard_normal((2, p)) + rng.standard_normal((n, p))).float()
                 for p in (12, 10)]

        def stat(vs, conf):
            res = fit("pca", vs, 1, confounds=conf)
            z = [x for x in pysimlr.confounds.clean_raw_views(res, vs)]
            a = torch.linalg.svd(z[0] - z[0].mean(0), full_matrices=False)[0][:, :3]
            b = torch.linalg.svd(z[1] - z[1].mean(0), full_matrices=False)[0][:, :3]
            return float(torch.linalg.svdvals(a.t() @ b)[0])

        obs = stat(views, c)
        null = []
        for _ in range(n_perm):
            p1 = rng.permutation(n)
            null.append(stat([views[0], views[1][torch.as_tensor(p1)]], permute_view_confounds(c, [None, p1])))
        ps.append((1 + sum(v >= obs for v in null)) / (1 + n_perm))
    ps = np.asarray(ps)
    frac = float(np.mean(ps <= 0.1))
    # Binomial(120, 0.1): 99.9% interval about [0.025, 0.2].
    assert 0.025 <= frac <= 0.2, f"fraction of p <= 0.1 is {frac:.3f}"
    assert stats.kstest(ps, "uniform").pvalue > 1e-3


# ----------------------------------------------------------------------------- downstream tools
def test_downstream_tools_carry_confounds_with_their_rows(data):
    """Each tool refits on row subsets; a confound matrix that did not follow its rows would raise."""
    xtr, xte, _, _, ctr, _ = data
    with _quiet():
        inf = loading_inference("simlr", xtr, 2, n_subsamples=2, n_permutations=2, iterations=15,
                                random_state=0, confounds=ctr)
        assert inf["result"]["confound_sink"] is not None and np.isfinite(inf["null_max_abs"])
        sr = select_rank("pca", xtr, [1, 2], n_subsamples=2, confounds=ctr)
        assert set(sr["criteria"]) >= {"component", "oos_coupling"}
        res = fit("simlr", xtr, 2, random_state=0, confounds=ctr, iterations=20)
        hc = heldout_coupling(res, xte)
        assert len(hc["cross_view_heldout"]) == 2
        y = xtr[0][:, :3].double().numpy().sum(1)
        y_te = xte[0][:, :3].double().numpy().sum(1)
        rel = feature_relevance(res, xtr, xte, outcome=y, outcome_test=y_te, n_permutations=200,
                                n_permutations_test=200, n_refits=2, refit_permutations=100, n_null_refits=1,
                                n_jobs=1)
        assert len(rel["features"]) == sum(x.shape[1] for x in xtr)
        att = component_attribution(res, xtr, xte, y, y_te, n_refits=2, n_jobs=1)
        assert np.isfinite(att["full_r2"]) and len(att["refits"]["default"]) == 2


def test_attribution_scores_the_cleaned_views(data):
    xtr, xte, _, _, ctr, cte = data
    res = fit("simlr", xtr, 2, random_state=0, confounds=ctr, iterations=20)
    y, y_te = ctr[:, 0], cte[:, 0]
    att = component_attribution(res, xtr, xte, y, y_te, n_refits=1, n_jobs=1)
    # The outcome is the confound itself: cleaned scores cannot predict it on held-out rows.
    assert att["full_r2"] < 0.15, att["full_r2"]


def test_apply_view_sink_rejects_wrong_width(data):
    xtr, _, _, _, ctr, _ = data
    _, rec = fit_confound_sink(xtr, ctr)
    with pytest.raises(ValueError, match="columns"):
        apply_view_sink(xtr[1], rec["views"][0])
