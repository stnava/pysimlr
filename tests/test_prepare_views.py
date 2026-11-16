"""Tests for `pysimlr.prepare_views` / `apply_prepared` (synthetic data only)."""
import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import apply_prepared, prepare_views

N = 90


def make(seed=0, n=N):
    """Two views; view 'a' is mostly nuisance (80% of its variance follows the nuisance), view 'b' is not."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n, 2))
    z = rng.standard_normal((n, 1))
    load = rng.standard_normal((2, 12))
    a = 3.0 * (c @ load) + z @ rng.standard_normal((1, 12)) + 0.5 * rng.standard_normal((n, 12))
    a = a * rng.uniform(0.5, 3.0, 12) + rng.uniform(-5, 5, 12)
    b = z @ rng.standard_normal((1, 8)) + rng.standard_normal((n, 8))
    return {"a": a, "b": b}, {"a": c}


def test_order_matters_skipping_restandardize_leaves_small_sd():
    v, nu = make()
    on = prepare_views(v, nuisance=nu)
    off = prepare_views(v, nuisance=nu, restandardize=False)
    sd_on = np.nanstd(on["views"]["a"], axis=0)
    sd_off = np.nanstd(off["views"]["a"], axis=0)
    # Criterion: with the second standardization every column has SD 1; without it the nuisance-heavy view's
    # median column SD is well below 1 (it lost most of its variance), and the other view is untouched.
    assert np.allclose(sd_on, 1.0, atol=1e-6)
    assert np.median(sd_off) < 0.6
    assert on["variance_removed"]["a"] > 0.5 and on["variance_removed"]["b"] == 0.0
    assert np.allclose(np.nanstd(off["views"]["b"], axis=0), 1.0)
    assert any("NOT standardized again" in r for r in off["record"])


def test_parameters_come_only_from_fit_rows():
    v, nu = make()
    fit = np.arange(N) < 60
    base = prepare_views(v, nuisance=nu, fit_rows=fit)
    v2 = {k: x.copy() for k, x in v.items()}
    v2["a"][60:] = v2["a"][60:] * 10 + 7          # change the non-fit rows
    v2["b"][60:, 0] = np.nan
    nu2 = {"a": nu["a"].copy()}
    nu2["a"][60:] = 99.0
    alt = prepare_views(v2, nuisance=nu2, fit_rows=fit)
    for k in v:
        assert np.array_equal(base["views"][k][:60], alt["views"][k][:60])
    assert base["variance_removed"] == alt["variance_removed"]
    assert np.allclose(base["views"]["b"][:60].mean(0), 0.0, atol=1e-10)


def test_apply_prepared_reproduces_fit_rows_and_handles_new_rows():
    v, nu = make()
    v["a"][3, 2] = np.nan
    v["b"][5] = np.nan
    fit = np.arange(N) < 60
    signs = {"a": np.where(np.arange(12) % 2, -1.0, 1.0)}
    p = prepare_views(v, nuisance=nu, fit_rows=fit, signs=signs)
    again = apply_prepared(p["params"], v)
    for k in v:
        assert np.array_equal(again[k], p["views"][k], equal_nan=True)
    new = {k: x[60:] for k, x in v.items()}
    out = apply_prepared(p["params"], new, nuisance={"a": nu["a"][60:]})
    for k in v:
        assert np.allclose(out[k], p["views"][k][60:], atol=1e-12, equal_nan=True)
    with pytest.raises(ValueError, match="columns"):
        apply_prepared(p["params"], {"a": new["a"][:, :5], "b": new["b"]})
    with pytest.raises(ValueError, match="names"):
        apply_prepared(p["params"], {"a": new["a"], "c": new["b"]})
    with pytest.raises(ValueError, match="mapping"):
        apply_prepared(p["params"], [new["a"], new["b"]])


def test_whole_view_missing_rows_stay_nan_and_isolated_gaps_are_filled():
    v, nu = make()
    v["a"][[4, 70]] = np.nan                      # whole view missing (one fit row, one non-fit row)
    v["a"][10, 3] = np.nan                        # isolated gap
    p = prepare_views(v, nuisance=nu, fit_rows=np.arange(N) < 60)
    a = p["views"]["a"]
    assert np.isnan(a[[4, 70]]).all()
    assert np.isfinite(np.delete(a, [4, 70], axis=0)).all()
    assert any("2 rows without the view" in r for r in p["record"])
    # the gap is filled with the fit-row median (before standardizing), so the standardized entry is tied to it
    p0 = prepare_views(v, fit_rows=np.arange(N) < 60)
    obs = np.isfinite(v["a"]).any(axis=1)
    fit_obs = (np.arange(N) < 60) & obs
    med = np.nanmedian(v["a"][fit_obs], axis=0)[3]
    filled = v["a"][fit_obs, 3].copy()
    filled[np.isnan(filled)] = med
    expect = (med - filled.mean()) / filled.std()
    assert np.isclose(p0["views"]["a"][10, 3], expect)


def test_max_missing_mask_and_empty_column_error():
    v, _ = make()
    v["a"][:40, 0] = np.nan                       # 40/90 missing in column 0
    v["a"][:5, 1] = np.nan                        # 5/90 in column 1
    p = prepare_views(v, max_missing=0.2)
    mask = p["feature_mask"]["a"]
    assert mask.shape == (12,) and not mask[0] and mask[1:].all()
    assert p["views"]["a"].shape == (N, 11) and p["views"]["b"].shape == (N, 8)
    assert p["feature_mask"]["b"].all()
    out = apply_prepared(p["params"], v)
    assert np.array_equal(out["a"], p["views"]["a"])
    v["a"][:, 4] = np.nan
    with pytest.raises(ValueError, match="no finite value"):
        prepare_views(v)
    assert prepare_views(v, max_missing=0.3)["feature_mask"]["a"].sum() == 10
    with pytest.raises(ValueError, match="drops every column"):
        prepare_views(v, max_missing=-0.1)


def test_constant_column_is_centered_not_scaled():
    v, _ = make()
    v["b"][:, 2] = 4.0
    p = prepare_views(v)
    col = p["views"]["b"][:, 2]
    assert np.all(col == 0.0) and p["views"]["b"].shape[1] == 8
    assert any("constant" in r for r in p["record"])
    assert np.allclose(np.nanstd(np.delete(p["views"]["b"], 2, axis=1), axis=0), 1.0)


def test_sd_convention_is_population_sd():
    v, _ = make()
    p = prepare_views(v)
    assert np.allclose(p["views"]["b"].std(axis=0, ddof=0), 1.0)
    assert not np.allclose(p["views"]["b"].std(axis=0, ddof=1), 1.0)


def test_list_mapping_tensor_and_single_inputs_agree():
    v, nu = make()
    signs = [np.ones(12), None]
    as_map = prepare_views(v, nuisance=nu, signs={"a": np.ones(12)})
    as_list = prepare_views([v["a"], v["b"]], nuisance=[nu["a"], None], signs=signs)
    as_idx = prepare_views([v["a"], v["b"]], nuisance={0: nu["a"]})
    tens = prepare_views([torch.as_tensor(v["a"]), torch.as_tensor(v["b"])], nuisance={"view0": nu["a"]})
    assert isinstance(as_map["views"], dict) and isinstance(as_list["views"], list)
    for i, k in enumerate("ab"):
        assert np.array_equal(as_map["views"][k], as_list["views"][i])
        assert np.array_equal(as_idx["views"][i], as_list["views"][i])
        assert np.array_equal(tens["views"][i], as_list["views"][i])
    assert as_list["variance_removed"][0] == as_map["variance_removed"]["a"]
    assert isinstance(prepare_views(v["b"])["views"], list)
    again = apply_prepared(as_list["params"], [v["a"], v["b"]])
    assert isinstance(again, list)


def test_signs_orient_columns():
    v, _ = make()
    s = np.ones(8)
    s[:3] = -1
    p = prepare_views(v, signs={"b": s})
    q = prepare_views(v)
    assert np.allclose(p["views"]["b"][:, :3], -q["views"]["b"][:, :3])
    assert np.allclose(p["views"]["b"][:, 3:], q["views"]["b"][:, 3:])
    with pytest.raises(ValueError, match="\\+1 or -1"):
        prepare_views(v, signs={"b": np.full(8, 2.0)})
    with pytest.raises(ValueError, match="entries"):
        prepare_views(v, signs={"b": np.ones(3)})


def test_determinism_and_input_not_modified():
    v, nu = make()
    before = {k: x.copy() for k, x in v.items()}
    p1 = prepare_views(v, nuisance=nu)
    p2 = prepare_views(v, nuisance=nu)
    for k in v:
        assert np.array_equal(p1["views"][k], p2["views"][k])
        assert np.array_equal(v[k], before[k])
    assert p1["record"] == p2["record"]


def test_bad_arguments_raise():
    v, nu = make()
    with pytest.raises(ValueError, match="unknown view key"):
        prepare_views(v, nuisance={"zzz": nu["a"]})
    with pytest.raises(ValueError, match="rows"):
        prepare_views(v, nuisance={"a": nu["a"][:10]})
    with pytest.raises(ValueError, match="fit_rows"):
        prepare_views(v, fit_rows=np.ones(5, bool))
    with pytest.raises(ValueError, match="fewer than 2"):
        prepare_views(v, fit_rows=np.arange(N) < 1)
    with pytest.raises(ValueError, match="different numbers of rows"):
        prepare_views({"a": v["a"], "b": v["b"][:10]})
    idx = prepare_views(v, fit_rows=np.arange(60))
    assert idx["fit_rows"].sum() == 60


def test_prepared_views_fit_with_scale_none():
    v, nu = make()
    v["a"][7] = np.nan
    p = prepare_views(v, nuisance=nu)
    res = pysimlr.fit("simlr", p["views"], 1, scale_list=["none"], missing_rows="mask", iterations=5)
    assert np.isfinite(res["best_energy"])
