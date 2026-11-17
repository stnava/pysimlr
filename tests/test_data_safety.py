"""Synthetic tests for `pysimlr.data_safety`, `prepare_views(..., safety=...)`, and `fit(..., safety=...)`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pysimlr


def _assert_reports_equal(r1: dict, r2: dict) -> None:
    assert r1["rules"] == r2["rules"]
    assert r1["view_names"] == r2["view_names"]
    assert np.array_equal(r1["fit_rows"], r2["fit_rows"])
    assert r1["warnings"] == r2["warnings"]
    assert r1["n_flagged_by_group"] == r2["n_flagged_by_group"]
    for key in ("mean_abs_z", "robust_z", "leverage_ratio", "md2"):
        c1, c2 = r1[key], r2[key]
        items = c1.keys() if isinstance(c1, dict) else range(len(c1))
        for k in items:
            np.testing.assert_allclose(c1[k], c2[k], rtol=0.0, atol=0.0, equal_nan=True)
    for k in (r1["flags"].keys() if isinstance(r1["flags"], dict) else range(len(r1["flags"]))):
        for rk in ("row_scale", "leverage", "md", "any"):
            assert np.array_equal(r1["flags"][k][rk], r2["flags"][k][rk])


def test_clean_data_no_gross_outlier_and_planted_row_scale_outlier_per_view():
    rng = np.random.default_rng(20261117)
    n = 180
    raw_clean = {
        "struct": rng.standard_normal((n, 80)),
        "func": rng.standard_normal((n, 60)),
        "cog": rng.standard_normal((n, 30)),
    }

    with warnings.catch_warnings(record=True) as caught_clean:
        warnings.simplefilter("always")
        prep_clean = pysimlr.prepare_views(raw_clean)
    safety_warns = [w for w in caught_clean if issubclass(w.category, pysimlr.DataSafetyWarning)]
    assert safety_warns == []
    rep_clean = prep_clean["data_safety"]
    assert rep_clean is not None
    assert rep_clean["flagged_rows"]["by_rule"]["row_scale"].size == 0
    for nm in ("struct", "func", "cog"):
        assert rep_clean["flagged_rows"]["by_view"][nm]["row_scale"].size == 0

    # Plant a +3.5 SD gross row outlier across features in "struct" row 17 only
    raw_bad = {k: v.copy() for k, v in raw_clean.items()}
    signs_row = rng.choice([-1.0, 1.0], size=raw_bad["struct"].shape[1])
    raw_bad["struct"][17] = 3.5 * signs_row
    raw_snapshot = {k: v.copy() for k, v in raw_bad.items()}

    with pytest.warns(pysimlr.DataSafetyWarning, match="struct") as caught_bad:
        prep_warn = pysimlr.prepare_views(raw_bad, safety="warn")
    assert len(caught_bad) == 1
    assert issubclass(caught_bad[0].category, RuntimeWarning)

    # Input and prepared data are never dropped or modified by safety checks
    for nm in raw_bad:
        np.testing.assert_array_equal(raw_bad[nm], raw_snapshot[nm])

    prep_off = pysimlr.prepare_views(raw_bad, safety="off")
    assert prep_off["data_safety"] is None
    for nm in raw_bad:
        np.testing.assert_allclose(prep_warn["views"][nm], prep_off["views"][nm], rtol=0.0, atol=0.0)

    rep = prep_warn["data_safety"]
    assert list(rep["flagged_rows"]["by_view"]["struct"]["row_scale"]) == [17]
    assert rep["flagged_rows"]["by_view"]["func"]["row_scale"].size == 0
    assert rep["flagged_rows"]["by_view"]["cog"]["row_scale"].size == 0
    assert list(rep["flagged_rows"]["by_rule"]["row_scale"]) == [17]
    assert rep["robust_z"]["struct"][17] > 5.0
    assert any("struct" in msg and "gross row outlier" in msg for msg in rep["warnings"])

    # Direct call on prepare_views dict reproduces prep_warn["data_safety"]
    rep_direct = pysimlr.data_safety(prep_warn)
    _assert_reports_equal(rep, rep_direct)

    with pytest.raises(ValueError, match="struct"):
        pysimlr.prepare_views(raw_bad, safety="raise")


def test_high_leverage_small_scale_row_flagged_by_rule2_or_3_not_rule1():
    rng = np.random.default_rng(42)
    n, p = 160, 48
    # Bulk variance sits on 3 strong factors (cols 0..35) plus 1 quiet factor on cols 36..39,
    # and independent unit noise on cols 40..47.
    f_strong = rng.standard_normal((n, 3))
    load_strong = np.zeros((3, p))
    load_strong[0, 0:12] = 3.0
    load_strong[1, 12:24] = 3.0
    load_strong[2, 24:36] = 3.0

    # Quiet PC4 direction concentrated on row 29 so row 29 dominates PC4 hat leverage & MCD distance,
    # while keeping its average |z| across all 48 columns right near the bulk median (~0.75).
    x = f_strong @ load_strong + 0.15 * rng.standard_normal((n, p))
    x[:, 40:48] = rng.standard_normal((n, 8))

    target_row = 29
    x[target_row, :] = 0.0
    x[target_row, 36:40] = np.array([1.8, -1.8, 1.8, -1.8])
    x[target_row, 0:36] = 0.55 * rng.choice([-1.0, 1.0], size=36)
    x[target_row, 40:48] = 0.55 * rng.choice([-1.0, 1.0], size=8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prep = pysimlr.prepare_views([x], safety="warn")
    # No Rule 1 gross outlier -> no DataSafetyWarning emitted by prepare_views
    assert [w for w in caught if issubclass(w.category, pysimlr.DataSafetyWarning)] == []

    rep = pysimlr.data_safety(prep["views"], n_pc=5)
    assert target_row not in set(rep["flagged_rows"]["by_view"][0]["row_scale"].tolist())
    assert abs(rep["robust_z"][0][target_row]) < 3.0
    flagged_2_or_3 = set(rep["flagged_rows"]["by_view"][0]["leverage"].tolist()) | set(
        rep["flagged_rows"]["by_view"][0]["md"].tolist()
    )
    assert target_row in flagged_2_or_3
    assert rep["flags"][0]["leverage"][target_row] or rep["flags"][0]["md"][target_row]
    assert not rep["flags"][0]["row_scale"][target_row]


def test_missing_views_fit_rows_columns_groups_and_determinism():
    rng = np.random.default_rng(99)
    n = 150
    v0 = rng.standard_normal((n, 24))
    v1 = rng.standard_normal((n, 20))

    # Whole-view missing rows in v1 must be ignored for v1
    missing_v1 = [3, 14, 88]
    v1[missing_v1] = np.nan

    # Column anomalies in v0:
    # col 0: constant
    v0[:, 0] = 5.0
    # col 1: near-constant raw SD (~1e-7)
    v0[:, 1] = 2.0 + 1e-7 * rng.standard_normal(n)
    # col 2: single extreme value on row 10
    v0[10, 2] = 14.0
    # col 3: isolated missing entries (10% on active fit rows)
    v0[np.arange(0, 30, 3), 3] = np.nan
    # col 5 == col 4 (exact duplicate), col 6 == -col 4 (opposite-sign, must NOT be marked duplicate)
    v0[:, 5] = v0[:, 4].copy()
    v0[:, 6] = -v0[:, 4].copy()

    # Held-out rows 120..149 have huge corruption, but fit_rows restricts diagnostics to 0..119
    fit_mask = np.arange(n) < 120
    v0[135] = 40.0
    v1[140] = -35.0

    # Plant 5 gross outliers concentrated inside group "site_B" (rows 40..64)
    groups = np.array(["site_A"] * 40 + ["site_B"] * 25 + ["site_C"] * 85, dtype=object)
    planted_b = [41, 44, 48, 52, 59]
    for r in planted_b:
        v1[r] = 4.5 * rng.choice([-1.0, 1.0], size=v1.shape[1])

    prep = pysimlr.prepare_views({"anat": v0, "dti": v1}, fit_rows=fit_mask, safety="off")
    rep1 = pysimlr.data_safety(prep, groups=groups)
    rep2 = pysimlr.data_safety(prep, groups=groups)
    _assert_reports_equal(rep1, rep2)

    # Whole-view missing rows in "dti" are NaN / False and never flagged
    for r in missing_v1:
        assert not rep1["active_rows"]["dti"][r]
        assert np.isnan(rep1["mean_abs_z"]["dti"][r])
        assert np.isnan(rep1["leverage_ratio"]["dti"][r])
        assert np.isnan(rep1["md2"]["dti"][r])
        assert not rep1["flags"]["dti"]["any"][r]
        assert r not in set(rep1["flagged_rows"]["by_view"]["dti"]["any"].tolist())

    # Rows outside fit_rows (120..149) are excluded and never flagged
    assert not np.any(rep1["flagged_rows"]["any"] >= 120)
    assert np.all(np.isnan(rep1["mean_abs_z"]["anat"][120:]))

    # Column findings in "anat"
    cols_anat = rep1["columns"]["anat"]
    assert 0 in set(cols_anat["constant"].tolist())
    assert 1 in set(cols_anat["near_constant"].tolist())
    assert 2 in set(cols_anat["single_extreme"].tolist())
    assert cols_anat["max_robust_z"][2] > 5.0
    assert np.isclose(cols_anat["missing_fraction"][3], 10.0 / 120.0)
    assert (4, 5) in cols_anat["duplicated"]
    assert (4, 6) not in cols_anat["duplicated"]
    assert (5, 6) not in cols_anat["duplicated"]

    # Group counts add up and detect concentration in "site_B"
    total_any = int(rep1["flagged_rows"]["any"].size)
    assert sum(rep1["n_flagged_by_group"].values()) == total_any
    for nm in ("anat", "dti"):
        assert sum(rep1["group_concentration"]["by_view"][nm].values()) == int(
            rep1["flagged_rows"]["by_view"][nm]["any"].size
        )
    assert rep1["group_concentration"]["concentrated"] is True
    assert rep1["group_concentration"]["group"] == "site_B"
    assert rep1["group_concentration"]["p_value"] < 0.05
    assert any("site_B" in w and "acquisition" in w for w in rep1["warnings"])


def test_loo_influence_identifies_solution_tipping_rows():
    rng = np.random.default_rng(7)
    n = 36
    # Two orthogonal latent factors on disjoint feature blocks (cols 0..5 vs cols 6..11).
    # On rows 0..33 both factors have identical energy; rows 34 and 35 carry extra amplitude on
    # factor 1 only, tipping the full k=1 PCA-initialized simlr fit onto factor 1 while leaving out
    # row 34 or 35 flips k=1 onto factor 2 (orthogonal, |r| << 0.95).
    t = np.linspace(0.0, 2.0 * np.pi, n - 2, endpoint=False)
    f1_base = np.sin(t)
    f2_base = np.cos(t)
    # Slightly favor f2 on the 34 base rows, then give rows 34, 35 a boost on f1 that just overcomes f2
    f1 = np.concatenate([0.96 * f1_base, np.array([2.05, -2.05])])
    f2 = np.concatenate([1.10 * f2_base, np.array([0.0, 0.0])])

    w1_a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w2_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    noise_a = 0.02 * rng.standard_normal((n, 12))
    noise_b = 0.02 * rng.standard_normal((n, 12))
    x_a = np.outer(f1, w1_a) + np.outer(f2, w2_a) + noise_a
    x_b = np.outer(f1, w1_a) + np.outer(f2, w2_a) + noise_b

    rep = pysimlr.data_safety(
        {"a": x_a, "b": x_b},
        influence=True,
        method="simlr",
        k=1,
        positivity="positive",
        energy_type="recon_r2",
        scale_list=["none"],
        iterations=80,
    )
    assert rep["loo_score_correlation"].shape == (n,)
    assert rep["n_other_solution"] == 2
    assert list(rep["other_solution_rows"]) == [34, 35]
    assert rep["loo_score_correlation"][34] < 0.1
    assert rep["loo_score_correlation"][35] < 0.1
    assert float(np.median(rep["loo_score_correlation"][:34])) > 0.99
    assert any("leave-one-out" in w for w in rep["warnings"])

    # k=2 and n_jobs=2 also return a 1-D (n,) array of minimum aligned component correlations
    rep_k2 = pysimlr.data_safety(
        {"a": x_a, "b": x_b},
        influence=True,
        method="pca",
        k=2,
    )
    rep_k2_par = pysimlr.data_safety(
        {"a": x_a, "b": x_b},
        influence=True,
        method="pca",
        k=2,
        n_jobs=2,
    )
    assert rep_k2["loo_score_correlation"].shape == (n,)
    assert np.all(rep_k2["loo_score_correlation"] > 0.90)
    np.testing.assert_allclose(rep_k2["loo_score_correlation"], rep_k2_par["loo_score_correlation"], atol=1e-10)


def test_fit_safety_modes_and_validation():
    rng = np.random.default_rng(123)
    v0 = rng.standard_normal((80, 20))
    v1 = rng.standard_normal((80, 20))
    v0[5] = 6.0

    # Default safety="off" on fit: no warning and data_safety is None
    with warnings.catch_warnings(record=True) as caught_off:
        warnings.simplefilter("always")
        res_off = pysimlr.fit("pca", [v0, v1], k=1)
    assert [w for w in caught_off if issubclass(w.category, pysimlr.DataSafetyWarning)] == []
    assert res_off["data_safety"] is None

    with pytest.warns(pysimlr.DataSafetyWarning, match="view0"):
        res_warn = pysimlr.fit("pca", [v0, v1], k=1, safety="warn")
    assert res_warn["data_safety"] is not None
    assert list(res_warn["data_safety"]["flagged_rows"]["by_view"][0]["row_scale"]) == [5]

    with pytest.raises(ValueError, match="view0"):
        pysimlr.fit("pca", [v0, v1], k=1, safety="raise")

    with pytest.raises(ValueError, match="safety must be one of"):
        pysimlr.prepare_views([v0, v1], safety="invalid")
    with pytest.raises(ValueError, match="safety must be one of"):
        pysimlr.fit("pca", [v0, v1], k=1, safety="invalid")
