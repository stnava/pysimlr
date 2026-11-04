"""Shrunk Johnson relative weights (`rw_shrinkage='auto'`) and outcome-blind wide-view screening (`D27`)."""
import inspect
import warnings

import numpy as np
import pytest
import torch
from sklearn.covariance import LedoitWolf

import pysimlr
from pysimlr.loadings import (
    _relative_weight_basis,
    _relative_weight_map,
    _relative_weight_shares,
    _zscore,
)


def test_shrunk_relative_weight_equals_joint_schur_r2():
    """Weights are non-negative and sum to 1 - 1 / inv(S_joint)[-1, -1] in [0, 1) for any p >= n."""
    rng = np.random.default_rng(11)
    for n, p in ((40, 120), (60, 200), (80, 15)):
        x = rng.standard_normal((n, p))
        if p < n:
            x[:, 1] = x[:, 0]  # rank-deficient collinearity
        t = 0.8 * x[:, 0:2] + rng.standard_normal((n, 2))
        xz, tz = _zscore(x), _zscore(t)

        # 1. Fixed float shrinkage
        lam = 0.25
        w_fix = _relative_weight_map(xz, tz, shrinkage=lam)
        assert (w_fix >= 0.0).all() and np.isfinite(w_fix).all()
        for j in range(2):
            joint = np.column_stack([xz, tz[:, j]])
            s = (1.0 - lam) * (joint.T @ joint / n) + lam * np.eye(p + 1)
            r2_schur = 1.0 - 1.0 / np.linalg.inv(s)[-1, -1]
            assert 0.0 <= r2_schur < 1.0
            assert np.isclose(w_fix[:, j].sum(), r2_schur, atol=1e-12)

        # 2. Automatic / Ledoit-Wolf shrinkage
        w_lw = _relative_weight_map(xz, tz, shrinkage="auto")
        assert (w_lw >= 0.0).all() and np.isfinite(w_lw).all()
        for j in range(2):
            joint = np.column_stack([xz, tz[:, j]])
            s_lw = LedoitWolf(assume_centered=True).fit(joint).covariance_
            r2_lw = 1.0 - 1.0 / np.linalg.inv(s_lw)[-1, -1]
            assert 0.0 <= r2_lw < 1.0
            assert np.isclose(w_lw[:, j].sum(), r2_lw, atol=1e-11)


def test_auto_shrinkage_is_exact_zero_on_full_rank_narrow_views():
    """`shrinkage='auto'` matches classical Johnson (2000) bit-for-bit when p < n and full rank."""
    rng = np.random.default_rng(19)
    xz = _zscore(rng.standard_normal((120, 18)))
    tz = _zscore(xz[:, :3] @ rng.standard_normal((3, 2)) + rng.standard_normal((120, 2)))
    w0 = _relative_weight_map(xz, tz, shrinkage=0.0)
    wa = _relative_weight_map(xz, tz, shrinkage="auto")
    assert np.array_equal(w0, wa)
    z0, l0 = _relative_weight_basis(xz, shrinkage=0.0)
    za, la = _relative_weight_basis(xz, shrinkage="auto")
    assert np.array_equal(z0, za) and np.array_equal(l0, la)
    sh = _relative_weight_shares(*_relative_weight_basis(_zscore(rng.standard_normal((30, 90))), "auto"),
                                 tz[:30])
    assert (sh >= 0.0).all() and np.allclose(sh.sum(0), 1.0, atol=1e-12)


def test_feature_maps_and_loading_inference_on_wide_views_with_auto_shrinkage():
    rng = np.random.default_rng(23)
    s = rng.standard_normal((60, 2))
    xs = [
        torch.tensor(s @ rng.standard_normal((2, 95)) + 0.7 * rng.standard_normal((60, 95))).float(),
        torch.tensor(s @ rng.standard_normal((2, 40)) + 0.7 * rng.standard_normal((60, 40))).float(),
    ]
    # Note: view 0 has p=95 > n=60; view 1 has p=40 > n_sub=30 at fraction=0.5!
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inf = pysimlr.loading_inference(
            "simlr", xs, 2, n_subsamples=4, fraction=0.5, iterations=25,
            maps=("relative_weight", "partial"),
            map_options={"rw_shrinkage": "auto"},
        )
    rw_entry = inf["maps"]["relative_weight"]
    for vi, x in enumerate(xs):
        assert tuple(rw_entry["map"][vi].shape) == (x.shape[1], 2)
        assert torch.isfinite(rw_entry["map"][vi]).all()
        assert (rw_entry["map"][vi] >= 0).all()
        assert (rw_entry["ci_low"][vi] >= -1e-9).all()
    enc = pysimlr.feature_map_values(inf["result"], inf=inf, view=0)
    assert np.isclose(np.sum(enc["relative_weight"]["values"]), 100.0, atol=1e-5)

    with pytest.raises(ValueError, match="rw_shrinkage"):
        pysimlr.feature_maps(inf["result"], xs, kinds=("relative_weight",), rw_shrinkage="bad")
    with pytest.raises(ValueError, match="rw_shrinkage has 3 entries"):
        pysimlr.feature_maps(inf["result"], xs, kinds=("relative_weight",), rw_shrinkage=["auto", "auto", "auto"])


def test_feature_relevance_recovers_planted_drivers_in_wide_view():
    """`feature_relevance` works when p >> n_train, p >> n_test, and p >> n_sub (refit_fraction=0.5)."""
    rng = np.random.default_rng(31)
    n_tr, n_te, p_wide = 80, 50, 200
    n = n_tr + n_te
    s = rng.standard_normal(n)
    w0 = np.zeros(p_wide)
    w0[:4] = 1.6
    xs = [
        torch.tensor(np.outer(s, w0) + rng.standard_normal((n, p_wide))).float(),
        torch.tensor(np.outer(s, rng.uniform(0.8, 1.3, 45)) + rng.standard_normal((n, 45))).float(),
        torch.tensor(np.outer(s, rng.uniform(0.8, 1.3, 18)) + rng.standard_normal((n, 18))).float(),
    ]
    xtr = [x[:n_tr] for x in xs]
    xte = [x[n_tr:] for x in xs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit("simlr", xtr, k=2, iterations=40, random_state=0)
        rel = pysimlr.feature_relevance(
            res, xtr, xte, outcome=s[:n_tr], outcome_test=s[n_tr:],
            views=(0, 1), n_permutations=2000, n_permutations_test=2000,
            n_refits=2, refit_permutations=500, refit_fraction=0.5,
            n_null_refits=2, n_jobs=1, random_state=7,
        )
    f0 = rel["features"].query("view == 0").reset_index(drop=True)
    assert len(f0) == p_wide
    assert np.isclose(f0.share_train.sum(), 1.0) and np.isclose(f0.share_test.sum(), 1.0)
    assert (f0.tier[:4] == 2).sum() >= 3
    assert (f0.tier[4:] == 2).sum() <= 3
    assert np.isfinite(f0.refit_p05_freq).all() and np.isfinite(f0.p_null_refit).all()


def test_screen_wide_views_is_outcome_blind_and_suppresses_noise_coupling():
    """`screen_wide_views` has no outcome parameter, detects pure-noise wide views, and boosts OOS recovery."""
    sig = inspect.signature(pysimlr.screen_wide_views)
    assert "outcome" not in sig.parameters and "y" not in sig.parameters

    # 1. Pure noise wide views (D25 fixture): omnibus test flags uncoupled views and eliminates WideViewWarning.
    rng = np.random.default_rng(3)
    noise_tr = [torch.tensor(rng.standard_normal((50, 120))).float() for _ in range(3)]
    gate_noise = pysimlr.screen_wide_views(
        noise_tr, k=2, n_permutations=300, n_subsamples=12, random_state=0,
    )
    assert gate_noise["view_coupled"] == [False, False, False]
    assert all(x.shape[1] < 50 for x in gate_noise["views"])
    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        res_noise = pysimlr.fit("simlr", gate_noise["views"], k=2, iterations=30, random_state=0)
    assert [w for w in w_rec if issubclass(w.category, pysimlr.WideViewWarning)] == []
    assert res_noise["wide_views"] == []

    # 2. Planted sparse signal in a wide view with exact duplicate collinearity.
    rng2 = np.random.default_rng(42)
    n_tr, n_te, p_wide = 80, 150, 360
    n = n_tr + n_te
    z_true = rng2.standard_normal((n, 2))
    w_wide = np.zeros((2, p_wide))
    w_wide[0, :5] = 1.4
    w_wide[1, 5:10] = 1.4
    x_wide = z_true @ w_wide + rng2.standard_normal((n, p_wide))
    # Duplicate col 0 into col 10 (r == 1.0) to test collinearity deduplication
    x_wide[:, 10] = x_wide[:, 0]
    x_v1 = z_true @ (0.9 * rng2.standard_normal((2, 24))) + rng2.standard_normal((n, 24))
    x_v2 = z_true @ (0.9 * rng2.standard_normal((2, 20))) + rng2.standard_normal((n, 20))
    all_views = [torch.tensor(m).float() for m in (x_wide, x_v1, x_v2)]
    tr = [v[:n_tr] for v in all_views]
    te = [v[n_tr:] for v in all_views]
    fnames = [[f"v{m}_{i}" for i in range(v.shape[1])] for m, v in enumerate(tr)]

    gate = pysimlr.screen_wide_views(
        tr, k=2, max_width_ratio=0.75, n_permutations=400, n_subsamples=20,
        max_collinearity=0.95, feature_names=fnames, view_names=("wide", "m1", "m2"),
        random_state=0,
    )
    assert gate["view_coupled"] == [True, True, True]
    kept0 = set(gate["indices"][0].tolist())
    # Drivers 0..9 recovered (at least 9/10), and duplicate col 10 excluded when col 0 is kept
    assert len(kept0 & set(range(10))) >= 9
    assert not ({0, 10} <= kept0)
    # Narrow views keep all columns untouched
    assert len(gate["indices"][1]) == 24 and len(gate["indices"][2]) == 20

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_raw = pysimlr.fit("simlr", tr, k=2, sparseness=0.8, iterations=60, random_state=0)
        res_gated = pysimlr.fit("simlr", gate["views"], k=2, sparseness=0.8, iterations=60, random_state=0)
    res_gated["view_names"] = gate["view_names"]

    te_gated = pysimlr.apply_view_screen(gate, te)
    pred_raw = pysimlr.predict(res_raw, te)
    pred_gated = pysimlr.predict(res_gated, te_gated)

    # Subset dict prediction + None missing-view handling through apply_view_screen
    te_sub_dict = pysimlr.apply_view_screen(gate, {"wide": te[0], "m1": te[1]})
    assert set(te_sub_dict) == {"wide", "m1"} and te_sub_dict["wide"].shape[1] == len(kept0)
    te_with_none = pysimlr.apply_view_screen(gate, [te[0], None, te[2]])
    assert te_with_none[1] is None and te_with_none[0].shape[1] == len(kept0)

    # Gating narrows the training-vs-held-out cross-view coupling gap and boosts wide-view OOS recovery
    z0_raw_te = pred_raw["latents"][0].detach().double().numpy()
    z0_gat_te = pred_gated["latents"][0].detach().double().numpy()
    r2_raw = np.linalg.svd(
        np.corrcoef(z0_raw_te.T, z_true[n_tr:].T)[:2, 2:], compute_uv=False
    ).mean()
    r2_gat = np.linalg.svd(
        np.corrcoef(z0_gat_te.T, z_true[n_tr:].T)[:2, 2:], compute_uv=False
    ).mean()
    assert r2_gat > r2_raw + 0.08, (r2_raw, r2_gat)
