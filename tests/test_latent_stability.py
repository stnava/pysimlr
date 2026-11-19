"""Tests for calibrated out-of-bag `pysimlr.latent_stability` across all weight constraints."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.latent_stability import (
    REMOVED_STABILITY_KEYS,
    compute_oob_icc,
    compute_trace_icc,
)

torch.set_num_threads(1)

CONSTRAINTS = ("linear", "conical", "convex", "mixed")


def _constraint_kwargs(constraint: str, n_views: int = 3, *, energy_type: str = "acc") -> dict:
    if constraint == "linear":
        return {"positivity": "linear", "energy_type": energy_type, "sparseness": 0.5}
    if constraint == "conical":
        return {"positivity": "conical", "energy_type": energy_type, "sparseness": 0.5}
    if constraint == "convex":
        return {"positivity": "convex", "energy_type": energy_type, "sparseness": 0.5}
    if constraint == "mixed":
        cycle = ["convex", "linear", "conical"]
        pvp = {
            m: {
                "positivity": cycle[m % 3],
                "energy_type": energy_type,
                "sparseness": 0.5,
            }
            for m in range(n_views)
        }
        return {"per_view_params": pvp}
    raise ValueError(constraint)


def _noise_views(seed: int = 0, n: int = 80, widths=(24, 22, 20)):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, p)) for p in widths]


def _planted_shared_views(
    seed: int = 0,
    n: int = 200,
    widths=(20, 18, 16),
    k: int = 1,
    shared_strength: float = 1.6,
    private_strength: float = 1.4,
    nonneg: bool = False,
    priv_rank: int = 2,
):
    rng = np.random.default_rng(seed)
    z_sh = rng.standard_normal((n, k))
    views = []
    for p in widths:
        w_sh = np.zeros((p, k))
        for j in range(k):
            sl = slice(j * 4, (j + 1) * 4)
            w_sh[sl, j] = rng.uniform(0.9, 1.2, size=4) if nonneg else rng.choice([-1.0, 1.0], size=4) * 1.0
        z_priv = rng.standard_normal((n, priv_rank)) * private_strength
        w_priv = rng.standard_normal((p, priv_rank)) / np.sqrt(max(p, 1))
        x = shared_strength * (z_sh @ w_sh.T) + z_priv @ w_priv.T + 0.5 * rng.standard_normal((n, p))
        views.append(x)
    return z_sh, views


def _single_view_dominant_views(seed: int = 5, n: int = 200, widths=(20, 16)):
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal((n, 1))
    w0 = np.zeros((widths[0], 1))
    w0[:14, 0] = 3.0
    v0 = z0 @ w0.T + 0.2 * rng.standard_normal((n, widths[0]))
    others = [rng.standard_normal((n, p)) for p in widths[1:]]
    return [v0] + others


def _wrong_oob_aligned_icc(
    views,
    k: int,
    *,
    n_subsamples: int = 14,
    fraction: float = 0.8,
    random_state: int = 0,
    constraint: str = "linear",
) -> float:
    """Deliberately WRONG variant: aligns each refit's OOB scores directly to `u_ref[oob_rows]`.

    Because the reference fit saw those rows and OOB rows are used to pick orientation/rotation,
    this leaks OOB information into the alignment and inflates trace ICC(1) even on pure noise.
    """
    from pysimlr.loadings import _procrustes

    n = views[0].shape[0]
    m = max(k + 2, int(round(fraction * n)))
    kw = _constraint_kwargs(constraint, len(views))
    ref = pysimlr.fit("simlr", views, k, random_state=random_state, iterations=15, **kw)
    u_ref = ref["u"].detach().double().numpy()
    u_ref = (u_ref - u_ref.mean(0)) / np.maximum(u_ref.std(0), 1e-12)
    rng = np.random.default_rng(random_state)
    all_rows = np.arange(n)
    y_oob = np.full((n_subsamples, n, k), np.nan, dtype=float)
    for b in range(n_subsamples):
        idx = np.sort(rng.choice(n, m, replace=False))
        oob = np.setdiff1d(all_rows, idx, assume_unique=True)
        sub_v = [v[idx] for v in views]
        oob_v = [v[oob] for v in views]
        rb = pysimlr.fit("simlr", sub_v, k, random_state=random_state, iterations=15, **kw)
        ut = rb["u"].detach().double().numpy()
        uo = pysimlr.predict(rb, oob_v)["u"].detach().double().numpy()
        uo = (uo - ut.mean(0)) / np.maximum(ut.std(0), 1e-12)
        # LEAK: align on OOB rows against u_ref[oob]!
        q_leak = _procrustes(uo - uo.mean(0), u_ref[oob] - u_ref[oob].mean(0))
        y_oob[b, oob] = uo @ q_leak
    return compute_trace_icc(y_oob)


def test_icc_unbalanced_hand_computation_and_negative_unclamped():
    # Small unbalanced table (4 raters x 5 targets) with missing entries (NaN).
    y = np.array([
        [[1.0, -2.0], [2.0, 0.0], [np.nan, np.nan], [4.0, 1.0], [np.nan, np.nan]],
        [[1.5, -1.0], [np.nan, np.nan], [3.0, 2.0], [3.5, 0.5], [5.0, 3.0]],
        [[np.nan, np.nan], [2.5, 1.0], [2.5, 1.5], [np.nan, np.nan], [4.5, 2.5]],
        [[0.5, -1.5], [1.5, 0.5], [3.5, 2.5], [4.5, 1.5], [5.5, 3.5]],
    ], dtype=float)

    # Hand ANOVA calculation for each column j=0, 1
    msb_hand = np.zeros(2)
    msw_hand = np.zeros(2)
    m_vec = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    total_m = float(m_vec.sum())
    n_t = 5.0
    k0_hand = (total_m - (m_vec ** 2).sum() / total_m) / (n_t - 1.0)
    for j in range(2):
        targets = [y[:, i, j][np.isfinite(y[:, i, j])] for i in range(5)]
        means = np.array([t.mean() for t in targets])
        grand = np.concatenate(targets).mean()
        ssb = sum(len(t) * (m_i - grand) ** 2 for t, m_i in zip(targets, means))
        ssw = sum(((t - m_i) ** 2).sum() for t, m_i in zip(targets, means))
        msb_hand[j] = ssb / (n_t - 1.0)
        msw_hand[j] = ssw / (total_m - n_t)
    icc1_hand = (msb_hand - msw_hand) / (msb_hand + (k0_hand - 1.0) * msw_hand)
    icc1k_hand = (msb_hand - msw_hand) / msb_hand
    trace_hand = (msb_hand.sum() - msw_hand.sum()) / (msb_hand.sum() + (k0_hand - 1.0) * msw_hand.sum())

    got = compute_oob_icc(y)
    np.testing.assert_allclose(got["msb"], msb_hand, atol=1e-12)
    np.testing.assert_allclose(got["msw"], msw_hand, atol=1e-12)
    np.testing.assert_allclose(got["latent_reliability"], icc1_hand, atol=1e-12)
    np.testing.assert_allclose(got["latent_reliability_averaged"], icc1k_hand, atol=1e-12)
    assert got["k0"] == pytest.approx(k0_hand, abs=1e-12)

    got_tr = compute_trace_icc(y)
    assert got_tr == pytest.approx(trace_hand, abs=1e-12)
    # Trace ICC(1) is strictly invariant under any O(k) orthogonal rotation.
    theta = 0.73
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert compute_trace_icc(y @ rot) == pytest.approx(got_tr, abs=1e-12)

    # Purely negative ICC(1) where targets share almost identical means and large within-target spread (MSW > MSB).
    y_neg = np.array([
        [[+3.0, -3.0], [+3.0, -3.0], [np.nan, np.nan], [-3.0, +3.0]],
        [[-3.0, +3.0], [-3.0, +3.0], [+3.0, -3.0], [+3.0, -3.0]],
        [[np.nan, np.nan], [+0.1, -0.1], [-3.0, +3.0], [np.nan, np.nan]],
    ], dtype=float)
    neg_res = compute_oob_icc(y_neg)
    assert np.all(neg_res["latent_reliability"] < 0.0)
    assert np.all(neg_res["latent_reliability_averaged"] < 0.0)
    assert compute_trace_icc(y_neg) < 0.0


@pytest.mark.parametrize("constraint", CONSTRAINTS)
def test_pure_noise_calibrated_and_unclamped(constraint):
    kw = _constraint_kwargs(constraint, 3)
    n_seeds = 10          # per-draw checks need many draws; the k = 2 calls use the first 6 only
    gains_k1 = []
    coups_k1 = []
    mv_gains_k2 = []
    coups_k2 = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(n_seeds):
            views = _noise_views(seed=seed, n=120, widths=(20, 18, 16))
            res1 = pysimlr.latent_stability(
                "simlr", views, 1, n_subsamples=10, fraction=0.8,
                baseline=True, n_baseline=3, random_state=seed, iterations=12, **kw,
            )
            assert res1["latent_reliability_multivariate"] == pytest.approx(
                float(res1["latent_reliability"][0]), abs=1e-12
            )
            gains_k1.append(res1["latent_reliability_gain"])
            coups_k1.append(res1["cross_view_coupling"])
            if seed < 6:
                res2 = pysimlr.latent_stability(
                    "simlr", views, 2, n_subsamples=10, fraction=0.8,
                    baseline=True, n_baseline=3, random_state=seed, iterations=12, **kw,
                )
                mv_gains_k2.append(res2["latent_reliability_multivariate_gain"])
                coups_k2.append(res2["cross_view_coupling"])
    assert np.all(np.abs(np.mean(gains_k1, axis=0)) <= 0.10)
    assert np.all(np.abs(np.mean(coups_k1, axis=0)) <= 0.15)
    assert abs(float(np.mean(mv_gains_k2))) <= 0.10
    assert np.all(np.abs(np.mean(coups_k2, axis=0)) <= 0.15)
    # Per-draw behaviour (a mean over seeds hides it). Measured over seeds 0-11 at this configuration:
    # per-draw sd of the reliability gain 0.18-0.22 and max |gain| 0.50; max |cross_view_coupling| 0.16.
    # The bounds sit at about twice the measured sd / 1.6x the measured maximum.
    g = np.asarray(gains_k1, dtype=float)[:, 0]
    c = np.asarray(coups_k1, dtype=float)[:, 0]
    assert float(np.std(g, ddof=1)) <= 0.40, (constraint, g.round(2).tolist())
    assert float(np.max(np.abs(g))) <= 0.80, (constraint, g.round(2).tolist())
    assert float(np.max(np.abs(c))) <= 0.35, (constraint, c.round(2).tolist())
    # Trace ICC(1) on white-noise OOB table is centered near 0 (not 1/k0 ~ 0.33), with negatives unclamped.
    rng = np.random.default_rng(123)
    tr_vals = []
    seen_neg = False
    for _ in range(40):
        y = rng.standard_normal((15, 40, 2))
        in_bag = rng.random((15, 40)) < 0.8
        y[in_bag] = np.nan
        if np.any(compute_oob_icc(y)["latent_reliability"] < 0.0):
            seen_neg = True
        tr_vals.append(compute_trace_icc(y))
    assert seen_neg
    assert abs(float(np.mean(tr_vals))) <= 0.05


@pytest.mark.parametrize("constraint", CONSTRAINTS)
def test_planted_signal_monotone_with_shared_strength(constraint):
    nonneg = constraint != "linear"
    kw = _constraint_kwargs(constraint, 3)
    gains = []
    for strength in (0.75, 0.40, 0.0):
        rng = np.random.default_rng(4)
        z_sh = rng.standard_normal((200, 1))
        views = []
        for p in (30, 28, 26):
            w_sh = np.zeros((p, 1))
            w_sh[:3, 0] = rng.uniform(0.9, 1.1, size=3) if nonneg else rng.choice([-1.0, 1.0], size=3)
            views.append(strength * (z_sh @ w_sh.T) + 1.25 * rng.standard_normal((200, p)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pysimlr.latent_stability(
                "simlr", views, 1, n_subsamples=14, fraction=0.8,
                baseline=True, n_baseline=3, random_state=0, iterations=20, **kw,
            )
        gains.append(float(out["latent_reliability_gain"][0]))
    assert gains[0] >= 0.30, (constraint, gains)
    assert gains[0] > gains[1] > gains[2], (constraint, gains)


@pytest.mark.parametrize("constraint", CONSTRAINTS)
def test_single_view_private_dominant_direction(constraint):
    views = _single_view_dominant_views(seed=5, n=200, widths=(20, 16))
    kw = _constraint_kwargs(constraint, len(views), energy_type="recon_r2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pysimlr.latent_stability(
            "simlr", views, 1, n_subsamples=25, fraction=0.8,
            baseline=True, n_baseline=3, random_state=0,
            tune_learning_rate="reference", iterations=25, **kw,
        )
    assert float(out["latent_reliability"][0]) > 0.50
    assert abs(float(out["latent_reliability_gain"][0])) <= 0.12
    assert abs(float(out["cross_view_coupling"][0])) <= 0.15


@pytest.mark.parametrize("constraint", CONSTRAINTS)
def test_k2_equal_strength_shared_factors(constraint):
    nonneg = constraint in ("conical", "convex")
    _, views = _planted_shared_views(
        seed=33, n=150, widths=(20, 18, 16), k=2,
        shared_strength=1.6, private_strength=0.9, nonneg=nonneg,
    )
    kw = _constraint_kwargs(constraint, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pysimlr.latent_stability(
            "simlr", views, 2, n_subsamples=14, fraction=0.8,
            baseline=True, n_baseline=3, random_state=0, iterations=30, **kw,
        )
    assert out["latent_reliability_multivariate_gain"] >= 0.30
    assert out["latent_reliability_gain"].shape == (2,)


@pytest.mark.parametrize("constraint", CONSTRAINTS)
def test_alignment_leak_guard_missing_rows_strata_determinism_n_jobs(constraint):
    views_null = _noise_views(seed=3, n=55, widths=(18, 16, 14))
    kw = _constraint_kwargs(constraint, len(views_null))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = pysimlr.latent_stability(
            "simlr", views_null, 2, n_subsamples=14, fraction=0.8,
            baseline=True, n_baseline=3, random_state=0, iterations=15, **kw,
        )
        leaked = _wrong_oob_aligned_icc(
            views_null, 2, n_subsamples=14, fraction=0.8, random_state=0, constraint=constraint,
        )
    assert leaked > float(clean["latent_reliability_multivariate_gain"]) + 0.15

    # Missing rows ("mask"), strata, determinism, and n_jobs=2 == n_jobs=1
    _, views = _planted_shared_views(
        seed=14, n=120, widths=(18, 16, 14), k=1,
        shared_strength=1.5, private_strength=0.8, nonneg=(constraint in ("conical", "convex")),
    )
    views_m = [v.copy() for v in views]
    views_m[1][:18] = np.nan
    strata = np.repeat([0, 1], [80, 40])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pysimlr.latent_stability(
            "simlr", views_m, 1, n_subsamples=12, fraction=0.5, strata=strata,
            baseline=True, n_baseline=2, random_state=0, draw_random_state=4,
            n_jobs=1, missing_rows="mask", iterations=20, **kw,
        )
        b = pysimlr.latent_stability(
            "simlr", views_m, 1, n_subsamples=12, fraction=0.5, strata=strata,
            baseline=True, n_baseline=2, random_state=0, draw_random_state=4,
            n_jobs=2, missing_rows="mask", iterations=20, **kw,
        )
    assert np.isfinite(a["latent_reliability"]).all()
    assert a["n_oob_per_row"]["min"] >= 1
    assert a["n_oob_per_row"]["median"] >= 1.0
    np.testing.assert_array_equal(a["latent_reliability"], b["latent_reliability"])
    np.testing.assert_array_equal(a["latent_reliability_baseline"], b["latent_reliability_baseline"])
    np.testing.assert_array_equal(a["feature_agreement"], b["feature_agreement"])
    np.testing.assert_array_equal(a["cross_view_coupling"], b["cross_view_coupling"])


def test_tune_learning_rate_reference_and_gauge_validation():
    _, views = _planted_shared_views(seed=2, n=90, widths=(14, 14, 12), k=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pysimlr.latent_stability(
            "simlr", views, 1, n_subsamples=5, fraction=0.8,
            baseline=True, n_baseline=2, random_state=0,
            tune_learning_rate="reference", iterations=15, positivity="conical",
        )
    lrs = out["refit_learning_rates"]
    assert len(lrs) == 5 and all(isinstance(x, float) for x in lrs) and len(set(lrs)) == 1
    assert len(out["provenance"]["baseline_learning_rates"]) == 2

    # Non-negative gauge check raises ValueError if a fit violates V >= 0.
    from pysimlr.latent_stability import _verify_nonneg_gauge

    bad_res = {
        "v": [torch.tensor([[-0.2], [0.5]])],
        "basis_is_joint": False,
    }
    with pytest.raises(ValueError, match="non-negative gauge"):
        _verify_nonneg_gauge(bad_res, ["positive"], [None])
    bad_convex = {
        "v": [torch.tensor([[0.4], [0.4]])],
        "basis_is_joint": False,
    }
    with pytest.raises(ValueError, match="simplex gauge"):
        _verify_nonneg_gauge(bad_convex, ["convex"], [None])


@pytest.mark.parametrize("constraint", ("linear", "conical", "convex"))
def test_deep_method_end_to_end(constraint):
    _, views = _planted_shared_views(
        seed=8, n=90, widths=(12, 12, 10), k=1,
        shared_strength=1.5, private_strength=0.6, nonneg=(constraint != "linear"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pysimlr.latent_stability(
            "lend", views, 1, n_subsamples=6, fraction=0.8,
            baseline=True, n_baseline=2, random_state=0,
            positivity=constraint, iterations=8,
        )
    assert out["alignment"]["feature_space"] == "structure"
    assert np.isfinite(out["latent_reliability"]).all()
    assert np.isfinite(out["cross_view_coupling"]).all()
    assert np.isfinite(out["feature_agreement"]).all()
    assert len(out["feature_agreement_by_view"]) == 3


def test_removed_keys_and_removed_criteria_errors():
    _, views = _planted_shared_views(seed=1, n=80, widths=(12, 10, 10), k=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls = pysimlr.latent_stability("pca", views, 1, n_subsamples=4, baseline=False)
        inf = pysimlr.loading_inference("pca", views, 1, n_subsamples=4)
        bag = pysimlr.bagged_fit("pca", views, 1, aggregate="average", n_subsamples=4)
    for res in (ls, inf, bag):
        for old_k, rep in REMOVED_STABILITY_KEYS.items():
            assert old_k not in res
            with pytest.raises(KeyError) as exc_info:
                _ = res[old_k]
            msg = exc_info.value.args[0]
            assert old_k in msg and "0.41.0" in msg and rep.split(" / ")[0] in msg

    for old_crit, rep in (
        ("component", "latent_reliability"),
        ("subspace", "latent_reliability"),
        ("subspace_corrected", "latent_reliability"),
        ("oos_coupling", "cross_view_coupling"),
    ):
        with pytest.raises(ValueError, match=rep):
            pysimlr.select_rank("pca", views, [1, 2], criterion=old_crit, n_subsamples=3)


STABILITY_KEYS = (
    "latent_reliability", "latent_reliability_averaged", "latent_reliability_multivariate",
    "cross_view_coupling", "feature_agreement", "n_oob_per_row", "n_refits_used",
    "converged_fraction", "alignment", "interpretation", "provenance",
)


def _small_views(seed: int = 1, n: int = 80, widths=(12, 10, 10)):
    return _planted_shared_views(seed=seed, n=n, widths=widths, k=1)[1]


def test_removed_keys_get_copy_dict_conversion_pickle():
    import copy
    import pickle

    views = _small_views()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls = pysimlr.latent_stability("pca", views, 1, n_subsamples=4, baseline=False)
        inf = pysimlr.loading_inference("pca", views, 1, n_subsamples=4)
        bag = pysimlr.bagged_fit("pca", views, 1, aggregate="average", n_subsamples=4)
    for res in (ls, inf, bag):
        for old_k in REMOVED_STABILITY_KEYS:
            # __getitem__ and .get both raise the migration message (get never returns None silently)
            for access in (lambda r, k=old_k: r[k], lambda r, k=old_k: r.get(k), lambda r, k=old_k: r.get(k, 0.0)):
                with pytest.raises(KeyError, match="removed in 0.41.0"):
                    access(res)
            # copy, copy.copy, deepcopy, pickle, and `|` keep the raising behaviour
            for dup in (res.copy(), copy.copy(res), copy.deepcopy(res), pickle.loads(pickle.dumps(res)), res | {}):
                with pytest.raises(KeyError, match="removed in 0.41.0"):
                    dup[old_k]
                with pytest.raises(KeyError, match="removed in 0.41.0"):
                    dup.get(old_k)
        # dict(res) and {**res} must be plain dicts: they carry the documented removed_keys table instead
        for plain in (dict(res), {**res}):
            assert type(plain) is dict
            table = plain["removed_keys"]
            assert set(table) == set(REMOVED_STABILITY_KEYS)
            assert all("0.41.0" in msg for msg in table.values())
        # an ordinary present key still reads normally through get
        assert res.get("no_such_key", 7) == 7


def test_select_model_and_plot_removed_arguments_raise():
    views = _small_views(n=60)
    with pytest.raises(ValueError, match="min_cross_view_coupling"):
        pysimlr.select_model(views, [{"label": "p", "method": "pca", "params": {}}], [1], min_oos_coupling=0.1, n_subsamples=2)
    with pytest.raises(ValueError, match="min_cross_view_coupling"):
        pysimlr.select_model(views, [{"label": "p", "method": "pca", "params": {"min_oos_coupling": 0.1}}], [1], n_subsamples=2)
    from pysimlr.plotting import plot_rank_selection
    with pytest.raises(ValueError, match="bars.*removed in 0.41.0.*latent_reliability_gain"):
        plot_rank_selection({"criterion": "latent_reliability"}, bars=[0.9, 0.8])
    from pysimlr.plotting import plot_feature_map_comparison, plot_feature_map_shift, plot_top_features
    for fn in (plot_top_features, plot_feature_map_shift, plot_feature_map_comparison):
        with pytest.raises(ValueError, match="bars.*removed in 0.41.0"):
            fn({}, bars=[0.9, 0.8])


def test_repeat_call_is_deterministic_and_groups_are_respected():
    views = _small_views(n=80)
    groups = np.repeat(np.arange(8), 10)
    kw = dict(n_subsamples=6, baseline=True, n_baseline=2, random_state=3, iterations=10, positivity="linear",
              sparseness=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pysimlr.latent_stability("simlr", views, 1, **kw)
        b = pysimlr.latent_stability("simlr", views, 1, **kw)
        g1 = pysimlr.latent_stability("simlr", views, 1, groups=groups, **kw)
        g2 = pysimlr.latent_stability("simlr", views, 1, groups=groups, **kw)
    for key in ("latent_reliability", "latent_reliability_baseline", "latent_reliability_gain",
                "cross_view_coupling", "cross_view_coupling_baseline", "feature_agreement",
                "feature_agreement_baseline"):
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(g1[key], g2[key])
    assert a["interpretation"] == b["interpretation"]
    assert g1["provenance"]["groups"] is not None and a["provenance"]["groups"] is None
    # a groups-respecting baseline permutes within group, so it differs from the ungrouped baseline draw
    assert not np.array_equal(g1["latent_reliability_baseline"], a["latent_reliability_baseline"])


def test_bagged_fit_select_interpretation_states_pre_selection():
    views = _small_views(n=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = pysimlr.bagged_fit("simlr", views, 1, aggregate="select", n_subsamples=5, iterations=8,
                                 sparseness=0.5, positivity="linear")
        avg = pysimlr.bagged_fit("simlr", views, 1, aggregate="average", n_subsamples=5, iterations=8,
                                 sparseness=0.5, positivity="linear")
    assert "pre-selection" in sel["interpretation"] and "result['v']" in sel["interpretation"]
    assert "pre-selection" not in avg["interpretation"]
    assert "select" in str(sel["provenance"]).lower()


def test_loading_inference_returns_same_stability_keys():
    views = _small_views(n=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls = pysimlr.latent_stability("simlr", views, 1, n_subsamples=4, baseline=False, iterations=8,
                                      sparseness=0.5, positivity="linear")
        inf = pysimlr.loading_inference("simlr", views, 1, n_subsamples=4, iterations=8,
                                        sparseness=0.5, positivity="linear")
        bag = pysimlr.bagged_fit("simlr", views, 1, aggregate="average", n_subsamples=4, iterations=8,
                                 sparseness=0.5, positivity="linear")
    for key in STABILITY_KEYS:
        assert key in ls and key in inf and key in bag, key
    assert set(ls["provenance"]) >= {"method", "k", "n_subsamples", "random_state"}


@pytest.mark.parametrize("constraint", ("linear", "conical", "convex"))
def test_default_tune_learning_rate_per_refit_on_private_direction(constraint):
    """The default ``tune_learning_rate="per_refit"`` on the single-view private-direction case.

    Measured (n=200, widths 20/16, B=25): reliability 0.92-0.94, baseline 0.82-0.86, gain 0.07-0.11,
    cross-view coupling 0.10-0.12, for all three constraints, the same as ``"reference"`` within 0.05.
    """
    views = _single_view_dominant_views(seed=5, n=200, widths=(20, 16))
    kw = _constraint_kwargs(constraint, len(views), energy_type="recon_r2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pysimlr.latent_stability(
            "simlr", views, 1, n_subsamples=25, fraction=0.8, baseline=True, n_baseline=3,
            random_state=0, iterations=25, **kw,
        )
    assert out["provenance"]["tune_learning_rate"] == "per_refit"
    assert float(out["latent_reliability"][0]) > 0.80
    assert abs(float(out["latent_reliability_gain"][0])) <= 0.25
    assert abs(float(out["cross_view_coupling"][0])) <= 0.25


@pytest.mark.parametrize("constraint", ("linear", "conical", "convex"))
def test_all_views_private_direction_gain_is_noisy_but_coupling_is_not(constraint):
    """Independent views, each with a strong private rank-1 direction: nothing is shared.

    ``latent_reliability_gain`` wanders from draw to draw here (a private direction is reproducible
    on its own), so it must be read with ``cross_view_coupling_gain``, which stays near 0. Measured
    on seeds 0-9 (n=60, three views of 20/18/16 features, B=10, n_baseline=2): per-draw
    reliability gain sd 0.24-0.26 and max |gain| 0.51; max |cross_view_coupling_gain| 0.36 and
    max |cross_view_coupling| 0.22. Bounds: the per-draw values below, never the mean.
    Rotating the baseline's fixed view was tried and did not reduce this spread.
    """
    kw = _constraint_kwargs(constraint, 3)
    lr_gain, cv_gain, cv = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(6):
            g = np.random.default_rng(500 + seed)
            n = 60
            views = []
            for p in (20, 18, 16):
                priv = g.standard_normal((n, 1)) * np.abs(g.standard_normal((1, p)))
                views.append(g.standard_normal((n, p)) + 2.5 * priv)
            out = pysimlr.latent_stability(
                "simlr", views, 1, n_subsamples=10, baseline=True, n_baseline=2,
                random_state=0, iterations=10, **kw,
            )
            lr_gain.append(float(out["latent_reliability_gain"][0]))
            cv_gain.append(float(out["cross_view_coupling_gain"][0]))
            cv.append(float(out["cross_view_coupling"][0]))
            assert "together with" in out["interpretation"]
    lr_gain, cv_gain, cv = map(np.asarray, (lr_gain, cv_gain, cv))
    assert float(np.max(np.abs(lr_gain))) <= 0.75, lr_gain.round(2).tolist()
    assert float(np.std(lr_gain, ddof=1)) <= 0.40, lr_gain.round(2).tolist()
    assert float(np.max(np.abs(cv_gain))) <= 0.50, cv_gain.round(2).tolist()
    assert float(np.max(np.abs(cv))) <= 0.35, cv.round(2).tolist()
    assert abs(float(np.mean(cv_gain))) <= 0.20
