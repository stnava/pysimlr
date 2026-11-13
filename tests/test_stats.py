"""``pysimlr.stats`` against the implementations it replaces.

Oracles are verbatim copies of the private helpers in antsxbrainmapping
(file:line cited at each) so that a change in pysimlr cannot drift from the
numbers already in that repo's ledgers. The golden test at the end recomputes
ledger columns from the October 2026 run when
``ANTSXBRAINMAPPING_OCTOBER_ROOT`` points at its scratch directory.
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

import numpy as np
import pytest
from scipy import integrate, special
from scipy import stats as sps

from pysimlr.stats import (
    bh_fdr,
    freedman_lane_partial_p,
    holm,
    icc_3,
    jzs_bf10_correlation,
    jzs_bf10_correlation_grid,
    jzs_bf10_one_sample_t,
    permutation_correlations,
    pooled_correlation_fisher_z,
    stouffer_z,
)

OCTOBER_ROOT_ENV = "ANTSXBRAINMAPPING_OCTOBER_ROOT"
OCTOBER_ROOT_DEFAULT = "/Users/stnava/data/processed/antsxbrainmapping/october_2026_run/scratch"


# ------------------------------------------------------------------------------ oracles
# antsxbrainmapping/src/antsxbrainmapping/discovery_transport.py:287
def _oracle_jzs_quad(r: float, n: int) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r):
        return 1.0, 1.0
    r_clip = float(np.clip(r, -0.999999, 0.999999))
    r2 = r_clip * r_clip
    log_prefactor = 0.5 * np.log(0.5 * n) - special.gammaln(0.5)

    def _integrand(g: float) -> float:
        if g <= 0.0:
            return 0.0
        log_val = (
            log_prefactor
            + 0.5 * (n - 2.0) * np.log1p(g)
            - 0.5 * (n - 1.0) * np.log1p((1.0 - r2) * g)
            - 1.5 * np.log(g)
            - (0.5 * n) / g
        )
        if log_val < -700.0:
            return 0.0
        return float(np.exp(min(log_val, 700.0)))

    integral, _ = integrate.quad(_integrand, 0.0, np.inf, limit=200, epsabs=1e-10, epsrel=1e-8)
    bf10 = float(max(integral, 1e-12))
    return bf10, float(1.0 / bf10)


# antsxbrainmapping/src/antsxbrainmapping/discovery_transport.py:319
def _oracle_holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = float(p_values[idx]) * float(m - rank)
        running_max = max(running_max, min(1.0, val))
        adjusted[idx] = running_max
    return adjusted


# antsxbrainmapping/src/antsxbrainmapping/discovery_transport.py:333
def _oracle_benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_min = 1.0
    for rev_rank in range(m - 1, -1, -1):
        idx = order[rev_rank]
        val = float(p_values[idx]) * float(m) / float(rev_rank + 1)
        running_min = min(running_min, min(1.0, val))
        adjusted[idx] = running_min
    return adjusted


# antsxbrainmapping/src/antsxbrainmapping/multiview_simlr.py:853
def _oracle_bh_fdr_adjust(p_vals: np.ndarray) -> np.ndarray:
    p = np.asarray(p_vals, dtype=np.float64)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = np.minimum(1.0, np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1])
    q_out = np.empty_like(p)
    q_out[order] = q_sorted
    return q_out


# antsxbrainmapping/src/antsxbrainmapping/multiview_simlr.py:867
def _oracle_jzs_grid(r_vals, n_samples: int) -> np.ndarray:
    r_arr = np.clip(np.asarray(r_vals, dtype=np.float64), -0.9999, 0.9999)
    if n_samples < 4:
        return np.ones_like(r_arr, dtype=np.float64)
    u_grid = np.linspace(-10.0, 12.0, 400, dtype=np.float64)
    du = float(u_grid[1] - u_grid[0])
    g = np.exp(u_grid)
    log_g = u_grid
    one_minus_r2 = np.maximum(1e-8, 1.0 - r_arr.reshape(-1, 1) ** 2)
    log_int = (
        0.5 * float(n_samples - 2) * np.log1p(g)[None, :]
        - 0.5 * float(n_samples - 1) * np.log1p(one_minus_r2 * g[None, :])
        - 0.5 * log_g[None, :]
        - float(n_samples) / (2.0 * g[None, :])
    )
    max_log = np.max(log_int, axis=1, keepdims=True)
    integral = np.sum(np.exp(log_int - max_log), axis=1) * du
    log_bf10 = (
        0.5 * np.log(float(n_samples) / (2.0 * np.pi)) + max_log.reshape(-1) + np.log(np.maximum(1e-300, integral))
    )
    bf10 = np.exp(np.clip(log_bf10, -50.0, 50.0))
    return bf10.reshape(np.asarray(r_vals).shape)


# antsxbrainmapping/src/antsxbrainmapping/multiview_simlr.py:926
def _oracle_vectorized_permutation_correlations(x_mat, y_vec, *, n_perms=10000, n_boot=2000, seed=42):
    n_samples, n_features = x_mat.shape
    y = np.asarray(y_vec, dtype=np.float64).reshape(-1)
    x_c = x_mat - np.mean(x_mat, axis=0, keepdims=True)
    x_sd = np.std(x_c, axis=0, ddof=1, keepdims=True)
    valid_x = (x_sd.reshape(-1) > 1e-10) & np.all(np.isfinite(x_mat), axis=0)
    x_sd_safe = np.where(x_sd > 1e-10, x_sd, 1.0)
    x_z = np.where(valid_x.reshape(1, -1), x_c / x_sd_safe, 0.0)
    y_c = y - np.mean(y)
    y_z = y_c / np.std(y_c, ddof=1)
    r_obs = (x_z.T @ y_z) / float(n_samples - 1)
    r_obs = np.clip(r_obs, -0.9999999, 0.9999999)
    df = max(1, n_samples - 2)
    t_obs = r_obs * np.sqrt(df / np.maximum(1e-12, 1.0 - r_obs**2))
    p_param = np.where(valid_x, 2.0 * sps.t.sf(np.abs(t_obs), df=df), 1.0)
    rng = np.random.default_rng(int(seed))
    y_perms = np.empty((n_samples, int(n_perms)), dtype=np.float64)
    for b in range(int(n_perms)):
        y_perms[:, b] = y_z[rng.permutation(n_samples)]
    r_null = np.abs((x_z.T @ y_perms) / float(n_samples - 1))
    abs_obs = np.abs(r_obs)[:, None]
    n_exceed = np.sum(r_null >= abs_obs, axis=1)
    p_perm = np.where(valid_x, (1.0 + n_exceed) / (1.0 + float(n_perms)), 1.0)
    q_fdr = _oracle_bh_fdr_adjust(p_perm)
    max_r_null = np.max(r_null, axis=0)
    n_exceed_fwer = np.sum(max_r_null[None, :] >= abs_obs, axis=1)
    p_fwer = np.where(valid_x, (1.0 + n_exceed_fwer) / (1.0 + float(n_perms)), 1.0)
    boot_idx = rng.integers(0, n_samples, size=(int(n_boot), n_samples))
    r_boot = np.empty((n_features, int(n_boot)), dtype=np.float64)
    for b in range(int(n_boot)):
        idx_b = boot_idx[b]
        xb = x_mat[idx_b]
        yb = y[idx_b]
        xb_c = xb - np.mean(xb, axis=0, keepdims=True)
        yb_c = yb - np.mean(yb)
        denom = np.sqrt(np.sum(xb_c**2, axis=0) * np.sum(yb_c**2))
        r_boot[:, b] = np.where(denom > 1e-12, (xb_c.T @ yb_c) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    ci95_low = np.where(valid_x, np.percentile(r_boot, 2.5, axis=1), 0.0)
    ci95_high = np.where(valid_x, np.percentile(r_boot, 97.5, axis=1), 0.0)
    pos_frac = np.mean(r_boot > 0.0, axis=1)
    neg_frac = np.mean(r_boot < 0.0, axis=1)
    prob_dir = np.where(valid_x, np.maximum(pos_frac, neg_frac), 0.5)
    bf10 = np.where(valid_x, _oracle_jzs_grid(r_obs, n_samples), 1.0)
    return {
        "r": r_obs, "p_param": p_param, "p_perm": p_perm, "q_fdr": q_fdr, "p_fwer": p_fwer,
        "ci95_low": ci95_low, "ci95_high": ci95_high, "prob_direction": prob_dir, "bf10": bf10,
    }


# antsxbrainmapping/src/antsxbrainmapping/multiview_simlr.py:1015
def _oracle_vectorized_paired_permutation_correlations(x_mat, y_mat, *, n_perms=10000, n_boot=2000, seed=42):
    n_samples, n_features = x_mat.shape
    x_c = x_mat - np.mean(x_mat, axis=0, keepdims=True)
    y_c = y_mat - np.mean(y_mat, axis=0, keepdims=True)
    x_sd = np.std(x_c, axis=0, ddof=1, keepdims=True)
    y_sd = np.std(y_c, axis=0, ddof=1, keepdims=True)
    valid = (x_sd.reshape(-1) > 1e-10) & (y_sd.reshape(-1) > 1e-10)
    x_z = np.where(valid.reshape(1, -1), x_c / np.where(x_sd > 1e-10, x_sd, 1.0), 0.0)
    y_z = np.where(valid.reshape(1, -1), y_c / np.where(y_sd > 1e-10, y_sd, 1.0), 0.0)
    r_obs = np.sum(x_z * y_z, axis=0) / float(n_samples - 1)
    r_obs = np.clip(r_obs, -0.9999999, 0.9999999)
    df = max(1, n_samples - 2)
    t_obs = r_obs * np.sqrt(df / np.maximum(1e-12, 1.0 - r_obs**2))
    p_param = np.where(valid, 2.0 * sps.t.sf(np.abs(t_obs), df=df), 1.0)
    rng = np.random.default_rng(int(seed))
    r_null = np.empty((n_features, int(n_perms)), dtype=np.float64)
    inv_denom = 1.0 / float(n_samples - 1)
    for b in range(int(n_perms)):
        perm_idx = rng.permutation(n_samples)
        r_null[:, b] = np.abs(np.sum(x_z * y_z[perm_idx, :], axis=0) * inv_denom)
    abs_obs = np.abs(r_obs)[:, None]
    n_exceed = np.sum(r_null >= abs_obs, axis=1)
    p_perm = np.where(valid, (1.0 + n_exceed) / (1.0 + float(n_perms)), 1.0)
    q_fdr = _oracle_bh_fdr_adjust(p_perm)
    max_r_null = np.max(r_null, axis=0)
    n_exceed_fwer = np.sum(max_r_null[None, :] >= abs_obs, axis=1)
    p_fwer = np.where(valid, (1.0 + n_exceed_fwer) / (1.0 + float(n_perms)), 1.0)
    boot_idx = rng.integers(0, n_samples, size=(int(n_boot), n_samples))
    r_boot = np.empty((n_features, int(n_boot)), dtype=np.float64)
    for b in range(int(n_boot)):
        idx_b = boot_idx[b]
        xb = x_mat[idx_b]
        yb = y_mat[idx_b]
        xb_c = xb - np.mean(xb, axis=0, keepdims=True)
        yb_c = yb - np.mean(yb, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(xb_c**2, axis=0) * np.sum(yb_c**2, axis=0))
        r_boot[:, b] = np.where(denom > 1e-12, np.sum(xb_c * yb_c, axis=0) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    ci95_low = np.where(valid, np.percentile(r_boot, 2.5, axis=1), 0.0)
    ci95_high = np.where(valid, np.percentile(r_boot, 97.5, axis=1), 0.0)
    prob_dir = np.where(valid, np.maximum(np.mean(r_boot > 0.0, axis=1), np.mean(r_boot < 0.0, axis=1)), 0.5)
    bf10 = np.where(valid, _oracle_jzs_grid(r_obs, n_samples), 1.0)
    return {
        "r": r_obs, "p_param": p_param, "p_perm": p_perm, "q_fdr": q_fdr, "p_fwer": p_fwer,
        "ci95_low": ci95_low, "ci95_high": ci95_high, "prob_direction": prob_dir, "bf10": bf10,
    }


# antsxbrainmapping/src/antsxbrainmapping/triangulation_variance.py:494
def _oracle_freedman_lane(x_target, z_nuis, y, *, n_permutations, rng):
    n = int(len(y))  # noqa: RUF046 -- verbatim copy of the source
    Z = np.column_stack([np.ones(n, dtype=np.float64), z_nuis])
    Z_pinv = np.linalg.pinv(Z)
    yhat_nuis = Z @ (Z_pinv @ y)
    r_y = y - yhat_nuis
    r_x = x_target - Z @ (Z_pinv @ x_target)
    sx = float(np.std(r_x))
    sy = float(np.std(r_y))
    rx_z = (r_x - float(np.mean(r_x))) / sx
    ry_z = (r_y - float(np.mean(r_y))) / sy
    part_r = float(np.clip(np.mean(rx_z * ry_z), -1.0, 1.0))
    Q_z, _, _ = np.linalg.svd(Z, full_matrices=False)
    perm_idx = np.array([rng.permutation(n) for _ in range(int(n_permutations))], dtype=np.int64)
    ry_perm = r_y[perm_idx]
    ry_perp = ry_perm - (ry_perm @ Q_z) @ Q_z.T
    norms = np.linalg.norm(ry_perp, axis=1)
    rx_unit = (r_x - float(np.mean(r_x))) / max(float(np.linalg.norm(r_x - float(np.mean(r_x)))), 1e-12)
    r_null = np.where(norms > 1e-12, (ry_perp @ rx_unit) / np.maximum(norms, 1e-12), 0.0)
    p_val = float((np.sum(np.abs(r_null) >= abs(part_r) - 1e-12) + 1) / (n_permutations + 1))
    return part_r, p_val


# antsxbrainmapping/src/antsxbrainmapping/triangulation_variance.py:715
def _oracle_one_sample_jzs(t_stat: float, n: int, r_scale: float = 0.7071067811865476) -> float:
    if n < 3 or not np.isfinite(t_stat):
        return 1.0
    nu = float(n - 1)
    t2 = float(t_stat) * float(t_stat)
    log_denom = -0.5 * (nu + 1.0) * np.log1p(t2 / nu)
    u = np.linspace(1e-4, 12.0, 1200, dtype=np.float64)
    g = 1.0 / (u * u)
    one_plus_ngr2 = 1.0 + float(n) * g * (r_scale * r_scale)
    log_integrand = (
        -0.5 * np.log(one_plus_ngr2)
        - 0.5 * (nu + 1.0) * np.log1p(t2 / (one_plus_ngr2 * nu))
        - 0.5 * u * u
        + 0.5 * np.log(2.0 / np.pi)
    )
    max_li = float(np.max(log_integrand))
    integral = float(np.trapezoid(np.exp(log_integrand - max_li), u)) * np.exp(max_li - log_denom)
    return float(np.clip(integral, 1e-12, 1e15))


# antsxbrainmapping/src/antsxbrainmapping/cognitive_networks.py:2306
def _oracle_cognitive_bh(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    if m == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(pvals)
    ranked_q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        orig_idx = order[i]
        q_val = min(prev, float(pvals[orig_idx]) * m / float(i + 1))
        ranked_q[orig_idx] = q_val
        prev = q_val
    return ranked_q


# antsxbrainmapping/scripts/run_four_hypothesis_triangulation.py:297
def _oracle_icc_3_k_across_days(mat_n_by_k: np.ndarray) -> tuple[float, float]:
    valid = np.all(np.isfinite(mat_n_by_k), axis=1)
    Y = mat_n_by_k[valid]
    n, k = Y.shape
    if n < 5 or k < 2:
        return float("nan"), float("nan")
    grand = float(np.mean(Y))
    row_m = np.mean(Y, axis=1, keepdims=True)
    col_m = np.mean(Y, axis=0, keepdims=True)
    ss_between = float(k * np.sum((row_m - grand) ** 2))
    ss_resid = float(np.sum((Y - row_m - col_m + grand) ** 2))
    ms_b = ss_between / max(n - 1, 1)
    ms_e = ss_resid / max((n - 1) * (k - 1), 1)
    denom_1 = ms_b + (k - 1) * ms_e
    icc_3_1 = float((ms_b - ms_e) / denom_1) if denom_1 > 1e-12 else 0.0
    icc_3_k = float((ms_b - ms_e) / ms_b) if ms_b > 1e-12 else 0.0
    return icc_3_1, icc_3_k


# antsxbrainmapping/scripts/run_four_hypothesis_triangulation.py:1361 (inline `_pool_r`)
def _oracle_pool_r(rs_raw, ns, expected_sign):
    w_iv = np.maximum(np.asarray(ns, dtype=np.float64) - 3.0, 1.0)
    sign_align = np.where(np.asarray(expected_sign, dtype=np.float64) < 0.0, -1.0, 1.0)
    rs = np.clip(sign_align * np.asarray(rs_raw, dtype=np.float64), -0.999, 0.999)
    zs = np.arctanh(rs)
    z_bar = float(np.sum(w_iv * zs) / np.sum(w_iv))
    se_z = float(1.0 / np.sqrt(np.sum(w_iv)))
    p_two = float(2.0 * sps.norm.sf(abs(z_bar / se_z)))
    return float(np.tanh(z_bar)), p_two


# antsxbrainmapping/scripts/run_four_hypothesis_triangulation.py:1374 (inline Stouffer block)
def _oracle_stouffer(p_dir, nw):
    nw = np.asarray(nw, dtype=np.float64)
    z_ind = sps.norm.isf(np.clip(np.asarray(p_dir, dtype=np.float64), 1e-15, 1.0 - 1e-15))
    z_stouffer = float(np.sum(np.sqrt(nw) * z_ind) / np.sqrt(np.sum(nw)))
    return z_stouffer, float(sps.norm.sf(z_stouffer))


# pysimlr/screening.py `_bh_qvalues` as of v0.22.2 (promoted into stats.bh_fdr)
def _oracle_screening_bh_qvalues(p: np.ndarray) -> np.ndarray:
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


# pysimlr/benchmarks/deep_ranking.py `holm_adjust` as of v0.22.2 (moved into stats.holm)
def _oracle_deep_ranking_holm(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, p[idx] * (m - rank))
        adj[idx] = min(running, 1.0)
    return adj


# ------------------------------------------------------------------------ multiplicity
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("m", [1, 2, 7, 50])
def test_bh_fdr_matches_every_source_implementation(seed, m):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=m) ** rng.uniform(0.3, 3.0)
    q = bh_fdr(p)
    for oracle in (_oracle_benjamini_hochberg, _oracle_bh_fdr_adjust, _oracle_cognitive_bh,
                   _oracle_screening_bh_qvalues):
        np.testing.assert_allclose(q, oracle(p.copy()), rtol=0, atol=1e-14)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("m", [1, 2, 7, 50])
def test_holm_matches_every_source_implementation(seed, m):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=m) ** rng.uniform(0.3, 3.0)
    h = holm(p)
    for oracle in (_oracle_holm_bonferroni, _oracle_deep_ranking_holm):
        np.testing.assert_allclose(h, oracle(p.copy()), rtol=0, atol=1e-14)


def test_bh_and_holm_on_known_p_values():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    np.testing.assert_allclose(bh_fdr(p), [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    np.testing.assert_allclose(holm(p), [0.04, 0.09, 0.09, 0.5])
    # Ties and the cap at 1.
    np.testing.assert_allclose(bh_fdr([0.5, 0.5, 0.5]), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(holm([0.5, 0.6, 0.9]), [1.0, 1.0, 1.0])
    assert bh_fdr([]).shape == (0,)
    assert holm([]).shape == (0,)


def test_bh_and_holm_are_monotone_and_bounded():
    p = np.random.default_rng(7).uniform(size=200)
    for f in (bh_fdr, holm):
        q = f(p)
        assert np.all(q >= p - 1e-15) and np.all(q <= 1.0)
        order = np.argsort(p)
        assert np.all(np.diff(q[order]) >= -1e-15)


@pytest.mark.parametrize("bad", [[0.1, np.nan], [0.1, 1.5], [[0.1, 0.2]], [-0.1]])
def test_multiplicity_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        bh_fdr(bad)
    with pytest.raises(ValueError):
        holm(bad)


def test_screening_and_deep_ranking_use_the_shared_helpers():
    from pysimlr import screening
    from pysimlr.benchmarks import deep_ranking

    assert screening.bh_fdr is bh_fdr
    assert not hasattr(screening, "_bh_qvalues")
    p = np.array([0.02, 0.4, 0.01, 0.2])
    np.testing.assert_array_equal(deep_ranking.holm_adjust(p), holm(p))


def test_stats_names_resolve_lazily_from_the_package():
    import pysimlr

    assert pysimlr.bh_fdr is bh_fdr
    assert pysimlr.icc_3 is icc_3
    assert pysimlr.stats.__name__ == "pysimlr.stats"
    assert "bh_fdr" in pysimlr.__all__


# ----------------------------------------------------------------------- Bayes factors
R_SWEEP = np.round(np.arange(-0.95, 0.951, 0.05), 2)
N_SWEEP = (10, 30, 52, 146)


@pytest.mark.parametrize("n", N_SWEEP)
def test_jzs_bf10_correlation_matches_the_quadrature_source(n):
    bf = jzs_bf10_correlation(R_SWEEP, n)
    expected = np.array([_oracle_jzs_quad(float(r), n)[0] for r in R_SWEEP])
    np.testing.assert_allclose(bf, expected, rtol=1e-12)
    assert isinstance(jzs_bf10_correlation(0.3, n), float)
    assert jzs_bf10_correlation(0.3, n) == jzs_bf10_correlation(-0.3, n)


@pytest.mark.parametrize("n", N_SWEEP)
def test_jzs_bf10_correlation_grid_matches_the_grid_source(n):
    bf = jzs_bf10_correlation_grid(R_SWEEP, n)
    np.testing.assert_allclose(bf, _oracle_jzs_grid(R_SWEEP, n), rtol=0, atol=0)
    assert isinstance(jzs_bf10_correlation_grid(0.3, n), float)


def test_the_two_correlation_implementations_disagree_as_documented():
    """The reason both are exported: no cell of the sweep agrees to 1e-6 and the grid clips at e^50."""
    worst = 0.0
    for n in N_SWEEP:
        q = jzs_bf10_correlation(R_SWEEP, n)
        g = jzs_bf10_correlation_grid(R_SWEEP, n)
        rel = np.abs(g - q) / q
        assert np.all(rel > 1e-6)
        worst = max(worst, float(rel.max()))
    assert worst > 0.99
    assert jzs_bf10_correlation_grid(0.95, 146) == pytest.approx(np.exp(50.0))
    assert jzs_bf10_correlation(0.95, 146) > 1e70


def test_jzs_bf10_correlation_against_a_fine_log_grid_reference():
    def ref(r, n):
        u = np.linspace(-30, 30, 400_001)
        g = np.exp(u)
        li = 0.5 * (n - 2) * np.log1p(g) - 0.5 * (n - 1) * np.log1p((1 - r * r) * g) - 0.5 * u - n / (2 * g)
        m = li.max()
        return np.exp(0.5 * np.log(n / 2) - special.gammaln(0.5) + m) * np.trapezoid(np.exp(li - m), u)

    for r, n in ((0.0, 10), (0.4, 27), (-0.6, 52), (0.2, 146)):
        assert jzs_bf10_correlation(r, n) == pytest.approx(ref(r, n), rel=1e-8)


def test_jzs_bf10_correlation_rscale_is_the_zellner_siow_scale():
    """``rscale`` rescales the InvGamma rate; equal to the explicit integral and monotone at r=0."""
    def ref(r, n, s):
        def f(g):
            return (1 + g) ** ((n - 2) / 2) * (1 + (1 - r * r) * g) ** (-(n - 1) / 2) * g ** -1.5 * np.exp(-n * s * s / (2 * g))
        val, _ = integrate.quad(f, 0, np.inf, limit=200)
        return np.sqrt(n * s * s / 2) / special.gamma(0.5) * val

    for s in (0.5, 1.0, 1.5):
        assert jzs_bf10_correlation(0.3, 30, rscale=s) == pytest.approx(ref(0.3, 30, s), rel=1e-7)
    # A wider prior spreads mass to large effects: BF10 at r=0 falls with rscale.
    assert jzs_bf10_correlation(0.0, 30, rscale=0.5) > jzs_bf10_correlation(0.0, 30, rscale=1.5)


@pytest.mark.parametrize("n", [5, 10, 30, 61, 146])
def test_jzs_bf10_one_sample_t_matches_the_source_to_its_grid_error(n):
    """The source integrates on a 1,200-point trapezoid grid; its measured error is 1.7e-4 relative."""
    for t in np.linspace(-8.0, 8.0, 9):
        assert jzs_bf10_one_sample_t(float(t), n) == pytest.approx(_oracle_one_sample_jzs(float(t), n), rel=5e-4)


def test_jzs_bf10_one_sample_t_against_direct_quadrature():
    def ref(t, n, s):
        nu = n - 1

        def f(g):
            a = 1 + n * g * s * s
            return a ** -0.5 * (1 + t * t / (a * nu)) ** (-(nu + 1) / 2) * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g))

        num, _ = integrate.quad(f, 0, np.inf, limit=400, epsabs=0, epsrel=1e-10)
        return num / (1 + t * t / nu) ** (-(nu + 1) / 2)

    for t, n, s in ((0.0, 10, np.sqrt(2) / 2), (2.5, 20, np.sqrt(2) / 2), (-4.0, 61, 1.0), (1.0, 5, 0.5)):
        assert jzs_bf10_one_sample_t(t, n, rscale=s) == pytest.approx(ref(t, n, s), rel=1e-7)
    assert jzs_bf10_one_sample_t(3.0, 20) == jzs_bf10_one_sample_t(-3.0, 20)
    assert jzs_bf10_one_sample_t(0.0, 20) < 1.0 < jzs_bf10_one_sample_t(4.0, 20)


def test_bayes_factors_reject_bad_input():
    with pytest.raises(ValueError):
        jzs_bf10_correlation(0.3, 3)
    with pytest.raises(ValueError):
        jzs_bf10_correlation_grid(0.3, 3)
    with pytest.raises(ValueError):
        jzs_bf10_correlation(np.nan, 10)
    with pytest.raises(ValueError):
        jzs_bf10_correlation(1.2, 10)
    with pytest.raises(ValueError):
        jzs_bf10_correlation(0.3, 10, rscale=0.0)
    with pytest.raises(ValueError):
        jzs_bf10_one_sample_t(1.0, 1)
    with pytest.raises(ValueError):
        jzs_bf10_one_sample_t(np.inf, 10)


# ------------------------------------------------------------- permutation inference
NON_BF_KEYS = ("r", "p_param", "p_perm", "q_fdr", "p_fwer", "ci95_low", "ci95_high", "prob_direction")


@pytest.mark.parametrize("seed", [3, 11])
def test_permutation_correlations_unpaired_matches_the_source(seed):
    rng = np.random.default_rng(seed)
    n, p = 24, 6
    X = rng.standard_normal((n, p))
    y = X[:, 0] * 0.8 + rng.standard_normal(n)
    X[:, 3] = 2.0  # degenerate column, kept and annotated
    got = permutation_correlations(X, y, n_perms=150, n_boot=60, seed=seed)
    want = _oracle_vectorized_permutation_correlations(X, y, n_perms=150, n_boot=60, seed=seed)
    for key in NON_BF_KEYS:
        np.testing.assert_allclose(got[key], want[key], rtol=0, atol=1e-13, err_msg=key)
    assert got["valid"].tolist() == [True, True, True, False, True, True]
    assert got["bf10"][3] == 1.0
    np.testing.assert_allclose(got["bf10"], np.where(got["valid"], jzs_bf10_correlation(got["r"], n), 1.0))
    # The source's BF10 came from the grid: agrees only to its quadrature error.
    np.testing.assert_allclose(got["bf10"], want["bf10"], rtol=2e-3)


@pytest.mark.parametrize("seed", [5, 13])
def test_permutation_correlations_paired_matches_the_source(seed):
    rng = np.random.default_rng(seed)
    n, p = 20, 5
    X = rng.standard_normal((n, p))
    Y = 0.6 * X + rng.standard_normal((n, p))
    Y[:, 1] = -1.0
    got = permutation_correlations(X, Y, n_perms=120, n_boot=50, seed=seed, paired=True)
    want = _oracle_vectorized_paired_permutation_correlations(X, Y, n_perms=120, n_boot=50, seed=seed)
    for key in NON_BF_KEYS:
        np.testing.assert_allclose(got[key], want[key], rtol=0, atol=1e-13, err_msg=key)
    assert got["valid"].tolist() == [True, False, True, True, True]
    np.testing.assert_allclose(got["bf10"], want["bf10"], rtol=2e-3)


def test_permutation_correlations_is_deterministic_and_skips_bootstrap_by_default():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 4))
    y = rng.standard_normal(15)
    a = permutation_correlations(X, y, n_perms=50, seed=9)
    b = permutation_correlations(X, y, n_perms=50, seed=9)
    c = permutation_correlations(X, y, n_perms=50, seed=10)
    for key in ("r", "p_perm", "p_fwer", "q_fdr", "bf10"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["p_perm"], c["p_perm"]) or not np.array_equal(a["p_fwer"], c["p_fwer"])
    for key in ("ci95_low", "ci95_high", "prob_direction"):
        assert np.all(np.isnan(a[key])) and a[key].shape == (4,)
    np.testing.assert_allclose(a["r"], [np.corrcoef(X[:, j], y)[0, 1] for j in range(4)], atol=1e-12)
    np.testing.assert_allclose(a["q_fdr"], bh_fdr(a["p_perm"]))
    assert np.all(a["p_perm"] >= 1.0 / 51.0) and np.all(a["p_fwer"] >= a["p_perm"] - 1e-15)


def test_permutation_correlations_rejects_bad_input():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 3))
    y = rng.standard_normal(10)
    with pytest.raises(ValueError):
        permutation_correlations(X[:3], y[:3], n_perms=10, seed=0)
    with pytest.raises(ValueError):
        permutation_correlations(X, y[:-1], n_perms=10, seed=0)
    with pytest.raises(ValueError):
        permutation_correlations(X, np.ones(10), n_perms=10, seed=0)
    with pytest.raises(ValueError):
        permutation_correlations(X, y, n_perms=0, seed=0)
    with pytest.raises(ValueError):
        permutation_correlations(X, y, n_perms=10, n_boot=-1, seed=0)
    with pytest.raises(ValueError):
        permutation_correlations(X, X[:, :2], n_perms=10, seed=0, paired=True)
    # a non-finite outcome row is masked: dropped with a warning (it used to raise); nothing left raises
    with pytest.warns(UserWarning, match="dropped 1 masked"):
        permutation_correlations(X, np.r_[y[:-1], np.nan], n_perms=10, seed=0)
    with pytest.raises(ValueError):
        permutation_correlations(X, np.full(10, np.nan), n_perms=10, seed=0)


@pytest.mark.parametrize("seed", [2, 8])
def test_freedman_lane_partial_p_matches_the_source(seed):
    rng = np.random.default_rng(seed)
    n = 30
    z = rng.standard_normal((n, 2))
    x = z @ [0.5, -0.3] + rng.standard_normal(n)
    y = 0.4 * x + z @ [1.0, 0.2] + rng.standard_normal(n)
    got = freedman_lane_partial_p(y, x, z, n_perms=200, seed=seed)
    want = _oracle_freedman_lane(x, z, y, n_permutations=200, rng=np.random.default_rng(seed))
    assert got == pytest.approx(want, rel=0, abs=1e-14)
    # Partial r equals the correlation of OLS residuals; a 1-d covariate is accepted.
    assert freedman_lane_partial_p(y, x, z[:, 0], n_perms=20, seed=0)[0] == pytest.approx(
        np.corrcoef(y - np.polyval(np.polyfit(z[:, 0], y, 1), z[:, 0]),
                    x - np.polyval(np.polyfit(z[:, 0], x, 1), z[:, 0]))[0, 1])
    assert freedman_lane_partial_p(y, x, z, n_perms=200, seed=seed) == got


def test_freedman_lane_partial_p_rejects_bad_input():
    rng = np.random.default_rng(0)
    y, x, z = rng.standard_normal(10), rng.standard_normal(10), rng.standard_normal(10)
    with pytest.raises(ValueError):
        freedman_lane_partial_p(y[:3], x[:3], z[:3], n_perms=10, seed=0)
    with pytest.raises(ValueError):
        freedman_lane_partial_p(y, x[:-1], z, n_perms=10, seed=0)
    with pytest.raises(ValueError):
        freedman_lane_partial_p(y, np.ones(10), z, n_perms=10, seed=0)
    with pytest.raises(ValueError):
        freedman_lane_partial_p(y, z, z, n_perms=10, seed=0)  # x in span(covariates)
    with pytest.raises(ValueError):
        freedman_lane_partial_p(y, x, z, n_perms=0, seed=0)


# ------------------------------------------------------------- meta-analytic pooling
@pytest.mark.parametrize("seed", [0, 4])
def test_pooled_correlation_fisher_z_matches_the_script(seed):
    rng = np.random.default_rng(seed)
    k = 5
    rs = rng.uniform(-0.7, 0.7, size=k)
    ns = rng.integers(2, 150, size=k)
    signs = rng.choice([-1, 0, 1], size=k)
    assert pooled_correlation_fisher_z(rs, ns, signs) == _oracle_pool_r(rs, ns, signs)
    assert pooled_correlation_fisher_z(rs, ns) == _oracle_pool_r(rs, ns, np.ones(k))


def test_pooled_correlation_fisher_z_analytic():
    r, p = pooled_correlation_fisher_z([0.5, 0.5], [30, 30])
    assert r == pytest.approx(0.5)
    assert p == pytest.approx(2 * sps.norm.sf(np.arctanh(0.5) * np.sqrt(54)))
    r_flip, _ = pooled_correlation_fisher_z([0.5, -0.5], [30, 30], signs=[1, -1])
    assert r_flip == pytest.approx(0.5)
    assert pooled_correlation_fisher_z([0.0, 0.0], [10, 10])[1] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 4])
def test_stouffer_z_matches_the_script(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=4)
    nw = rng.integers(5, 100, size=4)
    assert stouffer_z(p, nw) == _oracle_stouffer(p, nw)


def test_stouffer_z_analytic():
    # Four equal one-sided p: Z = 2 z_1 and p shrinks accordingly.
    z1 = sps.norm.isf(0.05)
    z, p = stouffer_z([0.05] * 4)
    assert z == pytest.approx(2 * z1)
    assert p == pytest.approx(sps.norm.sf(2 * z1))
    # Equal weights are a no-op versus the unweighted form.
    assert stouffer_z([0.05] * 4, [7, 7, 7, 7]) == pytest.approx(stouffer_z([0.05] * 4))
    assert stouffer_z([0.5, 0.5]) == pytest.approx((0.0, 0.5))


def test_pooling_rejects_bad_input():
    with pytest.raises(ValueError):
        pooled_correlation_fisher_z([], [])
    with pytest.raises(ValueError):
        pooled_correlation_fisher_z([0.1, 0.2], [10])
    with pytest.raises(ValueError):
        pooled_correlation_fisher_z([0.1, 0.2], [10, 20], signs=[1])
    with pytest.raises(ValueError):
        stouffer_z([0.1, 1.2])
    with pytest.raises(ValueError):
        stouffer_z([0.1, 0.2], [1.0, 0.0])
    with pytest.raises(ValueError):
        stouffer_z([])


# ---------------------------------------------------------------------- reliability
@pytest.mark.parametrize("seed", [0, 6])
def test_icc_3_matches_the_script(seed):
    rng = np.random.default_rng(seed)
    n, k = 25, 3
    subject = rng.standard_normal((n, 1))
    Y = subject + 0.3 * rng.standard_normal((n, k)) + np.array([[0.0, 0.5, -0.2]])
    Y[2, 1] = np.nan  # incomplete row is dropped, as in the script
    assert icc_3(Y) == pytest.approx(_oracle_icc_3_k_across_days(Y), rel=0, abs=1e-15)


def test_icc_3_on_a_known_table():
    # Shrout & Fleiss (1979) Table 2: 6 targets, 4 judges. ICC(3,1)=0.71, ICC(3,k)=0.91.
    Y = np.array([
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ], dtype=float)
    icc_1, icc_k = icc_3(Y)
    assert icc_1 == pytest.approx(0.7148, abs=5e-4)
    assert icc_k == pytest.approx(0.9093, abs=5e-4)
    # Perfect consistency: a column shift leaves ICC(3,.) at 1.
    base = np.arange(6.0)[:, None]
    assert icc_3(base + np.array([[0.0, 1.0, 2.0]])) == pytest.approx((1.0, 1.0))
    # Identical subjects: both undefined, annotated NaN.
    assert all(np.isnan(v) for v in icc_3(np.ones((6, 3))))


def test_icc_3_rejects_bad_input():
    with pytest.raises(ValueError):
        icc_3(np.ones((4, 3)))
    with pytest.raises(ValueError):
        icc_3(np.ones((6, 1)))
    with pytest.raises(ValueError):
        icc_3(np.ones(6))
    Y = np.random.default_rng(0).standard_normal((6, 2))
    Y[:2, 0] = np.nan
    with pytest.raises(ValueError):
        icc_3(Y)


# --------------------------------------------------------------------------- golden
def _october_root() -> str | None:
    root = os.environ.get(OCTOBER_ROOT_ENV)
    if root is None:
        return None
    return root or OCTOBER_ROOT_DEFAULT


def _ledgers() -> list[str]:
    root = _october_root()
    if root is None:
        return []
    return sorted(glob.glob(os.path.join(root, "four_study_transport", "results", "*_hypothesis_ledger.csv")))


@pytest.mark.golden
@pytest.mark.skipif(_october_root() is None, reason=f"{OCTOBER_ROOT_ENV} not set")
def test_golden_four_study_ledgers_are_reproduced_from_r_n_and_p():
    """``bf10``/``bf01_null`` from (``pearson_r``, ``n_subjects``); ``q_fdr_bh``/``p_fwer_holm`` per family.

    Column provenance: antsxbrainmapping/src/antsxbrainmapping/discovery_transport.py:1332-1368.
    The multiplicity columns are computed within ``family_id`` on the
    directional permutation p when ``expected_sign != 0`` and the two-sided p
    otherwise.
    """
    ledgers = _ledgers()
    assert ledgers, f"no *_hypothesis_ledger.csv under {_october_root()}"
    n_rows = 0
    for path in ledgers:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, path
        for row in rows:
            r = float(row["pearson_r"])
            n = int(row["n_subjects"])
            bf10 = jzs_bf10_correlation(r, n)
            assert bf10 == pytest.approx(float(row["bf10"]), rel=1e-9), (path, row["hypothesis_id"])
            assert 1.0 / bf10 == pytest.approx(float(row["bf01_null"]), rel=1e-9), (path, row["hypothesis_id"])
        families: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            families[row["family_id"]].append(row)
        for fam, members in families.items():
            p_use = np.array([
                float(m["p_perm_directional"]) if int(m["expected_sign"]) != 0 else float(m["p_perm_two_sided"])
                for m in members
            ])
            np.testing.assert_allclose(bh_fdr(p_use), [float(m["q_fdr_bh"]) for m in members], rtol=1e-12,
                                       err_msg=f"{path}:{fam}:q_fdr_bh")
            np.testing.assert_allclose(holm(p_use), [float(m["p_fwer_holm"]) for m in members], rtol=1e-12,
                                       err_msg=f"{path}:{fam}:p_fwer_holm")
        n_rows += len(rows)
    assert n_rows >= 7
