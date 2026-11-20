#!/usr/bin/env python3
"""Synthetic Theorem Verification (Props 1-3) & Multi-View Rank-1/Rank-2 Benchmark.

Verifies numerically:
- Prop 1: Order-statistic soft-thresholding (`_apply_quantile_sparseness(alg="soft")`)
  identically cancels uniform shrinkage alpha in [0, 1); `_apply_soft_threshold_sparseness`
  likewise cancels the uniform floor.
- Prop 2: Exact Euclidean/cosine projection onto C_m = {u >= 0 : ||u||_0 <= m, ||u||_2 = 1}
  is non-subtractive top-m restriction; legacy Euclidean simplex projection
  (`project_columns_to_simplex`) silently subtracts theta > 0 from unit-l2 columns even when
  `sparseness=None`, whereas native radial `_convexify` (`v_+ / 1^T v_+`) preserves exact
  support and direction.
- Prop 3: `ColumnSupportShrinkageOperator` with pairwise-disjoint supports commutes with
  exact conical orthogonality (`||V^T V - I_2||_F < 1e-14`) for every alpha in [0, 1],
  whereas global `UniformShrinkageOperator` collapses columns toward rank-1 as alpha -> 1.

Benchmarks (M=4 views, n_train=60, n_test=200, p=(80, 100, 60, 160), block-correlated noise):
- Design A: Rank-1 sparse (30% active positive features per view).
- Design B: Rank-2 disjoint non-negative modes (z_1 perp z_2; 25% + 25% disjoint active blocks).

Supports `--smoke` (<10s synchronous gate with n_seeds=2) and full 20-seed parallel run.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch
import warnings

# Guard against parent workspace cwd shadowing installed site-packages/nsa_flow
sys.path[:] = [
    _p
    for _p in sys.path
    if not (
        (Path(_p or ".").resolve() / "nsa_flow").is_dir()
        and not (Path(_p or ".").resolve() / "nsa_flow" / "__init__.py").is_file()
    )
]

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pysimlr  # noqa: E402
from pysimlr import sparsification as sp_mod  # noqa: E402
from pysimlr.uniform_shrinkage import (  # noqa: E402
    ColumnSupportShrinkageOperator,
    UniformShrinkageOperator,
)
import torch  # noqa: E402

DEFAULT_OUT_JSON = REPO_DIR / "paper" / "results_cache" / "v043_synthetic_theorems.json"
VIEW_DIMS = (80, 100, 60, 160)


def _exact_conical_topk_projection(y: np.ndarray, m: int) -> np.ndarray:
    """Exact Euclidean/cosine projection of ``y`` onto ``C_m = {u >= 0 : ||u||_0 <= m, ||u||_2 = 1}``."""
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    p = int(arr.size)
    m_i = int(m)
    if not (1 <= m_i <= p):
        raise ValueError(f"m must satisfy 1 <= m <= {p}; got {m}.")
    y_pos = np.maximum(arr, 0.0)
    if not np.any(y_pos > 0.0):
        raise ValueError("y has no strictly positive entries; conical top-m projection is undefined.")
    order = np.argsort(-y_pos, kind="mergesort")
    sel = order[:m_i]
    out = np.zeros(p, dtype=np.float64)
    out[sel] = y_pos[sel]
    nrm = float(np.linalg.norm(out))
    if nrm <= 0.0:
        raise ValueError("Top-m positive restriction has zero norm.")
    return out / nrm


@contextmanager
def _legacy_euclidean_convex_mode() -> Iterator[None]:
    """Patch `pysimlr.sparsification._convexify` to legacy Euclidean simplex projection."""
    with patch("pysimlr.sparsification._convexify", side_effect=lambda v: sp_mod.project_columns_to_simplex(v, support_only=True)):
        yield


def verify_propositions() -> dict[str, Any]:
    """Run exact numerical verifications for Propositions 1, 2, and 3."""
    rng = np.random.default_rng(20261119)

    # ------------------------------------------------------------------ Prop 1: alpha cancellation
    p1 = 100
    v_base = rng.lognormal(mean=0.0, sigma=0.7, size=(p1, 1)).astype(np.float64)
    v_base /= np.linalg.norm(v_base)
    q1 = 0.30
    alphas_grid = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.99]

    ref_q_soft = None
    max_diff_quantile_soft = 0.0
    soft_thresh_rows: list[dict[str, float]] = []
    for a in alphas_grid:
        op = UniformShrinkageOperator(p1, alpha=a, positive_part=True)
        v_tilde = torch.as_tensor(op @ v_base, dtype=torch.float64)
        u_q = sp_mod._apply_quantile_sparseness(v_tilde, q1, nonneg=True, alg="soft")
        u_q = (u_q / u_q.norm(dim=0, keepdim=True)).numpy()[:, 0]
        if ref_q_soft is None:
            ref_q_soft = u_q.copy()
        else:
            max_diff_quantile_soft = max(max_diff_quantile_soft, float(np.max(np.abs(u_q - ref_q_soft))))

        u_s = sp_mod._apply_soft_threshold_sparseness(v_tilde, q1, "conical")
        u_s_np = (u_s / u_s.norm(dim=0, keepdim=True)).numpy()[:, 0]
        pos_mask = u_s_np > 0.0
        v_t_np = v_tilde.numpy()[:, 0]
        tau_val = 0.5 * float(np.quantile(v_t_np, q1))
        floor_val = a * float(np.mean(v_base[:, 0]))
        soft_thresh_rows.append(
            {
                "alpha": float(a),
                "uniform_floor": floor_val,
                "soft_tau": tau_val,
                "cos_with_alpha0_quantile_soft": float(np.dot(u_q, ref_q_soft)),
                "cos_with_alpha0_soft_thresh": float(np.dot(u_s_np, ref_q_soft)),
                "active_fraction_soft_thresh": float(np.mean(pos_mask)),
            }
        )

    prop1 = {
        "p": p1,
        "q": q1,
        "alphas_tested": alphas_grid,
        "max_linf_diff_quantile_soft_vs_alpha0": float(max_diff_quantile_soft),
        "exact_alpha_cancellation_verified": bool(max_diff_quantile_soft < 1e-14),
        "soft_threshold_floor_erosion": soft_thresh_rows,
    }

    # ------------------------------------------------------------------ Prop 2: exact top-m & _convexify defect
    p2, m2 = 60, 20
    y2 = rng.standard_normal(p2) + 0.5
    u_exact = _exact_conical_topk_projection(y2, m2)
    y_pos = np.maximum(y2, 0.0)
    top_idx = np.argsort(-y_pos, kind="mergesort")[:m2]
    cs_bound = float(np.linalg.norm(y_pos[top_idx]))
    cs_err = abs(float(np.dot(y2, u_exact)) - cs_bound)

    y_m1 = float(np.sort(y_pos)[::-1][m2])
    u_soft_m = np.maximum(y_pos - y_m1, 0.0)
    u_soft_m /= np.linalg.norm(u_soft_m)
    euclid_exact = float(np.sum((y2 - u_exact) ** 2))
    euclid_soft = float(np.sum((y2 - u_soft_m) ** 2))
    cos_exact = float(np.dot(y2, u_exact) / np.linalg.norm(y2))
    cos_soft = float(np.dot(y2, u_soft_m) / np.linalg.norm(y2))

    max_random_violation = 0.0
    for _ in range(2000):
        idx_r = rng.choice(p2, size=m2, replace=False)
        u_r = np.zeros(p2, dtype=np.float64)
        w_r = np.abs(rng.standard_normal(m2))
        u_r[idx_r] = w_r / np.linalg.norm(w_r)
        viol = float(np.dot(y2, u_r) - np.dot(y2, u_exact))
        if viol > max_random_violation:
            max_random_violation = viol

    # Legacy Euclidean simplex vs native radial `_convexify` on unit-l2 positive vectors (sparseness=None!)
    p_cvx = 100
    u_l2 = np.abs(rng.standard_normal((p_cvx, 1))) + 0.05
    u_l2 = u_l2 / np.linalg.norm(u_l2, axis=0, keepdims=True)
    u_l2_t = torch.as_tensor(u_l2, dtype=torch.float64)
    w_euclid = sp_mod.project_columns_to_simplex(u_l2_t, support_only=True).numpy()[:, 0]
    w_radial = sp_mod._convexify(u_l2_t).numpy()[:, 0]

    active_e = w_euclid > 0.0
    theta_sub = float(np.mean(u_l2[active_e, 0] - w_euclid[active_e]))
    frac_zeroed_euclid = float(np.mean(~active_e))
    frac_zeroed_radial = float(np.mean(w_radial <= 0.0))
    cos_euclid_vs_u = float(np.dot(u_l2[:, 0], w_euclid) / (np.linalg.norm(u_l2[:, 0]) * np.linalg.norm(w_euclid)))
    cos_radial_vs_u = float(np.dot(u_l2[:, 0], w_radial) / (np.linalg.norm(u_l2[:, 0]) * np.linalg.norm(w_radial)))

    prop2 = {
        "p": p2,
        "m": m2,
        "cauchy_schwarz_abs_error": float(cs_err),
        "cauchy_schwarz_verified": bool(cs_err < 1e-14),
        "max_random_feasible_violation": float(max_random_violation),
        "euclid_sq_err_exact_topk": euclid_exact,
        "euclid_sq_err_soft_thresh": euclid_soft,
        "cos_sim_exact_topk": cos_exact,
        "cos_sim_soft_thresh": cos_soft,
        "convexify_defect_p": p_cvx,
        "unit_l2_l1_sum": float(np.sum(u_l2[:, 0])),
        "legacy_convexify_theta_subtracted": theta_sub,
        "legacy_convexify_fraction_zeroed_at_sparseness_none": frac_zeroed_euclid,
        "radial_convexify_fraction_zeroed_at_sparseness_none": frac_zeroed_radial,
        "legacy_convexify_cos_with_input": cos_euclid_vs_u,
        "radial_convexify_cos_with_input": cos_radial_vs_u,
    }

    # ------------------------------------------------------------------ Prop 3: k=2 commutation
    p3 = 100
    s1 = np.arange(0, 40, dtype=np.int64)
    s2 = np.arange(40, 80, dtype=np.int64)
    V0 = np.zeros((p3, 2), dtype=np.float64)
    V0[s1, 0] = rng.uniform(0.2, 1.5, size=s1.size)
    V0[s2, 1] = rng.uniform(0.2, 1.5, size=s2.size)
    V0 /= np.linalg.norm(V0, axis=0, keepdims=True)

    prop3_sweep: list[dict[str, float]] = []
    max_col_op_fro = 0.0
    for a in [0.0, 0.1, 0.25, 0.5, 0.75, 0.85, 1.0]:
        col_op = ColumnSupportShrinkageOperator(p3, supports=[s1, s2], alphas=[a, a], positive_part=True)
        Vc = col_op @ V0
        Vc /= np.linalg.norm(Vc, axis=0, keepdims=True)
        fro_c = float(np.linalg.norm(Vc.T @ Vc - np.eye(2), ord="fro"))
        cos_c = float(abs(np.dot(Vc[:, 0], Vc[:, 1])))
        max_col_op_fro = max(max_col_op_fro, fro_c)

        glob_op = UniformShrinkageOperator(p3, alpha=a, positive_part=True)
        Vg = glob_op @ V0
        Vg /= np.linalg.norm(Vg, axis=0, keepdims=True)
        fro_g = float(np.linalg.norm(Vg.T @ Vg - np.eye(2), ord="fro"))
        cos_g = float(abs(np.dot(Vg[:, 0], Vg[:, 1])))

        prop3_sweep.append(
            {
                "alpha": float(a),
                "col_support_fro_err": fro_c,
                "col_support_cos": cos_c,
                "global_smoother_fro_err": fro_g,
                "global_smoother_cos": cos_g,
            }
        )

    prop3 = {
        "p": p3,
        "support_sizes": [int(s1.size), int(s2.size)],
        "max_col_support_fro_error": float(max_col_op_fro),
        "exact_orthogonality_verified": bool(max_col_op_fro < 1e-14),
        "alpha_sweep": prop3_sweep,
    }

    return {"prop1": prop1, "prop2": prop2, "prop3": prop3}


def _block_correlated_noise(
    rng: np.random.Generator,
    n: int,
    p: int,
    *,
    block_size: int = 5,
    rho: float = 0.45,
) -> np.ndarray:
    """Generate unit-variance noise with intra-block correlation ``rho``."""
    out = np.empty((n, p), dtype=np.float64)
    s_shared = float(np.sqrt(rho))
    s_ind = float(np.sqrt(max(1.0 - rho, 1e-6)))
    for start in range(0, p, block_size):
        end = min(p, start + block_size)
        blk_z = rng.standard_normal((n, 1))
        out[:, start:end] = s_shared * blk_z + s_ind * rng.standard_normal((n, end - start))
    return out


def _make_design_a_data(
    seed: int,
    *,
    n_train: int = 60,
    n_test: int = 200,
    view_dims: tuple[int, ...] = VIEW_DIMS,
    active_frac: float = 0.30,
    signal_sd: float = 0.52,
    noise_rho: float = 0.45,
) -> dict[str, Any]:
    """Design A: Rank-1 sparse (30% active positive features per view) + block-correlated noise."""
    rng = np.random.default_rng(seed)
    n_tot = n_train + n_test
    z1 = rng.standard_normal(n_tot)
    z1 = (z1 - z1.mean()) / z1.std()

    views_tr: list[np.ndarray] = []
    views_te: list[np.ndarray] = []
    true_supports: list[np.ndarray] = []
    true_weights: list[np.ndarray] = []

    for p in view_dims:
        m_act = max(2, int(round(active_frac * p)))
        supp = np.arange(0, m_act, dtype=np.int64)
        w = np.zeros(p, dtype=np.float64)
        w[supp] = rng.uniform(0.5, 1.5, size=m_act)
        w[supp] = w[supp] / float(np.mean(w[supp])) * signal_sd
        eps = _block_correlated_noise(rng, n_tot, p, block_size=5, rho=noise_rho)
        X = np.outer(z1, w) + eps
        mu = X[:n_train].mean(axis=0, keepdims=True)
        sd = X[:n_train].std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xz = (X - mu) / sd
        views_tr.append(Xz[:n_train].copy())
        views_te.append(Xz[n_train:].copy())
        true_supports.append(supp)
        true_weights.append(w / max(float(np.linalg.norm(w)), 1e-12))

    return {
        "views_tr": views_tr,
        "views_te": views_te,
        "z_tr": z1[:n_train],
        "z_te": z1[n_train:],
        "true_supports": true_supports,
        "true_weights": true_weights,
    }


def _make_design_b_data(
    seed: int,
    *,
    n_train: int = 60,
    n_test: int = 200,
    view_dims: tuple[int, ...] = VIEW_DIMS,
    mode_frac: float = 0.25,
    signal_sds: tuple[float, float] = (0.62, 0.50),
    noise_rho: float = 0.45,
) -> dict[str, Any]:
    """Design B: Rank-2 disjoint non-negative modes (z_1 perp z_2; 25% + 25% active, 50% noise)."""
    rng = np.random.default_rng(seed)
    n_tot = n_train + n_test
    Z = rng.standard_normal((n_tot, 2))
    for sl in (slice(0, n_train), slice(n_train, n_tot)):
        Zs = Z[sl] - Z[sl].mean(axis=0, keepdims=True)
        q_mat, _ = np.linalg.qr(Zs)
        q_mat = q_mat * np.sqrt(Zs.shape[0])
        Z[sl] = q_mat

    views_tr: list[np.ndarray] = []
    views_te: list[np.ndarray] = []
    true_supports: list[list[np.ndarray]] = []
    true_weights: list[np.ndarray] = []

    for p in view_dims:
        m1 = max(2, int(round(mode_frac * p)))
        m2 = max(2, int(round(mode_frac * p)))
        s1 = np.arange(0, m1, dtype=np.int64)
        s2 = np.arange(m1, m1 + m2, dtype=np.int64)
        W = np.zeros((p, 2), dtype=np.float64)
        for c, (sc, sig) in enumerate(((s1, signal_sds[0]), (s2, signal_sds[1]))):
            wc = rng.uniform(0.5, 1.5, size=sc.size)
            wc = wc / float(np.mean(wc)) * sig
            W[sc, c] = wc
        eps = _block_correlated_noise(rng, n_tot, p, block_size=5, rho=noise_rho)
        X = Z @ W.T + eps
        mu = X[:n_train].mean(axis=0, keepdims=True)
        sd = X[:n_train].std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xz = (X - mu) / sd
        views_tr.append(Xz[:n_train].copy())
        views_te.append(Xz[n_train:].copy())
        true_supports.append([s1, s2])
        Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
        true_weights.append(Wn)

    return {
        "views_tr": views_tr,
        "views_te": views_te,
        "Z_tr": Z[:n_train],
        "Z_te": Z[n_train:],
        "true_supports": true_supports,
        "true_weights": true_weights,
    }


def _support_f1(pred_support: np.ndarray, true_support: np.ndarray) -> float:
    ps = set(int(x) for x in pred_support.tolist())
    ts = set(int(x) for x in true_support.tolist())
    if not ps or not ts:
        return 0.0
    inter = len(ps & ts)
    if inter == 0:
        return 0.0
    prec = inter / len(ps)
    rec = inter / len(ts)
    return float(2.0 * prec * rec / (prec + rec))


def _r2_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(ok.sum()) < 5 or np.std(y_pred[ok]) < 1e-12 or np.std(y_true[ok]) < 1e-12:
        return 0.0
    r = float(np.corrcoef(y_true[ok], y_pred[ok])[0, 1])
    return float(r * r)


def _predict_legacy_unshrunk(res: Mapping[str, Any], views_te: list[np.ndarray]) -> np.ndarray:
    vs = pysimlr.view_scores(res, views_te)["scores"]
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in vs], axis=1)  # (n, M, k)
    return np.nanmean(stack, axis=1)


def _run_design_a_seed(args: tuple[int, bool]) -> dict[str, dict[str, float]]:
    seed, smoke = args
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _make_design_a_data(seed)
        v_tr, v_te = data["views_tr"], data["views_te"]
        z_te = data["z_te"]
        true_supps = data["true_supports"]
        true_w = data["true_weights"]

        fit_extra: dict[str, Any] = {"iterations": 8, "learning_rate": 0.05} if smoke else {}

        def _eval_k1(u_te_1d: np.ndarray, v_mats: list[np.ndarray]) -> dict[str, float]:
            r2 = _r2_corr(z_te, u_te_1d)
            f1s = []
            cos_list = []
            active_fracs = []
            for m, V in enumerate(v_mats):
                vcol = np.maximum(V[:, 0], 0.0)
                supp = np.flatnonzero(vcol > 1e-8)
                f1s.append(_support_f1(supp, true_supps[m]))
                active_fracs.append(float(supp.size) / float(V.shape[0]))
                nrm = float(np.linalg.norm(vcol))
                cos_list.append(float(np.dot(vcol / nrm, true_w[m])) if nrm > 1e-12 else 0.0)
            return {
                "test_r2": r2,
                "support_f1": float(np.mean(f1s)),
                "weight_cos": float(np.mean(cos_list)),
                "active_frac": float(np.mean(active_fracs)),
            }

        out: dict[str, dict[str, float]] = {}

        # 1. Legacy unshrunk hard 0.5
        r_hard = pysimlr.fit(
            "simlr",
            v_tr,
            k=1,
            scale_list=["none"],
            positivity="conical",
            sparseness=0.5,
            sparseness_type="hard",
            energy_type="recon_r2",
            mixing_algorithm="svd",
            missing_rows="mask",
            random_state=seed,
            **fit_extra,
        )
        u_hard = _predict_legacy_unshrunk(r_hard, v_te)[:, 0]
        v_hard = [v.detach().cpu().double().numpy() for v in r_hard["v"]]
        out["legacy_hard_0p5"] = _eval_k1(u_hard, v_hard)

        # 2. Legacy soft 0.3 + global uniform shrinkage
        kw_soft = dict(
            pysimlr.uniform_shrinkage_params(
                v_tr, k=1, alpha="auto", positivity="conical", sparseness=0.3, sparseness_type="soft",
                energy_type="recon_r2", mixing_algorithm="svd",
            )
        )
        kw_soft.update(fit_extra)
        r_soft = pysimlr.fit("simlr", v_tr, k=1, random_state=seed, **kw_soft)
        u_soft = np.asarray(pysimlr.project_uniform_shrunk(r_soft, v_te, train_views=v_tr, post_shrink="auto")["u"]).reshape(-1)
        v_soft = [v.detach().cpu().double().numpy() for v in r_soft["v"]]
        out["legacy_soft_0p3"] = _eval_k1(u_soft, v_soft)

        # 3. Legacy Euclidean convex 0.3 + global uniform shrinkage
        with _legacy_euclidean_convex_mode():
            kw_ecvx = dict(
                pysimlr.uniform_shrinkage_params(
                    v_tr, k=1, alpha="auto", positivity="convex", sparseness=0.3, sparseness_type="soft",
                    energy_type="recon_r2", mixing_algorithm="svd",
                )
            )
            kw_ecvx.update(fit_extra)
            r_ecvx = pysimlr.fit("simlr", v_tr, k=1, random_state=seed, **kw_ecvx)
        u_ecvx = np.asarray(pysimlr.project_uniform_shrunk(r_ecvx, v_te, train_views=v_tr, post_shrink="auto")["u"]).reshape(-1)
        v_ecvx = [v.detach().cpu().double().numpy() for v in r_ecvx["v"]]
        out["legacy_euclidean_convex_0p3"] = _eval_k1(u_ecvx, v_ecvx)

        # 4. Global shrunk dense conical (sparseness=None)
        kw_gcon = dict(
            pysimlr.uniform_shrinkage_params(
                v_tr, k=1, alpha="auto", positivity="conical", sparseness=None, sparseness_type="soft",
                energy_type="recon_r2", mixing_algorithm="svd",
            )
        )
        kw_gcon.update(fit_extra)
        r_gcon = pysimlr.fit("simlr", v_tr, k=1, random_state=seed, **kw_gcon)
        u_gcon = np.asarray(pysimlr.project_uniform_shrunk(r_gcon, v_te, train_views=v_tr, post_shrink="auto")["u"]).reshape(-1)
        v_gcon = [v.detach().cpu().double().numpy() for v in r_gcon["v"]]
        out["global_shrunk_conical"] = _eval_k1(u_gcon, v_gcon)

        # 5. Global shrunk dense radial convex (sparseness=None; native _convexify is radial)
        kw_grcvx = dict(
            pysimlr.uniform_shrinkage_params(
                v_tr, k=1, alpha="auto", positivity="convex", sparseness=None, sparseness_type="soft",
                energy_type="recon_r2", mixing_algorithm="svd",
            )
        )
        kw_grcvx.update(fit_extra)
        r_grcvx = pysimlr.fit("simlr", v_tr, k=1, random_state=seed, **kw_grcvx)
        u_grcvx = np.asarray(pysimlr.project_uniform_shrunk(r_grcvx, v_te, train_views=v_tr, post_shrink="auto")["u"]).reshape(-1)
        v_grcvx = [v.detach().cpu().double().numpy() for v in r_grcvx["v"]]
        out["global_shrunk_radial_convex"] = _eval_k1(u_grcvx, v_grcvx)

        # 6-13. Two-Stage arms (conical & radial convex, plugin vs ric, q in {0.0, 0.3, 0.5})
        ts_specs = (
            ("twostage_conical_q0p0_ric", "conical", 0.0, "ric"),
            ("twostage_conical_q0p3_ric", "conical", 0.3, "ric"),
            ("twostage_conical_q0p3_plugin", "conical", 0.3, "plugin"),
            ("twostage_conical_q0p5_ric", "conical", 0.5, "ric"),
            ("twostage_conical_q0p5_plugin", "conical", 0.5, "plugin"),
            ("twostage_radial_convex_q0p3_ric", "convex", 0.3, "ric"),
            ("twostage_radial_convex_q0p5_ric", "convex", 0.5, "ric"),
            ("twostage_radial_convex_q0p5_plugin", "convex", 0.5, "plugin"),
        )
        for name, pos, q_val, amode in ts_specs:
            b = pysimlr.fit_uniform_shrunk(
                v_tr,
                k=1,
                sparseness=q_val,
                positivity=pos,
                alpha_mode=amode,
                energy_type="recon_r2",
                mixing_algorithm="svd",
                random_state=seed,
                **fit_extra,
            )
            proj = pysimlr.project_uniform_shrunk(b, v_te, train_views=v_tr, post_shrink="auto")
            v_mats = [v.detach().cpu().double().numpy() for v in b["v"]]
            out[name] = _eval_k1(np.asarray(proj["u"])[:, 0], v_mats)

    return out


def _run_design_b_seed(args: tuple[int, bool]) -> dict[str, dict[str, float]]:
    seed, smoke = args
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _make_design_b_data(seed)
        v_tr, v_te = data["views_tr"], data["views_te"]
        Z_te = data["Z_te"]
        true_supps = data["true_supports"]

        fit_extra: dict[str, Any] = {"iterations": 8, "learning_rate": 0.05} if smoke else {}

        def _eval_k2(U_te: np.ndarray, v_mats: list[np.ndarray]) -> dict[str, float]:
            r00 = _r2_corr(Z_te[:, 0], U_te[:, 0])
            r11 = _r2_corr(Z_te[:, 1], U_te[:, 1])
            r01 = _r2_corr(Z_te[:, 0], U_te[:, 1])
            r10 = _r2_corr(Z_te[:, 1], U_te[:, 0])
            if (r00 + r11) >= (r01 + r10):
                perm = (0, 1)
                r2_z1, r2_z2 = r00, r11
            else:
                perm = (1, 0)
                r2_z1, r2_z2 = r01, r10

            ok_u = np.isfinite(U_te).all(axis=1)
            if int(ok_u.sum()) > 5 and np.std(U_te[ok_u, 0]) > 1e-12 and np.std(U_te[ok_u, 1]) > 1e-12:
                abs_corr_u = float(abs(np.corrcoef(U_te[ok_u, 0], U_te[ok_u, 1])[0, 1]))
            else:
                abs_corr_u = 1.0

            f1s = []
            max_fro = 0.0
            max_cos = 0.0
            for m, V in enumerate(v_mats):
                norms = np.linalg.norm(V, axis=0, keepdims=True)
                Vn = V / np.where(norms > 1e-15, norms, 1.0)
                G = Vn.T @ Vn
                max_fro = max(max_fro, float(np.linalg.norm(G - np.eye(2), ord="fro")))
                max_cos = max(max_cos, float(abs(G[0, 1])))
                for true_c, pred_c in enumerate(perm):
                    supp_p = np.flatnonzero(np.maximum(V[:, pred_c], 0.0) > 1e-8)
                    f1s.append(_support_f1(supp_p, true_supps[m][true_c]))
            return {
                "test_r2_z1": r2_z1,
                "test_r2_z2": r2_z2,
                "test_r2_mean": 0.5 * (r2_z1 + r2_z2),
                "support_f1": float(np.mean(f1s)),
                "abs_corr_u1_u2": abs_corr_u,
                "max_orth_fro_err": max_fro,
                "max_col_cos": max_cos,
            }

        out: dict[str, dict[str, float]] = {}

        # 1. Legacy conical hard 0.5 (unshrunk nsa_flow)
        r_ch = pysimlr.fit(
            "simlr",
            v_tr,
            k=2,
            scale_list=["none"],
            positivity="conical",
            sparseness=0.5,
            sparseness_type="hard",
            nsa_w=0.5,
            initialization_type="nndsvd",
            energy_type="recon_r2",
            mixing_algorithm="svd",
            missing_rows="mask",
            random_state=seed,
            **fit_extra,
        )
        out["legacy_conical_hard_0p5"] = _eval_k2(
            _predict_legacy_unshrunk(r_ch, v_te), [v.detach().cpu().double().numpy() for v in r_ch["v"]]
        )

        # 2. Legacy conical soft 0.3 (unshrunk nsa_flow)
        r_cs = pysimlr.fit(
            "simlr",
            v_tr,
            k=2,
            scale_list=["none"],
            positivity="conical",
            sparseness=0.3,
            sparseness_type="soft",
            nsa_w=0.5,
            initialization_type="nndsvd",
            energy_type="recon_r2",
            mixing_algorithm="svd",
            missing_rows="mask",
            random_state=seed,
            **fit_extra,
        )
        out["legacy_conical_soft_0p3"] = _eval_k2(
            _predict_legacy_unshrunk(r_cs, v_te), [v.detach().cpu().double().numpy() for v in r_cs["v"]]
        )

        # 3. Global uniform smoother applied naively at k=2 (demonstrates mode collapse!)
        glob_ops = [UniformShrinkageOperator(a.shape[1], alpha=0.5, positive_part=True) for a in v_tr]
        r_gs = pysimlr.fit(
            "simlr",
            v_tr,
            k=2,
            scale_list=["none"],
            smoothing_matrices=glob_ops,
            positivity="conical",
            sparseness=None,
            sparseness_type="soft",
            nsa_w=0.5,
            initialization_type="nndsvd",
            energy_type="recon_r2",
            mixing_algorithm="svd",
            missing_rows="mask",
            random_state=seed,
            **fit_extra,
        )
        out["legacy_global_shrunk_k2"] = _eval_k2(
            _predict_legacy_unshrunk(r_gs, v_te), [v.detach().cpu().double().numpy() for v in r_gs["v"]]
        )

        # 4. Legacy Euclidean convex 0.5 (unshrunk nsa_flow)
        with _legacy_euclidean_convex_mode():
            r_ec = pysimlr.fit(
                "simlr",
                v_tr,
                k=2,
                scale_list=["none"],
                positivity="convex",
                sparseness=0.5,
                sparseness_type="hard",
                nsa_w=0.5,
                initialization_type="nndsvd",
                energy_type="recon_r2",
                mixing_algorithm="svd",
                missing_rows="mask",
                random_state=seed,
                **fit_extra,
            )
        out["legacy_euclidean_convex_0p5"] = _eval_k2(
            _predict_legacy_unshrunk(r_ec, v_te), [v.detach().cpu().double().numpy() for v in r_ec["v"]]
        )

        # 5-12. Two-Stage k=2 arms
        ts_specs = (
            ("twostage_conical_q0p0_ric", "conical", 0.0, "ric"),
            ("twostage_conical_q0p3_ric", "conical", 0.3, "ric"),
            ("twostage_conical_q0p3_plugin", "conical", 0.3, "plugin"),
            ("twostage_conical_q0p5_ric", "conical", 0.5, "ric"),
            ("twostage_conical_q0p5_plugin", "conical", 0.5, "plugin"),
            ("twostage_radial_convex_q0p0_ric", "convex", 0.0, "ric"),
            ("twostage_radial_convex_q0p3_ric", "convex", 0.3, "ric"),
            ("twostage_radial_convex_q0p5_ric", "convex", 0.5, "ric"),
        )
        for name, pos, q_val, amode in ts_specs:
            b = pysimlr.fit_uniform_shrunk(
                v_tr,
                k=2,
                sparseness=q_val,
                positivity=pos,
                alpha_mode=amode,
                stage1_nsa_w=0.5,
                stage1_init="nndsvd",
                energy_type="recon_r2",
                mixing_algorithm="svd",
                random_state=seed,
                **fit_extra,
            )
            proj = pysimlr.project_uniform_shrunk(b, v_te, train_views=v_tr, post_shrink="auto")
            v_mats = [v.detach().cpu().double().numpy() for v in b["v"]]
            out[name] = _eval_k2(np.asarray(proj["u"]), v_mats)

    return out


def _aggregate_seed_runs(seed_outputs: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, Any]]:
    arm_names = list(seed_outputs[0].keys())
    metric_names = list(seed_outputs[0][arm_names[0]].keys())
    summary: dict[str, dict[str, Any]] = {}
    for arm in arm_names:
        summary[arm] = {}
        for m in metric_names:
            vals = np.asarray([s[arm][m] for s in seed_outputs], dtype=np.float64)
            summary[arm][m] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast <10s gate (n_seeds=2, iterations=8)")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of Monte Carlo seeds (default: 20)")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel worker processes (default: 8)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (defaults to paper/results_cache/v043_synthetic_theorems.json, or temp path under --smoke)")
    args = parser.parse_args()

    t0 = time.time()
    smoke = bool(args.smoke)
    n_seeds = 2 if smoke else int(args.n_seeds)
    n_jobs = 1 if smoke else max(1, min(int(args.n_jobs), n_seeds))
    seeds = [100 + s for s in range(n_seeds)]
    out_path = args.output or (
        Path(tempfile.gettempdir()) / "pysimlr_smoke_synthetic_theorems.json"
        if smoke else DEFAULT_OUT_JSON
    )

    props = verify_propositions()
    t_props = round(time.time() - t0, 3)

    t1 = time.time()
    if n_jobs == 1:
        a_runs = [_run_design_a_seed((s, smoke)) for s in seeds]
        b_runs = [_run_design_b_seed((s, smoke)) for s in seeds]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            a_runs = list(ex.map(_run_design_a_seed, [(s, smoke) for s in seeds]))
            b_runs = list(ex.map(_run_design_b_seed, [(s, smoke) for s in seeds]))
    t_bench = round(time.time() - t1, 2)

    a_summary = _aggregate_seed_runs(a_runs)
    b_summary = _aggregate_seed_runs(b_runs)

    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cli_rel = " ".join(
        [str(Path(sys.argv[0]).resolve().relative_to(REPO_DIR))] + sys.argv[1:]
        if Path(sys.argv[0]).resolve().is_relative_to(REPO_DIR)
        else [Path(sys.argv[0]).name] + sys.argv[1:]
    )

    payload = {
        "provenance": {
            "cli": cli_rel,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "platform": f"{platform.system()} {platform.machine()}",
            "smoke": smoke,
            "n_seeds": n_seeds,
            "n_jobs": n_jobs,
            "view_dims": list(VIEW_DIMS),
            "n_train": 60,
            "n_test": 200,
            "energy_type": "recon_r2",
            "mixing_algorithm": "svd",
            "pysimlr_version": getattr(pysimlr, "__version__", "unknown"),
            "git_commit": commit,
            "propositions_sec": t_props,
            "benchmark_sec": t_bench,
            "total_sec": round(time.time() - t0, 2),
        },
        "propositions": props,
        "design_a_rank1_sparse": {
            "description": "M=4, n_train=60, n_test=200, p=(80,100,60,160), 30% active positive features, block-correlated noise (rho=0.45)",
            "summary": a_summary,
            "per_seed": a_runs,
        },
        "design_b_rank2_disjoint": {
            "description": "M=4, n_train=60, n_test=200, p=(80,100,60,160), two disjoint non-negative modes (25% + 25% active, 50% noise, z1 perp z2)",
            "summary": b_summary,
            "per_seed": b_runs,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "smoke": smoke,
                "n_seeds": n_seeds,
                "total_sec": payload["provenance"]["total_sec"],
                "prop1_max_diff": props["prop1"]["max_linf_diff_quantile_soft_vs_alpha0"],
                "prop2_cs_err": props["prop2"]["cauchy_schwarz_abs_error"],
                "prop2_euclid_zeroed": props["prop2"]["legacy_convexify_fraction_zeroed_at_sparseness_none"],
                "prop3_max_fro": props["prop3"]["max_col_support_fro_error"],
                "output": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
