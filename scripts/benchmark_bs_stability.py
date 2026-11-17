#!/usr/bin/env python3
"""Section 5 Benchmark Suite (v4): Calibrated Latent Stability, Optimizer, Initializer & Sample-Size Audit.

Implements the four synthetic benchmark problems and six evaluation criteria
specified in Section 5 of `/tmp/bs.md` (v4):

Problems:
  P1. Pure Gaussian noise (N=93, widths=[66, 98, 36, 100, 96])
  P2. Planted shared factor with strong private variance and block-correlated
      features across n in {60, 93, 186, 400} and decreasing shared strength
      alpha in {1.0, 0.65}
  P3. Two shared factors of nearly equal strength (k=1 competing basins vs
      k=2 multivariate trace ICC(1) & 2-D Procrustes subspace recovery)
  P4. Multi-view data with one all-missing view for ~15% of rows
      (missing_rows="mask")

Grid (no benchmark row exists for one constraint only):
  - Constraints: "linear", "conical", "convex", "mixed" (per_view_params)
  - Sparseness:  {0.5, 0.8}
  - Ranks:       k in {1, 2}

Candidates (10):
  baseline_0.40.0, B1_bc, B1_trust, B1_norm, B2_ref_lr, B3_feasible_aligned,
  B_combined, C1_graduated, C2_spectral_small, C3_soft_sparseness.

All stability measurements use `pysimlr.latent_stability(...)` exclusively
(no removed v2 keys anywhere). Supports `--smoke` (<15s) and full mode writing
`reports/bs_stability_benchmark.json` and `reports/bs_stability_benchmark_report.html`.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Ensure repo src/ is importable when invoked directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pysimlr  # noqa: E402
from pysimlr import latent_stability, simlr, solution_uniqueness  # noqa: E402
from pysimlr.multistart import SAME_SOLUTION_CORR  # noqa: E402

WIDTHS = [66, 98, 36, 100, 96]
CONSTRAINTS = ["linear", "conical", "convex", "mixed"]
MIXED_CYCLE = ["convex", "conical", "linear"]

CANDIDATES: List[Tuple[str, Dict[str, Any]]] = [
    (
        "baseline_0.40.0",
        {
            "optimizer_type": "hybrid_adam_legacy",
            "initialization_type": "pca_legacy",
            "tune_learning_rate": "per_refit",
            "phase": "Baseline",
            "addresses": "Reference (0.40.0)",
        },
    ),
    (
        "B1_bc",
        {
            "optimizer_type": "hybrid_adam_bc",
            "initialization_type": "pca_legacy",
            "tune_learning_rate": "per_refit",
            "phase": "Part B1",
            "addresses": "Moment bias correction",
        },
    ),
    (
        "B1_trust",
        {
            "optimizer_type": "hybrid_adam_trust",
            "initialization_type": "pca_legacy",
            "tune_learning_rate": "per_refit",
            "phase": "Part B1",
            "addresses": "Bias correction + step-0 trust cap",
        },
    ),
    (
        "B1_norm",
        {
            "optimizer_type": "hybrid_adam_norm",
            "initialization_type": "pca_legacy",
            "tune_learning_rate": "per_refit",
            "phase": "Part B1",
            "addresses": "Bias correction + weight-norm step",
        },
    ),
    (
        "B2_ref_lr",
        {
            "optimizer_type": "hybrid_adam_legacy",
            "initialization_type": "pca_legacy",
            "tune_learning_rate": "reference",
            "phase": "Part B2",
            "addresses": "Shared reference learning rate",
        },
    ),
    (
        "B3_feasible_aligned",
        {
            "optimizer_type": "hybrid_adam_legacy",
            "initialization_type": "pca_feasible",
            "align_initial_views": True,
            "tune_learning_rate": "per_refit",
            "phase": "Part B3",
            "addresses": "Feasible & aligned initial starts",
        },
    ),
    (
        "B_combined",
        {
            "optimizer_type": "hybrid_adam_trust",
            "initialization_type": "pca_feasible",
            "align_initial_views": True,
            "tune_learning_rate": "reference",
            "phase": "Part B (B1+B2+B3)",
            "addresses": "Trust + feasible/aligned + ref lr",
        },
    ),
    (
        "C1_graduated",
        {
            "optimizer_type": "hybrid_adam_trust",
            "initialization_type": "pca_feasible",
            "align_initial_views": True,
            "sparseness_schedule": 3,
            "tune_learning_rate": "reference",
            "phase": "Part C1",
            "addresses": "Graduated sparseness (3 stages)",
        },
    ),
    (
        "C2_spectral_small",
        {
            "optimizer_type": "hybrid_adam_trust",
            "initialization_type": "spectral",
            "align_initial_views": True,
            "tune_learning_rate": "reference",
            "phase": "Part C2",
            "addresses": "Spectral start + trust + ref lr",
        },
    ),
    (
        "C3_soft_sparseness",
        {
            "optimizer_type": "hybrid_adam_trust",
            "initialization_type": "pca_feasible",
            "align_initial_views": True,
            "sparseness_type": "soft",
            "tune_learning_rate": "reference",
            "phase": "Part C3",
            "addresses": "Continuous soft-threshold sparseness",
        },
    ),
]

CANDIDATE_MAP: Dict[str, Dict[str, Any]] = {name: cfg for name, cfg in CANDIDATES}
CANDIDATE_NAMES: List[str] = [name for name, _ in CANDIDATES]


# ---------------------------------------------------------------------------
# Constraint & Parameter Builders
# ---------------------------------------------------------------------------

def constraint_kwargs(constraint: str, n_views: int) -> Dict[str, Any]:
    """Return fit kwargs for 'linear', 'conical', 'convex', or 'mixed'."""
    if constraint in ("linear", "conical", "convex"):
        return {"positivity": constraint}
    if constraint == "mixed":
        pvp = {
            m: {"positivity": MIXED_CYCLE[m % len(MIXED_CYCLE)]}
            for m in range(n_views)
        }
        return {"positivity": "linear", "per_view_params": pvp}
    raise ValueError(f"Unknown constraint {constraint!r}")


def build_simlr_kwargs(
    cand_name: str,
    constraint: str,
    n_views: int,
    sparseness: float,
    iterations: int,
    **extra: Any,
) -> Dict[str, Any]:
    """Build consistent kwargs for pysimlr.fit('simlr', ...) or simlr(...)."""
    cfg = CANDIDATE_MAP[cand_name]
    kw: Dict[str, Any] = {
        "sparseness": sparseness,
        "iterations": iterations,
        "optimizer_type": cfg["optimizer_type"],
        "initialization_type": cfg["initialization_type"],
    }
    kw.update(constraint_kwargs(constraint, n_views))
    for opt_key in ("align_initial_views", "sparseness_schedule", "sparseness_type"):
        if opt_key in cfg:
            kw[opt_key] = cfg[opt_key]
    kw.update(extra)
    return kw


# ---------------------------------------------------------------------------
# Synthetic Data Generators (fixed seeds, zero external data)
# ---------------------------------------------------------------------------

def make_problem1_pure_noise(
    seed: int = 42,
    n: int = 93,
    widths: Sequence[int] = WIDTHS,
) -> List[np.ndarray]:
    """P1: Pure i.i.d. Gaussian noise with zero shared structure."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, p)) for p in widths]


def make_problem2_planted_shared(
    seed: int = 101,
    n: int = 93,
    widths: Sequence[int] = WIDTHS,
    alpha: float = 1.0,
    base_shared_amp: float = 0.25,
    priv_amps: Sequence[float] = (2.2, 1.6, 1.2),
    noise_sd: float = 0.5,
    block_corr: float = 0.75,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """P2: Planted rank-1 shared factor with strong view-private variance.

    Active features live in contiguous blocks with intra-block correlated noise
    (rho = block_corr) so C3 soft sparseness can be evaluated fairly alongside
    hard quantile thresholding under all constraints (linear, conical, convex,
    mixed). Loadings are non-negative so all constraint regimes share the same
    ground-truth latent and active support.
    """
    rng = np.random.default_rng(seed)
    z_true = rng.standard_normal((n, 1))
    shared_amp = float(alpha) * float(base_shared_amp)
    views: List[np.ndarray] = []
    w_true: List[np.ndarray] = []
    priv_arr = np.asarray(priv_amps, dtype=float)
    for p in widths:
        z_priv = rng.standard_normal((n, len(priv_arr))) * priv_arr
        w_priv = rng.standard_normal((p, len(priv_arr))) / np.sqrt(p)
        n_act = max(4, round(0.2 * p))
        w_sh = np.zeros((p, 1))
        w_sh[:n_act, 0] = shared_amp * (1.0 + 0.05 * rng.standard_normal(n_act))
        eps = rng.standard_normal((n, p))
        if block_corr > 0.0:
            s_other = np.sqrt(max(1.0 - block_corr ** 2, 0.0))
            for b0 in range(0, p - 3, 4):
                blk_z = rng.standard_normal((n, 1))
                eps[:, b0 : b0 + 4] = block_corr * blk_z + s_other * eps[:, b0 : b0 + 4]
        views.append(z_true @ w_sh.T + z_priv @ w_priv.T + noise_sd * eps)
        w_true.append(w_sh[:, 0])
    return z_true, w_true, views


def make_problem3_two_factors(
    seed: int = 202,
    n: int = 93,
    widths: Sequence[int] = WIDTHS,
    amp1: float = 0.45,
    amp2: float = 0.43,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """P3: Two orthogonal shared factors of nearly equal strength.

    At k=1, starts compete between z1 and z2 basins (multi-start distinctness).
    At k=2, both factors together form a well-determined 2-D latent subspace
    measured by trace ICC(1) (`latent_reliability_multivariate_gain`) and 2-D
    canonical subspace correlation.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n, 1))
    z2 = rng.standard_normal((n, 1))
    z2 = z2 - (z2.T @ z1) / (z1.T @ z1) * z1
    z2 = z2 / (np.std(z2) + 1e-12)
    z_true = np.hstack([z1, z2])
    views: List[np.ndarray] = []
    for p in widths:
        z_priv = rng.standard_normal((n, 2)) * np.array([1.8, 1.2])
        w_priv = rng.standard_normal((p, 2)) / np.sqrt(p)
        perm = rng.permutation(p)
        s_sz = max(3, round(0.18 * p))
        s1 = perm[:s_sz]
        s2 = perm[s_sz : 2 * s_sz]
        w1, w2 = np.zeros((p, 1)), np.zeros((p, 1))
        w1[s1, 0] = amp1
        w2[s2, 0] = amp2
        views.append(
            z1 @ w1.T + z2 @ w2.T + z_priv @ w_priv.T + 0.5 * rng.standard_normal((n, p))
        )
    return z_true, views


def make_problem4_missing_mask(
    seed: int = 303,
    n: int = 93,
    widths: Sequence[int] = WIDTHS,
    missing_frac: float = 0.15,
    missing_view: int = 2,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """P4: Planted rank-1 with ~15% all-missing (NaN) rows in one view."""
    z_true, w_true, views = make_problem2_planted_shared(
        seed=seed, n=n, widths=widths, alpha=alpha, block_corr=0.0
    )
    rng = np.random.default_rng(seed + 1000)
    n_miss = max(1, int(round(missing_frac * n)))
    miss_rows = rng.choice(n, size=n_miss, replace=False)
    mv = min(missing_view, len(views) - 1)
    views[mv] = views[mv].copy()
    views[mv][miss_rows, :] = np.nan
    return z_true, w_true, views


# ---------------------------------------------------------------------------
# Helpers & Metrics
# ---------------------------------------------------------------------------

def _to_np(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        return t.detach().cpu().double().numpy()
    return np.asarray(t, dtype=np.float64)


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    aa = a[ok] - a[ok].mean()
    bb = b[ok] - b[ok].mean()
    den = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    return float(abs((aa * bb).sum()) / den) if den > 1e-15 else float("nan")


def _subspace_recovery(u_hat: np.ndarray, z_true: np.ndarray) -> float:
    """Mean canonical correlation (in [0, 1]) between u_hat (n, k) and z_true (n, d).

    Reduces identically to |r(u[:, 0], z_true[:, 0])| when k == d == 1.
    """
    u_arr = np.asarray(u_hat, dtype=float)
    z_arr = np.asarray(z_true, dtype=float)
    if u_arr.ndim == 1:
        u_arr = u_arr[:, None]
    if z_arr.ndim == 1:
        z_arr = z_arr[:, None]
    ok = np.isfinite(u_arr).all(axis=1) & np.isfinite(z_arr).all(axis=1)
    if int(ok.sum()) < max(3, u_arr.shape[1] + 1):
        return float("nan")
    uc = u_arr[ok] - u_arr[ok].mean(axis=0, keepdims=True)
    zc = z_arr[ok] - z_arr[ok].mean(axis=0, keepdims=True)
    # Drop zero-variance columns cleanly
    u_keep = np.linalg.norm(uc, axis=0) > 1e-12
    z_keep = np.linalg.norm(zc, axis=0) > 1e-12
    if not u_keep.any() or not z_keep.any():
        return 0.0
    qu, _ = np.linalg.qr(uc[:, u_keep], mode="reduced")
    qz, _ = np.linalg.qr(zc[:, z_keep], mode="reduced")
    svals = np.linalg.svd(qu.T @ qz, compute_uv=False)
    svals = np.clip(svals, 0.0, 1.0)
    return float(np.mean(svals))


def _support_metrics(v_list: Sequence[Any], w_true: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Compute pooled support F1 and pooled absolute weight cosine against w_true."""
    v_cat = np.concatenate([np.linalg.norm(_to_np(v), axis=1) for v in v_list])
    w_cat = np.concatenate([np.abs(np.asarray(w, dtype=np.float64).ravel()) for w in w_true])
    pred_active = v_cat > 1e-6
    true_active = w_cat > 1e-12
    tp = int(np.sum(pred_active & true_active))
    fp = int(np.sum(pred_active & ~true_active))
    fn = int(np.sum(~pred_active & true_active))
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom > 0 else 0.0
    vn = float(np.linalg.norm(v_cat))
    wn = float(np.linalg.norm(w_cat))
    cos = float(np.dot(v_cat, w_cat) / (vn * wn)) if (vn > 1e-15 and wn > 1e-15) else 0.0
    return f1, cos


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    v = float(x)
    return v if np.isfinite(v) else None


def _arr_stat(x: Any, reduce: str = "mean") -> Optional[float]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float).ravel()
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return None
    if reduce == "first":
        return float(arr[0]) if np.isfinite(arr[0]) else None
    if reduce == "min":
        return float(np.min(fin))
    return float(np.mean(fin))


def _extract_stability_fields(stab: Dict[str, Any]) -> Dict[str, Any]:
    """Extract calibrated Part A stability fields from pysimlr.latent_stability()."""
    return {
        "latent_reliability": _arr_stat(stab.get("latent_reliability"), "mean"),
        "latent_reliability_min": _arr_stat(stab.get("latent_reliability"), "min"),
        "latent_reliability_baseline": _arr_stat(stab.get("latent_reliability_baseline"), "mean"),
        "latent_reliability_gain": _arr_stat(stab.get("latent_reliability_gain"), "mean"),
        "latent_reliability_gain_min": _arr_stat(stab.get("latent_reliability_gain"), "min"),
        "latent_reliability_averaged": _arr_stat(stab.get("latent_reliability_averaged"), "mean"),
        "latent_reliability_averaged_baseline": _arr_stat(stab.get("latent_reliability_averaged_baseline"), "mean"),
        "latent_reliability_averaged_gain": _arr_stat(stab.get("latent_reliability_averaged_gain"), "mean"),
        "latent_reliability_multivariate": _safe_float(stab.get("latent_reliability_multivariate")),
        "latent_reliability_multivariate_baseline": _safe_float(stab.get("latent_reliability_multivariate_baseline")),
        "latent_reliability_multivariate_gain": _safe_float(stab.get("latent_reliability_multivariate_gain")),
        "cross_view_coupling": _arr_stat(stab.get("cross_view_coupling"), "mean"),
        "cross_view_coupling_baseline": _arr_stat(stab.get("cross_view_coupling_baseline"), "mean"),
        "cross_view_coupling_gain": _arr_stat(stab.get("cross_view_coupling_gain"), "mean"),
        "feature_agreement": _arr_stat(stab.get("feature_agreement"), "mean"),
        "feature_agreement_baseline": _arr_stat(stab.get("feature_agreement_baseline"), "mean"),
        "feature_agreement_gain": _arr_stat(stab.get("feature_agreement_gain"), "mean"),
        "converged_fraction": _safe_float(stab.get("converged_fraction")),
        "n_refits_used": int(stab.get("n_refits_used") or 0),
        "n_unique_refit_lrs": len(
            {round(float(x), 8) for x in (stab.get("refit_learning_rates") or []) if x is not None}
        ),
    }


def _group_multistart_solutions(
    fits: Sequence[Dict[str, Any]],
) -> Tuple[int, Optional[float], float, int, float]:
    """Group multi-start fits via SAME_SOLUTION_CORR=0.95 connected components."""
    energies = [float(f["best_energy"]) for f in fits]
    finite = [i for i, e in enumerate(energies) if np.isfinite(e)]
    if not finite:
        return 0, None, float("nan"), 0, float("nan")
    chosen_i = min(finite, key=lambda i: (energies[i], i))
    scores = [_to_np(f["u"])[:, 0] for f in fits]
    parent = {i: i for i in finite}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in finite:
        for b in finite:
            if a < b and _abs_corr(scores[a], scores[b]) >= SAME_SOLUTION_CORR:
                parent[find(b)] = find(a)
    groups: Dict[int, List[int]] = {}
    for i in finite:
        groups.setdefault(find(i), []).append(i)
    best_by_group = sorted(min(energies[i] for i in g) for g in groups.values())
    gap = float(best_by_group[0] - best_by_group[1]) if len(best_by_group) > 1 else None
    dom_frac = float(max(len(g) for g in groups.values()) / len(finite))
    return len(groups), gap, energies[chosen_i], chosen_i, dom_frac


def _run_candidate_multistart(
    views: List[np.ndarray],
    cand_name: str,
    constraint: str,
    sparseness: float,
    k: int,
    iterations: int,
    n_starts: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run n_starts multi-start: start 0 uses candidate's init, rest are random.

    Per B2, random starts ALWAYS tune their own learning rate (never reusing a
    structured pca/spectral reference learning rate across random starts).
    """
    n_views = len(views)
    kw0 = build_simlr_kwargs(cand_name, constraint, n_views, sparseness, iterations, random_state=seed)
    t0 = time.perf_counter()
    first = simlr(views, k, **kw0)
    fits = [first]
    ss = np.random.SeedSequence(int(seed) + 77)
    seeds = [int(s) for s in ss.generate_state(max(0, n_starts - 1), dtype=np.uint32)]
    for s in seeds:
        kw = build_simlr_kwargs(cand_name, constraint, n_views, sparseness, iterations, random_state=s)
        kw["initialization_type"] = "random"
        fits.append(simlr(views, k, **kw))
    wall = time.perf_counter() - t0
    n_dist, gap, best_e, chosen_i, dom_frac = _group_multistart_solutions(fits)
    energies = [float(f["best_energy"]) for f in fits if np.isfinite(f["best_energy"])]
    return {
        "n_distinct_solutions": int(n_dist),
        "energy_gap": gap,
        "best_energy": best_e,
        "mean_start_energy": float(np.mean(energies)) if energies else float("nan"),
        "std_start_energy": float(np.std(energies)) if energies else float("nan"),
        "dominant_basin_fraction": dom_frac,
        "chosen_u": _to_np(fits[chosen_i]["u"]),
        "chosen_v": fits[chosen_i]["v"],
        "chosen_n_steps": int(fits[chosen_i]["convergence"]["n_steps"]),
        "default_energy": float(first["best_energy"]),
        "default_u": _to_np(first["u"]),
        "default_n_steps": int(first["convergence"]["n_steps"]),
        "wall_time": float(wall),
    }


def _measure_first_step_rel(
    views: List[np.ndarray],
    cand_name: str,
    constraint: str,
    sparseness: float,
    learning_rate: Any,
    seed: int = 42,
    force_feasible_start: bool = False,
) -> Dict[str, Optional[float]]:
    """Measure both pre- and post-projection first-step relative displacement on sweep 0.

    Post-projection displacement is ||P(v1) - P(v0)|| / ||P(v0)||, including
    simplex projection under 'convex' and per-view projections under 'mixed'.
    """
    simlr_mod = sys.modules["pysimlr.simlr"]
    orig_create = simlr_mod.create_optimizer
    orig_sp = simlr_mod.simlr_sparseness
    step0_io: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    proj_map: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    pos_list = (
        [MIXED_CYCLE[m % len(MIXED_CYCLE)] for m in range(len(views))]
        if constraint == "mixed"
        else [constraint] * len(views)
    )

    def wrapped_sp(v: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = orig_sp(v, *args, **kwargs)
        mi = kwargs.get("modality_index")
        if mi is not None:
            proj_map.append((int(mi), v.detach().clone(), out.detach().clone()))
        return out

    def wrapped_create(ot: str, vmats: List[torch.Tensor], **kw: Any):
        opt = orig_create(ot, vmats, **kw)
        orig_step = opt.step

        def spy_step(i: int, vc: torch.Tensor, dg: torch.Tensor, fe: Any = None) -> torch.Tensor:
            vn = orig_step(i, vc, dg, fe)
            if i not in step0_io:
                step0_io[i] = (vc.detach().clone(), vn.detach().clone())
            return vn

        opt.step = spy_step  # type: ignore[method-assign]
        return opt

    kw = build_simlr_kwargs(
        cand_name,
        constraint,
        len(views),
        sparseness,
        iterations=1,
        random_state=seed,
        learning_rate=learning_rate,
    )
    if force_feasible_start:
        kw["initialization_type"] = "pca_feasible"
    # Measure step 0 directly at target sparseness (without multi-stage schedule)
    kw.pop("sparseness_schedule", None)
    simlr_mod.create_optimizer = wrapped_create  # type: ignore[assignment]
    simlr_mod.simlr_sparseness = wrapped_sp  # type: ignore[assignment]
    try:
        res = simlr(views, 1, **kw)
    finally:
        simlr_mod.create_optimizer = orig_create  # type: ignore[assignment]
        simlr_mod.simlr_sparseness = orig_sp  # type: ignore[assignment]

    pre_rels: List[float] = []
    post_rels: List[float] = []
    for i in range(len(views)):
        if i not in step0_io:
            continue
        vc, vn = step0_io[i]
        denom_pre = float(vc.norm().clamp_min(1e-8).item())
        pre_rels.append(float((vn - vc).norm().item()) / denom_pre)

        p_vn = None
        for mi, pin, pout in reversed(proj_map):
            if mi == i and torch.equal(pin, vn):
                p_vn = pout
                break
        if p_vn is None:
            p_vn = orig_sp(
                vn,
                constraint_type="ortho",
                positivity=pos_list[i],
                sparseness=sparseness,
                sparseness_type=CANDIDATE_MAP[cand_name].get("sparseness_type", "hard"),
                modality_index=i,
            )
        if force_feasible_start or CANDIDATE_MAP[cand_name]["initialization_type"] in ("pca_feasible", "spectral"):
            p_v0 = vc
        else:
            p_v0 = orig_sp(
                vc,
                constraint_type="none",
                positivity=pos_list[i],
                sparseness=sparseness,
                sparseness_type="hard",
                modality_index=i,
                unit_columns=(pos_list[i] != "convex"),
            )
        denom_post = float(p_v0.norm().clamp_min(1e-8).item())
        post_rels.append(float((p_vn - p_v0).norm().item()) / denom_post)

    lr_used = float(res.get("learning_rate_used") or res.get("learning_rate") or 0.0)
    return {
        "pre_max_rel": _safe_float(np.max(pre_rels)) if pre_rels else None,
        "pre_med_rel": _safe_float(np.median(pre_rels)) if pre_rels else None,
        "post_max_rel": _safe_float(np.max(post_rels)) if post_rels else None,
        "post_med_rel": _safe_float(np.median(post_rels)) if post_rels else None,
        "lr_used": _safe_float(lr_used),
    }


# ---------------------------------------------------------------------------
# Worker Tasks (top-level for spawn multiprocessing)
# ---------------------------------------------------------------------------

def _worker_init() -> None:
    torch.set_num_threads(1)


def _task_first_step(args: Tuple[str, str, float, Sequence[int]]) -> Dict[str, Any]:
    cand_name, constraint, sp, widths = args
    _, _, views = make_problem2_planted_shared(seed=101, n=93, widths=widths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_lr1 = _measure_first_step_rel(views, cand_name, constraint, sp, learning_rate=1.0, seed=42)
        s_auto = _measure_first_step_rel(views, cand_name, constraint, sp, learning_rate="auto", seed=42)
        s_feas_lr1 = _measure_first_step_rel(
            views, cand_name, constraint, sp, learning_rate=1.0, seed=42, force_feasible_start=True
        )
    return {
        "candidate": cand_name,
        "constraint": constraint,
        "sparseness": float(sp),
        "pre_max_rel_lr1": s_lr1["pre_max_rel"],
        "post_max_rel_lr1": s_lr1["post_max_rel"],
        "post_max_rel_feas_lr1": s_feas_lr1["post_max_rel"],
        "post_med_rel_auto": s_auto["post_med_rel"],
        "pre_med_rel_auto": s_auto["pre_med_rel"],
        "lr_used_auto": s_auto["lr_used"],
    }


def _task_same_v0_energy(args: Tuple[str, float, int, int, int, Sequence[int]]) -> List[Dict[str, Any]]:
    constraint, sp, k, seed, iterations, widths = args
    _, _, views = make_problem2_planted_shared(seed=1000 + seed, n=93, widths=widths)
    n_views = len(views)
    ckw = constraint_kwargs(constraint, n_views)
    rows: List[Dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_init = simlr(
            views,
            k,
            sparseness=sp,
            iterations=0,
            random_state=seed,
            initialization_type="pca_feasible",
            **ckw,
        )
        v0 = [v.clone() for v in r_init["v"]]
        r_base = simlr(
            views,
            k,
            sparseness=sp,
            iterations=iterations,
            random_state=seed,
            initial_v=[v.clone() for v in v0],
            optimizer_type="hybrid_adam_legacy",
            **ckw,
        )
        e_base = float(r_base["best_energy"])
        lr_base = r_base.get("learning_rate_used")
        same_v0_cands = ["B1_bc", "B1_trust", "B1_norm", "B_combined", "C1_graduated", "C3_soft_sparseness"]
        for cname in same_v0_cands:
            cfg = CANDIDATE_MAP[cname]
            kw: Dict[str, Any] = {
                "sparseness": sp,
                "iterations": iterations,
                "random_state": seed,
                "initial_v": [v.clone() for v in v0],
                "optimizer_type": cfg["optimizer_type"],
                **ckw,
            }
            if cfg.get("tune_learning_rate") == "reference" and lr_base is not None:
                kw["learning_rate"] = float(lr_base)
            if "sparseness_schedule" in cfg:
                kw["sparseness_schedule"] = cfg["sparseness_schedule"]
            if "sparseness_type" in cfg:
                kw["sparseness_type"] = cfg["sparseness_type"]
            r_c = simlr(views, k, **kw)
            e_c = float(r_c["best_energy"])
            rows.append(
                {
                    "candidate": cname,
                    "constraint": constraint,
                    "sparseness": float(sp),
                    "k": int(k),
                    "seed": int(seed),
                    "baseline_energy": e_base,
                    "candidate_energy": e_c,
                    "delta_energy": e_c - e_base,
                    "non_degraded": bool(e_c <= e_base + 1e-6),
                }
            )
    return rows


def _run_stability_bundle(
    problem_label: str,
    views: List[np.ndarray],
    cand_name: str,
    constraint: str,
    sp: float,
    k: int,
    iterations: int,
    n_subsamples: int,
    n_baseline: int,
    z_true: Optional[np.ndarray] = None,
    w_true: Optional[List[np.ndarray]] = None,
    run_stability: bool = True,
    extra_fit_kw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run reference fit + optional pysimlr.latent_stability() on a dataset."""
    cfg = CANDIDATE_MAP[cand_name]
    ex = dict(extra_fit_kw or {})
    kw = build_simlr_kwargs(cand_name, constraint, len(views), sp, iterations, **ex)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = pysimlr.fit("simlr", views, k, random_state=42, **kw)
        stab_fields: Dict[str, Any] = {}
        if run_stability:
            stab = latent_stability(
                "simlr",
                views,
                k,
                n_subsamples=n_subsamples,
                fraction=0.8,
                baseline=True,
                n_baseline=n_baseline,
                random_state=42,
                n_jobs=1,
                tune_learning_rate=cfg["tune_learning_rate"],
                **kw,
            )
            stab_fields = _extract_stability_fields(stab)
    wall = time.perf_counter() - t0
    u_np = _to_np(ref["u"])
    row: Dict[str, Any] = {
        "problem": problem_label,
        "candidate": cand_name,
        "constraint": constraint,
        "sparseness": float(sp),
        "k": int(k),
        "has_stability": bool(run_stability),
        "best_energy": _safe_float(ref["best_energy"]),
        "n_steps": int(ref["convergence"]["n_steps"]),
        "lr_used": _safe_float(ref.get("learning_rate_used")),
        "wall_time": float(wall),
    }
    if z_true is not None:
        row["r_z_true"] = _safe_float(_subspace_recovery(u_np, z_true))
        if z_true.shape[1] >= 2 and k == 1:
            r1 = _abs_corr(u_np[:, 0], z_true[:, 0])
            r2 = _abs_corr(u_np[:, 0], z_true[:, 1])
            row["r_z1"] = _safe_float(r1)
            row["r_z2"] = _safe_float(r2)
            row["r_max_basin"] = _safe_float(max(r1, r2))
    if w_true is not None:
        f1, cos = _support_metrics(ref["v"], w_true)
        row["support_f1"] = _safe_float(f1)
        row["support_cosine"] = _safe_float(cos)
    row.update(stab_fields)
    return row


def _task_p1(args: Tuple[str, str, float, int, int, int, int, Sequence[int], bool]) -> Dict[str, Any]:
    cand_name, constraint, sp, k, iterations, n_subsamples, n_baseline, widths, run_stab = args
    views = make_problem1_pure_noise(seed=42, n=93, widths=widths)
    return _run_stability_bundle(
        "P1_pure_noise", views, cand_name, constraint, sp, k, iterations,
        n_subsamples, n_baseline, run_stability=run_stab,
    )


def _task_p2(args: Tuple[str, str, float, int, int, float, int, int, int, Sequence[int], bool]) -> Dict[str, Any]:
    cand_name, constraint, sp, k, n_rows, alpha, iterations, n_subsamples, n_baseline, widths, run_stab = args
    z_true, w_true, views = make_problem2_planted_shared(
        seed=101, n=n_rows, widths=widths, alpha=alpha
    )
    row = _run_stability_bundle(
        "P2_planted_shared", views, cand_name, constraint, sp, k, iterations,
        n_subsamples, n_baseline, z_true=z_true, w_true=w_true, run_stability=run_stab,
    )
    row["n"] = int(n_rows)
    row["alpha"] = float(alpha)
    return row


def _task_p3_stability(args: Tuple[str, str, float, int, int, int, int, Sequence[int]]) -> Dict[str, Any]:
    cand_name, constraint, sp, k, iterations, n_subsamples, n_baseline, widths = args
    z_true, views = make_problem3_two_factors(seed=202, n=93, widths=widths)
    return _run_stability_bundle(
        "P3_two_factors", views, cand_name, constraint, sp, k, iterations,
        n_subsamples, n_baseline, z_true=z_true, run_stability=True,
    )


def _task_p3_multistart(args: Tuple[str, str, float, int, int, int, Sequence[int]]) -> Dict[str, Any]:
    cand_name, constraint, sp, k, iterations, n_starts, widths = args
    z_true, views = make_problem3_two_factors(seed=202, n=93, widths=widths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = _run_candidate_multistart(
            views, cand_name, constraint, sp, k, iterations, n_starts=n_starts, seed=0
        )
    u_ch = ms["chosen_u"]
    row: Dict[str, Any] = {
        "problem": "P3_two_factors",
        "candidate": cand_name,
        "constraint": constraint,
        "sparseness": float(sp),
        "k": int(k),
        "n_starts": int(n_starts),
        "best_energy": _safe_float(ms["best_energy"]),
        "default_energy": _safe_float(ms["default_energy"]),
        "mean_start_energy": _safe_float(ms["mean_start_energy"]),
        "std_start_energy": _safe_float(ms["std_start_energy"]),
        "n_distinct_solutions": int(ms["n_distinct_solutions"]),
        "energy_gap": _safe_float(ms["energy_gap"]),
        "dominant_basin_fraction": _safe_float(ms["dominant_basin_fraction"]),
        "r_z_true": _safe_float(_subspace_recovery(u_ch, z_true)),
        "default_r_z_true": _safe_float(_subspace_recovery(ms["default_u"], z_true)),
        "n_steps": int(ms["default_n_steps"]),
        "wall_time": float(ms["wall_time"]),
    }
    if k == 1:
        r1 = _abs_corr(u_ch[:, 0], z_true[:, 0])
        r2 = _abs_corr(u_ch[:, 0], z_true[:, 1])
        row["r_z1"] = _safe_float(r1)
        row["r_z2"] = _safe_float(r2)
        row["r_max_basin"] = _safe_float(max(r1, r2))
    return row


def _task_p4(args: Tuple[str, str, float, int, int, int, int, Sequence[int], bool]) -> Dict[str, Any]:
    cand_name, constraint, sp, k, iterations, n_subsamples, n_baseline, widths, run_stab = args
    z_true, w_true, views = make_problem4_missing_mask(seed=303, n=93, widths=widths)
    return _run_stability_bundle(
        "P4_missing_mask", views, cand_name, constraint, sp, k, iterations,
        n_subsamples, n_baseline, z_true=z_true, w_true=w_true, run_stability=run_stab,
        extra_fit_kw={"missing_rows": "mask"},
    )


# ---------------------------------------------------------------------------
# Aggregation & Summaries
# ---------------------------------------------------------------------------

def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    arr = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(arr)) if arr else None


def _median(vals: Sequence[Optional[float]]) -> Optional[float]:
    arr = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(arr)) if arr else None


def _max_v(vals: Sequence[Optional[float]]) -> Optional[float]:
    arr = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.max(arr)) if arr else None


def _classify_regime(
    lr_val: Optional[float],
    lr_gain: Optional[float],
    coup_gain: Optional[float],
) -> str:
    """Section 8 regime label for the motivating N=93 setup."""
    if lr_gain is None or coup_gain is None:
        return "insufficient data"
    if lr_gain >= 0.15 and coup_gain >= 0.15:
        return "latent reliable and shared"
    if lr_val is not None and lr_val >= 0.30 and lr_gain < 0.10 and coup_gain < 0.10:
        return "reliable but not shared"
    return "neither (below calibration gain)"


def _compute_summaries(data: Dict[str, Any]) -> Dict[str, Any]:
    first_steps = data["first_step"]
    same_v0 = data["same_v0_energy"]
    p1 = data["p1_pure_noise"]
    p2 = data["p2_planted_shared"]
    p3_stab = data["p3_stability"]
    p3_ms = data["p3_multistart"]
    p4 = data["p4_missing_mask"]

    # Build paired baseline energy map across all single-fit runs (hard-objective comparison)
    base_e: Dict[Tuple[Any, ...], float] = {}
    for r in p1:
        if r["candidate"] == "baseline_0.40.0" and r["best_energy"] is not None:
            base_e[("P1", r["constraint"], r["sparseness"], r["k"])] = float(r["best_energy"])
    for r in p2:
        if r["candidate"] == "baseline_0.40.0" and r["best_energy"] is not None:
            base_e[("P2", r["n"], r["alpha"], r["constraint"], r["sparseness"], r["k"])] = float(r["best_energy"])
    for r in p3_ms:
        if r["candidate"] == "baseline_0.40.0" and r["default_energy"] is not None:
            base_e[("P3_def", r["constraint"], r["sparseness"], r["k"])] = float(r["default_energy"])
        if r["candidate"] == "baseline_0.40.0" and r["best_energy"] is not None:
            base_e[("P3_ms", r["constraint"], r["sparseness"], r["k"])] = float(r["best_energy"])
    for r in p4:
        if r["candidate"] == "baseline_0.40.0" and r["best_energy"] is not None:
            base_e[("P4", r["constraint"], r["sparseness"], r["k"])] = float(r["best_energy"])

    def _paired_for(cname: str, constraint_filter: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
        deltas: List[float] = []
        oks: List[bool] = []
        for r in p1:
            if r["candidate"] != cname or (constraint_filter and r["constraint"] != constraint_filter):
                continue
            eb = base_e.get(("P1", r["constraint"], r["sparseness"], r["k"]))
            if eb is not None and r["best_energy"] is not None:
                deltas.append(float(r["best_energy"]) - eb)
                oks.append(float(r["best_energy"]) <= eb + 1e-6)
        for r in p2:
            if r["candidate"] != cname or (constraint_filter and r["constraint"] != constraint_filter):
                continue
            eb = base_e.get(("P2", r["n"], r["alpha"], r["constraint"], r["sparseness"], r["k"]))
            if eb is not None and r["best_energy"] is not None:
                deltas.append(float(r["best_energy"]) - eb)
                oks.append(float(r["best_energy"]) <= eb + 1e-6)
        for r in p3_ms:
            if r["candidate"] != cname or (constraint_filter and r["constraint"] != constraint_filter):
                continue
            ebd = base_e.get(("P3_def", r["constraint"], r["sparseness"], r["k"]))
            if ebd is not None and r["default_energy"] is not None:
                deltas.append(float(r["default_energy"]) - ebd)
                oks.append(float(r["default_energy"]) <= ebd + 1e-6)
            ebm = base_e.get(("P3_ms", r["constraint"], r["sparseness"], r["k"]))
            if ebm is not None and r["best_energy"] is not None:
                deltas.append(float(r["best_energy"]) - ebm)
                oks.append(float(r["best_energy"]) <= ebm + 1e-6)
        for r in p4:
            if r["candidate"] != cname or (constraint_filter and r["constraint"] != constraint_filter):
                continue
            eb = base_e.get(("P4", r["constraint"], r["sparseness"], r["k"]))
            if eb is not None and r["best_energy"] is not None:
                deltas.append(float(r["best_energy"]) - eb)
                oks.append(float(r["best_energy"]) <= eb + 1e-6)
        if not oks:
            return None, None
        return 100.0 * sum(oks) / len(oks), float(np.median(deltas))

    # Candidate summary table (aggregated across all 4 constraints)
    summary_rows: List[Dict[str, Any]] = []
    for cname in CANDIDATE_NAMES:
        cfg = CANDIDATE_MAP[cname]
        fs_c = [r for r in first_steps if r["candidate"] == cname]
        sv_c = [r for r in same_v0 if r["candidate"] == cname]
        p1_c = [r for r in p1 if r["candidate"] == cname and r.get("has_stability")]
        p2_93_c = [
            r for r in p2
            if r["candidate"] == cname and r["n"] == 93 and abs(r["alpha"] - 1.0) < 1e-6
        ]
        p2_93_stab = [r for r in p2_93_c if r.get("has_stability") and r["k"] == 1 and abs(r["sparseness"] - 0.8) < 1e-6]
        p3_k1_ms = [r for r in p3_ms if r["candidate"] == cname and r["k"] == 1]
        p4_c = [r for r in p4 if r["candidate"] == cname]
        p4_stab = [r for r in p4_c if r.get("has_stability") and r["k"] == 1]

        same_v0_pct = (
            100.0 * sum(1 for r in sv_c if r["non_degraded"]) / len(sv_c)
            if sv_c
            else (100.0 if cname == "baseline_0.40.0" else None)
        )
        paired_pct, paired_med_de = _paired_for(cname)

        summary_rows.append(
            {
                "candidate": cname,
                "phase": cfg["phase"],
                "addresses": cfg["addresses"],
                "pre_step0_max_lr1": _max_v([r["pre_max_rel_lr1"] for r in fs_c]),
                "post_step0_max_lr1": _max_v([r["post_max_rel_lr1"] for r in fs_c]),
                "post_step0_feas_lr1": _max_v([r["post_max_rel_feas_lr1"] for r in fs_c]),
                "post_step0_med_auto": _median([r["post_med_rel_auto"] for r in fs_c]),
                "same_v0_nondegrade_pct": same_v0_pct,
                "same_v0_median_delta_e": _median([r["delta_energy"] for r in sv_c]) if sv_c else (0.0 if cname == "baseline_0.40.0" else None),
                "paired_nondegrade_pct": paired_pct,
                "paired_median_delta_e": paired_med_de,
                "p1_lr_gain": _median([r.get("latent_reliability_gain") for r in p1_c]),
                "p1_coup_gain": _median([r.get("cross_view_coupling_gain") for r in p1_c]),
                "p1_lr_avg": _median([r.get("latent_reliability_averaged") for r in p1_c]),
                "p2_r_z_true": _mean([r.get("r_z_true") for r in p2_93_c]),
                "p2_support_f1": _mean([r.get("support_f1") for r in p2_93_c]),
                "p2_lr_gain": _mean([r.get("latent_reliability_gain") for r in p2_93_stab]),
                "p2_coup_gain": _mean([r.get("cross_view_coupling_gain") for r in p2_93_stab]),
                "p2_fa_gain": _mean([r.get("feature_agreement_gain") for r in p2_93_stab]),
                "p3_n_distinct": _mean([float(r["n_distinct_solutions"]) for r in p3_k1_ms]),
                "p3_energy_gap": _median([r["energy_gap"] for r in p3_k1_ms]),
                "p4_r_z_true": _mean([r.get("r_z_true") for r in p4_c]),
                "p4_lr_gain": _mean([r.get("latent_reliability_gain") for r in p4_stab]),
                "p4_coup_gain": _mean([r.get("cross_view_coupling_gain") for r in p4_stab]),
                "mean_steps": _mean(
                    [float(r["n_steps"]) for r in p2_93_c + p4_c if r.get("n_steps") is not None]
                ),
                "mean_wall_s": _mean(
                    [float(r["wall_time"]) for r in p2_93_c + p4_c if not r.get("has_stability")]
                ),
            }
        )

    # Per-constraint breakdown (every candidate x constraint, never one-constraint-only)
    per_constraint_rows: List[Dict[str, Any]] = []
    for pos in CONSTRAINTS:
        for cname in CANDIDATE_NAMES:
            fs_cp = [r for r in first_steps if r["candidate"] == cname and r["constraint"] == pos]
            sv_cp = [r for r in same_v0 if r["candidate"] == cname and r["constraint"] == pos]
            p2_cp = [
                r for r in p2
                if r["candidate"] == cname and r["constraint"] == pos
                and r["n"] == 93 and abs(r["alpha"] - 1.0) < 1e-6
                and r.get("has_stability") and r["k"] == 1 and abs(r["sparseness"] - 0.8) < 1e-6
            ]
            p2_fit_cp = [
                r for r in p2
                if r["candidate"] == cname and r["constraint"] == pos
                and r["n"] == 93 and abs(r["alpha"] - 1.0) < 1e-6
            ]
            p4_cp = [
                r for r in p4
                if r["candidate"] == cname and r["constraint"] == pos and r.get("has_stability") and r["k"] == 1
            ]
            sv_pct = (
                100.0 * sum(1 for r in sv_cp if r["non_degraded"]) / len(sv_cp)
                if sv_cp else (100.0 if cname == "baseline_0.40.0" else None)
            )
            pr_pct, pr_de = _paired_for(cname, constraint_filter=pos)
            per_constraint_rows.append(
                {
                    "constraint": pos,
                    "candidate": cname,
                    "post_step0_feas_lr1": _max_v([r["post_max_rel_feas_lr1"] for r in fs_cp]),
                    "post_step0_own_lr1": _max_v([r["post_max_rel_lr1"] for r in fs_cp]),
                    "same_v0_nondegrade_pct": sv_pct,
                    "paired_nondegrade_pct": pr_pct,
                    "paired_median_delta_e": pr_de,
                    "p2_r_z_true": _mean([r.get("r_z_true") for r in p2_fit_cp]),
                    "p2_lr_gain": _mean([r.get("latent_reliability_gain") for r in p2_cp]),
                    "p2_coup_gain": _mean([r.get("cross_view_coupling_gain") for r in p2_cp]),
                    "p2_fa_gain": _mean([r.get("feature_agreement_gain") for r in p2_cp]),
                    "p4_lr_gain": _mean([r.get("latent_reliability_gain") for r in p4_cp]),
                    "p4_coup_gain": _mean([r.get("cross_view_coupling_gain") for r in p4_cp]),
                }
            )

    # P2 sample-size curve (n in {60, 93, 186, 400}, alpha=1.0, sp=0.8, k=1)
    ns_seen = sorted({int(r["n"]) for r in p2})
    sample_size_curve: List[Dict[str, Any]] = []
    for n_val in ns_seen:
        for cname in CANDIDATE_NAMES:
            sub = [
                r for r in p2
                if int(r["n"]) == n_val and r["candidate"] == cname
                and abs(float(r["alpha"]) - 1.0) < 1e-6
                and abs(float(r["sparseness"]) - 0.8) < 1e-6 and int(r["k"]) == 1
            ]
            if not sub:
                continue
            sub_stab = [r for r in sub if r.get("has_stability")]
            sample_size_curve.append(
                {
                    "n": n_val,
                    "candidate": cname,
                    "r_z_true": _mean([r.get("r_z_true") for r in sub]),
                    "support_f1": _mean([r.get("support_f1") for r in sub]),
                    "latent_reliability": _mean([r.get("latent_reliability") for r in sub_stab]),
                    "latent_reliability_gain": _mean([r.get("latent_reliability_gain") for r in sub_stab]),
                    "cross_view_coupling_gain": _mean([r.get("cross_view_coupling_gain") for r in sub_stab]),
                    "feature_agreement_gain": _mean([r.get("feature_agreement_gain") for r in sub_stab]),
                }
            )

    # P3 k=1 vs k=2 multivariate comparison
    p3_k_compare: List[Dict[str, Any]] = []
    for pos in CONSTRAINTS:
        for kv in sorted({int(r["k"]) for r in p3_stab}):
            for cname in sorted({r["candidate"] for r in p3_stab}):
                sub = [
                    r for r in p3_stab
                    if r["constraint"] == pos and int(r["k"]) == kv and r["candidate"] == cname
                ]
                ms_sub = [
                    r for r in p3_ms
                    if r["constraint"] == pos and int(r["k"]) == kv and r["candidate"] == cname
                ]
                if not sub:
                    continue
                r0 = sub[0]
                p3_k_compare.append(
                    {
                        "constraint": pos,
                        "k": kv,
                        "candidate": cname,
                        "lr_gain_mean": r0.get("latent_reliability_gain"),
                        "lr_gain_min": r0.get("latent_reliability_gain_min"),
                        "lr_mv_gain": r0.get("latent_reliability_multivariate_gain"),
                        "coup_gain": r0.get("cross_view_coupling_gain"),
                        "r_z_true": r0.get("r_z_true"),
                        "n_distinct": _mean([float(m["n_distinct_solutions"]) for m in ms_sub]),
                        "energy_gap": _median([m["energy_gap"] for m in ms_sub]),
                    }
                )

    # Section 8 Regime diagnosis for N=93 across all 4 constraints
    regime_diagnosis: List[Dict[str, Any]] = []
    for pos in CONSTRAINTS:
        # P1 Pure Noise (B_combined)
        p1_bc = [
            r for r in p1
            if r["constraint"] == pos and r["candidate"] == "B_combined"
            and r.get("has_stability") and r["k"] == 1 and abs(r["sparseness"] - 0.8) < 1e-6
        ]
        # P2 Borderline alpha=0.65 (B_combined)
        p2_weak = [
            r for r in p2
            if r["constraint"] == pos and r["candidate"] == "B_combined"
            and r["n"] == 93 and abs(r["alpha"] - 0.65) < 1e-6 and r.get("has_stability")
        ]
        # P2 Moderate alpha=1.0 (B_combined)
        p2_mod = [
            r for r in p2
            if r["constraint"] == pos and r["candidate"] == "B_combined"
            and r["n"] == 93 and abs(r["alpha"] - 1.0) < 1e-6
            and r.get("has_stability") and r["k"] == 1 and abs(r["sparseness"] - 0.8) < 1e-6
        ]
        for label, sub in [
            ("P1 Pure Noise (n=93, sp=0.8, k=1)", p1_bc),
            ("P2 Weak Shared alpha=0.65 (n=93, sp=0.8, k=1)", p2_weak),
            ("P2 Moderate Shared alpha=1.0 (n=93, sp=0.8, k=1)", p2_mod),
        ]:
            if not sub:
                continue
            r0 = sub[0]
            regime_diagnosis.append(
                {
                    "constraint": pos,
                    "case": label,
                    "latent_reliability": r0.get("latent_reliability"),
                    "latent_reliability_baseline": r0.get("latent_reliability_baseline"),
                    "latent_reliability_gain": r0.get("latent_reliability_gain"),
                    "latent_reliability_averaged_gain": r0.get("latent_reliability_averaged_gain"),
                    "cross_view_coupling_gain": r0.get("cross_view_coupling_gain"),
                    "feature_agreement_gain": r0.get("feature_agreement_gain"),
                    "regime": _classify_regime(
                        r0.get("latent_reliability"),
                        r0.get("latent_reliability_gain"),
                        r0.get("cross_view_coupling_gain"),
                    ),
                }
            )

    return {
        "candidate_summary": summary_rows,
        "per_constraint": per_constraint_rows,
        "sample_size_curve": sample_size_curve,
        "p3_k_compare": p3_k_compare,
        "regime_diagnosis": regime_diagnosis,
    }


# ---------------------------------------------------------------------------
# Self-Contained HTML Report Generator (SVG charts + tables)
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], digits: int = 3, pct: bool = False, signed: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if pct:
        return f"{v:.1f}%"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) > 0 and abs(v) < 1e-3:
        return f"{v:+.2e}" if signed else f"{v:.2e}"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _svg_grouped_bar_chart(
    title: str,
    categories: List[str],
    series: List[Tuple[str, str, List[Optional[float]]]],
    y_label: str,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    log_scale: bool = False,
    ref_line: Optional[Tuple[float, str]] = None,
) -> str:
    width, height = 780, 300
    pad_l, pad_r, pad_t, pad_b = 64, 20, 44, 78
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    all_vals = [
        float(v)
        for _, _, vs in series
        for v in vs
        if v is not None and np.isfinite(v) and (not log_scale or v > 0)
    ]
    if not all_vals:
        return f'<div class="chart-box"><h4>{html.escape(title)}</h4><p>No finite data</p></div>'

    if log_scale:
        tvals = [np.log10(max(v, 1e-4)) for v in all_vals]
        lo = float(np.floor(min(tvals))) if y_min is None else float(np.log10(max(y_min, 1e-4)))
        hi = float(np.ceil(max(tvals) + 0.1)) if y_max is None else float(np.log10(max(y_max, 1e-4)))
    else:
        lo = min(0.0, min(all_vals)) if y_min is None else float(y_min)
        hi = max(all_vals) * 1.15 if y_max is None else float(y_max)
        if hi <= lo:
            hi = lo + 1.0

    def y_px(val: float) -> float:
        tv = np.log10(max(val, 1e-4)) if log_scale else val
        frac = (tv - lo) / max(hi - lo, 1e-12)
        return pad_t + ph * (1.0 - max(0.0, min(1.0, frac)))

    lines = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{pad_l}" y="22" font-size="13" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
    ]
    lx = width - pad_r
    for s_name, color, _ in reversed(series):
        tw = len(s_name) * 6.4 + 18
        lx -= tw
        lines.append(f'<rect x="{lx}" y="11" width="10" height="10" rx="2" fill="{color}"/>')
        lines.append(f'<text x="{lx + 14}" y="20" font-size="10.5" fill="#334155">{html.escape(s_name)}</text>')
        lx -= 8

    n_ticks = 4
    for i in range(n_ticks + 1):
        tv = lo + (hi - lo) * i / n_ticks
        raw = (10.0 ** tv) if log_scale else tv
        yp = pad_t + ph * (1.0 - i / n_ticks)
        lbl = f"{raw:.2g}" if log_scale or abs(raw) >= 10 else f"{raw:.2f}"
        lines.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{width - pad_r}" y2="{yp:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{yp + 4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{lbl}</text>')

    if ref_line is not None:
        rval, rlbl = ref_line
        ry = y_px(rval)
        lines.append(f'<line x1="{pad_l}" y1="{ry:.1f}" x2="{width - pad_r}" y2="{ry:.1f}" stroke="#dc2626" stroke-width="1.4" stroke-dasharray="4,3"/>')
        lines.append(f'<text x="{width - pad_r - 2}" y="{ry - 4:.1f}" text-anchor="end" font-size="10" font-weight="600" fill="#dc2626">{html.escape(rlbl)}</text>')

    n_cat = len(categories)
    n_ser = len(series)
    grp_w = pw / max(n_cat, 1)
    bar_w = max(4.0, min(20.0, (grp_w * 0.78) / max(n_ser, 1)))
    zero_y = y_px(1e-4 if log_scale else max(0.0, lo))

    for ci, cat in enumerate(categories):
        gx = pad_l + ci * grp_w + (grp_w - bar_w * n_ser) / 2.0
        for si, (s_name, color, vals) in enumerate(series):
            val = vals[ci] if ci < len(vals) else None
            if val is None or not np.isfinite(val) or (log_scale and val <= 0):
                continue
            yp = y_px(val)
            bx = gx + si * bar_w
            bh = max(1.5, abs(zero_y - yp))
            by = min(yp, zero_y)
            lines.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w - 1.2:.1f}" height="{bh:.1f}" rx="2" fill="{color}">'
                f'<title>{html.escape(cat)} ({html.escape(s_name)}): {_fmt(val)}</title></rect>'
            )
        cx = pad_l + (ci + 0.5) * grp_w
        lines.append(
            f'<text x="{cx:.1f}" y="{height - pad_b + 14}" transform="rotate(-24 {cx:.1f} {height - pad_b + 14})" '
            f'text-anchor="end" font-size="10.5" fill="#1e293b">{html.escape(cat)}</text>'
        )

    lines.append(
        f'<text x="14" y="{pad_t + ph / 2:.1f}" transform="rotate(-90 14 {pad_t + ph / 2:.1f})" '
        f'text-anchor="middle" font-size="11" fill="#475569">{html.escape(y_label)}</text>'
    )
    lines.append("</svg>")
    return f'<div class="chart-box">{"".join(lines)}</div>'


def _svg_line_chart(
    title: str,
    x_vals: List[int],
    series: List[Tuple[str, str, List[Optional[float]]]],
    y_label: str,
    y_min: float = -0.1,
    y_max: Optional[float] = 1.05,
) -> str:
    width, height = 780, 280
    pad_l, pad_r, pad_t, pad_b = 62, 165, 40, 44
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    all_vals = [float(v) for _, _, vs in series for v in vs if v is not None and np.isfinite(v)]
    if not all_vals:
        return f'<div class="chart-box"><h4>{html.escape(title)}</h4><p>No finite data</p></div>'
    hi = max(all_vals) * 1.12 if y_max is None else float(y_max)
    lo = min(float(y_min), min(all_vals) - 0.05)
    if hi <= lo:
        hi = lo + 1.0

    def x_px(i: int) -> float:
        if len(x_vals) <= 1:
            return pad_l + pw / 2.0
        return pad_l + pw * (i / (len(x_vals) - 1))

    def y_px(val: float) -> float:
        frac = (val - lo) / max(hi - lo, 1e-12)
        return pad_t + ph * (1.0 - max(0.0, min(1.0, frac)))

    lines = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{pad_l}" y="22" font-size="13" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
    ]
    for i in range(5):
        tv = lo + (hi - lo) * i / 4
        yp = pad_t + ph * (1.0 - i / 4)
        lines.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{pad_l + pw}" y2="{yp:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{yp + 4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{tv:.2f}</text>')

    for idx, xv in enumerate(x_vals):
        xp = x_px(idx)
        lines.append(f'<line x1="{xp:.1f}" y1="{pad_t}" x2="{xp:.1f}" y2="{pad_t + ph}" stroke="#f1f5f9" stroke-width="1"/>')
        lines.append(f'<text x="{xp:.1f}" y="{height - 14}" text-anchor="middle" font-size="11" fill="#1e293b">n={xv}</text>')

    for si, (s_name, color, vals) in enumerate(series):
        pts = [(x_px(i), y_px(float(v)), float(v)) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]
        if pts:
            d = " ".join(("M" if k == 0 else "L") + f"{px:.1f},{py:.1f}" for k, (px, py, _) in enumerate(pts))
            lines.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
            for px, py, val in pts:
                lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{color}"><title>{html.escape(s_name)}: {_fmt(val)}</title></circle>')
        ly = pad_t + 10 + si * 18
        lines.append(f'<line x1="{pad_l + pw + 12}" y1="{ly}" x2="{pad_l + pw + 28}" y2="{ly}" stroke="{color}" stroke-width="2.4"/>')
        lines.append(f'<text x="{pad_l + pw + 34}" y="{ly + 4}" font-size="11" fill="#334155">{html.escape(s_name)}</text>')

    lines.append(
        f'<text x="14" y="{pad_t + ph / 2:.1f}" transform="rotate(-90 14 {pad_t + ph / 2:.1f})" '
        f'text-anchor="middle" font-size="11" fill="#475569">{html.escape(y_label)}</text>'
    )
    lines.append("</svg>")
    return f'<div class="chart-box">{"".join(lines)}</div>'


def render_html_report(data: Dict[str, Any]) -> str:
    meta = data["meta"]
    summary = data["summary"]["candidate_summary"]
    per_const = data["summary"]["per_constraint"]
    curve = data["summary"]["sample_size_curve"]
    p3_k = data["summary"]["p3_k_compare"]
    regimes = data["summary"]["regime_diagnosis"]
    first_steps = data["first_step"]

    sum_by_name = {r["candidate"]: r for r in summary}
    cats = [r["candidate"] for r in summary]

    # Chart 1: Post-projection first-step relative displacement by constraint (lr=1.0)
    fs_series = []
    for pos, col in [("linear", "#2563eb"), ("conical", "#059669"), ("convex", "#d97706"), ("mixed", "#7c3aed")]:
        vals = [
            _max_v([r["post_max_rel_feas_lr1"] for r in first_steps if r["candidate"] == c and r["constraint"] == pos])
            for c in cats
        ]
        fs_series.append((pos, col, vals))
    chart_step0 = _svg_grouped_bar_chart(
        "Criterion 5: Post-Projection Step-0 ||P(v1)-P(v0)|| / ||P(v0)|| (lr=1.0, feasible start, log scale)",
        cats,
        fs_series,
        "Max Post-Proj Step-0 (log10)",
        y_min=0.05,
        log_scale=True,
        ref_line=(1.0, "Trust cap = 1.0"),
    )

    # Chart 2: Energy non-degradation (%) per candidate
    nondeg_series = [
        ("Same-v0 runs (%)", "#0284c7", [sum_by_name[c]["same_v0_nondegrade_pct"] for c in cats]),
        ("All paired runs (%)", "#16a34a", [sum_by_name[c]["paired_nondegrade_pct"] for c in cats]),
    ]
    chart_energy = _svg_grouped_bar_chart(
        "Criterion 1: Energy Non-Degradation vs baseline_0.40.0 (<= baseline + 1e-6)",
        cats,
        nondeg_series,
        "% of runs <= baseline",
        y_min=0.0,
        y_max=108.0,
        ref_line=(95.0, "95% Bar"),
    )

    # Chart 3: Primary stability criteria on P2 (n=93, sp=0.8, k=1)
    p2_stab_series = [
        ("latent_reliability_gain", "#2563eb", [sum_by_name[c]["p2_lr_gain"] for c in cats]),
        ("cross_view_coupling_gain", "#059669", [sum_by_name[c]["p2_coup_gain"] for c in cats]),
        ("feature_agreement_gain", "#d97706", [sum_by_name[c]["p2_fa_gain"] for c in cats]),
    ]
    chart_p2_stab = _svg_grouped_bar_chart(
        "Criteria 2 & 6 (P2, n=93, sp=0.8, k=1): Calibrated Stability Gains across All 4 Constraints",
        cats,
        p2_stab_series,
        "Mean Gain vs Row-Permuted Baseline",
        y_min=-0.1,
        y_max=0.95,
    )

    # Chart 4: P1 Pure Noise calibration
    p1_cands = [r["candidate"] for r in summary if r["p1_lr_gain"] is not None]
    p1_series = [
        ("latent_reliability_gain", "#0ea5e9", [sum_by_name[c]["p1_lr_gain"] for c in p1_cands]),
        ("cross_view_coupling_gain", "#10b981", [sum_by_name[c]["p1_coup_gain"] for c in p1_cands]),
        ("latent_rel_averaged (NOT criterion)", "#94a3b8", [sum_by_name[c]["p1_lr_avg"] for c in p1_cands]),
    ]
    chart_p1 = _svg_grouped_bar_chart(
        "P1 (Pure Noise, N=93): Null Calibration of latent_reliability_gain & cross_view_coupling_gain",
        p1_cands,
        p1_series,
        "Median Metric Value",
        y_min=-0.25,
        y_max=0.6,
        ref_line=(0.0, "Zero Null"),
    )

    # Chart 5: Sample-size curve
    ns = sorted({int(r["n"]) for r in curve})
    curve_map = {(r["n"], r["candidate"]): r for r in curve}
    key_cands = [
        ("baseline_0.40.0", "#64748b"),
        ("B_combined", "#2563eb"),
    ]
    chart_n_stab = _svg_line_chart(
        "P2 Sample-Size Curve (sp=0.8, k=1): latent_reliability_gain & cross_view_coupling_gain",
        ns,
        [
            ("baseline lr_gain", "#94a3b8", [curve_map.get((nv, "baseline_0.40.0"), {}).get("latent_reliability_gain") for nv in ns]),
            ("B_combined lr_gain", "#2563eb", [curve_map.get((nv, "B_combined"), {}).get("latent_reliability_gain") for nv in ns]),
            ("baseline coup_gain", "#cbd5e1", [curve_map.get((nv, "baseline_0.40.0"), {}).get("cross_view_coupling_gain") for nv in ns]),
            ("B_combined coup_gain", "#059669", [curve_map.get((nv, "B_combined"), {}).get("cross_view_coupling_gain") for nv in ns]),
        ],
        "Mean Gain",
        y_min=-0.05,
        y_max=0.95,
    )
    chart_n_rec = _svg_line_chart(
        "P2 Sample-Size Curve (sp=0.8, k=1): Planted Factor Recovery |r(u, z_true)|",
        ns,
        [(cn, col, [curve_map.get((nv, cn), {}).get("r_z_true") for nv in ns]) for cn, col in key_cands]
        + [
            ("C1_graduated", "#059669", [curve_map.get((nv, "C1_graduated"), {}).get("r_z_true") for nv in ns]),
            ("C2_spectral_small", "#d97706", [curve_map.get((nv, "C2_spectral_small"), {}).get("r_z_true") for nv in ns]),
        ],
        "Mean |r(u, z_true)|",
        y_min=0.0,
        y_max=1.05,
    )

    # Chart 7: P3 k=1 vs k=2 (Competing basins vs multivariate trace ICC(1))
    p3_cats = [f"{pos} ({cn[:6]})" for pos in CONSTRAINTS for cn in ("baseline_0.40.0", "B_combined")]
    p3_map = {(r["constraint"], r["k"], r["candidate"]): r for r in p3_k}
    chart_p3 = _svg_grouped_bar_chart(
        "Problem 3 (Two Equal Factors): k=1 Rel Gain vs k=2 Multivariate Trace ICC(1) Gain",
        p3_cats,
        [
            ("k=1 lr_gain", "#94a3b8", [p3_map.get((pos, 1, cn), {}).get("lr_gain_mean") for pos in CONSTRAINTS for cn in ("baseline_0.40.0", "B_combined")]),
            ("k=2 mv_gain", "#2563eb", [p3_map.get((pos, 2, cn), {}).get("lr_mv_gain") for pos in CONSTRAINTS for cn in ("baseline_0.40.0", "B_combined")]),
            ("k=2 subspace_r", "#059669", [p3_map.get((pos, 2, cn), {}).get("r_z_true") for pos in CONSTRAINTS for cn in ("baseline_0.40.0", "B_combined")]),
        ],
        "Gain / Subspace Recovery",
        y_min=-0.2,
        y_max=1.05,
    )

    # Chart 8: P4 Missing Rows Mask across all 4 constraints
    pc_map = {(r["constraint"], r["candidate"]): r for r in per_const}
    p4_cands = [c for c in ("baseline_0.40.0", "B3_feasible_aligned", "B_combined") if any(pc_map.get((pos, c), {}).get("p4_lr_gain") is not None for pos in CONSTRAINTS)]
    chart_p4 = _svg_grouped_bar_chart(
        "Problem 4 (missing_rows='mask'): Stability Gains Across All 4 Constraints",
        CONSTRAINTS,
        [
            (f"{c[:8]} relG", col, [pc_map.get((pos, c), {}).get("p4_lr_gain") for pos in CONSTRAINTS])
            for c, col in zip(p4_cands, ["#64748b", "#0ea5e9", "#2563eb"])
        ]
        + [
            ("B_comb coupG", "#059669", [pc_map.get((pos, "B_combined"), {}).get("p4_coup_gain") for pos in CONSTRAINTS]),
        ],
        "Mean Gain vs Baseline",
        y_min=-0.1,
        y_max=0.95,
    )

    # Master §5 Summary Table
    master_rows_html = []
    for r in summary:
        sv_p = r["same_v0_nondegrade_pct"]
        pr_p = r["paired_nondegrade_pct"]
        badge = ""
        if r["candidate"] == "baseline_0.40.0":
            badge = '<span class="badge base">BASE</span>'
        elif sv_p is not None:
            badge = '<span class="badge pass">&ge;95%</span>' if sv_p >= 95.0 else '<span class="badge warn">&lt;95%</span>'
        master_rows_html.append(
            "<tr>"
            f"<td><strong>{html.escape(r['candidate'])}</strong><br><span class='sub'>{html.escape(r['phase'])}</span></td>"
            f"<td>{html.escape(r['addresses'])}</td>"
            f"<td>{_fmt(r['pre_step0_max_lr1'], 2)}</td>"
            f"<td>{_fmt(r['post_step0_feas_lr1'], 2)}</td>"
            f"<td>{_fmt(sv_p, pct=True)} {badge}</td>"
            f"<td>{_fmt(pr_p, pct=True)}</td>"
            f"<td>{_fmt(r['paired_median_delta_e'], 4, signed=True)}</td>"
            f"<td>{_fmt(r['p2_lr_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p2_coup_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p2_fa_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p2_r_z_true'])}</td>"
            f"<td>{_fmt(r['p3_n_distinct'], 2)}</td>"
            f"<td>{_fmt(r['p3_energy_gap'], 4)}</td>"
            f"<td>{_fmt(r['p4_lr_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p4_coup_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p4_r_z_true'])}</td>"
            f"<td>{_fmt(r['mean_steps'], 1)}</td>"
            f"<td>{_fmt(r['mean_wall_s'], 2)}s</td>"
            "</tr>"
        )

    # Per-Constraint Table (mandatory §5 / §8 per-constraint table)
    pc_rows_html = []
    for r in per_const:
        pc_rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(r['constraint'])}</code></td>"
            f"<td><strong>{html.escape(r['candidate'])}</strong></td>"
            f"<td>{_fmt(r['post_step0_feas_lr1'], 2)}</td>"
            f"<td>{_fmt(r['same_v0_nondegrade_pct'], pct=True)}</td>"
            f"<td>{_fmt(r['paired_nondegrade_pct'], pct=True)}</td>"
            f"<td>{_fmt(r['paired_median_delta_e'], 4, signed=True)}</td>"
            f"<td>{_fmt(r['p2_lr_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p2_coup_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p2_fa_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p2_r_z_true'])}</td>"
            f"<td>{_fmt(r['p4_lr_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['p4_coup_gain'], signed=True)}</td>"
            "</tr>"
        )

    # P3 k=1 vs k=2 table
    p3_rows_html = []
    for r in p3_k:
        p3_rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(r['constraint'])}</code></td>"
            f"<td>k={r['k']}</td>"
            f"<td><strong>{html.escape(r['candidate'])}</strong></td>"
            f"<td>{_fmt(r['lr_gain_mean'], signed=True)}</td>"
            f"<td>{_fmt(r['lr_gain_min'], signed=True)}</td>"
            f"<td>{_fmt(r['lr_mv_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['coup_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['r_z_true'])}</td>"
            f"<td>{_fmt(r['n_distinct'], 2)}</td>"
            f"<td>{_fmt(r['energy_gap'], 4)}</td>"
            "</tr>"
        )

    # Regime diagnosis table
    reg_rows_html = []
    for r in regimes:
        reg_rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(r['constraint'])}</code></td>"
            f"<td>{html.escape(r['case'])}</td>"
            f"<td>{_fmt(r['latent_reliability'])}</td>"
            f"<td>{_fmt(r['latent_reliability_baseline'])}</td>"
            f"<td>{_fmt(r['latent_reliability_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['latent_reliability_averaged_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['cross_view_coupling_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['feature_agreement_gain'], signed=True)}</td>"
            f"<td><strong>{html.escape(r['regime'])}</strong></td>"
            "</tr>"
        )

    # Sample-size table
    ss_rows_html = []
    for r in curve:
        ss_rows_html.append(
            "<tr>"
            f"<td>n={r['n']}</td>"
            f"<td><strong>{html.escape(r['candidate'])}</strong></td>"
            f"<td>{_fmt(r['r_z_true'])}</td>"
            f"<td>{_fmt(r['support_f1'])}</td>"
            f"<td>{_fmt(r['latent_reliability'])}</td>"
            f"<td>{_fmt(r['latent_reliability_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['cross_view_coupling_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['feature_agreement_gain'], signed=True)}</td>"
            "</tr>"
        )

    # Real-data outcome-free neuroimaging check (ADNI T1 vol, T1 thk, DTI FA; n=193)
    real_rows_html = []
    for r in data.get("real_data_check", []):
        real_rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(r['constraint'])}</code></td>"
            f"<td><strong>{html.escape(r['candidate'])}</strong></td>"
            f"<td>{_fmt(r['latent_reliability'])}</td>"
            f"<td>{_fmt(r['latent_reliability_baseline'])}</td>"
            f"<td>{_fmt(r['latent_reliability_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['latent_reliability_averaged'])}</td>"
            f"<td>{_fmt(r['cross_view_coupling'])}</td>"
            f"<td>{_fmt(r['cross_view_coupling_gain'], signed=True)}</td>"
            f"<td>{_fmt(r['feature_agreement'])}</td>"
            f"<td>{_fmt(r['feature_agreement_gain'], signed=True)}</td>"
            "</tr>"
        )
    real_section_html = (
        ""
        if not real_rows_html
        else f"""
  <h2>Real-Data Check (Outcome-Free 3-View Neuroimaging: T1 Volume, T1 Thickness, DTI FA; n=193)</h2>
  <div class="scroll-tbl">
    <table>
      <thead>
        <tr>
          <th>Constraint</th>
          <th>Candidate</th>
          <th>Latent Rel</th>
          <th>Rel Baseline</th>
          <th>Rel Gain</th>
          <th>Avg Score Rel</th>
          <th>Cross-View Coup</th>
          <th>Coup Gain</th>
          <th>Feature Agree</th>
          <th>FA Gain</th>
        </tr>
      </thead>
      <tbody>{''.join(real_rows_html)}</tbody>
    </table>
  </div>"""
    )

    b_row = sum_by_name["baseline_0.40.0"]
    bc_row = sum_by_name["B_combined"]
    c3_row = sum_by_name["C3_soft_sparseness"]
    su_demo = data.get("solution_uniqueness_demo", {})
    su_items = ", ".join(
        f"{pos}: unique={v.get('unique')} (n_distinct={v.get('n_distinct_solutions')}, gap={_fmt(v.get('energy_gap'), 4)})"
        for pos, v in su_demo.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SiMLR Section 5 Benchmark Report (v4) — Calibrated Latent Stability &amp; Part B/C</title>
<style>
  :root {{ --bg:#f8fafc; --card:#ffffff; --ink:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#1d4ed8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--ink); line-height:1.48; }}
  .wrap {{ max-width:1400px; margin:0 auto; padding:28px 24px 60px; }}
  h1 {{ margin:0 0 6px; font-size:24px; }}
  h2 {{ margin:28px 0 10px; font-size:18px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
  .meta {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin:16px 0 22px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
  .kpi .lbl {{ font-size:11.5px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; }}
  .kpi .val {{ font-size:21px; font-weight:700; margin:4px 0; color:#0f172a; }}
  .kpi .note {{ font-size:12.5px; color:var(--muted); }}
  .panel {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin-bottom:18px; }}
  .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(580px,1fr)); gap:16px; margin:14px 0; }}
  .chart-box {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }}
  .chart-svg {{ width:100%; height:auto; display:block; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }}
  th, td {{ border-bottom:1px solid var(--line); padding:6px 7px; text-align:left; vertical-align:top; }}
  th {{ background:#f1f5f9; font-weight:600; position:sticky; top:0; z-index:1; }}
  tr:hover td {{ background:#f8fafc; }}
  .scroll-tbl {{ max-height:460px; overflow:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }}
  .badge {{ display:inline-block; padding:2px 6px; border-radius:4px; font-size:10px; font-weight:700; margin-left:3px; }}
  .badge.pass {{ background:#dcfce7; color:#166534; }}
  .badge.warn {{ background:#fef3c7; color:#92400e; }}
  .badge.base {{ background:#e2e8f0; color:#334155; }}
  .sub {{ color:var(--muted); font-size:11px; }}
  .callout {{ border-left:4px solid var(--accent); background:#eff6ff; padding:12px 16px; border-radius:6px; margin:12px 0; }}
  .callout.warn {{ border-left-color:#d97706; background:#fffbeb; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>SiMLR Section 5 Benchmark Report (v4)</h1>
  <div class="meta">
    Mode: <strong>{html.escape(meta['mode'])}</strong> |
    Constraints: <code>{html.escape(', '.join(meta['constraints']))}</code> |
    Sparseness: <code>{html.escape(', '.join(str(s) for s in meta['sparseness_values']))}</code> |
    Ranks: <code>k={html.escape(', '.join(str(k) for k in meta['k_values']))}</code> |
    Subsamples: <code>{meta['n_subsamples']}</code> (n_baseline=<code>{meta['n_baseline']}</code>, fraction=0.8) |
    Multi-starts: <code>{meta['n_starts']}</code> |
    Runtime: <code>{meta['total_wall_s']:.1f}s</code>
  </div>

  <div class="kpi-grid">
    <div class="kpi">
      <div class="lbl">Post-Proj Step-0 Cap (lr=1.0)</div>
      <div class="val">{_fmt(b_row['post_step0_feas_lr1'], 2)} &rarr; {_fmt(bc_row['post_step0_feas_lr1'], 2)}</div>
      <div class="note">Max post-projection <code>||P(v1)-P(v0)|| / ||P(v0)||</code> across all 4 constraints</div>
    </div>
    <div class="kpi">
      <div class="lbl">Primary Stability Gains (P2, n=93)</div>
      <div class="val">rel {_fmt(bc_row['p2_lr_gain'], signed=True)} | coup {_fmt(bc_row['p2_coup_gain'], signed=True)}</div>
      <div class="note"><code>B_combined</code> mean <code>latent_reliability_gain</code> &amp; <code>cross_view_coupling_gain</code> vs baseline ({_fmt(b_row['p2_lr_gain'], signed=True)} / {_fmt(b_row['p2_coup_gain'], signed=True)})</div>
    </div>
    <div class="kpi">
      <div class="lbl">C3 Soft Feature Agreement Gain</div>
      <div class="val">{_fmt(bc_row['p2_fa_gain'], signed=True)} &rarr; {_fmt(c3_row['p2_fa_gain'], signed=True)}</div>
      <div class="note">P2 correlated-feature <code>feature_agreement_gain</code> (<code>B_combined</code> hard vs <code>C3_soft_sparseness</code>)</div>
    </div>
    <div class="kpi">
      <div class="lbl">P1 Null Calibration (Pure Noise)</div>
      <div class="val">rel {_fmt(bc_row['p1_lr_gain'], signed=True)} | coup {_fmt(bc_row['p1_coup_gain'], signed=True)}</div>
      <div class="note">Gains centered near 0 under null (whereas uncalibrated <code>latent_reliability_averaged</code> = {_fmt(bc_row['p1_lr_avg'])})</div>
    </div>
  </div>

  <div class="panel">
    <h2>Section 7 &amp; Section 8 Synthesis</h2>
    <div class="callout">
      <strong>Calibrated Stability &amp; Constraint Generality:</strong>
      Every metric is measured via <code>pysimlr.latent_stability(...)</code> across all four constraint regimes (<code>linear</code>, <code>conical</code>, <code>convex</code>, <code>mixed</code>). Under pure Gaussian noise (P1, <code>n=93</code>), <code>latent_reliability_gain</code> and <code>cross_view_coupling_gain</code> stay centered near zero while <code>latent_reliability_averaged</code> remains positive (confirming why averaged scores are reported but never used as a stability criterion). On correlated-feature views (P2), <code>C3_soft_sparseness</code> raises <code>feature_agreement_gain</code> from {_fmt(bc_row['p2_fa_gain'], signed=True)} to {_fmt(c3_row['p2_fa_gain'], signed=True)}.
    </div>
    <div class="callout warn">
      <strong>Small-Sample Regime (n=93, p=396) &amp; Multi-Start <code>solution_uniqueness</code>:</strong>
      When two shared directions compete (P3), <code>k=1</code> shows competing basins and erratic component-wise gains, whereas <code>k=2</code> recovers both directions jointly with high <code>latent_reliability_multivariate_gain</code>. Direct <code>solution_uniqueness</code> verification across constraints (<code>B_combined</code>, P3, sp=0.8, 5 starts): {html.escape(su_items)}.
    </div>
  </div>

  <h2>Visual Comparisons</h2>
  <div class="chart-grid">
    {chart_step0}
    {chart_energy}
    {chart_p2_stab}
    {chart_p1}
    {chart_n_stab}
    {chart_n_rec}
    {chart_p3}
    {chart_p4}
  </div>

  <h2>Section 5 Master Candidate Table (Aggregated Across Constraints, Sparseness &amp; Ranks)</h2>
  <div class="scroll-tbl">
    <table>
      <thead>
        <tr>
          <th>Candidate</th>
          <th>Change / Focus</th>
          <th>Pre-Step0 (lr=1)</th>
          <th>Post-Step0 (feas, lr=1)</th>
          <th>Same-v0 &le; Base</th>
          <th>Paired &le; Base</th>
          <th>Med &Delta;E</th>
          <th>P2 Rel Gain</th>
          <th>P2 Coup Gain</th>
          <th>P2 FA Gain</th>
          <th>P2 |r| / Subsp</th>
          <th>P3 Distinct</th>
          <th>P3 E-Gap</th>
          <th>P4 Rel Gain</th>
          <th>P4 Coup Gain</th>
          <th>P4 |r|</th>
          <th>Steps</th>
          <th>Wall</th>
        </tr>
      </thead>
      <tbody>{''.join(master_rows_html)}</tbody>
    </table>
  </div>

  <h2>Per-Constraint Breakdown (Every Candidate &times; Constraint)</h2>
  <div class="scroll-tbl">
    <table>
      <thead>
        <tr>
          <th>Constraint</th>
          <th>Candidate</th>
          <th>Post-Step0 (feas, lr=1)</th>
          <th>Same-v0 &le; Base</th>
          <th>Paired &le; Base</th>
          <th>Med &Delta;E</th>
          <th>P2 Rel Gain</th>
          <th>P2 Coup Gain</th>
          <th>P2 FA Gain</th>
          <th>P2 |r| / Subsp</th>
          <th>P4 Rel Gain</th>
          <th>P4 Coup Gain</th>
        </tr>
      </thead>
      <tbody>{''.join(pc_rows_html)}</tbody>
    </table>
  </div>

  <h2>Problem 3: Two Nearly Equal Shared Factors (k=1 Competing Basins vs k=2 Trace ICC(1))</h2>
  <div class="scroll-tbl">
    <table>
      <thead>
        <tr>
          <th>Constraint</th>
          <th>Rank</th>
          <th>Candidate</th>
          <th>Rel Gain (Mean)</th>
          <th>Rel Gain (Min)</th>
          <th>Multivariate Gain</th>
          <th>Coupling Gain</th>
          <th>Subspace Recovery</th>
          <th>Distinct Sols</th>
          <th>Energy Gap</th>
        </tr>
      </thead>
      <tbody>{''.join(p3_rows_html)}</tbody>
    </table>
  </div>

  <h2>Problem 2 Sample-Size Curve (n &isin; {{60, 93, 186, 400}}, sp=0.8, k=1)</h2>
  <div class="scroll-tbl">
    <table>
      <thead>
        <tr>
          <th>Sample Size</th>
          <th>Candidate</th>
          <th>|r(u, z_true)|</th>
          <th>Support F1</th>
          <th>Latent Rel</th>
          <th>Rel Gain</th>
          <th>Coup Gain</th>
          <th>FA Gain</th>
        </tr>
      </thead>
      <tbody>{''.join(ss_rows_html)}</tbody>
    </table>
  </div>

  <h2>Section 8 Regime Diagnosis (Motivating Dimensions n=93, widths=[66,98,36,100,96])</h2>
  <div class="scroll-tbl">
    <table>
      <thead>
        <tr>
          <th>Constraint</th>
          <th>Case (B_combined)</th>
          <th>Latent Rel</th>
          <th>Rel Baseline</th>
          <th>Rel Gain</th>
          <th>Avg Score Gain</th>
          <th>Coupling Gain</th>
          <th>FA Gain</th>
          <th>Section 8 Regime</th>
        </tr>
      </thead>
      <tbody>{''.join(reg_rows_html)}</tbody>
    </table>
  </div>{real_section_html}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def _run_real_data_check(smoke: bool = False) -> List[Dict[str, Any]]:
    """Run outcome-free real-data latent_stability check on bundled 3-view neuroimaging (n=193)."""
    import pandas as pd

    csv_path = _REPO_ROOT / "data/adni/adni_t1_dti_biomarker_merged.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    v0 = df[[c for c in df.columns if c.startswith("T1Hier__T1Hier_vol_")]].values.astype(np.float64)
    v1 = df[[c for c in df.columns if c.startswith("T1Hier__T1Hier_thk_")]].values.astype(np.float64)
    v2 = df[[c for c in df.columns if c.startswith("DTI_fa__")]].values.astype(np.float64)
    views = [v0[:60, :30], v1[:60, :30], v2[:60, :20]] if smoke else [v0, v1, v2]
    n_sub = 2 if smoke else 20
    n_base = 1 if smoke else 3
    iters = 4 if smoke else 25
    real_cands = ["baseline_0.40.0", "B_combined", "C1_graduated", "C3_soft_sparseness"]
    rows: List[Dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pos in CONSTRAINTS:
            ckw = (
                {
                    "per_view_params": {
                        0: {"positivity": "convex", "sparseness": 0.8},
                        1: {"positivity": "conical", "sparseness": 0.8},
                        2: {"positivity": "linear", "sparseness": 0.8},
                    }
                }
                if pos == "mixed"
                else {"positivity": pos, "sparseness": 0.8}
            )
            for cname in real_cands:
                cfg = {
                    kk: vv
                    for kk, vv in CANDIDATE_MAP[cname].items()
                    if kk not in ("phase", "addresses")
                }
                st = latent_stability(
                    "simlr",
                    views,
                    1,
                    n_subsamples=n_sub,
                    fraction=0.8,
                    baseline=True,
                    n_baseline=n_base,
                    random_state=0,
                    iterations=iters,
                    **ckw,
                    **cfg,
                )
                rows.append(
                    {
                        "constraint": pos,
                        "candidate": cname,
                        "latent_reliability": float(st["latent_reliability"][0]),
                        "latent_reliability_baseline": float(st["latent_reliability_baseline"][0]),
                        "latent_reliability_gain": float(st["latent_reliability_gain"][0]),
                        "latent_reliability_averaged": float(st["latent_reliability_averaged"][0]),
                        "cross_view_coupling": float(st["cross_view_coupling"][0]),
                        "cross_view_coupling_gain": float(st["cross_view_coupling_gain"][0]),
                        "feature_agreement": float(st["feature_agreement"][0]),
                        "feature_agreement_gain": float(st["feature_agreement_gain"][0]),
                    }
                )
    return rows


def run_benchmark(
    smoke: bool = False,
    out_json: Optional[str] = None,
    out_html: Optional[str] = None,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    if smoke:
        widths = [14, 16, 12]
        sparseness_values = [0.8]
        k_values = [1, 2]
        iterations = 5
        sv_iterations = 5
        n_subsamples = 4
        n_baseline = 1
        n_starts_p3_hi = 3
        n_starts_p3_lo = 2
        same_v0_seeds = [0]
        p2_curve_ns = [60]
        max_workers = workers or min(8, os.cpu_count() or 1)

        fs_cands = CANDIDATE_NAMES
        p1_stab_cands_k1 = ["baseline_0.40.0", "B_combined"]
        p1_stab_cands_k2 = ["B_combined"]
        p2_stab_cands_main = ["baseline_0.40.0", "B_combined", "C3_soft_sparseness"]
        p2_stab_cands_sp05: List[str] = []
        p2_stab_cands_k2: List[str] = []
        p2_stab_cands_curve = ["B_combined"]
        p2_stab_cands_alpha = ["B_combined"]
        p3_stab_cands = ["B_combined"]
        p3_ms_cands_hi = ["baseline_0.40.0", "B_combined", "C1_graduated", "C2_spectral_small"]
        p3_ms_cands_lo: List[str] = []
        p4_stab_cands_k1 = ["baseline_0.40.0", "B_combined"]
        p4_stab_cands_k2: List[str] = []
        full_grid_fit_cands = [
            "baseline_0.40.0", "B1_trust", "B_combined",
            "C1_graduated", "C2_spectral_small", "C3_soft_sparseness",
        ]
        smoke_fit_cands_p14 = ["baseline_0.40.0", "B_combined"]
    else:
        widths = list(WIDTHS)
        sparseness_values = [0.5, 0.8]
        k_values = [1, 2]
        iterations = 22
        sv_iterations = 50
        n_subsamples = 12
        n_baseline = 2
        n_starts_p3_hi = 20
        n_starts_p3_lo = 8
        same_v0_seeds = [0, 1, 2]
        p2_curve_ns = [60, 186, 400]
        max_workers = workers or min(10, os.cpu_count() or 1)

        fs_cands = CANDIDATE_NAMES
        p1_stab_cands_k1 = ["baseline_0.40.0", "B1_trust", "B_combined"]
        p1_stab_cands_k2 = ["baseline_0.40.0", "B_combined"]
        p2_stab_cands_main = CANDIDATE_NAMES
        p2_stab_cands_sp05 = ["baseline_0.40.0", "B_combined", "C3_soft_sparseness"]
        p2_stab_cands_k2 = ["baseline_0.40.0", "B_combined"]
        p2_stab_cands_curve = ["baseline_0.40.0", "B_combined"]
        p2_stab_cands_alpha = ["baseline_0.40.0", "B_combined"]
        p3_stab_cands = ["baseline_0.40.0", "B_combined"]
        p3_ms_cands_hi = CANDIDATE_NAMES
        p3_ms_cands_lo = ["baseline_0.40.0", "B_combined", "C1_graduated", "C2_spectral_small", "C3_soft_sparseness"]
        p4_stab_cands_k1 = ["baseline_0.40.0", "B3_feasible_aligned", "B_combined"]
        p4_stab_cands_k2 = ["baseline_0.40.0", "B_combined"]
        full_grid_fit_cands = CANDIDATE_NAMES
        smoke_fit_cands_p14 = CANDIDATE_NAMES

    # 1. Criterion 5: First-step pre/post-projection tasks
    fs_tasks = [
        (cname, pos, sp, widths)
        for pos in CONSTRAINTS
        for sp in sparseness_values
        for cname in fs_cands
    ]

    # 2. Criterion 1: Same-v0 energy tasks
    sv_tasks = [
        (pos, sp, kv, seed, sv_iterations, widths)
        for pos in CONSTRAINTS
        for sp in sparseness_values
        for kv in k_values
        for seed in same_v0_seeds
    ]

    smoke_stab_pos = ("linear", "mixed") if smoke else tuple(CONSTRAINTS)

    # 3. P1 Pure Noise: stability on selected candidates + single-fit across full grid
    p1_tasks = []
    for pos in CONSTRAINTS:
        for sp in sparseness_values:
            for kv in k_values:
                for cname in smoke_fit_cands_p14:
                    run_stab = (pos in smoke_stab_pos) and (
                        (kv == 1 and abs(sp - 0.8) < 1e-6 and cname in p1_stab_cands_k1)
                        or (kv == 2 and abs(sp - 0.8) < 1e-6 and cname in p1_stab_cands_k2)
                        or (not smoke and kv == 1 and abs(sp - 0.5) < 1e-6 and cname in p1_stab_cands_k2)
                    )
                    p1_tasks.append((cname, pos, sp, kv, iterations, n_subsamples, n_baseline, widths, run_stab))

    # 4. P2 Planted Shared Factor: full grid (n=93, alpha=1.0) + sample-size curve + alpha=0.65
    p2_tasks = []
    for pos in CONSTRAINTS:
        for sp in sparseness_values:
            for kv in k_values:
                for cname in full_grid_fit_cands:
                    if smoke and kv == 2 and cname not in ("baseline_0.40.0", "B_combined"):
                        continue
                    run_stab = (
                        (kv == 1 and abs(sp - 0.8) < 1e-6 and cname in p2_stab_cands_main)
                        or (kv == 1 and abs(sp - 0.5) < 1e-6 and cname in p2_stab_cands_sp05)
                        or (kv == 2 and abs(sp - 0.8) < 1e-6 and cname in p2_stab_cands_k2)
                    )
                    p2_tasks.append((cname, pos, sp, kv, 93, 1.0, iterations, n_subsamples, n_baseline, widths, run_stab))
    for nv in p2_curve_ns:
        for pos in smoke_stab_pos:
            for cname in ["baseline_0.40.0", "B_combined", "C1_graduated", "C2_spectral_small"]:
                if smoke and cname in ("C1_graduated", "C2_spectral_small"):
                    continue
                run_stab = cname in p2_stab_cands_curve
                p2_tasks.append((cname, pos, 0.8, 1, nv, 1.0, iterations, n_subsamples, n_baseline, widths, run_stab))
    for pos in smoke_stab_pos:
        for cname in ["baseline_0.40.0", "B_combined", "C1_graduated", "C2_spectral_small", "C3_soft_sparseness"]:
            if smoke and cname not in p2_stab_cands_alpha:
                continue
            run_stab = cname in p2_stab_cands_alpha
            p2_tasks.append((cname, pos, 0.8, 1, 93, 0.65, iterations, n_subsamples, n_baseline, widths, run_stab))

    # 5. P3 Two Equal-Strength Factors (k=1 competing basins vs k=2 multivariate ICC)
    p3_stab_tasks = [
        (cname, pos, 0.8, kv, iterations, n_subsamples, n_baseline, widths)
        for pos in smoke_stab_pos
        for kv in k_values
        for cname in p3_stab_cands
    ]
    p3_ms_tasks = [
        (cname, pos, 0.8, 1, iterations, n_starts_p3_hi, widths)
        for pos in CONSTRAINTS
        for cname in p3_ms_cands_hi
    ]
    if p3_ms_cands_lo:
        for pos in CONSTRAINTS:
            for cname in p3_ms_cands_lo:
                p3_ms_tasks.append((cname, pos, 0.5, 1, iterations, n_starts_p3_lo, widths))
                if cname in ("baseline_0.40.0", "B_combined", "C2_spectral_small"):
                    p3_ms_tasks.append((cname, pos, 0.8, 2, iterations, n_starts_p3_lo, widths))

    # 6. P4 Missing View Mask
    p4_tasks = []
    for pos in CONSTRAINTS:
        for sp in sparseness_values:
            for kv in k_values:
                for cname in smoke_fit_cands_p14:
                    if smoke and kv == 2:
                        continue
                    run_stab = (pos in smoke_stab_pos) and (
                        (kv == 1 and abs(sp - 0.8) < 1e-6 and cname in p4_stab_cands_k1)
                        or (kv == 2 and abs(sp - 0.8) < 1e-6 and cname in p4_stab_cands_k2)
                    )
                    p4_tasks.append((cname, pos, sp, kv, iterations, n_subsamples, n_baseline, widths, run_stab))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as ex:
        f_fs = [ex.submit(_task_first_step, t) for t in fs_tasks]
        f_sv = [ex.submit(_task_same_v0_energy, t) for t in sv_tasks]
        f_p1 = [ex.submit(_task_p1, t) for t in p1_tasks]
        f_p2 = [ex.submit(_task_p2, t) for t in p2_tasks]
        f_p3s = [ex.submit(_task_p3_stability, t) for t in p3_stab_tasks]
        f_p3m = [ex.submit(_task_p3_multistart, t) for t in p3_ms_tasks]
        f_p4 = [ex.submit(_task_p4, t) for t in p4_tasks]

        first_step_rows = [f.result() for f in f_fs]
        same_v0_rows = [row for f in f_sv for row in f.result()]
        p1_rows = [f.result() for f in f_p1]
        p2_rows = [f.result() for f in f_p2]
        p3_stab_rows = [f.result() for f in f_p3s]
        p3_ms_rows = [f.result() for f in f_p3m]
        p4_rows = [f.result() for f in f_p4]

    # Direct solution_uniqueness API verification across all 4 constraints
    _, v3_demo = make_problem3_two_factors(seed=202, n=93, widths=widths)
    su_demo: Dict[str, Dict[str, Any]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pos in CONSTRAINTS:
            su_res = solution_uniqueness(
                "simlr",
                v3_demo,
                1,
                n_starts=3 if smoke else 5,
                sparseness=0.8,
                iterations=iterations,
                optimizer_type="hybrid_adam_trust",
                align_initial_views=True,
                tune_learning_rate="per_refit",
                **constraint_kwargs(pos, len(v3_demo)),
            )
            su_demo[pos] = {
                "unique": bool(su_res["unique"]),
                "n_distinct_solutions": int(su_res["n_distinct_solutions"]),
                "energy_gap": _safe_float(su_res["energy_gap"]),
            }

    real_data_rows = _run_real_data_check(smoke=smoke)
    total_wall = time.perf_counter() - t_start
    payload: Dict[str, Any] = {
        "meta": {
            "mode": "smoke" if smoke else "full",
            "widths": list(widths),
            "constraints": list(CONSTRAINTS),
            "sparseness_values": sparseness_values,
            "k_values": k_values,
            "iterations": iterations,
            "n_subsamples": n_subsamples,
            "n_baseline": n_baseline,
            "n_starts": n_starts_p3_hi,
            "workers": max_workers,
            "total_wall_s": float(total_wall),
        },
        "first_step": first_step_rows,
        "same_v0_energy": same_v0_rows,
        "p1_pure_noise": p1_rows,
        "p2_planted_shared": p2_rows,
        "p3_stability": p3_stab_rows,
        "p3_multistart": p3_ms_rows,
        "p4_missing_mask": p4_rows,
        "solution_uniqueness_demo": su_demo,
        "real_data_check": real_data_rows,
    }
    payload["summary"] = _compute_summaries(payload)

    if out_json:
        p_json = Path(out_json)
        p_json.parent.mkdir(parents=True, exist_ok=True)
        p_json.write_text(json.dumps(payload, indent=2))
    if out_html:
        p_html = Path(out_html)
        p_html.parent.mkdir(parents=True, exist_ok=True)
        p_html.write_text(render_html_report(payload))

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Section 5 SiMLR calibrated stability & Part B/C benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run fast reduced benchmark (<15s)")
    parser.add_argument("--out-json", default=None, help="Output JSON report path")
    parser.add_argument("--out-html", default=None, help="Output HTML report path")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker processes")
    args = parser.parse_args()

    default_json = (
        "/tmp/bs_stability_smoke.json"
        if args.smoke
        else str(_REPO_ROOT / "reports/bs_stability_benchmark.json")
    )
    default_html = (
        "/tmp/bs_stability_smoke.html"
        if args.smoke
        else str(_REPO_ROOT / "reports/bs_stability_benchmark_report.html")
    )
    out_json = args.out_json or default_json
    out_html = args.out_html or default_html

    res = run_benchmark(
        smoke=args.smoke,
        out_json=out_json,
        out_html=out_html,
        workers=args.workers,
    )
    print(f"[{res['meta']['mode']}] Finished in {res['meta']['total_wall_s']:.2f}s -> JSON: {out_json} | HTML: {out_html}")
    for r in res["summary"]["candidate_summary"]:
        print(
            f"  {r['candidate']:19s} | post0={_fmt(r['post_step0_feas_lr1'], 2):>5s} | "
            f"sv0={_fmt(r['same_v0_nondegrade_pct'], pct=True):>6s} | "
            f"pair={_fmt(r['paired_nondegrade_pct'], pct=True):>6s} | "
            f"P2_relG={_fmt(r['p2_lr_gain'], signed=True):>6s} | "
            f"P2_coupG={_fmt(r['p2_coup_gain'], signed=True):>6s} | "
            f"P2_faG={_fmt(r['p2_fa_gain'], signed=True):>6s} | "
            f"P3_dist={_fmt(r['p3_n_distinct'], 2):>4s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
