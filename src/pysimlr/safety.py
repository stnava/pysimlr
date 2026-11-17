"""Multi-view data safety diagnostics (`pysimlr.data_safety`).

Standard row, column, group, and optional leave-one-row-out (LOO) sensitivity
diagnostics on prepared (or raw) multi-view matrices using ``numpy``,
``scipy.stats``, and ``sklearn.covariance.MinCovDet``. All diagnostics are
advisory: input data are never dropped or modified.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .multistart import SAME_SOLUTION_CORR
from .prepare import SD_FLOOR, _fit_mask, _pack, _split_views
from .utils import ConvergenceWarning, DataSafetyWarning, group_codes

SAFETY_MODES: Tuple[str, str, str] = ("warn", "raise", "off")
MAD_SCALE: float = 1.4826
NEAR_CONSTANT_SD: float = 1e-6
GROUP_CONCENTRATION_SHARE: float = 0.5
GROUP_CONCENTRATION_ALPHA: float = 0.05


def validate_safety_mode(safety: Any) -> str:
    """Validate ``safety`` in ``('warn', 'raise', 'off')``."""
    if not isinstance(safety, str) or safety not in SAFETY_MODES:
        raise ValueError(f"safety must be one of {SAFETY_MODES}; got {safety!r}.")
    return safety


def _is_prepared_result(obj: Any) -> bool:
    return (
        isinstance(obj, Mapping)
        and "views" in obj
        and "fit_rows" in obj
        and isinstance(obj.get("params"), Mapping)
        and "order" in obj["params"]
        and isinstance(obj["params"].get("views"), list)
    )


def _validate_thresholds(
    n_pc: Any,
    robust_z: Any,
    leverage_ratio: Any,
    md_quantile: Any,
) -> Tuple[int, float, float, float]:
    if isinstance(n_pc, bool) or not isinstance(n_pc, (int, np.integer)) or int(n_pc) < 1:
        raise ValueError(f"n_pc must be an integer >= 1; got {n_pc!r}.")
    if isinstance(robust_z, bool) or not isinstance(robust_z, (int, float, np.floating, np.integer)):
        raise ValueError(f"robust_z must be a positive float; got {robust_z!r}.")
    rz = float(robust_z)
    if not np.isfinite(rz) or rz <= 0.0:
        raise ValueError(f"robust_z must be a finite positive float; got {robust_z!r}.")
    if isinstance(leverage_ratio, bool) or not isinstance(leverage_ratio, (int, float, np.floating, np.integer)):
        raise ValueError(f"leverage_ratio must be a positive float; got {leverage_ratio!r}.")
    lr = float(leverage_ratio)
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"leverage_ratio must be a finite positive float; got {leverage_ratio!r}.")
    if isinstance(md_quantile, bool) or not isinstance(md_quantile, (int, float, np.floating, np.integer)):
        raise ValueError(f"md_quantile must be a float in (0, 1); got {md_quantile!r}.")
    mq = float(md_quantile)
    if not np.isfinite(mq) or mq <= 0.0 or mq >= 1.0:
        raise ValueError(f"md_quantile must be strictly between 0 and 1; got {md_quantile!r}.")
    return int(n_pc), rz, lr, mq


def _duplicated_pairs(zf: np.ndarray, non_const: np.ndarray) -> List[Tuple[int, int]]:
    """Exact standardized duplicate column pairs ``(j1, j2)`` with ``j1 < j2``.

    Opposite-sign copies (``r ~ -1``, such as intentional ``[x, -x]`` features)
    have negative inner product and are ignored.
    """
    if non_const.size < 2 or zf.shape[0] < 2:
        return []
    sub = zf[:, non_const]
    gram = (sub.T @ sub) / float(zf.shape[0])
    ii, jj = np.where(np.triu(gram > 1.0 - 1e-4, k=1))
    pairs: List[Tuple[int, int]] = []
    for a, b in zip(ii.tolist(), jj.tolist()):
        c1, c2 = int(non_const[a]), int(non_const[b])
        if np.allclose(zf[:, c1], zf[:, c2], rtol=1e-5, atol=1e-6):
            pairs.append((c1, c2))
    return pairs


def _check_one_view(
    x: np.ndarray,
    name: str,
    fit: np.ndarray,
    *,
    n_pc: int,
    robust_z: float,
    leverage_ratio: float,
    md_quantile: float,
    pre_miss: Optional[np.ndarray] = None,
    pre_sd: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n, p = x.shape
    xc = np.where(np.isfinite(x), x, np.nan)
    obs = np.isfinite(xc).any(axis=1)
    active = fit & obs
    n_obs = int(active.sum())

    mean_abs_z = np.full(n, np.nan, dtype=np.float64)
    row_rz = np.full(n, np.nan, dtype=np.float64)
    lev_ratio = np.full(n, np.nan, dtype=np.float64)
    md2 = np.full(n, np.nan, dtype=np.float64)

    f_scale = np.zeros(n, dtype=bool)
    f_lev = np.zeros(n, dtype=bool)
    f_md = np.zeros(n, dtype=bool)
    notes: List[str] = []

    if pre_miss is not None and pre_miss.shape == (p,):
        miss_frac = np.asarray(pre_miss, dtype=np.float64).copy()
    elif n_obs > 0:
        miss_frac = (~np.isfinite(xc[active])).mean(axis=0).astype(np.float64)
    else:
        miss_frac = np.full(p, np.nan, dtype=np.float64)

    max_col_rz = np.full(p, np.nan, dtype=np.float64)
    const_idx = np.empty(0, dtype=np.int64)
    near_const_idx = np.empty(0, dtype=np.int64)
    single_ext_idx = np.empty(0, dtype=np.int64)
    dup_pairs: List[Tuple[int, int]] = []

    if n_obs < 2:
        notes.append(f"{name}: fewer than 2 active rows ({n_obs}); row and column checks skipped.")
    else:
        xa = xc[active]
        has_val = np.isfinite(xa).any(axis=0)
        col_sd_full = np.zeros(p, dtype=np.float64)
        zf = np.zeros((n_obs, p), dtype=np.float64)
        z_nan = np.full((n_obs, p), np.nan, dtype=np.float64)

        if has_val.any():
            xa_ok = xa[:, has_val]
            col_med = np.nanmedian(xa_ok, axis=0)
            col_mean = np.nanmean(xa_ok, axis=0)
            col_sd = np.nanstd(xa_ok, axis=0, ddof=0)
            col_sd_full[has_val] = col_sd

            scale = np.where(col_sd > SD_FLOOR, col_sd, 1.0)
            z_ok = np.where(col_sd > SD_FLOOR, (xa_ok - col_mean) / scale, 0.0)
            z_ok = np.where(np.isfinite(xa_ok), z_ok, np.nan)
            z_nan[:, has_val] = z_ok

            filled_ok = xa_ok.copy()
            miss_ix = np.where(~np.isfinite(filled_ok))
            if miss_ix[0].size:
                filled_ok[miss_ix] = col_med[miss_ix[1]]
            f_mean = filled_ok.mean(axis=0)
            f_sd = filled_ok.std(axis=0, ddof=0)
            f_scale_col = np.where(f_sd > SD_FLOOR, f_sd, 1.0)
            zf[:, has_val] = np.where(f_sd > SD_FLOOR, (filled_ok - f_mean) / f_scale_col, 0.0)

            # Column robust z (Rule 4)
            abs_dev = np.abs(xa_ok - col_med)
            col_mad = MAD_SCALE * np.nanmedian(abs_dev, axis=0)
            col_rz = np.zeros_like(xa_ok, dtype=np.float64)
            for j in range(xa_ok.shape[1]):
                fin_j = np.isfinite(xa_ok[:, j])
                if not fin_j.any():
                    col_rz[:, j] = np.nan
                elif col_sd[j] <= SD_FLOOR:
                    col_rz[fin_j, j] = 0.0
                    col_rz[~fin_j, j] = np.nan
                elif col_mad[j] > SD_FLOOR:
                    col_rz[fin_j, j] = abs_dev[fin_j, j] / col_mad[j]
                    col_rz[~fin_j, j] = np.nan
                else:
                    col_rz[fin_j, j] = np.where(abs_dev[fin_j, j] > SD_FLOOR * col_sd[j], np.inf, 0.0)
                    col_rz[~fin_j, j] = np.nan

            max_ok = np.nanmax(col_rz, axis=0)
            max_col_rz[has_val] = max_ok
            above = np.where(np.isnan(col_rz), False, col_rz > robust_z)
            ok_cols = np.flatnonzero(has_val)
            single_ext_idx = ok_cols[np.sum(above, axis=0) == 1].astype(np.int64)

        ref_sd = np.asarray(pre_sd, dtype=np.float64) if (pre_sd is not None and pre_sd.shape == (p,)) else col_sd_full
        is_const = (~has_val) | (col_sd_full <= SD_FLOOR) | (ref_sd <= SD_FLOOR)
        is_near_const = (~is_const) & (ref_sd <= NEAR_CONSTANT_SD)
        const_idx = np.flatnonzero(is_const).astype(np.int64)
        near_const_idx = np.flatnonzero(is_near_const).astype(np.int64)

        non_const = np.flatnonzero(has_val & (col_sd_full > SD_FLOOR))
        dup_pairs = _duplicated_pairs(zf, non_const)

        # Rule 1: Row scale (mean |z| across features, robust z over active rows)
        if has_val.any():
            maz = np.nanmean(np.abs(z_nan[:, has_val]), axis=1)
            med_r = float(np.median(maz))
            mad_r = MAD_SCALE * float(np.median(np.abs(maz - med_r)))
            if mad_r > SD_FLOOR:
                rz = (maz - med_r) / mad_r
            else:
                rz = np.where(maz > med_r + SD_FLOOR, np.inf, 0.0)
            mean_abs_z[active] = maz
            row_rz[active] = rz
            f_scale[active] = np.where(np.isnan(rz), False, rz > robust_z)

        # Rule 2 (Leverage) & Rule 3 (MinCovDet squared Mahalanobis distance)
        zc = zf - zf.mean(axis=0)
        u_svd, s_svd, _ = np.linalg.svd(zc, full_matrices=False)
        s0 = float(s_svd[0]) if s_svd.size else 0.0
        n_pos = int((s_svd > SD_FLOOR * max(s0, 1.0)).sum())
        k_pc = min(n_pc, n_obs - 1, n_pos)

        if k_pc >= 1:
            h = (u_svd[:, :k_pc] ** 2).sum(axis=1)
            avg_h = float(k_pc) / float(n_obs)
            lr_vals = h / avg_h
            lev_ratio[active] = lr_vals
            f_lev[active] = lr_vals > leverage_ratio

            k_mcd = min(k_pc, (n_obs - 1) // 2)
            min_rows_mcd = max(4, 2 * max(k_mcd, 1) + 2)
            if k_mcd >= 1 and n_obs >= min_rows_mcd:
                from scipy.stats import chi2
                from sklearn.covariance import MinCovDet

                scores = u_svd[:, :k_mcd] * s_svd[:k_mcd]
                mcd_ok = False
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        mcd = MinCovDet(random_state=0).fit(scores)
                    cov = np.asarray(mcd.covariance_, dtype=np.float64)
                    if np.all(np.isfinite(cov)) and np.linalg.matrix_rank(cov) == k_mcd:
                        d2 = np.asarray(mcd.mahalanobis(scores), dtype=np.float64)
                        if np.all(np.isfinite(d2)):
                            crit = float(chi2.ppf(md_quantile, df=k_mcd))
                            md2[active] = d2
                            f_md[active] = d2 > crit
                            mcd_ok = True
                except (ValueError, np.linalg.LinAlgError):
                    mcd_ok = False
                if not mcd_ok:
                    notes.append(
                        f"{name}: robust Mahalanobis distance skipped because the MinCovDet subset covariance "
                        f"is singular on {k_mcd} principal component(s)."
                    )
            else:
                notes.append(
                    f"{name}: robust Mahalanobis distance skipped (need at least {min_rows_mcd} active rows "
                    f"for {max(k_mcd, 1)} PC(s); got {n_obs})."
                )
        else:
            notes.append(f"{name}: leverage and Mahalanobis checks skipped (zero positive singular values).")

        # Plain-sentence view warnings
        if f_scale.any():
            idx1 = np.flatnonzero(f_scale)
            notes.append(
                f"{name}: {idx1.size} gross row outlier(s) above robust_z={robust_z:g} on row scale "
                f"(rows {idx1.tolist()}; max robust_z={float(np.nanmax(row_rz[idx1])):.2f}, "
                f"max mean_abs_z={float(np.nanmax(mean_abs_z[idx1])):.2f})."
            )
        if f_lev.any() or f_md.any():
            notes.append(
                f"{name}: {int(f_lev.sum())} high-leverage row(s) (> {leverage_ratio:g}x average hat value) "
                f"and {int(f_md.sum())} multivariate-distance outlier(s) (> chi2({md_quantile:g}))."
            )
        if const_idx.size or near_const_idx.size:
            notes.append(
                f"{name}: {const_idx.size} constant column(s) and {near_const_idx.size} near-constant column(s)."
            )
        if single_ext_idx.size:
            notes.append(
                f"{name}: {single_ext_idx.size} column(s) with a single extreme value above robust_z={robust_z:g}."
            )
        if dup_pairs:
            notes.append(f"{name}: {len(dup_pairs)} duplicated column pair(s).")

    f_any = f_scale | f_lev | f_md
    return {
        "active": active,
        "n_obs": n_obs,
        "mean_abs_z": mean_abs_z,
        "robust_z": row_rz,
        "leverage_ratio": lev_ratio,
        "md2": md2,
        "flags": {
            "row_scale": f_scale,
            "leverage": f_lev,
            "md": f_md,
            "any": f_any,
        },
        "flagged_rows": {
            "row_scale": np.flatnonzero(f_scale).astype(np.int64),
            "leverage": np.flatnonzero(f_lev).astype(np.int64),
            "md": np.flatnonzero(f_md).astype(np.int64),
            "any": np.flatnonzero(f_any).astype(np.int64),
        },
        "columns": {
            "constant": const_idx,
            "near_constant": near_const_idx,
            "max_robust_z": max_col_rz,
            "single_extreme": single_ext_idx,
            "missing_fraction": miss_frac,
            "duplicated": dup_pairs,
        },
        "notes": notes,
    }


def _group_checks(
    groups: Any,
    n: int,
    fit: np.ndarray,
    names: List[str],
    form: str,
    view_checks: List[Dict[str, Any]],
    any_across: np.ndarray,
) -> Tuple[Dict[Any, int], Dict[str, Any], List[str]]:
    codes = group_codes(groups, n)
    if hasattr(groups, "to_numpy"):
        g_raw = np.asarray(groups.to_numpy())
    elif hasattr(groups, "detach"):
        g_raw = groups.detach().cpu().numpy()
    else:
        g_raw = np.asarray(groups)
    uniq_labels = np.unique(g_raw)
    labels = [lbl.item() if isinstance(lbl, np.generic) else lbl for lbl in uniq_labels]

    overall_counts: Dict[Any, int] = {}
    group_sizes: Dict[Any, int] = {}
    for code_i, lbl in enumerate(labels):
        in_g = fit & (codes == code_i)
        group_sizes[lbl] = int(in_g.sum())
        overall_counts[lbl] = int((any_across & in_g).sum())

    by_view_list: List[Dict[Any, int]] = []
    for vc in view_checks:
        f_v = vc["flags"]["any"]
        by_view_list.append(
            {lbl: int((f_v & fit & (codes == code_i)).sum()) for code_i, lbl in enumerate(labels)}
        )

    total_flagged = int(any_across.sum())
    n_fit = int(fit.sum())
    concentrated = False
    top_group: Any = None
    top_count = 0
    top_share = 0.0
    best_p = float("nan")
    notes: List[str] = []

    if total_flagged >= 2 and len(labels) >= 2 and n_fit >= 2:
        from scipy.stats import hypergeom

        best_rank = (-1, 2.0)
        for lbl in labels:
            k_g = overall_counts[lbl]
            n_g = group_sizes[lbl]
            if k_g <= 0 or n_g <= 0 or n_g >= n_fit:
                continue
            p_val = float(hypergeom.sf(k_g - 1, n_fit, n_g, total_flagged))
            if (k_g, -p_val) > (best_rank[0], -best_rank[1]):
                best_rank = (k_g, p_val)
                top_group = lbl
                top_count = k_g
                top_share = float(k_g) / float(total_flagged)
                best_p = p_val
        if top_group is not None and top_share >= GROUP_CONCENTRATION_SHARE and best_p < GROUP_CONCENTRATION_ALPHA:
            concentrated = True
            notes.append(
                f"groups: {top_count} of {total_flagged} flagged rows ({top_share:.0%}) fall in group "
                f"{top_group!r} (hypergeometric p={best_p:.3g}); concentrated flags in one group usually "
                f"indicate an acquisition or cohort difference rather than isolated outliers."
            )

    detail = {
        "by_view": _pack(names, by_view_list, form),
        "group_sizes": group_sizes,
        "concentrated": concentrated,
        "group": top_group if concentrated else None,
        "count": top_count if concentrated else 0,
        "total_flagged": total_flagged,
        "share": top_share if total_flagged > 0 else 0.0,
        "p_value": best_p,
    }
    return overall_counts, detail, notes


def _aligned_score_corr(u_loo: np.ndarray, u_ref: np.ndarray) -> float:
    """Minimum Pearson correlation across Procrustes-aligned consensus columns."""
    from .loadings import _procrustes

    ok = np.isfinite(u_loo).all(axis=1) & np.isfinite(u_ref).all(axis=1)
    if int(ok.sum()) < 3:
        return float("nan")
    a = u_loo[ok] - u_loo[ok].mean(axis=0, keepdims=True)
    b = u_ref[ok] - u_ref[ok].mean(axis=0, keepdims=True)
    if np.all(np.abs(a) <= SD_FLOOR) or np.all(np.abs(b) <= SD_FLOOR):
        return float("nan")
    rot = _procrustes(a, b)
    aligned = a @ rot
    corrs: List[float] = []
    for c in range(aligned.shape[1]):
        ac, bc = aligned[:, c], b[:, c]
        den = float(np.sqrt((ac * ac).sum() * (bc * bc).sum()))
        if den <= 0.0:
            return float("nan")
        corrs.append(float(np.clip((ac * bc).sum() / den, -1.0, 1.0)))
    return float(min(corrs))


def _loo_worker(task: Any) -> Tuple[float, bool, Optional[float]]:
    """Single leave-one-row-out refit worker (module-level for multiprocessing)."""
    from .api import fit
    from .loadings import _extract_learning_rate_used

    method, sub_views, k_int, params_i, u_ref_keep = task
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = fit(method, sub_views, k_int, safety="off", **params_i)
    u_i = res["u"].detach().cpu().double().numpy()
    conv = bool((res.get("convergence") or {}).get("converged", False))
    return _aligned_score_corr(u_i, u_ref_keep), conv, _extract_learning_rate_used(res)


def _slice_views_container(names: List[str], arrays: List[np.ndarray], form: str, idx: np.ndarray) -> Any:
    sliced = [a[idx] for a in arrays]
    return dict(zip(names, sliced)) if form == "mapping" else sliced


def _loo_influence(
    names: List[str],
    arrays: List[np.ndarray],
    form: str,
    fit: np.ndarray,
    *,
    method: str,
    k: int,
    n_jobs: int,
    tune_learning_rate: str,
    fit_params: Dict[str, Any],
) -> Tuple[np.ndarray, int, np.ndarray, float, List[Optional[float]], List[str]]:
    from .api import fit as run_fit, resolve_method
    from .confounds import slice_confounds
    from .loadings import _extract_learning_rate_used, _n_workers, _pool, _validate_tune_learning_rate

    _validate_tune_learning_rate(tune_learning_rate, method)
    n = arrays[0].shape[0]
    fit_idx = np.flatnonzero(fit)
    n_fit = int(fit_idx.size)
    if n_fit < 3:
        raise ValueError(f"influence=True requires at least 3 fit_rows; got {n_fit}.")
    k_int = int(k)
    if k_int < 1:
        raise ValueError(f"k must be >= 1; got {k!r}.")

    spec = resolve_method(method)
    base_params = dict(fit_params)
    has_missing_view_row = any((~np.isfinite(a[fit_idx]).any(axis=1)).any() for a in arrays)
    if has_missing_view_row and "missing_rows" not in base_params and "missing_rows" in spec.accepts:
        base_params["missing_rows"] = "mask"

    ref_params = dict(base_params)
    if ref_params.get("confounds") is not None:
        ref_params["confounds"] = slice_confounds(ref_params["confounds"], fit_idx)

    views_fit = _slice_views_container(names, arrays, form, fit_idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ref_res = run_fit(method, views_fit, k_int, safety="off", **ref_params)
    u_ref = ref_res["u"].detach().cpu().double().numpy()
    if tune_learning_rate == "reference":
        lr_ref = _extract_learning_rate_used(ref_res)
        if lr_ref is not None:
            base_params["learning_rate"] = lr_ref

    tasks = []
    for pos in range(n_fit):
        keep = np.arange(n_fit) != pos
        row_subset = fit_idx[keep]
        params_i = dict(base_params)
        if params_i.get("confounds") is not None:
            params_i["confounds"] = slice_confounds(params_i["confounds"], row_subset)
        sub_v = _slice_views_container(names, arrays, form, row_subset)
        tasks.append((method, sub_v, k_int, params_i, u_ref[keep]))

    workers = min(_n_workers(n_jobs), n_fit)
    if workers <= 1:
        outcomes = [_loo_worker(t) for t in tasks]
    else:
        with _pool(workers) as ex:
            outcomes = list(ex.map(_loo_worker, tasks))

    corrs = np.asarray([c for c, _, _ in outcomes], dtype=np.float64)
    convs = np.asarray([cv for _, cv, _ in outcomes], dtype=bool)
    refit_lrs: List[Optional[float]] = [lr for _, _, lr in outcomes]
    loo_corr_full = np.full(n, np.nan, dtype=np.float64)
    loo_corr_full[fit_idx] = corrs

    other_mask = np.where(np.isnan(corrs), False, corrs < SAME_SOLUTION_CORR)
    other_rows = fit_idx[other_mask].astype(np.int64)
    n_other = int(other_rows.size)
    conv_frac = float(convs.mean()) if convs.size else float("nan")

    notes: List[str] = []
    if n_other > 0:
        min_r = float(np.nanmin(corrs)) if np.isfinite(corrs).any() else float("nan")
        notes.append(
            f"influence: {n_other} of {n_fit} leave-one-out refits landed on a different solution "
            f"(score correlation < {SAME_SOLUTION_CORR:g} with the full fit; min r={min_r:.3f})."
        )
    return loo_corr_full, n_other, other_rows, conv_frac, refit_lrs, notes


def _data_safety_impl(
    views: Any,
    *,
    fit_rows: Any = None,
    groups: Any = None,
    n_pc: int = 5,
    robust_z: float = 5.0,
    leverage_ratio: float = 3.0,
    md_quantile: float = 0.999,
    influence: bool = False,
    method: str = "simlr",
    k: int = 1,
    n_jobs: int = 1,
    tune_learning_rate: str = "per_refit",
    fit_params: Optional[Dict[str, Any]] = None,
    pre_miss_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    pre_sd_list: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> Dict[str, Any]:
    from .loadings import TUNE_LEARNING_RATE_MODES, _validate_tune_learning_rate

    if not isinstance(tune_learning_rate, str) or tune_learning_rate not in TUNE_LEARNING_RATE_MODES:
        raise ValueError(
            f"tune_learning_rate must be one of {TUNE_LEARNING_RATE_MODES}; got {tune_learning_rate!r}."
        )
    if influence:
        _validate_tune_learning_rate(tune_learning_rate, method)
    n_pc_i, rz_f, lr_f, mq_f = _validate_thresholds(n_pc, robust_z, leverage_ratio, md_quantile)

    if _is_prepared_result(views):
        prep = views
        if fit_rows is None:
            fit_rows = prep["fit_rows"]
        vps = prep["params"]["views"]
        if pre_miss_list is None:
            pre_miss_list = [vp.get("missing_fraction") for vp in vps]
        if pre_sd_list is None:
            pre_sd_list = [vp.get("raw_sd") for vp in vps]
        views_in = prep["views"]
    else:
        views_in = views

    names, arrays, form = _split_views(views_in)
    n = arrays[0].shape[0]
    fit = _fit_mask(fit_rows, n)
    m = len(names)
    miss_l = list(pre_miss_list) if pre_miss_list is not None else [None] * m
    sd_l = list(pre_sd_list) if pre_sd_list is not None else [None] * m

    view_checks: List[Dict[str, Any]] = []
    all_warnings: List[str] = []
    for i, nm in enumerate(names):
        vc = _check_one_view(
            arrays[i],
            nm,
            fit,
            n_pc=n_pc_i,
            robust_z=rz_f,
            leverage_ratio=lr_f,
            md_quantile=mq_f,
            pre_miss=miss_l[i],
            pre_sd=sd_l[i],
        )
        view_checks.append(vc)
        all_warnings.extend(vc["notes"])

    any_scale = np.zeros(n, dtype=bool)
    any_lev = np.zeros(n, dtype=bool)
    any_md = np.zeros(n, dtype=bool)
    any_across = np.zeros(n, dtype=bool)
    for vc in view_checks:
        fl = vc["flags"]
        any_scale |= fl["row_scale"]
        any_lev |= fl["leverage"]
        any_md |= fl["md"]
        any_across |= fl["any"]

    if groups is not None:
        n_by_group, group_detail, group_notes = _group_checks(
            groups, n, fit, names, form, view_checks, any_across
        )
        all_warnings.extend(group_notes)
    else:
        n_by_group = None
        group_detail = None

    out: Dict[str, Any] = {
        "rules": {
            "n_pc": n_pc_i,
            "robust_z": rz_f,
            "leverage_ratio": lr_f,
            "md_quantile": mq_f,
            "same_solution_corr": SAME_SOLUTION_CORR,
        },
        "view_names": names,
        "fit_rows": fit,
        "active_rows": _pack(names, [vc["active"] for vc in view_checks], form),
        "n_obs": _pack(names, [vc["n_obs"] for vc in view_checks], form),
        "mean_abs_z": _pack(names, [vc["mean_abs_z"] for vc in view_checks], form),
        "robust_z": _pack(names, [vc["robust_z"] for vc in view_checks], form),
        "leverage_ratio": _pack(names, [vc["leverage_ratio"] for vc in view_checks], form),
        "md2": _pack(names, [vc["md2"] for vc in view_checks], form),
        "flags": _pack(names, [vc["flags"] for vc in view_checks], form),
        "flagged_rows": {
            "any": np.flatnonzero(any_across).astype(np.int64),
            "by_rule": {
                "row_scale": np.flatnonzero(any_scale).astype(np.int64),
                "leverage": np.flatnonzero(any_lev).astype(np.int64),
                "md": np.flatnonzero(any_md).astype(np.int64),
            },
            "by_view": _pack(names, [vc["flagged_rows"] for vc in view_checks], form),
        },
        "n_flagged_by_group": n_by_group,
        "group_concentration": group_detail,
        "columns": _pack(names, [vc["columns"] for vc in view_checks], form),
        "warnings": all_warnings,
    }

    if influence:
        loo_corr, n_other, other_rows, conv_frac, refit_lrs, loo_notes = _loo_influence(
            names,
            arrays,
            form,
            fit,
            method=method,
            k=k,
            n_jobs=n_jobs,
            tune_learning_rate=tune_learning_rate,
            fit_params=dict(fit_params or {}),
        )
        all_warnings.extend(loo_notes)
        out["loo_score_correlation"] = loo_corr
        out["n_other_solution"] = n_other
        out["other_solution_rows"] = other_rows
        out["loo_converged_fraction"] = conv_frac
        out["refit_learning_rates"] = refit_lrs

    return out


def apply_safety_policy(
    report: Dict[str, Any],
    safety: str,
    *,
    caller: str,
    stacklevel: int = 3,
) -> None:
    """Emit one `DataSafetyWarning` per view with a Rule 1 gross row outlier, or raise `ValueError`."""
    mode = validate_safety_mode(safety)
    if mode == "off":
        return
    rz_thresh = report["rules"]["robust_z"]
    names = report["view_names"]
    by_view = report["flagged_rows"]["by_view"]
    rz_all = report["robust_z"]
    maz_all = report["mean_abs_z"]
    is_map = isinstance(by_view, Mapping)

    messages: List[str] = []
    for i, nm in enumerate(names):
        key: Any = nm if is_map else i
        idx = by_view[key]["row_scale"]
        if idx.size == 0:
            continue
        rz_v = rz_all[key][idx]
        maz_v = maz_all[key][idx]
        messages.append(
            f"{caller}: view {nm!r} has {idx.size} gross row outlier(s) "
            f"(row scale robust_z > {rz_thresh:g}: rows {idx.tolist()}, "
            f"max robust_z={float(np.nanmax(rz_v)):.2f}, max mean_abs_z={float(np.nanmax(maz_v)):.2f}). "
            f"Data were not altered; inspect result['data_safety']."
        )
    if not messages:
        return
    if mode == "raise":
        raise ValueError("; ".join(messages))
    for msg in messages:
        warnings.warn(msg, DataSafetyWarning, stacklevel=stacklevel)


def data_safety(
    views: Any,
    *,
    fit_rows: Any = None,
    groups: Any = None,
    n_pc: int = 5,
    robust_z: float = 5.0,
    leverage_ratio: float = 3.0,
    md_quantile: float = 0.999,
    influence: bool = False,
    method: str = "simlr",
    k: int = 1,
    n_jobs: int = 1,
    tune_learning_rate: str = "per_refit",
    **fit_params: Any,
) -> Dict[str, Any]:
    """Standard row, column, group, and optional LOO sensitivity diagnostics for views.

    Evaluates each view on its observed ``fit_rows`` without dropping or altering
    any data. Accepts a matrix, list of matrices, mapping ``name -> matrix``, or
    the result dict returned by `prepare_views`.

    Parameters
    ----------
    views : matrix, list of matrices, mapping name -> matrix, or prepare_views dict
        ``(n, p_m)`` view matrices (typically after `prepare_views`), or the dict
        returned by `prepare_views`.
    fit_rows : None, boolean mask of length n, or 1-D integer indices
        Rows included in the diagnostic (default: all rows, or ``views["fit_rows"]``
        when ``views`` is a `prepare_views` dict). Rows missing an entire view
        (all NaN) are ignored for that view.
    groups : sequence of length n, optional
        Row group labels (e.g. site, scanner, cohort). Reports flag counts per
        group and checks whether flags concentrate in one group (>= 50% of
        flagged rows and one-sided hypergeometric ``p < 0.05``).
    n_pc : int, default=5
        Number of leading principal components for hat leverage and robust
        Mahalanobis distance.
    robust_z : float, default=5.0
        Upper-tail robust z threshold (median and ``1.4826 * MAD``) for Rule 1
        gross row scale outliers and Rule 4 single-extreme column checks.
    leverage_ratio : float, default=3.0
        Flag threshold on PC hat leverage divided by its mean ``k_pc / n_obs``.
    md_quantile : float, default=0.999
        Chi-square quantile for `sklearn.covariance.MinCovDet` squared Mahalanobis
        distances on the leading PC scores.
    influence : bool, default=False
        Run ``n_fit`` leave-one-row-out refits of ``pysimlr.fit(method, ..., k, **fit_params)``
        and report each row's Procrustes-aligned consensus score correlation with
        the full fit (`loo_score_correlation`) and the number of refits landing on
        a different solution (`n_other_solution`, informational count below ``0.95``).
    method : str, default="simlr"
        Estimator name passed to `pysimlr.fit` when ``influence=True``.
    k : int, default=1
        Shared latent rank passed to `pysimlr.fit` when ``influence=True``.
    n_jobs : int, default=1
        Worker processes for ``influence=True`` (``-1`` uses all CPU cores).
    **fit_params
        Forwarded to `pysimlr.fit` when ``influence=True`` (for example
        ``scale_list=["none"]`` on prepared views).

    Returns
    -------
    dict
        ``rules``, ``view_names``, ``fit_rows``, ``active_rows``, ``n_obs``,
        per-view ``mean_abs_z``, ``robust_z``, ``leverage_ratio``, ``md2`` and
        boolean ``flags`` (``row_scale``, ``leverage``, ``md``, ``any``),
        ``flagged_rows`` (``any``, ``by_rule``, ``by_view``),
        ``n_flagged_by_group`` (dict ``label -> count`` when ``groups`` is given,
        else ``None``), ``group_concentration``, ``columns``, ``warnings`` (list
        of plain sentences), and when ``influence=True``, ``loo_score_correlation``,
        ``n_other_solution``, ``other_solution_rows``, ``loo_converged_fraction``, and
        ``refit_learning_rates``.
    """
    return _data_safety_impl(
        views,
        fit_rows=fit_rows,
        groups=groups,
        n_pc=n_pc,
        robust_z=robust_z,
        leverage_ratio=leverage_ratio,
        md_quantile=md_quantile,
        influence=influence,
        method=method,
        k=k,
        n_jobs=n_jobs,
        tune_learning_rate=tune_learning_rate,
        fit_params=fit_params,
    )


__all__ = ["SAFETY_MODES", "apply_safety_policy", "data_safety", "validate_safety_mode"]
