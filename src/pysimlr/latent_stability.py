"""Calibrated out-of-bag latent stability for any registered PySiMLR method.

Computes out-of-bag intraclass correlation ``ICC(1)`` (``latent_reliability``),
``ICC(1, k0)`` (``latent_reliability_averaged``), rotation-invariant trace
``ICC(1)`` (``latent_reliability_multivariate``), out-of-bag leave-one-view-out
``cross_view_coupling``, and pairwise refit ``feature_agreement`` (plus
``feature_agreement_by_view``), each alongside an independent row-permutation
``_baseline`` and ``_gain``.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .positivity import NONNEG as _NONNEG
from .positivity import resolve_positivity
from .utils import group_codes, within_group_permutation

REMOVED_STABILITY_KEYS: Dict[str, str] = {
    "component_stability": "latent_reliability",
    "component_stability_null": "latent_reliability_baseline",
    "component_stability_corrected": "latent_reliability_gain",
    "subspace_stability": "latent_reliability / latent_reliability_multivariate",
    "subspace_stability_null": "latent_reliability_baseline / latent_reliability_multivariate_baseline",
    "subspace_stability_corrected": "latent_reliability_gain / latent_reliability_multivariate_gain",
    "subspace_stability_note": "latent_reliability (see latent_stability)",
    "weight_stability": "feature_agreement",
    "weight_stability_per_view": "feature_agreement_by_view",
    "oos_coupling": "cross_view_coupling",
    "oos_coupling_n": "cross_view_coupling_n",
    "oos_coupling_splits": "cross_view_coupling_splits",
}


def removed_key_message(key: str) -> str:
    """The migration message for a key removed in 0.41.0."""
    return f"{key!r} was removed in 0.41.0; use {REMOVED_STABILITY_KEYS[key]} (see latent_stability)"


class _RemovedKeysDict(dict):
    """Result dict that raises the migration message when a 0.41.0-removed key is read.

    ``res[key]`` and ``res.get(key)`` raise ``KeyError`` with the message for every
    key in ``REMOVED_STABILITY_KEYS`` (a plain ``dict.get`` would silently return
    ``None``). ``copy()``, ``copy.copy``, ``copy.deepcopy``, pickling and ``|`` keep
    the class. ``dict(res)`` and ``{**res}`` cannot return a subclass; with
    ``carry_table=True`` (the top-level results of ``latent_stability``,
    ``loading_inference`` and ``bagged_fit``) the dict therefore also holds a
    ``removed_keys`` entry, ``{old_name: message}``, which survives those
    conversions so a caller can still look the message up.
    """

    def __init__(self, *args: Any, carry_table: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if carry_table and "removed_keys" not in self:
            dict.__setitem__(self, "removed_keys", {k: removed_key_message(k) for k in REMOVED_STABILITY_KEYS})

    def __getitem__(self, key: Any) -> Any:
        if key in REMOVED_STABILITY_KEYS and not super().__contains__(key):
            raise KeyError(removed_key_message(key))
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if key in REMOVED_STABILITY_KEYS and not super().__contains__(key):
            raise KeyError(removed_key_message(key))
        return super().get(key, default)

    def copy(self) -> "_RemovedKeysDict":
        return type(self)(dict(self))

    __copy__ = copy

    def __deepcopy__(self, memo: Any) -> "_RemovedKeysDict":
        import copy as _copy
        return type(self)({k: _copy.deepcopy(v, memo) for k, v in dict.items(self)})

    def __reduce__(self) -> Any:
        return (type(self), (dict(self),))

    def __or__(self, other: Any) -> Any:
        if not isinstance(other, dict):
            return NotImplemented
        return type(self)({**dict(self), **dict(other)})

    def __ror__(self, other: Any) -> Any:
        if not isinstance(other, dict):
            return NotImplemented
        return type(self)({**dict(other), **dict(self)})


def _flat(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)) and all(x is None or isinstance(x, (bool, int, float, str)) for x in v):
        return list(v)
    if isinstance(v, np.ndarray) and v.ndim == 1 and v.size <= 1000:
        return v.tolist()
    return f"<{type(v).__name__}>"


def compute_oob_icc(y_oob: np.ndarray) -> Dict[str, Any]:
    """Unbalanced one-way random-effects ANOVA ``ICC(1)`` and ``ICC(1, k0)`` (never clamped at zero).

    Parameters
    ----------
    y_oob : np.ndarray
        Array of shape ``(B, n, k)`` containing out-of-bag scores and ``NaN``
        for in-bag / missing entries.
    """
    arr = np.asarray(y_oob, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"y_oob must be 3-D (B, n, k); got shape {arr.shape}.")
    _, _, k = arr.shape
    icc1 = np.full(k, np.nan, dtype=float)
    icc1k = np.full(k, np.nan, dtype=float)
    msb = np.full(k, np.nan, dtype=float)
    msw = np.full(k, np.nan, dtype=float)
    k0_out = float("nan")
    for j in range(k):
        yj = arr[:, :, j]
        fin = np.isfinite(yj)
        m = fin.sum(axis=0)
        mask_i = m >= 1
        n_t = int(mask_i.sum())
        m_t = m[mask_i].astype(float)
        total_m = float(m_t.sum())
        if n_t < 2 or (total_m - n_t) < 1.0:
            continue
        k0 = (total_m - float((m_t ** 2).sum()) / total_m) / (n_t - 1.0)
        if not np.isfinite(k0_out):
            k0_out = float(k0)
        y_sub = yj[:, mask_i]
        fin_sub = fin[:, mask_i]
        row_sum = np.where(fin_sub, y_sub, 0.0).sum(axis=0)
        y_bar_i = row_sum / m_t
        y_bar_all = float(row_sum.sum()) / total_m
        msb_j = float((m_t * ((y_bar_i - y_bar_all) ** 2)).sum()) / (n_t - 1.0)
        resid = np.where(fin_sub, y_sub - y_bar_i[None, :], 0.0)
        msw_j = float((resid ** 2).sum()) / (total_m - n_t)
        msb[j] = msb_j
        msw[j] = msw_j
        denom = msb_j + (k0 - 1.0) * msw_j
        if denom > 0.0:
            icc1[j] = float((msb_j - msw_j) / denom)
        if msb_j > 0.0:
            icc1k[j] = float((msb_j - msw_j) / msb_j)
    return {
        "latent_reliability": icc1,
        "latent_reliability_averaged": icc1k,
        "msb": msb,
        "msw": msw,
        "k0": k0_out,
    }


def compute_trace_icc(y_oob_mv: np.ndarray) -> float:
    """Rotation-invariant trace ``ICC(1)`` across all ``k`` components (never clamped at zero)."""
    arr = np.asarray(y_oob_mv, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"y_oob_mv must be 3-D (B, n, k); got shape {arr.shape}.")
    _, _, k = arr.shape
    fin_row = np.isfinite(arr).all(axis=2)
    m = fin_row.sum(axis=0)
    mask_i = m >= 1
    n_t = int(mask_i.sum())
    m_t = m[mask_i].astype(float)
    total_m = float(m_t.sum())
    if n_t < 2 or (total_m - n_t) < 1.0:
        return float("nan")
    k0 = (total_m - float((m_t ** 2).sum()) / total_m) / (n_t - 1.0)
    fin_sub = fin_row[:, mask_i]
    tr_msb = 0.0
    tr_msw = 0.0
    for j in range(k):
        y_sub = arr[:, mask_i, j]
        row_sum = np.where(fin_sub, y_sub, 0.0).sum(axis=0)
        y_bar_i = row_sum / m_t
        y_bar_all = float(row_sum.sum()) / total_m
        msb_j = float((m_t * ((y_bar_i - y_bar_all) ** 2)).sum()) / (n_t - 1.0)
        resid = np.where(fin_sub, y_sub - y_bar_i[None, :], 0.0)
        msw_j = float((resid ** 2).sum()) / (total_m - n_t)
        tr_msb += msb_j
        tr_msw += msw_j
    denom = tr_msb + (k0 - 1.0) * tr_msw
    if denom <= 0.0:
        return float("nan")
    return float((tr_msb - tr_msw) / denom)


def pairwise_cosine_agreement(mats: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine similarity across all ``B*(B-1)/2`` refit pairs per component."""
    arr = np.asarray(mats, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"mats must be 3-D (B, p, k); got shape {arr.shape}.")
    b_cnt, _, k = arr.shape
    out = np.full(k, np.nan, dtype=float)
    if b_cnt < 2:
        return out
    norms = np.linalg.norm(arr, axis=1)
    iu, ju = np.triu_indices(b_cnt, k=1)
    for j in range(k):
        nrm = norms[:, j]
        valid = np.isfinite(nrm) & (nrm > 1e-12) & np.isfinite(arr[:, :, j]).all(axis=1)
        if int(valid.sum()) < 2:
            continue
        unit = np.full((b_cnt, arr.shape[1]), np.nan, dtype=float)
        unit[valid] = arr[valid, :, j] / nrm[valid, None]
        gram = unit @ unit.T
        pairs = gram[iu, ju]
        fin_p = np.isfinite(pairs)
        if fin_p.any():
            out[j] = float(np.mean(pairs[fin_p]))
    return out


def _extract_nonneg_masks(res: Mapping[str, Any], n_views: int) -> List[Optional[np.ndarray]]:
    pvp = res.get("per_view_params") or ((res.get("parameters") or {}).get("used") or {}).get("per_view_params") or {}
    vnames = res.get("view_names") or []
    out: List[Optional[np.ndarray]] = []
    for m in range(n_views):
        nm_key = vnames[m] if m < len(vnames) else None
        ov = pvp.get(m) or pvp.get(f"view{m}") or (pvp.get(nm_key) if nm_key is not None else None) or {}
        nm = ov.get("nonneg_mask")
        out.append(None if nm is None else np.asarray(nm, dtype=bool))
    return out


def _inspect_constraint_regime(
    res: Mapping[str, Any],
    n_views: int,
) -> Tuple[bool, List[str], List[Optional[np.ndarray]]]:
    """Return ``(all_nonneg, pos_by_view, nonneg_masks)`` for a fitted result."""
    from .loadings import _result_view_settings

    pbv = (res.get("parameters") or {}).get("positivity_by_view") or res.get("positivity_by_view")
    if pbv is not None and len(pbv) == n_views and not any(v is None for v in pbv):
        pos_by_view = [resolve_positivity(v) for v in pbv]
    elif res.get("family") == "baseline" or res.get("basis_is_joint"):
        pos_by_view = ["either"] * n_views
    else:
        pos_by_view, _ = _result_view_settings(dict(res), n_views)
        pos_by_view = [resolve_positivity(v) for v in pos_by_view]
    nm_list = _extract_nonneg_masks(res, n_views)
    all_nonneg = (
        len(pos_by_view) == n_views
        and all(p in _NONNEG for p in pos_by_view)
        and not any(nm is not None and not bool(nm.all()) for nm in nm_list)
    )
    return all_nonneg, pos_by_view, nm_list


def _verify_nonneg_gauge(
    res: Mapping[str, Any],
    pos_by_view: Sequence[str],
    nm_list: Sequence[Optional[np.ndarray]],
) -> None:
    """Raise ``ValueError`` if any non-negative view weight matrix violates its constraint gauge."""
    if res.get("basis_is_joint") or not res.get("v"):
        return
    for m, v_t in enumerate(res["v"]):
        p_m = pos_by_view[m]
        if p_m not in _NONNEG:
            continue
        v_np = torch.as_tensor(v_t).detach().double().cpu().numpy()
        if not np.isfinite(v_np).all():
            raise ValueError(f"View {m} basis contains non-finite entries under positivity={p_m!r}.")
        nm_m = nm_list[m]
        if nm_m is not None:
            if nm_m.any() and float(v_np[nm_m].min()) < -1e-5:
                raise ValueError(
                    f"View {m} basis violates non-negative gauge on masked coordinates (min={v_np[nm_m].min():.3e})."
                )
        else:
            if float(v_np.min()) < -1e-5:
                raise ValueError(
                    f"View {m} basis violates non-negative gauge under positivity={p_m!r} (min={v_np.min():.3e})."
                )
            if p_m == "convex":
                for j in range(v_np.shape[1]):
                    col = v_np[:, j]
                    if float(np.linalg.norm(col)) > 1e-12 and abs(float(col.sum()) - 1.0) > 1e-3:
                        raise ValueError(
                            f"View {m} column {j} violates simplex gauge under positivity='convex' "
                            f"(sum={col.sum():.6g}, expected 1.0)."
                        )


def _replay_preprocessed_views(
    res: Mapping[str, Any],
    view_list: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Replay ``view_list`` through ``res``'s training preprocessing (``scale_list``, ``provenance_list``)."""
    from .api import _preprocess_views
    from .utils import preprocess_data

    scale_list = res.get("scale_list") or ["none"]
    prov_list = res.get("provenance_list") or []
    if res.get("missing_rows") == "mask":
        obs_masks = [~torch.isnan(v).all(dim=1) for v in view_list]
        xs = []
        for m_idx, (v, obs_m) in enumerate(zip(view_list, obs_masks)):
            v_obs = v[obs_m]
            if m_idx < len(prov_list):
                xs.append(preprocess_data(v_obs, scale_list, provenance=prov_list[m_idx]))
            else:
                xs.append(v_obs)
        return xs, obs_masks
    xs, _ = _preprocess_views(list(view_list), scale_list, prov_list)
    return xs, None


def _col_corr_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation between ``(n, k)`` arrays ``a`` and ``b``."""
    k = a.shape[1]
    out = np.zeros(k, dtype=float)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if int(ok.sum()) < 2:
        return out
    ac = a[ok] - a[ok].mean(axis=0, keepdims=True)
    bc = b[ok] - b[ok].mean(axis=0, keepdims=True)
    na = np.linalg.norm(ac, axis=0)
    nb = np.linalg.norm(bc, axis=0)
    den = na * nb
    mask = den > 1e-12
    if mask.any():
        out[mask] = (ac[:, mask] * bc[:, mask]).sum(axis=0) / den[mask]
    return out


def _consensus_sign_flips(
    res: Mapping[str, Any],
    train_views: Sequence[torch.Tensor],
    u_train: np.ndarray,
) -> np.ndarray:
    """Sign vector ``(k,)`` in ``{-1, +1}`` aligning ``u_train`` to non-negative view projections ``X_m V_m``."""
    k = int(u_train.shape[1])
    if res.get("basis_is_joint") or not res.get("v"):
        return np.ones(k, dtype=float)
    xs, obs_masks = _replay_preprocessed_views(res, train_views)
    corr_sum = np.zeros(k, dtype=float)
    for m, (x_m, v_t) in enumerate(zip(xs, res["v"])):
        v_np = torch.as_tensor(v_t).detach().double().cpu().numpy()
        z_m = x_m.detach().double().cpu().numpy() @ v_np
        if obs_masks is not None:
            obs_idx = obs_masks[m].detach().cpu().numpy()
            u_m = u_train[obs_idx]
        else:
            u_m = u_train
        corr_sum += _col_corr_pair(z_m, u_m)
    return np.where(corr_sum < 0.0, -1.0, 1.0)


def _standardize_by_train(
    u_train: np.ndarray,
    u_oob: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Standardize ``u_train`` and ``u_oob`` with ``u_train``'s own finite-row mean and SD."""
    fin = np.isfinite(u_train).all(axis=1)
    if int(fin.sum()) < 2:
        mu = np.zeros(u_train.shape[1], dtype=float)
        sd = np.ones(u_train.shape[1], dtype=float)
    else:
        mu = u_train[fin].mean(axis=0)
        sd = u_train[fin].std(axis=0, ddof=0)
    safe_sd = np.where(sd > 1e-12, sd, 1.0)
    z_train = np.where(sd[None, :] > 1e-12, (u_train - mu[None, :]) / safe_sd[None, :], 0.0)
    z_train = np.where(np.isfinite(u_train), z_train, np.nan)
    if u_oob is None:
        return z_train, None
    z_oob = np.where(sd[None, :] > 1e-12, (u_oob - mu[None, :]) / safe_sd[None, :], 0.0)
    z_oob = np.where(np.isfinite(u_oob), z_oob, np.nan)
    return z_train, z_oob


def _align_refit_to_ref(
    z_train_b_sh: np.ndarray,
    z_ref_sh: np.ndarray,
    all_nonneg: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cross-refit alignment ``(A_comp, A_mv)`` using shared training rows only."""
    from scipy.optimize import linear_sum_assignment
    from .loadings import _procrustes

    k = z_ref_sh.shape[1]
    ok = np.isfinite(z_train_b_sh).all(axis=1) & np.isfinite(z_ref_sh).all(axis=1)
    if int(ok.sum()) < max(2, k):
        eye = np.eye(k, dtype=float)
        return eye, eye
    a = z_train_b_sh[ok]
    b = z_ref_sh[ok]
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    if all_nonneg:
        if k == 1:
            eye = np.eye(1, dtype=float)
            return eye, eye
        na = np.linalg.norm(ac, axis=0, keepdims=True)
        nb = np.linalg.norm(bc, axis=0, keepdims=True)
        an = np.where(na > 1e-12, ac / np.where(na > 1e-12, na, 1.0), 0.0)
        bn = np.where(nb > 1e-12, bc / np.where(nb > 1e-12, nb, 1.0), 0.0)
        corr = np.nan_to_num(an.T @ bn, nan=0.0)
        r_ind, c_ind = linear_sum_assignment(-corr)
        perm = np.zeros((k, k), dtype=float)
        perm[r_ind, c_ind] = 1.0
        q_mv = _procrustes(ac, bc)
        return perm, q_mv
    q = _procrustes(ac, bc)
    return q, q


def _alignment_description(all_nonneg: bool, pos_by_view: Sequence[str], k: int, feature_space: str) -> Dict[str, Any]:
    if all_nonneg:
        per_comp = "identity" if k == 1 else "unsigned_column_permutation"
        mv_rule = "identity" if k == 1 else "train_procrustes"
        within_rule = "aligned_to_nonneg_view_projections_on_train"
    else:
        per_comp = "train_sign" if k == 1 else "train_procrustes"
        mv_rule = per_comp
        within_rule = "unconstrained"
    return {
        "regime": "nonnegative" if all_nonneg else "signed_or_mixed",
        "positivity_by_view": list(pos_by_view),
        "k": int(k),
        "within_fit_consensus_sign": within_rule,
        "per_component": per_comp,
        "multivariate": mv_rule,
        "alignment_rows": "shared_training_rows_only",
        "standardization": "per_refit_training_mean_and_sd",
        "feature_space": feature_space,
        "feature_pair_aggregation": "mean_over_all_refit_pairs",
    }


def _fmt_vec(arr: Optional[np.ndarray]) -> str:
    if arr is None:
        return "None"
    vals = np.asarray(arr, dtype=float).ravel()
    return "[" + ", ".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in vals) + "]"


def _fmt_gain(arr: Optional[np.ndarray]) -> str:
    if arr is None:
        return "None"
    vals = np.asarray(arr, dtype=float).ravel()
    return "[" + ", ".join(f"{v:+.3f}" if np.isfinite(v) else "nan" for v in vals) + "]"


def _build_stability_interpretation(
    metrics: Mapping[str, Any],
    *,
    baseline: bool,
    select_mode: bool = False,
) -> str:
    """Two plain factual sentences stating stability values, baselines, and gains (no verdict thresholds)."""
    lr = metrics["latent_reliability"]
    mv = metrics["latent_reliability_multivariate"]
    cv = metrics["cross_view_coupling"]
    fa = metrics["feature_agreement"]
    fspace = metrics["alignment"]["feature_space"]
    n_ref = metrics["n_refits_used"]
    oob = metrics["n_oob_per_row"]
    if baseline and metrics.get("latent_reliability_baseline") is not None:
        lr_b = metrics["latent_reliability_baseline"]
        lr_g = metrics["latent_reliability_gain"]
        mv_b = metrics["latent_reliability_multivariate_baseline"]
        mv_g = metrics["latent_reliability_multivariate_gain"]
        cv_part = (
            f"out-of-bag cross-view coupling is {_fmt_vec(cv)} "
            f"(baseline {_fmt_vec(metrics['cross_view_coupling_baseline'])}, "
            f"gain {_fmt_gain(metrics['cross_view_coupling_gain'])})"
            if cv is not None
            else "cross-view coupling is unavailable (fewer than 2 views or 3 OOB rows)"
        )
        s1 = (
            f"Out-of-bag latent reliability is {_fmt_vec(lr)} "
            f"(baseline {_fmt_vec(lr_b)}, gain {_fmt_gain(lr_g)}; "
            f"multivariate {mv:.3f}, baseline {mv_b:.3f}, gain {mv_g:+.3f}), "
            f"and {cv_part}; read the reliability gain together with the cross-view coupling gain, "
            "because a dominant direction private to one view is reproducible without being shared."
        )
        s2_tail = (
            f"Pairwise feature agreement in {fspace} space is {_fmt_vec(fa)} "
            f"(baseline {_fmt_vec(metrics['feature_agreement_baseline'])}, "
            f"gain {_fmt_gain(metrics['feature_agreement_gain'])}) "
            f"across {n_ref} refits (median {oob['median']:.1f}, min {oob['min']} OOB evaluations per row)"
        )
    else:
        cv_part = (
            f"out-of-bag cross-view coupling is {_fmt_vec(cv)}"
            if cv is not None
            else "cross-view coupling is unavailable (fewer than 2 views or 3 OOB rows)"
        )
        s1 = (
            f"Out-of-bag latent reliability is {_fmt_vec(lr)} (multivariate {mv:.3f}), and {cv_part}; "
            "reliability must be read together with cross-view coupling, because a dominant direction "
            "private to one view is reproducible without being shared."
        )
        no_base = "no permutation baseline was run (baseline=False)"
        s2_tail = (
            f"Pairwise feature agreement in {fspace} space is {_fmt_vec(fa)} across {n_ref} refits "
            f"(median {oob['median']:.1f}, min {oob['min']} OOB evaluations per row; {no_base})"
        )
    if select_mode:
        s2 = (
            f"{s2_tail}; these diagnostics describe the pre-selection all-feature refits "
            "rather than nested select-then-fit reliability of result['v']."
        )
    else:
        s2 = f"{s2_tail}."
    return f"{s1} {s2}"


def _compute_stability_from_refits(
    ref: Mapping[str, Any],
    train_views: Sequence[torch.Tensor],
    sub_idx: Sequence[np.ndarray],
    sub_records: Sequence[Mapping[str, Any]],
    *,
    oob_rows: Sequence[np.ndarray],
    exclude_unconverged: bool = False,
) -> Dict[str, Any]:
    """Compute all Part A stability metrics for one family of refits."""
    from scipy.optimize import linear_sum_assignment
    from .loadings import _nanmedian_cols

    n_views = len(train_views)
    n = int(train_views[0].shape[0])
    c_ref = [np.asarray(c, dtype=float) for c in ref["structure"]]
    stacked_ref = np.vstack(c_ref)
    k = int(stacked_ref.shape[1])

    if "u" in ref:
        all_nonneg, pos_by_view, nm_list = _inspect_constraint_regime(ref, n_views)
        _verify_nonneg_gauge(ref, pos_by_view, nm_list)
        u_ref_raw = torch.as_tensor(ref["u"]).detach().double().cpu().numpy()
        if all_nonneg:
            flips_ref = _consensus_sign_flips(ref, train_views, u_ref_raw)
            u_ref_raw = u_ref_raw * flips_ref[None, :]
        use_weights = (ref.get("family") == "linear") and (not ref.get("basis_is_joint")) and bool(ref.get("v"))
    else:
        all_nonneg = bool(ref["all_nonneg"])
        pos_by_view = list(ref["pos_by_view"])
        if all_nonneg and ref.get("nonneg_gauge_error") is not None:
            raise ValueError(str(ref["nonneg_gauge_error"]))
        u_ref_raw = np.asarray(ref["u_train"], dtype=float)
        if all_nonneg:
            flips_ref = np.asarray(ref["consensus_flips"], dtype=float)
            u_ref_raw = u_ref_raw * flips_ref[None, :]
        use_weights = bool(ref["use_weights"])
    z_ref, _ = _standardize_by_train(u_ref_raw, None)
    feature_space = "weights" if use_weights else "structure"

    conv_flags = [bool(rec["converged"]) for rec in sub_records]
    conv_frac = float(np.mean(conv_flags)) if conv_flags else 1.0

    y_comp_list: List[np.ndarray] = []
    y_mv_list: List[np.ndarray] = []
    feat_by_view_aligned: List[List[np.ndarray]] = [[] for _ in range(n_views)]
    feat_stacked_aligned: List[np.ndarray] = []
    coups: List[np.ndarray] = []
    coup_splits_all: List[Optional[List[float]]] = []
    coup_n: List[List[int]] = []
    oob_counts = np.zeros(n, dtype=int)

    for b_i, (idx_b, oob_b, rec) in enumerate(zip(sub_idx, oob_rows, sub_records)):
        keep_rec = (not exclude_unconverged) or bool(rec["converged"])
        st = np.vstack(rec["structure"])
        if keep_rec and rec.get("cross_view_coupling") is not None:
            _, col_ind = linear_sum_assignment(-np.abs(st.T @ stacked_ref))
            inv = np.empty_like(col_ind)
            inv[col_ind] = np.arange(k)
            c_aligned = np.asarray(rec["cross_view_coupling"], dtype=float)[inv]
            coups.append(c_aligned)
            coup_splits_all.append([float(x) for x in c_aligned])
            if rec.get("cross_view_coupling_n") is not None:
                coup_n.append(list(rec["cross_view_coupling_n"]))
        else:
            coup_splits_all.append(None)

        if not keep_rec:
            continue

        if all_nonneg and rec.get("nonneg_gauge_error") is not None:
            raise ValueError(str(rec["nonneg_gauge_error"]))

        u_train_b = np.asarray(rec["u_train"], dtype=float)
        u_oob_b = None if rec.get("u_oob") is None else np.asarray(rec["u_oob"], dtype=float)
        flips_b = np.asarray(rec.get("consensus_flips", np.ones(k)), dtype=float) if all_nonneg else np.ones(k, dtype=float)
        if all_nonneg:
            u_train_b = u_train_b * flips_b[None, :]
            if u_oob_b is not None:
                u_oob_b = u_oob_b * flips_b[None, :]

        z_train_b, z_oob_b = _standardize_by_train(u_train_b, u_oob_b)
        first_rows, first_pos = np.unique(idx_b, return_index=True)
        a_comp, a_mv = _align_refit_to_ref(z_train_b[first_pos], z_ref[first_rows], all_nonneg)

        y_comp_b = np.full((n, k), np.nan, dtype=float)
        y_mv_b = np.full((n, k), np.nan, dtype=float)
        if z_oob_b is not None and len(oob_b) > 0:
            y_comp_b[oob_b] = z_oob_b @ a_comp
            y_mv_b[oob_b] = z_oob_b @ a_mv
            fin_rows = np.isfinite(y_comp_b[oob_b]).all(axis=1)
            oob_counts[oob_b[fin_rows]] += 1
        y_comp_list.append(y_comp_b)
        y_mv_list.append(y_mv_b)

        if use_weights and rec.get("weights_raw") is not None:
            raw_mats = rec["weights_raw"]
        else:
            raw_mats = rec["structure_u"]
            if all_nonneg:
                raw_mats = [m_arr * flips_b[None, :] for m_arr in raw_mats]
        al_views = [np.asarray(m_arr, dtype=float) @ a_comp for m_arr in raw_mats]
        for vi, mat_v in enumerate(al_views):
            feat_by_view_aligned[vi].append(mat_v)
        feat_stacked_aligned.append(np.vstack(al_views))

    n_refits_used = len(y_comp_list)
    if n_refits_used > 0:
        y_comp = np.stack(y_comp_list, axis=0)
        y_mv = np.stack(y_mv_list, axis=0)
        icc_res = compute_oob_icc(y_comp)
        lat_rel = icc_res["latent_reliability"]
        lat_rel_avg = icc_res["latent_reliability_averaged"]
        lat_rel_mv = float(lat_rel[0]) if k == 1 else compute_trace_icc(y_mv)
        feat_agree = pairwise_cosine_agreement(np.stack(feat_stacked_aligned, axis=0))
        feat_agree_views = [pairwise_cosine_agreement(np.stack(fv, axis=0)) for fv in feat_by_view_aligned]
        n_oob_per_row = {"median": float(np.median(oob_counts)), "min": int(np.min(oob_counts))}
    else:
        lat_rel = np.full(k, np.nan, dtype=float)
        lat_rel_avg = np.full(k, np.nan, dtype=float)
        lat_rel_mv = float("nan")
        feat_agree = np.full(k, np.nan, dtype=float)
        feat_agree_views = [np.full(int(k), np.nan, dtype=float) for _ in range(n_views)]
        n_oob_per_row = {"median": 0.0, "min": 0}

    cv_coup = np.asarray(_nanmedian_cols(coups), dtype=float) if coups else None
    cv_coup_n = [float(x) for x in np.median(coup_n, axis=0)] if coup_n else None
    cv_coup_splits = coup_splits_all if coups else None

    return {
        "latent_reliability": lat_rel,
        "latent_reliability_averaged": lat_rel_avg,
        "latent_reliability_multivariate": lat_rel_mv,
        "cross_view_coupling": cv_coup,
        "cross_view_coupling_n": cv_coup_n,
        "cross_view_coupling_splits": cv_coup_splits,
        "feature_agreement": feat_agree,
        "feature_agreement_by_view": feat_agree_views,
        "n_oob_per_row": n_oob_per_row,
        "n_refits_used": int(n_refits_used),
        "converged_fraction": conv_frac,
        "alignment": _alignment_description(all_nonneg, pos_by_view, k, feature_space),
    }


def _average_baseline_metrics(base_list: Sequence[Mapping[str, Any]], k: int, n_views: int) -> Dict[str, Any]:
    """Average stability metrics over ``n_baseline`` independent permutation draws."""
    def _mean_arr(key: str) -> Optional[np.ndarray]:
        vals = [np.asarray(b[key], dtype=float) for b in base_list if b.get(key) is not None]
        if not vals:
            return None
        stk = np.stack(vals, axis=0)
        out = np.full(stk.shape[1:], np.nan, dtype=float)
        fin = np.isfinite(stk)
        any_f = fin.any(axis=0)
        if any_f.any():
            cnt = np.maximum(fin.sum(axis=0), 1)
            out[any_f] = (np.where(fin, stk, 0.0).sum(axis=0) / cnt)[any_f]
        return out

    lr_b = _mean_arr("latent_reliability")
    lra_b = _mean_arr("latent_reliability_averaged")
    mv_vals = [float(b["latent_reliability_multivariate"]) for b in base_list
               if np.isfinite(b["latent_reliability_multivariate"])]
    mv_b = float(np.mean(mv_vals)) if mv_vals else float("nan")
    cv_b = _mean_arr("cross_view_coupling")
    fa_b = _mean_arr("feature_agreement")
    fa_v_b: List[np.ndarray] = []
    for vi in range(n_views):
        vals_v = [np.asarray(b["feature_agreement_by_view"][vi], dtype=float) for b in base_list]
        stk_v = np.stack(vals_v, axis=0)
        out_v = np.full(k, np.nan, dtype=float)
        fin_v = np.isfinite(stk_v)
        any_v = fin_v.any(axis=0)
        if any_v.any():
            out_v[any_v] = (np.where(fin_v, stk_v, 0.0).sum(axis=0) / np.maximum(fin_v.sum(axis=0), 1))[any_v]
        fa_v_b.append(out_v)
    return {
        "latent_reliability_baseline": lr_b,
        "latent_reliability_averaged_baseline": lra_b,
        "latent_reliability_multivariate_baseline": mv_b,
        "cross_view_coupling_baseline": cv_b,
        "feature_agreement_baseline": fa_b,
        "feature_agreement_by_view_baseline": fa_v_b,
    }


def _attach_baseline_and_gains(
    real: Dict[str, Any],
    base_list: Optional[Sequence[Mapping[str, Any]]],
    k: int,
    n_views: int,
    *,
    select_mode: bool = False,
) -> Dict[str, Any]:
    """Combine real metrics and optional baseline draws into the public stability dict."""
    if not base_list:
        out = dict(real)
        out.update({
            "latent_reliability_baseline": None,
            "latent_reliability_gain": None,
            "latent_reliability_averaged_baseline": None,
            "latent_reliability_averaged_gain": None,
            "latent_reliability_multivariate_baseline": None,
            "latent_reliability_multivariate_gain": None,
            "cross_view_coupling_baseline": None,
            "cross_view_coupling_gain": None,
            "feature_agreement_baseline": None,
            "feature_agreement_gain": None,
            "feature_agreement_by_view_baseline": None,
            "feature_agreement_by_view_gain": None,
        })
        out["interpretation"] = _build_stability_interpretation(out, baseline=False, select_mode=select_mode)
        return out

    avg = _average_baseline_metrics(base_list, k, n_views)
    out = dict(real)
    out.update(avg)
    out["latent_reliability_gain"] = (
        real["latent_reliability"] - avg["latent_reliability_baseline"]
        if avg["latent_reliability_baseline"] is not None else None
    )
    out["latent_reliability_averaged_gain"] = (
        real["latent_reliability_averaged"] - avg["latent_reliability_averaged_baseline"]
        if avg["latent_reliability_averaged_baseline"] is not None else None
    )
    out["latent_reliability_multivariate_gain"] = (
        float(real["latent_reliability_multivariate"] - avg["latent_reliability_multivariate_baseline"])
        if avg["latent_reliability_multivariate_baseline"] is not None else None
    )
    out["cross_view_coupling_gain"] = (
        real["cross_view_coupling"] - avg["cross_view_coupling_baseline"]
        if (real["cross_view_coupling"] is not None and avg["cross_view_coupling_baseline"] is not None) else None
    )
    out["feature_agreement_gain"] = (
        real["feature_agreement"] - avg["feature_agreement_baseline"]
        if avg["feature_agreement_baseline"] is not None else None
    )
    out["feature_agreement_by_view_gain"] = (
        [a - b for a, b in zip(real["feature_agreement_by_view"], avg["feature_agreement_by_view_baseline"])]
        if avg["feature_agreement_by_view_baseline"] is not None else None
    )
    out["interpretation"] = _build_stability_interpretation(out, baseline=True, select_mode=select_mode)
    return out


def _normalize_fit_params_for_refits(
    views: Any,
    view_list: Sequence[torch.Tensor],
    ref: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    out = dict(params)
    if isinstance(views, Mapping):
        from .api import _resolve_fit_view_idx

        vnames_in = [str(nm) for nm in views.keys()]
        used_rec = (ref.get("parameters") or {}).get("used") or {}
        for nk in ("per_view_params", "positivity", "path_graph", "system_matrices"):
            if nk in out and nk in used_rec:
                out[nk] = used_rec[nk]
        if isinstance(out.get("confounds"), Mapping):
            out["confounds"] = {
                _resolve_fit_view_idx(kv, len(view_list), vnames_in): vv
                for kv, vv in out["confounds"].items()
            }
    return out


def _build_subsample_tasks(
    method: str,
    view_list: Sequence[torch.Tensor],
    k: int,
    random_state: int,
    refit_params: Mapping[str, Any],
    sub_idx: Sequence[np.ndarray],
    *,
    g_codes: Optional[np.ndarray] = None,
    scale_within_group: bool = False,
    map_opts: Optional[Dict[str, Any]] = None,
    probe_tensors: Optional[Sequence[Optional[torch.Tensor]]] = None,
    want: Tuple[str, ...] = (),
) -> Tuple[List[Any], List[np.ndarray]]:
    """Build `_fit_structure` tasks and OOB index arrays for one family of subsample refits."""
    from .confounds import slice_confounds

    n = int(view_list[0].shape[0])
    all_rows = np.arange(n)
    conf = refit_params.get("confounds")
    tasks, oob_rows = [], []
    use_extra = g_codes is not None or bool(want)
    for idx in sub_idx:
        oob = np.setdiff1d(all_rows, np.unique(idx), assume_unique=True)
        oob_rows.append(oob)
        t_idx = torch.as_tensor(idx)
        t_oob = torch.as_tensor(oob) if len(oob) >= 2 else None  # one row has no variance to score on
        sub_v = [x[t_idx].numpy() if x.device.type == "cpu" else x[t_idx] for x in view_list]
        oob_v = (
            [x[t_oob].numpy() if x.device.type == "cpu" else x[t_oob] for x in view_list]
            if t_oob is not None else None
        )
        sub_p = dict(refit_params) if conf is None else dict(refit_params, confounds=slice_confounds(conf, idx))
        task: Tuple[Any, ...] = (method, sub_v, oob_v, k, random_state, sub_p)
        if map_opts is not None:
            t_opts = (
                dict(map_opts, probe_views=[None if pt is None else pt[t_idx] for pt in probe_tensors])
                if probe_tensors is not None else map_opts
            )
            if g_codes is not None:
                t_opts = dict(t_opts, groups=g_codes[idx])
            task = task + (t_opts,)
        if use_extra:
            extra = {
                "groups": None if g_codes is None else g_codes[idx],
                "oob_groups": None if g_codes is None or len(oob) < 2 else g_codes[oob],
                "scale_within_group": bool(scale_within_group),
                "want": want,
            }
            task = task + ((None,) if len(task) == 6 else ()) + (extra,)
        tasks.append(task)
    return tasks, oob_rows


def _baseline_rngs(
    draw_seed: Any,
    parent_rng: np.random.Generator,
    n_baseline: int,
) -> List[np.random.Generator]:
    if isinstance(draw_seed, (int, np.integer)) and not isinstance(draw_seed, bool):
        return [
            np.random.default_rng(
                np.random.SeedSequence([abs(int(draw_seed)), 0 if int(draw_seed) >= 0 else 1, 0x5A17, d])
            )
            for d in range(n_baseline)
        ]
    return list(parent_rng.spawn(n_baseline))


def _prepare_baseline_draw(
    b_rng: np.random.Generator,
    view_list: Sequence[torch.Tensor],
    base_params: Mapping[str, Any],
    m: int,
    scheme: str,
    s_codes: Optional[np.ndarray],
    g_codes: Optional[np.ndarray],
    n_subsamples: int,
) -> Tuple[List[torch.Tensor], Dict[str, Any], List[np.ndarray]]:
    """Permute views ``1..M-1`` independently once for one baseline draw and draw its subsamples.

    View 0 stays in place on purpose (see the request, section 2.1): a dominant
    direction inside one view keeps its true reproducibility in the baseline.
    Rotating which view stays fixed was measured on all-private data (after 0.42.0)
    and did not reduce the per-draw spread of ``latent_reliability_gain``
    (sd 0.159 -> 0.161 linear, 0.212 -> 0.201 conical, 0.239 -> 0.204 convex;
    20 seeds, n=60, B=20), so it was not adopted.
    """
    from .confounds import permute_view_confounds
    from .loadings import _draw_subsample

    n = int(view_list[0].shape[0])
    if g_codes is None:
        perms = [b_rng.permutation(n) for _ in view_list[1:]]
    else:
        perms = [within_group_permutation(b_rng, g_codes) for _ in view_list[1:]]
    b_views: List[torch.Tensor] = [view_list[0]] + [
        v[torch.as_tensor(pi)] for v, pi in zip(view_list[1:], perms)
    ]
    conf = base_params.get("confounds")
    b_params = dict(base_params)
    if conf is not None:
        b_params["confounds"] = permute_view_confounds(conf, [None] + list(perms))
    b_sub_idx = [_draw_subsample(b_rng, n, m, scheme, s_codes) for _ in range(int(n_subsamples))]
    return b_views, b_params, b_sub_idx


def _run_baseline_families(
    method: str,
    view_list: Sequence[torch.Tensor],
    k: int,
    base_params: Mapping[str, Any],
    *,
    n_baseline: int,
    n_subsamples: int,
    m: int,
    scheme: str,
    s_codes: Optional[np.ndarray],
    g_codes: Optional[np.ndarray],
    scale_within_group: bool,
    random_state: int,
    draw_seed: Any,
    parent_rng: np.random.Generator,
    tune_learning_rate: str,
    n_jobs: int,
    executor: Any,
    exclude_unconverged: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Optional[float]]]:
    """Fit ``n_baseline`` independent row-permuted baseline families and compute their stability metrics.

    Each baseline draw fits its own permuted reference fit (tuning its own learning rate when
    ``tune_learning_rate='reference'``, never reusing the unshuffled real-data learning rate).
    """
    from .api import fit
    from .loadings import _extract_learning_rate_used, _run_fits

    b_rngs = _baseline_rngs(draw_seed, parent_rng, n_baseline)
    draws = [
        _prepare_baseline_draw(br, view_list, base_params, m, scheme, s_codes, g_codes, n_subsamples)
        for br in b_rngs
    ]

    # Fit each baseline draw's own reference fit on its permuted dataset.
    b_refs: List[Dict[str, Any]] = []
    b_ref_lrs: List[Optional[float]] = []
    all_sub_tasks: List[Any] = []
    draw_oobs: List[List[np.ndarray]] = []
    draw_slices: List[Tuple[int, int]] = []

    for b_views, b_params, b_sub_idx in draws:
        b_ref = fit(method, b_views, k, random_state=random_state, **b_params)
        b_refs.append(b_ref)
        b_sub_params = dict(b_params)
        lr_b = _extract_learning_rate_used(b_ref)
        b_ref_lrs.append(lr_b)
        if tune_learning_rate == "reference" and lr_b is not None:
            b_sub_params["learning_rate"] = lr_b
        t_b, oob_b = _build_subsample_tasks(
            method, b_views, k, random_state, b_sub_params, b_sub_idx,
            g_codes=g_codes, scale_within_group=scale_within_group,
        )
        start = len(all_sub_tasks)
        all_sub_tasks.extend(t_b)
        draw_slices.append((start, len(all_sub_tasks)))
        draw_oobs.append(oob_b)

    all_recs = _run_fits(all_sub_tasks, n_jobs, executor) if all_sub_tasks else []
    base_metrics: List[Dict[str, Any]] = []
    for (b_views, _, b_sub_idx), b_ref, (s0, s1), oob_b in zip(draws, b_refs, draw_slices, draw_oobs):
        bm = _compute_stability_from_refits(
            b_ref, b_views, b_sub_idx, all_recs[s0:s1],
            oob_rows=oob_b, exclude_unconverged=exclude_unconverged,
        )
        base_metrics.append(bm)
    return base_metrics, b_ref_lrs


def latent_stability(
    method: str,
    views: Any,
    k: int,
    *,
    n_subsamples: int = 50,
    fraction: float = 0.8,
    strata: Any = None,
    baseline: bool = True,
    n_baseline: int = 3,
    random_state: int = 0,
    draw_random_state: Optional[int] = None,
    n_jobs: int = 1,
    groups: Any = None,
    tune_learning_rate: str = "per_refit",
    _executor: Any = None,
    **fit_params: Any,
) -> Dict[str, Any]:
    """Calibrated out-of-bag latent stability report for any registered method.

    Refits ``method`` on ``n_subsamples`` random ``fraction``-subsamples drawn
    without replacement, standardizes each refit's out-of-bag scores using that
    refit's own training mean and SD, aligns refit latents on shared training
    rows only (respecting non-negative vs. signed/mixed weight constraints), and
    reports unbalanced one-way ANOVA ``ICC(1)`` (``latent_reliability``),
    ``ICC(1, k0)`` (``latent_reliability_averaged``), rotation-invariant trace
    ``ICC(1)`` (``latent_reliability_multivariate``), out-of-bag
    ``cross_view_coupling``, and pairwise refit ``feature_agreement``.
    When ``baseline=True`` and ``len(views) >= 2``, compares against the mean
    over ``n_baseline`` independent row-permutation draws (each permuted once
    before its subsamples and tuned on its own permuted reference fit).
    """
    import pysimlr
    from .api import as_views, fit, resolve_method
    from .loadings import (
        _draw_subsample,
        _extract_learning_rate_used,
        _n_workers,
        _pool,
        _run_fits,
        _validate_tune_learning_rate,
    )

    resolve_method(method)
    _validate_tune_learning_rate(tune_learning_rate, method)
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction}.")
    if int(n_subsamples) < 1:
        raise ValueError(f"n_subsamples must be >= 1; got {n_subsamples}.")
    if baseline and int(n_baseline) < 1:
        raise ValueError(f"n_baseline must be >= 1 when baseline=True; got {n_baseline}.")

    view_list = as_views(views)
    n_views = len(view_list)
    n = int(view_list[0].shape[0])
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")

    g_codes = None if groups is None else group_codes(groups, n)
    s_codes = None if strata is None else group_codes(strata, n, name="strata")
    draw_seed = random_state if draw_random_state is None else draw_random_state
    rng = np.random.default_rng(draw_seed)

    m = max(k + 2, int(round(float(fraction) * n)))
    if m >= n:
        raise ValueError(f"subsample size m={m} must be smaller than n={n} to leave out-of-bag rows.")

    ref = fit(method, views, k, random_state=random_state, **fit_params)
    base_params = _normalize_fit_params_for_refits(views, view_list, ref, fit_params)
    real_refit_params = dict(base_params)
    if tune_learning_rate == "reference":
        lr_ref = _extract_learning_rate_used(ref)
        if lr_ref is not None:
            real_refit_params["learning_rate"] = lr_ref

    sub_idx = [_draw_subsample(rng, n, m, "fraction", s_codes) for _ in range(int(n_subsamples))]
    real_tasks, real_oob = _build_subsample_tasks(
        method, view_list, k, random_state, real_refit_params, sub_idx,
        g_codes=g_codes,
    )

    workers = _n_workers(n_jobs)
    own_pool = (_executor is None) and (workers > 1)
    with (_pool(workers) if own_pool else contextlib.nullcontext()) as ex:
        pool_ex = ex if own_pool else _executor
        real_records = _run_fits(real_tasks, n_jobs, pool_ex)
        real_metrics = _compute_stability_from_refits(
            ref, view_list, sub_idx, real_records, oob_rows=real_oob,
        )

        base_list: Optional[List[Dict[str, Any]]] = None
        base_ref_lrs: Optional[List[Optional[float]]] = None
        if baseline and n_views >= 2:
            base_list, base_ref_lrs = _run_baseline_families(
                method, view_list, k, base_params,
                n_baseline=int(n_baseline), n_subsamples=int(n_subsamples),
                m=m, scheme="fraction", s_codes=s_codes, g_codes=g_codes,
                scale_within_group=False, random_state=random_state,
                draw_seed=draw_seed, parent_rng=rng,
                tune_learning_rate=tune_learning_rate,
                n_jobs=n_jobs, executor=pool_ex,
            )

    out = _attach_baseline_and_gains(real_metrics, base_list, k, n_views, select_mode=False)
    refit_lrs = [rec.get("learning_rate_used") for rec in real_records]
    out["refit_learning_rates"] = refit_lrs
    out["provenance"] = {
        "method": method,
        "k": int(k),
        "n_subsamples": int(n_subsamples),
        "fraction": float(fraction),
        "subsample_size": int(m),
        "subsample": "without replacement",
        "strata": _flat(strata),
        "groups": _flat(groups),
        "baseline": bool(baseline),
        "n_baseline": int(n_baseline) if (baseline and n_views >= 2) else 0,
        "random_state": random_state,
        "draw_random_state": draw_random_state,
        "n_jobs": int(n_jobs),
        "tune_learning_rate": tune_learning_rate,
        "baseline_learning_rates": base_ref_lrs,
        "fit_params": {kk: _flat(vv) for kk, vv in fit_params.items()},
        "n_rows": n,
        "view_widths": [int(x.shape[1]) for x in view_list],
        "pysimlr_version": pysimlr.__version__,
    }
    return _RemovedKeysDict(out, carry_table=True)


__all__ = [
    "REMOVED_STABILITY_KEYS",
    "_RemovedKeysDict",
    "compute_oob_icc",
    "compute_trace_icc",
    "pairwise_cosine_agreement",
    "latent_stability",
]
