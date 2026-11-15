"""Confound sink: one feature-space nuisance cleaner shared by every method.

`pysimlr.fit(method, views, k, confounds=C)` removes the part of each view
that known nuisance covariates ``C`` (site, scanner, motion, head size,
survey design, ...) explain, *before* the method sees the data, with an
operator that needs no confound values at prediction time. Because the
operator is part of each view's replayed preprocessing (its provenance), the
same cleaning reaches `pysimlr.predict`, missing-view prediction,
`loadings`, `feature_maps`, `heldout_coupling`, `feature_relevance` and the
refits of `loading_inference` / `select_rank` / `component_attribution`, for
every method family, with no per-method code.

The operator, per view ``m`` (training rows only)
-------------------------------------------------
`pysimlr.fit` executes an explicit three-stage preprocessing order recorded in
``provenance["pipeline_order"] == ("scale", "confound_projection", "rescale")``:

0. **Scale**: apply the view's ``scale_list`` (e.g. ``centerAndScale`` or
   ``none``) to obtain ``X_scaled``.
1. ``C_m`` (shared or per-view confounds) is standardised column-wise
   (finite-entry mean and SD; non-finite entries set to the column mean;
   constant columns dropped, all recorded) and reduced to an orthonormal,
   zero-mean basis ``Q_m`` (``n x k_c``) by SVD, keeping the numerical rank,
   capped at ``confound_rank`` or ``n // 4`` (a cap that binds warns and is
   recorded).
2. The scaled view's columns are filled at their training means ``mu_m`` where
   missing and centred: ``X_c = X_scaled - mu_m``.
3. ``H_m`` (``p_m x r``) is an orthonormal basis of the feature-space
   confound covariance ``X_c^T Q_m``: the feature directions along which the
   view covaries with the confounds (``r <= k_c``, numerical rank).
4. **Confound projection**: ``X_clean = X_scaled - gamma (X_scaled - mu_m) H_m H_m^T``,
   with the fraction of variance removed recorded in
   ``result["confound_sink"]["variance_removed"]``.
5. **Rescale**: controlled by ``confound_rescale`` (``"unit_variance"`` by
   default, independent of ``scale_list``, or ``"none"``). ``"unit_variance"``
   re-centres and divides each cleaned feature by its post-projection training
   SD so features whose variance was shrunk by confound removal do not drop out
   of within-view objectives (`recon_r2`, `pca`). Because ``Q_m`` is zero-mean
   and rescaling acts column by column, ``gamma = 1`` still yields exact zero
   sample covariance between every linear score and every confound column.
   When ``confound_rescale="none"`` and any view loses more than 30% of its
   variance, a `UserWarning` is emitted.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import threading
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

#: Explicit preprocessing stage names recorded in provenance and sink records.
PIPELINE_ORDER: tuple[str, str, str] = ("scale", "confound_projection", "rescale")
#: Post-projection feature rescaling modes.
CONFOUND_RESCALE_MODES: tuple[str, str] = ("unit_variance", "none")
#: Warn when `confound_rescale == "none"` and a view loses more than this fraction of variance.
CONFOUND_VARIANCE_WARN_THRESHOLD: float = 0.30

#: Parameters `pysimlr.fit` handles for every method.
SINK_PARAMS: dict[str, Any] = {
    "confounds": None,
    "confound_strength": 1.0,
    "confound_rank": None,
    "confound_rescale": "unit_variance",
}

_CONFOUND_STATE = threading.local()


def validate_confound_rescale(confound_rescale: Any) -> str:
    """Validate `confound_rescale` in `('unit_variance', 'none')`."""
    if not isinstance(confound_rescale, str) or confound_rescale not in CONFOUND_RESCALE_MODES:
        raise ValueError(
            f"confound_rescale must be one of {CONFOUND_RESCALE_MODES}; got {confound_rescale!r}."
        )
    return confound_rescale



# ----------------------------------------------------------------------------- validation
def validate_confound_params(confound_strength: Any, confound_rank: Any) -> tuple[float, int | None]:
    """Validate `confound_strength` (a float in [0, 1]) and `confound_rank` (positive int or None)."""
    if isinstance(confound_strength, bool) or not isinstance(confound_strength, (int, float, np.floating, np.integer)):
        raise ValueError(f"confound_strength must be a float in [0.0, 1.0]; got {confound_strength!r}.")  # noqa: TRY004 -- one ValueError for every bad value, as elsewhere in pysimlr
    gamma = float(confound_strength)
    if not np.isfinite(gamma) or gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"confound_strength must lie in [0.0, 1.0]; got {confound_strength!r}.")
    rank: int | None = None
    if confound_rank is not None:
        if isinstance(confound_rank, bool) or not isinstance(confound_rank, (int, np.integer)) or int(confound_rank) < 1:
            raise ValueError(f"confound_rank must be a positive integer or None; got {confound_rank!r}.")
        rank = int(confound_rank)
    return gamma, rank


def _is_single_matrix(obj: Any) -> bool:
    return isinstance(obj, (torch.Tensor, np.ndarray)) or hasattr(obj, "to_numpy")


def _to_numpy(mat: Any, label: str) -> np.ndarray:
    if hasattr(mat, "to_numpy"):
        arr = np.asarray(mat.to_numpy(), dtype=np.float64)
    elif isinstance(mat, torch.Tensor):
        arr = mat.detach().cpu().to(torch.float64).numpy()
    else:
        try:
            arr = np.asarray(mat, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric; {exc}") from exc
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{label} must be 1-D or 2-D (samples x confounds); got shape {tuple(arr.shape)}.")
    if arr.shape[1] == 0:
        raise ValueError(f"{label} has 0 columns.")
    return np.array(arr, dtype=np.float64, order="C", copy=True)


def _resolve_view_key(key: Any, n_views: int, view_names: Sequence[str] | None = None) -> int:
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool) and 0 <= int(key) < n_views:
        return int(key)
    if view_names is not None and isinstance(key, str) and key in view_names:
        idx = list(view_names).index(key)
        if 0 <= idx < n_views:
            return int(idx)
    if isinstance(key, str) and key.startswith("view") and key[4:].isdigit() and int(key[4:]) < n_views:
        return int(key[4:])
    names_hint = f", or one of {list(view_names)}" if view_names is not None else ""
    raise ValueError(f"unknown view key {key!r} in confounds; expected an index in 0..{n_views - 1} "
                     f"or 'view0'..'view{n_views - 1}'{names_hint}.")


def _unpack_confounds(confounds: Any, n_views: int,
                      view_names: Sequence[str] | None = None) -> tuple[list[tuple[str, Any]], bool]:
    if confounds is None:
        raise ValueError("confounds is None.")
    if _is_single_matrix(confounds):
        return [("confounds", confounds)] * n_views, True
    if isinstance(confounds, Mapping):
        if not confounds:
            raise ValueError("confounds mapping is empty.")
        items: list[tuple[str, Any]] = [("", None)] * n_views
        seen = [False] * n_views
        for key, val in confounds.items():
            idx = _resolve_view_key(key, n_views, view_names=view_names)
            if seen[idx]:
                raise ValueError(f"confounds gives view {idx} twice.")
            seen[idx] = True
            items[idx] = (f"confounds[{key!r}]", val)
        return items, False
    if isinstance(confounds, (list, tuple)):
        if len(confounds) != n_views:
            raise ValueError(f"per-view confounds has {len(confounds)} entries; there are {n_views} views.")
        return [(f"confounds[{i}]", val) for i, val in enumerate(confounds)], False
    raise ValueError(f"confounds must be an array, tensor, DataFrame/Series, a per-view list or a dict; "  # noqa: TRY004
                     f"got {type(confounds).__name__}.")


def per_view_confounds(confounds: Any, n_samples: int, n_views: int,
                       view_names: Sequence[str] | None = None) -> tuple[list[np.ndarray | None], bool]:
    """``confounds`` as one ``(n, q_m)`` array (or None) per view, and whether it was one shared matrix.

    Accepts a shared ``(n, q)`` / ``(n,)`` array, tensor, DataFrame or Series;
    a per-view sequence (``None`` for a view without confounds); or a dict
    keyed by view index, ``"view<i>"``, or a view name in ``view_names``.
    """
    items, shared = _unpack_confounds(confounds, n_views, view_names=view_names)
    if shared:
        c = _to_numpy(items[0][1], "confounds")
        if c.shape[0] != n_samples:
            raise ValueError(f"confounds has {c.shape[0]} rows; the views have {n_samples}.")
        return [c] * n_views, True
    out = [None if val is None else _to_numpy(val, lbl) for lbl, val in items]
    if all(c is None for c in out):
        raise ValueError("confounds gives no confound matrix for any view.")
    for i, c in enumerate(out):
        if c is not None and c.shape[0] != n_samples:
            raise ValueError(f"confounds for view {i} has {c.shape[0]} rows; the views have {n_samples}.")
    return out, False


# ----------------------------------------------------------------------------- row operations
def _slice_one(mat: Any, idx: np.ndarray) -> Any:
    if mat is None:
        return None
    if hasattr(mat, "iloc"):
        return mat.iloc[idx]
    if isinstance(mat, torch.Tensor):
        return mat[torch.as_tensor(idx, dtype=torch.long, device=mat.device)]
    return np.asarray(mat)[idx]


def slice_confounds(confounds: Any, indices: Any) -> Any:
    """Rows ``indices`` of `confounds` in any accepted form (the same form back)."""
    if confounds is None:
        return None
    idx = (indices.detach().cpu().numpy() if isinstance(indices, torch.Tensor)
           else np.asarray(indices, dtype=np.int64))
    if _is_single_matrix(confounds):
        return _slice_one(confounds, idx)
    if isinstance(confounds, Mapping):
        return {k: _slice_one(v, idx) for k, v in confounds.items()}
    if isinstance(confounds, (list, tuple)):
        return type(confounds)(_slice_one(v, idx) for v in confounds)
    raise ValueError(f"cannot slice confounds of type {type(confounds).__name__}.")


def permute_view_confounds(confounds: Any, perms: Sequence[np.ndarray | None],
                           view_names: Sequence[str] | None = None) -> list[Any]:
    """Per-view confounds whose rows follow each view's row permutation.

    ``perms[m]`` is the row order of view ``m`` in a null refit (``None``:
    unpermuted). A shared matrix becomes a per-view list of permuted copies;
    since each view is cleaned only against its own confound rows and the
    operator commutes with row permutations, every view of the null refit is
    cleaned exactly as in the observed fit (same basis rank, same penalty).
    """
    items, _ = _unpack_confounds(confounds, len(perms), view_names=view_names)
    return [v if (v is None or p is None) else _slice_one(v, np.asarray(p, dtype=np.int64))
            for (_, v), p in zip(items, perms)]


# ----------------------------------------------------------------------------- confound basis
def confound_basis(c_raw: np.ndarray, confound_rank: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    """Orthonormal zero-mean basis ``Q`` of the standardised confound columns, and what was done."""
    n, q = c_raw.shape
    info: dict[str, Any] = {"n_columns": int(q), "dropped_columns": [], "n_imputed": 0}
    cols = []
    for j in range(q):
        col = c_raw[:, j]
        fin = np.isfinite(col)
        if fin.sum() < 2:
            info["dropped_columns"].append({"column": j, "reason": "fewer than 2 finite values"})
            continue
        mu, sd = col[fin].mean(), col[fin].std(ddof=1)
        if not np.isfinite(sd) or sd <= 1e-12 * max(1.0, abs(mu)):
            info["dropped_columns"].append({"column": j, "reason": "constant"})
            continue
        info["n_imputed"] += int((~fin).sum())
        cols.append(np.where(fin, (col - mu) / sd, 0.0))
    if not cols:
        raise ValueError("confounds: every column is constant or missing; nothing to remove.")
    c_std = np.column_stack(cols)
    c_std = c_std - c_std.mean(axis=0, keepdims=True)
    u_svd, s_svd, _ = np.linalg.svd(c_std, full_matrices=False)
    numerical = int((s_svd > 1e-8 * s_svd[0]).sum())
    cap = max(1, n // 4) if confound_rank is None else int(confound_rank)
    cap = min(cap, n - 2)
    if cap < 1:
        raise ValueError(f"confounds: {n} rows are too few to estimate a confound sink.")
    k_c = min(numerical, cap)
    info.update({"numerical_rank": numerical, "rank": int(k_c), "rank_cap": int(cap),
                 "rank_truncated": bool(k_c < numerical)})
    if info["rank_truncated"]:
        warnings.warn(f"confounds: numerical rank {numerical} truncated to {k_c} "
                      f"({'confound_rank' if confound_rank is not None else 'n // 4'}); "
                      f"recorded in result['confound_sink'].", RuntimeWarning, stacklevel=4)
    qb = u_svd[:, :k_c]
    return qb - qb.mean(axis=0, keepdims=True), info


# ----------------------------------------------------------------------------- one view
def fit_view_sink(x: Any, c_raw: np.ndarray, gamma: float, confound_rank: int | None) -> dict[str, Any]:
    """Fit the cleaning operator of one view on its training rows (see module docstring)."""
    xa = (x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float64))
    if xa.shape[0] != c_raw.shape[0]:
        raise ValueError(f"view has {xa.shape[0]} rows; its confounds have {c_raw.shape[0]}.")
    q, binfo = confound_basis(c_raw, confound_rank)
    mu = np.nanmean(np.where(np.isfinite(xa), xa, np.nan), axis=0)
    if not np.all(np.isfinite(mu)):
        bad = np.where(~np.isfinite(mu))[0].tolist()
        raise ValueError(f"confound sink: view columns {bad[:10]} have no finite training value.")
    n_missing = int((~np.isfinite(xa)).sum())
    xf = np.where(np.isfinite(xa), xa, mu)
    xc = xf - mu
    u_b, s_b, _ = np.linalg.svd(xc.T @ q, full_matrices=False)
    r = int((s_b > 1e-8 * s_b[0]).sum()) if s_b.size and s_b[0] > 0 else 0
    if r == 0:
        raise ValueError("confound sink: a view has zero covariance with every confound on the training rows.")
    if r >= xa.shape[1]:
        raise ValueError(f"confound sink: the confounds span all {xa.shape[1]} feature directions of a view "
                         f"(rank {r}); cleaning would remove the whole view. Lower confound_rank or drop the view.")
    h = u_b[:, :r]
    z = xc @ h
    ss_before = float(np.sum(xc * xc))
    xc_clean = xc - float(gamma) * (z @ h.T)
    ss_after = float(np.sum(xc_clean * xc_clean))
    var_removed = float(np.clip(1.0 - ss_after / ss_before, 0.0, 1.0)) if ss_before > 0.0 else 0.0
    m_map = np.linalg.lstsq(z, q, rcond=None)[0]
    q_hat = z @ m_map
    fit_r = [float(np.corrcoef(q_hat[:, j], q[:, j])[0, 1]) if q_hat[:, j].std() > 0 else 0.0
             for j in range(q.shape[1])]
    return {"mean": mu, "H": h, "M": m_map, "gamma": float(gamma), "basis": q,
            "info": {**binfo, "removed_directions": r, "n_missing_filled": n_missing,
                     "variance_removed": var_removed,
                     "train_confound_fit_r": fit_r}}


def apply_view_sink(x: Any, sink: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cleaned view, image-derived confound scores (X - mu) H M)`` for any rows of one view."""
    xa = (x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float64))
    if xa.ndim != 2 or xa.shape[1] != sink["mean"].shape[0]:
        raise ValueError(f"view has shape {tuple(xa.shape)}; the confound sink expects "
                         f"{sink['mean'].shape[0]} columns.")
    row_nan = np.isnan(xa).all(axis=1) if sink.get("preserve_row_nans") else None
    mu = sink["mean"]
    xf = np.where(np.isfinite(xa), xa, mu)
    z = (xf - mu) @ sink["H"]
    cleaned = xf - sink["gamma"] * (z @ sink["H"].T)
    scores = z @ sink["M"]
    if row_nan is not None and row_nan.any():
        cleaned = cleaned.copy()
        scores = scores.copy()
        cleaned[row_nan] = np.nan
        scores[row_nan] = np.nan
    return torch.from_numpy(cleaned).float(), torch.from_numpy(scores).float()


# ----------------------------------------------------------------------------- all views
def _unit_variance_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.mean(x, dim=0)
    if x.shape[0] > 1:
        std = torch.std(x, dim=0)
        std = torch.where(torch.isnan(std) | (std < 1e-10), torch.ones_like(std), std)
    else:
        std = torch.ones_like(mean)
    return mean, std


def _warn_if_unrescaled_variance_loss(variance_removed: Sequence[float | None],
                                      rescale: str, *, stacklevel: int = 3) -> None:
    if rescale != "none":
        return
    heavy = [(i, v) for i, v in enumerate(variance_removed)
             if v is not None and v > CONFOUND_VARIANCE_WARN_THRESHOLD]
    if not heavy:
        return
    desc = ", ".join(f"view {i} ({100.0 * v:.1f}%)" for i, v in heavy)
    warnings.warn(
        f"confounds removed more than {int(100 * CONFOUND_VARIANCE_WARN_THRESHOLD)}% of variance "
        f"in {desc} with confound_rescale='none'; reduced-variance features contribute less to "
        f"within-view objectives such as 'recon_r2' and PCA. Pass confound_rescale='unit_variance' "
        f"(the default) to re-standardize cleaned features.",
        UserWarning,
        stacklevel=stacklevel,
    )


@dataclass
class ConfoundFitPlan:
    """Per-fit state consumed by `utils.preprocess_data` in `scale -> confound_projection -> rescale` order."""

    per_view: list[np.ndarray | None]
    shared: bool
    gamma: float
    rank: int | None
    rescale: str
    missing_rows: str
    obs_masks_np: list[np.ndarray] | None
    sinks: list[dict[str, Any] | None] = field(default_factory=list)
    cursor: int = 0


def active_confound_plan() -> ConfoundFitPlan | None:
    """Return the active thread-local `ConfoundFitPlan`, if any."""
    return getattr(_CONFOUND_STATE, "plan", None)


@contextlib.contextmanager
def confound_fit_context(plan: ConfoundFitPlan | None):
    """Install `plan` for `utils.preprocess_data` during a `pysimlr.fit` call."""
    prev = getattr(_CONFOUND_STATE, "plan", None)
    _CONFOUND_STATE.plan = plan
    try:
        yield
    finally:
        _CONFOUND_STATE.plan = prev


def prepare_confound_plan(
    views: Sequence[torch.Tensor],
    confounds: Any,
    confound_strength: Any = 1.0,
    confound_rank: Any = None,
    confound_rescale: Any = "unit_variance",
    *,
    missing_rows: str = "impute",
    view_names: Sequence[str] | None = None,
) -> ConfoundFitPlan:
    """Validate confound settings and build the `ConfoundFitPlan` for `pysimlr.fit`."""
    gamma, rank = validate_confound_params(confound_strength, confound_rank)
    rescale = validate_confound_rescale(confound_rescale)
    n = int(views[0].shape[0])
    per_view, shared = per_view_confounds(confounds, n, len(views), view_names=view_names)
    obs_np: list[np.ndarray] | None = None
    if missing_rows == "mask":
        obs_np = [(~torch.isnan(x).all(dim=1)).detach().cpu().numpy() for x in views]
    return ConfoundFitPlan(
        per_view=per_view,
        shared=bool(shared),
        gamma=gamma,
        rank=rank,
        rescale=rescale,
        missing_rows=missing_rows,
        obs_masks_np=obs_np,
    )


def apply_plan_step(
    x_scaled: torch.Tensor,
    scale_list: Sequence[str],
    pre_provenance: dict[str, Any],
    plan: ConfoundFitPlan,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run ``confound_projection -> rescale`` on one already-scaled training view."""
    idx = plan.cursor
    if idx >= len(plan.per_view):
        raise RuntimeError(
            f"confound_fit_context received more preprocess_data calls ({idx + 1}) "
            f"than views ({len(plan.per_view)})."
        )
    plan.cursor += 1
    prov = dict(pre_provenance)
    prov["pipeline_order"] = PIPELINE_ORDER
    prov["confound_rescale"] = plan.rescale
    c_m = plan.per_view[idx]
    if c_m is None:
        plan.sinks.append(None)
        return x_scaled, prov
    c_rows = c_m[plan.obs_masks_np[idx]] if plan.obs_masks_np is not None else c_m
    s = fit_view_sink(x_scaled, c_rows, plan.gamma, plan.rank)
    if plan.obs_masks_np is not None:
        s["preserve_row_nans"] = True
    s["pre_scale"] = {"scale_list": list(scale_list), "provenance": dict(pre_provenance)}
    plan.sinks.append(s)
    prov["confound_sink"] = s
    x_clean = apply_view_sink(x_scaled, s)[0].to(dtype=x_scaled.dtype, device=x_scaled.device)
    if plan.rescale == "unit_variance":
        rmean, rstd = _unit_variance_stats(x_clean)
        prov["confound_rescale_mean"] = rmean
        prov["confound_rescale_std"] = rstd
        x_clean = (x_clean - rmean) / rstd
    return x_clean, prov


def finalize_confound_plan(result: dict[str, Any], plan: ConfoundFitPlan) -> None:
    """Attach the completed `ConfoundFitPlan` record to `result` and warn if needed."""
    n_views = len(plan.per_view)
    if plan.cursor != n_views:
        raise RuntimeError(
            f"confound_fit_context processed {plan.cursor} of {n_views} views for "
            f"method {result.get('method')!r}."
        )
    c_all = [c for c in plan.per_view if c is not None]
    var_rem = [None if s is None else float(s["info"]["variance_removed"]) for s in plan.sinks]
    record = {
        "views": list(plan.sinks),
        "shared": bool(plan.shared),
        "confound_strength": plan.gamma,
        "confound_rank": [None if s is None else int(s["basis"].shape[1]) for s in plan.sinks],
        "confound_rescale": plan.rescale,
        "pipeline_order": PIPELINE_ORDER,
        "variance_removed": var_rem,
        "info": [None if s is None else s["info"] for s in plan.sinks],
        "_c_std": _standardise_for_report(np.column_stack(c_all) if not plan.shared else c_all[0]),
    }
    attach_sink(result, record)
    _warn_if_unrescaled_variance_loss(var_rem, plan.rescale, stacklevel=4)


def preprocess_with_view_sink(
    x: torch.Tensor,
    scale_list: Sequence[str],
    c_raw: np.ndarray,
    gamma: float,
    confound_rank: int | None,
    confound_rescale: str = "unit_variance",
    *,
    preserve_row_nans: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run ``scale -> confound_projection -> rescale`` for a single view (used by `extension`)."""
    from .utils import preprocess_data
    rescale = validate_confound_rescale(confound_rescale)
    sl = list(scale_list)
    with confound_fit_context(None):
        x_scaled, pre_prov = preprocess_data(x, sl)
    s = fit_view_sink(x_scaled, c_raw, gamma, confound_rank)
    if preserve_row_nans:
        s["preserve_row_nans"] = True
    s["pre_scale"] = {"scale_list": sl, "provenance": dict(pre_prov)}
    prov = dict(pre_prov)
    prov["pipeline_order"] = PIPELINE_ORDER
    prov["confound_rescale"] = rescale
    prov["confound_sink"] = s
    x_clean = apply_view_sink(x_scaled, s)[0].to(dtype=x_scaled.dtype, device=x_scaled.device)
    if rescale == "unit_variance":
        rmean, rstd = _unit_variance_stats(x_clean)
        prov["confound_rescale_mean"] = rmean
        prov["confound_rescale_std"] = rstd
        x_clean = (x_clean - rmean) / rstd
    return x_clean, prov


def fit_confound_sink(views: Sequence[torch.Tensor], confounds: Any, confound_strength: Any = 1.0,
                      confound_rank: Any = None, *, confound_rescale: str = "none",
                      missing_rows: str = "impute",
                      view_names: Sequence[str] | None = None) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Fit every view's operator on the training rows; return the cleaned views and the record."""
    gamma, rank = validate_confound_params(confound_strength, confound_rank)
    rescale = validate_confound_rescale(confound_rescale)
    n = int(views[0].shape[0])
    per_view, shared = per_view_confounds(confounds, n, len(views), view_names=view_names)
    sinks: list[dict[str, Any] | None] = []
    cleaned: list[torch.Tensor] = []
    mask_mode = (missing_rows == "mask")
    for x, c in zip(views, per_view):
        if c is None:
            sinks.append(None)
            cleaned.append(x)
            continue
        if mask_mode:
            obs_t = ~torch.isnan(x).all(dim=1)
            obs_np = obs_t.detach().cpu().numpy()
            s = fit_view_sink(x[obs_t], c[obs_np], gamma, rank)
            s["preserve_row_nans"] = True
        else:
            s = fit_view_sink(x, c, gamma, rank)
        sinks.append(s)
        xc = apply_view_sink(x, s)[0]
        if rescale == "unit_variance":
            if mask_mode:
                obs_t = ~torch.isnan(xc).all(dim=1)
                rmean, rstd = _unit_variance_stats(xc[obs_t])
                xc = xc.clone()
                xc[obs_t] = (xc[obs_t] - rmean) / rstd
            else:
                rmean, rstd = _unit_variance_stats(xc)
                xc = (xc - rmean) / rstd
        cleaned.append(xc)
    c_all = [c for c in per_view if c is not None]
    var_rem = [None if s is None else float(s["info"]["variance_removed"]) for s in sinks]
    record = {
        "views": sinks,
        "shared": bool(shared),
        "confound_strength": gamma,
        "confound_rank": [None if s is None else int(s["basis"].shape[1]) for s in sinks],
        "confound_rescale": rescale,
        "variance_removed": var_rem,
        "info": [None if s is None else s["info"] for s in sinks],
        "_c_std": _standardise_for_report(np.column_stack(c_all) if not shared else c_all[0]),
    }
    return cleaned, record


def _standardise_for_report(c: np.ndarray) -> np.ndarray:
    out = []
    for j in range(c.shape[1]):
        col = c[:, j]
        fin = np.isfinite(col)
        if fin.sum() < 2 or col[fin].std() <= 0:
            continue
        out.append(np.where(fin, col - col[fin].mean(), np.nan))
    return np.column_stack(out) if out else np.zeros((c.shape[0], 0))


def max_abs_corr(u: Any, c: Any) -> dict[str, float]:
    """``max`` and ``mean over confounds of the max over components`` of |Pearson r| (pairwise-complete)."""
    ua = np.asarray(u.detach().cpu().double() if isinstance(u, torch.Tensor) else u, dtype=np.float64)
    ca = np.asarray(c.detach().cpu().double() if isinstance(c, torch.Tensor) else c, dtype=np.float64)
    if ca.ndim == 1:
        ca = ca[:, None]
    if ua.size == 0 or ca.size == 0:
        return {"max": 0.0, "mean": 0.0}
    r = np.zeros((ua.shape[1], ca.shape[1]), dtype=np.float64)
    if ua.shape[0] >= 3 and np.all(np.isfinite(ua)) and np.all(np.isfinite(ca)):
        uc = ua - ua.mean(axis=0, keepdims=True)
        cc = ca - ca.mean(axis=0, keepdims=True)
        den = np.linalg.norm(uc, axis=0, keepdims=True).T @ np.linalg.norm(cc, axis=0, keepdims=True)
        np.divide(np.abs(uc.T @ cc), den, out=r, where=den > 0.0)
        np.clip(r, 0.0, 1.0, out=r)
    else:
        for j in range(ca.shape[1]):
            ok = np.isfinite(ca[:, j])
            if ok.sum() < 3 or ca[ok, j].std() == 0:
                continue
            cc = ca[ok, j] - ca[ok, j].mean()
            cn = float(np.linalg.norm(cc))
            if cn == 0.0:
                continue
            u_ok = ua[ok]
            uc = u_ok - u_ok.mean(axis=0, keepdims=True)
            un = np.linalg.norm(uc, axis=0)
            den = un * cn
            np.divide(np.abs(uc.T @ cc), den, out=r[:, j], where=(un > 0.0) & np.isfinite(un))
            np.clip(r[:, j], 0.0, 1.0, out=r[:, j])
    return {"max": float(r.max()), "mean": float(r.max(axis=0).mean())}


def confound_decorrelation(result: dict[str, Any], views: Any, confounds: Any) -> dict[str, float]:
    """|corr| between ``predict(result, views)["u"]`` and ``confounds`` (rows of `views`).

    The deployed estimator's decorrelation; use it on held-out rows. Works
    for any fit, with or without a confound sink, so the two can be compared.
    ``views`` may be a full sequence, a partial list with ``None`` for missing
    views, or any `Mapping` keyed by view name or index (routed through
    `predict_views`). ``confounds`` may be a shared matrix or a per-view list
    or `Mapping` (as in `pysimlr.fit`).
    """
    from .api import predict
    u = predict(result, views)["u"]
    n_rows = int(u.shape[0])
    n_views = len(result["v"])
    if isinstance(confounds, Mapping) or (
        isinstance(confounds, (list, tuple))
        and not _is_single_matrix(confounds)
        and len(confounds) == n_views
        and any(v is None or _is_single_matrix(v) for v in confounds)
    ):
        from .extension import view_names as _view_names
        per_view, shared = per_view_confounds(confounds, n_rows, n_views, view_names=_view_names(result)[:n_views])
        c_all = [cm for cm in per_view if cm is not None]
        same_non_null = shared or (
            len(c_all) > 1
            and all(cm.shape == c_all[0].shape and np.array_equal(cm, c_all[0], equal_nan=True) for cm in c_all[1:])
        )
        c = c_all[0] if same_non_null else np.column_stack(c_all)
    else:
        c = _to_numpy(confounds, "confounds")
        if c.shape[0] != n_rows:
            raise ValueError(f"confounds has {c.shape[0]} rows; the views have {n_rows}.")
    return max_abs_corr(u, c)


def sink_scores(record: dict[str, Any], views: Sequence[torch.Tensor | None]) -> dict[str, Any]:
    """Per-view estimated confound scores, and their average over the given views when shared."""
    from .utils import preprocess_data
    lat: dict[int, torch.Tensor] = {}
    for i, (x, s) in enumerate(zip(views, record["views"])):
        if x is None or s is None:
            continue
        pre = s.get("pre_scale")
        if pre is not None:
            xt = torch.as_tensor(x).float()
            row_nan = torch.isnan(xt).all(dim=1) if s.get("preserve_row_nans") else None
            x_in = preprocess_data(xt, pre["scale_list"], pre["provenance"])
            if row_nan is not None and bool(row_nan.any()):
                x_in = x_in.clone()
                x_in[row_nan] = float("nan")
            lat[i] = apply_view_sink(x_in, s)[1]
        else:
            lat[i] = apply_view_sink(x, s)[1]
    u_sink = None
    if record["shared"] and lat:
        stacked = torch.stack(list(lat.values()), dim=0)
        u_sink = torch.nanmean(stacked, dim=0) if torch.isnan(stacked).any() else stacked.mean(dim=0)
    return {"latents_sink": lat, "u_sink": u_sink}


def attach_sink(result: dict[str, Any], record: dict[str, Any]) -> None:
    """Put each view's operator into its replayed preprocessing and store the record."""
    prov = result.get("provenance_list")
    n_views = len(record["views"])
    if not isinstance(prov, list) or len(prov) != n_views or not all(isinstance(p, dict) for p in prov):
        raise ValueError(
            f"confounds need per-view preprocessing provenance to replay the sink at prediction; "
            f"method {result.get('method')!r} returned scale_list={result.get('scale_list')!r} with "
            f"{0 if not isinstance(prov, list) else len(prov)} provenance entries for {n_views} views.")
    for p, s in zip(prov, record["views"]):
        if s is not None:
            p["confound_sink"] = s
    result["confound_sink"] = record


def replay_sink(x: torch.Tensor, provenance: dict[str, Any] | None) -> torch.Tensor:
    """Apply a view's operator if its provenance carries one (called by `utils.preprocess_data`)."""
    if provenance is None or "confound_sink" not in provenance:
        return x
    return apply_view_sink(x, provenance["confound_sink"])[0].to(x.dtype)


def clean_raw_views(result: dict[str, Any], views: Sequence[Any | None]) -> list[Any | None]:
    """Apply a fit's cleaning pipeline to raw views; identity without a sink."""
    rec = result.get("confound_sink")
    if rec is None:
        return list(views)
    prov_list = result.get("provenance_list")
    scale_list = result.get("scale_list")
    use_pipeline = (
        rec.get("pipeline_order") == PIPELINE_ORDER
        and isinstance(prov_list, list)
        and len(prov_list) == len(rec["views"])
        and scale_list is not None
    )
    from .utils import preprocess_data
    out = []
    for idx, (x, s) in enumerate(zip(views, rec["views"])):
        if x is None or s is None:
            out.append(x)
        elif use_pipeline:
            cleaned = preprocess_data(torch.as_tensor(x).float(), scale_list, prov_list[idx])
            out.append(cleaned.double().numpy() if isinstance(x, np.ndarray) else cleaned)
        else:
            out.append(apply_view_sink(x, s)[0].double().numpy() if isinstance(x, np.ndarray)
                       else apply_view_sink(x, s)[0])
    return out


def refuse_probe_views(result: dict[str, Any], where: str) -> None:
    """Probe views are not cleaned by the sink; refuse rather than mix cleaned and uncleaned data."""
    if result.get("confound_sink") is not None:
        raise ValueError(
            f"{where}: probe_views are not cleaned by the confound sink of this fit (the sink is "
            f"fitted on the model's own views). Residualise the probe features on the same "
            f"confounds yourself, or fit without confounds.")
