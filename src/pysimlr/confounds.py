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
1. ``C_m`` (shared or per-view confounds) is standardised column-wise
   (finite-entry mean and SD; non-finite entries set to the column mean;
   constant columns dropped, all recorded) and reduced to an orthonormal,
   zero-mean basis ``Q_m`` (``n x k_c``) by SVD, keeping the numerical rank,
   capped at ``confound_rank`` or ``n // 4`` (a cap that binds warns and is
   recorded).
2. The view's columns are filled at their training means ``mu_m`` where
   missing and centred: ``X_c = X_m - mu_m``.
3. ``H_m`` (``p_m x r``) is an orthonormal basis of the feature-space
   confound covariance ``X_c^T Q_m``: the feature directions along which the
   view covaries with the confounds (``r <= k_c``, numerical rank).
4. The cleaned view is ``X_m - gamma (X_m - mu_m) H_m H_m^T``: those ``r``
   directions are projected out of the features (``gamma`` =
   ``confound_strength`` in ``[0, 1]`` damps the projection).

With ``gamma = 1`` every linear score ``X_clean w`` has exactly zero sample
covariance with every confound column on the training rows, and the
confound's feature pattern is removed from any row, so held-out decorrelation
needs no ``C_test`` -- it is limited only by how well ``X_c^T Q_m`` estimates
the population pattern. What is lost is the biology that lies in those ``r``
feature directions (all of it at ``gamma = 1``; biology that covaries with a
confound is partly kept at ``gamma < 1``). Nonlinear heads (deep and flow
models) see only the cleaned first-layer input but can still form nonlinear
functions of it that correlate with a confound; measure, do not assume.

The sink scores (``predict(...)["latents_sink"]``, ``u_sink``) are
``(X_m - mu_m) H_m M_m``, with ``M_m`` the least-squares map from the removed
directions to ``Q_m`` on the training rows: an image-derived estimate of the
confounds, averaged over the available views when the confounds are shared.

The same affine map is applied to the training rows and to new rows, so
``result["u"]`` equals ``predict(result, training_views)["u"]`` up to the
method's own prediction error, and a single subject (``N = 1``) is cleaned
exactly as in a batch. The training decorrelation is reported from the predict
path in ``result["confound_sink"]["train_max_abs_corr"]``; measure held-out
decorrelation with `confound_decorrelation` on rows that have confounds.

Row permutations commute with the operator (``clean(X[p], C[p]) =
clean(X, C)[p]``), which is what the permutation nulls rely on: each view is
permuted together with its own confound rows, so a null refit cleans each
view exactly as the observed fit does.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

#: Parameters `pysimlr.fit` handles for every method.
SINK_PARAMS: dict[str, Any] = {"confounds": None, "confound_strength": 1.0, "confound_rank": None}



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
    return np.array(arr, dtype=np.float64, copy=True)


def _resolve_view_key(key: Any, n_views: int) -> int:
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool) and 0 <= int(key) < n_views:
        return int(key)
    if isinstance(key, str) and key.startswith("view") and key[4:].isdigit() and int(key[4:]) < n_views:
        return int(key[4:])
    raise ValueError(f"unknown view key {key!r} in confounds; expected an index in 0..{n_views - 1} "
                     f"or 'view0'..'view{n_views - 1}'.")


def per_view_confounds(confounds: Any, n_samples: int, n_views: int) -> tuple[list[np.ndarray | None], bool]:
    """``confounds`` as one ``(n, q_m)`` array (or None) per view, and whether it was one shared matrix.

    Accepts a shared ``(n, q)`` / ``(n,)`` array, tensor, DataFrame or Series;
    a per-view sequence (``None`` for a view without confounds); or a dict
    keyed by view index or ``"view<i>"``.
    """
    if confounds is None:
        raise ValueError("confounds is None.")
    if _is_single_matrix(confounds):
        c = _to_numpy(confounds, "confounds")
        if c.shape[0] != n_samples:
            raise ValueError(f"confounds has {c.shape[0]} rows; the views have {n_samples}.")
        return [c] * n_views, True
    if isinstance(confounds, Mapping):
        if not confounds:
            raise ValueError("confounds mapping is empty.")
        out: list[np.ndarray | None] = [None] * n_views
        for key, val in confounds.items():
            idx = _resolve_view_key(key, n_views)
            if out[idx] is not None:
                raise ValueError(f"confounds gives view {idx} twice.")
            out[idx] = None if val is None else _to_numpy(val, f"confounds[{key!r}]")
    elif isinstance(confounds, (list, tuple)):
        if len(confounds) != n_views:
            raise ValueError(f"per-view confounds has {len(confounds)} entries; there are {n_views} views.")
        out = [None if val is None else _to_numpy(val, f"confounds[{i}]") for i, val in enumerate(confounds)]
    else:
        raise ValueError(f"confounds must be an array, tensor, DataFrame/Series, a per-view list or a dict; "  # noqa: TRY004
                         f"got {type(confounds).__name__}.")
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


def permute_view_confounds(confounds: Any, perms: Sequence[np.ndarray | None]) -> list[Any]:
    """Per-view confounds whose rows follow each view's row permutation.

    ``perms[m]`` is the row order of view ``m`` in a null refit (``None``:
    unpermuted). A shared matrix becomes a per-view list of permuted copies;
    since each view is cleaned only against its own confound rows and the
    operator commutes with row permutations, every view of the null refit is
    cleaned exactly as in the observed fit (same basis rank, same penalty).
    """
    n_views = len(perms)
    if _is_single_matrix(confounds):
        items: list[Any] = [confounds] * n_views
    elif isinstance(confounds, Mapping):
        items = [None] * n_views
        for key, val in confounds.items():
            items[_resolve_view_key(key, n_views)] = val
    elif isinstance(confounds, (list, tuple)):
        if len(confounds) != n_views:
            raise ValueError(f"per-view confounds has {len(confounds)} entries; there are {n_views} views.")
        items = list(confounds)
    else:
        raise ValueError(f"cannot permute confounds of type {type(confounds).__name__}.")
    return [v if (v is None or p is None) else _slice_one(v, np.asarray(p, dtype=np.int64))
            for v, p in zip(items, perms)]


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
    m_map = np.linalg.lstsq(z, q, rcond=None)[0]
    q_hat = z @ m_map
    fit_r = [float(np.corrcoef(q_hat[:, j], q[:, j])[0, 1]) if q_hat[:, j].std() > 0 else 0.0
             for j in range(q.shape[1])]
    return {"mean": mu, "H": h, "M": m_map, "gamma": float(gamma), "basis": q,
            "info": {**binfo, "removed_directions": r, "n_missing_filled": n_missing,
                     "train_confound_fit_r": fit_r}}


def apply_view_sink(x: Any, sink: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cleaned view, image-derived confound scores (X - mu) H M)`` for any rows of one view."""
    xa = (x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float64))
    if xa.ndim != 2 or xa.shape[1] != sink["mean"].shape[0]:
        raise ValueError(f"view has shape {tuple(xa.shape)}; the confound sink expects "
                         f"{sink['mean'].shape[0]} columns.")
    mu = sink["mean"]
    xf = np.where(np.isfinite(xa), xa, mu)
    z = (xf - mu) @ sink["H"]
    cleaned = xf - sink["gamma"] * (z @ sink["H"].T)
    return torch.from_numpy(cleaned).float(), torch.from_numpy(z @ sink["M"]).float()


# ----------------------------------------------------------------------------- all views
def fit_confound_sink(views: Sequence[torch.Tensor], confounds: Any, confound_strength: Any = 1.0,
                      confound_rank: Any = None) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Fit every view's operator on the training rows; return the cleaned views and the record."""
    gamma, rank = validate_confound_params(confound_strength, confound_rank)
    n = int(views[0].shape[0])
    per_view, shared = per_view_confounds(confounds, n, len(views))
    sinks: list[dict[str, Any] | None] = []
    cleaned: list[torch.Tensor] = []
    for x, c in zip(views, per_view):
        if c is None:
            sinks.append(None)
            cleaned.append(x)
            continue
        s = fit_view_sink(x, c, gamma, rank)
        sinks.append(s)
        cleaned.append(apply_view_sink(x, s)[0])
    c_all = [c for c in per_view if c is not None]
    record = {
        "views": sinks,
        "shared": bool(shared),
        "confound_strength": gamma,
        "confound_rank": [None if s is None else int(s["basis"].shape[1]) for s in sinks],
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
    r = np.zeros((ua.shape[1], ca.shape[1]))
    for j in range(ca.shape[1]):
        ok = np.isfinite(ca[:, j])
        if ok.sum() < 3 or ca[ok, j].std() == 0:
            continue
        for i in range(ua.shape[1]):
            if ua[ok, i].std() > 0:
                r[i, j] = abs(np.corrcoef(ua[ok, i], ca[ok, j])[0, 1])
    return {"max": float(r.max()) if r.size else 0.0,
            "mean": float(r.max(axis=0).mean()) if r.size else 0.0}


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
        per_view, _ = per_view_confounds(confounds, n_rows, n_views)
        c = np.column_stack([cm for cm in per_view if cm is not None])
    else:
        c = _to_numpy(confounds, "confounds")
        if c.shape[0] != n_rows:
            raise ValueError(f"confounds has {c.shape[0]} rows; the views have {n_rows}.")
    return max_abs_corr(u, c)


def sink_scores(record: dict[str, Any], views: Sequence[torch.Tensor | None]) -> dict[str, Any]:
    """Per-view estimated confound scores, and their average over the given views when shared."""
    lat: dict[int, torch.Tensor] = {}
    for i, (x, s) in enumerate(zip(views, record["views"])):
        if x is None or s is None:
            continue
        lat[i] = apply_view_sink(x, s)[1]
    u_sink = None
    if record["shared"] and lat:
        u_sink = torch.stack(list(lat.values()), dim=0).mean(dim=0)
    return {"latents_sink": lat, "u_sink": u_sink}


def attach_sink(result: dict[str, Any], record: dict[str, Any]) -> None:
    """Put each view's operator into its replayed preprocessing and store the record."""
    prov = result.get("provenance_list")
    n_views = len(record["views"])
    if not isinstance(prov, list) or len(prov) != n_views or not all(isinstance(p, dict) for p in prov):
        raise ValueError(
            f"confounds need per-view preprocessing provenance to replay the sink at prediction; "
            f"method {result.get('method')!r} returned scale_list={result.get('scale_list')!r} with "
            f"{0 if not isinstance(prov, list) else len(prov)} provenance entries for {n_views} views. "
            f"Use a scale_list other than ['none'].")
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
    """Apply a fit's operators to raw views (no other preprocessing); identity without a sink."""
    rec = result.get("confound_sink")
    if rec is None:
        return list(views)
    out = []
    for x, s in zip(views, rec["views"]):
        if x is None or s is None:
            out.append(x)
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
