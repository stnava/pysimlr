"""View preparation in one fixed order: ``prepare_views`` and ``apply_prepared``.

The order of the steps changes results, so it is fixed and documented here rather than re-implemented by every
study. For each view, on ``fit_rows`` (the rows that define every estimated quantity):

1. orient: multiply the columns by ``signs[view]`` (+1 / -1 per feature);
2. fill: inside rows that have the view (at least one finite value) replace isolated missing entries by the
   column median over ``fit_rows`` (optionally after dropping columns with more than ``max_missing`` missing
   among the ``fit_rows`` that have the view); rows without any value stay NaN;
3. standardize: subtract the column mean and divide by the column SD, both from ``fit_rows``;
4. nuisance: if ``nuisance[view]`` (rows x q) is given, remove what the view's own features predict of it with
   the package operator (``confounds.fit_view_sink(x, c, 1.0, None)`` fitted on ``fit_rows`` with the view,
   ``confounds.apply_view_sink`` applied to every row with the view);
5. standardize again on ``fit_rows`` (default: whenever step 4 ran).

Why step 5 matters. The nuisance operator removes variance in the directions the nuisance predicts. A view that
loses much of its variance there has columns with SD well below one afterwards; a consensus fit then weights it
by that smaller variance and the view drops out of the component. Re-standardizing restores unit scale per
column. Skipping step 5, or running step 3 only once before the nuisance step, is a different analysis.

Use the result with ``pysimlr.fit(..., scale_list=["none"])`` and no ``confounds``: the preparation is already
done, with parameters from ``fit_rows`` only. The fit's own centering and scaling would re-estimate them on every
row it is given (and with an unbiased SD, ``ddof=1``, against the ``ddof=0`` used here), and ``confounds`` would
apply the nuisance operator a second time. Rows where a whole view is missing stay NaN and are handled by the fit
(``missing_rows="mask"``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

ORDER = ("orient", "fill", "standardize", "nuisance", "standardize")
#: A column whose SD on ``fit_rows`` is not above this is constant: it is centered and not scaled (all zeros).
SD_FLOOR = 1e-12


# ----------------------------------------------------------------------------- input handling
def _to_array(x: Any, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().double().numpy()
    elif hasattr(x, "to_numpy"):
        a = np.asarray(x.to_numpy(), dtype=np.float64)
    else:
        a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"view {name!r} must be a 2-D (rows x features) array; got shape {a.shape}.")
    return a.copy()


def _split_views(views: Any) -> Tuple[List[str], List[np.ndarray], str]:
    """``(names, arrays, form)``; form is ``"mapping"`` or ``"list"`` (one matrix counts as a list of one view)."""
    if isinstance(views, Mapping):
        names = [str(k) for k in views]
        if len(set(names)) != len(names):
            raise ValueError(f"view names must be distinct; got {names}.")
        items, form = list(views.values()), "mapping"
    elif isinstance(views, (list, tuple)):
        items, form = list(views), "list"
        names = [f"view{i}" for i in range(len(items))]
    else:
        items, form, names = [views], "list", ["view0"]
    if not items:
        raise ValueError("views must contain at least one matrix.")
    arrays = [_to_array(x, nm) for x, nm in zip(items, names)]
    n = {a.shape[0] for a in arrays}
    if len(n) != 1:
        raise ValueError(f"views have different numbers of rows: {[a.shape[0] for a in arrays]}.")
    return names, arrays, form


def _per_view(arg: Any, names: List[str], what: str, *, scalar_ok: bool = False) -> List[Any]:
    """Resolve a per-view argument (None, mapping by name or index, list, or a scalar if allowed) to a list."""
    m = len(names)
    if arg is None:
        return [None] * m
    if isinstance(arg, Mapping):
        out: List[Any] = [None] * m
        for key, val in arg.items():
            if isinstance(key, (int, np.integer)) and not isinstance(key, bool) and 0 <= int(key) < m:
                idx = int(key)
            elif isinstance(key, str) and key in names:
                idx = names.index(key)
            else:
                raise ValueError(f"{what}: unknown view key {key!r}; expected one of {names} or an index 0..{m - 1}.")
            out[idx] = val
        return out
    if isinstance(arg, (list, tuple)):
        if len(arg) != m:
            raise ValueError(f"{what} has {len(arg)} entries for {m} views.")
        return list(arg)
    if scalar_ok:
        return [arg] * m
    raise TypeError(f"{what} must be None, a mapping by view name or index, or a list with one entry per view.")


def _fit_mask(fit_rows: Any, n: int) -> np.ndarray:
    if fit_rows is None:
        return np.ones(n, dtype=bool)
    a = np.asarray(fit_rows)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError(f"fit_rows is a boolean mask of length {a.size}; the views have {n} rows.")
        return a.copy()
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise TypeError("fit_rows must be None, a boolean mask of length n, or a 1-D array of integer row indices.")
    if a.size and (a.min() < -n or a.max() >= n):
        raise ValueError(f"fit_rows has an index outside 0..{n - 1}.")
    mask = np.zeros(n, dtype=bool)
    mask[a] = True
    return mask


def _check_signs(s: Any, p: int, name: str) -> np.ndarray:
    a = np.asarray(s, dtype=np.float64).ravel()
    if a.size != p:
        raise ValueError(f"signs for view {name!r} has {a.size} entries; the view has {p} columns.")
    if not np.all(np.abs(a) == 1.0):
        raise ValueError(f"signs for view {name!r} must be +1 or -1 per feature.")
    return a


def _sd_scale(sd: np.ndarray) -> np.ndarray:
    return np.where(sd > SD_FLOOR, sd, 1.0)


# ----------------------------------------------------------------------------- one view: steps shared by fit and apply
def _orient_fill_standardize(x: np.ndarray, vp: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Steps 1-3 with stored parameters: ``(standardized view, rows that have the view)``."""
    if x.shape[1] != vp["n_features_in"]:
        raise ValueError(f"view {vp['name']!r} has {x.shape[1]} columns; the preparation expects {vp['n_features_in']}.")
    if vp["signs"] is not None:
        x = x * vp["signs"]
    x = np.where(np.isfinite(x), x, np.nan)
    obs = np.isfinite(x).any(axis=1)
    x = x[:, vp["feature_mask"]]
    miss = ~np.isfinite(x) & obs[:, None]
    idx = np.where(miss)
    x[idx] = vp["median"][idx[1]]
    z = (x - vp["mean"]) / vp["scale"]
    z[~obs] = np.nan
    return z, obs


def _remove_nuisance(z: np.ndarray, obs: np.ndarray, vp: Dict[str, Any]) -> np.ndarray:
    from .confounds import apply_view_sink
    cleaned = np.full_like(z, np.nan)
    cleaned[obs] = apply_view_sink(z[obs], vp["sink"])[0].numpy().astype(np.float64)
    return cleaned


def _restandardize(cleaned: np.ndarray, vp: Dict[str, Any]) -> np.ndarray:
    return (cleaned - vp["mean2"]) / vp["scale2"]


def _prepare_one(x: np.ndarray, name: str, fit: np.ndarray, sign: Any, c: Any, max_missing: Optional[float],
                 restandardize: Optional[bool]) -> Tuple[np.ndarray, Dict[str, Any], float, List[str]]:
    n, p = x.shape
    rec: List[str] = []
    signs = None if sign is None else _check_signs(sign, p, name)
    if signs is not None:
        rec.append(f"{name}: oriented {int((signs < 0).sum())} of {p} columns (sign -1)")
    xo = x if signs is None else x * signs
    xo = np.where(np.isfinite(xo), xo, np.nan)
    obs = np.isfinite(xo).any(axis=1)
    fit_obs = fit & obs
    if int(fit_obs.sum()) < 2:
        raise ValueError(f"view {name!r}: fewer than 2 fit_rows have the view; nothing to estimate from.")

    miss_frac = (~np.isfinite(xo[fit_obs])).mean(axis=0)
    keep = np.ones(p, dtype=bool) if max_missing is None else miss_frac <= float(max_missing)
    if not keep.any():
        raise ValueError(f"view {name!r}: max_missing={max_missing} drops every column.")
    if max_missing is not None:
        rec.append(f"{name}: dropped {int((~keep).sum())} of {p} columns with more than {float(max_missing):.0%} "
                   f"missing among the {int(fit_obs.sum())} fit rows with the view")
    xk = xo[:, keep]
    empty = ~np.isfinite(xk[fit_obs]).any(axis=0)
    if empty.any():
        raise ValueError(f"view {name!r}: {int(empty.sum())} kept column(s) have no finite value on fit_rows "
                         f"(first: {np.where(empty)[0][:5].tolist()}); drop them with max_missing.")
    median = np.nanmedian(xk[fit_obs], axis=0)
    n_filled = int((~np.isfinite(xk) & obs[:, None]).sum())
    rec.append(f"{name}: filled {n_filled} isolated missing entries with the fit-row column median; "
               f"{int((~obs).sum())} rows without the view stay NaN")
    filled = xk.copy()
    idx = np.where(~np.isfinite(filled) & obs[:, None])
    filled[idx] = median[idx[1]]
    mean = filled[fit_obs].mean(axis=0)
    sd = filled[fit_obs].std(axis=0)
    n_const = int((sd <= SD_FLOOR).sum())
    vp: Dict[str, Any] = {
        "name": name, "n_features_in": p, "signs": signs, "feature_mask": keep, "median": median,
        "mean": mean, "scale": _sd_scale(sd), "sink": None, "mean2": None, "scale2": None, "restandardize": False,
    }
    rec.append(f"{name}: standardized {int(keep.sum())} columns on {int(fit_obs.sum())} fit rows (SD with ddof=0"
               + (f"; {n_const} constant column(s) centered and left unscaled" if n_const else "") + ")")
    z, obs2 = _orient_fill_standardize(x, vp)
    removed = 0.0
    if c is not None:
        from .confounds import fit_view_sink
        ca = np.asarray(c, dtype=np.float64)
        if ca.ndim == 1:
            ca = ca[:, None]
        if ca.ndim != 2 or ca.shape[0] != n:
            raise ValueError(f"nuisance for view {name!r} must be (rows x q) with {n} rows; got {ca.shape}.")
        try:
            sink = fit_view_sink(z[fit_obs], ca[fit_obs], 1.0, None)
        except ValueError as exc:
            raise ValueError(f"view {name!r}: {exc}") from exc
        vp["sink"] = {kk: sink[kk] for kk in ("mean", "H", "M", "gamma", "info")}
        cleaned = _remove_nuisance(z, obs2, vp)
        removed = float(1.0 - np.nanvar(cleaned[fit_obs], axis=0).sum() / np.nanvar(z[fit_obs], axis=0).sum())
        rec.append(f"{name}: nuisance ({ca.shape[1]} column(s)) removed {removed:.3f} of the variance "
                   f"({sink['info']['removed_directions']} directions)")
        do_again = True if restandardize is None else bool(restandardize)
        if do_again:
            mean2 = cleaned[fit_obs].mean(axis=0)
            sd2 = cleaned[fit_obs].std(axis=0)
            vp.update({"mean2": mean2, "scale2": _sd_scale(sd2), "restandardize": True})
            out = _restandardize(cleaned, vp)
            rec.append(f"{name}: standardized again on the fit rows after the nuisance step")
        else:
            out = cleaned
            rec.append(f"{name}: NOT standardized again after the nuisance step (restandardize=False)")
    else:
        out = z
        if restandardize:
            rec.append(f"{name}: restandardize ignored (no nuisance step for this view)")
    return out, vp, removed, rec


# ----------------------------------------------------------------------------- public
def _pack(names: List[str], values: List[Any], form: str) -> Any:
    return dict(zip(names, values)) if form == "mapping" else list(values)


def prepare_views(views: Any, *, fit_rows: Any = None, signs: Any = None, nuisance: Any = None,
                  max_missing: Any = None, restandardize: Optional[bool] = None) -> Dict[str, Any]:
    """Prepare views in the fixed order orient, fill, standardize, remove nuisance, standardize again.

    Every estimated quantity (column median, mean, SD, the nuisance operator) comes from ``fit_rows`` only and is
    applied to all rows; `apply_prepared` applies the stored parameters to new rows. Each step is optional and
    none is reordered. See the module docstring for the rationale.

    Parameters
    ----------
    views : matrix, list of matrices or mapping name -> matrix
        ``(n, p_m)`` arrays or tensors (one per view; NaN or inf marks a missing entry). The result has the
        same form (a single matrix gives a list of one view). Computation is in float64.
    fit_rows : None, boolean mask of length n, or integer row indices
        Rows that define the preparation (default all). For each view only the fit rows that have the view
        (at least one finite value) count.
    signs : None, mapping or list
        Per view an array of +1 / -1 per feature, multiplied into the columns first (views without an entry are
        not oriented). A mapping is keyed by view name or index.
    nuisance : None, mapping or list
        Per view an ``(n, q)`` array (or ``(n,)``). The view is cleaned with
        ``pysimlr.confounds.fit_view_sink(x, c, 1.0, None)`` fitted on the fit rows with the view and
        ``apply_view_sink(x, sink)[0]`` applied to all rows with the view (that operator works in float32, so
        cleaned values carry float32 rounding). ``c`` is standardized and rank-capped inside the sink, and its
        own missing entries are imputed there.
    max_missing : None, float, mapping or list
        Drop columns whose missing fraction among the fit rows with the view exceeds this value (between 0 and 1;
        one value for all views, or per view). Without it, a column with no finite value on the fit rows raises
        ``ValueError``.
    restandardize : None or bool
        Step 5. ``None`` (default): standardize again whenever the nuisance step ran for the view; ``False``
        skips it (reproduces the order effect: nuisance-heavy views keep SD below 1); ``True`` is the same as
        ``None`` (a view without a nuisance step is already standardized).

    Standard deviation convention. Population SD (``ddof=0``) on the fit rows with the view. A column whose SD is
    not above ``1e-12`` is constant: it is centered and not scaled, so it becomes all zeros (it stays in the
    view, so the column count is unchanged; remove it upstream if unwanted).

    Returns
    -------
    dict
        ``views`` (same form as the input, ``(n, kept_p)`` float64 arrays, all rows; rows without the view are
        NaN), ``feature_mask`` (per view, boolean over the input columns, False for columns dropped by
        ``max_missing``), ``variance_removed`` (per view, share of the fit-row variance the nuisance step removed,
        ``0.0`` when there was no nuisance step), ``params`` (everything `apply_prepared` needs), ``record``
        (list of strings, one per step and view), ``view_names`` and ``fit_rows`` (boolean mask).

    Pass ``views["views"]`` to ``pysimlr.fit`` with ``scale_list=["none"]``, no ``confounds``, and
    ``missing_rows="mask"`` if any row lacks a whole view.
    """
    names, arrays, form = _split_views(views)
    n = arrays[0].shape[0]
    fit = _fit_mask(fit_rows, n)
    signs_l = _per_view(signs, names, "signs")
    nuis_l = _per_view(nuisance, names, "nuisance")
    mm_l = _per_view(max_missing, names, "max_missing", scalar_ok=True)
    out_views, masks, removed, vps = [], [], [], []
    record = [(f"fit rows: {int(fit.sum())} of {n} rows define medians, means, SDs and nuisance operators; "
               f"order: {' > '.join(ORDER)}")]
    for i, nm in enumerate(names):
        o, vp, rem, rec = _prepare_one(arrays[i], nm, fit, signs_l[i], nuis_l[i], mm_l[i], restandardize)
        out_views.append(o)
        masks.append(vp["feature_mask"].copy())
        removed.append(rem)
        vps.append(vp)
        record.extend(rec)
    params = {"order": list(ORDER), "form": form, "view_names": names, "views": vps}
    return {"views": _pack(names, out_views, form), "feature_mask": _pack(names, masks, form),
            "variance_removed": _pack(names, removed, form), "params": params, "record": record,
            "view_names": names, "fit_rows": fit}


def apply_prepared(params: Dict[str, Any], views: Any, nuisance: Any = None) -> Any:
    """Apply the preparation stored by `prepare_views` to new rows, estimating nothing.

    ``views`` has the form used at fit time (a mapping with the same names, or a list with the same number of
    views) and full-width columns. Returns the prepared views in that form (float64, NaN rows for rows without
    the view). On the fit rows it reproduces ``prepare_views(...)["views"]`` exactly.

    ``nuisance`` is accepted so that call sites mirror `prepare_views`, and checked only for its row count: the
    operator is fixed at fit time and is a function of the view's own features (``apply_view_sink``), so new
    rows need no nuisance values.
    """
    names, arrays, form = _split_views(views)
    if form != params["form"]:
        raise ValueError(f"the preparation was fitted on a {params['form']} of views; got a {form}.")
    if form == "mapping":
        if set(names) != set(params["view_names"]):
            raise ValueError(f"view names {sorted(names)} differ from the fitted {sorted(params['view_names'])}.")
        order = [names.index(nm) for nm in params["view_names"]]
        arrays = [arrays[i] for i in order]
        names = list(params["view_names"])
    elif len(arrays) != len(params["views"]):
        raise ValueError(f"{len(arrays)} views given; the preparation was fitted on {len(params['views'])}.")
    if nuisance is not None:
        for nm, c in zip(names, _per_view(nuisance, names, "nuisance")):
            if c is not None and np.asarray(c).shape[0] != arrays[0].shape[0]:
                raise ValueError(f"nuisance for view {nm!r} has {np.asarray(c).shape[0]} rows; the views have "
                                 f"{arrays[0].shape[0]}.")
    out = []
    for x, vp in zip(arrays, params["views"]):
        z, obs = _orient_fill_standardize(x, vp)
        if vp["sink"] is not None:
            z = _remove_nuisance(z, obs, vp)
            if vp["restandardize"]:
                z = _restandardize(z, vp)
        out.append(z)
    return _pack(names, out, form)


__all__ = ["prepare_views", "apply_prepared"]
