r"""Which components carry a held-out prediction, and how stable is their support.

`component_attribution` asks three questions of a fitted multi-view model:

* **Which components predict the outcome.** Each view's first-layer projection ``X_m V_m`` is
  scored on held-out rows. Every component gets its held-out R^2 *alone*, and its *unique*
  contribution: the loss in held-out R^2 when that column is dropped from all of them. The
  read-out is a linear regression fitted on the training rows.
* **What a view's variance looks like.** For each view: the eigenvalue spectrum of its training
  correlation matrix (one dominant factor means the view is mostly one global axis), the mean
  inter-feature correlation, the correlation of each consensus component (rotated frame) with the
  view's subject mean, and, when the fit has OLEE ``loadings``, how many features are
  hard-assigned to each component.
* **Whether the predictive support replicates.** The model is refitted on random subsets of the
  training rows, with the same method and parameters (``refit_fraction``, ``n_refits``). In each
  refit and view, the component with the highest held-out R^2 alone is the *predictive column*;
  its support is recorded. ``variants`` repeats the refits with parameter overrides, e.g.
  ``{"collapse_weight=0": {"collapse_weight": 0.0}}``, to show how a setting changes the support.

Views are projected exactly as passed, so pass the same (preprocessed) arrays the model was
fitted on. The output contains only lists, numbers and strings, so it can be written to JSON
as it is.
`pysimlr.plotting.plot_component_attribution` draws it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

SUPPORT_TOL = 1e-6


def _np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=np.float64)


def _r2_from_pred(yte: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((yte - y_pred) ** 2))
    ss_tot = float(np.sum((yte - np.mean(yte)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _r2(ztr, ytr, zte, yte):
    ztr = np.asarray(ztr, dtype=np.float64)
    zte = np.asarray(zte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64).ravel()
    yte = np.asarray(yte, dtype=np.float64).ravel()
    ytr_mean = float(np.mean(ytr))
    if ztr.shape[1] == 0:
        return _r2_from_pred(yte, np.full_like(yte, ytr_mean))
    ztr_mean = np.mean(ztr, axis=0)
    ztr_c = ztr - ztr_mean
    ytr_c = ytr - ytr_mean
    if ztr.shape[1] == 1:
        denom = float(np.sum(ztr_c[:, 0] ** 2))
        slope = float((ztr_c[:, 0] @ ytr_c) / denom) if denom > 0.0 else 0.0
        return _r2_from_pred(yte, ytr_mean + (zte[:, 0] - ztr_mean[0]) * slope)
    beta = np.linalg.lstsq(ztr_c, ytr_c, rcond=None)[0]
    return _r2_from_pred(yte, ytr_mean + (zte - ztr_mean) @ beta)


def _r2_1d_cols(ztr: np.ndarray, ytr: np.ndarray, zte: np.ndarray, yte: np.ndarray) -> List[float]:
    """Vectorized 1-D OLS held-out ``R^2`` for every column of ``ztr`` / ``zte``."""
    d = ztr.shape[1]
    if d == 0:
        return []
    ztr_mean = np.mean(ztr, axis=0)
    ztr_c = ztr - ztr_mean
    ytr_mean = float(np.mean(ytr))
    ytr_c = ytr - ytr_mean
    denom = np.sum(ztr_c ** 2, axis=0)
    slope = np.where(denom > 0.0, (ztr_c.T @ ytr_c) / np.where(denom > 0.0, denom, 1.0), 0.0)
    preds = ytr_mean + (zte - ztr_mean) * slope
    ss_res = np.sum((yte[:, None] - preds) ** 2, axis=0)
    ss_tot = float(np.sum((yte - np.mean(yte)) ** 2))
    if ss_tot == 0.0:
        return [1.0 if float(sr) == 0.0 else 0.0 for sr in ss_res]
    return (1.0 - ss_res / ss_tot).tolist()


def _refit_task(args):
    method, k, params, train, rows, test, y_tr, y_te, views = args
    torch.set_num_threads(1)
    from .api import fit
    sub = [torch.as_tensor(np.asarray(x)[rows]).float() for x in train]
    if params.get("confounds") is not None:
        from .confounds import slice_confounds
        params = dict(params, confounds=slice_confounds(params["confounds"], rows))
    res = fit(method, sub, k, **params)
    # Score the views the refit saw: its own confound sink, if any, cleans them.
    from .confounds import clean_raw_views
    tr_c = clean_raw_views(res, [np.asarray(x, dtype=np.float64)[rows] for x in train])
    te_c = clean_raw_views(res, [None if x is None else np.asarray(x, dtype=np.float64) for x in test])
    y_tr_sub = y_tr[rows]
    out = {}
    for m in views:
        V = _np(res["v"][m])
        xs, xt = tr_c[m], te_c[m]
        r2 = _r2_1d_cols(xs @ V, y_tr_sub, xt @ V, y_te)
        j = int(np.argmax(r2))
        out[str(m)] = {"component": j + 1, "support": (V[:, j] > SUPPORT_TOL).astype(int).tolist(),
                       "r2": r2[j], "nnz": (V > SUPPORT_TOL).sum(0).astype(int).tolist()}
    return out


def component_attribution(result: Dict[str, Any], train_views: Sequence[Any], test_views: Sequence[Any],
                          outcome, outcome_test, *, views: Optional[Sequence[int]] = None,
                          view_names: Optional[Sequence[str]] = None, feature_names=None,
                          n_refits: int = 20, refit_fraction: float = 0.5,
                          refit_rows: Optional[Sequence[Sequence[int]]] = None,
                          variants: Optional[Mapping[str, Mapping[str, Any]]] = None,
                          refit_params: Optional[Mapping[str, Any]] = None,
                          random_state: int = 0, n_jobs: int = 8) -> Dict[str, Any]:
    """Held-out attribution of a fit's components, view spectra, and refit stability of the support.

    Parameters
    ----------
    result
        A `pysimlr.fit` result whose ``v`` are first-layer bases (linear `simlr`, or the linear
        entry of a deep/flow model).
    train_views, test_views
        The views the model was fitted on, and held-out rows of the same views, as the arrays
        that were fitted (they are projected as given). ``test_views`` entries may be ``None``
        for views you do not attribute.
    outcome, outcome_test
        The training and held-out outcome (1-D).
    views
        Views to attribute (default: every view present in ``test_views``).
    view_names, feature_names
        Optional labels; ``feature_names[m]`` lists view ``m``'s features.
    n_refits, refit_fraction, refit_rows
        Refits on ``refit_fraction`` of the training rows, drawn with ``random_state``; or the
        exact rows of each refit (``refit_rows``), which overrides both.
    variants
        ``{name: parameter overrides}``: each variant repeats the refits with those overrides.
        The refits without overrides are reported as ``"default"``.
    refit_params
        Parameters for the refits (default: ``result["parameters"]["used"]``, the fit's own).
    random_state, n_jobs
        Seed for the refit rows; worker processes.

    Returns
    -------
    A JSON-ready dict:
    * ``full_r2``.
    * ``components``, one entry per (view, component): ``view``, ``view_name``, ``component``
      (1-based), ``nnz``, ``alone``, ``unique``.
    * ``spectra``, per view: ``eigen_share``, ``mean_corr``, ``consensus_vs_mean``,
      ``hard_assignment``.
    * ``full_support``, per view: the full fit's predictive column support.
    * ``refits``: ``{variant: [ {view: {component, support, r2, nnz}} ]}``.
    * ``feature_names``, ``view_names``, ``views``, ``k``, ``n_train``, ``n_test``, ``n_refits``,
      ``refit_fraction``.
    """
    from .extension import view_names as _names
    from .relevance import _run
    train = [None if x is None else _np(x) for x in train_views]
    test = [None if x is None else _np(x) for x in test_views]
    n_views = len(result["v"])
    if len(train) != n_views or len(test) != n_views:
        raise ValueError(f"train_views and test_views need {n_views} entries (None for a missing test view); "
                         f"got {len(train)} and {len(test)}.")
    from .relevance import refuse_unobserved_rows
    refuse_unobserved_rows([None if x is None else torch.as_tensor(x) for x in train], "component_attribution(train_views)")
    refuse_unobserved_rows([None if x is None else torch.as_tensor(x) for x in test], "component_attribution(test_views)")
    views = [m for m in (views if views is not None else range(n_views)) if test[m] is not None]
    if not views:
        raise ValueError("no view to attribute: every test view is None.")
    y_tr, y_te = np.asarray(outcome, float).ravel(), np.asarray(outcome_test, float).ravel()
    if len(y_tr) != train[0].shape[0] or len(y_te) != test[views[0]].shape[0]:
        raise ValueError("outcome / outcome_test length does not match the views' rows.")
    vnames = list(view_names) if view_names is not None else list(_names(result))
    V = {m: _np(result["v"][m]) for m in views}
    k = V[views[0]].shape[1]
    cols = [(m, j) for m in views for j in range(k)]
    # The views as the fit saw them: its confound sink, if any, cleans train and test rows alike.
    from .confounds import clean_raw_views
    train_c, test_c = clean_raw_views(result, train), clean_raw_views(result, test)
    Ztr = np.hstack([train_c[m] @ V[m] for m in views]); Zte = np.hstack([test_c[m] @ V[m] for m in views])
    full = _r2(Ztr, y_tr, Zte, y_te)
    alone_all = _r2_1d_cols(Ztr, y_tr, Zte, y_te)
    comps = []
    for c, (m, j) in enumerate(cols):
        keep = [i for i in range(len(cols)) if i != c]
        comps.append({"view": m, "view_name": vnames[m], "component": j + 1, "nnz": int((V[m][:, j] > SUPPORT_TOL).sum()),
                      "alone": alone_all[c],
                      "unique": full - _r2(Ztr[:, keep], y_tr, Zte[:, keep], y_te)})
    rot = _np(result["loading_rotation"]) if result.get("loading_rotation") is not None else np.eye(k)
    U = _np(result["u"]) @ rot
    U_c = U - U.mean(axis=0, keepdims=True)
    u_norms = np.linalg.norm(U_c, axis=0)
    spectra, full_support = {}, {}
    for m in views:
        X = train_c[m]; n_x, p = X.shape
        X_c = X - X.mean(axis=0, keepdims=True)
        col_norms = np.linalg.norm(X_c, axis=0)
        if p > n_x and np.all(col_norms > 0.0):
            Z_corr = X_c / col_norms
            ev_nz = np.clip(np.linalg.eigvalsh(Z_corr @ Z_corr.T)[::-1], 0.0, None)
            ev = np.concatenate([ev_nz, np.zeros(p - n_x, dtype=np.float64)])
            c_sum = float(np.sum(Z_corr.sum(axis=1) ** 2))
        else:
            C = np.corrcoef(X.T)
            ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
            c_sum = float(C.sum())
        gm = X.mean(1)
        gm_c = gm - gm.mean()
        gm_norm = float(np.linalg.norm(gm_c))
        if gm_norm > 0.0 and np.all(u_norms > 0.0):
            cvm = ((U_c.T @ gm_c) / (u_norms * gm_norm)).tolist()
        else:
            cvm = [float(np.corrcoef(U[:, j], gm)[0, 1]) for j in range(k)]
        hard = None
        if result.get("loadings") is not None:
            L = _np(result["loadings"][m]); hard = (L > 0).sum(0).astype(int).tolist()
        spectra[str(m)] = {"eigen_share": (ev / ev.sum()).tolist(), "mean_corr": float((c_sum - p) / (p * (p - 1))),
                           "consensus_vs_mean": cvm,
                           "hard_assignment": hard}
        alone = [c["alone"] for c in comps if c["view"] == m]
        full_support[str(m)] = (V[m][:, int(np.argmax(alone))] > SUPPORT_TOL).astype(int).tolist()
    n = train[0].shape[0]
    if refit_rows is None:
        g = np.random.default_rng(random_state)
        refit_rows = [np.sort(g.choice(n, round(refit_fraction * n), replace=False)) for _ in range(n_refits)]
    refit_rows = [np.asarray(r, dtype=np.int64) for r in refit_rows]
    base = dict(refit_params if refit_params is not None else ((result.get("parameters") or {}).get("used") or {}))
    refits = {}
    for vname, over in {"default": {}, **dict(variants or {})}.items():
        params = dict(base, **dict(over))
        tasks = [(result["method"], k, params, train, rows, test, y_tr, y_te, views) for rows in refit_rows]
        refits[vname] = _run(_refit_task, tasks, n_jobs)
    fnames = {str(m): (list(feature_names[m]) if feature_names is not None else [f"v{m}.{i}" for i in range(train[m].shape[1])])
              for m in views}
    actual_frac = float(len(refit_rows[0]) / n) if refit_rows else float(round(refit_fraction * n) / n)
    return {"full_r2": full, "components": comps, "spectra": spectra, "full_support": full_support, "refits": refits,
            "feature_names": fnames, "view_names": vnames, "views": list(views), "k": int(k), "n_train": int(n),
            "n_test": len(y_te), "n_refits": len(refit_rows), "refit_fraction": actual_frac}
