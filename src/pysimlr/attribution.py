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

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

SUPPORT_TOL = 1e-6


def _np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=np.float64)


def _r2(ztr, ytr, zte, yte):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    return float(r2_score(yte, LinearRegression().fit(ztr, ytr).predict(zte)))


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
    out = {}
    for m in views:
        V = _np(res["v"][m])
        xs, xt = tr_c[m], te_c[m]
        r2 = [_r2(xs @ V[:, [j]], y_tr[rows], xt @ V[:, [j]], y_te) for j in range(V.shape[1])]
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
    comps = []
    for c, (m, j) in enumerate(cols):
        keep = [i for i in range(len(cols)) if i != c]
        comps.append({"view": m, "view_name": vnames[m], "component": j + 1, "nnz": int((V[m][:, j] > SUPPORT_TOL).sum()),
                      "alone": _r2(Ztr[:, [c]], y_tr, Zte[:, [c]], y_te),
                      "unique": full - _r2(Ztr[:, keep], y_tr, Zte[:, keep], y_te)})
    rot = _np(result["loading_rotation"]) if result.get("loading_rotation") is not None else np.eye(k)
    U = _np(result["u"]) @ rot
    spectra, full_support = {}, {}
    for m in views:
        X = train_c[m]; p = X.shape[1]
        C = np.corrcoef(X.T); ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
        gm = X.mean(1)
        hard = None
        if result.get("loadings") is not None:
            L = _np(result["loadings"][m]); hard = (L > 0).sum(0).astype(int).tolist()
        spectra[str(m)] = {"eigen_share": (ev / ev.sum()).tolist(), "mean_corr": float((C.sum() - p) / (p * (p - 1))),
                           "consensus_vs_mean": [float(np.corrcoef(U[:, j], gm)[0, 1]) for j in range(k)],
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
    return {"full_r2": full, "components": comps, "spectra": spectra, "full_support": full_support, "refits": refits,
            "feature_names": fnames, "view_names": vnames, "views": list(views), "k": int(k), "n_train": int(n),
            "n_test": len(y_te), "n_refits": len(refit_rows), "refit_fraction": float(len(refit_rows[0]) / n)}
