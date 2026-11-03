r"""Fused ridge regression: one coefficient vector per group, fused across neighbouring groups.

For groups :math:`g = 1..G` (strata, sites, ordered exposure thresholds), with
rows :math:`(X_g, y_g)` centred within each group, solve

.. math::

    \min_{w_1..w_G} \sum_g \frac{1}{n_g}\lVert y_g - X_g w_g\rVert^2
      + \lambda_1 \sum_g \lVert w_g\rVert^2
      + \lambda_2 \sum_{(a,b)\in E} \lVert w_a - w_b\rVert^2

where :math:`E` is the fusion graph (a chain over the given group order by
default). :math:`\lambda_2 = 0` gives independent per-group ridges;
:math:`\lambda_2 \to \infty` gives one shared ridge. Each group has its own
unpenalised intercept. The objective is a strictly convex quadratic for
:math:`\lambda_1 > 0`. It is solved exactly as one sparse linear system,
:math:`(\mathrm{blockdiag}(X_g^\top X_g/n_g) + \lambda_1 I + \lambda_2 L\otimes I)\,w = b`,
with :math:`L` the graph Laplacian of :math:`E`.

Groups are either a label per row, or a list of boolean row masks (masks may
overlap: for example the nested "exposure above percentile q" strata this was
first written for).
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["fit_fused_ridge", "predict_fused_ridge", "fused_ridge_cv"]


def _masks(groups, n: int) -> Tuple[List[np.ndarray], List[Any]]:
    """Normalise ``groups`` to (list of boolean masks, group names)."""
    if isinstance(groups, (list, tuple)) and len(groups) > 0 and np.ndim(groups[0]) == 1 \
            and len(groups[0]) == n and np.asarray(groups[0]).dtype == bool:
        masks = [np.asarray(m, bool) for m in groups]
        return masks, list(range(len(masks)))
    labels = np.asarray(groups)
    if labels.shape != (n,):
        raise ValueError(f"groups must be a label per row (length {n}) or a list of boolean masks.")
    names = list(dict.fromkeys(labels.tolist()))  # order of first appearance
    return [labels == g for g in names], names


def _edges(n_groups: int, fusion) -> List[Tuple[int, int]]:
    if fusion == "chain":
        return [(g, g + 1) for g in range(n_groups - 1)]
    if fusion == "all":
        return [(a, b) for a in range(n_groups) for b in range(a + 1, n_groups)]
    edges = [(int(a), int(b)) for a, b in fusion]
    for a, b in edges:
        if not (0 <= a < n_groups and 0 <= b < n_groups) or a == b:
            raise ValueError(f"invalid fusion edge {(a, b)} for {n_groups} groups.")
    return edges


def fit_fused_ridge(X, y, groups, *, lambda1: float = 0.1, lambda2: float = 1.0,
                    fusion: Union[str, Sequence[Tuple[int, int]]] = "chain",
                    feature_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Fit a fused ridge.

    Parameters
    ----------
    X : (n, p) array
    y : (n,) array
    groups : length-n labels, or list of G boolean row masks (may overlap)
    lambda1 : ridge penalty (> 0)
    lambda2 : fusion penalty (>= 0)
    fusion : "chain" (neighbours in group order), "all" (every pair), or a
        list of (a, b) group-index edges
    feature_names : optional names, kept for plotting

    Returns
    -------
    dict with ``coef`` (G, p), ``intercept`` (G,), ``x_mean`` (G, p),
    ``groups`` (names), ``edges``, ``lambda1``, ``lambda2``, ``n_per_group``,
    ``train_r2`` (per group) and ``feature_names``.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X has {n} rows but y has {y.shape[0]}.")
    if not lambda1 > 0:
        raise ValueError(f"lambda1 must be > 0 (it makes the system positive definite); got {lambda1}.")
    if lambda2 < 0:
        raise ValueError(f"lambda2 must be >= 0; got {lambda2}.")
    masks, names = _masks(groups, n)
    G = len(masks)
    if any(m.sum() < 2 for m in masks):
        raise ValueError("every group needs at least 2 rows.")
    edges = _edges(G, fusion)
    xm = np.stack([X[m].mean(0) for m in masks]); ym = np.array([y[m].mean() for m in masks])
    blocks, b = [], np.zeros(G * p)
    for g, m in enumerate(masks):
        Z = X[m] - xm[g]; t = y[m] - ym[g]; ng = m.sum()
        blocks.append(Z.T @ Z / ng)
        b[g * p:(g + 1) * p] = Z.T @ t / ng
    lap = np.zeros((G, G))
    for a, c in edges:
        lap[a, a] += 1; lap[c, c] += 1; lap[a, c] -= 1; lap[c, a] -= 1
    A = (sparse.block_diag(blocks) + lambda1 * sparse.identity(G * p)
         + lambda2 * sparse.kron(sparse.csr_matrix(lap), sparse.identity(p)))
    w = np.asarray(spsolve(sparse.csc_matrix(A), b)).reshape(G, p)
    intercept = ym - np.einsum("gp,gp->g", xm, w)
    r2 = []
    for g, m in enumerate(masks):
        resid = y[m] - (X[m] @ w[g] + intercept[g]); tot = ((y[m] - ym[g]) ** 2).sum()
        r2.append(float(1 - (resid ** 2).sum() / tot) if tot > 0 else float("nan"))
    return {"coef": w, "intercept": intercept, "x_mean": xm, "groups": names, "edges": edges,
            "lambda1": float(lambda1), "lambda2": float(lambda2),
            "n_per_group": [int(m.sum()) for m in masks], "train_r2": r2,
            "feature_names": list(feature_names) if feature_names is not None else None}


def predict_fused_ridge(result: Dict[str, Any], X, group=None) -> np.ndarray:
    """Predict with a fitted fused ridge.

    ``group`` is one group name (every row uses that group's model), a
    length-n array of group names (each row uses its own group), or ``None``,
    which returns an (n, G) array: every row under every group's model.
    """
    X = np.asarray(X, float)
    preds = X @ result["coef"].T + result["intercept"][None, :]  # (n, G)
    if group is None:
        return preds
    names = result["groups"]
    idx = {g: i for i, g in enumerate(names)}
    if np.ndim(group) == 0:
        if group not in idx:
            raise ValueError(f"unknown group {group!r}; groups are {names}.")
        return preds[:, idx[group]]
    group = np.asarray(group)
    if group.shape != (X.shape[0],):
        raise ValueError("group must be a single name or one name per row.")
    unknown = sorted(set(group.tolist()) - set(idx))
    if unknown:
        raise ValueError(f"unknown groups {unknown}; groups are {names}.")
    return preds[np.arange(X.shape[0]), [idx[g] for g in group.tolist()]]


def fused_ridge_cv(X, y, groups, *, lambda1s: Sequence[float] = (0.01, 0.1, 1.0),
                   lambda2s: Sequence[float] = (0.0, 0.1, 1.0, 10.0), n_folds: int = 5,
                   fusion="chain", random_state: int = 0) -> Dict[str, Any]:
    """Choose (lambda1, lambda2) by K-fold cross-validated mean squared error.

    Folds are drawn over rows. A held-out row is scored under every group it
    belongs to (overlapping masks count each membership). Returns ``best``
    (lambda1, lambda2), ``mse`` (len(lambda1s), len(lambda2s)) and its
    standard error, the grids, and ``result``: the refit on all rows at
    ``best``.
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel(); n = X.shape[0]
    masks, _ = _masks(groups, n)
    rng = np.random.default_rng(random_state)
    fold = rng.permutation(n) % int(n_folds)
    err = np.zeros((len(lambda1s), len(lambda2s), int(n_folds)))
    for f in range(int(n_folds)):
        tr, te = fold != f, fold == f
        tr_masks = [m & tr for m in masks]
        for i, l1 in enumerate(lambda1s):
            for j, l2 in enumerate(lambda2s):
                r = fit_fused_ridge(X, y, tr_masks, lambda1=l1, lambda2=l2, fusion=fusion)
                pred = predict_fused_ridge(r, X[te])  # (n_te, G)
                memb = np.stack([m[te] for m in masks], 1)
                e = (pred - y[te][:, None]) ** 2
                err[i, j, f] = e[memb].mean()
    mse = err.mean(2); se = err.std(2, ddof=1) / np.sqrt(int(n_folds))
    i, j = np.unravel_index(np.argmin(mse), mse.shape)
    best = (float(lambda1s[i]), float(lambda2s[j]))
    return {"best": best, "mse": mse, "se": se, "lambda1s": list(lambda1s), "lambda2s": list(lambda2s),
            "result": fit_fused_ridge(X, y, groups, lambda1=best[0], lambda2=best[1], fusion=fusion)}
