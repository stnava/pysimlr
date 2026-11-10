r"""Fused ridge: a multi-task penalised GLM whose task coefficients are fused across a graph.

Task t has rows R_t, a response y_t, penalised feature coefficients w_t and
unpenalised covariate coefficients v_t (intercept included). With z = X~ w_t + C v_t,

.. math::

    \min_{w, v}\ \frac{1}{T}\sum_t \frac{1}{n_t}\sum_{i\in R_t} \ell(y_{ti}, z_{ti})
      + \lambda_1\sum_t \lVert w_t\rVert^2
      + \lambda_2\sum_{(a,b)\in E}\lVert w_a - w_b\rVert^2

with :math:`\ell` the negative log-likelihood of ``family`` (gaussian
:math:`\tfrac12(z-y)^2`, binomial, poisson :math:`e^z - yz`), X~ the
standardised features (``standardize=True``: centre and scale with the
sample sd, as R's ``scale``), and E the fusion graph (a chain by default).
For outcome thresholds every task uses all N rows, so the data term is
:math:`\frac{1}{NJ}\sum\ell`, the objective of ``ANTsR::fusedRidge``.

Two ways to build the tasks (exactly one is given):

``groups``      **strata**: R_t = rows of group t (a label per row, or boolean
                masks, which may overlap); y_t = y.
``thresholds``  **outcome thresholds** (the ANTsR model): R_t = all rows;
                y_t = 1[y >= c_t]; ``family="binomial"`` by default, so
                neighbouring thresholds share slopes as lambda2 grows
                (cumulative-logit-like).

Solved by Newton/IRLS on the exact sparse block system, with step halving.
Gaussian converges in one step (``stop_reason="closed_form"``).

``top_k`` (``ANTsR`` ``topK``): in the fit, take each task's K largest
|standardised coefficient|, take the union across tasks, refit on those
columns and embed with zeros; in `predict_fused_ridge`, keep each task's K
largest coefficients and zero the rest, without a refit.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .defaults import DEFAULT_TOL

__all__ = ["fit_fused_ridge", "predict_fused_ridge", "fused_ridge_cv", "FAMILIES"]

FAMILIES = ("gaussian", "binomial", "poisson")
_DEFAULTS = dict(groups=None, thresholds=None, family=None, covariates=None, lambda1=0.1,
                 lambda2=1.0, fusion="chain", standardize=True, top_k=None, feature_names=None,
                 max_iter=100, tol=DEFAULT_TOL)


# ----------------------------------------------------------------------------- tasks
def _masks(groups, n: int) -> Tuple[List[np.ndarray], List[Any]]:
    if isinstance(groups, (list, tuple)) and len(groups) > 0 and np.ndim(groups[0]) == 1 \
            and len(groups[0]) == n and np.asarray(groups[0]).dtype == bool:
        return [np.asarray(m, bool) for m in groups], list(range(len(groups)))
    labels = np.asarray(groups)
    if labels.shape != (n,):
        raise ValueError(f"groups must be a label per row (length {n}) or a list of boolean masks.")
    names = list(dict.fromkeys(labels.tolist()))
    return [labels == g for g in names], names


def _tasks(y, n, groups, thresholds):
    """(model, task names, row masks, per-task responses)."""
    if (groups is None) == (thresholds is None):
        raise ValueError("give exactly one of groups= (strata) or thresholds= (outcome thresholds).")
    if groups is not None:
        masks, names = _masks(groups, n)
        return "strata", names, masks, [y] * len(masks)
    c = np.asarray(thresholds, float).ravel()
    if c.size < 1:
        raise ValueError("thresholds must be non-empty.")
    return "thresholds", c.tolist(), [np.ones(n, bool)] * c.size, [(y >= t).astype(float) for t in c]


def _edges(n_tasks: int, fusion) -> List[Tuple[int, int]]:
    if isinstance(fusion, str):
        if fusion == "chain":
            return [(t, t + 1) for t in range(n_tasks - 1)]
        if fusion == "all":
            return [(a, b) for a in range(n_tasks) for b in range(a + 1, n_tasks)]
        raise ValueError(f"fusion must be 'chain', 'all' or a list of edges; got {fusion!r}.")
    edges = [(int(a), int(b)) for a, b in fusion]
    for a, b in edges:
        if not (0 <= a < n_tasks and 0 <= b < n_tasks) or a == b:
            raise ValueError(f"invalid fusion edge {(a, b)} for {n_tasks} tasks.")
    return edges


# ----------------------------------------------------------------------------- family
def _mean(z, family):
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-z))
    if family == "poisson":
        return np.exp(np.minimum(z, 50.0))
    return z


def _loss(y, z, family):
    """Per-row negative log-likelihood (the R implementation's terms)."""
    if family == "binomial":
        return np.logaddexp(0.0, z) - y * z
    if family == "poisson":
        return np.exp(np.minimum(z, 50.0)) - y * z
    return 0.5 * (z - y) ** 2


def _weight(mu, family):
    if family == "binomial":
        return mu * (1.0 - mu)
    if family == "poisson":
        return mu
    return np.ones_like(mu)


# ----------------------------------------------------------------------------- solver
#: Newton systems up to this size (T * (p + q) unknowns) are solved dense. The
#: task blocks X_t' D X_t are dense, so SuperLU on the sparse form fills in and
#: was ~110 ms per step at T=5, p=300 (35 s of a 39 s CV), against a few ms for
#: a dense Cholesky solve. 4000 unknowns is a 128 MB float64 matrix.
DENSE_SOLVE_MAX = 4000


def _build_task_data(Xs, Cs, ys, masks, edges, family):
    T = len(masks)
    Xt = [np.hstack([Xs[m], Cs[m]]) for m in masks]
    yt = [ys[t][m] for t, m in enumerate(masks)]
    scale = [1.0 / (T * m.sum()) for m in masks]
    lap = np.zeros((T, T))
    for a, b in edges:
        lap[a, a] += 1; lap[b, b] += 1; lap[a, b] -= 1; lap[b, a] -= 1
    grams = [scale[t] * (Xt[t].T @ Xt[t]) for t in range(T)] if family == "gaussian" else None
    rhs = [scale[t] * (Xt[t].T @ yt[t]) for t in range(T)] if family == "gaussian" else None
    yty = sum(0.5 * scale[t] * float(yt[t] @ yt[t]) for t in range(T)) if family == "gaussian" else None
    return {"Xt": Xt, "yt": yt, "scale": scale, "lap": lap, "grams": grams, "rhs": rhs, "yty": yty}


def _subset_task_data(td, cols, p_full: int, q: int):
    idx = np.concatenate([cols, p_full + np.arange(q)])
    Xt_sub = [xt[:, idx] for xt in td["Xt"]]
    grams_sub = [G[np.ix_(idx, idx)] for G in td["grams"]] if td["grams"] is not None else None
    rhs_sub = [r[idx] for r in td["rhs"]] if td["rhs"] is not None else None
    return {"Xt": Xt_sub, "yt": td["yt"], "scale": td["scale"], "lap": td["lap"],
            "grams": grams_sub, "rhs": rhs_sub, "yty": td["yty"]}


def _solve(Xs, Cs, ys, masks, family, lambda1, lambda2, edges, max_iter, tol, *, _task_data=None):
    """Newton on the penalised objective. Returns (W (T,p), V (T,q), objective, n_steps, stop)."""
    import scipy.linalg
    T = len(masks); p = Xs.shape[1]; q = Cs.shape[1]; d = p + q
    td = _build_task_data(Xs, Cs, ys, masks, edges, family) if _task_data is None else _task_data
    Xt, yt, scale, lap = td["Xt"], td["yt"], td["scale"], td["lap"]
    pen_coeff = 2.0 * lambda1 * np.eye(T) + 2.0 * lambda2 * lap
    use_dense = (T * d <= DENSE_SOLVE_MAX)

    if use_dense:
        Hbase = np.zeros((T * d, T * d), dtype=np.float64)
        idx_p = np.arange(p)
        for a in range(T):
            for b in range(T):
                val = pen_coeff[a, b]
                if val != 0.0:
                    Hbase[a * d + idx_p, b * d + idx_p] = val
        Hbase.flat[::T * d + 1] += 1e-12
        Hpen = None
    else:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        pen_mask = np.concatenate([np.ones(p), np.zeros(q)])           # w penalised, v not
        P = sparse.kron(sparse.identity(T), sparse.diags(pen_mask))
        Lw = sparse.kron(sparse.csr_matrix(lap), sparse.diags(pen_mask))
        Hpen = 2.0 * lambda1 * P + 2.0 * lambda2 * Lw
        Hbase = None

    def _pen_grad_and_quad(th):
        th2 = th.reshape(T, d)
        gw = pen_coeff @ th2[:, :p]
        g_out = np.zeros((T, d), dtype=np.float64)
        g_out[:, :p] = gw
        return g_out.ravel(), 0.5 * float(np.sum(th2[:, :p] * gw))

    theta = np.zeros(T * d)
    if family != "gaussian":
        # Start each task's intercept (the last covariate column, added by
        # the caller) at its mean response on the link scale.
        for t in range(T):
            mu0 = float(yt[t].mean())
            if family == "binomial":
                mu0 = min(max(mu0, 1e-6), 1 - 1e-6); theta[t * d + d - 1] = np.log(mu0 / (1 - mu0))
            elif family == "poisson":
                theta[t * d + d - 1] = np.log(max(mu0, 1e-6))

    def objective(th):
        f = 0.0
        for t in range(T):
            z = Xt[t] @ th[t * d:(t + 1) * d]
            f += scale[t] * _loss(yt[t], z, family).sum()
        return f + _pen_grad_and_quad(th)[1]

    f = float(td["yty"]) if (family == "gaussian" and td["yty"] is not None) else objective(theta)
    stop, steps = "max_iter", 0
    for it in range(1, int(max_iter) + 1):
        steps = it
        if family == "gaussian" and use_dense and td["grams"] is not None:
            g = np.empty(T * d, dtype=np.float64)
            Hd = Hbase.copy()
            for t in range(T):
                sl = slice(t * d, (t + 1) * d)
                g[sl] = -td["rhs"][t]
                Hd[sl, sl] += td["grams"][t]
            try:
                step = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Hd, check_finite=False), -g,
                                              check_finite=False)
            except np.linalg.LinAlgError:
                step = scipy.linalg.solve(Hd, -g, assume_a="sym", check_finite=False)
        elif use_dense:
            g, _ = _pen_grad_and_quad(theta)
            Hd = Hbase.copy()
            for t in range(T):
                sl = slice(t * d, (t + 1) * d)
                z = Xt[t] @ theta[sl]; mu = _mean(z, family)
                g[sl] += scale[t] * (Xt[t].T @ (mu - yt[t]))
                Xw = Xt[t] * np.sqrt(scale[t] * _weight(mu, family))[:, None]
                Hd[sl, sl] += Xw.T @ Xw
            try:      # H is positive definite whenever the covariates have full rank
                step = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Hd, check_finite=False), -g,
                                              check_finite=False)
            except np.linalg.LinAlgError:   # rank-deficient covariates: same system, LDL'
                step = scipy.linalg.solve(Hd, -g, assume_a="sym", check_finite=False)
        else:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
            g = Hpen @ theta
            blocks = []
            for t in range(T):
                z = Xt[t] @ theta[t * d:(t + 1) * d]; mu = _mean(z, family)
                g[t * d:(t + 1) * d] += scale[t] * (Xt[t].T @ (mu - yt[t]))
                blocks.append(scale[t] * (Xt[t].T * _weight(mu, family)) @ Xt[t])
            H = sparse.block_diag(blocks) + Hpen + 1e-12 * sparse.identity(T * d)
            step = np.asarray(spsolve(sparse.csc_matrix(H), -g))
        if not np.all(np.isfinite(step)):
            stop = "non_finite"; break
        a, f_new = 1.0, objective(theta + step)
        while f_new > f + 1e-15 * max(1.0, abs(f)) and a > 1e-8:
            a *= 0.5; f_new = objective(theta + a * step)
        theta = theta + a * step
        if family == "gaussian":
            f = f_new; stop = "closed_form"; break
        rel = abs(f - f_new) / max(1.0, abs(f))
        f = f_new
        if rel < tol:
            stop = "converged"; break
    th = theta.reshape(T, d)
    return th[:, :p], th[:, p:], f, steps, stop


# ----------------------------------------------------------------------------- fit
def _prep(X, covariates, n, standardize, center=None, sd=None):
    X = np.asarray(X, float)
    if standardize:
        if center is None:
            center = X.mean(0); sd = X.std(0, ddof=1); sd = np.where(sd > 0, sd, 1.0)
        Xs = (X - center) / sd
    else:
        center, sd, Xs = np.zeros(X.shape[1]), np.ones(X.shape[1]), X
    C = np.zeros((n, 0)) if covariates is None else np.asarray(covariates, float).reshape(n, -1)
    return Xs, np.hstack([C, np.ones((n, 1))]), center, sd


def fit_fused_ridge(X, y, *, groups=None, thresholds=None, family: Optional[str] = None,
                    covariates=None, lambda1: float = 0.1, lambda2: float = 1.0,
                    fusion: Union[str, Sequence[Tuple[int, int]]] = "chain",
                    standardize: bool = True, top_k: Optional[int] = None,
                    feature_names: Optional[Sequence[str]] = None, max_iter: int = 100,
                    tol: float = DEFAULT_TOL, _cv_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fit a fused ridge (see the module docstring).

    ``family`` defaults to ``"binomial"`` for ``thresholds`` and
    ``"gaussian"`` for ``groups``. Returns a result dict: ``coef`` (T, p) on
    the original feature scale, ``coef_std`` on the standardised scale (what
    ``top_k`` ranks and the plots show), ``cov_coef`` (T, K+1; last column the
    intercept on the standardised scale), ``intercept`` (T,) on the original
    scale, ``tasks``, ``model``, ``family``, ``edges``, ``selected_features``,
    ``deviance`` (per-task mean loss), ``convergence`` and ``parameters``.
    """
    from .utils import warn_if_unconverged
    args = dict(groups=groups, thresholds=thresholds, family=family, covariates=covariates,
                lambda1=lambda1, lambda2=lambda2, fusion=fusion, standardize=standardize,
                top_k=top_k, feature_names=feature_names, max_iter=max_iter, tol=tol)
    if not lambda1 > 0:
        raise ValueError(f"lambda1 must be > 0 (it makes the system positive definite); got {lambda1}.")
    if lambda2 < 0:
        raise ValueError(f"lambda2 must be >= 0; got {lambda2}.")
    if _cv_cache is not None and "prep" in _cv_cache:
        model, names, masks, ys, family, edges, Xs, C, center, sd, td, n, p, n_per_task, fnames = _cv_cache["prep"]
    else:
        X = np.asarray(X, float); y = np.asarray(y, float).ravel(); n, p = X.shape
        if y.shape[0] != n:
            raise ValueError(f"X has {n} rows but y has {y.shape[0]}.")
        model, names, masks, ys = _tasks(y, n, groups, thresholds)
        family = family or ("binomial" if model == "thresholds" else "gaussian")
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}; got {family!r}.")
        if any(m.sum() < 2 for m in masks):
            raise ValueError("every task needs at least 2 rows.")
        edges = _edges(len(masks), fusion)
        Xs, C, center, sd = _prep(X, covariates, n, standardize)
        td = _build_task_data(Xs, C, ys, masks, edges, family)
        n_per_task = [int(m.sum()) for m in masks]
        fnames = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(p)]
        if _cv_cache is not None:
            _cv_cache["prep"] = (model, names, masks, ys, family, edges, Xs, C, center, sd, td, n, p, n_per_task, fnames)
            _cv_cache["base_solves"] = {}
    cols = np.arange(p)
    base_key = (float(lambda1), float(lambda2))
    if _cv_cache is not None and base_key in _cv_cache["base_solves"]:
        W, V, f, steps, stop = _cv_cache["base_solves"][base_key]
        W, V = W.copy(), V.copy()
    else:
        W, V, f, steps, stop = _solve(Xs, C, ys, masks, family, lambda1, lambda2, edges, max_iter, tol, _task_data=td)
        if _cv_cache is not None:
            _cv_cache["base_solves"][base_key] = (W.copy(), V.copy(), f, steps, stop)
    if top_k is not None:
        k = int(top_k)
        if k <= 0:
            raise ValueError("top_k must be a positive integer.")
        sel = set()
        for t in range(W.shape[0]):
            sel.update(np.argsort(-np.abs(W[t]))[:min(k, p)].tolist())
        cols = np.array(sorted(sel))
        td_sub = _subset_task_data(td, cols, p, C.shape[1])
        Ws, V, f, steps, stop = _solve(Xs[:, cols], C, ys, masks, family, lambda1, lambda2, edges,
                                       max_iter, tol, _task_data=td_sub)
        W = np.zeros_like(W); W[:, cols] = Ws
    coef = W / sd[None, :]
    intercept = V[:, -1] - coef @ center
    dev = []
    for t in range(len(masks)):
        th_t = np.concatenate([W[t], V[t]])
        z = td["Xt"][t] @ th_t
        dev.append(float(_loss(td["yt"][t], z, family).mean()))
    conv = {"converged": stop in ("converged", "closed_form"), "stop_reason": stop,
            "n_steps": steps, "max_steps": int(max_iter), "tol": float(tol), "final": float(f)}
    warn_if_unconverged(conv, "fused_ridge")
    return {"method": "fused_ridge", "model": model, "family": family, "tasks": names,
            "coef": coef, "coef_std": W, "cov_coef": V, "intercept": intercept,
            "x_center": center, "x_scale": sd, "standardize": bool(standardize),
            "edges": edges, "lambda1": float(lambda1), "lambda2": float(lambda2),
            "top_k": top_k, "selected_features": [int(c) for c in cols],
            "selected_feature_names": [fnames[c] for c in cols], "feature_names": fnames,
            "n_per_task": n_per_task, "deviance": dev,
            "n_covariates": int(C.shape[1] - 1), "convergence": conv,
            "parameters": {"used": args, "defaults": dict(_DEFAULTS)}}


def predict_fused_ridge(result: Dict[str, Any], X, covariates=None, task=None,
                        type: str = "link", top_k: Optional[int] = None) -> np.ndarray:
    """Predict with a fitted fused ridge.

    ``task``: ``None`` (an (n, T) array, every row under every task), one task
    name (one column) or a name per row. ``type``: ``"link"`` or
    ``"response"`` (probability for binomial, rate for poisson; for
    thresholds, P(y >= c_t)). ``top_k``: keep each task's K largest
    standardised coefficients and zero the rest (no refit).
    """
    if type not in ("link", "response"):
        raise ValueError(f"type must be 'link' or 'response'; got {type!r}.")
    X = np.asarray(X, float); n = X.shape[0]
    if covariates is None and result["n_covariates"] > 0:
        raise ValueError("this model was fitted with covariates; pass covariates=.")
    Xs, C, _, _ = _prep(X, covariates, n, result["standardize"], result["x_center"], result["x_scale"])
    W = np.array(result["coef_std"], float)
    if top_k is not None:
        k = int(top_k)
        if k <= 0:
            raise ValueError("top_k must be a positive integer.")
        for t in range(W.shape[0]):
            W[t, np.argsort(-np.abs(W[t]))[k:]] = 0.0
    z = Xs @ W.T + C @ np.asarray(result["cov_coef"]).T                # (n, T)
    out = _mean(z, result["family"]) if type == "response" else z
    if task is None:
        return out
    idx = {t: i for i, t in enumerate(result["tasks"])}
    if np.ndim(task) == 0:
        if task not in idx:
            raise ValueError(f"unknown task {task!r}; tasks are {result['tasks']}.")
        return out[:, idx[task]]
    task = list(np.asarray(task).tolist())
    if len(task) != n:
        raise ValueError("task must be one name or one name per row.")
    bad = sorted(set(task) - set(idx), key=str)
    if bad:
        raise ValueError(f"unknown tasks {bad}; tasks are {result['tasks']}.")
    return out[np.arange(n), [idx[t] for t in task]]


def fused_ridge_cv(X, y, *, groups=None, thresholds=None, family=None, covariates=None,
                   lambda1s: Sequence[float] = (0.01, 0.1, 1.0),
                   lambda2s: Sequence[float] = (0.0, 0.1, 1.0, 10.0),
                   top_ks: Sequence[Optional[int]] = (None,), n_folds: int = 5, foldid=None,
                   fusion="chain", standardize: bool = True, random_state: int = 0,
                   feature_names=None, max_iter: int = 100, tol: float = DEFAULT_TOL) -> Dict[str, Any]:
    """Choose (lambda1, lambda2, top_k) by K-fold cross-validated mean loss.

    Folds are over rows (``foldid`` overrides: one integer label per row).
    Each training fold refits the whole model, ``top_k`` selection included,
    so held-out loss is not biased by selecting on all rows. A held-out row is
    scored under every task it belongs to. Returns ``best`` (lambda1, lambda2,
    top_k), ``loss`` and ``se`` arrays (len(lambda1s), len(lambda2s),
    len(top_ks)), the grids, and ``result`` refit on all rows at ``best``.
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel(); n = X.shape[0]
    model, names, masks, ys = _tasks(y, n, groups, thresholds)
    fam = family or ("binomial" if model == "thresholds" else "gaussian")
    if foldid is None:
        fold = np.random.default_rng(random_state).permutation(n) % int(n_folds)
    else:
        fold = np.asarray(foldid)
        if fold.shape != (n,):
            raise ValueError("foldid must give one fold label per row.")
    folds = list(dict.fromkeys(fold.tolist()))
    C = None if covariates is None else np.asarray(covariates, float).reshape(n, -1)
    loss = np.zeros((len(lambda1s), len(lambda2s), len(top_ks), len(folds)))
    for fi, f in enumerate(folds):
        tr, te = fold != f, fold == f
        kw = dict(family=fam, fusion=fusion, standardize=standardize, max_iter=max_iter, tol=tol)
        if model == "strata":
            kw["groups"] = [m[tr] for m in masks]
        else:
            kw["thresholds"] = names
        memb = np.stack([m[te] for m in masks], 1)
        yte = np.stack([yy[te] for yy in ys], 1)
        X_tr, y_tr = X[tr], y[tr]
        C_tr = None if C is None else C[tr]
        X_te = X[te]
        C_te = None if C is None else C[te]
        fold_cache: Dict[str, Any] = {}
        for i, l1 in enumerate(lambda1s):
            for j, l2 in enumerate(lambda2s):
                for k, tk in enumerate(top_ks):
                    r = fit_fused_ridge(X_tr, y_tr, covariates=C_tr,
                                        lambda1=l1, lambda2=l2, top_k=tk, _cv_cache=fold_cache, **kw)
                    z = predict_fused_ridge(r, X_te, covariates=C_te)
                    loss[i, j, k, fi] = _loss(yte, z, fam)[memb].mean()
    mean = loss.mean(-1)
    se = loss.std(-1, ddof=1) / np.sqrt(len(folds)) if len(folds) > 1 else np.zeros_like(mean)
    i, j, k = np.unravel_index(np.argmin(mean), mean.shape)
    best = (float(lambda1s[i]), float(lambda2s[j]), top_ks[k])
    full = dict(family=fam, covariates=covariates, fusion=fusion, standardize=standardize,
                feature_names=feature_names, max_iter=max_iter, tol=tol)
    full["groups" if model == "strata" else "thresholds"] = groups if model == "strata" else thresholds
    return {"best": best, "loss": mean, "se": se, "lambda1s": list(lambda1s),
            "lambda2s": list(lambda2s), "top_ks": list(top_ks), "family": fam, "model": model,
            "result": fit_fused_ridge(X, y, lambda1=best[0], lambda2=best[1], top_k=best[2], **full)}
