"""Classical observed-variable path models (structural equation models with fit indices).

When to use which
-----------------
``pysimlr.sem`` (``path_model``, ``path_model_from_scores``, ``compare_path_models*``) tests graphs on
multi-column score BLOCKS by permutation (Pillai statistics). It gives permutation p-values, not classical
fit indices. This module is the complementary classical mode: every node is ONE observed variable (a
single measurement or a one-column score), the model is a recursive Gaussian path model, and it returns
the usual fit statistics (chi-square, RMSEA with interval, CFI, SRMR, AIC, BIC, weights). Use ``pysimlr.sem``
when nodes are blocks of several columns or when the Gaussian likelihood is doubtful; use this module for
single-variable nodes when chi-square based fit indices and model weights are wanted.

Model and estimator
-------------------
Variables are the columns of ``data`` (standardized on the rows used when ``standardize=True``, so the fit
works on the correlation matrix ``S``; otherwise ``S`` is the ML covariance, divisor ``n``). For ``edges``
(from, to) the recursive model is ``x = B x + e`` with ``cov(e) = Psi`` and implied covariance
``Sigma = (I - B)^-1 Psi (I - B)^-T``. ``Psi`` is diagonal for endogenous nodes, plus the requested
``residual_covariances``; exogenous nodes (``exogenous``) have their covariance among themselves fixed at
the sample value (their variances and covariances are counted as free parameters, as in the usual
treatment of fixed-x models). Nodes that appear in no edge are modelled as unrelated to everything.

* Discrepancy: Gaussian ML, ``F = log|Sigma| + tr(S Sigma^-1) - log|S| - p``.
* Estimator: when there is no residual covariance (or exactly one between two equations with identical
  predictor sets) the ML estimate is equation-by-equation least squares and is returned in closed form
  (``closed_form=True``). Otherwise ``F`` is minimized with scipy BFGS (gtol 1e-10, maxiter 2000) over
  coefficients, log residual variances and ``arctanh`` residual correlations, started at the least-squares
  solution. ``converged`` is True for closed form, and for BFGS when the optimizer reports success or
  the largest absolute gradient is below 1e-5; otherwise a ``RuntimeWarning`` is issued.
* ``chi2 = (n - 1) F`` against the saturated model, ``df = p(p+1)/2 - q`` with ``q`` the number of free
  parameters (coefficients, endogenous residual variances, residual covariances, and the
  ``ne(ne+1)/2`` variances and covariances of the ``ne`` exogenous nodes). ``p`` is the upper-tail
  chi-square probability.
* RMSEA ``sqrt(max(chi2 - df, 0) / (df (n - 1)))`` with a 90% interval from the noncentral chi-square.
* CFI ``1 - max(chi2 - df, 0) / max(chi2_null - df_null, chi2 - df, 1e-12)`` against the independence
  (diagonal) model, ``chi2_null = (n - 1) (sum log diag(S) - log|S|)`` and ``df_null = p(p-1)/2``.
* SRMR: root of the sum of squared elements of the lower triangle (diagonal included) of the
  correlation residuals ``corr(S) - Sigma / sqrt(diag(Sigma) diag(Sigma)^T)``, divided by ``p(p+1)/2``.
* AIC ``= n F + 2 q`` and BIC ``= n F + q ln n``: the constant ``-2 log L`` of the saturated model is
  dropped, so only differences between models fitted to the same rows are meaningful. This mixes ``n``
  (information criteria) with ``n - 1`` (chi-square, RMSEA); it follows the reference implementation this
  module generalizes and is not changed. The Akaike/Schwarz weights are ``exp(-d/2)`` normalised over all
  the models in one comparison (the saturated model included when it is given).
* When ``df == 0`` the model is saturated or just identified: ``p``, ``rmsea_lo`` and ``rmsea_hi`` are NaN,
  ``rmsea = 0`` and ``cfi = 1``. ``df < 0`` raises.

Only recursive (acyclic) models are supported. The data are assumed complete: NaN raises unless
``dropna=True``.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = [
    "compare_observed_path_models",
    "covariance_equivalence",
    "observed_path_model",
    "path_effect",
]

Edge = tuple[str, str]


# ---------------------------------------------------------------------------------------------- input handling
def _prepare(data: Any, names: Sequence[str] | None, dropna: bool, strata: Any = None):
    """Return (Z (n,p) float array, node names, strata or None) with complete rows only."""
    if hasattr(data, "columns") and hasattr(data, "to_numpy"):  # pandas DataFrame
        cols = [str(c) for c in data.columns]
        arr = {c: np.asarray(data[orig], dtype=float) for c, orig in zip(cols, data.columns)}
        node_names = list(names) if names is not None else cols
        Z = np.column_stack([arr[k] for k in node_names]) if all(k in arr for k in node_names) else None
        if Z is None:
            raise ValueError(f"names {list(names)} not all in DataFrame columns {cols}")
    elif isinstance(data, Mapping):
        node_names = list(names) if names is not None else [str(k) for k in data]
        missing = [k for k in node_names if k not in data]
        if missing:
            raise ValueError(f"names {missing} not in data keys {list(data)}")
        cols = [np.asarray(data[k], dtype=float) for k in node_names]
        if any(c.ndim != 1 for c in cols):
            raise ValueError("every mapping value must be a 1-D vector")
        if len({len(c) for c in cols}) != 1:
            raise ValueError("mapping vectors must all have the same length")
        Z = np.column_stack(cols)
    else:
        Z = np.asarray(data, dtype=float)
        if Z.ndim != 2:
            raise ValueError("data must be an (n, p) array or a mapping name -> (n,) vector")
        node_names = list(names) if names is not None else [f"x{i}" for i in range(Z.shape[1])]
        if len(node_names) != Z.shape[1]:
            raise ValueError(f"{len(node_names)} names for {Z.shape[1]} columns")
    if len(set(node_names)) != len(node_names):
        raise ValueError("node names must be unique")
    st = None
    if strata is not None:
        st = np.asarray(strata)
        if st.shape != (Z.shape[0],):
            raise ValueError(f"strata must have shape ({Z.shape[0]},), got {st.shape}")
    bad = ~np.isfinite(Z).all(axis=1)
    if bad.any():
        if not dropna:
            raise ValueError(
                f"{int(bad.sum())} of {Z.shape[0]} rows contain NaN or infinite values; the models are fitted to "
                "complete rows only. Remove them or pass dropna=True (the same rows are then used for every model)."
            )
        Z = Z[~bad]
        st = st[~bad] if st is not None else None
    if Z.shape[0] < len(node_names) + 2:
        raise ValueError(f"too few complete rows ({Z.shape[0]}) for {len(node_names)} variables")
    return Z, node_names, st


def _moment_matrix(Z: np.ndarray, standardize: bool) -> np.ndarray:
    if standardize:
        sd = Z.std(0)
        if np.any(sd <= 0):
            raise ValueError("a variable is constant on the rows used; it cannot be standardized")
        return np.corrcoef(Z.T)
    return np.cov(Z.T, ddof=0)


def _validate_spec(nodes: list[str], exog: Sequence[str], edges: Sequence[Edge], rescov: Sequence[Edge]):
    edges = [(str(a), str(b)) for a, b in edges]
    rescov = [(str(a), str(b)) for a, b in rescov]
    exog = [str(e) for e in exog]
    known = set(nodes)
    for e in exog:
        if e not in known:
            raise ValueError(f"exogenous node {e!r} is not a variable")
    if len(set(exog)) != len(exog):
        raise ValueError("duplicate exogenous nodes")
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate edges")
    for a, b in edges:
        if a not in known or b not in known:
            raise ValueError(f"edge {(a, b)} uses an unknown variable; variables are {nodes}")
        if a == b:
            raise ValueError(f"self loop {(a, b)}")
        if b in exog:
            raise ValueError(f"edge {(a, b)} points into exogenous node {b!r}")
    seen = set()
    for a, b in rescov:
        if a not in known or b not in known:
            raise ValueError(f"residual covariance {(a, b)} uses an unknown variable")
        if a == b:
            raise ValueError(f"residual covariance {(a, b)} joins a node to itself")
        if a in exog or b in exog:
            raise ValueError(f"residual covariance {(a, b)} involves an exogenous node")
        key = frozenset((a, b))
        if key in seen:
            raise ValueError(f"duplicate residual covariance {(a, b)}")
        seen.add(key)
    # acyclic (Kahn)
    indeg = {n: 0 for n in nodes}
    for _, b in edges:
        indeg[b] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    done = 0
    while queue:
        n = queue.pop()
        done += 1
        for a, b in edges:
            if a == n:
                indeg[b] -= 1
                if indeg[b] == 0:
                    queue.append(b)
    if done != len(nodes):
        raise ValueError("edges contain a directed cycle; only recursive models are supported")
    return edges, rescov, exog


# ---------------------------------------------------------------------------------------------- core fitting
def _parents(edges: Sequence[Edge]) -> dict[str, list[str]]:
    par: dict[str, list[str]] = {}
    for s, t in edges:
        par.setdefault(t, []).append(s)
    return par


def _ols(S: np.ndarray, idx: dict[str, int], edges: Sequence[Edge]) -> np.ndarray:
    p = S.shape[0]
    B = np.zeros((p, p))
    for t, ps in _parents(edges).items():
        P = [idx[s] for s in ps]
        B[idx[t], P] = np.linalg.solve(S[np.ix_(P, P)], S[P, idx[t]])
    return B


def _implied(B: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    A = np.linalg.inv(np.eye(len(B)) - B)
    return A @ Psi @ A.T


def _discrepancy(S: np.ndarray, Sig: np.ndarray, logdet_S: float) -> float:
    sgn, ld = np.linalg.slogdet(Sig)
    if sgn <= 0 or not np.isfinite(ld):
        return 1e6
    try:
        return float(ld + np.trace(np.linalg.solve(Sig, S)) - logdet_S - len(S))
    except np.linalg.LinAlgError:
        return 1e6


def _fit_cov(S, nodes, exog, edges, rescov, force_numeric=False):
    """ML fit of one recursive model to the moment matrix S (columns in the order of nodes)."""
    idx = {n: i for i, n in enumerate(nodes)}
    p, ne = len(nodes), len(exog)
    sgn, logdet_S = np.linalg.slogdet(S)
    if sgn <= 0:
        raise np.linalg.LinAlgError("moment matrix is not positive definite")
    par = _parents(edges)
    mask = np.eye(p, dtype=bool)
    for a, b in itertools.combinations(exog, 2):
        mask[idx[a], idx[b]] = mask[idx[b], idx[a]] = True
    for a, b in rescov:
        mask[idx[a], idx[b]] = mask[idx[b], idx[a]] = True
    B0 = _ols(S, idx, edges)
    I = np.eye(p)
    Psi0 = np.where(mask, (I - B0) @ S @ (I - B0).T, 0.0)
    q = len(edges) + (p - ne) + len(rescov) + ne * (ne + 1) // 2
    closed = len(rescov) == 0 or (len(rescov) == 1 and set(par.get(rescov[0][0], [])) == set(par.get(rescov[0][1], [])))
    converged = True
    if closed and not force_numeric:
        B, Psi = B0, Psi0
    else:
        free = [(idx[t], idx[s]) for s, t in edges]
        endo = [i for i in range(p) if nodes[i] not in exog]
        pairs = [(idx[a], idx[b]) for a, b in rescov]
        d0 = np.array([Psi0[i, i] for i in endo])
        if np.any(d0 <= 0):
            raise np.linalg.LinAlgError("non-positive starting residual variance")
        z0 = [np.arctanh(np.clip(Psi0[i, j] / np.sqrt(Psi0[i, i] * Psi0[j, j]), -0.95, 0.95)) for i, j in pairs]
        theta0 = np.concatenate([[B0[i, j] for i, j in free], np.log(d0), z0])
        ex = [idx[e] for e in exog]

        def build(th):
            Bm = np.zeros((p, p))
            for k, (i, j) in enumerate(free):
                Bm[i, j] = th[k]
            Ps = np.zeros((p, p))
            Ps[np.ix_(ex, ex)] = S[np.ix_(ex, ex)]
            dv = np.exp(th[len(free):len(free) + len(endo)])
            for k, i in enumerate(endo):
                Ps[i, i] = dv[k]
            for k, (i, j) in enumerate(pairs):
                Ps[i, j] = Ps[j, i] = np.tanh(th[len(free) + len(endo) + k]) * np.sqrt(Ps[i, i] * Ps[j, j])
            return Bm, Ps

        def obj(th):
            Bm, Ps = build(th)
            return _discrepancy(S, _implied(Bm, Ps), logdet_S)

        res = optimize.minimize(obj, theta0, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
        B, Psi = build(res.x)
        converged = bool(res.success or np.max(np.abs(res.jac)) < 1e-5)
    Sig = _implied(B, Psi)
    return {"B": B, "Psi": Psi, "Sigma": Sig, "F": _discrepancy(S, Sig, logdet_S), "q": int(q),
            "closed_form": bool(closed and not force_numeric), "converged": converged, "idx": idx}


def _rmsea_ci(chi2: float, df: int, n: int):
    point = float(np.sqrt(max(chi2 - df, 0.0) / (df * (n - 1))))
    lo_l = 0.0
    if stats.chi2.cdf(chi2, df) >= 0.95:
        lo_l = optimize.brentq(lambda lam: stats.ncx2.cdf(chi2, df, lam) - 0.95, 1e-9, max(chi2 * 4, 10.0))
    hi_l = 0.0
    if stats.chi2.cdf(chi2, df) > 0.05:
        hi_l = optimize.brentq(lambda lam: stats.ncx2.cdf(chi2, df, lam) - 0.05, 1e-9, max(chi2 * 6, 50.0))
    return point, float(np.sqrt(lo_l / (df * (n - 1)))), float(np.sqrt(hi_l / (df * (n - 1))))


def _indices(S, f, n):
    p = S.shape[0]
    n_obs = p * (p + 1) // 2
    df = n_obs - f["q"]
    if df < 0:
        raise ValueError(f"model has more free parameters ({f['q']}) than the {n_obs} distinct moments (df = {df}); it is not identified")
    chi2 = (n - 1) * f["F"]
    sd = np.sqrt(np.diag(S))
    d = np.sqrt(np.diag(f["Sigma"]))
    resid = S / np.outer(sd, sd) - f["Sigma"] / np.outer(d, d)
    srmr = float(np.sqrt(np.sum(resid[np.tril_indices(p)] ** 2) / n_obs))
    chi_b = (n - 1) * (np.sum(np.log(np.diag(S))) - np.linalg.slogdet(S)[1])
    df_b = p * (p - 1) // 2
    out = {"n": int(n), "q": int(f["q"]), "df": int(df), "chi2": float(chi2), "srmr": srmr,
           "aic": float(n * f["F"] + 2 * f["q"]), "bic": float(n * f["F"] + f["q"] * np.log(n)),
           "chi2_null": float(chi_b), "df_null": int(df_b)}
    if df > 0:
        out["p"] = float(stats.chi2.sf(chi2, df))
        out["rmsea"], out["rmsea_lo"], out["rmsea_hi"] = _rmsea_ci(chi2, df, n)
        out["cfi"] = float(1 - max(chi2 - df, 0.0) / max(chi_b - df_b, chi2 - df, 1e-12))
    else:
        out.update({"p": float("nan"), "rmsea": 0.0, "rmsea_lo": float("nan"), "rmsea_hi": float("nan"), "cfi": 1.0})
    return out


def _reach_pairs(nodes, edges):
    """Ordered pairs (a, b) joined by a directed path of length >= 2."""
    succ = {n: [] for n in nodes}
    for a, b in edges:
        succ[a].append(b)
    out = []
    for a in nodes:
        frontier = {c for b in succ[a] for c in succ[b]}
        seen = set(frontier)
        while frontier:
            frontier = {c for b in frontier for c in succ[b]} - seen
            seen |= frontier
        out.extend((a, b) for b in nodes if b in seen)
    return out


def _effects(B, nodes, edges):
    idx = {n: i for i, n in enumerate(nodes)}
    tot = np.linalg.inv(np.eye(len(nodes)) - B)
    rows = []
    for a, b in _reach_pairs(nodes, edges):
        t, d = float(tot[idx[b], idx[a]]), float(B[idx[b], idx[a]])
        rows.append({"from": a, "to": b, "indirect": t - d, "direct": d, "total": t})
    return rows


def _result(S, n, nodes, exog, edges, rescov, force_numeric=False):
    f = _fit_cov(S, nodes, exog, edges, rescov, force_numeric)
    res = _indices(S, f, n)
    idx = f["idx"]
    Psi = f["Psi"]
    res.update({
        "nodes": list(nodes),
        "coefficients": [{"from": s, "to": t, "estimate": float(f["B"][idx[t], idx[s]])} for s, t in edges],
        "residual_covariances": [{"a": a, "b": b, "covariance": float(Psi[idx[a], idx[b]]),
                                  "correlation": float(Psi[idx[a], idx[b]] / np.sqrt(Psi[idx[a], idx[a]] * Psi[idx[b], idx[b]]))}
                                 for a, b in rescov],
        "implied": f["Sigma"], "B": f["B"], "Psi": Psi, "fit_function": float(f["F"]),
        "indirect": _effects(f["B"], nodes, edges),
        "closed_form": f["closed_form"], "converged": f["converged"],
    })
    if not f["converged"]:
        warnings.warn("BFGS did not converge for a model with residual covariances; fit indices are unreliable",
                      RuntimeWarning, stacklevel=3)
    return res


# ---------------------------------------------------------------------------------------------- public API
def observed_path_model(data, edges, *, exogenous=(), residual_covariances=(), names=None, standardize=True,
                        dropna=False, force_numeric=False) -> dict:
    """Fit one recursive observed-variable path model by Gaussian maximum likelihood.

    See the module docstring for the likelihood, the free-parameter count, the AIC/BIC convention, the CFI
    null model, and when to prefer ``pysimlr.sem`` instead.

    Parameters
    ----------
    data : (n, p) array, DataFrame or mapping name -> (n,) vector
        Complete rows only. NaN or inf raises ``ValueError`` unless ``dropna=True`` (rows removed).
        Every column is a node, including columns that appear in no edge.
    edges : sequence of (from, to)
        Directed paths; the graph must be acyclic.
    exogenous : sequence of names
        Nodes with no incoming edge whose variances and mutual covariances are fixed at the sample values
        (treated as free parameters in ``q``). Covariates are entered this way and pointed at the
        outcomes through ``edges``; they are never regressed out beforehand.
    residual_covariances : sequence of (a, b)
        Free residual covariances between endogenous nodes.
    names : sequence of str, optional
        Column labels for an array; for a mapping or DataFrame, the subset and order of variables.
    standardize : bool
        True (default) fits the correlation matrix (coefficients are standardized); False fits the covariance.
    dropna : bool
        Drop incomplete rows instead of raising.
    force_numeric : bool
        Use BFGS even when the closed form applies (for checking).

    Returns
    -------
    dict with keys
        ``n``, ``q``, ``df``, ``chi2``, ``p``, ``rmsea``, ``rmsea_lo``, ``rmsea_hi``, ``cfi``, ``srmr``,
        ``aic``, ``bic``, ``chi2_null``, ``df_null``, ``fit_function`` (F),
        ``nodes`` (column order of the matrices),
        ``coefficients`` (list of ``{"from","to","estimate"}`` in the order of ``edges``),
        ``residual_covariances`` (list of ``{"a","b","covariance","correlation"}``),
        ``implied`` (model-implied correlation or covariance, (p, p) array), ``B`` (row = target) and ``Psi``,
        ``indirect`` (one ``{"from","to","indirect","direct","total"}`` per ordered pair joined by a
        directed path of length >= 2; ``indirect`` is the sum over those paths of the product of
        coefficients along each path, ``total`` the sum over all paths including the direct edge),
        ``closed_form`` (bool), ``converged`` (bool).
    """
    Z, nodes, _ = _prepare(data, names, dropna)
    edges, rescov, exog = _validate_spec(nodes, exogenous, edges, residual_covariances)
    S = _moment_matrix(Z, standardize)
    return _result(S, Z.shape[0], nodes, exog, edges, rescov, force_numeric)


def path_effect(B, nodes: Sequence[str], source: str, target: str, through: str | None = None) -> float:
    """Total effect of ``source`` on ``target`` from a coefficient matrix ``B`` (row = target).

    With ``through`` set, the part carried by directed paths that pass through that node: the total effect
    minus the total effect with the node's outgoing paths removed.
    """
    nodes = list(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    I = np.eye(len(nodes))
    tot = np.linalg.inv(I - np.asarray(B))[idx[target], idx[source]]
    if through is None:
        return float(tot)
    B2 = np.array(B, dtype=float)
    j = idx[through]
    if not np.any(B2[:, j]):
        return 0.0
    B2[:, j] = 0.0
    return float(tot - np.linalg.inv(I - B2)[idx[target], idx[source]])


def _norm_models(models: Mapping[str, Mapping[str, Any]]):
    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must be a non-empty mapping name -> {'edges': [...], 'residual_covariances': [...]}")
    out = {}
    for name, spec in models.items():
        if "edges" not in spec:
            raise ValueError(f"model {name!r} has no 'edges'")
        unknown = set(spec) - {"edges", "residual_covariances", "text"}
        if unknown:
            raise ValueError(f"model {name!r}: unknown keys {sorted(unknown)}")
        out[name] = {"edges": [tuple(e) for e in spec["edges"]],
                     "residual_covariances": [tuple(e) for e in spec.get("residual_covariances", [])],
                     "text": spec.get("text")}
    return out


def _resolve_through(
    through: Sequence[tuple[str, str, str]] | str | None,
    nodes: list[str],
    exog: list[str],
    checked: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str, str]] | None:
    """Validate and resolve ``through`` into unique ``(source, target, node)`` triples or ``None``."""
    if through is None:
        return None
    if isinstance(through, str):
        if through != "all_mediators":
            raise ValueError(
                f"unknown through specification {through!r}; expected 'all_mediators' or a sequence of (source, target, node) triples"
            )
        has_out = {a for sp in checked.values() for a, _ in sp["edges"]}
        has_in = {b for sp in checked.values() for _, b in sp["edges"]}
        sources = list(exog) if exog else [n for n in nodes if n in has_out and n not in has_in]
        source_set = set(sources)
        sinks = [n for n in nodes if n not in source_set and n in has_in and n not in has_out]
        sink_set = set(sinks)
        intermediates = [n for n in nodes if n not in source_set and n not in sink_set and n in has_in and n in has_out]
        return [(s, t, m) for s in sources for t in sinks for m in intermediates]
    if isinstance(through, (bytes, Mapping)) or not hasattr(through, "__iter__"):
        raise ValueError("through must be None, 'all_mediators', or a sequence of (source, target, node) triples")
    known = set(nodes)
    triples: list[tuple[str, str, str]] = []
    for item in through:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 3:
            raise ValueError(f"invalid through triple {item!r}; expected a 3-tuple (source, target, node)")
        s, t, m = (str(item[0]), str(item[1]), str(item[2]))
        triple = (s, t, m)
        if s not in known or t not in known or m not in known:
            raise ValueError(f"through triple {triple} uses an unknown variable; variables are {nodes}")
        if s == m or t == m or s == t:
            raise ValueError(f"through triple {triple} must have distinct source, target, and node")
        triples.append(triple)
    return list(dict.fromkeys(triples))


def _reachable_from(succ: Mapping[str, Sequence[str]], start: str) -> set[str]:
    """Nodes reachable from ``start`` along one or more directed edges."""
    seen = set(succ[start])
    frontier = set(seen)
    while frontier:
        frontier = {c for b in frontier for c in succ[b]} - seen
        seen |= frontier
    return seen


def _active_through_triples(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    triples: Sequence[tuple[str, str, str]],
) -> set[tuple[str, str, str]]:
    """Subset of ``triples`` for which ``edges`` contains a directed path ``source ->+ node ->+ target``."""
    succ: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        succ[a].append(b)
    reach_cache: dict[str, set[str]] = {}

    def reaches(u: str) -> set[str]:
        if u not in reach_cache:
            reach_cache[u] = _reachable_from(succ, u)
        return reach_cache[u]

    active: set[tuple[str, str, str]] = set()
    for s, t, m in triples:
        if succ[m] and m in reaches(s) and t in reaches(m):
            active.add((s, t, m))
    return active


def _through_stat_map(
    B: np.ndarray,
    nodes: Sequence[str],
    triples: Sequence[tuple[str, str, str]],
    active_triples: set[tuple[str, str, str]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for s, t, m in triples:
        key = f"through:{s}>{t}|{m}"
        out[key] = path_effect(B, nodes, s, t, through=m) if (s, t, m) in active_triples else 0.0
    return out


def _stat_vector(res: dict) -> dict[str, float]:
    d = {f"{c['from']}>{c['to']}": c["estimate"] for c in res["coefficients"]}
    for r in res["residual_covariances"]:
        d[f"{r['a']}~~{r['b']}"] = r["correlation"]
    for r in res["indirect"]:
        d[f"indirect:{r['from']}>{r['to']}"] = r["indirect"]
        d[f"total:{r['from']}>{r['to']}"] = r["total"]
    for r in res.get("through", ()):
        d[f"through:{r['from']}>{r['to']}|{r['through']}"] = r["estimate"]
    return d


def _boot_chunk(args):
    Z, idx_list, nodes, exog, specs, standardize, triples, active_by_model = args
    out = []
    for ix in idx_list:
        try:
            Zb = Z[ix]
            if not np.isfinite(Zb).all() or np.any(Zb.std(0) <= 0):
                raise np.linalg.LinAlgError("constant variable in resample")
            S = _moment_matrix(Zb, standardize)
            row = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # counted through the converged flag below
                for name, sp in specs.items():
                    r = _result(S, len(ix), nodes, exog, sp["edges"], sp["residual_covariances"])
                    sv = _stat_vector(r)
                    if triples:
                        sv.update(_through_stat_map(r["B"], nodes, triples, active_by_model[name]))
                    row[name] = (r["aic"], r["bic"], r["converged"], sv)
            out.append(row)
        except np.linalg.LinAlgError:
            out.append(None)  # singular resample; counted and reported as n_boot_failed
    return out


def _resample_indices(n, strata, n_boot, random_state):
    rng = np.random.default_rng(random_state)
    groups = None if strata is None else [np.flatnonzero(strata == g) for g in np.unique(strata)]
    out = []
    for _ in range(n_boot):
        if groups is None:
            out.append(rng.choice(n, size=n, replace=True))
        else:
            out.append(np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in groups]))
    return out


def compare_observed_path_models(data, models, *, exogenous=(), n_boot=0, strata=None, random_state=0, n_jobs=1,
                                 standardize=True, names=None, dropna=False, ci=0.95,
                                 through: Sequence[tuple[str, str, str]] | str | None = None,
                                 return_draws: bool = False) -> dict:
    """Fit several recursive path models to the SAME complete rows and compare them.

    Parameters
    ----------
    data, exogenous, standardize, names, dropna
        As in :func:`observed_path_model`. ``exogenous`` applies to every model.
    models : mapping name -> {"edges": [...], "residual_covariances": [...], "text": optional}
    n_boot : int
        0 (default) skips the bootstrap. Otherwise a nonparametric bootstrap of the estimator: rows are
        resampled WITH replacement (within ``strata`` when given, so every stratum keeps its size), each
        resample is re-standardized, and every model is refitted. Intervals are percentile intervals at
        level ``ci``. Singular resamples are skipped and counted in ``n_boot_failed``.
    strata : array-like of length n, optional
    random_state : int
        Resample indices are all drawn up front in the parent process from ``np.random.default_rng``;
        results do not depend on ``n_jobs``.
    n_jobs : int
        Processes for the bootstrap (1 runs inline).
    through : sequence of (source, target, node) triples or "all_mediators", optional
        Specific-mediator effects to track across every model via ``path_effect(..., through=node)``.
        Unlike ``indirect:source>target`` (the sum over all directed paths of length >= 2 between
        ``source`` and ``target``), ``through:source>target|node`` isolates only the directed paths
        that pass through ``node`` (they agree only when every indirect path passes through ``node``).
        Passing ``"all_mediators"`` expands to every ``(source, target, node)`` triple with ``source``
        in ``exogenous`` (or graph roots with outgoing and no incoming edges across the models when
        ``exogenous`` is empty), ``target`` a sink across all models (not a source and with no outgoing
        edge), and ``node`` an intermediate node. Models without a directed path
        ``source ->+ node ->+ target`` get exact ``0.0`` (and ``lo = hi = 0.0``).
    return_draws : bool
        When True and ``n_boot > 0``, also return ``bootstrap_draws`` mapping
        ``model -> {key: 1-D float ndarray of valid resample draws}``. Raises ``ValueError`` when
        ``return_draws=True`` and ``n_boot <= 0``.

    Returns
    -------
    dict with
        ``n``, ``n_boot``, ``n_boot_failed``, ``nodes``,
        ``table``: one dict per model (input order) with ``model``, ``text``, ``n``, ``q``, ``df``,
        ``chi2``, ``p``, ``rmsea``, ``rmsea_lo``, ``rmsea_hi``, ``cfi``, ``srmr``, ``aic``, ``bic``,
        ``closed_form``, ``converged``, ``daic``, ``dbic`` (differences from the best of all models given),
        ``w_aic``, ``w_bic`` (Akaike/Schwarz weights, sum to 1), ``rejected`` (``p < 0.05``; False when
        ``df == 0``) and, with ``n_boot > 0``, ``best_share_aic`` / ``best_share_bic`` (fraction of used
        resamples in which the model has the smallest AIC / BIC; ties go to the first model);
        ``best_by_aic``, ``best_by_bic``: model names (saturated model included if supplied);
        ``fits``: name -> full :func:`observed_path_model` result (plus ``through`` when ``through`` is given,
        a list of ``{"from", "to", "through", "estimate"}`` dicts);
        ``bootstrap`` (only when ``n_boot > 0``): name -> {key: {"estimate","lo","hi"}} with keys
        ``"a>b"`` (coefficient), ``"a~~b"`` (residual correlation), ``"indirect:a>b"``, ``"total:a>b"``,
        and ``"through:a>b|node"`` (when ``through`` is given);
        ``bootstrap_draws`` (only when ``n_boot > 0`` and ``return_draws=True``): name -> {key: 1-D ndarray}.
    """
    if return_draws and int(n_boot) <= 0:
        raise ValueError("return_draws=True requires n_boot > 0")
    Z, nodes, st = _prepare(data, names, dropna, strata)
    specs = _norm_models(models)
    exog = [str(e) for e in exogenous]
    checked = {}
    for name, sp in specs.items():
        e, r, _ = _validate_spec(nodes, exog, sp["edges"], sp["residual_covariances"])
        checked[name] = {"edges": e, "residual_covariances": r, "text": sp["text"]}
    triples = _resolve_through(through, nodes, exog, checked)
    active_by_model = (
        {name: _active_through_triples(nodes, sp["edges"], triples) for name, sp in checked.items()}
        if triples else {}
    )
    n = Z.shape[0]
    S = _moment_matrix(Z, standardize)
    fits = {name: _result(S, n, nodes, exog, sp["edges"], sp["residual_covariances"]) for name, sp in checked.items()}
    if triples is not None:
        for name, f in fits.items():
            tmap = _through_stat_map(f["B"], nodes, triples, active_by_model[name]) if triples else {}
            f["through"] = [
                {"from": s, "to": t, "through": m, "estimate": tmap[f"through:{s}>{t}|{m}"]}
                for s, t, m in triples
            ]
    rows = []
    for name, f in fits.items():
        rows.append({"model": name, "text": checked[name]["text"], **{k: f[k] for k in (
            "n", "q", "df", "chi2", "p", "rmsea", "rmsea_lo", "rmsea_hi", "cfi", "srmr", "aic", "bic", "closed_form", "converged")}})
    for key, w in (("aic", "w_aic"), ("bic", "w_bic")):
        v = np.array([r[key] for r in rows])
        ww = np.exp(-(v - v.min()) / 2)
        for r, wi, di in zip(rows, ww / ww.sum(), v - v.min()):
            r[w] = float(wi)
            r["d" + key] = float(di)
    for r in rows:
        r["rejected"] = bool(r["df"] > 0 and r["p"] < 0.05)
    out = {"n": n, "n_boot": int(n_boot), "n_boot_failed": 0, "nodes": nodes, "table": rows,
           "best_by_aic": min(rows, key=lambda r: r["aic"])["model"],
           "best_by_bic": min(rows, key=lambda r: r["bic"])["model"], "fits": fits}
    if n_boot > 0:
        idx_list = _resample_indices(n, st, int(n_boot), random_state)
        if n_jobs > 1:
            chunks = [idx_list[i::n_jobs] for i in range(n_jobs)]
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                parts = list(pool.map(
                    _boot_chunk,
                    [(Z, c, nodes, exog, checked, standardize, triples, active_by_model) for c in chunks],
                ))
            res = [None] * len(idx_list)
            for i, part in enumerate(parts):
                res[i::n_jobs] = part
        else:
            res = _boot_chunk((Z, idx_list, nodes, exog, checked, standardize, triples, active_by_model))
        used = [r for r in res if r is not None]
        out["n_boot_failed"] = len(res) - len(used)
        if not used:
            raise ValueError("every bootstrap resample was singular")
        mnames = list(checked)
        lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
        boot = {}
        draws = {} if return_draws else None
        for m in mnames:
            est_map = _stat_vector(fits[m])
            keys = list(used[0][m][3])
            boot[m] = {}
            if draws is not None:
                draws[m] = {}
            for k in keys:
                arr = np.array([r[m][3][k] for r in used], dtype=float)
                boot[m][k] = {
                    "estimate": est_map[k],
                    "lo": float(np.percentile(arr, lo_q)),
                    "hi": float(np.percentile(arr, hi_q)),
                }
                if draws is not None:
                    draws[m][k] = arr
        out["bootstrap"] = boot
        if draws is not None:
            out["bootstrap_draws"] = draws
        for crit, j in (("aic", 0), ("bic", 1)):
            wins = np.zeros(len(mnames))
            for r in used:
                wins[int(np.argmin([r[m][j] for m in mnames]))] += 1
            for row, wv in zip(rows, wins / len(used)):
                row["best_share_" + crit] = float(wv)
        out["n_boot_not_converged"] = int(sum(1 for r in used for m in mnames if not r[m][2]))
    return out


# ---------------------------------------------------------------------------------------------- equivalence
def _dag_signature(edges, exog):
    skel = {frozenset(e) for e in edges}
    skel |= {frozenset(c) for c in itertools.combinations(exog, 2)}
    v = set()
    for t, ps in _parents(edges).items():
        for a, b in itertools.combinations(sorted(ps), 2):
            if frozenset((a, b)) not in skel:
                v.add((a, t, b))
    return frozenset(skel), frozenset(v)


def _random_sigma(nodes, exog, edges, rescov, rng):
    idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)
    ex = [idx[e] for e in exog]
    for _ in range(500):
        B = np.zeros((p, p))
        for a, b in edges:
            B[idx[b], idx[a]] = rng.uniform(0.25, 0.7) * rng.choice([-1, 1])
        Psi = np.diag(rng.uniform(0.6, 1.4, p))
        for i, j in itertools.combinations(ex, 2):
            Psi[i, j] = Psi[j, i] = rng.uniform(0.1, 0.3) * rng.choice([-1, 1]) * np.sqrt(Psi[i, i] * Psi[j, j])
        for a, b in rescov:
            Psi[idx[a], idx[b]] = Psi[idx[b], idx[a]] = rng.uniform(0.1, 0.3) * rng.choice([-1, 1]) * np.sqrt(
                Psi[idx[a], idx[a]] * Psi[idx[b], idx[b]])
        if np.linalg.eigvalsh(Psi).min() > 0.05:
            Sig = _implied(B, Psi)
            d = np.sqrt(np.diag(Sig))
            return Sig / np.outer(d, d)
    raise RuntimeError("could not draw a positive-definite covariance")


def covariance_equivalence(models, *, exogenous=(), n_draws: int = 5, random_state: int = 0, tol: float = 1e-6) -> dict:
    """Which models imply the same set of covariance matrices.

    Numerical test as in the reference: draw ``n_draws`` correlation matrices from each model (random
    coefficients, residual variances and residual covariances), fit every other model to each draw, and
    say model ``a`` is inside ``b`` when ``b`` reproduces all draws from ``a`` with discrepancy ``F < tol``.
    Equivalent means mutually inside. Models are compared on the union of the variables they mention
    (plus ``exogenous``); a variable absent from a model's edges is unrelated to everything in it.

    Parameters
    ----------
    models : mapping name -> {"edges": [...], "residual_covariances": [...]}
    exogenous : names fixed-x in every model (their mutual covariance is free in the draw and the fit).

    Returns
    -------
    dict with ``classes`` (list of lists of model names, input order), ``pairs`` (one dict per unordered
    pair: ``a``, ``b``, ``equivalent``, ``a_inside_b``, ``b_inside_a``, ``graphical`` (skeleton and
    v-structure criterion, True/False; None when either model has residual covariances)),
    ``n_draws``, ``graphical_agrees`` (every pair with a graphical answer agrees with the numerical one) and
    ``n_pairs_graphical``. The draws are random; with few draws a non-equivalence can go undetected only
    if the extra freedom of the larger model is not exercised, which has probability zero for generic draws.
    """
    specs = _norm_models(models)
    exog = [str(e) for e in exogenous]
    nodes: list[str] = list(dict.fromkeys(
        [e for e in exog] + [v for sp in specs.values() for ed in (sp["edges"], sp["residual_covariances"]) for pair in ed for v in pair]))
    for sp in specs.values():
        _validate_spec(nodes, exog, sp["edges"], sp["residual_covariances"])
    rng = np.random.default_rng(random_state)
    names = list(specs)
    draws = {a: [_random_sigma(nodes, exog, specs[a]["edges"], specs[a]["residual_covariances"], rng) for _ in range(n_draws)] for a in names}
    contains: dict[tuple[str, str], bool] = {}
    for a, b in itertools.permutations(names, 2):
        fmax = 0.0
        for R in draws[a]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a non-converged refit shows up as F >= tol
                f = _fit_cov(R, nodes, exog, specs[b]["edges"], specs[b]["residual_covariances"])
            fmax = max(fmax, f["F"])
        contains[(a, b)] = fmax < tol
    parent = {n: n for n in names}

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x

    pairs, agree = [], []
    for a, b in itertools.combinations(names, 2):
        eq = contains[(a, b)] and contains[(b, a)]
        if eq:
            parent[root(a)] = root(b)
        g = None
        if not specs[a]["residual_covariances"] and not specs[b]["residual_covariances"]:
            g = _dag_signature(specs[a]["edges"], exog) == _dag_signature(specs[b]["edges"], exog)
            agree.append(bool(g == eq))
        pairs.append({"a": a, "b": b, "equivalent": bool(eq), "a_inside_b": bool(contains[(a, b)]),
                      "b_inside_a": bool(contains[(b, a)]), "graphical": g})
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(root(n), []).append(n)
    return {"classes": list(groups.values()), "pairs": pairs, "n_draws": int(n_draws),
            "graphical_agrees": bool(all(agree)), "n_pairs_graphical": len(agree)}
