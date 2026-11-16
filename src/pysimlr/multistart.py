"""``multistart_fit``: fit from several starting points and keep the lowest energy.

The linear fit can have several local solutions, and the default start (``initialization_type="pca"``) is
deterministic, so a single fit does not show whether another solution exists. `multistart_fit` runs
`pysimlr.fit` from the default start and from random starts, returns the fit with the lowest ``best_energy``
and reports how many distinct solutions the starts found.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

#: Two starts are the same solution when the absolute correlation of their first consensus component is at least this.
SAME_SOLUTION_CORR = 0.95


def _seeds(random_state: int, n: int) -> List[int]:
    """``n`` distinct seeds derived deterministically from ``random_state``."""
    if n == 0:
        return []
    ss = np.random.SeedSequence(int(random_state))
    out: List[int] = []
    for s in ss.generate_state(4 * n + 8, dtype=np.uint32):
        s = int(s)
        if s not in out:
            out.append(s)
        if len(out) == n:
            return out
    raise RuntimeError("could not derive distinct seeds")  # pragma: no cover


def _run_start(task: Any) -> Dict[str, Any]:
    from .api import fit
    method, views, k, params, seed = task
    kw = dict(params)
    if seed is not None:
        kw.update(initialization_type="random", random_state=seed)
    return fit(method, views, k, **kw)


def _energy(method: str, res: Dict[str, Any]) -> float:
    if "best_energy" not in res or res["best_energy"] is None:
        raise NotImplementedError(
            f"multistart_fit: method {method!r} reports no 'best_energy' (the similarity energy that `simlr` "
            f"minimises), so fits from different starts have no comparable energy.")
    return float(res["best_energy"])


def _first_score(res: Dict[str, Any]) -> np.ndarray:
    u = res["u"]
    u = u.detach().cpu().double().numpy() if hasattr(u, "detach") else np.asarray(u, dtype=np.float64)
    return u[:, 0]


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float(abs((a * b).sum() / den)) if den > 0 else float("nan")


def multistart_fit(method: str, views: Any, k: int, *, n_starts: int = 10, include_default: bool = True,
                   random_state: int = 0, n_jobs: int = 1, **params: Any) -> Dict[str, Any]:
    """Fit ``method`` from the default start and ``n_starts`` random starts; return the lowest-energy fit.

    Parameters
    ----------
    method, views, k, **params
        As for `pysimlr.fit`. ``initialization_type`` and ``random_state`` are set by this function and may
        not appear in ``params``.
    n_starts : int, default=10
        Number of random starts (``initialization_type="random"``), each with its own seed derived from
        ``random_state`` (``numpy.random.SeedSequence``); the seeds are distinct.
    include_default : bool, default=True
        Also fit once with ``params`` unchanged, exactly as ``pysimlr.fit(method, views, k, **params)`` would
        (this start has ``seed=None``). With ``n_starts=0`` the result is that plain fit plus the summary keys.
    random_state : int, default=0
        Root of the derived seeds.
    n_jobs : int, default=1
        Worker processes for the starts (``-1``: all cores); the result does not depend on it beyond
        floating-point differences from the thread count inside a worker.

    Supported methods. A start can be varied when the method accepts ``initialization_type`` and
    ``random_state``, and starts are comparable when the result has ``best_energy`` (read from the first fit).
    Otherwise ``NotImplementedError`` names the method and the missing piece. Of the registered methods this
    holds for ``simlr``; ``pca`` and ``nsa_pipeline`` accept no ``initialization_type`` (closed-form or
    deterministic), and ``lend``, ``ned``, ``ned_shared_private`` and ``flow_v`` report no ``best_energy``
    (their training loss includes decoder terms and is not the similarity energy).

    Returns
    -------
    dict
        The chosen fit (every key `pysimlr.fit` returns) plus ``starts`` (one dict per start in run order:
        ``label`` ("default" or "random_<i>"), ``seed`` (None for the default), ``energy`` (``best_energy``),
        ``converged``, ``n_steps`` (from ``result["convergence"]``) and ``score_correlation_with_chosen``
        (absolute correlation of the first consensus component ``u[:, 0]`` with the chosen fit's)), ``chosen``
        (label of the chosen start), ``chosen_index`` (its position in ``starts``), ``n_distinct_solutions``
        (number of groups of starts linked by ``|corr| >= 0.95``, linking through chains), ``energy_gap`` (the
        lowest energy minus the lowest energy among the other distinct solutions: zero or negative, ``None``
        with a single solution) and ``multistart`` (the settings). Lower energy is better; ties go to the
        earlier start. A start with a non-finite energy is excluded from the choice with a ``RuntimeWarning``;
        all non-finite raises ``RuntimeError``.
    """
    from .api import METHODS, resolve_method
    from .loadings import _n_workers, _pool

    spec = resolve_method(method)
    if int(n_starts) < 0 or int(n_starts) != n_starts:
        raise ValueError(f"n_starts must be a non-negative integer; got {n_starts}.")
    n_starts = int(n_starts)
    if n_starts == 0 and not include_default:
        raise ValueError("nothing to run: n_starts=0 and include_default=False.")
    clash = sorted({"initialization_type", "random_state"} & set(params))
    if clash:
        raise ValueError(f"multistart_fit sets {clash} itself; remove {'it' if len(clash) == 1 else 'them'} "
                         f"from the arguments.")
    missing = [p for p in ("initialization_type", "random_state") if p not in spec.accepts]
    if missing:
        raise NotImplementedError(
            f"multistart_fit: method {method!r} does not accept {missing}, so its start cannot be varied "
            f"(initialization_type='random' with a seed). Methods that accept initialization_type: "
            f"{sorted(n for n, s in METHODS.items() if 'initialization_type' in s.accepts)}.")

    seeds: List[Optional[int]] = ([None] if include_default else []) + list(_seeds(random_state, n_starts))
    labels = (["default"] if include_default else []) + [f"random_{i}" for i in range(n_starts)]
    tasks = [(method, views, k, params, s) for s in seeds]

    first = _run_start(tasks[0])
    _energy(method, first)                       # raises NotImplementedError before the other starts run
    rest = tasks[1:]
    workers = min(_n_workers(n_jobs), max(len(rest), 1))
    if workers > 1 and rest:
        with _pool(workers) as ex:
            fits = [first] + list(ex.map(_run_start, rest))
    else:
        fits = [first] + [_run_start(t) for t in rest]

    energies = [_energy(method, f) for f in fits]
    finite = [i for i, e in enumerate(energies) if np.isfinite(e)]
    if len(finite) < len(fits):
        bad = [labels[i] for i in range(len(fits)) if i not in finite]
        if not finite:
            raise RuntimeError(f"multistart_fit: every start has a non-finite energy ({bad}).")
        warnings.warn(f"multistart_fit: starts {bad} have a non-finite energy and are excluded from the choice.",
                      RuntimeWarning, stacklevel=2)
    chosen_i = min(finite, key=lambda i: (energies[i], i))
    scores = [_first_score(f) for f in fits]

    # distinct solutions: connected components of the graph |corr| >= SAME_SOLUTION_CORR among the usable starts
    parent = {i: i for i in finite}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in finite:
        for b in finite:
            if a < b and _abs_corr(scores[a], scores[b]) >= SAME_SOLUTION_CORR:
                parent[find(b)] = find(a)
    groups: Dict[int, List[int]] = {}
    for i in finite:
        groups.setdefault(find(i), []).append(i)
    best_by_group = sorted(min(energies[i] for i in g) for g in groups.values())
    gap = float(best_by_group[0] - best_by_group[1]) if len(best_by_group) > 1 else None

    starts = []
    for i, f in enumerate(fits):
        conv = f.get("convergence", {})
        starts.append({
            "label": labels[i], "seed": seeds[i], "energy": energies[i],
            "converged": conv.get("converged"), "n_steps": conv.get("n_steps"),
            "score_correlation_with_chosen": _abs_corr(scores[i], scores[chosen_i]),
        })
    out = dict(fits[chosen_i])
    out.update({
        "starts": starts, "chosen": labels[chosen_i], "chosen_index": chosen_i,
        "n_distinct_solutions": len(groups), "energy_gap": gap,
        "multistart": {"method": method, "n_starts": n_starts, "include_default": bool(include_default),
                       "random_state": int(random_state), "n_jobs": n_jobs,
                       "same_solution_corr": SAME_SOLUTION_CORR},
    })
    return out


__all__ = ["multistart_fit"]
