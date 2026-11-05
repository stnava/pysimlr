r"""Which features carry a component: a hard relevance cutoff.

Feature maps (`structure`, relative weights, ...) are continuous; on their own they do not say
which features matter. `feature_relevance` turns a map into a decision with an explicit null and
an explicit replication:

* **Statistic.** For view m and component j, each feature's share of the variance that the view
  explains in a target t (Johnson's relative weights, normalized to sum to 1). The target is the
  mean of the *other* views' standardized, sign-aligned scores on component j, so the test never
  regresses a linear view's own score on itself. The *fair share* is 1/p: what every feature would
  carry if none mattered more than another.
* **Training test.** Permute t against the view's features ``n_permutations`` times; a feature's
  p-value is P(null share >= observed share). Groups of features (lobes, networks, bilateral
  pairs) are tested the same way on the sum of their shares.
* **Held-out replication.** The fitted model is applied to held-out subjects (never used in
  fitting); shares and their own permutation null are recomputed there. Views missing at test
  time are passed as None; the target is built from the same views in training and held-out data,
  and the component's sign is learned in training.
* **Refit stability.** ``n_refits`` half-sample refits of the training rows, each with its own
  permutation test; the frequency of p < alpha.
* **Tiers.** 2 = p < alpha in training AND held-out ("replicated"), 1 = training only, 0 = neither.
  p-values are uncorrected; ``summary`` reports the counts against what chance gives (alpha * p).

**What the test can and cannot say.** Shares are normalized, so the test asks whether a feature
carries more than its fair share of what the view explains, not how much the view explains. Under
the null a single feature's share is close to Beta(1/2, (p - 1)/2) whatever the signal strength,
so p depends on concentration alone: with p = 160 a feature needs a share of about 2.4% (3.8x fair)
for p < 0.05, and when k features carry a signal equally each share is about 1/k, which bounds
their p-values (four equal drivers among 24 features: p ~ 0.01 however strong the signal). More
permutations sharpen p, not power. A diffuse signal (every feature equally relevant) passes no
feature by design; group tests on summed shares are the tool for it. Read ``r_outcome_test`` for
strength.

The permutation null holds the fitted model fixed. ``n_null_refits`` adds a stricter, slower null
that refits the whole model with the tested view's training rows shuffled against the other views
(``p_null_refit``). On ADNI-MCI (300 refits per view) it reproduced the permutation results and did
not change the tiers (``docs/audit/evidence/relevance``), so it is off by default and does not
enter the tiers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .loadings import (
    _relative_weight_basis,
    _relative_weight_basis_lam,
    _relative_weight_shares,
    _zscore,
)

_CHUNK = 5_000


def _perm_pvalues(zo, lam2, t, obs, groups: Mapping[str, Sequence[int]], n_perm: int, rng):
    """Permutation p-values (uncorrected) of each feature's and each group's share."""
    ge = np.zeros(len(obs))
    geg = {g: 0 for g in groups}
    for s in range(0, n_perm, _CHUNK):
        k = min(_CHUNK, n_perm - s)
        null = _relative_weight_shares(zo, lam2, rng.permuted(np.tile(t[:, None], (1, k)), axis=0))
        ge += (null >= obs[:, None]).sum(1)
        for g, idx in groups.items():
            geg[g] += int((null[idx].sum(0) >= obs[idx].sum()).sum())
    return (1 + ge) / (n_perm + 1), {g: (1 + c) / (n_perm + 1) for g, c in geg.items()}


def _as_tensor(x):
    if x is None:
        return None
    return x.detach().float() if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x)).float()


def _targets(result: Dict[str, Any], views: List[Optional[torch.Tensor]], avail: List[int],
             eval_views: Sequence[int], component: Optional[int], outcome, signs=None):
    """Component j, per-view signs, and {m: (standardized X_m, target t)} on the given rows."""
    from .api import _preprocess_views, predict
    from .extension import view_names as _names
    rot = result["loading_rotation"].detach().double().numpy()
    call = [views[i] if i in avail else None for i in range(len(views))]
    pred = predict(result, call)
    lat = pred["latents"]
    if isinstance(lat, dict):
        names = _names(result)
        sc = {i: np.asarray(lat[names[i]].detach().double()) @ rot for i in avail}
    else:
        sc = {i: np.asarray(lat[i].detach().double()) @ rot for i in avail}
    u = np.asarray(pred["u"].detach().double()) @ rot
    if component is None:
        if outcome is None:
            raise ValueError("pass component= or outcome= (the component most correlated with it is used).")
        y = np.asarray(outcome, dtype=float).ravel()
        if len(y) != u.shape[0]:
            raise ValueError(f"outcome has {len(y)} rows; the views have {u.shape[0]}.")
        component = int(np.argmax([abs(np.corrcoef(u[:, c], y)[0, 1]) for c in range(u.shape[1])]))
    if signs is None:
        uc = u - u.mean(0)
        signs = {i: np.where(((z - z.mean(0)) * uc).sum(0) >= 0, 1.0, -1.0) for i, z in sc.items()}
    sc = {i: _zscore(z * signs[i]) for i, z in sc.items()}
    xs, _ = _preprocess_views([views[i] for i in avail], result["scale_list"],
                              [result["provenance_list"][i] for i in avail])
    xs = {i: x.double().numpy() for i, x in zip(avail, xs)}
    out = {}
    for m in eval_views:
        others = [i for i in avail if i != m]
        if not others:
            raise ValueError(f"view {m}: no other available view to build the target from.")
        out[m] = (_zscore(xs[m]), _zscore(sum(sc[i] for i in others))[:, component])
    return component, signs, out


def _group_index(names: Sequence[str], labels: Mapping[str, Mapping[str, str]], min_size: int):
    """{"<grouping> <label>": [feature indices]} for groups with at least ``min_size`` members."""
    out: Dict[str, List[int]] = {}
    for grouping, lab in labels.items():
        tmp: Dict[str, List[int]] = {}
        for i, n in enumerate(names):
            g = lab.get(n)
            if g is not None:
                tmp.setdefault(f"{grouping} {g}", []).append(i)
        out.update({k: v for k, v in tmp.items() if len(v) >= min_size})
    return out


def _view_test(xz, t, names, labels, min_size, n_perm, rng, shrinkage: Any = "auto"):
    live = xz.std(0) > 0
    nm = [n for n, k in zip(names, live) if k]
    zo, lam2, lam = _relative_weight_basis_lam(xz[:, live], shrinkage)
    obs = _relative_weight_shares(zo, lam2, t[:, None])[:, 0]
    groups = _group_index(nm, labels, min_size)
    p, pg = _perm_pvalues(zo, lam2, t, obs, groups, n_perm, rng)
    return nm, obs, p, groups, pg, lam


def _refit_task(args):
    (method, k, params, train, avail, eval_views, outcome, component, rows, names, labels,
     min_size, n_perm, shrinkages, seed) = args
    torch.set_num_threads(1)
    from .api import fit
    sub = [None if v is None else v[torch.as_tensor(rows)] for v in train]
    res = fit(method, sub, k, **params)
    y = None if outcome is None else np.asarray(outcome)[rows]
    _, _, per = _targets(res, sub, avail, eval_views, component, y)
    rng = np.random.default_rng(seed)
    out = {}
    for m, (xz, t) in per.items():
        nm, _obs, p, _groups, pg, _lam = _view_test(xz, t, names[m], labels[m], min_size, n_perm, rng, shrinkages[m])
        out[m] = (nm, p, pg)
    return out


def _null_refit_task(args):
    """Shares of view m after a refit with m's training rows shuffled against the other views."""
    method, k, params, train, avail, m, outcome, component, live, shrinkage, seed = args
    torch.set_num_threads(1)
    from .api import fit
    sub = list(train)
    sub[m] = sub[m][torch.as_tensor(np.random.default_rng(seed).permutation(sub[m].shape[0]))]
    res = fit(method, sub, k, **params)
    _, _, per = _targets(res, sub, avail, [m], component, outcome)
    xz, t = per[m]
    zo, lam2 = _relative_weight_basis(xz[:, live], shrinkage)
    return m, _relative_weight_shares(zo, lam2, t[:, None])[:, 0]


def _run(fn, tasks, n_jobs):
    from .loadings import _n_workers, _pool
    workers = min(_n_workers(n_jobs), len(tasks))
    if workers > 1:
        with _pool(workers) as ex:
            return list(ex.map(fn, tasks))
    return [fn(t) for t in tasks]


def feature_relevance(result: Dict[str, Any], train_views: Sequence[Any], test_views: Optional[Sequence[Any]] = None,
                      *, outcome=None, outcome_test=None, component: Optional[int] = None,
                      views: Optional[Sequence[int]] = None, feature_names=None, view_names=None,
                      groups: Optional[Mapping[str, Any]] = None, min_group_size: int = 2, alpha: float = 0.05,
                      shrinkage: Any = "auto",
                      n_permutations: int = 100_000, n_permutations_test: int = 50_000, n_refits: int = 10,
                      refit_permutations: int = 20_000, refit_fraction: float = 0.5, n_null_refits: int = 0,
                      random_state: int = 2026, n_jobs: int = 8) -> Dict[str, Any]:
    """A hard relevance cutoff for the features of a fitted component (see the module docstring).

    Parameters
    ----------
    result
        A `pysimlr.fit` result.
    train_views
        The views the model was fitted on (all of them, in order).
    test_views
        Held-out subjects' views, in the same order; ``None`` for a view missing at test time. Without
        it there is no replication and no feature can reach tier 2.
    outcome, outcome_test
        Optional per-subject outcome on the training / held-out rows. ``outcome`` chooses the
        component (the one most correlated with it) unless ``component`` is given; ``outcome_test``
        adds each feature's held-out correlation with it (``r_outcome_test``).
    views
        Which views to test (default: every view available at test time, or all).
    feature_names, view_names
        Per-view feature names (list of lists) and view names.
    groups
        ``{grouping: f}`` with ``f`` a callable ``feature name -> label or None`` or a dict; for example
        lobes and bilateral pairs. Groups with fewer than ``min_group_size`` features are dropped.
        Grouping names must not contain spaces.
    shrinkage
        Shrinkage for the relative-weight basis: ``"auto"`` (default: classical ``0.0`` when full-rank
        ``p < n``, else ``"ledoit_wolf"`` so wide views and half-sample refits with ``p >= n_sub``
        remain well-defined), ``"ledoit_wolf"``, ``0.0``, or a float in ``[0, 1)`` (or a per-view list).
        ``"ledoit_wolf"`` here estimates lam from the view alone, so it does not depend on the target
        and the permutation null is exact. The lam used is returned (``shrinkage_used``; also the
        ``shrinkage`` column of ``summary``). Held-out shares reuse the training lam; only where that
        estimator is undefined on the held-out rows is lam re-resolved with ``"auto"``
        (``shrinkage_used_test`` records it). Half-sample refits resolve lam on their own rows.
    n_permutations, n_permutations_test
        Permutations of the target in training and in held-out data.
    n_refits, refit_permutations, refit_fraction
        Half-sample refits (``refit_fraction`` of the training rows, ``refit_permutations`` each)
        for ``refit_p05_freq``, the fraction of refits with p < alpha. 0 skips them.
    n_null_refits
        Whole-model refits per tested view with that view's rows shuffled (``p_null_refit``;
        0 skips them, and the column is NaN). Reported beside the tiers, not part of them.
    random_state, n_jobs
        Seed for every permutation and refit; worker processes for the refits.

    Returns
    -------
    dict with ``features`` and ``groups`` (pandas DataFrames: shares in training and held-out data,
    ratio to the fair share, ``p_train``, ``p_test``, ``refit_p05_freq``, ``p_null_refit``,
    ``r_outcome_test``, ``tier``; NaN where a test was not run),
    ``summary`` (counts against chance per view), ``component`` (0-based), ``alpha`` and settings.
    """
    import pandas as pd

    from .extension import view_names as _names
    train = [_as_tensor(v) for v in train_views]
    n_views = len(result["v"])
    if len(train) != n_views:
        raise ValueError(f"train_views has {len(train)} views; the fit has {n_views}.")
    if test_views is not None:
        test = [_as_tensor(v) for v in test_views]
        if len(test) != n_views:
            raise ValueError(f"test_views has {len(test)} entries; the fit has {n_views} (use None for missing).")
        avail = [i for i, v in enumerate(test) if v is not None]
    else:
        test, avail = None, list(range(n_views))
    eval_views = list(views) if views is not None else list(avail)
    bad = [m for m in eval_views if m not in avail]
    if bad:
        raise ValueError(f"views {bad} are not available at test time.")
    if isinstance(shrinkage, (list, tuple)):
        if len(shrinkage) != n_views:
            raise ValueError(f"shrinkage has {len(shrinkage)} entries; the fit has {n_views} views.")
        shrinkages = list(shrinkage)
    else:
        shrinkages = [shrinkage] * n_views
    vnames = list(view_names) if view_names is not None else list(_names(result))
    names = {m: (list(feature_names[m]) if feature_names is not None else
                 [f"v{m}.{i}" for i in range(train[m].shape[1])]) for m in eval_views}
    bad = [g for g in (groups or {}) if " " in str(g)]
    if bad:
        raise ValueError(f"grouping names must not contain spaces: {bad}")
    labels: Dict[int, Dict[str, Dict[str, str]]] = {}
    for m in eval_views:
        labels[m] = {}
        for grouping, f in (groups or {}).items():
            get = f.get if isinstance(f, Mapping) else f
            labels[m][grouping] = {n: get(n) for n in names[m] if get(n) is not None}

    rng = np.random.default_rng(random_state)
    j, signs, tr = _targets(result, train, avail, eval_views, component, outcome)
    te = None
    if test is not None:
        _, _, te = _targets(result, test, avail, eval_views, j, None, signs=signs)
    ref = {}
    for m in eval_views:
        nm, obs, p, groups_m, pg, lam_tr = _view_test(
            *tr[m], names[m], labels[m], min_group_size, n_permutations, rng, shrinkages[m])
        entry = dict(names=nm, obs=obs, p=p, groups=groups_m, pg=pg, obs_te=None, p_te=None, pg_te=None, r=None,
                     lam=lam_tr, lam_te=np.nan)
        if te is not None:
            xz_te, t_te = te[m]
            live = np.isin(names[m], nm)
            xz_te = xz_te[:, live]
            # The training estimator (its lam) is applied to the held-out rows, so the two shares are
            # comparable; only if it is undefined there (classical, but the held-out view is
            # rank-deficient) is lam re-resolved with 'auto'. Both are recorded.
            try:
                zo, lam2, lam_te = _relative_weight_basis_lam(xz_te, lam_tr)
            except ValueError:
                zo, lam2, lam_te = _relative_weight_basis_lam(xz_te, "auto")
            obs_te = _relative_weight_shares(zo, lam2, t_te[:, None])[:, 0]
            p_te, pg_te = _perm_pvalues(zo, lam2, t_te, obs_te, groups_m, n_permutations_test, rng)
            entry.update(obs_te=obs_te, p_te=p_te, pg_te=pg_te, lam_te=lam_te)
            if outcome_test is not None:
                y = np.asarray(outcome_test, dtype=float).ravel()
                entry["r"] = np.array([np.corrcoef(xz_te[:, i], y)[0, 1] for i in range(xz_te.shape[1])])
        ref[m] = entry

    freq = {m: {} for m in eval_views}
    gfreq = {m: {} for m in eval_views}
    n_done = 0
    if n_refits > 0:
        n = train[0].shape[0]
        k = int(result["u"].shape[1])
        params = dict((result.get("parameters") or {}).get("used") or {})
        seeds = np.random.SeedSequence(random_state).spawn(n_refits)
        tasks = []
        for r, ss in enumerate(seeds):
            g = np.random.default_rng(ss)
            rows = np.sort(g.choice(n, round(refit_fraction * n), replace=False))
            tasks.append((result["method"], k, params, train, avail, eval_views, outcome,
                          None if outcome is not None else j, rows, names, labels, min_group_size,
                          refit_permutations, shrinkages, int(g.integers(2 ** 31))))
        outs = _run(_refit_task, tasks, n_jobs)
        n_done = len(outs)
        for o in outs:
            for m, (nm, p, pg) in o.items():
                for n_, pp in zip(nm, p):
                    freq[m][n_] = freq[m].get(n_, 0) + int(pp < alpha)
                for g_, pp in pg.items():
                    gfreq[m][g_] = gfreq[m].get(g_, 0) + int(pp < alpha)

    nulls = {m: [] for m in eval_views}
    if n_null_refits > 0:
        k = int(result["u"].shape[1])
        params = dict((result.get("parameters") or {}).get("used") or {})
        seeds = np.random.SeedSequence([random_state, 1]).spawn(n_null_refits * len(eval_views))
        tasks = [(result["method"], k, params, train, avail, m, outcome, None if outcome is not None else j,
                  np.isin(names[m], ref[m]["names"]), shrinkages[m],
                  int(np.random.default_rng(seeds[i * len(eval_views) + q]).integers(2 ** 31)))
                 for i in range(n_null_refits) for q, m in enumerate(eval_views)]
        for m, sh in _run(_null_refit_task, tasks, n_jobs):
            nulls[m].append(sh)

    def p_null(m, idx):
        if not nulls[m]:
            return np.nan
        nl = np.asarray(nulls[m])[:, idx].sum(1)
        return float((1 + (nl >= ref[m]["obs"][idx].sum()).sum()) / (len(nl) + 1))

    def tier(pt, pte):
        if not np.isfinite(pte):
            return int(pt < alpha)
        return 2 if (pt < alpha and pte < alpha) else int(pt < alpha)

    rows_f, rows_g, summ = [], [], []
    for m in eval_views:
        e = ref[m]
        fair = 1.0 / len(e["names"])
        for i, n_ in enumerate(e["names"]):
            pte = np.nan if e["p_te"] is None else float(e["p_te"][i])
            rows_f.append(dict(view=m, view_name=vnames[m], feature=n_, share_train=float(e["obs"][i]),
                               ratio_fair=float(e["obs"][i]) / fair, p_train=float(e["p"][i]),
                               share_test=np.nan if e["obs_te"] is None else float(e["obs_te"][i]), p_test=pte,
                               refit_p05_freq=(freq[m].get(n_, 0) / n_done) if n_done else np.nan,
                               p_null_refit=p_null(m, [i]),
                               r_outcome_test=np.nan if e["r"] is None else float(e["r"][i]),
                               tier=tier(float(e["p"][i]), pte)))
        for g_, idx in e["groups"].items():
            pte = np.nan if e["pg_te"] is None else float(e["pg_te"][g_])
            o = float(e["obs"][idx].sum())
            grouping, _, label = g_.partition(" ")
            rows_g.append(dict(view=m, view_name=vnames[m], grouping=grouping, group=label, n=len(idx),
                               share_train=o, ratio_fair=o / (len(idx) * fair), p_train=float(e["pg"][g_]),
                               share_test=np.nan if e["obs_te"] is None else float(e["obs_te"][idx].sum()),
                               p_test=pte, refit_p05_freq=(gfreq[m].get(g_, 0) / n_done) if n_done else np.nan,
                               p_null_refit=p_null(m, idx),
                               tier=tier(float(e["pg"][g_]), pte)))
        f_m = [r for r in rows_f if r["view"] == m]
        summ.append(dict(view=m, view_name=vnames[m], n_features=len(f_m),
                         n_train=sum(r["p_train"] < alpha for r in f_m),
                         n_test=sum(r["p_test"] < alpha for r in f_m),
                         n_replicated=sum(r["tier"] == 2 for r in f_m), expected_by_chance=alpha * len(f_m),
                         shrinkage=float(e["lam"]), shrinkage_test=float(e["lam_te"])))
    return {"features": pd.DataFrame(rows_f), "groups": pd.DataFrame(rows_g), "summary": pd.DataFrame(summ),
            "component": j, "alpha": alpha, "n_refits": n_done, "n_null_refits": n_null_refits, "n_permutations": n_permutations,
            "n_permutations_test": n_permutations_test if test is not None else 0,
            "view_names": vnames,
            "shrinkage_used": {m: float(ref[m]["lam"]) for m in eval_views},
            "shrinkage_used_test": {m: float(ref[m]["lam_te"]) for m in eval_views}}
