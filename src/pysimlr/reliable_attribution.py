r"""Attribute a target to the features of several views, with the checks that make the answer readable.

`attribute_target` is one call for the question "which features, in which views, move with this target?".
It sets up the fit itself (the target is the *root* view of a target-rooted graph, one component,
conical weights, ``nc_abs``), so a caller does not copy fit settings from another package, and it returns

* ``maps``: one signed ``(p_m,)`` array per view, oriented so that **positive means the feature rises with
  the target** (covariates and group means removed);
* ``score``: the out-of-fold correlation between the latent score and the target, with a flag when the
  latent does not predict (a map of a latent that does not predict describes nothing);
* ``generic``: per view, how much the map and the latent look like the view's leading principal component,
  and, with ``n_null > 0``, what the same quantities are when the target is permuted;
* optionally (``n_bag > 0``) a bagged map with delete-d rescaled intervals and a selection calibrated on
  permuted targets, with the expected number of false selections.

What the map is: the association (Pearson correlation) of each feature with the cross-validated latent
that was fitted to the target. It is not a causal effect and not a unique contribution: a feature that
only tracks a driver scores as high as the driver. See ``docs/reliable_attribution.md``.

``reference=`` freezes a fit made on a large sample: the small sample is only projected on it and the
map is the structure coefficient of the projected score.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .utils import MissingRowsWarning, center_within_groups, group_codes, within_group_permutation

METHODS = ("structure", "univariate")
_Z975 = 1.959963984540054
_GENERIC_MIN_ALIGN = 0.5   # a map is only called generic when it is at least this aligned with the view's PC1


# ---------------------------------------------------------------------------------------------- inputs
def _to_arrays(views: Any) -> Tuple[List[np.ndarray], List[Optional[List[str]]]]:
    if hasattr(views, "shape") and not isinstance(views, (list, tuple)):
        views = [views]
    arrs: List[np.ndarray] = []
    names: List[Optional[List[str]]] = []
    for v in views:
        if hasattr(v, "columns"):
            names.append([str(c) for c in v.columns])
            arrs.append(v.to_numpy(dtype=np.float64))
        else:
            names.append(None)
            a = v.detach().cpu().double().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float64)
            arrs.append(a)
    n = arrs[0].shape[0]
    for m, a in enumerate(arrs):
        if a.ndim != 2:
            raise ValueError(f"view {m} must be 2-D (n, p); got shape {a.shape}.")
        if a.shape[0] != n:
            raise ValueError(f"view {m} has {a.shape[0]} rows; view 0 has {n}.")
        part = np.isnan(a).any(1) & ~np.isnan(a).all(1)
        if part.any() or np.isinf(a).any():
            raise ValueError(f"view {m} has partially missing rows or infinite values; only whole rows may be "
                             "missing (all-NaN), as in `fit(..., missing_rows='mask')`. Impute or drop those rows.")
    return arrs, names


def _target_array(target: Any, n: int) -> np.ndarray:
    y = target.to_numpy(dtype=np.float64) if hasattr(target, "to_numpy") else np.asarray(
        target.detach().cpu().double().numpy() if torch.is_tensor(target) else target, dtype=np.float64)
    y = y.reshape(-1) if y.ndim == 2 and y.shape[1] == 1 else y
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"target must have one value per row ({n}); got shape {y.shape}.")
    return y


def _cov_array(cov: Any, n: int) -> Optional[np.ndarray]:
    if cov is None:
        return None
    c = cov.to_numpy(dtype=np.float64) if hasattr(cov, "to_numpy") else np.asarray(cov, dtype=np.float64)
    c = c[:, None] if c.ndim == 1 else c
    if c.shape[0] != n:
        raise ValueError(f"covariates must have one row per observation ({n}); got {c.shape[0]}.")
    if not np.isfinite(c).all():
        raise ValueError("covariates contain non-finite values; impute or drop those rows first.")
    return c


# ------------------------------------------------------------------------------------ preprocessing
class _Frame:
    """Within-group centering, covariate residualisation and z-scoring of the views and the target,
    estimated on the rows it is fitted on and applied unchanged to other rows (frozen reference)."""

    def __init__(self) -> None:
        self.state: List[Dict[str, np.ndarray]] = []   # one per view, then the target

    @staticmethod
    def _design(cov: Optional[np.ndarray], codes: Optional[np.ndarray], obs: np.ndarray) -> np.ndarray:
        d = np.ones((int(obs.sum()), 1))
        if cov is not None:
            c = cov[obs]
            if codes is not None:
                c = center_within_groups(c, codes[obs])
            d = np.column_stack([d, c])
        return d

    def _one(self, a: np.ndarray, cov, codes, st: Optional[Dict[str, np.ndarray]]):
        obs = ~np.isnan(a).all(1)
        out = np.full(a.shape, np.nan)
        ao = a[obs]
        if codes is not None:
            ao = center_within_groups(ao, codes[obs])
        d = self._design(cov, codes, obs)
        if st is None:
            coef = np.linalg.lstsq(d, ao, rcond=None)[0]
            r = ao - d @ coef
            mu = r.mean(0)
            sd = r.std(0, ddof=1) if r.shape[0] > 1 else np.ones(r.shape[1])
            st = {"coef": coef, "mu": mu, "sd": np.where(sd > 0, sd, 1.0)}
        else:
            r = ao - d @ st["coef"]
        out[obs] = (r - st["mu"]) / st["sd"]
        return out, st

    def fit_transform(self, arrs: Sequence[np.ndarray], y: np.ndarray, cov, codes):
        self.state = []
        zs = []
        for a in list(arrs) + [y[:, None]]:
            z, st = self._one(a, cov, codes, None)
            self.state.append(st)
            zs.append(z)
        return zs[:-1], zs[-1][:, 0]

    def transform(self, arrs: Sequence[np.ndarray], y: Optional[np.ndarray], cov, codes):
        zs = [self._one(a, cov, codes, st)[0] for a, st in zip(arrs, self.state[:-1])]
        zy = None if y is None else self._one(y[:, None], cov, codes, self.state[-1])[0][:, 0]
        return zs, zy


# ----------------------------------------------------------------------------------------- fit pieces
def target_root_fit_params(n_views: int, sparseness: float = 0.8, iterations: int = 300) -> Dict[str, Any]:
    """The `pysimlr.fit("simlr", ...)` keyword arguments that make the last view (index ``n_views``) the
    root: every feature view is coupled to the target and to nothing else (``topology="graph"``), conical
    weights, ``nc_abs``, PCA initialisation, whole-row missingness masked. The target view has no sparseness.
    """
    root = int(n_views)
    pvp = {i: {"positivity": "conical", "sparseness": float(sparseness), "energy_type": "nc_abs"} for i in range(root)}
    pvp[root] = {"positivity": "conical", "sparseness": None, "energy_type": "nc_abs"}
    graph = {i: [root] for i in range(root)}
    graph[root] = list(range(root))
    return {"topology": "graph", "path_graph": graph, "per_view_params": pvp, "initialization_type": "pca",
            "iterations": int(iterations), "missing_rows": "mask", "dynamic_weights": False}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _col_corr(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-column correlation of ``x`` (rows with NaN skipped as whole rows) with ``t``."""
    ok = ~np.isnan(x).all(1) & np.isfinite(t)
    xc = x[ok] - x[ok].mean(0)
    tc = t[ok] - t[ok].mean()
    nx = np.sqrt((xc ** 2).sum(0))
    nt = float(np.sqrt((tc ** 2).sum()))
    if nt == 0:
        return np.zeros(x.shape[1])
    return np.where(nx > 0, (xc.T @ tc) / np.where(nx > 0, nx, 1.0) / nt, 0.0)


def _fit_root(zs: Sequence[np.ndarray], yz: np.ndarray, *, sparseness: float, seed: int,
              fit_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from .api import fit
    t = [torch.as_tensor(z, dtype=torch.float32) for z in zs] + [torch.as_tensor(yz[:, None], dtype=torch.float32)]
    params = target_root_fit_params(len(zs), sparseness)
    params.update(fit_options or {})
    return fit("simlr", t, 1, random_state=int(seed), **params)


def _signed_fit(zs, yz, *, sparseness, seed, fit_options):
    """Fit; return ``(result, u (n,), target_sign, per-view maps, per-view signed weights)``.

    ``target_sign`` makes the latent correlate positively with the target. The per-view weight sign is the
    sign of that view's own coupling to the latent (``corr(X_m v_m, u)``): conical weights are non-negative,
    yet a view can couple negatively to the consensus, and its raw weights then describe the opposite
    direction."""
    res = _fit_root(zs, yz, sparseness=sparseness, seed=seed, fit_options=fit_options)
    u = res["u"].detach().cpu().double().numpy()[:, 0]
    c = _corr(u, yz)
    s = -1.0 if (np.isfinite(c) and c < 0) else 1.0
    t = [torch.as_tensor(z, dtype=torch.float32) for z in zs]
    masks = [torch.as_tensor(~np.isnan(z).all(1)) for z in zs]
    use_masks = None if all(bool(m.all()) for m in masks) else masks
    from .loadings import structure_coefficients
    ob = ~np.isnan(zs[0]).all(1) if use_masks is None else None
    # structure on the full latent column; rows unobserved in a view are skipped by obs_masks
    st = structure_coefficients(t if use_masks is None else [torch.nan_to_num(x) for x in t],
                                torch.as_tensor(u[:, None]), obs_masks=use_masks)
    maps = [s * a.double().numpy()[:, 0] for a in st]
    weights = []
    for m, z in enumerate(zs):
        v = res["v"][m].detach().cpu().double().numpy()[:, 0]
        coup = _corr(np.nan_to_num(z) @ v, u) if use_masks is None else _corr(np.where(masks[m].numpy(), np.nan_to_num(z) @ v, np.nan), u)
        weights.append(s * (-1.0 if (np.isfinite(coup) and coup < 0) else 1.0) * v)
    return res, u, s, maps, weights


# --------------------------------------------------------------------------------- generic variance
def _pc1(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(loading map (p,), score (n,))`` of the view's first principal component on its observed rows."""
    obs = ~np.isnan(z).all(1)
    x = z[obs] - z[obs].mean(0)
    uu, s, vt = np.linalg.svd(x, full_matrices=False)
    score = np.full(z.shape[0], np.nan)
    score[obs] = uu[:, 0] * s[0]
    return vt[0], score


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    c = _corr(a, b)
    return abs(c) if np.isfinite(c) else float("nan")


def _generic_report(zs, maps, u, null_maps, null_us, tol) -> Dict[str, Any]:
    rows = []
    for m, z in enumerate(zs):
        load, score = _pc1(z)
        row = {"map_pc1_absr": _abs_corr(maps[m], load), "latent_pc1_absr": _abs_corr(u, score)}
        if null_maps:
            row["null_map_pc1_absr"] = float(np.mean([_abs_corr(nm[m], load) for nm in null_maps]))
            row["null_latent_pc1_absr"] = float(np.mean([_abs_corr(un, score) for un in null_us]))
            row["map_excess"] = row["map_pc1_absr"] - row["null_map_pc1_absr"]
            row["latent_excess"] = row["latent_pc1_absr"] - row["null_latent_pc1_absr"]
            row["generic"] = bool(row["map_pc1_absr"] >= _GENERIC_MIN_ALIGN and row["map_excess"] < tol)
        else:
            row.update(null_map_pc1_absr=None, null_latent_pc1_absr=None, map_excess=None, latent_excess=None, generic=None)
        rows.append(row)
    return {"views": rows, "n_null": len(null_maps or []), "tolerance": float(tol),
            "doc": "map_pc1_absr: |corr| of the map with the view's PC1 loading map; latent_pc1_absr: |corr| of "
                   "the latent with the view's PC1 score; null_*: the same, mean over refits on permuted targets; "
                   "excess = observed - null; generic = |corr| with PC1 of at least 0.5 and excess below tolerance (the "
                   "map is as PC1-like as what a target with no signal produces)."}


# ------------------------------------------------------------------------------------ cross-validation
def _fit_core(method: str, zs, yz, *, sparseness, seed, fit_options):
    """``(maps, u, target_sign, weights, result)`` of one fit on prepared rows."""
    if method == "structure":
        res, u, s, maps, weights = _signed_fit(zs, yz, sparseness=sparseness, seed=seed, fit_options=fit_options)
        return maps, u, s, weights, res
    maps = [_col_corr(z, yz) for z in zs]
    u = sum(np.nan_to_num(z) @ mp for z, mp in zip(zs, maps))
    return maps, u, 1.0, maps, None


def _score_new(method: str, core, zs_new) -> np.ndarray:
    """Out-of-sample latent score of prepared rows, sign-oriented to the target."""
    maps, _, s, _, res = core
    if method == "structure":
        from .api import predict
        t = [torch.as_tensor(z, dtype=torch.float32) for z in zs_new] + [None]
        return s * predict(res, t)["u"].detach().cpu().double().numpy()[:, 0]
    return sum(np.nan_to_num(z) @ mp for z, mp in zip(zs_new, maps))


def _oof_score(method, arrs, y, cov, codes, rows, n_folds, seed, sparseness, fit_options) -> np.ndarray:
    idx = np.flatnonzero(rows)
    n_folds = int(max(2, min(n_folds, len(idx))))
    folds = np.array_split(np.random.default_rng(seed + 1).permutation(idx), n_folds)
    pred = np.full(len(y), np.nan)
    sub = lambda a, i: None if a is None else a[i]
    for f in folds:
        tr = np.setdiff1d(idx, f)
        fr = _Frame()
        zs_tr, yz_tr = fr.fit_transform([a[tr] for a in arrs], y[tr], sub(cov, tr), sub(codes, tr))
        core = _fit_core(method, zs_tr, yz_tr, sparseness=sparseness, seed=seed, fit_options=fit_options)
        zs_te, _ = fr.transform([a[f] for a in arrs], None, sub(cov, f), sub(codes, f))
        pred[f] = _score_new(method, core, zs_te)
    return pred


# ------------------------------------------------------------------------------------- bagged uncertainty
def _bag_maps(zs, yz, strata, *, n_bag, fraction, seed, sparseness, fit_options, n_jobs):
    """``(F (B, p_total) oriented refit structure maps, converged_fraction, m, n)`` from `loading_inference`
    refits on ``fraction`` of the rows (stratified by ``strata``), the target as root view."""
    from .loadings import loading_inference
    t = [torch.as_tensor(z, dtype=torch.float32) for z in zs] + [torch.as_tensor(yz[:, None], dtype=torch.float32)]
    params = target_root_fit_params(len(zs), sparseness)
    params.update(fit_options or {})
    n = int(yz.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # the unconverged-fraction warning is reported in the output
        li = loading_inference("simlr", t, 1, n_subsamples=int(n_bag), fraction=float(fraction), subsample="fraction",
                               strata=strata, return_refits=("rows",), random_state=int(seed),
                               draw_random_state=int(seed) + 101, n_jobs=int(n_jobs), **params)
    s_all = np.asarray(li["refits"]["structure"], dtype=np.float64)[:, :, 0]          # (B, p_total + 1)
    sgn = -1.0 if s_all[:, -1].mean() < 0 else 1.0
    m = max(3, int(round(fraction * n)))
    return sgn * s_all[:, :-1], float(li["converged_fraction"]), m, n


def _delete_d_summary(f: np.ndarray, m: int, n: int) -> Dict[str, np.ndarray]:
    """Bagged map and delete-d rescaled standard error of ``B`` subsample maps of ``m`` of ``n`` rows.

    Delete-d jackknife variance (Shao and Tu 1995): ``(n - d) / (d B) sum_b (f_b - mean)^2`` with
    ``d = n - m``; the plain spread of subsample maps understates the variability at ``n`` by that factor."""
    d = max(n - m, 1)
    mean = f.mean(0)
    se = np.sqrt((m / d) * f.var(0))
    ratio = np.where(se > 1e-12, mean / np.where(se > 1e-12, se, 1.0), 0.0)
    return {"mean": mean, "se": se, "ci_low": mean - _Z975 * se, "ci_high": mean + _Z975 * se, "ratio": ratio}


def _calibrated_selection(ratio: np.ndarray, null_ratio: np.ndarray, fdr: float) -> Dict[str, Any]:
    """Select the largest set of top-``|ratio|`` features whose permuted-target expectation of exceedances
    (mean over null runs of the count of ``|null ratio| >= threshold``) is at most ``fdr`` times its size."""
    a = np.abs(ratio)
    order = np.argsort(-a, kind="stable")
    nulls = np.sort(np.abs(null_ratio), axis=1)
    p = len(a)
    thr = a[order]
    exp_false = np.mean([p - np.searchsorted(row, thr, side="left") for row in nulls], axis=0)   # at each cut
    ok = exp_false / np.arange(1, p + 1) <= fdr
    best = int(np.flatnonzero(ok).max()) + 1 if ok.any() else 0
    sel = np.zeros(p, dtype=bool)
    sel[order[:best]] = True
    return {"selected": sel, "n_selected": best, "expected_false_selections": float(exp_false[best - 1]) if best else 0.0,
            "threshold": float(thr[best - 1]) if best else float("inf")}


# ------------------------------------------------------------------------------------------- the call
def _split(vec: np.ndarray, sizes: Sequence[int]) -> List[np.ndarray]:
    return [vec[a:b] for a, b in zip(np.cumsum([0] + list(sizes))[:-1], np.cumsum(list(sizes)))]


def _permute(rng: np.random.Generator, n: int, codes: Optional[np.ndarray]) -> np.ndarray:
    return rng.permutation(n) if codes is None else within_group_permutation(rng, codes)


def _reference_parts(reference: Any):
    """``(result, frame or None, target_sign)`` of a frozen reference: an `attribute_target` output or a raw
    `pysimlr.fit` result of a target-rooted fit (target = last view)."""
    if isinstance(reference, dict) and "frame" in reference and "result" in reference:
        if reference.get("method") != "structure" or reference["result"] is None:
            raise ValueError("reference must come from attribute_target(method='structure').")
        return reference["result"], reference["frame"], float(reference["target_sign"])
    if isinstance(reference, dict) and "v" in reference and "u" in reference:
        u = reference["u"].detach().cpu().double().numpy()[:, 0]
        root = reference["latents"][-1].detach().cpu().double().numpy()[:, 0]
        c = _corr(u, root)
        return reference, None, -1.0 if (np.isfinite(c) and c < 0) else 1.0
    raise TypeError("reference must be an attribute_target() result or a pysimlr.fit result (target as last view).")


def attribute_target(views: Any, target: Any, *, covariates: Any = None, groups: Any = None,
                     method: str = "structure", reference: Any = None, n_null: int = 0, n_bag: int = 0,
                     sparseness: float = 0.8, n_folds: int = 5, bag_fraction: float = 0.8, fdr: float = 0.1,
                     generic_tol: float = 0.1, random_state: int = 0, n_jobs: int = 1,
                     fit_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r"""Attribute ``target`` to the features of ``views``: signed per-view maps and the checks to read them.

    Parameters
    ----------
    views : list of ``(n, p_m)`` arrays / tensors / DataFrames with matched rows
        A row may be all-NaN in a view (``missing_rows="mask"``); partial NaN rows are refused.
    target : ``(n,)``
        Rows with a non-finite target are dropped with a ``MissingRowsWarning``.
    covariates : ``(n,)`` or ``(n, c)``, optional
        Removed from every feature and from the target (linear residualisation, estimated on the rows
        fitted) before anything else. The map is then the association given the covariates.
    groups : length-n labels, optional
        Site, cohort, scanner. Features, target and covariates are centered within group, the latent
        correlation is therefore within group, and every permutation moves rows only within their group.
    method : {"structure", "univariate"}
        ``"structure"`` (default): fit the target as the root view of a one-component target-rooted
        `simlr` model (`target_root_fit_params`) and map each feature by its correlation with the latent
        (`structure_coefficients`). ``"univariate"``: the covariate-adjusted correlation of each feature with
        the target (a t-map in disguise; cheap, no latent; with ``n_null > 0`` also permutation p-values and
        FDR q-values per view from `stats.permutation_correlations`, ``n_null`` then being the number of
        permutations).
    reference : `attribute_target` output (``method="structure"``) or `pysimlr.fit` result, optional
        Frozen reference fitted on a large sample. The sample is only *projected* on it (no fit); the map
        is the structure coefficient of the projected score, with Fisher-z 95% intervals. A raw `fit`
        result carries no covariate model: pass ``covariates=None`` and pre-adjust the data yourself.
    n_null : int
        Number of permuted-target refits (``method="structure"``) used for the ``generic`` null and, with
        ``n_bag > 0``, the selection calibration. 0 (default) skips them.
    n_bag : int
        0 (default): no uncertainty output. >0: ``n_bag`` refits on ``bag_fraction`` of the rows
        (`loading_inference`) give a bagged map with delete-d rescaled intervals; with ``n_null > 0`` each
        permuted target is bagged too and features are selected so that the expected number of false
        selections is at most ``fdr`` times the number selected. Cost: ``n_bag * (1 + n_null)`` fits.
        In the benchmark it did not improve the map's accuracy (same as the single fit at 60x the cost).
    sparseness : float
        Quantile sparseness of the feature weights (0.2-0.8 made no difference to the maps).
    n_folds : int
        Folds of the out-of-fold score-target correlation.
    generic_tol : float
        A view is flagged ``generic`` when its map's |correlation| with the view's PC1 loading map is at
        least 0.5 and exceeds the permuted-target value by less than this.
    fit_options : dict, optional
        Overrides of the target-root `fit` keyword arguments.

    Returns
    -------
    dict with
    ``maps`` (list of ``(p_m,)`` arrays, **positive = the feature rises with the target**), ``map``
    (concatenated), ``feature_names``, ``weights`` (raw per-view weights with the per-view coupling sign
    applied; for reference only, they agree with the truth far worse than ``maps``), ``view_signs``,
    ``target_sign``; ``score`` (``r``: out-of-fold score-target correlation, or the frozen-reference
    projection's correlation in reference mode; ``kind``; ``bar``: one-sided p=0.01 null value for that n;
    ``predictive``: ``r > bar``; ``oof_scores``); ``generic`` (per view ``map_pc1_absr``,
    ``latent_pc1_absr`` and, with ``n_null``, their permuted-target means, ``map_excess``, ``generic``);
    ``uncertainty`` (``None`` or ``map``/``se``/``ci_low``/``ci_high``/``ratio``, ``selected``,
    ``n_selected``, ``expected_false_selections``); ``inference`` (univariate with ``n_null``);
    ``result`` and ``frame`` (the fit and the preprocessing; pass the whole dict as ``reference=`` later);
    ``rows_used``, ``mode``, ``method``, ``settings``.

    The map is the association of each feature with the cross-validated latent, not a causal effect and
    not a unique contribution. If ``score["predictive"]`` is false it describes nothing; if
    ``generic[view]["generic"]`` is true it is no more specific than what a target with no signal gives.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}.")
    n_null, n_bag = int(n_null), int(n_bag)
    if n_null < 0 or n_bag < 0:
        raise ValueError("n_null and n_bag must be >= 0.")
    if n_bag and not 0.5 <= float(bag_fraction) < 1.0:
        raise ValueError(f"bag_fraction must be in [0.5, 1); got {bag_fraction}.")
    arrs, names = _to_arrays(views)
    n = arrs[0].shape[0]
    y = _target_array(target, n)
    cov = _cov_array(covariates, n)
    codes = None if groups is None else group_codes(groups, n)
    observed_any = np.any([~np.isnan(a).all(1) for a in arrs], axis=0)
    rows = np.isfinite(y) & observed_any
    if not rows.all():
        warnings.warn(f"attribute_target: dropped {int((~rows).sum())} row(s) with a non-finite target or no observed "
                      f"view; {int(rows.sum())} remain.", MissingRowsWarning, stacklevel=2)
    if rows.sum() < 8:
        raise ValueError(f"only {int(rows.sum())} usable rows; attribution needs at least 8.")
    a_r = [a[rows] for a in arrs]
    y_r = y[rows]
    c_r = None if cov is None else cov[rows]
    g_r = None if codes is None else codes[rows]
    n_r = int(rows.sum())
    sizes = [a.shape[1] for a in a_r]
    n_eff = n_r if g_r is None else n_r - int(g_r.max())      # n - G + 1 within-group rows
    from .extension import _null_bar
    bar = _null_bar(n_eff)
    rng = np.random.default_rng(random_state)
    out: Dict[str, Any] = {"method": method, "n": n, "n_used": n_r, "rows_used": rows, "view_sizes": sizes,
                           "feature_names": names, "uncertainty": None, "inference": None}
    settings = {"sparseness": float(sparseness), "n_folds": int(n_folds), "n_null": n_null, "n_bag": n_bag,
                "bag_fraction": float(bag_fraction), "fdr": float(fdr), "random_state": int(random_state),
                "covariates": cov is not None, "groups": codes is not None,
                "fit_params": None if method != "structure" else target_root_fit_params(len(a_r), sparseness)}
    out["settings"] = settings

    def full(v):                        # a length-n_r vector on all n rows
        f = np.full(n, np.nan)
        f[rows] = v
        return f

    # ------------------------------------------------------------------------------ frozen reference
    if reference is not None:
        if method != "structure":
            raise ValueError("reference= requires method='structure'.")
        if n_bag:
            raise ValueError("n_bag is not available with reference=: the score is frozen, so resampling the small "
                             "sample only reproduces the Fisher-z intervals returned in uncertainty.")
        if n_null:
            warnings.warn("n_null is ignored with reference=: the frozen score does not depend on the target.", UserWarning, stacklevel=2)
        ref_res, frame, sgn = _reference_parts(reference)
        if frame is None:
            if cov is not None:
                raise ValueError("a raw fit result carries no covariate model; pass covariates=None (pre-adjust the data) "
                                 "or pass the attribute_target() output as reference.")
            zs = [a.copy() for a in a_r]
            if g_r is not None:
                zs = [np.where(np.isnan(z), np.nan, center_within_groups(np.nan_to_num(z), g_r)) for z in zs]
            yz = y_r - y_r.mean() if g_r is None else center_within_groups(y_r, g_r)
        else:
            zs, yz = frame.transform(a_r, y_r, c_r, g_r)
        from .api import predict
        z = sgn * predict(ref_res, [torch.as_tensor(v, dtype=torch.float32) for v in zs] + [None])["u"].detach().cpu().double().numpy()[:, 0]
        maps = [_col_corr(v, z) for v in zs]
        r = _corr(z, yz)
        se = 1.0 / np.sqrt(max(n_eff - 3, 1))
        zc = [np.arctanh(np.clip(m, -0.999999, 0.999999)) for m in maps]
        out["uncertainty"] = {"kind": "fisher_z_95", "ci_low": [np.tanh(c - _Z975 * se) for c in zc],
                              "ci_high": [np.tanh(c + _Z975 * se) for c in zc], "n_eff": int(n_eff)}
        out.update(mode="reference", maps=maps, map=np.concatenate(maps), weights=None, view_signs=None, target_sign=sgn,
                   result=ref_res, frame=frame)
        out["score"] = {"r": r, "kind": "frozen_reference", "bar": bar, "predictive": bool(np.isfinite(r) and r > bar),
                        "oof_scores": full(z)}
        out["generic"] = _generic_report(zs, maps, z, None, None, generic_tol)
        if not out["score"]["predictive"]:
            warnings.warn(f"attribute_target: the frozen reference's score correlates {r:.3f} with the target here (null bar "
                          f"{bar:.3f}); the map describes a latent that does not predict this target.", UserWarning, stacklevel=2)
        return out

    # ------------------------------------------------------------------------------------------ fit
    frame = _Frame()
    zs, yz = frame.fit_transform(a_r, y_r, c_r, g_r)
    core = _fit_core(method, zs, yz, sparseness=sparseness, seed=random_state, fit_options=fit_options)
    maps, u, sgn, weights, res = core
    view_signs = None
    if res is not None:
        view_signs = [float(np.sign(w @ res["v"][m].detach().cpu().double().numpy()[:, 0]) or 1.0) * sgn
                      for m, w in enumerate(weights)]
    oof = _oof_score(method, a_r, y_r, c_r, g_r, np.ones(n_r, dtype=bool), n_folds, random_state, sparseness, fit_options)
    r = _corr(oof, yz)
    out.update(mode="fit", maps=maps, map=np.concatenate(maps), weights=weights, view_signs=view_signs, target_sign=sgn,
               result=res, frame=frame)
    out["score"] = {"r": r, "kind": "out_of_fold", "bar": bar, "predictive": bool(np.isfinite(r) and r > bar),
                    "oof_scores": full(oof), "n_folds": int(n_folds)}
    if not out["score"]["predictive"]:
        warnings.warn(f"attribute_target: the out-of-fold correlation of the latent with the target is {r:.3f} (null bar "
                      f"{bar:.3f}); the map describes a latent that does not predict this target.", UserWarning, stacklevel=2)
    null_maps, null_us = [], []
    for j in range(n_null):
        yp = yz[_permute(rng, n_r, g_r)]
        nm, nu, _, _, _ = _fit_core(method, zs, yp, sparseness=sparseness, seed=int(random_state) + 1 + j, fit_options=fit_options)
        null_maps.append(nm)
        null_us.append(nu)
    out["generic"] = _generic_report(zs, maps, u, null_maps, null_us, generic_tol)
    if method == "univariate" and n_null:
        from .stats import permutation_correlations
        out["inference"] = [permutation_correlations(z, yz, n_perms=n_null, seed=int(random_state), groups=g_r)
                            for z in zs]
    if n_bag:
        if method != "structure":
            raise ValueError("n_bag needs method='structure' (the univariate map has no refits to bag).")
        out["uncertainty"] = _uncertainty(zs, yz, g_r, sizes, n_bag, n_null, bag_fraction, fdr, sparseness, random_state,
                                          fit_options, n_jobs, rng)
    return out


def _uncertainty(zs, yz, g_r, sizes, n_bag, n_null, fraction, fdr, sparseness, seed, fit_options, n_jobs, rng):
    f, conv, m, n = _bag_maps(zs, yz, g_r, n_bag=n_bag, fraction=fraction, seed=seed, sparseness=sparseness,
                              fit_options=fit_options, n_jobs=n_jobs)
    s = _delete_d_summary(f, m, n)
    un: Dict[str, Any] = {"kind": "delete_d_rescaled", "n_bag": int(n_bag), "subsample_rows": int(m), "delete_d": int(n - m),
                          "converged_fraction": conv, "selected": None, "n_selected": None,
                          "expected_false_selections": None, "fdr": float(fdr), "n_null": int(n_null)}
    for key in ("mean", "se", "ci_low", "ci_high", "ratio"):
        un[key] = _split(s[key], sizes)
    if n_null:
        nulls = []
        for j in range(n_null):
            yp = yz[_permute(rng, len(yz), g_r)]
            fn, _, _, _ = _bag_maps(zs, yp, g_r, n_bag=n_bag, fraction=fraction, seed=int(seed) + 1000 + j,
                                    sparseness=sparseness, fit_options=fit_options, n_jobs=n_jobs)
            nulls.append(_delete_d_summary(fn, m, n)["ratio"])
        sel = _calibrated_selection(s["ratio"], np.array(nulls), fdr)
        un["selected"] = _split(sel["selected"], sizes)
        un.update(n_selected=sel["n_selected"], expected_false_selections=sel["expected_false_selections"],
                  ratio_threshold=sel["threshold"])
    return un
