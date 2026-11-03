r"""Extend a fitted model to modalities it has never seen.

`extend(result, views, new_views)` adds one or more new views (modalities)
to a `pysimlr.fit` result, for any method, and returns a new result; the input
is not modified. Two ways (``how=``):

``how="anchored"`` (default)
    The fitted model is frozen. Its consensus ``t = predict(result, views)["u"]``
    on the subjects that have the new modality is the target, and each new view
    gets a linear encoder ``V_new`` with ``X_new V_new`` matching ``t`` under the
    fit's own energy, positivity, sparseness and NSA-Flow weight. The new view
    lives in the original component frame: its ``structure`` and ``loadings``
    use the fit's ``loading_rotation``, components keep their numbering, and
    `pysimlr.predict` accepts it, alone or with any subset of the original
    views. Predictions from all original views are unchanged, bit for bit.

    For deep and flow results the new view still gets a *linear* encoder into
    the frozen latent space: the target is the model's own consensus, so this
    is exactly what the linear first layer of those models is, without a
    nonlinear head of its own.

``how="reembed"``
    The algorithm of ANTsR's ``extend_simlr_embedding_with_new_modalities``,
    with every option: the existing views' scores (from `predict`, so correct
    for deep and flow models) and the new views' raw features are fitted
    together as a new joint embedding. The original model is not extended; the
    re-embedding is returned as its own result. See `reembed`.

Consensus from a subset of views
--------------------------------
With every original view present, `predict` uses the fit's frozen consensus
map. With a subset (or with extended views), the consensus is a precision-
weighted average of per-view affine maps fitted on the training rows,
``u_hat = sum_m w_m (z_m B_m + b_m) / sum_m w_m`` over the available views m,
with ``B_m, b_m`` the least-squares map from view m's latent to the training
consensus and ``w_m = R^2_m / (1 - R^2_m)`` from that fit. A view that
predicts the consensus well dominates; one that does not is down-weighted,
never dropped.
"""
import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .defaults import DEFAULT_SIMLR_ITERATIONS, DEFAULT_TOL

EXTEND_MODES = ("anchored", "reembed")
#: Largest mean gap (in correlation) between a new view's encoder and the
#: least-squares bound for it to count as converged; see `fit_anchored_encoder`.
BOUND_GAP_TOL = 0.01


# ----------------------------------------------------------------------------- views and names
def _default_names(m: int) -> List[str]:
    return [f"view{i}" for i in range(m)]


def view_names(result: Dict[str, Any]) -> List[str]:
    """Names of every view a result knows: the original views (``result["view_names"]``,
    or ``view0 ...``), then any extended views, in order."""
    m = len(result["v"])
    names = list(result.get("view_names") or _default_names(m))[:m]
    return names + [n for n in (result.get("extensions") or {})]


def _to_tensor(m) -> torch.Tensor:
    if hasattr(m, "to_numpy"):
        m = m.to_numpy()
    t = m.detach() if isinstance(m, torch.Tensor) else torch.as_tensor(np.asarray(m, dtype=np.float64))
    t = t.float()
    if t.ndim != 2:
        raise ValueError(f"each view must be 2-D (samples x features); got shape {tuple(t.shape)}.")
    return t


def split_views(result: Dict[str, Any], views: Any) -> Tuple[List[Optional[torch.Tensor]], Dict[str, torch.Tensor]]:
    """``views`` as (original views in fit order, ``None`` where missing; extended views by name).

    ``views`` is a list (original order; ``None`` for a missing view; no
    extended views) or a dict keyed by view name (see `view_names`) or by
    original index.
    """
    m = len(result["v"])
    orig_names = view_names(result)[:m]
    ext_names = list((result.get("extensions") or {}).keys())
    if isinstance(views, dict):
        orig: List[Optional[torch.Tensor]] = [None] * m
        ext: Dict[str, torch.Tensor] = {}
        for key, x in views.items():
            if x is None:
                continue
            if isinstance(key, (int, np.integer)) and 0 <= int(key) < m:
                orig[int(key)] = _to_tensor(x)
            elif key in orig_names:
                orig[orig_names.index(key)] = _to_tensor(x)
            elif key in ext_names:
                ext[key] = _to_tensor(x)
            else:
                raise ValueError(f"unknown view {key!r}; this result has views {orig_names + ext_names}.")
    else:
        items = list(views) if isinstance(views, (list, tuple)) else [views]
        if len(items) != m:
            raise ValueError(f"expected {m} original views (None for a missing one); got {len(items)}. "
                             f"Pass a dict keyed by view name to include extended views.")
        orig = [None if x is None else _to_tensor(x) for x in items]
        ext = {}
    rows = {t.shape[0] for t in orig if t is not None} | {t.shape[0] for t in ext.values()}
    if not rows:
        raise ValueError("no views given.")
    if len(rows) != 1:
        raise ValueError(f"views have different numbers of rows: {sorted(rows)}.")
    return orig, ext


# ----------------------------------------------------------------------------- per-view consensus maps
def _affine_map(z: torch.Tensor, u: torch.Tensor) -> Dict[str, Any]:
    """Least-squares ``u ~ z B + b`` and the precision weight ``R^2 / (1 - R^2)``."""
    z64, u64 = z.double(), u.double()
    zm, um = z64.mean(0, keepdim=True), u64.mean(0, keepdim=True)
    zc, uc = z64 - zm, u64 - um
    B = torch.linalg.lstsq(zc, uc, driver="gelsd").solution
    res = uc - zc @ B
    r2 = 1.0 - float((res ** 2).sum()) / max(float((uc ** 2).sum()), 1e-300)
    r2 = min(max(r2, 0.0), 1.0 - 1e-6)
    return {"B": B, "b": um - zm @ B, "r2": r2, "weight": r2 / (1.0 - r2)}


def consensus_maps(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per original view, the affine map from its training latent to the training consensus."""
    cached = result.get("_view_maps")
    if cached is not None:
        return cached
    u = torch.as_tensor(result["u"])
    lat = result["latents"][:len(result["v"])]
    if len(lat) != len(result["v"]):
        raise ValueError(f"{result.get('method')} has a joint latent; per-view prediction from a subset "
                         f"of views is not defined.")
    return [_affine_map(torch.as_tensor(z), u) for z in lat]


def subset_consensus(latents: Dict[int, torch.Tensor], maps: Dict[int, Dict[str, Any]]) -> torch.Tensor:
    """Precision-weighted average of the mapped latents of the available views."""
    num, den = None, 0.0
    for key, z in latents.items():
        mp = maps[key]
        if mp["weight"] <= 0:
            continue
        pred = z.double() @ mp["B"] + mp["b"]
        num = pred * mp["weight"] if num is None else num + pred * mp["weight"]
        den += mp["weight"]
    if num is None:
        raise ValueError("none of the available views predicts the consensus (all weights are 0).")
    return (num / den).float()


def _fill_missing(result: Dict[str, Any], orig: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
    """Stand-ins for missing original views (all NaN, filled by the replayed
    preprocessing), so the method's predictor runs. Their latents are discarded."""
    n = next(t.shape[0] for t in orig if t is not None)
    sizes = [int(v.shape[0]) for v in result["v"]]
    return [t if t is not None else torch.full((n, sizes[i]), float("nan")) for i, t in enumerate(orig)]


def _ext_latent(result: Dict[str, Any], name: str, x: torch.Tensor) -> torch.Tensor:
    from .utils import preprocess_data
    e = result["extensions"][name]
    xp = preprocess_data(x, e["scale_list"], e["provenance"])
    return xp.float() @ e["v"].float()


def predict_views(result: Dict[str, Any], views: Any) -> Dict[str, Any]:
    """`pysimlr.predict` for any available subset of original and extended views.

    Returns ``u`` (consensus from the available views), ``latents`` (a dict
    keyed by view name, available views only), ``views_used``, and ``r2`` /
    ``mse`` / ``reconstructions`` for the original views that were given.
    """
    from .api import predict as _predict
    orig, ext = split_views(result, views)
    names = view_names(result); m = len(result["v"])
    have = [i for i, t in enumerate(orig) if t is not None]
    base = _predict({k: v for k, v in result.items() if k not in ("extensions",)}, _fill_missing(result, orig)) \
        if have else None
    lat: Dict[Any, torch.Tensor] = {}
    if base is not None:
        for i in have:
            lat[i] = torch.as_tensor(base["latents"][i]).float()
    for nm, x in ext.items():
        lat[nm] = _ext_latent(result, nm, x)
    maps: Dict[Any, Dict[str, Any]] = {}
    if have:
        om = consensus_maps(result)
        maps.update({i: om[i] for i in have})
    for nm in ext:
        maps[nm] = result["extensions"][nm]["map"]
    u = subset_consensus(lat, maps)
    out = {"u": u, "latents": {(names[k] if isinstance(k, int) else k): v for k, v in lat.items()},
           "views_used": [names[i] for i in have] + list(ext)}
    if base is not None:
        out["r2"] = {names[i]: base["r2"][i] for i in have}
        out["mse"] = {names[i]: base["mse"][i] for i in have}
        out["reconstructions"] = {names[i]: base["reconstructions"][i] for i in have}
    return out


# ----------------------------------------------------------------------------- anchored encoder
def _used(result: Dict[str, Any], key: str, default=None):
    prm = result.get("parameters", {}) or {}
    used, defaults = prm.get("used", {}) or {}, prm.get("defaults", {}) or {}
    return used.get(key, defaults.get(key, result.get(key, default)))


def _corr_cols(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a = a.double() - a.double().mean(0); b = b.double() - b.double().mean(0)
    den = torch.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)).clamp_min(1e-300)
    return ((a * b).sum(0) / den).numpy()


def _null_bar(n: int, alpha: float = 0.01) -> float:
    """One-sided Fisher-z critical r for ``n`` rows (the `select_rank` oos_coupling bar)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - alpha)
    return float(math.tanh(z / math.sqrt(max(n - 3, 1))))


def fit_anchored_encoder(x: torch.Tensor, target: torch.Tensor, *, energy_type: str = "nc",
                         positivity: str = "either", sparseness=None, nsa_w: float = 0.5,
                         iterations: int = DEFAULT_SIMLR_ITERATIONS, tol: float = DEFAULT_TOL,
                         patience: int = 10, learning_rate: float = 1.0) -> Dict[str, Any]:
    """A linear basis ``V`` (p x k) with ``x V`` matching the frozen ``target`` (n x k).

    Projected gradient with backtracking on ``calculate_simlr_energy(V, x,
    target, energy_type)``; after every step ``V`` is projected exactly as
    `simlr` projects a view (`sparsification.simlr_sparseness`: NSA-Flow
    retraction with weight ``nsa_w``, ``positivity``, quantile
    ``sparseness``). Starts from the cross-covariance ``x' target``. Stops when
    the relative energy change stays below ``tol`` for ``patience`` steps
    (`utils.ConvergenceMonitor`), or when no step size decreases the energy.

    Certificate. The NSA-Flow retraction is not a convex projection, so a
    stalled step (``no_descent``) is not a stationarity certificate (the
    gradient map does not shrink there). For ``energy_type="nc"`` there is an
    objective bound instead: no linear encoder correlates with target column j
    more than the least-squares fit of that column on ``x``. The fit is
    ``converged`` if it stopped by plateau, or if its mean gap to that bound is
    at most ``BOUND_GAP_TOL``; ``bound_gap`` is recorded either way. With
    ``p >= n`` the bound is trivially 1 and certifies nothing.
    """
    from .simlr import calculate_simlr_energy
    from .sparsification import simlr_sparseness
    from .utils import ConvergenceMonitor
    x64, t64 = x.double(), target.double()
    ctype = "ortho" if nsa_w and nsa_w > 0 else "none"

    def project(v):
        return simlr_sparseness(v, constraint_type=ctype, positivity=positivity, sparseness=sparseness,
                                constraint_weight=nsa_w if ctype == "ortho" else None,
                                energy_type=energy_type).double()

    def energy(v):
        return float(calculate_simlr_energy(v, x64, t64, energy_type))

    v = x64.T @ (t64 - t64.mean(0))
    v = project(v / v.norm(dim=0, keepdim=True).clamp_min(1e-12))
    e = energy(v); hist = [e]; lr = float(learning_rate)
    mon = ConvergenceMonitor(tol=tol, patience=patience)
    stop, steps = "max_iter", 0
    for it in range(int(iterations)):
        steps = it + 1
        vv = v.clone().requires_grad_(True)
        calculate_simlr_energy(vv, x64, t64, energy_type).backward()
        g = vv.grad.detach()
        gn = float(g.norm())
        if not math.isfinite(gn) or gn == 0.0:
            stop = "converged" if gn == 0.0 else "non_finite"; break
        step = lr * float(v.norm()) / gn
        accepted = False
        for _ in range(20):
            cand = project(v - step * g)
            ec = energy(cand)
            if math.isfinite(ec) and ec < e:
                accepted = True; break
            step *= 0.5
        if not accepted:
            stop = "no_descent"; break
        v, e = cand, ec; hist.append(e)
        lr = min(1.0, lr * 1.5)
        if mon.update(e):
            stop = "converged"; break
    gap = None
    if energy_type == "nc" and x64.shape[1] < x64.shape[0]:
        tc = t64 - t64.mean(0)
        ls = x64 @ torch.linalg.lstsq(x64, tc).solution
        bound = _corr_cols(ls, tc); got = _corr_cols(x64 @ v, tc)
        gap = float(np.mean(bound - got))
    converged = stop == "converged" or (gap is not None and gap <= BOUND_GAP_TOL)
    return {"v": v.float(), "energy": hist, "convergence": {
        "converged": converged, "stop_reason": stop if stop != "converged" or converged else stop,
        "n_steps": steps, "max_steps": int(iterations), "tol": tol, "patience": patience,
        "bound_gap": gap, "certificate": "objective_bound" if (gap is not None and gap <= BOUND_GAP_TOL) else None}}


def _anchored(result: Dict[str, Any], views: Any, new_views: Dict[str, Any], *, holdout: float = 0.2,
              energy_type: Optional[str] = None, positivity: Optional[str] = None, sparseness=None,
              nsa_w: Optional[float] = None, iterations: int = DEFAULT_SIMLR_ITERATIONS,
              random_state: int = 0, warn: bool = True) -> Dict[str, Any]:
    from .loadings import olee_loadings, structure_coefficients
    from .similarity import check_fit_energy
    from .utils import preprocess_data
    if not isinstance(new_views, dict) or not new_views:
        raise ValueError("new_views must be a non-empty dict mapping a new view name to its matrix.")
    known = view_names(result)
    clash = [nm for nm in new_views if nm in known]
    if clash:
        raise ValueError(f"new view names {clash} already exist in this result ({known}).")
    target = predict_views(result, views)["u"]
    n, k = target.shape
    e_used = energy_type or _used(result, "energy_type", "nc")
    try:
        e_used = check_fit_energy(e_used, path="linear")
    except ValueError:
        warnings.warn(f"energy {e_used!r} has no linear form; the new view's encoder is fitted with 'nc'.",
                      RuntimeWarning, stacklevel=3)
        e_used = "nc"
    pos = positivity or _used(result, "positivity", "either")
    sp = sparseness if sparseness is not None else _used(result, "sparseness", None)
    if isinstance(sp, (list, tuple)):
        sp = sp[0]
    w = nsa_w if nsa_w is not None else _used(result, "nsa_w", 0.5)
    R = torch.as_tensor(result["loading_rotation"]).double()
    scale_list = result["scale_list"]
    out = copy.copy(result)
    out["view_names"] = view_names(result)[:len(result["v"])]
    out["extensions"] = dict(result.get("extensions") or {})
    for key in ("structure", "loadings", "loading_margin"):
        if out.get(key) is not None:
            out[key] = list(out[key])
    rng = np.random.default_rng(random_state)
    for nm, raw in new_views.items():
        x = _to_tensor(raw)
        if x.shape[0] != n:
            raise ValueError(f"new view {nm!r} has {x.shape[0]} rows; the original views have {n}.")
        xp, prov = preprocess_data(x, scale_list)
        settings = dict(energy_type=e_used, positivity=pos, sparseness=sp, nsa_w=w, iterations=iterations)
        hold = None
        if holdout and 0 < holdout < 1 and n >= 20:
            te = rng.choice(n, max(3, int(round(holdout * n))), replace=False)
            tr = np.setdiff1d(np.arange(n), te)
            xtr, ptr = preprocess_data(x[tr], scale_list)
            fh = fit_anchored_encoder(xtr, target[tr], **settings)
            xte = preprocess_data(x[te], scale_list, ptr)
            r = _corr_cols((xte.float() @ fh["v"]).double() @ R, target[te].double() @ R)
            hold = {"coupling": r.tolist(), "n": int(len(te)), "null_bar": _null_bar(len(te))}
            if warn and float(np.nanmax(r)) < hold["null_bar"]:
                warnings.warn(f"new view {nm!r} does not align with the model: held-out coupling "
                              f"{np.round(r, 3).tolist()} is below the null bar {hold['null_bar']:.3f} "
                              f"on every component.", RuntimeWarning, stacklevel=3)
        fit = fit_anchored_encoder(xp, target, **settings)
        z = xp.float() @ fit["v"]
        struct = structure_coefficients([xp], target.double() @ R)[0]
        loads, margins = olee_loadings([struct], pos)
        out["extensions"][nm] = {
            "v": fit["v"], "scale_list": list(scale_list), "provenance": prov,
            "map": _affine_map(z, target), "energy_type": e_used, "positivity": pos, "sparseness": sp,
            "nsa_w": w, "convergence": fit["convergence"], "energy": fit["energy"],
            "coupling": _corr_cols(z.double() @ R, target.double() @ R).tolist(), "holdout": hold,
            "n_subjects": int(n)}
        for key, val in (("structure", struct), ("loadings", loads[0]),
                         ("loading_margin", None if margins is None else margins[0])):
            if out.get(key) is not None:
                out[key].append(val)
        if not fit["convergence"]["converged"]:
            from .utils import warn_if_unconverged
            warn_if_unconverged(fit["convergence"], f"extend[{nm}]")
    out["_view_maps"] = consensus_maps(result)
    return out


# ----------------------------------------------------------------------------- entry point
def extend(result: Dict[str, Any], views: Any, new_views: Dict[str, Any], *, how: str = "anchored",
           **options) -> Dict[str, Any]:
    """Add new views (modalities) to a fitted model; see the module docstring.

    Parameters
    ----------
    result : dict
        A `pysimlr.fit` result (any method with per-view latents).
    views : list or dict
        The original views for the subjects that have the new modalities: a
        list in fit order (``None`` for a missing view) or a dict keyed by
        view name (``result["view_names"]`` or ``view0 ...``).
    new_views : dict
        New view name -> (n x p) matrix, rows matching ``views``.
    how : {"anchored", "reembed"}
        ``"anchored"`` freezes the model and maps the new views into its
        component frame; ``"reembed"`` fits a new joint embedding (ANTsR's
        algorithm, whose own ``mode`` option is ``"concatenate"`` /
        ``"split"`` / ``"auto"``; see `reembed`).
    **options
        anchored: ``holdout`` (fraction held out for the coupling check,
        default 0.2), ``energy_type``, ``positivity``, ``sparseness``,
        ``nsa_w`` (default: the fit's own), ``iterations``,
        ``random_state``, ``warn``. reembed: see `reembed`.

    Returns
    -------
    dict
        anchored: a copy of ``result`` with ``extensions`` (per new view: its
        encoder ``v``, preprocessing, consensus map, energy, convergence,
        in-sample and held-out ``coupling``), ``view_names``, and the new
        views appended to ``structure``, ``loadings`` and ``loading_margin``.
        `pysimlr.predict` then accepts the new views. reembed: see `reembed`.
    """
    if how not in EXTEND_MODES:
        raise ValueError(f"how must be one of {EXTEND_MODES}; got {how!r}.")
    if how == "anchored":
        return _anchored(result, views, new_views, **options)
    return reembed(result, views, new_views, **options)


REEMBED_MODES = ("concatenate", "split", "auto")


def reembed(result: Dict[str, Any], views: Any, new_views: Dict[str, Any], *, mode: str = "concatenate",
            split_prefixes: Optional[Sequence[str]] = None, adjacency_type: str = "feature_correlation",
            cor_threshold: float = 0.8, use_abs_correlation: bool = True,
            k_new: Optional[Union[int, Dict[str, int]]] = None, k_method: str = "cumulative",
            cumvar_threshold: float = 0.9, min_k: int = 2, joint_k_policy: str = "max",
            preprocess: str = "drop", method: str = "simlr", standardize: bool = True,
            verbose: bool = False, **kwargs) -> Dict[str, Any]:
    """ANTsR's ``extend_simlr_embedding_with_new_modalities`` for any fitted model.

    The existing views' scores (`pysimlr.predict` latents, so deep and flow
    models are handled correctly) and the new views' raw features are fitted
    together as a new joint embedding with ``method``. Every ANTsR option is
    kept, with the same defaults:

    ``mode``
        ``"concatenate"``: all existing scores as one block (``"sim"``);
        ``"split"``: one block per existing view named in ``split_prefixes``
        (view names; default all); ``"auto"``: one block per existing view.
    ``adjacency_type``, ``cor_threshold``, ``use_abs_correlation``
        Per-block feature-correlation adjacency (|r| >= ``cor_threshold``),
        passed as ``smoothing_matrices`` when ``method`` accepts them.
    ``k_new``, ``k_method``, ``cumvar_threshold``, ``min_k``
        Rank of each new view: given (one int, or a dict by name), or chosen by
        cumulative variance / elbow, never below ``min_k``.
    ``joint_k_policy``
        ``"max"`` / ``"median"`` / ``"min"`` of the new views' ranks: the joint
        rank of the re-embedding.
    ``preprocess``
        ``"drop"`` or ``"zero"`` all-NaN and constant columns (recorded).

    Returns ``reembedded`` (the new fit: a full result, usable with
    `pysimlr.predict` and the plots), ``blocks``, ``block_names``,
    ``adjacency``, ``k_new_used``, ``joint_k``, ``diagnostics``, and
    ``updated_result``: ANTsR's copy of ``result`` with the new views' bases
    appended to ``v``. Those bases live in the re-embedding's component frame
    and rank (``joint_k``), not the original's; the original model's ``u``,
    ``structure`` and `predict` do not use them. Use ``how="anchored"`` in
    `extend` to add views to the original model itself.
    """
    from .api import fit as _fit, predict as _predict, resolve_method
    from .nnh import (_build_feature_correlation_adjacency, _combine_joint_k, _preprocess_matrix,
                      _resolve_k_new)
    if mode not in REEMBED_MODES:
        raise ValueError(f"mode must be one of {REEMBED_MODES}; got {mode!r}.")
    if adjacency_type != "feature_correlation":
        raise ValueError("adjacency_type must be 'feature_correlation'.")
    if k_method not in ("cumulative", "elbow"):
        raise ValueError("k_method must be 'cumulative' or 'elbow'.")
    if joint_k_policy not in ("max", "median", "min"):
        raise ValueError("joint_k_policy must be 'max', 'median' or 'min'.")
    if preprocess not in ("drop", "zero"):
        raise ValueError("preprocess must be 'drop' or 'zero'.")
    if not 0.0 <= cor_threshold <= 1.0:
        raise ValueError("cor_threshold must be in [0, 1].")
    if not 0.0 < cumvar_threshold <= 1.0:
        raise ValueError("cumvar_threshold must be in (0, 1].")
    if int(min_k) < 1:
        raise ValueError("min_k must be a positive integer.")
    if not isinstance(new_views, dict) or not new_views:
        raise ValueError("new_views must be a non-empty dict mapping a new view name to its matrix.")
    names = view_names(result)[:len(result["v"])]
    clash = [nm for nm in new_views if nm in view_names(result)]
    if clash:
        raise ValueError(f"new view names {clash} already exist in this result.")
    orig, _ = split_views(result, views)
    if isinstance(views, dict) or any(t is None for t in orig):
        lat = predict_views(result, views)["latents"]
    else:
        pr = _predict(result, orig)
        lat = {names[i]: torch.as_tensor(z).float() for i, z in enumerate(pr["latents"])}
    blocks_raw: Dict[str, np.ndarray] = {}
    block_cols: Dict[str, List[str]] = {}
    if mode == "concatenate":
        cols = [f"{nm}PC{j + 1}" for nm, z in lat.items() for j in range(z.shape[1])]
        blocks_raw["sim"] = np.hstack([z.double().numpy() for z in lat.values()]); block_cols["sim"] = cols
    else:
        keep = list(lat) if mode == "auto" or split_prefixes is None else \
            [nm for nm in lat if any(nm.startswith(p) for p in split_prefixes)]
        if not keep:
            raise ValueError(f"no existing view matches split_prefixes={list(split_prefixes)}; views are {list(lat)}.")
        for nm in keep:
            z = lat[nm]
            blocks_raw[nm] = z.double().numpy(); block_cols[nm] = [f"{nm}PC{j + 1}" for j in range(z.shape[1])]
    n = next(iter(blocks_raw.values())).shape[0]
    for nm, x in new_views.items():
        a = _to_tensor(x).double().numpy()
        if a.shape[0] != n:
            raise ValueError(f"new view {nm!r} has {a.shape[0]} rows; the existing views have {n}.")
        if standardize:
            sd = np.nanstd(a, axis=0, ddof=1); sd = np.where(sd > 0, sd, 1.0)
            a = (a - np.nanmean(a, axis=0)) / sd
        blocks_raw[nm] = a; block_cols[nm] = [f"{nm}_{j}" for j in range(a.shape[1])]
    blocks, prep_diag = {}, {}
    for nm, mat in blocks_raw.items():
        prep = _preprocess_matrix(mat, nm, preprocess_policy=preprocess, verbose=verbose)
        blocks[nm] = prep["matrix"]
        cols = block_cols[nm]
        prep_diag[nm] = {"dropped_columns": [cols[i] for i in prep["dropped"]],
                         "zeroed_columns": [cols[i] for i in prep["zeroed"]],
                         "n_rows": int(prep["matrix"].shape[0]), "n_cols": int(prep["matrix"].shape[1])}
    adjacency = {nm: _build_feature_correlation_adjacency(m, threshold=cor_threshold, use_abs=use_abs_correlation)
                 for nm, m in blocks.items()}
    new_names = list(new_views)
    k_used = _resolve_k_new(k_new=k_new, new_modality_names=new_names, blocks=blocks, k_method=k_method,
                            cumvar_threshold=cumvar_threshold, min_k=int(min_k))
    joint_k = _combine_joint_k(k_used, policy=joint_k_policy, min_k=int(min_k))
    order = list(blocks)
    fit_kwargs = dict(kwargs)
    if "smoothing_matrices" in resolve_method(method).accepts:
        fit_kwargs.setdefault("smoothing_matrices", [torch.tensor(adjacency[nm]).float() for nm in order])
    re = _fit(method, [torch.tensor(blocks[nm]).float() for nm in order], int(joint_k), verbose=verbose,
              **fit_kwargs)
    re["view_names"] = order
    updated = copy.copy(result)
    updated["v"] = list(result["v"]) + [re["v"][order.index(nm)] for nm in new_names]
    updated["view_names"] = names + new_names
    return {"reembedded": re, "updated_result": updated, "blocks": blocks, "block_names": order,
            "adjacency": adjacency, "k_new_used": dict(k_used), "joint_k": int(joint_k),
            "diagnostics": {"mode_used": mode, "split_prefixes_used": None if mode == "concatenate" else
                            [nm for nm in order if nm not in new_names],
                            "adjacency_type": adjacency_type, "cor_threshold": cor_threshold,
                            "use_abs_correlation": use_abs_correlation, "preprocess": preprocess,
                            "n_subjects": int(n), "block_dims": {nm: tuple(blocks[nm].shape) for nm in order},
                            "preprocessing": prep_diag, "k_method": k_method, "cumvar_threshold": cumvar_threshold,
                            "min_k": int(min_k), "joint_k_policy": joint_k_policy, "method": method}}
