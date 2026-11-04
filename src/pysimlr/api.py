"""
One interface to every multi-view method in pysimlr.

The methods (linear `simlr`, the deep `lend`/`ned`/`ned_shared_private`
models, the flow model `flow_v`, and two joint baselines) share one call
shape, one parameter vocabulary and one result contract, so a benchmark is a
loop with no per-method branch::

    import pysimlr
    for name in pysimlr.method_names():
        res = pysimlr.fit(name, views, k=3, iterations=50, random_state=0,
                          energy_type="acc", nsa_w=0.5)
        pred = pysimlr.predict(res, test_views)
        print(name, res["parameters"]["ignored"], pred["u"].shape)

Parameters
----------
Each method accepts exactly the keywords of its native entry point (the
registry derives them from the signature, so it cannot drift). A keyword
that some *other* method accepts but this one does not is recorded in
``result["parameters"]["ignored"]`` with the reason and warned about once
per call site, as Python's default warning filter does (``strict=True``
raises instead). A keyword that *no* method
accepts can only be a misspelling and always raises `TypeError`.

Result contract
---------------
Every result carries ``method``, ``family``, ``parameters``, ``u`` (an
``(n, k)`` tensor), ``v`` (one ``(p_m, k)`` block per view), ``latents``,
``convergence``, ``sparseness``, ``scale_list``, ``provenance_list``,
``random_state`` and rank diagnostics -- per view (``effective_rank``,
``numerical_rank``, ``condition_number``, ``max_column_overlap``), or
``joint_*`` with ``basis_is_joint=True`` for the baselines that fit one basis
on the concatenated views, and the interpretation keys ``structure`` and
``loadings`` (one ``(p_m, k)`` block per view; see `pysimlr.loadings`).
`validate_result` checks it.

``v`` is the encoder that scores new subjects; ``structure`` (feature-latent
correlations) and ``loadings`` (their OLEE summary) are what to read feature
by feature, because ``v`` is weakly identified in the low-variance directions
of each view.

For `simlr`, ``u`` is the star consensus of the fitted views under every
topology; the per-view consensus targets of the ``"loo"`` and ``"graph"``
topologies are in ``u_per_modality``.
"""
from __future__ import annotations

import dataclasses
import inspect
import types
import warnings
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import torch

from .deep import lend_simr, ned_simr, ned_simr_shared_private, predict_deep
from .flows import flow_simr_v
from .optimizers import SIMLR_OPTIMIZER_DEFAULTS
from .simlr import SIMLR_OPTIMIZER_KEYS, predict_simlr, simlr
from .utils import basis_rank_report
from .defaults import DEFAULT_NSA_W, check_nsa_w

FAMILIES = ("linear", "deep", "flow", "baseline")

#: Keys every `fit` result carries.
CONTRACT_KEYS = ("method", "family", "parameters", "u", "v", "latents", "convergence",
                 "sparseness", "scale_list", "provenance_list", "random_state",
                 "structure", "loadings")
RANK_KEYS = ("effective_rank", "numerical_rank", "condition_number", "max_column_overlap")
CONVERGENCE_KEYS = ("converged", "stop_reason", "certificate", "grad_map", "n_steps",
                    "max_steps", "tol", "patience", "best", "final")
#: The only values ``convergence["stop_reason"]`` takes, for every method.
#: ``converged``: the energy plateau rule fired (certificate
#: ``"energy_plateau"``) or, for NSA-Flow, its stationarity certificate;
#: ``max_iter``: the budget ran out; ``no_descent``: no view found a descent
#: step (`simlr`); ``non_finite``: the objective became NaN/Inf;
#: ``closed_form``: an exact eigendecomposition (PCA).
STOP_REASONS = ("converged", "max_iter", "no_descent", "non_finite", "closed_form")

#: Arguments every native fit takes positionally; not parameters.
_POSITIONAL = ("data_matrices", "k")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def as_views(views: Any) -> List[torch.Tensor]:
    """Convert input to a list of 2-D float32 tensors with matched rows.

    Accepts one matrix (a single view) or a list/tuple of matrices; each may
    be a `torch.Tensor`, `np.ndarray` or anything with ``to_numpy()``.
    """
    items = list(views) if isinstance(views, (list, tuple)) else [views]
    if not items:
        raise ValueError("views must contain at least one matrix.")
    out = []
    for i, m in enumerate(items):
        if hasattr(m, "to_numpy"):
            m = m.to_numpy()
        t = m.detach() if isinstance(m, torch.Tensor) else torch.as_tensor(np.asarray(m, dtype=np.float64))
        t = t.float()
        if t.ndim != 2:
            raise ValueError(f"view {i} must be 2-D (samples x features); got shape {tuple(t.shape)}.")
        out.append(t)
    rows = {t.shape[0] for t in out}
    if len(rows) != 1:
        raise ValueError(f"Size mismatch: views have different numbers of rows: "
                         f"{[t.shape[0] for t in out]}.")
    return out


# ---------------------------------------------------------------------------
# baselines: one basis fitted on the concatenated views
# ---------------------------------------------------------------------------
def _closed_form_convergence() -> Dict[str, Any]:
    return {"converged": True, "stop_reason": "closed_form", "certificate": "exact",
            "grad_map": 0.0, "n_steps": 0, "max_steps": 0, "tol": None, "patience": None,
            "best": float("nan"), "final": float("nan")}


def _joint_scale_list(scale: bool) -> List[str]:
    return ["centerAndScale"] if scale else ["center"]


def _preprocess_views(views: List[torch.Tensor], scale_list: List[str],
                      provenance: Optional[List[Dict[str, Any]]] = None):
    """Per-view `utils.preprocess_data`, so the baselines share the package's
    preprocessing (NaN fill, unbiased std) and their provenance replays."""
    from .utils import preprocess_data
    if provenance is None:
        out = [preprocess_data(v, scale_list) for v in views]
        return [o[0] for o in out], [o[1] for o in out]
    return [preprocess_data(v, scale_list, prov) for v, prov in zip(views, provenance)], provenance


def _joint_result(views: List[torch.Tensor], k: int, scale: bool, fit_transform,
                  convergence: Dict[str, Any], sparseness, random_state,
                  extra: Dict[str, Any]) -> Dict[str, Any]:
    """Fit a joint basis on the concatenated, preprocessed views and return the
    result contract. ``fit_transform(xs) -> (u, v_tot, transform)``."""
    scale_list = _joint_scale_list(scale)
    xs_list, provenance = _preprocess_views(views, scale_list)
    xs = torch.cat(xs_list, dim=1).double().numpy()
    u, v_tot, transform = fit_transform(xs)
    sizes = [int(t.shape[1]) for t in views]
    offs = np.cumsum([0] + sizes)
    v_tot_t = torch.as_tensor(v_tot).float()
    u_t = torch.as_tensor(u).float()
    res = {
        "u": u_t, "v": [v_tot_t[offs[i]:offs[i + 1]] for i in range(len(sizes))],
        "v_tot": v_tot_t, "basis_is_joint": True, "latents": [u_t],
        "convergence": convergence, "sparseness": sparseness,
        "scale_list": scale_list, "provenance_list": provenance,
        "random_state": random_state, "view_sizes": sizes, "_transform": transform,
        **{f"joint_{kk}": vv for kk, vv in basis_rank_report([v_tot_t], k, warn=False).items()},
    }
    res.update(extra)
    return res


def pca_joint(data_matrices, k: int, random_state: Optional[int] = None,
              scale: bool = True) -> Dict[str, Any]:
    """PCA on the column-concatenated views, each standardized with
    `utils.preprocess_data` (a baseline).

    One orthonormal basis for all views jointly; ``v`` holds its per-view row
    blocks and the rank diagnostics are reported for the joint basis only
    (``basis_is_joint=True``), because a block of an orthonormal matrix is not
    itself a basis. ``random_state`` is recorded only when scikit-learn chose
    its randomized solver; the full SVD it uses otherwise is deterministic.
    """
    from sklearn.decomposition import PCA
    holder = {}

    def fit_transform(xs):
        pca = PCA(n_components=k, random_state=random_state).fit(xs)
        holder["pca"] = pca
        return pca.transform(xs), pca.components_.T, pca.transform

    res = _joint_result(as_views(data_matrices), k, scale, fit_transform,
                        _closed_form_convergence(), None, None, {})
    pca = holder["pca"]
    res["estimator"] = pca
    if getattr(pca, "_fit_svd_solver", None) == "randomized":
        res["random_state"] = random_state
    return res


def nsa_pipeline(data_matrices, k: int, nsa_w: float = DEFAULT_NSA_W, consolidate: bool = False,
                 sparseness: Optional[float] = None, scale: bool = True) -> Dict[str, Any]:
    """NSA-Flow dimensionality reduction on the concatenated views (a baseline).

    The `nsa_flow` scikit-learn estimator (``mode="auto"``) on the
    concatenation of the views, each standardized with `utils.preprocess_data`;
    ``sparseness`` must be a scalar or ``None`` because the basis is joint.
    Deterministic: there is no ``random_state``.
    """
    from .nsa_backend import load_nsa_estimator
    nsa_w = check_nsa_w(nsa_w)
    cls = load_nsa_estimator()
    if cls is None:
        raise ImportError("nsa_pipeline needs the nsa_flow package (pip install nsa-flow).")
    if isinstance(sparseness, (list, tuple)):
        raise ValueError("nsa_pipeline fits one joint basis; sparseness must be a scalar or None.")
    holder = {}

    def fit_transform(xs):
        est = cls(n_components=k, w=nsa_w, mode="auto", consolidate=consolidate, sparseness=sparseness)
        est.fit(xs)
        holder["est"] = est
        return est.transform(xs), est.components_.T, est.transform

    res = _joint_result(as_views(data_matrices), k, scale, fit_transform,
                        _closed_form_convergence(), sparseness, None, {})
    est = holder["est"]
    converged = bool(getattr(est, "converged_", False))
    res["convergence"] = {
        "converged": converged, "stop_reason": "converged" if converged else "max_iter",
        "certificate": getattr(est, "certificate_", None),
        "grad_map": float(getattr(est, "grad_map_", float("nan"))),
        "n_steps": getattr(est, "n_iter_", None), "max_steps": None, "tol": None,
        "patience": None, "best": float("nan"), "final": float("nan")}
    res["estimator"] = est
    from .utils import warn_if_unconverged
    warn_if_unconverged(res["convergence"], "nsa_pipeline")
    return res


def _predict_joint(views: List[torch.Tensor], result: Dict[str, Any]) -> Dict[str, Any]:
    """Project and reconstruct in the preprocessed (standardized) space, the
    space every other method reconstructs in."""
    xs_list, _ = _preprocess_views(views, result["scale_list"], result["provenance_list"])
    xs = torch.cat(xs_list, dim=1).double().numpy()
    u = result["_transform"](xs)
    rec = u @ result["v_tot"].double().numpy().T
    offs = np.cumsum([0] + result["view_sizes"])
    recs = [torch.as_tensor(rec[:, offs[i]:offs[i + 1]]).float() for i in range(len(views))]
    u_t = torch.as_tensor(u).float()
    errors = [float(torch.norm(x.float() - r) / (torch.norm(x.float()) + 1e-10)) for x, r in zip(xs_list, recs)]
    return {"u": u_t, "latents": [u_t], "reconstructions": recs, "errors": errors}


def _predict_linear(views, result):
    return predict_simlr(views, result)


def _predict_deep(views, result):
    return predict_deep(views, result)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MethodSpec:
    """One registered method.

    ``accepts`` and ``defaults`` are derived from the signature of ``fit``
    (plus ``extra_params`` for keywords it takes through ``**kwargs``), so the
    registry cannot disagree with the function it calls.
    """
    name: str
    family: str
    fit: Callable[..., Dict[str, Any]]
    predict: Callable[[List[torch.Tensor], Dict[str, Any]], Dict[str, Any]]
    budget_unit: str
    description: str
    presets: Mapping[str, Any] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))
    extra_params: Mapping[str, Any] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))

    @property
    def defaults(self) -> Dict[str, Any]:
        sig = inspect.signature(self.fit)
        out = {n: p.default for n, p in sig.parameters.items()
               if n not in _POSITIONAL and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        out.update(self.extra_params)
        return out

    @property
    def accepts(self) -> frozenset:
        """Parameters a caller may set; a preset is fixed, so it is not one."""
        return frozenset(self.defaults) - frozenset(self.presets)


def _simlr_extra(keys=None) -> Dict[str, Any]:
    """`simlr`'s optimizer hyperparameters as registry extras: all of them for
    `simlr` (whose optimizer is itself a parameter), or only ``keys`` for a
    preset whose fixed optimizer reads nothing else."""
    keys = SIMLR_OPTIMIZER_KEYS if keys is None else keys
    extra = {kk: SIMLR_OPTIMIZER_DEFAULTS[kk] for kk in keys}
    if "learning_rate" in extra:
        extra["learning_rate"] = "auto"  # `simlr`'s own default, not the optimizer's
    return extra


def _frozen(d: Mapping[str, Any]) -> Mapping[str, Any]:
    return types.MappingProxyType(dict(d))


_SPECS = [
    MethodSpec("simlr", "linear", simlr, _predict_linear, "iterations",
               "Linear SiMLR: one sparse, NSA-Flow-constrained basis per view.", extra_params=_frozen(_simlr_extra())),
    MethodSpec("lend", "deep", lend_simr, _predict_deep, "epochs",
               "LEND: linear first layer, nonlinear decoders."),
    MethodSpec("ned", "deep", ned_simr, _predict_deep, "epochs",
               "NED: linear first layer, nonlinear heads and decoders."),
    MethodSpec("ned_shared_private", "deep", ned_simr_shared_private, _predict_deep, "epochs",
               "NED++: NED with per-view private latents."),
    MethodSpec("flow_v", "flow", flow_simr_v, _predict_deep, "epochs",
               "Flow-SiMR-V: linear first layer, RealNVP flow per view."),
    MethodSpec("pca", "baseline", pca_joint, _predict_joint, "none",
               "PCA of the concatenated standardized views (joint basis)."),
    MethodSpec("nsa_pipeline", "baseline", nsa_pipeline, _predict_joint, "none",
               "NSA-Flow on the concatenated standardized views (joint basis)."),
]

#: name -> spec. The single source of truth for what a method is called,
#: what it accepts and how it predicts.
METHODS: Mapping[str, MethodSpec] = types.MappingProxyType({s.name: s for s in _SPECS})
_ALL_PARAMS = frozenset().union(*(s.accepts for s in _SPECS))

REMOVED_METHODS: Mapping[str, str] = types.MappingProxyType({
    "simlr_lbfgsb": ("in 0.6.0; it was not competitive with 'hybrid_adam' on the default-selection "
                     "benchmark (docs/tracking_amigo.md, D5). Use 'simlr' (default optimizer 'hybrid_adam')."),
})


def method_names(family: Optional[Union[str, Iterable[str]]] = None) -> List[str]:
    """Registered method names, optionally restricted to one or more families."""
    if family is None:
        return list(METHODS)
    fams = {family} if isinstance(family, str) else set(family)
    bad = fams - set(FAMILIES)
    if bad:
        raise ValueError(f"unknown family {sorted(bad)}; choose from {FAMILIES}.")
    return [n for n, s in METHODS.items() if s.family in fams]


def resolve_method(name: str) -> MethodSpec:
    """The spec for a registered method name (there are no aliases)."""
    if name in REMOVED_METHODS:
        raise ValueError(f"method {name!r} was removed: {REMOVED_METHODS[name]}")
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; registered: {list(METHODS)}.")
    return METHODS[name]


def describe_method(name: str) -> Dict[str, Any]:
    """Family, budget unit, presets and every accepted parameter with its default."""
    s = resolve_method(name)
    return {"name": s.name, "family": s.family, "budget_unit": s.budget_unit,
            "description": s.description,
            "presets": dict(s.presets),
            "parameters": {kk: vv for kk, vv in sorted(s.defaults.items()) if kk not in s.presets}}


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------
def _ignored_reason(spec: MethodSpec, name: str) -> str:
    if name in spec.presets:
        return f"fixed to {spec.presets[name]!r} by method {spec.name!r}"
    users = [s.name for s in _SPECS if name in s.accepts]
    return f"{spec.name!r} has no parameter {name!r} (accepted by {users})"


def _same(a: Any, b: Any) -> bool:
    """Equality for preset values, which are plain scalars or strings."""
    scalar = (str, bool, int, float, type(None))
    return isinstance(a, scalar) and isinstance(b, scalar) and a == b


def fit(method: str, views: Any, k: int, *, strict: bool = False, **params: Any) -> Dict[str, Any]:
    """Fit a registered method with the shared vocabulary.

    Parameters
    ----------
    method : str
        A name from `method_names()`.
    views : matrix or list of matrices
        One ``(n, p_m)`` matrix per view (see `as_views`).
    k : int
        Dimensionality of the shared latent space.
    strict : bool, default=False
        Raise `TypeError` instead of recording and warning when a parameter
        does not apply to this method.
    **params
        Any parameter of any registered method (see `describe_method`).

    Returns
    -------
    dict
        The native result plus the contract keys (module docstring) and
        ``parameters``: ``requested`` (as passed), ``used`` (what reached the
        method, presets included), ``ignored`` (name -> reason), ``defaults``
        (accepted parameters left at their default), ``random_state`` (the
        seed actually used, ``None`` for deterministic fits), ``budget_unit``
        and ``preset``.
    """
    spec = resolve_method(method)
    unknown = sorted(set(params) - _ALL_PARAMS)
    if unknown:
        raise TypeError(f"no pysimlr method has parameter(s) {unknown}; "
                        f"see pysimlr.describe_method(name) for each method's parameters.")
    used: Dict[str, Any] = {}
    ignored: Dict[str, str] = {}
    for name, value in params.items():
        if name in spec.presets:
            if _same(value, spec.presets[name]):
                continue
            ignored[name] = _ignored_reason(spec, name)
        elif name in spec.accepts:
            used[name] = value
        else:
            ignored[name] = _ignored_reason(spec, name)
    if ignored and strict:
        raise TypeError(f"{spec.name}: " + "; ".join(ignored.values()))
    for name, why in ignored.items():
        # Python's default warning filter shows each distinct message once per
        # call site; `warnings.simplefilter("always")` shows every one.
        warnings.warn(f"pysimlr.fit: parameter {name!r} ignored: {why}.",
                      RuntimeWarning, stacklevel=2)
    call = {**used, **spec.presets}
    view_list = as_views(views)
    res = spec.fit(view_list, int(k), **call)
    _normalize(res)
    res["method"] = spec.name
    res["family"] = spec.family
    res["parameters"] = {
        "requested": dict(params),
        "used": call,
        "ignored": ignored,
        "defaults": {n: d for n, d in spec.defaults.items() if n not in call},
        "random_state": res.get("random_state"),
        "budget_unit": spec.budget_unit,
        "preset": dict(spec.presets),
    }
    if spec.family in ("deep", "flow"):
        res["convergence"]["grad_map"] = _basis_grad_map(res, view_list)
    from .loadings import loadings as _loadings
    lo = _loadings(res, view_list)
    res["structure"], res["loadings"] = lo["structure"], lo["loadings"]
    res["loading_margin"], res["loading_rotation"] = lo["loading_margin"], lo["loading_rotation"]
    res["coupling_signs"] = lo["coupling_signs"]
    validate_result(res, n=view_list[0].shape[0], k=int(k), n_views=len(view_list))
    return res


def _basis_grad_map(res: Dict[str, Any], views: List[torch.Tensor]) -> Optional[float]:
    """Face residual (`utils.stationarity_residual`) of a deep/flow fit's
    first-layer basis ``V``, the object every method has.

    The same functional `simlr` reports per sweep, evaluated once on the
    returned ``V`` and consensus ``u``, on the views as the model saw them
    (its own scaling replayed), for the energy and feasible set it used. It
    certifies the basis, not the decoder weights. Largest over views; ``None``
    only when the result has no per-view basis of matching shape.
    """
    from .utils import stationarity_residual
    from .similarity import SimilarityContext, resolve_energy_name, similarity_gradient
    vs, u = res.get("v"), res.get("u")
    if not vs or u is None:
        return None
    used, defaults = res["parameters"]["used"], res["parameters"]["defaults"]
    energy_type = used.get("energy_type", defaults.get("energy_type"))
    positivity = used.get("positivity", defaults.get("positivity", "either"))
    nonneg = positivity in ("positive", "hard", "nonnegative", "nonneg", "softplus")
    name = resolve_energy_name(energy_type, path="deep")
    xs, _ = _preprocess_views(views, res["scale_list"], res["provenance_list"])
    u_t = torch.as_tensor(u).detach().float()
    terms = []
    for x, v in zip(xs, vs):
        v = torch.as_tensor(v).detach().float()
        x = torch.as_tensor(x).detach().float()
        if x.shape[0] != u_t.shape[0] or v.shape[1] != u_t.shape[1] or x.shape[1] != v.shape[0]:
            continue
        g = x.t() @ similarity_gradient(name, x @ v, u_t, SimilarityContext(x=x, v=v), wrt="s")
        terms.append(stationarity_residual(v, g, nonneg=nonneg)[0])
    return max(terms) if terms else None


def _normalize(out: Dict[str, Any]) -> None:
    u = out.get("u")
    if isinstance(u, (list, tuple)):
        if out.get("u_per_modality") is None:
            out["u_per_modality"] = list(u)
        out["u"] = u[0]
    out.setdefault("random_state", None)


def predict(result: Dict[str, Any], views: Any) -> Dict[str, Any]:
    """Out-of-sample projection with a `fit` result.

    Returns ``u`` (an ``(n, k)`` tensor), ``latents``, ``reconstructions``,
    ``errors`` (per-view ``||X - X_hat||_F / ||X||_F``), and per view ``r2``
    (``1 - ||X - X_hat||^2 / ||X||^2``) and ``mse``, plus whatever the method's
    own predictor adds. All of these are in the method's preprocessed space:
    each view standardized with its *training* mean and standard deviation
    (``centerAndScale``; `simlr` also divides by a constant, which R^2 ignores).
    ``r2`` is therefore held-out variance explained relative to the training
    mean, on the same per-feature scale for every method.

    ``views`` may also be a dict keyed by view name (``result["view_names"]``,
    or ``view0 ...``, plus any views added by `pysimlr.extend`), or a list
    with ``None`` for a missing view. Then ``u`` is built from the available
    views only (`pysimlr.extension.predict_views`) and ``latents``, ``r2``,
    ``mse`` and ``reconstructions`` are dicts keyed by view name.
    """
    if "method" not in result:
        raise ValueError("predict needs a result from pysimlr.fit (it carries 'method').")
    # A subset of views (a dict, or None for a missing view) or extended views
    # (`pysimlr.extend`): the consensus is built from the views available.
    if isinstance(views, dict) or (isinstance(views, (list, tuple)) and any(v is None for v in views)):
        from .extension import predict_views
        return predict_views(result, views)
    spec = resolve_method(result["method"])
    view_list = as_views(views)
    out = spec.predict(view_list, result)
    _normalize(out)
    out.pop("random_state", None)
    missing = [kk for kk in ("u", "latents", "reconstructions", "errors") if kk not in out]
    if missing:
        raise RuntimeError(f"{spec.name} predictor did not return {missing}.")
    xs, _ = _preprocess_views(view_list, result["scale_list"], result["provenance_list"])
    out["r2"], out["mse"] = [], []
    for x, r in zip(xs, out["reconstructions"]):
        x = x.double(); r = torch.as_tensor(r).double()
        ss_res = float(((x - r) ** 2).sum())
        out["r2"].append(1.0 - ss_res / (float((x ** 2).sum()) + 1e-12))
        out["mse"].append(ss_res / max(1, x.numel()))
    return out


def validate_result(result: Dict[str, Any], n: Optional[int] = None, k: Optional[int] = None,
                    n_views: Optional[int] = None) -> None:
    """Raise `ValueError` if a result does not satisfy the shared contract."""
    missing = [kk for kk in CONTRACT_KEYS if kk not in result]
    if result.get("basis_is_joint"):
        missing += [f"joint_{kk}" for kk in RANK_KEYS if f"joint_{kk}" not in result]
    else:
        missing += [kk for kk in RANK_KEYS if kk not in result]
    conv = result.get("convergence")
    if isinstance(conv, dict):
        missing += [f"convergence.{kk}" for kk in CONVERGENCE_KEYS if kk not in conv]
    elif "convergence" in result:
        missing.append("convergence (not a dict)")
    if missing:
        raise ValueError(f"result of {result.get('method')!r} is missing {missing}.")
    if conv["stop_reason"] not in STOP_REASONS:
        raise ValueError(f"result of {result.get('method')!r} has stop_reason "
                         f"{conv['stop_reason']!r}, not one of {STOP_REASONS}.")
    u = result["u"]
    if not isinstance(u, torch.Tensor) or u.ndim != 2:
        raise ValueError(f"u must be a 2-D tensor; got {type(u).__name__}.")
    if n is not None and k is not None and tuple(u.shape) != (n, k):
        raise ValueError(f"u has shape {tuple(u.shape)}, expected {(n, k)}.")
    if n_views is not None and len(result["v"]) != n_views:
        raise ValueError(f"v has {len(result['v'])} blocks for {n_views} views.")
    for key in ("structure", "loadings"):
        blocks = result[key]
        if n_views is not None and len(blocks) != n_views:
            raise ValueError(f"{key} has {len(blocks)} blocks for {n_views} views.")
        for i, b in enumerate(blocks):
            if b.ndim != 2 or (k is not None and b.shape[1] != k) or not torch.isfinite(b).all():
                raise ValueError(f"{key}[{i}] must be a finite (p_m, {k}) tensor; "
                                 f"got shape {tuple(b.shape)}.")


def parameter_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """A flat, table-friendly view of ``result["parameters"]``.

    Scalars are kept; anything else (tensors, graphs, lists of matrices) is
    replaced by its type name so the row can be written to CSV.
    """
    rec = result["parameters"]

    def flat(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (list, tuple)) and all(x is None or isinstance(x, (bool, int, float, str)) for x in v):
            return list(v)
        return f"<{type(v).__name__}>"

    return {"method": result["method"], "family": result["family"],
            "budget_unit": rec["budget_unit"], "random_state": rec["random_state"],
            "used": {kk: flat(vv) for kk, vv in rec["used"].items()},
            "ignored": sorted(rec["ignored"])}


__all__ = ["FAMILIES", "CONTRACT_KEYS", "METHODS", "MethodSpec", "as_views", "describe_method",
           "fit", "method_names", "nsa_pipeline", "parameter_summary", "pca_joint", "predict",
           "resolve_method", "validate_result"]
