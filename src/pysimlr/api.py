"""
One interface to every multi-view method in pysimlr.

The methods (linear `simlr`, the deep `lend`/`ned`/`ned_shared_private`
models, the flow model `flow_v`, and two joint baselines) share one call
shape, one parameter vocabulary and one result contract, so a benchmark is a
loop with no per-method branch::

    import pysimlr
    for name in pysimlr.method_names():
        etype = pysimlr.matched_energy(name)
        kw = {"energy_type": etype} if etype is not None else {}
        res = pysimlr.fit(name, views, k=3, iterations=50, random_state=0,
                          nsa_w=0.5, **kw)
        pred = pysimlr.predict(res, test_views)
        print(name, res["parameters"]["ignored"], pred["u"].shape)

Matched energy pairing
----------------------
Holding ``energy_type`` constant across linear and deep/flow families does not
compare matched objectives:

* **Linear** (``"simlr"``): ``energy_type="recon_r2"`` reconstructs each view
  directly from the shared consensus (`similarity.energy_recon_r2`). Passing
  ``"align"`` to ``"simlr"`` aligns view projections without a reconstruction
  term and emits a one-time `UserWarning`.
* **Deep / Flow** (``"lend"``, ``"ned"``, ``"ned_shared_private"``,
  ``"flow_v"``): reconstruction is handled by the decoder / normalizing-flow
  loss, so the matched similarity term is ``energy_type="align"`` (and
  data-dependent linear terms such as ``"recon_r2"`` are refused).
* **Baselines** (``"pca"``, ``"nsa_pipeline"``): take no ``energy_type``
  (`matched_energy` returns ``None``).

Use `pysimlr.matched_energy(method)` (also reported in
``describe_method(method)["matched_energy"]``) when comparing across families.

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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from .deep import _predict_deep_impl, lend_simr, ned_simr, ned_simr_shared_private, predict_deep
from .flows import flow_simr_v
from .optimizers import SIMLR_OPTIMIZER_DEFAULTS
from .simlr import SIMLR_OPTIMIZER_KEYS, _predict_simlr_impl, predict_simlr, simlr
from .utils import basis_rank_report
from .defaults import DEFAULT_NSA_W, check_nsa_w
from .confounds import SINK_PARAMS

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

    Accepts one matrix (a single view), a list/tuple of matrices, or a mapping
    of ``view_name -> matrix``; each may be a `torch.Tensor`, `np.ndarray` or
    anything with ``to_numpy()``.
    """
    from .extension import _to_tensor
    if isinstance(views, Mapping):
        items = list(views.values())
    elif isinstance(views, (list, tuple)):
        items = list(views)
    else:
        items = [views]
    if not items:
        raise ValueError("views must contain at least one matrix.")
    out = [_to_tensor(m, f"view {i}") for i, m in enumerate(items)]
    rows = {t.shape[0] for t in out}
    if len(rows) != 1:
        raise ValueError(f"Size mismatch: views have different numbers of rows: "
                         f"{[t.shape[0] for t in out]}.")
    return out


def _resolve_fit_view_idx(key: Any, n_views: int, view_names: Optional[List[str]]) -> int:
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool) and 0 <= int(key) < n_views:
        return int(key)
    if view_names is not None and isinstance(key, str) and key in view_names:
        return int(view_names.index(key))
    if isinstance(key, str) and key.startswith("view") and key[4:].isdigit() and int(key[4:]) < n_views:
        return int(key[4:])
    names_hint = f", or one of {view_names}" if view_names is not None else ""
    raise ValueError(
        f"unknown view key {key!r}; expected 0..{n_views - 1} or 'view0'..'view{n_views - 1}'{names_hint}."
    )


def _normalize_fit_path_graph(path_graph: Any, n_views: int,
                              view_names: Optional[List[str]]) -> Dict[int, List[int]]:
    if isinstance(path_graph, Mapping):
        out: Dict[int, List[int]] = {}
        for k_raw, nbrs in path_graph.items():
            ia = _resolve_fit_view_idx(k_raw, n_views, view_names)
            out[ia] = [_resolve_fit_view_idx(nb, n_views, view_names) for nb in nbrs]
        return out
    adj: Dict[int, List[int]] = {i: [] for i in range(n_views)}
    for a_raw, b_raw in path_graph:
        ia = _resolve_fit_view_idx(a_raw, n_views, view_names)
        ib = _resolve_fit_view_idx(b_raw, n_views, view_names)
        if ib not in adj[ia]:
            adj[ia].append(ib)
        if ia not in adj[ib]:
            adj[ib].append(ia)
    return adj


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
    return {"u": u_t, "latents": [u_t], "reconstructions": recs, "errors": errors, "_preprocessed_views": xs_list}


def _predict_linear(views, result):
    return _predict_simlr_impl(views, result, return_preprocessed=True)


def _predict_deep(views, result):
    return _predict_deep_impl(views, result, return_preprocessed=True)


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
               "Linear SiMLR: one sparse, NSA-Flow-constrained basis per view "
               "(matched energy: 'recon_r2'; counterpart to deep/flow 'align').",
               extra_params=_frozen(_simlr_extra())),
    MethodSpec("lend", "deep", lend_simr, _predict_deep, "epochs",
               "LEND: linear first layer V, nonlinear decoders "
               "(matched energy: 'align'; counterpart to linear 'recon_r2')."),
    MethodSpec("ned", "deep", ned_simr, _predict_deep, "epochs",
               "NED: linear first layer, nonlinear heads and decoders "
               "(matched energy: 'align'; counterpart to linear 'recon_r2')."),
    MethodSpec("ned_shared_private", "deep", ned_simr_shared_private, _predict_deep, "epochs",
               "NED++: NED with per-view private latents "
               "(matched energy: 'align'; counterpart to linear 'recon_r2')."),
    MethodSpec("flow_v", "flow", flow_simr_v, _predict_deep, "epochs",
               "Flow-SiMR-V: linear first layer V, RealNVP flow per view "
               "(matched energy: 'align'; counterpart to linear 'recon_r2')."),
    MethodSpec("pca", "baseline", pca_joint, _predict_joint, "none",
               "PCA of the concatenated standardized views (joint basis; matched_energy: None)."),
    MethodSpec("nsa_pipeline", "baseline", nsa_pipeline, _predict_joint, "none",
               "NSA-Flow on the concatenated standardized views (joint basis; matched_energy: None)."),
]

#: name -> spec. The single source of truth for what a method is called,
#: what it accepts and how it predicts.
METHODS: Mapping[str, MethodSpec] = types.MappingProxyType({s.name: s for s in _SPECS})
#: Every method also takes the confound-sink parameters (`pysimlr.confounds`),
#: which `fit` applies at the shared preprocessing layer.
_ALL_PARAMS = frozenset().union(*(s.accepts for s in _SPECS)) | frozenset(SINK_PARAMS)

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


def matched_energy(name: str) -> Optional[str]:
    """Return the matched cross-family ``energy_type`` for registered method ``name``.

    Returns ``"recon_r2"`` for ``"simlr"`` (linear path, where the similarity
    term itself reconstructs each view from the consensus), ``"align"`` for
    deep and flow models (``"lend"``, ``"ned"``, ``"ned_shared_private"``,
    ``"flow_v"``, where reconstruction is the decoder/flow loss and the
    similarity term aligns latents), and ``None`` for joint baselines
    (``"pca"``, ``"nsa_pipeline"``) that take no ``energy_type``.
    """
    spec = resolve_method(name)
    if spec.family == "linear":
        return "recon_r2"
    if spec.family in ("deep", "flow"):
        return "align"
    return None


def describe_method(name: str) -> Dict[str, Any]:
    """Family, budget unit, matched energy, presets and every accepted parameter with its default.

    Notes
    -----
    When testing an exogenous block (e.g. an exposure) against multi-view scores
    (`pysimlr.path_model` / `pysimlr.path_model_from_scores`), three designs
    answer distinct questions: (1) fit without the exposure and test on the
    exposure-free scores via `path_model_from_scores` (`null="fixed"`),
    (2) include the exposure as an exogenous root via
    ``topology="graph", path_graph=fork_graph(..., exogenous_root=True)``, or
    (3) include all blocks symmetrically in `fit`, which shares the consensus
    across every block and requires ``null="refit"`` on training rows.
    """
    s = resolve_method(name)
    return {"name": s.name, "family": s.family, "budget_unit": s.budget_unit,
            "matched_energy": matched_energy(s.name),
            "description": s.description,
            "presets": dict(s.presets),
            "parameters": {kk: vv for kk, vv in sorted({**s.defaults, **SINK_PARAMS}.items())
                           if kk not in s.presets}}


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


def fit(method: str, views: Any, k: int, *, strict: bool = False, safety: str = "off", **params: Any) -> Dict[str, Any]:
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
    safety : {"off", "warn", "raise"}, default="off"
        Optional `pysimlr.data_safety` check before fitting (``"off"`` by
        default because `prepare_views` already checks with ``safety="warn"``
        and internal CV / LOO loops call `fit` repeatedly). When ``"warn"`` or
        ``"raise"``, attaches ``result["data_safety"]`` and emits one
        `DataSafetyWarning` per view with a gross row outlier (or raises
        `ValueError` when ``"raise"``).
    **params
        Any parameter of any registered method (see `describe_method`).
        ``confounds``, ``confound_strength`` and ``confound_rank`` apply to
        every method: each view is cleaned of the nuisance its own features
        predict, by an operator fitted on these rows and replayed by
        `predict` without confound values (`pysimlr.confounds`). The record
        is ``result["confound_sink"]`` (``None`` without confounds).

        Per-view overrides (``method="simlr"``): pass ``per_view_params`` as a
        mapping ``{view: {...}}`` keyed by integer index ``0..M-1``,
        ``"view0".."view{M-1}"``, or the view's own name when ``views`` is a
        mapping. Allowed keys per view are ``positivity``, ``energy_type``,
        ``sparseness``, ``nsa_w``, ``varimax``, and ``nonneg_mask`` (a 1-D
        boolean mask of features constrained to be non-negative under
        ``positivity="positive"``/``"conical"``, leaving unmasked features
        signed; ``positivity="convex"`` refuses a partial mask). ``positivity``
        itself may also be passed as a per-view list or mapping on ``"simlr"``,
        for example ``pysimlr.fit("simlr", views, k=1, positivity="convex",
        per_view_params={1: {"positivity": "either"}})``. This is useful when
        a view holds relative features (each subject's values expressed against
        their own mean), which have no meaningful one-sided orientation and
        should be fitted signed while absolute views stay non-negative/convex.

    Returns
    -------
    dict
        The native result plus the contract keys (module docstring) and
        ``parameters``: ``requested`` (as passed), ``used`` (what reached the
        method, presets included), ``ignored`` (name -> reason), ``defaults``
        (accepted parameters left at their default), ``positivity_by_view``
        (resolved constraint per view, or ``[None, ...]`` when not applicable),
        ``positivity_resolved`` (that shared string, or ``"mixed"`` when views
        differ), ``random_state`` (the seed actually used, ``None`` for
        deterministic fits), ``budget_unit``, ``preset``, ``optimizer_variant``,
        ``initialization_variant``, ``learning_rate_rule``,
        ``learning_rate_used``, ``sparseness_schedule``, and
        ``sparseness_type``.

    See Also
    --------
    pysimlr.solution_uniqueness : Multi-start diagnostic checking whether
        independent starts reach one or several distinct energy basins.
    """
    from .safety import validate_safety_mode
    safety_mode = validate_safety_mode(safety)
    spec = resolve_method(method)
    sink_req = {kk: params.pop(kk) for kk in list(params) if kk in SINK_PARAMS}
    if sink_req.get("confounds") is None and any(
            kk in sink_req and sink_req[kk] != SINK_PARAMS[kk]
            for kk in ("confound_strength", "confound_rank", "confound_rescale")):
        raise ValueError(
            "confound_strength / confound_rank / confound_rescale were set but confounds is None."
        )
    unknown = sorted(set(params) - _ALL_PARAMS)
    if unknown:
        raise TypeError(f"no pysimlr method has parameter(s) {unknown}; "
                        f"see pysimlr.describe_method(name) for each method's parameters.")
    used: Dict[str, Any] = {}
    ignored: Dict[str, str] = {}
    from .positivity import resolve_positivity
    positivity_resolved = None
    if "positivity" in params:
        raw_pos = params["positivity"]
        if isinstance(raw_pos, (list, tuple, Mapping)):
            pos_vals = list(raw_pos.values()) if isinstance(raw_pos, Mapping) else list(raw_pos)
            resolved_vals = [resolve_positivity(v) for v in pos_vals]
            if "convex" in resolved_vals and "positivity" not in spec.accepts:
                raise ValueError(f"{spec.name} has no basis constraint, so positivity='convex' "
                                 "cannot be applied; use a linear, deep or flow method.")
            if "positivity" in spec.accepts and "per_view_params" not in spec.accepts:
                raise ValueError(
                    f"{spec.name} takes one positivity constraint across views (got {type(raw_pos).__name__}); "
                    "for a different constraint per view use method='simlr' with positivity=[...] or "
                    "per_view_params={i: {'positivity': ...}}."
                )
            if "positivity" in spec.accepts:
                positivity_resolved = (
                    resolved_vals[0] if (resolved_vals and len(set(resolved_vals)) == 1) else "mixed"
                )
        else:
            positivity_resolved = resolve_positivity(raw_pos)   # unknown values raise
            if positivity_resolved == "convex" and "positivity" not in spec.accepts:
                raise ValueError(f"{spec.name} has no basis constraint, so positivity='convex' "
                                 "cannot be applied; use a linear, deep or flow method.")
    elif "positivity" in spec.accepts:
        positivity_resolved = resolve_positivity(spec.defaults.get("positivity", "either"))
    for name, value in params.items():
        if name in ("initial_v", "sparseness_schedule") and name not in spec.accepts and value is not None:
            raise ValueError(
                f"{spec.name} does not support {name!r}; {name!r} is only supported by "
                f"methods that accept it (such as 'simlr')."
            )
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
    fit_view_names: Optional[List[str]] = None
    if isinstance(views, Mapping):
        fit_view_names = [str(nm) for nm in views.keys()]
        if len(set(fit_view_names)) != len(fit_view_names) or {"outcome", "consensus"} & set(fit_view_names):
            raise ValueError(
                f"view names must be distinct and not 'outcome' or 'consensus'; got {fit_view_names}."
            )
    view_list = as_views(views)
    n_views = len(view_list)
    if call.get("path_graph") is not None:
        call["path_graph"] = _normalize_fit_path_graph(call["path_graph"], n_views, fit_view_names)
    if call.get("per_view_params") is not None and isinstance(call["per_view_params"], Mapping):
        norm_pvp: Dict[int, Dict[str, Any]] = {}
        for kv, vv in call["per_view_params"].items():
            idx_v = _resolve_fit_view_idx(kv, n_views, fit_view_names)
            if idx_v in norm_pvp:
                raise ValueError(f"per_view_params gives view {idx_v} twice.")
            if not isinstance(vv, Mapping):
                raise TypeError(f"per_view_params[{kv!r}] must be a dict; got {type(vv).__name__}.")
            norm_pvp[idx_v] = dict(vv)
        call["per_view_params"] = norm_pvp
    if isinstance(call.get("positivity"), Mapping):
        norm_pos: Dict[int, Any] = {}
        for kv, vv in call["positivity"].items():
            idx_v = _resolve_fit_view_idx(kv, n_views, fit_view_names)
            if idx_v in norm_pos:
                raise ValueError(f"positivity gives view {idx_v} twice.")
            norm_pos[idx_v] = vv
        call["positivity"] = norm_pos
    _resolved_auto_sp: Optional[List[Optional[float]]] = None
    if call.get("sparseness") == "auto" or (
        isinstance(call.get("sparseness"), (list, tuple))
        and any(x == "auto" for x in call["sparseness"])
    ):
        from .presets import resolve_auto_sparseness
        _pos_hint = call.get("positivity", "either")
        _pos_str = _pos_hint if isinstance(_pos_hint, str) else "either"
        _auto_q = resolve_auto_sparseness(
            view_list, int(k), positivity=_pos_str, path_graph=call.get("path_graph")
        )
        if call["sparseness"] == "auto":
            call["sparseness"] = _auto_q
        else:
            call["sparseness"] = [
                (_auto_q[i] if i < len(_auto_q) else None) if sq == "auto" else sq
                for i, sq in enumerate(call["sparseness"])
            ]
        _resolved_auto_sp = list(call["sparseness"])
    if call.get("system_matrices") is not None and isinstance(call["system_matrices"], Mapping):
        from .system_prior import normalize_system_matrices_arg
        call["system_matrices"] = normalize_system_matrices_arg(
            call["system_matrices"], n_views, view_names=fit_view_names
        )
    if sink_req.get("confounds") is not None and isinstance(sink_req["confounds"], Mapping):
        sink_req["confounds"] = {
            _resolve_fit_view_idx(kv, n_views, fit_view_names): vv
            for kv, vv in sink_req["confounds"].items()
        }
    if safety_mode == "off":
        safety_report = None
    else:
        from .safety import apply_safety_policy, data_safety
        safety_report = data_safety(views)
        apply_safety_policy(safety_report, safety_mode, caller="pysimlr.fit", stacklevel=3)
    plan = None
    if sink_req.get("confounds") is not None:
        from .confounds import confound_fit_context, finalize_confound_plan, prepare_confound_plan
        plan = prepare_confound_plan(
            view_list, **sink_req,
            missing_rows=str(call.get("missing_rows", "impute")),
            view_names=fit_view_names,
        )
        with confound_fit_context(plan):
            res = spec.fit(view_list, int(k), **call)
    else:
        res = spec.fit(view_list, int(k), **call)
    _normalize(res)
    if _resolved_auto_sp is not None:
        res["sparseness_resolved"] = _resolved_auto_sp
    if fit_view_names is not None:
        res["view_names"] = fit_view_names
    res["method"] = spec.name
    res["family"] = spec.family
    res["confound_sink"] = None
    res["data_safety"] = safety_report
    if plan is not None:
        # From here on every replay of the training preprocessing (predict,
        # loadings, the grad-map certificate) runs `scale -> confound_projection -> rescale`.
        finalize_confound_plan(res, plan)
    sink_used = {kk: sink_req.get(kk, SINK_PARAMS[kk]) for kk in SINK_PARAMS} if plan is not None else {}
    _opt_variant = (
        call.get("optimizer_type", spec.defaults.get("optimizer_type"))
        if ("optimizer_type" in spec.accepts or "optimizer_type" in spec.defaults)
        else None
    )
    if call.get("initial_v") is not None:
        _init_variant = "initial_v"
    elif "initialization_type" in spec.accepts or "initialization_type" in spec.defaults:
        _init_variant = call.get("initialization_type", spec.defaults.get("initialization_type"))
    else:
        _init_variant = None
    if "learning_rate" in spec.accepts or "learning_rate" in spec.defaults:
        _lr_req = call.get("learning_rate", spec.defaults.get("learning_rate"))
        _lr_rule: Optional[str] = "auto" if _lr_req == "auto" else "fixed"
    else:
        _lr_req = None
        _lr_rule = None
    _lr_used_val = res.get("learning_rate_used")
    if _lr_used_val is None and isinstance(_lr_req, (int, float)) and not isinstance(_lr_req, bool):
        _lr_used_val = float(_lr_req)
    _sp_sched = res.get("sparseness_schedule")
    _sp_type = (
        call.get("sparseness_type", spec.defaults.get("sparseness_type"))
        if ("sparseness_type" in spec.accepts or "sparseness_type" in spec.defaults)
        else None
    )
    res["parameters"] = {
        "requested": {**params, **sink_req},
        "used": {**call, **sink_used},
        "ignored": ignored,
        "defaults": {n: d for n, d in {**spec.defaults, **SINK_PARAMS}.items()
                     if n not in call and n not in sink_used},
        "random_state": res.get("random_state"),
        "budget_unit": spec.budget_unit,
        "preset": dict(spec.presets),
        "optimizer_variant": _opt_variant,
        "initialization_variant": _init_variant,
        "learning_rate_rule": _lr_rule,
        "learning_rate_used": float(_lr_used_val) if _lr_used_val is not None else None,
        "sparseness_schedule": _sp_sched,
        "sparseness_type": _sp_type,
    }
    # `used` keeps the caller's spelling (refits replay it); `positivity_by_view`
    # records the canonical value every view actually used, and
    # `positivity_resolved` is that shared value or `"mixed"` when views differ
    # (`None` / `[None, ...]` for a method without the parameter).
    if res.get("positivity_by_view") is not None:
        pos_by_view: List[Optional[str]] = [str(p) for p in res["positivity_by_view"]]
        positivity_resolved = pos_by_view[0] if len(set(pos_by_view)) == 1 else "mixed"
    elif "positivity" in spec.accepts:
        pos_by_view = [positivity_resolved] * n_views
    else:
        pos_by_view = [None] * n_views
        positivity_resolved = None
    res["parameters"]["positivity_by_view"] = pos_by_view
    res["parameters"]["positivity_resolved"] = positivity_resolved
    if spec.family in ("deep", "flow"):
        res["convergence"]["grad_map"] = _basis_grad_map(res, view_list)
    from .loadings import loadings as _loadings
    lo = _loadings(res, view_list)
    res["structure"], res["loadings"] = lo["structure"], lo["loadings"]
    res["loading_margin"], res["loading_rotation"] = lo["loading_margin"], lo["loading_rotation"]
    res["coupling_signs"] = lo["coupling_signs"]
    validate_result(res, n=view_list[0].shape[0], k=int(k), n_views=len(view_list))
    if plan is not None:
        # Decorrelation of the deployed estimator on the training rows: the
        # predict path, not a projection of `u` (which would be exact by
        # construction and say nothing about new subjects).
        from .confounds import max_abs_corr
        u_pred = predict(res, view_list)["u"]
        rec = res["confound_sink"]
        dec = max_abs_corr(u_pred, rec.pop("_c_std"))
        rec["train_max_abs_corr"], rec["train_mean_abs_corr"] = dec["max"], dec["mean"]
        rec["u_predict_max_abs_diff"] = float((u_pred.double() - res["u"].double()).abs().max())
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
    from .loadings import _result_view_settings
    from .positivity import is_nonneg
    from .utils import stationarity_residual
    from .similarity import (SimilarityContext, resolve_energy_name,
                             similarity_gradient, similarity_needs_data)
    vs, u = res.get("v"), res.get("u")
    if not vs or u is None:
        return None
    pos_list, et_list = _result_view_settings(res, len(vs))
    pvp = res.get("per_view_params") or {}
    u_t = torch.as_tensor(u).detach().float()
    terms = []
    if res.get("missing_rows") == "mask":
        from .utils import preprocess_data
        scale_list = res["scale_list"]
        prov_list = res["provenance_list"]
        for m_idx, (raw_v, v) in enumerate(zip(views, vs)):
            om = ~torch.isnan(raw_v).all(dim=1)
            v = torch.as_tensor(v).detach().float()
            x = preprocess_data(raw_v[om], scale_list, prov_list[m_idx]).detach().float()
            u_m = u_t[om]
            if x.shape[0] < 3 or x.shape[0] != u_m.shape[0] or v.shape[1] != u_m.shape[1] or x.shape[1] != v.shape[0]:
                continue
            name = resolve_energy_name(et_list[m_idx], path="deep")
            needs_data = similarity_needs_data(name)
            nonneg_m = is_nonneg(pos_list[m_idx])
            nm_m = (pvp.get(m_idx) or pvp.get(f"view{m_idx}") or {}).get("nonneg_mask")
            ctx = SimilarityContext(x=x, v=v)
            if needs_data:
                g = similarity_gradient(name, x @ v, u_m, ctx, wrt="v")
            else:
                g = x.t() @ similarity_gradient(name, x @ v, u_m, ctx, wrt="s")
            terms.append(stationarity_residual(v, g, nonneg=nonneg_m, nonneg_mask=nm_m)[0])
        return max(terms) if terms else None
    xs, _ = _preprocess_views(views, res["scale_list"], res["provenance_list"])
    for m_idx, (x, v) in enumerate(zip(xs, vs)):
        v = torch.as_tensor(v).detach().float()
        x = torch.as_tensor(x).detach().float()
        if x.shape[0] != u_t.shape[0] or v.shape[1] != u_t.shape[1] or x.shape[1] != v.shape[0]:
            continue
        name = resolve_energy_name(et_list[m_idx], path="deep")
        needs_data = similarity_needs_data(name)
        nonneg_m = is_nonneg(pos_list[m_idx])
        nm_m = (pvp.get(m_idx) or pvp.get(f"view{m_idx}") or {}).get("nonneg_mask")
        ctx = SimilarityContext(x=x, v=v)
        if needs_data:
            g = similarity_gradient(name, x @ v, u_t, ctx, wrt="v")
        else:
            g = x.t() @ similarity_gradient(name, x @ v, u_t, ctx, wrt="s")
        terms.append(stationarity_residual(v, g, nonneg=nonneg_m, nonneg_mask=nm_m)[0])
    return max(terms) if terms else None


def _normalize(out: Dict[str, Any]) -> None:
    u = out.get("u")
    if isinstance(u, (list, tuple)):
        if out.get("u_per_modality") is None:
            out["u_per_modality"] = list(u)
        out["u"] = u[0]
    out.setdefault("random_state", None)


def predict(
    result: Dict[str, Any],
    views: Any,
    *,
    exclude_views: Optional[Sequence[Union[int, str]]] = None,
) -> Dict[str, Any]:
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

    Pass ``exclude_views`` (a sequence of view names or integer indices, e.g.
    ``exclude_views=["Root"]`` on a `pysimlr.fork_graph` fit) to mask out
    exogenous root or target views at test time without mutating ``views``.
    """
    if "method" not in result:
        raise ValueError("predict needs a result from pysimlr.fit (it carries 'method').")
    if result.get("feature_mask") is not None:
        # select-mode `bagged_fit`: the model was fitted on the kept features; subset full-width views to them.
        from .bagging import mask_views
        views = mask_views(result, views)
    if result.get("aggregate") == "average":
        if exclude_views:
            raise NotImplementedError("exclude_views is not supported for an average-mode bagged_fit result.")
        from .bagging import bagged_predict
        return bagged_predict(result, views)
    if exclude_views is not None:
        if isinstance(exclude_views, (str, bytes)):
            raise TypeError("exclude_views must be a sequence of view names or indices, not a bare string.")
        if len(exclude_views) > 0:
            from .extension import predict_views
            return predict_views(result, views, exclude_views=exclude_views)
    # A subset of views (a dict, or None for a missing view) or extended views
    # (`pysimlr.extend`): the consensus is built from the views available.
    if isinstance(views, Mapping) or (isinstance(views, (list, tuple)) and any(v is None for v in views)):
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
    if result.get("confound_sink") is not None:
        from .confounds import sink_scores
        sc = sink_scores(result["confound_sink"], view_list)
        out["latents_sink"] = [sc["latents_sink"].get(i) for i in range(len(view_list))]
        out["u_sink"] = sc["u_sink"]
    r2_mse = out.pop("_r2_mse", None)
    xs = out.pop("_preprocessed_views", None)
    if r2_mse is not None:
        out["r2"], out["mse"] = r2_mse
    else:
        if xs is None:
            xs, _ = _preprocess_views(view_list, result["scale_list"], result["provenance_list"])
        from .utils import _recon_metrics
        _, (out["r2"], out["mse"]) = _recon_metrics(xs, out["reconstructions"], include_r2_mse=True)
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
           "fit", "matched_energy", "method_names", "nsa_pipeline", "parameter_summary", "pca_joint",
           "predict", "resolve_method", "validate_result"]
