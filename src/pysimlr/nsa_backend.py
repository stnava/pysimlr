"""Single resolution point for the optional NSA-Flow dependency.

Resolution is deferred and cached: the backend pulls in matplotlib.pyplot
through its own ``utils`` module, so importing it eagerly re-introduced about
0.7s of package import time even after the plotting entry points here were made
lazy.

Three modules used to import this package three different ways -- ``import
nsa`` (sparsification), ``import nsa_flow as nsa`` (deep) and ``from nsa_flow
import nsa_flow_orth`` (optimizers) -- so at most one could have been correct,
and which retraction path ran depended on which name happened to resolve.
Centralising the lookup makes the resolved backend inspectable.

The package is now declared as the ``nsa`` extra in ``pyproject.toml``. It is
still optional, so results differ between environments that do and do not have
it installed -- the constrained branches fall back to the SVD polar factor
without it. That difference is intended but not self-evident, so record
:func:`backend_report` alongside any results.
"""
import functools
from functools import lru_cache
from typing import Any, Callable, Optional

#: Module names tried, in order.
CANDIDATE_MODULES = ("nsa_flow", "nsa")

#: Name of the resolved module, or None if no backend is available.
RESOLVED_MODULE: Optional[str] = None


@lru_cache(maxsize=1)
def load_nsa_backend() -> Optional[Any]:
    """
    Import the NSA-Flow backend module, or return None if unavailable.

    Returns
    -------
    module or None
        The first importable candidate module, or None. Sets
        `RESOLVED_MODULE` as a side effect so callers and bug reports can see
        which backend (if any) was picked up.
    """
    global RESOLVED_MODULE
    import importlib

    for name in CANDIDATE_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        RESOLVED_MODULE = name
        return module
    RESOLVED_MODULE = None
    return None



def _rng_preserving(fn: Callable) -> Callable:
    """
    Wrap `fn` so it cannot disturb the caller's global RNG state.

    The NSA-Flow entry point takes a ``seed`` argument (default 42) and seeds
    the global generator with it, which is a side effect on process-wide state.
    Because `simlr_sparseness` calls it once per view per iteration, every
    `torch` random draw made *after* a `simlr()` call was reseeded to the same
    fixed state. Concretely, this made `torch.randperm` return an identical
    permutation every time, which silently reduced every permutation null
    distribution to a single repeated value -- `simlr_perm`,
    `paths.permutation_test` and `estimate_rank` all draw their nulls that way,
    as does any user code that generates randomness after fitting.

    `NSAFlowOptimizer` already carried a manual save/restore for this; doing it
    here covers every call site instead of one.

    Parameters
    ----------
    fn : callable
        The backend function to wrap.

    Returns
    -------
    callable
        `fn`, with the CPU (and CUDA, where present) RNG states restored
        afterwards. The backend's own internal determinism is unaffected.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import torch
        cpu_state = torch.get_rng_state()
        cuda_states = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
    return wrapper

@lru_cache(maxsize=1)
def load_nsa_flow() -> Optional[Any]:
    """
    Resolve the NSA-Flow retraction entry point.

    Returns
    -------
    callable or None
        ``nsa_flow(target, w=..., *, nonneg=..., max_iter=..., dtype=...)``,
        wrapped so it cannot disturb the caller's global RNG state (see
        :func:`_rng_preserving`), or None when no backend is installed.

    Notes
    -----
    Earlier releases exposed ``nsa_flow_orth`` with a different keyword set
    (``retraction``, ``precision``, ``apply_nonneg``). That name is gone; the
    legacy names are still probed so an older backend keeps working, but the
    modern signature is preferred.
    """
    module = load_nsa_backend()
    if module is None:
        return None
    for attr in ("nsa_flow", "nsa_flow_orth"):
        fn = getattr(module, attr, None)
        if callable(fn):
            return _rng_preserving(fn)
    return None


@lru_cache(maxsize=1)
def load_nsa_flow_data() -> Optional[Any]:
    """
    Resolve the data-driven solver, ``nsa_flow_data(X, k=..., w=...)``.

    Returns
    -------
    callable or None
        The solver, RNG-preserving, or None if unavailable.

    Notes
    -----
    This form solves for the basis from the raw data rather than refining a
    candidate. The anchored form stays *near* whatever candidate it is given,
    so a candidate derived from ``abs()`` of a signed matrix has its sign
    damage faithfully preserved; the data form does not inherit that.
    """
    module = load_nsa_backend()
    if module is None:
        return None
    fn = getattr(module, "nsa_flow_data", None)
    return _rng_preserving(fn) if callable(fn) else None


@lru_cache(maxsize=1)
def load_gradient_mapping() -> Optional[Any]:
    """
    Resolve NSA-Flow's scale-invariant stationarity certificate.

    Returns
    -------
    callable or None
        ``gradient_mapping(Y, grad, proj=...) -> float``, evaluating
        ``||P(Y - ||Y||_F^2 grad) - Y||_F / ||Y||_F``, or None when the backend
        is absent or too old to expose it.

    Notes
    -----
    This is the quantity that makes one ``tol`` mean the same thing on every
    problem: the fixed step ``t = ||Y||_F^2`` cancels the scale of ``Y``, so
    the measure is dimensionless and vanishes exactly at a stationary point of
    the *constrained* problem. It lives in ``nsa_flow.diagnostics``, which is
    not re-exported from the package root, so it is probed on the submodule.

    pysimlr needs it because `simlr` had no certificate at all: it stopped when
    the total energy stopped moving, and the total energy is renormalised every
    iteration by ``normalizing_weights``, which pins it to 1.0. The test
    therefore fired at iteration 3 on every problem and ``iterations`` never
    bound. A certificate makes the stopping claim checkable.
    """
    module = load_nsa_backend()
    if module is None:
        return None
    fn = getattr(module, "gradient_mapping", None)
    if not callable(fn):
        try:
            from importlib import import_module
            fn = getattr(import_module(f"{module.__name__}.diagnostics"),
                         "gradient_mapping", None)
        except ImportError:
            return None
    return fn if callable(fn) else None


@lru_cache(maxsize=1)
def load_lbfgsb() -> Optional[Any]:
    """
    Resolve NSA-Flow's pure-PyTorch bound-constrained L-BFGS-B.

    Returns
    -------
    callable or None
        ``lbfgsb_minimize(x0, fun_grad, fun=None, *, lower=0.0, upper=None,
        mask=None, max_grad=..., tol=..., certificate=...)`` returning a dict
        with ``x, f, n_grad, iters, stop, grad_map``; None when unavailable.

    Notes
    -----
    Byrd-Lu-Nocedal-Zhu, written against torch so it stays on device. The
    reason to prefer it over the projected-gradient and ``torch_lbfgs`` paths
    here is the generalized Cauchy point: projected gradient is confined to a
    diagonal metric, and two-metric projection releases at most one bound per
    iteration, whereas the Cauchy search can identify the whole active set in
    one step. On these objectives roughly half the coordinates sit at zero, so
    that is the difference between moving the support and freezing it.
    """
    module = load_nsa_backend()
    if module is None:
        return None
    fn = getattr(module, "lbfgsb_minimize", None)
    if not callable(fn):
        try:
            from importlib import import_module
            fn = getattr(import_module(f"{module.__name__}.lbfgsb"),
                         "lbfgsb_minimize", None)
        except ImportError:
            return None
    return fn if callable(fn) else None


@lru_cache(maxsize=1)
def load_nsa_flow_orth() -> Optional[Any]:
    """Deprecated alias of :func:`load_nsa_flow`, kept for callers."""
    return load_nsa_flow()


@lru_cache(maxsize=1)
def load_polar_factor() -> Optional[Any]:
    """
    Resolve the Sylvester-based polar factor retraction, ``polar_factor(Y, eps=...)``.

    From the NSA-Flow guide:
    In iterative optimization loops, avoid SVD/QR decompositions. For polar
    factor derivatives and retractions, use ``nsa_flow.polar_factor``.

    Returns
    -------
    callable or None
        The polar factor operator, or None if unavailable.
    """
    module = load_nsa_backend()
    if module is None:
        return None
    fn = getattr(module, "polar_factor", None)
    return fn if callable(fn) else None


@lru_cache(maxsize=1)
def load_nsa_estimator() -> Optional[Any]:
    """
    Resolve the scikit-learn compatible estimator ``NSAFlow``.

    Returns
    -------
    class or None
        The ``NSAFlow`` class (subclass of BaseEstimator, TransformerMixin),
        or None if unavailable.
    """
    module = load_nsa_backend()
    if module is None:
        return None
    cls = getattr(module, "NSAFlow", None)
    return cls if isinstance(cls, type) else None


@lru_cache(maxsize=1)
def load_stiefel_defect_normalised() -> Optional[Any]:
    """
    Resolve the scale-invariant normalized Stiefel defect diagnostic.

    Returns
    -------
    callable or None
        ``stiefel_defect_normalised(Y)``, or None if unavailable.
    """
    module = load_nsa_backend()
    if module is None:
        return None
    fn = getattr(module, "stiefel_defect_normalised", None)
    if callable(fn):
        return fn
    energy_mod = getattr(module, "energy", None)
    if energy_mod is not None:
        fn = getattr(energy_mod, "stiefel_defect_normalised", None)
        if callable(fn):
            return fn
    return None


@lru_cache(maxsize=1)
def load_consolidate_supports() -> Optional[Any]:
    """
    Resolve the signed support consolidation utility.

    Returns
    -------
    callable or None
        ``consolidate_supports(V_pos, V_neg)``, or None if unavailable.
    """
    module = load_nsa_backend()
    if module is None:
        return None
    fn = getattr(module, "consolidate_supports", None)
    if callable(fn):
        return fn
    signed_mod = getattr(module, "signed", None)
    if signed_mod is not None:
        fn = getattr(signed_mod, "consolidate_supports", None)
        if callable(fn):
            return fn
    return None


#: Backend functions found but refused because their signature cannot take
#: the keyword arguments pysimlr passes, keyed by name, with the reason.
#: Surfaced by `backend_report(extended=True)` so the choice of implementation
#: is visible rather than inferred.
INCOMPATIBLE_BACKEND_FUNCTIONS: dict = {}


def _accepts_keywords(fn: Callable, name: str, keywords: tuple) -> bool:
    """True if ``fn(E, **{k: ... for k in keywords})`` binds.

    A backend whose signature has drifted is refused here, once, with a
    warning and a record in `INCOMPATIBLE_BACKEND_FUNCTIONS` -- rather than
    being called, failing with a ``TypeError``, and having a blanket
    ``except`` quietly substitute the PyTorch implementation on every call,
    which also hid genuine errors raised inside the backend.
    """
    import inspect
    import warnings
    try:
        inspect.signature(fn).bind(None, **{k: None for k in keywords})
    except TypeError as exc:
        reason = f"signature {inspect.signature(fn)} cannot bind {list(keywords)}: {exc}"
        INCOMPATIBLE_BACKEND_FUNCTIONS[name] = reason
        warnings.warn(
            f"nsa_flow {name} is incompatible with pysimlr ({reason}); using "
            f"pysimlr's PyTorch implementation.", RuntimeWarning, stacklevel=3)
        return False
    except ValueError:
        # No introspectable signature (builtin/C extension): trust the call.
        return True
    return True


def _resolve_backend_function(name: str, keywords: tuple) -> Optional[Callable]:
    module = load_nsa_backend()
    if module is None:
        return None
    fn = getattr(module, name, None)
    if not callable(fn):
        linalg_mod = getattr(module, "linalg", None)
        fn = getattr(linalg_mod, name, None) if linalg_mod is not None else None
    if not callable(fn):
        return None
    return fn if _accepts_keywords(fn, name, keywords) else None


@lru_cache(maxsize=1)
def load_varimax_rotation() -> Optional[Any]:
    """
    Resolve the Orthogonal Varimax rotation matrix operator.

    Returns
    -------
    callable or None
        ``varimax_rotation(E, normalize=True, eps=1e-6, max_iter=100)``,
        or None if unavailable or incompatible (see `_accepts_keywords`).
    """
    return _resolve_backend_function("varimax_rotation",
                                     ("normalize", "eps", "max_iter"))


@lru_cache(maxsize=1)
def load_varimax() -> Optional[Any]:
    """
    Resolve the Orthogonal Varimax subspace rotation operator.

    Returns
    -------
    callable or None
        ``varimax(E, normalize=True, eps=1e-6, max_iter=100, canonical_sign=True)``,
        or None if unavailable or incompatible (see `_accepts_keywords`).
    """
    return _resolve_backend_function("varimax",
                                     ("normalize", "eps", "max_iter", "canonical_sign"))


def backend_report(extended: bool = False) -> dict:
    """
    Describe the resolved NSA-Flow backend, for diagnostics and provenance.

    Parameters
    ----------
    extended : bool, default=False
        If True, includes extended provenance keys (version, has_polar_factor,
        has_estimator, default_optimizer). If False, returns the canonical
        trio {"available", "module", "entry_point"} for backwards compatibility.

    Returns
    -------
    dict
        Keys "available" (bool), "module" (str or None) and "entry_point"
        (str or None), plus extended keys if requested.
    """
    module = load_nsa_backend()
    fn = load_nsa_flow_orth()
    rep = {
        "available": fn is not None,
        "module": RESOLVED_MODULE,
        "entry_point": getattr(fn, "__name__", None) if fn is not None else None,
    }
    if extended:
        version = getattr(module, "__version__", None) if module is not None else None
        has_polar = load_polar_factor() is not None
        has_est = load_nsa_estimator() is not None
        v_tuple = tuple(int(x) for x in version.split(".")[:2] if x.isdigit()) if version else ()
        if v_tuple >= (3, 0):
            opt = "lbfgsb"
        elif v_tuple >= (2, 11):
            opt = "torch_lbfgs"
        else:
            opt = None
        load_varimax()
        load_varimax_rotation()
        rep.update({
            "incompatible_functions": dict(INCOMPATIBLE_BACKEND_FUNCTIONS),
            "version": version,
            "has_polar_factor": has_polar,
            "has_estimator": has_est,
            "default_optimizer": opt,
        })
    return rep
