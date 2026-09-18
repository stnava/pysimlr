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
        opt = "torch_lbfgs" if version and tuple(int(x) for x in version.split(".")[:2] if x.isdigit()) >= (2, 11) else None
        rep.update({
            "version": version,
            "has_polar_factor": has_polar,
            "has_estimator": has_est,
            "default_optimizer": opt,
        })
    return rep
