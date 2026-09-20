r"""The similarity terms, defined once.

SiMLR asks how well a view's projection agrees with the shared latent. That
question had two independent answers in this code base: `simlr.calculate_simlr_energy`
scored ``(X, V, u)`` for the linear path, and `deep.calculate_sim_loss` scored
``(z, u)`` for the deep and flow paths. They shared parameter *names* and not
definitions:

===============  ==========================================  ==============================================
``energy_type``  linear                                      deep / flow
===============  ==========================================  ==============================================
``regression``   :math:`\lVert X - uV^\top\rVert_F^2`         :math:`\mathrm{mean}((z/\sigma_z - u/\sigma_u)^2)`
``nc``           :math:`-\langle u, XV\rangle/(\lVert u\rVert\lVert XV\rVert)`  :math:`-\mathrm{tr}(u^\top z)/\lVert u^\top z\rVert_F`
``acc``          :math:`-\sum|u^\top XV|/(n-1)`               :math:`-\sum|u^\top z|/(n-1)`
===============  ==========================================  ==============================================

So a benchmark that held ``energy_type`` fixed and varied the model was varying
the objective at the same time, and every cross-model comparison of "which
energy wins" was comparing different functions. This is the third instance of
one-name-several-definitions in the package, after the five orthogonality
functionals in `utils` and the sixth frame defect inlined in the benchmark
runner.

The two ``regression`` terms are both defensible and genuinely different --
one reconstructs the data, the other aligns latents -- so they get separate
names here (``recon`` and ``align``) rather than one silently shadowing the
other.

Design
------
Each term is a plain differentiable function of ``(s, u, ctx)``:

``s``
    The view's representation entering the comparison: ``X V`` for a linear
    model, ``g(X V)`` when a warp is present. This is the *only* place a warp
    enters the objective, which is what makes "does the warp help" a
    controlled comparison rather than a change of objective.
``u``
    The shared latent from the mixing method.
``ctx``
    Optional `SimilarityContext` carrying ``X`` and ``V`` for the terms that
    need the data itself (``recon``) rather than just the projection.

Gradients come from autograd. The hand-written gradients this replaces were
wrong in two ways that a finite-difference check caught: ``regression`` dropped
the :math:`V u^\top u` curvature term and was *orthogonal* to true descent in
the regime the library runs in (cosine -0.0045), and six of twelve declared
objectives returned an identically zero gradient. Differentiating the written
energy makes both failures unrepresentable.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

__all__ = [
    "SimilarityContext", "SIMILARITY", "similarity_names",
    "similarity_energy", "similarity_gradient", "describe_similarity",
    "LEGACY_ENERGY_ALIASES", "resolve_energy_name",
]


@dataclass
class SimilarityContext:
    """What a similarity term may need beyond ``(s, u)``.

    Attributes
    ----------
    x : torch.Tensor, optional
        The view, ``(n, p)``. Needed by ``recon``.
    v : torch.Tensor, optional
        The loading matrix, ``(p, k)``. Needed by ``recon``.
    eps : float
        Floor for denominators, so a degenerate iterate gives a large finite
        value rather than a NaN that propagates silently.
    """

    x: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    eps: float = 1e-10


def _require(ctx: Optional[SimilarityContext], *fields: str, name: str) -> SimilarityContext:
    if ctx is None:
        raise ValueError(
            f"similarity {name!r} needs a SimilarityContext carrying "
            f"{', '.join(fields)}; it is a function of the data, not only of "
            f"the projection."
        )
    for f in fields:
        if getattr(ctx, f, None) is None:
            raise ValueError(f"similarity {name!r} needs ctx.{f}, which is None.")
    return ctx


def _std(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Column standard deviation, biased, with the floor inside the root.

    ``sqrt(var + eps)`` rather than ``clamp(std, eps)``: the clamped form is
    non-differentiable where it binds and jumps as a column approaches
    constant, while this is smooth everywhere. Biased (``unbiased=False``)
    because this standardises a representation rather than estimating a
    population parameter, and because it is what the deep path already used --
    matching it keeps the identity-equivalence test exact.
    """
    return torch.sqrt(z.var(dim=0, keepdim=True, unbiased=False) + eps)


#: Rows needed before a centred, standardised cross-moment means anything.
#: At n = 1 centring sends every such term to exactly 0 with a zero gradient,
#: and at n = 2 standardising pins every value to +-1 so the term is constant
#: (measured gradient magnitude ~1e-6). Both are silent no-ops of exactly the
#: kind this registry exists to make impossible, so they are refused.
MIN_ROWS_FOR_MOMENT = 3


def _check_rows(s: torch.Tensor, name: str) -> None:
    n = int(s.shape[0])
    if n < MIN_ROWS_FOR_MOMENT:
        raise ValueError(
            f"similarity {name!r} needs at least {MIN_ROWS_FOR_MOMENT} rows to "
            f"form a centred cross-moment, got n={n}. Below that the term is "
            f"identically constant and its gradient is exactly zero, so the "
            f"optimiser would run and report success having moved nothing. "
            f"Use 'recon', which is defined for any n."
        )


def _centre(z: torch.Tensor) -> torch.Tensor:
    """Remove the column means.

    Every cross-moment term below is a *covariance*, so both arguments are
    centred here rather than relying on the caller. The linear path omitted
    this and coincided with the deep path only because `simlr` happens to
    centre upstream in `preprocess_data`; a caller who passed uncentred views
    silently got a second-moment matrix instead.
    """
    return z - z.mean(dim=0, keepdim=True)


# --------------------------------------------------------------------------
# the terms
# --------------------------------------------------------------------------
def energy_recon(s: torch.Tensor, u: torch.Tensor,
                 ctx: Optional[SimilarityContext] = None) -> torch.Tensor:
    r"""Data reconstruction, :math:`\lVert X - u V^\top\rVert_F^2`.

    The linear path's historical ``regression``. Its unconstrained minimiser is
    :math:`V^\star = X^\top u (u^\top u)^{-1}`, so :math:`u^\top u` is the
    curvature of the block problem rather than an ignorable scale -- and it is
    never the identity here, because every mixing method returns
    variance-standardised scores, giving :math:`u^\top u = (n-1) I`.

    Note this is the one term whose value depends on ``V`` and not only on the
    representation ``s``, so a warp cannot change it except through ``u``.
    """
    c = _require(ctx, "x", "v", name="recon")
    return torch.sum((c.x - u @ c.v.t()) ** 2)


def energy_align(s: torch.Tensor, u: torch.Tensor,
                 ctx: Optional[SimilarityContext] = None) -> torch.Tensor:
    r"""Standardised latent agreement, :math:`\mathrm{mean}((s/\sigma_s - u/\sigma_u)^2)`.

    The deep path's historical ``regression``. Standardising both sides makes
    it scale free in each argument, so unlike ``recon`` it says nothing about
    the magnitude of the representation -- only about its direction pattern.
    """
    _check_rows(s, "align")
    sc, uc = _centre(s), _centre(u)
    return torch.mean((sc / _std(sc) - uc / _std(uc)) ** 2)


def energy_acc(s: torch.Tensor, u: torch.Tensor,
               ctx: Optional[SimilarityContext] = None) -> torch.Tensor:
    r"""Negative absolute cross-covariance, :math:`-\sum_{jl}|u^\top s|_{jl}/(n-1)`.

    Degree-1 homogeneous in ``s`` and unbounded below, so the column scale must
    be fixed elsewhere (see `sparsification.GAUGE_FREE_ENERGIES`) or the
    optimiser improves it forever by inflating ``V``.
    """
    _check_rows(s, "acc")
    n = max(int(s.shape[0]) - 1, 1)
    return -torch.sum(torch.abs(_centre(u).t() @ _centre(s))) / n


def energy_nc(s: torch.Tensor, u: torch.Tensor,
              ctx: Optional[SimilarityContext] = None) -> torch.Tensor:
    r"""Negative normalised correlation, :math:`-\langle u, s\rangle/(\lVert u\rVert\lVert s\rVert)`.

    Degree-0 homogeneous: by Euler's theorem :math:`\langle\nabla E, s\rangle = 0`,
    so the gradient is everywhere orthogonal to ``s`` and the objective is a
    function on the sphere quotient. A reported energy change of exactly zero
    under a purely radial step is therefore correct behaviour, not a stalled
    solver -- a distinction that cost real debugging time here.
    """
    _check_rows(s, "nc")
    eps = ctx.eps if ctx is not None else 1e-10
    uc, sc = _centre(u), _centre(s)
    num = torch.sum(uc * sc)
    den = torch.linalg.vector_norm(uc) * torch.linalg.vector_norm(sc) + eps
    return -num / den


def energy_procrustes(s: torch.Tensor, u: torch.Tensor,
                      ctx: Optional[SimilarityContext] = None) -> torch.Tensor:
    r"""Negative Procrustes correlation, :math:`-\mathrm{tr}(u^\top s)/\lVert u^\top s\rVert_F`.

    The deep path's historical ``nc``. Kept under its own name because it is a
    different function from `energy_nc`: it compares the *trace* of the cross
    covariance to its Frobenius norm, which rewards a diagonally dominant
    cross covariance -- component-wise correspondence -- where ``nc`` only asks
    that the flattened representations point the same way.
    """
    _check_rows(s, "procrustes")
    eps = ctx.eps if ctx is not None else 1e-10
    cross = _centre(u).t() @ _centre(s)
    return -torch.trace(cross) / (torch.linalg.matrix_norm(cross) + eps)


def _ica_contrast(s: torch.Tensor, u: torch.Tensor, kind: str,
                  eps: float) -> torch.Tensor:
    """Negentropy-style contrast on the standardised cross correlation.

    The linear path evaluated the same contrast on the *raw* ``u' X V`` cross
    covariance with no standardisation, so its value scaled with the magnitude
    of ``X`` and ``V`` -- on one test matrix ``kurtosis`` read -1.3e7 rather
    than a bounded contrast, and the number was not comparable between two
    models whose bases differ in scale. Standardising makes the contrast a
    function of the cross *correlation*, which is what the negentropy
    interpretation requires.

    A consequence worth stating: standardising drops these terms from degree-1
    to degree-0 homogeneous in ``s``, so they no longer need the column gauge
    that `sparsification.GAUGE_FREE_ENERGIES` was fixing for them.
    """
    _check_rows(s, kind)
    n = max(int(s.shape[0]), 1)
    sc, uc = _centre(s), _centre(u)
    z = sc / _std(sc)
    w = uc / _std(uc)
    c = (w.t() @ z) / n
    a = torch.abs(c)
    if kind == "logcosh":
        # log cosh|c|, written so the exponential cannot overflow.
        g = a - float(np.log(2.0)) + torch.log1p(torch.exp(-2.0 * a))
    elif kind == "exp":
        g = -torch.exp(-0.5 * c ** 2)
    elif kind == "gauss":
        g = -torch.exp(-0.5 * c ** 2)
    elif kind == "kurtosis":
        g = c ** 4
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(kind)
    return -torch.sum(g)


def energy_logcosh(s, u, ctx=None):
    """Robust ICA contrast; see `_ica_contrast`."""
    return _ica_contrast(s, u, "logcosh", ctx.eps if ctx else 1e-10)


def energy_exp(s, u, ctx=None):
    """Exponential ICA contrast; see `_ica_contrast`."""
    return _ica_contrast(s, u, "exp", ctx.eps if ctx else 1e-10)


def energy_gauss(s, u, ctx=None):
    """Gaussian ICA contrast; see `_ica_contrast`."""
    return _ica_contrast(s, u, "gauss", ctx.eps if ctx else 1e-10)


def energy_kurtosis(s, u, ctx=None):
    """Kurtosis ICA contrast; see `_ica_contrast`."""
    return _ica_contrast(s, u, "kurtosis", ctx.eps if ctx else 1e-10)


#: name -> (function, needs-data, one-line description). The single source of
#: truth: adding a similarity is an entry here, not an edit in two modules that
#: can drift apart.
SIMILARITY: Dict[str, tuple] = {
    "recon": (energy_recon, True,
              "||X - u V'||_F^2 : reconstruct the data from the shared latent"),
    "align": (energy_align, False,
              "mean((s/sd(s) - u/sd(u))^2) : standardised latent agreement"),
    "acc": (energy_acc, False,
            "-sum|u's|/(n-1) : absolute cross covariance"),
    "nc": (energy_nc, False,
           "-<u,s>/(||u|| ||s||) : normalised correlation, scale invariant"),
    "procrustes": (energy_procrustes, False,
                   "-tr(u's)/||u's||_F : rewards component-wise correspondence"),
    "logcosh": (energy_logcosh, False, "ICA negentropy contrast (log cosh)"),
    "exp": (energy_exp, False, "ICA negentropy contrast (exponential)"),
    "gauss": (energy_gauss, False, "ICA negentropy contrast (Gaussian)"),
    "kurtosis": (energy_kurtosis, False, "ICA negentropy contrast (kurtosis)"),
}

#: Old names, and which term they actually meant in each path. ``regression``
#: is deliberately *not* mapped: it meant two different functions, so resolving
#: it silently would pick a winner and hide the change. Callers must say which.
LEGACY_ENERGY_ALIASES = {
    "normalized_correlation": "nc",
}


#: Names whose *deep-path* meaning differed from the unified one. Resolving
#: these warns instead of changing results quietly.
_DEEP_RENAMED = {
    "nc": ("procrustes",
           "the deep path's 'nc' was -tr(u's)/||u's||_F, the Procrustes "
           "correlation, not the normalised correlation -<u,s>/(||u|| ||s||) "
           "that the linear path and the registry mean by that name"),
}


def resolve_energy_name(name: str, path: Optional[str] = None) -> str:
    """Map a legacy ``energy_type`` onto a registry key.

    Parameters
    ----------
    name : str
        The requested energy.
    path : {"linear", "deep"}, optional
        Which historical definition to assume for a genuinely ambiguous name.
        ``"regression"`` meant ``recon`` in the linear path and ``align`` in
        the deep path; without ``path`` it raises rather than guess.
    """
    if name == "recon" and path == "deep":
        raise ValueError(
            "similarity 'recon' is ||X - uV'||^2, a function of the data and "
            "the loading matrix, and the deep path does not carry either into "
            "the similarity term -- its reconstruction is a separate decoder "
            "loss. Use 'align' for latent agreement on a deep model."
        )
    if name in SIMILARITY:
        if path == "deep" and name in _DEEP_RENAMED:
            old, why = _DEEP_RENAMED[name]
            warnings.warn(
                f"energy_type={name!r} on a deep model now resolves to the "
                f"registry's {name!r}: {why}. Results will differ from previous "
                f"releases. Pass {old!r} for the previous behaviour.",
                DeprecationWarning, stacklevel=3,
            )
        return name
    if name in LEGACY_ENERGY_ALIASES:
        return LEGACY_ENERGY_ALIASES[name]
    if name == "regression":
        if path == "linear":
            return "recon"
        if path == "deep":
            return "align"
        raise ValueError(
            "energy_type='regression' is ambiguous: it meant ||X - uV'||^2 in "
            "the linear path and standardised latent MSE in the deep path, and "
            "those are different objectives. Ask for 'recon' or 'align'."
        )
    raise ValueError(
        f"unknown similarity {name!r}; available: {sorted(SIMILARITY)}"
    )


def similarity_names() -> list:
    """Registry keys, sorted."""
    return sorted(SIMILARITY)


def describe_similarity(name: str) -> str:
    """One-line description of a registry entry."""
    return SIMILARITY[resolve_energy_name(name)][2]


def similarity_energy(name: str, s: torch.Tensor, u: torch.Tensor,
                      ctx: Optional[SimilarityContext] = None,
                      path: Optional[str] = None) -> torch.Tensor:
    """Evaluate a similarity term by name."""
    fn, _needs, _doc = SIMILARITY[resolve_energy_name(name, path)]
    return fn(s, u, ctx)


def similarity_gradient(name: str, s: torch.Tensor, u: torch.Tensor,
                        ctx: Optional[SimilarityContext] = None,
                        wrt: str = "s", path: Optional[str] = None) -> torch.Tensor:
    r"""Gradient of a similarity term, by autograd.

    Parameters
    ----------
    wrt : {"s", "v"}
        Differentiate with respect to the representation, or with respect to
        ``ctx.v`` (which ``recon`` needs, since it is the variable the linear
        block step actually updates).

    Returns
    -------
    torch.Tensor
        :math:`\partial E/\partial(\cdot)`. Callers wanting a *descent*
        direction negate it; the historical hand-written functions returned the
        negated form, which is a convention worth stating rather than
        inferring.
    """
    key = resolve_energy_name(name, path)
    fn, _needs, _doc = SIMILARITY[key]
    if wrt == "v":
        c = _require(ctx, "x", "v", name=key)
        v = c.v.detach().clone().requires_grad_(True)
        local = SimilarityContext(x=c.x, v=v, eps=c.eps)
        e = fn(s, u, local)
        (g,) = torch.autograd.grad(e, v)
        return g
    t = s.detach().clone().requires_grad_(True)
    e = fn(t, u, ctx)
    (g,) = torch.autograd.grad(e, t)
    return g
