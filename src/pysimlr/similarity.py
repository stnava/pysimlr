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


def energy_recon_r2(s: torch.Tensor, u: torch.Tensor,
                    ctx: Optional[SimilarityContext] = None) -> torch.Tensor:
    r"""Profiled reconstruction error, :math:`\min_D \lVert X - uDV^\top\rVert_F^2 / \lVert X\rVert_F^2`.

    The same residual as `energy_recon`, evaluated at the best per-component
    scale :math:`D = \mathrm{diag}(d)` instead of at whatever scale the
    projection happened to leave on ``V``, and expressed as a fraction of the
    view's variance -- i.e. :math:`1 - R^2`, in :math:`[0, 1]`.

    Why the plain residual is not usable as shipped
    -----------------------------------------------
    ``recon`` is degree-1.87 homogeneous in ``V``, so it carries a free scale
    that the optimiser is not permitted to set: `simlr_sparseness` re-pins the
    column gauge every sweep. Under the library default
    ``scale_list=["centerAndScale", "np"]`` -- which divides ``X`` by
    :math:`np`, leaving :math:`\lVert X\rVert_F = 0.0074` against
    :math:`u^\top u = (n-1)I` -- the residual decomposes as

        E = ||X||^2 - 2<X,uV'> + ||uV'||^2 = 1.3e-04 - 9.5e-01 + 1.79e+03

    so **99.95% of the objective does not involve the data at all**, the
    gradient is :math:`\approx 2(n-1)V` (pure rescaling; the component able to
    move the support is 2e-4 of its norm), and the optimal scale is
    :math:`d \approx 2.4\times10^{-4}` -- a 4000x correction the projection
    prevents the iterate from making.

    Eliminating the nuisance scale analytically (Golub-Pereyra variable
    projection) rather than asking the optimiser to chase it gives, with
    :math:`Q = (V^\top V)\odot(u^\top u)` and :math:`b_j = (V^\top X^\top u)_{jj}`,

        d* = Q^{-1} b,    E = 1 - b'd* / ||X||^2 .

    ``Q`` is PSD by the Schur product theorem, so :math:`b'Q^{-1}b \ge 0` and
    the value lies in :math:`[0, 1]`; :math:`D = 0` attains the upper end.

    Measured against `energy_recon` on a planted disjoint basis: invariant to
    ``scale_list`` to 8 digits (1793 / 1681 / 18794 before), invariant to a
    ``diag(1, 10, 100)`` regauge of ``V`` (1793 -> 6.0e6 before), gradient
    component along the gauge cos 4e-12 (fraction 1.000000 before), off-orbit
    fraction 0.99 (2e-4 before), and separation between the true partition and
    a row-permuted one of 0.879 vs 0.066 per view -- against 0.03% for
    `energy_recon`, the flatness recorded as GT2 in
    ``tests/test_support_identifiability.py``.

    The scale it profiles out is a gauge, not information: ``D`` rescales
    columns, so the support pattern and every column's direction survive
    untouched. Note ``d`` is not sign-constrained here -- see
    `recon_r2_scales` -- so under ``positivity='positive'`` a negative
    :math:`d_j` would reconstruct from a negatively-loaded component.
    """
    c = _require(ctx, "x", "v", name="recon_r2")
    x, v = c.x, c.v
    xx = torch.sum(x * x)
    if float(xx) <= 0.0:
        raise ValueError(
            "recon_r2 needs a view with non-zero variance; ||X||_F^2 is 0, so "
            "the fraction of explained variance is undefined. A constant view "
            "should be dropped rather than scored."
        )
    q = (v.t() @ v) * (u.t() @ u)
    b = torch.diagonal(v.t() @ x.t() @ u)
    eye = torch.eye(q.shape[0], dtype=q.dtype, device=q.device)
    # Relative ridge: Q is PSD but singular whenever two columns coincide or a
    # column dies, which `enforce_column_floor` makes rare rather than absent.
    ridge = 1e-10 * torch.clamp(q.diagonal().mean().abs(), min=1e-30)
    d = torch.linalg.solve(q + ridge * eye, b)
    # Not clamped to [0, 1]: the bound is guaranteed by the Schur product
    # theorem, and a clamp would zero the gradient exactly where a degenerate
    # view needs it most.
    return (xx - b @ d) / xx


def recon_r2_scales(x: torch.Tensor, u: torch.Tensor,
                    v: torch.Tensor) -> torch.Tensor:
    """The profiled per-component scales ``d*`` that `energy_recon_r2` uses.

    Exposed for diagnostics: ``d`` far from 1 means the iterate's gauge is far
    from the one the data wants, and a negative entry means the profile is
    reconstructing from a sign-flipped component.
    """
    q = (v.t() @ v) * (u.t() @ u)
    b = torch.diagonal(v.t() @ x.t() @ u)
    eye = torch.eye(q.shape[0], dtype=q.dtype, device=q.device)
    ridge = 1e-10 * torch.clamp(q.diagonal().mean().abs(), min=1e-30)
    return torch.linalg.solve(q + ridge * eye, b)


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
        # Hyvarinen's G2. The literature names it both "exponential" and
        # "Gaussian"; there was a separate `kind == "gauss"` branch here
        # evaluating this identical expression, and a separate registry entry
        # for it, so every sweep over the energies ran this contrast twice and
        # reported the two runs as independent results. `gauss` is now an
        # alias in LEGACY_ENERGY_ALIASES and there is one implementation.
        g = -torch.exp(-0.5 * c ** 2)
    elif kind == "kurtosis":
        g = c ** 4
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(kind)
    return -torch.sum(g)


#: E[log cosh(nu)] for a standard normal nu. The reference value Hyvarinen's
#: negentropy approximation subtracts; computed as \int logcosh(y) phi(y) dy.
GAUSSIAN_LOGCOSH = 0.37457


def energy_negentropy(s: torch.Tensor, u: torch.Tensor,
                      ctx: Optional[SimilarityContext] = None) -> torch.Tensor:
    r"""Hyvarinen negentropy on the *marginals*, :math:`-\sum_j (E[G(z_j)] - E[G(\nu)])^2`.

    With :math:`z_j` the standardised j-th latent, :math:`G = \log\cosh` and
    :math:`\nu` standard normal. Minimising this maximises how far each
    component's own distribution is from Gaussian.

    Why it is here, and why it is not one of the four terms above
    -------------------------------------------------------------
    ``logcosh``, ``exp`` and ``kurtosis`` apply :math:`G` to the entries of the
    :math:`k\times k` cross-correlation :math:`c = u^\top z / n` (see
    `_ica_contrast`). That is a criterion on how *matched* two representations
    are; nothing in it looks at a distribution, so it cannot supply ICA
    identifiability however it is weighted.

    This term is what Comon's (1994) theorem is about: with independent
    latents, at most one Gaussian, the mixing is identifiable up to permutation
    and scaling. That matters here because a reconstruction term is provably
    *unable* to identify the support -- :math:`(uT^{-1})(VT^\top)^\top`
    reconstructs identically for any invertible :math:`T`, the rotation
    indeterminacy of factor analysis -- so identification has to come from a
    term that is not :math:`GL(k)`-invariant. Non-negativity is the other
    candidate, but the NMF uniqueness results (separability, sufficiently
    scattered, and min-volume as their surrogate) all require *both* factors
    non-negative, and the consensus returns a signed, whitened ``u``
    (measured: 56-58% negative entries, :math:`u^\top u = (n-1)I`). SiMLR is a
    semi-NMF, so those results do not apply; non-Gaussianity needs no cone and
    does.

    Measured on a planted disjoint basis, 12 random starts at ``w=0.5``,
    added to ``recon_r2`` with weight :math:`\mu`:

    ======  ===================  ===========
    mu      support recovery     agreement
    ======  ===================  ===========
    0       0.6524 +- 0.0260     0.9245
    30      0.7780 +- 0.0123     0.9270
    100     0.8806 +- 0.0901     0.9333
    300     0.9159 +- 0.1234     0.8801
    ======  ===================  ===========

    Alone, with no data term, it scores 0.4313 -- the data term fixes the
    subspace and this picks the rotation within it; neither does the job by
    itself. The spread narrows up to ``mu`` ~ 30 and widens past ~100, so the
    peak at 300 is not the safe operating point.

    The planted latents are :math:`|N(0,1)|`, strongly non-Gaussian by
    construction, which is the regime where this should win. It says nothing
    about how much non-Gaussianity real data carries.
    """
    _check_rows(s, "negentropy")
    z = _centre(s) / _std(_centre(s))
    j = (torch.log(torch.cosh(z)).mean(dim=0) - GAUSSIAN_LOGCOSH) ** 2
    return -torch.sum(j)


def energy_logcosh(s, u, ctx=None):
    """Robust ICA contrast; see `_ica_contrast`."""
    return _ica_contrast(s, u, "logcosh", ctx.eps if ctx else 1e-10)


def energy_exp(s, u, ctx=None):
    """Exponential ICA contrast; see `_ica_contrast`."""
    return _ica_contrast(s, u, "exp", ctx.eps if ctx else 1e-10)


def energy_kurtosis(s, u, ctx=None):
    """Kurtosis ICA contrast; see `_ica_contrast`."""
    return _ica_contrast(s, u, "kurtosis", ctx.eps if ctx else 1e-10)


#: name -> (function, needs-data, one-line description). The single source of
#: truth: adding a similarity is an entry here, not an edit in two modules that
#: can drift apart.
SIMILARITY: Dict[str, tuple] = {
    "recon": (energy_recon, True,
              "||X - u V'||_F^2 : reconstruct the data from the shared latent"),
    "recon_r2": (energy_recon_r2, True,
                 "min_D ||X - u D V'||^2 / ||X||^2 : profiled reconstruction, 1 - R^2 in [0,1]"),
    "align": (energy_align, False,
              "mean((s/sd(s) - u/sd(u))^2) : standardised latent agreement"),
    "acc": (energy_acc, False,
            "-sum|u's|/(n-1) : absolute cross covariance"),
    "nc": (energy_nc, False,
           "-<u,s>/(||u|| ||s||) : normalised correlation, scale invariant"),
    "procrustes": (energy_procrustes, False,
                   "-tr(u's)/||u's||_F : rewards component-wise correspondence"),
    "negentropy": (energy_negentropy, False,
                   "-sum_j (E[G(z_j)] - E[G(nu)])^2 : Hyvarinen negentropy on the marginals"),
    "logcosh": (energy_logcosh, False,
                "-sum logcosh|c| on the k x k cross correlation c = u'z/n (NOT negentropy)"),
    "exp": (energy_exp, False,
            "sum exp(-c^2/2) on the k x k cross correlation (NOT negentropy)"),
    "kurtosis": (energy_kurtosis, False,
                 "-sum c^4 on the k x k cross correlation (NOT negentropy)"),
}

#: Old names, and which term they actually meant in each path. ``regression``
#: is deliberately *not* mapped: it meant two different functions, so resolving
#: it silently would pick a winner and hide the change. Callers must say which.
#: Names already announced as deprecated, so the notice is emitted once.
_DEPRECATION_ANNOUNCED: set = set()

LEGACY_ENERGY_ALIASES = {
    "normalized_correlation": "nc",
    # `gauss` and `exp` were two registry entries evaluating the identical
    # expression -exp(-c^2/2) -- verified bit-identical, 8.956381485164 both --
    # under two descriptions ("Gaussian" and "exponential"). They are the same
    # contrast: Hyvarinen's G2 is named both ways in the literature. Keeping
    # both as entries meant every sweep over `similarity_names()` ran one
    # objective twice and reported it as two independent results. `gauss` now
    # resolves to `exp` and is listed once.
    "gauss": "exp",
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
    if path == "deep" and name in SIMILARITY and SIMILARITY[name][1]:
        raise ValueError(
            f"similarity {name!r} is a function of the data and the loading "
            f"matrix, and the deep path does not carry either into the "
            f"similarity term -- its reconstruction is a separate decoder "
            f"loss. Use 'align' for latent agreement on a deep model."
        )
    if path == "linear" and name in ("regression", "recon"):
        # `recon` is degenerate as an objective: it is minimised by V -> 0.
        # Driven hard on a planted basis it reaches E = 0.0001 by shrinking the
        # basis away, against 1793 at the true basis, and under the library's
        # own default scaling 99.95% of its value is ||u V'||^2 -- a term that
        # does not involve the data at all. `recon_r2` is the same residual
        # with the nuisance scale profiled out and normalised by ||X||^2, so it
        # has a finite non-degenerate minimiser and lies in [0, 1].
        #
        # The registry entry stays: `recon` is still a well-defined quantity
        # and the identifiability ground truths are stated in terms of it.
        # What is deprecated is *selecting it as an objective*, which is what
        # `path="linear"` means here.
        # Once per process, not once per evaluation: this resolves on every
        # energy and gradient call, for every view, on every sweep -- 32k
        # warnings for one test run, which buries everything else.
        if name not in _DEPRECATION_ANNOUNCED:
            _DEPRECATION_ANNOUNCED.add(name)
            warnings.warn(
                f"energy_type={name!r} is deprecated: it is minimised by "
                f"driving the basis to zero. Resolving to 'recon_r2', the "
                f"scale-profiled form. Pass 'recon_r2' explicitly to silence "
                f"this.",
                DeprecationWarning, stacklevel=3,
            )
        return "recon_r2"
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


#: Terms already expressed on a common, absolute per-view scale. Everything
#: else is in the view's own units, so `simlr` divides each view by its own
#: initial energy to make the sum meaningful -- a normalisation that is
#: relative to iteration 0 and therefore says nothing absolute. These need no
#: such treatment and must not receive it: dividing `recon_r2` by its own
#: initial value would discard exactly the interpretation (fraction of
#: variance unexplained) that it exists to provide.
SELF_NORMALISED = frozenset({"recon_r2"})


def similarity_is_self_normalised(name: str) -> bool:
    """True when per-view values are already comparable across views."""
    return resolve_energy_name(name) in SELF_NORMALISED


def similarity_needs_data(name: str) -> bool:
    """True when the term is a function of ``X`` and ``V``, not only of ``s``.

    Two call sites used to hardcode ``name == "recon"`` for this: the linear
    gradient's choice of ``wrt`` and the deep path's refusal. Adding a second
    data term (`recon_r2`) would have silently taken the wrong branch in both
    -- differentiating w.r.t. ``s`` and then applying the chain rule through
    ``X``, which is not the gradient of a term that also depends on ``V``
    directly. The registry already records the flag; this reads it.
    """
    return bool(SIMILARITY[resolve_energy_name(name)][1])


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
