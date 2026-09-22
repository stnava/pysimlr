import warnings

import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from .svd import ba_svd, safe_pca
from .utils import basis_rank_report
from .optimizers import create_optimizer
from .sparsification import orthogonalize_and_q_sparsify, simlr_sparseness, NSA_DEFAULT_W
from .utils import (set_seed_based_on_time, adjusted_rvcoef, safe_svd,
                    invariant_orthogonality_defect, l1_normalize_features, orthogonality_defect,
                    gradient_orthogonality_defect,
                    orthogonality_summary, preprocess_data)
from .consensus import compute_shared_consensus

#: The default `simlr` constraint. Named so that "did the caller choose this
#: weight, or inherit it?" is answerable -- which decides whether an `nsa_w`
#: that disagrees with it is a conflict worth warning about.
#:
#: The weight was 0.1 here while `NSA_DEFAULT_W` was 0.5, so "the default w"
#: was two different numbers depending on which path reached the projection.
#: Both are now 0.5, chosen for conditioning rather than for accuracy. Below
#: about 0.2 the returned basis is rank deficient (effective rank 2.27 of 3 on
#: BRCA4, condition number ~2.8e2); 0.35 sits at the knee (2.83, 1.62) and 0.5
#: is comfortably inside the safe region (2.94, 1.27). Accuracy does not
#: settle the choice: over 40 paired seeds on Diabetes w in [0.30, 0.45] beat
#: 0.25 by +0.027 to +0.069 test R^2 (Wilcoxon p <= 0.002, Friedman
#: p = 3.4e-16), while on BRCA4 w was not distinguishable at all (Friedman
#: p = 0.43) and test accuracy was flat across the whole range. Prediction
#: does not reveal the collapse; only the rank does.
DEFAULT_CONSTRAINT = "orthox0.5x1"


def parse_constraint(constraint_str: str) -> Dict[str, Any]:
    """
    Parse a constraint string into type, weight, and iterations.

    The constraint string follows the format "Type[xWeight[xIterations]]",
    e.g., "Stiefel", "Stiefel x 0.5", or "Stiefel x 0.5 x 5".

    Parameters
    ----------
    constraint_str : str
        The constraint string to parse.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys:
        - "type": The name of the constraint (e.g., "Stiefel", "Grassmann").
        - "weight": The constraint weight (float).
        - "iterations": The number of projection iterations (int).

    Raises
    ------
    TypeError
        If the input is not a string.
    """
    parts = constraint_str.split('x')
    constraint_type = parts[0].strip()

    # Default weight depends on type. Every soft-orthogonality branch in
    # simlr_sparseness is gated on `weight > 0`, so defaulting the "ortho" and
    # "nsaflow" families to 0.0 made a bare constraint="ortho" apply no
    # constraint whatsoever -- and zero the orthogonality penalty in simlr's
    # energy as well. An explicit "orthox0" still selects no constraint.
    if constraint_type in ("Stiefel", "Grassmann", "Stiefel_ns", "Grassmann_ns",
                           "Stiefel_polar", "Grassmann_polar", "NewtonSchulz"):
        weight = 1.0
    elif constraint_type in ("ortho", "nsaflow", "ortho_ns", "nsaflow_ns",
                             "ortho_polar", "nsaflow_polar"):
        # One default weight, shared with `NSA_DEFAULT_W`; see
        # `DEFAULT_CONSTRAINT` for the evidence and how thin it is.
        weight = 0.5
    elif constraint_type == "none":
        weight = 0.0
    else:
        warnings.warn(
            f"Unrecognised constraint type {constraint_type!r} in "
            f"{constraint_str!r}; no manifold constraint will be applied. "
            f"Known types: Stiefel, Grassmann, NewtonSchulz, ortho, nsaflow "
            f"(optionally suffixed _ns or _polar), none.",
            UserWarning, stacklevel=2,
        )
        weight = 0.0

    iterations = 1
    if len(parts) > 1:
        try: weight = float(parts[1])
        except ValueError: pass
    if len(parts) > 2:
        try: iterations = int(parts[2])
        except ValueError: pass
        if iterations != 1:
            # The third field of the constraint string does nothing. It is
            # forwarded to `simlr_sparseness` as `constraint_iterations`,
            # which has ignored it since the projection became the single
            # constraint operator -- measured, "orthox0.5x1" and "orthox0.5x9"
            # produce a bit-identical basis. The documented format keeps the
            # field so existing strings still parse, but a caller who sets it
            # is asking for repeated projection and not getting it.
            warnings.warn(
                f"constraint={constraint_str!r}: the iterations field "
                f"({iterations}) has no effect -- the projection is applied "
                f"once per sweep and is solved to tolerance. Use the weight "
                f"field to change the constraint's strength.",
                DeprecationWarning, stacklevel=2,
            )
    return {"type": constraint_type, "weight": weight, "iterations": iterations}

def project_gradient(v_grad: torch.Tensor, v_current: torch.Tensor, constraint_type: str) -> torch.Tensor:
    """
    Project the gradient onto the tangent space of a specific manifold.

    This ensures that optimization updates remain valid with respect to 
    manifold constraints (e.g., Stiefel or Grassmann manifolds), following 
    the theory in Edelman et al. (1998).

    Parameters
    ----------
    v_grad : torch.Tensor
        The raw gradient of the energy function.
    v_current : torch.Tensor
        The current value of the basis matrix (point on the manifold).
    constraint_type : str
        The type of manifold/constraint ("Stiefel", "Grassmann", etc.).

    Returns
    -------
    torch.Tensor
        The projected gradient.

    Raises
    ------
    TypeError
        If inputs are not tensors.
    """
    if constraint_type == "Grassmann":
        # Project onto Grassmann tangent space: G = G - V(V^T G)
        return v_grad - v_current @ (v_current.t() @ v_grad)
    elif constraint_type == "Stiefel":
        # Project onto Stiefel tangent space: G = G - V sym(V^T G)
        vtg = v_current.t() @ v_grad
        sym_vtg = 0.5 * (vtg + vtg.t())
        return v_grad - v_current @ sym_vtg
    return v_grad

def calculate_u(projections: List[torch.Tensor], 
                mixing_algorithm: str = "newton", 
                k: Optional[int] = None,
                orthogonalize: bool = False) -> torch.Tensor:
    """
    Deprecated: Use compute_shared_consensus from .consensus instead.

    Parameters
    ----------
    projections : List[torch.Tensor]
        List of projected data matrices (N x K).
    mixing_algorithm : str, default="svd"
        The algorithm used to mix projections ("svd", "pca", "ica", "avg", "newton").
        The algorithm to use for consensus computation.
    k : int, optional
        Target rank for the shared latent space.
    orthogonalize : bool, default=False
        Whether to orthogonalize the resulting consensus matrix.

    Returns
    -------
    torch.Tensor
        The shared consensus latent matrix (N x K).

    Raises
    ------
    TypeError
        If projections is not a list of tensors.
    """
    return compute_shared_consensus(projections, mixing_algorithm, k, orthogonalize)

#: Every strategy `initialize_simlr` implements. Kept as one tuple so the
#: error message and the dispatcher can't drift apart.
INITIALIZATION_TYPES = ("pca", "random", "gcca", "nndsvd", "perturbed_pca",
                        "domain", "best_of_n")


def initialize_simlr(data_matrices: List[torch.Tensor],
                     k: int,
                     initialization_type: str = "pca",
                     joint_reduction: bool = True,
                     positivity: str = "either",
                     generator: Optional[torch.Generator] = None,
                     domain_matrices: Optional[List[Optional[torch.Tensor]]] = None,
                     n_candidates: int = 5,
                     perturbation_scale: float = 0.05) -> List[torch.Tensor]:
    """
    Initialize the basis matrices (V) for SiMLR.

    Provides various initialization strategies, primarily based on SVD/PCA,
    to ensure the optimization starts from a reasonable point.

    Parameters
    ----------
    data_matrices : List[torch.Tensor]
        List of input data matrices (N x P_i).
    k : int
        Target rank for the latent space.
    initialization_type : str, default="pca"
        ``"pca"`` -- truncated SVD of each view. Deterministic: it is a
        function of the data alone, so repeated calls return bit-identical
        bases whatever the seed.

        ``"random"`` -- a random feasible basis, drawn from ``generator``.

        The distinction is load-bearing rather than cosmetic. Under ``"pca"``
        a "multi-start" experiment is vacuous: measured over 5 seeds the
        initial bases were identical to 0.00e+00, so any agreement statistic
        across starts is 1.0 by construction and says nothing. It also means
        SiMLR *begins* at PCA and, at the default ``learning_rate=0.001``
        (LARS moves ``V`` by exactly ``lr`` of its norm per sweep), ends
        within ~0.4% of it -- which is most of why PCA is hard to beat in the
        benchmarks.

        ``"random"`` exists so that identifiability can be measured: whether
        independent starts reach the same support is the empirical content of
        "the constraint identifies the factorisation", and it cannot be asked
        of a deterministic initialiser.

        ``"gcca"`` -- a joint (generalized-CCA, MAXVAR) start. Per-view PCA is
        blind to which of a view's directions are shared with the other
        views; where the shared signal is a minority of any one view's own
        variance (private structure dominates -- the regime
        `test_positivity_and_gradient_consistency.py`'s xfail measures a real
        but small optimizer gain in), that makes PCA a poor start for the
        objective SiMLR actually maximizes. This instead finds the sample-space
        subspace best explained *jointly* by every view's column space, and
        regresses each view onto it. Costs one extra SVD and an eigendecomposition
        of an (N x N) matrix, so it is not free on large N.

        Measured (`scripts/compare_initializations.py`, 3 views, N=120, K=3,
        10 seeds, consensus via `compute_shared_consensus`): in a "buried"
        regime where per-view private structure dominates the shared signal,
        PCA's own consensus starts at Procrustes R^2 0.033 while `"gcca"`
        starts at 0.992 -- confirming the failure mode this is meant for is
        real and that this fixes it *at initialization*. But `simlr`'s own
        15-sweep fit then erodes most of that advantage (0.992 -> 0.846),
        while starting from PCA the fit only reaches 0.054 -- so `"gcca"`
        still wins after fitting, just by much less than its head start would
        suggest, and the sweep is actively working against a good start here,
        not merely failing to improve on a bad one. In the well-conditioned
        "easy" regime (PCA already near the optimum) every strategy tested
        starts and finishes within noise of 0.99, so `"gcca"` is not worse
        there either. Only two regimes, only these generators -- this is
        evidence the direction is worth pursuing, not a general benchmark.

        ``"nndsvd"`` -- Boutsidis & Gallopoulos (2008) non-negative double SVD.
        For the non-negative case, an alternative to `initial_basis_for_view`'s
        `nsa_flow_data` fit that needs no extra backend: it builds a
        non-negative basis from the sign-split SVD components rather than
        rectifying signed loadings (which, per that function's docstring,
        breaks orthogonality outright).

        ``"perturbed_pca"`` -- PCA plus a small random perturbation, retracted
        to feasibility. A middle ground between ``"pca"`` (identical across
        seeds, so multi-start is vacuous) and ``"random"`` (throws away PCA's
        near-optimality in the well-conditioned case entirely): stays in PCA's
        basin while still giving genuine seed-to-seed variation for
        identifiability testing. Scale set by `perturbation_scale`.

        ``"domain"`` -- seeds each view's basis from the leading directions of
        that view's `domain_matrices` entry (the same prior used by the
        ``"dat"`` energy term) instead of from the data at all. Needs
        `domain_matrices`.

        ``"best_of_n"`` -- draws `n_candidates` candidates (PCA plus
        `n_candidates - 1` random feasible starts) and keeps whichever scores
        highest on a cheap joint-agreement proxy (projecting every view onto a
        rough shared direction and summing the alignment). The proxy is not
        SiMLR's actual energy -- it is data-only and ignores `energy_type`,
        `constraint` and `positivity`'s interaction with the optimizer -- so
        treat this as a way to avoid an unlucky random draw, not as a
        surrogate for running `simlr` itself.

        Empirically this has not paid off: 0 wins out of 21 method-by-dataset
        blocks across every full real-data sweep run in this session (Heart,
        Diabetes, mfeat, ADNI+PPMI, TCGA-KIRC x SiMLR/LEND/NED/Flow-SiMR-V),
        even after fixing `_score_joint_basis`'s scale bias (see its
        docstring). The proxy apparently does not track downstream task
        performance closely enough to be a useful selector. Left implemented
        and tested -- the mechanism may still be useful with a better proxy --
        but benchmark scripts in this repo exclude it by default; see
        `scripts/real_data_init_comparison.py`'s `BENCHMARK_INITIALIZATION_TYPES`.
    joint_reduction : bool, default=True
        Reserved for a future joint-reduction initializer and currently
        ignored. Previously both of these arguments were accepted and silently
        discarded, so callers could believe they had selected a strategy that
        did not exist.
    positivity : str, default="either"
        When this requests a non-negative basis and the NSA-Flow backend is
        installed, the basis is fitted directly from the data instead of being
        taken from the signed PCA loadings.
    generator : torch.Generator, optional
        Source of randomness for ``initialization_type`` in
        ``{"random", "perturbed_pca", "best_of_n"}``. A local generator, never
        the global stream: seeding globally inside a library function makes a
        caller's own ``seed=`` inert and breaks the independence of repeated
        benchmark runs.
    domain_matrices : list of torch.Tensor, optional
        Per-view prior matrices for ``initialization_type="domain"``, one per
        view, each sharing that view's feature dimension (the same convention
        as the ``"dat"`` energy's `prior_matrix`). A ``None`` entry falls back
        to PCA for that view.
    n_candidates : int, default=5
        Number of candidates drawn for ``initialization_type="best_of_n"``.
    perturbation_scale : float, default=0.05
        Standard deviation, relative to the PCA basis itself, of the noise
        added for ``initialization_type="perturbed_pca"``.

    Returns
    -------
    List[torch.Tensor]
        Initial basis matrices (P_i x K) for each view.

    Raises
    ------
    TypeError
        If input matrices are not tensors.

    Notes
    -----
    Under a non-negative constraint the signed PCA loadings are not a usable
    starting point, and the fix is not to rectify them. `simlr_sparseness`
    applies the positivity constraint by reflection, so a signed
    initialization reached the first retraction as ``abs(V_pca)``: a matrix
    whose orthogonality has been destroyed outright (normalized Stiefel defect
    1.60, against 0.00 for the signed loadings it came from), and the anchored
    retraction then stays near that damaged target rather than repairing it.

    Fitting the basis from ``X`` with `nsa_flow_data` avoids the rectification
    entirely. Measured over 10 seeds against a known non-negative basis, in
    recovery and in defect:

    ==============================  ==========  ==========  ==========  ======
    candidate                       disjoint    defect      overlapping defect
    ==============================  ==========  ==========  ==========  ======
    ``abs(PCA)`` then retract       0.9162      0.62        0.8112      0.72
    signed PCA then retract         0.7451      0.43        0.7691      0.46
    ``nsa_flow_data(X)``            **0.9945**  **0.05**    **0.9032**  0.13
    ==============================  ==========  ==========  ==========  ======

    Note that feeding the *signed* loadings to the non-negative solver is worse
    than rectifying them, not better: the solve is anchored, so a signed target
    pushes entries to zero wherever the anchor is negative rather than where
    the data says the basis is small. Solving from the data is what removes the
    problem. The synthetic truth used above is itself non-negative, which
    flatters a non-negative fit; the defect column does not depend on that,
    since ``abs(PCA)`` simply is not orthogonal.
    """
    if initialization_type not in INITIALIZATION_TYPES:
        raise ValueError(
            f"initialization_type={initialization_type!r} is not implemented; "
            f"choose one of {INITIALIZATION_TYPES}."
        )
    if initialization_type in ("random", "perturbed_pca", "best_of_n") and generator is None:
        raise ValueError(
            f"initialization_type={initialization_type!r} needs an explicit "
            f"`generator`. Falling back to the global RNG would make the "
            f"result depend on unrelated draws elsewhere in the process, "
            f"which is exactly what a multi-start experiment must not do."
        )
    if initialization_type == "domain":
        if domain_matrices is None:
            raise ValueError(
                "initialization_type='domain' needs `domain_matrices`, one "
                "entry per view."
            )
        return _domain_informed_basis(data_matrices, domain_matrices, k, positivity)
    if initialization_type == "gcca":
        return _gcca_basis(data_matrices, k, positivity)
    if initialization_type == "nndsvd":
        return [_nndsvd_basis_for_view(x, k) for x in data_matrices]
    if initialization_type == "perturbed_pca":
        base = [initial_basis_for_view(x, k, positivity=positivity,
                                       initialization_type="pca")
                for x in data_matrices]
        out = []
        for v in base:
            noise = torch.randn(v.shape, generator=generator, dtype=v.dtype)
            raw = v + perturbation_scale * v.norm() * noise / (noise.norm() + 1e-12)
            out.append(simlr_sparseness(raw, constraint_type="ortho", positivity=positivity))
        return out
    if initialization_type == "best_of_n":
        return _best_of_n_basis(data_matrices, k, positivity, generator, n_candidates)
    return [initial_basis_for_view(x, k, positivity=positivity,
                                   initialization_type=initialization_type,
                                   generator=generator)
            for x in data_matrices]


def _gcca_basis(data_matrices: List[torch.Tensor], k: int,
                positivity: str) -> List[torch.Tensor]:
    """Joint (generalized-CCA, MAXVAR) initialization.

    Finds the top-``k`` eigenvectors of ``sum_i P_i``, where ``P_i`` projects
    onto view ``i``'s column space -- the sample-space subspace best agreed on
    by every view jointly -- then regresses each view onto it by least
    squares to recover a per-view basis. See `initialize_simlr`'s docstring
    for why this targets the buried-shared-signal regime that per-view PCA
    does not.
    """
    n = data_matrices[0].shape[0]
    joint = torch.zeros(n, n, dtype=torch.float64)
    for x in data_matrices:
        xd = x.detach().double()
        u, s, _ = safe_svd(xd, full_matrices=False)
        tol = float(s.max()) * max(xd.shape) * torch.finfo(torch.float64).eps
        rank = max(1, int((s > tol).sum()))
        u_r = u[:, :rank]
        joint += u_r @ u_r.t()
    eigvals, eigvecs = torch.linalg.eigh(joint)
    # `eigh` returns ascending eigenvalues; the top-k shared directions are
    # the last k columns, largest first.
    shared_u = eigvecs[:, -k:].flip(dims=[1]).contiguous()
    out = []
    for x in data_matrices:
        xd = x.detach().double()
        v = torch.linalg.lstsq(xd, shared_u).solution
        # The regression solution's scale reflects how well this view
        # predicts the shared code, which can be far from unit -- rescale to
        # unit columns first so every initializer hands the retraction a
        # basis on the same footing as PCA's (already unit-norm) columns,
        # rather than a magnitude the retraction's fixed weight only partly
        # corrects.
        v = v / (v.norm(dim=0, keepdim=True) + 1e-12)
        out.append(simlr_sparseness(v.to(x.dtype), constraint_type="ortho",
                                    positivity=positivity))
    return out


def _nndsvd_basis_for_view(x: torch.Tensor, k: int) -> torch.Tensor:
    """NNDSVD (Boutsidis & Gallopoulos, 2008), adapted to this module's
    ``(features, k)`` basis convention.

    For each rank-1 SVD term ``s_j * u_j @ v_j.T``, splits ``u_j`` and ``v_j``
    into their non-negative and non-positive parts and keeps whichever pairing
    (positive/positive or negative/negative -- the two that are sign-
    consistent with a non-negative product) carries more of the term's norm.
    Unlike rectifying signed PCA loadings with `abs` or `clamp`, this is a
    property of the SVD term itself, not a repair applied after the fact.
    """
    xd = x.detach().double()
    u, s, vt = safe_svd(xd, full_matrices=False)
    k_eff = min(k, s.shape[0])
    v = torch.zeros(x.shape[1], k, dtype=torch.float64)
    for j in range(k_eff):
        uj, vj = u[:, j], vt[j, :]
        up, un = torch.clamp(uj, min=0.0), torch.clamp(-uj, min=0.0)
        vp, vn = torch.clamp(vj, min=0.0), torch.clamp(-vj, min=0.0)
        norm_up, norm_un = torch.norm(up), torch.norm(un)
        norm_vp, norm_vn = torch.norm(vp), torch.norm(vn)
        term_p = (norm_up * norm_vp).item()
        term_n = (norm_un * norm_vn).item()
        if max(term_p, term_n) <= 0:
            continue
        if term_p >= term_n:
            scale = (float(s[j]) * term_p) ** 0.5
            v[:, j] = vp / norm_vp * scale
        else:
            scale = (float(s[j]) * term_n) ** 0.5
            v[:, j] = vn / norm_vn * scale
    if k_eff < k:
        # Fewer usable singular directions than requested rank: pad with a
        # tiny non-negative perturbation rather than leave a dead column, the
        # same convention `initial_basis_for_view`'s PCA path uses.
        v[:, k_eff:] = 1e-4 * torch.rand(x.shape[1], k - k_eff, dtype=torch.float64)
    return v.to(x.dtype)


def _domain_informed_basis(data_matrices: List[torch.Tensor],
                           domain_matrices: List[Optional[torch.Tensor]],
                           k: int, positivity: str) -> List[torch.Tensor]:
    """Seed each view's basis from its domain prior's own leading directions.

    ``domain_matrices[i]`` must share view ``i``'s feature dimension, the same
    convention the ``"dat"`` energy term uses for `prior_matrix`.
    """
    if len(domain_matrices) != len(data_matrices):
        raise ValueError(
            f"domain_matrices has {len(domain_matrices)} entries but there "
            f"are {len(data_matrices)} views."
        )
    out = []
    for x, dm in zip(data_matrices, domain_matrices):
        if dm is None:
            out.append(initial_basis_for_view(x, k, positivity=positivity,
                                              initialization_type="pca"))
            continue
        dm_t = torch.as_tensor(dm)
        if dm_t.shape[-1] != x.shape[1]:
            raise ValueError(
                f"domain_matrices entry has {dm_t.shape[-1]} columns but its "
                f"view has {x.shape[1]} features; a 'domain' initializer's "
                f"prior must share the view's feature dimension, the same "
                f"convention the 'dat' energy uses."
            )
        _, _, vt = safe_svd(dm_t.double(), full_matrices=False)
        k_eff = min(k, vt.shape[0])
        v = vt[:k_eff, :].t().contiguous()
        if k_eff < k:
            pad = 1e-4 * torch.randn(x.shape[1], k - k_eff, dtype=torch.float64)
            v = torch.cat([v, pad], dim=1)
        out.append(simlr_sparseness(v.to(x.dtype), constraint_type="ortho",
                                    positivity=positivity))
    return out


def _score_joint_basis(data_matrices: List[torch.Tensor],
                       v_mats: List[torch.Tensor]) -> float:
    """Cheap, data-only proxy for how much shared structure a candidate basis
    exposes: projects every view through its candidate basis, takes a rough
    consensus direction (the top singular vectors of the summed projections),
    and sums each view's alignment with it. Not `simlr`'s actual energy --
    see `initialize_simlr`'s docstring for what this proxy does and does not
    stand in for.

    Each view's projection is rescaled to unit Frobenius norm before scoring.
    Without this, `"random"` candidates (whose raw basis norm ends up several
    times PCA's, since `simlr_sparseness`'s retraction does not renormalize to
    a fixed scale) win purely on magnitude: `||u'z||_F` scales with `||z||_F`
    regardless of how well `z` actually agrees with the consensus direction
    `u`. Measured on real data (`scripts/real_data_init_x_method.py`) this was
    not a theoretical concern -- `best_of_n` underperformed its own `"pca"`
    candidate in every one of 12 method-by-dataset cells checked, which is
    only possible if the proxy it selects by does not track the thing being
    optimised for.
    """
    zs = [x @ v for x, v in zip(data_matrices, v_mats)]
    zs = [z / (torch.norm(z, p='fro') + 1e-12) for z in zs]
    stacked = torch.stack(zs, dim=0).sum(dim=0)
    u, _, _ = safe_svd(stacked, full_matrices=False)
    return float(sum(torch.norm(u.t() @ z, p='fro').item() for z in zs))


def _best_of_n_basis(data_matrices: List[torch.Tensor], k: int, positivity: str,
                     generator: torch.Generator, n_candidates: int) -> List[torch.Tensor]:
    """Draw `n_candidates` feasible starts (PCA plus random draws) and keep
    the one `_score_joint_basis` ranks highest."""
    candidates = [initialize_simlr(data_matrices, k, initialization_type="pca",
                                   positivity=positivity)]
    for _ in range(max(0, n_candidates - 1)):
        candidates.append(initialize_simlr(data_matrices, k, initialization_type="random",
                                           positivity=positivity, generator=generator))
    scores = [_score_joint_basis(data_matrices, c) for c in candidates]
    return candidates[int(np.argmax(scores))]


def initialize_deep_encoders(encoders, data_matrices: List[torch.Tensor], k: int,
                             initialization_type: str = "pca",
                             generator: Optional[torch.Generator] = None,
                             domain_matrices: Optional[List[Optional[torch.Tensor]]] = None,
                             n_candidates: int = 5,
                             perturbation_scale: float = 0.05) -> None:
    """Seed a deep model's per-view first-layer encoders, in place.

    Shared by `LENDSiMRModel`, `NEDSiMRModel`, `NEDSharedPrivateSiMRModel` and
    `FlowSiMRVModel`'s `initialize_v` so every `INITIALIZATION_TYPES` strategy
    -- not only PCA -- is available to the deep entry points too, rather than
    the deep path being permanently stuck on the linear path's original
    default. `positivity` is read from `encoders[0]` rather than taken as a
    parameter: every encoder in a model is built from the same top-level
    `positivity` argument, so they agree by construction, and this keeps that
    invariant enforced in one place instead of asking each caller to pass it
    again.

    Parameters
    ----------
    encoders : sequence of encoder modules
        Each must expose `.positivity` (str) and `.v_raw` (the raw,
        unconstrained parameter -- constraints such as `positivity` are
        applied downstream by the encoder's own `.v` property, the same
        `initial_basis_for_view` relies on for the linear path).
    data_matrices, k, initialization_type, generator, domain_matrices,
    n_candidates, perturbation_scale : see `initialize_simlr`.
    """
    positivity = encoders[0].positivity
    v_mats = initialize_simlr(data_matrices, k, initialization_type=initialization_type,
                              positivity=positivity, generator=generator,
                              domain_matrices=domain_matrices, n_candidates=n_candidates,
                              perturbation_scale=perturbation_scale)
    with torch.no_grad():
        for enc, x, v in zip(encoders, data_matrices, v_mats):
            enc.v_raw.copy_(v.to(x.dtype))


def initial_basis_for_view(x: torch.Tensor, k: int,
                           positivity: str = "either",
                           initialization_type: str = "pca",
                           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Build the initial ``(features, k)`` basis for one view.

    Shared by `initialize_simlr` and the deep models' ``initialize_v`` so that
    every entry point starts from the same basis for a given constraint.

    Under a non-negative constraint the basis is fitted from ``x`` with
    `nsa_flow_data` when the backend is available. Otherwise it is the signed
    PCA loadings, with each column's sign resolved so that its sum is positive
    -- a sign flip preserves orthogonality exactly, unlike the rectification
    that a non-negative constraint would otherwise apply downstream.

    The deep encoders need this as much as the linear model does: they rectify
    at use time (`LENDNSAEncoder.v` clamps negatives) rather than at init, so a
    signed start reaches the first forward pass as ``clamp(V_pca)``, which is
    the same loss of orthogonality as ``abs(V_pca)`` by another route.

    Parameters
    ----------
    x : torch.Tensor
        The view, shape (samples, features).
    k : int
        Target rank.
    positivity : str, default="either"
        When non-negative, fit from the data rather than rectifying PCA.

    Returns
    -------
    torch.Tensor
        Basis of shape (features, k) in `x`'s dtype.
    """
    from .nsa_backend import load_nsa_flow_data
    from .sparsification import NSA_DEFAULT_W, _usable_retraction

    want_nonneg = positivity in ('positive', 'hard', 'nonnegative', 'nonneg',
                                 'softplus')

    if initialization_type == "random":
        if generator is None:
            raise ValueError("initialization_type='random' needs a `generator`")
        # Drawn feasible, not drawn then repaired: a raw draw is neither
        # near-orthogonal nor sparse, so the first retraction would move it
        # much further than any subsequent sweep and the "start" recorded by
        # the caller would not be the point the optimiser actually began from.
        from .sparsification import simlr_sparseness
        raw = (torch.rand(x.shape[1], k, generator=generator, dtype=x.dtype)
               if want_nonneg else
               torch.randn(x.shape[1], k, generator=generator, dtype=x.dtype))
        return simlr_sparseness(raw, constraint_type="ortho",
                                positivity=positivity).to(x.dtype)
    if want_nonneg and x.shape[1] > k:
        fit_from_data = load_nsa_flow_data()
        if fit_from_data is not None:
            fitted = _nonnegative_basis_from_data(
                fit_from_data, x, k, NSA_DEFAULT_W, _usable_retraction)
            if fitted is not None:
                return fitted.to(x.dtype)

    u, s, v = ba_svd(x, nu=0, nv=k)
    if v.shape[1] < k:
        padding = torch.randn(v.shape[0], k - v.shape[1],
                              dtype=v.dtype, device=v.device) * 1e-4
        v = torch.cat([v, padding], dim=1)
    if want_nonneg:
        # Resolve each column's sign so downstream rectification removes as
        # little as possible. This is a sign flip, not a rectification, so the
        # basis stays exactly as orthogonal as PCA made it.
        for j in range(v.shape[1]):
            if v[:, j].sum() < 0:
                v[:, j] *= -1
    return v.to(x.dtype)


def _nonnegative_basis_from_data(fit_from_data, x: torch.Tensor, k: int,
                                 w: float, usable) -> Optional[torch.Tensor]:
    """
    Fit a non-negative, near-orthogonal basis for `x` directly from the data.

    Returns None if the backend raises or returns something unusable, so the
    caller falls back to the PCA initializer rather than starting from a
    degenerate basis. A wide view (``p <= k``) is left to PCA: no basis with
    more columns than rows can be orthogonal, so there is nothing to fit.
    """
    reference = torch.ones(x.shape[1], k, dtype=torch.float64)
    try:
        result = fit_from_data(x.detach().double(), k=k, w=float(w))
    except Exception:
        return None
    candidate = None
    if hasattr(result, 'get'):
        candidate = result.get('V') or result.get('Y')
    if candidate is None:
        candidate = getattr(result, 'V', None) or getattr(result, 'Y', None)
    if candidate is None:
        return None
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    candidate = torch.clamp_min(candidate, 0.0)
    if not usable(candidate, reference):
        return None
    return candidate


def nsa_contrast_transform(x: Union[torch.Tensor, np.ndarray],
                           k: int = 6,
                           w: float = 0.5,
                           consolidate: bool = True,
                           optimizer: str = "torch_lbfgs",
                           max_iter: Optional[int] = None,
                           tol: Optional[float] = None) -> Dict[str, Any]:
    """
    Fit an NSA-Flow Signed Contrast representation (Recipe B / Guide v2.11.0+).

    Lifts standardized feature profiles into non-overlapping positive and negative
    lobes (V = V^+ - V^-) with strictly disjoint supports when `consolidate=True`.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input data matrix of shape (n_samples, n_features).
    k : int, default=6
        Number of components to extract.
    w : float, default=0.5
        Trade-off weight in [0, 1]. w=0 maximizes reconstruction fidelity,
        w=1 maximizes orthogonality (disjoint supports).
    consolidate : bool, default=True
        Guarantees strictly disjoint supports (zero lobe overlap).
    optimizer : str, default="torch_lbfgs"
        Optimization algorithm ("torch_lbfgs", "spg", or "lbfgs").
    max_iter : int, optional
        Maximum iterations cap.
    tol : float, optional
        Stationarity tolerance.

    Returns
    -------
    dict
        Dictionary containing:
        - 'v': Component loading matrix [features, k]
        - 'scores': Projection scores [samples, k]
        - 'result': Underlying NSAResult object
    """
    from .nsa_backend import load_nsa_flow
    fn = load_nsa_flow()
    if fn is None:
        raise RuntimeError("NSA-Flow is not installed. Install with `pip install nsa-flow`.")
    x_t = torch.as_tensor(x, dtype=torch.float32)
    res = fn(x_t, k=k, w=w, mode="signed", consolidate=consolidate,
             optimizer=optimizer, max_iter=max_iter, tol=tol)
    v = getattr(res, 'V', None)
    if v is None:
        v = getattr(res, 'Y', None)
    if v is None and hasattr(res, 'get'):
        v = res.get('V') if res.get('V') is not None else res.get('Y')
    scores = x_t @ v
    return {
        "v": v,
        "scores": scores,
        "result": res,
    }


def nsa_nonnegative_transform(x: Union[torch.Tensor, np.ndarray],
                              k: int = 6,
                              w: float = 0.5,
                              optimizer: str = "torch_lbfgs",
                              max_iter: Optional[int] = None,
                              tol: Optional[float] = None) -> Dict[str, Any]:
    """
    Fit an NSA-Flow Non-Negative representation on physical quantities (Recipe A).

    Extracts physically realizable non-negative constituent spectra without negative
    loadings, maintaining frame defect D ≈ 0. Do not mean-center input features.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input data matrix of shape (n_samples, n_features), where x >= 0.
    k : int, default=6
        Number of components to extract.
    w : float, default=0.5
        Trade-off weight in [0, 1].
    optimizer : str, default="torch_lbfgs"
        Optimization algorithm ("torch_lbfgs", "spg", or "lbfgs").
    max_iter : int, optional
        Maximum iterations cap.
    tol : float, optional
        Stationarity tolerance.

    Returns
    -------
    dict
        Dictionary containing:
        - 'v': Non-negative constituent loading matrix [features, k]
        - 'scores': Projection scores [samples, k]
        - 'result': Underlying NSAResult object
    """
    from .nsa_backend import load_nsa_flow
    fn = load_nsa_flow()
    if fn is None:
        raise RuntimeError("NSA-Flow is not installed. Install with `pip install nsa-flow`.")
    x_t = torch.as_tensor(x, dtype=torch.float32)
    res = fn(x_t, k=k, w=w, mode="data", nonneg=True,
             optimizer=optimizer, max_iter=max_iter, tol=tol)
    v = getattr(res, 'V', None)
    if v is None:
        v = getattr(res, 'Y', None)
    if v is None and hasattr(res, 'get'):
        v = res.get('V') if res.get('V') is not None else res.get('Y')
    if v is not None:
        v = torch.clamp_min(v, 0.0)
    scores = x_t @ v
    return {
        "v": v,
        "scores": scores,
        "result": res,
    }

def calculate_ica_energy(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nonlinearity: str = "logcosh", a: float = 1.0) -> torch.Tensor:
    """
    Compute the ICA-based energy (negentropy) for a given projection.

    Measures how non-Gaussian the projected data is, which is a common 
    objective in Independent Component Analysis.

    Parameters
    ----------
    x : torch.Tensor
        Input data matrix (N x P).
    u : torch.Tensor
        Shared latent space (N x K).
    v : torch.Tensor
        Basis matrix (P x K).
    nonlinearity : str, default="logcosh"
        The contrast function to use. Options: 'logcosh', 'exp', 'gauss', 'kurtosis'.
    a : float, default=1.0
        Hyperparameter for the contrast function (used by 'gauss').

    Returns
    -------
    torch.Tensor
        The computed energy value (scalar).

    Raises
    ------
    TypeError
        If inputs are not valid tensors.
    """
    s = (u.t() @ x) @ v
    n = x.shape[0]
    if nonlinearity == "logcosh":
        abs_s = torch.abs(s)
        return -torch.sum(abs_s - np.log(2.0) + torch.log1p(torch.exp(-2.0 * abs_s))) / n
    elif nonlinearity == "exp": return -torch.sum(-torch.exp(-s**2 / 2.0)) / n
    elif nonlinearity == "gauss": return -torch.sum(-0.5 * torch.exp(-a * s**2)) / n
    elif nonlinearity == "kurtosis": return -torch.sum((s**4.0) / 4.0) / n
    return torch.tensor(0.0, dtype=x.dtype, device=x.device)

def calculate_ica_gradient(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nonlinearity: str = "logcosh", a: float = 1.0) -> torch.Tensor:
    """
    Compute the gradient of the ICA-based energy with respect to the basis matrix V.

    Used by gradient-based optimizers to update the projection weights towards 
    maximum non-Gaussianity.

    Parameters
    ----------
    x : torch.Tensor
        Input data matrix (N x P).
    u : torch.Tensor
        Shared latent space (N x K).
    v : torch.Tensor
        Basis matrix (P x K).
    nonlinearity : str, default="logcosh"
        The contrast function used.
    a : float, default=1.0
        Hyperparameter for the contrast function.

    Returns
    -------
    torch.Tensor
        The gradient matrix (P x K).

    Raises
    ------
    TypeError
        If inputs are not valid tensors.
    """
    s = (u.t() @ x) @ v
    # Must match the normalizer used by calculate_ica_energy, which divides by
    # the sample count. This previously divided by s.shape[0] == k, so the
    # gradient was off by a factor of n/k relative to the energy it belongs to
    # -- enough to invalidate any Armijo sufficient-decrease test.
    n = x.shape[0]
    if nonlinearity == "logcosh": return (1.0 / n) * (x.t() @ u @ torch.tanh(s))
    elif nonlinearity == "exp": return (1.0 / n) * (x.t() @ u @ (s * torch.exp(-s**2 / 2.0)))
    elif nonlinearity == "gauss": return (1.0 / n) * (x.t() @ u @ (a * s * torch.exp(-a * s**2)))
    elif nonlinearity == "kurtosis": return (1.0 / n) * (x.t() @ u @ (s**3))
    return torch.zeros_like(v)

#: Objectives with both an energy and a matching gradient. Anything else is
#: refused by both, rather than silently evaluating to zero.
SUPPORTED_ENERGY_TYPES = frozenset({
    "regression", "acc", "logcosh", "exp", "gauss", "kurtosis",
    "nc", "normalized_correlation", "dat",
})


def calculate_simlr_energy(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, energy_type: str = "regression", lambda_val: float = 0.0, prior_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the energy (loss) for a single modality in SiMLR.

    Supports various objective functions including reconstruction error, 
    canonical correlation (ACC), ICA-based negentropy, and domain-knowledge 
    alignment.

    Parameters
    ----------
    v : torch.Tensor
        Basis matrix (P x K) for the current modality.
    x : torch.Tensor
        Input data matrix (N x P) for the current modality.
    u : torch.Tensor
        Shared latent space (N x K).
    energy_type : str, default="regression"
        The objective function to use. Options:
        - 'regression': Reconstruction MSE (||X - UV^T||^2).
        - 'acc': Maximum absolute covariance (canonical correlation).
        - 'logcosh', 'exp', 'gauss', 'kurtosis': ICA-based negentropy.
        - 'nc' or 'normalized_correlation': Cosine similarity.
        - 'dat': Alignment with domain knowledge `prior_matrix`.
    lambda_val : float, default=0.0
        Scaling factor for 'dat' energy.
    prior_matrix : Optional[torch.Tensor], default=None
        Domain knowledge matrix (K x P) for 'dat' energy.

    Returns
    -------
    torch.Tensor
        The computed energy value (scalar).

    Raises
    ------
    TypeError
        If inputs are not valid tensors.
    """
    from .similarity import (SimilarityContext, resolve_energy_name,
                             similarity_energy)
    u = u.to(x.dtype); v = v.to(x.dtype)
    if energy_type == "dat":
        # Domain regulariser, not a similarity: it scores V against a prior
        # matrix rather than against the shared latent, so it stays here.
        if prior_matrix is None:
            raise ValueError(
                "energy_type='dat' is a domain regulariser and requires "
                "prior_matrix. It used to return a constant 0.0, so a caller "
                "who passed it as a primary objective optimised nothing."
            )
        alignment = prior_matrix.to(x.dtype) @ v
        return -lambda_val * torch.sum(alignment ** 2)
    name = resolve_energy_name(energy_type, path="linear")
    return similarity_energy(name, x @ v, u, SimilarityContext(x=x, v=v))

def calculate_simlr_gradient(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, 
                             energy_type: str = "regression", lambda_val: float = 0.0, 
                             prior_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate the gradient of the SiMLR energy function for a single view.

    Parameters
    ----------
    v : torch.Tensor
        The view-specific basis matrix (p_i x k).
    x : torch.Tensor
        The view-specific data matrix (n x p_i).
    u : torch.Tensor
        The shared consensus latent matrix (n x k).
    energy_type : str, default="regression"
        The type of energy function used for gradient calculation.
        Options: "regression", "acc", "logcosh", "exp", "gauss", "kurtosis", "dat".
    lambda_val : float, default=0.0
        Regularization parameter for "dat" (Directed Alignment Transfer) energy.
    prior_matrix : torch.Tensor, optional
        Prior alignment matrix for "dat" energy.

    Returns
    -------
    torch.Tensor
        The computed gradient (p_i x k).

    Raises
    ------
    TypeError
        If inputs are not valid tensors.
    """
    from .similarity import (SimilarityContext, resolve_energy_name,
                             similarity_gradient, similarity_needs_data)
    u = u.to(x.dtype); v = v.to(x.dtype)
    if energy_type == "dat":
        if prior_matrix is None:
            raise ValueError(
                "energy_type='dat' is a domain regulariser and requires "
                "prior_matrix."
            )
        prior_matrix = prior_matrix.to(x.dtype)
        return 2 * lambda_val * (prior_matrix.t() @ prior_matrix @ v)

    name = resolve_energy_name(energy_type, path="linear")
    ctx = SimilarityContext(x=x, v=v)
    s_rep = x @ v
    if similarity_needs_data(name):
        # These depend on V directly, not only through s = XV.
        g_v = similarity_gradient(name, s_rep, u, ctx, wrt="v")
    else:
        # Chain rule through the linear encoder: dE/dV = X' dE/ds.
        g_v = x.t() @ similarity_gradient(name, s_rep, u, ctx, wrt="s")
    # Historical convention, preserved: this function returns a *descent*
    # direction (-dE/dV), not the gradient. Callers add it to V.
    return -g_v

def _nonneg_consensus(u):
    """Project the consensus onto the non-negative NSA-Flow set.

    Same operator the basis uses, applied to ``u`` instead of ``V``: the point
    is to make both factors of ``X ~ u V'`` non-negative, which is the premise
    every NMF identifiability result rests on and which SiMLR does not
    currently satisfy. Falls back to the unprojected consensus if the backend
    refuses it, because a failed projection must not silently become a
    different mixing method.
    """
    from .sparsification import _nsa_retract, NSA_DEFAULT_W

    def one(z):
        out = _nsa_retract(z, w=NSA_DEFAULT_W, nonneg=True)
        return z if out is None else out.to(z.dtype)

    return [one(z) for z in u] if isinstance(u, list) else one(u)


def simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
          k: int,
          iterations: int = 100,
          initialization_type: str = "pca",
          init_seed: Optional[int] = None,
          negentropy_weight: float = 0.0,
          nonneg_u: bool = False,
          optimizer_type: str = "armijo_gradient",
          energy_type: str = "acc",
          constraint: str = DEFAULT_CONSTRAINT,
          mixing_algorithm: str = "newton",
          positivity: str = "either",
          smoothing_matrices: Optional[List[torch.Tensor]] = None,
          domain_matrices: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
          domain_lambdas: Optional[Union[float, List[float]]] = None,
          orthogonalize_u: bool = False,
          topology: str = "star",
          path_graph: Optional[Dict[int, List[int]]] = None,
          scale_list: List[str] = ["centerAndScale", "np"],
          consolidate: bool = False,
          nsa_w: Optional[float] = None,
          tol: float = 1e-6,
          verbose: bool = False,
          **opt_params) -> Dict[str, Any]:
    """
    Perform Similarity-driven Multi-view Linear Representation (SiMLR).

    SiMLR identifies a shared latent subspace across multiple data modalities
    by optimizing an energy function subject to manifold constraints and sparsity.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    k : int
        The dimensionality of the shared latent space.
    iterations : int, default=100
        Maximum number of optimization iterations.
    nonneg_u : bool, default=False
        Project the shared consensus onto the non-negative NSA-Flow set each
        sweep. Experimental.

        The motivation is identifiability. NMF uniqueness -- separability,
        the sufficiently-scattered condition, and min-volume as their
        algorithmic surrogate -- requires *both* factors non-negative. SiMLR
        constrains ``V`` but every mixing method returns a variance-
        standardised ``u`` (measured: 56-58% negative entries,
        ``u'u = (n-1)I``), so the factorisation is a *semi*-NMF, for which no
        uniqueness result holds. Making ``u`` non-negative is what would bring
        those results into scope.

        Note this is not free: the consensus is standardised precisely so that
        per-view energies are comparable, and rectifying it changes what the
        mixing method returns. Measure before adopting.
    nsa_w : float, optional
        The NSA-Flow weight, in ``[0, 1]``. Defaults to the weight parsed from
        ``constraint`` (both are 0.35; see `DEFAULT_CONSTRAINT`).

        Its primary job is keeping the basis from collapsing. The gradient
        step drives the columns together; ``w`` holds them apart. Measured on
        BRCA4 (p <= 200, k = 3, 6 seeds, ``consolidate=False``):

        ======  ==================  ===========  ==============
        w       eff. rank (of 3)    cond number  col. overlap
        ======  ==================  ===========  ==============
        0.05    2.291               4.9e+02      0.417
        0.10    2.270               2.8e+02      0.456
        0.20    2.448               6.2e+01      0.421
        0.35    2.826               1.62         0.290
        0.50    2.943               1.27         0.165
        0.90    3.000               1.02         0.013
        0.99    3.000               1.00         0.003
        ======  ==================  ===========  ==============

        Below about 0.2 the fit returns a rank-deficient basis: two of the
        three columns have become collinear, so a caller asking for k = 3
        components receives roughly 2. `simlr` now reports this as
        ``effective_rank`` / ``numerical_rank`` and warns. 0.35 sits at the
        knee and the default is 0.5, chosen for conditioning: prediction is
        flat across the whole range on BRCA4 (0.759 to 0.799), so accuracy
        does not reveal the collapse and only the rank does. Raise it toward
        0.9 for sparser, better-separated columns; lower it only with
        `effective_rank` in the result checked.

        Two things make this easy to mis-measure, and both produced wrong
        conclusions here before the numbers above were taken properly.

        **`consolidate=True` erases it.** The final `consolidate_supports`
        pass forces strictly disjoint supports whatever ``w`` was, so the
        returned basis has column overlap 0.0000 at *every* ``w`` and the
        structural effect is invisible. The benchmark harness passes
        ``consolidate=True``, so any structural claim about ``w`` measured
        through it is a claim about the consolidation step.

        **The iterate oscillates; it does not ratchet.** The gradient step
        between projections fills in *every* zero: measured on BRCA4, the
        fraction of exact zeros entering the projection was 0.000 on every
        sweep at every ``w``, with column overlap arriving at 0.3-1.0. The
        projection restores sparsity from a dense matrix each time, so there
        is no compounding. Applying the projection repeatedly *without* the
        intervening gradient step does drive overlap to zero for any positive
        ``w``, but the algorithm never does that and it is not evidence about
        ``w``. Whether the loop is stable is exactly the rank question above:
        at ``w=0.1`` the incoming overlap climbs 0.54, 0.79, 0.97, 1.00 and
        stays at 1.00 -- the projection can no longer separate the columns.

        Separately from structure, ``w`` throttles the optimiser: energy
        reduction fell from 1.00 at ``w=0.1`` to 0.02 at ``w=0.99`` on
        Diabetes, and runtime rose 4.6x from ``w=0.05`` to ``w=0.99`` on
        BRCA4. Predictive accuracy peaked mid-range (Diabetes, 40 paired
        seeds: ``w`` in [0.30, 0.45] beat 0.25 by +0.027 to +0.069 test R^2,
        Wilcoxon p <= 0.002; BRCA4 showed no resolvable difference at all,
        Friedman p = 0.43). ``w = 0`` is worth avoiding outright -- it gave a
        near rank-deficient basis on BRCA4 (condition number 7.5e28).
    negentropy_weight : float, default=0.0
        Weight ``mu`` on a Hyvarinen negentropy term added to the similarity;
        ``0`` disables it and is the historical behaviour. See
        `similarity.energy_negentropy`.

        This is the term that can identify the support. A reconstruction
        criterion cannot: ``(uT^-1)(VT')'`` reconstructs identically for any
        invertible ``T``, so it fixes only ``span(V)``. Measured on a planted
        disjoint basis over 12 random starts at ``w=0.5``, support recovery
        went 0.6524 +- 0.0260 at ``mu=0`` to 0.8806 +- 0.0901 at ``mu=100``
        and 0.9159 +- 0.1234 at ``mu=300``; the term alone, with no data
        term, scores 0.4313. The useful values are large because the contrast
        itself is small -- O(1e-4) for a half-normal latent against a data
        term of O(1e-2) -- not because the term is being over-weighted.

        Needs `initialization_type="random"` to be meaningful as an
        identifiability claim: under the default PCA start every run begins
        from the same basis.
    initialization_type : str, default="pca"
        One of `INITIALIZATION_TYPES`: ``"pca"`` (deterministic, the
        historical behaviour), ``"random"``, ``"gcca"``, ``"nndsvd"``,
        ``"perturbed_pca"``, ``"domain"`` (needs `domain_matrices`) or
        ``"best_of_n"``. See `initialize_simlr` for what each does and why;
        only ``"pca"`` and ``"random"`` have any measurement behind them here
        -- the rest are untested strategies, not benchmarked replacements.
    init_seed : int, optional
        Seed for a random start. Supplying it selects ``"random"``
        automatically, since a seed handed to the deterministic initialiser
        would silently have no effect. Drawn from a local `torch.Generator`,
        so the global RNG stream is untouched.
    optimizer_type : str, default="armijo_gradient"
        The optimizer to use. This default has changed twice (`lars` ->
        `hybrid_adam` -> `armijo_gradient`) without a controlled ablation ever
        settling which is best. The one comparison on record
        (`CORRECTNESS_AUDIT.md`, 10 seeds, post-bugfix) has `hybrid_adam`,
        `lars` and `armijo_gradient` within noise of each other (0.9822-0.9848
        Procrustes R^2 depending on `positivity`), so `armijo_gradient` is
        chosen for its line search's sufficient-decrease guarantee rather than
        for a measured accuracy advantage -- treat it as a reasonable default,
        not a proven one, until that ablation exists.
    energy_type : str, default="acc"
        The similarity/reconstruction objective to minimize.
    constraint : str, default="Stiefel"
        The manifold constraint on basis matrices ("Stiefel", "Grassmann", "NewtonSchulz").

    mixing_algorithm : str, default="svd"
        The algorithm used to mix projections ("svd", "pca", "ica", "avg", "newton").

    positivity : str, default="either"
        Sign constraint on the basis ("either", "positive", "negative").
    smoothing_matrices : List[torch.Tensor], optional
        Spatial/prior smoothing operators for each modality.
    domain_matrices : List[Union[torch.Tensor, np.ndarray]], optional
        Matrices to align with for directed domain knowledge.
    domain_lambdas : Union[float, List[float]], optional
        Weight(s) for the domain knowledge alignment objective.
    orthogonalize_u : bool, default=False
        Whether to enforce orthogonality on the consensus matrix U.
    topology : str, default="star"
        The consensus topology ("star", "loo", "graph").
    path_graph : Dict[int, List[int]], optional
        Adjacency list for "graph" topology.
    scale_list : List[str], default=["centerAndScale", "np"]
        Preprocessing methods to apply to input data.
    tol : float, default=1e-6
        Convergence tolerance for the optimization.
    verbose : bool, default=False
        Whether to print convergence details.
    **opt_params : dict
        Additional parameters to pass to the optimizer or constraint function.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the fitted model parts:
        - "u": Shared latent consensus (N x K).
        - "v": List of view-specific basis matrices (P_i x K).
        - "w": Reconstruction mapping matrices.
        - "energy": The total energy after each iteration.
        - "converged_iter": The iteration at which convergence occurred.
        - "best_energy": The lowest total energy reached.
        - "best_iteration": Index into "energy" of that lowest value. The
          returned "v" and "u" come from this iterate, because alternating
          minimization is not monotone in the joint objective -- `u` is
          recomputed after each sweep over the views, so the energy typically
          drops steeply and then drifts slowly upward.

    Raises
    ------
    TypeError
        If the inputs are not valid data structures.
    """
    # Budget aliasing, so one caller can drive every method:
    #     for alg in algorithms: evaluate(run(alg, data, budget))
    # `simlr` counts `iterations` and the deep models count `epochs`. They are
    # not the same unit, but they occupy the same slot -- "how long may this
    # run" -- and requiring the caller to know which name each method uses is
    # the one thing that stopped the entry points being interchangeable.
    if "epochs" in opt_params:
        iterations = int(opt_params.pop("epochs"))

    # Removed parameters. Both were accepted, documented and swept, and
    # neither did anything:
    #
    #   sparseness_quantile (and its aliases `sparsity` / `sparseness`)
    #       Sparsity is a consequence of the NSA-Flow weight `w`. The value was
    #       forwarded to `simlr_sparseness`, which has ignored it (with a
    #       DeprecationWarning) since the projection became the single
    #       constraint operator.
    #   use_nsa
    #       Reached `SIMLR_OPTIMIZER_DEFAULTS` and stopped -- no optimizer
    #       reads it -- so `use_nsa=False` returned a bit-identical fit for
    #       every optimizer, including `nsa_flow`. There is nothing here to
    #       switch off: `simlr_sparseness` *is* the projection and has no
    #       non-NSA fallback by design.
    #
    # They are intercepted rather than left to fall into `**opt_params`, where
    # they would reach `create_optimizer` and draw a misleading "unrecognised
    # optimizer parameter" warning instead of naming the real replacement.
    for _gone, _use in (("sparseness_quantile", "nsa_w (w -> 1 gives disjoint supports)"),
                        ("sparsity", "nsa_w"), ("sparseness", "nsa_w"),
                        ("use_nsa", "constraint='none' to fit without a projection")):
        if _gone in opt_params:
            opt_params.pop(_gone)
            warnings.warn(
                f"simlr: {_gone!r} has been removed because it had no effect. "
                f"Use {_use}.",
                DeprecationWarning, stacklevel=2,
            )

    # `learning_rate="auto"` resolves here, before anything is built, so the
    # probes are ordinary `simlr` calls with an explicit float -- the probe is
    # literally the consumer, which is the only way to tune a step size across
    # optimizers whose steps mean different things (LARS displaces
    # `lr * ||V||` regardless of the gradient; Adam ~`lr` per coordinate).
    # `learning_rate` arrives through **opt_params rather than as a named
    # parameter, so its default is set here. "auto" tunes it per problem by
    # running short probes of the real fit; see `optimizers.tune_learning_rate`.
    # The fixed 0.001 it replaces is not a neutral choice: LARS and the
    # line-search optimizers move V by about `lr` of its norm per sweep, so a
    # short fit ended within ~0.4% of its initialisation and no energy,
    # constraint or regulariser could change the answer.
    opt_params.setdefault("learning_rate", "auto")
    if isinstance(opt_params.get("learning_rate", None), str):
        if opt_params["learning_rate"] != "auto":
            raise ValueError(
                f"learning_rate must be a float or 'auto'; got "
                f"{opt_params['learning_rate']!r}."
            )
        # Selected by the signature, not by taking whatever is in locals().
        # The frame also holds loop variables and imports by this point, and
        # both have leaked into the recursive call before: first the imported
        # `tune_learning_rate`, then `_gone` and `_use` from the removed-
        # parameter loop above, each arriving at `create_optimizer` as an
        # "unrecognised optimizer parameter".
        import inspect as _inspect
        _names = {kk for kk, vv in _inspect.signature(simlr).parameters.items()
                  if vv.kind is not vv.VAR_KEYWORD}
        _frame = locals()
        _bound = {kk: _frame[kk] for kk in _names if kk in _frame}
        from .optimizers import tune_learning_rate
        _iters = _bound.pop("iterations")
        _rest = {kk: vv for kk, vv in opt_params.items() if kk != "learning_rate"}

        def _probe(learning_rate, iterations):
            return simlr(**_bound, **_rest, iterations=iterations,
                         learning_rate=learning_rate)

        _lr, _scores = tune_learning_rate(
            _probe, probe_iterations=min(3, max(1, _iters)), verbose=verbose)
        _out = simlr(**_bound, **_rest, iterations=_iters, learning_rate=_lr)
        _out["learning_rate"] = _lr
        _out["learning_rate_scores"] = _scores
        return _out

    # Contract Validation: data_matrices structure
    if not isinstance(data_matrices, (list, tuple)) or len(data_matrices) == 0:
        raise ValueError("data_matrices must be a non-empty list or tuple of matrices/arrays.")

    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")

    torch_mats = []
    for idx, m in enumerate(data_matrices):
        if not isinstance(m, (torch.Tensor, np.ndarray)):
            raise TypeError(f"View {idx} must be a torch.Tensor or np.ndarray, got {type(m).__name__}.")
        m_t = torch.as_tensor(m).float()
        if m_t.ndim != 2:
            raise ValueError(
                f"Each view in data_matrices must be 2D (samples x features). "
                f"View {idx} has {m_t.ndim} dimension(s) with shape {tuple(m_t.shape)}."
            )
        if m_t.shape[0] == 0:
            raise ValueError(f"View {idx} has 0 samples.")
        if m_t.shape[1] == 0:
            raise ValueError(f"View {idx} has 0 features.")
        torch_mats.append(m_t)

    # Contract Validation: sample count consistency across all views
    n_samples_list = [m.shape[0] for m in torch_mats]
    if len(set(n_samples_list)) > 1:
        details = ", ".join(f"view {i}: {n}" for i, n in enumerate(n_samples_list))
        raise ValueError(
            f"All data matrices must have identical number of samples (rows), "
            f"but found mismatched sample counts: {details}."
        )
    
    provenance_list = []
    if scale_list is not None and len(scale_list) > 0 and scale_list[0] != "none":
        scaled_mats = []
        for m in torch_mats:
            m_scaled, prov = preprocess_data(m, scale_list)
            scaled_mats.append(m_scaled)
            provenance_list.append(prov)
        torch_mats = scaled_mats
        
    n_modalities = len(torch_mats)
    orig_dtype = torch_mats[0].dtype
    # `init_seed` implies a random start: passing a seed to a deterministic
    # initialiser and getting the same basis back is the trap this avoids.
    if init_seed is not None and initialization_type == "pca":
        initialization_type = "random"
    init_gen = None
    if initialization_type in ("random", "perturbed_pca", "best_of_n"):
        init_gen = torch.Generator()
        if init_seed is not None:
            init_gen.manual_seed(int(init_seed))
    v_mats = initialize_simlr(torch_mats, k, positivity=positivity,
                              initialization_type=initialization_type,
                              generator=init_gen,
                              domain_matrices=domain_matrices)
    # One weight.  `constraint` sets the prox weight; the "nsa_flow" optimizer's
    # intermediate soft retraction used to default to a DIFFERENT weight (0.5
    # against 0.1), so one call applied two different operators.  Unless the
    # caller separates them explicitly, they agree.  (parse_constraint is pure;
    # the full parse below is unchanged.)
    nsa_w_given = nsa_w is not None
    if nsa_w is None:
        # `parse_constraint` already applies the type-dependent default, so a
        # zero reaching here is one the caller asked for -- `constraint="none"`
        # or an explicit `"orthox0"` -- not an unset value. Substituting
        # NSA_DEFAULT_W made those two requests apply a half-strength
        # constraint instead of none.
        nsa_w = parse_constraint(constraint)["weight"]
    opt_params.setdefault("nsa_w", nsa_w)
    optimizer = create_optimizer(optimizer_type, v_mats, **opt_params)
    
    constraint_info = parse_constraint(constraint)
    constraint_type = constraint_info["type"]
    constraint_weight = constraint_info["weight"]
    constraint_iterations = constraint_info["iterations"]
    if nsa_w_given:
        # `nsa_w` used to reach `opt_params` only -- the optimizer's internal
        # retraction -- while the prox that runs every sweep took its weight
        # from `constraint`. With `optimizer_type="lars"` (the default when
        # this was measured), which has no retraction at all, `nsa_w`
        # therefore did *nothing*: measured
        # across nsa_w in {0.05, 0.3, 0.5, 0.9} the fitted basis was identical
        # to 0.00e+00, which is the "flat w sweep" that made w look like an
        # irrelevant hyperparameter. It is not irrelevant -- the same weight
        # passed through `constraint` moves the projection's zero count from
        # 10 to 75 -- it was simply not being applied.
        # The *default* constraint string carries a weight too, so a caller
        # who passed only `nsa_w` must not be warned about a conflict with a
        # value they never chose.
        explicit_in_string = (len(str(constraint).split('x')) > 1
                              and str(constraint) != DEFAULT_CONSTRAINT)
        if explicit_in_string and abs(float(constraint_weight) - float(nsa_w)) > 1e-12:
            warnings.warn(
                f"nsa_w={nsa_w!r} and constraint={constraint!r} (weight "
                f"{constraint_weight}) specify different NSA-Flow weights. "
                f"nsa_w wins for the projection. Pass only one.",
                RuntimeWarning, stacklevel=2,
            )
        constraint_weight = float(nsa_w)
    
    torch_domains = [torch.as_tensor(dm).float() if dm is not None else None for dm in domain_matrices] if domain_matrices else None
    if domain_lambdas is not None and not isinstance(domain_lambdas, (list, tuple)):
        # Accept any scalar, not just float; `domain_lambdas=1` used to reach
        # `domain_lambdas[i]` and raise "'int' object is not subscriptable".
        domain_lambdas = [float(domain_lambdas)] * n_modalities
    elif isinstance(domain_lambdas, (list, tuple)):
        domain_lambdas = list(domain_lambdas)
        if len(domain_lambdas) != n_modalities:
            raise ValueError(
                f"domain_lambdas has {len(domain_lambdas)} entries but there are "
                f"{n_modalities} views."
            )
    if torch_domains is not None and domain_lambdas is None:
        domain_lambdas = [1.0] * n_modalities
    
    from .nsa_backend import load_gradient_mapping
    _grad_mapping = load_gradient_mapping()
    # The feasible set the certificate projects onto must match the one the
    # iterate actually lives in, or the measure is of a different problem.
    _want_nonneg = positivity in ('positive', 'hard', 'nonnegative', 'nonneg',
                                  'softplus')
    _cert_proj = (lambda z: torch.clamp(z, min=0.0)) if _want_nonneg else None

    energy_history = []
    grad_map_history = []
    n_degenerate_iterates = 0
    stop_reason = "max_iter"
    from .utils import ConvergenceMonitor, CONVERGENCE_PATIENCE
    monitor = ConvergenceMonitor(tol=tol, patience=int(opt_params.pop(
        "patience", CONVERGENCE_PATIENCE)), max_steps=iterations)
    certificate = None
    prev_total_energy = float('inf')
    prev_grad_map = float('inf')
    converged_iter = iterations
    # Alternating minimization is not monotone in the joint objective: each V_i
    # is optimized against a fixed u_i, and then u is recomputed from the new
    # projections, which can raise the total. In practice the energy drops
    # sharply over the first few iterations and then drifts slowly upward, so
    # the final iterate is not the best one found. Keep the best.
    best_total_energy = float('inf')
    best_v_mats = None
    # The solver's self-report for each modality's final projection. Kept
    # alongside the best iterate so the reported diagnostics describe the basis
    # actually returned, not whichever iteration the loop happened to stop on.
    retraction_diags = [{} for _ in range(n_modalities)]
    best_retraction_diags = None
    
    normalizing_weights = [1.0] * n_modalities
    orth_weights = [1.0] * n_modalities
    domain_weights = [1.0] * n_modalities
    # Calibrated at it=0 like `orth_weights`, so `negentropy_weight` is a
    # *relative* weight: mu=1 means "as important as the data term on this
    # dataset". Uncalibrated it is not comparable between datasets and barely
    # between views -- the contrast is O(1e-4) for a half-normal latent while
    # `recon_r2` is O(1e-2), so the raw mu that does anything is in the
    # hundreds, and the same raw mu that helps one view swamps another.
    neg_weights = [1.0] * n_modalities
    
    prev_resid = [None] * n_modalities   # last prox-gradient fixed-point residual per view
    for it in range(iterations):
        cert_terms = []
        projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
        u = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u, topology=topology, path_graph=path_graph)
        if nonneg_u:
            u = _nonneg_consensus(u)
        # Capture the linear map this consensus used, so that
        # `predict_shared_latent` can apply it to new data instead of
        # deriving a fresh one. See `_capture_consensus_anchor`.
        consensus_anchor = _capture_consensus_anchor(
            projections, mixing_algorithm, k, orthogonalize_u, topology, path_graph)
        for i in range(n_modalities):
            u_i = u[i] if isinstance(u, list) else u
            # Local energy function that incorporates sparsification/retraction
            # Single definition of "the feasible point corresponding to v", so
            # the energy and the gradient are evaluated at the *same* place.
            # Previously the energy was measured after sparsification/retraction
            # while the gradient was taken at the raw iterate, which left every
            # Armijo/backtracking optimizer testing a sufficient-decrease
            # condition against the slope of a different function.
            def to_feasible(v_cand):
                return simlr_sparseness(
                    v_cand.to(orig_dtype), constraint_type=constraint_type,
                    smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None,
                    positivity=positivity,
                    constraint_weight=constraint_weight, constraint_iterations=constraint_iterations,
                    energy_type=energy_type, modality_index=i)

            def smooth_energy_fn(v_cand):
                v_sp = to_feasible(v_cand)
                sim_e = calculate_simlr_energy(v_sp, torch_mats[i], u_i, energy_type) * normalizing_weights[i]
                dom_e = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_e = calculate_simlr_energy(v_sp, torch_mats[i], u_i, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]) * domain_weights[i]
                orth_e = 0.0
                if constraint_type == "ortho": orth_e = orthogonality_defect(v_sp) * constraint_weight * orth_weights[i]
                neg_e = 0.0
                if negentropy_weight != 0.0:
                    neg_e = (calculate_simlr_energy(v_sp, torch_mats[i], u_i, "negentropy")
                             * negentropy_weight * neg_weights[i])
                return (sim_e + dom_e + orth_e + neg_e).item()

            # Local gradient function that also incorporates manifold projection
            raw_grad_holder = [None]

            def smooth_gradient_fn(v_curr):
                # Evaluate at the feasible point the energy also uses.
                v_feas = to_feasible(v_curr)

                sim_grad = calculate_simlr_gradient(v_feas, torch_mats[i], u_i, energy_type) * normalizing_weights[i]
                # Kept in step with `smooth_energy_fn`: a term present in one
                # and absent from the other makes every line search test a
                # sufficient-decrease condition against the wrong slope.
                if negentropy_weight != 0.0:
                    sim_grad = sim_grad + calculate_simlr_gradient(
                        v_feas, torch_mats[i], u_i, "negentropy"
                    ) * negentropy_weight * neg_weights[i]
                dom_grad = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_grad = calculate_simlr_gradient(v_feas, torch_mats[i], u_i, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]) * domain_weights[i]
                
                # The energy `smooth_energy_fn` also carries an orthogonality
                # term, so omitting it here made energy and gradient describe
                # two different functions. Any line search or curvature
                # estimate built from the pair was then inconsistent, and the
                # stationarity certificate certified the wrong objective. The
                # mismatch was survivable only while the fidelity gradient was
                # itself ~(n-1) times too small; correcting that (see the
                # `regression` branch of `calculate_simlr_gradient`) made the
                # inconsistency dominate and drove `energy_reduction` to zero.
                orth_grad = 0.0
                if constraint_type == "ortho" and constraint_weight != 0.0:
                    orth_grad = (gradient_orthogonality_defect(v_feas)
                                 * constraint_weight * orth_weights[i])

                total_grad = sim_grad + dom_grad - orth_grad
                # Keep the analytic gradient before it is projected onto the
                # tangent space and retracted. The retraction orthonormalises
                # the direction and so throws its magnitude away, which is
                # correct for a search direction and useless for a certificate:
                # scoring the retracted direction returns a constant (3.0000 on
                # the 3-view case, independent of how far from stationary the
                # iterate is). The certificate has to see the real gradient.
                raw_grad_holder[0] = total_grad.detach().clone()
                # Project gradient onto tangent space
                total_grad = project_gradient(total_grad, v_feas, constraint_type)
                
                # The search direction is NOT retracted.  Proximal gradient is
                # `prox(v + eta * g)`: one operator, applied to the point, after
                # the step.  The direction used to be passed through
                # simlr_sparseness too (orthonormalising it and discarding its
                # magnitude); CORRECTNESS_AUDIT.md measured that as changing
                # latent recovery by < 0.0004, i.e. an extra solve per view per
                # iteration that bought nothing, and it is not part of any
                # proximal-gradient method.
                return total_grad.contiguous()

            total_grad = smooth_gradient_fn(v_mats[i])
            # Stationarity certificate at the iterate entering this sweep,
            # using the gradient that was going to be computed anyway. See
            # `nsa_backend.load_gradient_mapping`: energy plateau is not
            # stationarity, and this loop previously claimed convergence with
            # nothing measuring the latter.
            # Stationarity certificate for THE PROBLEM BEING SOLVED.  This loop
            # is proximal gradient on E + w*Dtilde over the feasible set, whose
            # fixed points satisfy v = prox(v + eta*d(v)).  The right residual is
            # therefore the relative fixed-point residual ||v+ - v|| / ||v||,
            # computed after the step and prox below.  The gradient mapping of
            # E alone under the sign projection -- what used to be certified --
            # is NOT zero at that fixed point (it settled at 6.2 on a problem
            # the iteration had converged on) and is kept only as a diagnostic.
            gm_E = None
            if _grad_mapping is not None and raw_grad_holder[0] is not None:
                try:
                    gm_E = float(_grad_mapping(v_mats[i], raw_grad_holder[0],
                                               _cert_proj))
                except Exception:
                    gm_E = None
            gm_i = prev_resid[i]
            # A view certified stationary takes no step and is not re-projected.
            # Proximal gradient has nothing to do at a stationary point, and
            # doing it anyway is not free: the prox is solved to `tol`, so
            # re-applying it to its own output moves the basis by O(tol) per
            # sweep -- measured as a 5e-5 drift in the overall energy over 12
            # sweeps from an already-optimal initialisation, which the
            # first-vs-last monotonicity tests caught.  The first sweep always
            # projects, so a raw initialisation never bypasses the constraint.
            if gm_i is not None and gm_i <= tol and it > 0:
                cert_terms.append(gm_i)
                retraction_diags[i] = {"skipped": "stationary", "fixed_point_residual": gm_i,
                                       "grad_map_E": gm_E}
                continue
            v_before = v_mats[i].detach().clone()
            # Optimizers that re-evaluate along their own line search (LBFGS)
            # need to recompute the analytic gradient at each trial point.
            if hasattr(optimizer, "gradient_function"):
                optimizer.gradient_function = smooth_gradient_fn
            v_updated = optimizer.step(i, v_mats[i], total_grad, smooth_energy_fn)
            
            # Apply final projection
            retraction_diags[i] = {}
            v_mats[i] = simlr_sparseness(v_updated, constraint_type=constraint_type, smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None, positivity=positivity, constraint_weight=constraint_weight, constraint_iterations=constraint_iterations, energy_type=energy_type, modality_index=i, retraction_diagnostics=retraction_diags[i])
            resid = float(torch.linalg.norm(v_mats[i] - v_before)
                          / torch.linalg.norm(v_before).clamp_min(1e-300))
            prev_resid[i] = resid
            cert_terms.append(resid)
            retraction_diags[i]["fixed_point_residual"] = resid
            retraction_diags[i]["grad_map_E"] = gm_E
            
        if it == 0:
            from .similarity import similarity_is_self_normalised
            try:
                self_norm = similarity_is_self_normalised(energy_type)
            except (ValueError, KeyError):
                self_norm = False
            for i in range(n_modalities):
                u_i = u[i] if isinstance(u, list) else u
                sim_e = calculate_simlr_energy(v_mats[i], torch_mats[i], u_i, energy_type).item()
                # A self-normalised term is already on an absolute per-view
                # scale, so the views are averaged and the total stays in the
                # term's own range ([0, 1] for `recon_r2`). Dividing by each
                # view's initial energy, as the general path does, would make
                # the total 1.0 at iteration 0 by construction and relative
                # thereafter -- fine for an arbitrarily-scaled term, but it
                # would throw away an interpretable one.
                normalizing_weights[i] = (1.0 / n_modalities if self_norm
                                          else 1.0 / (abs(sim_e) * n_modalities + 1e-10))
                orth_e = orthogonality_defect(v_mats[i]).item()
                if orth_e > 1e-10: orth_weights[i] = abs(sim_e) * normalizing_weights[i] / orth_e
                else: orth_weights[i] = 0.0
                if negentropy_weight != 0.0:
                    neg_e0 = abs(calculate_simlr_energy(
                        v_mats[i], torch_mats[i], u_i, "negentropy").item())
                    neg_weights[i] = (abs(sim_e) * normalizing_weights[i] / neg_e0
                                      if neg_e0 > 1e-12 else 0.0)
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_e_raw = calculate_simlr_energy(v_mats[i], torch_mats[i], u_i, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]).item()
                    if abs(dom_e_raw) > 1e-10: domain_weights[i] = abs(sim_e * normalizing_weights[i]) / abs(dom_e_raw)
                    else: domain_weights[i] = 1.0
        
        total_energy = 0.0
        for i in range(n_modalities):
            u_i = u[i] if isinstance(u, list) else u
            total_energy += calculate_simlr_energy(v_mats[i], torch_mats[i], u_i, energy_type).item() * normalizing_weights[i]
            
        energy_history.append(total_energy)

        # A modality whose basis is identically zero contributes zero energy,
        # so the cheapest way to reduce a *sum* of per-view energies is to kill
        # a view. Selecting the lowest-energy iterate therefore selects the
        # degenerate one: on the 3-view case the loop returned V with view 0
        # entirely zero at total energy 2/3 (exactly one of three views
        # surviving), and reported it as `best_iteration=1`.
        #
        # A rank-deficient basis is not a solution to a rank-k problem, so such
        # an iterate is not eligible to be "best" however low its energy.
        iterate_is_degenerate = any(
            not bool((v.detach().abs() > 1e-12).any(dim=0).all()) for v in v_mats
        )
        if total_energy < best_total_energy and not iterate_is_degenerate:
            best_total_energy = total_energy
            best_v_mats = [v.clone() for v in v_mats]
            best_retraction_diags = [dict(d) for d in retraction_diags]
        if iterate_is_degenerate:
            n_degenerate_iterates += 1
        
        # Stopping. An energy plateau is not stationarity: the total energy is
        # normalised to 1.0 at iteration 0, so a loop that is barely descending
        # produces relative changes below `tol` within two or three sweeps and
        # the old test then declared convergence. Measured on the 3-view case
        # it fired at iteration 3 for every problem and every `iterations`,
        # so the argument never bound; with the test disabled one seed carried
        # on improving from R^2 0.300 to 0.366.
        #
        # `grad_map` is the certificate: dimensionless, scale invariant, and
        # zero exactly at a stationary point of the constrained problem. Only
        # it may claim convergence. A plateau still stops the loop -- there is
        # no point burning sweeps that do nothing -- but it is certified only
        # when the certificate is also within a wide band of stationarity,
        # following nsa_flow's `CERTIFICATES` contract.
        this_grad_map = max(cert_terms) if cert_terms else float("nan")
        grad_map_history.append(this_grad_map)

        # The plateau test is now `utils.ConvergenceMonitor`, the same rule the
        # deep trainer uses, so `tol` and `patience` mean one thing across the
        # package. The bespoke test it replaces compared consecutive energies
        # against `tol` and fired within two or three sweeps on a normalised
        # energy. The certificate below is unchanged and is still the only
        # thing allowed to claim *stationarity*, which is strictly stronger
        # than "stopped improving".
        plateaued = monitor.update(total_energy)

        if cert_terms and monitor.certify("grad_map", this_grad_map, tol):
            stop_reason, certificate = "grad_map", "stationary"
            converged_iter = it + 1
            if verbose: print(f"Stationary at iteration {it}: grad_map {this_grad_map:.3e}")
            break

        if plateaued:
            # `monitor.update` returns True both for a real plateau and for
            # the budget running out. Only the first is a plateau; calling an
            # exhausted budget one claims the fit finished when it was cut off.
            if monitor.stop_reason != "converged":
                break
            stop_reason = "plateau"
            # Weaker than stationary, and only claimed near it.
            certificate = ("numerical_floor"
                           if cert_terms and this_grad_map <= 1e3 * tol else None)
            converged_iter = it + 1
            if verbose:
                print(f"Plateau at iteration {it}: energy {total_energy}, "
                      f"grad_map {this_grad_map:.3e}, certificate {certificate}")
            break

        prev_total_energy = total_energy
        prev_grad_map = this_grad_map
        
        if verbose and it % 10 == 0: print(f"Iteration {it}: Total Energy {total_energy}")
        
    # Return the lowest-energy iterate rather than whichever one the loop
    # happened to stop on.
    if best_v_mats is not None:
        v_mats = best_v_mats
        if best_retraction_diags is not None:
            retraction_diags = best_retraction_diags

    if consolidate:
        from .nsa_backend import load_consolidate_supports
        cons_fn = load_consolidate_supports()
        if cons_fn is not None:
            consolidated_v = []
            for v_mat in v_mats:
                if positivity == "positive":
                    v_c = cons_fn(v_mat.double()).to(orig_dtype)
                    v_c = torch.nn.functional.normalize(v_c, p=2, dim=0, eps=1e-8)
                    consolidated_v.append(v_c)
                else:
                    k_v = v_mat.shape[1]
                    v_p = torch.clamp_min(v_mat, 0.0)
                    v_n = torch.clamp_min(-v_mat, 0.0)
                    W = torch.cat([v_p, v_n], dim=1)
                    W_c = cons_fn(W.double()).to(orig_dtype)
                    v_c = W_c[:, :k_v] - W_c[:, k_v:]
                    v_c = torch.nn.functional.normalize(v_c, p=2, dim=0, eps=1e-8)
                    consolidated_v.append(v_c)
            v_mats = consolidated_v

    # Re-calculate final shared consensus after the last V update
    projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
    u = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u, topology=topology, path_graph=path_graph)
    # Capture the linear map this consensus used, so that
    # `predict_shared_latent` can apply it to new data instead of
    # deriving a fresh one. See `_capture_consensus_anchor`.
    consensus_anchor = _capture_consensus_anchor(
        projections, mixing_algorithm, k, orthogonalize_u, topology, path_graph)
    
    v_summaries = [orthogonality_summary(v) for v in v_mats]
    
    # Compute reconstruction weights W_i such that X_i approx U @ W_i
    # W_i = pinv(U) @ X_i
    w_mats = []
    for i, x in enumerate(torch_mats):
        u_i = u[i] if isinstance(u, list) else u
        try:
            u_pinv = torch.linalg.pinv(u_i)
            w_mats.append(u_pinv @ x)
        except Exception:
            # Fallback if pinv fails
            w_mats.append(torch.zeros(k, x.shape[1], dtype=u_i.dtype, device=u_i.device))

    return {
        "u": u, "v": v_mats, "w": w_mats, "energy": energy_history, 
        "normalizing_weights": normalizing_weights, 
        "orth_weights": orth_weights, "domain_weights": domain_weights, 
        "converged_iter": converged_iter,
        # The stopping claim, and what backs it. `converged` is True only with
        # a certificate; `stop_reason` says why iteration ceased either way,
        # and `energy_reduction` answers "did this optimise anything at all",
        # which a bare `converged_iter` never could.
        "stop_reason": stop_reason,
        # The uniform report; see `utils.ConvergenceMonitor`. `stop_reason`
        # above keeps its own vocabulary for compatibility.
        "convergence": {**monitor.report(), "stop_reason": stop_reason},
        "certificate": certificate,
        "converged": certificate is not None,
        "grad_map": (grad_map_history[-1] if grad_map_history else None),
        "grad_map_history": grad_map_history,
        # How many sweeps produced a rank-deficient basis. Nonzero means the
        # objective is rewarding collapse and the result should be read with
        # that in mind, even though such iterates can no longer be returned.
        "n_degenerate_iterates": n_degenerate_iterates,
        "energy_start": (energy_history[0] if energy_history else None),
        "energy_reduction": ((energy_history[0] - min(energy_history))
                             if energy_history else None),
        "v_orthogonality": v_summaries,
        "best_energy": best_total_energy,
        "best_iteration": (int(np.argmin(energy_history)) if energy_history else None),
        # What the retraction solver reported for each modality: the stopping
        # rule, the stationarity certificate, the effective rank, the scale
        # drift and which fidelity it chose. Empty when no backend ran.
        "v_retraction": retraction_diags,
        # The consensus map fitted on the training projections. Required for
        # out-of-sample prediction: without it `predict_shared_latent`
        # re-derives the basis on the new data, and for svd/pca/ica that basis
        # is defined only up to rotation, sign and permutation.
        "consensus_anchor": consensus_anchor,
        "mixing_algorithm": mixing_algorithm,
        "orthogonalize_u": orthogonalize_u,
        "topology": topology,
        "path_graph": path_graph,
        "energy_type": energy_type,
        "scale_list": scale_list,
        "provenance_list": provenance_list,
        "consolidated": consolidate,
        # How many components actually came back. See `basis_rank_report`:
        # a weak constraint lets the columns merge, and nothing else in this
        # dict would show it.
        **basis_rank_report(v_mats, k),
    }


def pairwise_matrix_similarity(mat_list: List[torch.Tensor], v_list: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute pairwise similarity (Adjusted RV Coefficient) between all pairs of modalities.

    Parameters
    ----------
    mat_list : List[torch.Tensor]
        List of data matrices (N x P_i).
    v_list : List[torch.Tensor]
        List of basis matrices (P_i x K).

    Returns
    -------
    Dict[str, float]
        Dictionary where keys are "sim_i_j" and values are the similarity scores.

    Raises
    ------
    TypeError
        If inputs are not valid lists of tensors.
    """
    n_modalities = len(mat_list); similarities = {}
    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            l_i = mat_list[i] @ v_list[i]; l_j = mat_list[j] @ v_list[j]
            similarities[f"sim_{i}_{j}"] = adjusted_rvcoef(l_i, l_j)
    return similarities

def simlr_perm(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, n_perms: int = 50, verbose: bool = False, **simlr_params) -> Dict[str, Any]:
    """
    Perform permutation testing for SiMLR to assess the significance of shared latent structures.

    This function runs the SiMLR algorithm on the original data and then repeatedly on 
    permuted versions of the data (where rows of each modality are independently shuffled)
    to build a null distribution of the cross-modality similarity (ACC).

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        A list of data matrices (one for each modality).
    k : int
        The number of shared latent components to extract.
    n_perms : int, optional
        The number of permutations to perform (default is 50).
    verbose : bool, optional
        Whether to print progress during the SiMLR optimization (default is False).
    **simlr_params : dict
        Additional parameters passed to the `simlr` function (e.g., energy_type, constraint, optimizer_type).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "simlr_result": The result of SiMLR on the original data.
        - "stats": Per-modality-pair permutation statistics, keyed "sim_i_j",
          each a dict with "observed", "p_value", "n_permutations",
          "n_exceeding", "null_mean", "null_sd", "z_score" and
          "null_distribution".
        - "n_permutations": The requested number of permutations.

    Notes
    -----
    The p-value is the add-one permutation estimate
    ``(1 + #{null >= observed}) / (1 + n_perms)`` (Phipson & Smyth, 2010).

    Earlier revisions reported ``scipy.stats.ttest_1samp(null, observed,
    alternative='less')``, which tests whether the *mean* of the null lies below
    the observed value rather than how far into the null's upper tail the
    observation falls. That statistic assumes a normally distributed null,
    ignores the tail entirely, and shrinks without bound as `n_perms` grows, so
    it reported arbitrarily small p-values for data with no shared structure.
    The `t_stat` key it produced has been replaced by `z_score`, a plain
    standardized effect size that makes no distributional claim.

    Raises
    ------
    TypeError
        If the inputs are of an invalid type.
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]

    def _fit_and_score(mats):
        """Fit SiMLR and score pairwise similarity on the same matrices the fit saw."""
        fit = simlr(mats, k=k, verbose=False, **simlr_params)
        # The basis V is learned on preprocessed data, so the similarity must be
        # evaluated on preprocessed data too. Scoring raw matrices against a V
        # fitted on centered/scaled ones mixes two different feature scalings.
        scale_list = fit.get('scale_list') or []
        prov_list = fit.get('provenance_list') or []
        if scale_list and scale_list[0] != "none" and len(prov_list) == len(mats):
            scored = [preprocess_data(m, scale_list, provenance=pv)
                      for m, pv in zip(mats, prov_list)]
        else:
            scored = mats
        v_norm = [l1_normalize_features(v) for v in fit['v']]
        return fit, pairwise_matrix_similarity(scored, v_norm)

    res, obs_sims = _fit_and_score(torch_mats)
    if verbose:
        print(f"Observed pairwise similarities: {obs_sims}")

    null_results = {key: [] for key in obs_sims}
    for p in range(n_perms):
        if verbose and p % 10 == 0:
            print(f"Permutation {p}/{n_perms}")
        # Independently shuffling rows of every view destroys cross-view
        # correspondence while preserving each view's marginal distribution.
        mats_perm = [m[torch.randperm(m.shape[0])] for m in torch_mats]
        _, sims_p = _fit_and_score(mats_perm)
        for k_sim, v_sim in sims_p.items():
            null_results[k_sim].append(v_sim)

    stats = {}
    for k_sim, obs in obs_sims.items():
        null_dist = np.asarray(null_results[k_sim], dtype=float)
        null_dist = null_dist[np.isfinite(null_dist)]
        n_eff = int(null_dist.size)
        if n_eff == 0:
            stats[k_sim] = {"observed": obs, "p_value": float('nan'),
                            "n_permutations": 0, "null_mean": float('nan'),
                            "null_sd": float('nan'), "z_score": float('nan'),
                            "null_distribution": []}
            continue
        # Add-one (Phipson & Smyth 2010) permutation p-value: the observed
        # statistic is itself one realisation under the null, so the estimate is
        # (1 + #{null >= obs}) / (1 + n_perm). This is bounded below by
        # 1/(1 + n_perm) and never returns an impossible p of exactly 0.
        n_exceed = int(np.sum(null_dist >= obs))
        p_val = (1.0 + n_exceed) / (1.0 + n_eff)
        null_mean = float(null_dist.mean())
        null_sd = float(null_dist.std(ddof=1)) if n_eff > 1 else 0.0
        if null_sd > 0.0:
            z_score = (float(obs) - null_mean) / null_sd
        else:
            z_score = float('inf') if float(obs) > null_mean else 0.0
        stats[k_sim] = {
            "observed": float(obs),
            "p_value": float(p_val),
            "n_permutations": n_eff,
            "n_exceeding": n_exceed,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "z_score": float(z_score),
            "null_distribution": null_dist.tolist(),
        }
    return {"simlr_result": res, "stats": stats, "n_permutations": n_perms}

def _capture_consensus_anchor(projections: List[torch.Tensor],
                              mixing_algorithm: str,
                              k: int,
                              orthogonalize_u: bool,
                              topology: str,
                              path_graph: Optional[Dict[int, List[int]]]) -> Optional[torch.Tensor]:
    """
    Return the linear map the consensus used, or None if it has none.

    `compute_shared_consensus` returns ``(u, anchor)`` when asked in training
    mode. The anchor takes the concatenated projections to the shared latent,
    and it is what makes the consensus reproducible on data it was not fitted
    on.

    Only ``svd``, ``pca`` and ``ica`` have such a map, and only those need one:
    they derive a basis whose rotation, column signs and ordering are
    arbitrary, so re-deriving it on a new sample yields axes that need not
    correspond to the ones a downstream model was fitted against. ``avg`` and
    ``newton`` are fixed functions of their input and return None here.

    Returns None rather than raising if no anchor is available, since the fit
    itself does not depend on one.
    """
    try:
        result = compute_shared_consensus(
            projections, mixing_algorithm=mixing_algorithm, k=k,
            orthogonalize=orthogonalize_u, training=True,
            topology=topology, path_graph=path_graph)
    except Exception:
        return None
    if not isinstance(result, tuple) or len(result) != 2:
        return None
    anchor = result[1]
    return anchor.detach().clone() if isinstance(anchor, torch.Tensor) else None


def predict_shared_latent(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                          simlr_result: Dict[str, Any]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Compute the shared latent basis U for new data using the trained SIMLR model.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of new data matrices (one for each modality).
    simlr_result : Dict[str, Any]
        The result dictionary from a previous `simlr` call.

    Returns
    -------
    Union[torch.Tensor, List[torch.Tensor]]
        The shared consensus latent matrix (N x K) or list of matrices for the new data.

    Raises
    ------
    TypeError
        If inputs are of invalid type.
    """
    # 1. Contract Validation
    if not isinstance(data_matrices, (list, tuple)) or len(data_matrices) == 0:
        raise ValueError("data_matrices must be a non-empty list or tuple of matrices/arrays.")

    v_mats = simlr_result.get('v')
    if v_mats is not None and len(data_matrices) != len(v_mats):
        raise ValueError(
            f"Expected {len(v_mats)} data matrices matching model views, "
            f"but received {len(data_matrices)}."
        )

    for idx, m in enumerate(data_matrices):
        m_t = torch.as_tensor(m)
        if m_t.ndim != 2:
            raise ValueError(
                f"Each view in data_matrices must be 2D (samples x features). "
                f"View {idx} has {m_t.ndim} dimension(s) with shape {tuple(m_t.shape)}."
            )

    n_samples_list = [torch.as_tensor(m).shape[0] for m in data_matrices]
    if len(set(n_samples_list)) > 1:
        details = ", ".join(f"view {i}: {n}" for i, n in enumerate(n_samples_list))
        raise ValueError(
            f"All data matrices must have identical number of samples (rows), "
            f"but found mismatched sample counts: {details}."
        )

    # 2. Preprocess data matrices
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    scale_list = simlr_result.get('scale_list', [])
    provenance_list = simlr_result.get('provenance_list', [])
    if scale_list and scale_list[0] != "none":
        scaled_mats = []
        for i, m in enumerate(torch_mats):
            prov = provenance_list[i] if i < len(provenance_list) else None
            m_scaled = preprocess_data(m, scale_list, provenance=prov)
            scaled_mats.append(m_scaled)
        torch_mats = scaled_mats

    v_mats = simlr_result['v']
    mixing_alg = simlr_result.get('mixing_algorithm', 'svd')
    orthogonalize_u = simlr_result.get('orthogonalize_u', False)
    topology = simlr_result.get('topology', 'star')
    path_graph = simlr_result.get('path_graph', None)
    k = v_mats[0].shape[1]
    
    # 2. Project to shared space
    projections = [x @ v.to(x.dtype) for x, v in zip(torch_mats, v_mats)]
    
    # 3. Compute consensus U using the original mixing settings
    # Apply the consensus map fitted at training time rather than deriving a
    # fresh one. For svd/pca/ica the derived basis is arbitrary up to rotation,
    # sign and permutation, so re-deriving it here placed held-out samples in
    # different axes than the training latent: on Diabetes a regression fitted
    # on the training U scored -0.67 out of sample, against +0.40 through the
    # fixed map X @ V. `compute_shared_consensus` reuses the anchor only when
    # `training=False` and the algorithm is one of the volatile ones.
    anchor = simlr_result.get('consensus_anchor')
    u_new = compute_shared_consensus(projections, mixing_algorithm=mixing_alg, k=k, orthogonalize=orthogonalize_u, training=False, anchor=anchor, topology=topology, path_graph=path_graph)
    return u_new

def reconstruct_from_learned_maps(u: Union[torch.Tensor, List[torch.Tensor]], 
                                  simlr_result: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Reconstruct all data matrices (modalities) from the shared latent basis U.

    Parameters
    ----------
    u : Union[torch.Tensor, List[torch.Tensor]]
        The shared latent matrix (N x K) or list of matrices.
    simlr_result : Dict[str, Any]
        The result dictionary from a previous `simlr` call, which must 
        contain the learned reconstruction weights "w".

    Returns
    -------
    List[torch.Tensor]
        List of reconstructed data matrices (one for each modality).

    Raises
    ------
    TypeError
        If inputs are of invalid type.
    """
    if 'w' not in simlr_result:
        # For backward compatibility, but this should be avoided
        return []
    
    w_mats = simlr_result['w']
    reconstructions = []
    for i, w in enumerate(w_mats):
        u_i = u[i] if isinstance(u, list) else u
        x_pred = u_i @ w.to(u_i.dtype)
        reconstructions.append(x_pred)
    return reconstructions

def predict_simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                  simlr_result: Dict[str, Any],
                  allow_legacy_refit: bool = False) -> Dict[str, Any]:
    """
    Predict using a trained SiMLR model on new data matrices.

    Generates the shared latent representation `U` and, optionally, the 
    reconstructed data inputs from the new modalities based on the learned 
    model mappings.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of new data matrices to predict on.
    simlr_result : Dict[str, Any]
        The output dictionary from a previous `simlr` fit.
    allow_legacy_refit : bool, default=False
        Whether to allow least-squares estimation of reconstructions 
        if learned weights 'w' are missing.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "u": Predicted shared latent representation.
        - "reconstructions": Reconstructed input matrices.
        - "errors": Normalized Frobenius reconstruction error per modality.

    Raises
    ------
    ValueError
        If learned weights are missing and legacy refit is not allowed.
    TypeError
        If the inputs are not valid formats.
    """
    # 1. Contract Validation
    if not isinstance(data_matrices, (list, tuple)) or len(data_matrices) == 0:
        raise ValueError("data_matrices must be a non-empty list or tuple of matrices/arrays.")

    v_mats = simlr_result.get('v')
    if v_mats is not None and len(data_matrices) != len(v_mats):
        raise ValueError(
            f"Expected {len(v_mats)} data matrices matching model views, "
            f"but received {len(data_matrices)}."
        )

    for idx, m in enumerate(data_matrices):
        m_t = torch.as_tensor(m)
        if m_t.ndim != 2:
            raise ValueError(
                f"Each view in data_matrices must be 2D (samples x features). "
                f"View {idx} has {m_t.ndim} dimension(s) with shape {tuple(m_t.shape)}."
            )

    n_samples_list = [torch.as_tensor(m).shape[0] for m in data_matrices]
    if len(set(n_samples_list)) > 1:
        details = ", ".join(f"view {i}: {n}" for i, n in enumerate(n_samples_list))
        raise ValueError(
            f"All data matrices must have identical number of samples (rows), "
            f"but found mismatched sample counts: {details}."
        )

    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    
    scale_list = simlr_result.get('scale_list', [])
    provenance_list = simlr_result.get('provenance_list', [])
    if scale_list and scale_list[0] != "none":
        scaled_mats = []
        for i, m in enumerate(torch_mats):
            prov = provenance_list[i] if i < len(provenance_list) else None
            m_scaled = preprocess_data(m, scale_list, provenance=prov)
            scaled_mats.append(m_scaled)
        torch_mats = scaled_mats

    if 'model' in simlr_result:
        # Fallback to predict_deep logic if a deep model is detected
        from .deep import predict_deep
        return predict_deep(data_matrices, simlr_result)
    
    # For standard SIMLR:
    u_new = predict_shared_latent(data_matrices, simlr_result)
    
    if 'w' in simlr_result:
        reconstructions = reconstruct_from_learned_maps(u_new, simlr_result)
    elif allow_legacy_refit:
        # Fallback to old suspicious least-squares behavior for backward compatibility
        reconstructions = []
        v_mats = simlr_result['v']
        mixing_alg = simlr_result.get('mixing_algorithm', 'svd')
        orthogonalize_u = simlr_result.get('orthogonalize_u', False)
        for i, x in enumerate(torch_mats):
            u_i = u_new[i] if isinstance(u_new, list) else u_new
            u_ortho = orthogonalize_u or (mixing_alg in ["svd", "pca"])
            if u_ortho:
                weights = u_i.t() @ x
                x_pred = u_i @ weights
            else:
                u_pinv = torch.linalg.pinv(u_i)
                weights = u_pinv @ x
                x_pred = u_i @ weights
            reconstructions.append(x_pred)
    else:
        raise ValueError("Learned reconstruction weights 'w' missing from simlr_result. "
                         "Legacy refit is disabled by default. Set allow_legacy_refit=True to enable.")
    
    errors = []
    for i, x in enumerate(torch_mats):
        x_pred = reconstructions[i]
        err = torch.norm(x - x_pred, p='fro').item() / (torch.norm(x, p='fro').item() + 1e-10)
        errors.append(err)
        
    latents = [x @ v.to(x.dtype) for x, v in zip(torch_mats, simlr_result.get("v", []))]
    return {"u": u_new, "latents": latents, "reconstructions": reconstructions, "errors": errors}


def estimate_rank(data_matrices: List[Union[torch.Tensor, np.ndarray]], n_permutations: int = 20, var_threshold: float = 0.99) -> int:
    """
    Estimate the optimal shared rank `k` across multiple data modalities.

    Uses a heuristic approach relying on the singular value spectrum and 
    cross-modality alignment (RV coefficient) to suggest the number of 
    shared latent components, potentially augmented by permutation testing.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    n_permutations : int, default=20
        Number of random permutations for building a null distribution. 
        If 0, skips permutation testing and uses a fast heuristic.
    var_threshold : float, default=0.99
        Cumulative variance threshold to bound the maximum searched rank.

    Returns
    -------
    int
        The estimated optimal rank `k`.

    Raises
    ------
    TypeError
        If input types are invalid.
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    n_modalities = len(torch_mats); k_max_list = []
    for x in torch_mats:
        x_centered = x - torch.mean(x, dim=0); _, s, _ = safe_svd(x_centered, full_matrices=False)
        eigenvalues = s**2; prop_var = torch.cumsum(eigenvalues, dim=0) / (torch.sum(eigenvalues) + 1e-10)
        reached = torch.where(prop_var >= var_threshold)[0]
        # A degenerate view (near-zero total variance) never crosses the
        # threshold because of the 1e-10 floor in the denominator, and indexing
        # the empty result raised IndexError. Fall back to the full spectrum.
        k_max_list.append(int(reached[0].item()) + 1 if reached.numel() else int(s.numel()))
    k_max = min(k_max_list) if k_max_list else 1
    if k_max < 1: k_max = 1
    def calculate_rv_curve(mats, km):
        u_list = [safe_svd(m, full_matrices=False)[0][:, :km] for m in mats]
        scores = []
        for curr_k in range(1, km + 1):
            mod_scores = []
            for i in range(n_modalities):
                y_target = u_list[i][:, :curr_k]; other_inds = [j for j in range(n_modalities) if j != i]
                u_other = torch.cat([u_list[j][:, :curr_k] for j in other_inds], dim=1)
                consensus, _, _ = safe_svd(u_other, full_matrices=False)
                consensus = consensus[:, :curr_k]; mod_scores.append(adjusted_rvcoef(y_target, consensus))
            scores.append(np.mean(mod_scores))
        return scores
    proc_mats = [(m - torch.mean(m, dim=0)) / (torch.norm(m - torch.mean(m, dim=0), p='fro') + 1e-10) for m in torch_mats]
    real_curve = calculate_rv_curve(proc_mats, k_max)
    if n_permutations > 0 and n_modalities >= 2:
        null_curves = []
        for _ in range(n_permutations):
            perm_mats = [proc_mats[0]] + [m[torch.randperm(m.shape[0])] for m in proc_mats[1:]]
            null_curves.append(calculate_rv_curve(perm_mats, k_max))
        null_curve_mean = np.mean(null_curves, axis=0); signal = np.array(real_curve) - null_curve_mean; optimal_k = np.argmax(signal) + 1
    else:
        if len(real_curve) < 3: optimal_k = 1
        else:
            y = np.array(real_curve); x_vals = np.linspace(0, 1, len(y)); y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
            optimal_k = np.argmax(y_norm - x_vals) + 1
    return int(optimal_k)

def decompose_energy(data_matrices: List[Union[torch.Tensor, np.ndarray]], simlr_result: Dict[str, Any], energy_type: str = "acc") -> Dict[str, Any]:
    """
    Decompose the SiMLR objective energy across modalities and features.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    simlr_result : Dict[str, Any]
        The result dictionary from a fitted SiMLR model.
    energy_type : str, default="acc"
        The energy function type to evaluate.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "modality_energies": List of energy values for each modality.
        - "feature_importances": List of gradient-based importance arrays 
          per modality feature.

    Raises
    ------
    TypeError
        If input types are invalid.
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]

    # V and U were fitted on preprocessed data, so the energy has to be
    # evaluated on preprocessed data too. Skipping this made the returned
    # decomposition inconsistent with the model's own energies.
    scale_list = simlr_result.get('scale_list') or []
    prov_list = simlr_result.get('provenance_list') or []
    if scale_list and scale_list[0] != "none" and len(prov_list) == len(torch_mats):
        torch_mats = [preprocess_data(m, scale_list, provenance=pv)
                      for m, pv in zip(torch_mats, prov_list)]

    u_all = simlr_result['u']; v_mats = simlr_result['v']
    modality_energies = []; feature_importances = []
    for i, (x, v) in enumerate(zip(torch_mats, v_mats)):
        u = u_all[i] if isinstance(u_all, list) else u_all
        mod_energy = calculate_simlr_energy(v, x, u, energy_type).item()
        grad = calculate_simlr_gradient(v, x, u, energy_type)
        # .detach().cpu() is required: a bare .numpy() raises on CUDA/MPS.
        feat_imp = torch.sum(torch.abs(grad), dim=1).detach().cpu().numpy()
        modality_energies.append(mod_energy); feature_importances.append(feat_imp)
    return {"modality_energies": modality_energies, "feature_importances": feature_importances}
