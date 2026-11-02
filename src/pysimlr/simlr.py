
import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any
from .svd import ba_svd
from .utils import basis_rank_report
from .optimizers import create_optimizer, SIMLR_OPTIMIZER_DEFAULTS
from .sparsification import simlr_sparseness, DEFAULT_SPARSENESS, energy_needs_unit_columns
from .utils import (adjusted_rvcoef, safe_svd, correlation_penalty, stationarity_residual,
                    l1_normalize_features,
                    orthogonality_summary, preprocess_data)
from .consensus import compute_shared_consensus
from .defaults import (DEFAULT_ENERGY_TYPE, DEFAULT_NSA_W, DEFAULT_TOPOLOGY, DEFAULT_LINEAR_OPTIMIZER,
                       DEFAULT_COLLAPSE_WEIGHT, DEFAULT_MIXING_ALGORITHM, DEFAULT_SIMLR_ITERATIONS,
                       DEFAULT_TOL, check_nsa_w)

#: How the default NSA-Flow weight `DEFAULT_NSA_W` was chosen.
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

#: Keys `simlr` accepts through ``**opt_params``: the optimizer
#: hyperparameters, minus the two the optimizer takes from `simlr`'s own named
#: parameters (forwarding them twice would let the two disagree).
SIMLR_OPTIMIZER_KEYS = frozenset(set(SIMLR_OPTIMIZER_DEFAULTS) - {"positivity"})

#: Names that existed in earlier releases. Passing one raises `TypeError`
#: naming the replacement; nothing is aliased.
SIMLR_RENAMED_PARAMETERS = {
    "epochs": "iterations",
    "init_seed": "random_state",
    "sparseness_quantile": "sparseness",
    "sparsity": "sparseness",
    "use_nsa": "removed (it had no effect); use nsa_w=0 to fit without a projection",
    "constraint": "removed; the one orthogonality weight is nsa_w (0 = no orthogonality constraint)",
    "beta": "the RMSProp decay is now 'rmsprop_beta'",
}



#: Every strategy `initialize_simlr` implements. Kept as one tuple so the
#: error message and the dispatcher can't drift apart.
INITIALIZATION_TYPES = ("pca", "varimax", "random", "gcca", "nndsvd", "perturbed_pca",
                        "domain", "best_of_n", "ica")


def initialize_simlr(data_matrices: List[torch.Tensor],
                     k: int,
                     initialization_type: str = "pca",
                     joint_reduction: bool = True,
                     positivity: str = "either",
                     generator: Optional[torch.Generator] = None,
                     domain_matrices: Optional[List[Optional[torch.Tensor]]] = None,
                     n_candidates: int = 5,
                     perturbation_scale: float = 0.05,
                     sparseness: Optional[Union[float, List[Optional[float]]]] = DEFAULT_SPARSENESS,
                     varimax: Optional[bool] = None) -> List[torch.Tensor]:
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
        `nsa_w` and `positivity`'s interaction with the optimizer -- so
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
    sparseness : float or list, optional
        Column-wise quantile sparseness applied by the initial retraction.
    varimax : bool or None, default=None
        Apply an Orthogonal Varimax rotation at initialization. ``None``
        rotates only when ``positivity`` is non-negative (see
        `resolve_varimax` for the measurements behind that default); an
        explicit ``True``/``False`` always wins. Semantics when rotating, by
        ``initialization_type``:

        * ``"pca"``, ``"perturbed_pca"`` -- per-view Varimax on each view's
          leading ``k`` right singular vectors (signed and non-negative alike;
          under a non-negative constraint with NSA-Flow installed the
          rotation is ``nsa_flow_data(init="varimax")``), then the views'
          columns are matched to view 0 by `_align_multiview_basis_columns`.
          Per-view Varimax is O(k)-arbitrary per view, so without the
          alignment column ``j`` of one view need not correspond to column
          ``j`` of another.
        * ``"gcca"`` -- one Varimax rotation ``R`` of the joint MAXVAR
          sample-space subspace ``shared_u`` (``n x k``), applied before each
          view is regressed onto it. Every view targets the same rotated
          columns, so the views are column-aligned by construction and no
          assignment step is needed.
        * ``"varimax"`` -- always rotates (per view, then aligned), whatever
          ``positivity`` is; an explicit ``varimax=False`` is refused.
        * ``"random"``, ``"nndsvd"``, ``"ica"`` -- unaffected. ``"domain"``
          forwards it to the PCA fallback used for a ``None`` prior, and
          ``"best_of_n"`` to its PCA candidate.

        ``varimax=False`` never rotates: under a non-negative constraint the
        NSA-Flow fit uses ``init="pca_clamp"`` and the PCA fallback keeps the
        unrotated loadings.

    Returns
    -------
    List[torch.Tensor]
        Initial basis matrices (P_i x K) for each view.

    Raises
    ------
    TypeError
        If input matrices are not tensors.
    ValueError
        If ``initialization_type="varimax"`` is combined with an explicit
        ``varimax=False``,
        or if the views do not share a sample count.

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
    n_rows = [int(x.shape[0]) for x in data_matrices]
    if len(set(n_rows)) > 1:
        raise ValueError(
            f"Size mismatch: every view must have the same number of samples "
            f"(rows); got {n_rows}."
        )
    if initialization_type == "varimax" and varimax is False:
        raise ValueError(
            "initialization_type='varimax' contradicts varimax=False; use "
            "initialization_type='pca' with varimax=False for unrotated PCA "
            "loadings, or leave varimax at its default."
        )
    varimax = resolve_varimax(varimax, positivity)
    if initialization_type == "domain":
        if domain_matrices is None:
            raise ValueError(
                "initialization_type='domain' needs `domain_matrices`, one "
                "entry per view."
            )
        return _domain_informed_basis(data_matrices, domain_matrices, k, positivity,
                                      sparseness=sparseness, varimax=varimax)
    if initialization_type == "gcca":
        return _gcca_basis(data_matrices, k, positivity, sparseness=sparseness,
                           varimax=varimax)
    if initialization_type == "nndsvd":
        return [
            simlr_sparseness(_nndsvd_basis_for_view(x, k), constraint_type="ortho",
                             positivity=positivity, sparseness=sparseness,
                             modality_index=idx_m).to(x.dtype)
            if sparseness is not None and x.shape[1] > k
            else _nndsvd_basis_for_view(x, k)
            for idx_m, x in enumerate(data_matrices)
        ]
    if initialization_type == "ica":
        return [_ica_basis_for_view(x, k, positivity=positivity, sparseness=sparseness, modality_index=idx_m)
                for idx_m, x in enumerate(data_matrices)]
    if initialization_type == "perturbed_pca":
        base = [initial_basis_for_view(x, k, positivity=positivity,
                                       initialization_type="pca",
                                       sparseness=sparseness,
                                       modality_index=idx_m,
                                       varimax=varimax)
                for idx_m, x in enumerate(data_matrices)]
        if varimax and len(base) > 1 and k > 1:
            base = _align_multiview_basis_columns(data_matrices, base,
                                                  nonneg=_wants_nonneg(positivity))
        out = []
        for idx_m, v in enumerate(base):
            noise = torch.randn(v.shape, generator=generator, dtype=v.dtype)
            raw = v + perturbation_scale * v.norm() * noise / (noise.norm() + 1e-12)
            out.append(simlr_sparseness(raw, constraint_type="ortho", positivity=positivity,
                                        sparseness=sparseness, modality_index=idx_m))
        return out
    if initialization_type == "best_of_n":
        return _best_of_n_basis(data_matrices, k, positivity, generator, n_candidates,
                                sparseness=sparseness, varimax=varimax)
    v_mats = [initial_basis_for_view(x, k, positivity=positivity,
                                     initialization_type=initialization_type,
                                     generator=generator,
                                     sparseness=sparseness,
                                     modality_index=idx_m,
                                     varimax=varimax)
              for idx_m, x in enumerate(data_matrices)]
    rotated = initialization_type == "varimax" or (varimax and initialization_type == "pca")
    if len(v_mats) > 1 and k > 1 and rotated:
        v_mats = _align_multiview_basis_columns(data_matrices, v_mats,
                                                nonneg=_wants_nonneg(positivity))
    return v_mats


def _wants_nonneg(positivity: str) -> bool:
    """True when ``positivity`` requests a non-negative basis."""
    return positivity in ('positive', 'hard', 'nonnegative', 'nonneg', 'softplus')


def resolve_varimax(varimax: Optional[bool], positivity: str) -> bool:
    """Resolve the ``varimax`` default against ``positivity``.

    ``None`` (the default on every entry point) rotates only a non-negative
    basis. An explicit ``True`` or ``False`` always wins.

    The split follows measurement, not symmetry. Under a non-negative
    constraint the Varimax start was never worse in the sweeps run for this
    change (NED collapse basin 4/11 seeds against 5/11 without it). On a signed
    problem the rotated start was itself never worse than PCA, but the fits
    that followed were mixed: NED fell into the collapse basin on 6/11 seeds
    against 4/11 (`test_latent_collapse`'s generator, seeds 0-9 and 42), and
    linear `simlr` at ``orthox0.9x1`` dropped one of six seeds from 0.986 to
    0.755 latent recovery. Signed rotation is therefore opt-in.
    """
    if varimax is None:
        return _wants_nonneg(positivity)
    return bool(varimax)


def _align_multiview_basis_columns(data_matrices: List[torch.Tensor],
                                   v_mats: List[torch.Tensor],
                                   nonneg: bool = False) -> List[torch.Tensor]:
    """Align column permutations across views to view 0 in sample space.

    Per-view Varimax (and per-view ``nsa_flow_data``) solves an O(k)-invariant
    objective independently per modality, so component columns are unordered
    across views. Matching view m's sample projections ``X_m V_m`` to view 0's
    ``X_0 V_0`` via linear assignment puts corresponding localized factors in
    the same column slot before sweep 0 computes the shared consensus ``U``.
    """
    from scipy.optimize import linear_sum_assignment
    p0 = (data_matrices[0].detach().double() @ v_mats[0].detach().double())
    p0_c = p0 - p0.mean(dim=0, keepdim=True)
    p0_n = p0_c / torch.linalg.vector_norm(p0_c, dim=0, keepdim=True).clamp_min(1e-12)
    aligned = [v_mats[0]]
    for m in range(1, len(v_mats)):
        pm = (data_matrices[m].detach().double() @ v_mats[m].detach().double())
        pm_c = pm - pm.mean(dim=0, keepdim=True)
        pm_n = pm_c / torch.linalg.vector_norm(pm_c, dim=0, keepdim=True).clamp_min(1e-12)
        cross = (p0_n.t() @ pm_n).cpu().numpy()
        score = cross if nonneg else np.abs(cross)
        _, col_ind = linear_sum_assignment(-score)
        perm = torch.as_tensor(col_ind, dtype=torch.long, device=v_mats[m].device)
        v_m = v_mats[m][:, perm]
        if not nonneg:
            signs = [1.0 if cross[j, col_ind[j]] >= 0.0 else -1.0 for j in range(len(col_ind))]
            sign_t = torch.tensor(signs, dtype=v_m.dtype, device=v_m.device).unsqueeze(0)
            v_m = v_m * sign_t
        aligned.append(v_m)
    return aligned


def _ica_basis_for_view(x: torch.Tensor, k: int, positivity: str = "either",
                        sparseness: Optional[Union[float, List[Optional[float]]]] = DEFAULT_SPARSENESS,
                        modality_index: Optional[int] = None) -> torch.Tensor:
    """Deterministic spatial ICA initialization on the top-``k`` singular vectors.

    Rotates the leading ``k`` right singular vectors of ``x`` via FastICA
    (seeded deterministically with ``w_init = I_k``) to maximize spatial
    non-Gaussianity across features before applying `simlr_sparseness`.
    Columns are sign-canonicalized by third-moment skewness and ordered by
    descending explained variance ``||x @ v_j||_2^2``.
    """
    xd = x.detach().cpu().double().numpy()
    _, s, vt = np.linalg.svd(xd, full_matrices=False)
    k_eff = min(k, vt.shape[0])
    v_svd = vt[:k_eff, :].T.copy()
    for j in range(k_eff):
        idx_max = int(np.argmax(np.abs(v_svd[:, j])))
        if v_svd[idx_max, j] < 0:
            v_svd[:, j] *= -1.0

    if k_eff > 1 and v_svd.shape[0] > k_eff:
        # A FastICA failure propagates; it used to return the SVD basis under the name 'ica'.
        from sklearn.decomposition import FastICA
        ica = FastICA(
            n_components=k_eff,
            algorithm="parallel",
            whiten="unit-variance",
            fun="logcosh",
            w_init=np.eye(k_eff, dtype=np.float64),
            max_iter=500,
            tol=1e-5,
            random_state=0,
        )
        v_ica = ica.fit_transform(v_svd)
    else:
        v_ica = v_svd

    for j in range(k_eff):
        col = v_ica[:, j]
        if np.sum(col ** 3) < 0:
            v_ica[:, j] *= -1.0
        nrm = np.linalg.norm(v_ica[:, j])
        if nrm > 1e-12:
            v_ica[:, j] /= nrm

    var_scores = [np.sum((xd @ v_ica[:, j]) ** 2) for j in range(k_eff)]
    order = np.argsort(var_scores)[::-1]
    v_ica = v_ica[:, order]

    if k_eff < k:
        pad = 1e-4 * np.ones((x.shape[1], k - k_eff), dtype=np.float64)
        v_ica = np.hstack([v_ica, pad])

    v_t = torch.tensor(v_ica, dtype=x.dtype, device=x.device)
    return simlr_sparseness(v_t, constraint_type="ortho", positivity=positivity,
                            sparseness=sparseness, modality_index=modality_index)


def _gcca_basis(data_matrices: List[torch.Tensor], k: int,
                positivity: str,
                sparseness: Optional[Union[float, List[Optional[float]]]] = DEFAULT_SPARSENESS,
                varimax: bool = True) -> List[torch.Tensor]:
    """Joint (generalized-CCA, MAXVAR) initialization.

    Finds the top-``k`` eigenvectors of ``sum_i P_i``, where ``P_i`` projects
    onto view ``i``'s column space -- the sample-space subspace best agreed on
    by every view jointly -- then regresses each view onto it by least
    squares to recover a per-view basis. See `initialize_simlr`'s docstring
    for why this targets the buried-shared-signal regime that per-view PCA
    does not.

    With ``varimax=True`` the shared subspace is rotated once,
    ``shared_u <- shared_u @ R`` with ``R = varimax_rotation(shared_u)``,
    before the regressions. ``R`` is orthogonal, so the span (and hence the
    MAXVAR objective) is unchanged; only the axes within it move. Because
    every view regresses onto the same rotated columns, column ``j`` means
    the same shared direction in every view.
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
    if varimax and k > 1 and n > k:
        from .svd import varimax_rotation
        shared_u = (shared_u @ varimax_rotation(shared_u)).contiguous()
    out = []
    for idx_m, x in enumerate(data_matrices):
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
                                    positivity=positivity,
                                    sparseness=sparseness,
                                    modality_index=idx_m))
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
                           k: int, positivity: str,
                           sparseness: Optional[Union[float, List[Optional[float]]]] = DEFAULT_SPARSENESS,
                           varimax: bool = True) -> List[torch.Tensor]:
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
    for idx_m, (x, dm) in enumerate(zip(data_matrices, domain_matrices)):
        if dm is None:
            out.append(initial_basis_for_view(x, k, positivity=positivity,
                                              initialization_type="pca",
                                              sparseness=sparseness,
                                              modality_index=idx_m,
                                              varimax=varimax))
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
                                    positivity=positivity,
                                    sparseness=sparseness,
                                    modality_index=idx_m))
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
                     generator: torch.Generator, n_candidates: int,
                     sparseness: Optional[Union[float, List[Optional[float]]]] = DEFAULT_SPARSENESS,
                     varimax: bool = True) -> List[torch.Tensor]:
    """Draw `n_candidates` feasible starts (PCA plus random draws) and keep
    the one `_score_joint_basis` ranks highest."""
    candidates = [initialize_simlr(data_matrices, k, initialization_type="pca",
                                   positivity=positivity, sparseness=sparseness,
                                   varimax=varimax)]
    for _ in range(max(0, n_candidates - 1)):
        candidates.append(initialize_simlr(data_matrices, k, initialization_type="random",
                                           positivity=positivity, generator=generator,
                                           sparseness=sparseness))
    scores = [_score_joint_basis(data_matrices, c) for c in candidates]
    return candidates[int(np.argmax(scores))]


def initialize_deep_encoders(encoders, data_matrices: List[torch.Tensor], k: int,
                             initialization_type: str = "pca",
                             generator: Optional[torch.Generator] = None,
                             domain_matrices: Optional[List[Optional[torch.Tensor]]] = None,
                             n_candidates: int = 5,
                             perturbation_scale: float = 0.05,
                             varimax: Optional[bool] = None) -> None:
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
    n_candidates, perturbation_scale, varimax : see `initialize_simlr`.

    Varimax acts here, on the initial first-layer basis, and nowhere after
    training: the deep decoders ``f_m(X_m V_m)`` and the RealNVP couplings
    are not O(k)-equivariant, so rotating a trained ``V_m`` would detach it
    from the model that was fitted with it.
    """
    positivity = encoders[0].positivity
    enc_sparseness = [getattr(enc, "sparseness", DEFAULT_SPARSENESS) for enc in encoders]
    v_mats = initialize_simlr(data_matrices, k, initialization_type=initialization_type,
                              positivity=positivity, generator=generator,
                              domain_matrices=domain_matrices, n_candidates=n_candidates,
                              perturbation_scale=perturbation_scale,
                              sparseness=enc_sparseness, varimax=varimax)
    with torch.no_grad():
        for enc, x, v in zip(encoders, data_matrices, v_mats):
            enc.v_raw.copy_(v.to(x.dtype))


def initial_basis_for_view(x: torch.Tensor, k: int,
                           positivity: str = "either",
                           initialization_type: str = "pca",
                           generator: Optional[torch.Generator] = None,
                           sparseness: Optional[Union[float, List[Optional[float]]]] = DEFAULT_SPARSENESS,
                           modality_index: Optional[int] = None,
                           varimax: Optional[bool] = None) -> torch.Tensor:
    """
    Build the initial ``(features, k)`` basis for one view.

    Shared by `initialize_simlr` and the deep models' ``initialize_v`` so that
    every entry point starts from the same basis for a given constraint.

    Under a non-negative constraint the basis is fitted from ``x`` with
    `nsa_flow_data` when the backend is available. Otherwise it is the
    (optionally Varimax-rotated) PCA loadings, with each column's sign resolved so that its sum is positive
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
    varimax : bool or None, default=None
        Rotate the PCA loadings by Orthogonal Varimax, or request
        ``init="varimax"`` from `nsa_flow_data`. ``None`` resolves through
        `resolve_varimax` (non-negative only).
        ``initialization_type="varimax"`` rotates regardless. Column
        correspondence across views is the caller's job; see
        `_align_multiview_basis_columns`.

    Returns
    -------
    torch.Tensor
        Basis of shape (features, k) in `x`'s dtype.
    """
    from .nsa_backend import load_nsa_flow_data
    from .sparsification import NSA_DEFAULT_W, _resolve_quantile, _usable_retraction, simlr_sparseness

    want_nonneg = _wants_nonneg(positivity)
    varimax = resolve_varimax(varimax, positivity)
    q_init = _resolve_quantile(sparseness, modality_index=modality_index)

    if initialization_type == "random":
        if generator is None:
            raise ValueError("initialization_type='random' needs a `generator`")
        # Drawn feasible, not drawn then repaired: a raw draw is neither
        # near-orthogonal nor sparse, so the first retraction would move it
        # much further than any subsequent sweep and the "start" recorded by
        # the caller would not be the point the optimiser actually began from.
        raw = (torch.rand(x.shape[1], k, generator=generator, dtype=x.dtype)
               if want_nonneg else
               torch.randn(x.shape[1], k, generator=generator, dtype=x.dtype))
        return simlr_sparseness(raw, constraint_type="ortho",
                                positivity=positivity,
                                sparseness=q_init).to(x.dtype)
    if want_nonneg and x.shape[1] > k:
        fit_from_data = load_nsa_flow_data()
        if fit_from_data is not None:
            q_data = (q_init if (q_init is not None and k > 1 and q_init > 1.0 - 1.0 / k)
                      else None)
            fitted = _nonnegative_basis_from_data(
                fit_from_data, x, k, NSA_DEFAULT_W, _usable_retraction,
                sparseness=q_data, varimax=varimax)
            if fitted is not None:
                if q_data is not None:
                    from .sparsification import _apply_quantile_sparseness
                    fitted = _apply_quantile_sparseness(
                        fitted, q_data, nonneg=True, only_if_denser=True)
                return fitted.to(x.dtype)

    u, s, v = ba_svd(x, nu=0, nv=k)
    if v.shape[1] < k:
        padding = torch.randn(v.shape[0], k - v.shape[1],
                              dtype=v.dtype, device=v.device) * 1e-4
        v = torch.cat([v, padding], dim=1)
    if (varimax or initialization_type == "varimax") and v.shape[0] > k and k > 1:
        from .svd import varimax as _varimax
        v = _varimax(v)
    if want_nonneg:
        # Resolve each column's sign so downstream rectification removes as
        # little as possible. This is a sign flip, not a rectification, so the
        # basis stays exactly as orthogonal as PCA/Varimax made it.
        for j in range(v.shape[1]):
            if v[:, j].sum() < 0:
                v[:, j] *= -1
    return v.to(x.dtype)


def _nonnegative_basis_from_data(fit_from_data, x: torch.Tensor, k: int,
                                 w: float, usable,
                                 sparseness: Optional[float] = None,
                                 varimax: bool = True) -> Optional[torch.Tensor]:
    """
    Fit a non-negative, near-orthogonal basis for `x` directly from the data.

    Returns None if the backend raises or returns something unusable, so the
    caller falls back to the PCA initializer rather than starting from a
    degenerate basis. A wide view (``p <= k``) is left to PCA: no basis with
    more columns than rows can be orthogonal, so there is nothing to fit.
    """
    reference = torch.ones(x.shape[1], k, dtype=torch.float64)
    init_mode = "varimax" if varimax else "pca_clamp"
    # A backend failure propagates; it used to return None and the caller silently used ba_svd.
    try:
        result = fit_from_data(x.detach().double(), k=k, w=float(w),
                               init=init_mode, sparseness=sparseness)
    except TypeError:
        try:
            result = fit_from_data(x.detach().double(), k=k, w=float(w),
                                   sparseness=sparseness)
        except TypeError:
            result = fit_from_data(x.detach().double(), k=k, w=float(w))
    from .nsa_backend import result_matrix
    candidate = result_matrix(result)
    if candidate is None:
        return None
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    candidate = torch.clamp_min(candidate, 0.0)
    if not usable(candidate, reference):
        return None
    return candidate


def _nsa_data_transform(x, nonneg: bool, **nsa_kwargs) -> Dict[str, Any]:
    """Run NSA-Flow on a data matrix and return its basis, scores and result.
    Shared by `nsa_contrast_transform` (signed) and `nsa_nonnegative_transform`
    (non-negative, basis clamped at 0)."""
    from .nsa_backend import load_nsa_flow, result_matrix
    fn = load_nsa_flow()
    if fn is None:
        raise RuntimeError("NSA-Flow is not installed. Install with `pip install nsa-flow`.")
    x_t = torch.as_tensor(x, dtype=torch.float32)
    res = fn(x_t, nonneg=True, **nsa_kwargs) if nonneg else fn(x_t, **nsa_kwargs)
    v = result_matrix(res)
    if v is None:
        raise RuntimeError("NSA-Flow returned a result with neither 'V' nor 'Y'.")
    if nonneg:
        v = torch.clamp_min(v, 0.0)
    return {"v": v, "scores": x_t @ v, "result": res}


def nsa_contrast_transform(x: Union[torch.Tensor, np.ndarray],
                           k: int = 6,
                           w: float = DEFAULT_NSA_W,
                           consolidate: bool = True,
                           optimizer: Optional[str] = None,
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
    optimizer : str, optional
        NSA-Flow optimizer; None uses NSA-Flow's default (L-BFGS-B). NSA-Flow
        deprecated "torch_lbfgs", which was the default here before 0.4.0.
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
    return _nsa_data_transform(x, nonneg=False, k=k, w=w, mode="signed", consolidate=consolidate,
                               optimizer=optimizer, max_iter=max_iter, tol=tol)


def nsa_nonnegative_transform(x: Union[torch.Tensor, np.ndarray],
                              k: int = 6,
                              w: float = DEFAULT_NSA_W,
                              optimizer: Optional[str] = None,
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
    optimizer : str, optional
        NSA-Flow optimizer; None uses NSA-Flow's default (L-BFGS-B). NSA-Flow
        deprecated "torch_lbfgs", which was the default here before 0.4.0.
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
    return _nsa_data_transform(x, nonneg=True, k=k, w=w, mode="data",
                               optimizer=optimizer, max_iter=max_iter, tol=tol)

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
    "recon_r2", "acc", "logcosh", "exp", "kurtosis", "nc", "align", "procrustes",
    "negentropy", "dat",
})


def calculate_simlr_energy(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, energy_type: str = DEFAULT_ENERGY_TYPE, lambda_val: float = 0.0, prior_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    energy_type : str, default=DEFAULT_ENERGY_TYPE ("nc")
        Similarity term (`pysimlr.defaults`, `similarity.SIMILARITY`).
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


def _calibrate_view_scale(v_m: torch.Tensor, x_m: torch.Tensor, u_m: torch.Tensor,
                          energy_type: str, lambda_val: float = 0.0,
                          prior_matrix: Optional[torch.Tensor] = None) -> float:
    """Evaluate initial view scale for similarity normalization."""
    sim_e = calculate_simlr_energy(v_m, x_m, u_m, energy_type, lambda_val=lambda_val, prior_matrix=prior_matrix)
    return abs(float(sim_e.item()))


def calculate_simlr_gradient(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, 
                             energy_type: str = DEFAULT_ENERGY_TYPE, lambda_val: float = 0.0, 
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
    energy_type : str, default=DEFAULT_ENERGY_TYPE ("nc")
        Similarity term (`pysimlr.defaults`, `similarity.SIMILARITY`).
        Options: `SUPPORTED_ENERGY_TYPES`.
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


def _check_views(data_matrices, n_views: Optional[int] = None,
                 fitting: bool = False) -> List[torch.Tensor]:
    """Validate a list of views and return them as float32 tensors.

    Shared by `simlr`, `predict_shared_latent` and `predict_simlr`. Requires a
    non-empty list or tuple of 2-D matrices with equal row counts and, when
    `n_views` is given, exactly that many. For a fit (`fitting=True`) each
    view must also be a tensor or array with at least one row and column.
    """
    if not isinstance(data_matrices, (list, tuple)) or len(data_matrices) == 0:
        raise ValueError("data_matrices must be a non-empty list or tuple of matrices/arrays.")
    if n_views is not None and len(data_matrices) != n_views:
        raise ValueError(f"Expected {n_views} data matrices matching model views, "
                         f"but received {len(data_matrices)}.")
    mats = []
    for idx, m in enumerate(data_matrices):
        if fitting and not isinstance(m, (torch.Tensor, np.ndarray)):
            raise TypeError(f"View {idx} must be a torch.Tensor or np.ndarray, got {type(m).__name__}.")
        m_t = torch.as_tensor(m).float()
        if m_t.ndim != 2:
            raise ValueError(
                f"Each view in data_matrices must be 2D (samples x features). "
                f"View {idx} has {m_t.ndim} dimension(s) with shape {tuple(m_t.shape)}.")
        if fitting and m_t.shape[0] == 0:
            raise ValueError(f"View {idx} has 0 samples.")
        if fitting and m_t.shape[1] == 0:
            raise ValueError(f"View {idx} has 0 features.")
        mats.append(m_t)
    n_samples_list = [m.shape[0] for m in mats]
    if len(set(n_samples_list)) > 1:
        details = ", ".join(f"view {i}: {n}" for i, n in enumerate(n_samples_list))
        raise ValueError(
            f"All data matrices must have identical number of samples (rows), "
            f"but found mismatched sample counts: {details}.")
    return mats


def _scale_like_training(torch_mats: List[torch.Tensor], simlr_result: Dict[str, Any]) -> List[torch.Tensor]:
    """Apply the fit's `scale_list` with its recorded per-view provenance
    (training means and scales), so new data land in the training space."""
    scale_list = simlr_result.get('scale_list', [])
    provenance_list = simlr_result.get('provenance_list', [])
    if not scale_list or scale_list[0] == "none":
        return torch_mats
    return [preprocess_data(m, scale_list, provenance=provenance_list[i] if i < len(provenance_list) else None)
            for i, m in enumerate(torch_mats)]


# Private channel from the `learning_rate="auto"` branch of `simlr` into its
# own probe and final fits: `opt_params["_internal_probe"] = (token, views,
# provenance, v0, probe_only)`. The views are prepared and the start basis drawn
# once, so every probe and the final fit start from the same point and none
# repeats the preprocessing. Only a tuple headed by this object is honoured;
# users cannot construct it.
_INTERNAL_PROBE_TOKEN = object()


def _prepare_fit_inputs(data_matrices, k, scale_list, positivity, initialization_type,
                        random_state, domain_matrices, sparseness, varimax):
    """Validate and scale the views and draw the starting basis for `simlr`.

    Returns ``(torch_mats, provenance_list, v0)``. Every initializer gets a
    generator seeded from `random_state`; the deterministic ones ignore it.
    It does not change `initialization_type` (it used to switch "pca" to
    "random").
    """
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")
    torch_mats = _check_views(data_matrices, fitting=True)
    provenance_list = []
    if scale_list is not None and len(scale_list) > 0 and scale_list[0] != "none":
        scaled = []
        for m in torch_mats:
            m_scaled, prov = preprocess_data(m, scale_list)
            scaled.append(m_scaled)
            provenance_list.append(prov)
        torch_mats = scaled
    init_gen = torch.Generator()
    init_gen.manual_seed(random_state)
    v0 = initialize_simlr(torch_mats, k, positivity=positivity,
                          initialization_type=initialization_type,
                          generator=init_gen, domain_matrices=domain_matrices,
                          sparseness=sparseness, varimax=varimax)
    return torch_mats, provenance_list, v0


def simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
          k: int,
          iterations: int = DEFAULT_SIMLR_ITERATIONS,
          initialization_type: str = "pca",
          random_state: Optional[int] = None,
          negentropy_weight: float = 0.0,
          collapse_weight: float = DEFAULT_COLLAPSE_WEIGHT,
          nonneg_u: bool = False,
          optimizer_type: str = DEFAULT_LINEAR_OPTIMIZER,
          energy_type: str = DEFAULT_ENERGY_TYPE,
          mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
          positivity: str = "either",
          smoothing_matrices: Optional[List[torch.Tensor]] = None,
          domain_matrices: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
          domain_lambdas: Optional[Union[float, List[float]]] = None,
          orthogonalize_u: bool = False,
          topology: str = DEFAULT_TOPOLOGY,
          path_graph: Optional[Dict[int, List[int]]] = None,
          scale_list: Optional[List[str]] = None,
          consolidate: bool = False,
          nsa_w: float = DEFAULT_NSA_W,
          sparseness: Optional[Union[float, List[Optional[float]]]] = DEFAULT_SPARSENESS,
          varimax: Optional[bool] = None,
          dynamic_weights: bool = False,
          mai_metric: str = "procrustes_r2",
          use_rank_mai: bool = False,
          tol: float = DEFAULT_TOL,
          patience: Optional[int] = None,
          verbose: bool = False,
          **opt_params) -> Dict[str, Any]:
    """
    Perform Similarity-driven Multi-view Linear Representation (SiMLR).

    SiMLR identifies a shared latent subspace across multiple data modalities
    by optimizing an energy function subject to manifold constraints and sparsity.
    Scale-free similarity energies receive equal weight (1/M) across modalities,
    while non-scale-free energies calibrate weights relative to initial empirical energy.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    k : int
        The dimensionality of the shared latent space.
    iterations : int, default=DEFAULT_SIMLR_ITERATIONS (300)
        Maximum number of optimization sweeps. At the default `tol` the fits
        measured in `pysimlr.defaults` stop after ~73; the rest is headroom.
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
    nsa_w : float, default=DEFAULT_NSA_W (0.5)
        The one orthogonality weight, in ``[0, 1]``: the weight of the
        NSA-Flow retraction applied after every step. The objective is the
        similarity energy alone, minimised subject to that projection (the
        original SiMLR design; the NSA-Flow paper calibrated w = 0.5 for the
        projection alone). ``0`` fits without an orthogonality constraint
        (positivity and ``sparseness`` still apply).

        An orthogonality-defect penalty used to be added to the objective as
        well. Removing it (`scripts/constraint_selection.py`, 20 paired
        synthetic fits) raised support F1 0.621 -> 0.646 (paired t 2.6) and
        made every fit converge (20/20 against 14/20); frame defect stayed
        near zero (0.0043) and column support overlap rose 0.155 -> 0.244,
        a difference confined to signed fits, where the contract is
        orthogonality rather than disjointness.

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
    collapse_weight : float, default=DEFAULT_COLLAPSE_WEIGHT (1.0)
        Weight on `utils.correlation_penalty` of each view's scores ``X V``,
        added to the objective. Disjoint loadings on correlated features can
        still give collinear scores, which an orthogonalised consensus hides
        and "nc" does not penalise. ``0`` disables it.
    initialization_type : str, default="pca"
        One of `INITIALIZATION_TYPES`: ``"pca"`` (deterministic, the
        historical behaviour), ``"random"``, ``"gcca"``, ``"nndsvd"``,
        ``"perturbed_pca"``, ``"domain"`` (needs `domain_matrices`) or
        ``"best_of_n"``. See `initialize_simlr` for what each does and why;
        only ``"pca"`` and ``"random"`` have any measurement behind them here
        -- the rest are untested strategies, not benchmarked replacements.
    random_state : int, optional
        Seed for the initializer, through a local `torch.Generator`, so the
        global RNG stream is untouched; deterministic starts (``"pca"``)
        ignore it. It does not change `initialization_type`. When ``None``,
        one seed is drawn from the global torch RNG before anything else
        (including the learning-rate probes, which share it) and is always
        returned in ``result["random_state"]``.
    optimizer_type : str, default=DEFAULT_LINEAR_OPTIMIZER ("hybrid_adam")
        The optimizer to use; the default and the benchmark behind it are in
        `pysimlr.defaults`. Before 0.4.0 the default was `armijo_gradient`.
    energy_type : str, default=DEFAULT_ENERGY_TYPE ("nc")
        The similarity/reconstruction objective to minimize. Scale-free
        energies (`similarity.SCALE_FREE_ENERGIES`, including `nc`, `align`,
        `procrustes`, `recon_r2`, and ICA contrasts) are weighted equally
        with weight 1/M across the M modalities, avoiding arbitrary
        initialization-dependent weight spreads (previously up to 279x across
        views) and bounding total energy min >= -1 for `nc`. Other energies
        are calibrated relative to their initial empirical energy.
    mixing_algorithm : str, default="svd"
        The algorithm used to mix projections ("svd", "pca", "ica", "avg", "newton").

    positivity : str, default="either"
        Sign constraint on the basis ("either", "positive", "negative").
    sparseness : float or List[float] or None, default=0.5
        R ``ANTsR::simlr``-style column-wise quantile sparsification target in
        ``(0, 1)``, or ``None`` for soft NSA-Flow orthogonality without hard
        cardinality thresholding.
    consolidate : bool, default=True
        Whether to apply post-hoc disjoint support consolidation across columns.
    varimax : bool or None, default=None
        ``None`` rotates only a non-negative basis (``positivity="positive"``
        and its aliases); pass ``True`` to rotate a signed basis too, or
        ``False`` to never rotate (see `resolve_varimax`). When rotating,
        applies Orthogonal Varimax subspace rotation at initialization (``init="varimax"`` in
        ``nsa_flow_data`` and :func:`pysimlr.svd.varimax` on PCA bases, signed
        or non-negative; one shared rotation of the MAXVAR subspace for
        ``initialization_type="gcca"``), redistributing any Perron-Frobenius
        global/mean mode across all ``k`` axes into localized regional
        factors. See `initialize_simlr` for the exact semantics per
        ``initialization_type``. Set ``varimax=False``
        (``init="pca_clamp"``) on raw abundance, compositional, or dietary
        intake tables where the global mean mode itself carries the primary
        signal of interest.
    smoothing_matrices : List[torch.Tensor], optional
        Spatial/prior smoothing operators for each modality.
    domain_matrices : List[Union[torch.Tensor, np.ndarray]], optional
        Matrices to align with for directed domain knowledge.
    domain_lambdas : Union[float, List[float]], optional
        Weight(s) for the domain knowledge alignment objective.
    orthogonalize_u : bool, default=False
        Whether to enforce orthogonality on the consensus matrix U.
    topology : str, default=DEFAULT_TOPOLOGY ("loo")
        The consensus topology ("star", "loo", "graph").
    path_graph : Dict[int, List[int]], optional
        Adjacency list for "graph" topology.
    scale_list : List[str], optional
        Preprocessing methods to apply to input data. ``None`` means
        ``["centerAndScale", "np"]``.
    tol : float, default=DEFAULT_TOL (1e-4)
        Relative-improvement threshold of the energy plateau rule: the fit
        stops, converged (``certificate="energy_plateau"``), after `patience`
        sweeps that improve the best energy by less than `tol` relative. This
        is the only stopping tolerance. A stationarity residual on the iterate's
        current face (`utils.stationarity_residual`) is reported per sweep in
        ``grad_map_history`` (and ``grad_map``, its last value) as a diagnostic;
        it does not stop the fit.
    patience : int, optional
        Iterations without relative improvement above `tol` before stopping.
        ``None`` means `utils.CONVERGENCE_PATIENCE`.
    verbose : bool, default=False
        Whether to print convergence details.
    **opt_params : dict
        Optimizer hyperparameters: the keys of
        `optimizers.SIMLR_OPTIMIZER_DEFAULTS` except ``positivity`` and
        ``nsa_w``, which come from this function's own parameters, plus
        ``learning_rate`` (float or ``"auto"``). Any other key raises
        `TypeError`.

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
    # Keyword validation. Anything not a named parameter arrives here, and the
    # only legitimate content is optimizer hyperparameters. Unknown keys used
    # to be forwarded to the optimizer and merely warned about, so a typo or a
    # parameter meant for a different method changed nothing and the fit
    # still reported success. Renamed parameters raise with the new name.
    if scale_list is None:
        scale_list = ["centerAndScale", "np"]
    _internal = opt_params.get("_internal_probe", None)
    if isinstance(_internal, tuple) and len(_internal) == 5 and _internal[0] is _INTERNAL_PROBE_TOKEN:
        opt_params.pop("_internal_probe")
        _, _prepped_mats, _prepped_prov, _prepped_v, _probe_only = _internal
    else:
        _prepped_mats = _prepped_prov = _prepped_v = None
        _probe_only = False
    _bad = sorted(set(opt_params) - set(SIMLR_OPTIMIZER_KEYS))
    if _bad:
        _hints = [f"{kk!r} -> {SIMLR_RENAMED_PARAMETERS[kk]}"
                  for kk in _bad if kk in SIMLR_RENAMED_PARAMETERS]
        raise TypeError(
            f"simlr() got unexpected keyword argument(s) {_bad}."
            + (f" Renamed or removed: {'; '.join(_hints)}." if _hints else "")
            + f" Optimizer hyperparameters accepted through **opt_params: "
            f"{sorted(SIMLR_OPTIMIZER_KEYS)}."
        )

    # One seed per call, resolved before the learning-rate probes so that the
    # probes and the final fit share it, and recorded in the result. It used to
    # stay None for "deterministic" starts, which handed `initialize_simlr`
    # generator=None: starts that do draw random numbers (e.g. "gcca") then
    # used the global stream, whose state the probes leave optimizer-dependent
    # (matched |r| 0.74/0.58 under hybrid_adam vs 0.94/0.96 for any seed).
    if random_state is None:
        random_state = int(torch.randint(0, 2**31 - 1, (1,)).item())
    random_state = int(random_state)

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
    from .optimizers import LEARNING_RATE_FREE_OPTIMIZERS
    if optimizer_type in LEARNING_RATE_FREE_OPTIMIZERS and opt_params["learning_rate"] == "auto":
        # Nothing to tune: these optimizers never read the learning rate.
        opt_params["learning_rate"] = SIMLR_OPTIMIZER_DEFAULTS["learning_rate"]
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
        _prep_mats, _prep_prov, _prep_v = _prepare_fit_inputs(
            data_matrices, k, scale_list, positivity, initialization_type,
            random_state, domain_matrices, sparseness, varimax)
        from .optimizers import tune_learning_rate
        _iters = _bound.pop("iterations")
        _rest = {kk: vv for kk, vv in opt_params.items() if kk != "learning_rate"}

        def _probe(learning_rate, iterations):
            return simlr(
                **_bound, **_rest, iterations=iterations,
                learning_rate=learning_rate,
                _internal_probe=(_INTERNAL_PROBE_TOKEN, _prep_mats, _prep_prov, _prep_v, True),
            )

        _lr, _scores = tune_learning_rate(
            _probe, probe_iterations=min(3, max(1, _iters)), verbose=verbose)
        _out = simlr(
            **_bound, **_rest, iterations=_iters, learning_rate=_lr,
            _internal_probe=(_INTERNAL_PROBE_TOKEN, _prep_mats, _prep_prov, _prep_v, False),
        )
        _out["learning_rate"] = _lr
        _out["learning_rate_scores"] = _scores
        return _out

    if _prepped_mats is not None:
        torch_mats, provenance_list = _prepped_mats, list(_prepped_prov)
        v_mats = [v.clone() for v in _prepped_v]
    else:
        torch_mats, provenance_list, v_mats = _prepare_fit_inputs(
            data_matrices, k, scale_list, positivity, initialization_type,
            random_state, domain_matrices, sparseness, varimax)
    n_modalities = len(torch_mats)
    orig_dtype = torch_mats[0].dtype
    # One orthogonality weight. It drives the NSA-Flow retraction applied after
    # every step and the calibrated orthogonality-defect term of the objective;
    # 0 removes both (no orthogonality constraint, sparsity still applied).
    nsa_w = check_nsa_w(nsa_w)
    constraint_type = "ortho" if nsa_w > 0.0 else "none"
    constraint_weight = nsa_w
    # `positivity` reaches the optimizer: without it the bound-constrained
    # `nsa_lbfgsb` ran with no lower bound under positivity="positive".
    opt_params["positivity"] = positivity
    optimizer = create_optimizer(optimizer_type, v_mats, **opt_params)
    
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
    
    _want_nonneg = positivity in ('positive', 'hard', 'nonnegative', 'nonneg',
                                  'softplus')

    energy_history = []
    grad_map_history = []
    n_degenerate_iterates = 0
    stop_reason = "max_iter"
    from .utils import ConvergenceMonitor, CONVERGENCE_PATIENCE
    monitor = ConvergenceMonitor(
        tol=tol,
        patience=int(CONVERGENCE_PATIENCE if patience is None else patience),
        max_steps=iterations)
    certificate = None
    prev_total_energy = float('inf')
    converged_iter = iterations
    # Alternating minimization is not monotone in the joint objective: each V_i
    # is optimized against a fixed u_i, and then u is recomputed from the new
    # projections, which can raise the total. In practice the energy drops
    # sharply over the first few iterations and then drifts slowly upward, so
    # the final iterate is not the best one found. Keep the best.
    best_total_energy = float('inf')
    best_iteration = None
    best_v_mats = None
    # The solver's self-report for each modality's final projection. Kept
    # alongside the best iterate so the reported diagnostics describe the basis
    # actually returned, not whichever iteration the loop happened to stop on.
    retraction_diags = [{} for _ in range(n_modalities)]
    best_retraction_diags = None
    
    normalizing_weights = [1.0] * n_modalities
    domain_weights = [1.0] * n_modalities
    neg_weights = [1.0] * n_modalities

    from .similarity import similarity_is_scale_free
    try:
        _self_norm = similarity_is_scale_free(energy_type)
    except (ValueError, KeyError):
        _self_norm = False

    def _calibrate_weights(curr_v_mats, curr_u):
        for idx_m in range(n_modalities):
            u_m = curr_u[idx_m] if isinstance(curr_u, list) else curr_u
            scale_val = _calibrate_view_scale(curr_v_mats[idx_m], torch_mats[idx_m], u_m, energy_type)
            normalizing_weights[idx_m] = (1.0 / n_modalities if _self_norm
                                          else 1.0 / (scale_val * n_modalities + 1e-10))
            if negentropy_weight != 0.0:
                neg_e0 = abs(calculate_simlr_energy(
                    curr_v_mats[idx_m], torch_mats[idx_m], u_m, "negentropy").item())
                # For scale-free energies normalizing_weights[idx_m] is 1/M, so neg_weights
                # scales relative to |E0| / M.
                neg_weights[idx_m] = (scale_val * normalizing_weights[idx_m] / neg_e0
                                      if neg_e0 > 1e-12 else 0.0)
            if torch_domains is not None and torch_domains[idx_m] is not None:
                dom_e_raw = calculate_simlr_energy(curr_v_mats[idx_m], torch_mats[idx_m], u_m, "dat", lambda_val=domain_lambdas[idx_m], prior_matrix=torch_domains[idx_m]).item()
                if abs(dom_e_raw) > 1e-10:
                    # For scale-free energies normalizing_weights[idx_m] is 1/M, so domain_weights
                    # scales relative to |E0| / M.
                    domain_weights[idx_m] = abs(scale_val * normalizing_weights[idx_m]) / abs(dom_e_raw)
                else:
                    domain_weights[idx_m] = 1.0

    init_total_energy = 0.0
    if iterations == 0:
        init_projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
        init_u = compute_shared_consensus(
            init_projections, mixing_algorithm=mixing_algorithm, k=k,
            orthogonalize=orthogonalize_u, topology=topology, path_graph=path_graph)
        if nonneg_u:
            init_u = _nonneg_consensus(init_u)
        _calibrate_weights(v_mats, init_u)
        init_total_energy = sum(
            calculate_simlr_energy(v_mats[idx_m], torch_mats[idx_m],
                                   init_u[idx_m] if isinstance(init_u, list) else init_u,
                                   energy_type).item() * normalizing_weights[idx_m]
            + collapse_weight * float(correlation_penalty(torch_mats[idx_m] @ v_mats[idx_m].to(torch_mats[idx_m].dtype)))
            for idx_m in range(n_modalities)
        )

    mai_ema = torch.ones(n_modalities, dtype=orig_dtype) / n_modalities
    mai_mod_weights = torch.ones(n_modalities, dtype=orig_dtype) / n_modalities

    def _update_simlr_mai(projs, step_idx, max_steps):
        # Scores and gate shared with the deep and flow models
        # (`consensus.loo_mai_scores`, `mai_gated_target`); the schedule is
        # per sweep here: EMA 0.8, gate and transition both over half the run.
        if not dynamic_weights:
            return None
        from .consensus import loo_mai_scores, mai_gated_target
        with torch.no_grad():
            mai_t = torch.tensor(loo_mai_scores(projs, mai_metric, use_rank=use_rank_mai), dtype=orig_dtype)
            mai_ema.copy_(0.8 * mai_ema + 0.2 * mai_t)
            half = max(1.0, float(max_steps) * 0.5)
            target_w = mai_gated_target(mai_ema, 5.0 + 10.0 * min(1.0, max(0, step_idx) / half))
            uniform_w = torch.ones_like(target_w) / len(target_w)
            rho = min(1.0, max(0.0, float(step_idx) / half))
            mai_mod_weights.copy_((1.0 - rho) * uniform_w + rho * target_w)
            return mai_mod_weights

    torch_mats_64 = [x.double() for x in torch_mats]
    torch_domains_64 = (
        [d.double() if d is not None else None for d in torch_domains]
        if torch_domains is not None else None
    )
    _gauge = energy_needs_unit_columns(energy_type)

    def _view_oracles(i, u_i, _loop_sp):
        """Objective, smooth energy, exact gradient and projection for view
        `i`. Built per view so each closure captures its own view index, data
        and consensus: defined directly in the sweep loop they read the loop's
        `i` and data when *called*, so an optimizer that kept a callback past
        its `step` would silently evaluate another view."""
        # The smooth oracle every optimizer sees: one scalar function of V,
        # evaluated in float64, with its exact gradient by autograd. For
        # energies with a scale gauge (`energy_needs_unit_columns`) it is
        # the energy of the column-normalised basis -- otherwise
        # -sum|U'XV| falls without bound as ||V|| grows -- and that
        # normalisation is smooth away from a zero column. The rest of the
        # constraint set (sparsity, positivity, NSA-Flow retraction) is
        # applied once after the step (`simlr_sparseness` below), as in
        # proximal gradient.
        #
        # It used to score the fully *projected* point while returning a
        # tangent-projected gradient taken there, in float32: the slope a
        # line search tested was off by 2-14x, and the energy only moved in
        # float32 spacings (4.7e-10 at E = -6e-3). L-BFGS-B then built its
        # curvature pairs from rounding noise, and inputs 1e-8 apart reached
        # bases 0.07-1.1 apart after one sweep.
        x64 = torch_mats_64[i]
        u64 = u_i.double()
        dom64 = torch_domains_64[i] if torch_domains_64 is not None else None
        sm_i = smoothing_matrices[i] if smoothing_matrices else None
        _last_proj = [None, None, None]

        def _smooth_objective(v64):
            w64 = v64 / v64.norm(dim=0, keepdim=True).clamp_min(1e-300) if _gauge else v64
            e = calculate_simlr_energy(w64, x64, u64, energy_type) * normalizing_weights[i]
            if dom64 is not None:
                e = e + calculate_simlr_energy(w64, x64, u64, "dat", lambda_val=domain_lambdas[i],
                                               prior_matrix=dom64) * domain_weights[i]
            if negentropy_weight != 0.0:
                e = e + (calculate_simlr_energy(w64, x64, u64, "negentropy")
                         * negentropy_weight * neg_weights[i])
            if collapse_weight != 0.0:
                e = e + collapse_weight * correlation_penalty(x64 @ w64)
            return e

        def smooth_energy_fn(v_cand):
            with torch.no_grad():
                return float(_smooth_objective(v_cand.detach().double()))

        def _projection(v_cand):
            """The constraint set's projection P, exactly as applied after each step."""
            v_in = v_cand.to(orig_dtype)
            d = None if _probe_only else {}
            v_proj = simlr_sparseness(
                v_in, constraint_type=constraint_type,
                smoothing_matrix=sm_i,
                positivity=positivity, sparseness=_loop_sp,
                constraint_weight=constraint_weight,
                energy_type=energy_type, modality_index=i,
                retraction_diagnostics=d)
            _last_proj[0] = v_in.detach()
            _last_proj[1] = v_proj
            _last_proj[2] = d
            return v_proj

        def composite_energy_fn(v_cand):
            """The objective itself, E(P(v)): the smooth energy at the
            constrained point the step will actually produce. Line searches
            test trial points against this, in float64."""
            return smooth_energy_fn(_projection(v_cand))

        def smooth_gradient_fn(v_curr):
            """Descent direction ``-grad E(v)`` at `v_curr`, in its dtype."""
            with torch.enable_grad():
                v64 = v_curr.detach().double().requires_grad_(True)
                (g,) = torch.autograd.grad(_smooth_objective(v64), v64)
            return (-g).to(v_curr.dtype).contiguous()

        # The step direction may be tangent-projected for manifold
        # constraints: for an orthogonal projector P, <grad, P grad> =
        # |P grad|^2, so a line search along it still tests the true slope.
        # The oracle above stays the exact gradient.
        return smooth_energy_fn, smooth_gradient_fn, composite_energy_fn, _projection, _last_proj

    total_sweeps = (iterations + 1) if iterations > 0 else 0
    for raw_it in range(total_sweeps):
        is_warm_start = (raw_it == 0)
        it = raw_it - 1
        # Views whose optimizer step returned the iterate unchanged (the line
        # search found no decrease). Such a step has a fixed-point residual of
        # exactly 0 without the iterate being stationary, so it must not count
        # toward the certificate.
        zero_step_views = 0
        sweep_grad_map = []
        projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
        cur_mod_w = _update_simlr_mai(projections, max(0, it), max(1, iterations))
        u = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u, topology=topology, path_graph=path_graph, modality_weights=cur_mod_w)
        if nonneg_u:
            u = _nonneg_consensus(u)
        for i in range(n_modalities):
            u_i = u[i] if isinstance(u, list) else u
            # Local energy function that incorporates sparsification/retraction
            from .sparsification import _resolve_quantile
            _q_mod = _resolve_quantile(sparseness, modality_index=i)
            _loop_sp = (None if (_want_nonneg and k > 1 and _q_mod is not None
                                 and _q_mod < 1.0 - 1.0 / k)
                        else sparseness)
            smooth_energy_fn, smooth_gradient_fn, composite_energy_fn, _projection, _last_proj = \
                _view_oracles(i, u_i, _loop_sp)
            _exact = smooth_gradient_fn(v_mats[i])
            # Stationarity diagnostic of the iterate entering this sweep, on its
            # current face (`utils.stationarity_residual`). Reported, not used
            # to stop: convergence is the energy plateau rule.
            if _probe_only:
                gm_E, gm_off = 0.0, 0.0
            else:
                gm_E, gm_off = stationarity_residual(v_mats[i], -_exact, nonneg=_want_nonneg)
                sweep_grad_map.append(gm_E)
            total_grad = _exact.contiguous()
            v_before = v_mats[i].detach().clone()
            # Two oracles, the same for every optimizer: the objective E(P(v))
            # for line searches (passed to `step`), and the smooth part E(v)
            # with its exact gradient for quasi-Newton solvers, whose curvature
            # pairs need a gradient that is the derivative of the function.
            if hasattr(optimizer, "gradient_function"):
                optimizer.gradient_function = smooth_gradient_fn
                optimizer.smooth_energy_function = smooth_energy_fn
            v_updated = optimizer.step(i, v_mats[i], total_grad, composite_energy_fn)
            zero_step = bool(torch.equal(v_updated, v_before))

            # Apply final projection (reusing the exact candidate already
            # projected during line search when `v_updated` matches it).
            if (_last_proj[0] is not None
                    and v_updated.dtype == _last_proj[0].dtype
                    and v_updated.shape == _last_proj[0].shape
                    and torch.equal(v_updated, _last_proj[0])):
                retraction_diags[i] = {} if _last_proj[2] is None else dict(_last_proj[2])
                v_mats[i] = _last_proj[1].clone()
            else:
                retraction_diags[i] = {}
                v_mats[i] = simlr_sparseness(v_updated, constraint_type=constraint_type, smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None, positivity=positivity, sparseness=_loop_sp, constraint_weight=constraint_weight, energy_type=energy_type, modality_index=i, retraction_diagnostics=None if _probe_only else retraction_diags[i])
            resid = float(torch.linalg.norm(v_mats[i] - v_before)
                          / torch.linalg.norm(v_before).clamp_min(1e-300))
            retraction_diags[i]["fixed_point_residual"] = resid
            retraction_diags[i]["grad_map_E"] = gm_E
            retraction_diags[i]["off_support"] = gm_off
            if zero_step:
                retraction_diags[i]["zero_step"] = True
                zero_step_views += 1
            
        if is_warm_start:
            _calibrate_weights(v_mats, u)
            init_total_energy = sum(
                calculate_simlr_energy(v_mats[idx_m], torch_mats[idx_m],
                                       u[idx_m] if isinstance(u, list) else u,
                                       energy_type).item() * normalizing_weights[idx_m]
                + collapse_weight * float(correlation_penalty(torch_mats[idx_m] @ v_mats[idx_m].to(torch_mats[idx_m].dtype)))
                for idx_m in range(n_modalities)
            )
            best_v_mats = [v.clone() for v in v_mats]
            best_retraction_diags = [dict(d) for d in retraction_diags]
            continue
        
        total_energy = 0.0
        for i in range(n_modalities):
            u_i = u[i] if isinstance(u, list) else u
            total_energy += calculate_simlr_energy(v_mats[i], torch_mats[i], u_i, energy_type).item() * normalizing_weights[i]
            if collapse_weight != 0.0:
                total_energy += collapse_weight * float(correlation_penalty(torch_mats[i] @ v_mats[i].to(torch_mats[i].dtype)))
            
        energy_history.append(total_energy)
        grad_map_history.append(max(sweep_grad_map) if sweep_grad_map else float("nan"))

        iterate_is_degenerate = any(
            not bool((v.detach().abs() > 1e-12).any(dim=0).all()) for v in v_mats
        )
        if total_energy < best_total_energy and not iterate_is_degenerate:
            best_total_energy = total_energy
            best_iteration = it
            best_v_mats = [v.clone() for v in v_mats]
            best_retraction_diags = [dict(d) for d in retraction_diags]
        if iterate_is_degenerate:
            n_degenerate_iterates += 1
        
        plateaued = monitor.update(total_energy)

        if zero_step_views == n_modalities:
            # No view found a descent step. This is a line-search failure,
            # not a certificate: the fit stops, uncertified, at its best iterate.
            stop_reason, certificate = "no_descent", None
            converged_iter = it + 1
            if verbose: print(f"No descent step at iteration {it}")
            break

        # Convergence is the energy plateau rule, governed by `tol`. The face
        # residual in `grad_map_history` is a diagnostic only; whether it can
        # certify is open (it falls >= 2 decades where the energy settles, see
        # utils.stationarity_residual).
        if plateaued:
            if monitor.stop_reason != "converged":
                break
            stop_reason = "converged"
            certificate = "energy_plateau"
            converged_iter = it + 1
            if verbose:
                print(f"Plateau at iteration {it}: energy {total_energy}, "
                      f"certificate {certificate}")
            break

        prev_total_energy = total_energy
        
        if verbose and it % 10 == 0: print(f"Iteration {it}: Total Energy {total_energy}")
        
    # Return the lowest-energy iterate rather than whichever one the loop
    # happened to stop on.
    if best_v_mats is not None:
        v_mats = best_v_mats
        if best_retraction_diags is not None:
            retraction_diags = best_retraction_diags
    if sparseness is not None:
        from .sparsification import _apply_quantile_sparseness, _resolve_quantile, _unit_normalize_columns
        for idx_m in range(n_modalities):
            q_m = _resolve_quantile(sparseness, modality_index=idx_m)
            if q_m is not None and _want_nonneg and k > 1 and q_m < 1.0 - 1.0 / k:
                v_sp = _apply_quantile_sparseness(v_mats[idx_m], q_m, nonneg=True, only_if_denser=True)
                if energy_needs_unit_columns(energy_type):
                    v_sp = _unit_normalize_columns(v_sp)
                v_mats[idx_m] = v_sp.to(orig_dtype)
    if not np.isfinite(best_total_energy):
        best_total_energy = min(energy_history) if energy_history else init_total_energy
    if best_iteration is None:
        best_iteration = int(np.argmin(energy_history)) if energy_history else 0
    if _probe_only:
        return {"best_energy": best_total_energy, "v": v_mats, "energy": energy_history}

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
    # `topology` sets the training objective only: under "loo"/"graph" each
    # view was scored against a consensus of the other views. The reported
    # shared latent, the reconstruction weights and the prediction map are
    # always the all-view ("star") consensus, as on the deep and flow paths;
    # the per-view training targets are returned as `u_per_modality`.
    u_per_modality = None
    if topology != "star":
        u_targets = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u, topology=topology, path_graph=path_graph)
        u_per_modality = [_nonneg_consensus(t) if nonneg_u else t for t in u_targets]
    u_raw = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u, topology="star")
    u = _nonneg_consensus(u_raw) if nonneg_u else u_raw
    # Capture the linear map this consensus used, so that
    # `predict_shared_latent` can apply it to new data instead of
    # deriving a fresh one. See `_capture_consensus_anchor`.
    consensus_anchor = _capture_consensus_anchor(
        projections, mixing_algorithm, k, orthogonalize_u, "star", None)
    consensus_stats = _capture_consensus_stats(
        projections, u_raw, mixing_algorithm, consensus_anchor, "star")
    # The general frozen map covers what the star statistics do not
    # ("loo"/"graph" topologies, an anchorless SVD/PCA/ICA consensus); without
    # it those predictions were re-derived from each test batch.
    from .consensus import capture_consensus_map
    consensus_map = capture_consensus_map(projections, u_raw)
    
    v_summaries = [orthogonality_summary(v) for v in v_mats]
    
    # Compute reconstruction weights W_i such that X_i approx U @ W_i
    # W_i = pinv(U) @ X_i
    w_mats = []
    for i, x in enumerate(torch_mats):
        u_i = u[i] if isinstance(u, list) else u
        # A pinv failure (non-finite u) propagates; it used to store W = 0, so
        # every later reconstruction was zero without any sign of failure.
        w_mats.append(torch.linalg.pinv(u_i) @ x)

    return {
        "u": u, "u_per_modality": u_per_modality, "v": v_mats, "w": w_mats, "energy": energy_history,
        # Per-view first-layer projections of the (preprocessed) training
        # data, the same quantity `predict_simlr` returns as "latents".
        "latents": [x @ v.to(x.dtype) for x, v in zip(torch_mats, v_mats)],
        "normalizing_weights": normalizing_weights, 
        "domain_weights": domain_weights, 
        "converged_iter": converged_iter,
        "stop_reason": stop_reason,
        "convergence": {**monitor.report(), "stop_reason": stop_reason,
                        "converged": stop_reason == "converged", "certificate": certificate,
                        "grad_map": (grad_map_history[-1] if grad_map_history else None)},
        "certificate": certificate,
        "converged": certificate is not None,
        "n_degenerate_iterates": n_degenerate_iterates,
        "grad_map": (grad_map_history[-1] if grad_map_history else None),
        "grad_map_history": grad_map_history,
        "energy_start": (energy_history[0] if energy_history else init_total_energy),
        "energy_reduction": ((energy_history[0] - best_total_energy)
                             if energy_history else 0.0),
        "v_orthogonality": v_summaries,
        "best_energy": best_total_energy,
        "best_iteration": best_iteration,
        "v_retraction": retraction_diags,
        "consensus_anchor": consensus_anchor,
        "consensus_stats": consensus_stats,
        "consensus_map": consensus_map,
        "mixing_algorithm": mixing_algorithm,
        "orthogonalize_u": orthogonalize_u,
        "nonneg_u": bool(nonneg_u),
        "topology": topology,
        "path_graph": path_graph,
        "energy_type": energy_type,
        "scale_list": scale_list,
        "provenance_list": provenance_list,
        "consolidated": consolidate,
        "sparseness": sparseness,
        "random_state": random_state,
        "initialization_type": initialization_type,
        "modality_weights": mai_mod_weights.detach().cpu().numpy(),
        "mai": mai_ema.detach().cpu().numpy(),
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
        Additional parameters passed to the `simlr` function (e.g., energy_type, nsa_w, optimizer_type).

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
    # A failure propagates: the same consensus just succeeded during the fit.
    result = compute_shared_consensus(
        projections, mixing_algorithm=mixing_algorithm, k=k,
        orthogonalize=orthogonalize_u, training=True,
        topology=topology, path_graph=path_graph)
    if not isinstance(result, tuple) or len(result) != 2:
        return None
    anchor = result[1]
    return anchor.detach().clone() if isinstance(anchor, torch.Tensor) else None


def _capture_consensus_stats(projections: List[torch.Tensor],
                             u_raw: Union[torch.Tensor, List[torch.Tensor]],
                             mixing_algorithm: str,
                             anchor: Optional[torch.Tensor],
                             topology: str) -> Optional[Dict[str, Any]]:
    """Capture training moments so out-of-sample prediction is batch-size (N=1) invariant."""
    if topology != "star" or isinstance(u_raw, list) or not projections:
        return None
    # A failure propagates; returning None made prediction fall back to the batch-dependent consensus.
    from .utils import align_column_signs
    proj_rms = []
    norm_projs = []
    for p in projections:
        p_safe = torch.nan_to_num(p.detach().double(), nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(torch.norm(p_safe, p="fro") / np.sqrt(max(1, p_safe.shape[0])))
        proj_rms.append(rms)
        norm_projs.append(p_safe / (rms + 1e-8) if rms > 1e-6 else torch.zeros_like(p_safe))
    if mixing_algorithm in ("svd", "pca", "ica"):
        if anchor is None:
            return None
        u_pre = torch.cat(norm_projs, dim=1) @ anchor.detach().double()
        view_signs = None
    else:
        aligned = align_column_signs(norm_projs)
        view_signs = []
        for p_orig, p_al in zip(norm_projs, aligned):
            dots = torch.sum(p_orig * p_al, dim=0, keepdim=True)
            view_signs.append(torch.where(dots < 0.0, -torch.ones_like(dots), torch.ones_like(dots)))
        u_pre = torch.mean(torch.stack(aligned), dim=0)
    mu_k = u_pre.mean(dim=0, keepdim=True)
    u_pre_c = u_pre - mu_k
    lstsq_res = torch.linalg.lstsq(u_pre_c, u_raw.detach().double())
    r_k = lstsq_res.solution
    sv = torch.linalg.svdvals(u_pre_c)
    cond = float((sv[0] / sv[-1].clamp_min(1e-15)).item()) if sv.numel() > 0 else 1.0
    rank = int(lstsq_res.rank.item()) if lstsq_res.rank.numel() else int((sv > 1e-10 * sv[:1]).sum().item())
    return {
        "proj_rms": proj_rms,
        "view_signs": view_signs,
        "mu_k": mu_k,
        "r_k": r_k,
        "rank": rank,
        "condition_number": cond,
    }


def _apply_consensus_stats(projections: List[torch.Tensor],
                           stats: Dict[str, Any],
                           mixing_algorithm: str,
                           anchor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Apply captured training consensus statistics row-wise (invariant to N_test)."""
    # A failure propagates; returning None made prediction fall back to the batch-dependent consensus.
    dtype = projections[0].dtype
    device = projections[0].device
    proj_rms = stats["proj_rms"]
    norm_projs = []
    for p, rms in zip(projections, proj_rms):
        p_safe = torch.nan_to_num(p.double(), nan=0.0, posinf=0.0, neginf=0.0)
        norm_projs.append(p_safe / (rms + 1e-8) if rms > 1e-6 else torch.zeros_like(p_safe))
    if mixing_algorithm in ("svd", "pca", "ica"):
        if anchor is None:
            return None
        u_pre = torch.cat(norm_projs, dim=1) @ anchor.to(device=device, dtype=torch.float64)
    else:
        view_signs = stats.get("view_signs")
        if not view_signs:
            return None
        aligned = [p * s.to(device=device, dtype=torch.float64) for p, s in zip(norm_projs, view_signs)]
        u_pre = torch.mean(torch.stack(aligned), dim=0)
    mu_k = stats["mu_k"].to(device=device, dtype=torch.float64)
    r_k = stats["r_k"].to(device=device, dtype=torch.float64)
    return ((u_pre - mu_k) @ r_k).to(dtype=dtype)


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
    torch_mats = _check_views(data_matrices, n_views=len(simlr_result['v']) if simlr_result.get('v') is not None else None)

    # 2. Preprocess with the training statistics
    torch_mats = _scale_like_training(torch_mats, simlr_result)

    v_mats = simlr_result['v']
    mixing_alg = simlr_result.get('mixing_algorithm', 'svd')
    simlr_result.get('orthogonalize_u', False)
    v_mats[0].shape[1]
    
    # 2. Project to shared space
    projections = [x @ v.to(x.dtype) for x, v in zip(torch_mats, v_mats)]
    
    # 3. Compute consensus U using the original mixing settings
    anchor = simlr_result.get('consensus_anchor')
    stats = simlr_result.get('consensus_stats')
    u_new = None
    if stats is not None and (mixing_alg not in ("svd", "pca", "ica") or anchor is not None):
        u_new = _apply_consensus_stats(projections, stats, mixing_alg, anchor)
    if u_new is None and simlr_result.get('consensus_map') is not None:
        from .consensus import apply_consensus_map
        u_new = apply_consensus_map(projections, simlr_result['consensus_map'])
    if u_new is None:
        raise ValueError("simlr_result carries neither consensus statistics nor a consensus "
                         "map (a result from before 0.3.0); refit it -- re-deriving the "
                         "consensus from the prediction batch made it batch-dependent.")
    if simlr_result.get('nonneg_u', False):
        u_new = _nonneg_consensus(u_new)
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
        raise ValueError("simlr_result has no learned reconstruction weights 'w'; refit it.")
    
    w_mats = simlr_result['w']
    reconstructions = []
    for i, w in enumerate(w_mats):
        u_i = u[i] if isinstance(u, list) else u
        x_pred = u_i @ w.to(u_i.dtype)
        reconstructions.append(x_pred)
    return reconstructions

def predict_simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                  simlr_result: Dict[str, Any]) -> Dict[str, Any]:
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
        If the result has no learned reconstruction weights or no frozen consensus.
    TypeError
        If the inputs are not valid formats.
    """
    # 1. Contract Validation
    torch_mats = _check_views(data_matrices, n_views=len(simlr_result['v']) if simlr_result.get('v') is not None else None)

    torch_mats = _scale_like_training(torch_mats, simlr_result)

    if 'model' in simlr_result:
        # Fallback to predict_deep logic if a deep model is detected
        from .deep import predict_deep
        return predict_deep(data_matrices, simlr_result)
    
    # For standard SIMLR:
    u_new = predict_shared_latent(data_matrices, simlr_result)
    
    reconstructions = reconstruct_from_learned_maps(u_new, simlr_result)
    
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

def decompose_energy(data_matrices: List[Union[torch.Tensor, np.ndarray]], simlr_result: Dict[str, Any], energy_type: str = DEFAULT_ENERGY_TYPE) -> Dict[str, Any]:
    """
    Decompose the SiMLR objective energy across modalities and features.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    simlr_result : Dict[str, Any]
        The result dictionary from a fitted SiMLR model.
    energy_type : str, default=DEFAULT_ENERGY_TYPE ("nc")
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
