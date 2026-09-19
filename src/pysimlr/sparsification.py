import warnings
import torch
import numpy as np
from typing import Optional, List, Union, Dict, Any
from .utils import safe_svd

from .nsa_backend import load_nsa_flow, load_polar_factor

#: Retraction weight used when a constraint string does not name one. The
#: backend's cost is driven by `w`, not by problem size: w=0.5 converges in
#: ~22 iterations (~6 ms) with a Stiefel defect near 1e-1, whereas w=0.99 needs
#: ~1435 iterations (~559 ms) for 5.7e-5. Moderate w also proved more
#: reproducible -- at high w the solution approaches a combinatorial vertex and
#: small data changes flip which vertex is selected.
NSA_DEFAULT_W = 0.5

#: Largest retraction weight handed to the backend, for every constraint family.
#: At w=1 the fidelity term drops out entirely: the solver no longer stays near
#: the candidate, the scale of the returned Y is unconstrained, and -- the
#: decisive point -- every scaled Stiefel matrix is optimal, so *nothing selects
#: among clusterings*. The answer is then arbitrary rather than merely rescaled.
#: `parse_constraint` assigns weight 1.0 to every hard-manifold constraint, so
#: "Stiefel" and "Grassmann" hit exactly that case and must be capped too.
#:
#: Measured over 10 paired seeds, holding w at 0.99 rather than 1.0 raised
#: subspace recovery from 0.842 to 0.876 on one design and 0.720 to 0.738 on
#: another. Recovery keeps improving as w falls further (0.885 at w=0.9 on the
#: first design), and the backend's own experiments found moderate w better than
#: aggressive w on both accuracy and reproducibility, because a high w pins the
#: solution near a combinatorial vertex where small data changes flip which
#: vertex is chosen. The cap is therefore set below the knee rather than just
#: inside it.
#:
#: The cost is that ``V'V = I`` is no longer exact for a hard manifold
#: constraint -- the normalized defect settles near 1e-2 rather than 0. A caller
#: who needs exactness should orthogonalize without the non-negative solver,
#: since exact orthogonality *and* non-negativity is disjoint supports, i.e. a
#: clustering, which is the degenerate case this cap exists to avoid.
NSA_MAX_W = 0.95

#: Smallest retraction weight handed to the backend. At w=0 the constraint term
#: drops out and the solve is a no-op, so a caller who selected the backend
#: gets nothing; the branches here only call the backend when they want some
#: retraction.
NSA_MIN_W = 1e-3


class _LazyNsaFlowOrth:
    """Defers the NSA-Flow import until a retraction actually needs it.

    Compares equal to None (``is not None`` still discriminates) by reporting
    truthiness from the resolved backend, so existing call sites keep working
    while the import stays off the package-import path.
    """

    def __bool__(self):
        return load_nsa_flow() is not None

    def __call__(self, *args, **kwargs):
        fn = load_nsa_flow()
        if fn is None:
            raise RuntimeError("No NSA-Flow backend is installed.")
        return fn(*args, **kwargs)


nsa_flow_orth = _LazyNsaFlowOrth()

def _assignment_indicator(m: torch.Tensor) -> torch.Tensor:
    """
    Binary indicator maximizing ``sum(m * I)`` with at most one entry per row/column.

    Solves the linear assignment problem exactly via
    :func:`scipy.optimize.linear_sum_assignment`, falling back to a greedy
    column sweep if SciPy is unavailable.

    Parameters
    ----------
    m : torch.Tensor
        Score matrix of shape (rows, cols).

    Returns
    -------
    torch.Tensor
        Matrix of the same shape as `m` containing only 0.0 and 1.0, with at
        most one non-zero per row and at most one per column.
    """
    indicator = torch.zeros_like(m)
    m_safe = torch.nan_to_num(m, nan=float('-inf'), posinf=float('inf'), neginf=float('-inf'))
    try:
        from scipy.optimize import linear_sum_assignment
        # linear_sum_assignment minimizes, so negate to maximize sum(m * I).
        cost = -torch.nan_to_num(m_safe, nan=0.0, posinf=1e30, neginf=-1e30)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        indicator[torch.as_tensor(rows, dtype=torch.long),
                  torch.as_tensor(cols, dtype=torch.long)] = 1.0
    except Exception:
        # Greedy fallback: one pass over columns, claiming the best free row.
        row_used = torch.zeros(m_safe.shape[0], dtype=torch.bool, device=m.device)
        for j in range(m_safe.shape[1]):
            available = m_safe[:, j].clone()
            available[row_used] = float('-inf')
            max_val, selected_row = torch.max(available, dim=0)
            if torch.isfinite(max_val):
                indicator[selected_row, j] = 1.0
                row_used[selected_row] = True
    return indicator


def _nsa_retract(v: torch.Tensor,
                 w: float,
                 nonneg: bool,
                 max_iter: int = 5000,
                 max_w: float = None,
                 diagnostics: Optional[dict] = None) -> Optional[torch.Tensor]:
    """
    Retract `v` toward a non-negative, near-orthogonal basis via NSA-Flow.

    Parameters
    ----------
    v : torch.Tensor
        Candidate basis of shape (features, components).
    w : float
        Retraction weight. Larger values drive the Stiefel defect lower at
        steeply increasing cost, and reduce reproducibility; see
        `NSA_DEFAULT_W`.
    nonneg : bool
        Whether to constrain the result to be non-negative.
    max_iter : int, default=5000
        Iteration cap handed to the solver. The solver typically stops far
        short of this; `stop_reason` reports which condition ended it.
    max_w : float, optional
        Upper bound applied to `w`, defaulting to `NSA_MAX_W`. Every constraint
        family uses that default; the parameter exists so a caller can bound a
        single solve more tightly, not so the cap can be lifted.
    diagnostics : dict, optional
        If given, updated in place with what the solver reported: the stopping
        rule, the stationarity certificate, the effective rank, the scale
        drift, which fidelity it chose and the target's negative mass. These
        are the quantities that say whether a retraction is trustworthy, and
        they were previously read only to extract `Y` and then discarded.

    Returns
    -------
    torch.Tensor or None
        The retracted basis in `v`'s dtype, or None if no backend is available
        or the result fails :func:`_usable_retraction`.

    Notes
    -----
    Solved in float64 regardless of `v`'s dtype and cast back. The tolerance
    floor scales with precision -- float32 bottoms out near 1e-6 rather than
    1e-9 -- and the solve is a few milliseconds either way at these sizes, so
    there is no reason to accept the coarser answer.

    `align` is left at its default of False: its justification rested on a
    functional the solver no longer uses, and the gain measured under the old
    one has not been re-established.

    The candidate is rescaled to unit RMS column norm before the solve and the
    scale is restored afterwards. The solver's fidelity term is not
    scale-invariant, so without this the answer depends on the caller's
    arbitrary column scale. On one 10x3 candidate whose column norms differed
    by 50x, the Stiefel defect measured 0.828 at unit scale but 2.449 at both
    1e+6 and 1e+8, converging to a visibly different solution (``||Y||`` 95.4
    rather than 71.3); on a better-conditioned candidate the solver instead hit
    its iteration cap at 1e+8. Under the gauge fix the stop reason, defect and
    ``||Y||`` are identical across twenty orders of magnitude of input scale,
    and identical to the ungauged result at unit scale -- so this changes
    nothing in the normal regime and removes a failure mode at the extremes.
    The rescaling is global rather than per column: normalizing each column
    separately would reweight them against each other inside the fidelity term,
    which changes the problem rather than just its gauge.
    """
    fn = load_nsa_flow()
    if fn is None:
        return None

    w = _clamp_retraction_weight(w, max_w=max_w)
    v_detached = v.detach()
    n_cols = max(1, v_detached.shape[1])
    scale = float(torch.linalg.norm(v_detached.double()) / (n_cols ** 0.5))
    if not np.isfinite(scale) or scale <= 0.0:
        # An all-zero or non-finite candidate; the backend rejects both, and
        # there is no gauge to normalize by.
        return None
    target = v_detached.double() / scale

    # This call is the proximal operator of the SiMLR outer loop:
    #     prox(z) = argmin_{Y >= 0} (1-w) ||Y - z||^2 / ||z||^2 + w Dtilde(Y)
    # so the fidelity MUST be the entrywise Euclidean distance to z.  Left at
    # "auto", nsa_flow switches to its sign-blind subspace fidelity whenever the
    # target has negative mass > 1%, which a post-gradient-step iterate always
    # does; that operator is invariant under X0 -> X0 M for any invertible M,
    # discards where the step moved within the span, and is not a prox of
    # anything -- and the switch is a threshold on the iterate, so it can flip
    # between iterations.  "subspace" is the right choice for one-shot basis
    # recovery from a signed initialiser; it is the wrong one inside a loop.
    # mode="anchored" is what "no k" already implies; saying it makes the call
    # immune to changes in nsa_flow's auto-dispatch.
    try:
        result = fn(target, w=float(w), mode="anchored", fidelity="anchor",
                    nonneg=bool(nonneg), max_iter=int(max_iter))
    except Exception:
        return None

    candidate = None
    if hasattr(result, 'get'):
        candidate = result.get('V') or result.get('Y')
    if candidate is None:
        candidate = getattr(result, 'V', None) or getattr(result, 'Y', None)
    if diagnostics is not None:
        diagnostics.update(_retraction_diagnostics(result, w))
    if candidate is None:
        return None

    _warn_if_unconverged(result)

    if nonneg and isinstance(candidate, torch.Tensor):
        candidate = torch.clamp_min(candidate, 0.0)

    candidate = (candidate * scale).to(v.dtype)
    return candidate if _usable_retraction(candidate, v, nonneg=nonneg) else None


def _resolve_column_sign_gauge(z: torch.Tensor) -> torch.Tensor:
    """
    Fix the per-column sign gauge so the non-negative constraint is least destructive.

    Every SiMLR energy is even in each column of ``V`` (they act on ``X V``
    through squares, covariances or a regression whose coefficient absorbs the
    sign), so ``V -> V diag(s)``, ``s in {-1,+1}^k``, is a symmetry of ``E``.
    Choosing ``s`` is therefore a gauge choice, not a change of the iterate, and
    the prox applied to ``z diag(s)`` is still the prox step of the outer loop
    -- on the quotient by that symmetry.

    Pick, per column, the sign under which more of the column's mass survives
    projection onto the orthant: flip column ``j`` iff
    ``||max(0, -z_j)|| > ||max(0, z_j)||``. A column that is entirely
    non-positive becomes entirely non-negative, so it is never zeroed by the
    constraint for a reason that was only ever a sign convention. Columns of
    mixed sign keep their orientation and are projected honestly: a feature
    that loads negatively is zeroed, not reflected. This replaces the old
    ``abs(v)`` fallback, which is not a symmetry of anything and reports a
    negatively fitted feature as a positive one.
    """
    pos = torch.linalg.vector_norm(z.clamp_min(0.0), dim=0)
    neg = torch.linalg.vector_norm((-z).clamp_min(0.0), dim=0)
    sign = torch.where(neg > pos, -torch.ones_like(pos), torch.ones_like(pos))
    return z * sign.unsqueeze(0)


def _retraction_candidate(v_signed: torch.Tensor, v_rectified: torch.Tensor,
                          nonneg: bool) -> torch.Tensor:
    """
    Choose what to hand the retraction: the signed iterate, or the rectified one.

    When the solver will enforce non-negativity itself, give it the signed
    candidate. Rectifying first throws away the sign information the solver
    would otherwise use, and the backend's sign-blind subspace fidelity exists
    precisely to consume a signed target: it anchors to ``range(X0)`` rather
    than to ``X0`` entrywise, so arbitrary column signs cost nothing.

    Measured over 10 seeds against a known non-negative basis, recovery of the
    true basis by what was handed to the solver:

    ============================  ==========  ===========
    candidate                     disjoint    overlapping
    ============================  ==========  ===========
    ``abs(v)``                    0.9162      0.8112
    projected ``v``               0.9467      0.8774
    signed ``v``                  **0.9930**  **0.9526**
    ============================  ==========  ===========

    with the normalized Stiefel defect falling from 0.53 to 0.11 and 0.64 to
    0.42 respectively. The signed candidate also beat fitting from the data
    outright on overlapping supports (0.9526 against 0.9032).

    This only holds with a backend that implements the sign-blind fidelity.
    Against the older entrywise-only fidelity the same comparison ran the other
    way -- signed 0.745 against 0.916 for ``abs`` -- because anchoring a
    non-negative solution to a signed target charges it for negative entries it
    cannot reach; those charges are constant, so they do not steer the solve and
    the optimum degenerates toward ``max(0, X0)``. Older backends therefore get
    the rectified candidate, detected by whether the backend accepts the
    ``fidelity`` keyword.

    When the solver is not enforcing non-negativity there is nothing to choose:
    the rectified candidate already carries whatever sign the caller asked for.
    """
    # Retained for import compatibility.  The prox always receives the signed
    # iterate now (see simlr_sparseness); the fidelity is fixed to "anchor"
    # explicitly rather than inferred from the backend's signature.
    return v_rectified if not nonneg else v_signed


def _backend_has_sign_blind_fidelity() -> bool:
    """Whether the installed backend selects a sign-blind fidelity for signed
    targets. Cached, since it is a static property of the installed version."""
    global _SIGN_BLIND_FIDELITY
    if _SIGN_BLIND_FIDELITY is None:
        fn = load_nsa_flow()
        if fn is None:
            _SIGN_BLIND_FIDELITY = False
        else:
            try:
                import inspect
                _SIGN_BLIND_FIDELITY = (
                    'fidelity' in inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                _SIGN_BLIND_FIDELITY = False
    return _SIGN_BLIND_FIDELITY


_SIGN_BLIND_FIDELITY = None


def _clamp_retraction_weight(w: float, max_w: float = None) -> float:
    """
    Clamp a retraction weight into the range the solver is defined on.

    At ``w = 0`` the constraint term vanishes and the solve is a no-op, so a
    caller who selected the backend gets nothing. At ``w = 1`` the fidelity
    term vanishes instead; that is degenerate for a soft constraint but is
    exactly what a hard manifold constraint wants, so `max_w` decides whether
    it is allowed through. The weight also arrives from user-supplied strings
    such as ``"orthox1.5"``, which are not restricted to [0, 1] at all.

    See `NSA_MAX_W` for the measured cost of ``w = 1`` on the soft family.
    """
    w = float(w)
    if not np.isfinite(w):
        return NSA_DEFAULT_W
    upper = NSA_MAX_W if max_w is None else float(max_w)
    return float(min(max(w, NSA_MIN_W), upper))


#: Solver-reported fields worth keeping. `stop_reason` and `converged` say
#: whether the solve finished, `grad_map` how close to stationary it got,
#: `effective_rank` whether the basis collapsed, `scale_ratio` how far the
#: scale drifted, and `fidelity_mode`/`target_negative_mass` which notion of
#: closeness the backend chose and why.
_RETRACTION_DIAGNOSTIC_FIELDS = (
    'stop_reason', 'converged', 'certificate', 'grad_map', 'tol', 'iters',
    'n_grad', 'n_energy', 'defect', 'defect_D', 'defect_Cg',
    'effective_rank', 'scale_ratio', 'fidelity_mode', 'target_negative_mass',
    'fidelity', 'energy', 'energy_start', 'energy_reduction', 'seconds',
    'lobe_overlap', 'consolidated', 'optimizer',
)


def _retraction_diagnostics(result, w: float) -> Dict[str, Any]:
    """
    Extract the solver's self-report, tolerating fields a version may not have.

    Returns a plain dict so it can be carried in a result payload without
    keeping a reference to the backend's own object.
    """
    def field(name):
        if hasattr(result, 'get'):
            return result.get(name)
        return getattr(result, name, None)

    out = {'w': float(w)}
    for name in _RETRACTION_DIAGNOSTIC_FIELDS:
        value = field(name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.tolist()
        out[name] = value
    return out


def _warn_if_unconverged(result) -> None:
    """
    Warn when the solver stopped on its iteration cap rather than converging.

    nsa_flow >= 3 sets ``converged`` only when a certificate was earned
    (``certificate`` is ``"stationary"`` or ``"numerical_floor"``); a
    ``line_search`` or ``max_iter`` stop is not converged.  A capped solve
    still returns a structurally valid basis, so the result is used rather
    than discarded -- but silently accepting an unconverged retraction is how
    a degenerate basis previously reached the caller unnoticed. The
    message is constant, so Python's default filter reports it once per call
    site instead of once per SiMLR iteration.
    """
    def _field(name):
        if hasattr(result, 'get'):
            return result.get(name)
        return getattr(result, name, None)

    converged = _field('converged')
    stop_reason = _field('stop_reason')
    if converged is None and stop_reason is not None:
        converged = stop_reason in ('grad_map', 'plateau')
    if converged is False:
        import warnings
        warnings.warn(
            "NSA-Flow stopped without a convergence certificate "
            f"(stop_reason={stop_reason!r}, |Gmap|={_field('grad_map')}); the "
            "retraction may be far from the constraint set. Raise max_iter or "
            "loosen the retraction weight.",
            RuntimeWarning, stacklevel=3,
        )


def _unit_normalize_columns(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Scale each column of `v` to unit L2 norm, leaving all-zero columns alone.

    Parameters
    ----------
    v : torch.Tensor
        Matrix of shape (features, components).
    eps : float, default=1e-12
        Columns with norm at or below this are returned unchanged.

    Returns
    -------
    torch.Tensor
        A matrix of the same shape with unit-norm (or zero) columns.
    """
    norms = torch.linalg.vector_norm(v, dim=0, keepdim=True)
    return v / torch.where(norms > eps, norms, torch.ones_like(norms))


def _usable_retraction(candidate: Optional[torch.Tensor],
                       reference: torch.Tensor,
                       nonneg: bool = False) -> bool:
    """
    Whether a retraction backend's output can be accepted in place of `reference`.

    The NSA-Flow backend returns a result dict whose ``Y`` is not guaranteed to
    be a usable basis, and a zero basis destroys the model silently: every
    projection becomes zero, the latent carries no signal, and downstream
    R-squared is exactly 0 while nothing raises. Checking only
    ``Y is not None``, as this code previously did, does not catch that. That
    is the failure this guard exists for, and it is worth keeping regardless of
    which backend version is installed.

    History, since the specific collapse it was written against no longer
    reproduces. An all-zero ``Y`` was observed for a 5x2 non-negative input at
    ``w`` of 0.1 and 0.3 with ``apply_nonneg="hard"``, on the backend version
    current when this guard was added; it is what took SiMLR's Diabetes
    R-squared from +0.3932 to +0.0000 with min and max identical across all 160
    runs. On 2.7.0 the same call returns ``||Y||`` of 2.02 and 1.55 with no zero
    entries. The likely mechanism is the signed-target pathology the backend
    since fixed: a non-negative ``Y`` anchored entrywise to a signed target is
    charged for negative entries it cannot reach, and the optimum degenerates
    toward ``max(0, X0)``. 2.7.0 routes a signed target to its sign-blind
    subspace fidelity instead, which is why the collapse is gone rather than
    merely rarer.

    Parameters
    ----------
    candidate : torch.Tensor or None
        The backend's proposed basis.
    reference : torch.Tensor
        The input it was asked to retract, used for shape and rank comparison.

    Returns
    -------
    bool
        True if `candidate` is finite, non-degenerate, correctly shaped, and
        does not lose rank relative to `reference`.
    """
    if candidate is None:
        return False
    if candidate.shape != reference.shape:
        return False
    if not bool(torch.isfinite(candidate).all()):
        return False
    if float(torch.linalg.vector_norm(candidate)) <= 0.0:
        return False
    # A zero column means a component with no loading at all.
    if bool((torch.linalg.vector_norm(candidate, dim=0) <= 0.0).any()):
        return False
    # Rank loss is judged against the rank the FEASIBLE SET can support near
    # the input, not against the input itself.  The prox now receives the
    # signed iterate; a full-rank signed square matrix (e.g. a 5x5 orthogonal
    # Q) has no full-rank non-negative neighbour in general -- clamp(Q) is
    # typically rank 4 -- so demanding rank(Y) >= rank(z) would reject the
    # correct prox output.  The old code passed this check only because it
    # compared against the already-clamped candidate.
    rank_ref = reference.clamp_min(0.0) if nonneg else reference
    try:
        if int(torch.linalg.matrix_rank(candidate)) < int(torch.linalg.matrix_rank(rank_ref)):
            return False
    except RuntimeError:
        return False
    return True


def _svd_polar(v: torch.Tensor) -> torch.Tensor:
    """
    Polar retraction onto the Stiefel manifold.

    Prefers the Sylvester-based `polar_factor` from NSA-Flow (which avoids
    SVD/QR in solver inner loops and provides smooth derivatives per the
    NSA-Flow guide), falling back to SVD if unavailable.
    """
    polar_fn = load_polar_factor()
    if polar_fn is not None:
        try:
            return polar_fn(v)
        except Exception:
            pass
    u, _, vh = safe_svd(v, full_matrices=False)
    return u @ vh


def apply_positivity(v: torch.Tensor, positivity: str) -> torch.Tensor:
    """
    Constrain the sign of `v` under one of two distinct semantics.

    Parameters
    ----------
    v : torch.Tensor
        Basis matrix of shape (features, components).
    positivity : str
        - ``"positive"`` / ``"hard"``: project onto the non-negative orthant,
          falling back to reflection (``abs(v)``) only for the matrices where
          projecting would zero a column or drop the rank.
        - ``"negative"``: the same, mirrored to non-positive.
        - ``"nonnegative"`` / ``"project"``: project onto the non-negative
          orthant, resolving column sign ambiguity first and then clamping.
        - anything else (e.g. ``"either"``): returned unchanged.

    Returns
    -------
    torch.Tensor
        A matrix of the same shape satisfying the requested sign.

    Notes
    -----
    The two semantics are not interchangeable, and which is appropriate depends
    on the shape of `v`.

    *Reflection* is information-preserving in the narrow sense that it is a
    bijection on magnitudes, so no feature is dropped and the rank of `v` is
    unchanged. Its cost is that it is not a projection: a feature whose fitted
    association is strongly negative is reported as a strong *positive*
    contributor, asserting the opposite of what was fitted. It also damages
    orthogonality roughly twice as much as clamping (see
    :func:`_project_nonnegative` for the measurements).

    *Projection* (``"nonnegative"``) is the honest sign constraint: a feature
    that loads negatively is set to zero rather than sign-flipped. Its cost is
    structural, and it is severe when the number of features is close to the
    number of components. Clamping can zero an entire row -- deleting a feature
    from every component -- and can reduce the rank of `v`. On a 5-feature view
    with k=5 this was measured to take the basis from rank 5 (effectively a
    permutation matrix, first-layer R-squared 0.39) to rank 4 with a fully zero
    row and R-squared 0.00.

    ``"positive"`` therefore projects and checks, reflecting only the matrices
    where the projection actually degenerates -- which across 200 random
    matrices per shape happened only at ``p == k`` (11 of 200 at (5, 5)) and
    never once for ``p > k``. ``"nonnegative"`` projects unconditionally, with
    no fallback, for callers who would rather see the degeneracy than have it
    silently repaired; :func:`positivity_diagnostics` reports zero rows, zero
    columns and rank loss.

    Reflection was the unconditional behaviour of ``"positive"`` through the
    published results, but switching to projection does not move them: across
    160 real-data cells, 159 were numerically identical and predictive accuracy
    changed by a mean of -0.00003.

    That is not because the constraint is inert. Over a 100-iteration Diabetes
    run, 45% of the calls received a candidate containing negative entries
    (15.6% of entries on average) -- but those negatives are small residuals
    left by the non-negative retraction, so reflecting and clamping differ by a
    relative norm of only 0.0005. The initializer is what made them small: the
    damaging case was a *signed* candidate, where reflection took a basis of
    normalized Stiefel defect 0.00 to 1.60. With the initializer fitting a
    non-negative basis from the data, that case no longer arises here, and this
    change removes the remaining exposure to it rather than correcting a live
    error.

    Examples
    --------
    >>> import torch
    >>> v = torch.tensor([[1.0, -3.0], [-0.5, -2.0]])
    >>> apply_positivity(v, "positive")     # projects; col 1 flips, then clamps
    tensor([[1., 3.],
            [0., 2.]])
    >>> apply_positivity(v, "nonnegative")  # same, with no degeneracy fallback
    tensor([[1., 3.],
            [0., 2.]])

    Reflection is used only where projecting would degenerate. Here column 1
    is already zero, so clamping it would leave a zero column and the whole
    matrix falls back to ``abs``:

    >>> w = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    >>> apply_positivity(w, "positive")
    tensor([[1., 0.],
            [2., 0.],
            [3., 0.]])
    """
    if positivity in ('positive', 'hard'):
        return _project_nonnegative(v, sign=1.0)
    if positivity == 'negative':
        return -_project_nonnegative(-v, sign=1.0)
    if positivity in ('nonnegative', 'nonneg', 'project'):
        return _sign_resolved_clamp(v, sign=1.0)
    if positivity in ('nonpositive', 'project_negative'):
        pos_mass = torch.sum(torch.clamp(v, min=0.0), dim=0, keepdim=True)
        neg_mass = torch.sum(torch.clamp(-v, min=0.0), dim=0, keepdim=True)
        v = torch.where(pos_mass > neg_mass, -v, v)
        return torch.clamp(v, max=0.0)
    return v


def _sign_resolved_clamp(v: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
    """
    Project onto the non-negative orthant, resolving each column's sign first.

    A basis column's overall sign is arbitrary, so the column is flipped when
    most of its mass is negative before clamping. Without that, clamping would
    discard the informative half of a column that happened to be fitted with
    the opposite sign.
    """
    pos_mass = torch.sum(torch.clamp(v, min=0.0), dim=0, keepdim=True)
    neg_mass = torch.sum(torch.clamp(-v, min=0.0), dim=0, keepdim=True)
    v = torch.where(neg_mass > pos_mass, -v, v)
    return torch.clamp(v, min=0.0)


def _project_nonnegative(v: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
    """
    Enforce non-negativity by projection, reflecting only if that would break `v`.

    Reflection (``abs``) is not a projection: it maps a strongly negative
    loading to an equally strong *positive* one, asserting the opposite of what
    the fit found. It also damages orthogonality more than clamping does -- over
    200 random matrices per shape, the normalized Stiefel defect after
    reflection against after projection was 1.04 vs 0.85 at (5, 2), 1.72 vs
    1.24 at (10, 3) and 2.86 vs 1.51 at (300, 5) -- and the reflected matrix
    also moves further from the original.

    Projection has one failure mode that reflection does not: clamping can zero
    a column outright or drop the rank. Measured across the same shapes, that
    happens only when the number of features equals the number of components
    (at (5, 5), mean rank 4.95 of 5); for every shape with more features than
    components, rank was preserved in all 200 trials. Rather than guess from
    the shape, the projection is computed and checked, and reflection is used
    only for the matrices where it actually degenerates -- so ``abs`` no longer
    runs on a well-posed basis, which is the case that matters.
    """
    projected = _sign_resolved_clamp(v, sign=sign)

    if bool((projected.abs().sum(dim=0) == 0).any()):
        return torch.abs(v)
    if v.shape[1] > 1:
        try:
            if (torch.linalg.matrix_rank(projected.float())
                    < torch.linalg.matrix_rank(v.float())):
                return torch.abs(v)
        except Exception:
            # A rank query that fails is not a reason to reject the projection.
            pass
    return projected


def positivity_diagnostics(v: torch.Tensor) -> Dict[str, Any]:
    """
    Report the structural damage a sign projection may have done to `v`.

    Projection onto a sign orthant can delete features and reduce rank, which
    is silent at the API level but destroys the basis. This surfaces it.

    Parameters
    ----------
    v : torch.Tensor
        Basis matrix of shape (features, components).

    Returns
    -------
    Dict[str, Any]
        "zero_rows": features with no loading on any component;
        "zero_cols": components with no loading at all;
        "rank" and "full_rank": numerical rank against min(shape);
        "rank_deficient": whether rank is below min(shape).
    """
    zero_rows = int((v.abs().sum(dim=1) == 0).sum())
    zero_cols = int((v.abs().sum(dim=0) == 0).sum())
    rank = int(torch.linalg.matrix_rank(v)) if min(v.shape) > 0 else 0
    full = min(v.shape)
    return {
        "zero_rows": zero_rows,
        "zero_cols": zero_cols,
        "rank": rank,
        "full_rank": full,
        "rank_deficient": rank < full,
    }


def optimize_indicator_matrix(m: torch.Tensor, 
                              preprocess: bool = True, 
                              max_iter: int = 20, 
                              tol: float = 1e-4, 
                              verbose: bool = False) -> torch.Tensor:
    """
    Mask a matrix down to an optimal one-per-row/column selection of entries.

    Chooses the binary indicator `I` that maximizes ``sum(m * I)`` subject to at
    most one selected entry per row and at most one per column, then returns the
    *masked values* ``m * I`` (not the indicator itself -- use
    :func:`_assignment_indicator` for that).

    Parameters
    ----------
    m : torch.Tensor
        The input matrix (usually a cross-covariance or projection).
    preprocess : bool, default=True
        Whether to flip row signs so each row carries mostly positive mass.
    max_iter : int, default=20
        Unused; retained for backwards compatibility. The assignment is solved
        exactly in one step, so no iteration is required.
    tol : float, default=1e-4
        Unused; retained for backwards compatibility.
    verbose : bool, default=False
        Whether to print the achieved objective value.

    Returns
    -------
    torch.Tensor
        ``m * I`` -- the input values with all unselected entries zeroed, where
        rows may have been sign-flipped if `preprocess` is True. Same shape as `m`.

    Notes
    -----
    Because the return value carries magnitudes rather than 0/1 flags, callers
    must not treat it as an indicator matrix; doing so squares the entries.

    Examples
    --------
    >>> import torch
    >>> m = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    >>> optimize_indicator_matrix(m, preprocess=False)
    tensor([[2., 0.],
            [0., 3.]])

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m).float()
        
    m_opt = m.clone()
    if preprocess:
        for i in range(m_opt.shape[0]):
            if torch.sum(m_opt[i, :] < 0) > torch.sum(m_opt[i, :] > 0):
                m_opt[i, :] = -m_opt[i, :]

    indicator = _assignment_indicator(m_opt)
    if verbose:
        print(f"Assignment objective: {torch.sum(m_opt * indicator).item()}")
    return torch.where(indicator > 0, m_opt, torch.zeros_like(m_opt))

def indicator_opt_both_ways(m: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """
    Sparsify a matrix to a one-per-row/column selection, trying both sign orientations.

    Latent components are only defined up to sign, so this selects the entry
    assignment that captures the most mass under either `m` or `-m`, and returns
    the original values of `m` restricted to the winning selection.

    Parameters
    ----------
    m : torch.Tensor
        The input matrix to sparsify.
    verbose : bool, optional
        Whether to print the objective value of each orientation (default False).

    Returns
    -------
    torch.Tensor
        ``m * I`` for the winning indicator `I`: the original entries of `m`,
        magnitudes and signs preserved, with everything unselected set to zero.

    Notes
    -----
    Earlier revisions passed the *masked values* returned by
    :func:`optimize_indicator_matrix` back in as if they were a 0/1 indicator,
    which squared every retained entry (``m**2 * I``), discarded the sign, and
    made the negative orientation unreachable.

    Examples
    --------
    >>> import torch
    >>> m = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    >>> indicator_opt_both_ways(m)
    tensor([[2., 0.],
            [0., 3.]])
    >>> # a predominantly negative matrix keeps its negative entries
    >>> indicator_opt_both_ways(-m)
    tensor([[-2.,  0.],
            [ 0., -3.]])

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m).float()

    indicator_pos = _assignment_indicator(m)
    captured_pos = torch.sum(m * indicator_pos)

    indicator_neg = _assignment_indicator(-m)
    captured_neg = torch.sum(-m * indicator_neg)

    if verbose:
        print(f"positive orientation: {captured_pos.item()}, "
              f"negative orientation: {captured_neg.item()}")

    indicator = indicator_pos if captured_pos >= captured_neg else indicator_neg
    return torch.where(indicator > 0, m, torch.zeros_like(m))

def rank_based_matrix_segmentation(v: torch.Tensor, 
                                   sparseness_quantile: float, 
                                   basic: bool = False, 
                                   positivity: str = "positive", 
                                   transpose: bool = False) -> torch.Tensor:
    """
    Apply rank-based segmentation to a matrix to enforce sparsity.

    Retains only the top percentile of values (based on absolute magnitude) 
    for each row or column, setting others to zero.

    Parameters
    ----------
    v : torch.Tensor
        The matrix to segment.
    sparseness_quantile : float
        The quantile of elements to set to zero (0.0 to 1.0).
    basic : bool, default=False
        If False, uses assignment-based segmentation (`indicator_opt_both_ways`),
        retaining a single entry per row and column; `sparseness_quantile` is
        not used in this mode because the assignment fixes the retained count.
        If True, uses quantile-based thresholding.
    positivity : str, default="positive"
        Constraint on sign. "positive" retains only non-negative entries,
        "negative" only non-positive entries, and "either" ranks by absolute
        magnitude regardless of sign.
    transpose : bool, default=False
        Whether to apply segmentation to columns (False) or rows (True).
        Honoured in both `basic` modes.

    Returns
    -------
    torch.Tensor
        The segmented sparse matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(v).float()
        
    if transpose:
        v = v.t()

    if not basic:
        # Assignment-based segmentation: one retained entry per row/column.
        # `positivity` still applies, so a "positive" request cannot come back
        # negative; `sparseness_quantile` has no meaning here because the
        # assignment already fixes how many entries survive.
        if positivity == "positive":
            candidate = torch.clamp(v, min=0.0)
        elif positivity == "negative":
            candidate = torch.clamp(v, max=0.0)
        else:
            candidate = v
        outmat = indicator_opt_both_ways(candidate)
        return outmat.t() if transpose else outmat

    outmat = torch.zeros_like(v)
    n_to_keep = int(round(v.shape[1] * (1.0 - sparseness_quantile)))
    n_to_keep = max(0, min(n_to_keep, v.shape[1]))
    if n_to_keep == 0:
        return outmat.t() if transpose else outmat

    for k in range(v.shape[0]):
        row_values = v[k, :].clone()
        if torch.all(row_values == 0):
            continue

        if positivity == "positive":
            # honour the requested sign: drop negatives, rank the rest
            row_values = torch.clamp(row_values, min=0.0)
            _, loc_ord = torch.topk(row_values, k=n_to_keep)
        elif positivity == "negative":
            row_values = torch.clamp(row_values, max=0.0)
            _, loc_ord = torch.topk(-row_values, k=n_to_keep)
        else:
            _, loc_ord = torch.topk(torch.abs(row_values), k=n_to_keep)

        outmat[k, loc_ord] = row_values[loc_ord]

    if transpose:
        return outmat.t()
    return outmat

def orthogonalize_and_q_sparsify(v: torch.Tensor, 
                                 sparseness_quantile: float = 0.0, 
                                 positivity: str = "either",
                                 orthogonalize: bool = True,
                                 unit_norm: bool = True,
                                 soft_thresholding: bool = False,
                                 sparseness_alg: Optional[str] = None) -> torch.Tensor:
    """
    Orthogonalize and/or sparsify a projection matrix.

    A comprehensive utility for enforcing constraints on basis matrices, 
    including Stiefel manifold projection (SVD-based) and quantile-based 
    sparsification.

    Parameters
    ----------
    v : torch.Tensor
        The input matrix (features x components).
    sparseness_quantile : float, default=0.0
        Proportion of elements to zero out.
    positivity : str, default="either"
        Sign constraints: "positive", "negative", or "either".
    orthogonalize : bool, default=True
        Whether to project the matrix onto the Stiefel manifold (V^T V = I).
    unit_norm : bool, default=True
        Whether to normalize each component to unit L2 norm.
    soft_thresholding : bool, default=False
        If True, uses soft-thresholding (shrinkage) instead of hard zeroing.
    sparseness_alg : str, optional
        Override algorithm: "orthorank" or "basic" for rank-based segmentation.

    Returns
    -------
    torch.Tensor
        The constrained and sparsified matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if sparseness_alg == "orthorank":
        return rank_based_matrix_segmentation(v, sparseness_quantile, basic=False, positivity=positivity, transpose=True)
    elif sparseness_alg == "basic":
        return rank_based_matrix_segmentation(v, sparseness_quantile, basic=True, positivity=positivity, transpose=True)

    if torch.all(v == 0):
        return v.clone()

    v_out = v.clone()
    orig_dtype = v_out.dtype
    n, k = v_out.shape
    
    if orthogonalize and k > 1:
        try:
            nonneg = positivity in ('positive', 'hard', 'nonnegative', 'nonneg')
            retracted = _nsa_retract(v_out, w=NSA_DEFAULT_W, nonneg=nonneg)
            v_out = retracted if retracted is not None else _svd_polar(v_out)
        except Exception: pass
        
    sparsify = (isinstance(sparseness_quantile, (list, torch.Tensor, np.ndarray))
                or sparseness_quantile > 0)

    if not sparsify:
        # `unit_norm` used to be honoured only on the sparsification path, so
        # with sparseness_quantile == 0 the columns were left unnormalized no
        # matter what the caller asked for. Column scale is arbitrary for a
        # basis matrix, and leaving it free lets covariance-style energies grow
        # without bound.
        if unit_norm:
            v_out = _unit_normalize_columns(v_out)
        return v_out.to(orig_dtype)

    if sparsify:
        for vv in range(k):
            local_v = v_out[:, vv]
            if positivity == "positive":
                local_v = torch.clamp(local_v, min=0.0)
            elif positivity == "negative":
                local_v = torch.clamp(local_v, max=0.0)
            
            sq = sparseness_quantile[vv] if isinstance(sparseness_quantile, (list, torch.Tensor, np.ndarray)) else sparseness_quantile
            if sq <= 0:
                if unit_norm:
                    norm = torch.norm(local_v)
                    if norm > 0: local_v = local_v / norm
                v_out[:, vv] = local_v
                continue
                
            if soft_thresholding:
                thresh = torch.quantile(torch.abs(local_v), sq)
                local_v = torch.sign(local_v) * torch.clamp(torch.abs(local_v) - thresh, min=0.0)
            else:
                thresh = torch.quantile(torch.abs(local_v), sq)
                local_v[torch.abs(local_v) < thresh] = 0.0
                
            if unit_norm:
                norm = torch.norm(local_v)
                if norm > 0:
                    local_v = local_v / norm
            v_out[:, vv] = local_v
            
    return v_out.to(orig_dtype)

def project_to_orthonormal_nonnegative(x: torch.Tensor, 
                                       max_iter: int = 100, 
                                       tol: float = 1e-4, 
                                       constraint: str = 'positive') -> torch.Tensor:
    """
    Alternate projections onto the Stiefel manifold and the sign-constrained orthant.

    Iteratively alternates between projecting onto the Stiefel manifold
    (orthogonality) and the non-negative orthant (positivity).

    Warnings
    --------
    The result is generally **not** orthonormal. Both sets are non-convex, so
    alternating projection carries no convergence guarantee, and the loop
    applies the sign projection last -- which destroys orthogonality unless the
    columns happen to have disjoint supports (the only way a matrix can be both
    orthonormal and non-negative). This is a heuristic that trades the two
    constraints off, not a projection onto their intersection, and it is not
    Dykstra's algorithm, which would require maintaining per-set correction
    terms.

    Parameters
    ----------
    x : torch.Tensor
        The input matrix.
    max_iter : int, default=100
        Maximum number of projection cycles.
    tol : float, default=1e-4
        Convergence tolerance for the difference between iterations.
    constraint : str, default='positive'
        "positive" or "negative".

    Returns
    -------
    torch.Tensor
        A sign-constrained matrix that is approximately orthonormal; check
        `pysimlr.utils.stiefel_defect` if orthogonality matters.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x).float()
        
    v_out = x.clone()
    for _ in range(max_iter):
        v_prev = v_out.clone()
        # Orthogonality projection
        v_out = _svd_polar(v_out)
        # Positivity projection
        if constraint == 'positive':
            v_out = torch.clamp(v_out, min=0.0)
        elif constraint == 'negative':
            v_out = torch.clamp(v_out, max=0.0)
            
        if torch.norm(v_out - v_prev) < tol:
            break
    return v_out

def project_to_partially_orthonormal_nonnegative(x: torch.Tensor, 
                                               max_iter: int = 10, 
                                               constraint: str = 'positive', 
                                               ortho_strength: float = 1.0) -> torch.Tensor:
    """
    Project a matrix towards the Stiefel manifold with a controlled strength.

    Blends the original matrix with its polar projection onto the Stiefel
    manifold, `max_iter` times.

    Warnings
    --------
    The blend is re-applied every iteration, so the deviation from the manifold
    shrinks geometrically as ``(1 - ortho_strength) ** max_iter``. With the
    default `max_iter` of 10, any `ortho_strength` above roughly 0.3 is
    indistinguishable from a full projection -- the parameter controls the rate
    of approach, not the final distance from the manifold. Use `max_iter=1` for
    a genuinely partial projection. There is also no convergence check; the loop
    always runs `max_iter` times.

    Parameters
    ----------
    x : torch.Tensor
        The input matrix.
    max_iter : int, default=10
        Maximum number of projection cycles.
    constraint : str, default='positive'
        "positive", "negative", or "either".
    ortho_strength : float, default=1.0
        The blend factor (0.0 to 1.0).

    Returns
    -------
    torch.Tensor
        The partially projected matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x).float()
        
    v_out = x.clone()
    for _ in range(max_iter):
        v_ortho = _svd_polar(v_out)
        v_out = (1 - ortho_strength) * v_out + ortho_strength * v_ortho
        
        if constraint == 'positive':
            v_out = torch.clamp(v_out, min=0.0)
        elif constraint == 'negative':
            v_out = torch.clamp(v_out, max=0.0)
    return v_out

#: Energies whose value is unchanged or driven to -inf by rescaling ``V``, so
#: the column scale is a free gauge that the objective cannot pin down.
#: Measured homogeneity degree of ``E(cV)`` in ``c``:
#:
#:     regression  1.87  finite minimiser, scale is set by the data
#:     acc         1.00  -> -inf, unbounded below
#:     logcosh     1.01  -> -inf, unbounded below
#:     kurtosis    4.00  -> -inf, unbounded below
#:     nc          0.00  scale-invariant
#:     exp/gauss   0.00  scale-invariant
#:
#: For the unbounded ones the optimiser can "improve" the energy forever by
#: inflating ``||V||`` -- magnitudes around 1e8 were observed -- so the gauge
#: has to be fixed or the objective is meaningless. For the scale-invariant
#: ones fixing it is free and removes a null direction.
#:
#: ``regression`` is the exception and must be left alone: its minimiser is
#: ``V* = X^T u (u^T u)^{-1}``, whose column norms are determined by the data.
#: The previous code unit-normalised **every** energy on every sweep, which
#: deleted the very scale the default objective was trying to find.
GAUGE_FREE_ENERGIES = frozenset({"regression"})


def energy_needs_unit_columns(energy_type: Optional[str]) -> bool:
    """Whether ``energy_type`` requires the column scale to be fixed."""
    if energy_type is None:
        return False
    return energy_type not in GAUGE_FREE_ENERGIES


def simlr_sparseness(v: torch.Tensor,
                     constraint_type: str = "none",
                     smoothing_matrix: Optional[torch.Tensor] = None,
                     positivity: str = 'either',
                     sparseness_quantile: float = 0.0,
                     constraint_weight: float = 0.0,
                     constraint_iterations: int = 1,
                     sparseness_alg: str = 'soft',
                     energy_type: Optional[str] = None,
                     modality_index: Optional[int] = None,
                     unit_columns: Optional[bool] = None,
                     retraction_diagnostics: Optional[dict] = None) -> torch.Tensor:
    """
    Project a basis onto the NSA-Flow feasible set.

    This is the single projection operator for SiMLR. It applies the optional
    smoothing prior, resolves the sign convention, and calls NSA-Flow. There is
    nothing else: the non-negativity, the near-orthogonality and the sparsity
    are all properties of the set NSA-Flow solves onto, controlled by one
    weight ``w``.

    Why it is only a call
    ---------------------
    This function used to be a five-branch dispatch that applied
    ``apply_positivity``, then a manifold retraction (NSA-Flow, or an SVD polar
    factor when the backend was missing, which its own docstring admitted
    "changes numerical results"), then quantile soft-thresholding, then a
    column renormalisation. Four operators, each with its own notion of what
    the feasible set is, composed in an order nothing verified.

    NSA-Flow already solves for that set in one step. ``w`` is a genuine convex
    weight: ``w = 0`` returns ``max(0, V)`` and ``w = 1`` gives orthogonal
    columns, which under non-negativity means *disjoint supports* -- a hard
    clustering of the features. Sparsity is therefore a consequence of ``w``,
    not a separate quantile applied afterwards, and the two cannot disagree
    because there is only one of them.

    Parameters
    ----------
    v : torch.Tensor
        Candidate basis, shape (features, components).
    constraint_type : str, default="none"
        Retained for call compatibility. ``"none"`` returns the (optionally
        smoothed, sign-resolved) input unprojected; anything else projects.
    smoothing_matrix : torch.Tensor, optional
        Spatial/graph prior applied as ``S @ V`` before projection. This is a
        prior on the basis, not part of the feasible set, so it stays outside
        the solver. Used by `nnh_embed` and by the operators in `sparse`.
    positivity : str, default='either'
        ``'positive'``/``'hard'``/``'nonnegative'`` constrain ``V >= 0``;
        ``'negative'`` solves the reflected problem and negates the result;
        anything else leaves the sign free.
    sparseness_quantile, sparseness_alg, constraint_iterations : deprecated
        Sparsity is set by ``w``. Passing a non-default value warns rather than
        being silently dropped, because a parameter that looks respected and is
        not is how a benchmark ends up reporting a setting it never ran.
    constraint_weight : float, default=0.0
        The NSA-Flow weight ``w``. ``0`` means "use `NSA_DEFAULT_W`".
    energy_type, modality_index : optional
        ``modality_index`` indexes a per-view ``constraint_weight`` list.
        ``energy_type`` selects the gauge: see `GAUGE_FREE_ENERGIES`.
    unit_columns : bool, optional
        Rescale the projected columns to unit norm. ``None`` derives it from
        ``energy_type`` via `energy_needs_unit_columns`, which is the only
        correct default: the covariance and negentropy energies are unbounded
        below under rescaling and need the gauge fixed, while ``regression``
        has a finite minimiser whose scale carries information and must not be
        renormalised.
    retraction_diagnostics : dict, optional
        Updated in place with the solver's self-report.

    Returns
    -------
    torch.Tensor
        The projected basis, in the input dtype.

    Raises
    ------
    ImportError
        If the NSA-Flow backend is unavailable and a projection was requested.
        There is deliberately no fallback: substituting a different operator
        silently changes the feasible set.
    """
    orig_dtype = v.dtype
    v_out = v.clone()
    if torch.isnan(v_out).any():
        v_out = torch.nan_to_num(v_out, nan=0.0)

    if smoothing_matrix is not None:
        # No dtype coercion here: `sparse` supplies matrix-free operators
        # (e.g. SparseGraphResolvent) that implement `@` but not `.to()`.
        v_out = smoothing_matrix @ v_out

    for name, value, default in (("sparseness_quantile", sparseness_quantile, 0.0),
                                 ("sparseness_alg", sparseness_alg, "soft"),
                                 ("constraint_iterations", constraint_iterations, 1)):
        if isinstance(value, (list, tuple, np.ndarray)) or value != default:
            warnings.warn(
                f"simlr_sparseness: {name}={value!r} is ignored. Sparsity is "
                f"now a consequence of the NSA-Flow weight w "
                f"(constraint_weight); w -> 1 gives disjoint supports. "
                f"Set w rather than {name}.",
                DeprecationWarning, stacklevel=2,
            )

    nonneg = positivity in ('positive', 'hard', 'nonnegative', 'nonneg', 'softplus')
    negative = (positivity == 'negative')

    if constraint_type == "none":
        # The sign constraint is part of the feasible set whether or not the
        # orthogonality projection runs, so it is applied on this path too.
        # Returning `v` untouched here made `positivity='positive'` silently
        # inert for the default `constraint_type`.
        return apply_positivity(v_out, positivity).to(orig_dtype)

    if negative:
        # The solver only knows V >= 0; solve the reflected problem.
        v_out = -v_out
        nonneg = True

    w = constraint_weight
    if isinstance(w, (list, tuple, np.ndarray)) and modality_index is not None:
        w = w[modality_index]
    w = float(w) if w and float(w) > 0 else NSA_DEFAULT_W

    # The solver receives the SIGNED iterate.  With nonneg=True the feasible
    # set Y >= 0 is enforced by the solver's projection; rectifying first
    # computes prox(clamp(z)) rather than prox(z), which is not the proximal
    # operator and breaks the descent argument for the outer loop.  (The
    # rectified path had in fact been the live one on nsa_flow 3.0.x, because
    # the backend moved `fidelity` into **kwargs and the signature probe that
    # gated the signed path went False.)  Without nonneg there is nothing to
    # choose: the sign is whatever the caller asked for.
    candidate = (_resolve_column_sign_gauge(v_out) if nonneg
                 else apply_positivity(v_out.clone(), positivity))
    projected = _nsa_retract(candidate, w=w, nonneg=nonneg,
                             diagnostics=retraction_diagnostics)
    if projected is None and load_nsa_flow() is None:
        raise ImportError(
            "simlr_sparseness requires the NSA-Flow backend to project onto "
            "the constrained set. Install nsa_flow, or pass "
            "constraint_type='none' to skip projection. The previous SVD-polar "
            "fallback projected onto a different set and was applied silently."
        )
    if projected is None:
        # The backend is installed but returned something unusable (all-zero,
        # NaN, wrong shape, or it raised). That is a runtime failure of the
        # projection, not a missing install, and it must not be papered over
        # with a different operator -- a zero basis used to propagate silently.
        raise RuntimeError(
            f"NSA-Flow returned an unusable projection for a "
            f"{tuple(v_out.shape)} basis at w={w:.3g}, nonneg={nonneg}. "
            f"No substitute is applied, because a different projection means a "
            f"different feasible set. Inspect retraction_diagnostics, or pass "
            f"constraint_type='none'."
        )
    if unit_columns is None:
        unit_columns = energy_needs_unit_columns(energy_type)
    if unit_columns:
        projected = _unit_normalize_columns(projected)

    if negative:
        projected = -projected
    return projected.to(orig_dtype)
