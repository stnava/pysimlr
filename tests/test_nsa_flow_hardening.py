"""
Hardening tests for `simlr` with NSA-Flow as the sparsification engine.

These exercise the *real* backend (skipped when it is not installed) rather
than a mock, and assert properties of the contract rather than specific
numbers, so they survive backend releases:

* a degenerate retraction is never silently adopted as the basis;
* the result is invariant to the caller's arbitrary column scale;
* dtype, shape, finiteness and the positivity/unit-norm contracts hold;
* repeated runs agree, and the backend does not disturb the caller's RNG.

The regression behind most of this: a backend version returned an all-zero
``Y`` for narrow inputs, the call site tested only ``Y is not None``, and SiMLR
produced an identically-zero basis on every run with no error anywhere.
"""

import warnings

import numpy as np
import pytest
import torch

from pysimlr import simlr
from pysimlr.nsa_backend import load_nsa_flow
from pysimlr.sparsification import _nsa_retract, _usable_retraction

pytestmark = pytest.mark.skipif(load_nsa_flow() is None,
                                reason="no NSA-Flow backend installed")


def orthogonality_defect(v: torch.Tensor) -> float:
    """
    ``||V'V/mean(diag) - I||`` -- zero exactly when the columns are orthogonal
    and equinormed.

    Normalizing by the mean diagonal rather than adding an epsilon keeps the
    measure scale-free. An additive epsilon silently dominates the Gram matrix
    for small-magnitude bases: at ``||V|| ~ 1e-12`` the Gram entries are
    ~1e-24, so a 1e-12 floor reported a defect of 1.7 for a basis whose true
    defect was 0.83.
    """
    gram = v.double().T @ v.double()
    mean_diag = torch.diag(gram).mean()
    if mean_diag <= 0:
        return float("inf")
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    return float(torch.norm(gram / mean_diag - eye))


def make_case(seed=0, n=80, ps=(30, 24), k=3):
    """Non-negative basis with block-disjoint supports, the regime the
    non-negative retraction is designed for."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, k, generator=g)
    mats, bases = [], []
    for p in ps:
        v = torch.rand(p, k, generator=g)
        mask = torch.zeros(p, k)
        block = p // k
        for j in range(k):
            mask[j * block:(j + 1) * block, j] = 1.0
        v = v * mask
        v = v / v.norm(dim=0, keepdim=True)
        mats.append(u @ v.T + 0.3 * torch.randn(n, p, generator=g))
        bases.append(v)
    return mats, bases


# --------------------------------------------------------------------------
# `_nsa_retract`: degenerate candidates
# --------------------------------------------------------------------------

def _degenerate_candidates():
    g = torch.Generator().manual_seed(0)
    rand = lambda p, k: torch.rand(p, k, generator=g, dtype=torch.float64)

    all_zero = torch.zeros(10, 3, dtype=torch.float64)

    zero_col = rand(10, 3)
    zero_col[:, 1] = 0.0

    single_entry = torch.zeros(10, 3, dtype=torch.float64)
    single_entry[0, 0] = 1.0

    has_nan = rand(10, 3)
    has_nan[0, 0] = float("nan")

    has_inf = rand(10, 3)
    has_inf[0, 0] = float("inf")

    return [
        pytest.param(all_zero, id="all-zero"),
        pytest.param(zero_col, id="zero-column"),
        pytest.param(single_entry, id="single-nonzero-entry"),
        pytest.param(has_nan, id="contains-nan"),
        pytest.param(has_inf, id="contains-inf"),
    ]


@pytest.mark.parametrize("candidate", _degenerate_candidates())
def test_degenerate_candidates_are_rejected_not_returned(candidate):
    """
    A candidate that cannot yield a full-rank basis must come back as None so
    the caller falls back, rather than as a rank-deficient or zero matrix.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _nsa_retract(candidate, w=0.5, nonneg=True)
    assert out is None, (
        "a degenerate candidate produced an accepted retraction: "
        f"{None if out is None else out}"
    )


@pytest.mark.parametrize("shape", [(5, 2), (10, 3), (13, 2), (25, 3), (64, 5)])
@pytest.mark.parametrize("w", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("nonneg", [True, False])
def test_well_posed_candidates_retract_without_degenerating(shape, w, nonneg):
    """
    Across the shape/weight grid that previously collapsed to zeros, the
    retraction must be finite, full-rank, free of zero columns, and
    non-negative when asked.
    """
    p, k = shape
    g = torch.Generator().manual_seed(p * 100 + k)
    v = torch.rand(p, k, generator=g, dtype=torch.float64)
    if not nonneg:
        v = v - 0.5

    out = _nsa_retract(v, w=w, nonneg=nonneg)
    assert out is not None, f"backend rejected a well-posed {shape} candidate at w={w}"
    assert out.shape == v.shape
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0
    assert (out.abs().sum(dim=0) > 0).all(), "retraction produced a zero column"
    assert torch.linalg.matrix_rank(out.float()) == k, "retraction lost rank"
    if nonneg:
        assert (out >= 0).all(), "nonneg=True returned negative entries"


@pytest.mark.parametrize("scale", [1e-8, 1e-3, 1.0, 1e3, 1e8])
def test_retraction_is_invariant_to_the_candidates_scale(scale):
    """
    A basis matrix's column scale is an arbitrary gauge -- every call site
    renormalizes afterwards -- so the retraction must not depend on it.
    Ungauged, the solver returned a visibly different solution at large scale
    (defect 2.45 rather than 0.83) and could stall on its iteration cap.
    """
    g = torch.Generator().manual_seed(7)
    base = torch.rand(10, 3, generator=g, dtype=torch.float64)
    base[:, 0] *= 50.0  # deliberately unequal column norms

    reference = _nsa_retract(base, w=0.5, nonneg=True)
    scaled = _nsa_retract(base * scale, w=0.5, nonneg=True)
    assert reference is not None and scaled is not None

    assert orthogonality_defect(scaled) == pytest.approx(
        orthogonality_defect(reference), rel=1e-6, abs=1e-9)
    # The scale is restored on the way out, so the result tracks the input.
    assert float(scaled.norm()) == pytest.approx(
        float(reference.norm()) * scale, rel=1e-6)


def test_retraction_preserves_the_candidates_dtype():
    """Solved in float64 for tolerance headroom, returned in the input dtype."""
    g = torch.Generator().manual_seed(3)
    v32 = torch.rand(20, 3, generator=g)
    out = _nsa_retract(v32, w=0.5, nonneg=True)
    assert out is not None and out.dtype == torch.float32

    out64 = _nsa_retract(v32.double(), w=0.5, nonneg=True)
    assert out64 is not None and out64.dtype == torch.float64


def test_retraction_is_deterministic():
    g = torch.Generator().manual_seed(11)
    v = torch.rand(25, 3, generator=g, dtype=torch.float64)
    first = _nsa_retract(v, w=0.5, nonneg=True)
    second = _nsa_retract(v, w=0.5, nonneg=True)
    assert torch.equal(first, second)


def test_retraction_does_not_disturb_the_callers_rng_stream():
    """
    The backend seeds the global generator internally. Left unguarded that
    collapsed `simlr_perm`'s permutations to a single repeated draw -- every
    permutation identical, the null distribution a constant -- while the whole
    test suite still passed.
    """
    g = torch.Generator().manual_seed(5)
    v = torch.rand(20, 3, generator=g, dtype=torch.float64)

    torch.manual_seed(1234)
    expected = torch.randn(6)

    torch.manual_seed(1234)
    _nsa_retract(v, w=0.5, nonneg=True)
    observed = torch.randn(6)

    assert torch.equal(expected, observed), (
        "the retraction advanced or reseeded the global RNG stream")


# --------------------------------------------------------------------------
# `simlr` end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("constraint", ["orthox0.1", "orthox0.5", "orthox0.9",
                                        "nsaflow", "Stiefel", "Stiefel_ns",
                                        "Grassmann"])
@pytest.mark.parametrize("positivity", ["positive", "either"])
def test_simlr_returns_a_usable_basis_for_every_constraint(constraint, positivity):
    mats, _ = make_case()
    res = simlr(mats, k=3, iterations=8, constraint=constraint,
                positivity=positivity, verbose=False)

    for i, v in enumerate(res["v"]):
        assert v.shape == (mats[i].shape[1], 3)
        assert torch.isfinite(v).all(), f"modality {i} basis is not finite"
        assert v.abs().sum() > 0, f"modality {i} basis collapsed to zero"
        assert (v.abs().sum(dim=0) > 0).all(), f"modality {i} has a zero column"
        if positivity == "positive":
            assert (v >= -1e-8).all(), f"modality {i} violated positivity"


@pytest.mark.parametrize("positivity", ["positive", "either"])
def test_hard_manifold_constraint_delivers_unit_norm_columns(positivity):
    """Stiefel/Grassmann promise an orthonormal basis; the column norms are the
    part a soft constraint does not pin down, so they are asserted explicitly."""
    mats, _ = make_case()
    res = simlr(mats, k=3, iterations=8, constraint="Stiefel",
                positivity=positivity, verbose=False)
    for v in res["v"]:
        norms = v.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), norms


def test_hard_constraint_with_positivity_is_actually_orthogonal():
    """
    The behaviour the backend exists for: non-negativity and orthogonality
    together force near-disjoint column supports.

    This used to compare the backend against the SVD-polar fallback. There is
    no fallback now -- `simlr_sparseness` is a call to NSA-Flow and nothing
    else -- so the assertion is absolute rather than relative.
    """
    mats, _ = make_case()
    res = simlr(mats, k=3, iterations=8, constraint="Stiefel",
                positivity="positive", verbose=False)
    defect = float(np.mean([orthogonality_defect(v) for v in res["v"]]))
    assert defect < 0.1, f"non-negative basis is not near-orthogonal: {defect}"
    for v in res["v"]:
        assert (v >= -1e-6).all()


def test_simlr_requires_the_backend_for_a_constrained_fit():
    """nsa_flow is a hard dependency: it defines the feasible set.

    Previously it was optional and its absence silently substituted an SVD
    polar factor, so the same call ran a different algorithm depending on the
    install. That is now an error rather than a quiet difference.
    """
    from unittest.mock import patch

    mats, _ = make_case()
    with patch("pysimlr.sparsification.load_nsa_flow", return_value=None):
        with pytest.raises(ImportError, match="requires the NSA-Flow backend"):
            simlr(mats, k=3, iterations=2, constraint="Stiefel",
                  positivity="positive", verbose=False)

def test_simlr_is_reproducible_with_the_backend_engaged():
    mats, _ = make_case()
    kwargs = dict(k=3, iterations=8, constraint="orthox0.9",
                  positivity="positive", verbose=False)
    first = simlr(mats, **kwargs)
    second = simlr(mats, **kwargs)
    for a, b in zip(first["v"], second["v"]):
        assert torch.allclose(a, b, atol=1e-6), "repeated simlr runs disagree"


@pytest.mark.parametrize("weight", [0.0, 1.0, 1.5, -0.5])
def test_out_of_range_constraint_weights_do_not_crash_simlr(weight):
    """`w` outside (0, 1) is meaningless to the solver but reachable from a
    user-supplied constraint string, so it must degrade rather than explode."""
    mats, _ = make_case()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = simlr(mats, k=3, iterations=5, constraint=f"orthox{weight}",
                    positivity="positive", verbose=False)
    for v in res["v"]:
        assert torch.isfinite(v).all()
        assert v.abs().sum() > 0


@pytest.mark.parametrize("requested,expected", [
    (1.0, "NSA_MAX_W"), (1.5, "NSA_MAX_W"), (2.0, "NSA_MAX_W"),
    (0.0, "NSA_MIN_W"), (-0.5, "NSA_MIN_W"),
    (0.5, 0.5), (0.9, 0.9),
])
def test_retraction_weight_is_clamped_off_the_degenerate_endpoints(requested, expected):
    """
    ``w = 1`` drops the fidelity term, so the retraction stops staying near the
    candidate; ``w = 0`` drops the constraint term, so it does nothing. Both
    are reachable -- ``parse_constraint`` assigns 1.0 to every hard-manifold
    constraint -- and a user string like ``"orthox1.5"`` is not bounded at all.
    """
    from pysimlr.sparsification import (NSA_MAX_W, NSA_MIN_W,
                                        _clamp_retraction_weight)

    bound = {"NSA_MAX_W": NSA_MAX_W, "NSA_MIN_W": NSA_MIN_W}.get(expected, expected)
    assert _clamp_retraction_weight(requested) == pytest.approx(bound)
    assert 0.0 < _clamp_retraction_weight(requested) < 1.0


def test_every_constraint_family_is_capped_below_w_equals_one():
    """
    No constraint reaches w=1, including the hard manifold ones.

    `parse_constraint` assigns weight 1.0 to "Stiefel" and "Grassmann", and at
    w=1 the fidelity term drops out: the scale of the result is unconstrained
    and every scaled Stiefel matrix is optimal, so nothing selects among
    clusterings and the answer is arbitrary rather than merely rescaled.

    The hard branch used to be exempted so that ``V'V = I`` stayed exact. That
    was the wrong trade: recovery measured 0.842 at w=1 against 0.876 at 0.99
    and 0.885 at 0.9 over 10 paired seeds, and exact orthogonality *with*
    non-negativity is disjoint supports -- the clustering the cap exists to
    avoid.
    """
    from unittest.mock import patch

    from pysimlr.sparsification import NSA_MAX_W

    real = load_nsa_flow()

    def run(constraint):
        seen = []

        def spy(target, **kwargs):
            seen.append(kwargs.get("w"))
            return real(target, **kwargs)

        mats, _ = make_case()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch("pysimlr.sparsification.load_nsa_flow", return_value=spy):
                simlr(mats, k=3, iterations=5, constraint=constraint,
                      positivity="positive", verbose=False)
        assert seen, f"the backend was never called for {constraint!r}"
        return seen

    for constraint in ("Stiefel", "Grassmann", "orthox1.0", "nsaflowx1.5"):
        seen = run(constraint)
        assert max(seen) <= NSA_MAX_W, (
            f"{constraint!r} reached w={max(seen)}, above the cap {NSA_MAX_W}")
        assert all(0.0 < w < 1.0 for w in seen)

    assert NSA_MAX_W == pytest.approx(0.95)


def test_usable_retraction_rejects_what_it_claims_to():
    """Unit coverage of the guard itself, independent of any backend."""
    reference = torch.rand(10, 3, dtype=torch.float64) + 0.1

    assert not _usable_retraction(None, reference)
    assert not _usable_retraction(torch.zeros(10, 3, dtype=torch.float64), reference)
    assert not _usable_retraction(torch.rand(7, 3, dtype=torch.float64), reference)
    assert not _usable_retraction(
        torch.full((10, 3), float("nan"), dtype=torch.float64), reference)

    zero_col = torch.rand(10, 3, dtype=torch.float64) + 0.1
    zero_col[:, 1] = 0.0
    assert not _usable_retraction(zero_col, reference)

    good = torch.eye(10, 3, dtype=torch.float64) + 0.01
    assert _usable_retraction(good, reference)


if __name__ == "__main__":
    pytest.main([__file__])


# --------------------------------------------------------------------------
# The backend's layers and its solver disagree on `nonneg=True`
# --------------------------------------------------------------------------

def test_encoder_requests_hard_nonnegativity_not_softplus():
    """
    `nsa_flow(V, nonneg=True)` means a hard non-negativity constraint, but
    `NSAFlowLinear` maps `nonneg=True` to softplus. Passing the boolean turned
    the encoder's effective basis dense and near-uniform -- softplus sends
    every zero to log(2) = 0.693 -- so a basis with 27 of 60 entries non-zero
    and a normalized defect of 0.10 became fully dense with a defect of 2.38
    and every entry above 0.689.
    """
    from pysimlr.deep import LENDNSAEncoder
    from pysimlr.simlr import initial_basis_for_view

    torch.manual_seed(0)
    x = torch.randn(60, 20)

    encoder = LENDNSAEncoder(20, latent_dim=3, nsa_w=0.5,
                             positivity="positive", use_nsa=True)
    if encoder.nsa_linear is None:
        pytest.skip("backend did not provide NSAFlowLinear")

    assert encoder.nsa_linear.nonneg == "hard", (
        f"layer was configured with nonneg={encoder.nsa_linear.nonneg!r}; "
        "True means softplus to this layer, not hard non-negativity")

    with torch.no_grad():
        encoder.v_raw.copy_(initial_basis_for_view(x, 3, positivity="positive"))
    basis = encoder.v.detach()

    assert (basis >= 0).all()
    # Softplus would leave no entry below log(2); a hard clamp keeps the zeros.
    assert float(basis.min()) < 0.5, (
        f"every entry is at least {float(basis.min()):.3f}; the basis was "
        "passed through softplus rather than clamped")
    assert float((basis.abs() > 1e-10).float().mean()) < 0.9, (
        "the effective basis is dense; sparsity from the initializer was lost")
    assert orthogonality_defect(basis) < 0.5, (
        f"effective basis defect {orthogonality_defect(basis):.3f} is far from "
        "the initializer's")


def test_softplus_positivity_is_not_applied_twice():
    """`LENDNSAEncoder.v` applies its own `softplus(v - 4)`, so the layer must
    not also be asked for softplus."""
    from pysimlr.deep import LENDNSAEncoder

    encoder = LENDNSAEncoder(20, latent_dim=3, nsa_w=0.5,
                             positivity="softplus", use_nsa=True)
    if encoder.nsa_linear is None:
        pytest.skip("backend did not provide NSAFlowLinear")
    assert encoder.nsa_linear.nonneg in (None, "none", False), (
        f"layer was configured with nonneg={encoder.nsa_linear.nonneg!r}, which "
        "would apply softplus on top of the encoder's own")
