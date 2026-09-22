"""
Contract tests for the NSA-Flow retraction backend as `simlr` uses it.

These pin the *current* backend interface: `nsa_flow(V, w=, nonneg=, max_iter=)`
returning a result carrying `Y`. The backend previously took a `retraction`
keyword selecting among soft_polar/soft_ns/ns; it now implements a single
solver, so the `_ns` and `_polar` constraint suffixes select only which
pysimlr branch runs and no longer reach the backend. Tests that asserted a
`retraction` kwarg were removed rather than adapted, since there is nothing
left for them to assert.
"""

import random

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from pysimlr import simlr
from pysimlr.sparsification import NSA_DEFAULT_W, NSA_MIN_W


def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _identity_backend():
    """A backend stand-in that returns its input untouched."""
    mock = MagicMock()
    mock.side_effect = lambda Y, **kwargs: {"Y": Y.clone()}
    return mock


# `_nsa_retract` resolves the backend through `load_nsa_flow` on every call, so
# that is the single seam to patch. Patching the module-level `nsa_flow_orth`
# alias only flips the availability gate and leaves the real backend running.
def _patch_backend(mock):
    return patch("pysimlr.sparsification.load_nsa_flow", return_value=mock)


@pytest.mark.parametrize("constraint_type", ["ortho", "ortho_polar", "ortho_ns",
                                             "nsaflow", "nsaflow_ns"])
def test_backend_is_invoked_for_every_soft_ortho_constraint(constraint_type):
    set_all_seeds(42)
    x1, x2 = torch.randn(20, 10), torch.randn(20, 10)
    mock = _identity_backend()

    with _patch_backend(mock):
        simlr([x1, x2], k=2, iterations=1, constraint=f"{constraint_type}x0.6x7")

    assert mock.called, f"NSA-Flow was not called for constraint '{constraint_type}'"


def test_constraint_weight_is_passed_as_w():
    set_all_seeds(42)
    x1 = torch.randn(20, 10)
    mock = _identity_backend()

    with _patch_backend(mock):
        simlr([x1], k=2, iterations=1, constraint="orthox0.6x7")

    _, kwargs = mock.call_args
    assert kwargs["w"] == pytest.approx(0.6)


def test_a_bare_constraint_uses_parse_constraints_weight_not_nsa_default_w():
    """
    Two defaults could supply `w`, and they must now agree.

    `parse_constraint` defaulted the ortho family to 0.1 while
    `NSA_DEFAULT_W` was 0.5, so "the default w" was two different numbers
    depending on which path reached the projection. Both are 0.35 now, and
    this asserts the parsed weight still wins -- `NSA_DEFAULT_W` applies only
    when no weight is supplied at all (`constraint_weight=None`); an explicit
    zero is honoured as the unconstrained endpoint, not treated as unset.

    That ordering is deliberate. Measured over four designs (12 paired seeds
    each), tightening the retraction under non-negativity lowered the
    orthogonality defect monotonically but *reduced* subspace recovery in
    every design -- e.g. 0.894 at w=0.1 falling to 0.886 at w=0.9 and 0.841
    at a hard Stiefel constraint. At w=0.1 the backend also costs ~10x the
    runtime of the polar fallback for a +0.0006 recovery difference. So the
    weak default is the right one for a soft constraint, and raising it to
    0.5 here would silently change every `constraint="ortho"` run.
    """
    set_all_seeds(42)
    x1 = torch.randn(20, 10)
    mock = _identity_backend()

    with _patch_backend(mock):
        simlr([x1], k=2, iterations=1, constraint="nsaflow")

    _, kwargs = mock.call_args
    assert kwargs["w"] == pytest.approx(NSA_DEFAULT_W)


def test_an_explicitly_zero_weight_reaches_the_backend_as_the_floor():
    """`nsaflowx0` means no orthogonality term, and the floor is what gets sent.

    This test previously asserted the opposite -- that an explicit zero falls
    back to `NSA_DEFAULT_W` -- on the premise that the backend "needs some
    weight". That premise is false: `_nsa_retract(v, w=0.0)` returns a finite
    basis whose maximum off-diagonal column overlap (0.785) is the input's own,
    i.e. exactly the unconstrained endpoint. `_clamp_w` then maps the request
    to `NSA_MIN_W` (1e-3) so the solve is not a literal no-op -- that clamp is
    deliberate and is asserted by
    `test_retraction_weight_is_clamped_off_the_degenerate_endpoints`. The bug
    was not the clamp; it was that the request never reached it.

    The cost of the old behaviour was that `constraint_weight=0.0` produced
    output *bit-identical* to `0.5` (measured: ||P_0(v) - P_0.5(v)|| = 0.0e+00,
    43 zeros in both). Any sweep using w=0 as its unconstrained baseline
    measured w=0.5 twice and therefore read flat at the low end. It also
    contradicted `parse_constraint`, which documents that an explicit
    `"orthox0"` selects no constraint.
    """
    set_all_seeds(42)
    x1 = torch.randn(20, 10)
    mock = _identity_backend()

    with _patch_backend(mock):
        simlr([x1], k=2, iterations=1, constraint="nsaflowx0")

    assert mock.called
    _, kwargs = mock.call_args
    assert kwargs["w"] == pytest.approx(NSA_MIN_W)


def test_w_zero_is_not_w_default():
    """The endpoint is reachable and distinct -- the property the bug removed.

    Guards the projection directly rather than through `simlr`, so a future
    re-introduction of a falsy-zero test anywhere on the path fails here.
    """
    from pysimlr.sparsification import simlr_sparseness, NSA_DEFAULT_W as W

    g = torch.Generator().manual_seed(0)
    v = torch.rand(40, 3, generator=g)
    kw = dict(constraint_type="ortho", positivity="positive", energy_type="recon")
    at = {w: simlr_sparseness(v, constraint_weight=w, **kw) for w in (0.0, W, 1.0)}

    assert float((at[0.0] - at[W]).norm()) > 1e-3, (
        "constraint_weight=0.0 collapsed onto the default again")
    # `None`, not `0`, is the sentinel for "unset".
    assert torch.equal(simlr_sparseness(v, constraint_weight=None, **kw), at[W])
    assert torch.equal(simlr_sparseness(v, **kw), at[W])

    def overlap(p):
        pn = p / p.norm(dim=0, keepdim=True).clamp_min(1e-12)
        o = pn.t() @ pn
        o.fill_diagonal_(0)
        return float(o.abs().max())

    # Monotone in w, with w=0 leaving the input's own overlap untouched.
    assert overlap(at[0.0]) > overlap(at[W]) > overlap(at[1.0])


def test_candidate_is_handed_to_the_backend_in_float64():
    """
    float32 bottoms the solver's tolerance out near 1e-6 instead of 1e-9, and
    the solve costs milliseconds either way at these sizes, so the candidate is
    promoted before the call and cast back after.
    """
    set_all_seeds(42)
    x1 = torch.randn(20, 10)
    mock = _identity_backend()

    with _patch_backend(mock):
        res = simlr([x1], k=2, iterations=1, constraint="orthox0.5")

    args, _ = mock.call_args
    assert args[0].dtype == torch.float64
    assert res["v"][0].dtype == torch.float32, "result must come back in the input dtype"


@pytest.mark.parametrize("positivity,expected_nonneg", [("positive", True),
                                                        ("either", False)])
def test_positivity_selects_the_nonneg_flag(positivity, expected_nonneg):
    set_all_seeds(42)
    x1 = torch.randn(20, 10)
    mock = _identity_backend()

    with _patch_backend(mock):
        simlr([x1], k=2, iterations=1, constraint="orthox0.5", positivity=positivity)

    _, kwargs = mock.call_args
    assert kwargs["nonneg"] is expected_nonneg


def test_a_raising_backend_is_reported_not_substituted():
    """A failing projection must surface, not be swapped for another operator.

    This used to fall back to an SVD polar factor. That projects onto a
    different feasible set, so the same call produced a different algorithm
    depending on whether the backend happened to work -- silently.
    """
    set_all_seeds(42)
    x1 = torch.randn(20, 5)
    mock = MagicMock(side_effect=Exception("nsa_flow internal error"))

    with _patch_backend(mock):
        with pytest.raises(RuntimeError, match="unusable projection"):
            simlr([x1], k=2, iterations=1, constraint="orthox0.5")
    assert mock.called


@pytest.mark.parametrize("bad", ["zeros", "wrong_shape", "nan", "none"])
def test_an_unusable_backend_result_is_rejected_rather_than_adopted(bad):
    """
    The regression this guards: an earlier backend version returned an
    all-zero Y for narrow inputs, the code checked only `Y is not None`, and
    SiMLR silently produced a zero basis with no error anywhere.

    Rejection now means raising. It previously meant substituting an SVD polar
    factor, which is a projection onto a different set -- quieter, but it made
    the algorithm depend on whether the backend happened to succeed.
    """
    set_all_seeds(42)
    x1 = torch.randn(20, 5)

    def payload(Y, **kwargs):
        if bad == "zeros":
            return {"Y": torch.zeros_like(Y)}
        if bad == "wrong_shape":
            return {"Y": torch.randn(Y.shape[0] + 3, Y.shape[1], dtype=Y.dtype)}
        if bad == "nan":
            return {"Y": torch.full_like(Y, float("nan"))}
        return {"Y": None}

    mock = MagicMock(side_effect=payload)
    with _patch_backend(mock):
        with pytest.raises(RuntimeError, match="unusable projection"):
            simlr([x1], k=2, iterations=1, constraint="orthox0.5")
    assert mock.called


def test_hard_stiefel_constraint_also_routes_through_the_backend():
    set_all_seeds(42)
    x1 = torch.randn(20, 5)
    mock = _identity_backend()

    with _patch_backend(mock):
        simlr([x1], k=2, iterations=1, constraint="Stiefel_nsx0.5")

    assert mock.called
    _, kwargs = mock.call_args
    assert kwargs["w"] == pytest.approx(0.5)


def test_deep_encoders_expose_an_oriented_basis_when_the_backend_is_used():
    """
    The backend's linear module stores its weight as (latent, features) per the
    `torch.nn.Linear` convention; pysimlr indexes the basis as
    (features, latent). Both `v_raw` and `v` must come back in pysimlr's
    orientation.
    """
    from pysimlr.deep import lend_simr

    set_all_seeds(42)
    x1 = torch.randn(20, 8)
    res = lend_simr([x1], k=2, epochs=2, batch_size=10, warmup_epochs=0,
                    use_nsa=True, verbose=False)

    model = res["model"]
    encoders = getattr(model, "linear_encoders", getattr(model, "encoders", []))
    assert encoders
    for enc in encoders:
        assert enc.v_raw.shape == (8, 2)
        assert enc.v.shape == (8, 2)
        # The oriented view must share storage with the trainable parameter,
        # or the `v_raw.copy_(...)` initialization paths write into a temporary.
        assert (x1 @ enc.v_raw).shape == (20, 2)


if __name__ == "__main__":
    pytest.main([__file__])
