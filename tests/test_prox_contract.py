"""The retraction is the proximal operator of the outer loop -- verified, not assumed.

SiMLR takes a gradient step on its coupled energy and then calls
``simlr_sparseness`` on the result.  That is proximal gradient, and it only
converges to stationary points of ``E + w Dtilde`` if the retraction is the
Euclidean prox: ``argmin_{Y>=0} (1-w)||Y - z||^2/||z||^2 + w Dtilde(Y)`` at the
*signed* post-step point ``z``.  Two things broke that silently before:
the input was rectified first (``prox(clamp(z))``), and the backend's
``fidelity="auto"`` could pick its sign-blind subspace term, which is not a
prox of anything.  These tests pin both.
"""
import inspect

import numpy as np
import pytest
import torch

from pysimlr.nsa_backend import load_nsa_flow
from pysimlr.sparsification import simlr_sparseness

pytestmark = pytest.mark.skipif(load_nsa_flow() is None, reason="nsa_flow required")


def _z(seed=0, p=40, k=4):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(p, k, generator=g, dtype=torch.float64)      # signed, as a step is


def test_backend_exposes_fidelity_explicitly():
    import nsa_flow
    assert "fidelity" in inspect.signature(nsa_flow.nsa_flow).parameters


def test_retraction_is_the_anchored_prox_of_the_signed_point():
    import nsa_flow
    z = _z().abs() * torch.sign(torch.randn(40, 4, generator=torch.Generator().manual_seed(1)))
    z = z * torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    # make every column already in its canonical gauge so no flip occurs
    from pysimlr.sparsification import _resolve_column_sign_gauge
    z = _resolve_column_sign_gauge(z)
    diag = {}
    y = simlr_sparseness(z, constraint_type="orth", positivity="positive",
                         constraint_weight=0.5, energy_type="regression",
                         retraction_diagnostics=diag)
    # the same prox, called directly on the SIGNED z with the anchor fidelity
    ref = nsa_flow.nsa_flow(z / (float(torch.linalg.norm(z)) / z.shape[1] ** 0.5),
                            w=0.5, mode="anchored", fidelity="anchor", nonneg=True)
    ref_y = ref.Y * (float(torch.linalg.norm(z)) / z.shape[1] ** 0.5)
    assert torch.allclose(y, ref_y, atol=1e-8, rtol=1e-6)
    assert diag["fidelity_mode"] == "anchor"
    assert (y >= 0).all()


def test_prox_of_signed_point_differs_from_prox_of_clamped_point():
    """If these were equal the fix would be vacuous.  They are not."""
    import nsa_flow
    z = _z(seed=3)
    a = nsa_flow.nsa_flow(z, w=0.5, mode="anchored", fidelity="anchor", nonneg=True).Y
    b = nsa_flow.nsa_flow(z.clamp_min(0), w=0.5, mode="anchored", fidelity="anchor",
                          nonneg=True).Y
    assert not torch.allclose(a, b, atol=1e-6)


def test_fidelity_never_flips_across_iterates():
    """auto would pick subspace here (negative mass ~0.5); the loop must not."""
    for seed in range(5):
        diag = {}
        simlr_sparseness(_z(seed), constraint_type="orth", positivity="positive",
                         constraint_weight=0.5, energy_type="regression",
                         retraction_diagnostics=diag)
        assert diag["fidelity_mode"] == "anchor"
        assert diag["target_negative_mass"] > 0.3          # it really was signed


def test_diagnostics_carry_the_certificate():
    diag = {}
    simlr_sparseness(_z(), constraint_type="orth", positivity="positive",
                     constraint_weight=0.5, energy_type="regression",
                     retraction_diagnostics=diag)
    for key in ("certificate", "converged", "grad_map", "n_grad", "defect_D"):
        assert key in diag, key
    assert diag["certificate"] in ("stationary", "numerical_floor")
    assert diag["converged"] is True


def test_negative_positivity_is_the_reflected_prox():
    z = _z(seed=7)
    y_neg = simlr_sparseness(z, constraint_type="orth", positivity="negative",
                             constraint_weight=0.5, energy_type="regression")
    y_pos = simlr_sparseness(-z, constraint_type="orth", positivity="positive",
                             constraint_weight=0.5, energy_type="regression")
    assert torch.allclose(y_neg, -y_pos, atol=1e-8)
    assert (y_neg <= 0).all()


def test_column_sign_gauge_is_a_symmetry_of_the_retraction():
    """Negating any column of the input must not change the output."""
    z = _z(seed=11)
    y0 = simlr_sparseness(z, constraint_type="orth", positivity="positive",
                          constraint_weight=0.5, energy_type="regression")
    flip = z.clone(); flip[:, 1] *= -1; flip[:, 3] *= -1
    y1 = simlr_sparseness(flip, constraint_type="orth", positivity="positive",
                          constraint_weight=0.5, energy_type="regression")
    assert torch.allclose(y0, y1, atol=1e-8)


def test_all_negative_column_is_flipped_not_zeroed_not_reflected():
    z = _z(seed=2)
    z[:, 0] = -z[:, 0].abs()                          # entirely non-positive column
    y = simlr_sparseness(z, constraint_type="orth", positivity="positive",
                         constraint_weight=0.5, energy_type="regression")
    assert float(torch.linalg.vector_norm(y[:, 0])) > 0.0
    # the mixed-sign columns are projected, not reflected: a negative entry
    # cannot reappear as a positive one
    mixed = z[:, 1]
    assert (y[:, 1][mixed < 0] == 0).all() or float(torch.linalg.vector_norm(
        (-mixed).clamp_min(0))) > float(torch.linalg.vector_norm(mixed.clamp_min(0)))


# ------------------------------------------------ consistency across methods

def test_nsa_flow_optimizer_step_is_the_signfree_anchored_prox():
    import nsa_flow
    from pysimlr.optimizers import NSAFlowOptimizer
    v = [torch.randn(30, 4, dtype=torch.float64)]
    g = torch.randn(30, 4, dtype=torch.float64)
    opt = NSAFlowOptimizer("nsa_flow", v, learning_rate=0.1, nsa_w=0.3)
    out = opt.step(0, v[0], g)
    ref = nsa_flow.nsa_flow((v[0] + 0.1 * g).double(), w=0.3, mode="anchored",
                            fidelity="anchor", nonneg=False).Y
    assert torch.allclose(out.double(), ref, atol=1e-8)


def test_simlr_defaults_do_not_trigger_the_deprecated_path():
    import warnings
    from pysimlr.simlr import simlr
    torch.manual_seed(0)
    X = [torch.randn(60, 8), torch.randn(60, 6)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simlr(X, k=2, iterations=3, verbose=False)
    assert not [x for x in w if issubclass(x.category, DeprecationWarning)
                and "simlr_sparseness" in str(x.message)]


def test_simlr_one_weight_unless_separated():
    """nsa_w defaults to the prox weight parsed from `constraint`."""
    from pysimlr.simlr import parse_constraint
    assert parse_constraint("orthox0.3x1")["weight"] == 0.3


def test_deep_result_bases_are_prox_fixed_points():
    """The returned deep bases lie on the same feasible set as simlr's:
    applying the prox again changes nothing."""
    from pysimlr import lend_simr
    torch.manual_seed(0)
    X = [torch.randn(80, 10), torch.randn(80, 7)]
    res = lend_simr(X, k=3, epochs=3, positivity="positive", nsa_w=0.3,
                    verbose=False)
    assert "retraction_diagnostics" in res
    for v, d in zip(res["v"], res["retraction_diagnostics"]):
        assert d.get("fidelity_mode") == "anchor", d
        again = simlr_sparseness(v.double(), constraint_type="orth", positivity="positive",
                                 constraint_weight=0.3, energy_type="regression",
                                 unit_columns=True)
        assert torch.allclose(again, v.double(), atol=1e-6, rtol=1e-5)
        assert (v >= 0).all()
