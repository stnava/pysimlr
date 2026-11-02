"""Regression tests for the 2026-09 correctness audit (docs/audit/AUDIT_2026_09.md).

Each test reproduces the audit's measurement of one finding and asserts the
fixed behaviour. Gradients are checked against central finite differences of
the energy, not against autograd, because the linear gradients are themselves
produced by autograd: agreement with autograd would only show the code calls
autograd, not that the energy and the step direction describe one function.
"""
import numpy as np
import torch
import pytest

from pysimlr.simlr import (SUPPORTED_ENERGY_TYPES, calculate_simlr_energy,
                           calculate_simlr_gradient)
from pysimlr.utils import gradient_orthogonality_defect, orthogonality_defect

N, P, K = 60, 12, 3


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, P, generator=g, dtype=torch.float64)
    v = torch.randn(P, K, generator=g, dtype=torch.float64)
    # Column-standardised consensus, as compute_shared_consensus returns:
    # U'U = (n - 1) I, the regime in which the A1 bug was invisible to a
    # check made with orthonormal U.
    u = torch.randn(N, K, generator=g, dtype=torch.float64)
    u = (u - u.mean(0)) / u.std(0)
    return x, v, u


def _fd_gradient(f, v, h=1e-6):
    grad = torch.zeros_like(v)
    for idx in torch.cartesian_prod(torch.arange(v.shape[0]), torch.arange(v.shape[1])):
        i, j = int(idx[0]), int(idx[1])
        e = torch.zeros_like(v); e[i, j] = h
        grad[i, j] = (f(v + e) - f(v - e)) / (2 * h)
    return grad


def _cos(a, b):
    return float(torch.sum(a * b) / (a.norm() * b.norm()))


SMOOTH = sorted(SUPPORTED_ENERGY_TYPES - {"dat"})


@pytest.mark.parametrize("energy_type", SMOOTH)
def test_every_supported_energy_has_the_gradient_of_its_own_energy(energy_type):
    """A1 (regression direction dropped U'U) and A2 (zero gradients), for every
    objective the linear path accepts: the returned descent direction equals
    minus the finite-difference gradient of the energy."""
    x, v, u = _fixture()
    fd = _fd_gradient(lambda vv: float(calculate_simlr_energy(vv, x, u, energy_type)), v)
    step = calculate_simlr_gradient(v, x, u, energy_type)
    assert fd.norm() > 0, f"{energy_type}: energy is flat at the fixture"
    assert _cos(-fd, step) > 0.9999, energy_type
    assert torch.allclose(step, -fd, rtol=1e-4, atol=1e-6 * float(fd.norm())), energy_type


def test_regression_direction_depends_on_the_iterate():
    """A1: the shipped direction reduced to the constant 2X'U. The fixed one
    changes when V changes, because the true gradient carries V U'U."""
    x, v, u = _fixture()
    d1 = calculate_simlr_gradient(v, x, u, "recon_r2")
    d2 = calculate_simlr_gradient(v @ torch.diag(torch.tensor([1.0, 2.0, 0.5], dtype=v.dtype)) + 0.1, x, u, "recon_r2")
    assert not torch.allclose(d1, d2)


def test_dat_gradient_matches_its_energy():
    x, v, u = _fixture()
    prior = torch.randn(K, P, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    f = lambda vv: float(calculate_simlr_energy(vv, x, u, "dat", lambda_val=0.7, prior_matrix=prior))
    fd = _fd_gradient(f, v)
    step = calculate_simlr_gradient(v, x, u, "dat", lambda_val=0.7, prior_matrix=prior)
    assert _cos(-fd, step) > 0.9999


@pytest.mark.parametrize("energy_type", ["cca", "pca", "ica"])
def test_objectives_without_a_gradient_are_refused(energy_type):
    """A2: these used to return an identically zero gradient, so the optimiser
    never ran while V moved by retraction alone."""
    x, v, u = _fixture()
    with pytest.raises(ValueError):
        calculate_simlr_energy(v, x, u, energy_type)
    with pytest.raises(ValueError):
        calculate_simlr_gradient(v, x, u, energy_type)


def test_orthogonality_term_gradient_matches_its_energy():
    """A4: the 'ortho' constraint adds orthogonality_defect to the energy; the
    step must subtract its gradient. The combined direction used by `simlr`
    (similarity step minus weighted defect gradient) is checked against the
    finite-difference gradient of the combined energy."""
    x, v, u = _fixture()
    w = 0.3
    f = lambda vv: float(calculate_simlr_energy(vv, x, u, "acc") + w * orthogonality_defect(vv))
    fd = _fd_gradient(f, v)
    step = calculate_simlr_gradient(v, x, u, "acc") - w * gradient_orthogonality_defect(v)
    assert _cos(-fd, step) > 0.9999


def test_the_oracle_simlr_hands_its_optimizer_is_consistent():
    """A4 at the source, and the 0.4.0 oracle fix: inside a real `simlr`
    sweep (constraint "ortho", so the orthogonality term is live), the smooth
    energy and the gradient given to the quasi-Newton solver must describe one
    function in float64. They used to score the projected point in float32
    while returning a tangent-projected gradient: slopes 2-14x off."""
    import importlib
    O = importlib.import_module("pysimlr.optimizers")
    simlr = importlib.import_module("pysimlr.simlr").simlr
    seen = []
    orig = O.NSALBFGSB.step

    def spy(self, i, v, g, f=None):
        if not seen:
            E, grad = self.smooth_energy_function, self.gradient_function
            v0 = v.detach().double()
            d = torch.randn(v0.shape, generator=torch.Generator().manual_seed(3), dtype=v0.dtype)
            d /= d.norm()
            h = 1e-5
            fd = (E(v0 + h * d) - E(v0 - h * d)) / (2 * h)
            seen.append((float((-grad(v0) * d).sum()), fd))
        return orig(self, i, v, g, f)

    O.NSALBFGSB.step = spy
    try:
        g = torch.Generator().manual_seed(0)
        z = torch.randn(50, 2, generator=g)
        views = [z @ torch.randn(2, p, generator=g) + 0.3 * torch.randn(50, p, generator=g) for p in (8, 6)]
        simlr(views, k=2, iterations=1, optimizer_type="nsa_lbfgsb", constraint="orthox0.3",
              energy_type="acc", random_state=0)
    finally:
        O.NSALBFGSB.step = orig
    slope, fd = seen[0]
    assert slope == pytest.approx(fd, rel=1e-5, abs=1e-12)


def test_one_orthogonality_defect_for_energy_losses_and_reports():
    """A3: the energy, the deep and flow losses, and the benchmark's reported
    frame defect all use `orthogonality_defect`. On the audit's disjoint
    non-negative basis with unequal column norms the reported number is that
    function's value, not the unnormalised ||V'V - I|| the runner used to compute."""
    import inspect
    from pysimlr import deep, flows
    from pysimlr.benchmarks import metrics, runner
    v = torch.zeros(9, 3, dtype=torch.float64)
    v[0:3, 0] = 1.0; v[3:6, 1] = 3.0; v[6:9, 2] = 10.0
    assert float(metrics.calculate_v_orthogonality([v])) == pytest.approx(float(orthogonality_defect(v)))
    for module in (deep, flows, runner):
        src = inspect.getsource(module)
        for other in ("stiefel_defect(", "angle_defect(", "mean_orthogonality_defect(",
                      "invariant_orthogonality_defect("):
            assert other not in src.replace("import", ""), (module.__name__, other)


def test_view_callbacks_stay_bound_to_their_view():
    """The energy and gradient callbacks are built per view; called after the
    sweep they must still evaluate their own view. Defined in the loop they read
    the loop index late, and view 0's gradient computed with view 2's data
    raised a shape mismatch."""
    import importlib
    O = importlib.import_module("pysimlr.optimizers")
    simlr = importlib.import_module("pysimlr.simlr").simlr
    kept = {}
    orig = O.NSALBFGSB.step

    def spy(self, i, v, g, f=None):
        kept.setdefault(i, (self.smooth_energy_function, self.gradient_function, v.detach().clone()))
        return orig(self, i, v, g, f)

    O.NSALBFGSB.step = spy
    try:
        g = torch.Generator().manual_seed(0)
        z = torch.randn(40, 2, generator=g)
        views = [z @ torch.randn(2, p, generator=g) + 0.3 * torch.randn(40, p, generator=g) for p in (8, 6, 4)]
        simlr(views, k=2, iterations=1, optimizer_type="nsa_lbfgsb", random_state=0)
    finally:
        O.NSALBFGSB.step = orig
    for i, (E, grad, v) in kept.items():
        assert grad(v).shape == v.shape, i
        assert np.isfinite(E(v)), i
