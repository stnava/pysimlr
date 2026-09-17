"""Finite-difference / autograd validation of the hand-written defect gradients.

`gradient_invariant_orthogonality_defect` previously returned only the inner
term `4 * Ap @ M`, dropping the `1/||A||_F` factor and the projection term
introduced by the Frobenius normalization. The result was not even a positive
multiple of the true gradient (components could differ in sign), and it
violated the scale-invariance identity <A, grad> == 0. Nothing in the suite
validated it.
"""
import pytest
import torch

from pysimlr.utils import (
    gradient_invariant_orthogonality_defect,
    gradient_mean_orthogonality_defect,
    invariant_orthogonality_defect,
    mean_orthogonality_defect,
)

SHAPES = [(6, 3), (10, 4), (4, 4), (20, 5), (3, 7), (8, 2)]

PAIRS = [
    (invariant_orthogonality_defect, gradient_invariant_orthogonality_defect),
    (mean_orthogonality_defect, gradient_mean_orthogonality_defect),
]
IDS = ["invariant", "mean"]


@pytest.mark.parametrize("fn,grad_fn", PAIRS, ids=IDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_manual_gradient_matches_autograd(fn, grad_fn, shape):
    torch.manual_seed(hash(shape) % 10_000)
    a = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    fn(a).backward()
    manual = grad_fn(a.detach())
    assert manual.shape == a.shape
    assert torch.allclose(manual, a.grad, atol=1e-9, rtol=1e-6), (
        f"max abs error {float((manual - a.grad).abs().max()):.3e}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_invariant_gradient_has_no_radial_component(shape):
    """The defect is invariant to rescaling A, so its gradient must be
    orthogonal to A. The old implementation gave <A, grad> = 4*s*defect."""
    torch.manual_seed(7)
    a = torch.randn(*shape, dtype=torch.float64)
    grad = gradient_invariant_orthogonality_defect(a)
    assert abs(float(torch.sum(a * grad))) < 1e-9


@pytest.mark.parametrize("scale", [0.01, 7.0, 1000.0])
def test_invariant_defect_is_scale_invariant_and_gradient_scales_inversely(scale):
    torch.manual_seed(8)
    a = torch.randn(9, 3, dtype=torch.float64)
    assert torch.allclose(
        invariant_orthogonality_defect(a * scale),
        invariant_orthogonality_defect(a),
        atol=1e-10,
    )
    assert torch.allclose(
        gradient_invariant_orthogonality_defect(a * scale),
        gradient_invariant_orthogonality_defect(a) / scale,
        atol=1e-10,
    )


@pytest.mark.parametrize("fn,grad_fn", PAIRS, ids=IDS)
def test_gradient_descends_the_defect(fn, grad_fn):
    """A small step along -grad must decrease the defect."""
    torch.manual_seed(9)
    a = torch.randn(12, 4, dtype=torch.float64)
    g = grad_fn(a)
    before = float(fn(a))
    step = 1e-4 / (float(g.norm()) + 1e-12)
    after = float(fn(a - step * g))
    assert after < before, f"{before:.6e} -> {after:.6e}"


@pytest.mark.parametrize("fn,grad_fn", PAIRS, ids=IDS)
def test_orthonormal_input_is_a_stationary_point(fn, grad_fn):
    q = torch.linalg.qr(torch.randn(10, 4, dtype=torch.float64))[0]
    assert float(fn(q)) < 1e-18
    assert float(grad_fn(q).abs().max()) < 1e-9


@pytest.mark.parametrize("fn,grad_fn", PAIRS, ids=IDS)
def test_degenerate_inputs_are_safe(fn, grad_fn):
    z = torch.zeros(5, 3, dtype=torch.float64)
    assert float(fn(z)) == 0.0
    assert torch.equal(grad_fn(z), torch.zeros_like(z))
    single = torch.randn(5, 1, dtype=torch.float64)
    assert torch.isfinite(torch.as_tensor(float(fn(single))))
    assert torch.isfinite(grad_fn(single)).all()
