"""Autograd validation of the hand-written defect gradients."""
import pytest
import torch

from pysimlr.utils import (
    gradient_mean_orthogonality_defect,
    mean_orthogonality_defect,
)

SHAPES = [(6, 3), (10, 4), (4, 4), (20, 5), (3, 7), (8, 2)]

PAIRS = [
    (mean_orthogonality_defect, gradient_mean_orthogonality_defect),
]
IDS = ["mean"]


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
