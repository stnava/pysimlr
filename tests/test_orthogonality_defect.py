"""
The orthogonality penalty must not be minimised by rank collapse.

`invariant_orthogonality_defect` normalises by the global Frobenius norm and
sums the off-diagonal Gram entries. A matrix with one non-zero column and
``k - 1`` zero columns attains exactly 0.0 under it, because a zero column is
orthogonal to everything -- so the penalty SiMLR added to its energy, and the
deep models added to their losses, was minimised by discarding components. It
is also not a function of the angles: normalising globally rather than per
column lets it be reduced by inflating a single column.

`orthogonality_defect` is the trace-normalised defect ``||G - I/k||_F^2`` with
``G = A'A/tr(A'A)``, which obeys ``D >= 1/r - 1/k`` at rank ``r``.
"""

import numpy as np
import pytest
import torch

from pysimlr.utils import (gradient_orthogonality_defect,
                           invariant_orthogonality_defect,
                           orthogonality_defect)


def test_rank_collapse_is_penalised_not_rewarded():
    """The defect the old measure got exactly wrong."""
    collapsed = torch.zeros(10, 3)
    collapsed[:, 0] = torch.randn(10)

    assert float(invariant_orthogonality_defect(collapsed)) == pytest.approx(0.0), (
        "fixture no longer reproduces the old measure's zero")
    assert float(orthogonality_defect(collapsed)) == pytest.approx(1.0), (
        "a rank-one basis must score the ceiling, not the floor")


@pytest.mark.parametrize("k,rank", [(3, 1), (4, 2), (5, 3), (5, 4)])
def test_the_rank_floor_holds(k, rank):
    """``D >= 1/r - 1/k`` at rank ``r``, normalized by ``1 - 1/k``."""
    torch.manual_seed(k * 10 + rank)
    v = torch.zeros(20, k, dtype=torch.float64)
    v[:, :rank] = torch.randn(20, rank, dtype=torch.float64)
    floor = (1.0 / rank - 1.0 / k) / (1.0 - 1.0 / k)
    assert float(orthogonality_defect(v)) >= floor - 1e-9


def test_zero_exactly_on_an_orthonormal_basis():
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(12, 4, dtype=torch.float64))
    assert float(orthogonality_defect(q)) < 1e-12


def test_orthogonal_but_unequal_norms_is_not_zero():
    """``D`` charges norm imbalance as well as correlation -- its zero set is
    orthonormality, which is what "Stiefel" promises."""
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(12, 3, dtype=torch.float64))
    q[:, 0] *= 5.0
    assert float(orthogonality_defect(q)) > 0.1


def test_invariant_to_a_global_rescaling():
    torch.manual_seed(1)
    v = torch.randn(20, 4, dtype=torch.float64)
    base = float(orthogonality_defect(v))
    for scale in [1e-6, 0.1, 10.0, 1e6]:
        assert float(orthogonality_defect(v * scale)) == pytest.approx(base, rel=1e-9)


def test_cannot_be_gamed_by_inflating_one_column():
    """
    The old measure fell from 2.9e-15 to 6.0e-19 when one column of an
    orthonormal basis was scaled by 100, with every angle unchanged. A measure
    used as a penalty must not reward that.
    """
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(10, 3, dtype=torch.float64))
    inflated = q.clone()
    inflated[:, 0] *= 100.0

    assert float(invariant_orthogonality_defect(inflated)) < \
        float(invariant_orthogonality_defect(q)), (
        "fixture no longer reproduces the old measure's gameability")
    assert float(orthogonality_defect(inflated)) > \
        float(orthogonality_defect(q))


def test_matches_the_spectral_identity():
    """``D_raw = 1/EffRank - 1/k`` with ``EffRank = (tr S)^2 / ||S||_F^2``."""
    torch.manual_seed(2)
    for p, k in [(20, 3), (50, 5), (12, 4)]:
        v = torch.randn(p, k, dtype=torch.float64)
        s_mat = v.t() @ v
        eff_rank = float(torch.diagonal(s_mat).sum() ** 2 / torch.sum(s_mat ** 2))
        expected = 1.0 / eff_rank - 1.0 / k
        assert float(orthogonality_defect(v, normalized=False)) == \
            pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("shape", [(8, 3), (20, 4), (50, 2), (13, 5)])
@pytest.mark.parametrize("normalized", [True, False])
def test_closed_form_gradient_agrees_with_autograd(shape, normalized):
    torch.manual_seed(sum(shape))
    a = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    orthogonality_defect(a, normalized=normalized).backward()
    manual = gradient_orthogonality_defect(a.detach(), normalized=normalized)
    assert torch.allclose(a.grad, manual, atol=1e-12)


def test_gradient_satisfies_the_euler_identity():
    """``D`` is homogeneous of degree zero, so ``<grad D, A> = 0`` and the
    defect term cannot alter ``||A||_F``."""
    torch.manual_seed(3)
    for p, k in [(10, 3), (30, 4)]:
        a = torch.randn(p, k, dtype=torch.float64)
        g = gradient_orthogonality_defect(a)
        assert abs(float((g * a).sum())) < 1e-10 * float(a.norm()) ** 2 + 1e-12


def test_k_equals_one_is_vacuous():
    """Orthogonality of a single column is undefined; report zero rather than
    dividing by ``1 - 1/k = 0``."""
    v = torch.randn(10, 1)
    assert float(orthogonality_defect(v)) == 0.0
    assert torch.equal(gradient_orthogonality_defect(v), torch.zeros_like(v))


def test_all_zero_basis_reports_the_ceiling_not_a_nan():
    v = torch.zeros(10, 3)
    d = float(orthogonality_defect(v))
    assert np.isfinite(d) and d == pytest.approx(1.0)
    assert torch.isfinite(gradient_orthogonality_defect(v)).all()


def test_agrees_with_the_backend_when_it_is_installed():
    nsa_flow = pytest.importorskip("nsa_flow")
    torch.manual_seed(4)
    for p, k in [(10, 3), (30, 4), (76, 5)]:
        v = torch.randn(p, k, dtype=torch.float64)
        assert float(orthogonality_defect(v)) == pytest.approx(
            float(nsa_flow.stiefel_defect_normalised(v)), rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__])
