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


# --------------------------------------------------------------------------
# The angle defect C, reported beside D
# --------------------------------------------------------------------------

def test_angle_defect_is_invariant_to_per_column_rescaling():
    """``C`` measures only the angles, which is the point of having it: the
    column scale of a basis feeding a layer that can rescale it is a gauge."""
    from pysimlr.utils import angle_defect

    torch.manual_seed(0)
    v = torch.randn(20, 4, dtype=torch.float64)
    base = float(angle_defect(v))
    scaled = v * torch.tensor([1.0, 10.0, 0.1, 100.0], dtype=torch.float64)
    assert float(angle_defect(scaled)) == pytest.approx(base, rel=1e-9)


def test_angle_defect_separates_decorrelation_from_norm_balance():
    """The reason to report both: ``D`` conflates the two and ``C`` does not."""
    from pysimlr.utils import angle_defect

    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(12, 3, dtype=torch.float64))
    unequal = q.clone()
    unequal[:, 0] *= 50.0

    assert float(orthogonality_defect(unequal)) > 0.5, "D should charge the imbalance"
    assert float(angle_defect(unequal)) < 1e-12, "C should not"


def test_angle_defect_still_penalises_a_dead_column():
    """
    The subtlety: written as a sum over ``i != j``, ``C`` scores a
    rank-collapsed matrix 0.0, because a zero column has zero cosine against
    everything -- the same failure that makes the old invariant defect unusable
    as a penalty. Subtracting the identity charges a floored column
    ``1/(k(k-1))`` for its missing unit diagonal.
    """
    from pysimlr.utils import angle_defect

    collapsed = torch.zeros(10, 3, dtype=torch.float64)
    collapsed[:, 0] = torch.randn(10, dtype=torch.float64)

    assert float(angle_defect(collapsed)) == pytest.approx(1.0 / 3.0), (
        "a rank-one basis must be charged for its two dead columns")
    assert float(angle_defect(collapsed, diagonal=False)) == pytest.approx(0.0), (
        "without the diagonal term the dead-column penalty is gone, which is "
        "why it is on by default")

    one_dead = torch.randn(10, 3, dtype=torch.float64)
    one_dead[:, 1] = 0.0
    assert float(angle_defect(one_dead)) > 0.15


@pytest.mark.parametrize("shape", [(9, 3), (20, 4), (50, 2)])
@pytest.mark.parametrize("diagonal", [True, False])
def test_angle_defect_gradient_agrees_with_autograd(shape, diagonal):
    from pysimlr.utils import angle_defect, gradient_angle_defect

    torch.manual_seed(sum(shape))
    a = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    angle_defect(a, diagonal=diagonal).backward()
    manual = gradient_angle_defect(a.detach(), diagonal=diagonal)
    assert torch.allclose(a.grad, manual, atol=1e-12)


def test_angle_defect_matches_the_backend_including_degenerate_cases():
    """Random matrices are not enough here: the two definitions agree on those
    and diverge exactly where a column norm is floored."""
    nsa_flow = pytest.importorskip("nsa_flow")
    from pysimlr.utils import angle_defect

    torch.manual_seed(0)
    cases = [torch.randn(10, 3, dtype=torch.float64)]
    q, _ = torch.linalg.qr(torch.randn(10, 3, dtype=torch.float64))
    cases.append(q)
    rescaled = q.clone(); rescaled[:, 0] *= 100.0
    cases.append(rescaled)
    collapsed = torch.zeros(10, 3, dtype=torch.float64)
    collapsed[:, 0] = torch.randn(10, dtype=torch.float64)
    cases.append(collapsed)
    one_zero = torch.randn(10, 3, dtype=torch.float64); one_zero[:, 1] = 0.0
    cases.append(one_zero)

    for v in cases:
        assert float(angle_defect(v)) == pytest.approx(
            float(nsa_flow.angle_defect(v)), rel=1e-9, abs=1e-15)


def test_summary_reports_both_defects():
    from pysimlr.utils import orthogonality_summary

    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(10, 3))
    summary = orthogonality_summary(q)
    assert "invariant_defect" in summary and "angle_defect" in summary
    assert summary["invariant_defect"] < 1e-6
    assert summary["angle_defect"] < 1e-6
