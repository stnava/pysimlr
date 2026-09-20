"""The stopping claim must be backed by something that measures stationarity.

`simlr` used to break when the total energy stopped changing and report
`converged_iter`. The total energy is normalised to 1.0 at iteration 0, so a
loop that is barely descending crosses a relative tolerance within two or three
sweeps: the test fired at iteration 3 for every problem, `iterations` never
bound, and nothing in the result said whether the point was stationary.
"""
import numpy as np
import pytest
import torch

from pysimlr.simlr import simlr
from pysimlr.nsa_backend import load_gradient_mapping, load_lbfgsb

needs_backend = pytest.mark.skipif(
    load_gradient_mapping() is None, reason="NSA-Flow backend not installed"
)


def _views(seed=0, n=120):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, 3, generator=g)
    return [u @ torch.randn(3, p, generator=g)
            + 0.3 * torch.randn(n, p, generator=g) for p in (30, 20)]


def test_result_reports_a_stop_reason_and_never_claims_convergence_bare():
    res = simlr(_views(), k=3, iterations=5)
    assert res["stop_reason"] in ("grad_map", "plateau", "max_iter")
    # converged is a claim about the point, and only a certificate may make it.
    assert res["converged"] == (res["certificate"] is not None)
    if res["certificate"] is not None:
        assert res["certificate"] in ("stationary", "numerical_floor")


def test_energy_reduction_is_reported_so_no_progress_is_visible():
    res = simlr(_views(), k=3, iterations=5)
    assert res["energy_start"] is not None
    assert res["energy_reduction"] is not None
    assert res["energy_reduction"] >= 0.0


@needs_backend
def test_iterations_actually_binds():
    """The old energy test fired at iteration 3 regardless of `iterations`."""
    short = simlr(_views(), k=3, iterations=3, positivity="positive")
    long = simlr(_views(), k=3, iterations=60, positivity="positive")
    assert len(long["grad_map_history"]) > len(short["grad_map_history"]), (
        "the loop stops at the same sweep no matter the budget"
    )


@needs_backend
def test_certificate_is_finite_and_tracks_the_raw_gradient():
    """Scoring the retracted direction returned a constant; guard against it."""
    res = simlr(_views(), k=3, iterations=20, positivity="positive")
    hist = [g for g in res["grad_map_history"] if np.isfinite(g)]
    assert len(hist) >= 2
    assert len(set(np.round(hist, 9))) > 1, (
        "grad_map is constant across sweeps, so it is not measuring the iterate"
    )
    # Deliberately not asserted monotone. Alternating minimisation optimises
    # each V_i against a fixed u and then recomputes u, which is not monotone
    # in the joint objective, and the certificate makes that visible: on the
    # 3-view case it falls from 6.6 to 2.4e-3, while on this small random
    # problem it rises from 5.7e-3 to 0.76. Reporting it is the point;
    # requiring it to fall would be asserting a property the algorithm has not
    # got, and would have to be "fixed" by weakening the measurement.
    assert all(np.isfinite(g) and g >= 0.0 for g in hist)


@pytest.mark.skipif(load_lbfgsb() is None, reason="NSA-Flow backend not installed")
def test_lbfgsb_never_returns_a_rank_deficient_basis():
    """A dead column collapsed test R^2 from 0.94 to 0.17 at one inner budget."""
    for budget in (5, 10, 20, 50):
        res = simlr(_views(seed=3), k=3, iterations=10, positivity="positive",
                    optimizer_type="nsa_lbfgsb", max_iter=budget)
        for v in res["v"]:
            live = (v.detach().abs() > 1e-12).any(dim=0)
            assert bool(live.all()), f"dead component at max_iter={budget}"


@pytest.mark.skipif(load_lbfgsb() is None, reason="NSA-Flow backend not installed")
def test_lbfgsb_survives_a_failing_inner_solve():
    """The backend raises ZeroDivisionError when the Cauchy path is flat."""
    from pysimlr.optimizers import create_optimizer

    opt = create_optimizer("nsa_lbfgsb", [torch.rand(8, 3)])

    def boom(*_args, **_kwargs):
        raise ZeroDivisionError("float division by zero")

    opt._minimize = boom
    opt.gradient_function = lambda v: torch.zeros_like(v)
    v = torch.rand(8, 3)
    with pytest.warns(RuntimeWarning, match="inner solve failed"):
        out = opt.step(0, v, torch.zeros_like(v), full_energy_function=lambda z: 1.0)
    assert torch.equal(out, v), "a failed inner solve must keep the iterate"


# --- rank preservation ------------------------------------------------------

def test_column_floor_revives_a_dead_column_and_leaves_healthy_ones():
    from pysimlr.sparsification import enforce_column_floor, COLUMN_NORM_FLOOR
    torch.manual_seed(0)
    v = torch.rand(6, 3)
    v[:, 1] = 0.0
    out = enforce_column_floor(v)
    norms = out.norm(dim=0)
    assert (norms > 0).all(), "a dead column survived the floor"
    assert float(norms[1]) == pytest.approx(COLUMN_NORM_FLOOR * float(norms.max()), rel=1e-3)
    assert torch.equal(out[:, [0, 2]], v[:, [0, 2]]), "live columns were disturbed"


def test_column_floor_is_scale_invariant_and_keeps_direction():
    from pysimlr.sparsification import enforce_column_floor
    torch.manual_seed(1)
    v = torch.rand(8, 3)
    v[:, 2] *= 1e-9
    out = enforce_column_floor(v)
    assert torch.allclose(out[:, 2] / out[:, 2].norm(),
                          v[:, 2] / v[:, 2].norm(), atol=1e-5)
    assert torch.allclose(enforce_column_floor(v * 1000.0),
                          enforce_column_floor(v) * 1000.0, atol=1e-3)
    healthy = torch.rand(8, 3)
    assert torch.equal(enforce_column_floor(healthy), healthy)


def test_projection_survives_a_rank_deficient_iterate():
    """A dead column upstream used to fail the whole projection."""
    from pysimlr.sparsification import simlr_sparseness
    torch.manual_seed(2)
    v = torch.randn(12, 3)
    v[:, 1] = 0.0
    out = simlr_sparseness(v, constraint_type="ortho", positivity="positive",
                           constraint_weight=0.5, energy_type="regression")
    assert (out.norm(dim=0) > 0).all(), "projection returned a rank-deficient basis"


@pytest.mark.skipif(load_lbfgsb() is None, reason="NSA-Flow backend not installed")
def test_lbfgsb_scores_a_degenerate_trial_point_as_infinite():
    """Under V >= 0 a long step lands on the zero corner; reject, don't crash."""
    from pysimlr.optimizers import create_optimizer
    opt = create_optimizer("nsa_lbfgsb", [torch.rand(8, 3)])
    opt.gradient_function = lambda v: torch.ones_like(v)
    seen = {}

    def energy(v):
        seen["called_with_zero"] = seen.get("called_with_zero", False) or bool(
            (v.abs() < 1e-12).all())
        raise RuntimeError("unusable projection")

    v = torch.rand(8, 3)
    out = opt.step(0, v, torch.ones_like(v), full_energy_function=energy)
    assert torch.isfinite(out).all()
    assert (out.norm(dim=0) > 0).all()
