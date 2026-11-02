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


_DEFAULT_CERT_CACHE = {}


def _default_cert_fit():
    if "res" not in _DEFAULT_CERT_CACHE:
        _DEFAULT_CERT_CACHE["res"] = simlr(_views(), k=3, iterations=5)
    return _DEFAULT_CERT_CACHE["res"]


@pytest.mark.xdist_group("solver_cert_default")
def test_result_reports_a_stop_reason_and_never_claims_convergence_bare():
    res = _default_cert_fit()
    assert res["stop_reason"] in ("converged", "max_iter", "no_descent")
    # converged is a claim about the point, and only a certificate may make it.
    assert res["converged"] == (res["certificate"] is not None)
    if res["certificate"] is not None:
        assert res["certificate"] in ("stationary", "numerical_floor")


def test_a_zero_step_is_not_a_stationarity_certificate(monkeypatch):
    """A line search that finds no decrease returns step 0, so the iterate is
    unchanged and its fixed-point residual is exactly 0. That used to be
    certified "stationary" (seen under topology='loo' with tol=1e-12, where the
    gradient mapping at the certified point was 3.7-7.8)."""
    from pysimlr import optimizers
    monkeypatch.setattr(optimizers.ArmijoGradient, "step",
                        lambda self, i, v, g, f=None: v)
    res = simlr(_views(), k=3, iterations=10, tol=1e-12, optimizer_type="armijo_gradient")
    assert res["stop_reason"] == "no_descent"
    assert res["certificate"] is None and res["converged"] is False
    assert all(d.get("zero_step") for d in res["v_retraction"])


@pytest.mark.xdist_group("solver_cert_default")
def test_energy_reduction_is_reported_so_no_progress_is_visible():
    res = _default_cert_fit()
    assert res["energy_start"] is not None
    assert res["energy_reduction"] is not None
    assert res["energy_reduction"] >= 0.0


@needs_backend
def test_iterations_actually_binds():
    """The old energy test fired at iteration 3 regardless of `iterations`."""
    short = simlr(_views(), k=3, iterations=3, positivity="positive")
    long = simlr(_views(), k=3, iterations=60, positivity="positive")
    assert len(long["energy"]) > len(short["energy"]), (
        "the loop stops at the same sweep no matter the budget"
    )


def test_face_residual_is_zero_on_a_stationary_face_and_flags_the_support():
    """`utils.stationarity_residual`: zero where the gradient vanishes on the
    support, whatever it is off the support; the off-support ratio reports
    a coordinate the gradient would admit."""
    from pysimlr.utils import stationarity_residual
    v = torch.tensor([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0]], dtype=torch.float64)
    g = torch.zeros_like(v)
    g[2, 0] = -0.4          # off the support of column 0, pointing inward
    on, off = stationarity_residual(v, g, nonneg=True)
    assert on == 0.0
    t = float(v.norm() ** 2)
    assert off == pytest.approx(t * 0.4 / 0.5)
    g[0, 0] = 0.1           # on the support: no longer stationary
    assert stationarity_residual(v, g, nonneg=True)[0] > 0.0
    # scale invariance of the face residual for a 1/scale gradient
    on1 = stationarity_residual(v, g)[0]
    on2 = stationarity_residual(3.0 * v, g / 3.0)[0]
    assert on1 == pytest.approx(on2, rel=1e-9)


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
                           constraint_weight=0.5, energy_type="recon")
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


@pytest.mark.skipif(load_lbfgsb() is None, reason="NSA-Flow backend not installed")
def test_lbfgsb_refuses_missing_energy_or_gradient_callable():
    """NSALBFGSB.step must raise rather than silently taking an unconstrained step."""
    from pysimlr.optimizers import create_optimizer

    v = torch.rand(8, 3)
    opt = create_optimizer("nsa_lbfgsb", [v])
    with pytest.raises(ValueError, match="requires both an energy function and gradient_function"):
        opt.step(0, v, torch.ones_like(v), full_energy_function=lambda z: 1.0)

    opt.gradient_function = lambda z: torch.ones_like(z)
    with pytest.raises(ValueError, match="requires both an energy function and gradient_function"):
        opt.step(0, v, torch.ones_like(v), full_energy_function=None)


@pytest.mark.skipif(load_lbfgsb() is None, reason="NSA-Flow backend not installed")
@pytest.mark.parametrize("positivity", ["positive", "either"])
def test_lbfgsb_is_reproducible_under_float32_input_perturbations(positivity):
    """Inputs 1e-8 apart used to diverge to 0.06-0.83 subspace gaps under
    unanchored quasi-Newton steps over the indefinite block objective."""
    rng = np.random.default_rng(0)
    u = rng.standard_normal((60, 2))
    views = [
        torch.tensor(np.abs(u @ rng.standard_normal((2, p))) + 0.2 * rng.standard_normal((60, p))).float()
        for p in (10, 8, 6)
    ]
    kw = dict(k=3, iterations=20, optimizer_type="nsa_lbfgsb",
              positivity=positivity, random_state=0, verbose=False)
    base = simlr(views, **kw)["v"]
    for seed in range(3):
        g = torch.Generator().manual_seed(100 + seed)
        pert = [v + 1e-8 * torch.randn(v.shape, generator=g) for v in views]
        got = simlr(pert, **kw)["v"]
        for va, vb in zip(base, got):
            qa, _ = torch.linalg.qr(va.double())
            qb, _ = torch.linalg.qr(vb.double())
            subspace_gap = float(torch.linalg.matrix_norm(qa @ qa.T - qb @ qb.T, ord=2))
            assert subspace_gap < 1e-4, (
                f"nsa_lbfgsb ({positivity}, seed={seed}) subspace gap {subspace_gap:.2e} >= 1e-4"
            )

