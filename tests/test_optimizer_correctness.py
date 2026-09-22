"""Optimizer contract tests.

Defects covered:
  * `filter_params` built its result solely from a `defaults` dict that omitted
    `decay_rate`, `beta`, `k` and `alpha`, so those four hyperparameters were
    discarded even when passed explicitly -- LARS, RMSProp and Lookahead were
    permanently stuck on their fallbacks. Unknown keys (typos) vanished too.
  * `bidirectional_linesearch` chose its direction by comparing the two *step
    sizes*, which says nothing about which direction descends further.
  * `create_optimizer` silently substituted HybridAdam for an unrecognised name.
  * `BidirectionalArmijoGradient` updated a momentum buffer and then ignored it.
  * `larslow`'s trust-ratio fallback omitted `trust_coefficient`, giving any
    zero-initialised parameter 1000x the intended learning rate.
"""
import warnings

import numpy as np
import pytest
import torch

from pysimlr import simlr
from pysimlr.optimizers import (
    SIMLR_OPTIMIZER_DEFAULTS,
    bidirectional_linesearch,
    create_optimizer,
)
from pysimlr.utils import procrustes_r2

ALL_OPTIMIZERS = [
    "hybrid_adam", "adam", "nadam", "rmsprop", "gd", "armijo_gradient",
    "bidirectional_armijo_gradient", "lookahead", "bidirectional_lookahead",
    "nsa_flow", "torch_adamw", "torch_adagrad", "torch_nadam", "torch_lbfgs",
    "lars",
]


# ------------------------------------------------------------ hyperparameters

@pytest.mark.parametrize("opt,key,value", [
    ("lars", "decay_rate", 0.25),
    ("rmsprop", "beta", 0.5),
    ("lookahead", "k", 3),
    ("lookahead", "alpha", 0.75),
    ("adam", "beta1", 0.5),
    ("adam", "learning_rate", 0.123),
    ("nsa_flow", "nsa_w", 0.42),
])
def test_hyperparameters_reach_the_optimizer(opt, key, value):
    o = create_optimizer(opt, [torch.randn(8, 3)], **{key: value})
    assert o.params[key] == value, (
        f"{opt}: {key}={value} was dropped (got {o.params.get(key)!r})"
    )


def test_lookahead_uses_the_supplied_period_and_interpolation():
    o = create_optimizer("lookahead", [torch.randn(8, 3)], k=3, alpha=0.75)
    assert o.k == 3 and o.alpha == 0.75


def test_defaults_cover_every_key_the_optimizers_read():
    for key in ("learning_rate", "beta1", "beta2", "beta", "epsilon",
                "weight_decay", "amsgrad", "momentum", "nsa_w", "decay_rate",
                "k", "alpha"):
        assert key in SIMLR_OPTIMIZER_DEFAULTS, key


def test_unknown_hyperparameter_warns_rather_than_vanishing():
    with pytest.warns(UserWarning, match="Unrecognised optimizer parameter"):
        create_optimizer("adam", [torch.randn(4, 2)], learnign_rate=0.5)


def test_known_hyperparameters_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        create_optimizer("lars", [torch.randn(4, 2)], learning_rate=0.1, decay_rate=0.2)


def test_unknown_optimizer_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        create_optimizer("hybird_adam", [torch.randn(4, 2)])


# ------------------------------------------------------------- line search

def test_bidirectional_search_picks_the_lower_energy_direction():
    """A landscape where the wrong-signed direction admits the larger step.

    The energy decreases steeply for negative offsets and only gently for
    positive ones, so a step-size comparison prefers the shallow direction.
    """
    v0 = torch.zeros(1, 1)

    def energy(v):
        t = float(v.reshape(-1)[0])
        return 100.0 * t if t > 0 else 10.0 * t   # decreasing as t -> -inf

    direction = torch.ones(1, 1)
    step, chosen = bidirectional_linesearch(
        v0, direction, -direction, energy, initial_step_size=1.0
    )
    assert step > 0
    moved = v0 + step * chosen
    assert energy(moved) < energy(v0), (
        f"line search moved uphill: {energy(v0)} -> {energy(moved)}"
    )
    assert float(chosen.reshape(-1)[0]) < 0, "should have flipped the direction"


def test_bidirectional_search_returns_zero_when_no_descent_exists():
    v0 = torch.zeros(2, 2)
    step, _ = bidirectional_linesearch(
        v0, torch.ones(2, 2), -torch.ones(2, 2),
        lambda v: float(torch.sum(v ** 2)) + 1.0, initial_step_size=1.0
    )
    assert step == 0.0


# --------------------------------------------------------- end-to-end sanity

def _coupled(seed=0, n=120, k=3):
    torch.manual_seed(seed)
    z = torch.randn(n, k)
    return z, [
        z @ torch.randn(k, 25) + 0.4 * torch.randn(n, 25),
        z @ torch.randn(k, 18) + 0.4 * torch.randn(n, 18),
    ]


@pytest.mark.parametrize("optimizer_type", ALL_OPTIMIZERS)
def test_every_optimizer_recovers_a_strong_shared_latent(optimizer_type):
    """The consensus is pinned so this measures the optimizer.

    `u` is produced by the mixing method, not by the optimizer, so leaving
    `mixing_algorithm` at its default made this test move whenever that
    default moved -- and it did, from "svd" to "newton". Measured over 8 seeds
    on this fixture, latent recovery is 0.987 under svd and ica but 0.784
    under newton and avg, so every optimizer "failed" a 0.8 threshold that had
    been calibrated against a different consensus. Which mixing recovers this
    latent best is a real question, and a separate one from whether an
    optimizer works; it belongs in the design sweep, not here.
    """
    z, x = _coupled()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = simlr(x, k=3, iterations=12, optimizer_type=optimizer_type,
                    energy_type="acc", mixing_algorithm="svd")
    u = res["u"]
    u = u[0] if isinstance(u, list) else u
    assert torch.isfinite(u).all()
    for v in res["v"]:
        assert torch.isfinite(v).all()
    r2 = procrustes_r2(z, u)
    assert r2 > 0.8, f"{optimizer_type}: latent recovery only {r2:.4f}"


@pytest.mark.parametrize("optimizer_type", ["armijo_gradient",
                                            "bidirectional_armijo_gradient"])
def test_momentum_buffer_is_actually_used(optimizer_type):
    """Both Armijo variants smooth the direction; a zero gradient followed by a
    non-zero one must still move the iterate on the second step."""
    o = create_optimizer(optimizer_type, [torch.zeros(6, 2)], learning_rate=0.1)
    v = torch.zeros(6, 2)
    g = torch.ones(6, 2)
    v1 = o.step(0, v, g)
    assert not torch.allclose(v1, v), "first step did not move"
    # with a zero gradient the momentum buffer should still carry the direction
    v2 = o.step(0, v1, torch.zeros(6, 2))
    assert not torch.allclose(v2, v1), (
        f"{optimizer_type}: zero gradient produced no motion, so the momentum "
        f"buffer is not being used as the search direction"
    )


def test_larslow_trust_ratio_fallback_is_scaled():
    """A zero-initialised parameter must not receive an unscaled step."""
    import torch.nn as nn
    from pysimlr.deep import larslow

    p = nn.Parameter(torch.zeros(4, 3))
    opt = larslow([p], lr=1.0, trust_coefficient=1e-3, weight_decay=0.0, momentum=0.0)
    p.grad = torch.ones(4, 3)
    opt.step()
    moved = float(p.detach().abs().max())
    assert moved == pytest.approx(1e-3, rel=1e-6), (
        f"expected a trust_coefficient-scaled step of 1e-3, got {moved:.3e} "
        f"(1.0 would mean the fallback dropped trust_coefficient)"
    )


# ----------------------------------------------- constraint string parsing

@pytest.mark.parametrize("spec,expected_type,expected_weight", [
    ("ortho", "ortho", 0.5),
    ("nsaflow", "nsaflow", 0.5),
    ("Stiefel", "Stiefel", 1.0),
    ("Grassmann", "Grassmann", 1.0),
    ("NewtonSchulz", "NewtonSchulz", 1.0),
    ("none", "none", 0.0),
    ("orthox0", "ortho", 0.0),
    ("orthox0.5x3", "ortho", 0.5),
])
def test_constraint_defaults(spec, expected_type, expected_weight):
    from pysimlr.simlr import parse_constraint

    got = parse_constraint(spec)
    assert got["type"] == expected_type
    assert got["weight"] == pytest.approx(expected_weight)


def test_bare_ortho_actually_constrains():
    """constraint='ortho' defaulted to weight 0.0, and every soft-orthogonality
    branch is gated on weight > 0, so it applied nothing at all."""
    from pysimlr.simlr import parse_constraint

    assert parse_constraint("ortho")["weight"] > 0.0


def test_unknown_constraint_warns():
    from pysimlr.simlr import parse_constraint

    with pytest.warns(UserWarning, match="Unrecognised constraint type"):
        parse_constraint("Orthoo")


def test_parse_constraint_iterations():
    from pysimlr.simlr import parse_constraint

    assert parse_constraint("orthox0.3x7")["iterations"] == 7
    assert parse_constraint("ortho")["iterations"] == 1
