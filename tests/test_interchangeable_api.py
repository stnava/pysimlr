"""Can an evaluation loop treat the methods as interchangeable?

The usability contract this pins:

    for alg in ALGORITHMS:
        result = run(alg, data)
        evaluate(result, data)

with no per-method branch anywhere in it. That requires two things to hold,
and neither is guaranteed by the individual unit tests: the entry points must
accept one common parameter set, and they must return one common result
structure rich enough to score.

This module found a real bug rather than only documenting a contract. Under
the default `newton` mixing, `ned_simr` scored 0.034 here -- noise -- while
its own per-view latents scored 0.784 and 0.785. The cause was sign
cancellation in the averaging consensus; see `test_sign_alignment.py`. With
that fixed the same call scores 0.795.

It is a usability test, not a benchmark. The performance floor is deliberately
loose -- it catches a method that returns noise, not a method that is 0.05
behind another. Hard comparisons belong in the design sweeps, where budgets,
seeds and cohorts are controlled.

Kept fast on purpose (about three seconds for all five) so it can run on every
edit; a contract test nobody runs is not a contract.
"""
import time

import numpy as np
import pytest
import torch

from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private
from pysimlr.flows import flow_simr_v
from pysimlr.simlr import simlr

pytest.importorskip("sklearn")

#: Every multi-view entry point that should be drop-in interchangeable.
#: `pca` and `nsa_pipeline` are deliberately absent: they are sklearn
#: pipelines reached through `benchmarks.runner`, not pysimlr model functions,
#: and they fit one joint basis rather than a basis per view.
ALGORITHMS = [simlr, lend_simr, ned_simr, ned_simr_shared_private, flow_simr_v]

#: The parameter set the loop is allowed to use. Anything not here would make
#: the caller special-case a method.
COMMON_PARAMS = dict(k=2, energy_type="acc", mixing_algorithm="newton",
                     nsa_w=0.5, positivity="positive", verbose=False)

#: Result keys every method must provide for a generic `evaluate` to work.
REQUIRED_KEYS = {"u", "v", "convergence", "effective_rank", "numerical_rank",
                 "condition_number", "max_column_overlap"}

K = 2
BUDGET = 25


def _planted(seed=0, n=80, dims=(10, 8)):
    """A strong, easy signal: the floor below should be trivial to clear."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, K))
    views = [torch.tensor(u @ rng.standard_normal((K, p))
                          + 0.15 * rng.standard_normal((n, p))).float()
             for p in dims]
    y = u[:, 0] + 0.5 * u[:, 1]
    return views, y


def run(alg, views, budget=BUDGET):
    """One call shape for every algorithm."""
    torch.manual_seed(0)
    np.random.seed(0)
    return alg(views, iterations=budget, **COMMON_PARAMS)


def evaluate(result, y):
    """One scorer, using only keys in the shared contract."""
    from sklearn.linear_model import LinearRegression
    u = result["u"]
    u = u[0] if isinstance(u, list) else u
    u = u.detach().cpu().numpy()
    return float(LinearRegression().fit(u, y).score(u, y))


# --------------------------------------------------------------------------
# the loop itself, which is the actual subject of this module
# --------------------------------------------------------------------------
def test_the_evaluation_loop_runs_with_no_per_method_branch():
    views, y = _planted()
    scores = {}
    for alg in ALGORITHMS:                      # exactly the documented loop
        scores[alg.__name__] = evaluate(run(alg, views), y)
    assert len(scores) == len(ALGORITHMS)
    for name, s in scores.items():
        assert np.isfinite(s), f"{name} produced a non-finite score"


@pytest.mark.parametrize("alg", ALGORITHMS, ids=lambda f: f.__name__)
def test_every_algorithm_accepts_the_common_parameters(alg):
    views, _ = _planted()
    run(alg, views)          # raises if a parameter is not accepted


@pytest.mark.parametrize("alg", ALGORITHMS, ids=lambda f: f.__name__)
def test_either_budget_name_is_accepted(alg):
    """`simlr` counts iterations and the deep models count epochs. They are
    different units, but they occupy the same slot, and a caller should not
    have to know which name a method uses to give it a budget."""
    views, _ = _planted()
    torch.manual_seed(0); np.random.seed(0)
    a = alg(views, iterations=8, **COMMON_PARAMS)
    torch.manual_seed(0); np.random.seed(0)
    b = alg(views, epochs=8, **COMMON_PARAMS)
    ua = a["u"][0] if isinstance(a["u"], list) else a["u"]
    ub = b["u"][0] if isinstance(b["u"], list) else b["u"]
    assert ua.shape == ub.shape


@pytest.mark.parametrize("alg", ALGORITHMS, ids=lambda f: f.__name__)
def test_every_algorithm_returns_the_shared_structure(alg):
    views, _ = _planted()
    res = run(alg, views)
    missing = sorted(REQUIRED_KEYS - set(res))
    assert not missing, f"{alg.__name__} does not return {missing}"

    u = res["u"]
    u = u[0] if isinstance(u, list) else u
    assert u.shape == (views[0].shape[0], K), f"{alg.__name__}: u is {tuple(u.shape)}"
    assert torch.isfinite(u).all()

    assert len(res["v"]) == len(views)
    for v, m in zip(res["v"], views):
        assert v.shape == (m.shape[1], K)
        assert torch.isfinite(v).all()
        assert (v.abs().sum(dim=0) > 0).all(), f"{alg.__name__} returned a dead column"

    for field in ("effective_rank", "numerical_rank", "condition_number",
                  "max_column_overlap"):
        assert len(res[field]) == len(views)


@pytest.mark.parametrize("alg", ALGORITHMS, ids=lambda f: f.__name__)
def test_every_algorithm_reports_convergence_the_same_way(alg):
    views, _ = _planted()
    c = run(alg, views)["convergence"]
    for field in ("converged", "stop_reason", "n_steps", "max_steps", "tol",
                  "patience", "best", "final"):
        assert field in c, f"{alg.__name__} convergence is missing {field}"
    assert isinstance(c["converged"], (bool, np.bool_))
    assert isinstance(c["stop_reason"], str)
    assert 0 < c["n_steps"] <= c["max_steps"]


@pytest.mark.parametrize("alg", ALGORITHMS, ids=lambda f: f.__name__)
def test_every_algorithm_recovers_an_easy_planted_signal(alg):
    """A floor, not a ranking.

    The latent is two orthogonal factors observed through two noisy linear
    views; anything that finds structure at all clears this. Measured at this
    budget: simlr 0.22, ned 0.77, nedpp 0.81, lend 0.88, flow 0.88 -- so 0.15
    fails only a method returning noise. The spread is real and worth knowing,
    but it is a question for the design sweeps, not for a contract test.
    """
    views, y = _planted()
    score = evaluate(run(alg, views), y)
    assert score > 0.15, (
        f"{alg.__name__} scored {score:.3f} on a strongly planted signal, "
        f"which is the level of a method returning noise")


def test_the_shared_contract_has_not_shrunk():
    """Anti-rot: the intersection of the result keys must still cover the
    contract, so a method cannot quietly stop returning part of it."""
    views, _ = _planted()
    shared = None
    for alg in ALGORITHMS:
        keys = set(run(alg, views))
        shared = keys if shared is None else (shared & keys)
    missing = sorted(REQUIRED_KEYS - shared)
    assert not missing, (
        f"these keys are no longer returned by every algorithm: {missing}")
