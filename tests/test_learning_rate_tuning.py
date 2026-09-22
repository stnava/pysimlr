"""`tune_learning_rate`: pick a step size by running the real model.

The default ``learning_rate=0.001`` is not right for the three optimizers this
package ships, and cannot be: measured on one planted fixture with a three-
sweep probe, the best candidate is 1.0 for `simlr`, 0.01 for `lend_simr`
(an interior optimum -- 3.80 / 3.53 / 3.64 either side) and 0.001 for
`flow_simr_v`, whose loss rises monotonically with the step and reaches
3.7e+09 at lr=1. No single constant serves all of them.

The tuner therefore probes by *running the model*. The tempting alternative --
choose ``lr`` by evaluating :math:`f(v - lr\\,g)` -- silently assumes plain
gradient descent. LARS steps :math:`lr\\,(\\lVert v\\rVert/\\lVert g\\rVert)\\hat g`,
so its displacement is ``lr * ||v||`` *independently of the gradient*, and
Adam's is ~``lr`` per coordinate. An analytic probe returns a confident number
that means nothing to the optimizer consuming it, and nothing about the result
would look wrong.

One thing these tests deliberately do not assert: that tuning improves the
*answer*. Measured over 6 seeds on a planted basis it halves the energy
(0.4916 -> 0.1911) and **lowers** support recovery (0.6713 -> 0.5615). The
tuner is a fix for an optimiser that does not move, not for an objective that
points the wrong way, and the inert default was masking the second problem
rather than avoiding it. That is why ``learning_rate="auto"`` is opt-in.
"""
import numpy as np
import pytest
import torch

from pysimlr.optimizers import LR_PROBE_GRID, tune_learning_rate
from pysimlr.simlr import simlr

N, DIMS, K = 40, (9, 6), 3


def _views(seed=0):
    rng = np.random.default_rng(seed)
    u = np.abs(rng.standard_normal((N, K)))
    out = []
    for p in DIMS:
        v = np.zeros((p, K))
        blk = p // K
        for j in range(K):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        out.append(torch.tensor(u @ v.T + 0.05 * rng.standard_normal((N, p))).float())
    return out


def _fake(scores, key="best_energy", v=None):
    """A stand-in fit whose energy is a lookup, so selection logic is testable
    without paying for a solve."""
    v = v if v is not None else [torch.ones(4, 2)]
    def fit(learning_rate, iterations):
        return {key: scores[learning_rate], "v": v}
    return fit


# --------------------------------------------------------------------------
# selection logic
# --------------------------------------------------------------------------
def test_it_picks_the_lowest_scoring_candidate():
    scores = {1.0: 5.0, 0.1: 1.0, 0.01: 3.0}
    best, got = tune_learning_rate(_fake(scores), candidates=list(scores))
    assert best == 0.1
    assert got == scores


def test_a_history_is_scored_by_its_last_entry_not_its_minimum():
    """The deep models return their *final* weights, so the final loss is the
    one describing what the caller gets. Scoring by the minimum would credit a
    run for an iterate it discarded -- and would systematically prefer a step
    size that spikes low once and then diverges.
    """
    fit = _fake({1.0: [9.0, 0.01, 50.0], 0.1: [9.0, 2.0, 1.0]}, key="loss_history")
    best, scores = tune_learning_rate(fit, candidates=[1.0, 0.1])
    assert scores[1.0] == 50.0 and scores[0.1] == 1.0
    assert best == 0.1, "a diverging run was selected on the strength of one iterate"


def test_candidates_that_blow_up_or_collapse_are_rejected_not_scored():
    """A diverging probe must not win by returning nan (which compares oddly)
    or by collapsing the basis to zero (which can make an energy look small)."""
    dead = [torch.zeros(4, 2)]
    good = [torch.ones(4, 2)]

    def fit(learning_rate, iterations):
        if learning_rate == 1.0:
            return {"best_energy": float("nan"), "v": good}
        if learning_rate == 0.3:
            return {"best_energy": -1e9, "v": dead}      # collapsed basis
        if learning_rate == 0.1:
            raise RuntimeError("solver failed")
        return {"best_energy": 2.0, "v": good}

    best, scores = tune_learning_rate(fit, candidates=[1.0, 0.3, 0.1, 0.01])
    assert best == 0.01
    assert set(scores) == {0.01}, f"a degenerate candidate was scored: {scores}"


def test_all_candidates_failing_reraises_the_real_error():
    """When every candidate raises, the failure is not about the step size.

    This used to assert the generic "no learning rate produced a usable fit".
    That message names the wrong component: once `learning_rate="auto"` became
    the default, a broken backend or a misspelled energy reached the user as a
    complaint about learning rates, pointing them at the tuner instead of at
    their actual mistake.
    """
    def fit(learning_rate, iterations):
        raise RuntimeError("the backend is on fire")
    with pytest.raises(RuntimeError, match="the backend is on fire"):
        tune_learning_rate(fit, candidates=[1.0, 0.1])


def test_no_usable_score_without_an_error_still_reports_the_tuner():
    """The other branch: probes that *run* but return nothing scorable."""
    def fit(learning_rate, iterations):
        return {"best_energy": float("nan"), "v": [torch.ones(4, 2)]}
    with pytest.raises(RuntimeError, match="no learning rate"):
        tune_learning_rate(fit, candidates=[1.0, 0.1])


def test_the_grid_starts_at_one_and_descends():
    assert LR_PROBE_GRID[0] == 1.0
    assert list(LR_PROBE_GRID) == sorted(LR_PROBE_GRID, reverse=True)


# --------------------------------------------------------------------------
# end to end through simlr
# --------------------------------------------------------------------------
def test_auto_beats_the_default_on_the_energy_it_optimises():
    views = _views()
    kw = dict(k=K, iterations=6, positivity="positive",
              energy_type="recon_r2", verbose=False)
    torch.manual_seed(0); np.random.seed(0)
    default = simlr(views, **kw)["best_energy"]
    torch.manual_seed(0); np.random.seed(0)
    auto = simlr(views, learning_rate="auto", **kw)["best_energy"]
    assert auto <= default, (
        f"auto tuning ({auto:.6g}) did worse than the default ({default:.6g}) "
        f"on the objective it selects for, which means the probe and the fit "
        f"have diverged")


def test_auto_returns_a_usable_fit():
    views = _views()
    res = simlr(views, k=K, iterations=6, positivity="positive",
                energy_type="recon_r2", learning_rate="auto", verbose=False)
    assert len(res["v"]) == len(views)
    for v, m in zip(res["v"], views):
        assert v.shape == (m.shape[1], K)
        assert torch.isfinite(v).all()
        assert float(v.abs().sum()) > 0


def test_an_unknown_learning_rate_string_is_refused():
    with pytest.raises(ValueError, match="must be a float or 'auto'"):
        simlr(_views(), k=K, iterations=2, learning_rate="fast", verbose=False)


def test_auto_is_deterministic():
    views = _views()
    kw = dict(k=K, iterations=5, positivity="positive", energy_type="recon_r2",
              learning_rate="auto", verbose=False)
    torch.manual_seed(0); np.random.seed(0); a = simlr(views, **kw)["v"][0]
    torch.manual_seed(0); np.random.seed(0); b = simlr(views, **kw)["v"][0]
    assert torch.equal(a, b)
