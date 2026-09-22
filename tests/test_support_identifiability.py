r"""Can the objective identify the support? Ground truths with known answers.

pysimlr reports sparsity and disjoint supports as results. These tests ask
whether anything in the *objective* rewards getting the support right, or
whether it arrives from the initialisation and the projection and is merely
preserved -- or lost -- by optimisation.

The answer, established here, is the latter. That is not a bug in any solver;
it is a property of the similarity terms, and it explains a cluster of
otherwise baffling benchmark results:

* ``SiMLR-LBFGSB`` scores 0.415 support recovery where every other constrained
  method scores ~0.97, on all 8 seeds, with a normal sparsity of 0.667.
* ``SiMLR`` and ``SiMLR-LBFGS`` agree to four decimals on every metric.
* Across the whole benchmark the least-converged optimiser tends to score best.

Each test below fixes a quantity whose true value is known by construction, so
a regression shows up as a number moving away from a *derivable* answer rather
than away from a recorded one.
"""
import numpy as np
import pytest
import torch

from pysimlr.benchmarks.metrics import support_recovery_score
from pysimlr.benchmarks.synthetic_cases import build_case
from pysimlr.consensus import compute_shared_consensus
from pysimlr.similarity import SimilarityContext, similarity_energy
from pysimlr.simlr import initialize_simlr, simlr
from pysimlr.utils import preprocess_data


def planted(n=300, k=3, dims=(60, 45), noise=0.05, seed=0):
    """Views generated from a known disjoint non-negative basis."""
    rng = np.random.default_rng(seed)
    u = np.abs(rng.standard_normal((n, k)))
    views, truth = [], []
    for p in dims:
        v = np.zeros((p, k))
        blk = p // k
        for j in range(k):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        truth.append(torch.tensor(v).float())
        views.append(torch.tensor(u @ v.T + noise * rng.standard_normal((n, p))).float())
    return views, truth


# --------------------------------------------------------------------------
# GT1: latent-only similarities cannot see the support, by construction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("energy,floor", [
    ("recon_r2", 0.55),
    ("align", 0.70),
    ("nc", 0.85),
])
def test_the_pipeline_recovers_support_under_every_energy(energy, floor):
    r"""This test used to assert the opposite, and the opposite was an artifact.

    It recorded that latent-only terms score 0.5000 -- chance -- against
    1.0000 for ``recon``, and concluded that ``align`` and ``nc`` compare two
    k-dimensional objects and so cannot express a preference between bases
    that span the same subspace with different supports.

    The reasoning about the *energies* is still sound. The measurement was
    not. Under the defaults of the time the three energies scored **exactly**
    0.5000, all three, to four decimals -- and three different objectives
    agreeing exactly is the signature of an optimiser that never moved, not of
    three objectives that cannot tell supports apart. It was: `learning_rate`
    defaulted to 0.001, which moves ``V`` about 0.4% of its norm over a short
    fit, so every run returned its initialisation, whose support is 0.5 on
    this fixture.

    With `learning_rate="auto"` and the current consensus the same call gives
    recon_r2 0.6688, nc 0.9783, align 0.8189 -- they separate, and none is at
    chance. Note the ordering: the *data* term does worst and the latent-only
    ``nc`` does best, the reverse of what the original framing predicts. One
    fixture at one seed, so it is recorded rather than interpreted. So the pipeline does recover the planted support under a
    latent-only energy. That does not resurrect the energies as identifiers:
    the constraint and the initialisation supply the structure (see GT3), and
    the energy then has to not destroy it, which is a weaker claim and the one
    this now checks.
    """
    views, truth = planted()
    torch.manual_seed(0)
    np.random.seed(0)
    res = simlr(views, k=3, iterations=25, optimizer_type="lars",
                energy_type=energy, positivity="positive",
                constraint="orthox0.5x1")
    got = support_recovery_score([v.detach() for v in res["v"]], truth)
    assert got > floor, (
        f"{energy} recovered only {got:.4f} of the planted support, below the "
        f"{floor} recorded for it. If every energy has dropped to the same "
        f"value, check whether the optimiser is moving at all before "
        f"concluding anything about the energies -- that is the mistake this "
        f"test was originally written around."
    )


# --------------------------------------------------------------------------
# GT2: the objective is nearly flat across supports
# --------------------------------------------------------------------------
def test_objective_barely_distinguishes_true_support_from_wrong_one():
    """The load-bearing measurement of this whole module.

    Evaluating ``recon`` at the planted basis and at a support-scrambled basis
    that spans a similar subspace gives energies within a fraction of a percent
    of each other. An objective that cannot separate them cannot be credited
    with recovering the support when a fit happens to.
    """
    views, truth = planted()
    scaled = [preprocess_data(m, ["centerAndScale", "np"])[0] for m in views]
    v_true = [v / v.norm(dim=0, keepdim=True) for v in truth]

    # A basis with deliberately scrambled support spanning a similar subspace:
    # permute rows within each view, which preserves column norms and roughly
    # the spanned subspace while destroying which feature loads on which mode.
    g = torch.Generator().manual_seed(1)
    v_scrambled = [v[torch.randperm(v.shape[0], generator=g)] for v in v_true]

    def total(vs):
        proj = [x @ v for x, v in zip(scaled, vs)]
        u = compute_shared_consensus(proj, mixing_algorithm="newton", k=3)
        u = u[0] if isinstance(u, list) else u
        return sum(float(similarity_energy("recon", x @ v, u,
                                           SimilarityContext(x=x, v=v)))
                   for x, v in zip(scaled, vs))

    e_true, e_scram = total(v_true), total(v_scrambled)
    sup_scram = support_recovery_score(v_scrambled, truth)
    rel = abs(e_scram - e_true) / max(abs(e_true), 1e-12)

    assert sup_scram < 0.6, "the scrambled basis should have poor support"
    assert rel < 0.05, (
        f"recon separates true from scrambled support by only {rel:.2%} "
        f"(E_true={e_true:.2f}, E_scrambled={e_scram:.2f}, scrambled support "
        f"{sup_scram:.3f}). If this ever fails because `rel` grew, the "
        f"objective has become support-sensitive, which would be an "
        f"improvement worth noticing rather than a regression."
    )


# --------------------------------------------------------------------------
# GT3: the initialisation, not the objective, supplies the structure
# --------------------------------------------------------------------------
def test_initialisation_already_carries_most_of_the_support():
    """Zero optimisation steps recover most of the planted support.

    Measured 0.7885 at init on the `nonneg_parts` case against 0.9944 after a
    barely-moving `lars` fit and 0.2790 after a hard-optimising `nsa_lbfgsb`
    fit. Optimisation is not what finds the structure.
    """
    case = build_case(kind="nonneg_parts", seed=42, n_samples=400)
    scaled = [preprocess_data(m, ["centerAndScale", "np"])[0] for m in case["data"]]
    truth = [torch.as_tensor(v).float() for v in case["true_v"]]
    v0 = initialize_simlr(scaled, 3, positivity="positive")
    at_init = support_recovery_score(v0, truth)
    assert at_init > 0.6, (
        f"initialisation support dropped to {at_init:.4f}; the projection or "
        f"the initialiser has regressed, and every downstream support number "
        f"depends on it"
    )


# --------------------------------------------------------------------------
# GT4: optimisers must actually differ, and must actually optimise
# --------------------------------------------------------------------------
def test_optimisers_are_not_silently_equivalent():
    """`SiMLR` and `SiMLR-LBFGS` matched to four decimals on every benchmark
    metric because both stall ~4 orders from stationarity at a similar point.
    This pins that they at least move, and that one of them converges."""
    views, _truth = planted(dims=(40, 30))
    out = {}
    for opt in ("lars", "nsa_lbfgsb"):
        torch.manual_seed(0)
        np.random.seed(0)
        r = simlr(views, k=3, iterations=20, optimizer_type=opt,
                  energy_type="acc", positivity="positive",
                  constraint="orthox0.5x1")
        out[opt] = (torch.cat([v.detach() for v in r["v"]]),
                    float(r["grad_map"]))

    delta = float((out["lars"][0] - out["nsa_lbfgsb"][0]).norm())
    assert delta > 1e-6, (
        "two different optimisers produced bit-identical bases; at least one "
        "is not running"
    )
    assert out["nsa_lbfgsb"][1] < out["lars"][1], (
        f"nsa_lbfgsb should reach a smaller stationarity residual than lars "
        f"(got {out['nsa_lbfgsb'][1]:.2e} vs {out['lars'][1]:.2e})"
    )


def test_better_optimisation_does_not_silently_improve_support():
    """A guard on the conclusion itself, not on a number.

    Support recovery is currently *anti*-correlated with optimiser quality: on
    `nonneg_parts` the harder-converging solver scores 0.279 against 0.994.
    If a future objective change makes the better optimiser also recover the
    better support, this test fails -- and that failure is the signal that the
    objective has become support-identifying, which is the fix this module
    argues for.
    """
    case = build_case(kind="nonneg_parts", seed=42, n_samples=400)
    truth = [torch.as_tensor(v).float() for v in case["true_v"]]
    scaled = [preprocess_data(m, ["centerAndScale", "np"])[0] for m in case["data"]]
    got = {}
    for opt in ("lars", "nsa_lbfgsb"):
        torch.manual_seed(42)
        np.random.seed(42)
        r = simlr(scaled, k=3, iterations=30, optimizer_type=opt,
                  energy_type="recon_r2", positivity="positive",
                  nsa_w=0.5, consolidate=True)
        got[opt] = (support_recovery_score([v.detach() for v in r["v"]], truth),
                    float(r["energy_reduction"]))

    # Neither optimiser meaningfully reduces this energy, and that is the
    # point rather than a caveat. Measured at the `nsa_w=0.5` this test asks
    # for: lars 2.38e-07, nsa_lbfgsb exactly 0.0. At `w=0.1` -- which is what
    # this test *used* to run, because `nsa_w` reached only the optimizer's
    # internal retraction and never the prox, so the projection silently used
    # the default constraint weight -- nsa_lbfgsb managed 1.26e-05 and this
    # assertion read `nsa_lbfgsb > lars`. Once `nsa_w` actually applied, the
    # harder optimiser stalled completely.
    #
    # The support gap is unaffected by any of that (0.545 vs 0.989 at w=0.5,
    # 0.549 vs 0.989 at w=0.1), which is the claim the module is really about.
    # The anti-correlation is gone, and this assertion is the reverse of what
    # it was. It used to `pytest.fail` if the harder-optimising solver did as
    # well as the weaker one, on the recorded finding that support recovery
    # was *anti*-correlated with optimiser quality (0.279 against 0.994). That
    # finding was taken under `energy_type="recon"` -- an objective minimised
    # by driving the basis to zero, so optimising it harder did move the
    # answer further from the truth. Under `recon_r2`, which has a finite
    # non-degenerate minimiser, both solvers land near the truth: lars 0.9892,
    # nsa_lbfgsb 1.0000.
    #
    # Neither reduces the energy measurably (0.000e+00 for both), so this is
    # not yet evidence that optimisation *finds* the support -- the
    # initialisation and the projection still supply it, as GT3 says. What
    # changed is that optimising no longer destroys it.
    for opt in ("lars", "nsa_lbfgsb"):
        assert got[opt][0] > 0.95, (
            f"{opt} recovered only {got[opt][0]:.4f}; under a non-degenerate "
            f"objective both solvers should land near the planted support"
        )
    assert got["nsa_lbfgsb"][0] >= got["lars"][0] - 0.05, (
        f"the harder-optimising solver is again markedly worse "
        f"({got['nsa_lbfgsb'][0]:.4f} vs {got['lars'][0]:.4f}), which is the "
        f"anti-correlation this module recorded under the degenerate `recon` "
        f"objective. Check the energy before relaxing this."
    )
