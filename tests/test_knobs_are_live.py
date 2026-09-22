"""Every knob must change the answer. Fast enough to run on every edit.

This module exists because of a class of bug that all of the repository's
other tests pass over: a parameter that is accepted, documented, and swept in
a benchmark, but which never reaches the code it names. Nothing errors, every
existing test passes, and the sweep reports a real-looking table of numbers
that are all the same underlying fit.

Three that shipped simultaneously:

* ``nsa_w`` reached the optimizer's internal retraction only, never the prox
  that runs every sweep -- and with the default ``optimizer_type="lars"``,
  which has no retraction, it did nothing at all. Measured across
  ``nsa_w in {0.05, 0.3, 0.5, 0.9}`` the fitted basis was identical to
  ``0.00e+00``. This is what made ``w`` look like an irrelevant
  hyperparameter across an entire design sweep.
* ``filter_kwargs`` in the benchmark runner kept only *named* parameters,
  while `simlr` takes optimizer hyperparameters through ``**opt_params``, so
  ``learning_rate`` was silently dropped from every benchmark call ever run.
* ``gauss`` and ``exp`` were two registry entries evaluating the identical
  expression, so every sweep over the energies ran one objective twice and
  reported it as two independent results.

The assertions below are deliberately of the form "this must not be
*bit-identical*" rather than "this must differ by at least X". The failure
mode is exact inertness, so an exact test catches it without being sensitive
to problem size, seed, or how far the optimiser happens to travel.
"""
import inspect

import numpy as np
import pytest
import torch

from pysimlr.benchmarks.runner import filter_kwargs
from pysimlr.consensus import compute_shared_consensus
from pysimlr.similarity import SIMILARITY, SimilarityContext, similarity_energy
from pysimlr.simlr import simlr

# Small enough that the whole module runs in about a second, with a planted
# signal so the optimiser has something to descend.
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


# A learning rate the optimiser can actually act on. LARS moves V by exactly
# `lr` of its norm per sweep, so at the 0.001 default a short fit ends where it
# started and *no* knob can change anything -- these tests would pass vacuously.
BASE = dict(k=K, iterations=4, positivity="positive", energy_type="recon_r2",
            learning_rate=0.5, verbose=False)


def _fit(views, **kw):
    torch.manual_seed(0)
    np.random.seed(0)
    res = simlr(views, **{**BASE, **kw})
    return torch.cat([v.detach().flatten() for v in res["v"]])


def _differs(a, b):
    return float((a - b).norm()) > 1e-9 * max(1.0, float(a.norm()))


# --------------------------------------------------------------------------
# the general property: a knob that does nothing is a bug
# --------------------------------------------------------------------------
@pytest.mark.parametrize("knob,values", [
    ("energy_type", ["recon_r2", "acc", "nc", "align"]),
    ("mixing_algorithm", ["newton", "avg", "svd"]),
    ("nsa_w", [0.1, 0.5, 0.9]),
    ("optimizer_type", ["lars", "adam"]),
    ("positivity", ["positive", "either"]),
    ("negentropy_weight", [0.0, 5.0]),
    ("nonneg_u", [False, True]),
])
def test_changing_a_knob_changes_the_fitted_basis(knob, values):
    """Each setting must produce a distinct basis.

    A knob whose settings all return the identical fit is either unwired or
    redundant, and in both cases every benchmark row that varies it is a
    duplicate row wearing a different label.
    """
    views = _views()
    fits = {v: _fit(views, **{knob: v}) for v in values}
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            assert _differs(fits[a], fits[b]), (
                f"{knob}={a!r} and {knob}={b!r} produced a bit-identical "
                f"basis, so this knob is not reaching the code it names")


def test_learning_rate_changes_the_fit():
    """Separate from the sweep above because it travels a different route.

    `simlr` takes it through ``**opt_params``, not as a named parameter, which
    is exactly why the benchmark harness dropped it.
    """
    views = _views()
    assert _differs(_fit(views, learning_rate=0.01), _fit(views, learning_rate=0.5))


def test_the_optimiser_actually_moves_the_basis():
    """Guards the precondition every test above depends on.

    If the fit does not move off its initialisation, no objective and no
    regulariser can change it, and the knob tests would all pass vacuously
    while measuring nothing.
    """
    from pysimlr.simlr import initialize_simlr
    from pysimlr.utils import preprocess_data

    views = _views()
    scaled = [preprocess_data(m, ["centerAndScale", "np"])[0] for m in views]
    v0 = torch.cat([v.flatten() for v in initialize_simlr(scaled, K, positivity="positive")])
    moved = float((_fit(views) - v0).norm() / v0.norm())
    assert moved > 1e-3, (
        f"the fit travelled {moved:.2e} from its initialisation; at the "
        f"default learning_rate=0.001 LARS moves V by `lr` of its norm per "
        f"sweep, so a short fit returns the initialisation and every knob "
        f"above becomes untestable")


def test_the_default_learning_rate_contributes_something():
    """Everything above overrides `learning_rate`, so nothing there guards the
    *default* -- the configuration every benchmark actually ran.

    The comparison is against ``learning_rate=0``, not against the
    initialisation. Measuring distance from the initialisation does not work:
    the prox runs every sweep and moves ``V`` on its own, so a fit with a
    completely dead optimiser still lands far from where it started. A first
    attempt at this test asserted exactly that and passed happily with the
    default learning rate mutated to zero.

    The bound is loose on purpose and is not a claim the default is well
    chosen -- it buys 1.8e-3 of relative displacement here, and LARS moves
    ``V`` by exactly ``lr`` of its norm per sweep, so at 0.001 the fit is
    substantially its initialisation. That is a known, separate problem. This
    pins only that the optimiser is contributing at all.
    """
    views = _views()
    torch.manual_seed(0); np.random.seed(0)
    default = simlr(views, k=K, iterations=6, positivity="positive",
                    energy_type="recon_r2", verbose=False)
    default = torch.cat([v.detach().flatten() for v in default["v"]])
    torch.manual_seed(0); np.random.seed(0)
    frozen = simlr(views, k=K, iterations=6, positivity="positive",
                   energy_type="recon_r2", learning_rate=0.0, verbose=False)
    frozen = torch.cat([v.detach().flatten() for v in frozen["v"]])
    assert _differs(default, frozen), (
        "the fit at the default learning rate is bit-identical to one taken "
        "with a zero step, so the optimiser contributes nothing and every "
        "benchmark row is the initialisation put through the projection")


def test_a_larger_learning_rate_travels_further():
    """Catches `learning_rate` being accepted and then ignored, which is
    distinct from it being dropped before the call (covered separately)."""
    from pysimlr.simlr import initialize_simlr
    from pysimlr.utils import preprocess_data

    views = _views()
    scaled = [preprocess_data(m, ["centerAndScale", "np"])[0] for m in views]
    v0 = torch.cat([v.flatten() for v in initialize_simlr(scaled, K, positivity="positive")])
    dist = []
    for lr in (0.001, 0.5):
        dist.append(float((_fit(views, learning_rate=lr) - v0).norm() / v0.norm()))
    assert dist[1] > dist[0] * 5, (
        f"a 500x larger learning rate moved the basis {dist[1]:.2e} against "
        f"{dist[0]:.2e}; the step size is not reaching the update")


# --------------------------------------------------------------------------
# the specific mechanisms, pinned individually
# --------------------------------------------------------------------------
def test_nsa_w_and_the_constraint_string_mean_the_same_weight():
    """`nsa_w` must drive the prox, not only the optimizer's own retraction.

    It previously reached `opt_params` alone. Under `optimizer_type="lars"`,
    which has no retraction, that made it inert.
    """
    views = _views()
    via_arg = _fit(views, nsa_w=0.9)
    via_str = _fit(views, constraint="orthox0.9x1")
    assert not _differs(via_arg, via_str), (
        "nsa_w=0.9 and constraint='orthox0.9x1' gave different fits; the two "
        "spellings of one weight have diverged again")


def test_conflicting_weight_spellings_warn():
    with pytest.warns(RuntimeWarning, match="different NSA-Flow weights"):
        _fit(_views(), nsa_w=0.9, constraint="orthox0.2x1")


def test_filter_kwargs_keeps_what_the_callee_takes_via_var_keyword():
    """The harness bug, tested without running a fit.

    Keeping only named parameters silently discarded every optimizer
    hyperparameter `simlr` accepts through ``**opt_params``.
    """
    kept = filter_kwargs(simlr, {"learning_rate": 0.2, "energy_type": "recon_r2"})
    assert "learning_rate" in kept, (
        "learning_rate was dropped on the way to simlr; benchmark conditions "
        "that differ only in it are not distinct conditions")
    assert "energy_type" in kept


def test_filter_kwargs_warns_rather_than_dropping_in_silence():
    with pytest.warns(RuntimeWarning, match="cannot accept"):
        filter_kwargs(simlr, {"a_key_simlr_has_never_heard_of": 1})


def test_modality_weights_change_the_consensus():
    """MAI acts only through the consensus, so if that ignored the weights the
    whole mechanism would be decorative."""
    g = torch.Generator().manual_seed(0)
    lat = [torch.randn(30, K, generator=g) for _ in range(3)]
    flat = compute_shared_consensus(lat, mixing_algorithm="newton", k=K,
                                    modality_weights=torch.tensor([1 / 3, 1 / 3, 1 / 3]))
    tilt = compute_shared_consensus(lat, mixing_algorithm="newton", k=K,
                                    modality_weights=torch.tensor([0.9, 0.05, 0.05]))
    flat = flat[0] if isinstance(flat, list) else flat
    tilt = tilt[0] if isinstance(tilt, list) else tilt
    assert _differs(flat, tilt), "compute_shared_consensus ignored modality_weights"


# --------------------------------------------------------------------------
# the registry: two names for one function is a duplicated benchmark row
# --------------------------------------------------------------------------
def test_no_two_registry_terms_compute_the_same_function():
    """`exp` and `gauss` were bit-identical (8.956381485164) under two
    different descriptions."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 7, generator=g, dtype=torch.float64)
    v = torch.randn(7, 3, generator=g, dtype=torch.float64)
    u = torch.randn(40, 3, generator=g, dtype=torch.float64)
    ctx = SimilarityContext(x=x, v=v)
    vals = {n: float(similarity_energy(n, x @ v, u, ctx)) for n in SIMILARITY}
    names = sorted(vals)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert vals[a] != vals[b], (
                f"registry entries {a!r} and {b!r} returned the identical "
                f"value {vals[a]!r}; if they are the same contrast, one should "
                f"be an alias so a sweep runs it once")


# --------------------------------------------------------------------------
# learning_rate must reach every optimizer that documents it
# --------------------------------------------------------------------------
ALL_OPTIMIZERS = ["lars", "adam", "nadam", "rmsprop", "gd", "hybrid_adam",
                  "armijo_gradient", "bidirectional_armijo_gradient",
                  "lookahead", "bidirectional_lookahead", "nsa_flow",
                  "torch_adamw", "torch_adagrad", "torch_nadam"]

#: `nsa_lbfgsb` chooses its own step by line search inside the backend, so a
#: caller-supplied `learning_rate` genuinely has nothing to set. Every other
#: optimizer documents the parameter and must honour it.
LR_EXEMPT_OPTIMIZERS = {"nsa_lbfgsb": "L-BFGS-B picks its step by line search "
                                      "in the backend; `max_iter` is its live "
                                      "budget control"}


@pytest.mark.parametrize("opt", ALL_OPTIMIZERS)
def test_learning_rate_reaches_every_optimizer(opt):
    """Five optimizers silently ignored it.

    `hybrid_adam`, `armijo_gradient`, `bidirectional_armijo_gradient`,
    `lookahead` and `bidirectional_lookahead` seed their line search from
    ``state['last_step_size']``, which was initialised to a hardcoded 0.01.
    Because that key therefore always existed, the
    ``state.get('last_step_size', torch.tensor(lr))`` fallbacks never fired,
    and `learning_rate` was read only in the ``full_energy_function is None``
    branch -- which `simlr` never takes, because it always passes one.

    Measured across lr from 1e-4 to 1.0 the fitted basis was *bit-identical*
    for all five, so on the Diabetes sweep they returned the same test R2 at
    every learning rate: 0.3725, 0.3724, 0.3714, 0.3709, 0.3695 repeated
    across the whole grid.
    """
    views = _views()
    torch.manual_seed(0); np.random.seed(0)
    lo = _fit(views, optimizer_type=opt, learning_rate=1e-4)
    torch.manual_seed(0); np.random.seed(0)
    hi = _fit(views, optimizer_type=opt, learning_rate=1.0)
    assert _differs(lo, hi), (
        f"optimizer_type={opt!r} produced a bit-identical basis at "
        f"learning_rate 1e-4 and 1.0, so the step size is not reaching it")


def test_the_learning_rate_exemptions_are_still_exemptions():
    """Anti-rot: if an exempt optimizer starts honouring `learning_rate`, it
    should move into the parametrized test above rather than stay excused."""
    views = _views()
    for opt, why in LR_EXEMPT_OPTIMIZERS.items():
        torch.manual_seed(0); np.random.seed(0)
        lo = _fit(views, optimizer_type=opt, learning_rate=1e-4)
        torch.manual_seed(0); np.random.seed(0)
        hi = _fit(views, optimizer_type=opt, learning_rate=1.0)
        assert not _differs(lo, hi), (
            f"{opt} now responds to learning_rate ({why}); move it into "
            f"ALL_OPTIMIZERS")
