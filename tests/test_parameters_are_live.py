"""Every parameter must do something, and new ones must say which.

The bugs this guards are the cheap, humiliating kind: a parameter that is
accepted, documented, swept in a benchmark, and never read. Nothing raises,
the suite stays green, and the sweep produces a table of numbers that are all
the same fit under different labels. Four shipped at once and were found only
by asking "does the fitted basis actually change?":

* ``nsa_w`` reached the optimizer's internal retraction and never the prox --
  and under the default ``optimizer_type="lars"``, which has no retraction, it
  did nothing whatsoever. Identical basis to ``0.00e+00`` across
  ``{0.05, 0.3, 0.5, 0.9}``. This is what made ``w`` look like an irrelevant
  hyperparameter across an entire published design sweep.
* ``use_nsa`` reached ``SIMLR_OPTIMIZER_DEFAULTS`` and stopped. No optimizer
  reads it, so ``simlr(use_nsa=False)`` was a no-op on the linear path.
* ``learning_rate`` was dropped by the benchmark harness before the call.
* ``gauss`` and ``exp`` were two registry names for one expression.

The load-bearing test here is `test_every_parameter_is_classified`. Asserting
that a *chosen* list of parameters works cannot catch the next one someone
adds; requiring every parameter in the signature to be either exercised or
exempted-with-a-reason can, and turns "I forgot to wire it up" into a failing
test at the moment the signature changes.
"""
import inspect

import numpy as np
import pytest
import torch

from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private
from pysimlr.flows import flow_simr_v
from pysimlr.simlr import simlr

N, DIMS, K = 40, (9, 6), 3


def _views(seed=0, n=N):
    rng = np.random.default_rng(seed)
    u = np.abs(rng.standard_normal((n, K)))
    out = []
    for p in DIMS:
        v = np.zeros((p, K))
        blk = p // K
        for j in range(K):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        out.append(torch.tensor(u @ v.T + 0.05 * rng.standard_normal((n, p))).float())
    return out


_SM = [torch.eye(p) * 0.5 + 0.5 / p for p in DIMS]
_DOM = [torch.randn(2, p, generator=torch.Generator().manual_seed(1)) for p in DIMS]

#: parameter -> two values that must produce different fits.
EXERCISED = {
    "init_seed": (1, 2),
    "initialization_type": ({"initialization_type": "pca"},
                            {"initialization_type": "random", "init_seed": 3}),
    "energy_type": ("recon_r2", "acc"),
    "optimizer_type": ("lars", "adam"),
    "mixing_algorithm": ("newton", "svd"),
    "nsa_w": (0.1, 0.9),
    "constraint": ("orthox0.2x1", "none"),
    "positivity": ("positive", "either"),
    "nonneg_u": (False, True),
    "negentropy_weight": (0.0, 5.0),
    "orthogonalize_u": (True, False),
    "scale_list": (["centerAndScale", "np"], ["centerAndScale"]),
    "consolidate": (True, False),
    "smoothing_matrices": (None, _SM),
    "domain_matrices": ({"domain_matrices": None},
                        {"domain_matrices": _DOM, "domain_lambdas": 1.0}),
    "topology": ({"topology": "star"}, {"topology": "loo"}),
    # `iterations` needs a fit with somewhere to go. From the default PCA
    # start on this fixture the *first* iterate is already the best, so
    # `best_v_mats` returns it whatever the budget -- measured identical
    # `best_energy` at 1, 3 and 30 iterations for `hybrid_adam` and `lars`
    # alike. That is correct behaviour rather than an inert parameter, so the
    # test starts somewhere the optimiser has to work from.
    "iterations": ({"iterations": 1, "learning_rate": 0.01, "init_seed": 5},
                   {"iterations": 30, "learning_rate": 0.01, "init_seed": 5}),
}

#: parameter -> why changing it cannot or should not change the fitted basis.
#: Each entry is a claim someone had to make deliberately.
EXEMPT = {
    "data_matrices": "the input itself, not a setting",
    "k": "changes the output shape, so the bases are not comparable; shape is "
         "asserted in test_auto_returns_a_usable_fit and the runner tests",
    "verbose": "prints; must not change numerics",
    "domain_lambdas": "only meaningful with domain_matrices, exercised there",
    "learning_rate": "arrives via **opt_params, covered in "
                     "test_learning_rate_tuning.py and test_knobs_are_live.py",
    "tol": "gates early stopping, so it acts on `stop_reason` rather than on "
           "the basis: `simlr` returns its *best* iterate, and stopping "
           "sooner returns the same one unless a later iterate would have "
           "beaten it. Asserted in test_tol_controls_early_stopping.",
    "path_graph": "read only by topology='graph', where with this module's "
                  "two-view fixture the only possible graph is the complete "
                  "one -- which is leave-one-out, so no two graphs can differ. "
                  "It is live and exercised on three views in "
                  "test_the_complete_graph_reproduces_leave_one_out and the "
                  "consensus tests. Supplying it under another topology now "
                  "warns rather than being silently discarded.",
}

BASE = dict(k=K, iterations=4, positivity="positive", energy_type="recon_r2",
            learning_rate=0.5, verbose=False)


def _fit(views, overrides):
    torch.manual_seed(0)
    np.random.seed(0)
    res = simlr(views, **{**BASE, **overrides})
    return torch.cat([v.detach().flatten() for v in res["v"]])


def _as_overrides(param, value):
    return dict(value) if isinstance(value, dict) else {param: value}


# --------------------------------------------------------------------------
# the anti-rot guard
# --------------------------------------------------------------------------
def test_every_parameter_is_classified():
    """A new `simlr` parameter must be exercised or exempted, never neither.

    Without this, every test below is a list someone curated once. With it,
    adding a parameter and forgetting to wire it up fails here immediately.
    """
    sig = inspect.signature(simlr)
    params = {p for p, q in sig.parameters.items()
              if q.kind not in (q.VAR_KEYWORD, q.VAR_POSITIONAL)}
    classified = set(EXERCISED) | set(EXEMPT)
    missing = sorted(params - classified)
    assert not missing, (
        f"unclassified simlr parameter(s) {missing}: add two values to "
        f"EXERCISED that must change the fit, or an entry to EXEMPT saying "
        f"why they cannot. An unclassified parameter is one nobody has "
        f"checked is wired up.")
    # `**opt_params` names are real settings that are not named parameters;
    # `learning_rate` is the one that matters and is covered elsewhere.
    from pysimlr.optimizers import SIMLR_OPTIMIZER_DEFAULTS
    stale = sorted(classified - params - set(SIMLR_OPTIMIZER_DEFAULTS))
    assert not stale, f"EXERCISED/EXEMPT name parameters simlr no longer has: {stale}"


def test_the_two_tables_do_not_overlap():
    both = sorted(set(EXERCISED) & set(EXEMPT))
    assert not both, f"{both} are both exercised and exempt; pick one"


# --------------------------------------------------------------------------
# every exercised parameter must change the answer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("param", sorted(EXERCISED))
def test_changing_a_parameter_changes_the_fit(param):
    a, b = EXERCISED[param]
    views = _views()
    with pytest.warns(None) if False else _quiet():
        fa = _fit(views, _as_overrides(param, a))
        fb = _fit(views, _as_overrides(param, b))
    assert float((fa - fb).norm()) > 1e-9 * max(1.0, float(fa.norm())), (
        f"simlr({param}={a!r}) and simlr({param}={b!r}) produced a "
        f"bit-identical basis. Either the parameter is not reaching the code "
        f"it names, or these two values are not actually different settings.")


class _quiet:
    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore")
        return self
    def __exit__(self, *a):
        return self._c.__exit__(*a)


#: Parameters deleted from `simlr` because they provably did nothing. Each
#: must still be *recognised* -- accepted, warned about, and not forwarded --
#: so an old call site gets told what to use instead of a confusing
#: "unrecognised optimizer parameter" from three layers down.
REMOVED = {
    "sparseness_quantile": "nsa_w",
    "sparsity": "nsa_w",
    "sparseness": "nsa_w",
    "use_nsa": "constraint",
}


@pytest.mark.parametrize("gone", sorted(REMOVED))
def test_a_removed_parameter_is_named_not_silently_swallowed(gone):
    """Deleting a dead parameter is only half the job.

    Left to fall into ``**opt_params`` these would reach `create_optimizer`
    and produce "Unrecognised optimizer parameter", which tells a caller
    nothing about why their setting stopped mattering -- or, worse, nothing
    at all if the optimizer accepted the name.
    """
    with pytest.warns(DeprecationWarning, match=f"{gone!r} has been removed"):
        simlr(_views(), k=K, iterations=2, verbose=False, **{gone: 0.5})


@pytest.mark.parametrize("gone", sorted(REMOVED))
def test_a_removed_parameter_does_not_reach_the_optimizer(gone):
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        simlr(_views(), k=K, iterations=2, verbose=False, **{gone: 0.5})
    assert not [m for m in caught
                if "Unrecognised optimizer parameter" in str(m.message)], (
        f"{gone} was forwarded into opt_params instead of being intercepted")


@pytest.mark.parametrize("gone", sorted(REMOVED))
def test_a_removed_parameter_changes_nothing(gone):
    """The justification for deleting them: they had no effect. If one of
    these ever starts mattering, it was not dead and should come back."""
    views = _views()
    base = _fit(views, {})
    with _quiet():
        got = _fit(views, {gone: 0.5})
    assert torch.equal(base, got), (
        f"{gone} changed the fit, so removing it was wrong")


# --------------------------------------------------------------------------
# the same property across the other entry points
# --------------------------------------------------------------------------
DEEP_KNOBS = {
    "energy_type": ("align", "acc"),
    "mixing_algorithm": ("newton", "svd"),
    "dynamic_weights": (False, True),
    "nsa_w": (0.1, 0.9),
    "positivity": ("positive", "either"),
}

#: MAI ramps in over a fraction of the run and has nothing to say unless the
#: views actually differ in quality, so the deep fixture adds a pure-noise
#: third view and uses a budget past the ramp. At 2 epochs with two symmetric
#: views `dynamic_weights` is legitimately inert and the test is vacuous.
DEEP_EPOCHS = 8


def _deep_views(seed=0, n=48):
    views = _views(seed=seed, n=n)
    rng = np.random.default_rng(seed + 99)
    views.append(torch.tensor(rng.standard_normal((n, 7))).float())
    return views


def _deep_fit(fn, views, overrides):
    torch.manual_seed(0)
    np.random.seed(0)
    res = fn(views, k=K, epochs=DEEP_EPOCHS, verbose=False, **overrides)
    return torch.cat([v.detach().flatten() for v in res["v"]])


@pytest.mark.parametrize("fn", [ned_simr, ned_simr_shared_private])
@pytest.mark.parametrize("param", sorted(DEEP_KNOBS))
def test_deep_knobs_change_the_fit(fn, param):
    """`ned_simr` and `ned_simr_shared_private` are fast enough for the default
    suite; `lend_simr` (~0.8s/fit) and `flow_simr_v` (~2s/fit) are the same
    test under `slow`."""
    a, b = DEEP_KNOBS[param]
    views = _deep_views()
    with _quiet():
        fa = _deep_fit(fn, views, {param: a})
        fb = _deep_fit(fn, views, {param: b})
    assert float((fa - fb).norm()) > 1e-9 * max(1.0, float(fa.norm())), (
        f"{fn.__name__}({param}={a!r}) and ({param}={b!r}) gave an identical "
        f"basis")


@pytest.mark.slow
@pytest.mark.parametrize("fn", [lend_simr, flow_simr_v])
@pytest.mark.parametrize("param", sorted(DEEP_KNOBS))
def test_slow_deep_knobs_change_the_fit(fn, param):
    a, b = DEEP_KNOBS[param]
    views = _deep_views()
    with _quiet():
        fa = _deep_fit(fn, views, {param: a})
        fb = _deep_fit(fn, views, {param: b})
    assert float((fa - fb).norm()) > 1e-9 * max(1.0, float(fa.norm()))


@pytest.mark.parametrize("fn", [ned_simr, ned_simr_shared_private])
def test_deep_models_respond_to_their_initialisation(fn):
    """Different starting points must give different answers.

    `initialize_simlr` is PCA and deterministic, so a model that only ever
    starts there has one trajectory no matter what seed the caller passes --
    and multi-start, hence every empirical identifiability claim, is
    meaningless. Here the seed drives the network init, which must be enough.
    """
    views = _views(n=30)
    with _quiet():
        torch.manual_seed(0); np.random.seed(0)
        a = torch.cat([v.detach().flatten() for v in fn(views, k=K, epochs=2, verbose=False)["v"]])
        torch.manual_seed(7); np.random.seed(7)
        b = torch.cat([v.detach().flatten() for v in fn(views, k=K, epochs=2, verbose=False)["v"]])
    assert float((a - b).norm()) > 1e-9, f"{fn.__name__} ignores its seed"


# --------------------------------------------------------------------------
# incompatible combinations must refuse, not quietly do something else
# --------------------------------------------------------------------------
def test_a_data_term_is_refused_on_the_deep_path():
    """`recon` and `recon_r2` need X and V; the deep path carries neither into
    the similarity. Silently substituting another term would make
    `energy_type` mean different objectives on different models -- which is
    the bug the shared registry was built to end."""
    from pysimlr.similarity import resolve_energy_name
    for name in ("recon", "recon_r2"):
        with pytest.raises(ValueError, match="does not carry"):
            resolve_energy_name(name, path="deep")


def test_an_ambiguous_energy_name_is_refused():
    from pysimlr.similarity import resolve_energy_name
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_energy_name("regression")


def test_an_unknown_energy_is_refused_not_silently_defaulted():
    with pytest.raises(ValueError, match="unknown similarity"):
        simlr(_views(), k=K, iterations=2, energy_type="nope", verbose=False)


def test_an_unknown_optimizer_is_refused():
    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        simlr(_views(), k=K, iterations=2, optimizer_type="nope", verbose=False)


def test_an_unknown_initialisation_is_refused():
    with pytest.raises(ValueError, match="not implemented"):
        simlr(_views(), k=K, iterations=2, initialization_type="nmf", verbose=False)


def test_an_unrecognised_optimizer_hyperparameter_warns():
    """Accepting a misspelled hyperparameter in silence is how a sweep ends up
    reporting a setting it never ran."""
    with pytest.warns(UserWarning, match="Unrecognised optimizer parameter"):
        simlr(_views(), k=K, iterations=2, verbose=False, lerning_rate=0.1)


# --------------------------------------------------------------------------
# MAI: "the knob changes something" is not enough, it must do the right thing
# --------------------------------------------------------------------------
def _procrustes_r2(Z, U):
    """The MAI similarity, reproduced so the metric can be tested without
    training a network.

    This is a *copy* of the library expression, not a call into it: the metric
    is written inline inside update_mai, of which there are five identical
    copies across deep.py and flows.py. So the test below documents the
    property and pins the algebra, but cannot by itself catch a regression in
    the library -- test_mai_downweights_a_pure_noise_view is what does
    that (verified: reintroducing the clamping form fails three tests).
    Collapsing those five copies into one shared function would let this test
    guard the real thing.
    """
    Z = Z - Z.mean(0); U = U - U.mean(0)
    Z = Z / Z.norm(); U = U / U.norm()
    _, s, _ = torch.linalg.svd(Z.t() @ U, full_matrices=False)
    return float(s.sum().item()) ** 2


def test_the_mai_metric_does_not_saturate_to_zero_on_ordinary_inputs():
    """The bug this pins, in one assertion.

    The shipped form was ``max(0, 1 - ||Z@Omega - u||^2 / ||u||^2)``. With both
    arguments at unit Frobenius norm and ``Omega`` orthogonal that identity is
    ``2*sum(s) - 1``, which is exactly 0 for every ``sum(s) < 0.5`` -- and at
    k=3 every view sits below that cliff. So all modality scores were 0, the
    gate saw no spread, and the weights stayed uniform forever: measured
    [0.333, 0.333, 0.333] at 12 epochs with one view of pure noise.

    Nothing failed, nothing warned, and the MAI arm of the design sweep
    faithfully reported that the choice of MAI metric does not matter.
    """
    g = torch.Generator().manual_seed(0)
    n, k = 60, 3
    signal = torch.randn(n, k, generator=g)
    noise = torch.randn(n, k, generator=g)

    aligned = _procrustes_r2(signal, signal)
    noisy = _procrustes_r2(signal + 1.0 * torch.randn(n, k, generator=g), signal)
    unrelated = _procrustes_r2(noise, signal)

    assert aligned == pytest.approx(1.0, abs=1e-6)
    assert unrelated > 0.0, (
        "an unrelated view scores exactly 0, so it is indistinguishable from "
        "every other unrelated view and the gate has nothing to rank")
    assert aligned > noisy > unrelated, (
        f"the metric does not order signal ({aligned:.4f}) above noisy "
        f"({noisy:.4f}) above unrelated ({unrelated:.4f})")


@pytest.mark.parametrize("fn", [ned_simr, ned_simr_shared_private])
def test_mai_downweights_a_pure_noise_view(fn):
    """The end-to-end claim MAI exists to make.

    A third view of pure noise shares nothing with the other two, so
    leave-one-out agreement should rank it last. Before the metric fix the
    weights were uniform to three decimals however many epochs it ran.
    """
    views = _deep_views()
    with _quiet():
        torch.manual_seed(0); np.random.seed(0)
        res = fn(views, k=K, epochs=12, dynamic_weights=True, verbose=False)
    w = np.asarray(res["modality_weights"], dtype=float)
    assert w.shape == (3,)
    assert w[2] == pytest.approx(w.min()), (
        f"the pure-noise view did not get the smallest weight: {np.round(w, 4)}")
    assert w.max() - w.min() > 0.02, (
        f"weights are effectively uniform ({np.round(w, 4)}); MAI is running "
        f"but discriminating nothing")


def test_mai_weights_stay_a_normalised_distribution():
    views = _deep_views()
    with _quiet():
        torch.manual_seed(0); np.random.seed(0)
        w = np.asarray(ned_simr(views, k=K, epochs=8, dynamic_weights=True,
                                verbose=False)["modality_weights"], dtype=float)
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# a setting that is not understood must be refused, not approximated
# --------------------------------------------------------------------------
def test_an_unknown_topology_is_refused():
    """Every unrecognised topology used to fall through to the `star`
    consensus and return a confident answer for a model nobody asked for.
    A typo, a wrong case, or a plausible-sounding name all did it silently.
    """
    from pysimlr.consensus import compute_shared_consensus
    g = torch.Generator().manual_seed(0)
    lat = [torch.randn(30, K, generator=g) for _ in range(3)]
    for bad in ("banana", "LOO", "path", ""):
        with pytest.raises(ValueError, match="unknown topology"):
            compute_shared_consensus(lat, mixing_algorithm="newton", k=K, topology=bad)


def test_a_path_graph_that_cannot_be_used_warns():
    """`path_graph` is read only by topology='graph'; supplying it elsewhere
    was accepted and discarded."""
    from pysimlr.consensus import compute_shared_consensus
    g = torch.Generator().manual_seed(0)
    lat = [torch.randn(30, K, generator=g) for _ in range(3)]
    with pytest.warns(RuntimeWarning, match="does not read it"):
        compute_shared_consensus(lat, mixing_algorithm="newton", k=K,
                                 topology="loo", path_graph={0: [1]})


def test_the_complete_graph_reproduces_leave_one_out():
    """A property the implementation must satisfy, not a recorded number: if
    every modality neighbours every other, 'graph' *is* leave-one-out."""
    from pysimlr.consensus import compute_shared_consensus
    g = torch.Generator().manual_seed(0)
    lat = [torch.randn(30, K, generator=g) for _ in range(3)]
    loo = compute_shared_consensus(lat, mixing_algorithm="newton", k=K, topology="loo")
    full = compute_shared_consensus(lat, mixing_algorithm="newton", k=K, topology="graph",
                                    path_graph={0: [1, 2], 1: [0, 2], 2: [0, 1]})
    for a, b in zip(loo, full):
        assert torch.allclose(a, b, atol=1e-6)


def test_the_constraint_strings_iterations_field_is_declared_dead():
    """`Type x Weight x Iterations` -- the third field does nothing.

    It reaches `simlr_sparseness` as `constraint_iterations`, which has
    ignored it since the projection became the single constraint operator:
    "orthox0.5x1" and "orthox0.5x9" give a bit-identical basis. The field
    still parses so old strings keep working, but setting it now says so.
    """
    from pysimlr.simlr import parse_constraint
    with pytest.warns(DeprecationWarning, match="iterations field"):
        parse_constraint("orthox0.5x9")
    with _quiet():
        assert parse_constraint("orthox0.5x9")["iterations"] == 9
    views = _views()
    with _quiet():
        a = _fit(views, {"constraint": "orthox0.5x1"})
        b = _fit(views, {"constraint": "orthox0.5x9"})
    assert torch.equal(a, b), (
        "the iterations field now changes the fit, so it is no longer dead "
        "and the warning above should be removed")


# --------------------------------------------------------------------------
# MAI metrics: live, valid, and gauge-invariant
# --------------------------------------------------------------------------
MAI_METRICS = ["procrustes_r2", "procrustes_r2_sharp", "cca", "rvcoef"]


def test_the_mai_metric_list_matches_the_library():
    """Anti-rot, as for the simlr parameters.

    The gauge-invariance test below can only check metrics it knows how to
    score. Pinning the library's set to this list means adding a metric fails
    here until someone has checked it is gauge-invariant -- which is the
    property `trace` failed.
    """
    from pysimlr.deep import VALID_MAI_METRICS
    assert set(VALID_MAI_METRICS) == set(MAI_METRICS), (
        f"the library offers {sorted(VALID_MAI_METRICS)} but this module "
        f"checks {sorted(MAI_METRICS)}; a metric has been added or removed "
        f"without its invariance being verified")
    from pysimlr.flows import VALID_MAI_METRICS as FLOW_METRICS
    assert set(FLOW_METRICS) == set(VALID_MAI_METRICS), (
        "deep.py and flows.py offer different MAI metrics")


def test_every_mai_metric_ranks_a_pure_noise_view_last():
    """Distinctness is not enough -- each metric must be *right*.

    `trace` passed a distinctness check and still ranked a planted pure-noise
    view above a signal view on `flow_simr_v` (0.276 against 0.246).
    """
    views = _deep_views()
    for metric in MAI_METRICS:
        with _quiet():
            torch.manual_seed(0); np.random.seed(0)
            w = np.asarray(ned_simr(views, k=K, epochs=14, dynamic_weights=True,
                                    mai_metric=metric, verbose=False)["modality_weights"],
                           dtype=float)
        assert w.argmin() == 2, (
            f"mai_metric={metric!r} gave the pure-noise view weight "
            f"{w[2]:.3f} out of {np.round(w, 3)}; it is not the smallest")


def test_mai_metrics_are_invariant_to_the_latent_gauge():
    """The property that disqualified `trace`.

    The k latent coordinates carry an arbitrary rotation Q. A similarity
    between two latents must not depend on it, or it scores the coordinate
    system rather than the views. Measured for the removed `trace`:
    0.0348, 0.1319 and 0.0000 for one fixed pair under three rotations.
    """
    g = torch.Generator().manual_seed(0)
    n, k = 80, 3
    Z = torch.randn(n, k, generator=g); U = torch.randn(n, k, generator=g)
    Z = (Z - Z.mean(0)); U = (U - U.mean(0)); Z = Z / Z.norm(); U = U / U.norm()

    def scores(Z, U):
        c = Z.t() @ U
        _, s, _ = torch.linalg.svd(c, full_matrices=False)
        return {"procrustes_r2": float(s.sum()) ** 2,
                "cca": float(s.mean()),
                "rvcoef": float((c.norm() ** 2) /
                                ((Z.t() @ Z).norm() * (U.t() @ U).norm() + 1e-8))}

    base = scores(Z, U)
    for seed in (1, 2, 3):
        q, _ = torch.linalg.qr(torch.randn(k, k, generator=torch.Generator().manual_seed(seed)))
        rotated = scores(Z @ q, U)
        for name, val in base.items():
            assert rotated[name] == pytest.approx(val, rel=1e-6), (
                f"{name} changed from {val:.6f} to {rotated[name]:.6f} under a "
                f"rotation of the latent coordinates, so it measures the "
                f"coordinate system rather than agreement between views")


@pytest.mark.parametrize("bad", ["trace", "banana", "Procrustes_R2"])
def test_an_unknown_mai_metric_is_refused(bad):
    """It must raise, not fall through.

    The unknown-name branch used to *be* `trace`, so a typo silently selected
    the one metric that does not work. And the per-view computation is wrapped
    in `except Exception: mais.append(0.0)`, which would turn the error into
    "this view agrees with nothing" and return plausible uniform weights --
    hence the name is validated before that loop, not inside it.
    """
    with pytest.raises(ValueError, match="unknown mai_metric"):
        with _quiet():
            ned_simr(_deep_views(), k=K, epochs=2, dynamic_weights=True,
                     mai_metric=bad, verbose=False)


def test_tol_controls_early_stopping():
    """`tol` acts on when the loop stops, which is `stop_reason`, not the basis.

    It sat in EXERCISED comparing fitted bases and looked inert. It is not:
    at tol=1e-12 the fit runs to `max_iter` with grad_map 3.9e-04, and at
    tol=10 it stops on `grad_map`. The returned basis is unchanged because
    `simlr` returns its best iterate and the best one here is early, so a
    later stop has nothing better to offer.
    """
    views = _views()
    torch.manual_seed(0); np.random.seed(0)
    tight = simlr(views, k=K, iterations=30, positivity="positive",
                  energy_type="recon_r2", learning_rate=0.01, tol=1e-12,
                  verbose=False)
    torch.manual_seed(0); np.random.seed(0)
    loose = simlr(views, k=K, iterations=30, positivity="positive",
                  energy_type="recon_r2", learning_rate=0.01, tol=10.0,
                  verbose=False)
    # `tol` now drives the shared `ConvergenceMonitor` as well as the
    # stationarity certificate, so a tight tolerance no longer runs to
    # `max_iter` -- the plateau rule stops it first, which is the point of
    # having one criterion. What `tol` still controls, and what this checks,
    # is *how* it stops: an absurd tolerance certifies stationarity almost
    # immediately, a realistic one stops on the plateau much later.
    assert loose["stop_reason"] == "grad_map", (
        f"a tolerance of 10 did not certify immediately "
        f"(stop_reason={loose['stop_reason']!r}); `tol` is not being read")
    assert tight["stop_reason"] == "plateau"
    assert tight["convergence"]["n_steps"] > loose["convergence"]["n_steps"], (
        f"a tight tolerance ({tight['convergence']['n_steps']} steps) did not "
        f"run longer than a loose one ({loose['convergence']['n_steps']})")


# --------------------------------------------------------------------------
# one convergence criterion, set once, meaning the same thing everywhere
# --------------------------------------------------------------------------
CONVERGENCE_KEYS = {"converged", "stop_reason", "certificate", "n_steps",
                    "max_steps", "tol", "patience", "best", "final",
                    "recent_relative_range"}


def test_every_method_reports_the_same_convergence_keys():
    """`simlr` reported `stop_reason`/`certificate`/`grad_map`; the deep
    trainer reported `converged_iter`. Two vocabularies meant no single check
    could answer "did this fit finish?" across the package."""
    from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private
    from pysimlr.flows import flow_simr_v
    views = _views()
    got = {}
    with _quiet():
        got["simlr"] = simlr(views, k=K, iterations=40, positivity="positive",
                             energy_type="recon_r2", verbose=False)
        for fn in (lend_simr, ned_simr, ned_simr_shared_private):
            got[fn.__name__] = fn(views, k=K, epochs=40, verbose=False)
    for name, res in got.items():
        assert "convergence" in res, f"{name} does not report `convergence`"
        assert CONVERGENCE_KEYS <= set(res["convergence"]), (
            f"{name} is missing {sorted(CONVERGENCE_KEYS - set(res['convergence']))}")


def test_an_exhausted_budget_is_not_reported_as_convergence():
    """The distinction the old rules blurred: `simlr`'s plateau test fired
    within three sweeps on a normalised energy, and the deep trainer's
    absolute `tol=1e-6` never fired at all, so both "converged" meant
    "finished running"."""
    views = _views()
    with _quiet():
        starved = simlr(views, k=K, iterations=2, positivity="positive",
                        energy_type="recon_r2", verbose=False)
    c = starved["convergence"]
    assert c["converged"] is False
    assert c["stop_reason"] == "max_iter"
    assert c["n_steps"] == c["max_steps"]


def test_convergence_is_not_an_artifact_of_the_budget():
    """Two generous budgets must reach the same place.

    They did not: on one fixture the deep trainer reported "converged" at a
    loss of 4.45 with a 120-epoch cap and 3.71 with 300, because whether ten
    consecutive non-improvements happened early was down to per-epoch noise.
    The monitor tests a trailing mean for that reason.
    """
    views = _views()
    with _quiet():
        a = simlr(views, k=K, iterations=100, positivity="positive",
                  energy_type="recon_r2", verbose=False)
        b = simlr(views, k=K, iterations=400, positivity="positive",
                  energy_type="recon_r2", verbose=False)
    assert a["convergence"]["converged"] and b["convergence"]["converged"]
    assert a["convergence"]["n_steps"] == b["convergence"]["n_steps"]
    assert a["best_energy"] == pytest.approx(b["best_energy"], rel=1e-6)


def test_every_method_returns_its_best_iterate_not_its_last():
    """`simlr` returned `best_v_mats`; the deep trainers returned their last
    weights. Measured post-warmup on ADNI the last was worse than the best by
    +0.17 (NED), +0.08 (NEDPP) and +0.04 (LEND) -- small, but there is no
    reason to hand back a model the run already knew was worse, and the
    asymmetry made `simlr` and the deep models answer different questions.

    (An earlier version of this measurement compared across the warmup
    boundary and read the gap as +7.63. It is not: the similarity term is off
    during warmup, so pre-warmup epochs optimise a different objective -- the
    loss goes 2.96 at epoch 19 to 10.49 at epoch 20 for that reason alone.)
    """
    from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private
    from pysimlr.flows import flow_simr_v
    views = _views()
    with _quiet():
        for fn in (lend_simr, ned_simr, ned_simr_shared_private, flow_simr_v):
            res = fn(views, k=K, epochs=60, verbose=False)
            c = res["convergence"]
            assert c["returned"] == "best", (
                f"{fn.__name__} returned its final weights, not its best")
            w = c["warmup_epochs"]
            post = [l for i, l in enumerate(res["loss_history"]) if i >= w]
            assert c["best_epoch"] is None or c["best_epoch"] >= w, (
                f"{fn.__name__} selected a best epoch inside warmup "
                f"({c['best_epoch']} < {w}), where the objective is different")
            if post:
                # `best_loss` is the raw loss of the restored iterate;
                # `best` is the monitor's smoothed criterion value, a
                # different quantity.
                assert c.get("best_loss", float("nan")) == pytest.approx(
                    min(post), rel=1e-6), (
                    f"{fn.__name__} restored an iterate whose loss is not the "
                    f"minimum of its post-warmup history")


def test_the_deep_schedule_does_not_depend_on_the_budget():
    """`CosineAnnealingLR(T_max=epochs)` made the epoch cap a hyperparameter:
    at the same step and seed a 120-epoch run sat well down the cosine while a
    600-epoch run was near its initial rate, so identical step counts gave
    different losses (4.47 / 4.29 / 4.26). `ReduceLROnPlateau` responds to the
    loss instead, so a cap is only a cap."""
    from pysimlr.deep import ned_simr
    views = _views()
    with _quiet():
        torch.manual_seed(0); np.random.seed(0)
        short = ned_simr(views, k=K, epochs=80, verbose=False)
        torch.manual_seed(0); np.random.seed(0)
        long = ned_simr(views, k=K, epochs=400, verbose=False)
    assert short["convergence"]["converged"] and long["convergence"]["converged"]
    assert short["convergence"]["n_steps"] == long["convergence"]["n_steps"]
    assert short["convergence"]["best"] == pytest.approx(
        long["convergence"]["best"], rel=1e-6), (
        "two generous budgets reached different optima, so the cap is still "
        "shaping the optimisation rather than bounding it")
