"""The central interface: one registry, one `fit`, one `predict`, one record
of which parameters each method used.

What this pins:
  * every registered method fits and predicts through `pysimlr.fit` /
    `pysimlr.predict` with one call shape, and its result passes
    `validate_result`;
  * a method's accepted parameters are exactly its native signature (no
    ``**kwargs`` catch-alls left to swallow a setting);
  * a parameter that applies to another method but not this one is recorded
    in ``parameters["ignored"]`` with a reason and warned about once;
    ``strict=True`` raises; a parameter no method has is a typo and raises;
  * every parameter shared by two or more method families -- the vocabulary
    a benchmark loop actually passes -- is either shown to change the fit or
    exempted with a stated reason (`test_every_shared_parameter_is_classified`).
"""
import functools
import inspect
import warnings
from collections import defaultdict

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import api

K = 2
FAST = dict(iterations=4, batch_size=16)


def _views(n=48, dims=(10, 8, 6), seed=0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, K))
    return [torch.tensor(np.abs(u @ rng.standard_normal((K, p)))
                         + 0.2 * rng.standard_normal((n, p))).float() for p in dims]


def _quiet_fit(name, views, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pysimlr.fit(name, views, K, **kw)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_the_package_exports_the_interface():
    for name in ("fit", "predict", "method_names", "describe_method", "validate_result"):
        assert callable(getattr(pysimlr, name))
    assert pysimlr.method_names() == list(api.METHODS)


def test_families_partition_the_methods():
    names = set()
    for fam in api.FAMILIES:
        got = set(pysimlr.method_names(fam))
        assert got, f"family {fam} is empty"
        assert not (names & got)
        names |= got
    assert names == set(pysimlr.method_names())


@pytest.mark.parametrize("name", list(api.METHODS))
def test_accepts_is_the_native_signature(name):
    spec = api.METHODS[name]
    sig = inspect.signature(spec.fit)
    assert not any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()) or name.startswith("simlr"), (
        f"{name}'s native fit has a **kwargs catch-all; a setting could vanish into it")
    named = {n for n, q in sig.parameters.items()
             if n not in ("data_matrices", "k") and q.kind is not q.VAR_KEYWORD}
    assert spec.accepts == (named | set(spec.extra_params)) - set(spec.presets)


def test_only_registered_names_resolve_and_unknown_names_list_the_registry():
    assert api.resolve_method("simlr").name == "simlr"
    for removed in ("linear", "shared_private", "flow_simr_v", "nsaflow"):
        with pytest.raises(ValueError, match="registered"):
            api.resolve_method(removed)
    with pytest.raises(ValueError, match="in 0.6.0; it was not competitive.*Use 'simlr'"):
        api.resolve_method("simlr_lbfgsb")
    with pytest.raises(ValueError, match="registered"):
        api.resolve_method("nope")


def test_describe_method_reports_defaults_and_budget_unit():
    d = pysimlr.describe_method("lend")
    assert d["family"] == "deep" and d["budget_unit"] == "epochs"
    from pysimlr.defaults import DEFAULT_DEEP_ITERATIONS
    assert d["parameters"]["iterations"] == DEFAULT_DEEP_ITERATIONS == 300
    assert "random_state" in d["parameters"]


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(api.METHODS))
def test_every_method_fits_and_predicts_through_one_call_shape(name):
    views = _views()
    tr, te = [v[:36] for v in views], [v[36:] for v in views]
    res = _quiet_fit(name, tr, random_state=0, energy_type="acc", nsa_w=0.5, **FAST)
    api.validate_result(res, n=36, k=K, n_views=3)
    assert res["method"] == name
    pred = pysimlr.predict(res, te)
    assert tuple(pred["u"].shape) == (12, K)
    assert torch.isfinite(pred["u"]).all()
    assert len(pred["reconstructions"]) == 3 and len(pred["errors"]) == 3


def test_validate_result_rejects_a_missing_contract_key():
    res = _quiet_fit("simlr", _views(), iterations=2, learning_rate=0.05)
    broken = dict(res)
    broken.pop("convergence")
    with pytest.raises(ValueError, match="convergence"):
        api.validate_result(broken)


# --------------------------------------------------------------------------
# the parameter record
# --------------------------------------------------------------------------
def test_a_parameter_for_another_method_is_recorded_and_warned():
    views = _views()
    with pytest.warns(RuntimeWarning, match="hidden_dims"):
        res = pysimlr.fit("simlr", views, K, iterations=2, learning_rate=0.05, hidden_dims=(8,))
    assert "hidden_dims" in res["parameters"]["ignored"]
    assert "lend" in res["parameters"]["ignored"]["hidden_dims"]
    assert "hidden_dims" not in res["parameters"]["used"]


def test_removed_method_simlr_lbfgsb_raises_on_fit():
    with pytest.raises(ValueError, match="in 0.6.0; it was not competitive.*Use 'simlr'"):
        pysimlr.fit("simlr_lbfgsb", _views(), k=K)


def test_strict_raises_on_a_parameter_this_method_cannot_use():
    with pytest.raises(TypeError, match="hidden_dims"):
        pysimlr.fit("simlr", _views(), K, strict=True, iterations=2, hidden_dims=(8,))


def test_a_parameter_no_method_has_is_a_typo_and_raises():
    with pytest.raises(TypeError, match="iteratons"):
        pysimlr.fit("simlr", _views(), K, iteratons=2)


def test_the_record_separates_requested_used_and_defaults():
    res = _quiet_fit("ned", _views(), random_state=3, **FAST)
    rec = res["parameters"]
    assert rec["requested"] == {"random_state": 3, **FAST}
    assert rec["used"] == {"random_state": 3, **FAST}
    assert "hidden_dims" in rec["defaults"] and "iterations" not in rec["defaults"]
    assert rec["random_state"] == 3 and rec["budget_unit"] == "epochs"
    row = api.parameter_summary(res)
    assert row["method"] == "ned" and row["ignored"] == []


@pytest.mark.parametrize("name", list(api.METHODS))
def test_random_state_reproduces_every_method(name):
    views = _views()
    kw = dict(random_state=11, **FAST)
    if api.METHODS[name].family == "linear":
        kw["initialization_type"] = "random"
    a = _quiet_fit(name, views, **kw)
    b = _quiet_fit(name, views, **kw)
    assert torch.equal(a["u"], b["u"])


def test_random_state_does_not_change_the_linear_initializer():
    res = _quiet_fit("simlr", _views(), iterations=2, learning_rate=0.05, random_state=1)
    assert res["initialization_type"] == "pca"
    # Always recorded: the seed reaches every initializer (and the probes).
    assert res["parameters"]["random_state"] == 1


# --------------------------------------------------------------------------
# the shared vocabulary is live
# --------------------------------------------------------------------------
#: param -> {family: (a, b)}: two values that must produce different fits.
#: Each family is tested on one representative method.
EXERCISED = {
    "energy_type": {f: ("acc", "align") for f in ("linear", "deep", "flow")},
    "mixing_algorithm": {f: ("newton", "svd") for f in ("linear", "deep", "flow")},
    "positivity": {f: ("either", "positive") for f in ("linear", "deep", "flow")},
    "nsa_w": {"linear": (0.1, 0.9), "deep": (0.1, 0.9), "flow": (0.1, 0.9), "baseline": (0.1, 0.9)},
    "sparseness": {"linear": (None, 0.8), "deep": (None, 0.8), "flow": (None, 0.8), "baseline": (None, 0.8)},
    "iterations": {"linear": (1, 8), "deep": (1, 6), "flow": (1, 6)},
    "random_state": {"linear": (1, 2), "deep": (1, 2), "flow": (1, 2)},
    "initialization_type": {f: ("pca", "random") for f in ("linear", "deep", "flow")},
    "optimizer_type": {"linear": ("lars", "adam"), "deep": ("adam", "larslow"), "flow": ("adam", "larslow")},
    "learning_rate": {"linear": (0.01, 0.5), "deep": (1e-4, 1e-2), "flow": (1e-4, 1e-2)},
    "varimax": {f: (False, True) for f in ("linear", "deep", "flow")},
    "weight_decay": {"deep": (0.0, 0.5), "flow": (0.0, 0.5)},
    "sim_weight": {"deep": (0.0, 5.0), "flow": (0.0, 5.0)},
    "batch_size": {"deep": (8, 48), "flow": (8, 48)},
    "warmup_epochs": {"deep": (0, 5), "flow": (0, 5)},
    "topology": {f: ("star", "loo") for f in ("linear", "deep", "flow")},
    "dynamic_weights": {"deep": (False, True), "flow": (False, True)},
    "consolidate": {"baseline": (False, True)},
    # 0 charges any non-zero score correlation; 0.9 charges almost none.
    "collapse_threshold": {f: (0.0, 0.9) for f in ("linear", "deep", "flow")},
}

#: param -> reason it is not in EXERCISED for some or all families.
EXEMPT = {
    "device": "selects hardware; must not change numerics beyond the device noise floor",
    "verbose": "prints only",
    "tol": "acts on when training stops, not on the basis for a fixed budget; "
           "test_parameters_are_live::test_tol_controls_early_stopping",
    "patience": "acts on when training stops, with tol",
    "domain_matrices": "read only by initialization_type='domain'; test_alternative_initializations",
    "n_candidates": "read only by initialization_type='best_of_n'; test_alternative_initializations",
    "perturbation_scale": "read only by initialization_type='perturbed_pca'; test_alternative_initializations",
    "path_graph": "read only by topology='graph'; test_consensus_graph and test_flow_mai",
    "mai_metric": "read only with dynamic_weights=True on >=3 views; test_mai_metrics",
    "use_rank_mai": "read only with dynamic_weights=True; test_mai_metrics",
    "prune_threshold": "read only by the graph topology's edge pruning; test_latent_pruning",
    "sign_alignment": "changes the fit only for 3+ views whose latents disagree in sign, which is a property of "
                      "the data, not a switch to toggle on one fixture; tests/test_view_order_sign_alignment.py "
                      "(linear) and tests/test_view_order_deep.py (deep and flow)",
    "first_layer_mode": "first-layer projection schedule; test_deep_first_layer_contract",
    "nsa_iterations": "first-layer projection schedule; test_deep_first_layer_contract",
    "use_nsa": "first-layer projection switch; test_deep_first_layer_contract",
    "stabilization_start_epoch": "schedule offset, inert inside a 4-epoch budget; test_deep_first_layer_contract",
    "stabilization_ramp_epochs": "schedule offset, inert inside a 4-epoch budget; test_deep_first_layer_contract",
}

#: (param, family) pairs exercised elsewhere or inert by construction.
EXEMPT_FOR = {
    ("random_state", "baseline"): "PCA uses it only in the randomized SVD solver, which sklearn "
                                  "picks for large inputs; NSA-Flow does not take it",
    ("weight_decay", "linear"): "read only by the adaptive optimizers (adam, lars); "
                                "test_knobs_are_live::test_learning_rate_reaches_every_optimizer",
    ("dynamic_weights", "linear"): "test_parameters_are_live EXEMPT: symmetric on two views; test_flow_mai",
    ("consolidate", "linear"): "test_parameters_are_live::EXERCISED['consolidate']",
}

REPRESENTATIVE = {"linear": "simlr", "deep": "ned", "flow": "flow_v", "baseline": "nsa_pipeline"}


def _shared_vocabulary():
    fams = defaultdict(set)
    for s in api.METHODS.values():
        for p in s.accepts:
            fams[p].add(s.family)
    return {p: f for p, f in fams.items() if len(f) >= 2}


def test_every_shared_parameter_is_classified():
    """A shared parameter nobody has checked is wired up on each family is
    how a sweep reports a condition that never ran."""
    unclassified = []
    for p, fams in sorted(_shared_vocabulary().items()):
        if p in EXEMPT:
            continue
        for f in sorted(fams):
            if f in EXERCISED.get(p, {}) or (p, f) in EXEMPT_FOR:
                continue
            unclassified.append((p, f))
    assert not unclassified, (
        f"unclassified (parameter, family) pairs {unclassified}: add two values to EXERCISED "
        f"that must change the fit, or a reason to EXEMPT / EXEMPT_FOR.")


def _cases():
    out = []
    for p, by_family in EXERCISED.items():
        for f, (a, b) in by_family.items():
            out.append(
                pytest.param(
                    p, f, a, b, id=f"{p}-{f}", marks=pytest.mark.xdist_group(f"vocab_{f}")
                )
            )
    return out


@functools.lru_cache(maxsize=64)
def _cached_vocab_fit_vec(name: str, eff_items: tuple) -> torch.Tensor:
    res = _quiet_fit(name, _views(), **dict(eff_items))
    return torch.cat([v.detach().flatten() for v in res["v"]] + [res["u"].flatten()])


def _vocab_fit_vec(name: str, kw: dict) -> torch.Tensor:
    eff = {**pysimlr.describe_method(name)["parameters"], **kw}
    try:
        key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in eff.items()))
        hash(key)
    except TypeError:
        res = _quiet_fit(name, _views(), **kw)
        return torch.cat([v.detach().flatten() for v in res["v"]] + [res["u"].flatten()])
    return _cached_vocab_fit_vec(name, key)


@pytest.mark.parametrize("param,family,a,b", _cases())
def test_a_shared_parameter_changes_the_fit(param, family, a, b):
    name = REPRESENTATIVE[family]
    assert param in api.METHODS[name].accepts, f"{name} does not accept {param}"
    base = dict(FAST) if family in ("deep", "flow") else {}
    if family == "linear":
        base = dict(iterations=4, learning_rate=0.5, initialization_type="random", random_state=0)
    elif family in ("deep", "flow"):
        base["random_state"] = 0
    if family == "linear" and param in ("initialization_type", "varimax"):
        # Varimax rotates the PCA/GCCA start (`initialize_simlr`), so it is
        # exercised from that start.
        base.pop("initialization_type")
        base.pop("random_state")
    va = _vocab_fit_vec(name, {**base, param: a})
    vb = _vocab_fit_vec(name, {**base, param: b})
    assert va.shape != vb.shape or not torch.equal(va, vb), (
        f"{name}({param}={a!r}) and ({param}={b!r}) returned a bit-identical fit")


# --------------------------------------------------------------------------
# prediction: one space, batch invariance, baseline provenance
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(api.METHODS))
def test_prediction_does_not_depend_on_the_batch(name):
    """Out-of-sample latents must not depend on which other rows are in the
    batch. Before the frozen consensus map, `lend` moved by up to 2.2 and
    `simlr` under topology="loo" by 0.58 between a 5-row and a 30-row batch."""
    views = _views(n=60)
    tr, te = [v[:40] for v in views], [v[40:] for v in views]
    res = _quiet_fit(name, tr, random_state=0, **FAST)
    full = pysimlr.predict(res, te)["u"]
    part = pysimlr.predict(res, [v[:5] for v in te])["u"]
    one = pysimlr.predict(res, [v[:1] for v in te])["u"]
    assert torch.allclose(full[:5], part, atol=1e-4)
    assert torch.allclose(full[:1], one, atol=1e-4)


def test_linear_loo_prediction_does_not_depend_on_the_batch():
    views = _views(n=60)
    tr, te = [v[:40] for v in views], [v[40:] for v in views]
    res = _quiet_fit("simlr", tr, iterations=4, learning_rate=0.05, topology="loo")
    full = pysimlr.predict(res, te)["u"]
    part = pysimlr.predict(res, [v[:5] for v in te])["u"]
    assert torch.allclose(full[:5], part, atol=1e-4)


#: Methods whose fit is not stable under float32 rounding of the input. One
#: list, one reason, the same settings as every other method: no per-optimizer
#: defaults or tolerances.
_ROUNDING_UNSTABLE = {
    "ned": (
        "Not a shift bug: a +1e-6 perturbation moves NED's R^2 exactly as far "
        "as the +20 shift (0.0036 on component 1, both runs giving the same "
        "fit). That is float32 training amplifying input rounding. The latent "
        "collapse penalty reduces it: with collapse weight 0 the same 1e-6 "
        "perturbation moves R^2 by 0.084."),
}


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=_ROUNDING_UNSTABLE[n]))
    if n in _ROUNDING_UNSTABLE else n for n in api.METHODS])
def test_reconstruction_r2_is_in_one_space_for_every_method(name):
    """A constant shift of the input must not change any method's R^2. The
    baselines used to score in raw units, so on data with a large mean they
    looked near perfect (0.97 against 0.64 for simlr at a +20 shift)."""
    views = _views(n=60)
    shifted = [v + 20.0 for v in views]
    kw = dict(random_state=0, **FAST)
    a = pysimlr.predict(_quiet_fit(name, views, **kw), views)["r2"]
    b = pysimlr.predict(_quiet_fit(name, shifted, **kw), shifted)["r2"]
    assert np.allclose(a, b, atol=1e-3), f"{name}: r2 {a} vs {b} after a +20 shift"
    assert all(-1.0 < r <= 1.0 for r in a)


@pytest.mark.parametrize("name", ["pca", "nsa_pipeline"])
def test_baseline_provenance_replays_through_preprocess_data(name):
    from pysimlr.utils import preprocess_data
    views = _views()
    res = _quiet_fit(name, views)
    for v, prov in zip(views, res["provenance_list"]):
        preprocess_data(v, res["scale_list"], prov)


def test_pca_records_a_seed_only_when_it_used_one():
    res = _quiet_fit("pca", _views(), random_state=3)
    solver = res["estimator"]._fit_svd_solver
    assert res["random_state"] == (3 if solver == "randomized" else None)
