"""`varimax` means the same thing on every multi-view entry point.

`simlr` exposed ``varimax`` while the deep and flow entry points routed their
initialization through `initialize_deep_encoders`, which had no such argument
and let ``**kwargs`` swallow it -- so ``lend_simr(..., varimax=False)`` ran the
default and said nothing. Under ``positivity="either"`` the flag was also inert
for ``"pca"`` and ``"gcca"`` starts even on `simlr` itself.

The contract pinned here:

* every entry point accepts ``varimax`` and hands it to `initialize_simlr`;
* the default (``None``) rotates a non-negative basis and leaves a signed one
  unrotated; an explicit ``True``/``False`` overrides it;
* ``varimax=True`` changes the start (and therefore the fit) relative to
  ``varimax=False`` under ``gcca``/``either`` and ``pca``/``either|positive``;
* with ``varimax=True`` the views' columns correspond -- the linear assignment
  of view ``m``'s sample projections onto view 0's is the identity;
* ``gcca`` gets one shared rotation of the MAXVAR subspace, not one per view;
* ``varimax=False`` never rotates, and contradicts ``initialization_type="varimax"``;
* the trained flow's bijection and the returned basis are untouched by the
  initialization choice (no post-hoc rotation of ``V``).
"""
import sys
import warnings

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from pysimlr.deep import (LENDSiMRModel, NEDSharedPrivateSiMRModel, NEDSiMRModel,
                          lend_simr, ned_simr, ned_simr_shared_private)
from pysimlr.flows import FlowSiMRVModel, flow_simlr_v, flow_simr_v
from pysimlr.simlr import initialize_simlr


def _module(name):
    """The live module object. `pysimlr` re-exports the `simlr` function under
    the submodule's name, so `import pysimlr.simlr as m` binds the function;
    and other tests reload modules, so this is looked up at call time."""
    return sys.modules[name]

K = 3
DIMS = (12, 9, 9)

#: (initialization_type, positivity) pairs the varimax flag must act on.
CONFIGS = [("gcca", "either"), ("pca", "either"), ("pca", "positive")]

ENTRY_POINTS = {
    "simlr": lambda X, **kw: _module("pysimlr.simlr").simlr(X, k=K, iterations=2, verbose=False, **kw),
    "flow_simlr_v": lambda X, **kw: flow_simlr_v(X, k=K, epochs=2, batch_size=16, verbose=False, **kw),
    "lend_simr": lambda X, **kw: lend_simr(X, k=K, epochs=2, batch_size=16, verbose=False, **kw),
    "ned_simr": lambda X, **kw: ned_simr(X, k=K, epochs=2, batch_size=16, verbose=False, **kw),
    "ned_simr_shared_private": lambda X, **kw: ned_simr_shared_private(
        X, k=K, epochs=2, batch_size=16, verbose=False, **kw),
}


def _views(seed=0, n=48, dims=DIMS, noise=0.3):
    """Three views sharing ``K`` block-localized factors."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, K))
    out = []
    for p in dims:
        v = np.zeros((p, K))
        blk = p // K
        for j in range(K):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        x = u @ v.T + noise * rng.standard_normal((n, p))
        x = (x - x.mean(0)) / x.std(0)
        out.append(torch.tensor(x).float())
    return out


def _cross_view_assignment(data, v_mats):
    """Optimal column assignment (and matched |r|) of each view onto view 0."""
    proj = []
    for x, v in zip(data, v_mats):
        z = x.double() @ v.detach().double()
        z = z - z.mean(0, keepdim=True)
        proj.append(z / z.norm(dim=0, keepdim=True).clamp_min(1e-12))
    out = []
    for z in proj[1:]:
        c = np.abs((proj[0].t() @ z).numpy())
        _, col = linear_sum_assignment(-c)
        out.append((col.tolist(), c[np.arange(len(col)), col]))
    return out


def _assert_aligned(data, v_mats, label, min_r=0.8):
    for m, (perm, matched) in enumerate(_cross_view_assignment(data, v_mats), start=1):
        assert perm == list(range(K)), (
            f"{label}: view {m}'s columns map to view 0's as {perm}, not the "
            f"identity -- column j does not mean the same factor in every view")
        assert matched.min() > min_r, f"{label}: view {m} matched |r| {matched}"


def _flat(v_mats):
    return torch.cat([v.detach().double().flatten() for v in v_mats])


# --------------------------------------------------------------------------
# initialize_simlr semantics
# --------------------------------------------------------------------------
@pytest.mark.parametrize("init,pos", CONFIGS + [("perturbed_pca", "either")])
def test_varimax_changes_the_start_and_aligns_columns(init, pos):
    X = _views()
    out = {}
    for vm in (True, False):
        gen = torch.Generator().manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[vm] = initialize_simlr(X, K, initialization_type=init,
                                       positivity=pos, varimax=vm, generator=gen)
    delta = float((_flat(out[True]) - _flat(out[False])).norm())
    assert delta > 1e-3, f"{init}/{pos}: varimax=True and False gave the same start"
    _assert_aligned(X, out[True], f"initialize_simlr({init}, {pos})")


def test_gcca_rotates_the_shared_subspace_once(monkeypatch):
    """One rotation of the (n x k) MAXVAR subspace, not one per view."""
    X = _views()
    calls = []
    svd_mod = _module("pysimlr.svd")
    real = svd_mod.varimax_rotation

    def spy(E, *a, **kw):
        calls.append(tuple(E.shape))
        return real(E, *a, **kw)

    monkeypatch.setattr(svd_mod, "varimax_rotation", spy)
    initialize_simlr(X, K, initialization_type="gcca", positivity="either", varimax=True)
    assert calls == [(X[0].shape[0], K)], calls


@pytest.mark.parametrize("init", ["gcca", "pca", "perturbed_pca"])
def test_varimax_false_never_rotates_a_signed_start(monkeypatch, init):
    calls = []
    svd_mod = _module("pysimlr.svd")
    monkeypatch.setattr(svd_mod, "varimax_rotation",
                        lambda *a, **kw: calls.append("rotation"))
    monkeypatch.setattr(svd_mod, "varimax", lambda *a, **kw: calls.append("varimax"))
    initialize_simlr(_views(), K, initialization_type=init, positivity="either",
                     varimax=False, generator=torch.Generator().manual_seed(0))
    assert calls == [], f"{init} rotated with varimax=False: {calls}"


@pytest.mark.parametrize("init", ["pca", "gcca"])
@pytest.mark.parametrize("pos,expected", [("either", False), ("positive", True)])
def test_the_default_rotates_only_a_nonnegative_basis(init, pos, expected):
    """``varimax=None`` resolves through `resolve_varimax`: rotate a
    non-negative basis, leave a signed one as PCA/GCCA gave it."""
    X = _views()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = initialize_simlr(X, K, initialization_type=init, positivity=pos)
        explicit = initialize_simlr(X, K, initialization_type=init, positivity=pos,
                                    varimax=expected)
    assert torch.equal(_flat(default), _flat(explicit)), (
        f"{init}/{pos}: default did not match varimax={expected}")


def test_the_varimax_initialisation_rotates_under_the_default():
    X = _views()
    rotated = initialize_simlr(X, K, initialization_type="varimax", positivity="either")
    assert torch.equal(_flat(rotated), _flat(initialize_simlr(
        X, K, initialization_type="pca", positivity="either", varimax=True)))


def test_varimax_init_type_contradicting_the_flag_is_refused():
    with pytest.raises(ValueError, match="contradicts varimax=False"):
        initialize_simlr(_views(), K, initialization_type="varimax", varimax=False)


# --------------------------------------------------------------------------
# the deep models' own initialize_v path
# --------------------------------------------------------------------------
def _deep_model(cls, dims, positivity):
    if cls is FlowSiMRVModel:
        return cls(list(dims), K, num_layers=2, hidden_dim=16, positivity=positivity)
    if cls is NEDSharedPrivateSiMRModel:
        return cls(list(dims), K, 1, [16], 0.0, 0.1, positivity, 0.5, "newton")
    return cls(list(dims), K, [16], 0.0, 0.1, positivity, 0.5, "newton")


def _init_model(model, X, **kw):
    if isinstance(model, FlowSiMRVModel):
        model.initialize_weights(X, **kw)
        return [enc.v_raw.detach().clone() for enc in model.linear_encoders]
    model.initialize_v(X, K, **kw)
    encs = model.encoders if isinstance(model, LENDSiMRModel) else model.linear_encoders
    return [enc.v_raw.detach().clone() for enc in encs]


@pytest.mark.parametrize("cls", [LENDSiMRModel, NEDSiMRModel,
                                 NEDSharedPrivateSiMRModel, FlowSiMRVModel])
@pytest.mark.parametrize("init,pos", CONFIGS)
def test_deep_encoder_seeding_honours_varimax(cls, init, pos):
    X = _views()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seeded = {vm: _init_model(_deep_model(cls, DIMS, pos), X,
                                  initialization_type=init, varimax=vm)
                  for vm in (True, False)}
        reference = initialize_simlr(X, K, initialization_type=init, positivity=pos,
                                     varimax=True, sparseness=[0.5] * len(X))
    assert float((_flat(seeded[True]) - _flat(seeded[False])).norm()) > 1e-3, (
        f"{cls.__name__}: varimax did not reach the encoder seed")
    assert torch.allclose(_flat(seeded[True]), _flat(reference), atol=1e-5), (
        f"{cls.__name__}: encoder seed differs from initialize_simlr's start")
    _assert_aligned(X, seeded[True], f"{cls.__name__}({init}, {pos})")


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("init,pos", CONFIGS)
def test_entry_points_forward_varimax_and_change_the_fit(monkeypatch, name, init, pos):
    # `simlr`'s default learning-rate tuner runs short probe fits before the
    # real one, each calling `initialize_simlr`; every call must carry the
    # caller's value.
    seen = {True: [], False: []}
    current = []
    simlr_mod = _module("pysimlr.simlr")
    real = simlr_mod.initialize_simlr

    def spy(data, k, *a, **kw):
        v = real(data, k, *a, **kw)
        current.append((kw.get("varimax"), [d.detach().clone() for d in data], v))
        return v

    monkeypatch.setattr(simlr_mod, "initialize_simlr", spy)
    X = _views()
    fits = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for vm in (True, False):
            torch.manual_seed(0)
            np.random.seed(0)
            current.clear()
            res = ENTRY_POINTS[name](X, initialization_type=init, positivity=pos, varimax=vm)
            seen[vm] = list(current)
            fits[vm] = res["v"]

    for vm in (True, False):
        passed = [s[0] for s in seen[vm]]
        assert passed and all(p is vm for p in passed), (
            f"{name}(varimax={vm}) passed varimax={passed} to initialize_simlr")
    _, data_true, v_true = seen[True][-1]
    _assert_aligned(data_true, v_true, f"{name}({init}, {pos}) start")
    assert float((_flat(fits[True]) - _flat(fits[False])).norm()) > 1e-6, (
        f"{name}: varimax=True and False returned the same fitted basis")


@pytest.mark.parametrize("fn", [lend_simr, ned_simr, ned_simr_shared_private, flow_simr_v])
def test_an_unknown_keyword_warns_instead_of_vanishing(fn):
    with pytest.warns(UserWarning, match="varimx"):
        fn(_views(), k=K, epochs=1, batch_size=16, verbose=False, varimx=False)


# --------------------------------------------------------------------------
# the flow bijection and the returned basis are not rotated after training
# --------------------------------------------------------------------------
@pytest.mark.parametrize("vm", [True, False])
def test_flow_invariants_hold_under_either_initialisation(vm):
    X = _views()
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = flow_simr_v(X, k=K, epochs=3, batch_size=16, verbose=False,
                          initialization_type="gcca", varimax=vm)
    model = res["model"].eval()
    with torch.no_grad():
        for m, (flow, enc) in enumerate(zip(model.flows, model.linear_encoders)):
            v = res["v"][m]
            assert torch.equal(v, enc._finalized_v.cpu()), (
                f"view {m}: returned V is not the encoder's finalized basis")
            z = (X[m].to(v.dtype) @ v).float()
            fz, log_det = flow.forward_and_log_det(z)
            back, inv_log_det = flow.inverse_and_log_det(fz)
            assert torch.allclose(back, z, atol=1e-4), f"view {m}: inverse(forward(z)) != z"
            assert torch.allclose(log_det, -inv_log_det, atol=1e-4), (
                f"view {m}: forward and inverse log|det J| do not cancel")
            assert torch.allclose(enc(X[m].to(v.dtype)), X[m].to(v.dtype) @ v, atol=1e-5), (
                f"view {m}: the eval encoder does not project with the returned V")
