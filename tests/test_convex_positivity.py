"""positivity='linear' | 'conical' | 'convex' (D36).

Convex constrains every basis column to the probability simplex inside the
optimisation of every family that has a positivity parameter; linear and
conical are exact aliases of 'either' and 'positive'; unknown values raise.
"""
import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.positivity import is_convex, is_nonneg, resolve_positivity
from pysimlr.sparsification import project_columns_to_simplex, simlr_sparseness

TOL = 1e-6


def _views(seed=0, n=90, dims=(24, 18, 10)):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 2, generator=g)
    return [z @ torch.randn(2, p, generator=g) + 0.5 * torch.randn(n, p, generator=g) for p in dims]


def _assert_simplex(vs):
    for v in vs:
        v = torch.as_tensor(v).detach().double()
        assert torch.isfinite(v).all()
        assert float(v.min()) >= 0.0
        assert float((v.sum(0) - 1.0).abs().max()) <= TOL


# --- the projection -----------------------------------------------------------------------------
@pytest.mark.parametrize("support_only", [False, True])
def test_simplex_projection_properties(support_only):
    g = torch.Generator().manual_seed(1)
    for _ in range(60):
        p, k = int(torch.randint(2, 120, (1,), generator=g)), int(torch.randint(1, 6, (1,), generator=g))
        v = torch.randn(p, k, generator=g, dtype=torch.float64) * float(10 ** (torch.rand(1, generator=g) * 4 - 2))
        v = v.abs() if support_only else v
        w = project_columns_to_simplex(v, support_only=support_only)
        assert float(w.min()) >= 0.0
        assert float((w.sum(0) - 1).abs().max()) <= 1e-10
        assert torch.allclose(project_columns_to_simplex(w, support_only=support_only), w, atol=1e-12)
        if support_only:
            assert bool(((w > 0) <= (v > 0)).all())


def test_simplex_projection_matches_bisection_reference():
    c = torch.tensor([0.9, -0.2, 0.4, 0.05, 0.3], dtype=torch.float64)
    lo, hi = float(c.min()) - 1.0, float(c.max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(torch.clamp(c - mid, min=0).sum()) > 1 else (lo, mid)
    ref = torch.clamp(c - 0.5 * (lo + hi), min=0)
    got = project_columns_to_simplex(c[:, None], support_only=False)[:, 0]
    assert torch.allclose(got, ref, atol=1e-10)


def test_support_restricted_projection_never_densifies():
    v = torch.tensor([[0.2], [0.0], [0.05], [0.0]], dtype=torch.float64)
    w = project_columns_to_simplex(v)
    assert (w[[1, 3]] == 0).all() and abs(float(w.sum()) - 1.0) < 1e-12


def test_simplex_projection_refuses_a_dead_column():
    with pytest.raises(ValueError, match="no positive entry"):
        project_columns_to_simplex(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))


def test_simplex_projection_gradcheck():
    x = (torch.rand(10, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) + 0.1)
    assert torch.autograd.gradcheck(project_columns_to_simplex, (x.requires_grad_(True),), eps=1e-7, atol=1e-6)


@pytest.mark.parametrize("w", [0.0, 0.5])
def test_simlr_sparseness_convex_is_on_the_simplex(w):
    v = torch.rand(40, 4, generator=torch.Generator().manual_seed(3)) - 0.3
    out = simlr_sparseness(v, constraint_type="ortho" if w > 0 else "none", positivity="convex",
                           sparseness=0.5, constraint_weight=w, energy_type="nc")
    _assert_simplex([out])


# --- vocabulary ---------------------------------------------------------------------------------
def test_resolver_aliases_and_errors():
    assert resolve_positivity("linear") == "either"
    assert resolve_positivity("conical") == "positive"
    assert resolve_positivity("Convex") == "convex"
    assert resolve_positivity(True) == "positive" and resolve_positivity(False) == "either"
    assert is_nonneg("convex") and is_nonneg("conical") and not is_nonneg("linear")
    assert is_convex("convex") and not is_convex("positive")
    for bad in ("bogus", "conic", ""):
        with pytest.raises(ValueError):
            resolve_positivity(bad)
    with pytest.raises(TypeError):
        resolve_positivity(3)


def test_unknown_positivity_raises_at_fit():
    with pytest.raises(ValueError, match="unknown positivity"):
        pysimlr.fit("simlr", _views(), 2, positivity="conicall", iterations=3, random_state=0)


@pytest.mark.parametrize("method", ["pca", "nsa_pipeline"])
def test_baseline_refuses_convex(method):
    with pytest.raises(ValueError, match="convex"):
        pysimlr.fit(method, _views(), 2, positivity="convex")


@pytest.mark.parametrize("alias,canonical", [("linear", "either"), ("conical", "positive")])
def test_aliases_are_bit_identical_and_recorded(alias, canonical):
    xs = _views()
    a = pysimlr.fit("simlr", xs, 2, positivity=alias, iterations=15, random_state=0)
    b = pysimlr.fit("simlr", xs, 2, positivity=canonical, iterations=15, random_state=0)
    for va, vb in zip(a["v"], b["v"]):
        assert torch.equal(torch.as_tensor(va), torch.as_tensor(vb))
    assert a["parameters"]["used"]["positivity"] == alias
    assert a["parameters"]["positivity_resolved"] == canonical
    assert b["parameters"]["positivity_resolved"] == canonical


@pytest.mark.parametrize("method", ["lend", "flow_v"])
def test_aliases_bit_identical_deep(method):
    xs = _views()
    a = pysimlr.fit(method, xs, 2, positivity="conical", iterations=4, random_state=0)
    b = pysimlr.fit(method, xs, 2, positivity="positive", iterations=4, random_state=0)
    for va, vb in zip(a["v"], b["v"]):
        assert torch.equal(torch.as_tensor(va), torch.as_tensor(vb))


# --- every family ---------------------------------------------------------------------------------
@pytest.mark.parametrize("energy", ["nc", "nc_abs", "acc", "align", "recon_r2", "procrustes", "logcosh", "kurtosis"])
def test_linear_convex_all_energies(energy):
    xs = _views()
    r = pysimlr.fit("simlr", xs, 2, positivity="convex", energy_type=energy, iterations=15, random_state=0)
    _assert_simplex(r["v"])
    assert r["parameters"]["positivity_resolved"] == "convex"


@pytest.mark.parametrize("method", ["lend", "ned", "ned_shared_private", "flow_v"])
def test_deep_and_flow_convex(method):
    xs = _views()
    r = pysimlr.fit(method, xs, 2, positivity="convex", iterations=6, random_state=0)
    _assert_simplex(r["v"])
    pr = pysimlr.predict(r, xs)
    assert np.isfinite(np.asarray(torch.as_tensor(pr["u"]).detach(), float)).all()


def test_deep_eval_basis_is_the_finalized_simplex_basis():
    xs = _views()
    r = pysimlr.fit("lend", xs, 2, positivity="convex", iterations=6, random_state=0)
    model = r["model"]
    encs = getattr(model, "linear_encoders", None) or model.encoders
    for enc, v in zip(encs, r["v"]):
        enc.eval()
        assert torch.allclose(enc.v.detach().double().cpu(), torch.as_tensor(v).double(), atol=1e-6)


def test_deep_encoder_train_basis_is_on_the_simplex_and_differentiable():
    from pysimlr.deep import LENDNSAEncoder
    enc = LENDNSAEncoder(30, 3, positivity="convex", sparseness=0.5)
    enc.train()
    v = enc.v
    _assert_simplex([v])
    (v * torch.linspace(0.0, 1.0, v.numel()).reshape(v.shape)).sum().backward()   # sum(v) is constant on the simplex
    grads = [q.grad for q in enc.parameters() if q.grad is not None]
    assert grads and all(torch.isfinite(gq).all() for gq in grads) and any(gq.abs().sum() > 0 for gq in grads)


def test_convex_is_not_a_rescaled_conical_fit():
    xs = _views()
    c = pysimlr.fit("simlr", xs, 2, positivity="convex", iterations=40, random_state=0)
    p = pysimlr.fit("simlr", xs, 2, positivity="positive", iterations=40, random_state=0)
    def unit(v):
        t = torch.as_tensor(v).double()
        return t / t.norm(dim=0, keepdim=True)

    diff = max(float((unit(a) - unit(b)).abs().max()) for a, b in zip(c["v"], p["v"]))
    assert diff > 1e-3


def test_predict_and_missing_view_predict_under_convex():
    xs = _views()
    r = pysimlr.fit("simlr", xs, 2, positivity="convex", iterations=15, random_state=0)
    full = pysimlr.predict(r, xs)
    part = pysimlr.predict(r, [xs[0], None, xs[2]])
    for out in (full, part):
        assert np.isfinite(np.asarray(torch.as_tensor(out["u"]).detach(), float)).all()
    _assert_simplex(r["v"])
    hc = pysimlr.heldout_coupling(r, xs)
    assert np.isfinite(np.nanmean(hc["cross_view_heldout"]))


def test_anchored_encoder_and_extend_under_convex():
    from pysimlr.extension import fit_anchored_encoder
    xs = _views()
    r = pysimlr.fit("simlr", xs[:2], 2, positivity="convex", iterations=15, random_state=0)
    fa = fit_anchored_encoder(xs[2], torch.as_tensor(r["u"]).detach(), positivity="convex", sparseness=0.5)
    _assert_simplex([fa["v"]])
    ext = pysimlr.extend(r, xs[:2], {"new": xs[2]}, how="anchored", iterations=15, random_state=0, warn=False)
    _assert_simplex([ext["extensions"]["new"]["v"]])
    pr = pysimlr.predict(ext, {"view0": xs[0], "view1": xs[1], "new": xs[2]})
    assert np.isfinite(np.asarray(torch.as_tensor(pr["u"]).detach(), float)).all()


def test_feature_relevance_refits_under_convex():
    xs = _views(n=60)
    r = pysimlr.fit("simlr", xs, 2, positivity="convex", iterations=10, random_state=0)
    out = pysimlr.feature_relevance(r, xs, component=0, n_permutations=99, n_refits=2, refit_permutations=99,
                                    random_state=0, n_jobs=1)
    assert out is not None


def test_loading_inference_refits_under_convex():
    li = pysimlr.loading_inference("simlr", _views(n=60), 2, n_subsamples=3, positivity="convex",
                                   iterations=8, random_state=0)
    assert li is not None


def test_consolidate_refuses_convex():
    with pytest.raises(ValueError, match="consolidate"):
        pysimlr.fit("simlr", _views(), 2, positivity="convex", consolidate=True, iterations=3, random_state=0)


def test_nsa_w_cap_still_enforced_under_convex():
    with pytest.raises(ValueError, match="nsa_w"):
        pysimlr.fit("simlr", _views(), 2, positivity="convex", nsa_w=1.0, iterations=3, random_state=0)
