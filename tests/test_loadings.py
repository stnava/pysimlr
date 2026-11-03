"""`pysimlr.loadings`: the identified, interpretable summary every fit carries.

Measured on the planted linear case (`build_case("linear", n=300, seed=0,
private_scale=0.5)`, two refits on 98% of the rows): the encoder V's span moved
by mean sin^2 0.453 (signed) / 0.161 (positive; 0.030 under the earlier
objective with an orthogonality-defect penalty) with matched-column |r| 0.60 /
0.89, while `loadings` moved 0.0002 / 0.0001 with matched-column |r| 0.9999.
The thresholds below sit between the two.
"""
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

import pysimlr
from pysimlr import api
from pysimlr.benchmarks.synthetic_cases import build_case
from pysimlr.loadings import olee_loadings, structure_coefficients, varimax_rotation


def _small(seed=0, n=120):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, 2, generator=g)
    return [u @ torch.randn(2, p, generator=g) + 0.3 * torch.randn(n, p, generator=g) for p in (20, 15)]


def _planted(seed=0):
    c = build_case("linear", n_samples=300, seed=seed, private_scale=0.5)
    return [torch.as_tensor(np.asarray(x)).float() for x in c["data"]], np.asarray(c["true_u"]).shape[1]


def _span_instability(a, b):
    out = []
    for x, y in zip(a, b):
        qa, _ = np.linalg.qr(np.asarray(x, float)); qb, _ = np.linalg.qr(np.asarray(y, float))
        s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)
        out.append(float((1 - s ** 2).mean()))
    return float(np.mean(out))


def _matched_column_r(a, b):
    A = np.vstack([np.asarray(x, float) for x in a]); B = np.vstack([np.asarray(x, float) for x in b])
    k = A.shape[1]
    m = np.abs(np.corrcoef(A.T, B.T)[:k, k:])
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].mean())


@pytest.mark.parametrize("method", api.method_names())
def test_every_method_carries_structure_and_loadings(method):
    views = _small()
    res = pysimlr.fit(method, views, 2, iterations=5, random_state=0)
    for key in ("structure", "loadings"):
        assert len(res[key]) == 2
        for x, b in zip(views, res[key]):
            assert b.shape == (x.shape[1], 2) and torch.isfinite(b).all()
    r = res["loading_rotation"].double()
    assert torch.allclose(r.T @ r, torch.eye(2, dtype=r.dtype), atol=1e-5)


@pytest.mark.parametrize("method", ["simlr", "lend"])
def test_loadings_replay_reproduces_the_fit(method):
    """simlr scales by n*p ("np"), the deep models do not; replay uses each
    result's own scale_list and provenance."""
    views = _small()
    res = pysimlr.fit(method, views, 2, iterations=5, random_state=0)
    again = pysimlr.loadings(res, views)
    for key in ("structure", "loadings"):
        assert all(torch.equal(a, b) for a, b in zip(again[key], res[key]))


def test_structure_is_the_feature_latent_correlation_in_the_reported_basis():
    views = _small()
    res = pysimlr.fit("simlr", views, 2, iterations=5, random_state=0)
    u = (res["u"].double() @ res["loading_rotation"].double()).numpy()
    for x, c in zip(views, res["structure"]):
        x = x.double().numpy()
        want = np.array([[np.corrcoef(x[:, f], u[:, j])[0, 1] for j in range(2)] for f in range(x.shape[1])])
        assert np.allclose(c.numpy(), want, atol=1e-5)


def test_signed_loadings_have_exactly_orthogonal_columns():
    c = [torch.randn(30, 3), torch.randn(20, 3)]
    load, margin = olee_loadings(c, "either")
    assert margin is None
    for l, ci in zip(load, c):
        g = l.double().T @ l.double()
        assert torch.allclose(g - torch.diag(torch.diag(g)), torch.zeros(3, 3, dtype=g.dtype), atol=1e-5)
        assert torch.allclose(l.norm(dim=0), ci.norm(dim=0), atol=1e-5)


def test_positive_loadings_are_a_disjoint_nonnegative_assignment():
    c = [torch.randn(30, 3), torch.randn(20, 3)]
    load, margin = olee_loadings(c, "positive")
    for l, ci, mg in zip(load, c, margin):
        assert (l >= 0).all()
        assert ((l > 0).sum(1) <= 1).all()                 # each feature on at most one component
        cp = ci.clamp_min(0)
        kept = cp.max(1).values > 0
        assert torch.equal((l > 0).any(1), kept)
        assert torch.allclose(l.max(1).values[kept], cp.max(1).values[kept])
        assert (mg >= 0).all()


def test_varimax_basis_does_not_depend_on_the_input_rotation():
    torch.manual_seed(1)
    c = [torch.randn(40, 3).double(), torch.randn(25, 3).double()]
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    a = [ci @ varimax_rotation(c) for ci in c]
    cr = [ci @ q for ci in c]
    b = [ci @ varimax_rotation(cr) for ci in cr]
    assert _matched_column_r(a, b) > 0.9999


@pytest.mark.parametrize("positivity", ["either", "positive"])
def test_loadings_are_stable_where_the_encoder_is_not(positivity):
    views, k = _planted(0)
    rng = np.random.default_rng(1)
    idxs = [torch.as_tensor(np.sort(rng.choice(300, 294, replace=False))) for _ in range(2)]
    a, b = [pysimlr.fit("simlr", [x[i] for x in views], k, positivity=positivity, random_state=0) for i in idxs]
    v_a, v_b = [v.detach() for v in a["v"]], [v.detach() for v in b["v"]]
    if positivity == "either":
        # Signed V moves under both objectives. Under positivity V's
        # instability depends on the design: 0.030 with the old
        # orthogonality-defect penalty, 0.161 without it; the loadings are
        # stable either way, which is what this pins.
        assert _span_instability(v_a, v_b) > 0.1
    assert _span_instability(a["loadings"], b["loadings"]) < 0.01
    assert _matched_column_r(a["loadings"], b["loadings"]) > 0.99


def test_loadings_are_deterministic():
    views = _small()
    a = pysimlr.fit("simlr", views, 2, iterations=5, random_state=0)
    b = pysimlr.fit("simlr", views, 2, iterations=5, random_state=0)
    assert all(torch.equal(x, y) for x, y in zip(a["loadings"], b["loadings"]))


def test_held_out_loadings_use_the_predicted_latent():
    views = _small(n=150)
    tr, te = [x[:120] for x in views], [x[120:] for x in views]
    res = pysimlr.fit("simlr", tr, 2, iterations=5, random_state=0)
    out = pysimlr.loadings(res, te, u=pysimlr.predict(res, te)["u"])
    assert [tuple(b.shape) for b in out["structure"]] == [(20, 2), (15, 2)]


def test_structure_coefficients_reject_mismatched_rows():
    with pytest.raises(ValueError, match="rows"):
        structure_coefficients([torch.randn(10, 3)], torch.randn(9, 2))


def test_loading_inference_reports_intervals_stability_and_selection():
    views = _small(n=100)
    out = pysimlr.loading_inference("simlr", views, 2, n_subsamples=4, n_permutations=2,
                                    iterations=5, random_state=0)
    for vi, x in enumerate(views):
        p = x.shape[1]
        assert out["ci_low"][vi].shape == (p, 2) and (out["ci_low"][vi] <= out["ci_high"][vi]).all()
        assert len(out["component_stability"][vi]) == 2
        f = out["selection_frequency"][vi]
        assert ((f >= 0) & (f <= 1)).all() and out["selected"][vi].dtype == torch.bool
        q = out["q"][vi]
        assert q == int(np.floor(np.sqrt(1.0 * (2 * 0.8 - 1) * p)))
        assert out["expected_false_selections"][vi] == pytest.approx(q ** 2 / ((2 * 0.8 - 1) * p))
        assert out["expected_false_selections"][vi] <= 1.0
    assert out["null_max_abs"] > 0


def test_loading_inference_rejects_a_threshold_without_an_error_bound():
    with pytest.raises(ValueError, match="threshold"):
        pysimlr.loading_inference("simlr", _small(), 2, threshold=0.5)


def test_select_rank_reports_per_component_stability():
    out = pysimlr.select_rank("simlr", _small(n=80), [1, 2], n_subsamples=3, iterations=5, random_state=0)
    assert [r["k"] for r in out["table"]] == [1, 2]
    for r in out["table"]:
        assert len(r["component_stability"]) == r["k"]
        assert r["all_stable"] == (r["min_stability"] >= 0.9)
    assert out["k"] in (None, 1, 2)


@pytest.mark.slow
def test_select_rank_recovers_a_planted_rank():
    """Planted linear case with k = 3 (seed 0): measured component stability
    k=2 [0.996, 0.891], k=3 [0.998, 0.997, 0.997], k=4 [.., 0.66, ..],
    k=5 [.., 0.717, .., 0.347] with 10 half-sample refits."""
    views, k_true = _planted(0)
    out = pysimlr.select_rank("simlr", views, [2, 3, 4], n_subsamples=10, random_state=0)
    assert k_true == 3 and out["k"] == 3


def test_loading_inference_does_not_depend_on_n_jobs():
    views = _small(n=80)
    kw = dict(n_subsamples=3, n_permutations=1, iterations=5, random_state=0)
    a = pysimlr.loading_inference("simlr", views, 2, n_jobs=1, **kw)
    b = pysimlr.loading_inference("simlr", views, 2, n_jobs=2, **kw)
    for key in ("ci_low", "ci_high", "selection_frequency", "selected"):
        for x, y in zip(a[key], b[key]):
            assert torch.equal(x, y), key
    assert a["component_stability"] == b["component_stability"]
    assert a["null_max_abs"] == b["null_max_abs"]
    with pytest.raises(ValueError, match="n_jobs"):
        pysimlr.loading_inference("simlr", views, 2, n_jobs=0, **kw)
    sa = pysimlr.select_rank("simlr", views, [1, 2], n_subsamples=2, iterations=5,
                             random_state=0, n_jobs=1)
    sb = pysimlr.select_rank("simlr", views, [1, 2], n_subsamples=2, iterations=5,
                             random_state=0, n_jobs=2)
    assert sa == sb


@pytest.mark.parametrize("mixing", ["newton", "avg"])
def test_structure_is_the_correlation_with_the_rotated_scores(mixing):
    """`structure` is corr(X, u @ R) exactly, for every mixing. C0 @ R agrees
    only when u's columns are uncorrelated with equal variance."""
    from pysimlr.api import _preprocess_views
    views = _small(n=120)
    res = pysimlr.fit("simlr", views, 2, iterations=10, random_state=0, mixing_algorithm=mixing)
    xs, _ = _preprocess_views(views, res["scale_list"], res["provenance_list"])
    u_rot = res["u"].double() @ res["loading_rotation"].double()
    for got, want in zip(res["structure"], structure_coefficients(xs, u_rot)):
        assert torch.allclose(got, want, atol=1e-6)
