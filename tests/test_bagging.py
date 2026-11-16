"""Tests for `pysimlr.bagged_fit` / `bagged_predict` (synthetic data only)."""
import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.sem import view_scores

N_PLANTED, P = 5, 30


def planted(n=120, seed=0, n_views=3):
    """One shared latent loads on the first 5 of 30 features of every view; the other 25 are noise."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    views = [np.hstack([z @ rng.uniform(0.8, 1.2, (1, N_PLANTED)) + 0.8 * rng.standard_normal((n, N_PLANTED)),
                        rng.standard_normal((n, P - N_PLANTED))]) for _ in range(n_views)]
    return z[:, 0], views


def corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


@pytest.fixture(scope="module")
def data():
    return planted(seed=0), planted(n=60, seed=1)


@pytest.fixture(scope="module")
def sel(data):
    (z, v), _ = data
    return pysimlr.bagged_fit("simlr", v, 1, aggregate="select", n_subsamples=16)


@pytest.fixture(scope="module")
def avg(data):
    (z, v), _ = data
    return pysimlr.bagged_fit("simlr", v, 1, aggregate="average", n_subsamples=40)


def test_select_recovers_planted_features_and_score(data, sel):
    (z, v), (z2, v2) = data
    # Criterion: at least 4 of the 5 planted features kept and at most 3 noise features kept, in every view.
    for mask in sel["feature_mask"]:
        assert mask.shape == (P,) and mask.dtype == bool
        assert mask[:N_PLANTED].sum() >= 4
        assert mask[N_PLANTED:].sum() <= 3
    assert abs(corr(sel["scores"], z)) >= 0.9
    pred = pysimlr.predict(sel, v2)["u"].numpy()[:, 0]            # full-width new rows
    assert abs(corr(pred, z2)) >= 0.9
    assert abs(corr(pysimlr.bagged_predict(sel, v2)["u"].numpy()[:, 0], z2)) >= 0.9
    vs = view_scores(sel, v2)                                       # full-width views
    assert len(vs["scores"]) == 3 and all(abs(corr(s, z2)) >= 0.8 for s in vs["scores"])
    assert [int(m.sum()) for m in sel["feature_mask"]] == [fm.sum() for fm in sel["feature_mask"]]
    assert all(1.0 <= e <= m.sum() + 1e-9 for e, m in zip(sel["effective_features"], sel["feature_mask"]))
    assert sel["aggregate"] == "select" and len(sel["selection_frequency"]) == 3


def test_select_accepts_subset_views_and_rejects_other_widths(data, sel):
    _, (z2, v2) = data
    full = pysimlr.predict(sel, v2)["u"].numpy()
    sub = [x[:, m] for x, m in zip(v2, sel["feature_mask"])]
    np.testing.assert_allclose(pysimlr.predict(sel, sub)["u"].numpy(), full, atol=1e-6)
    # A dict of views takes `predict`'s dict path (`predict_views`, which differs from the list path for any fit):
    # full-width and kept-feature dicts must agree with each other.
    d_full = pysimlr.predict(sel, {f"view{i}": x for i, x in enumerate(v2)})["u"].numpy()
    d_sub = pysimlr.predict(sel, {f"view{i}": x for i, x in enumerate(sub)})["u"].numpy()
    np.testing.assert_allclose(d_full, d_sub, atol=1e-6)
    with pytest.raises(ValueError, match="view 1 has 17 columns"):
        pysimlr.predict(sel, [v2[0], v2[1][:, :17], v2[2]])


def test_average_scores_oob_and_between_run_agreement(data, avg):
    (z, v), (z2, v2) = data
    assert abs(corr(avg["scores"], z)) >= 0.9
    # fraction 0.8, 40 subsamples: P(a row is in every fit) = 0.8**40 = 1.3e-4, so no row is expected to be NaN.
    oob = avg["oob_scores"]
    assert oob.shape == (120, 1) and not np.isnan(oob).any()
    assert abs(corr(oob, z)) >= 0.85
    assert avg["feature_mask"][0].all() and avg["effective_features"][0] > 1.0
    pred = pysimlr.predict(avg, v2)
    assert abs(corr(pred["u"].numpy(), z2)) >= 0.9 and pred["n_fits"] == 40
    # training rows through predict == the stored ensemble scores
    np.testing.assert_allclose(pysimlr.predict(avg, v)["u"].numpy(), avg["scores"], atol=1e-4)
    # second independent run: same fits' seed, different row draws
    other = pysimlr.bagged_fit("simlr", v, 1, aggregate="average", n_subsamples=16, draw_random_state=7)
    assert abs(corr(avg["scores"], other["scores"])) >= 0.95
    few = pysimlr.bagged_fit("pca", v, 1, aggregate="average", n_subsamples=4)   # 0.8**4 = 41% of rows in every fit
    assert np.isnan(few["oob_scores"]).any() and np.isfinite(few["oob_scores"]).any()
    assert abs(corr(pysimlr.predict(avg, v2)["u"].numpy(), pysimlr.predict(other, v2)["u"].numpy())) >= 0.95


@pytest.mark.parametrize("method", ["simlr", "pca"])
def test_averaged_weights_equal_averaged_scores(method, data):
    (z, v), (z2, v2) = data
    r = pysimlr.bagged_fit(method, v, 2, aggregate="average", n_subsamples=6 if method == "simlr" else 12)
    assert r["weights_averaged"] is True
    lat = pysimlr.bagged_predict(r, v2)["latents"]
    for m in range(3):
        # Definition: mean over fits of the aligned per-view score (no per-view sign, no standardisation) equals
        # X_m @ weights_raw[m] + intercepts[m] (each fit's affine standardisation folded into the weights).
        got = v2[m] @ r["weights_raw"][m] + r["intercepts"][m]
        np.testing.assert_allclose(got, lat[m].numpy(), atol=1e-5 * (1 + np.abs(got).max()))
        assert r["averaged_weights"][m].shape == (P, 2)
    assert len(r["effective_features"]) == 3


def test_deep_methods_average_scores_only():
    z, v = planted(n=60, seed=3)
    z2, v2 = planted(n=30, seed=4)
    # `iterations` is reduced from the method default (300) for speed; this tests plumbing, not the fit quality.
    r = pysimlr.bagged_fit("lend", v, 1, aggregate="average", n_subsamples=3, iterations=15)
    assert r["weights_averaged"] is False and "averaged_weights" not in r
    assert "lend" in r["weights_note"] and r["effective_features"] == [None] * 3
    assert pysimlr.predict(r, v2)["u"].shape == (30, 1)
    with pytest.raises(NotImplementedError, match="average-mode"):
        view_scores(r, v2)


@pytest.mark.parametrize("method,kw", [("pca", {}), ("lend", {"iterations": 15}), ("flow_v", {"iterations": 15})])
@pytest.mark.parametrize("aggregate", ["select", "average"])
def test_both_modes_end_to_end(method, kw, aggregate):
    z, v = planted(n=60, seed=3)
    z2, v2 = planted(n=30, seed=4)
    # deep/flow `iterations` reduced from the default (300) for speed: plumbing test only.
    r = pysimlr.bagged_fit(method, v, 1, aggregate=aggregate, n_subsamples=4, **kw)
    assert r["aggregate"] == aggregate and r["method"] == method and r["k"] == 1
    p = pysimlr.predict(r, v2)["u"]
    assert tuple(p.shape) == (30, 1) and torch.isfinite(p).all()
    assert np.asarray(r["scores"]).shape == (60, 1) and np.asarray(r["oob_scores"]).shape == (60, 1)
    assert len(r["selection_frequency"]) == 3 and len(r["feature_mask"]) == 3
    assert 0.0 <= r["converged_fraction"] <= 1.0 and "provenance" in r and "between_run_note" in r
    if aggregate == "select":
        vs = view_scores(r, v2)
        assert len(vs["scores"]) == 3 and vs["scores"][0].shape == (30, 1)
        assert all(e > 0 for e in r["effective_features"])
    if method == "pca":
        assert abs(corr(r["scores"], z)) >= 0.8 if aggregate == "average" else True


def test_select_raises_when_a_view_keeps_nothing():
    rng = np.random.default_rng(0)
    v = [rng.standard_normal((60, 12)) for _ in range(3)]       # no shared signal: nothing is stable
    with pytest.raises(ValueError, match=r"kept no feature.*maximum selection frequency.*threshold is 1\.0"):
        pysimlr.bagged_fit("pca", v, 1, aggregate="select", n_subsamples=6, threshold=1.0)


def test_default_aggregate_is_select(data):
    import inspect
    (z, v), _ = data
    assert inspect.signature(pysimlr.bagged_fit).parameters["aggregate"].default == "select"
    assert pysimlr.bagged_fit("pca", v, 1, n_subsamples=4)["aggregate"] == "select"


def test_argument_validation(data):
    (z, v), _ = data
    with pytest.raises(ValueError, match=r"aggregate must be one of \('select', 'average'\)"):
        pysimlr.bagged_fit("pca", v, 1, aggregate="median")
    with pytest.raises(ValueError, match="threshold"):
        pysimlr.bagged_fit("pca", v, 1, threshold=0.4)
    with pytest.raises(ValueError, match="fraction"):
        pysimlr.bagged_fit("pca", v, 1, fraction=1.0)
    with pytest.raises(NotImplementedError):
        pysimlr.predict(pysimlr.bagged_fit("pca", v, 1, aggregate="average", n_subsamples=4), v, exclude_views=[0])


@pytest.mark.parametrize("aggregate", ["select", "average"])
def test_determinism_and_n_jobs(aggregate, data):
    (z, v), (z2, v2) = data
    kw = dict(aggregate=aggregate, n_subsamples=6, random_state=3, draw_random_state=5)
    a = pysimlr.bagged_fit("pca", v, 1, **kw)
    b = pysimlr.bagged_fit("pca", v, 1, **kw)
    c = pysimlr.bagged_fit("pca", v, 1, n_jobs=2, **kw)
    for other in (b, c):
        np.testing.assert_array_equal(a["scores"], other["scores"])
        np.testing.assert_array_equal(a["oob_scores"], other["oob_scores"])
        for f1, f2 in zip(a["selection_frequency"], other["selection_frequency"]):
            np.testing.assert_array_equal(f1, f2)
    assert not np.array_equal(a["oob_scores"], pysimlr.bagged_fit("pca", v, 1, **{**kw, "draw_random_state": 6})["oob_scores"])


def test_determinism_simlr_select(data):
    (z, v), _ = data
    kw = dict(aggregate="select", n_subsamples=5, random_state=1)
    a, b = pysimlr.bagged_fit("simlr", v, 1, **kw), pysimlr.bagged_fit("simlr", v, 1, **kw)
    np.testing.assert_array_equal(a["scores"], b["scores"])
    assert all((x == y).all() for x, y in zip(a["feature_mask"], b["feature_mask"]))


@pytest.mark.parametrize("aggregate", ["select", "average"])
def test_missing_rows_mask_runs(aggregate):
    z, v = planted(n=90, seed=2)
    v = [x.copy() for x in v]
    v[1][:12] = np.nan                                          # view 1 unobserved on 12 rows
    r = pysimlr.bagged_fit("simlr", v, 1, aggregate=aggregate, n_subsamples=5, missing_rows="mask")
    assert np.asarray(r["scores"]).shape == (90, 1) and np.isfinite(r["scores"]).all()
    if aggregate == "select":
        assert pysimlr.predict(r, v)["u"].shape == (90, 1)


def test_other_registered_methods_run_both_modes():
    """Every registered method supports both modes (none raises NotImplementedError)."""
    z, v = planted(n=60, seed=5)
    for method in ("ned", "ned_shared_private", "nsa_pipeline"):
        kw = {} if method == "nsa_pipeline" else {"iterations": 10}
        for aggregate in ("select", "average"):
            r = pysimlr.bagged_fit(method, v, 1, aggregate=aggregate, n_subsamples=3, **kw)
            assert pysimlr.predict(r, v)["u"].shape[0] == 60


def test_loading_inference_models_refits(data):
    (z, v), _ = data
    inf = pysimlr.loading_inference("pca", v, 1, n_subsamples=4, return_refits=("models", "rows"), random_state=0)
    models = inf["refits"]["models"]
    assert len(models) == 4 and all(m["method"] == "pca" for m in models)
    with pytest.raises(ValueError, match="unknown return_refits"):
        pysimlr.loading_inference("pca", v, 1, n_subsamples=2, return_refits=("nope",))


def test_bound_reported_only_at_half_fraction(data):
    (z, v), _ = data
    r = pysimlr.bagged_fit("pca", v, 1, n_subsamples=4)
    assert all(np.isnan(x) for x in r["expected_false_selections"]) and r["expected_false_selections_reason"]
    r5 = pysimlr.bagged_fit("pca", v, 1, n_subsamples=4, fraction=0.5)
    assert all(np.isfinite(x) for x in r5["expected_false_selections"])
    assert r5["provenance"]["subsample"] == "without replacement"
