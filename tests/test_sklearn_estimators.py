"""`MultiViewSiMR`: one scikit-learn estimator for every pysimlr method.

Regressions pinned here:
  * single-view `transform` projected *raw* X onto the loadings although `fit`
    had standardized X, so it disagreed with the library's own predictor (max
    abs difference 30.0 on a shifted, scaled input) and `fit_transform`
    disagreed with `result_["u"]`;
  * the LEND/NED wrappers passed both `iterations` and `epochs`, and the
    callee kept `iterations`, so the `epochs` argument never did anything;
  * `random_state` seeded the global torch/numpy RNG instead of the fit.
"""
import numpy as np
import pytest
import torch

import pysimlr

_LAZY_SKLEARN_IMPORTS = {
    "clone": ("sklearn.base", "clone"),
    "GridSearchCV": ("sklearn.model_selection", "GridSearchCV"),
    "cross_val_score": ("sklearn.model_selection", "cross_val_score"),
    "MultiViewPCA": ("pysimlr.sklearn", "MultiViewPCA"),
    "MultiViewSiMR": ("pysimlr.sklearn", "MultiViewSiMR"),
    "build_simlr_pipeline": ("pysimlr.sklearn", "build_simlr_pipeline"),
}


@pytest.fixture(scope="module", autouse=True)
def _lazy_module_imports():
    import importlib
    for _name, (_mod, _attr) in _LAZY_SKLEARN_IMPORTS.items():
        globals()[_name] = getattr(importlib.import_module(_mod), _attr)

K = 2


def _views(n=60, dims=(8, 6), seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, K))
    return [u @ rng.standard_normal((K, p)) * (1 + np.arange(p)) + shift
            + 0.2 * rng.standard_normal((n, p)) for p in dims]


def test_single_view_transform_uses_the_fitted_preprocessing():
    X = _views(dims=(8,), shift=10.0)[0]
    est = MultiViewSiMR(n_components=K, iterations=5, learning_rate=0.05, random_state=0).fit(X)
    z = est.transform(X)
    ref = pysimlr.predict(est.result_, [X])["u"].numpy()
    np.testing.assert_allclose(z, ref, atol=1e-6)
    assert abs(z.mean()) < 1.0, "latents of standardized input should be near zero-mean"
    np.testing.assert_allclose(clone(est).fit_transform(X), z, atol=1e-6)
    assert est.components_.shape == (K, 8) and est.loadings_.shape == (8, K)


def test_multi_view_list_input():
    views = _views()
    est = MultiViewSiMR(n_components=K, iterations=5, learning_rate=0.05, random_state=0).fit(views)
    assert isinstance(est.components_, list) and [c.shape for c in est.components_] == [(K, 8), (K, 6)]
    assert est.transform(views).shape == (60, K)
    assert est.n_features_in_ == 14 and est.family_ == "linear"


def test_view_sizes_reproduces_the_list_form():
    views = _views()
    a = MultiViewSiMR(n_components=K, iterations=5, learning_rate=0.05, random_state=0).fit(views).transform(views)
    X = np.hstack(views)
    est = MultiViewSiMR(n_components=K, iterations=5, learning_rate=0.05, random_state=0, view_sizes=[8, 6]).fit(X)
    np.testing.assert_allclose(est.transform(X), a, atol=1e-6)
    with pytest.raises(ValueError, match="sum to"):
        MultiViewSiMR(view_sizes=[8, 5]).fit(X)


@pytest.mark.parametrize("method", ["lend", "ned", "flow_v", "pca"])
def test_other_methods_through_the_same_estimator(method):
    views = _views()
    est = MultiViewSiMR(method=method, n_components=K, iterations=3, batch_size=16,
                        random_state=0).fit(views)
    assert est.transform(views).shape == (60, K)
    assert np.isfinite(est.score(views))


def test_the_budget_reaches_the_deep_models():
    """The old wrappers' `epochs` argument never reached training."""
    views = _views()
    short = MultiViewSiMR(method="ned", n_components=K, iterations=2, random_state=0).fit(views)
    long = MultiViewSiMR(method="ned", n_components=K, iterations=7, random_state=0).fit(views)
    assert len(short.result_["loss_history"]) == 2
    assert len(long.result_["loss_history"]) == 7


def test_a_parameter_the_method_cannot_use_is_recorded():
    with pytest.warns(RuntimeWarning, match="hidden_dims"):
        est = MultiViewSiMR(n_components=K, iterations=3, learning_rate=0.05, hidden_dims=(8,)).fit(_views())
    assert "hidden_dims" in est.parameters_ignored_
    with pytest.raises(TypeError):
        MultiViewSiMR(n_components=K, iterations=3, learning_rate=0.05, hidden_dims=(8,), strict=True).fit(_views())


def test_random_state_reproduces_and_leaves_the_global_stream_alone():
    views = _views()
    torch.manual_seed(5)
    expected = torch.randn(3)
    torch.manual_seed(5)
    a = MultiViewSiMR(method="lend", n_components=K, iterations=3, random_state=4).fit_transform(views)
    assert torch.equal(torch.randn(3), expected)
    b = MultiViewSiMR(method="lend", n_components=K, iterations=3, random_state=4).fit_transform(views)
    np.testing.assert_array_equal(a, b)


def test_grid_search_loops_over_methods():
    views = _views(n=48)
    X = np.hstack(views)
    est = MultiViewSiMR(n_components=K, iterations=3, batch_size=16, random_state=0, view_sizes=[8, 6])
    gs = GridSearchCV(est, {"method": ["simlr", "lend", "flow_v", "pca"]}, cv=2).fit(X)
    assert set(gs.cv_results_["param_method"]) == {"simlr", "lend", "flow_v", "pca"}
    assert np.isfinite(gs.cv_results_["mean_test_score"]).all()


def test_clone_and_get_params_round_trip():
    est = MultiViewSiMR(method="ned", n_components=3, hidden_dims=(16, 8), extra_params={"sim_weight": 2.0})
    c = clone(est)
    assert c.get_params() == est.get_params()


def test_inverse_transform_is_linear_only():
    views = _views()
    est = MultiViewSiMR(n_components=K, iterations=5, learning_rate=0.05, random_state=0).fit(views)
    recs = est.inverse_transform(est.transform(views))
    assert [r.shape for r in recs] == [(60, 8), (60, 6)]
    deep = MultiViewSiMR(method="lend", n_components=K, iterations=2).fit(views)
    with pytest.raises(NotImplementedError):
        deep.inverse_transform(deep.transform(views))


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_build_simlr_pipeline_cross_validates(task):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 10))
    y = (X[:, 0] > 0).astype(int) if task == "classification" else X[:, 0] + 0.1 * rng.standard_normal(60)
    pipe = build_simlr_pipeline(n_components=K, task=task, iterations=4, learning_rate=0.05, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=3)
    assert np.isfinite(scores).all()


def test_multiview_pca_view_sizes_matches_list_input():
    views = _views()
    a = MultiViewPCA(n_components=K).fit_transform(views)
    b = MultiViewPCA(n_components=K, view_sizes=[8, 6]).fit_transform(np.hstack(views))
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("method", ["simlr", "pca"])
def test_inverse_transform_returns_input_units_and_shape(method):
    """It returned a list for a concatenated input and, for simlr, values
    about 1/(n p) of the input scale (the "np" preprocessing step was not
    undone)."""
    views = _views(shift=20.0)
    X = np.hstack(views)
    est = MultiViewSiMR(method=method, n_components=K, iterations=8, random_state=0,
                        view_sizes=[8, 6]).fit(X)
    rec = est.inverse_transform(est.transform(X))
    assert isinstance(rec, np.ndarray) and rec.shape == X.shape
    rel = np.linalg.norm(rec - X) / np.linalg.norm(X - X.mean(0))
    assert rel < 1.0, f"{method}: reconstruction error {rel:.3f} of the centred signal"
    assert abs(rec.mean() - X.mean()) < 1.0


def test_score_compares_methods_on_shifted_data():
    """With a +20 mean, the baselines used to win any grid because their
    error was divided by the raw (mean-inflated) norm."""
    views = _views(n=60, shift=20.0)
    s_raw = {m: MultiViewSiMR(method=m, n_components=K, iterations=8, random_state=0).fit(views).score(views)
             for m in ("simlr", "pca")}
    centred = [v - v.mean(0) for v in views]
    s_c = {m: MultiViewSiMR(method=m, n_components=K, iterations=8, random_state=0).fit(centred).score(centred)
           for m in ("simlr", "pca")}
    for m in s_raw:
        assert abs(s_raw[m] - s_c[m]) < 1e-3


def test_build_simlr_pipeline_takes_a_list_of_views():
    views = _views()
    y = views[0][:, 0] > 0
    pipe = build_simlr_pipeline(n_components=K, iterations=4, learning_rate=0.05, random_state=0).fit(views, y)
    assert pipe.predict(views).shape == (60,)
