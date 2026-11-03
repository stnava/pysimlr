"""Fused ridge: exact solution, the two limits, prediction routing, CV, plots."""
import numpy as np
import pytest

import pysimlr
from pysimlr.fused_ridge import fit_fused_ridge, fused_ridge_cv, predict_fused_ridge


def _data(seed=0, n=240, p=6, G=4, shared=True):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    g = np.repeat(np.arange(G), n // G)
    base = rng.standard_normal(p)
    W = np.stack([base + (0 if shared else rng.standard_normal(p)) for _ in range(G)])
    y = np.einsum("np,np->n", X, W[g]) + 0.5 * rng.standard_normal(n) + g  # group intercepts
    return X, y, g, W


def _objective(X, y, masks, w, b, l1, l2, edges):
    f = 0.0
    for k, m in enumerate(masks):
        f += ((y[m] - X[m] @ w[k] - b[k]) ** 2).mean() + l1 * (w[k] ** 2).sum()
    return f + l2 * sum(((w[a] - w[c]) ** 2).sum() for a, c in edges)


def test_solution_is_the_minimiser():
    X, y, g, _ = _data()
    r = fit_fused_ridge(X, y, g, lambda1=0.1, lambda2=2.0)
    masks = [g == k for k in r["groups"]]
    f0 = _objective(X, y, masks, r["coef"], r["intercept"], 0.1, 2.0, r["edges"])
    rng = np.random.default_rng(1)
    for _ in range(5):
        dw = 1e-3 * rng.standard_normal(r["coef"].shape); db = 1e-3 * rng.standard_normal(len(masks))
        assert _objective(X, y, masks, r["coef"] + dw, r["intercept"] + db, 0.1, 2.0, r["edges"]) > f0


def test_lambda2_zero_gives_independent_ridges_and_large_lambda2_one_shared():
    X, y, g, _ = _data(shared=False)
    r0 = fit_fused_ridge(X, y, g, lambda1=0.3, lambda2=0.0)
    for k, m in enumerate([g == j for j in r0["groups"]]):
        Z = X[m] - X[m].mean(0); t = y[m] - y[m].mean(); n = m.sum()
        w = np.linalg.solve(Z.T @ Z / n + 0.3 * np.eye(X.shape[1]), Z.T @ t / n)
        assert np.allclose(r0["coef"][k], w, atol=1e-8)
    rbig = fit_fused_ridge(X, y, g, lambda1=0.3, lambda2=1e6)
    assert np.abs(rbig["coef"] - rbig["coef"].mean(0)).max() < 1e-4


def test_overlapping_masks_and_custom_fusion_graph():
    X, y, g, _ = _data()
    t = X[:, 0]
    masks = [t > q for q in np.quantile(t, [0.1, 0.3, 0.5])]  # nested strata
    r = fit_fused_ridge(X, y, masks, lambda1=0.1, lambda2=1.0, fusion=[(0, 2)])
    assert r["coef"].shape == (3, X.shape[1]) and r["edges"] == [(0, 2)]
    with pytest.raises(ValueError, match="fusion edge"):
        fit_fused_ridge(X, y, masks, fusion=[(0, 5)])
    with pytest.raises(ValueError, match="lambda1"):
        fit_fused_ridge(X, y, g, lambda1=0.0)


def test_predict_routes_rows_to_their_group():
    X, y, g, _ = _data()
    r = fit_fused_ridge(X, y, g)
    allg = predict_fused_ridge(r, X)
    assert allg.shape == (len(y), len(r["groups"]))
    assert np.allclose(predict_fused_ridge(r, X, group=g), allg[np.arange(len(y)), g])
    assert np.allclose(predict_fused_ridge(r, X, group=2), allg[:, 2])
    with pytest.raises(ValueError, match="unknown group"):
        predict_fused_ridge(r, X, group=9)


def test_cv_prefers_fusion_when_groups_share_coefficients():
    X, y, g, _ = _data(n=120, p=10, G=6, shared=True)
    cv = fused_ridge_cv(X, y, g, lambda1s=(0.01, 0.1), lambda2s=(0.0, 10.0), n_folds=4)
    assert cv["best"][1] == 10.0 and cv["mse"].shape == (2, 2)
    assert cv["result"]["lambda2"] == 10.0


def test_plots_return_figures_and_accept_an_axis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, y, g, _ = _data()
    r = fit_fused_ridge(X, y, g, feature_names=[f"f{i}" for i in range(X.shape[1])])
    cv = fused_ridge_cv(X, y, g, lambda1s=(0.1,), lambda2s=(0.0, 1.0), n_folds=3)
    for fig in (pysimlr.plot_fused_coefficients(r), pysimlr.plot_coefficient_paths(r, features=["f0", 2]),
                pysimlr.plot_fused_ridge_cv(cv),
                pysimlr.plot_predictions(y, predict_fused_ridge(r, X, group=g))):
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
    fig, ax = plt.subplots()
    assert pysimlr.plot_coefficient_paths(r, ax=ax) is fig
    plt.close(fig)
