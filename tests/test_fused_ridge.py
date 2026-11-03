"""Fused ridge: strata and outcome-threshold models, GLM families, covariates, top_k, CV,
ANTsR parity, and plots."""
import shutil
import subprocess
import warnings

import numpy as np
import pytest

import pysimlr
from pysimlr.fused_ridge import (_loss, _mean, fit_fused_ridge, fused_ridge_cv,
                                 predict_fused_ridge)


def _strata(seed=0, n=240, p=6, G=4, shared=True):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)); g = np.repeat(np.arange(G), n // G)
    base = rng.standard_normal(p)
    W = np.stack([base + (0 if shared else rng.standard_normal(p)) for _ in range(G)])
    y = np.einsum("np,np->n", X, W[g]) + 0.5 * rng.standard_normal(n) + g
    return X, y, g


def _ordinal(seed=0, n=400, p=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)); b = np.array([1.0, -0.8, 0.5, 0.0, 0.0])[:p]
    y = X @ b + rng.logistic(size=n)
    return X, y, np.quantile(y, [0.25, 0.5, 0.75])


def _penalised_grad(r, X, y, groups=None, thresholds=None, covariates=None):
    """The gradient of the documented objective at the fitted point (standardised scale)."""
    n = X.shape[0]
    Xs = (X - r["x_center"]) / r["x_scale"]
    C = np.hstack([np.zeros((n, 0)) if covariates is None else covariates, np.ones((n, 1))])
    if thresholds is not None:
        masks = [np.ones(n, bool)] * len(thresholds); ys = [(y >= c).astype(float) for c in thresholds]
    else:
        masks = [groups == t for t in r["tasks"]]; ys = [y] * len(masks)
    T = len(masks); lap = np.zeros((T, T))
    for a, b in r["edges"]:
        lap[a, a] += 1; lap[b, b] += 1; lap[a, b] -= 1; lap[b, a] -= 1
    W, V = r["coef_std"], r["cov_coef"]; gw = 2 * r["lambda1"] * W + 2 * r["lambda2"] * lap @ W
    gv = np.zeros_like(V)
    for t, m in enumerate(masks):
        mu = _mean(Xs[m] @ W[t] + C[m] @ V[t], r["family"]); res = (mu - ys[t][m]) / (T * m.sum())
        gw[t] += Xs[m].T @ res; gv[t] += C[m].T @ res
    return np.abs(np.concatenate([gw.ravel(), gv.ravel()])).max()


# --- strata (gaussian, exact) -------------------------------------------------
def test_gaussian_strata_is_exact_and_has_the_two_limits():
    X, y, g = _strata(shared=False)
    r = fit_fused_ridge(X, y, groups=g, lambda1=0.1, lambda2=2.0)
    assert r["convergence"]["stop_reason"] == "closed_form" and r["model"] == "strata"
    assert _penalised_grad(r, X, y, groups=g) < 1e-10
    r0 = fit_fused_ridge(X, y, groups=g, lambda1=0.3, lambda2=0.0, standardize=False)
    for t, m in enumerate([g == j for j in r0["tasks"]]):   # independent ridges
        Z = X[m] - X[m].mean(0); s = y[m] - y[m].mean(); T = len(r0["tasks"]); nm = m.sum()
        w = np.linalg.solve(Z.T @ Z / (T * nm) + 2 * 0.3 * np.eye(X.shape[1]), Z.T @ s / (T * nm))
        assert np.allclose(r0["coef_std"][t], w, atol=1e-8)
    big = fit_fused_ridge(X, y, groups=g, lambda1=0.3, lambda2=1e6)
    assert np.abs(big["coef_std"] - big["coef_std"].mean(0)).max() < 1e-4


def test_overlapping_masks_custom_edges_and_errors():
    X, y, _ = _strata()
    masks = [X[:, 0] > q for q in np.quantile(X[:, 0], [0.1, 0.3, 0.5])]
    r = fit_fused_ridge(X, y, groups=masks, fusion=[(0, 2)])
    assert r["coef"].shape == (3, X.shape[1]) and r["edges"] == [(0, 2)]
    with pytest.raises(ValueError, match="fusion edge"):
        fit_fused_ridge(X, y, groups=masks, fusion=[(0, 5)])
    with pytest.raises(ValueError, match="exactly one"):
        fit_fused_ridge(X, y)
    with pytest.raises(ValueError, match="exactly one"):
        fit_fused_ridge(X, y, groups=masks, thresholds=[0.0])
    with pytest.raises(ValueError, match="lambda1"):
        fit_fused_ridge(X, y, groups=masks, lambda1=0.0)


def test_predict_routes_rows_and_returns_original_scale():
    X, y, g = _strata()
    r = fit_fused_ridge(X, y, groups=g)
    allt = predict_fused_ridge(r, X)
    assert allt.shape == (len(y), len(r["tasks"]))
    assert np.allclose(allt, X @ r["coef"].T + r["intercept"][None, :])   # original-scale coef
    assert np.allclose(predict_fused_ridge(r, X, task=g), allt[np.arange(len(y)), g])
    with pytest.raises(ValueError, match="unknown task"):
        predict_fused_ridge(r, X, task=9)


# --- thresholds (binomial, the ANTsR model) and poisson ------------------------
def test_binomial_thresholds_converge_to_the_minimiser_and_tie_slopes():
    X, y, c = _ordinal()
    r = fit_fused_ridge(X, y, thresholds=c, lambda1=0.01, lambda2=0.0, tol=1e-12)
    assert r["family"] == "binomial" and r["convergence"]["stop_reason"] == "converged"
    assert _penalised_grad(r, X, y, thresholds=c) < 1e-6
    tied = fit_fused_ridge(X, y, thresholds=c, lambda1=0.01, lambda2=1e4)
    assert np.abs(tied["coef_std"] - tied["coef_std"].mean(0)).max() < 1e-3
    p = predict_fused_ridge(tied, X, type="response")     # P(y >= c_t): decreasing in t
    assert np.all(np.diff(p.mean(0)) < 0)


def test_poisson_strata_and_unpenalised_covariates():
    rng = np.random.default_rng(3); n = 300
    X = rng.standard_normal((n, 4)); g = rng.integers(0, 3, n); cov = rng.standard_normal((n, 1))
    y = rng.poisson(np.exp(0.3 * X[:, 0] + 0.8 * cov[:, 0] + 0.2))
    r = fit_fused_ridge(X, y, groups=g, family="poisson", covariates=cov, lambda1=0.05, tol=1e-12)
    assert _penalised_grad(r, X, y, groups=g, covariates=cov) < 1e-6
    shrunk = fit_fused_ridge(X, y, groups=g, family="poisson", covariates=cov, lambda1=1e4)
    assert np.abs(shrunk["coef_std"]).max() < 1e-3
    assert np.all(np.abs(shrunk["cov_coef"][:, 0]) > 0.5)    # covariate effect is unpenalised


def test_unconverged_glm_warns():
    X, y, c = _ordinal()
    with pytest.warns(pysimlr.ConvergenceWarning):
        fit_fused_ridge(X, y, thresholds=c, max_iter=1)


# --- top_k ----------------------------------------------------------------------
def test_top_k_selects_the_union_and_refits():
    X, y, c = _ordinal(p=5)
    full = fit_fused_ridge(X, y, thresholds=c, lambda1=0.01, lambda2=0.1)
    r = fit_fused_ridge(X, y, thresholds=c, lambda1=0.01, lambda2=0.1, top_k=2)
    union = sorted({int(j) for t in range(len(c)) for j in np.argsort(-np.abs(full["coef_std"][t]))[:2]})
    assert r["selected_features"] == union
    assert np.all(r["coef_std"][:, [j for j in range(5) if j not in union]] == 0)
    direct = fit_fused_ridge(X[:, union], y, thresholds=c, lambda1=0.01, lambda2=0.1)
    assert np.allclose(r["coef"][:, union], direct["coef"], atol=1e-8)


def test_top_k_in_predict_truncates_without_refit():
    X, y, c = _ordinal()
    r = fit_fused_ridge(X, y, thresholds=c, lambda1=0.01, lambda2=0.1)
    W = r["coef_std"].copy()
    for t in range(W.shape[0]):
        W[t, np.argsort(-np.abs(W[t]))[1:]] = 0
    manual = ((X - r["x_center"]) / r["x_scale"]) @ W.T + r["cov_coef"][:, -1][None, :]
    assert np.allclose(predict_fused_ridge(r, X, top_k=1), manual)


def test_cv_chooses_fusion_and_selects_inside_folds(monkeypatch):
    X, y, g = _strata(n=120, p=10, G=6, shared=True)
    import pysimlr.fused_ridge as fr
    seen, real = [], fr.fit_fused_ridge

    def spy(X_, y_, **kw):
        seen.append((X_.shape[0], kw.get("top_k")))
        return real(X_, y_, **kw)
    monkeypatch.setattr(fr, "fit_fused_ridge", spy)
    cv = fr.fused_ridge_cv(X, y, groups=g, lambda1s=(0.01,), lambda2s=(0.0, 10.0),
                           top_ks=(None, 3), n_folds=4)
    assert cv["best"][1] == 10.0 and cv["loss"].shape == (1, 2, 2)
    fold_fits = seen[:-1]                                      # last call: refit on all rows
    assert all(n_ < X.shape[0] for n_, _ in fold_fits)         # top_k selection never sees all rows
    assert any(tk == 3 for _, tk in fold_fits)
    folds = np.arange(len(y)) % 3
    cv2 = fused_ridge_cv(X, y, groups=g, lambda1s=(0.1,), lambda2s=(1.0,), foldid=folds)
    assert cv2["se"].shape == (1, 1, 1)


# --- ANTsR parity (gated) ---------------------------------------------------------
def _has_antsr():
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(["Rscript", "-e", "cat(requireNamespace('ANTsR', quietly=TRUE))"],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip().endswith("TRUE")


def _antsr(tmp_path, X, y, c, body):
    np.savetxt(tmp_path / "X.csv", X, delimiter=","); np.savetxt(tmp_path / "y.csv", y, delimiter=",")
    script = (f"suppressWarnings(suppressMessages(library(ANTsR)))\n"
              f"X <- as.matrix(read.csv('{tmp_path}/X.csv', header=FALSE)); "
              f"y <- read.csv('{tmp_path}/y.csv', header=FALSE)[,1]; th <- c({','.join(map(repr, c.tolist()))})\n"
              + body + f"\nwrite.table(B, '{tmp_path}/B.csv', sep=',', row.names=FALSE, col.names=FALSE)")
    subprocess.run(["Rscript", "-e", script], check=True, capture_output=True, timeout=300)
    return np.atleast_2d(np.loadtxt(tmp_path / "B.csv", delimiter=","))


@pytest.mark.slow
@pytest.mark.skipif(not _has_antsr(), reason="Rscript with ANTsR is not available")
def test_matches_antsr_fused_ridge_direct(tmp_path):
    """Same objective as ANTsR::fusedRidgeDirect (the documented one), same solution."""
    X, y, c = _ordinal(n=300, p=5)
    B = _antsr(tmp_path, X, y, c, "B <- fusedRidgeDirect(X, y, thresholds=th, lambda1=0.05, lambda2=0.5, "
               "family='binomial', standardize=TRUE, cv=FALSE, "
               "optim_control=list(maxit=5000, pgtol=1e-10))$coefs_full")
    r = fit_fused_ridge(X, y, thresholds=c, lambda1=0.05, lambda2=0.5, tol=1e-12)
    assert np.allclose(r["coef_std"].T, B, atol=1e-3)


@pytest.mark.slow
@pytest.mark.skipif(not _has_antsr(), reason="Rscript with ANTsR is not available")
def test_top_k_matches_antsr_selection_and_refit(tmp_path):
    """ANTsR topK semantics: the selected set equals ANTsR::fusedRidge(topK=) (fusedRidgeDirect's
    topK path errors in ANTsR 0.6.7), and the refit equals fusedRidgeDirect on those columns."""
    X, y, c = _ordinal(n=300, p=5)
    B_sel = _antsr(tmp_path, X, y, c, "B <- fusedRidge(X, y, thresholds=th, lambda1=0.05, lambda2=0.5, "
                   "family='binomial', cv=FALSE, topK=2)$coefs_full")
    selected = sorted(np.flatnonzero(np.abs(B_sel).sum(1) > 0).tolist())
    r = fit_fused_ridge(X, y, thresholds=c, lambda1=0.05, lambda2=0.5, top_k=2, tol=1e-12)
    assert r["selected_features"] == selected
    cols = "c(" + ",".join(str(j + 1) for j in selected) + ")"
    B = _antsr(tmp_path, X, y, c, f"B <- fusedRidgeDirect(X[, {cols}, drop=FALSE], y, thresholds=th, "
               "lambda1=0.05, lambda2=0.5, family='binomial', standardize=TRUE, cv=FALSE, "
               "optim_control=list(maxit=5000, pgtol=1e-10))$coefs_full")
    assert np.allclose(r["coef_std"][:, selected].T, B, atol=1e-3)


# --- plots ----------------------------------------------------------------------
def test_fused_ridge_plots_return_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, y, c = _ordinal()
    r = fit_fused_ridge(X, y, thresholds=c, top_k=3, feature_names=[f"f{i}" for i in range(X.shape[1])])
    cv = fused_ridge_cv(X, y, thresholds=c, lambda1s=(0.1,), lambda2s=(0.0, 1.0), top_ks=(None, 2), n_folds=3)
    for fig in (pysimlr.plot_fused_coefficients(r), pysimlr.plot_coefficient_paths(r, features=["f0", 2]),
                pysimlr.plot_fused_ridge_cv(cv),
                pysimlr.plot_predictions(y >= c[1], predict_fused_ridge(r, X, task=c[1], type="response"))):
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
