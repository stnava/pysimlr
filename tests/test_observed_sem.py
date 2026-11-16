"""Tests of pysimlr.observed_sem on synthetic data only."""
import numpy as np
import pytest
from scipy import optimize, stats

import pysimlr
from pysimlr import observed_sem as osem
from pysimlr.observed_sem import (
    compare_observed_path_models,
    covariance_equivalence,
    observed_path_model,
)

CHAIN = [("A", "B"), ("B", "C")]


def chain_data(n=4000, seed=0, a=0.6, b=0.5):
    r = np.random.default_rng(seed)
    A = r.standard_normal(n)
    B = a * A + r.standard_normal(n) * np.sqrt(1 - a * a)
    C = b * B + r.standard_normal(n) * np.sqrt(1 - b * b)
    return {"A": A, "B": B, "C": C}


def ml_chi2(R_sample, Sigma, n):
    """Independent chi2 from the ML discrepancy formula."""
    p = R_sample.shape[0]
    F = np.log(np.linalg.det(Sigma)) + np.trace(R_sample @ np.linalg.inv(Sigma)) - np.log(np.linalg.det(R_sample)) - p
    return (n - 1) * F


def test_saturated_recursive_model():
    d = chain_data(500)
    m = observed_path_model(d, [("A", "B"), ("B", "C"), ("A", "C")])
    assert m["df"] == 0 and m["q"] == 6
    assert abs(m["chi2"]) < 1e-9 and m["cfi"] == 1.0 and m["rmsea"] == 0.0
    assert np.isnan(m["p"])


def test_planted_chain_recovered_and_misspecification_rejected():
    d = chain_data(20000, seed=1)
    cmp = compare_observed_path_models(d, {"chain": {"edges": CHAIN}, "no_BC": {"edges": [("A", "B")]}})
    rows = {r["model"]: r for r in cmp["table"]}
    assert not rows["chain"]["rejected"] and rows["no_BC"]["rejected"]
    fit = cmp["fits"]["chain"]
    est = {(c["from"], c["to"]): c["estimate"] for c in fit["coefficients"]}
    assert est[("A", "B")] == pytest.approx(0.6, abs=0.03)
    assert est[("B", "C")] == pytest.approx(0.5, abs=0.03)
    (ind,) = fit["indirect"]
    assert (ind["from"], ind["to"]) == ("A", "C")
    assert ind["indirect"] == pytest.approx(est[("A", "B")] * est[("B", "C")], abs=1e-12)
    assert ind["direct"] == 0.0 and ind["total"] == ind["indirect"]
    assert cmp["best_by_bic"] == "chain"


def test_recursive_is_equationwise_ols_and_chi2_matches_independent_formula():
    r = np.random.default_rng(3)
    n = 300
    Z = r.standard_normal((n, 4))
    Z[:, 1] += 0.5 * Z[:, 0]
    Z[:, 2] += 0.4 * Z[:, 0] + 0.3 * Z[:, 1]
    Z[:, 3] += 0.5 * Z[:, 2]
    edges = [("x0", "x1"), ("x0", "x2"), ("x1", "x2"), ("x2", "x3")]
    m = observed_path_model(Z, edges)
    assert m["closed_form"] and m["converged"]
    Zs = (Z - Z.mean(0)) / Z.std(0)
    for tgt, pars in {"x1": ["x0"], "x2": ["x0", "x1"], "x3": ["x2"]}.items():
        j = int(tgt[1])
        X = Zs[:, [int(q[1]) for q in pars]]
        beta = np.linalg.lstsq(X, Zs[:, j], rcond=None)[0]
        got = [c["estimate"] for c in m["coefficients"] if c["to"] == tgt]
        assert np.allclose(got, beta, atol=1e-10)
    R = np.corrcoef(Z.T)
    # independent implied matrix: residual variances from OLS residuals
    B = np.zeros((4, 4))
    for c in m["coefficients"]:
        B[int(c["to"][1]), int(c["from"][1])] = c["estimate"]
    psi = np.diag([np.var(Zs[:, j] - Zs @ B[j]) for j in range(4)])
    Ainv = np.linalg.inv(np.eye(4) - B)
    Sigma = Ainv @ psi @ Ainv.T
    assert m["chi2"] == pytest.approx(ml_chi2(R, Sigma, n), rel=1e-9, abs=1e-9)
    assert np.allclose(m["implied"], Sigma, atol=1e-10)
    assert m["df"] == 10 - (4 + 4)
    assert m["p"] == pytest.approx(stats.chi2.sf(m["chi2"], m["df"]))


def test_residual_covariance_matches_brute_force_optimisation():
    r = np.random.default_rng(5)
    n = 800
    A = r.standard_normal(n)
    e = r.standard_normal((n, 2))
    B = 0.5 * A + e[:, 0]
    C = 0.4 * B + 0.6 * e[:, 0] + e[:, 1]  # correlated disturbance and a different predictor set
    D = r.standard_normal(n) + 0.3 * A
    data = {"A": A, "B": B, "C": C, "D": D}
    edges, rc = [("A", "B"), ("B", "C"), ("A", "D")], [("C", "D")]
    m = observed_path_model(data, edges, residual_covariances=rc)
    assert not m["closed_form"] and m["converged"]
    Z = np.column_stack([A, B, C, D])
    R = np.corrcoef(Z.T)
    idx = {k: i for i, k in enumerate("ABCD")}

    def F(th):
        Bm = np.zeros((4, 4))
        for k, (s, t) in enumerate(edges):
            Bm[idx[t], idx[s]] = th[k]
        Psi = np.diag(np.exp(th[3:7]))
        Psi[2, 3] = Psi[3, 2] = th[7]
        if np.linalg.eigvalsh(Psi).min() <= 0:
            return 1e6
        Ai = np.linalg.inv(np.eye(4) - Bm)
        Sg = Ai @ Psi @ Ai.T
        return np.log(np.linalg.det(Sg)) + np.trace(R @ np.linalg.inv(Sg)) - np.log(np.linalg.det(R)) - 4

    best = None
    for start in range(3):
        th0 = np.concatenate([np.full(3, 0.2), np.zeros(4), [0.05 * start]])
        res = optimize.minimize(F, th0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 20000, "maxfev": 20000})
        res = optimize.minimize(F, res.x, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    assert m["fit_function"] == pytest.approx(best.fun, abs=1e-7)
    assert m["chi2"] == pytest.approx((n - 1) * best.fun, abs=1e-4)
    assert m["residual_covariances"][0]["covariance"] == pytest.approx(best.x[7], abs=1e-3)
    # forcing the iterative path on a closed-form model agrees with the closed form
    m1 = observed_path_model(data, edges)
    m2 = observed_path_model(data, edges, force_numeric=True)
    assert m1["chi2"] == pytest.approx(m2["chi2"], abs=1e-6)


def test_markov_equivalence():
    d = chain_data(1000, seed=2)
    models = {"fwd": {"edges": CHAIN}, "rev": {"edges": [("C", "B"), ("B", "A")]}, "collider": {"edges": [("A", "B"), ("C", "B")]}}
    cmp = compare_observed_path_models(d, models)
    t = {r["model"]: r for r in cmp["table"]}
    assert t["fwd"]["chi2"] == pytest.approx(t["rev"]["chi2"], abs=1e-9)
    assert t["fwd"]["aic"] == pytest.approx(t["rev"]["aic"], abs=1e-9)
    assert t["collider"]["chi2"] != pytest.approx(t["fwd"]["chi2"], abs=1e-3)
    eq = covariance_equivalence(models)
    assert sorted(map(sorted, eq["classes"])) == [["collider"], ["fwd", "rev"]]
    assert eq["graphical_agrees"] and eq["n_pairs_graphical"] == 3
    pair = {(p["a"], p["b"]): p for p in eq["pairs"]}
    assert pair[("fwd", "rev")]["equivalent"] and not pair[("fwd", "collider")]["equivalent"]


def test_weights_bootstrap_determinism_jobs_and_strata():
    d = chain_data(400, seed=4)
    models = {"chain": {"edges": CHAIN}, "sat": {"edges": CHAIN + [("A", "C")]}, "fork": {"edges": [("B", "A"), ("B", "C")]},
              "rc": {"edges": [("A", "B")], "residual_covariances": [("B", "C")]}}
    strata = np.repeat(["s1", "s2", "s3"], [100, 150, 150])
    kw = {"n_boot": 6, "strata": strata, "random_state": 7}
    a = compare_observed_path_models(d, models, **kw)
    assert sum(r["w_aic"] for r in a["table"]) == pytest.approx(1.0)
    assert sum(r["w_bic"] for r in a["table"]) == pytest.approx(1.0)
    best = min(a["table"], key=lambda r: r["aic"])
    assert best["daic"] == 0.0 and a["best_by_aic"] == best["model"]
    assert min(r["dbic"] for r in a["table"]) == 0.0
    assert sum(r["best_share_aic"] for r in a["table"]) == pytest.approx(1.0)
    assert sum(r["best_share_bic"] for r in a["table"]) == pytest.approx(1.0)
    lo_hi = a["bootstrap"]["chain"]["A>B"]
    assert lo_hi["lo"] <= lo_hi["estimate"] + 0.2 and lo_hi["lo"] < lo_hi["hi"]
    assert "indirect:A>C" in a["bootstrap"]["chain"] and "B~~C" in a["bootstrap"]["rc"]
    b = compare_observed_path_models(d, models, **kw)
    assert _same(a, b)
    c = compare_observed_path_models(d, models, n_jobs=2, **kw)
    assert _same(a, c)
    other = compare_observed_path_models(d, models, **{**kw, "random_state": 8})
    assert not _same(a, other)
    # strata sizes preserved by the resampler
    for ix in osem._resample_indices(400, strata, 5, 0):
        assert np.array_equal(np.bincount(np.searchsorted(["s1", "s2", "s3"], strata[ix])), [100, 150, 150])
    # missing data
    d_nan = {k: v.copy() for k, v in d.items()}
    d_nan["B"][3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        observed_path_model(d_nan, CHAIN)
    assert observed_path_model(d_nan, CHAIN, dropna=True)["n"] == 399


def _same(a, b):
    ra, rb = a["table"], b["table"]
    for x, y in zip(ra, rb):
        for k in x:
            if isinstance(x[k], float) and not (x[k] == y[k] or (np.isnan(x[k]) and np.isnan(y[k]))):
                return False
    return a["bootstrap"] == b["bootstrap"]


def test_validation_errors_and_exogenous_covariate():
    d = chain_data(300)
    with pytest.raises(ValueError, match="cycle"):
        observed_path_model(d, [("A", "B"), ("B", "A")])
    with pytest.raises(ValueError, match="unknown"):
        observed_path_model(d, [("A", "Z")])
    with pytest.raises(ValueError, match="exogenous"):
        observed_path_model(d, [("B", "A")], exogenous=["A"])
    r = np.random.default_rng(123)
    d2 = {**d, "age": r.standard_normal(300) + 0.3 * d["A"]}
    m = observed_path_model(d2, CHAIN + [("age", "A"), ("age", "B"), ("age", "C")], exogenous=["age"])
    # q: 5 coefficients, 3 endogenous variances, 1 exogenous variance
    assert m["q"] == 9 and m["df"] == 10 - 9
    # covariate path equals regression on the covariate jointly with the other parents
    Z = np.column_stack([d2[k] for k in ("A", "B", "C", "age")])
    Z = (Z - Z.mean(0)) / Z.std(0)
    beta = np.linalg.lstsq(Z[:, [0, 3]], Z[:, 1], rcond=None)[0]
    got = {c["from"]: c["estimate"] for c in m["coefficients"] if c["to"] == "B"}
    assert got["A"] == pytest.approx(beta[0]) and got["age"] == pytest.approx(beta[1])


def test_path_effect_and_unstandardized():
    d = chain_data(500, seed=9)
    m = observed_path_model(d, CHAIN + [("A", "C")])
    tot = osem.path_effect(m["B"], m["nodes"], "A", "C")
    via = osem.path_effect(m["B"], m["nodes"], "A", "C", through="B")
    row = m["indirect"][0]
    assert tot == pytest.approx(row["total"]) and via == pytest.approx(row["indirect"])
    scaled = {"A": 3 * d["A"] + 1, "B": 2 * d["B"], "C": d["C"]}
    u = observed_path_model(scaled, CHAIN, standardize=False)
    s = observed_path_model(scaled, CHAIN)
    assert u["chi2"] == pytest.approx(s["chi2"], abs=1e-8)
    assert u["coefficients"][0]["estimate"] != pytest.approx(s["coefficients"][0]["estimate"])
    assert pysimlr.observed_path_model is observed_path_model


def test_optional_semopy_crosscheck():
    semopy = pytest.importorskip("semopy")
    d = chain_data(1000, seed=11)
    import pandas as pd
    df = pd.DataFrame(d)
    mod = semopy.Model("B ~ A\nC ~ B")
    mod.fit(df)
    stat = semopy.calc_stats(mod)
    mine = observed_path_model(df, CHAIN, standardize=False)
    assert float(stat["DoF"].iloc[0]) == mine["df"]
    # semopy's chi2 uses N-1 or N depending on version; allow either
    chi = float(stat["chi2"].iloc[0])
    assert min(abs(chi - mine["chi2"]), abs(chi - mine["chi2"] * 1000 / 999)) < 1e-2
