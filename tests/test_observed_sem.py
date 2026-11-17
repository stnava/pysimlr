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


def _parallel_mediators_data(n=800, seed=17):
    r = np.random.default_rng(seed)
    X = r.standard_normal(n)
    B = 0.5 * X + np.sqrt(1.0 - 0.25) * r.standard_normal(n)
    C = 0.4 * X + np.sqrt(1.0 - 0.16) * r.standard_normal(n)
    # Var(0.4*B + 0.35*C + 0.25*X):
    # = 0.16 + 0.1225 + 0.0625 + 2*(0.4*0.35*0.2 + 0.4*0.25*0.5 + 0.35*0.25*0.4) = 0.571
    M = 0.4 * B + 0.35 * C + 0.25 * X + np.sqrt(1.0 - 0.571) * r.standard_normal(n)
    return {"X": X, "B": B, "C": C, "M": M}


def test_through_chain_zero_edge_parallel_mediators_and_invariance():
    # 1) Planted chain X -> B -> M with direct X -> M: through == (X>B)*(B>M), total == direct + through, CI covers planted
    r = np.random.default_rng(21)
    n = 800
    X = r.standard_normal(n)
    B = 0.5 * X + np.sqrt(1.0 - 0.25) * r.standard_normal(n)
    M = 0.4 * B + 0.3 * X + np.sqrt(1.0 - 0.37) * r.standard_normal(n)
    d = {"X": X, "B": B, "M": M}
    models = {
        "med": {"edges": [("X", "B"), ("B", "M"), ("X", "M")]},
        "no_out": {"edges": [("X", "B"), ("X", "M")]},  # B has no outgoing edge
    }

    base = compare_observed_path_models(d, models, exogenous=["X"], n_boot=120, random_state=5)
    with_t = compare_observed_path_models(
        d, models, exogenous=["X"], n_boot=120, random_state=5, through=[("X", "M", "B")],
    )

    # Zero drift on existing keys and table
    assert "through" not in base["fits"]["med"]
    for row_a, row_b in zip(base["table"], with_t["table"]):
        assert set(row_a) == set(row_b)
        for k in row_a:
            va, vb = row_a[k], row_b[k]
            assert va == vb or (isinstance(va, float) and np.isnan(va) and np.isnan(vb))
    assert base["n_boot_failed"] == with_t["n_boot_failed"] == 0
    for mname in models:
        for k, v in base["bootstrap"][mname].items():
            assert with_t["bootstrap"][mname][k] == v

    # Chain identity and CI coverage
    med_b = with_t["bootstrap"]["med"]
    assert med_b["through:X>M|B"]["estimate"] == pytest.approx(
        med_b["X>B"]["estimate"] * med_b["B>M"]["estimate"], abs=1e-12,
    )
    assert med_b["total:X>M"]["estimate"] == pytest.approx(
        med_b["X>M"]["estimate"] + med_b["through:X>M|B"]["estimate"], abs=1e-12,
    )
    assert med_b["through:X>M|B"]["lo"] <= 0.20 <= med_b["through:X>M|B"]["hi"]
    assert with_t["fits"]["med"]["through"] == [
        {"from": "X", "to": "M", "through": "B", "estimate": med_b["through:X>M|B"]["estimate"]}
    ]

    # Model where B has no outgoing edge: exact 0.0 and zero-width interval
    no_out_b = with_t["bootstrap"]["no_out"]
    assert no_out_b["through:X>M|B"] == {"estimate": 0.0, "lo": 0.0, "hi": 0.0}
    assert with_t["fits"]["no_out"]["through"] == [
        {"from": "X", "to": "M", "through": "B", "estimate": 0.0}
    ]

    # 2) Two parallel mediators: through B + through C == indirect:X>M; also test disconnected path via B
    dp = _parallel_mediators_data(n=500, seed=18)
    pmodels = {
        "dual": {"edges": [("X", "B"), ("B", "M"), ("X", "C"), ("C", "M"), ("X", "M")]},
        "b_unreachable": {"edges": [("X", "C"), ("C", "M"), ("B", "M"), ("X", "M")]},  # B->M exists, X->+B does not
    }
    pcmp = compare_observed_path_models(
        dp, pmodels, exogenous=["X"], n_boot=0, through=[("X", "M", "B"), ("X", "M", "C")],
    )
    assert "bootstrap" not in pcmp and "bootstrap_draws" not in pcmp
    est_by_m = {row["through"]: row["estimate"] for row in pcmp["fits"]["dual"]["through"]}
    ind = next(row for row in pcmp["fits"]["dual"]["indirect"] if row["from"] == "X" and row["to"] == "M")
    assert est_by_m["B"] + est_by_m["C"] == pytest.approx(ind["indirect"], abs=1e-12)
    b_unreach = {row["through"]: row["estimate"] for row in pcmp["fits"]["b_unreachable"]["through"]}
    assert b_unreach["B"] == 0.0


def test_through_matches_manual_resample_loop_jobs_determinism_draws_and_all_mediators():
    dp = _parallel_mediators_data(n=240, seed=31)
    strata = np.repeat(["s0", "s1", "s2"], [60, 90, 90])
    models = {
        "dual": {"edges": [("X", "B"), ("B", "M"), ("X", "C"), ("C", "M"), ("X", "M")]},
        "only_b": {"edges": [("X", "B"), ("B", "M"), ("X", "M")], "residual_covariances": [("B", "C")]},
    }
    triples = [("X", "M", "B"), ("X", "M", "C")]

    res1 = compare_observed_path_models(
        dp, models, exogenous=["X"], n_boot=12, strata=strata, random_state=11,
        n_jobs=1, through=triples, return_draws=True,
    )
    res2 = compare_observed_path_models(
        dp, models, exogenous=["X"], n_boot=12, strata=strata, random_state=11,
        n_jobs=2, through=triples, return_draws=True,
    )
    res_nodraws = compare_observed_path_models(
        dp, models, exogenous=["X"], n_boot=12, strata=strata, random_state=11,
        n_jobs=1, through=triples,
    )
    assert "bootstrap_draws" not in res_nodraws
    assert _same(res1, res2)
    for mname in models:
        for k, arr in res1["bootstrap_draws"][mname].items():
            assert isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.dtype == float
            assert np.array_equal(arr, res2["bootstrap_draws"][mname][k])

    # Manual loop over the exact same _resample_indices
    Z, nodes, st = osem._prepare(dp, None, False, strata)
    idx_list = osem._resample_indices(len(Z), st, 12, 11)
    for mname, spec in models.items():
        manual_draws = {m: [] for _, _, m in triples}
        for ix in idx_list:
            fit = observed_path_model(
                Z[ix], spec["edges"], exogenous=["X"],
                residual_covariances=spec.get("residual_covariances", ()), names=nodes,
            )
            for _, _, m in triples:
                manual_draws[m].append(osem.path_effect(fit["B"], nodes, "X", "M", through=m))
        for _, _, m in triples:
            key = f"through:X>M|{m}"
            arr = np.asarray(manual_draws[m], dtype=float)
            assert np.max(np.abs(res1["bootstrap_draws"][mname][key] - arr)) <= 1e-12
            assert res1["bootstrap"][mname][key]["lo"] == pytest.approx(float(np.percentile(arr, 2.5)), abs=1e-12)
            assert res1["bootstrap"][mname][key]["hi"] == pytest.approx(float(np.percentile(arr, 97.5)), abs=1e-12)

    # Different random_state moves intervals
    res_other = compare_observed_path_models(
        dp, models, exogenous=["X"], n_boot=12, strata=strata, random_state=12, through=triples,
    )
    assert not _same(res1, res_other)

    # "all_mediators" matches explicit triples (both with exogenous and when exogenous is omitted)
    res_auto = compare_observed_path_models(
        dp, models, exogenous=["X"], n_boot=12, strata=strata, random_state=11, through="all_mediators",
    )
    assert _same(res_nodraws, res_auto)
    res_noexog_exp = compare_observed_path_models(
        dp, models, n_boot=0, through=triples,
    )
    res_noexog_auto = compare_observed_path_models(
        dp, models, n_boot=0, through="all_mediators",
    )
    assert res_noexog_auto["fits"]["dual"]["through"] == res_noexog_exp["fits"]["dual"]["through"]
    assert res_noexog_auto["fits"]["only_b"]["through"] == res_noexog_exp["fits"]["only_b"]["through"]


def test_through_validation_errors():
    d = chain_data(120, seed=7)
    models = {"chain": {"edges": CHAIN}}
    with pytest.raises(ValueError, match=r"\('A', 'C', 'Z'\)"):
        compare_observed_path_models(d, models, through=[("A", "C", "Z")])
    with pytest.raises(ValueError, match=r"\('Z', 'C', 'B'\)"):
        compare_observed_path_models(d, models, through=[("Z", "C", "B")])
    with pytest.raises(ValueError, match=r"\('A', 'C', 'A'\)"):
        compare_observed_path_models(d, models, through=[("A", "C", "A")])
    with pytest.raises(ValueError, match=r"\('A', 'C', 'C'\)"):
        compare_observed_path_models(d, models, through=[("A", "C", "C")])
    with pytest.raises(ValueError, match=r"\('A', 'A', 'B'\)"):
        compare_observed_path_models(d, models, through=[("A", "A", "B")])
    with pytest.raises(ValueError):
        compare_observed_path_models(d, models, through="not_a_mode")
    with pytest.raises(ValueError):
        compare_observed_path_models(d, models, through=("A", "C", "B"))
    with pytest.raises(ValueError):
        compare_observed_path_models(d, models, through=[("A", "C")])
    with pytest.raises(ValueError, match="n_boot > 0"):
        compare_observed_path_models(d, models, n_boot=0, return_draws=True)
    with pytest.raises(ValueError, match="n_boot > 0"):
        compare_observed_path_models(d, models, n_boot=-1, return_draws=True)
