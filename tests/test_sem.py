"""`pysimlr.sem`: score-level path model (plan docs/plans/2026-11-06_path_model.md, section 7)."""
import time
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import sem
from pysimlr.benchmarks.synthetic_cases import build_path_graph_case
from pysimlr.loadings import _oob_metrics

CHAIN = [(0, 1), (1, 2)]


def _fit(method, views, k, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pysimlr.fit(method, views, k, random_state=0, **kw)


def _split(case, n_tr):
    v = case["views"]
    return [x[:n_tr] for x in v], [x[n_tr:] for x in v]


def _edge(pm, a, b):
    e = pm["edges"]
    return e[(e.a == a) & (e.b == b)].iloc[0]


def _pm(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sem.path_model(*args, **kw)


# ----------------------------------------------------------------------------- 1
def _old_oob_scores(res, sub_views, oob_views):
    """The score extraction `_oob_metrics` carried before the refactor (oracle)."""
    from pysimlr.api import _preprocess_views
    pred = pysimlr.predict(res, oob_views)
    rot = res["loading_rotation"].detach().double().numpy()
    u_tr = res["u"].detach().double().numpy() @ rot
    if not res.get("basis_is_joint") and len(res["latents"]) == len(sub_views):
        z_tr = [z.detach().double().numpy() @ rot for z in res["latents"]]
        z_oob = [z.detach().double().numpy() @ rot for z in pred["latents"]]
    else:
        xs_tr, _ = _preprocess_views(list(sub_views), res["scale_list"], res["provenance_list"])
        xs_oob, _ = _preprocess_views(list(oob_views), res["scale_list"], res["provenance_list"])
        v_rot = [v.detach().double().numpy() @ rot for v in res["v"]]
        z_tr = [x.double().numpy() @ vr for x, vr in zip(xs_tr, v_rot)]
        z_oob = [x.double().numpy() @ vr for x, vr in zip(xs_oob, v_rot)]
    out = []
    u_tr_c = u_tr - u_tr.mean(0, keepdims=True)
    for zm_tr, zm_oob in zip(z_tr, z_oob):
        sgn = np.where(((zm_tr - zm_tr.mean(0, keepdims=True)) * u_tr_c).sum(0) >= 0.0, 1.0, -1.0)
        zc = zm_oob * sgn[None, :]
        zc = zc - zc.mean(0, keepdims=True)
        sd = zc.std(0, keepdims=True)
        out.append(np.where(sd > 0, zc / np.where(sd > 0, sd, 1.0), 0.0))
    return out


@pytest.mark.parametrize("method", ["simlr", "pca"])
def test_view_scores_matches_oob_metrics_frame(method):
    case = build_path_graph_case(CHAIN, 300, 2, (12, 10, 8), 0.5, random_state=1)
    tr, te = _split(case, 200)
    res = _fit(method, tr, 2, **({"iterations": 30} if method == "simlr" else {}))
    want = _old_oob_scores(res, tr, te)
    got = sem.view_scores(res, te, train_views=tr)
    assert got["source"] == ("joint_block" if method == "pca" else "latents") and got["sign_rows"] == "train"
    for a, b in zip(got["scores"], want):
        assert np.array_equal(a, b)
    _, coup, scores = _oob_metrics(res, tr, te, return_scores=True)
    tot = sum(want)
    ref = np.mean([np.sum(z * (tot - z), 0) / np.sqrt((z ** 2).sum(0) * ((tot - z) ** 2).sum(0)) for z in want], 0)
    assert np.array_equal(np.asarray(coup), ref)
    assert all(np.array_equal(s, w.astype(np.float32)) for s, w in zip(scores, want))


# ----------------------------------------------------------------------------- 2
def test_pillai_is_rotation_invariant():
    g = np.random.default_rng(0)
    n, k = 200, 4
    L = g.standard_normal((n, k))
    Za = sem._zscore(L + 0.8 * g.standard_normal((n, k)))
    Zb = sem._zscore(L + 0.8 * g.standard_normal((n, k)))
    C = g.standard_normal((n, 2))
    N, _, _ = sem._nuisance(n, C)
    o1 = sem._observed(sem._edge_design({"a": Za, "b": Zb}, N, "a", "b", []))
    Rq, _ = np.linalg.qr(g.standard_normal((k, k)))
    o2 = sem._observed(sem._edge_design({"a": Za, "b": Zb @ Rq}, N, "a", "b", []))
    assert abs(o1["pillai"] - o2["pillai"]) < 1e-10
    assert np.max(np.abs(o1["r"] - o2["r"])) > 1e-3
    # equals the mean squared canonical correlation computed directly
    Q = np.linalg.qr(N)[0]
    Ra, Rb = Za - Q @ (Q.T @ Za), Zb - Q @ (Q.T @ Zb)
    rho = np.linalg.svd(np.linalg.qr(Ra)[0].T @ np.linalg.qr(Rb)[0], compute_uv=False)
    assert abs(o1["pillai"] - np.mean(rho ** 2)) < 1e-10


# ----------------------------------------------------------------------------- 3
def test_freedman_lane_removes_covariate_inflation():
    n_seeds = 30
    family, inflated, rejected = 0, 0, 0
    for seed in range(n_seeds):
        case = build_path_graph_case([], 400, 2, (12, 10, 8), 0.5, covariate_effect=1.0, random_state=seed)
        tr, te = _split(case, 200)
        res = _fit("simlr", tr, 2, iterations=30)
        raw = _pm(res, te, n_permutations=999, n_bootstrap=0, random_state=seed)
        adj = _pm(res, te, covariates=case["C"][200:], n_permutations=999, n_bootstrap=0, random_state=seed)
        inflated += int(raw["edges"].pillai.min() > 0.2 and raw["edges"].p_perm.max() < 0.01)
        family += int(adj["edges"].p_maxt.min() < 0.05)
        rejected += int((adj["edges"].p_perm < 0.05).sum())
    assert inflated == n_seeds
    # Plan deviation (WP1 report): "every edge p_perm > 0.05 in 9 of 10 seeds" is failed by a
    # calibrated test (3 null edges: P(all pass) = 0.857 per seed, P(>= 9 of 10) = 0.55; seeds 0-9
    # give 6 of 10 while 100 seeds give per-edge rates 0.03/0.06/0.06 and max-T 0.07). Rates over
    # 30 seeds are bounded instead, at about 2.2 binomial SD above nominal.
    assert rejected / (3 * n_seeds) <= 0.10
    assert family / n_seeds <= 0.15


# ----------------------------------------------------------------------------- 4
def test_absent_edges_pass_given_parents_and_fail_pairwise():
    ok = 0
    for seed in range(10):
        case = build_path_graph_case(CHAIN, 500, 3, (20, 25, 18), 0.3, random_state=seed)
        tr, te = _split(case, 350)
        res = _fit("simlr", tr, 3, iterations=50)
        pw = _pm(res, te, graph=CHAIN, conditioning="pairwise", n_permutations=999, n_bootstrap=0)
        pa = _pm(res, te, graph=CHAIN, conditioning="parents", n_permutations=999, n_bootstrap=0)
        ok += int(_edge(pw, "view0", "view2").p_perm < 0.05 and _edge(pa, "view0", "view2").p_perm >= 0.05)
        assert pa["fit"]["absent_p"] is not None and "view0 - view2" in pa["fit"]["absent_p"]
    assert ok >= 8      # plan F2: estimated scores attenuate, not remove, an absent edge


def test_sink_fit_and_covariates_compose():
    """Plan F3: covariates leaking into every view; the sink at fit time, covariates= in path_model."""
    ok = 0
    for seed in range(10):
        case = build_path_graph_case(CHAIN, 500, 3, (20, 25, 18), 0.5, covariate_effect=0.8, random_state=seed)
        tr, te = _split(case, 350)
        res = _fit("simlr", tr, 3, iterations=50, confounds=case["C"][:350])
        pm = _pm(res, te, graph=CHAIN, covariates=case["C"][350:], conditioning="parents",
                 n_permutations=999, n_bootstrap=0)
        assert pm["nuisance_names"] == ["intercept", "cov0", "cov1"]
        floor = 1.0 / 1000
        pres = pm["edges"][pm["edges"].present]
        ok += int((pres.p_perm == floor).all() and (pm["edges"][~pm["edges"].present].p_perm > 0.05).all())
    assert ok >= 8


# ----------------------------------------------------------------------------- 5
@pytest.mark.slow
def test_null_calibration_fixed_model():
    rej = {}
    fam = 0
    n_data = 200
    for seed in range(n_data):
        case = build_path_graph_case([], 240, 2, (8, 7, 6), 0.5, random_state=1000 + seed)
        tr, te = _split(case, 120)
        res = _fit("simlr", tr, 2, iterations=10)
        pm = _pm(res, te, n_permutations=500, n_bootstrap=0, random_state=seed)
        for r in pm["edges"].itertuples():
            rej[(r.a, r.b)] = rej.get((r.a, r.b), 0) + int(r.p_perm < 0.05)
        fam += int(pm["edges"].p_maxt.min() < 0.05)
    rates = {e: v / n_data for e, v in rej.items()}
    print(f"fixed_calibration: per_edge={list(rates.values())} maxt_family={fam / n_data:.3f}")
    assert all(0.03 <= r <= 0.07 for r in rates.values()), rates
    assert 0.03 <= fam / n_data <= 0.07, fam / n_data


# ----------------------------------------------------------------------------- 7
KEYS = {"nodes", "graph", "directed", "conditioning", "rows", "n", "k", "edges", "edge_components", "moderation",
        "outcome_edges", "omnibus", "fit", "families", "edge_models", "feature_contributions", "null",
        "scores_source", "signs", "parameters", "warnings"}


@pytest.mark.parametrize("method", ["simlr", "lend", "ned", "flow_v", "pca"])
def test_every_family_runs(method):
    case = build_path_graph_case(CHAIN, 160, 2, (8, 7, 6), 0.5, random_state=0)
    tr, te = _split(case, 100)
    res = _fit(method, tr, 2, **({} if method == "pca" else {"iterations": 3}))
    pm = _pm(res, te, graph=CHAIN, outcome=case["outcome"][100:], groups=case["G"][100:], conditioning="parents",
             n_permutations=99, n_bootstrap=50, train_views=tr)
    assert KEYS <= set(pm)
    ps = np.concatenate([pm["edges"].p_perm, pm["edges"].p_maxt, pm["edge_components"].p_perm,
                         pm["moderation"].p_perm, pm["outcome_edges"].p_holm])
    assert np.all((ps > 0) & (ps <= 1))
    assert pm["scores_source"] == ("joint_block" if method == "pca" else "latents")
    assert pm["edges"].ci_low.notna().all()


# ----------------------------------------------------------------------------- 8
def test_moderation_detects_planted_group_effect():
    ok = 0
    for seed in range(10):
        case = build_path_graph_case(CHAIN, 600, 2, (12, 10, 8), 0.3, group_effect=0.5, random_state=seed)
        tr, te = _split(case, 300)
        res = _fit("simlr", tr, 2, iterations=50)
        pm = _pm(res, te, graph=CHAIN, groups=case["G"][300:], n_permutations=999, n_bootstrap=0)
        m = pm["moderation"]
        m = m[m.component == "sum"]
        p01 = m[(m.a == "view0") & (m.b == "view1")].p_perm.iloc[0]
        p12 = m[(m.a == "view1") & (m.b == "view2")].p_perm.iloc[0]
        ok += int(p01 < 0.05 and p12 > 0.05)
    assert ok >= 8


# ----------------------------------------------------------------------------- 9
def test_outcome_edges_are_the_ancova():
    case = build_path_graph_case(CHAIN, 400, 3, (12, 10, 8), 0.5, covariate_effect=0.3, random_state=3)
    tr, te = _split(case, 250)
    res = _fit("simlr", tr, 3, iterations=30)
    C, y, G = case["C"][250:], case["outcome"][250:], case["G"][250:]
    pm = _pm(res, te, covariates=C, groups=G, outcome=y, n_permutations=99, n_bootstrap=0)
    z = sem.view_scores(res, te)["scores"][2]
    X = np.column_stack([np.ones(len(y)), C, G == 1])
    res_ = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    ry = res_(y)
    oc = pm["edge_components"]
    oc = oc[(oc.a == "view2") & (oc.b == "outcome")].sort_values("component")
    want = [np.corrcoef(res_(z[:, j]), ry)[0, 1] for j in range(3)]
    assert np.allclose(oc.r.to_numpy(float), want, atol=1e-8)
    assert set(pm["outcome_edges"].a) == {"view0", "view1", "view2"}


# ----------------------------------------------------------------------------- 15
def test_wide_view_training_fixed_null_is_refused():
    g = np.random.default_rng(0)
    xs = [torch.tensor(g.standard_normal((60, p))).float() for p in (8, 80)]
    res = _fit("simlr", xs, 2, iterations=10)
    with pytest.warns(RuntimeWarning, match="wide views"):
        pm = sem.path_model(res, xs, rows="train", null="fixed", n_permutations=99, n_bootstrap=0)
    assert pm["edges"].p_perm.isna().all() and pm["omnibus"]["p_max"] is None and pm["p_reason"]
    with pytest.warns(RuntimeWarning, match="allow_train_fixed=True"):
        pm = sem.path_model(res, xs, rows="train", null="fixed", n_permutations=99, n_bootstrap=0,
                            allow_train_fixed=True)
    assert pm["edges"].p_perm.notna().all()
    pm = sem.path_model(res, xs, rows="train", n_refits=3, n_bootstrap=0)     # default null on training rows
    assert pm["parameters"]["null"] == "refit" and pm["edges"].p_refit.notna().all() and not pm["warnings"]


# ----------------------------------------------------------------------------- 17
def test_fixed_null_cost():
    case = build_path_graph_case(None, 600, 5, (20, 20, 20, 20), 0.5, random_state=0)
    tr, te = _split(case, 300)
    res = _fit("simlr", tr, 5, iterations=10)
    t0 = time.perf_counter()
    pm = _pm(res, te, n_permutations=10_000, n_bootstrap=0)
    dt = time.perf_counter() - t0
    assert len(pm["edges"]) == 6 and pm["parameters"]["n_permutations"] == 10_000
    assert dt < 30.0, dt


# ----------------------------------------------------------------------------- extras
def test_p_values_are_add_one_and_nulls_stored():
    case = build_path_graph_case(CHAIN, 300, 2, (12, 10, 8), 0.3, random_state=0)
    tr, te = _split(case, 150)
    res = _fit("simlr", tr, 2, iterations=30)
    pm = _pm(res, te, graph=CHAIN, n_permutations=199, n_bootstrap=0)
    assert pm["edges"].p_perm.min() >= 1 / 200
    assert all(len(v) == 199 for v in pm["null"].values())
    assert pm["omnibus"]["p_max"] == 1 / 200


def test_bootstrap_interval_covers_the_estimate():
    case = build_path_graph_case(CHAIN, 400, 2, (12, 10, 8), 0.5, random_state=2)
    tr, te = _split(case, 200)
    res = _fit("simlr", tr, 2, iterations=30)
    pm = _pm(res, te, graph=CHAIN, n_permutations=0, n_bootstrap=400)
    e = pm["edges"]
    assert ((e.ci_low <= e.pillai + 1e-12) & (e.pillai <= e.ci_high + 0.05)).all()
    assert e.p_perm.isna().all() and pm["p_reason"].startswith("n_permutations = 0")
    c = pm["edge_components"]
    assert ((c.beta_ci_low <= c.beta) & (c.beta <= c.beta_ci_high)).all()


def test_inference_hook_gives_refit_intervals():
    case = build_path_graph_case(CHAIN, 300, 2, (12, 10, 8), 0.3, random_state=0)
    tr, te = _split(case, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inf = pysimlr.loading_inference("simlr", tr, 2, n_subsamples=6, iterations=20)
    assert len(inf["oob_scores"]) == 6 and inf["n_rows"] == 200
    pm = _pm(inf["result"], te, graph=CHAIN, inference=inf, n_permutations=99, n_bootstrap=0)
    e = pm["edges"]
    assert e.ci_refit_low.notna().all() and (e.ci_refit_low <= e.ci_refit_high).all()
    assert pm["parameters"]["n_refit_pillai"] == 6
    with pytest.raises(ValueError, match="train_covariates"):
        _pm(inf["result"], te, covariates=case["C"][200:], inference=inf, n_permutations=9, n_bootstrap=0)


def test_undirected_bic_counts_each_edge_once():
    Z = {"a": np.zeros((10, 3)), "b": np.zeros((10, 3)), "c": np.zeros((10, 2))}
    models = {nm: {"W": None, "resid_var": np.ones(z.shape[1]), "n_nuisance": 2} for nm, z in Z.items()}
    adj = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    assert sem._n_params(models, Z, adj, False) == (2 * 8 + 8) + 9 + 6


def test_compare_path_models_parsimony_rule():
    case = build_path_graph_case(CHAIN, 500, 3, (20, 25, 18), 0.3, random_state=0)
    tr, te = _split(case, 350)
    res = _fit("simlr", tr, 3, iterations=50)
    cmp = sem.compare_path_models(res, tr, te, {"chain": CHAIN, "complete": None, "empty": []},
                                  n_bootstrap=500, n_permutations=199)
    rows = {r["graph"]: r for r in cmp["comparison"]}
    assert cmp["best"] == "chain" and cmp["best_by_loglik"] in rows
    assert rows[cmp["best_by_loglik"]]["delta_vs_top"] == 0.0
    assert rows["chain"]["ci_low"] <= 0.0 <= rows["chain"]["ci_high"]


# ----------------------------------------------------------------------------- WP2: refit null
def test_refit_null_on_training_rows():
    case = build_path_graph_case(CHAIN, 160, 2, (8, 7, 6), 0.4, random_state=0)
    res = _fit("simlr", case["views"], 2, iterations=5, learning_rate=0.05)
    pm = _pm(res, case["views"], rows="train", graph=CHAIN, outcome=case["outcome"], null="both",
             n_permutations=199, n_refits=9, n_bootstrap=0, conditioning="parents", allow_train_fixed=True)
    e = pm["edges"]
    assert pm["parameters"]["null"] == "both" and e.p_perm.notna().all()
    assert e.p_refit.notna().all() and (e.p_refit >= 1 / 10).all() and (e.p_refit <= 1).all()
    assert e.p_refit_holm.notna().all() and pm["omnibus"]["p_mean_refit"] >= 1 / 10
    assert len(pm["refit_null"]["per_edge"][("view0", "view1")]) == 9
    pr = _pm(res, case["views"], rows="train", graph=CHAIN, n_refits=4, n_bootstrap=0, conditioning="parents")
    assert pr["parameters"]["null"] == "refit" and pr["edges"].p_perm.isna().all()
    assert pr["fit"]["null"] == "refit" and pr["fit"]["max_absent_p"] is None and pr["fit"]["reason"]
    with pytest.raises(ValueError, match="rows='train'"):
        sem.path_model(res, case["views"], null="refit")


# ----------------------------------------------------------------------------- 6
@pytest.mark.slow
def test_null_calibration_refit():
    """100 datasets with three independent views, n_refits = 100, simlr with 5 iterations."""
    rej = np.zeros(3)
    n_data = 100
    for seed in range(n_data):
        case = build_path_graph_case([], 100, 2, (8, 7, 6), 0.5, random_state=2000 + seed)
        res = _fit("simlr", case["views"], 2, iterations=5, learning_rate=0.05)
        pm = _pm(res, case["views"], rows="train", null="refit", n_refits=100, n_bootstrap=0,
                 refit_scope=("edges",), random_state=seed, n_jobs=4)
        rej += (pm["edges"].p_refit.to_numpy(float) < 0.05)
    rates = rej / n_data
    print(f"refit_calibration: per_edge={rates.tolist()} mean={rates.mean():.3f} max={rates.max():.3f}")
    assert 0.02 <= rates.mean() <= 0.09, rates
    assert rates.max() <= 0.12, rates


# ----------------------------------------------------------------------------- 14
def test_refit_null_permutes_confounds_with_the_view(monkeypatch):
    import pysimlr.api as api
    case = build_path_graph_case(CHAIN, 120, 2, (8, 7, 6), 0.5, covariate_effect=0.5, random_state=0)
    views, C = case["views"], case["C"]
    res = _fit("simlr", views, 2, iterations=5, learning_rate=0.05, confounds=C)
    calls, real = [], api.fit

    def spy(method, vs, k, **kw):
        calls.append(([v.clone() for v in vs], kw.get("confounds")))
        return real(method, vs, k, **kw)

    monkeypatch.setattr(api, "fit", spy)
    _pm(res, views, rows="train", graph=CHAIN, covariates=C, null="refit", n_refits=2, n_bootstrap=0)
    assert len(calls) == 2 * 2 + 2                 # nodes view1, view2 (2 refits each) + 2 global refits
    for vs, conf in calls:
        assert isinstance(conf, list) and len(conf) == 3
        for m in range(3):
            key = {float(x): i for i, x in enumerate(views[m][:, 0])}
            perm = np.array([key[float(x)] for x in vs[m][:, 0]])
            assert np.array_equal(np.asarray(conf[m]), C[perm])
    moved = [sum(not torch.equal(vs[m], views[m]) for m in range(3)) for vs, _ in calls]
    assert moved == [1, 1, 1, 1, 2, 2]


# ----------------------------------------------------------------------------- 13
def test_permutation_test_routes_through_fit():
    case = build_path_graph_case([(0, 1)], 60, 2, (6, 5), 0.3, random_state=0)
    res = pysimlr.permutation_test(case["views"], 2, n_permutations=4, method="lend", iterations=2,
                                   batch_size=20, warmup_epochs=0)
    assert {"observed_similarity", "null_similarities", "p_value", "n_permutations", "n_exceeding"} <= set(res)
    assert res["result"]["method"] == "lend" and len(res["null_similarities"]) == 4
    assert res["p_value"] == pytest.approx((1 + res["n_exceeding"]) / 5) and res["p_value"] >= 1 / 5
    assert len(res["edges"]) == 1


# ----------------------------------------------------------------------------- 10
def test_compare_path_models_selects_planted_graph():
    hits, fractions = 0, []
    graphs = {"chain": CHAIN, "star": [(0, 1), (0, 2)], "complete": None, "empty": []}
    for seed in range(10):
        case = build_path_graph_case(CHAIN, 500, 3, (20, 25, 18), 0.3, random_state=seed)
        tr, te = _split(case, 350)
        res = _fit("simlr", tr, 3, iterations=50)
        cmp = sem.compare_path_models(res, tr, te, graphs, n_bootstrap=500, n_permutations=199, random_state=seed)
        rows = {r["graph"]: r for r in cmp["comparison"]}
        hits += int(cmp["best"] == "chain")
        fractions.append(rows["chain"]["best_fraction"])
        assert rows[cmp["best_by_loglik"]]["delta_vs_top"] == 0.0
    assert hits >= 8, (hits, fractions)


# ----------------------------------------------------------------------------- 11
def test_compare_training_topologies_runs_and_is_paired(monkeypatch):
    seen = []
    real = sem.path_model

    def spy(result, views, **kw):
        seen.append(views[0][:, 0].clone())
        return real(result, views, **kw)

    monkeypatch.setattr(sem, "path_model", spy)
    case = build_path_graph_case(CHAIN, 200, 2, (8, 7, 6), 0.4, random_state=0)
    out = sem.compare_path_models(method="simlr", k=2, views=case["views"], graphs={"chain": CHAIN, "empty": []},
                                  n_splits=2, fit_params={"iterations": 10}, outcome=case["outcome"],
                                  n_permutations=19)
    assert out["mode"] == "training_topology" and out["splits"].shape[0] == 4
    assert list(out["splits"].split) == [0, 0, 1, 1] and out["splits"].outcome_r2.notna().all()
    assert out["summary"].shape[0] == 2 and len(out["test_rows"]) == 2
    assert torch.equal(seen[0], seen[1]) and torch.equal(seen[2], seen[3]) and not torch.equal(seen[0], seen[2])
    assert all(len(t) == 60 for t in out["test_rows"])


# ----------------------------------------------------------------------------- 12
def test_predict_path_full_views_equals_view_scores():
    case = build_path_graph_case(CHAIN, 300, 2, (12, 10, 8), 0.3, random_state=0)
    tr, te = _split(case, 200)
    res = _fit("simlr", tr, 2, iterations=30)
    pm = _pm(res, te, graph=CHAIN, n_permutations=0, n_bootstrap=0)
    pp = sem.predict_path(pm, te)
    vs = sem.view_scores(res, te)["scores"]
    assert pp["predicted_from"] == {} and all(np.array_equal(pp["scores"][f"view{m}"], vs[m]) for m in range(3))


def test_predict_path_missing_view_is_predicted_from_neighbours():
    case = build_path_graph_case(CHAIN, 500, 3, (20, 25, 18), 0.3, random_state=0)
    tr, te = _split(case, 350)
    res = _fit("simlr", tr, 3, iterations=50)
    pm = _pm(res, te, graph=CHAIN, outcome=case["outcome"][350:], n_permutations=0, n_bootstrap=0)
    pp = sem.predict_path(pm, [te[0], None, te[2]], views_true=te)
    assert pp["predicted_from"]["view1"] == ["view0", "view2"]
    assert pp["r2_per_node"]["view1"] > 0.5
    assert pp["outcome_pred"].shape == (150, 1)
    lone = sem.predict_path(pm, [te[0], None, None], outcome=False)
    assert lone["outcome_pred"] is None and lone["predicted_from"]["view1"] == ["view0"]


# ----------------------------------------------------------------------------- WP2: joint fits, feature_maps
def test_joint_fit_predicts_from_a_subset_of_views():
    case = build_path_graph_case(CHAIN, 200, 2, (8, 7, 6), 0.4, random_state=0)
    tr, te = _split(case, 120)
    res = _fit("pca", tr, 2)
    pred = pysimlr.predict(res, [te[0], None, te[2]])
    assert set(pred["latents"]) == {"view0", "view2"} and tuple(pred["u"].shape) == (80, 2)
    full = sem.view_scores(res, te, train_views=tr)
    part = sem.view_scores(res, [te[0], None, te[2]], train_views=tr)
    assert part["scores"][1] is None and np.array_equal(part["scores"][0], full["scores"][0])
    h = pysimlr.heldout_coupling(res, [te[0], None, te[2]], train_views=tr)
    assert h["views"] == [0, 2] and len(h["cross_view_heldout"]) == 2


def test_feature_maps_scores_come_from_view_scores():
    """The folded extraction keeps feature_maps' convention bitwise: signs learned on the given rows
    against the given u (oracle: the removed `loadings._view_scores`)."""
    from pysimlr.api import _preprocess_views
    from pysimlr.loadings import _zscore
    case = build_path_graph_case(CHAIN, 120, 2, (8, 7, 6), 0.4, random_state=0)
    for method in ("simlr", "pca"):
        res = _fit(method, case["views"], 2, **({} if method == "pca" else {"iterations": 10}))
        rot = res["loading_rotation"].double().numpy()
        u = pysimlr.predict(res, case["views"])["u"]
        if method == "pca":
            xs = _preprocess_views(case["views"], res["scale_list"], res["provenance_list"])[0]
            zs = [x.double().numpy() @ (v.double().numpy() @ rot) for x, v in zip(xs, res["v"])]
        else:
            zs = [np.asarray(z.double()) @ rot for z in pysimlr.predict(res, case["views"])["latents"]]
        uc = np.asarray(u.double()) @ rot
        uc = uc - uc.mean(0)
        want = [_zscore(z * np.where(((z - z.mean(0)) * uc).sum(0) >= 0, 1.0, -1.0)) for z in zs]
        got = sem.view_scores(res, case["views"], sign_against=u)
        assert got["sign_rows"] == "evaluation"
        assert all(np.array_equal(a, b) for a, b in zip(got["scores"], want))


def test_adni_pilot_provenance_guards(tmp_path):
    """D1 regression guard: --figures-only refuses missing CSVs and _verify_p4_consistency
    rejects internally inconsistent (n_ge_refit=0 with q95_refit >= pillai) records."""
    import pandas as pd
    from scripts.adni_path_model_pilot import _verify_p4_consistency, run_pilot

    with pytest.raises(FileNotFoundError, match="requires all live CSV outputs"):
        run_pilot([0], ["simlr"], out_dir=tmp_path, figures_only=True)

    valid = pd.DataFrame([{
        "method": "simlr", "a": "v0", "b": "v1", "pillai": 0.89,
        "n_ge_fixed": 0, "p_perm_fixed": 1 / 51,
        "n_ge_refit": 0, "p_refit": 1 / 51,
        "q95_fixed": 0.007, "q95_refit": 0.33,
    }])
    _verify_p4_consistency(valid, 50)

    impossible = valid.copy()
    impossible["q95_refit"] = 0.91  # > pillai while n_ge_refit == 0
    with pytest.raises(ValueError, match="Impossible refit null record"):
        _verify_p4_consistency(impossible, 50)


# ----------------------------------------------------------------------------- T1: training rows
def test_training_rows_fixed_null_not_reported_for_any_view():
    """rows='train' with the fixed-model null reports no p-values, wide view or not, unless
    allow_train_fixed=True; the refit null is the default there."""
    case = build_path_graph_case(CHAIN, 160, 2, (8, 7, 6), 0.4, random_state=3)
    res = _fit("simlr", case["views"], 2, iterations=5, learning_rate=0.05)
    with pytest.warns(RuntimeWarning, match="p-values are not reported"):
        pm = sem.path_model(res, case["views"], rows="train", null="fixed", n_permutations=99, n_bootstrap=0)
    assert pm["wide_views"] == [] and pm["edges"].p_perm.isna().all() and pm["edges"].p_holm.isna().all()
    assert pm["omnibus"]["p_max"] is None and pm["fit"]["reason"] and "allow_train_fixed" in pm["p_reason"]
    with pytest.warns(RuntimeWarning, match="allow_train_fixed=True"):
        pa = sem.path_model(res, case["views"], rows="train", null="fixed", n_permutations=99, n_bootstrap=0,
                            allow_train_fixed=True)
    assert pa["edges"].p_perm.notna().all() and pa["omnibus"]["p_max"] is not None
    ph = _pm(res, case["views"], n_permutations=99, n_bootstrap=0)          # held-out rows: reported, no warning
    assert ph["edges"].p_perm.notna().all() and not ph["warnings"]


# ----------------------------------------------------------------------------- T2: scores from any source
def test_path_model_from_scores_matches_path_model():
    case = build_path_graph_case(CHAIN, 300, 2, (8, 7, 6), 0.5, covariate_effect=0.5, random_state=0)
    tr, te = _split(case, 200)
    res = _fit("simlr", tr, 2, iterations=5, learning_rate=0.05, confounds=case["C"][:200])
    C, y = case["C"][200:], case["outcome"][200:]
    kw = dict(outcome=y, covariates=C, groups=case["G"][200:],
              conditioning="parents", n_permutations=199, n_bootstrap=60, random_state=4)
    pm = _pm(res, te, graph=[(0, 1), (1, 2), (2, 3)], **kw)
    names = ["view0", "view1", "view2"]
    sc = dict(zip(names, sem.view_scores(res, te)["scores"]))
    ps = sem.path_model_from_scores(sc, graph=[("view0", "view1"), ("view1", "view2"), ("view2", "outcome")], **kw)
    for col in ("pillai", "rv", "p_perm", "p_maxt", "p_holm", "q_bh", "ci_low", "ci_high"):
        assert np.allclose(pm["edges"][col].astype(float), ps["edges"][col].astype(float), equal_nan=True), col
    for col in ("r", "beta", "p_perm", "ci_low", "beta_ci_high"):
        assert np.allclose(pm["edge_components"][col].astype(float), ps["edge_components"][col].astype(float),
                           equal_nan=True), col
    assert pm["omnibus"]["p_max"] == ps["omnibus"]["p_max"]
    for key in ("present_pass", "absent_pass", "max_absent_p", "rejected", "absent_p"):
        assert pm["fit"][key] == ps["fit"][key], key
    assert np.isclose(pm["fit"]["max_absent_pillai"], ps["fit"]["max_absent_pillai"])
    assert np.allclose(pm["outcome_edges"].p_perm, ps["outcome_edges"].p_perm)
    assert ps["result"] is None and ps["scores_source"] == "given" and len(ps["feature_contributions"]) == 0
    assert ps["parameters"]["scores_given"] and ps["parameters"]["null"] == "fixed"
    # prediction through the graph from a mapping with a missing block equals the view-based one
    G = case["G"][200:]
    a = sem.predict_path(pm, [te[0], None, te[2]], covariates=C, groups=G)
    b = sem.predict_path(ps, {"view0": sc["view0"], "view1": None, "view2": sc["view2"]}, covariates=C, groups=G)
    assert np.allclose(a["scores"]["view1"], b["scores"]["view1"]) and np.allclose(a["outcome_pred"], b["outcome_pred"])
    assert b["predicted_from"]["view1"] == ["view0", "view2"]
    # blocks of different width: pillai only
    wide3 = np.hstack([sc["view1"], sc["view2"][:, :1]])
    pw = sem.path_model_from_scores({"a": sc["view0"], "b": wide3}, n_permutations=49, n_bootstrap=0)
    assert len(pw["edges"]) == 1 and len(pw["edge_components"]) == 0 and pw["edges"].p_perm.notna().all()
    p1 = sem.path_model_from_scores({"a": sc["view0"], "b": sc["view1"][:, :1]}, n_permutations=49, n_bootstrap=0)
    assert len(p1["edge_components"]) == 2                   # a one-column block: each component against it
    # training-row declaration follows the same policy; no refit null, no consensus, no inference
    with pytest.warns(RuntimeWarning, match="allow_train_fixed"):
        pt = sem.path_model_from_scores(sc, rows="train", n_permutations=49, n_bootstrap=0)
    assert pt["edges"].p_perm.isna().all()
    with pytest.raises(ValueError, match="no measurement model"):
        sem.path_model_from_scores(sc, null="refit")
    with pytest.raises(ValueError, match="same number of rows"):
        sem.path_model_from_scores({"a": sc["view0"], "b": sc["view1"][:-1]})
    with pytest.raises(ValueError, match="non-finite"):
        sem.path_model_from_scores({"a": sc["view0"], "b": np.full_like(sc["view1"], np.nan)})
    with pytest.raises(ValueError, match="unknown score name"):
        sem.predict_path(ps, {"nope": sc["view0"]}, covariates=C, groups=G)
    assert pysimlr.path_model_from_scores is sem.path_model_from_scores
