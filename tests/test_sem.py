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
    with pytest.raises(NotImplementedError):
        sem.path_model(res, xs, rows="train", n_permutations=9)


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
