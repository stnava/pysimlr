"""Fast unit and contract tests (< 2 s total) for `pysimlr.presets`, zero-threshold
`sparseness='auto'`, `alpha_mode='auto'` (`jk_js`), and projected-mode STE gradients.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.deep import LENDNSAEncoder
from pysimlr.simlr import initialize_deep_encoders


def test_preset_catalog_and_lookup_contracts() -> None:
    items = pysimlr.list_presets()
    names = [p["name"] for p in items]
    assert len(items) == 9
    assert names[0] == "general_sparse"
    assert {"dense_consensus", "sparse_focal", "directional_shrunk", "nonlinear_ned"} <= set(names)
    for item in items:
        got = pysimlr.get_preset(item["name"])
        assert got["name"] == item["name"]
        assert got["summary"] and got["when_to_prefer"]
        # Mutating returned static_params copy must not mutate catalog
        got["static_params"]["sparseness"] = 0.123
        assert pysimlr.get_preset(item["name"])["static_params"]["sparseness"] != 0.123

    with pytest.raises(ValueError, match="unknown preset"):
        pysimlr.get_preset("not_a_preset")


def test_analyze_views_speed_determinism_and_guards() -> None:
    rng = np.random.default_rng(7)
    views = [rng.normal(size=(80, 24)), rng.normal(size=(80, 30)), rng.normal(size=(80, 18))]
    t0 = time.perf_counter()
    p1 = pysimlr.analyze_views(views, k=2)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    p2 = pysimlr.analyze_views(views, k=2)
    assert elapsed_ms < 120.0
    for key in ("recommended_k", "signal_detected", "loo_r2", "nonlinearity", "per_view_sparseness_1se"):
        assert p1[key] == p2[key]

    with pytest.raises(ValueError, match="at least 6 active rows"):
        pysimlr.analyze_views([rng.normal(size=(5, 8)), rng.normal(size=(5, 8))])
    with pytest.raises(ValueError, match="mismatched row counts"):
        pysimlr.analyze_views([rng.normal(size=(12, 6)), rng.normal(size=(14, 6))])


def test_null_noise_sets_signal_false_k1_and_general_sparse_top1() -> None:
    rng = np.random.default_rng(101)
    noise_views = [rng.normal(size=(96, 30)), rng.normal(size=(96, 36)), rng.normal(size=(96, 25))]
    rep = pysimlr.recommend_presets(noise_views)
    prof = rep["profile"]
    assert prof["signal_detected"] is False
    assert prof["recommended_k"] == 1
    assert rep["recommended"]["name"] == "general_sparse"
    assert "Fisher-z" in rep["recommended"]["why"]


def test_wachter_gcca_rank_recovers_multi_factor_subspace() -> None:
    rng = np.random.default_rng(202)
    n = 120
    z = rng.normal(size=(n, 3))
    views = [
        z @ rng.normal(size=(3, 28)) + 0.35 * rng.normal(size=(n, 28)),
        z @ rng.normal(size=(3, 32)) + 0.35 * rng.normal(size=(n, 32)),
        z @ rng.normal(size=(3, 24)) + 0.35 * rng.normal(size=(n, 24)),
    ]
    prof = pysimlr.analyze_views(views)
    assert prof["signal_detected"] is True
    assert prof["recommended_k"] == 3


def test_whole_view_missing_rows_route_to_mask_in_auto_and_none() -> None:
    rng = np.random.default_rng(303)
    z = rng.normal(size=(48, 2))
    v0 = z @ rng.normal(size=(2, 12)) + 0.4 * rng.normal(size=(48, 12))
    v1 = z @ rng.normal(size=(2, 14)) + 0.4 * rng.normal(size=(48, 14))
    v1[:10, :] = np.nan

    rep = pysimlr.recommend_presets({"mri": v0, "pet": v1}, k=2)
    assert rep["profile"]["has_whole_view_missing_rows"] is True
    assert rep["profile"]["whole_missing_views"] == ["pet"]
    assert rep["recommended"]["params"]["missing_rows"] == "mask"
    assert "missing_rows='mask'" in rep["recommended"]["why"]

    res_none = pysimlr.fit_preset(
        {"mri": v0, "pet": v1}, k=2, preset=None, iterations=2, learning_rate=0.05, random_state=0
    )
    assert res_none["preset_used"]["params"]["missing_rows"] == "mask"
    assert "preset_report" not in res_none


def test_prepare_views_signs_preserve_directional_cone() -> None:
    rng = np.random.default_rng(404)
    z = rng.normal(size=(50, 1))
    raw_a = 0.55 * z @ np.array([[1.0, 1.0, -1.0, -1.0, 1.0, -1.0]]) + rng.normal(size=(50, 6))
    raw_b = 0.55 * z @ np.ones((1, 8)) + rng.normal(size=(50, 8))
    prep = pysimlr.prepare_views(
        {"a": raw_a, "b": raw_b},
        signs={"a": np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0]), "b": np.ones(8)},
    )
    rep = pysimlr.recommend_presets(prep, k=1)
    assert rep["profile"]["is_prepared"] is True
    assert rep["profile"]["domain_signs_supplied"] is True
    assert rep["recommended"]["name"] == "directional_shrunk"
    assert rep["recommended"]["params"]["energy_type"] == "nc"

    res = pysimlr.fit_preset(prep, k=1, preset="auto", analysis=rep, iterations=3, learning_rate=0.05)
    assert res["preset_used"]["name"] == "directional_shrunk"
    assert "uniform_shrinkage_provenance" in res


def test_directional_polarity_detects_co_positive_vs_anticorrelated_cones() -> None:
    rng = np.random.default_rng(505)
    z = rng.normal(size=(64, 1))
    co_views = [
        0.5 * z @ np.ones((1, 60)) + rng.normal(size=(64, 60)),
        0.5 * z @ np.ones((1, 70)) + rng.normal(size=(64, 70)),
    ]
    anti_views = [co_views[0], -co_views[1]]

    rep_co = pysimlr.recommend_presets(co_views, k=1)
    rep_anti = pysimlr.recommend_presets(anti_views, k=1)
    assert rep_co["recommended"]["name"] == "directional_shrunk"
    assert rep_co["recommended"]["params"]["energy_type"] == "nc"
    assert rep_anti["recommended"]["name"] == "directional_shrunk"
    assert rep_anti["recommended"]["params"]["energy_type"] == "nc_abs"


def test_quadratic_nonlinearity_and_private_variance_separation() -> None:
    rng = np.random.default_rng(606)
    n = 160
    z = rng.normal(size=(n, 2))
    z_quad = (z ** 2 - 1.0) / np.sqrt(2.0)
    wa = rng.normal(size=(16, 2)) * 0.9
    wb = rng.normal(size=(18, 2)) * 0.9
    pure_nl = [
        z @ wa.T + 0.25 * rng.normal(size=(n, 16)),
        z_quad @ wb.T + 0.25 * rng.normal(size=(n, 18)),
    ]
    rep_nl = pysimlr.recommend_presets(pure_nl, k=2)
    assert rep_nl["recommended"]["name"] == "nonlinear_ned"
    assert rep_nl["profile"]["nonlinearity"]["ned_press_gain"] > 0.20

    # Add strong uncoupled view-private factors -> nonlinear_shared_private outranks nonlinear_ned
    priv_a = 2.8 * rng.normal(size=(n, 2)) @ rng.normal(size=(2, 12))
    priv_b = 2.8 * rng.normal(size=(n, 2)) @ rng.normal(size=(2, 12))
    sp_views = [
        np.hstack([priv_a, z[:, :1] @ np.ones((1, 8))]) + 0.25 * rng.normal(size=(n, 20)),
        np.hstack([priv_b, z_quad[:, :1] @ np.ones((1, 8))]) + 0.25 * rng.normal(size=(n, 20)),
    ]
    rep_sp = pysimlr.recommend_presets(sp_views, k=1)
    names_ranked = [r["name"] for r in rep_sp["recommendations"]]
    assert rep_sp["profile"]["private_variance_fraction"] > 0.60
    assert names_ranked.index("nonlinear_shared_private") < names_ranked.index("nonlinear_ned")


def test_format_preset_menu_and_allow_nonlinear_false() -> None:
    rng = np.random.default_rng(707)
    views = [rng.normal(size=(40, 10)), rng.normal(size=(40, 12))]
    rep_lin = pysimlr.recommend_presets(views, k=2, allow_nonlinear=False)
    assert all(r["complexity_tier"] == 1 for r in rep_lin["recommendations"])
    txt = pysimlr.format_preset_menu(rep_lin, max_rows=3)
    assert "PySiMLR Data Profile" in txt
    assert "#1" in txt and "#3" in txt and "#4" not in txt
    assert "Why:" in txt


def test_fit_preset_all_three_tiers_and_predict() -> None:
    rng = np.random.default_rng(808)
    z = rng.normal(size=(44, 2))
    views = [
        z @ rng.normal(size=(2, 14)) + 0.4 * rng.normal(size=(44, 14)),
        z @ rng.normal(size=(2, 16)) + 0.4 * rng.normal(size=(44, 16)),
    ]
    tr = [v[:32] for v in views]
    te = [v[32:] for v in views]

    # Tier 1: preset=None (no analyzer)
    r_none = pysimlr.fit_preset(tr, k=2, preset=None, iterations=2, learning_rate=0.05, random_state=0)
    assert r_none["preset_used"]["name"] == "general_sparse"
    assert r_none["preset_used"]["rank"] is None
    assert "preset_report" not in r_none
    pred_none = pysimlr.predict(r_none, te)
    assert pred_none["u"].shape == (12, 2)

    # Tier 2: preset="auto" (automated Rank #1)
    r_auto = pysimlr.fit_preset(
        tr, k="auto", preset="auto", allow_nonlinear=False, iterations=2, learning_rate=0.05, random_state=0
    )
    assert r_auto["preset_used"]["rank"] == 1
    assert "preset_report" in r_auto

    # Tier 3: explicit human choice (with and without analyze)
    r_exp = pysimlr.fit_preset(
        tr, k=2, preset="dense_consensus", analyze=False, iterations=2, learning_rate=0.05, random_state=0
    )
    assert r_exp["preset_used"]["name"] == "dense_consensus"
    assert r_exp["preset_used"]["params"]["sparseness"] is None

    with pytest.raises(ValueError, match="unknown preset"):
        pysimlr.fit_preset(tr, k=2, preset="invalid_name")


def test_single_view_and_fit_rows_subsetting() -> None:
    rng = np.random.default_rng(909)
    single = rng.normal(size=(36, 10))
    rep1 = pysimlr.recommend_presets(single, k=1)
    assert rep1["profile"]["n_views"] == 1
    assert rep1["recommended"]["k"] == 1

    v_a = rng.normal(size=(40, 10))
    v_b = rng.normal(size=(40, 12))
    mask = np.zeros(40, dtype=bool)
    mask[:24] = True
    res_sub = pysimlr.fit_preset(
        [v_a, v_b], k=1, preset="dense_consensus", fit_rows=mask, iterations=2, learning_rate=0.05, random_state=0
    )
    assert res_sub["u"].shape == (24, 1)


def test_sparseness_auto_in_simlr_and_deep_entry_points() -> None:
    rng = np.random.default_rng(2024)
    n = 120
    z = rng.normal(size=(n, 2))
    # View 0 is narrow (p=6 -> None), View 1 is wide with 8 focal features + 72 noise features
    w_narrow = np.array([[1.5, 0.0, 1.2, 0.0, 1.0, 0.0], [0.0, 1.5, 0.0, 1.2, 0.0, 1.0]])
    v_narrow = z @ w_narrow + 0.2 * rng.normal(size=(n, 6))
    w_wide = np.zeros((80, 2))
    w_wide[:4, 0] = 2.0
    w_wide[4:8, 1] = 2.0
    v_wide = z @ w_wide.T + rng.normal(size=(n, 80))

    res_lin = pysimlr.fit(
        "simlr", [v_narrow, v_wide], k=2, sparseness="auto", iterations=2, learning_rate=0.05, random_state=0
    )
    assert "sparseness_resolved" in res_lin
    q0, q1 = res_lin["sparseness_resolved"]
    assert q0 is None
    assert q1 is not None and 0.2 <= q1 <= 0.95

    # Direct simlr() call with mixed list ["auto", 0.5]
    res_direct = pysimlr.simlr(
        [v_narrow, v_wide], k=2, sparseness=[None, "auto"], iterations=2, learning_rate=0.05, random_state=0
    )
    assert res_direct["sparseness_resolved"][0] is None
    assert res_direct["sparseness_resolved"][1] == q1

    # Deep entry point via pysimlr.fit("lend", ..., sparseness="auto")
    res_deep = pysimlr.fit(
        "lend", [v_narrow, v_wide], k=2, sparseness="auto", iterations=2, batch_size=32, random_state=0
    )
    assert res_deep["sparseness_resolved"] == [q0, q1]


def test_uniform_shrinkage_alpha_mode_auto_jk_js_and_zero_post_shrink() -> None:
    rng = np.random.default_rng(222)
    z = rng.normal(size=(50, 1))
    views = [
        0.45 * z @ np.ones((1, 30)) + rng.normal(size=(50, 30)),
        0.45 * z @ np.ones((1, 35)) + rng.normal(size=(50, 35)),
    ]
    res_k1 = pysimlr.fit_uniform_shrunk(
        views, k=1, alpha_mode="auto", iterations=4, learning_rate=0.05, random_state=0
    )
    prov = res_k1["uniform_shrinkage_provenance"]
    assert prov["alpha_mode"] == "auto"
    assert all(0.0 <= a <= 1.0 for a in prov["alphas"])
    # PCA(1) positive view-sum orientation: corr(u, plain view mean) > 0
    u_np = res_k1["u"].detach().cpu().double().numpy()[:, 0]
    plain_mean = 0.5 * (views[0].mean(axis=1) + views[1].mean(axis=1))
    assert float(np.corrcoef(u_np, plain_mean)[0, 1]) > 0.5

    proj = pysimlr.project_uniform_shrunk(res_k1, views, post_shrink="auto")
    assert proj["post_shrink"] == 0.0

    # Two-stage k=2 + sparseness="auto" + alpha_mode="auto"
    z2 = rng.normal(size=(56, 2))
    views_k2 = []
    for p in (28, 32):
        w = np.zeros((p, 2))
        w[:6, 0] = 0.9
        w[6:12, 1] = 0.9
        views_k2.append(z2 @ w.T + 0.6 * rng.normal(size=(56, p)))
    res_k2 = pysimlr.fit_uniform_shrunk(
        views_k2, k=2, alpha_mode="auto", sparseness="auto", iterations=4, learning_rate=0.05, random_state=0
    )
    assert res_k2["uniform_shrinkage_provenance"]["stage_mode"] == "two_stage_support_restricted"
    proj_k2 = pysimlr.project_uniform_shrunk(res_k2, views_k2, post_shrink="auto")
    assert proj_k2["post_shrink"] == 0.0
    assert proj_k2["u"].shape == (56, 2)


def test_deep_projected_mode_ste_gradients_and_dense_vraw_init() -> None:
    rng = np.random.default_rng(333)
    x0 = torch.tensor(rng.normal(size=(24, 16)), dtype=torch.float32)
    x1 = torch.tensor(rng.normal(size=(24, 14)), dtype=torch.float32)

    for use_nsa in (True, False):
        enc0 = LENDNSAEncoder(
            16, 2, nsa_w=0.5, positivity="either", sparseness=0.7,
            use_nsa=use_nsa, first_layer_mode="projected",
        )
        enc1 = LENDNSAEncoder(
            14, 2, nsa_w=0.5, positivity="either", sparseness=0.7,
            use_nsa=use_nsa, first_layer_mode="projected",
        )
        initialize_deep_encoders([enc0, enc1], [x0, x1], k=2, initialization_type="pca")
        # v_raw starts dense (> 95% non-zero), whereas projected .v enforces sparseness=0.7
        assert float((enc0.v_raw.abs() > 1e-7).float().mean().item()) > 0.95
        assert float((enc0.v.abs() > 1e-7).float().mean().item()) < 0.50

        enc0.train()
        scores = enc0(x0)
        loss = (scores ** 2).sum()
        loss.backward()
        leaf_param = next(enc0.parameters())
        assert leaf_param.grad is not None
        # Straight-through estimator delivers gradient to 100% of first-layer leaf parameters
        assert int((leaf_param.grad.abs() > 0).sum().item()) == leaf_param.numel()


def test_path_graph_routing_in_recommend_and_fit_preset() -> None:
    rng = np.random.default_rng(444)
    z = rng.normal(size=(40, 1))
    views = {
        "v0": 0.5 * z @ np.ones((1, 8)) + rng.normal(size=(40, 8)),
        "v1": 0.5 * z @ np.ones((1, 10)) + rng.normal(size=(40, 10)),
        "v2": 0.5 * z @ np.ones((1, 9)) + rng.normal(size=(40, 9)),
    }
    graph = {"v0": ["v1"], "v1": ["v0", "v2"], "v2": ["v1"]}
    rep = pysimlr.recommend_presets(views, k=1, path_graph=graph)
    for rec in rep["recommendations"]:
        assert rec["params"]["topology"] == "graph"
        assert rec["params"]["path_graph"] == graph

    res_none = pysimlr.fit_preset(
        views, k=1, preset=None, path_graph=graph, iterations=2, learning_rate=0.05, random_state=0
    )
    assert res_none["preset_used"]["params"]["topology"] == "graph"


def test_jackknife_js_alpha_small_n_vs_large_n_monotone() -> None:
    rng = np.random.default_rng(555)
    # Non-uniform positive loadings so D^2 > 0; small n has much larger jackknife variance than large n
    load = np.linspace(0.2, 1.5, 40)
    z_s = rng.normal(size=24)
    x_s = np.outer(z_s, load) + 1.5 * rng.normal(size=(24, 40))
    t_s = z_s + 0.3 * rng.normal(size=24)

    z_l = rng.normal(size=600)
    x_l = np.outer(z_l, load) + 1.5 * rng.normal(size=(600, 40))
    t_l = z_l + 0.3 * rng.normal(size=600)

    a_small = pysimlr.jackknife_js_alpha(x_s, t_s, allow_sign_flip=False)
    a_large = pysimlr.jackknife_js_alpha(x_l, t_l, allow_sign_flip=False)
    assert 0.0 <= a_large < a_small <= 1.0


def test_jackknife_js_alpha_degenerate_and_tiny_guards() -> None:
    rng = np.random.default_rng(666)
    # p < 3 or n < 6 -> 0.0
    assert pysimlr.jackknife_js_alpha(rng.normal(size=(20, 2)), rng.normal(size=20)) == 0.0
    assert pysimlr.jackknife_js_alpha(rng.normal(size=(5, 10)), rng.normal(size=5)) == 0.0
    # Constant target -> 1.0
    assert pysimlr.jackknife_js_alpha(rng.normal(size=(20, 10)), np.ones(20)) == 1.0


def test_isolated_nans_and_tensor_inputs_handled_cleanly() -> None:
    rng = np.random.default_rng(777)
    a = rng.normal(size=(40, 12))
    b = rng.normal(size=(40, 14))
    a[2, 3] = np.nan
    b[7, 1] = np.nan
    prof = pysimlr.analyze_views([torch.tensor(a), torch.tensor(b)], k=2)
    assert prof["has_isolated_nans"] is True
    assert prof["has_whole_view_missing_rows"] is False
    assert prof["n_complete_rows"] == 40


def test_skewed_heavy_tailed_views_elevate_non_gaussianity() -> None:
    rng = np.random.default_rng(888)
    gauss_v = [rng.normal(size=(120, 16)), rng.normal(size=(120, 16))]
    skew_v = [np.exp(1.4 * rng.normal(size=(120, 16))), np.exp(1.4 * rng.normal(size=(120, 16)))]
    p_g = pysimlr.analyze_views(gauss_v, k=2)
    p_s = pysimlr.analyze_views(skew_v, k=2)
    assert p_s["mean_non_gaussianity"] > p_g["mean_non_gaussianity"] + 1.0


def test_precomputed_analysis_reused_without_reprofiling() -> None:
    rng = np.random.default_rng(999)
    views = [rng.normal(size=(36, 10)), rng.normal(size=(36, 12))]
    rep = pysimlr.recommend_presets(views, k=2, allow_nonlinear=False)
    res = pysimlr.fit_preset(
        views, k=2, preset="sparse_focal", analysis=rep, iterations=2, learning_rate=0.05, random_state=0
    )
    assert res["preset_used"]["name"] == "sparse_focal"
    assert res["preset_report"]["recommended"]["name"] == rep["recommended"]["name"]


def test_uniform_shrinkage_invalid_alpha_mode_raises() -> None:
    rng = np.random.default_rng(123)
    views = [rng.normal(size=(24, 10)), rng.normal(size=(24, 10))]
    with pytest.raises(ValueError, match="alpha_mode"):
        pysimlr.fit_uniform_shrunk(views, k=1, alpha_mode="unknown_mode", iterations=2)


def test_per_view_params_sparseness_auto_in_simlr() -> None:
    rng = np.random.default_rng(321)
    n = 110
    z = rng.normal(size=(n, 1))
    v0 = 1.5 * z @ np.ones((1, 8)) + 0.2 * rng.normal(size=(n, 8))
    w1 = np.zeros((70, 1))
    w1[:5, 0] = 2.0
    v1 = z @ w1.T + rng.normal(size=(n, 70))
    res = pysimlr.fit(
        "simlr",
        [v0, v1],
        k=1,
        sparseness=None,
        per_view_params={1: {"sparseness": "auto"}},
        iterations=2,
        learning_rate=0.05,
        random_state=0,
    )
    assert res["sparseness_resolved"][0] is None
    assert res["sparseness_resolved"][1] is not None and 0.2 <= res["sparseness_resolved"][1] <= 0.95


def test_recommend_presets_k1_monotone_skewed_views_detect_positive_flow_gain() -> None:
    rng = np.random.default_rng(77)
    n = 160
    z = rng.normal(size=(n, 1))
    w0 = rng.normal(size=(1, 10))
    w1 = rng.normal(size=(1, 10))
    # Monotone heavy-tailed warping exp(1.35 * z) degrades linear R^2 while normal-scores recover z
    s0 = np.exp(1.35 * z)
    s1 = np.exp(-1.25 * z)
    v0 = s0 @ w0 + 0.15 * rng.normal(size=(n, 10))
    v1 = s1 @ w1 + 0.15 * rng.normal(size=(n, 10))
    rep = pysimlr.recommend_presets([v0, v1], k=1, allow_nonlinear=True)
    assert rep["profile"]["nonlinearity"]["flow_press_gain"] > 0.05
    flow_entry = next(r for r in rep["recommendations"] if r["name"] == "nonlinear_flow")
    assert flow_entry["metrics"]["delta_vs_best_linear"] > 0.05

