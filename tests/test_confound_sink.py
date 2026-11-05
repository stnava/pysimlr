"""Tests for the first-class multi-view confound sink (`simlr(..., confounds=...)`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import pysimlr
from pysimlr.api import fit, predict, validate_result
from pysimlr.attribution import component_attribution
from pysimlr.confounds import max_abs_confound_corr
from pysimlr.loadings import loading_inference, select_rank
from pysimlr.relevance import feature_relevance
from pysimlr.simlr import predict_simlr, simlr
from pysimlr.utils import heldout_coupling


def _pearson_abs(a: np.ndarray, b: np.ndarray) -> float:
    a1 = np.asarray(a, dtype=np.float64).ravel()
    b1 = np.asarray(b, dtype=np.float64).ravel()
    a1 = a1 - a1.mean()
    b1 = b1 - b1.mean()
    den = float(np.linalg.norm(a1) * np.linalg.norm(b1))
    if den <= 1e-15:
        return 0.0
    return float(abs(np.dot(a1, b1)) / den)


def _make_dipole_confounded_benchmark(
    n_train: int = 120,
    n_test: int = 60,
    dims: tuple[int, ...] = (40, 35, 30),
    confound_var_ratio: float = 3.5,
    seed: int = 2026,
):
    """3-view benchmark with a signed dipole confound >3x stronger than non-negative biology."""
    rng = np.random.default_rng(seed)
    n_tot = n_train + n_test

    # Two shared biological factors (uncorrelated with the scanner/motion confound).
    z_bio = rng.standard_normal((n_tot, 2))
    z_bio = (z_bio - z_bio.mean(axis=0, keepdims=True)) / z_bio.std(axis=0, ddof=1, keepdims=True)
    # Emphasize factor 0 slightly so component 1 ordering is unambiguous once clean.
    bio_scales = np.array([1.25, 0.85], dtype=np.float64)

    # Signed dipole nuisance covariate (e.g. head motion FD / scanner drift).
    c = rng.standard_normal((n_tot, 1))
    # Orthogonalize c against z_bio on the full generating population so ground truth is clean.
    c = c - z_bio @ (np.linalg.lstsq(z_bio, c, rcond=None)[0])
    c = (c - c.mean(axis=0, keepdims=True)) / c.std(axis=0, ddof=1, keepdims=True)

    train_views, test_views = [], []
    for p in dims:
        # Disjoint non-negative regional supports for biology:
        w_bio = np.zeros((p, 2), dtype=np.float64)
        s0 = max(4, p // 5)
        s1 = max(4, p // 5)
        w_bio[:s0, 0] = 0.6 + 0.4 * rng.random(s0)
        w_bio[s0:s0 + s1, 1] = 0.6 + 0.4 * rng.random(s1)
        w_bio /= np.linalg.norm(w_bio, axis=0, keepdims=True)

        # Signed dipole confound pattern (half positive, half negative, overlapping bio supports):
        w_conf = np.zeros((p, 1), dtype=np.float64)
        half = p // 2
        w_conf[:half, 0] = +(0.5 + 0.5 * rng.random(half))
        w_conf[half:, 0] = -(0.5 + 0.5 * rng.random(p - half))
        w_conf /= np.linalg.norm(w_conf, axis=0, keepdims=True)

        sig_bio = (z_bio * bio_scales) @ w_bio.T
        bio_var = float(np.var(sig_bio))
        sig_conf_unit = c @ w_conf.T
        conf_amp = np.sqrt(confound_var_ratio * bio_var / float(np.var(sig_conf_unit)))
        sig_conf = conf_amp * sig_conf_unit
        noise = 0.08 * np.sqrt(bio_var) * rng.standard_normal((n_tot, p))
        x = (sig_bio + sig_conf + noise).astype(np.float32)

        train_views.append(x[:n_train])
        test_views.append(x[n_train:])

    return (
        train_views,
        test_views,
        z_bio[:n_train],
        z_bio[n_train:],
        c[:n_train],
        c[n_train:],
    )


def test_confounds_none_preserves_default_contract():
    train_views, test_views, *_ = _make_dipole_confounded_benchmark(n_train=48, n_test=16, dims=(18, 15, 12))
    res = fit("simlr", train_views, k=2, iterations=12, random_state=7)
    validate_result(res, n=48, k=2, n_views=3)
    assert res["confound_sink"] is None
    pred = predict(res, test_views)
    assert "u_sink" not in pred
    assert "latents_sink" not in pred


def test_confound_parameter_validation_errors():
    views, *_ = _make_dipole_confounded_benchmark(n_train=36, n_test=8, dims=(12, 10))
    c = np.linspace(-1.0, 1.0, 36)

    with pytest.raises(ValueError, match="confound_strength"):
        simlr(views, k=2, iterations=3, confounds=c, confound_strength=-0.1)
    with pytest.raises(ValueError, match="confound_strength"):
        simlr(views, k=2, iterations=3, confounds=c, confound_strength=1.2)
    with pytest.raises(ValueError, match="confound_strength"):
        simlr(views, k=2, iterations=3, confounds=c, confound_strength=True)
    with pytest.raises(ValueError, match="confound_rank"):
        simlr(views, k=2, iterations=3, confounds=c, confound_rank=0)
    with pytest.raises(ValueError, match="confound_rank"):
        simlr(views, k=2, iterations=3, confounds=c, confound_rank=False)
    with pytest.raises(ValueError, match="confound_partial_out"):
        simlr(views, k=2, iterations=3, confounds=c, confound_partial_out=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rows"):
        simlr(views, k=2, iterations=3, confounds=np.ones((10, 2)))
    with pytest.raises(ValueError, match="entries"):
        simlr(views, k=2, iterations=3, confounds=[c, c, c])
    with pytest.raises(ValueError, match="empty"):
        simlr(views, k=2, iterations=3, confounds={})
    with pytest.raises(ValueError, match="only None"):
        simlr(views, k=2, iterations=3, confounds=[None, None])
    with pytest.raises(ValueError, match="unknown view key"):
        simlr(views, k=2, iterations=3, confounds={"view9": c})


def test_input_formats_nan_handling_and_per_view_confounds():
    rng = np.random.default_rng(11)
    views, test_views, _, _, c_tr, _ = _make_dipole_confounded_benchmark(
        n_train=60, n_test=15, dims=(20, 18, 16), seed=11
    )
    # 1. DataFrame with NaNs + constant column (dropped cleanly)
    df_conf = pd.DataFrame({
        "fd": c_tr[:, 0].copy(),
        "site_const": np.ones(60),
        "tsnr": 0.5 * c_tr[:, 0] + 0.5 * rng.standard_normal(60),
    })
    df_conf.iloc[[2, 17, 41], 0] = np.nan
    df_conf.iloc[[5, 29], 2] = np.nan

    res_df = fit("simlr", views, k=2, iterations=20, positivity="positive",
                 confounds=df_conf, confound_strength=1.0, random_state=3)
    validate_result(res_df, n=60, k=2, n_views=3)
    cs = res_df["confound_sink"]
    assert cs is not None
    assert cs["confound_rank"] == 2
    assert torch.isfinite(res_df["u"]).all()
    assert cs["max_abs_corr_after"] < 1e-5

    # 2. 1-D Series
    s_conf = pd.Series(c_tr[:, 0], name="motion_fd")
    res_s = fit("simlr", views, k=2, iterations=15, confounds=s_conf, random_state=3)
    assert res_s["confound_sink"]["confound_rank"] == 1
    assert res_s["confound_sink"]["max_abs_corr_after"] < 1e-5

    # 3. Per-view list [C_0, None, C_2] and dict {"view0": C_0, 2: C_2}
    c0 = c_tr
    c2 = rng.standard_normal((60, 2))
    res_list = fit("simlr", views, k=2, iterations=20, positivity="positive",
                   confounds=[c0, None, c2], random_state=5)
    sink_l = res_list["confound_sink"]
    assert sink_l["v_sink"][0].shape == (20, 1)
    assert sink_l["v_sink"][1].shape == (18, 0)
    assert sink_l["v_sink"][2].shape == (16, 2)
    assert sink_l["partial_weights"][1].shape == (0, 2)
    assert sink_l["max_abs_corr_after"] < 1e-5

    res_dict = fit("simlr", views, k=2, iterations=20, positivity="positive",
                   confounds={"view0": c0, 2: c2}, random_state=5)
    assert torch.allclose(res_list["u"], res_dict["u"], atol=1e-5)

    pred_l = predict(res_list, test_views)
    assert pred_l["u_sink"].shape == (15, sink_l["confound_rank"])
    assert pred_l["latents_sink"][0].shape == (15, 1)
    assert pred_l["latents_sink"][1].shape == (15, 0)
    assert pred_l["latents_sink"][2].shape == (15, 2)


@pytest.mark.parametrize("topology", ["star", "loo", "graph"])
@pytest.mark.parametrize("mixing", ["svd", "pca", "avg", "newton"])
def test_exact_orthogonality_across_topologies_and_mixings(topology: str, mixing: str):
    views, _, _, _, c_tr, _ = _make_dipole_confounded_benchmark(
        n_train=56, n_test=10, dims=(18, 16, 14), seed=19
    )
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]} if topology == "graph" else None
    res = fit(
        "simlr",
        views,
        k=2,
        iterations=25,
        positivity="positive",
        topology=topology,
        path_graph=graph,
        mixing_algorithm=mixing,
        confounds=c_tr,
        confound_strength=1.0,
        random_state=19,
    )
    # Strict non-negativity preserved on biological basis V:
    for v in res["v"]:
        assert int((v < 0).sum().item()) == 0
    # Sink encoder is signed (captures dipole with both signs):
    for vs in res["confound_sink"]["v_sink"]:
        assert bool((vs > 0).any()) and bool((vs < 0).any())
    # Exact orthogonality of consensus U to C:
    corr_after = max_abs_confound_corr(res["u"], torch.as_tensor(c_tr, dtype=torch.float64))
    assert corr_after < 1e-5
    assert res["confound_sink"]["max_abs_corr_after"] < 1e-5
    assert res["confound_sink"]["max_abs_corr_before"] > 0.5


def test_soft_confound_strength_monotonic_interpolation():
    views, _, _, _, c_tr, _ = _make_dipole_confounded_benchmark(
        n_train=80, n_test=20, dims=(24, 20, 18), confound_var_ratio=3.0, seed=31
    )
    c_t = torch.as_tensor(c_tr, dtype=torch.float64)
    corrs = []
    for gamma in (0.0, 0.5, 1.0):
        res = fit(
            "simlr",
            views,
            k=2,
            iterations=35,
            positivity="positive",
            confounds=c_tr,
            confound_strength=gamma,
            random_state=31,
        )
        corrs.append(max_abs_confound_corr(res["u"], c_t))
    assert corrs[0] > 0.65
    assert corrs[0] > corrs[1] > corrs[2]
    assert corrs[1] > 0.05
    assert corrs[2] < 1e-5


def test_high_confound_recovery_zero_ctest_n1_invariance_and_missing_views():
    train_views, test_views, _z_bio_tr, z_bio_te, c_tr, c_te = _make_dipole_confounded_benchmark(
        n_train=120, n_test=60, dims=(40, 35, 30), confound_var_ratio=3.5, seed=2026
    )

    # (A) Rank k=1 primary axis C1: unadjusted non-negative SiMLR is hijacked by the 3.5x dipole,
    # whereas confound-sink SiMLR places true biology directly on C1 without C_test.
    res_unadj_k1 = fit(
        "simlr", train_views, k=1, iterations=50, positivity="positive",
        energy_type="nc_abs", sparseness=0.5, random_state=42
    )
    pred_unadj_k1 = predict(res_unadj_k1, test_views)
    assert _pearson_abs(res_unadj_k1["u"][:, 0].numpy(), c_tr[:, 0]) > 0.85
    assert _pearson_abs(pred_unadj_k1["u"][:, 0].numpy(), z_bio_te[:, 0]) < 0.25

    res_sink_k1 = fit(
        "simlr", train_views, k=1, iterations=50, positivity="positive",
        energy_type="nc_abs", sparseness=0.5, confounds=c_tr, confound_strength=1.0, random_state=42
    )
    pred_sink_k1 = predict(res_sink_k1, test_views)
    assert _pearson_abs(pred_sink_k1["u"][:, 0].numpy(), z_bio_te[:, 0]) > 0.90
    assert _pearson_abs(pred_sink_k1["u"][:, 0].numpy(), c_te[:, 0]) < 0.20

    # (B) Rank k=2 multi-component recovery: unadjusted loses the 2nd biological factor to the confound,
    # whereas confound-sink SiMLR recovers both biological factors (>0.90) on held-out test subjects!
    res_unadj = fit(
        "simlr", train_views, k=2, iterations=60, positivity="positive",
        energy_type="nc_abs", sparseness=0.5, random_state=42
    )
    assert max(_pearson_abs(res_unadj["u"][:, j].numpy(), c_tr[:, 0]) for j in range(2)) > 0.70

    res_sink = fit(
        "simlr", train_views, k=2, iterations=60, positivity="positive",
        energy_type="nc_abs", sparseness=0.5, confounds=c_tr, confound_strength=1.0, random_state=42
    )
    for v in res_sink["v"]:
        assert int((v < 0).sum().item()) == 0
    assert res_sink["confound_sink"]["max_abs_corr_after"] < 1e-5

    pred_batch = predict(res_sink, test_views)
    assert "u_sink" in pred_batch and "latents_sink" in pred_batch
    bio0_rec = max(_pearson_abs(pred_batch["u"][:, j].numpy(), z_bio_te[:, 0]) for j in range(2))
    bio1_rec = max(_pearson_abs(pred_batch["u"][:, j].numpy(), z_bio_te[:, 1]) for j in range(2))
    sink_conf_te = max(_pearson_abs(pred_batch["u"][:, j].numpy(), c_te[:, 0]) for j in range(2))
    captured_conf_te = _pearson_abs(pred_batch["u_sink"][:, 0].numpy(), c_te[:, 0])
    assert bio0_rec > 0.90, f"expected held-out bio0 |r| > 0.90, got {bio0_rec:.3f}"
    assert bio1_rec > 0.90, f"expected held-out bio1 |r| > 0.90, got {bio1_rec:.3f}"
    assert sink_conf_te < 0.20, f"expected held-out U confound |r| < 0.20, got {sink_conf_te:.3f}"
    assert captured_conf_te > 0.90, f"expected u_sink to capture held-out confound (>0.90), got {captured_conf_te:.3f}"

    # Direct `predict_simlr` contract check:
    direct = predict_simlr(test_views, res_sink)
    assert torch.allclose(direct["u"], pred_batch["u"], atol=1e-6)
    assert torch.allclose(direct["u_sink"], pred_batch["u_sink"], atol=1e-6)

    # Exact N=1 single-subject batch invariance across all outputs:
    u_rows, usink_rows = [], []
    lat_rows = [[] for _ in test_views]
    lsink_rows = [[] for _ in test_views]
    for idx in range(12):
        one = [v[idx:idx + 1] for v in test_views]
        p1 = predict(res_sink, one)
        u_rows.append(p1["u"])
        usink_rows.append(p1["u_sink"])
        for m in range(len(test_views)):
            lat_rows[m].append(p1["latents"][m])
            lsink_rows[m].append(p1["latents_sink"][m])
    assert float((torch.cat(u_rows, dim=0) - pred_batch["u"][:12]).abs().max().item()) < 1e-5
    assert float((torch.cat(usink_rows, dim=0) - pred_batch["u_sink"][:12]).abs().max().item()) < 1e-5
    for m in range(len(test_views)):
        assert float((torch.cat(lat_rows[m], dim=0) - pred_batch["latents"][m][:12]).abs().max().item()) < 1e-5
        assert float((torch.cat(lsink_rows[m], dim=0) - pred_batch["latents_sink"][m][:12]).abs().max().item()) < 1e-5

    # Missing-view subset prediction (`subset_consensus`) without C_test, plus N=1 invariance:
    sub_batch = predict(res_sink, {0: test_views[0], 1: test_views[1]})
    assert set(sub_batch["latents"].keys()) == {"view0", "view1"}
    assert set(sub_batch["latents_sink"].keys()) == {"view0", "view1"}
    assert sub_batch["u_sink"].shape == (60, 1)
    sub_bio_te = max(
        _pearson_abs(sub_batch["u"][:, 0].numpy(), z_bio_te[:, 0]),
        _pearson_abs(sub_batch["u"][:, 0].numpy(), z_bio_te[:, 1]),
    )
    assert sub_bio_te > 0.78
    sub_one = predict(res_sink, {0: test_views[0][:1], 1: test_views[1][:1]})
    assert float((sub_one["u"] - sub_batch["u"][:1]).abs().max().item()) < 1e-5
    assert float((sub_one["u_sink"] - sub_batch["u_sink"][:1]).abs().max().item()) < 1e-5


def test_downstream_inference_select_rank_relevance_and_attribution():
    train_views, test_views, z_bio_tr, z_bio_te, c_tr, _ = _make_dipole_confounded_benchmark(
        n_train=64, n_test=24, dims=(18, 16, 14), seed=77
    )
    df_c = pd.DataFrame({"fd": c_tr[:, 0]})

    # 1. loading_inference with shared DataFrame confounds and per-view confounds
    inf = loading_inference(
        "simlr",
        train_views,
        k=2,
        iterations=15,
        positivity="positive",
        confounds=df_c,
        n_subsamples=4,
        n_permutations=2,
        n_jobs=1,
        random_state=77,
    )
    assert inf["result"]["confound_sink"] is not None
    assert inf["result"]["confound_sink"]["max_abs_corr_after"] < 1e-5
    assert inf["oos_coupling"] is not None and len(inf["oos_coupling"]) == 2
    assert np.isfinite(inf["null_max_abs"])

    inf_pv = loading_inference(
        "simlr",
        train_views,
        k=2,
        iterations=12,
        confounds=[c_tr, None, c_tr],
        n_subsamples=3,
        n_permutations=2,
        n_jobs=1,
        random_state=77,
    )
    assert inf_pv["result"]["confound_sink"]["v_sink"][1].shape[1] == 0

    # 2. select_rank + heldout_coupling
    sr = select_rank(
        "simlr",
        train_views,
        ks=[1, 2],
        iterations=12,
        positivity="positive",
        confounds=c_tr,
        n_subsamples=4,
        n_jobs=1,
        random_state=77,
    )
    assert set(sr["criteria"].keys()) == {"component", "subspace", "oos_r2", "oos_r2_1se", "oos_coupling"}
    hc = heldout_coupling(inf["result"], test_views)
    assert len(hc["cross_view_heldout"]) == 2

    # 3. feature_relevance + component_attribution subsample/null refit slicing
    rel = feature_relevance(
        inf["result"],
        train_views,
        test_views,
        outcome=z_bio_tr[:, 0],
        outcome_test=z_bio_te[:, 0],
        n_permutations=200,
        n_permutations_test=200,
        n_refits=2,
        refit_permutations=100,
        n_null_refits=1,
        n_jobs=1,
        random_state=77,
    )
    assert len(rel["features"]) == sum(v.shape[1] for v in train_views)

    attr = component_attribution(
        inf["result"],
        train_views,
        test_views,
        outcome=z_bio_tr[:, 0],
        outcome_test=z_bio_te[:, 0],
        n_refits=2,
        n_jobs=1,
        random_state=77,
    )
    assert len(attr["refits"]["default"]) == 2
    assert hasattr(pysimlr, "fit")
