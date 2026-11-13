"""Regression and contract tests for positive-sparse calibrated consensus.

Covers the two failure modes that previously put out-of-sample ``u`` in tension
with modality projections ``X_m V_m`` under ``positivity="positive"``:

1. Uncalibrated in-sample ``R^2 / (1 - R^2)`` weights in ``extension._affine_map``
   let ``p >> n`` views with ``dof >= n`` seize >95% of ``subset_consensus`` by
   interpolating training noise, silencing low-dimensional views.
2. Signed ``energy_type="nc"`` and step-0 positive clamping zeroed out
   non-negative bases ``V_m >= 0`` whose biological signal coupled negatively
   to the shared axis (e.g. regional hypoperfusion opposing hyperconnectivity).
"""
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.extension import consensus_maps


def _planted_block_weights(rng: np.random.Generator, k: int, p: int, active_per_col: int) -> np.ndarray:
    w = np.zeros((k, p), dtype=np.float64)
    for j in range(k):
        start = j * active_per_col
        w[j, start:start + active_per_col] = 0.7 + 0.6 * rng.random(active_per_col)
    return w


def _best_matched_f1(matrix: np.ndarray, planted_mask: np.ndarray, threshold: float = 1e-6) -> float:
    k = matrix.shape[1]
    sel = matrix > threshold
    scores = []
    for j in range(k):
        best = 0.0
        for c in range(k):
            tp = float((sel[:, j] & planted_mask[:, c]).sum())
            denom = float(sel[:, j].sum() + planted_mask[:, c].sum())
            if denom > 0:
                best = max(best, 2.0 * tp / denom)
        scores.append(best)
    return float(np.mean(scores))


def _best_matched_abs_r(u_pred: np.ndarray, z_true: np.ndarray) -> float:
    k = z_true.shape[1]
    vals = []
    for c in range(k):
        best = max(abs(float(np.corrcoef(u_pred[:, j], z_true[:, c])[0, 1])) for j in range(u_pred.shape[1]))
        vals.append(best)
    return float(np.mean(vals))


def test_nc_abs_recovers_anticorrelated_positive_view_and_coupling_signs():
    rng = np.random.default_rng(17)
    n, k = 180, 3
    z = rng.standard_normal((n, k))
    w0 = _planted_block_weights(rng, k, 36, 8)
    w1 = _planted_block_weights(rng, k, 36, 8)
    w2 = _planted_block_weights(rng, k, 36, 8)

    x0 = (z @ w0 + 0.35 * rng.standard_normal((n, 36))).astype(np.float32)
    x1 = (z @ w1 + 0.35 * rng.standard_normal((n, 36))).astype(np.float32)
    # View 2 couples negatively to +z with strictly non-negative parcel weights w2 >= 0.
    x2 = (-z @ w2 + 0.35 * rng.standard_normal((n, 36))).astype(np.float32)

    res_abs = pysimlr.fit(
        "simlr", [x0, x1, x2], k,
        positivity="positive", energy_type="nc_abs", sparseness=0.65, random_state=0,
    )
    res_nc = pysimlr.fit(
        "simlr", [x0, x1, x2], k,
        positivity="positive", energy_type="nc", sparseness=0.65, random_state=0,
    )

    v2 = res_abs["v"][2]
    l2 = res_abs["loadings"][2]
    assert bool((v2 >= 0).all()), "encoder V_2 must be strictly non-negative"
    assert bool((l2 >= 0).all()), "OLEE loadings[2] must remain strictly non-negative"
    assert bool(((l2 > 0).sum(dim=1) <= 1).all()), "OLEE loadings[2] must have disjoint column supports"

    # Views 0 and 1 set the positive orientation of u @ R, so View 2's coupling signs must be -1.
    assert torch.all(res_abs["coupling_signs"][0] > 0)
    assert torch.all(res_abs["coupling_signs"][1] > 0)
    assert torch.all(res_abs["coupling_signs"][2] < 0)

    planted2 = (w2.T > 0)
    # Unthresholded OLEE assigns positive noise correlations (~10 of 12 background features),
    # capping unthresholded F1 at ~0.82 even at 100% recall; check both unthresholded (>= 0.80)
    # and substantive correlation (>= 0.25) / sparse encoder V_2 support recovery (>= 0.90).
    f1_abs_raw = _best_matched_f1(l2.numpy(), planted2, threshold=1e-6)
    f1_nc_raw = _best_matched_f1(res_nc["loadings"][2].numpy(), planted2, threshold=1e-6)
    f1_abs_strong = _best_matched_f1(l2.numpy(), planted2, threshold=0.25)
    # Encoder V_2 lives in the unrotated SVD basis of U; its rotated latent scores X_2 V_2 R
    # must recover all k planted factors with |r| >= 0.90 while OLEE loadings recover >= 0.95 F1.
    z2_rot = (res_abs["latents"][2].double() @ res_abs["loading_rotation"].double()).numpy()
    z2_nc_rot = (res_nc["latents"][2].double() @ res_nc["loading_rotation"].double()).numpy()
    r_z2_abs = _best_matched_abs_r(z2_rot, z)
    r_z2_nc = _best_matched_abs_r(z2_nc_rot, z)
    f1_nc_strong = _best_matched_f1(res_nc["loadings"][2].numpy(), planted2, threshold=0.25)
    assert f1_abs_raw >= 0.80 and f1_abs_raw > f1_nc_raw + 0.25
    assert f1_abs_strong >= 0.95 and f1_nc_strong <= 0.15, (
        f"expected >=0.95 substantive OLEE F1 ({f1_abs_strong:.3f}) vs <=0.15 for signed nc ({f1_nc_strong:.3f})"
    )
    assert r_z2_abs >= 0.90 and r_z2_abs > r_z2_nc + 0.30, (
        f"expected X_2 V_2 latent recovery >=0.90 ({r_z2_abs:.3f}) beating signed nc ({r_z2_nc:.3f})"
    )


@pytest.mark.parametrize("sign_alignment", ["greedy", "spectral"])
def test_calibrated_subset_consensus_prevents_high_p_overfitting_takeover(sign_alignment):
    # The ">0.90 uncalibrated share" precondition below was measured with the greedy sign rule, under which
    # this seed's view 2 is fitted badly (in-sample R^2 0.79 against 0.98 for views 0 and 1) because of an
    # order-dependent sign choice among its k=3 columns. With the default spectral rule the same seed fits
    # all three views at R^2 >= 0.99 and the uncalibrated share is 0.38 (seeds 23-28: greedy 0.20-0.96,
    # spectral 0.25-0.43). The precondition and the margin over the uncalibrated consensus (both need the
    # takeover) are therefore asserted for "greedy" only, which is the fixture they describe; for both rules
    # the calibrated high-p share must stay < 0.25 and the held-out recovery >= 0.65.

    rng = np.random.default_rng(23)
    n_tr, n_te, k = 80, 140, 3
    n = n_tr + n_te
    z = rng.standard_normal((n, k))

    # Views 0 and 1 are high-dimensional (p >> n_tr) and heavily noise-dominated;
    # View 2 is low-dimensional (p = 36 < n_tr) with stronger per-feature signal-to-noise.
    w0 = np.zeros((k, 600), dtype=np.float64)
    w1 = np.zeros((k, 720), dtype=np.float64)
    w2 = np.zeros((k, 36), dtype=np.float64)
    for j in range(k):
        w0[j, j * 20:(j + 1) * 20] = 0.35
        w1[j, j * 20:(j + 1) * 20] = 0.35
        w2[j, j * 8:(j + 1) * 8] = 0.85

    x0 = (z @ w0 + 1.60 * rng.standard_normal((n, 600))).astype(np.float32)
    x1 = (z @ w1 + 1.60 * rng.standard_normal((n, 720))).astype(np.float32)
    x2 = (z @ w2 + 0.95 * rng.standard_normal((n, 36))).astype(np.float32)

    tr, te = slice(0, n_tr), slice(n_tr, None)
    res = pysimlr.fit(
        "simlr", [x0[tr], x1[tr], x2[tr]], k,
        positivity="positive", energy_type="nc", sparseness=0.5, random_state=0,
        sign_alignment=sign_alignment,
    )

    maps = consensus_maps(res)
    # Uncalibrated in-sample R^2 on p >> n views reaches ~0.98 (>95% of precision share),
    # whereas DOF-calibrated weights curb p >> n self-fitting (<20% share).
    uncal_w = [maps[i]["r2"] / max(1.0 - maps[i]["r2"], 1e-6) for i in range(3)]
    uncal_high_p_share = (uncal_w[0] + uncal_w[1]) / sum(uncal_w)
    cal_high_p_share = (maps[0]["weight"] + maps[1]["weight"]) / sum(m["weight"] for m in maps)
    if sign_alignment == "greedy":
        assert uncal_high_p_share > 0.90, f"expected p>>n overfitting in uncalibrated R^2, got {uncal_high_p_share:.3f}"
    assert cal_high_p_share < 0.25, (
        f"calibrated high-p share ({cal_high_p_share:.3f}) did not curb overfitting ({uncal_high_p_share:.3f})"
    )

    pred_sub = pysimlr.predict(res, {"view0": x0[te], "view1": x1[te], "view2": x2[te]})
    r_cal = _best_matched_abs_r(pred_sub["u"].numpy(), z[te])

    z_te_latents = [pred_sub["latents"][f"view{i}"].double() for i in range(3)]
    u_uncal = sum((z_te_latents[i] @ maps[i]["B"] + maps[i]["b"]) * uncal_w[i] for i in range(3))
    u_uncal = (u_uncal / sum(uncal_w)).numpy()
    r_uncal = _best_matched_abs_r(u_uncal, z[te])

    assert r_cal >= 0.65, f"expected strong held-out consensus recovery, got {r_cal:.3f}"
    if sign_alignment == "greedy":
        # Needs the overfitting takeover that only the greedy fit of this seed produces (see above); under
        # spectral the uncalibrated weights do not take over and r_uncal is high as well.
        assert r_cal > r_uncal + 0.25, f"calibrated u ({r_cal:.3f}) should decisively beat uncalibrated u ({r_uncal:.3f})"


def test_anchored_extend_recovers_anticorrelated_positive_view_and_small_n_oof_weights():
    rng = np.random.default_rng(31)
    n_fit, n_bridge, n_te, k = 150, 30, 80, 3
    n = n_fit + n_bridge + n_te
    z = rng.standard_normal((n, k))

    w0 = _planted_block_weights(rng, k, 30, 7)
    w1 = _planted_block_weights(rng, k, 30, 7)
    w_asl = _planted_block_weights(rng, k, 27, 6)

    x0 = (z @ w0 + 0.35 * rng.standard_normal((n, 30))).astype(np.float32)
    x1 = (z @ w1 + 0.35 * rng.standard_normal((n, 30))).astype(np.float32)
    x_asl = (-z @ w_asl + 0.35 * rng.standard_normal((n, 27))).astype(np.float32)
    x_noise = rng.standard_normal((n, 45)).astype(np.float32)

    s_fit = slice(0, n_fit)
    s_br = slice(n_fit, n_fit + n_bridge)
    s_te = slice(n_fit + n_bridge, None)

    res = pysimlr.fit(
        "simlr", [x0[s_fit], x1[s_fit]], k,
        positivity="positive", energy_type="nc_abs", sparseness=0.65, random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ext = pysimlr.extend(
            res,
            [x0[s_br], x1[s_br]],
            {"ASL_neg": x_asl[s_br], "Noise": x_noise[s_br]},
            random_state=0,
        )
    assert any("does not align" in str(w.message) for w in caught)

    e_asl = ext["extensions"]["ASL_neg"]
    e_noise = ext["extensions"]["Noise"]
    assert e_asl["holdout"]["n"] == n_bridge
    assert min(e_asl["holdout"]["coupling"]) > e_asl["holdout"]["null_bar"]
    assert torch.all(e_asl["coupling_signs"] < 0)

    asl_idx = ext["view_names"].index("view1") + 1
    l_asl = ext["loadings"][asl_idx]
    assert bool((l_asl >= 0).all()) and bool(((l_asl > 0).sum(dim=1) <= 1).all())
    assert _best_matched_f1(l_asl.numpy(), w_asl.T > 0, threshold=0.25) >= 0.90

    # OOF calibration assigns strong precision weight to ASL_neg and near-zero weight to pure Noise.
    assert e_asl["map"]["weight"] > 0.5
    assert e_noise["map"]["weight"] < 0.08 * e_asl["map"]["weight"]

    u_asl_only = pysimlr.predict(ext, {"ASL_neg": x_asl[s_te]})["u"].numpy()
    u_asl_plus_noise = pysimlr.predict(ext, {"ASL_neg": x_asl[s_te], "Noise": x_noise[s_te]})["u"].numpy()
    r_only = _best_matched_abs_r(u_asl_only, z[s_te])
    r_both = _best_matched_abs_r(u_asl_plus_noise, z[s_te])
    assert r_only >= 0.75
    assert r_both >= r_only - 0.03


def test_signed_nc_with_positive_emits_one_time_warning_recommending_nc_abs():
    from pysimlr.similarity import _reset_nc_positive_warning
    from pysimlr.consensus import mai_dof_adjust
    from pysimlr.extension import _affine_map

    rng = np.random.default_rng(7)
    x0 = rng.standard_normal((50, 16)).astype(np.float32)
    x1 = rng.standard_normal((50, 16)).astype(np.float32)

    _reset_nc_positive_warning()
    with warnings.catch_warnings(record=True) as w_first:
        warnings.simplefilter("always")
        pysimlr.fit("simlr", [x0, x1], 2, positivity="positive", energy_type="nc", iterations=6, random_state=0)
    nc_warns_first = [w for w in w_first if issubclass(w.category, UserWarning) and "nc_abs" in str(w.message)]
    assert len(nc_warns_first) == 1

    # Second call in the same process must not re-emit the warning.
    with warnings.catch_warnings(record=True) as w_second:
        warnings.simplefilter("always")
        pysimlr.fit("simlr", [x0, x1], 2, positivity="positive", energy_type="nc", iterations=6, random_state=1)
    nc_warns_second = [w for w in w_second if issubclass(w.category, UserWarning) and "nc_abs" in str(w.message)]
    assert len(nc_warns_second) == 0

    # Resetting and fitting with energy_type="nc_abs" or positivity="either" emits zero warnings.
    _reset_nc_positive_warning()
    with warnings.catch_warnings(record=True) as w_clean:
        warnings.simplefilter("always")
        pysimlr.fit("simlr", [x0, x1], 2, positivity="positive", energy_type="nc_abs", iterations=6, random_state=0)
        pysimlr.fit("simlr", [x0, x1], 2, positivity="either", energy_type="nc", iterations=6, random_state=0)
    nc_warns_clean = [w for w in w_clean if issubclass(w.category, UserWarning) and "nc_abs" in str(w.message)]
    assert len(nc_warns_clean) == 0

    # Verify _affine_map per-component r2_per_component matches mai_dof_adjust on assessable views.
    u_t = torch.from_numpy(rng.standard_normal((100, 3)))
    z_t = u_t * torch.tensor([[1.0, 0.5, 0.0]]) + 0.6 * torch.from_numpy(rng.standard_normal((100, 3)))
    mp = _affine_map(z_t, u_t, dof=25.0)
    zc = z_t.double() - z_t.double().mean(0, keepdim=True)
    uc = u_t.double() - u_t.double().mean(0, keepdim=True)
    ss_tot = (uc ** 2).sum(0)
    ss_res = ((uc - zc @ mp["B"]) ** 2).sum(0)
    r2_raw = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0 - 1e-6)
    expected = [mai_dof_adjust(float(r2_raw[j].item()), "procrustes_r2", 100, 25.0) for j in range(3)]
    assert np.allclose(mp["r2_per_component"], [min(e, 1.0 - 1e-4) for e in expected], atol=1e-10)

