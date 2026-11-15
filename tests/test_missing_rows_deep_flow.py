"""Tests for `missing_rows='mask'` across deep and normalizing-flow SiMLR models."""

import math
import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.flows import flow_simr


DEEP_FLOW_API_MODELS = ("lend", "ned", "ned_shared_private", "flow_v")


def _planted_missing_views(n: int = 72, seed: int = 11):
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    views = []
    for p, s1, s2 in ((12, 2.2, 0.5), (10, 0.4, 2.1), (14, 1.8, -1.6)):
        x = 0.35 * rng.standard_normal((n, p))
        x[:, :4] += s1 * z1[:, None]
        x[:, 4:8] += s2 * z2[:, None]
        views.append(torch.tensor(x, dtype=torch.float32))
    masked = [v.clone() for v in views]
    if n >= 45:
        # Block + scattered unobserved rows; every subject observed in >= 1 view
        masked[0][0:10] = float("nan")
        masked[1][12:22] = float("nan")
        masked[2][25:33] = float("nan")
        masked[0][40] = float("nan")
        masked[2][40] = float("nan")  # row 40 observed only in view 1
    else:
        masked[0][0:4] = float("nan")
        masked[1][5:9] = float("nan")
    return views, masked, np.column_stack([z1, z2])


@pytest.mark.parametrize("method", DEEP_FLOW_API_MODELS)
def test_deep_flow_missing_rows_mask_contract_and_downstream(method):
    full_views, masked_views, true_z = _planted_missing_views()
    res = pysimlr.fit(
        method,
        masked_views,
        k=2,
        missing_rows="mask",
        iterations=35,
        batch_size=24,
        random_state=7,
    )
    assert res["missing_rows"] == "mask"
    assert res["parameters"]["used"]["missing_rows"] == "mask"
    assert res["parameters"]["ignored"] == {}
    assert res["obs_mask"] is not None and len(res["obs_mask"]) == 3

    for m_idx, (v_in, lat, mask_list) in enumerate(zip(masked_views, res["latents"], res["obs_mask"])):
        expected_obs = ~torch.isnan(v_in).all(dim=1)
        assert torch.equal(torch.tensor(mask_list, dtype=torch.bool), expected_obs)
        assert bool(torch.isfinite(lat[expected_obs]).all()), f"{method} view {m_idx} observed latents non-finite"
        assert bool(torch.isnan(lat[~expected_obs]).all()), f"{method} view {m_idx} unobserved latents not NaN"

    assert bool(torch.isfinite(res["u"]).all())
    assert all(math.isfinite(float(e)) for e in res["errors"])
    gm = res["convergence"]["grad_map"]
    assert gm is not None and math.isfinite(float(gm))
    pysimlr.validate_result(res, n=full_views[0].shape[0], k=2, n_views=3)

    # Planted signal recovery on consensus U
    u_np = res["u"].detach().cpu().double().numpy()
    proj = u_np @ np.linalg.lstsq(u_np, true_z, rcond=None)[0]
    r2_signal = 1.0 - np.sum((true_z - proj) ** 2) / np.sum((true_z - true_z.mean(0)) ** 2)
    assert r2_signal > 0.55, f"{method} signal R2 too low under missing_rows='mask': {r2_signal:.3f}"

    # Out-of-sample predict with fresh unobserved rows + subset views
    te_views = [v.clone()[:24] for v in full_views]
    te_views[0][2:6] = float("nan")
    te_views[2][8:11] = float("nan")
    pred = pysimlr.predict(res, te_views)
    assert bool(torch.isfinite(torch.as_tensor(pred["u"])).all())
    assert bool(torch.isnan(torch.as_tensor(pred["latents"][0])[2:6]).all())
    assert bool(torch.isfinite(torch.as_tensor(pred["latents"][0])[6:]).all())
    assert all(math.isfinite(float(x)) for x in pred["r2"])
    assert all(math.isfinite(float(x)) for x in pred["mse"])

    sub_pred = pysimlr.predict(res, [te_views[0], None, te_views[2]])
    assert bool(torch.isfinite(torch.as_tensor(sub_pred["u"])).all())

    # Downstream loadings and feature_maps
    ld = pysimlr.loadings(res, masked_views)
    for s_mat in ld["structure"]:
        assert bool(torch.isfinite(torch.as_tensor(s_mat)).all())
    fmap = pysimlr.feature_maps(res, masked_views, target="other_views")
    assert "structure" in fmap and len(fmap["structure"]) == 3
    for fm in fmap["structure"]:
        assert bool(torch.isfinite(torch.as_tensor(fm)).all())


@pytest.mark.parametrize("method", ("lend", "flow_v"))
@pytest.mark.parametrize("energy_type", ("align", "nc"))
def test_align_and_nc_compose_with_missing_rows_mask_and_loo(method, energy_type):
    _, masked_views, _ = _planted_missing_views(n=64, seed=19)
    res = pysimlr.fit(
        method,
        masked_views,
        k=2,
        energy_type=energy_type,
        topology="loo",
        missing_rows="mask",
        iterations=25,
        batch_size=20,
        random_state=3,
    )
    assert res["missing_rows"] == "mask"
    assert bool(torch.isfinite(res["u"]).all())
    assert res["u_per_modality"] is not None
    assert all(bool(torch.isfinite(u_m).all()) for u_m in res["u_per_modality"])
    pred = pysimlr.predict(res, masked_views)
    assert bool(torch.isfinite(torch.as_tensor(pred["u"])).all())


@pytest.mark.parametrize("method", ("lend", "flow_v"))
def test_confounds_compose_with_missing_rows_mask_on_deep_and_flow(method):
    _, masked_views, _ = _planted_missing_views(n=60, seed=29)
    rng = np.random.default_rng(29)
    conf = rng.standard_normal((60, 2)).astype(np.float32)
    res = pysimlr.fit(
        method,
        masked_views,
        k=2,
        missing_rows="mask",
        confounds=conf,
        iterations=20,
        batch_size=20,
        random_state=5,
    )
    assert res["confound_sink"] is not None
    pred = pysimlr.predict(res, masked_views)
    assert "u_sink" in pred
    assert bool(torch.isfinite(torch.as_tensor(pred["u"])).all())


def test_native_flow_simr_missing_rows_mask():
    _, masked_views, _ = _planted_missing_views(n=56, seed=41)
    res = flow_simr(
        masked_views,
        k=2,
        missing_rows="mask",
        iterations=20,
        batch_size=20,
        random_state=9,
    )
    assert res["missing_rows"] == "mask"
    assert bool(torch.isfinite(res["u"]).all())
    assert bool(torch.isnan(res["latents"][0][:10]).all())
    assert bool(torch.isfinite(res["latents"][0][10:40]).all())
    pred = pysimlr.predict_deep(masked_views, res)
    assert bool(torch.isfinite(pred["u"]).all())
    assert bool(torch.isnan(pred["latents"][0][:10]).all())


@pytest.mark.parametrize("method", ("lend", "ned", "flow_v"))
def test_missing_rows_validation_errors_match_simlr(method):
    full_views, _, _ = _planted_missing_views(n=30, seed=5)
    with pytest.raises(ValueError, match="missing_rows"):
        pysimlr.fit(method, full_views, k=2, missing_rows="drop", iterations=5)

    too_few = [v.clone() for v in full_views]
    too_few[1][4:] = float("nan")  # only 4 observed rows (< max(k+2, 6) = 6)
    with pytest.raises(ValueError, match=r"observed.*rows"):
        pysimlr.fit(method, too_few, k=2, missing_rows="mask", iterations=5)

    orphan = [v.clone() for v in full_views]
    for m in range(3):
        orphan[m][7] = float("nan")
    with pytest.raises(ValueError, match="observed in at least one view"):
        pysimlr.fit(method, orphan, k=2, missing_rows="mask", iterations=5)


@pytest.mark.parametrize("method", ("lend", "flow_v"))
def test_impute_default_reproducibility_unchanged(method):
    full_views, _, _ = _planted_missing_views(n=48, seed=13)
    r1 = pysimlr.fit(method, full_views, k=2, iterations=12, random_state=21)
    r2 = pysimlr.fit(method, full_views, k=2, missing_rows="impute", iterations=12, random_state=21)
    assert r1["missing_rows"] == "impute" and r1["obs_mask"] is None
    assert torch.equal(r1["u"], r2["u"])
    for v1, v2 in zip(r1["v"], r2["v"]):
        assert torch.equal(v1, v2)
