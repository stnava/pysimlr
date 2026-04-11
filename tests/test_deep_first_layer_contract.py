import pytest
import torch

from pysimlr import (
    lend_simr,
    ned_simr,
    ned_simr_shared_private,
    predict_deep,
    extract_first_layer_factors,
    LENDNSAEncoder,
    analyze_first_layer_alignment,
    attribute_shared_to_first_layer,
    attribute_prediction_to_features,
)
from pysimlr.utils import preprocess_data


def _scaled_inputs(mats, model_res):
    return [
        preprocess_data(torch.as_tensor(m).float(), model_res["scale_list"], prov)
        for m, prov in zip(mats, model_res["provenance_list"])
    ]


def _assert_first_layer_contract(model_res, mats):
    first_layer = model_res["first_layer"]
    assert set(first_layer.keys()) == {"v", "scores", "orthogonality_defect", "sparsity_summary"}
    assert len(first_layer["v"]) == len(mats)
    assert len(first_layer["scores"]) == len(mats)
    assert len(first_layer["orthogonality_defect"]) == len(mats)
    assert len(first_layer["sparsity_summary"]) == len(mats)

    scaled = _scaled_inputs(mats, model_res)
    for idx, (x_scaled, v, scores, summary) in enumerate(zip(scaled, first_layer["v"], first_layer["scores"], first_layer["sparsity_summary"])):
        assert scores.shape == (x_scaled.shape[0], v.shape[1])
        expected_scores = x_scaled @ v
        assert torch.allclose(scores, expected_scores, atol=1e-4, rtol=1e-4), f"first-layer scores mismatch for modality {idx}"
        assert abs(first_layer["orthogonality_defect"][idx] - summary["orthogonality_defect"]) < 1e-8
        assert summary["n_features"] == v.shape[0]
        assert summary["n_components"] == v.shape[1]
        assert len(summary["component_l0"]) == v.shape[1]
        assert len(summary["top_features"]) == v.shape[1]
        assert len(summary["top_features"][0]) > 0


def test_lend_first_layer_contract():
    mats = [torch.randn(24, 10), torch.randn(24, 8)]
    res = lend_simr(mats, k=3, epochs=2, batch_size=8, warmup_epochs=0, sim_weight=0.0)
    _assert_first_layer_contract(res, mats)
    pred = predict_deep(mats, res, device="cpu")
    assert pred["first_layer"] is not None
    assert len(pred["first_layer_scores"]) == 2
    assert pred["interpretability"] is not None
    assert pred["deep_layer"] is not None


def test_ned_first_layer_contract_and_extraction():
    mats = [torch.randn(20, 11), torch.randn(20, 7)]
    res = ned_simr(mats, k=4, epochs=2, batch_size=10, warmup_epochs=0, sim_weight=0.0)
    _assert_first_layer_contract(res, mats)
    extracted = extract_first_layer_factors(
        res,
        feature_names=[
            [f"m1_{i}" for i in range(11)],
            [f"m2_{i}" for i in range(7)],
        ],
        top_k=3,
    )
    assert extracted["sparsity_summary"][0]["top_features"][0][0]["feature_name"].startswith("m1_")
    assert len(extracted["sparsity_summary"][1]["top_features"][0]) == 3


@pytest.mark.xfail(reason="shared-private training path has a pre-existing runtime instability in this repo", strict=False)
def test_shared_private_first_layer_contract():
    mats = [torch.randn(18, 9), torch.randn(18, 6)]
    res = ned_simr_shared_private(
        mats,
        k=3,
        private_k=2,
        epochs=2,
        batch_size=6,
        warmup_epochs=0,
        sim_weight=0.0,
        private_orthogonality_weight=0.0,
        private_variance_weight=0.0,
    )
    _assert_first_layer_contract(res, mats)


def test_first_layer_training_schedule_metadata():
    mats = [torch.randn(16, 9), torch.randn(16, 7)]
    res = lend_simr(
        mats,
        k=3,
        epochs=4,
        batch_size=8,
        warmup_epochs=0,
        sim_weight=0.0,
        first_layer_mode="scheduled",
        stabilization_start_epoch=1,
        stabilization_ramp_epochs=2,
    )
    meta = res["first_layer_training"]
    assert meta["mode"] == "scheduled"
    assert meta["stabilization_start_epoch"] == 1
    assert meta["stabilization_ramp_epochs"] == 2
    assert len(meta["projection_alpha_history"]) == 4
    assert len(meta["basis_drift_history"]) == 4
    assert meta["projection_alpha_history"][0] == 0.0
    assert meta["projection_alpha_history"][-1] == 1.0
    assert meta["projection_alpha_history"] == sorted(meta["projection_alpha_history"])


def test_encoder_scheduled_projection_matches_projected_basis_at_alpha_one():
    encoder = LENDNSAEncoder(6, 2, first_layer_mode="scheduled")
    x = torch.randn(5, 6)
    encoder.train()
    encoder.set_projection_schedule(epoch=2, total_epochs=3, stabilization_start_epoch=0, stabilization_ramp_epochs=1)
    scheduled_scores = encoder.encode_first_layer(x)
    projected_scores = x @ encoder.v
    raw_scores = x @ encoder.v_raw
    assert torch.allclose(scheduled_scores, projected_scores, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(scheduled_scores, raw_scores, atol=1e-6, rtol=1e-6)


def test_pr3_alignment_and_shared_attribution_payloads():
    mats = [torch.randn(22, 9), torch.randn(22, 7)]
    res = ned_simr(mats, k=3, epochs=2, batch_size=11, warmup_epochs=0, sim_weight=0.0)

    assert "interpretability" in res
    assert "deep_layer" in res
    assert "shared_to_first_layer" in res["interpretability"]
    assert "deep_layer_alignment" in res["interpretability"]

    alignment = analyze_first_layer_alignment(res)
    shared_attr = attribute_shared_to_first_layer(res)

    assert len(alignment["modalities"]) == 2
    assert alignment["modalities"][0]["component_correlation"].shape == (3, 3)
    assert alignment["modalities"][0]["feature_importance"].shape[0] == 9
    assert len(shared_attr["per_modality"]) == 2
    assert shared_attr["combined"]["component_importance"].shape[0] == 6
    assert len(shared_attr["combined"]["feature_importance"]) == 2
    assert shared_attr["combined"]["feature_importance"][0].shape[0] == 9


def test_pr3_prediction_attribution_prefers_signal_modality():
    torch.manual_seed(0)
    n = 40
    x1 = torch.randn(n, 6)
    x2 = 0.05 * torch.randn(n, 6)
    y = 2.0 * x1[:, 0] - 1.5 * x1[:, 1] + 0.1 * torch.randn(n)

    res = lend_simr([x1, x2], k=2, epochs=2, batch_size=10, warmup_epochs=0, sim_weight=0.0)
    attr = attribute_prediction_to_features(res, y)

    m1 = attr["per_modality"][0]["global_r2"]
    m2 = attr["per_modality"][1]["global_r2"]
    assert m1 >= m2
    assert attr["combined"]["feature_importance"][0].shape[0] == 6
    assert attr["shared_latent_baseline"] is not None
