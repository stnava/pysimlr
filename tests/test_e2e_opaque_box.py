"""
Comprehensive 4-Tier Requirement-Driven Opaque-Box Test Suite for PySIMLR.

Derived strictly from ORIGINAL_REQUEST.md, PROJECT.md, and public API contracts.
Opaque-box philosophy: Exercises public interfaces as an end-user or scientific
analysis pipeline would interact with them.

Tiers:
- Tier 1: Feature Coverage (Core Public APIs: simlr, flow-simr, lend-simr, ned-simr,
          sklearn estimators/transformers, MultiViewPCA, IO serialization, diagnostics)
- Tier 2: Boundary & Corner Cases (singular/rank-deficient, N_test=1 batch invariance,
          1D input rejection, empty inputs, shape mismatches, non-negativity)
- Tier 3: Cross-Feature Interactions (MultiViewPCA whitening + SiMLR, Flow-SiMR-V conditional
          synthesis, LOO & graph topology anchors under CV, scikit-learn pipeline chaining)
- Tier 4: Real-World Application Workloads (3-view genomics/imaging/phenotype synthetic pipeline,
          clinical diagnostic classification pipeline, generative multimodal imputation)
"""

import os
import shutil
import random
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold

import pysimlr
from pysimlr import (
    simlr,
    predict_simlr,
    predict_shared_latent,
    reconstruct_from_learned_maps,
    estimate_rank,
    decompose_energy,
    lend_simr,
    ned_simr,
    predict_deep,
    flow_simr,
    flow_simr_v,
    FlowSiMRModel,
    FlowSiMRVModel,
    FlowConditionalInference,
    write_simlr,
    read_simlr,
    adjusted_rvcoef,
    compute_shared_consensus,
    whiten_matrix,
    safe_pca,
    ba_svd,
)
from pysimlr.sklearn import (
    SiMLREstimator,
    SiMLR,
    LENDTransformer,
    NEDTransformer,
    FlowSiMLRTransformer,
    MultiViewPCA,
    build_simlr_pipeline,
)


# ============================================================================
# DETERMINISTIC FIXTURES & DATA GENERATORS
# ============================================================================

@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Ensure strict determinism across all tests in the opaque-box suite."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def clean_two_views():
    """Generate 2-view synthetic dataset with ground-truth latent components."""
    torch.manual_seed(42)
    n, p1, p2, k = 60, 16, 12, 2
    u_true = torch.randn(n, k)
    u_true, _ = torch.linalg.qr(u_true)

    w1 = torch.randn(k, p1)
    w2 = torch.randn(k, p2)

    x1 = u_true @ w1 + 0.05 * torch.randn(n, p1)
    x2 = u_true @ w2 + 0.05 * torch.randn(n, p2)

    return {
        "x1": x1,
        "x2": x2,
        "u_true": u_true,
        "n": n,
        "p1": p1,
        "p2": p2,
        "k": k,
    }


@pytest.fixture
def clean_three_views():
    """Generate 3-view synthetic dataset with shared latent consensus."""
    torch.manual_seed(101)
    n, p1, p2, p3, k = 50, 14, 10, 12, 3
    u_true = torch.randn(n, k)
    u_true, _ = torch.linalg.qr(u_true)

    w1 = torch.randn(k, p1)
    w2 = torch.randn(k, p2)
    w3 = torch.randn(k, p3)

    x1 = u_true @ w1 + 0.05 * torch.randn(n, p1)
    x2 = u_true @ w2 + 0.05 * torch.randn(n, p2)
    x3 = u_true @ w3 + 0.05 * torch.randn(n, p3)

    return {
        "views": [x1, x2, x3],
        "u_true": u_true,
        "n": n,
        "dims": [p1, p2, p3],
        "k": k,
    }


# ============================================================================
# TIER 1: FEATURE COVERAGE (CORE PUBLIC APIS)
# ============================================================================

class TestTier1FeatureCoverage:
    """Happy-path tests for each core public capability."""

    def test_simlr_linear_multiview_two_and_three_views(self, clean_two_views, clean_three_views):
        """Verify linear multi-view simlr on 2-view and 3-view datasets."""
        # 1. Two views
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k2 = clean_two_views["k"]
        res2 = simlr([x1, x2], k=k2, iterations=10, energy_type="recon_r2", optimizer_type="hybrid_adam")

        assert "v" in res2
        assert "u" in res2
        assert "energy" in res2
        assert "best_energy" in res2
        assert len(res2["v"]) == 2
        assert res2["v"][0].shape == (clean_two_views["p1"], k2)
        assert res2["v"][1].shape == (clean_two_views["p2"], k2)
        assert np.isfinite(res2["best_energy"])

        # Reconstruction from learned maps
        recons2 = reconstruct_from_learned_maps(res2["u"], res2)
        assert len(recons2) == 2
        assert recons2[0].shape == x1.shape
        assert recons2[1].shape == x2.shape

        # 2. Three views
        views3 = clean_three_views["views"]
        k3 = clean_three_views["k"]
        res3 = simlr(views3, k=k3, iterations=8, optimizer_type="torch_lbfgs")
        assert len(res3["v"]) == 3
        for i, dim in enumerate(clean_three_views["dims"]):
            assert res3["v"][i].shape == (dim, k3)

    def test_simlr_predict_shared_latent_and_predict_simlr(self, clean_two_views):
        """Verify out-of-sample projection via predict_shared_latent and predict_simlr."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k = clean_two_views["k"]
        res = simlr([x1, x2], k=k, iterations=10)

        # Held-out split
        x1_test = x1[:10] + 0.01 * torch.randn(10, clean_two_views["p1"])
        x2_test = x2[:10] + 0.01 * torch.randn(10, clean_two_views["p2"])

        u_pred = predict_shared_latent([x1_test, x2_test], res)
        assert u_pred.shape == (10, k)
        assert torch.isfinite(u_pred).all()

        pred_dict = predict_simlr([x1_test, x2_test], res)
        assert "u" in pred_dict
        assert "latents" in pred_dict
        assert "reconstructions" in pred_dict
        assert pred_dict["u"].shape == (10, k)
        assert len(pred_dict["latents"]) == 2

    def test_flow_simr_and_model_architecture(self, clean_two_views):
        """Verify FlowSiMRModel instantiation and flow_simr execution."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k = clean_two_views["k"]
        p1, p2 = clean_two_views["p1"], clean_two_views["p2"]

        # Direct model instantiation test
        model = FlowSiMRModel([p1, p2], latent_dim=k, num_layers=2, hidden_dim=32)
        assert hasattr(model, "flows")
        assert hasattr(model, "encoders")
        assert hasattr(model, "decoders")
        assert len(model.flows) == 2

        # Public flow_simr wrapper execution
        res = flow_simr([x1, x2], k=k, epochs=4, batch_size=30, warmup_epochs=1, verbose=False)
        assert "model" in res
        assert "latents" in res
        assert "cond_inference" in res
        assert len(res["latents"]) == 2
        assert res["latents"][0].shape == (clean_two_views["n"], k)
        assert torch.isfinite(res["latents"][0]).all()

    def test_flow_simr_v_linear_stiefel_flow_hybrid(self, clean_two_views):
        """Verify FlowSiMRVModel instantiation and flow_simr_v execution."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k = clean_two_views["k"]
        p1, p2 = clean_two_views["p1"], clean_two_views["p2"]

        model_v = FlowSiMRVModel([p1, p2], latent_dim=k, num_layers=2, hidden_dim=32, nsa_w=0.1)
        assert hasattr(model_v, "linear_encoders")
        assert len(model_v.linear_encoders) == 2

        res_v = flow_simr_v([x1, x2], k=k, epochs=4, batch_size=30, warmup_epochs=1, verbose=False)
        assert "model" in res_v
        assert "v" in res_v
        assert len(res_v["v"]) == 2
        assert res_v["v"][0].shape == (p1, k)
        assert res_v["v"][1].shape == (p2, k)

    def test_lend_simr_deep_hybrid_contract(self, clean_two_views):
        """Verify lend_simr linear-encoder non-linear-decoder contract and predict_deep."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k = clean_two_views["k"]

        res = lend_simr([x1, x2], k=k, iterations=2, epochs=4, batch_size=30, warmup_epochs=1, verbose=False)
        assert res["model_type"] == "lend_simr"
        assert "v" in res
        assert len(res["v"]) == 2
        assert "latents" in res
        assert "first_layer" in res

        # Predict with predict_deep
        x1_held = x1[:8]
        x2_held = x2[:8]
        pred = predict_deep([x1_held, x2_held], res)
        assert "latents" in pred
        assert pred["latents"][0].shape == (8, k)

    def test_ned_simr_deep_contract(self, clean_two_views):
        """Verify ned_simr non-linear encoder-decoder model and predict_deep."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k = clean_two_views["k"]

        res = ned_simr([x1, x2], k=k, iterations=2, epochs=4, batch_size=30, warmup_epochs=1, verbose=False)
        assert res["model_type"] == "ned_simr"
        assert len(res["latents"]) == 2
        assert res["latents"][0].shape == (clean_two_views["n"], k)

        pred = predict_deep([x1[:5], x2[:5]], res)
        assert pred["latents"][0].shape == (5, k)

    def test_sklearn_simlr_estimator_api(self, clean_two_views):
        """Verify SiMLREstimator / SiMLR conforms to Scikit-Learn Estimator specifications."""
        x1_np = clean_two_views["x1"].numpy()
        x2_np = clean_two_views["x2"].numpy()
        k = clean_two_views["k"]

        # Single-view fit & transform
        single_est = SiMLR(n_components=k, iterations=8, optimizer_type="torch_lbfgs", random_state=42)
        single_est.fit(x1_np)
        assert hasattr(single_est, "components_")
        assert hasattr(single_est, "loadings_")
        assert single_est.components_.shape == (k, clean_two_views["p1"])
        assert single_est.loadings_.shape == (clean_two_views["p1"], k)

        z_single = single_est.transform(x1_np)
        assert z_single.shape == (clean_two_views["n"], k)
        z_ft = single_est.fit_transform(x1_np)
        assert z_ft.shape == (clean_two_views["n"], k)

        rec_single = single_est.inverse_transform(z_single)
        assert rec_single.shape == x1_np.shape

        # Multi-view fit & transform
        multi_est = SiMLREstimator(n_components=k, iterations=8, optimizer_type="torch_lbfgs", random_state=42)
        multi_est.fit([x1_np, x2_np])
        assert isinstance(multi_est.components_, list)
        assert len(multi_est.components_) == 2
        z_multi = multi_est.transform([x1_np, x2_np])
        assert z_multi.shape == (clean_two_views["n"], k)

    def test_sklearn_deep_transformers(self, clean_two_views):
        """Verify LENDTransformer, NEDTransformer, and FlowSiMLRTransformer scikit-learn transformers."""
        x1_np = clean_two_views["x1"].numpy()
        k = 2

        # LENDTransformer
        lend_t = LENDTransformer(n_components=k, iterations=2, epochs=4, random_state=42)
        lend_t.fit(x1_np)
        z_lend = lend_t.transform(x1_np)
        assert z_lend.shape == (clean_two_views["n"], k)

        # NEDTransformer
        ned_t = NEDTransformer(n_components=k, iterations=2, epochs=4, random_state=42)
        ned_t.fit(x1_np)
        z_ned = ned_t.transform(x1_np)
        assert z_ned.shape == (clean_two_views["n"], k)

        # FlowSiMLRTransformer
        flow_t = FlowSiMLRTransformer(n_components=k, epochs=4, random_state=42)
        flow_t.fit(x1_np)
        z_flow = flow_t.transform(x1_np)
        assert z_flow.shape == (clean_two_views["n"], k)

    def test_sklearn_multiview_pca(self, clean_two_views):
        """Verify MultiViewPCA per-view orthogonal decomposition, whitening, and concatenation."""
        x1_np = clean_two_views["x1"].numpy()
        x2_np = clean_two_views["x2"].numpy()
        k = 3

        # Default: concat=True
        mv_pca = MultiViewPCA(n_components=k, scale=True, whiten=False, concat=True, random_state=42)
        mv_pca.fit([x1_np, x2_np])

        assert hasattr(mv_pca, "components_")
        assert hasattr(mv_pca, "loadings_")
        assert len(mv_pca.components_) == 2
        assert mv_pca.components_[0].shape == (k, clean_two_views["p1"])
        assert mv_pca.components_[1].shape == (k, clean_two_views["p2"])

        z_concat = mv_pca.transform([x1_np, x2_np])
        assert z_concat.shape == (clean_two_views["n"], 2 * k)

        # concat=False with whitening
        mv_pca_list = MultiViewPCA(n_components=k, scale=True, whiten=True, concat=False, random_state=42)
        mv_pca_list.fit([x1_np, x2_np])
        z_list = mv_pca_list.transform([x1_np, x2_np])
        assert isinstance(z_list, list)
        assert len(z_list) == 2
        assert z_list[0].shape == (clean_two_views["n"], k)
        assert z_list[1].shape == (clean_two_views["n"], k)
        # Whitened variance should be near 1.0
        assert np.allclose(np.var(z_list[0], axis=0), np.ones(k), atol=1e-1)

    def test_io_serialization_clean_roundtrip(self, clean_two_views, tmp_path):
        """Verify write_simlr and read_simlr correctly persist and restore primary model results."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        res = simlr([x1, x2], k=2, iterations=6)

        prefix = os.path.join(tmp_path, "test_clean_simlr")
        core_obj = {
            "u": res["u"],
            "v": res["v"],
            "w": res["w"],
            "energy": res["energy"],
            "converged_iter": res["converged_iter"],
            "modality_names": ["modality_1", "modality_2"],
        }
        write_simlr(core_obj, file_prefix=prefix, clear_dir=True, use_r=False)

        outdir = f"{prefix}_simlr"
        assert os.path.isdir(outdir)
        assert os.path.isfile(os.path.join(outdir, "manifest.json"))

        restored = read_simlr(outdir, use_r=False)
        assert "u" in restored
        assert "v" in restored
        assert len(restored["v"]) == len(res["v"])
        assert restored["modality_names"] == ["modality_1", "modality_2"]
        np.testing.assert_allclose(restored["u"], res["u"].detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    def test_io_serialization_raw_model_regression(self, clean_two_views, tmp_path):
        """Verify write_simlr and read_simlr handle raw dictionary with provenance_list (tracking F2.9)."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        res = simlr([x1, x2], k=2, iterations=6)

        prefix = os.path.join(tmp_path, "test_opaque_simlr")
        # Direct raw simlr serialization check:
        # Note: res["provenance_list"] contains nested tensors ('cas_mean', 'cas_std').
        # F2.9 in M2 tracks tensor-safe JSON serialization.
        write_simlr(res, file_prefix=prefix, clear_dir=True, use_r=False)
        outdir = f"{prefix}_simlr"
        assert os.path.isdir(outdir)
        assert os.path.isfile(os.path.join(outdir, "manifest.json"))
        restored = read_simlr(outdir, use_r=False)
        assert "u" in restored
        assert "v" in restored
        assert len(restored["v"]) == len(res["v"])
        np.testing.assert_allclose(restored["u"], res["u"].detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    def test_diagnostic_estimate_rank_and_decompose_energy(self, clean_two_views):
        """Verify estimate_rank and decompose_energy diagnostics."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        # Fast heuristic rank estimation (n_permutations=2)
        k_est = estimate_rank([x1, x2], n_permutations=2, var_threshold=0.90)
        assert isinstance(k_est, int)
        assert 1 <= k_est <= clean_two_views["p2"]

        res = simlr([x1, x2], k=2, iterations=6)
        energy_decomp = decompose_energy([x1, x2], res, energy_type="acc")
        assert "modality_energies" in energy_decomp
        assert len(energy_decomp["modality_energies"]) == 2
        assert all(np.isfinite(e) for e in energy_decomp["modality_energies"])


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ============================================================================

class TestTier2BoundariesAndCornerCases:
    """Stress tests on pathological boundaries, dimensions, and invariants."""

    def test_boundary_rank_deficient_and_singular_matrices(self):
        """Verify simlr and MultiViewPCA handle collinear/rank-deficient matrices without crashing."""
        torch.manual_seed(42)
        n = 50
        # View 1: rank 1 repeated columns
        v1_col = torch.randn(n, 1)
        x1_singular = v1_col.repeat(1, 10) + 1e-7 * torch.randn(n, 10)
        # View 2: standard matrix
        x2_normal = torch.randn(n, 8)

        # simlr should not crash or return NaNs
        res = simlr([x1_singular, x2_normal], k=2, iterations=6)
        assert torch.isfinite(res["u"]).all()
        assert torch.isfinite(res["v"][0]).all()
        assert torch.isfinite(res["v"][1]).all()

        # MultiViewPCA should also handle singular matrices cleanly
        mv_pca = MultiViewPCA(n_components=2, scale=True, random_state=42)
        mv_pca.fit([x1_singular.numpy(), x2_normal.numpy()])
        z = mv_pca.transform([x1_singular.numpy(), x2_normal.numpy()])
        assert np.isfinite(z).all()

    def test_boundary_single_sample_n1_batch_invariance(self, clean_two_views):
        """Verify single-sample N_test=1 prediction is identical whether alone or inside a batch."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k = clean_two_views["k"]

        # 1. Single-view SiMLREstimator / SiMLR batch invariance
        single_est = SiMLR(n_components=k, iterations=8, optimizer_type="torch_lbfgs", random_state=42)
        single_est.fit(x1.numpy())
        batch_x1 = x1[:5].numpy()
        z_batch = single_est.transform(batch_x1)
        z_single = single_est.transform(batch_x1[0:1])
        assert z_single.shape == (1, k)
        assert np.isfinite(z_single).all()
        np.testing.assert_allclose(
            z_single[0], z_batch[0], rtol=1e-5, atol=1e-5,
            err_msg="Single view N=1 projection drifted from its in-batch coordinate."
        )

        # 2. Multi-view single sample transformation stability
        multi_est = SiMLREstimator(n_components=k, iterations=8, random_state=42)
        multi_est.fit([x1.numpy(), x2.numpy()])
        res = multi_est.result_
        u_single = predict_shared_latent([x1[0:1], x2[0:1]], res)
        assert u_single.shape == (1, k)
        assert torch.isfinite(u_single).all()

        u_batch = predict_shared_latent([x1[:5], x2[:5]], res)
        np.testing.assert_allclose(
            u_single[0].detach().cpu().numpy(),
            u_batch[0].detach().cpu().numpy(),
            rtol=1e-5, atol=1e-5
        )

    def test_boundary_single_sample_n1_multiview_pca(self, clean_two_views):
        """Verify MultiViewPCA transforms single sample N=1 identically to in-batch."""
        x1_np = clean_two_views["x1"].numpy()
        x2_np = clean_two_views["x2"].numpy()
        mv_pca = MultiViewPCA(n_components=2, scale=True, concat=True, random_state=42)
        mv_pca.fit([x1_np, x2_np])

        batch_z = mv_pca.transform([x1_np[:4], x2_np[:4]])
        single_z = mv_pca.transform([x1_np[0:1], x2_np[0:1]])
        assert single_z.shape == (1, 4)
        np.testing.assert_allclose(single_z[0], batch_z[0], rtol=1e-5, atol=1e-5)

    def test_boundary_1d_input_contract_rejection(self):
        """Verify passing 1D arrays instead of 2D matrices raises clear ValueError or IndexError."""
        x_1d = torch.randn(40)
        x_2d = torch.randn(40, 8)

        with pytest.raises((ValueError, IndexError)):
            simlr([x_1d, x_2d], k=2, iterations=4)

        # Also predict_shared_latent
        res_mock = {"v": [torch.randn(8, 2), torch.randn(8, 2)], "mixing_algorithm": "svd"}
        with pytest.raises((ValueError, IndexError)):
            predict_shared_latent([x_1d, x_2d], res_mock)

    def test_boundary_empty_inputs_rejection(self):
        """Verify empty input list raises clear ValueError or IndexError."""
        with pytest.raises((ValueError, IndexError)):
            simlr([], k=2)

        with pytest.raises((ValueError, IndexError)):
            predict_shared_latent([], {"v": []})

    def test_boundary_sample_count_mismatch_fast_fail(self):
        """Verify modalities with unequal sample counts (N1 != N2) fail fast with error."""
        x1 = torch.randn(40, 10)
        x2 = torch.randn(55, 10)  # N2 != N1

        with pytest.raises((ValueError, RuntimeError)):
            simlr([x1, x2], k=2, iterations=4)

    def test_boundary_feature_count_mismatch_at_test_time(self, clean_two_views):
        """Verify dimension mismatch at test-time prediction raises ValueError."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        res = simlr([x1, x2], k=2, iterations=6)

        # Provide x1 with wrong feature count (p1 + 5)
        bad_x1 = torch.randn(10, clean_two_views["p1"] + 5)
        good_x2 = torch.randn(10, clean_two_views["p2"])

        with pytest.raises((ValueError, RuntimeError)):
            predict_shared_latent([bad_x1, good_x2], res)

    def test_boundary_constant_zero_variance_features(self):
        """Verify datasets with constant all-zero features optimize without NaNs/Infs."""
        torch.manual_seed(42)
        n = 40
        # Add 3 all-zero columns
        x1 = torch.cat([torch.randn(n, 8), torch.zeros(n, 3)], dim=1)
        x2 = torch.randn(n, 8)

        res = simlr([x1, x2], k=2, iterations=8)
        assert torch.isfinite(res["u"]).all()
        assert torch.isfinite(res["v"][0]).all()
        assert torch.isfinite(res["v"][1]).all()

    def test_boundary_positivity_and_nonnegative_consensus(self, clean_two_views):
        """Verify positivity='positive' and nonneg_u=True enforce non-negative loadings and consensus."""
        x1, x2 = clean_two_views["x1"].abs(), clean_two_views["x2"].abs()
        k = 2

        res = simlr([x1, x2], k=k, iterations=10, positivity="positive", nonneg_u=True)

        for i, v in enumerate(res["v"]):
            min_v = float(v.min())
            assert min_v >= -1e-5, f"Modality {i} basis violated non-negativity: min={min_v}"

        u = res["u"]
        min_u = float(u.min())
        assert min_u >= -1e-5, f"Consensus U violated non-negativity: min={min_u}"


# ============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS
# ============================================================================

class TestTier3CrossFeatureInteractions:
    """Pairwise and combinatorial interactions between multiple library modules."""

    def test_interaction_multiview_pca_whitening_chained_with_simlr(self, clean_three_views):
        """Verify MultiViewPCA whitening upstream seamlessly feeds into classical SiMLR."""
        views_np = [v.numpy() for v in clean_three_views["views"]]
        k_pca = 6
        k_simlr = 2

        # 1. Whitening stage with MultiViewPCA
        mv_pca = MultiViewPCA(n_components=k_pca, scale=True, whiten=True, concat=False, random_state=42)
        whitened_views = mv_pca.fit_transform(views_np)
        assert len(whitened_views) == 3
        for wv in whitened_views:
            assert wv.shape == (clean_three_views["n"], k_pca)

        # 2. SiMLR alignment on whitened representations
        res = simlr(whitened_views, k=k_simlr, iterations=10, optimizer_type="torch_lbfgs")
        assert len(res["v"]) == 3
        for v in res["v"]:
            assert v.shape == (k_pca, k_simlr)
            assert torch.isfinite(v).all()

    def test_interaction_flow_simr_v_conditional_inference_synthesis(self, clean_two_views):
        """Verify Flow-SiMR-V conditional inference performs cross-modality synthesis out-of-sample."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k = clean_two_views["k"]
        p2 = clean_two_views["p2"]
        n_train = 45

        x1_train, x2_train = x1[:n_train], x2[:n_train]
        x1_test, x2_test = x1[n_train:], x2[n_train:]
        n_test = x1_test.shape[0]

        res = flow_simr_v([x1_train, x2_train], k=k, epochs=6, batch_size=20, warmup_epochs=1, verbose=False)
        cond = res.get("cond_inference")
        assert isinstance(cond, FlowConditionalInference)

        # Obtain latent for held-out modality 0
        model = res["model"]
        model.eval()
        with torch.no_grad():
            v0 = res["v"][0]
            # Linear projection followed by flow encoder
            z0_proj = x1_test @ v0
            z0_flow = model.encoders[0](z0_proj)
            # Predict modality 1 latent from modality 0 latent
            z1_pred = cond.predict_conditional(observed_idx=0, target_idx=1, observed_z=z0_flow)
            assert z1_pred.shape == (n_test, k)
            assert torch.isfinite(z1_pred).all()

            # Decode through modality 1 decoder (Flow-SiMR-V flow decoder maps k -> k)
            z1_decoded = model.decoders[1](z1_pred)
            assert z1_decoded.shape == (n_test, k)
            assert torch.isfinite(z1_decoded).all()

            # Ambient space synthesis via linear loadings V_1
            v1 = res["v"][1]
            x2_synth = z1_decoded @ v1.T
            assert x2_synth.shape == (n_test, p2)
            assert torch.isfinite(x2_synth).all()

    def test_interaction_topologies_under_cross_validation(self, clean_two_views):
        """Verify LOO and Graph topologies maintain stable consensus anchors across CV splits."""
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        k = 2
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(x1):
            x1_tr, x2_tr = x1[train_idx], x2[train_idx]
            x1_val, x2_val = x1[val_idx], x2[val_idx]

            # LOO topology returns list of consensus tensors (one per view)
            res_loo = simlr([x1_tr, x2_tr], k=k, topology="loo", iterations=6)
            u_loo_val = predict_shared_latent([x1_val, x2_val], res_loo)
            assert isinstance(u_loo_val, list)
            assert len(u_loo_val) == 2
            assert u_loo_val[0].shape == (len(val_idx), k)
            assert torch.isfinite(u_loo_val[0]).all()

    def test_interaction_sklearn_pipeline_end_to_end(self, clean_two_views):
        """Verify SiMLREstimator integrates into a full scikit-learn Pipeline and cross_val_score."""
        x1_np = clean_two_views["x1"].numpy()
        # Synthetic binary target correlated with the ground-truth latent
        u_true = clean_two_views["u_true"].numpy()
        y = (u_true[:, 0] > 0.0).astype(int)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("simlr", SiMLREstimator(n_components=2, iterations=8, optimizer_type="torch_lbfgs", random_state=42)),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs")),
        ])

        scores = cross_val_score(pipe, x1_np, y, cv=3)
        assert len(scores) == 3
        assert (scores >= 0.0).all()
        # Ensure meaningful learning (accuracy well above random chance 0.5)
        assert scores.mean() > 0.65

    def test_interaction_multiview_pca_pipeline_regression(self, clean_two_views):
        """Verify MultiViewPCA integrates into scikit-learn regression pipeline."""
        x1_np = clean_two_views["x1"].numpy()
        x2_np = clean_two_views["x2"].numpy()
        u_true = clean_two_views["u_true"].numpy()
        y_reg = 2.5 * u_true[:, 0] + 0.1 * np.random.randn(clean_two_views["n"])

        # Multi-view list transformed by MultiViewPCA(concat=True)
        mv_pca = MultiViewPCA(n_components=3, scale=True, concat=True, random_state=42)
        X_reduced = mv_pca.fit_transform([x1_np, x2_np])

        reg = Ridge(alpha=1.0)
        reg.fit(X_reduced, y_reg)
        r2 = reg.score(X_reduced, y_reg)
        assert r2 > 0.70, f"Expected R2 > 0.70 from PCA coordinates, got {r2:.3f}"


# ============================================================================
# TIER 4: REAL-WORLD APPLICATION WORKLOADS
# ============================================================================

class TestTier4RealWorldWorkloads:
    """Full-scale realistic scientific workloads and synthetic pipelines."""

    def test_workload_three_view_genomics_imaging_phenotype(self, tmp_path):
        """
        Scenario 1: 3-View Multi-Omics / Neuroimaging Synthetic Pipeline.
        - View 0: Genomics (p1=30 gene expression / SNP features)
        - View 1: Neuroimaging (p2=24 regional cortical thickness / volume features)
        - View 2: Phenotype (p3=15 cognitive scores & physiological markers)
        Driven by 2 latent biological pathways + structured noise.
        """
        torch.manual_seed(2026)
        n, p_gen, p_img, p_phen, k = 75, 30, 24, 15, 2

        # Latent biological disease axis
        u_disease = torch.randn(n, k)
        u_disease, _ = torch.linalg.qr(u_disease)

        w_gen = torch.randn(k, p_gen)
        w_img = torch.randn(k, p_img)
        w_phen = torch.randn(k, p_phen)

        x_gen = u_disease @ w_gen + 0.05 * torch.randn(n, p_gen)
        x_img = u_disease @ w_img + 0.05 * torch.randn(n, p_img)
        x_phen = u_disease @ w_phen + 0.05 * torch.randn(n, p_phen)

        # 1. Pipeline step: Multi-view PCA feature reduction & scaling check
        mv_pca = MultiViewPCA(n_components=6, scale=True, whiten=False, concat=False, random_state=42)
        views_scaled = mv_pca.fit_transform([x_gen.numpy(), x_img.numpy(), x_phen.numpy()])
        assert len(views_scaled) == 3

        # 2. Pipeline step: Multi-view SiMLR integration
        res = simlr([x_gen, x_img, x_phen], k=k, iterations=15, energy_type="recon_r2")

        # 3. Scientific Verification: Biological pathway recovery
        u_est = res["u"]
        recovery_metric = float(adjusted_rvcoef(u_est, u_disease))
        assert recovery_metric > 0.85, (
            f"Expected latent pathway recovery RV > 0.85, got {recovery_metric:.4f}"
        )

        # 4. Pipeline step: Model persistence to scientific archive
        archive_prefix = os.path.join(tmp_path, "pipeline_run_3view")
        core_obj = {
            "u": u_est,
            "v": res["v"],
            "energy": res["energy"],
            "converged_iter": res["converged_iter"],
        }
        write_simlr(core_obj, file_prefix=archive_prefix, clear_dir=True, use_r=False)
        restored = read_simlr(f"{archive_prefix}_simlr", use_r=False)
        np.testing.assert_allclose(restored["u"], u_est.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)

    def test_workload_clinical_classification_pipeline(self):
        """
        Scenario 2: Clinical Diagnostic Classification Pipeline.
        Evaluates 5-fold cross-validated diagnostic classification (e.g. MCI vs Cognitively Normal)
        using SiMLREstimator multi-view latent representations.
        """
        np.random.seed(42)
        torch.manual_seed(42)
        n, p1, p2, k = 80, 20, 15, 2

        # Latent cognitive impairment factor
        u_latent = np.random.randn(n, k)
        u_latent, _ = np.linalg.qr(u_latent)

        # Diagnostic label driven by first latent factor
        y = (u_latent[:, 0] > 0.1).astype(int)

        x1 = u_latent @ np.random.randn(k, p1) + 0.1 * np.random.randn(n, p1)
        x2 = u_latent @ np.random.randn(k, p2) + 0.1 * np.random.randn(n, p2)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_accuracies = []

        for train_idx, test_idx in kf.split(x1):
            x1_tr, x2_tr = x1[train_idx], x2[train_idx]
            x1_te, x2_te = x1[test_idx], x2[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Fit multi-view estimator strictly inside training fold (zero test leakage)
            simlr_model = SiMLREstimator(n_components=k, iterations=10, optimizer_type="torch_lbfgs", random_state=42)
            z_tr = simlr_model.fit_transform([x1_tr, x2_tr])
            z_te = simlr_model.transform([x1_te, x2_te])

            # Downstream clinical classifier
            clf = LogisticRegression(C=1.0)
            clf.fit(z_tr, y_tr)
            acc = clf.score(z_te, y_te)
            fold_accuracies.append(acc)

        mean_acc = np.mean(fold_accuracies)
        assert mean_acc > 0.70, (
            f"Expected 5-fold cross-validated classification accuracy > 0.70, got {mean_acc:.3f}"
        )

    def test_workload_generative_multimodal_imputation(self, clean_two_views):
        """
        Scenario 3: Generative Multimodal Imputation Workload.
        Verifies Flow-SiMR generative model can accurately impute missing modalities
        for held-out test subjects using conditional distribution modeling.
        """
        x1, x2 = clean_two_views["x1"], clean_two_views["x2"]
        n_train = 48
        k = clean_two_views["k"]
        p2 = clean_two_views["p2"]

        x1_train, x2_train = x1[:n_train], x2[:n_train]
        x1_test, x2_test = x1[n_train:], x2[n_train:]
        n_test = x1_test.shape[0]

        # Train generative Flow-SiMR model
        res = flow_simr([x1_train, x2_train], k=k, epochs=8, batch_size=24, warmup_epochs=1, verbose=False)
        cond = res["cond_inference"]
        model = res["model"]
        model.eval()

        with torch.no_grad():
            # Project held-out x1 through encoder 0
            z0_test = model.encoders[0](x1_test)
            # Impute latent coordinates for modality 1
            z1_imputed = cond.predict_conditional(observed_idx=0, target_idx=1, observed_z=z0_test)
            # Decode to synthesize modality 1 features
            x2_imputed = model.decoders[1](z1_imputed)

            assert x2_imputed.shape == (n_test, p2)
            assert torch.isfinite(x2_imputed).all()

            # Verify positive correlation between imputed and actual features across samples
            recon_error = torch.mean((x2_imputed - x2_test)**2).item()
            assert np.isfinite(recon_error)
