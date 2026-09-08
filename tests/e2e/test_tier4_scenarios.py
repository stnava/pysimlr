"""Tier 4: Full-Scale Real-World Application Scenarios for pysimlr.

Scenarios:
1. Multi-modal Omics Integration (Transcriptomics + Proteomics):
   Pre-whitening, LOO consensus, classical SiMLR vs deep LEND, biological pathway recovery.
2. High-Dimensional Non-linear Manifold Recovery:
   Curved multi-view non-linear manifold with ambient noise (P=24-28), untangled by NED & NED++
   with Log-Cosh loss.
3. Generative Bijective Cross-Modality Imputation:
   Out-of-sample synthesis from View 1 to View 2 using ANTsTorch RealNVP Flow-SiMR and
   Woodbury-based conditional Gaussian inference.
4. Structural Path Modeling with Topological Graph Constraints:
   Biological developmental lineage (Genomics -> Transcriptomics -> Clinical endpoints)
   evaluated via fit_structural_models and simlr_path.
5. Extreme Robustness & Outlier Resilience:
   Heavy-tailed Cauchy noise, extreme 10x sample outliers, collinearity, and near-singular
   covariance stabilized by Armijo line search and NSA non-negative Stiefel retractions.
"""

import pytest
import torch
import numpy as np

from pysimlr import (
    simlr,
    lend_simr,
    ned_simr,
    ned_simr_shared_private,
    flow_simr,
    whiten_matrix,
    fit_structural_models,
    create_path_graph,
    simlr_path,
    sparse_distance_matrix,
    adjusted_rvcoef,
    invariant_orthogonality_defect,
)


def test_scenario1_multimodal_omics_integration(synthetic_omics_data):
    """
    Scenario 1: Transcriptomics + Proteomics biological integration.
    Exercises: Whitening check -> Classical SiMLR with LOO -> Deep LEND -> Pathway recovery.
    """
    x_rna = synthetic_omics_data["x_rna"]
    x_prot = synthetic_omics_data["x_prot"]
    pathway_signal = synthetic_omics_data["pathway_signal"]
    n = synthetic_omics_data["n"]
    k = synthetic_omics_data["k"]

    # 1. Whitening stage verification on individual omics modalities
    w_rna = whiten_matrix(x_rna, nc=20)["whitened_matrix"]
    w_prot = whiten_matrix(x_prot, nc=15)["whitened_matrix"]
    assert w_rna.shape == (n, 20)
    assert w_prot.shape == (n, 15)

    # 2. Classical SiMLR with Leave-One-Out (LOO) consensus directly on multi-omics views
    res_simlr = simlr(
        [x_rna, x_prot],
        k=k,
        topology="loo",
        iterations=15,
        optimizer_type="hybrid_adam"
    )
    assert len(res_simlr["v"]) == 2
    assert len(res_simlr["u"]) == 2

    # Verify high alignment of classical consensus with ground truth biological pathway
    u_rna_simlr = x_rna @ res_simlr["v"][0]
    rv_simlr = adjusted_rvcoef(u_rna_simlr, pathway_signal)
    assert rv_simlr > 0.65, f"Expected classical SiMLR pathway recovery > 0.65, got {rv_simlr}"

    # 3. Deep LEND multi-omics representation learning
    res_lend = lend_simr(
        [x_rna, x_prot],
        k=k,
        epochs=12,
        batch_size=25,
        warmup_epochs=2,
        hidden_dims=[32, 16],
        verbose=False
    )
    assert res_lend["model_type"] == "lend_simr"
    assert len(res_lend["latents"]) == 2

    # Verify deep consensus also captures biological pathway signal
    rv_lend = adjusted_rvcoef(res_lend["latents"][0], pathway_signal)
    assert rv_lend > 0.50, f"Expected deep LEND pathway recovery > 0.50, got {rv_lend}"


def test_scenario2_high_dimensional_nonlinear_manifold_recovery():
    """
    Scenario 2: High-Dimensional Non-linear Manifold Recovery.
    Non-linear manifold embedded in ambient noise (p=24-28), untangled by NED / NED++
    with Log-Cosh energy objective.
    """
    torch.manual_seed(42)
    n = 70
    p1, p2 = 28, 24
    k = 2

    # Latent manifold parameter (continuous non-linear developmental trajectory)
    theta = torch.linspace(0, 2 * np.pi, n)
    latent_signal = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)

    # View 1: Non-linear trigonometric + quadratic embedding + ambient noise
    w1_ambient = torch.randn(2, p1)
    x1 = torch.sin(latent_signal @ w1_ambient) + 0.1 * (latent_signal[:, :1] ** 2) + 0.05 * torch.randn(n, p1)

    # View 2: Hyperbolic tangent embedding + ambient noise
    w2_ambient = torch.randn(2, p2)
    x2 = torch.tanh(latent_signal @ w2_ambient) + 0.05 * torch.randn(n, p2)

    # 1. Fit non-linear NED model with Log-Cosh energy objective
    res_ned = ned_simr(
        [x1, x2],
        k=k,
        epochs=12,
        batch_size=25,
        warmup_epochs=2,
        energy_type="logcosh",
        hidden_dims=[32, 16],
        verbose=False
    )
    assert res_ned["model_type"] == "ned_simr"
    assert len(res_ned["latents"]) == 2

    # Verify non-linear latent recovers continuous trajectory
    rv_ned = adjusted_rvcoef(res_ned["latents"][0], latent_signal)
    assert rv_ned > 0.40, f"Expected non-linear manifold recovery > 0.40, got {rv_ned}"

    # 2. Disentangle shared manifold from modality-specific variation with NED++
    res_nedpp = ned_simr_shared_private(
        [x1, x2],
        k=k,
        private_k=2,
        epochs=10,
        batch_size=25,
        warmup_epochs=2,
        energy_type="logcosh",
        hidden_dims=[32, 16],
        verbose=False
    )
    assert res_nedpp["model_type"] == "ned_shared_private"
    assert res_nedpp["latents"][0].shape == (n, k)
    assert res_nedpp["private_latents"][0].shape == (n, 2)


def test_scenario3_generative_bijective_cross_modality_imputation():
    """
    Scenario 3: Out-of-sample Generative Bijective Imputation.
    Modality 2 is missing on held-out test cohort; synthesize X2 from X1 using Flow-SiMR
    and Woodbury conditional Gaussian inference.
    """
    torch.manual_seed(42)
    n_train = 60
    n_test = 20
    n_total = n_train + n_test
    p1, p2, k = 10, 8, 3

    # Shared generator
    u_all = torch.randn(n_total, k)
    w1 = torch.randn(k, p1)
    w2 = torch.randn(k, p2)

    x1_all = u_all @ w1 + 0.05 * torch.randn(n_total, p1)
    x2_all = u_all @ w2 + 0.05 * torch.randn(n_total, p2)

    # Train / Test partition
    x1_train, x1_test = x1_all[:n_train], x1_all[n_train:]
    x2_train, x2_test = x2_all[:n_train], x2_all[n_train:]

    # Fit Flow-SiMR on training cohort
    res = flow_simr(
        [x1_train, x2_train],
        k=k,
        epochs=10,
        batch_size=20,
        warmup_epochs=2,
        num_layers=2,
        hidden_dim=16,
        verbose=False
    )

    flow_model = res["model"]
    cond = res["cond_inference"]

    # Imputation phase: View 2 is completely missing for the test cohort
    flow_model.eval()
    with torch.no_grad():
        # Encode test View 1 through flow encoder wrapper
        z1_test = flow_model.encoders[0](x1_test)

        # Infer test View 2 latent representation conditionally
        z2_inferred = cond.predict_conditional(observed_idx=0, target_idx=1, observed_z=z1_test)
        assert z2_inferred.shape == (n_test, k)

        # Synthesize View 2 through flow decoder
        x2_synthesized = flow_model.decoders[1](z2_inferred)
        assert x2_synthesized.shape == x2_test.shape

    # Assess synthesis fidelity against true held-out data
    mse_synth = torch.mean((x2_test - x2_synthesized) ** 2).item()
    var_test = torch.var(x2_test).item()
    nmse = mse_synth / (var_test + 1e-8)

    assert nmse < 0.90, f"Expected normalized imputation error < 0.90, got {nmse}"
    rv_synth = adjusted_rvcoef(x2_test, x2_synthesized)
    assert rv_synth > 0.60, f"Expected cross-modality synthesis RV correlation > 0.60, got {rv_synth}"
    assert not torch.isnan(x2_synthesized).any()


def test_scenario4_structural_path_modeling_with_graph_constraints():
    """
    Scenario 4: Multi-stage Developmental / Causal Lineage Path Modeling.
    Genomics (Mod 0) -> Transcriptomics (Mod 1) -> Clinical Phenotype (Mod 2).
    Compares true causal chain against independent model via fit_structural_models and simlr_path.
    """
    torch.manual_seed(42)
    n = 60
    p0, p1, p2 = 14, 20, 8
    k = 2

    # Step-wise signal propagation: Mod 0 drives Mod 1, Mod 1 drives Mod 2
    u0 = torch.randn(n, k)
    x0 = u0 @ torch.randn(k, p0) + 0.1 * torch.randn(n, p0)

    u1 = 0.8 * u0 + 0.2 * torch.randn(n, k)
    x1 = u1 @ torch.randn(k, p1) + 0.1 * torch.randn(n, p1)

    u2 = 0.8 * u1 + 0.2 * torch.randn(n, k)
    x2 = u2 @ torch.randn(k, p2) + 0.1 * torch.randn(n, p2)

    views = [x0, x1, x2]

    # Create topological path graphs
    # True causal chain: 0 <-> 1 and 1 <-> 2
    chain_graph = create_path_graph([(0, 1), (1, 2)], n_modalities=3)
    # Incorrect graph: 0 <-> 2 only (skipping mediator 1)
    skip_graph = create_path_graph([(0, 2)], n_modalities=3)

    models = {
        "true_lineage_chain": chain_graph,
        "incorrect_skip": skip_graph
    }

    results = fit_structural_models(
        views,
        k=k,
        models=models,
        model_type="lend",
        epochs=8,
        batch_size=20,
        warmup_epochs=1,
        hidden_dims=[32, 16],
        verbose=False
    )

    assert len(results["comparison"]) == 2
    comp_dict = {c["model"]: c for c in results["comparison"]}
    assert "true_lineage_chain" in comp_dict
    assert "incorrect_skip" in comp_dict

    # Sequential addition along the lineage using simlr_path
    path_model = [[0, 1], [0, 1, 2]]
    path_res = simlr_path(views, k=k, path_model=path_model, iterations=8)
    assert len(path_res["path_results"]) == 2
    assert "consensus_correlations" in path_res


def test_scenario5_extreme_robustness_outlier_resilience():
    """
    Scenario 5: Extreme Robustness & Outlier Resilience.
    Adversarial setup: Heavy-tailed Cauchy noise, 5% extreme 10x sample outliers,
    collinear feature blocks, and sparse kernel boundary.
    Verifies that Armijo gradient line search and LEND non-negative Stiefel retractions
    maintain bounded defect and prevent gradient explosion.
    """
    torch.manual_seed(42)
    n, p1, p2, k = 50, 16, 14, 2

    # Underlying signal
    u_true = torch.randn(n, k)
    x1 = u_true @ torch.randn(k, p1)
    x2 = u_true @ torch.randn(k, p2)

    # 1. Add heavy-tailed Cauchy noise perturbations
    cauchy_noise1 = torch.distributions.Cauchy(0.0, 0.05).sample((n, p1))
    cauchy_noise2 = torch.distributions.Cauchy(0.0, 0.05).sample((n, p2))
    x1 = x1 + cauchy_noise1.clamp(-2.0, 2.0)
    x2 = x2 + cauchy_noise2.clamp(-2.0, 2.0)

    # 2. Inject 5% extreme outliers (samples 0 and 1 scaled by 10x)
    x1[:2, :] *= 10.0
    x2[:2, :] *= 10.0

    # 3. Create collinear feature block in view 1
    x1[:, 2] = x1[:, 1] * 2.0
    x1[:, 3] = x1[:, 1] * -1.5

    # Fit classical SiMLR with Armijo gradient line search
    res_simlr = simlr(
        [x1, x2],
        k=k,
        iterations=12,
        optimizer_type="armijo_gradient",
        constraint="Stiefel"
    )

    # Verify no NaN explosion despite extreme outliers
    assert not torch.isnan(res_simlr["u"]).any(), "Armijo optimizer produced NaNs under outliers"
    assert not torch.isinf(res_simlr["u"]).any()
    assert not torch.isnan(res_simlr["v"][0]).any()

    # Verify Stiefel invariant defect remains bounded
    defect = invariant_orthogonality_defect(res_simlr["v"][0])
    assert defect < 0.25, f"Expected bounded orthogonality defect < 0.25 under outliers, got {defect}"

    # Test LEND NSA neural model under identical adversarial conditions
    res_lend = lend_simr(
        [x1, x2],
        k=k,
        epochs=8,
        batch_size=20,
        warmup_epochs=1,
        hidden_dims=[32, 16],
        verbose=False
    )
    assert not torch.isnan(res_lend["latents"][0]).any()
    assert not np.isnan(res_lend["loss_history"][-1])
