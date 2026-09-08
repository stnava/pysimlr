"""Tier 3: Pairwise and Cross-Feature Combinations for pysimlr public APIs.

Exercises:
- Flows + Newton/SVD consensus topologies.
- LEND neural architecture + Topological graph constraints.
- NED neural architecture + Structural equation path modeling.
- Data whitening + Flow-SiMR cross-view conditional synthesis.
- Multiscale SVD geometric features + Classical SiMLR alignment.
- Flow-SiMR-V expanded bottleneck + Dynamic modality weights.
- Classical SiMLR + Leave-One-Out (LOO) consensus + Armijo gradient line search.
- NED neural architecture + Log-Cosh energy objective + Variance pruning.
- Whitened representations + Sparse k-NN affinity distance matrix.
- Sequential path exploration (simlr_path) + Permutation statistical testing.
- Deep SiMR wrapper + Comprehensive interpretability contract.
- Classical SiMLR projections + Neuroimaging NNH DataFrame feature extension.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from pysimlr import (
    simlr,
    deep_simr,
    lend_simr,
    ned_simr,
    flow_simr,
    flow_simr_v,
    whiten_matrix,
    multiscale_svd,
    sparse_distance_matrix,
    fit_structural_models,
    create_path_graph,
    simlr_path,
    permutation_test,
    apply_simlr_matrices,
    FlowConditionalInference,
    adjusted_rvcoef,
)


def test_combo_flow_simr_with_newton_and_svd_consensus(synthetic_two_views):
    """Verify Flow-SiMR couples with diverse consensus algorithms (Newton vs SVD)."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = 2

    # Newton consensus
    res_newton = flow_simr(
        [x1, x2],
        k=k,
        epochs=5,
        batch_size=20,
        warmup_epochs=1,
        mixing_algorithm="newton",
        verbose=False
    )
    assert res_newton["latents"][0].shape == (synthetic_two_views["n"], k)
    assert not torch.isnan(res_newton["latents"][0]).any()

    # SVD consensus
    res_svd = flow_simr(
        [x1, x2],
        k=k,
        epochs=5,
        batch_size=20,
        warmup_epochs=1,
        mixing_algorithm="svd",
        verbose=False
    )
    assert res_svd["latents"][0].shape == (synthetic_two_views["n"], k)
    assert not torch.isnan(res_svd["latents"][0]).any()


def test_combo_lend_with_graph_topology(synthetic_three_views):
    """Verify LEND neural architecture respects topological graph constraints across 3 views."""
    views = synthetic_three_views["views"]
    k = 2
    # Chain graph: 0 <-> 1 <-> 2
    graph = {0: [1], 1: [0, 2], 2: [1]}

    res = lend_simr(
        views,
        k=k,
        epochs=8,
        batch_size=25,
        warmup_epochs=2,
        topology="graph",
        path_graph=graph,
        hidden_dims=[32, 16],
        verbose=False
    )

    assert res["model_type"] == "lend_simr"
    assert len(res["latents"]) == 3
    # Latents should be aligned to neighbors
    assert res["latents"][0].shape == (synthetic_three_views["n"], k)


def test_combo_ned_with_structural_models(synthetic_three_views):
    """Verify fit_structural_models evaluates structural path graph hypotheses using NED."""
    views = synthetic_three_views["views"]
    k = 2

    models = {
        "chain_hypothesis": {0: [1], 1: [0, 2], 2: [1]},
        "star_hypothesis": {0: [1, 2], 1: [0], 2: [0]}
    }

    results = fit_structural_models(
        views,
        k=k,
        models=models,
        model_type="ned",
        epochs=6,
        batch_size=25,
        warmup_epochs=1,
        hidden_dims=[32, 16],
        verbose=False
    )

    assert "models" in results
    assert "comparison" in results
    assert len(results["comparison"]) == 2
    for comp in results["comparison"]:
        assert "model" in comp
        assert "total_loss" in comp
        assert "recon_loss" in comp
        assert "sim_loss" in comp


def test_combo_whitened_inputs_to_flow_simr(synthetic_two_views):
    """Verify multi-stage pipeline: Whiten -> Flow-SiMR -> Conditional Synthesis."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]

    # 1. Whitening stage
    w1 = whiten_matrix(x1, nc=synthetic_two_views["p1"])["whitened_matrix"]
    w2 = whiten_matrix(x2, nc=synthetic_two_views["p2"])["whitened_matrix"]

    # 2. Flow-SiMR stage
    res = flow_simr(
        [w1, w2],
        k=2,
        epochs=6,
        batch_size=20,
        warmup_epochs=1,
        num_layers=2,
        hidden_dim=16,
        verbose=False
    )

    # 3. Conditional Inference stage
    cond = res["cond_inference"]
    z1 = res["latents"][0]
    z2_pred = cond.predict_conditional(0, 1, z1)
    assert z2_pred.shape == (synthetic_two_views["n"], 2)

    w2_synth = res["model"].decoders[1](z2_pred)
    assert w2_synth.shape == w2.shape
    assert not torch.isnan(w2_synth).any()


def test_combo_multiscale_svd_features_into_simlr():
    """Verify multi-scale singular value profiles can be aligned across domains with SiMLR."""
    torch.manual_seed(42)
    n = 40
    # Two distinct spatial/feature domains
    x_dom1 = torch.randn(n, 12)
    x_dom2 = torch.randn(n, 14)

    r = torch.tensor([0.5, 1.0, 2.0])
    nev = 4

    # Extract multi-scale singular value features for each domain
    feat1 = multiscale_svd(x_dom1, r=r, locn=n, nev=nev, knn=0)["evals_vs_scale"]
    feat2 = multiscale_svd(x_dom2, r=r, locn=n, nev=nev, knn=0)["evals_vs_scale"]

    # Align scale profiles using SiMLR
    res = simlr([feat1, feat2], k=2, iterations=8)
    assert res["u"].shape == (len(r), 2)
    assert res["v"][0].shape == (nev, 2)


def test_combo_flow_simr_v_with_dynamic_weights(synthetic_two_views):
    """Verify Flow-SiMR-V with dynamic MAI modality weighting runs stably."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]

    res = flow_simr_v(
        [x1, x2],
        k=4,
        epochs=5,
        batch_size=20,
        warmup_epochs=1,
        dynamic_weights=True,
        mai_metric="procrustes_r2",
        verbose=False
    )

    assert res["model"] is not None
    assert res["latents"][0].shape == (synthetic_two_views["n"], 4)
    assert len(res["reconstructions"]) == 2


def test_combo_simlr_with_loo_consensus_and_armijo(synthetic_three_views):
    """Verify SiMLR with LOO topology and Armijo gradient line search optimizer."""
    views = synthetic_three_views["views"]
    k = 2

    res = simlr(
        views,
        k=k,
        topology="loo",
        optimizer_type="armijo_gradient",
        iterations=10
    )

    assert "u" in res
    assert isinstance(res["u"], list)
    assert len(res["u"]) == 3
    for u_i in res["u"]:
        assert u_i.shape == (synthetic_three_views["n"], k)


def test_combo_ned_with_logcosh_and_pruning(synthetic_two_views):
    """Verify NED architecture with Log-Cosh energy objective and variance-weighted pruning."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]

    res = ned_simr(
        [x1, x2],
        k=2,
        epochs=6,
        batch_size=20,
        warmup_epochs=1,
        energy_type="logcosh",
        prune_threshold=0.2,
        hidden_dims=[32, 16],
        verbose=False
    )

    assert res["model_type"] == "ned_simr"
    assert len(res["loss_history"]) == 6
    assert not np.isnan(res["loss_history"][-1])


def test_combo_whitened_data_with_sparse_distance_matrix():
    """Verify whitened features fed into sparse_distance_matrix yield well-conditioned affinity."""
    torch.manual_seed(42)
    x = torch.randn(50, 8)

    whitened = whiten_matrix(x, nc=6)["whitened_matrix"]
    smat = sparse_distance_matrix(whitened, k=5, sigma=1.0)

    assert smat.shape == (50, 50)
    assert torch.all(smat >= 0.0)
    # Diagonal affinity of Gaussian kernel with zero distance should be 1.0
    diag = torch.diag(smat)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-3)


def test_combo_simlr_path_with_permutation_test(synthetic_three_views):
    """Verify simlr_path sequential exploration combined with permutation significance test."""
    views = synthetic_three_views["views"]
    k = 2

    # Sequential addition: [view0, view1] -> [view0, view1, view2]
    path_model = [[0, 1], [0, 1, 2]]
    path_res = simlr_path(views, k=k, path_model=path_model, iterations=6)

    assert "path_results" in path_res
    assert len(path_res["path_results"]) == 2

    # Permutation significance testing
    perm = permutation_test(views, k=k, n_permutations=3)
    assert "observed_similarity" in perm
    assert "null_similarities" in perm
    assert "p_value" in perm
    assert 0.0 <= perm["p_value"] <= 1.0


def test_combo_deep_simr_with_interpretability_pipeline(synthetic_two_views):
    """Verify deep_simr end-to-end training generates complete interpretability contract."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]

    res = deep_simr(
        [x1, x2],
        k=2,
        epochs=6,
        batch_size=20,
        warmup_epochs=1,
        verbose=False
    )

    assert "interpretability" in res
    assert "first_layer" in res
    assert "deep_layer_alignment" in res["interpretability"]
    assert "shared_to_first_layer" in res["interpretability"]


def test_combo_nnh_apply_simlr_matrices(synthetic_two_views):
    """Verify applying learned SiMLR basis matrices to new data in pandas DataFrame."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = 2

    res = simlr([x1, x2], k=k, iterations=10)

    # Convert test samples to DataFrame
    p1 = synthetic_two_views["p1"]
    p2 = synthetic_two_views["p2"]
    feature_names_1 = [f"mod1_feat_{i}" for i in range(p1)]
    feature_names_2 = [f"mod2_feat_{i}" for i in range(p2)]

    df = pd.DataFrame(
        torch.cat([x1, x2], dim=1).numpy(),
        columns=feature_names_1 + feature_names_2
    )

    extended_df, added_names = apply_simlr_matrices(
        df,
        simlr_v=res["v"],
        feature_names=[feature_names_1, feature_names_2],
        modality_names=["mod1", "mod2"]
    )

    assert isinstance(extended_df, pd.DataFrame)
    assert len(added_names) > 0
    # Check that projected columns are added
    for col in added_names:
        assert col in extended_df.columns
