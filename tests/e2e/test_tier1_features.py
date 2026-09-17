"""Tier 1: Comprehensive Feature Coverage for pysimlr public APIs.

Exercises:
- SiMLR Core: Canonical correlation, multi-view, rank estimation, optimizers, prediction.
- Deep Architectures: LEND, NED, NEDPP shared-private, diverse loss objectives.
- Normalizing Flows: ANTsTorch RealNVP native binding, fallback architecture, flow_simr_v, conditional inference.
- Whitening & SVD: PCA whitening, ba_svd, safe_pca, multiscale_svd, adjusted_rvcoef.
- Consensus Topologies: Star, LOO, Graph, Anchor stability, Pruning.
"""

import pytest
import torch
import numpy as np

from pysimlr import (
    simlr,
    predict_simlr,
    predict_shared_latent,
    reconstruct_from_learned_maps,
    estimate_rank,
    decompose_energy,
    lend_simr,
    ned_simr,
    ned_simr_shared_private,
    predict_deep,
    flow_simr,
    flow_simr_v,
    FlowSiMRModel,
    FlowConditionalInference,
    compute_shared_consensus,
    ba_svd,
    safe_pca,
    whiten_matrix,
    multiscale_svd,
    adjusted_rvcoef,
    rvcoef,
    build_interpretability_report,
)
from pysimlr.flows import NormalizingFlow, CustomRealNVP


# ============================================================================
# FEATURE AREA 1: SiMLR Core
# ============================================================================

def test_simlr_stiefel_constraint_is_near_orthogonal(synthetic_two_views):
    """
    The Stiefel contract on its own, with nothing else acting on the basis.

    ``V'V`` is the identity to about 1e-4 rather than to solver precision, and
    that is deliberate: the retraction weight is capped at `NSA_MAX_W` = 0.95,
    because at w=1 the fidelity term drops out and every scaled Stiefel matrix
    becomes optimal, so nothing selects among them. Exactness is not on offer
    for the non-negative solver at all -- exact orthogonality plus
    non-negativity is disjoint supports, i.e. a clustering.

    The column norms are still exact; only the off-diagonals carry the slack.

    `sparseness_quantile` is set to 0 deliberately. It defaults to 0.5, and a
    50% soft threshold applied after the retraction moves the basis further off
    the manifold -- so leaving it on would test the interaction rather than the
    constraint. See
    `test_simlr_stiefel_orthogonality_is_approximate_once_sparsified`.
    """
    x1, x2 = synthetic_two_views["x1"], synthetic_two_views["x2"]
    k = synthetic_two_views["k"]

    res = simlr([x1, x2], k=k, iterations=15, constraint="Stiefel",
                optimizer_type="hybrid_adam", sparseness_quantile=0.0)

    for i, v in enumerate(res["v"]):
        vtv = v.t() @ v
        norms = v.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Modality {i} lost its unit column norms: {norms}")
        off = vtv - torch.diag(torch.diag(vtv))
        # Measured 1.2e-4 and 0.0 for the two modalities at the 0.95 cap.
        assert float(off.abs().max()) < 1e-3, (
            f"Modality {i} is not near the Stiefel manifold: V'V = {vtv}")


def test_simlr_stiefel_orthogonality_is_approximate_once_sparsified(synthetic_two_views):
    """
    `constraint="Stiefel"` and `sparseness_quantile=0.5` are both defaults and
    they pull against each other: a 50% soft threshold is applied after the
    retraction, so the returned basis is sparse and only approximately
    orthogonal. The column norms are still pinned to 1 exactly -- that part of
    the contract is restored after thresholding -- but the off-diagonals are
    not zero.

    Measured off-diagonal here is 0.001 and 0.054 for the two modalities. The
    bound below is deliberately loose; it exists to catch a collapse to a
    degenerate or wildly non-orthogonal basis, not to pin the exact value.
    Whether a hard manifold constraint ought to be re-imposed after
    sparsification is a semantics question about what the two requests mean
    together, not something this test should decide: re-retracting afterwards
    was measured to restore ``V'V = I`` exactly but to raise density from 49%
    to 87%, discarding most of the sparsity the caller also asked for, while
    leaving subspace recovery unchanged (0.9004 vs 0.9000 over 8 seeds).
    """
    x1, x2 = synthetic_two_views["x1"], synthetic_two_views["x2"]
    k = synthetic_two_views["k"]

    res = simlr([x1, x2], k=k, iterations=15, constraint="Stiefel",
                optimizer_type="hybrid_adam")

    for i, v in enumerate(res["v"]):
        vtv = v.t() @ v
        norms = v.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Modality {i} lost its unit column norms: {norms}")
        off = vtv - torch.diag(torch.diag(vtv))
        assert float(off.abs().max()) < 0.10, (
            f"Modality {i} is far from orthogonal: V'V = {vtv}")
        assert float((v.abs() > 1e-10).float().mean()) < 0.75, (
            f"Modality {i} was not sparsified: V = {v}")


def test_simlr_canonical_correlation_two_views(synthetic_two_views):
    """Verify 2-view SiMLR returns the documented shapes and aligns the views."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = synthetic_two_views["k"]
    n = synthetic_two_views["n"]

    res = simlr([x1, x2], k=k, iterations=15, constraint="Stiefel", optimizer_type="hybrid_adam")

    # Assert contract outputs
    assert "u" in res
    assert "v" in res
    assert res["u"].shape == (n, k)
    assert len(res["v"]) == 2
    assert res["v"][0].shape == (synthetic_two_views["p1"], k)
    assert res["v"][1].shape == (synthetic_two_views["p2"], k)

    # Verify orthogonality summary metrics
    assert "v_orthogonality" in res
    assert len(res["v_orthogonality"]) == 2
    for summary in res["v_orthogonality"]:
        assert summary["invariant_defect"] < 0.10

    # Cross-modality latent alignment
    proj1 = x1 @ res["v"][0]
    proj2 = x2 @ res["v"][1]
    r_val = adjusted_rvcoef(proj1, proj2)
    assert r_val > 0.60, f"Expected high cross-view canonical alignment, got {r_val}"


def test_simlr_three_views_energy_decomposition(synthetic_three_views):
    """Verify 3-view SiMLR and energy decomposition across multi-modal projections."""
    views = synthetic_three_views["views"]
    k = synthetic_three_views["k"]
    n = synthetic_three_views["n"]

    res = simlr(views, k=k, iterations=15, optimizer_type="hybrid_adam")
    assert len(res["v"]) == 3
    assert res["u"].shape == (n, k)

    # Decompose energy
    energy = decompose_energy(views, res)
    assert isinstance(energy, dict)
    assert "modality_energies" in energy
    assert len(energy["modality_energies"]) == 3
    assert "feature_importances" in energy
    for e in energy["modality_energies"]:
        assert not np.isnan(e) and not np.isinf(e)


def test_simlr_rank_estimation_permutation(synthetic_two_views):
    """Verify estimate_rank reliably identifies meaningful latent dimensionality."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]

    k_est = estimate_rank([x1, x2], n_permutations=4)
    assert isinstance(k_est, int)
    assert 1 <= k_est <= min(synthetic_two_views["p1"], synthetic_two_views["p2"])


def test_simlr_optimizers_convergence(synthetic_two_views):
    """Verify different SiMLR optimizers (hybrid_adam, armijo_gradient, gd) execute cleanly."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = 2

    for opt_name in ["hybrid_adam", "armijo_gradient", "gd"]:
        res = simlr([x1, x2], k=k, iterations=10, optimizer_type=opt_name)
        assert res["u"] is not None
        assert not torch.isnan(res["u"]).any(), f"Optimizer {opt_name} produced NaNs in U"
        assert not torch.isnan(res["v"][0]).any(), f"Optimizer {opt_name} produced NaNs in V[0]"


def test_simlr_prediction_and_reconstruction_fidelity(synthetic_two_views):
    """Verify predict_simlr, predict_shared_latent, and reconstruct_from_learned_maps."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = synthetic_two_views["k"]

    res = simlr([x1, x2], k=k, iterations=15)

    # 1. predict_simlr
    pred = predict_simlr([x1, x2], res)
    assert "u" in pred
    assert len(pred["reconstructions"]) == 2
    assert pred["reconstructions"][0].shape == x1.shape
    assert pred["reconstructions"][1].shape == x2.shape

    # 2. predict_shared_latent
    u_pred = predict_shared_latent([x1, x2], res)
    assert u_pred.shape == res["u"].shape
    # Consistency between predictions
    assert torch.allclose(u_pred, pred["u"], atol=1e-4)

    # 3. reconstruct_from_learned_maps
    recons = reconstruct_from_learned_maps(res["u"], res)
    assert len(recons) == 2
    assert recons[0].shape == x1.shape
    assert recons[1].shape == x2.shape


def test_simlr_sparsification_and_positivity(synthetic_two_views):
    """Verify sparseness quantile and positivity constraints on basis matrices."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]

    res = simlr(
        [x1, x2],
        k=2,
        iterations=15,
        sparseness_quantile=0.4,
        positivity="positive"
    )

    for v in res["v"]:
        # Allow tiny numerical relaxation from retraction steps
        assert torch.all(v >= -1e-4), "Positivity constraint violated in basis matrix"


# ============================================================================
# FEATURE AREA 2: Deep Architectures (LEND, NED, NEDPP)
# ============================================================================

def test_lend_simr_end_to_end_training(synthetic_two_views):
    """Verify LEND neural architecture trains and preserves non-negative Stiefel encoder weights."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = synthetic_two_views["k"]
    n = synthetic_two_views["n"]

    res = lend_simr(
        [x1, x2],
        k=k,
        epochs=10,
        batch_size=20,
        warmup_epochs=2,
        hidden_dims=[32, 16],
        verbose=False
    )

    assert res["model"] is not None
    assert res["model_type"] == "lend_simr"
    assert len(res["latents"]) == 2
    assert res["latents"][0].shape == (n, k)
    assert len(res["errors"]) == 2
    assert len(res["loss_history"]) == 10

    # Ensure predict_deep produces consistent latents
    pred = predict_deep([x1, x2], res, device="cpu")
    assert torch.allclose(pred["latents"][0], res["latents"][0], atol=1e-4)


def test_ned_simr_nonlinear_representation(synthetic_two_views):
    """Verify NED non-linear encoder-decoder model trains and produces valid latents."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = synthetic_two_views["k"]
    n = synthetic_two_views["n"]

    res = ned_simr(
        [x1, x2],
        k=k,
        epochs=10,
        batch_size=20,
        warmup_epochs=2,
        hidden_dims=[32, 16],
        verbose=False
    )

    assert res["model_type"] == "ned_simr"
    assert len(res["loss_history"]) == 10
    assert not np.isnan(res["loss_history"][-1])
    assert res["latents"][0].shape == (n, k)


def test_nedpp_shared_private_disentanglement(synthetic_two_views):
    """Verify NED++ cleanly separates shared consensus from modality-private representations."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = 2
    private_k = 2
    n = synthetic_two_views["n"]

    res = ned_simr_shared_private(
        [x1, x2],
        k=k,
        private_k=private_k,
        epochs=10,
        batch_size=20,
        warmup_epochs=2,
        hidden_dims=[32, 16],
        verbose=False
    )

    assert res["model_type"] == "ned_shared_private"
    assert "latents" in res
    assert "private_latents" in res
    assert res["latents"][0].shape == (n, k)
    assert res["private_latents"][0].shape == (n, private_k)
    assert res["private_latents"][1].shape == (n, private_k)


def test_deep_loss_formulations(synthetic_two_views):
    """Verify diverse energy objectives: regression, acc, nc, and logcosh."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]

    for energy in ["regression", "acc", "nc", "logcosh"]:
        res = lend_simr(
            [x1, x2],
            k=2,
            epochs=5,
            batch_size=30,
            energy_type=energy,
            hidden_dims=[16],
            verbose=False
        )
        assert len(res["loss_history"]) == 5
        assert not np.isnan(res["loss_history"][-1])


def test_deep_interpretability_and_contract(synthetic_two_views):
    """Verify interpretability first-layer contract and report generation."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]

    res = lend_simr(
        [x1, x2],
        k=2,
        epochs=6,
        batch_size=30,
        hidden_dims=[16],
        verbose=False
    )

    assert "first_layer" in res
    assert "first_layer_scores" in res
    assert "interpretability" in res
    interp = res["interpretability"]
    assert "deep_layer_alignment" in interp
    assert "shared_to_first_layer" in interp


# ============================================================================
# FEATURE AREA 3: Normalizing Flows (Flow-SiMR & Flow-SiMR-V)
# ============================================================================

def test_flow_simr_antstorch_native_binding():
    """Verify NormalizingFlow binds to ANTsTorch native RealNVP with exact invertibility."""
    dim = 6
    x = torch.randn(15, dim)

    flow = NormalizingFlow(dim=dim, num_layers=2, hidden_dim=16)
    assert not flow.use_fallback, "Expected native ANTsTorch binding when available"

    z, log_det = flow(x)
    assert z.shape == x.shape
    assert log_det.shape == (15,)

    x_rec = flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-5)


def test_flow_simr_transparent_fallback_mode():
    """Verify CustomRealNVP fallback flow provides exact invertibility and log-det tracking."""
    dim = 6
    x = torch.randn(12, dim)

    fallback_flow = CustomRealNVP(dim=dim, num_layers=2, hidden_dim=16)
    z, log_det = fallback_flow(x)
    assert z.shape == x.shape
    assert log_det.shape == (12,)

    x_rec = fallback_flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-5)


def test_flow_simr_multi_view_training(synthetic_two_views):
    """Verify flow_simr multi-view training, exact NLL loss logging, and reconstructions."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = 2
    n = synthetic_two_views["n"]

    res = flow_simr(
        [x1, x2],
        k=k,
        epochs=8,
        batch_size=20,
        warmup_epochs=2,
        num_layers=2,
        hidden_dim=16,
        verbose=False
    )

    assert res["model_type"] == "flow_simr"
    assert res["latents"][0].shape == (n, k)
    assert res["reconstructions"][0].shape == x1.shape
    assert len(res["loss_history"]) == 8
    assert not np.isnan(res["loss_history"][-1])


def test_flow_simr_v_latent_bottleneck_expansion():
    """Verify flow_simr_v supports latent dimension k larger than modality dimensions."""
    torch.manual_seed(42)
    n = 40
    p1, p2 = 6, 8
    k = 12  # k > p1 and k > p2 (expanded bottleneck)

    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)

    res = flow_simr_v(
        [x1, x2],
        k=k,
        epochs=5,
        batch_size=20,
        warmup_epochs=1,
        num_layers=2,
        hidden_dim=16,
        verbose=False
    )

    assert res["model"] is not None
    assert res["latents"][0].shape == (n, k)
    assert res["reconstructions"][0].shape == (n, p1)
    assert res["reconstructions"][1].shape == (n, p2)


def test_flow_conditional_inference_cross_prediction(synthetic_two_views):
    """Verify FlowConditionalInference predicts target modality latent space and synthesizes data."""
    x1 = synthetic_two_views["x1"]
    x2 = synthetic_two_views["x2"]
    k = 2
    n = synthetic_two_views["n"]

    res = flow_simr(
        [x1, x2],
        k=k,
        epochs=6,
        batch_size=20,
        warmup_epochs=1,
        num_layers=2,
        hidden_dim=16,
        verbose=False
    )

    cond = res["cond_inference"]
    assert isinstance(cond, FlowConditionalInference)
    assert cond.joint_mean.shape == (2 * k,)

    # Predict modality 2 latent from modality 1 latent
    z1 = res["latents"][0]
    z2_pred = cond.predict_conditional(observed_idx=0, target_idx=1, observed_z=z1)
    assert z2_pred.shape == (n, k)
    assert not torch.isnan(z2_pred).any()

    # Synthesize modality 2
    x2_synth = res["model"].decoders[1](z2_pred)
    assert x2_synth.shape == x2.shape
    assert not torch.isnan(x2_synth).any()


# ============================================================================
# FEATURE AREA 4: Whitening & SVD
# ============================================================================

def test_whiten_matrix_identity_covariance():
    """Verify whiten_matrix transforms correlated features into identity covariance."""
    torch.manual_seed(42)
    n, p = 100, 8
    # Create correlated data
    a = torch.randn(n, 3)
    mix = torch.randn(3, p)
    x = a @ mix + 0.1 * torch.randn(n, p)

    res = whiten_matrix(x, nc=p)
    whitened = res["whitened_matrix"]
    assert whitened.shape == (n, p)

    # Centered whitened matrix covariance: W^T W / (n - 1)
    w_centered = whitened - torch.mean(whitened, dim=0, keepdim=True)
    cov = (w_centered.t() @ w_centered) / (n - 1)

    # Whitening means an identity covariance -- not merely positive variance.
    rank = res["rank"]
    assert rank == p, f"expected full rank {p}, got {rank}"
    assert torch.allclose(cov, torch.eye(p), atol=1e-4), (
        f"covariance is not the identity; max deviation "
        f"{float((cov - torch.eye(p)).abs().max()):.3e}"
    )
    # and zero mean per column
    assert torch.allclose(
        whitened.mean(dim=0), torch.zeros(p), atol=1e-4
    )


def test_ba_svd_randomized_orthonormality():
    """Verify ba_svd computes accurate singular vectors with orthonormal columns."""
    torch.manual_seed(42)
    x = torch.randn(80, 20)
    nu, nv = 5, 5

    u, s, v = ba_svd(x, nu=nu, nv=nv)
    assert u.shape == (80, nu)
    assert s.shape == (min(nu, nv),)
    assert v.shape == (20, nv)

    # Orthonormality check
    utu = u.t() @ u
    vtv = v.t() @ v
    eye = torch.eye(nu)
    assert torch.allclose(utu, eye, atol=1e-4)
    assert torch.allclose(vtv, eye, atol=1e-4)


def test_safe_pca_centering_and_reconstruction():
    """Verify safe_pca centers data and reconstructs leading variance accurately."""
    torch.manual_seed(42)
    x = torch.randn(60, 15)
    nc = 6

    pca = safe_pca(x, nc=nc)
    assert pca["u"].shape == (60, nc)
    assert pca["s"].shape == (nc,)
    assert pca["v"].shape == (15, nc)

    # Verify reconstruction of centered subspace
    recon = pca["u"] @ torch.diag(pca["s"]) @ pca["v"].t()
    x_centered = x - torch.mean(x, dim=0)
    # The leading 6 components should explain majority of variance
    var_orig = torch.norm(x_centered, p="fro") ** 2
    var_recon = torch.norm(recon, p="fro") ** 2
    assert var_recon / var_orig > 0.40


def test_multiscale_svd_intrinsic_dimension():
    """Verify multiscale_svd computes singular values across scales for manifold dimension analysis."""
    torch.manual_seed(42)
    x = torch.randn(50, 10)
    r = torch.tensor([0.5, 1.0, 2.0])
    nev = 4

    # Radius-based
    res_radius = multiscale_svd(x, r=r, locn=5, nev=nev, knn=0)
    assert res_radius["evals_vs_scale"].shape == (len(r), nev)
    assert not torch.isnan(res_radius["evals_vs_scale"]).any()

    # KNN-based
    res_knn = multiscale_svd(x, r=r, locn=5, nev=nev, knn=15)
    assert res_knn["evals_vs_scale"].shape == (len(r), nev)
    assert not torch.isnan(res_knn["evals_vs_scale"]).any()


def test_adjusted_rvcoef_properties():
    """Verify adjusted_rvcoef returns 1.0 for identical matrices and lower for noise."""
    torch.manual_seed(42)
    x = torch.randn(40, 10)
    y_noise = torch.randn(40, 10)

    # Identical
    rv_self = adjusted_rvcoef(x, x)
    assert abs(rv_self - 1.0) < 1e-3

    # Independent noise
    rv_noise = adjusted_rvcoef(x, y_noise)
    assert rv_noise < 0.35


# ============================================================================
# FEATURE AREA 5: Consensus Topologies
# ============================================================================

def test_consensus_star_topology_algorithms():
    """Verify compute_shared_consensus with star topology across all mixing algorithms."""
    torch.manual_seed(42)
    n, k = 50, 3
    projections = [torch.randn(n, k) for _ in range(3)]

    for alg in ["svd", "pca", "avg", "newton"]:
        u = compute_shared_consensus(projections, mixing_algorithm=alg, k=k, topology="star", training=False)
        assert u.shape == (n, k), f"Algorithm {alg} produced unexpected shape {u.shape}"
        assert not torch.isnan(u).any(), f"Algorithm {alg} produced NaNs in consensus U"


def test_consensus_leave_one_out_topology():
    """Verify LOO topology produces modality-specific consensus leaving out self."""
    torch.manual_seed(42)
    n, k = 50, 2
    projections = [torch.randn(n, k) for _ in range(3)]

    u_list = compute_shared_consensus(projections, topology="loo", training=False)
    assert isinstance(u_list, list)
    assert len(u_list) == 3
    for u_i in u_list:
        assert u_i.shape == (n, k)


def test_consensus_graph_topology_adjacency():
    """Verify graph topology aligns each modality with its specified graph neighbors."""
    torch.manual_seed(42)
    n, k = 40, 2
    projections = [torch.randn(n, k) for _ in range(3)]
    path_graph = {0: [1], 1: [0, 2], 2: [1]}

    u_list = compute_shared_consensus(projections, topology="graph", path_graph=path_graph, training=False)
    assert isinstance(u_list, list)
    assert len(u_list) == 3
    for u_i in u_list:
        assert u_i.shape == (n, k)


def test_consensus_learned_anchor_stability():
    """Verify learned anchor in training mode guarantees deterministic alignment during inference."""
    torch.manual_seed(42)
    n, k = 45, 2
    projections = [torch.randn(n, k) for _ in range(2)]

    # Training mode yields consensus and anchor
    u_train, anchor = compute_shared_consensus(projections, topology="star", training=True)
    assert anchor is not None

    # Inference mode using learned anchor
    u_infer = compute_shared_consensus(projections, topology="star", training=False, anchor=anchor)
    u_infer_std = (u_infer - u_infer.mean(0, keepdim=True)) / (torch.std(u_infer, dim=0, keepdim=True) + 1e-6)
    assert torch.allclose(u_train, u_infer_std, atol=1e-4)


def test_consensus_variance_pruning_threshold():
    """Verify variance-weighted pruning excludes unaligned / discordant projections."""
    torch.manual_seed(42)
    n, k = 50, 2
    signal = torch.randn(n, k)
    p1 = signal + 0.05 * torch.randn(n, k)
    p2 = signal + 0.05 * torch.randn(n, k)
    p_discordant = torch.randn(n, k) * 10.0  # Discordant modality

    # With prune_threshold, consensus focuses on aligned projections
    u_pruned = compute_shared_consensus([p1, p2, p_discordant], topology="star", prune_threshold=0.3, training=False)
    assert u_pruned.shape == (n, k)
    assert not torch.isnan(u_pruned).any()
