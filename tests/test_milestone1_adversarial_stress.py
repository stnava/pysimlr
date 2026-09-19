"""Adversarial Stress Test Suite for Milestone 1 Optimizers, Paths, and Contracts.

Targeted audits:
1. ArmijoGradient & BidirectionalArmijoGradient:
   - Objective (cross-view correlation) monotonic increase and step shrinkage under pathological / steep gradients.
   - Gradient direction alignment matching correlation ascent (+grad corr = -grad E).
   - Vanishing / near-zero (< 1e-8) and zero gradients leaving parameters unchanged.
   - BidirectionalArmijoGradient exploration of inverted gradients to decrease energy.
   - Ill-conditioned data execution with ArmijoGradient and BidirectionalArmijoGradient across energy types.
2. simlr_path and permutation_test:
   - Multi-topology stability (star, loo, graph) across varying view counts (2, 3, 5 views).
   - Single-path steps (e.g. [[0, 1]]) and multi-step expanding paths.
   - Permutation tests across all topologies.
   - Empty permutations (n_permutations=0) handling without unhandled traceback crashes.
3. simlr and predict_simlr contracts:
   - Invalid shapes: 1D tensors/arrays, 3D tensors/arrays cleanly raising ValueError.
   - 0 samples and 0 features cleanly raising ValueError.
   - Mismatched sample counts across modalities (N_i != N_j) raising descriptive ValueError.
   - Invalid or non-positive latent dimensions k (k=0, k=-2, k=2.5, k="2") raising ValueError.
   - Non-sequence or empty data_matrices inputs ([], (), None, int) raising ValueError.
   - predict_simlr and predict_shared_latent input contracts (view counts, 1D/3D shapes, mismatched samples, 0 samples).
"""

import pytest
import torch
import numpy as np

from pysimlr import (
    simlr,
    predict_simlr,
    predict_shared_latent,
)
from pysimlr.optimizers import (
    ArmijoGradient,
    BidirectionalArmijoGradient,
    backtracking_linesearch,
)
from pysimlr.paths import simlr_path, permutation_test


# ============================================================================
# 1. ArmijoGradient & BidirectionalArmijoGradient Adversarial Stress Tests
# ============================================================================

def test_stress_armijo_gradient_steep_gradient_shrinkage():
    """Verify ArmijoGradient backtracks to shrink step size on steep quartic energy functions."""
    torch.manual_seed(42)
    P, K = 6, 2
    V = torch.tensor([[1.0, 0.5], [0.5, 1.0], [-0.5, 0.2], [0.1, -0.4], [0.8, -0.2], [-0.3, 0.7]])

    # Quartic bowl with steep barrier: E(V) = ||V||^2 + 100 * ||V||^4
    def energy_fn(v):
        norm_sq = torch.sum(v**2)
        return norm_sq + 100.0 * (norm_sq**2)

    norm_sq_V = torch.sum(V**2)
    grad_E = 2.0 * V + 400.0 * norm_sq_V * V
    descent_grad = -grad_E  # steepest descent direction for energy

    opt = ArmijoGradient("armijo", [V], learning_rate=1.0)
    v_next = opt.step(0, V, descent_grad, energy_fn)

    e_init = energy_fn(V).item()
    e_next = energy_fn(v_next).item()

    # Backtracking must shrink step size so that energy decreases
    assert e_next < e_init, f"Step did not decrease energy: init={e_init}, next={e_next}"
    assert not torch.isnan(v_next).any()
    assert not torch.isinf(v_next).any()


def test_stress_armijo_gradient_pathological_scale_invariance():
    """Verify ArmijoGradient survives huge gradient magnitudes (1e8) without NaNs or divergence."""
    torch.manual_seed(42)
    P, K = 5, 2
    V = torch.randn(P, K)

    def energy_fn(v):
        return torch.sum(v**2)

    # Huge gradient magnitude: 1e8
    huge_grad = -2.0 * V * 1e8
    opt = ArmijoGradient("armijo", [V], learning_rate=1.0)
    v_next = opt.step(0, V, huge_grad, energy_fn)

    assert not torch.isnan(v_next).any(), "ArmijoGradient produced NaNs on 1e8 gradient"
    assert energy_fn(v_next) <= energy_fn(V), "ArmijoGradient increased energy under huge gradient"


def test_stress_armijo_gradient_zero_and_near_zero_gradients():
    """Verify ArmijoGradient returns unchanged parameters on zero or sub-epsilon gradients."""
    P, K = 4, 2
    V = torch.randn(P, K)

    def energy_fn(v):
        return torch.sum(v**2)

    # Fresh optimizer with exact zero gradient
    opt_zero = ArmijoGradient("armijo", [V], learning_rate=0.5, epsilon=1e-8)
    v_zero = opt_zero.step(0, V, torch.zeros(P, K), energy_fn)
    assert torch.equal(v_zero, V), "Zero gradient must not alter parameters"

    # Sub-epsilon gradient (1e-12)
    opt_tiny = ArmijoGradient("armijo", [V], learning_rate=0.5, epsilon=1e-8)
    v_tiny = opt_tiny.step(0, V, torch.randn(P, K) * 1e-12, energy_fn)
    assert torch.equal(v_tiny, V), "Sub-epsilon gradient must not alter parameters"


def test_stress_armijo_gradient_at_strict_local_minimum():
    """Verify ArmijoGradient refuses uphill steps when parameters are already at a local minimum."""
    P, K = 4, 2
    V = torch.randn(P, K)

    # V is exact local minimum (energy increases for any v != V)
    def energy_fn(v):
        return torch.sum((v - V)**2)

    # Perturbed gradient attempting to push away from minimum
    grad = torch.randn(P, K)
    opt = ArmijoGradient("armijo", [V], learning_rate=1.0)
    v_next = opt.step(0, V, grad, energy_fn)

    assert torch.equal(v_next, V), "ArmijoGradient must not take steps that increase energy from minimum"


def test_stress_bidirectional_armijo_inverts_uphill_gradient():
    """Verify BidirectionalArmijoGradient successfully discovers negative direction when gradient points uphill."""
    P, K = 4, 2
    V = torch.tensor([[1.0, 0.5], [0.5, 1.0], [-0.5, 0.2], [0.1, -0.4]])

    # Quadratic bowl: min at 0
    def energy_fn(v):
        return torch.sum(v**2)

    # Uphill gradient (+grad E) pointing away from 0
    uphill_grad = 2.0 * V
    opt = BidirectionalArmijoGradient("bidi_armijo", [V], learning_rate=0.5)
    v_next = opt.step(0, V, uphill_grad, energy_fn)

    e_init = energy_fn(V).item()
    e_next = energy_fn(v_next).item()
    assert e_next < e_init, f"BidirectionalArmijo failed to invert uphill gradient: init={e_init}, next={e_next}"


def test_stress_simlr_armijo_monotonic_correlation_ascent():
    """Verify full SiMLR with armijo_gradient improves cross-view correlation on ill-conditioned data, with a bounded, settling objective.

    Run without the orthogonality constraint: with it, the iteration correctly
    converges to the optimum of E + w*Dtilde, whose correlation energy is
    slightly above the unconstrained optimum the SVD initialisation already
    sits at, so first-vs-last monotonicity of E alone is not a property of the
    constrained method (see docs/theory/SIMLR_THEORY.md S8).
    """
    torch.manual_seed(123)
    n = 60
    # Ill-conditioned data: condition number > 1e4
    u_true = torch.randn(n, 2)
    scales = torch.tensor([1000.0, 100.0, 1.0, 0.1, 0.01])
    x1 = (u_true @ torch.randn(2, 5)) * scales + torch.randn(n, 5) * 0.1
    x2 = (u_true @ torch.randn(2, 6)) + torch.randn(n, 6) * 0.1

    res = simlr([x1, x2], k=2, iterations=12, optimizer_type="armijo_gradient", energy_type="acc",
                constraint="orthox0")  # the optimizer is under test, not the constraint
    energy_history = res["energy"]

    # In acc mode, energy = -sum(|cov|), so decreasing energy == increasing correlation
    assert energy_history[-1] <= energy_history[0], "Overall energy did not decrease across iterations"
    # Strict step-wise monotonicity is NOT a property of this algorithm and is
    # deliberately not asserted. simlr alternates between optimizing each V_i
    # against a fixed u_i and recomputing u from the updated projections; that
    # second step can raise the total, so the joint objective is guaranteed to
    # improve overall, not at every sweep.
    #
    # These tests previously asserted per-step monotonicity, which held only
    # because the objective was diverging: with the basis column scale left
    # free, -sum|U'XV| is unbounded below. The measured trajectory ran
    # -1.0, -1.38, ... -8.11 with *growing* increments at iteration 12 and
    # reached ~-2.6e6 by iteration 40 -- monotone, but because ||V|| was
    # blowing up, not because the correlation was improving. With the columns
    # unit-normalized the energy now converges near -3.5.
    energies = np.asarray(energy_history, dtype=float)
    assert np.all(np.isfinite(energies))
    assert np.abs(energies).max() < 1e3, (
        f"energy reached {np.abs(energies).max():.3e}; the objective is "
        f"diverging rather than converging"
    )
    if len(energies) >= 6:
        tail = energies[len(energies) // 2:]
        assert np.ptp(tail) < 0.5 * abs(float(energies.min())), (
            f"energy still moving by {np.ptp(tail):.4f} over the second half "
            f"of the run; it has not settled"
        )
    assert res["best_energy"] == pytest.approx(float(energies.min()), rel=1e-9)


def test_stress_simlr_bidirectional_armijo_acc_correlation_ascent():
    """Verify full SiMLR with bidirectional_armijo_gradient improves cross-view correlation (acc), with a bounded, settling objective."""
    torch.manual_seed(42)
    n = 45
    u_true = torch.randn(n, 2)
    x1 = u_true @ torch.randn(2, 8) + torch.randn(n, 8) * 0.1
    x2 = u_true @ torch.randn(2, 6) + torch.randn(n, 6) * 0.1

    res = simlr([x1, x2], k=2, iterations=10, optimizer_type="bidirectional_armijo_gradient", energy_type="acc",
                constraint="orthox0")  # the optimizer is under test, not the constraint
    assert not torch.isnan(res["u"]).any()
    energy_history = res["energy"]
    assert energy_history[-1] <= energy_history[0], "Overall energy did not decrease across iterations"
    # Strict step-wise monotonicity is NOT a property of this algorithm and is
    # deliberately not asserted. simlr alternates between optimizing each V_i
    # against a fixed u_i and recomputing u from the updated projections; that
    # second step can raise the total, so the joint objective is guaranteed to
    # improve overall, not at every sweep.
    #
    # These tests previously asserted per-step monotonicity, which held only
    # because the objective was diverging: with the basis column scale left
    # free, -sum|U'XV| is unbounded below. The measured trajectory ran
    # -1.0, -1.38, ... -8.11 with *growing* increments at iteration 12 and
    # reached ~-2.6e6 by iteration 40 -- monotone, but because ||V|| was
    # blowing up, not because the correlation was improving. With the columns
    # unit-normalized the energy now converges near -3.5.
    energies = np.asarray(energy_history, dtype=float)
    assert np.all(np.isfinite(energies))
    assert np.abs(energies).max() < 1e3, (
        f"energy reached {np.abs(energies).max():.3e}; the objective is "
        f"diverging rather than converging"
    )
    if len(energies) >= 6:
        tail = energies[len(energies) // 2:]
        assert np.ptp(tail) < 0.5 * abs(float(energies.min())), (
            f"energy still moving by {np.ptp(tail):.4f} over the second half "
            f"of the run; it has not settled"
        )
    assert res["best_energy"] == pytest.approx(float(energies.min()), rel=1e-9)


# ============================================================================
# 2. simlr_path and permutation_test Multi-Topology Stress Tests
# ============================================================================

@pytest.mark.parametrize("n_views", [2, 3, 5])
@pytest.mark.parametrize("topology", ["star", "loo", "graph"])
def test_stress_simlr_path_topologies_and_views(n_views, topology):
    """Verify simlr_path across star, loo, and graph topologies with 2, 3, and 5 modalities."""
    torch.manual_seed(100 + n_views)
    mats = [torch.randn(25, 4 + i) for i in range(n_views)]

    # Single-step path
    single_path = [list(range(min(2, n_views)))]
    # Multi-step path
    multi_path = [list(range(i + 1)) for i in range(1, n_views)]

    graph = {i: [(i + 1) % n_views, (i - 1) % n_views] for i in range(n_views)}
    kwargs = {"path_graph": graph} if topology == "graph" else {}

    # Single path step
    res_single = simlr_path(data_matrices=mats, path_model=single_path, k=2, iterations=3, topology=topology, **kwargs)
    assert len(res_single["consensus_correlations"]) == 1
    assert not np.isnan(res_single["consensus_correlations"][0])
    assert np.isclose(res_single["consensus_correlations"][0], 1.0, atol=1e-3)

    # Multi-path steps
    res_multi = simlr_path(data_matrices=mats, path_model=multi_path, k=2, iterations=3, topology=topology, **kwargs)
    assert len(res_multi["consensus_correlations"]) == len(multi_path)
    assert not any(np.isnan(res_multi["consensus_correlations"]))
    assert np.isclose(res_multi["consensus_correlations"][-1], 1.0, atol=1e-3)


@pytest.mark.parametrize("n_views", [2, 3])
@pytest.mark.parametrize("topology", ["star", "loo", "graph"])
def test_stress_permutation_test_topologies_and_views(n_views, topology):
    """Verify permutation_test across topologies with 2 and 3 views."""
    torch.manual_seed(200 + n_views)
    mats = [torch.randn(20, 5) for _ in range(n_views)]
    graph = {i: [(i + 1) % n_views, (i - 1) % n_views] for i in range(n_views)}
    kwargs = {"path_graph": graph} if topology == "graph" else {}

    perm_res = permutation_test(data_matrices=mats, k=2, n_permutations=3, iterations=2, topology=topology, **kwargs)
    assert "observed_similarity" in perm_res
    assert "null_similarities" in perm_res
    assert len(perm_res["null_similarities"]) == 3
    assert not np.isnan(perm_res["observed_similarity"])
    assert 0.0 <= perm_res["p_value"] <= 1.0


def test_stress_permutation_test_empty_permutations():
    """Verify permutation_test handles n_permutations=0 without unhandled exception."""
    x1 = torch.randn(15, 4)
    x2 = torch.randn(15, 4)
    # n_permutations=0 should return empty null_similarities and nan p_value cleanly
    res = permutation_test([x1, x2], k=2, n_permutations=0, iterations=2)
    assert res["null_similarities"] == []
    assert not np.isnan(res["observed_similarity"])
    assert np.isnan(res["p_value"])


# ============================================================================
# 3. simlr and predict_simlr Contract Validation Stress Tests
# ============================================================================

def test_stress_contract_simlr_invalid_input_types_and_shapes():
    """Verify simlr raises descriptive ValueError or TypeError on invalid container types and shapes."""
    # 1. Empty data_matrices
    for empty_container in [[], ()]:
        with pytest.raises(ValueError, match=r"non-empty list or tuple"):
            simlr(empty_container, k=2)

    # 2. Non-container types
    for non_container in [None, "string_view", 42, {"a": torch.randn(10, 2)}]:
        with pytest.raises(ValueError, match=r"non-empty list or tuple"):
            simlr(non_container, k=2)

    # 3. Elements not tensors/ndarrays
    with pytest.raises(TypeError, match=r"must be a torch.Tensor or np.ndarray"):
        simlr(["invalid_element_1", "invalid_element_2"], k=2)

    # 4. 1D arrays
    with pytest.raises(ValueError, match=r"must be 2D"):
        simlr([torch.randn(20), torch.randn(20, 5)], k=2)
    with pytest.raises(ValueError, match=r"must be 2D"):
        simlr([np.random.randn(20), torch.randn(20, 5)], k=2)

    # 5. 3D arrays
    with pytest.raises(ValueError, match=r"must be 2D"):
        simlr([torch.randn(20, 5, 2), torch.randn(20, 5)], k=2)
    with pytest.raises(ValueError, match=r"must be 2D"):
        simlr([np.random.randn(20, 5, 2), torch.randn(20, 5)], k=2)

    # 6. Zero samples
    with pytest.raises(ValueError, match=r"0 samples"):
        simlr([torch.empty(0, 5), torch.empty(0, 5)], k=2)

    # 7. Zero features
    with pytest.raises(ValueError, match=r"0 features"):
        simlr([torch.empty(10, 0), torch.empty(10, 5)], k=2)

    # 8. Mismatched sample count across views
    with pytest.raises(ValueError, match=r"identical number of samples|mismatched sample counts"):
        simlr([torch.randn(10, 5), torch.randn(15, 5)], k=2)


def test_stress_contract_simlr_invalid_k_dimensions():
    """Verify simlr raises descriptive ValueError on non-positive, float, or string k."""
    x1 = torch.randn(15, 5)
    x2 = torch.randn(15, 5)

    for invalid_k in [0, -1, -50, 2.5, "2", None]:
        with pytest.raises(ValueError, match=r"positive integer"):
            simlr([x1, x2], k=invalid_k)


def test_stress_contract_predict_simlr_and_predict_shared_latent():
    """Verify predict_simlr and predict_shared_latent enforce strict validation contracts."""
    torch.manual_seed(42)
    x1 = torch.randn(20, 5)
    x2 = torch.randn(20, 6)
    model = simlr([x1, x2], k=2, iterations=2)

    for predict_fn in [predict_simlr, predict_shared_latent]:
        # 1. Empty data_matrices
        for empty_in in [[], ()]:
            with pytest.raises(ValueError, match=r"non-empty list or tuple"):
                predict_fn(empty_in, model)

        # 2. Non-container types
        for non_container in [None, "invalid_view", 100]:
            with pytest.raises(ValueError, match=r"non-empty list or tuple"):
                predict_fn(non_container, model)

        # 3. View count mismatch
        with pytest.raises(ValueError, match=r"Expected 2 data matrices"):
            predict_fn([x1], model)
        with pytest.raises(ValueError, match=r"Expected 2 data matrices"):
            predict_fn([x1, x2, torch.randn(20, 4)], model)

        # 4. 1D view
        with pytest.raises(ValueError, match=r"must be 2D"):
            predict_fn([torch.randn(20), x2], model)

        # 5. 3D view
        with pytest.raises(ValueError, match=r"must be 2D"):
            predict_fn([torch.randn(20, 5, 2), x2], model)

        # 6. Mismatched samples at predict time
        with pytest.raises(ValueError, match=r"identical number of samples|mismatched sample counts"):
            predict_fn([torch.randn(12, 5), torch.randn(18, 6)], model)

    # 7. Predict with 0 samples succeeds cleanly returning empty tensors
    pred_0 = predict_simlr([torch.randn(0, 5), torch.randn(0, 6)], model)
    assert pred_0["u"].shape == (0, 2)
    assert pred_0["latents"][0].shape == (0, 2)
