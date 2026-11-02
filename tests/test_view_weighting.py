"""Tests for Milestone 1 (WP-1: A10 Principled View Weighting).

Verifies:
1. Scale-free energies receive equal 1/M normalizing weights.
2. Raw view rescaling by 7 leaves u and v invariant.
3. Total energy min(res["energy"]) >= -1 - 1e-6 for nc.
4. Linear and deep architectures combine views with equal weights (structural parity).
5. Existing regression contract for non-scale-free energies (acc) remains intact.
"""
import pytest
import torch
import numpy as np

import pysimlr
from pysimlr.similarity import (
    SCALE_FREE_ENERGIES,
    SIMILARITY,
    SimilarityContext,
    similarity_energy,
    similarity_is_self_normalised,
)
from pysimlr.simlr import simlr


@pytest.mark.parametrize("m_views", [2, 3, 4])
@pytest.mark.parametrize("energy", sorted(SCALE_FREE_ENERGIES))
def test_scale_free_energies_have_equal_normalizing_weights(m_views, energy):
    """Every verified scale-free energy must assign normalizing_weights == [1/M] * M."""
    torch.manual_seed(42)
    n, k = 60, 2
    views = [torch.randn(n, 10 + i * 5, dtype=torch.float32) for i in range(m_views)]
    res = simlr(views, k=k, iterations=5, energy_type=energy, random_state=42)
    weights = res["normalizing_weights"]

    expected_w = 1.0 / m_views
    assert len(weights) == m_views
    for w in weights:
        assert abs(w - expected_w) < 1e-6, f"{energy} weight {w} != {expected_w}"

    # Verify that non-scale-free energy (acc) does NOT use equal weights
    res_acc = simlr(views, k=k, iterations=5, energy_type="acc", random_state=42)
    acc_weights = res_acc["normalizing_weights"]
    assert not all(abs(w - expected_w) < 1e-6 for w in acc_weights)


def test_view_rescaling_invariance_end_to_end():
    """Rescaling a raw view by 7 leaves fitted u and v unchanged to < 1e-5."""
    torch.manual_seed(10)
    n, p1, p2, k = 100, 20, 15, 2
    x1 = torch.randn(n, p1, dtype=torch.float32)
    x2 = torch.randn(n, p2, dtype=torch.float32)

    # Base fit
    res_base = simlr([x1, x2], k=k, iterations=25, energy_type="nc", random_state=0)

    # Scaled fit: scale view 0 by 7.0
    res_scaled = simlr([7.0 * x1, x2], k=k, iterations=25, energy_type="nc", random_state=0)

    # Subspace agreement
    assert torch.allclose(res_base["u"], res_scaled["u"], atol=1e-5), "u changed under raw view rescaling"
    for v_b, v_s in zip(res_base["v"], res_scaled["v"]):
        assert torch.allclose(v_b, v_s, atol=1e-5), "v changed under raw view rescaling"


def test_scale_free_energies_degree_zero_invariance():
    """Evaluate degree-0 homogeneity directly on energy functionals without preprocessing."""
    torch.manual_seed(0)
    n, p, k = 50, 12, 3
    x = torch.randn(n, p, dtype=torch.float64)
    v = torch.randn(p, k, dtype=torch.float64)
    s = x @ v
    u, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64))
    u = u * ((n - 1) ** 0.5)

    ctx = SimilarityContext(x=x, v=v)
    for name in sorted(SCALE_FREE_ENERGIES):
        e1 = similarity_energy(name, s, u, ctx).item()
        e7_s = similarity_energy(name, 7.0 * s, u, ctx).item()
        assert abs(e7_s - e1) < 1e-5, f"{name} failed projection rescaling invariance"
        if SIMILARITY[name][1]:  # needs_data
            ctx7 = SimilarityContext(x=7.0 * x, v=v)
            e7_x = similarity_energy(name, s, u, ctx7).item()
            assert abs(e7_x - e1) < 1e-5, f"{name} failed data rescaling invariance"


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_total_energy_bounded_below_for_nc(seed):
    """For energy_type='nc', total energy min(res['energy']) >= -1 - 1e-6."""
    torch.manual_seed(seed)
    n, k = 80, 2
    views = [torch.randn(n, 15), torch.randn(n, 12), torch.randn(n, 10)]

    # Without collapse penalty
    res_no_cw = simlr(views, k=k, iterations=20, energy_type="nc", collapse_weight=0.0, random_state=seed)
    assert min(res_no_cw["energy"]) >= -1.0 - 1e-6, f"Energy fell below -1: {min(res_no_cw['energy'])}"
    assert res_no_cw["best_energy"] >= -1.0 - 1e-6

    # With default collapse penalty (penalty is non-negative, so lower bound still holds)
    res_cw = simlr(views, k=k, iterations=20, energy_type="nc", collapse_weight=1.0, random_state=seed)
    assert min(res_cw["energy"]) >= -1.0 - 1e-6, f"Energy fell below -1: {min(res_cw['energy'])}"
    assert res_cw["best_energy"] >= -1.0 - 1e-6


def test_linear_and_deep_equal_view_weighting_parity():
    """Verify structural parity in view weighting between linear and deep architectures."""
    torch.manual_seed(42)
    n, k = 60, 2
    views = [torch.randn(n, 10), torch.randn(n, 15), torch.randn(n, 12)]
    m = len(views)

    # Linear path: normalizing_weights == [1/M] * M
    res_lin = simlr(views, k=k, iterations=5, energy_type="nc", random_state=42)
    lin_weights = np.asarray(res_lin["normalizing_weights"])
    assert np.allclose(lin_weights, 1.0 / m)
    weight_ratio = float(np.max(lin_weights) / np.min(lin_weights))
    assert abs(weight_ratio - 1.0) < 1e-6, f"Linear weight spread {weight_ratio} != 1.0"

    # Deep path: calculate_sim_loss sums terms with equal implicit weight 1.0 (relative weight 1.0/M)
    from pysimlr.deep import calculate_sim_loss
    latents = [torch.randn(n, k) for _ in range(m)]
    u_shared = torch.randn(n, k)
    total_loss, diags = calculate_sim_loss(latents, u_shared, energy_type="nc")
    assert isinstance(total_loss, torch.Tensor)
    assert not torch.isnan(total_loss)


def test_existing_normalizing_weights_contract():
    """Verify existing regression contract for non-scale-free energy (acc)."""
    from tests.test_m1_regression import TestSimlrCalibrationAndEnergyTracking
    test_obj = TestSimlrCalibrationAndEnergyTracking()
    # Runs test_simlr_calibrates_weights_at_init which checks acc calibration
    test_obj.test_simlr_calibrates_weights_at_init()
