"""Tests for Milestone 1 (WP-1: A10 Principled View Weighting).

Verifies:
1. Scale-free energies receive equal 1/M normalizing weights.
2. Raw view rescaling by 7 leaves u and v invariant.
3. Total energy min(res["energy"]) >= -1 - 1e-6 for nc.
4. Linear and deep architectures combine views with equal weights (structural parity).
5. Non-scale-free energies (acc) change under rescaling and use calibrated weights.
"""
import pytest
import torch
import numpy as np

import pysimlr
from pysimlr.similarity import (
    AUXILIARY_ENERGIES,
    SCALE_FREE_ENERGIES,
    SIMILARITY,
    SimilarityContext,
    similarity_energy,
    similarity_is_scale_free,
)
from pysimlr.simlr import simlr


@pytest.mark.parametrize("m_views", [2, 3, 4])
@pytest.mark.parametrize("energy", sorted(SCALE_FREE_ENERGIES - set(AUXILIARY_ENERGIES)))
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


def test_non_scale_free_energy_acc_uses_calibrated_weights():
    """Verify that non-scale-free energy (acc) does not use equal weights."""
    torch.manual_seed(42)
    n, k, m_views = 60, 2, 3
    views = [torch.randn(n, 10 + i * 5, dtype=torch.float32) for i in range(m_views)]
    res_acc = simlr(views, k=k, iterations=5, energy_type="acc", random_state=42)
    acc_weights = res_acc["normalizing_weights"]
    expected_w = 1.0 / m_views
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
    # Positive half: all energies in SCALE_FREE_ENERGIES are invariant under rescaling
    for name in sorted(SCALE_FREE_ENERGIES):
        e1 = similarity_energy(name, s, u, ctx).item()
        e7_s = similarity_energy(name, 7.0 * s, u, ctx).item()
        assert abs(e7_s - e1) < 1e-5, f"{name} failed projection rescaling invariance"
        if SIMILARITY[name][1]:  # needs_data
            ctx7 = SimilarityContext(x=7.0 * x, v=v)
            e7_x = similarity_energy(name, s, u, ctx7).item()
            assert abs(e7_x - e1) < 1e-5, f"{name} failed data rescaling invariance"

    # Negative half: non-scale-free energies change under rescaling (acc s ratio 7, recon x ratio 11.86)
    non_scale_free = sorted(set(SIMILARITY) - set(SCALE_FREE_ENERGIES))
    for name in non_scale_free:
        e1 = similarity_energy(name, s, u, ctx).item()
        e7_s = similarity_energy(name, 7.0 * s, u, ctx).item()
        s_changed = abs(e7_s - e1) > 1e-3
        x_changed = False
        if SIMILARITY[name][1]:  # needs_data
            ctx7 = SimilarityContext(x=7.0 * x, v=v)
            e7_x = similarity_energy(name, s, u, ctx7).item()
            x_changed = abs(e7_x - e1) > 1e-3
        assert s_changed or x_changed, f"Non-scale-free energy {name} unexpectedly invariant under rescaling"

    # Verify specific measured ratios for non-scale-free energies
    e_acc = similarity_energy("acc", s, u, ctx).item()
    e_acc_7s = similarity_energy("acc", 7.0 * s, u, ctx).item()
    assert abs(e_acc_7s / e_acc - 7.0) < 1e-4, f"acc s ratio {e_acc_7s / e_acc} != 7.0"

    e_recon = similarity_energy("recon", s, u, ctx).item()
    ctx7_x = SimilarityContext(x=7.0 * x, v=v)
    e_recon_7x = similarity_energy("recon", s, u, ctx7_x).item()
    assert abs(e_recon_7x / e_recon - 11.86) < 0.1, f"recon x ratio {e_recon_7x / e_recon} != 11.86"


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

    # Linear path: normalizing_weights == [1/M] * M (weights times M are all 1)
    res_lin = simlr(views, k=k, iterations=5, energy_type="nc", random_state=42)
    lin_weights = np.asarray(res_lin["normalizing_weights"])
    assert np.allclose(lin_weights * m, 1.0, atol=1e-12)
    weight_ratio = float(np.max(lin_weights) / np.min(lin_weights))
    assert abs(weight_ratio - 1.0) < 1e-6, f"Linear weight spread {weight_ratio} != 1.0"

    # Deep path: calculate_sim_loss equals sum of similarity_energy over views to 1e-12 in float64
    from pysimlr.deep import calculate_sim_loss
    latents = [torch.randn(n, k, dtype=torch.float64) for _ in range(m)]
    # Standardize latents as calculate_sim_loss does for energy_needs_unit_columns
    latents_std = [(z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-6) for z in latents]
    u_shared = torch.randn(n, k, dtype=torch.float64)
    ctx = SimilarityContext()
    expected_sum = sum(similarity_energy("nc", z, u_shared, ctx) for z in latents_std)
    total_loss, diags = calculate_sim_loss(latents, u_shared, energy_type="nc",
                                           weights={"sim": 1.0, "var": 0.0, "collapse": 0.0, "u_var": 0.0})
    diff = abs((total_loss - expected_sum).item())
    assert diff < 1e-12, f"Structural similarity loss mismatch: {diff} >= 1e-12"


def test_view_energy_scale_calibrated_vs_candidate():
    """Verify _view_energy_scale calibrated scale and normalizing_weights at init."""
    from pysimlr.simlr import _view_energy_scale, calculate_simlr_energy

    torch.manual_seed(42)
    n, p, k = 50, 10, 2
    x = torch.randn(n, p)
    u = torch.randn(n, k)
    assert _view_energy_scale("acc", x, u, -0.42) == 0.42

    views = [torch.randn(n, 12, dtype=torch.float32), torch.randn(n, 8, dtype=torch.float32)]
    m = len(views)
    res = simlr(views, k=k, iterations=0, energy_type="acc", topology="star",
                scale_list=["none"], random_state=42)
    u_init = res["u"]
    v_init = res["v"]
    weights = res["normalizing_weights"]

    assert len(weights) == m
    for idx in range(m):
        e0_m = calculate_simlr_energy(v_init[idx], views[idx], u_init, "acc").item()
        expected_w = 1.0 / (abs(e0_m) * m)
        assert abs(weights[idx] - expected_w) < 1e-6


