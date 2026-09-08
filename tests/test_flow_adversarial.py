"""
Adversarial Stress Test Suite for Normalizing Flows.

Targeting:
- Invertibility precision under native RealNVP vs fallback: ||x - f^{-1}(f(x))||_inf < 10^{-5}
  across varying dims (D=2, 8, 32, 64) and odd dims.
- Log-determinant Jacobian consistency: log |det J_{f^{-1}}| = -log |det J_f|.
- Autograd analytical Jacobian oracle comparison with logabsdet.
- Extreme inputs: high dynamic range, large values, sub-normal values, all-zeros.
- Single sample input (N=1) behavior and ActNorm data-dependent initialization constraints.
- Force fallback toggle: contract and mathematical behavior equivalence.
- Autograd gradient flow and parameter optimization.
- FlowWhitener invertibility in latent space.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pysimlr import NormalizingFlow, FlowWhitener, flow_whiten_matrix


@pytest.mark.parametrize("dim", [2, 3, 8, 17, 32, 64])
@pytest.mark.parametrize("force_fallback", [False, True])
def test_adversarial_invertibility_dims_native_and_fallback(dim, force_fallback):
    """Stress test forward-then-inverse invertibility ||x - f^{-1}(f(x))||_inf < 10^{-5}."""
    torch.manual_seed(42)
    flow = NormalizingFlow(dim=dim, num_layers=4, hidden_dim=32, force_fallback=force_fallback)
    if not force_fallback:
        # Warmup ActNorm with standard batch
        flow(torch.randn(10, dim))

    x = torch.randn(20, dim)
    z, log_det = flow(x)
    x_rec = flow.inverse(z)

    max_err = torch.max(torch.abs(x - x_rec)).item()
    assert max_err < 1e-5, f"Invertibility failed for D={dim}, fallback={force_fallback}: {max_err}"
    assert z.shape == (20, dim)
    assert log_det.shape == (20,)


@pytest.mark.parametrize("dim", [2, 8, 32, 64])
@pytest.mark.parametrize("force_fallback", [False, True])
def test_adversarial_invertibility_inverse_first(dim, force_fallback):
    """Stress test inverse-then-forward invertibility ||z - f(f^{-1}(z))||_inf < 10^{-5}."""
    torch.manual_seed(101)
    flow = NormalizingFlow(dim=dim, num_layers=4, hidden_dim=32, force_fallback=force_fallback)
    if not force_fallback:
        flow(torch.randn(10, dim))

    z = torch.randn(20, dim)
    x = flow.inverse(z)
    z_rec, log_det = flow(x)

    max_err = torch.max(torch.abs(z - z_rec)).item()
    assert max_err < 1e-5, f"Inverse-first invertibility failed for D={dim}, fallback={force_fallback}: {max_err}"
    assert x.shape == (20, dim)


@pytest.mark.parametrize("dim", [2, 8, 32, 64])
@pytest.mark.parametrize("force_fallback", [False, True])
def test_adversarial_log_det_jacobian_consistency(dim, force_fallback):
    """Verify log |det J_{f^{-1}}| = -log |det J_f| across forward and inverse transformations."""
    torch.manual_seed(202)
    flow = NormalizingFlow(dim=dim, num_layers=4, hidden_dim=32, force_fallback=force_fallback)
    if not force_fallback:
        flow(torch.randn(10, dim))

    x = torch.randn(15, dim)
    z, ld_fwd = flow.forward_and_log_det(x)
    x_rec, ld_inv = flow.inverse_and_log_det(z)

    # Invertibility check
    assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-5)

    # Log-determinant consistency: ld_fwd + ld_inv == 0
    consistency_err = torch.max(torch.abs(ld_fwd + ld_inv)).item()
    assert consistency_err < 1e-5, (
        f"Log-det consistency violated for D={dim}, fallback={force_fallback}: sum={consistency_err}"
    )


@pytest.mark.parametrize("dim", [2, 4, 8])
@pytest.mark.parametrize("force_fallback", [False, True])
def test_adversarial_autograd_jacobian_oracle(dim, force_fallback):
    """
    Adversarial oracle test: compute exact analytical Jacobian via autograd
    and verify slogdet(J) logabsdet matches flow's log_det.
    """
    torch.manual_seed(303)
    flow = NormalizingFlow(dim=dim, num_layers=2, hidden_dim=16, force_fallback=force_fallback)
    if not force_fallback:
        flow(torch.randn(10, dim))

    x = torch.randn(1, dim)
    z, reported_ld = flow(x)

    # Oracle Jacobian for forward pass
    def f_single(v):
        out, _ = flow(v.unsqueeze(0))
        return out.squeeze(0)

    J_fwd = torch.autograd.functional.jacobian(f_single, x.squeeze(0))
    sign_fwd, true_ld_fwd = torch.linalg.slogdet(J_fwd)

    # Determinant must be non-zero (invertible map)
    assert torch.abs(sign_fwd).item() == 1.0, "Forward Jacobian must be non-singular"
    if not force_fallback:
        # Native RealNVP uses alternating masks without coordinate permutations (orientation preserving)
        assert sign_fwd.item() > 0, "Native RealNVP must be orientation-preserving"

    err_fwd = torch.abs(reported_ld.squeeze() - true_ld_fwd).item()
    assert err_fwd < 1e-5, f"Forward log-det diverges from true Jacobian: reported={reported_ld.item()}, true={true_ld_fwd.item()}"

    # Oracle Jacobian for inverse pass
    def f_inv_single(v):
        out, _ = flow.inverse_and_log_det(v.unsqueeze(0))
        return out.squeeze(0)

    J_inv = torch.autograd.functional.jacobian(f_inv_single, z.squeeze(0))
    sign_inv, true_ld_inv = torch.linalg.slogdet(J_inv)

    assert torch.abs(sign_inv).item() == 1.0, "Inverse Jacobian must be non-singular"
    _, reported_ld_inv = flow.inverse_and_log_det(z)
    err_inv = torch.abs(reported_ld_inv.squeeze() - true_ld_inv).item()
    assert err_inv < 1e-5, f"Inverse log-det diverges from true Jacobian: reported={reported_ld_inv.item()}, true={true_ld_inv.item()}"


@pytest.mark.parametrize("force_fallback", [False, True])
def test_adversarial_extreme_inputs(force_fallback):
    """Stress test high dynamic range, large values, sub-normal values, and all-zeros."""
    torch.manual_seed(404)
    dim = 8
    flow = NormalizingFlow(dim=dim, force_fallback=force_fallback)
    if not force_fallback:
        flow(torch.randn(10, dim))

    # 1. All Zeros
    x_zeros = torch.zeros(4, dim)
    z_z, ld_z = flow(x_zeros)
    x_rec_z = flow.inverse(z_z)
    assert not torch.isnan(z_z).any()
    assert not torch.isnan(ld_z).any()
    assert torch.allclose(x_zeros, x_rec_z, atol=1e-5)

    # 2. Sub-normal / tiny values
    x_tiny = torch.randn(4, dim) * 1e-15
    z_t, ld_t = flow(x_tiny)
    x_rec_t = flow.inverse(z_t)
    assert not torch.isnan(z_t).any()
    assert not torch.isnan(ld_t).any()
    assert torch.allclose(x_tiny, x_rec_t, atol=1e-5)

    # 3. High Dynamic Range (spanning 8 orders of magnitude across features)
    x_hdr = torch.randn(5, dim)
    x_hdr[:, 0] *= 1e4
    x_hdr[:, 1] *= 1e-4
    x_hdr[:, 2] *= 1e2
    x_hdr[:, 3] *= 1e-2
    z_h, ld_h = flow(x_hdr)
    x_rec_h = flow.inverse(z_h)
    rel_err = (torch.norm(x_hdr - x_rec_h) / torch.norm(x_hdr)).item()
    assert rel_err < 1e-4, f"HDR relative invertibility error too large: {rel_err}"

    # 4. Large values (1e3): verify no NaNs, and float32 relative error within bounds
    x_large = torch.randn(4, dim) * 1e3
    z_l, ld_l = flow(x_large)
    x_rec_l = flow.inverse(z_l)
    assert not torch.isnan(z_l).any()
    assert not torch.isnan(ld_l).any()
    assert not torch.isnan(x_rec_l).any()
    rel_err_large = (torch.norm(x_large - x_rec_l) / torch.norm(x_large)).item()
    assert rel_err_large < 1e-4, f"Large value relative invertibility error too large: {rel_err_large}"


def test_adversarial_single_sample_n1_behavior():
    """
    Stress test single sample (N=1) input.
    - Fallback CustomRealNVP directly supports N=1 on fresh initialization.
    - Native RealNVP with ActNorm requires N >= 2 on fresh initialization
      for data-dependent variance initialization (Bessel's correction).
    - Native RealNVP with pre-initialized ActNorm supports N=1 at inference time.
    """
    torch.manual_seed(505)
    dim = 8
    x_single = torch.randn(1, dim)

    # Fallback handles N=1 on fresh initialization
    flow_fb = NormalizingFlow(dim=dim, force_fallback=True)
    z_fb, ld_fb = flow_fb(x_single)
    x_rec_fb = flow_fb.inverse(z_fb)
    assert z_fb.shape == (1, dim)
    assert ld_fb.shape == (1,)
    assert torch.allclose(x_single, x_rec_fb, atol=1e-5)

    # Native fresh initialization with N=1 fails predictably due to ActNorm variance degrees of freedom
    flow_nat_fresh = NormalizingFlow(dim=dim, force_fallback=False)
    with pytest.raises(RuntimeError, match=r"\[ActNorm\] non-finite values in 'std'"):
        flow_nat_fresh(x_single)

    # Native with pre-warmed ActNorm handles N=1 at inference time
    flow_nat = NormalizingFlow(dim=dim, force_fallback=False)
    flow_nat(torch.randn(4, dim))  # warmup data-dependent init
    z_nat, ld_nat = flow_nat(x_single)
    x_rec_nat = flow_nat.inverse(z_nat)
    assert z_nat.shape == (1, dim)
    assert ld_nat.shape == (1,)
    assert torch.allclose(x_single, x_rec_nat, atol=1e-5)

    # Verify log-det consistency on N=1 inference
    _, ld_inv_nat = flow_nat.inverse_and_log_det(z_nat)
    assert torch.allclose(ld_nat + ld_inv_nat, torch.zeros(1), atol=1e-5)


def test_adversarial_force_fallback_toggle():
    """Verify force_fallback flag behavior, warnings, and contract equivalence."""
    dim = 6
    x = torch.randn(8, dim)

    # force_fallback=True emits UserWarning
    with pytest.warns(UserWarning, match="Using local fallback CustomRealNVP"):
        f_fb = NormalizingFlow(dim=dim, force_fallback=True)
    assert f_fb.use_fallback is True

    # force_fallback=False binds to ANTsTorch
    f_nat = NormalizingFlow(dim=dim, force_fallback=False)
    assert f_nat.use_fallback is False

    # Contract equivalence: method signatures, return shapes, types
    z_fb, ld_fb = f_fb(x)
    f_nat(torch.randn(10, dim))  # warmup
    z_nat, ld_nat = f_nat(x)

    assert isinstance(z_fb, torch.Tensor) and isinstance(z_nat, torch.Tensor)
    assert isinstance(ld_fb, torch.Tensor) and isinstance(ld_nat, torch.Tensor)
    assert z_fb.shape == z_nat.shape == (8, dim)
    assert ld_fb.shape == ld_nat.shape == (8,)

    # inverse_and_log_det API check
    rec_fb, ld_inv_fb = f_fb.inverse_and_log_det(z_fb)
    rec_nat, ld_inv_nat = f_nat.inverse_and_log_det(z_nat)
    assert rec_fb.shape == rec_nat.shape == (8, dim)
    assert ld_inv_fb.shape == ld_inv_nat.shape == (8,)


@pytest.mark.parametrize("force_fallback", [False, True])
def test_adversarial_gradient_flow_and_optimization(force_fallback):
    """Verify autograd computation graph integrity, gradient flow, and parameter updates."""
    torch.manual_seed(606)
    dim = 4
    flow = NormalizingFlow(dim=dim, num_layers=2, hidden_dim=16, force_fallback=force_fallback)
    if not force_fallback:
        flow(torch.randn(10, dim))

    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    x = torch.randn(12, dim)

    # Forward pass and NLL loss: 0.5 * ||z||^2 - log_det
    z, log_det = flow(x)
    loss = 0.5 * torch.sum(z ** 2, dim=1) - log_det
    total_loss = loss.mean()

    total_loss.backward()

    # Verify gradients exist and are finite
    has_grad = False
    for p in flow.parameters():
        if p.requires_grad and p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all(), "Gradient contains non-finite elements"
    assert has_grad, "No parameter received gradients"

    optimizer.step()
    optimizer.zero_grad()


def test_adversarial_flow_whitener_latent_invertibility():
    """Verify FlowWhitener exact latent invertibility (output_space='z') achieves < 1e-5."""
    torch.manual_seed(707)
    n, p = 30, 4
    x = torch.randn(n, p)

    whitener = FlowWhitener(
        K=4,
        max_iter=15,
        batch_size=15,
        base_distribution="GaussianPCA",
        pca_latent_dimension=p,
        output_space="z"
    )
    whitener.fit(x)

    z = whitener.transform(x, output_space="z")
    x_rec = whitener.inverse_transform(z, input_space="z")

    max_diff = torch.max(torch.abs(x - x_rec)).item()
    assert max_diff < 1e-5, f"FlowWhitener latent inversion error too large: {max_diff}"
