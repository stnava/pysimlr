import pytest
import torch
from pysimlr.flows import (
    AffineCouplingLayer,
    PermutationLayer,
    NormalizingFlow,
    flow_simr,
    FlowConditionalInference
)
from pysimlr import predict_deep

def test_affine_coupling_invertibility():
    torch.manual_seed(42)
    dim = 8
    x = torch.randn(10, dim)
    
    # Block mask coupling
    layer_block = AffineCouplingLayer(dim=dim, hidden_dim=16, mask_type="block", index=0)
    z_b, log_det_b = layer_block(x)
    x_rec_b = layer_block.inverse(z_b)
    
    assert torch.allclose(x, x_rec_b, atol=1e-5, rtol=1e-5)
    assert log_det_b.shape == (10,)
    
    # Alternate mask coupling
    layer_alt = AffineCouplingLayer(dim=dim, hidden_dim=16, mask_type="alternate", index=1)
    z_a, log_det_a = layer_alt(x)
    x_rec_a = layer_alt.inverse(z_a)
    
    assert torch.allclose(x, x_rec_a, atol=1e-5, rtol=1e-5)
    assert log_det_a.shape == (10,)

def test_permutation_layer():
    torch.manual_seed(42)
    dim = 6
    x = torch.randn(5, dim)
    
    layer = PermutationLayer(dim=dim)
    z, log_det = layer(x)
    x_rec = layer.inverse(z)
    
    assert torch.allclose(x, x_rec, atol=1e-7)
    assert torch.all(log_det == 0.0)

def test_normalizing_flow():
    torch.manual_seed(42)
    dim = 8
    x = torch.randn(15, dim)
    
    flow = NormalizingFlow(dim=dim, num_layers=4, hidden_dim=16)
    assert flow.use_fallback is False
    z, log_det = flow(x)
    x_rec = flow.inverse(z)
    
    # Exact invertibility verification (scientific validity floor)
    assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-5)
    assert log_det.shape == (15,)

def test_flow_simr_training_and_imputation():
    torch.manual_seed(42)
    n = 60
    p1 = 12
    p2 = 10
    k = 3
    
    # Create aligned synthetic multi-view data
    u_true = torch.randn(n, k)
    w1 = torch.randn(k, p1)
    w2 = torch.randn(k, p2)
    
    # Modalities as linear projections of shared consensus + noise
    x1 = u_true @ w1 + 0.05 * torch.randn(n, p1)
    x2 = u_true @ w2 + 0.05 * torch.randn(n, p2)
    
    # Fit Flow-SiMR model for a few epochs
    res = flow_simr(
        [x1, x2], 
        k=k, 
        iterations=10, 
        batch_size=20, 
        warmup_epochs=2, 
        num_layers=2, 
        hidden_dim=16, 
        verbose=False
    )
    
    # Verify outputs
    assert res["model"] is not None
    assert res["model_type"] == "flow_simr"
    for f in res["model"].flows:
        assert f.use_fallback is False
    assert len(res["latents"]) == 2
    assert res["latents"][0].shape == (n, k)
    assert len(res["reconstructions"]) == 2
    assert res["reconstructions"][0].shape == (n, p1)
    assert len(res["reconstructions"][1].shape) == 2
    assert len(res["errors"]) == 2
    
    # Verify exact log-likelihood (NLL) optimization path
    assert len(res["loss_history"]) == 10
    assert len(res["recon_history"]) == 10
    assert len(res["sim_history"]) == 10
    
    # Verify standard prediction wrapper consistency
    pred_res = predict_deep([x1, x2], res, device="cpu")
    assert pred_res["latents"] is not None
    assert torch.allclose(pred_res["latents"][0], res["latents"][0], atol=1e-5)
    
    # Verify Woodbury-based Conditional Inference
    cond = res["cond_inference"]
    assert isinstance(cond, FlowConditionalInference)
    assert cond.joint_mean.shape == (2 * k,)
    assert cond.joint_cov.shape == (2 * k, 2 * k)
    
    # Conditional prediction: predict modality 2's shared latent space from modality 1's shared latent space
    z1 = res["latents"][0]
    z2_pred = cond.predict_conditional(observed_idx=0, target_idx=1, observed_z=z1)
    
    assert z2_pred.shape == (n, k)
    
    # Decoded target prediction (cross-view synthesis)
    x2_synth = res["model"].decoders[1](z2_pred)
    assert x2_synth.shape == (n, p2)
    assert not torch.isnan(x2_synth).any()

def test_flow_simr_v_bottleneck():
    torch.manual_seed(42)
    n = 30
    p1 = 8
    p2 = 10
    k = 15 # K is larger than modality features
    
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)
    
    from pysimlr.flows import flow_simr_v
    
    # This should execute successfully without PyTorch shape mismatches
    res = flow_simr_v(
        [x1, x2],
        k=k,
        iterations=3,
        batch_size=10,
        warmup_epochs=1,
        num_layers=2,
        hidden_dim=8,
        verbose=False
    )
    
    assert res["model"] is not None
    for f in res["model"].flows:
        assert f.use_fallback is False
    assert res["latents"][0].shape == (n, k)
    assert res["reconstructions"][0].shape == (n, p1)
    assert res["reconstructions"][1].shape == (n, p2)


def test_normalizing_flow_forced_fallback():
    torch.manual_seed(42)
    dim = 8
    x = torch.randn(15, dim)
    
    # Must emit UserWarning on fallback
    with pytest.warns(UserWarning, match="Using local fallback CustomRealNVP"):
        flow = NormalizingFlow(dim=dim, num_layers=4, hidden_dim=16, force_fallback=True)
    
    assert flow.use_fallback is True
    
    # Forward pass
    z, log_det = flow(x)
    assert z.shape == (15, dim)
    assert log_det.shape == (15,)
    
    # Exact invertibility verification for fallback CustomRealNVP
    x_rec = flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-5)
    
    # forward_and_log_det consistency
    z2, log_det2 = flow.forward_and_log_det(x)
    assert torch.allclose(z, z2, atol=1e-5)
    assert torch.allclose(log_det, log_det2, atol=1e-5)
    
    # inverse_and_log_det consistency
    x_rec2, log_det_inv = flow.inverse_and_log_det(z)
    assert torch.allclose(x, x_rec2, atol=1e-5)
    assert log_det_inv.shape == (15,)


def test_elementwise_rq_spline_and_positive_actnorm_contracts():
    from scipy.stats import spearmanr
    from pysimlr.flows import (
        ActNormElementwiseRQSplineFlow,
        ElementwiseRQSplineFlow,
        PositiveActNorm1d,
        exact_pcf_penalty_bpd,
    )

    torch.manual_seed(7)
    dim = 3
    x_raw = torch.randn(24, dim) * torch.tensor([[9.0, 0.4, 3.5]]) + torch.tensor([[2.5, -1.2, 0.7]])

    # Epoch-0 identity on ElementwiseRQSplineFlow
    sp0 = ElementwiseRQSplineFlow(dim=dim, num_layers=2, num_bins=8)
    z0, ld0 = sp0(torch.clamp(x_raw, -3.0, 3.0))
    assert float((z0 - torch.clamp(x_raw, -3.0, 3.0)).abs().max()) < 1e-5
    assert float(ld0.abs().max()) < 1e-5

    # PositiveActNorm1d + perturbed spline: invertibility, analytical log_det, zero off-diag Jacobian, monotonicity
    fl = ActNormElementwiseRQSplineFlow(dim=dim, num_layers=2, num_bins=8)
    assert fl.use_fallback is False
    fl.train()
    z_init, _ = fl(x_raw)
    # First batch standardization check
    assert torch.allclose(z_init.mean(dim=0), torch.zeros(dim), atol=1e-4)
    assert torch.allclose(z_init.std(dim=0, unbiased=False), torch.ones(dim), atol=5e-2)

    with torch.no_grad():
        for layer in fl.layers:
            layer.unnorm_widths.normal_(0.0, 0.35)
            layer.unnorm_heights.normal_(0.0, 0.35)
            layer.unnorm_derivs.add_(0.25 * torch.randn_like(layer.unnorm_derivs))
            layer.raw_log_scale.normal_(0.0, 0.2)
            layer.shift.normal_(0.0, 0.15)
    fl.eval()

    z, ld = fl.forward_and_log_det(x_raw)
    x_rec, ld_inv = fl.inverse_and_log_det(z)
    assert float((x_rec - x_raw).abs().max()) < 1e-4
    assert float((ld + ld_inv).abs().max()) < 1e-4

    # Analytical log_det vs exact autograd Jacobian & zero off-diagonal Jacobian
    for row_idx in range(4):
        xi = x_raw[row_idx:row_idx + 1].detach().clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(lambda inp: fl.forward_and_log_det(inp)[0].squeeze(0), xi.squeeze(0))
        off_diag = J - torch.diag(torch.diag(J))
        assert float(off_diag.abs().max()) == 0.0
        assert bool((torch.diag(J) > 0).all())
        _, logabsdet_auto = torch.linalg.slogdet(J)
        assert abs(float(ld[row_idx]) - float(logabsdet_auto)) < 1e-4

    # PCF penalty is identically 0 for coordinate-wise spline flow
    assert abs(float(exact_pcf_penalty_bpd(fl, x_raw))) < 1e-6

    # Strict coordinate-wise monotonicity (Spearman rho == 1.0)
    for j in range(dim):
        rho = float(spearmanr(x_raw[:, j].detach().numpy(), z[:, j].detach().numpy()).statistic)
        assert rho > 0.999999


def test_flow_simr_v_k1_auto_spline1d_fixes_affine_collapse():
    import numpy as np
    import pysimlr
    from pysimlr.flows import flow_simr_v, resolve_flow_type

    assert resolve_flow_type("auto", 1) == "spline1d"
    assert resolve_flow_type("auto", 2) == "realnvp"
    assert resolve_flow_type("spline1d", 3) == "spline1d"
    assert resolve_flow_type("realnvp", 1) == "realnvp"
    with pytest.raises(ValueError, match="flow_type"):
        resolve_flow_type("nope", 1)

    rng = np.random.default_rng(19)
    n = 64
    u = rng.standard_normal((n, 1))
    v0 = rng.standard_normal((1, 8))
    v1 = rng.standard_normal((1, 7))
    # Strongly skewed nonlinear view 0 + cubic view 1
    x0 = torch.tensor(np.exp(0.9 * u) @ v0 + 0.1 * rng.standard_normal((n, 8)), dtype=torch.float32)
    x1 = torch.tensor((u + 0.35 * u ** 3) @ v1 + 0.1 * rng.standard_normal((n, 7)), dtype=torch.float32)

    res_auto = pysimlr.fit("flow_v", [x0, x1], k=1, iterations=25, batch_size=32, warmup_epochs=4, random_state=3)
    res_rnvp = pysimlr.fit(
        "flow_v", [x0, x1], k=1, iterations=25, batch_size=32, warmup_epochs=4, random_state=3, flow_type="realnvp"
    )
    assert res_auto["flow_type"] == "spline1d"
    assert res_rnvp["flow_type"] == "realnvp"

    grid = torch.linspace(-2.2, 2.2, 41).unsqueeze(1)
    with torch.no_grad():
        y_sp, _ = res_auto["model"].flows[0](grid)
        y_rn, _ = res_rnvp["model"].flows[0](grid)
    d2_sp = float((y_sp[2:] - 2.0 * y_sp[1:-1] + y_sp[:-2]).abs().max())
    d2_rn = float((y_rn[2:] - 2.0 * y_rn[1:-1] + y_rn[:-2]).abs().max())
    assert d2_rn < 1e-5, f"RealNVP at dim=1 should be affine, got d2={d2_rn}"
    assert d2_sp > 1e-4, f"Spline1D at dim=1 must learn genuine nonlinear curvature, got d2={d2_sp}"

    pred = pysimlr.predict(res_auto, [x0[:12], x1[:12]])
    assert torch.allclose(pred["u"], res_auto["u"][:12], atol=1e-4)

    # Masked unobserved rows work cleanly with spline1d
    x0_mask = x0.clone()
    x0_mask[3:7] = float("nan")
    res_m = pysimlr.fit(
        "flow_v", [x0_mask, x1], k=1, iterations=8, batch_size=32, warmup_epochs=2,
        missing_rows="mask", random_state=4,
    )
    assert torch.isnan(res_m["latents"][0][3:7]).all()
    assert torch.isfinite(res_m["u"]).all()


def test_flow_simr_v_pcf_alpha_parity_and_jacobian_regularization():
    import pysimlr
    from pysimlr.flows import exact_pcf_penalty_bpd, flow_simr_v

    with pytest.raises(ValueError, match="pcf_alpha"):
        flow_simr_v([torch.randn(20, 6), torch.randn(20, 5)], k=2, iterations=2, pcf_alpha=-0.1)

    torch.manual_seed(11)
    n, k = 56, 2
    u = torch.randn(n, k)
    x0 = (u + 0.3 * u.pow(2)) @ torch.randn(k, 8) + 0.1 * torch.randn(n, 8)
    x1 = torch.sinh(0.8 * u) @ torch.randn(k, 7) + 0.1 * torch.randn(n, 7)

    r_def = pysimlr.fit("flow_v", [x0, x1], k=k, iterations=18, batch_size=28, warmup_epochs=3, random_state=5)
    r_zro = pysimlr.fit(
        "flow_v", [x0, x1], k=k, iterations=18, batch_size=28, warmup_epochs=3, random_state=5, pcf_alpha=0.0
    )
    assert torch.equal(r_def["u"], r_zro["u"])

    r_pcf = pysimlr.fit(
        "flow_v", [x0, x1], k=k, iterations=18, batch_size=28, warmup_epochs=3, random_state=5, pcf_alpha=1.0
    )
    assert not torch.equal(r_def["u"], r_pcf["u"])

    # Evaluate PCF penalty on linear projections
    from pysimlr.api import _preprocess_views
    xs_std, _ = _preprocess_views([x0, x1], r_def["scale_list"], r_def["provenance_list"])
    pcf_unreg = sum(
        float(exact_pcf_penalty_bpd(fl, xm @ vm))
        for fl, xm, vm in zip(r_def["model"].flows, xs_std, r_def["v"])
    )
    pcf_reg = sum(
        float(exact_pcf_penalty_bpd(fl, xm @ vm))
        for fl, xm, vm in zip(r_pcf["model"].flows, xs_std, r_pcf["v"])
    )
    assert pcf_reg <= pcf_unreg + 1e-5


def test_flow_simr_scale_bound_gaussian_pca_and_post_flow_varimax():
    import inspect
    from pysimlr.flows import PostRotatedFlowWrapper, flow_simr

    assert inspect.signature(flow_simr).parameters["scale_bound"].default == 0.5
    with pytest.raises(ValueError, match="base_distribution"):
        flow_simr([torch.randn(24, 6), torch.randn(24, 5)], k=2, iterations=2, base_distribution="invalid")

    torch.manual_seed(21)
    n, k = 54, 3
    u = torch.randn(n, k)
    x0 = u @ torch.randn(k, 9) + 0.1 * torch.randn(n, 9)
    x1 = u @ torch.randn(k, 8) + 0.1 * torch.randn(n, 8)

    # 1. DiagGaussian + post-flow Varimax: exact reconstruction invariance & rotated latents
    res_d0 = flow_simr([x0, x1], k=k, iterations=8, batch_size=27, warmup_epochs=2, random_state=9, varimax=False)
    res_d1 = flow_simr([x0, x1], k=k, iterations=8, batch_size=27, warmup_epochs=2, random_state=9, varimax=True)
    assert all(isinstance(fl, PostRotatedFlowWrapper) for fl in res_d1["model"].flows)
    assert not torch.allclose(res_d0["u"], res_d1["u"], atol=1e-3)
    for r0, r1 in zip(res_d0["reconstructions"], res_d1["reconstructions"]):
        assert torch.allclose(r0, r1, atol=1e-5)
    pred_d1 = predict_deep([x0[:10], x1[:10]], res_d1, device="cpu")
    assert torch.allclose(pred_d1["u"], res_d1["u"][:10], atol=1e-4)

    # 2. GaussianPCA + post-flow Varimax (W <- R^T W): exact log_prob and reconstruction invariance
    res_g0 = flow_simr(
        [x0, x1], k=k, iterations=8, batch_size=27, warmup_epochs=2, random_state=12,
        base_distribution="gaussian_pca", varimax=False,
    )
    res_g1 = flow_simr(
        [x0, x1], k=k, iterations=8, batch_size=27, warmup_epochs=2, random_state=12,
        base_distribution="gaussian_pca", varimax=True,
    )
    assert not torch.allclose(res_g0["u"], res_g1["u"], atol=1e-3)
    for r0, r1 in zip(res_g0["reconstructions"], res_g1["reconstructions"]):
        assert torch.allclose(r0, r1, atol=1e-5)

    from pysimlr.api import _preprocess_views
    xs_std, _ = _preprocess_views([x0, x1], res_g0["scale_list"], res_g0["provenance_list"])
    with torch.no_grad():
        for m_i, xm in enumerate(xs_std):
            z0, ld0 = res_g0["model"].flows[m_i](xm.float())
            z1, ld1 = res_g1["model"].flows[m_i](xm.float())
            lp0 = res_g0["model"].gpca_bases[m_i].log_prob(z0) + ld0
            lp1 = res_g1["model"].gpca_bases[m_i].log_prob(z1) + ld1
            assert torch.allclose(lp0, lp1, atol=1e-5)

    pred_g1 = predict_deep([x0[:10], x1[:10]], res_g1, device="cpu")
    assert torch.allclose(pred_g1["u"], res_g1["u"][:10], atol=1e-4)
    z1_cond = res_g1["cond_inference"].predict_conditional(0, 1, res_g1["latents"][0][:5])
    x1_syn = res_g1["model"].decoders[1](z1_cond)
    assert x1_syn.shape == (5, 8) and torch.isfinite(x1_syn).all()


