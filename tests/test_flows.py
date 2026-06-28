import pytest
import torch
import numpy as np
from pysimlr.flows import (
    AffineCouplingLayer,
    PermutationLayer,
    NormalizingFlow,
    FlowSiMRModel,
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
    
    # Fit Flow-SiMLR model for a few epochs
    res = flow_simr(
        [x1, x2], 
        k=k, 
        epochs=10, 
        batch_size=20, 
        warmup_epochs=2, 
        num_layers=2, 
        hidden_dim=16, 
        verbose=False
    )
    
    # Verify outputs
    assert res["model"] is not None
    assert res["model_type"] == "flow_simr"
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
        epochs=3,
        batch_size=10,
        warmup_epochs=1,
        num_layers=2,
        hidden_dim=8,
        verbose=False
    )
    
    assert res["model"] is not None
    assert res["latents"][0].shape == (n, k)
    assert res["reconstructions"][0].shape == (n, p1)
    assert res["reconstructions"][1].shape == (n, p2)
