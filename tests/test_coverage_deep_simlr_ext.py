import pytest
import torch
import numpy as np
from unittest.mock import patch
import sys

def test_imports_fallback():
    with patch.dict('sys.modules', {'sklearn.decomposition': None, 'nsa_flow': None}):
        if 'pysimlr.simlr' in sys.modules: del sys.modules['pysimlr.simlr']
        if 'pysimlr.deep' in sys.modules: del sys.modules['pysimlr.deep']
        import pysimlr.simlr
        import pysimlr.deep
        assert getattr(pysimlr.simlr, 'FastICA', None) is None
        assert getattr(pysimlr.deep, 'nsa', None) is None
    # restore modules to prevent issues with other tests
    if 'pysimlr.simlr' in sys.modules: del sys.modules['pysimlr.simlr']
    if 'pysimlr.deep' in sys.modules: del sys.modules['pysimlr.deep']

from pysimlr.simlr import (
    parse_constraint,
    initialize_simlr,
    calculate_ica_gradient,
    calculate_simlr_energy,
    simlr,
    reconstruct_from_learned_maps,
    predict_simlr,
    estimate_rank,
    decompose_energy
)

from pysimlr.deep import (
    _svd_project_columns,
    _newton_step_ortho,
    _normalize_rows,
    _covariance_penalty,
    _cross_covariance_penalty,
    _standardize_deep,
    LENDNSAEncoder,
    ModalityDecoder,
    LENDSiMRModel,
    NEDSiMRModel,
    NEDSharedPrivateSiMRModel,
    calculate_sim_loss,
    deep_simr,
    predict_deep,
    lend_simr,
    ned_simr,
    ned_simr_shared_private
)

def test_simlr_parse_constraint_exceptions():
    res = parse_constraint("Stiefel x not_a_float x not_an_int")
    assert res["weight"] == 1.0
    assert res["iterations"] == 1

def test_simlr_initialize_simlr_padding():
    x = torch.randn(10, 2)
    v_mats = initialize_simlr([x], k=5)
    assert v_mats[0].shape == (2, 5)

def test_simlr_ica_gradient_kurtosis():
    x = torch.randn(10, 5)
    u = torch.randn(10, 2)
    v = torch.randn(5, 2)
    grad = calculate_ica_gradient(x, u, v, nonlinearity="kurtosis")
    assert grad.shape == (5, 2)

def test_simlr_sparseness_alias():
    x1 = torch.randn(10, 5)
    x2 = torch.randn(10, 6)
    res = simlr([x1, x2], k=2, iterations=2, sparseness=0.1, positivity="negative")
    assert "u" in res

def test_simlr_reconstruct_missing_w():
    u = torch.randn(10, 2)
    simlr_result = {"v": [torch.randn(5, 2)]}
    recons = reconstruct_from_learned_maps(u, simlr_result)
    assert recons == []

def test_simlr_predict_simlr_legacy_refit():
    x1 = torch.randn(10, 5)
    simlr_result = {
        "v": [torch.randn(5, 2)],
        "mixing_algorithm": "svd",
        "orthogonalize_u": True,
        "scale_list": ["none"]
    }
    with pytest.raises(ValueError):
        predict_simlr([x1], simlr_result)
    
    res = predict_simlr([x1], simlr_result, allow_legacy_refit=True)
    assert "reconstructions" in res

def test_simlr_estimate_rank_zero_perms():
    x1 = torch.randn(20, 5)
    x2 = torch.randn(20, 6)
    k = estimate_rank([x1, x2], n_permutations=0)
    assert k >= 1

def test_simlr_decompose_energy():
    x1 = torch.randn(10, 5)
    x2 = torch.randn(10, 6)
    res = simlr([x1, x2], k=2, iterations=2)
    decomp = decompose_energy([x1, x2], res, energy_type="acc")
    assert "modality_energies" in decomp

def test_deep_svd_project_columns_fallback():
    u = torch.tensor([[float('nan'), 1.0], [1.0, 1.0]])
    res = _svd_project_columns(u)
    assert torch.isnan(res).any()

def test_deep_newton_step_ortho():
    u = torch.randn(5, 2)
    res = _newton_step_ortho(u)
    assert res.shape == (5, 2)

def test_deep_normalize_rows():
    x = torch.randn(5, 2)
    res = _normalize_rows(x)
    assert res.shape == (5, 2)

def test_deep_covariance_penalties_small_n():
    z1 = torch.randn(1, 2)
    z2 = torch.randn(1, 2)
    assert _covariance_penalty(z1).item() == 0.0
    assert _cross_covariance_penalty(z1, z2).item() == 0.0

def test_deep_standardize_constant_col():
    x = torch.ones(10, 5)
    scaled, _ = _standardize_deep([x])
    assert scaled[0].shape == (10, 5)

def test_deep_encoder_branches():
    enc = LENDNSAEncoder(input_dim=5, latent_dim=2, first_layer_mode="projected", nsa_iterations=2)
    x = torch.randn(10, 5)
    
    enc.set_projection_schedule(1, 10, 5, 2)
    enc.encode_first_layer(x, use_projected=True)
    enc.encode_first_layer(x, use_projected=False)
    enc.first_layer_outputs(x, use_projected=True)
    enc.get_projector()

def test_deep_decoder_no_dropout():
    dec = ModalityDecoder(2, 5, dropout=0.0)
    z = torch.randn(10, 2)
    res = dec(z)
    assert res.shape == (10, 5)

def test_deep_lend_simr_model_branches():
    model = LENDSiMRModel(input_dims=[5, 6], latent_dim=2, sparseness_quantile=0.1)
    x1 = torch.randn(10, 1)
    x2 = torch.randn(10, 1)
    model.initialize_v([x1, x2], k=2)

def test_deep_ned_simr_transform():
    model = NEDSiMRModel(input_dims=[5], latent_dim=2)
    x = torch.randn(10, 5)
    u = model.transform([x])
    assert u.shape[0] == 10

def test_deep_ned_shared_private_transform():
    model = NEDSharedPrivateSiMRModel(input_dims=[5], shared_latent_dim=2, private_latent_dim=2)
    x = torch.randn(10, 5)
    u = model.transform([x])
    assert u.shape[0] == 10

def test_deep_calculate_sim_loss_nc_logcosh():
    latents = [torch.randn(10, 2), torch.randn(10, 2)]
    u_shared = torch.randn(10, 2)
    loss, diag = calculate_sim_loss(latents, u_shared, energy_type="nc")
    loss, diag = calculate_sim_loss(latents, u_shared, energy_type="logcosh")
    
    zero_latents = [torch.zeros(10, 2)]
    zero_u = torch.zeros(10, 2)
    loss, diag = calculate_sim_loss(zero_latents, zero_u, energy_type="nc")

def test_deep_train_loop_freeze_private():
    x1 = torch.randn(20, 5)
    res = ned_simr_shared_private([x1], k=2, private_k=2, epochs=2, shared_warmup_epochs=1, device="cpu")
    assert "u" in res

def test_deep_lend_simr_sparseness_alias():
    x1 = torch.randn(10, 5)
    res = lend_simr([x1], k=2, epochs=1, sparseness=0.1, device="cpu")
    assert "u" in res

def test_deep_deep_simr_alias():
    x1 = torch.randn(10, 5)
    res = deep_simr([x1], k=2, epochs=1, device="cpu")
    assert "u" in res

def test_deep_predict_deep_ned():
    x1 = torch.randn(10, 5)
    res = ned_simr([x1], k=2, epochs=1, device="cpu")
    pred = predict_deep([x1], res, device="cpu")
    assert "u" in pred
