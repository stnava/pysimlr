"""Adversarial stress tests for Nomenclature Harmonization and backward compatibility in pysimlr.

Tests verify:
1. Top-level and module-level importability of legacy aliases:
   FlowSiMLRModel, FlowSiMLRVModel, flow_simlr, flow_simlr_v
2. Identity equivalence between legacy aliases and canonical Flow-SiMR classes/functions.
3. Instantiation and forward/backward passes of legacy model classes.
4. Execution of legacy functional APIs (flow_simlr, flow_simlr_v) with synthetic multi-modal data.
5. Downstream dispatch in nnh.py with legacy method='flow_simlr_v'.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from typing import List


def test_legacy_alias_imports_and_identity():
    """Verify legacy aliases can be imported from top-level pysimlr and pysimlr.flows."""
    import pysimlr
    from pysimlr import (
        FlowSiMRModel, FlowSiMRVModel, flow_simr, flow_simr_v,
        FlowSiMLRModel, FlowSiMLRVModel, flow_simlr, flow_simlr_v
    )
    from pysimlr.flows import (
        FlowSiMRModel as FlowSiMRModel_mod,
        FlowSiMRVModel as FlowSiMRVModel_mod,
        flow_simr as flow_simr_mod,
        flow_simr_v as flow_simr_v_mod,
        FlowSiMLRModel as FlowSiMLRModel_mod,
        FlowSiMLRVModel as FlowSiMLRVModel_mod,
        flow_simlr as flow_simlr_mod,
        flow_simlr_v as flow_simlr_v_mod,
    )

    # Verify top-level __all__ exports
    for sym in ['FlowSiMLRModel', 'FlowSiMLRVModel', 'flow_simlr', 'flow_simlr_v']:
        assert sym in pysimlr.__all__, f"Symbol {sym} missing from pysimlr.__all__"
        assert hasattr(pysimlr, sym), f"Symbol {sym} not found on pysimlr package"

    # Verify strict identity
    assert FlowSiMLRModel is FlowSiMRModel
    assert FlowSiMLRVModel is FlowSiMRVModel
    assert flow_simlr is flow_simr
    assert flow_simlr_v is flow_simr_v

    assert FlowSiMLRModel_mod is FlowSiMRModel_mod
    assert FlowSiMLRVModel_mod is FlowSiMRVModel_mod
    assert flow_simlr_mod is flow_simr_mod
    assert flow_simlr_v_mod is flow_simr_v_mod


@pytest.mark.parametrize("force_fallback", [False, True])
@pytest.mark.parametrize("mixing", ["newton", "medoid"])
def test_legacy_flow_simlr_model_instantiation_and_forward(force_fallback, mixing):
    """Stress test FlowSiMLRModel instantiation, forward pass, and backward pass."""
    from pysimlr import FlowSiMLRModel

    torch.manual_seed(42)
    n_samples = 24
    input_dims = [8, 12]
    latent_dim = 4

    model = FlowSiMLRModel(
        input_dims=input_dims,
        latent_dim=latent_dim,
        num_layers=2,
        hidden_dim=16,
        mixing_algorithm=mixing,
        force_fallback=force_fallback
    )

    X = [torch.randn(n_samples, d) for d in input_dims]

    # Training forward pass
    model.train()
    latents, reconstructions, u_shared = model(X)

    assert len(latents) == len(input_dims)
    assert len(reconstructions) == len(input_dims)

    # Verify backward pass through latents and reconstructions
    loss = sum(torch.mean(l**2) for l in latents) + sum(torch.mean(r**2) for r in reconstructions)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were computed"
    for g in grads:
        assert not torch.isnan(g).any(), "NaN gradient detected"
        assert not torch.isinf(g).any(), "Inf gradient detected"

    for i, d in enumerate(input_dims):
        assert latents[i].shape == (n_samples, latent_dim)
        assert reconstructions[i].shape == (n_samples, d)

    # Evaluation / inference pass
    model.eval()
    with torch.no_grad():
        eval_latents, eval_recons, eval_u = model(X)
        assert len(eval_latents) == len(input_dims)
        for i, d in enumerate(input_dims):
            assert eval_recons[i].shape == (n_samples, d)


@pytest.mark.parametrize("positivity", ["positive", "either"])
@pytest.mark.parametrize("retraction", ["soft_polar", "soft_ns"])
def test_legacy_flow_simlr_v_model_instantiation_and_forward(positivity, retraction):
    """Stress test FlowSiMLRVModel instantiation, initialization, forward pass, and backward pass."""
    from pysimlr import FlowSiMLRVModel

    torch.manual_seed(42)
    n_samples = 24
    input_dims = [10, 14]
    k = 3

    model = FlowSiMLRVModel(
        input_dims=input_dims,
        latent_dim=k,
        num_layers=2,
        hidden_dim=16,
        positivity=positivity,
        retraction_type=retraction,
        dynamic_weights=True,
        use_rank_mai=False
    )

    X = [torch.randn(n_samples, d) for d in input_dims]

    # Initialize weights via SVD
    model.initialize_weights(X)

    model.train()
    latents, reconstructions, u_shared = model(X)

    assert len(latents) == len(input_dims)
    assert len(reconstructions) == len(input_dims)

    # Backward pass
    loss = sum(torch.mean(l**2) for l in latents) + sum(torch.mean(r**2) for r in reconstructions)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were computed"
    for g in grads:
        assert not torch.isnan(g).any(), "NaN gradient detected"

    for i in range(len(input_dims)):
        assert latents[i].shape == (n_samples, k)
        assert reconstructions[i].shape == (n_samples, input_dims[i])


def test_legacy_functional_flow_simlr():
    """Verify execution of flow_simlr legacy functional entry point."""
    from pysimlr import flow_simlr

    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 30
    mats = [np.random.randn(n_samples, 8), np.random.randn(n_samples, 6)]

    result = flow_simlr(
        data_matrices=mats,
        k=3,
        epochs=3,
        batch_size=16,
        warmup_epochs=1,
        verbose=False
    )

    assert isinstance(result, dict)
    assert "model" in result
    assert "latents" in result
    assert "reconstructions" in result
    assert "loss_history" in result
    assert result.get("model_type") == "flow_simr"
    assert len(result["latents"]) == 2
    assert result["latents"][0].shape == (n_samples, 3)
    assert result["latents"][1].shape == (n_samples, 3)


def test_legacy_functional_flow_simlr_v():
    """Verify execution of flow_simlr_v legacy functional entry point."""
    from pysimlr import flow_simlr_v

    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 30
    mats = [np.random.randn(n_samples, 8), np.random.randn(n_samples, 6)]

    result = flow_simlr_v(
        data_matrices=mats,
        k=3,
        epochs=3,
        batch_size=16,
        warmup_epochs=1,
        verbose=False
    )

    assert isinstance(result, dict)
    assert "model" in result
    assert "v" in result
    assert "latents" in result
    assert "reconstructions" in result
    assert "loss_history" in result
    assert result.get("model_type") == "flow_simr_v"
    assert len(result["v"]) == 2
    assert result["v"][0].shape == (8, 3)
    assert result["v"][1].shape == (6, 3)


def test_nnh_extend_simlr_embedding_with_flow_simlr_v_epochs():
    """Verify extend_simlr_embedding_with_new_modalities with method='flow_simlr_v' using epochs."""
    from pysimlr.nnh import extend_simlr_embedding_with_new_modalities

    n = 20
    pymm = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y1': np.random.randn(n)
    })
    simlr_result = {
        'v': [pd.DataFrame(np.random.randn(2, 2), index=['x1', 'x2'], columns=['mod1_1', 'mod1_2'])],
        'feature_names': [['x1', 'x2']],
        'modality_names': ['mod1']
    }
    new_modalities = {'mod2': ['y1']}

    res = extend_simlr_embedding_with_new_modalities(
        pymm, simlr_result, new_modalities, method='flow_simlr_v', epochs=2
    )
    assert isinstance(res, dict)
    assert "updated_simlr_result" in res
    assert "simlr_fit" in res
    assert "blocks" in res


def test_nnh_extend_simlr_embedding_with_flow_simlr_v_iterations():
    """Adversarial stress test: pass iterations to extend_simlr_embedding_with_new_modalities with legacy method='flow_simlr_v'.
    
    Verifies that line 1149 in src/pysimlr/nnh.py correctly remaps iterations to epochs.
    """
    from pysimlr.nnh import extend_simlr_embedding_with_new_modalities

    n = 20
    pymm = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y1': np.random.randn(n)
    })
    simlr_result = {
        'v': [pd.DataFrame(np.random.randn(2, 2), index=['x1', 'x2'], columns=['mod1_1', 'mod1_2'])],
        'feature_names': [['x1', 'x2']],
        'modality_names': ['mod1']
    }
    new_modalities = {'mod2': ['y1']}

    # Should succeed without TypeError if backward compatibility were complete
    extend_simlr_embedding_with_new_modalities(
        pymm, simlr_result, new_modalities, method='flow_simlr_v', iterations=2
    )
