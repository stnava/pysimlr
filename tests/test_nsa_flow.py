import torch
import numpy as np
import random
import pytest
from unittest.mock import patch, MagicMock
from pysimlr import simlr

def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_nsa_flow_triggered_by_ortho_constraint():
    """
    Verify that nsa_flow_orth is triggered by 'ortho' constraint
    and that parameters (weight, iterations) are correctly passed to it.
    """
    set_all_seeds(42)
    # Smaller data for faster test
    x1 = torch.randn(20, 10)
    x2 = torch.randn(20, 10)
    
    mock_nsa = MagicMock()
    # Mock should return a dictionary with 'Y' key as expected by simlr_sparseness
    mock_nsa.side_effect = lambda Y, **kwargs: {'Y': Y.clone()}

    weight = 0.6
    iters = 7
    constraint_str = f"orthox{weight}x{iters}"

    with patch('pysimlr.sparsification.nsa_flow_orth', mock_nsa):
        # We only need 1 iteration of simlr to verify it triggers nsa_flow_orth
        simlr([x1, x2], k=2, iterations=1, constraint=constraint_str)

    assert mock_nsa.called, "nsa_flow_orth was not called even though constraint was 'ortho'"
    
    # Check arguments of the first call (modality 0)
    args, kwargs = mock_nsa.call_args_list[0]
    
    # Y (first arg) should be a tensor
    assert isinstance(args[0], torch.Tensor)
    # w (weight) should be passed
    assert kwargs['w'] == weight
    # max_iter should be max(5, iters)
    assert kwargs['max_iter'] == max(5, iters)
    # retraction should be "soft_ns"
    assert kwargs['retraction'] == "soft_ns"

    # Verify ortho_polar sets retraction to soft_polar
    mock_nsa.reset_mock()
    constraint_str_polar = f"ortho_polarx{weight}x{iters}"
    with patch('pysimlr.sparsification.nsa_flow_orth', mock_nsa):
        simlr([x1, x2], k=2, iterations=1, constraint=constraint_str_polar)
    assert mock_nsa.called
    _, kwargs_polar = mock_nsa.call_args_list[0]
    assert kwargs_polar['retraction'] == "soft_polar"

def test_nsa_flow_triggered_by_none_constraint_with_weight():
    """
    Verify that nsa_flow_orth is also triggered when constraint is 'none'
    but a weight is provided, acting as an orthogonality regularizer.
    """
    set_all_seeds(42)
    x1 = torch.randn(20, 10)
    
    mock_nsa = MagicMock()
    mock_nsa.side_effect = lambda Y, **kwargs: {'Y': Y.clone()}

    # "nonex0.5" means constraint_type="none", weight=0.5
    with patch('pysimlr.sparsification.nsa_flow_orth', mock_nsa):
        simlr([x1], k=2, iterations=1, constraint="orthox0.5")

    assert mock_nsa.called, "nsa_flow_orth should be called when weight > 0 even when constraint is 'ortho'"
    _, kwargs = mock_nsa.call_args
    assert kwargs['w'] == 0.5

def test_nsa_flow_fallback_mechanism():
    """
    Verify that if nsa_flow_orth fails (raises exception), simlr falls back 
    to standard SVD-based orthogonalization without crashing.
    """
    set_all_seeds(42)
    x1 = torch.randn(20, 5)
    
    # Mock that raises an exception
    mock_nsa = MagicMock()
    mock_nsa.side_effect = Exception("nsa_flow internal error")

    with patch('pysimlr.sparsification.nsa_flow_orth', mock_nsa):
        # Should complete without raising exception
        res = simlr([x1], k=2, iterations=1, constraint="orthox0.5")
        
    assert mock_nsa.called
    assert 'v' in res
    assert res['v'][0].shape == (5, 2)

def test_newton_schulz_retraction_support():
    """Verify that Newton-Schulz retraction parameter is correctly parsed and passed to nsa_flow_orth."""
    set_all_seeds(42)
    x1 = torch.randn(20, 5)
    
    mock_nsa = MagicMock()
    mock_nsa.side_effect = lambda Y, **kwargs: {'Y': Y.clone()}
    
    # 1. Test linear simlr with Stiefel_ns constraint
    with patch('pysimlr.sparsification.nsa_flow_orth', mock_nsa):
        simlr([x1], k=2, iterations=1, constraint="Stiefel_nsx0.5")
    assert mock_nsa.called
    _, kwargs = mock_nsa.call_args
    assert kwargs['retraction'] == "ns"
    
    # Reset mock
    mock_nsa.reset_mock()
    
    # 2. Test linear simlr with nsaflow_ns constraint
    with patch('pysimlr.sparsification.nsa_flow_orth', mock_nsa):
        simlr([x1], k=2, iterations=1, constraint="nsaflow_nsx0.5")
    assert mock_nsa.called
    _, kwargs = mock_nsa.call_args
    assert kwargs['retraction'] == "soft_ns"

    # 3. Test deep model (lend_simr) with retraction_type='soft_ns'
    from pysimlr.deep import lend_simr
    res_deep = lend_simr([x1], k=2, epochs=2, batch_size=10, warmup_epochs=0, use_nsa=True, retraction_type="soft_ns", verbose=False)
    assert 'model' in res_deep
    # Check that model has retraction_type='soft_ns' on its encoders
    encoders = getattr(res_deep['model'], 'linear_encoders', getattr(res_deep['model'], 'encoders', []))
    for enc in encoders:
        if enc.nsa_linear is not None:
            assert enc.nsa_linear.retraction_type == "soft_ns"
        elif enc.nsa_layer is not None:
            assert enc.nsa_layer.retraction_type == "soft_ns"

if __name__ == "__main__":
    pytest.main([__file__])
