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
    # retraction should be "soft_polar"
    assert kwargs['retraction'] == "soft_polar"

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
        simlr([x1], k=2, iterations=1, constraint="nonex0.5")

    assert mock_nsa.called, "nsa_flow_orth should be called when weight > 0 even if constraint is 'none'"
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

if __name__ == "__main__":
    pytest.main([__file__])
