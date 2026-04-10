import torch
import numpy as np
import random
import pytest
from unittest.mock import patch
from pysimlr import simlr

def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_nsa_flow_impacts_v():
    set_all_seeds(42)
    x1 = torch.randn(50, 20)
    x2 = torch.randn(50, 15)
    
    # SVD Fallback (nsa_flow_orth is None)
    set_all_seeds(42)
    with patch('pysimlr.sparsification.nsa_flow_orth', None):
        res_svd = simlr([x1, x2], k=3, iterations=2, constraint="orthox0.5x5")
        v_svd = res_svd['v']

    # Mock nsa_flow_orth
    def dummy_nsa_flow_orth(Y, w=0.5, **kwargs):
        # We must return a modified Y that doesn't trigger the try-except fallback
        return {'Y': Y * 0.5}

    set_all_seeds(42)
    with patch('pysimlr.sparsification.nsa_flow_orth', dummy_nsa_flow_orth):
        res_nsa = simlr([x1, x2], k=3, iterations=2, constraint="orthox0.5x5")
        v_nsa = res_nsa['v']

    diff_v1 = torch.max(torch.abs(v_svd[0] - v_nsa[0])).item()
    diff_v2 = torch.max(torch.abs(v_svd[1] - v_nsa[1])).item()

    assert diff_v1 > 1e-4, f"v1 was not impacted by nsa_flow (diff: {diff_v1})"
    assert diff_v2 > 1e-4, f"v2 was not impacted by nsa_flow (diff: {diff_v2})"
    
if __name__ == "__main__":
    pytest.main([__file__])
