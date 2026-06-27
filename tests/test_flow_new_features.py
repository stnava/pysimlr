import pytest
import torch
import numpy as np
from pysimlr.flows import flow_simr, flow_simr_v

def test_flow_simr_v_positivity_either():
    """Test that positivity='either' allows negative weights in linear projection V."""
    torch.manual_seed(42)
    xs = [torch.randn(30, 5) for _ in range(3)]
    
    # Run Flow-SiMLR-V with positivity='either'
    res = flow_simr_v(xs, k=2, epochs=3, batch_size=15, warmup_epochs=0, positivity='either', verbose=False)
    
    # Extract projection matrices V and convert to numpy
    v_mats = [v.numpy() if hasattr(v, 'numpy') else np.array(v) for v in res['v']]
    assert len(v_mats) == 3
    
    # Verify that at least one weight is negative (since positivity='either' allows negative weights)
    has_negative = False
    for v in v_mats:
        if np.any(v < 0.0):
            has_negative = True
            break
    assert has_negative, "positivity='either' should allow negative weights in V matrix."

def test_flow_simr_v_positivity_positive():
    """Test that positivity='positive' clamps weights in V to be non-negative."""
    torch.manual_seed(42)
    xs = [torch.randn(30, 5) for _ in range(3)]
    
    # Run Flow-SiMLR-V with default positivity='positive'
    res = flow_simr_v(xs, k=2, epochs=3, batch_size=15, warmup_epochs=0, positivity='positive', verbose=False)
    
    # Extract projection matrices V and convert to numpy
    v_mats = [v.numpy() if hasattr(v, 'numpy') else np.array(v) for v in res['v']]
    for v in v_mats:
        # Check that all elements are non-negative
        assert np.all(v >= -1e-7), "positivity='positive' should restrict weights to be non-negative."

def test_flow_weight_history_logging():
    """Test that weight history is logged in flow_simr and flow_simr_v when dynamic_weights is True."""
    torch.manual_seed(42)
    xs = [torch.randn(30, 5) for _ in range(3)]
    
    # Test flow_simr
    res_flow = flow_simr(xs, k=2, epochs=4, batch_size=15, warmup_epochs=0, dynamic_weights=True, verbose=False)
    assert 'weight_history' in res_flow
    assert len(res_flow['weight_history']) == 4
    for entry in res_flow['weight_history']:
        assert len(entry) == 3
        assert np.isclose(np.sum(entry), 1.0)
        
    # Test flow_simr_v
    res_flow_v = flow_simr_v(xs, k=2, epochs=4, batch_size=15, warmup_epochs=0, dynamic_weights=True, verbose=False)
    assert 'weight_history' in res_flow_v
    assert len(res_flow_v['weight_history']) == 4
    for entry in res_flow_v['weight_history']:
        assert len(entry) == 3
        assert np.isclose(np.sum(entry), 1.0)

def test_delayed_dynamic_weights_start_epoch():
    """Test that dynamic gating weights remain exactly uniform until the dynamic_weights_start epoch is reached."""
    u = torch.randn(30, 2)
    xs = [
        u @ torch.randn(2, 5) + 0.01 * torch.randn(30, 5),
        u @ torch.randn(2, 5) + 0.5 * torch.randn(30, 5),
        torch.randn(30, 5)
    ]
    
    # Run Flow-SiMLR-V with dynamic_weights_start = 3 and epochs = 5
    res = flow_simr_v(xs, k=2, epochs=5, batch_size=15, warmup_epochs=0, dynamic_weights=True, dynamic_weights_start=3, verbose=False)
    
    weight_hist = res['weight_history']
    assert len(weight_hist) == 5
    
    # For epochs 1, 2, and 3 (0-indexed 0, 1, 2), weights must be exactly uniform (1/3 each)
    for t in range(3):
        assert np.allclose(weight_hist[t], np.ones(3)/3)
        
    # For epoch 5 (0-indexed 4), after delay, the weights can start updating and diverge from uniform
    last_weights = weight_hist[-1]
    is_uniform = np.allclose(last_weights, np.ones(3)/3)
    assert not is_uniform, "Gating weights should update and diverge from uniform after dynamic_weights_start epoch."
