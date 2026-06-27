import pytest
import torch
import numpy as np
from pysimlr.flows import flow_simr, flow_simr_v

def test_flow_simr_dynamic_weights():
    torch.manual_seed(42)
    xs = [torch.randn(30, 5) for _ in range(3)]
    res = flow_simr(xs, k=2, epochs=5, batch_size=15, warmup_epochs=0, dynamic_weights=True, verbose=False)
    
    assert 'u' in res
    assert 'modality_weights' in res
    assert 'mai' in res
    
    weights = res['modality_weights']
    mai = res['mai']
    
    assert len(weights) == 3
    assert len(mai) == 3
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert np.isclose(np.sum(weights), 1.0)

def test_flow_simr_v_dynamic_weights():
    torch.manual_seed(42)
    xs = [torch.randn(30, 5) for _ in range(3)]
    res = flow_simr_v(xs, k=2, epochs=5, batch_size=15, warmup_epochs=0, dynamic_weights=True, verbose=False)
    
    assert 'u' in res
    assert 'modality_weights' in res
    assert 'mai' in res
    
    weights = res['modality_weights']
    mai = res['mai']
    
    assert len(weights) == 3
    assert len(mai) == 3
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert np.isclose(np.sum(weights), 1.0)

def test_flow_simr_different_metrics():
    torch.manual_seed(42)
    xs = [torch.randn(30, 5) for _ in range(3)]
    for metric in ['procrustes_r2', 'procrustes_r2_sharp', 'cca', 'rvcoef']:
        res = flow_simr(xs, k=2, epochs=2, batch_size=15, warmup_epochs=0, dynamic_weights=True, mai_metric=metric, verbose=False)
        assert res['model'].mai_metric == metric
        assert 'modality_weights' in res

def test_delayed_dynamic_weights_warmup():
    torch.manual_seed(42)
    xs = [torch.randn(30, 5) for _ in range(3)]
    
    # If dynamic_weights_start is 10 and we run only 5 epochs, the weights should be uniform (1/3 each)
    res = flow_simr(xs, k=2, epochs=5, batch_size=15, warmup_epochs=0, dynamic_weights=True, dynamic_weights_start=10, verbose=False)
    weights = res['modality_weights']
    assert np.allclose(weights, np.ones(3)/3)

