import torch
import numpy as np
from pysimlr.deep import lend_simr

def test_dynamic_weights_initialization():
    torch.manual_seed(42)
    xs = [torch.randn(20, 5) for _ in range(3)]
    res = lend_simr(xs, k=2, epochs=2, batch_size=10, topology="star", dynamic_weights=True, verbose=False)
    
    assert 'u' in res
    assert 'modality_weights' in res
    assert 'mci' in res
    
    weights = res['modality_weights']
    mci = res['mci']
    
    assert len(weights) == 3
    assert len(mci) == 3
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert np.isclose(np.sum(weights), 1.0)
    
def test_dynamic_weights_pruning_off():
    # Verify that dynamic_weights can run without prune_threshold
    torch.manual_seed(42)
    xs = [torch.randn(20, 5) for _ in range(3)]
    res = lend_simr(xs, k=2, epochs=2, batch_size=10, topology="loo", dynamic_weights=True, prune_threshold=None, verbose=False)
    
    assert 'modality_weights' in res
    assert len(res['modality_weights']) == 3

def test_dynamic_weights_with_pruning():
    # Verify that dynamic_weights can run WITH prune_threshold safely
    torch.manual_seed(42)
    xs = [torch.randn(20, 5) for _ in range(3)]
    res = lend_simr(xs, k=2, epochs=2, batch_size=10, topology="loo", dynamic_weights=True, prune_threshold=0.1, verbose=False)
    
    assert 'modality_weights' in res
    assert len(res['modality_weights']) == 3
