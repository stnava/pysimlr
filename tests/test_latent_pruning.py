import torch
import numpy as np
from pysimlr.consensus import compute_shared_consensus
from pysimlr.deep import lend_simr

def test_consensus_pruning():
    torch.manual_seed(42)
    # 3 identical modalities and 1 corrupt modality
    u_true = torch.randn(10, 2)
    u_corrupt = torch.randn(10, 2)
    
    projs = [u_true + torch.randn(10, 2)*0.01 for _ in range(3)]
    projs.append(u_corrupt)
    
    # Without pruning
    u_star_no_prune = compute_shared_consensus(projs, topology="star")
    # With pruning
    u_star_prune = compute_shared_consensus(projs, topology="star", prune_threshold=0.1)
    
    assert u_star_prune is not None
    assert u_star_prune.shape == (10, 2)
    
    # With LOO
    u_loo_no_prune = compute_shared_consensus(projs, topology="loo")
    u_loo_prune = compute_shared_consensus(projs, topology="loo", prune_threshold=0.1)
    
    assert len(u_loo_prune) == 4
    
def test_lend_simr_pruning():
    torch.manual_seed(42)
    xs = [torch.randn(20, 5) for _ in range(4)]
    res = lend_simr(xs, k=2, epochs=2, batch_size=10, topology="star", prune_threshold=0.5, verbose=False)
    assert 'u' in res
