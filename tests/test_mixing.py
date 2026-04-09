import torch
import numpy as np
from pysimlr import lend_simr, predict_simlr

def test_lend_mix_ica():
    torch.manual_seed(42)
    n, p1, p2, k = 100, 20, 20, 2
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)
    
    print("Running LEND SiMR with mixing_algorithm='ica'...")
    # Smaller epochs for quick test
    res = lend_simr([x1, x2], k=k, epochs=5, mixing_algorithm='ica', verbose=True)
    
    assert 'u' in res
    assert res['u'].shape == (n, k)
    print("LEND Mix ICA test PASSED")

def test_simlr_mix_pca():
    from pysimlr import simlr
    torch.manual_seed(42)
    n, p1, p2, k = 100, 20, 20, 2
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)
    
    print("Running SiMLR with mixing_algorithm='pca'...")
    res = simlr([x1, x2], k=k, iterations=5, mixing_algorithm='pca', verbose=True)
    
    assert 'u' in res
    assert res['u'].shape == (n, k)
    print("SiMLR Mix PCA test PASSED")

if __name__ == "__main__":
    test_lend_mix_ica()
    test_simlr_mix_pca()
