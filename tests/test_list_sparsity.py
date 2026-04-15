import torch
import numpy as np
from pysimlr import simlr, lend_simr

def test_simlr_list_sparsity():
    n, p1, p2 = 100, 50, 40
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)
    
    # Test with sparseness_quantile as a list
    res = simlr([x1, x2], k=5, iterations=2, sparseness_quantile=[0.1, 0.2], verbose=False)
    assert len(res['v']) == 2
    
    # Test with sparsity as a list
    res = simlr([x1, x2], k=5, iterations=2, sparsity=[0.3, 0.4], verbose=False)
    assert len(res['v']) == 2

    # Test with sparseness as a list
    res = simlr([x1, x2], k=5, iterations=2, sparseness=[0.5, 0.6], verbose=False)
    assert len(res['v']) == 2

def test_lend_simr_list_sparsity():
    n, p1, p2 = 100, 50, 40
    x1 = torch.randn(n, p1)
    x2 = torch.randn(n, p2)
    
    # Test with sparseness_quantile as a list
    res = lend_simr([x1, x2], k=5, epochs=2, sparseness_quantile=[0.1, 0.2], verbose=False)
    assert len(res['v']) == 2
    
    # Test with sparsity as a list
    res = lend_simr([x1, x2], k=5, epochs=2, sparsity=[0.3, 0.4], verbose=False)
    assert len(res['v']) == 2

    # Test with sparseness as a list
    res = lend_simr([x1, x2], k=5, epochs=2, sparseness=[0.5, 0.6], verbose=False)
    assert len(res['v']) == 2

if __name__ == "__main__":
    test_simlr_list_sparsity()
    test_lend_simr_list_sparsity()
    print("Tests passed!")
