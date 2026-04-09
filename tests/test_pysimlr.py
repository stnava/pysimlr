import os
import torch
import pandas as pd
import numpy as np
from pysimlr import (
    simlr,
    predict_simlr,
    estimate_rank,
    ba_svd,
    safe_pca,
    sparse_distance_matrix,
    smooth_regression
)

def test_simlr_basic():
    # Setup small synthetic data
    torch.manual_seed(42)
    n, p1, p2, k = 50, 20, 15, 2
    u_true = torch.randn(n, k)
    v1_true = torch.randn(p1, k)
    v2_true = torch.randn(p2, k)
    
    x1 = u_true @ v1_true.t() + 0.1 * torch.randn(n, p1)
    x2 = u_true @ v2_true.t() + 0.1 * torch.randn(n, p2)
    
    # Test estimate_rank
    k_est = estimate_rank([x1, x2], n_permutations=5)
    assert isinstance(k_est, int)
    assert k_est > 0
    
    # Test simlr
    res = simlr([x1, x2], k=k, iterations=10)
    assert "u" in res
    assert "v" in res
    assert len(res["v"]) == 2
    assert res["u"].shape == (n, k)
    
    # Test predict_simlr
    pred = predict_simlr([x1, x2], res)
    assert "u" in pred
    assert len(pred["reconstructions"]) == 2
    assert len(pred["errors"]) == 2

def test_svd():
    x = torch.randn(100, 20)
    u, s, v = ba_svd(x, nu=5, nv=5)
    assert u.shape == (100, 5)
    assert s.shape == (5,)
    assert v.shape == (20, 5)
    
    res = safe_pca(x, nc=5)
    assert res["u"].shape == (100, 5)
    assert res["v"].shape == (20, 5)

def test_sparse():
    x = torch.randn(10, 5)
    # Corrected call signature (no kmetric)
    smat = sparse_distance_matrix(x, k=2)
    assert smat.shape == (10, 10)

def test_regression():
    x = torch.randn(20, 10)
    y = x[:, 0] * 2 + x[:, 1] * -1 + torch.randn(20) * 0.1
    res = smooth_regression(x, y, iterations=5, nv=1)
    assert res["v"].shape == (10, 1)
