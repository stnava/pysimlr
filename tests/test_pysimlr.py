import torch
import pandas as pd
import pytest
import numpy as np
from pysimlr import (
    set_seed_based_on_time,
    multigrep,
    get_names_from_dataframe,
    ba_svd,
    whiten_matrix,
    sparse_distance_matrix,
    orthogonalize_and_q_sparsify,
    smooth_regression,
    simlr
)

def test_utils():
    # Test set_seed_based_on_time
    seed = set_seed_based_on_time()
    assert isinstance(seed, int)
    
    # Test multigrep
    desc = ["apple", "banana", "cherry", "date"]
    patterns = ["a", "e"]
    res = multigrep(patterns, desc, intersect=False)
    assert isinstance(res, torch.Tensor)
    assert len(res) == 4
    
    res = multigrep(patterns, desc, intersect=True)
    assert len(res) == 2
    
    # Test get_names_from_dataframe
    df = pd.DataFrame(columns=["test_1", "test_2", "other_1"])
    nms = get_names_from_dataframe(["test"], df)
    assert nms == ["test_1", "test_2"]

def test_svd():
    # Test ba_svd
    x = torch.randn(10, 5)
    u, s, v = ba_svd(x, nu=3, nv=2)
    assert u.shape == (10, 3)
    assert s.shape == (2,)
    assert v.shape == (5, 2)
    
    # Test whiten_matrix
    x = torch.randn(20, 5)
    res = whiten_matrix(x)
    whitened = res["whitened_matrix"]
    assert whitened.shape == (20, 5)

def test_sparse():
    x = torch.randn(10, 5)
    smat = sparse_distance_matrix(x, k=2, kmetric="euclidean")
    assert smat.shape == (5, 5)
    assert torch.any(smat != 0)

def test_sparsification():
    v = torch.randn(10, 3)
    v_sparse = orthogonalize_and_q_sparsify(v, sparseness_quantile=0.5)
    assert v_sparse.shape == (10, 3)
    assert torch.any(v_sparse == 0)

def test_regression():
    x = torch.randn(20, 10)
    y = x[:, 0] * 2 + x[:, 1] * -1 + torch.randn(20) * 0.1
    res = smooth_regression(x, y, iterations=5, nv=1)
    assert res["v"].shape == (10, 1)

def test_simlr_smoke():
    x1 = torch.randn(20, 10)
    x2 = torch.randn(20, 10)
    res = simlr([x1, x2], k=2, iterations=3)
    assert res["u"].shape == (20, 2)
    assert len(res["v"]) == 2
    assert res["v"][0].shape == (10, 2)
