import torch
import pytest
import numpy as np
from pysimlr.sparsification import (
    optimize_indicator_matrix,
    indicator_opt_both_ways,
    rank_based_matrix_segmentation,
    orthogonalize_and_q_sparsify,
    project_to_orthonormal_nonnegative,
    project_to_partially_orthonormal_nonnegative,
    simlr_sparseness
)

def test_optimize_indicator_matrix():
    m = torch.randn(10, 5)
    m_opt = optimize_indicator_matrix(m, verbose=True)
    assert m_opt.shape == m.shape
    # Each column should have at most one 1
    I = (m_opt != 0).float()
    assert torch.all(torch.sum(I, dim=0) <= 1.0)

def test_indicator_opt_both_ways():
    m = torch.randn(10, 5)
    m_opt = indicator_opt_both_ways(m)
    assert m_opt.shape == m.shape

def test_rank_based_matrix_segmentation():
    v = torch.randn(10, 5)
    v_seg = rank_based_matrix_segmentation(v, sparseness_quantile=0.5, basic=True)
    assert v_seg.shape == v.shape
    
    v_seg_either = rank_based_matrix_segmentation(v, sparseness_quantile=0.5, basic=True, positivity="either")
    assert v_seg_either.shape == v.shape

    v_seg_transpose = rank_based_matrix_segmentation(v, sparseness_quantile=0.5, basic=True, transpose=True)
    assert v_seg_transpose.shape == v.shape

def test_orthogonalize_and_q_sparsify():
    v = torch.randn(20, 5)
    # Test different algs
    v_ortho = orthogonalize_and_q_sparsify(v, sparseness_alg="orthorank")
    assert v_ortho.shape == v.shape
    
    v_basic = orthogonalize_and_q_sparsify(v, sparseness_alg="basic")
    assert v_basic.shape == v.shape
    
    v_q = orthogonalize_and_q_sparsify(v, sparseness_quantile=0.5, soft_thresholding=True)
    assert v_q.shape == v.shape
    
    # All zeros
    v_zero = torch.zeros(10, 5)
    assert torch.all(orthogonalize_and_q_sparsify(v_zero) == 0)

def test_project_to_orthonormal_nonnegative():
    x = torch.randn(10, 5)
    v = project_to_orthonormal_nonnegative(x, max_iter=10)
    assert v.shape == x.shape
    assert torch.all(v >= -1e-7)

def test_project_to_partially_orthonormal_nonnegative():
    x = torch.randn(10, 5)
    v = project_to_partially_orthonormal_nonnegative(x, max_iter=2)
    assert v.shape == x.shape

def test_simlr_sparseness():
    v = torch.randn(10, 5)
    # Test different combinations
    v_out = simlr_sparseness(v, constraint_type="Stiefel", sparseness_alg="nnorth", positivity="positive")
    assert v_out.shape == v.shape
    assert torch.all(v_out >= -1e-7)
    
    v_out2 = simlr_sparseness(v, constraint_type="ortho", constraint_weight=0.5)
    assert v_out2.shape == v.shape
    
    v_out3 = simlr_sparseness(v, energy_type="cca")
    assert v_out3.shape == v.shape

def test_simlr_sparseness_with_smoothing():
    v = torch.randn(10, 5)
    smoothing = torch.eye(10)
    v_out = simlr_sparseness(v, smoothing_matrix=smoothing)
    assert v_out.shape == v.shape
