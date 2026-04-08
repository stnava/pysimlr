import torch
import pandas as pd
import pytest
import numpy as np
from pysimlr import (
    set_seed_based_on_time,
    multigrep,
    get_names_from_dataframe,
    map_asym_var,
    map_lr_average_var,
    ba_svd,
    safe_pca,
    whiten_matrix,
    multiscale_svd,
    sparse_distance_matrix,
    sparse_distance_matrix_xy,
    optimize_indicator_matrix,
    indicator_opt_both_ways,
    rank_based_matrix_segmentation,
    orthogonalize_and_q_sparsify,
    smooth_matrix_prediction,
    smooth_regression,
    simlr
)

def test_utils_comprehensive():
    # set_seed_based_on_time
    seed = set_seed_based_on_time()
    assert isinstance(seed, int)

    # multigrep empty
    assert len(multigrep(["xyz"], ["abc"])) == 0

    # map_asym_var
    df = pd.DataFrame({
        'left_1': [1, 2, 3],
        'right_1': [3, 2, 1],
        'other': [0, 0, 0]
    })
    res = map_asym_var(df, ['left_1'])
    assert 'Asym_1' in res.columns
    assert res['Asym_1'].tolist() == [2, 0, 2]
    
    # map_lr_average_var
    res = map_lr_average_var(df, ['left_1'])
    assert 'LRAVG_1' in res.columns
    assert res['LRAVG_1'].tolist() == [2.0, 2.0, 2.0]

def test_svd_comprehensive():
    # safe_pca with scaling and centering
    x = torch.randn(20, 10)
    res = safe_pca(x, nc=3, center=True, scale=True)
    assert res['u'].shape == (20, 3)
    
    # ba_svd divide_by_max
    u, s, v = ba_svd(x, divide_by_max=True)
    assert u.shape[0] == 20
    
    # multiscale_svd with knn
    r = torch.tensor([0.5, 1.0])
    res = multiscale_svd(x, r, locn=2, nev=2, knn=5)
    assert res['evals_vs_scale'].shape == (2, 2)

def test_sparse_comprehensive():
    x = torch.randn(10, 5)
    y = torch.randn(10, 3)
    
    # sparse_distance_matrix with various metrics
    for metric in ["euclidean", "correlation", "covariance", "gaussian"]:
        smat = sparse_distance_matrix(x, k=2, kmetric=metric, sigma=1.0 if metric=="gaussian" else None)
        assert smat.shape == (5, 5)
    
    # sinkhorn normalization
    smat = sparse_distance_matrix(x, k=2, sinkhorn=True)
    assert smat.shape == (5, 5)
    
    # sparse_distance_matrix_xy correlation
    smat_xy = sparse_distance_matrix_xy(x, y, k=2, kmetric="correlation")
    assert smat_xy.shape == (5, 3)

def test_sparsification_comprehensive():
    m = torch.randn(5, 5)
    # positivity constraints
    for pos in ["positive", "negative", "either"]:
        res = rank_based_matrix_segmentation(m, sparseness_quantile=0.5, basic=True, positivity=pos)
        assert res.shape == (5, 5)
        
    # orthogonalize_and_q_sparsify variations
    res = orthogonalize_and_q_sparsify(m, orthogonalize=True, unit_norm=True)
    assert res.shape == (5, 5)
    
    res = orthogonalize_and_q_sparsify(m, positivity="positive")
    assert torch.all(res >= 0)

def test_regression_comprehensive():
    x = torch.randn(20, 10)
    basis_df = pd.DataFrame({'b1': np.random.randn(20), 'b2': np.random.randn(20)})
    
    # smooth_matrix_prediction verbose
    res = smooth_matrix_prediction(x, basis_df, iterations=2, verbose=True)
    assert res['v'].shape == (10, 2)
    
    # smooth_regression verbose
    y = torch.randn(20)
    res = smooth_regression(x, y, iterations=2, verbose=True)
    assert res['v'].shape == (10, 2)

def test_optimizers_comprehensive():
    v = [torch.randn(10, 2)]
    from pysimlr.optimizers import create_optimizer
    
    opts = ["hybrid_adam", "adam", "rmsprop", "gd", 
            "torch_adamw", "torch_adagrad", "torch_nadam", "torch_lbfgs"]
    
    for opt_type in opts:
        opt = create_optimizer(opt_type, v)
        grad = torch.randn(10, 2)
        def energy_fn(v_cand): return torch.sum(v_cand**2)
        
        updated_v = opt.step(0, v[0], grad, energy_fn)
        assert updated_v.shape == (10, 2)

def test_simlr_comprehensive():
    x1 = torch.randn(20, 10)
    x2 = torch.randn(20, 10)
    # Test with different optimizer type
    res = simlr([x1, x2], k=2, iterations=2, optimizer_type="torch_adamw")
    assert res['u'].shape == (20, 2)
