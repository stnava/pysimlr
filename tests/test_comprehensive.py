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
    smooth_matrix_prediction,
    smooth_regression,
    nnh_embed,
    simlr,
    predict_simlr,
    estimate_rank,
    decompose_energy,
    simlr_path,
    permutation_test,
    deep_simr,
    lend_simr,
    ned_simr,
    ned_simr_shared_private,
    adjusted_rvcoef
)

def test_utils_comprehensive():
    seed = set_seed_based_on_time()
    assert isinstance(seed, int)
    desc = ["T1_vol", "DTI_fa", "rsfMRI_conn", "T1_thick"]
    patterns = ["T1", "vol"]
    res = multigrep(patterns, desc, intersect=True)
    assert len(res) == 1
    df = pd.DataFrame(columns=["T1_vol", "DTI_fa", "rsfMRI_conn", "age"])
    names = get_names_from_dataframe(["T1", "DTI"], df)
    assert "T1_vol" in names
    df_asym = pd.DataFrame({"left_hippo": [1.0, 2.0], "right_hippo": [1.1, 1.9]})
    df_res = map_asym_var(df_asym, ["left_hippo"])
    assert "Asym_hippo" in df_res.columns
    df_avg = pd.DataFrame({"left_hippo": [1.0, 2.0], "right_hippo": [1.2, 2.2]})
    df_res_avg = map_lr_average_var(df_avg, ["left_hippo"])
    assert "LRAVG_hippo" in df_res_avg.columns

def test_svd_comprehensive():
    x = torch.randn(20, 10)
    u, s, v = ba_svd(x, nu=5, nv=5)
    assert u.shape == (20, 5)
    pca_res = safe_pca(x, nc=3)
    assert pca_res['u'].shape == (20, 3)
    white_res = whiten_matrix(x)
    assert white_res['whitened_matrix'].shape == (20, 10)
    r = torch.tensor([0.5, 1.0, 2.0])
    ms_res = multiscale_svd(x, r, locn=5, nev=2)
    assert ms_res['evals_vs_scale'].shape == (3, 2)

def test_sparse_comprehensive():
    x = torch.randn(10, 5)
    y = torch.randn(12, 5)
    dist = sparse_distance_matrix(x, k=3)
    assert dist.shape == (10, 10)
    dist_xy = sparse_distance_matrix_xy(x, y, k=3)
    assert dist_xy.shape == (10, 12)

def test_regression_comprehensive():
    x = torch.randn(20, 10)
    y = torch.randn(20, 5)
    u_np = np.random.randn(20, 2)
    pred = smooth_matrix_prediction(x, u_np)
    assert pred.shape == (20, 2)
    reg_res = smooth_regression(x, y, iterations=2)
    assert 'v' in reg_res

def test_nnh_comprehensive():
    df = pd.DataFrame({'T1Hier_vol1': np.random.randn(20), 'T1Hier_vol2': np.random.randn(20), 'DTI_fa1': np.random.randn(20), 'age': np.random.randn(20)})
    res = nnh_embed(df, nsimlr=2, iterations=2, covariates=['age'])
    assert 'u' in res

def test_simlr_core_comprehensive():
    x1, x2 = torch.randn(50, 20), torch.randn(50, 15)
    k = estimate_rank([x1, x2], n_permutations=2)
    res = simlr([x1, x2], k=3, iterations=5)
    assert 'u' in res
    pred = predict_simlr([x1, x2], res)
    assert 'u' in pred
    decomp = decompose_energy([x1, x2], res)
    assert len(decomp['modality_energies']) == 2

def test_path_comprehensive():
    x1, x2 = torch.randn(30, 10), torch.randn(30, 8)
    res = simlr_path([x1, x2], k=2, iterations=2, path_model=[[1], [0]])
    assert len(res) > 0
    perm_res = permutation_test([x1, x2], k=2, n_permutations=2, iterations=2)
    assert 'p_value' in perm_res

def test_deep_simr_comprehensive():
    x1, x2 = torch.randn(50, 20), torch.randn(50, 15)
    res = deep_simr([x1, x2], k=5, epochs=2, batch_size=10, warmup_epochs=0, sim_weight=0.0, verbose=True)
    assert 'u' in res
    pred = predict_simlr([x1, x2], res)
    assert u_is_stable(pred['u'])

def test_lend_simr_comprehensive():
    x1, x2 = torch.randn(50, 20), torch.randn(50, 15)
    res = lend_simr([x1, x2], k=5, epochs=2, batch_size=10, sparseness_quantile=0.0, warmup_epochs=0, sim_weight=0.0, verbose=True)
    assert 'u' in res
    pred = predict_simlr([x1, x2], res)
    assert u_is_stable(pred['u'])

def test_ned_simr_comprehensive():
    x1, x2 = torch.randn(50, 20), torch.randn(50, 15)
    res = ned_simr([x1, x2], k=3, epochs=2, batch_size=10, warmup_epochs=0, sim_weight=0.0, verbose=True)
    assert 'u' in res
    pred = predict_simlr([x1, x2], res)
    assert u_is_stable(pred['u'])

def test_ned_shared_private_comprehensive():
    x1, x2 = torch.randn(50, 20), torch.randn(50, 15)
    res = ned_simr_shared_private([x1, x2], k=3, private_k=2, epochs=2, batch_size=10, warmup_epochs=0, 
                                  sim_weight=0.0, private_orthogonality_weight=0.0, private_variance_weight=0.0, verbose=True)
    assert 'u' in res
    pred = predict_simlr([x1, x2], res)
    assert u_is_stable(pred['u'])

def u_is_stable(u):
    return not torch.any(torch.isnan(u))

if __name__ == "__main__":
    pytest.main([__file__])
