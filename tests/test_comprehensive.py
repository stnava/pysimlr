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
    simlr,
    nnh_embed,
    predict_simlr,
    estimate_rank,
    decompose_energy,
    rvcoef,
    adjusted_rvcoef,
    simlr_path,
    permutation_test
)

def test_utils_comprehensive():
    seed = set_seed_based_on_time()
    assert isinstance(seed, int)
    desc = ["apple", "banana", "cherry", "date"]
    assert len(multigrep(["a", "e"], desc, intersect=True)) == 2
    assert len(multigrep(["xyz"], desc)) == 0
    df = pd.DataFrame({'left_1': [1, 2, 3], 'right_1': [3, 2, 1], 'other': [5, 5, 5]})
    res = map_asym_var(df, ['left_1'])
    assert 'Asym_1' in res.columns
    res = map_lr_average_var(df, ['left_1'])
    assert 'LRAVG_1' in res.columns
    nms = get_names_from_dataframe(["left", "right"], df)
    assert len(nms) == 2
    nms_ex = get_names_from_dataframe(["_1"], df, exclusions=["right"])
    assert nms_ex == ["left_1"]
    x = torch.randn(10, 5)
    y = torch.randn(10, 5)
    rv = rvcoef(x, y)
    assert 0 <= rv <= 1.0
    adj_rv = adjusted_rvcoef(x, y)
    assert isinstance(adj_rv, float)

def test_svd_comprehensive():
    x = torch.randn(50, 20)
    u, s, v = ba_svd(x, nu=5, nv=5, na_to_noise=True)
    assert u.shape == (50, 5)
    x_nan = x.clone()
    x_nan[0, 0] = float('nan')
    u2, s2, v2 = ba_svd(x_nan, na_to_noise=True)
    assert not torch.any(torch.isnan(u2))
    u3, s3, v3 = ba_svd(x, divide_by_max=True, nv=100)
    assert s3.shape[0] <= 20
    res = safe_pca(x, nc=100, center=True, scale=True)
    assert res['u'].shape[1] <= 20
    res_w = whiten_matrix(x)
    assert res_w['whitened_matrix'].shape == (50, 20)
    r = torch.tensor([0.5, 1.0])
    res_ms = multiscale_svd(x, r, locn=2, nev=2, knn=5, verbose=True)
    assert res_ms['evals_vs_scale'].shape == (2, 2)

def test_sparse_comprehensive():
    x = torch.randn(10, 5)
    y = torch.randn(10, 3)
    for metric in ["euclidean", "correlation", "covariance", "gaussian"]:
        smat = sparse_distance_matrix(x, k=2, kmetric=metric, sigma=1.0 if metric=="gaussian" else None, r=0.5)
        assert smat.shape == (5, 5)
    smat = sparse_distance_matrix(x, k=2, sinkhorn=True)
    assert smat.shape == (5, 5)
    smat_xy = sparse_distance_matrix_xy(x, y, k=2, kmetric="correlation", r=0.1)
    assert smat_xy.shape == (5, 3)

def test_sparsification_comprehensive():
    m = torch.randn(5, 5)
    res_ind = optimize_indicator_matrix(m, preprocess=True, verbose=True)
    assert res_ind.shape == (5, 5)
    res_both = indicator_opt_both_ways(m, verbose=True)
    assert res_both.shape == (5, 5)
    for pos in ["positive", "negative", "either"]:
        res = rank_based_matrix_segmentation(m, sparseness_quantile=0.5, basic=True, positivity=pos)
        assert res.shape == (5, 5)
    res = orthogonalize_and_q_sparsify(m, orthogonalize=True, unit_norm=True)
    assert res.shape == (5, 5)
    res_pos = orthogonalize_and_q_sparsify(m, positivity="positive", sparseness_quantile=0.1)
    assert torch.all(res_pos >= 0)
    res_neg = orthogonalize_and_q_sparsify(m, positivity="negative", sparseness_quantile=0.1)
    assert torch.all(res_neg <= 0)
    res_basic = orthogonalize_and_q_sparsify(m, sparseness_alg="basic")
    assert res_basic.shape == (5, 5)
    m_zero = torch.zeros(5, 5)
    res_zero = orthogonalize_and_q_sparsify(m_zero)
    assert torch.all(res_zero == 0)

def test_regression_comprehensive():
    x = torch.randn(20, 10)
    basis_df = pd.DataFrame({'b1': np.random.randn(20), 'b2': np.random.randn(20)})
    res = smooth_matrix_prediction(x, basis_df, iterations=2, verbose=True)
    assert res['v'].shape == (10, 2)
    y = torch.randn(20)
    res = smooth_regression(x, y, iterations=2, verbose=True)
    assert res['v'].shape == (10, 2)

def test_optimizers_comprehensive():
    v = [torch.randn(10, 2)]
    from pysimlr.optimizers import create_optimizer
    opts = ["hybrid_adam", "adam", "rmsprop", "gd", "nsa_flow",
            "torch_adamw", "torch_adagrad", "torch_nadam", "torch_lbfgs"]
    for opt_type in opts:
        try:
            opt = create_optimizer(opt_type, v, learning_rate=0.01)
            grad = torch.randn(10, 2)
            def energy_fn(v_cand): return torch.sum(v_cand**2)
            updated_v = opt.step(0, v[0], grad, energy_fn)
            assert updated_v.shape == (10, 2)
        except ImportError:
            pytest.skip("optional package not installed")

def test_simlr_comprehensive():
    x1 = torch.randn(20, 10)
    x2 = torch.randn(20, 10)
    d1 = torch.randn(3, 10)
    d2 = torch.randn(3, 10)
    res = simlr([x1, x2], k=3, iterations=5, 
               energy_type="acc",
               domain_matrices=[d1, d2],
               domain_lambdas=0.1,
               optimizer_type="hybrid_adam",
               verbose=True)
    assert res['u'].shape == (20, 3)
    
    # Test non-linear ICA energy types
    for nle in ["logcosh", "exp", "gauss", "kurtosis"]:
        res_nle = simlr([x1, x2], k=2, iterations=2, energy_type=nle)
        assert len(res_nle['energy']) == 2
        
    # Energy decomposition
    dec = decompose_energy([x1, x2], res, energy_type="acc")
    assert len(dec['modality_energies']) == 2
    assert dec['feature_importances'][0].shape[0] == 10

def test_nnh_embed_comprehensive():
    n = 20
    df = pd.DataFrame({
        'T1Hier_vol1': np.random.randn(n),
        'T1Hier_vol2': np.random.randn(n),
        'DTI_fa1': np.random.randn(n),
        'rsfMRI_conn1': np.random.randn(n),
        'T1Hier_resnetGrade': np.linspace(1.0, 1.1, n),
        'age': np.random.randn(n)
    })
    res = nnh_embed(df, nsimlr=2, iterations=2, covariates=['mean'], verbose=True)
    assert 'u' in res
    assert res['u'].shape[1] == 2

def test_predict_rank_comprehensive():
    x1 = torch.randn(30, 10)
    x2 = torch.randn(30, 8)
    k_elbow = estimate_rank([x1, x2], n_permutations=0)
    assert isinstance(k_elbow, int)
    k_perm = estimate_rank([x1, x2], n_permutations=2)
    assert isinstance(k_perm, int)
    res = simlr([x1, x2], k=3, iterations=2)
    pred = predict_simlr([x1, x2], res)
    assert 'u' in pred
    assert len(pred['errors']) == 2

def test_paths_comprehensive():
    x1 = torch.randn(30, 10)
    x2 = torch.randn(30, 10)
    x3 = torch.randn(30, 10)
    pm = [[1], [0, 2], [1]]
    res = simlr_path([x1, x2, x3], k=2, path_model=pm, iterations=5, verbose=True)
    assert len(res['u']) == 3
    assert res['u'][0].shape == (30, 2)
    pt_res = permutation_test([x1, x2], k=2, n_permutations=5, iterations=2)
    assert 'p_value' in pt_res
    assert isinstance(pt_res['p_value'], float)
