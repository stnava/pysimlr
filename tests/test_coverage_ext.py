import torch
import numpy as np
import pandas as pd
import pytest
from pysimlr import (
    simlr,
    lend_simr,
    ned_simr,
    ned_simr_shared_private,
    create_optimizer,
    simlr_sparseness,
    orthogonalize_and_q_sparsify,
    simlr_perm,
    decompose_energy,
    ba_svd,
    safe_pca,
    whiten_matrix,
    multiscale_svd,
    adjusted_rvcoef
)

def test_optimizer_coverage():
    v_mats = [torch.randn(5, 2)]
    opts = ["adam", "nadam", "rmsprop", "gd", "armijo_gradient", 
            "bidirectional_armijo_gradient", "lookahead", 
            "bidirectional_lookahead", "hybrid_adam",
            "torch_adamw", "torch_adagrad"]
    
    for opt_type in opts:
        optimizer = create_optimizer(opt_type, v_mats, learning_rate=0.01)
        grad = torch.randn(5, 2)
        v_next = optimizer.step(0, v_mats[0], grad)
        assert v_next.shape == (5, 2)

def test_sparsification_coverage():
    from pysimlr.sparsification import optimize_indicator_matrix, rank_based_matrix_segmentation
    v = torch.randn(10, 3)
    res_soft = simlr_sparseness(v, sparseness_alg='soft', sparseness_quantile=0.5)
    assert res_soft.shape == (10, 3)
    res_nnorth = simlr_sparseness(v, sparseness_alg='nnorth', positivity='positive')
    assert res_nnorth.shape == (10, 3)
    assert torch.all(res_nnorth >= -1e-6)
    res_ortho = simlr_sparseness(v, constraint_type='ortho', constraint_weight=0.5)
    assert res_ortho.shape == (10, 3)
    res_basic = orthogonalize_and_q_sparsify(v, sparseness_alg='basic', sparseness_quantile=0.5)
    assert res_basic.shape == (10, 3)
    res_orthorank = orthogonalize_and_q_sparsify(v, sparseness_alg='orthorank', sparseness_quantile=0.5)
    assert res_orthorank.shape == (10, 3)
    
    # Indicator coverage
    m = torch.randn(10, 5)
    m_opt = optimize_indicator_matrix(m)
    assert m_opt.shape == (10, 5)
    
    # Rank based extra coverage
    rb_res = rank_based_matrix_segmentation(v, 0.5, basic=True, positivity='negative')
    assert rb_res.shape == (10, 3)

def test_simlr_energy_types():
    x1 = torch.randn(50, 10)
    x2 = torch.randn(50, 8)
    energies = ["regression", "acc", "logcosh", "exp", "nc", "gauss", "kurtosis", "dat"]
    for e in energies:
        domain_mats = [torch.randn(2, 10), torch.randn(2, 8)] if e == "dat" else None
        domain_lambdas = 0.1 if e == "dat" else None
        res = simlr([x1, x2], k=2, iterations=2, energy_type=e, 
                    domain_matrices=domain_mats, domain_lambdas=domain_lambdas)
        assert 'u' in res

def test_simlr_perm_and_decomp():
    x1 = torch.randn(30, 10)
    x2 = torch.randn(30, 8)
    res = simlr([x1, x2], k=2, iterations=2)
    decomp = decompose_energy([x1, x2], res)
    assert 'modality_energies' in decomp
    perm_res = simlr_perm([x1, x2], k=2, n_perms=2, iterations=2)
    assert 'stats' in perm_res

def test_deep_mixing_algs():
    x1 = torch.randn(50, 10)
    x2 = torch.randn(50, 8)
    mix_algs = ["avg", "newton", "pca", "svd", "stochastic"]
    for mix in mix_algs:
        res = ned_simr([x1, x2], k=2, epochs=2, mixing_algorithm=mix, warmup_epochs=0)
        assert 'u' in res

def test_svd_edge_cases():
    # Test ba_svd with NaNs
    x = torch.randn(20, 10)
    x[0, 0] = float('nan')
    u, s, v = ba_svd(x, nu=5, nv=5)
    assert not torch.any(torch.isnan(u))
    
    # Test whiten
    w_res = whiten_matrix(x)
    assert w_res['whitened_matrix'].shape == (20, 10)
    
    # Test multiscale
    r = torch.tensor([0.5, 1.0])
    ms_res = multiscale_svd(x, r, locn=2, nev=2, knn=5)
    assert ms_res['evals_vs_scale'].shape == (2, 2)

def test_utils_extra():
    from pysimlr.utils import rvcoef, adjusted_rvcoef
    # Test trace path (N < P+Q)
    x1 = torch.randn(10, 20)
    y1 = torch.randn(10, 20)
    r1 = rvcoef(x1, y1)
    
    # Test gram path (N >= P+Q)
    x2 = torch.randn(50, 5)
    y2 = torch.randn(50, 5)
    r2 = rvcoef(x2, y2)
    
    assert isinstance(r1, float)
    assert isinstance(r2, float)
    
    # Map vars coverage
    df = pd.DataFrame({
        'left_vol': [1, 2],
        'right_vol': [1.1, 1.9]
    })
    from pysimlr.utils import map_asym_var, map_lr_average_var
    df_asym = map_asym_var(df, ['left_vol'])
    assert 'Asym_vol' in df_asym.columns
    df_avg = map_lr_average_var(df, ['left_vol'])
    assert 'LRAVG_vol' in df_avg.columns
