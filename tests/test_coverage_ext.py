import torch
import numpy as np
import pandas as pd
import pytest
import os
import matplotlib
matplotlib.use('Agg')
from pysimlr import (
    simlr,
    lend_simr,
    ned_simr,
    ned_simr_shared_private,
    deep_simr,
    create_optimizer,
    simlr_sparseness,
    orthogonalize_and_q_sparsify,
    simlr_perm,
    decompose_energy,
    ba_svd,
    safe_pca,
    whiten_matrix,
    multiscale_svd,
    adjusted_rvcoef,
    optimize_indicator_matrix,
    rank_based_matrix_segmentation,
    nnh_embed
)

def test_optimizer_coverage():
    v_mats = [torch.randn(5, 2)]
    opts = ["adam", "nadam", "rmsprop", "gd", "armijo_gradient", 
            "bidirectional_armijo_gradient", "lookahead", 
            "bidirectional_lookahead", "hybrid_adam",
            "torch_adamw", "torch_adagrad"]
    for opt_type in opts:
        optimizer = create_optimizer(opt_type, v_mats, learning_rate=0.01, momentum=0.9, beta1=0.9, beta2=0.999)
        v_next = optimizer.step(0, v_mats[0], torch.randn(5, 2))
        assert v_next.shape == (5, 2)

def test_sparsification_coverage():
    from pysimlr.sparsification import (optimize_indicator_matrix, 
                                        rank_based_matrix_segmentation,
                                        project_to_partially_orthonormal_nonnegative)
    v = torch.randn(10, 3)
    assert simlr_sparseness(v, sparseness_alg='soft', sparseness_quantile=0.5).shape == (10, 3)
    assert torch.all(simlr_sparseness(v, sparseness_alg='nnorth', positivity='positive') >= -1e-6)
    assert simlr_sparseness(v, constraint_type='ortho', constraint_weight=0.5).shape == (10, 3)
    assert rank_based_matrix_segmentation(v, 0.5, basic=True, positivity='either', transpose=False).shape == (10, 3)
    assert rank_based_matrix_segmentation(v, 0.5, basic=True, positivity='positive', transpose=True).shape == (10, 3)
    assert optimize_indicator_matrix(torch.randn(10, 5), preprocess=True).shape == (10, 5)
    v_neg = project_to_partially_orthonormal_nonnegative(torch.randn(5, 2), constraint='negative')
    assert torch.all(v_neg <= 1e-6)
    assert rank_based_matrix_segmentation(torch.zeros(5, 2), 0.5, basic=True).shape == (5, 2)

def test_simlr_energy_types():
    x1, x2 = torch.randn(50, 10), torch.randn(50, 8)
    for e in ["regression", "acc", "logcosh", "exp", "nc", "gauss", "kurtosis", "dat"]:
        dm = [torch.randn(2, 10), torch.randn(2, 8)] if e == "dat" else None
        assert 'u' in simlr([x1, x2], k=2, iterations=2, energy_type=e, domain_matrices=dm, domain_lambdas=0.1, verbose=True)

def test_simlr_perm_and_decomp():
    x1, x2 = torch.randn(30, 10), torch.randn(30, 8)
    res = simlr([x1, x2], k=2, iterations=2)
    assert 'modality_energies' in decompose_energy([x1, x2], res)
    assert 'stats' in simlr_perm([x1, x2], k=2, n_perms=2, iterations=2)

def test_nnh_extra():
    df = pd.DataFrame({'T1_1': np.random.randn(20), 'T1_2': np.random.randn(20), 'DTI_1': np.random.randn(20), 'age': np.random.randn(20)})
    assert 'u' in nnh_embed(df, nsimlr=3, iterations=2, covariates=['age'])

def test_deep_extra():
    x1, x2 = torch.randn(50, 10), torch.randn(50, 8)
    # Test different energy types in deep models
    for e in ["acc", "logcosh"]:
        assert 'u' in deep_simr([x1, x2], k=2, epochs=2, energy_type=e, warmup_epochs=0)
    # Test ICA mixing in deep models
    assert 'u' in lend_simr([x1, x2], k=2, epochs=2, mixing_algorithm="ica", warmup_epochs=0)

def test_deep_mixing_algs():
    x1, x2 = torch.randn(50, 10), torch.randn(50, 8)
    for mix in ["avg", "newton", "pca", "svd", "stochastic"]:
        assert 'u' in ned_simr([x1, x2], k=2, epochs=2, mixing_algorithm=mix, warmup_epochs=0)

def test_svd_extra():
    x = torch.randn(20, 10)
    assert ba_svd(x, nu=0, nv=0)[0].shape[0] == 20
    assert safe_pca(x, nc=1)['u'].shape == (20, 1)
    assert 'evals_vs_scale' in multiscale_svd(x, torch.tensor([0.5, 1.0]), locn=[1, 5, 10], nev=2)
    assert multiscale_svd(x, torch.tensor([0.5]), locn=2, nev=2, knn=5)['evals_vs_scale'].shape == (1, 2)
    x_flat = torch.ones(10, 5)
    assert torch.all(safe_pca(x_flat, nc=2)['u'] == 0)

def test_visualization_coverage():
    from pysimlr.visualization import (plot_energy, plot_latent_2d, plot_v_matrix, 
                                       plot_lend_simr_architecture, plot_ned_simr_architecture,
                                       plot_ned_shared_private_architecture, generate_all_architecture_graphs)
    assert plot_energy([1.0, 0.5]) is not None
    assert plot_latent_2d(torch.randn(10, 2)) is not None
    assert plot_v_matrix(torch.randn(5, 2)) is not None
    assert plot_lend_simr_architecture() is not None
    assert plot_ned_simr_architecture() is not None
    assert plot_ned_shared_private_architecture() is not None
    test_dir = "test_viz_output"
    generate_all_architecture_graphs(test_dir)
    assert os.path.exists(os.path.join(test_dir, "detailed_lend_simr.pdf"))
    import shutil
    shutil.rmtree(test_dir)

def test_utils_extra():
    from pysimlr.utils import rvcoef, adjusted_rvcoef, map_asym_var, map_lr_average_var, l1_normalize_features
    assert isinstance(rvcoef(torch.randn(10, 20), torch.randn(10, 20)), float)
    df = pd.DataFrame({'left_vol': [1, 2], 'right_vol': [1.1, 1.9]})
    assert 'Asym_vol' in map_asym_var(df, ['left_vol']).columns
    assert 'LRAVG_vol' in map_lr_average_var(df, ['left_vol']).columns
    v = torch.randn(10, 2)
    v_norm = l1_normalize_features(v)
    assert torch.allclose(torch.sum(torch.abs(v_norm), dim=0), torch.ones(2))
