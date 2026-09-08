"""
Authentic parity unit tests for algorithms ported from R SIMLR and ANTsR.
Addresses Feature 18 in PROJECT.md (Milestone 3).
"""
import torch
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from pysimlr import (
    get_names_from_dataframe,
    map_asym_var,
    map_lr_average_var,
    rvcoef,
    adjusted_rvcoef,
    simlr_sparseness,
    orthogonalize_and_q_sparsify,
    sparse_distance_matrix,
    invariant_orthogonality_defect,
)
from pysimlr.utils import preprocess_data, procrustes_r2


def test_parity_dataframe_column_selection():
    """Verify R-parity for regex dataframe column inclusion and exclusion."""
    df = pd.DataFrame(columns=['brain_thick', 'brain_vol', 'age_years', 'blood_pressure', 'brain_surface'])
    # Inclusion matching
    res_inc = get_names_from_dataframe(['brain'], df)
    assert res_inc == ['brain_surface', 'brain_thick', 'brain_vol']
    # Exclusion matching
    res_exc = get_names_from_dataframe(['brain'], df, exclusions=['thick'])
    assert res_exc == ['brain_surface', 'brain_vol']
    # Multiple patterns
    res_multi = get_names_from_dataframe(['age', 'blood'], df)
    assert res_multi == ['age_years', 'blood_pressure']
    # Empty match
    assert get_names_from_dataframe(['nonexistent'], df) == []


def test_parity_neuroimaging_asymmetry_and_averaging():
    """Verify ANTsR parity for bilateral neuroimaging asymmetry and average calculations."""
    df = pd.DataFrame({
        'left_hippo': [3.0, 3.2, 2.8],
        'right_hippo': [3.1, 3.2, 2.9],
        'left_amygdala': [1.5, 1.4, 1.6],
        'right_amygdala': [1.6, 1.4, 1.5],
        'covar': [10, 20, 30]
    })
    asym_df = map_asym_var(df, ['left_hippo', 'left_amygdala'])
    assert 'Asym_hippo' in asym_df.columns
    assert 'Asym_amygdala' in asym_df.columns
    np.testing.assert_allclose(asym_df['Asym_hippo'].values, [0.1, 0.0, 0.1], atol=1e-5)
    np.testing.assert_allclose(asym_df['Asym_amygdala'].values, [0.1, 0.0, 0.1], atol=1e-5)

    avg_df = map_lr_average_var(df, ['left_hippo', 'left_amygdala'])
    assert 'LRAVG_hippo' in avg_df.columns
    assert 'LRAVG_amygdala' in avg_df.columns
    np.testing.assert_allclose(avg_df['LRAVG_hippo'].values, [3.05, 3.2, 2.85], atol=1e-5)
    np.testing.assert_allclose(avg_df['LRAVG_amygdala'].values, [1.55, 1.4, 1.55], atol=1e-5)


def test_parity_preprocess_data_centering_and_scaling():
    """Verify R scale() parity: zero mean and unit sample variance with provenance support."""
    torch.manual_seed(42)
    mat = torch.randn(60, 5) * 4.0 + 7.0
    scaled, prov = preprocess_data(mat, ['centerAndScale'])
    np.testing.assert_allclose(scaled.mean(dim=0).numpy(), 0.0, atol=1e-5)
    np.testing.assert_allclose(scaled.std(dim=0).numpy(), 1.0, atol=1e-4)

    # Provenance reproducibility
    test_mat = torch.randn(10, 5) * 4.0 + 7.0
    scaled_test = preprocess_data(test_mat, ['centerAndScale'], provenance=prov)
    expected_test = (test_mat - prov['cas_mean']) / prov['cas_std']
    np.testing.assert_allclose(scaled_test.numpy(), expected_test.numpy(), atol=1e-5)


def test_parity_adjusted_rv_coefficient():
    """Verify Robert & Escoufier RV coefficient properties: identity, scale, and rotation invariance."""
    torch.manual_seed(42)
    x = torch.randn(50, 4)
    # Self RV is exactly 1.0
    np.testing.assert_allclose(rvcoef(x, x), 1.0, atol=1e-5)
    np.testing.assert_allclose(adjusted_rvcoef(x, x), 1.0, atol=1e-5)
    # Scale invariance
    np.testing.assert_allclose(rvcoef(2.5 * x, -3.0 * x), 1.0, atol=1e-5)
    # Rotation invariance
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    np.testing.assert_allclose(rvcoef(x, x @ q), 1.0, atol=1e-4)
    np.testing.assert_allclose(adjusted_rvcoef(x, x @ q), 1.0, atol=1e-4)
    # Orthogonal basis has near-zero RV
    u_orth, _ = torch.linalg.qr(torch.randn(50, 8))
    rv_val = rvcoef(u_orth[:, :4], u_orth[:, 4:])
    assert 0.0 <= rv_val < 0.01


def test_parity_simlr_sparseness_constraints():
    """Verify R simlr.sparseness parity: positive/negative clamping, quantile thresholding and unit norm."""
    torch.manual_seed(42)
    v = torch.randn(30, 4)
    # Positive constraint + unit norm
    v_pos = orthogonalize_and_q_sparsify(v, sparseness_quantile=0.5, positivity='positive', unit_norm=True)
    assert (v_pos < -1e-6).sum() == 0, 'Negative values found'
    np.testing.assert_allclose(torch.norm(v_pos, p=2, dim=0).numpy(), 1.0, atol=1e-5)
    assert ((v_pos == 0).float().mean(dim=0) >= 0.45).all()

    # Negative constraint + unit norm
    v_neg = orthogonalize_and_q_sparsify(v, sparseness_quantile=0.5, positivity='negative', unit_norm=True)
    assert (v_neg > 1e-6).sum() == 0, 'Positive values found'
    np.testing.assert_allclose(torch.norm(v_neg, p=2, dim=0).numpy(), 1.0, atol=1e-5)
    assert ((v_neg == 0).float().mean(dim=0) >= 0.45).all()

    # simlr_sparseness entrypoint
    v_sim = simlr_sparseness(v, sparseness_quantile=0.5, positivity='positive')
    assert (v_sim < -1e-6).sum() == 0
    assert ((v_sim == 0).float().mean(dim=0) >= 0.45).all()


def test_parity_procrustes_alignment():
    """Verify Procrustes alignment R2 invariance under orthogonal rotations and consistency with scipy."""
    torch.manual_seed(42)
    u = torch.randn(50, 3)
    np.testing.assert_allclose(procrustes_r2(u, u), 1.0, atol=1e-5)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    u_rot = u @ q
    np.testing.assert_allclose(procrustes_r2(u, u_rot), 1.0, atol=1e-4)
    # Cross-validation with scipy orthogonal_procrustes
    r_scipy, _ = orthogonal_procrustes(u.numpy(), u_rot.numpy())
    aligned_diff = np.linalg.norm(u.numpy() - u_rot.numpy() @ r_scipy.T)
    assert aligned_diff < 1e-4


def test_parity_sparse_distance_matrix():
    """Verify R sparseDistanceMatrix parity: KNN graph distance and Gaussian affinity kernel properties."""
    torch.manual_seed(42)
    x = torch.randn(20, 4)
    k = 4
    # Distance matrix (without sigma)
    s_dist = sparse_distance_matrix(x, k=k)
    assert s_dist.shape == (20, 20)
    np.testing.assert_allclose(torch.diagonal(s_dist).numpy(), 0.0, atol=1e-6)
    assert ((s_dist > 0).sum(dim=1) == k).all()

    # Affinity matrix (with sigma)
    s_aff = sparse_distance_matrix(x, k=k, sigma=1.0)
    assert s_aff.shape == (20, 20)
    np.testing.assert_allclose(torch.diagonal(s_aff).numpy(), 1.0, atol=1e-6)
    assert ((s_aff > 0).sum(dim=1) == k + 1).all()
    assert ((s_aff >= 0.0) & (s_aff <= 1.0)).all()
