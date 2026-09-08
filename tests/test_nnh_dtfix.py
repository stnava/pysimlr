"""
Unit tests for pysimlr.nnh: apply_simlr_matrices_dtfix and residualization.
Addresses Feature 15 in PROJECT.md (Milestone 3).
"""
import pytest
import pandas as pd
import numpy as np
import torch
from pysimlr.nnh import (
    apply_simlr_matrices,
    apply_simlr_matrices_dtfix,
    extend_simlr_embedding_with_new_modalities,
    nnh_update_residuals,
    _preprocess_matrix,
    _build_feature_correlation_adjacency,
    _compute_recommended_k,
    _combine_joint_k,
    _detect_prefixes,
)


def test_apply_simlr_matrices_dtfix_standard_shortened_mismatch():
    """Verify dtfix successfully matches shortened projection matrix row names against full ANTsPyMM DTI names."""
    df = pd.DataFrame({
        "DTI_mean_fa_Left": [0.5, 0.6, 0.7],
        "DTI_mean_fa_Right": [0.52, 0.61, 0.73],
        "T1Hier_Left-Caudate_vol": [12.0, 14.0, 16.0],
        "age": [60, 65, 70]
    })

    # Projection matrix has shortened row names from _shorten_pymm_names: 'dti.fa.left', 'dti.fa.right'
    W_dt = pd.DataFrame(
        [[1.0, 0.2], [0.5, 0.8]],
        index=["dti.fa.left", "dti.fa.right"],
        columns=["PC1", "PC2"]
    )
    matrices = {"dt": W_dt}

    # 1. Verify standard apply_simlr_matrices fails to find overlap and yields 0 added columns
    df_std, added_std = apply_simlr_matrices(df, matrices)
    assert len(added_std) == 0

    # 2. Verify apply_simlr_matrices_dtfix detects correspondence and adds projections
    df_fix, added_fix = apply_simlr_matrices_dtfix(df, matrices, verbose=False)
    assert set(added_fix) == {"dt_PC1", "dt_PC2"}
    assert "dt_PC1" in df_fix.columns
    assert "dt_PC2" in df_fix.columns

    # 3. Verify original column names are preserved/restored (not left as shortened names)
    assert "DTI_mean_fa_Left" in df_fix.columns
    assert "DTI_mean_fa_Right" in df_fix.columns
    assert "dti.fa.left" not in df_fix.columns

    # 4. Verify authentic numerical computation: Y = X @ W
    X = df[["DTI_mean_fa_Left", "DTI_mean_fa_Right"]].values
    expected_PC1 = X @ W_dt["PC1"].values
    expected_PC2 = X @ W_dt["PC2"].values
    np.testing.assert_allclose(df_fix["dt_PC1"].values, expected_PC1, rtol=1e-5)
    np.testing.assert_allclose(df_fix["dt_PC2"].values, expected_PC2, rtol=1e-5)


def test_apply_simlr_matrices_dtfix_dta_asymmetry_block():
    """Verify dtfix detects and handles dta asymmetry block shortening correspondence."""
    df = pd.DataFrame({
        "DTI_Asym_fa_Left": [0.02, 0.05, 0.01],
        "age": [50, 55, 60]
    })

    W_dta = pd.DataFrame(
        [[1.5]],
        index=["dti.asym.fa.left"],
        columns=["PC1"]
    )
    matrices = {"dta": W_dta}

    df_fix, added_fix = apply_simlr_matrices_dtfix(df, matrices)
    assert "dta_PC1" in added_fix
    assert "DTI_Asym_fa_Left" in df_fix.columns
    np.testing.assert_allclose(df_fix["dta_PC1"].values, df["DTI_Asym_fa_Left"].values * 1.5, rtol=1e-5)


def test_apply_simlr_matrices_dtfix_already_matching_names():
    """Verify dtfix operates normally without unnecessary renaming when names already match."""
    df = pd.DataFrame({
        "DTI_mean_fa_Left": [0.5, 0.6],
        "DTI_mean_fa_Right": [0.52, 0.61]
    })

    W_dt = pd.DataFrame(
        [[1.0], [2.0]],
        index=["DTI_mean_fa_Left", "DTI_mean_fa_Right"],
        columns=["PC1"]
    )
    matrices = {"dt": W_dt}

    df_fix, added_fix = apply_simlr_matrices_dtfix(df, matrices)
    assert added_fix == ["dt_PC1"]
    expected = df["DTI_mean_fa_Left"] * 1.0 + df["DTI_mean_fa_Right"] * 2.0
    np.testing.assert_allclose(df_fix["dt_PC1"].values, expected.values, rtol=1e-5)


def test_apply_simlr_matrices_dtfix_non_dataframe_and_n_limit():
    """Verify dtfix handles list of tensor matrices with feature_names and respects n_limit."""
    df = pd.DataFrame({
        "DTI_mean_fa_Left": [0.5, 0.6, 0.7],
        "DTI_mean_fa_Right": [0.52, 0.61, 0.73]
    })

    W = torch.tensor([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1]]).float()
    matrices_list = [W]
    feature_names = [["dti.fa.left", "dti.fa.right"]]
    modality_names = ["dt"]

    df_fix, added_fix = apply_simlr_matrices_dtfix(
        df,
        matrices_list,
        feature_names=feature_names,
        modality_names=modality_names,
        n_limit=1
    )

    # Only 1 PC should be added due to n_limit=1
    assert added_fix == ["dt_PC1"]
    assert "dt_PC1" in df_fix.columns
    assert "dt_PC2" not in df_fix.columns


def test_apply_simlr_matrices_name_collision_versioning():
    """Verify apply_simlr_matrices tags colliding column names with version prefix."""
    df = pd.DataFrame({
        "t1_f1": [1.0, 2.0],
        "t1_PC1": [99.0, 99.0]  # Already existing column
    })

    W = pd.DataFrame([[2.0]], index=["t1_f1"], columns=["PC1"])
    df_res, added = apply_simlr_matrices(df, {"t1": W})

    assert len(added) == 1
    assert added[0] == "t1_PCr00.1"
    assert "t1_PCr00.1" in df_res.columns
    # Original column untouched
    assert df_res["t1_PC1"].iloc[0] == 99.0


def test_nnh_update_residuals_mean_covariate():
    """Verify nnh_update_residuals correctly residualizes features against row-wise mean signal."""
    mat = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [2.0, 4.0, 6.0]
    ]).float()
    dummy_df = pd.DataFrame()

    res = nnh_update_residuals(mat, dummy_df, covariate_cols="mean")
    assert res.shape == (3, 3)

    # Column 1 (index 1) equals the row means exactly: [2.0, 5.0, 4.0] -> residual must be 0.0
    np.testing.assert_allclose(res[:, 1].numpy(), np.zeros(3), atol=1e-5)

    # Verify that each residual column is orthogonal to the regressor (row-wise mean)
    row_means = torch.mean(mat, dim=1).numpy()
    for col in range(3):
        col_res = res[:, col].numpy()
        dot_product = np.dot(row_means - np.mean(row_means), col_res)
        np.testing.assert_allclose(dot_product, 0.0, atol=1e-5)


def test_nnh_update_residuals_multi_column_orthogonality():
    """Verify nnh_update_residuals makes multivariate feature columns strictly orthogonal to covariates."""
    np.random.seed(42)
    n = 50
    age = np.random.randn(n)
    sex = 0.5 * age + np.random.randn(n)
    cov_df = pd.DataFrame({"age": age, "sex": sex})

    # 3 features correlated with age and sex
    f1 = 1.5 * age + 0.5 * sex + np.random.randn(n)
    f2 = -2.0 * age + 1.0 * sex + np.random.randn(n)
    f3 = 0.8 * sex + np.random.randn(n)
    mat = torch.tensor(np.column_stack([f1, f2, f3])).float()

    res = nnh_update_residuals(mat, cov_df, ["age", "sex"])
    assert res.shape == (n, 3)

    # Invariant: Residuals must be orthogonal to span(age, sex, 1)
    X = np.column_stack([np.ones(n), age, sex])
    for j in range(3):
        residuals_col = res[:, j].numpy()
        proj = X.T @ residuals_col
        np.testing.assert_allclose(proj, np.zeros(3), atol=1e-4)


def test_nnh_preprocess_matrix_policies():
    """Verify _preprocess_matrix drops or zeroes invalid columns and standardizes finite values."""
    # Matrix: Col 0 valid, Col 1 all-NaN, Col 2 constant 0 variance
    M = np.array([
        [1.0, np.nan, 5.0],
        [2.0, np.nan, 5.0],
        [3.0, np.nan, 5.0],
        [4.0, np.nan, 5.0]
    ])

    # 1. Drop policy
    res_drop = _preprocess_matrix(M, "test_block", preprocess_policy="drop")
    assert res_drop["matrix"].shape == (4, 1)
    assert set(res_drop["dropped"]) == {1, 2}
    np.testing.assert_allclose(np.mean(res_drop["matrix"], axis=0), 0.0, atol=1e-5)

    # 2. Zero policy
    res_zero = _preprocess_matrix(M, "test_block", preprocess_policy="zero")
    assert res_zero["matrix"].shape == (4, 3)
    assert set(res_zero["zeroed"]) == {1, 2}
    np.testing.assert_allclose(res_zero["matrix"][:, 1:], np.zeros((4, 2)), atol=1e-5)

    # 3. Error on completely empty block
    M_invalid = np.full((5, 2), np.nan)
    with pytest.raises(ValueError, match="no usable columns"):
        _preprocess_matrix(M_invalid, "empty_block", preprocess_policy="drop")


def test_nnh_auxiliary_helpers():
    """Verify _build_feature_correlation_adjacency, _compute_recommended_k, _combine_joint_k, _detect_prefixes."""
    np.random.seed(42)
    M = np.random.randn(30, 4)

    # Adjacency thresholding
    A = _build_feature_correlation_adjacency(M, threshold=0.5, use_abs=True)
    assert A.shape == (4, 4)
    np.testing.assert_allclose(np.diag(A), np.ones(4))
    assert np.all(A >= 0.0)

    # Recommended k
    k_cum = _compute_recommended_k(M, method="cumulative", cumulative_thresh=0.8, min_k=2)
    k_elb = _compute_recommended_k(M, method="elbow", min_k=2)
    assert 2 <= k_cum <= 4
    assert 2 <= k_elb <= 4

    # Combine joint k
    k_dict = {"t1": 2, "dt": 4, "rsf": 3}
    assert _combine_joint_k(k_dict, policy="max") == 4
    assert _combine_joint_k(k_dict, policy="median") == 3
    assert _combine_joint_k(k_dict, policy="min") == 2

    # Detect prefixes
    simnames = ["t1_PC1", "t1_PC2", "dt_PC1", "rsf_C1"]
    prefixes = _detect_prefixes(simnames)
    assert set(prefixes) == {"t1_", "dt_", "rsf_"}


def test_extend_simlr_embedding_with_flow_simlr_v_iterations():
    """Verify extend_simlr_embedding_with_new_modalities remaps iterations -> epochs for legacy method='flow_simlr_v'."""
    np.random.seed(42)
    torch.manual_seed(42)
    n = 25
    pymm = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "y1": np.random.randn(n),
        "y2": np.random.randn(n)
    })
    simlr_result = {
        "v": [pd.DataFrame(np.random.randn(2, 2), index=["x1", "x2"], columns=["mod1_1", "mod1_2"])],
        "feature_names": [["x1", "x2"]],
        "modality_names": ["mod1"]
    }
    new_modalities = {"mod2": ["y1", "y2"]}

    res = extend_simlr_embedding_with_new_modalities(
        pymm,
        simlr_result,
        new_modalities,
        method="flow_simlr_v",
        iterations=5,
        verbose=False
    )

    assert isinstance(res, dict)
    assert "updated_simlr_result" in res
    assert "simlr_fit" in res
    assert "blocks" in res
    assert "mod2" in res["simlr_fit"]["v_dict"]
    assert res["simlr_fit"]["v_dict"]["mod2"].shape[0] == 2
    # Confirms iterations=5 was genuinely remapped to epochs=5 and trained for 5 epochs
    assert len(res["simlr_fit"]["loss_history"]) == 5

