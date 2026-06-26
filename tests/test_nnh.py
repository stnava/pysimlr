import pandas as pd
import numpy as np
import torch
import pytest
from pysimlr.nnh import (
    antspymm_predictors,
    _get_names_from_dataframe,
    mapAsymVar,
    mapLRAverageVar,
    nnh_embed,
    nnh_update_residuals
)

def test_get_names_from_dataframe():
    df = pd.DataFrame(columns=[
        "T1Hier_Left-Caudate_vol", "T1Hier_Right-Caudate_vol",
        "T1Hier_Left-Caudate_tissues", "T1w_SNR", "DTI_evr",
        "bn_str_ca_deep_something", "bn_str_ca_surface_something"
    ])
    
    # 1. Single search term
    names = _get_names_from_dataframe("T1Hier_", df)
    assert set(names) == {"T1Hier_Left-Caudate_vol", "T1Hier_Right-Caudate_vol", "T1Hier_Left-Caudate_tissues"}
    
    # 2. Multiple search terms (AND logic)
    names = _get_names_from_dataframe(["T1Hier_", "vol"], df)
    assert set(names) == {"T1Hier_Left-Caudate_vol", "T1Hier_Right-Caudate_vol"}
    
    # 3. Exclusions
    names = _get_names_from_dataframe("T1Hier_", df, exclusions="tissues")
    assert set(names) == {"T1Hier_Left-Caudate_vol", "T1Hier_Right-Caudate_vol"}
    
    # 4. Multiple exclusions
    names = _get_names_from_dataframe("T1Hier_", df, exclusions=["tissues", "Right"])
    assert set(names) == {"T1Hier_Left-Caudate_vol"}

def test_map_asym_var_and_lr_average_var():
    data = {
        "T1Hier_left-Caudate_vol": [10.0, 20.0],
        "T1Hier_right-Caudate_vol": [12.0, 18.0]
    }
    df = pd.DataFrame(data)
    
    # test mapAsymVar
    df_asym = mapAsymVar(df, ["T1Hier_left-Caudate_vol"], leftname="left", rightname="right", replacer="Asym")
    assert "T1Hier_Asym-Caudate_vol" in df_asym.columns
    # |10 - 12| = 2; |20 - 18| = 2
    np.testing.assert_array_almost_equal(df_asym["T1Hier_Asym-Caudate_vol"].values, [2.0, 2.0])
    
    # test mapLRAverageVar
    df_avg = mapLRAverageVar(df, ["T1Hier_left-Caudate_vol"], leftname="left", rightname="right", replacer="LRAVG")
    assert "T1Hier_LRAVG-Caudate_vol" in df_avg.columns
    # (10 + 12)/2 = 11; (20 + 18)/2 = 19
    np.testing.assert_array_almost_equal(df_avg["T1Hier_LRAVG-Caudate_vol"].values, [11.0, 19.0])

def test_antspymm_predictors_return_colnames():
    cols = [
        "T1w_some_feature", "mtl_feature", "cerebellum_feature",
        "T1Hier_left-Caudate_vol", "T1Hier_right-Caudate_vol",
        "DTI_some_feature", "rsfMRI_fcnxpro_feature", "perf_some_feature",
        "SNR", "hier_id", "unrelated_column"
    ]
    df = pd.DataFrame(columns=cols)
    
    # By default, return_colnames should be True
    selected = antspymm_predictors(df)
    assert isinstance(selected, list)
    assert "T1w_some_feature" in selected
    assert "mtl_feature" in selected
    assert "cerebellum_feature" in selected
    assert "T1Hier_left-Caudate_vol" in selected
    assert "T1Hier_right-Caudate_vol" in selected
    assert "DTI_some_feature" in selected
    assert "rsfMRI_fcnxpro_feature" in selected
    assert "perf_some_feature" in selected
    # exclusions
    assert "SNR" not in selected
    assert "hier_id" not in selected
    assert "unrelated_column" not in selected

def test_antspymm_predictors_modify_df():
    cols = [
        "T1Hier_Left-Caudate_vol", "T1Hier_Right-Caudate_vol",
        "DTI_left_md", "DTI_right_md", "unrelated_column"
    ]
    data = {
        "T1Hier_Left-Caudate_vol": [10.0, 20.0],
        "T1Hier_Right-Caudate_vol": [12.0, 18.0],
        "DTI_left_md": [0.5, 0.6],
        "DTI_right_md": [0.55, 0.58],
        "unrelated_column": [1, 2]
    }
    df = pd.DataFrame(data)
    
    # return_colnames=False, doasym=True
    df_mod = antspymm_predictors(df, doasym=True, return_colnames=False)
    assert isinstance(df_mod, pd.DataFrame)
    
    # 1. Capitalization should be standardized (Left -> left, Right -> right)
    assert "T1Hier_left-Caudate_vol" in df_mod.columns
    assert "T1Hier_right-Caudate_vol" in df_mod.columns
    assert "T1Hier_Left-Caudate_vol" not in df_mod.columns
    
    # 2. Asymmetry columns should be present
    assert "T1Hier_Asym-Caudate_vol" in df_mod.columns
    # DTI has left_md (contains 'left' inside it)
    assert "DTI_Asym_md" in df_mod.columns
    
    # 3. L/R Average columns should be present
    assert "T1Hier_LRAVG-Caudate_vol" in df_mod.columns
    assert "DTI_LRAVG_md" in df_mod.columns
    
    # 4. Values check
    np.testing.assert_array_almost_equal(df_mod["T1Hier_Asym-Caudate_vol"].values, [2.0, 2.0])
    np.testing.assert_array_almost_equal(df_mod["T1Hier_LRAVG-Caudate_vol"].values, [11.0, 19.0])

def test_nnh_embed_flow():
    # Construct a small dataframe
    df = pd.DataFrame({
        'T1Hier_vol1': np.random.randn(20),
        'T1Hier_vol2': np.random.randn(20),
        'DTI_fa1': np.random.randn(20),
        'age': np.random.randn(20)
    })
    
    # 1. Test use_flow=False (defaults to simlr)
    res_simlr = nnh_embed(df, nsimlr=2, iterations=2, covariates=['age'], use_flow=False)
    assert 'u' in res_simlr
    assert 'v' in res_simlr
    assert 'w' in res_simlr
    
    # 2. Test use_flow=True (runs flow_simr_v)
    res_flow = nnh_embed(df, nsimlr=2, iterations=2, covariates=['age'], use_flow=True)
    assert 'u' in res_flow
    assert 'v' in res_flow
    assert 'w' in res_flow
    assert 'model' in res_flow
    assert res_flow['model_type'] == 'flow_simr_v'

def test_joint_residualization():
    # Create data with correlated covariates
    np.random.seed(42)
    n_samples = 50
    # Covariates age and sex are correlated
    age = np.random.randn(n_samples)
    sex = 0.5 * age + np.random.randn(n_samples)
    
    # Feature y depends on both
    y = 2.0 * age + 3.0 * sex + np.random.randn(n_samples)
    
    df = pd.DataFrame({'age': age, 'sex': sex})
    mat = torch.tensor(y[:, None]).float()
    
    # 1. Joint residualization
    res_joint = nnh_update_residuals(mat, df, ['age', 'sex'])
    
    # 2. Manual joint residuals calculation
    from sklearn.linear_model import LinearRegression
    X = df[['age', 'sex']].values
    reg = LinearRegression().fit(X, y)
    expected_residuals = y - reg.predict(X)
    
    np.testing.assert_array_almost_equal(res_joint.numpy()[:, 0], expected_residuals, decimal=5)
    
    # 3. Verify formula string notation
    res_formula = nnh_update_residuals(mat, df, 'age + sex')
    np.testing.assert_array_almost_equal(res_formula.numpy(), res_joint.numpy(), decimal=5)
