import pandas as pd
import numpy as np
import torch
import pytest
from pysimlr.nnh import (
    nnh_embed,
    _shorten_pymm_names,
    apply_simlr_matrices,
    apply_simlr_matrices_dtfix,
    extend_simlr_embedding_with_new_modalities
)

def test_shorten_pymm_names():
    names = [
        "T1Hier_Left-Caudate_vol",
        "DTI_mean_fa_Left",
        "perf_cbf_mean_Right",
        "rsfmri_fcnxpro129"
    ]
    shortened = _shorten_pymm_names(names)
    assert shortened[0] == "t1.left-caudate.vol"
    assert shortened[1] == "dti.fa.left"
    assert shortened[2] == "cbf.right"
    assert shortened[3] == "rsf.p2"

def test_apply_simlr_matrices():
    # Setup dataframe
    np.random.seed(42)
    df = pd.DataFrame({
        "t1_f1": np.random.randn(10),
        "t1_f2": np.random.randn(10),
        "dt_f1": np.random.randn(10)
    })
    
    # Projection matrices
    W_t1 = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["t1_f1", "t1_f2"],
        columns=["PC1", "PC2"]
    )
    W_dt = pd.DataFrame(
        [[2.0]],
        index=["dt_f1"],
        columns=["PC1"]
    )
    
    simlr_v = {
        "t1": W_t1,
        "dt": W_dt
    }
    
    extended, added_names = apply_simlr_matrices(df, simlr_v)
    assert "t1PC1" in extended.columns
    assert "t1PC2" in extended.columns
    assert "dtPC1" in extended.columns
    assert len(added_names) == 3
    
    np.testing.assert_array_almost_equal(extended["t1PC1"].values, df["t1_f1"].values)
    np.testing.assert_array_almost_equal(extended["dtPC1"].values, df["dt_f1"].values * 2.0)

def test_extend_simlr_embedding():
    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Create a synthetic dataset
    df = pd.DataFrame({
        "t1_f1": np.random.randn(20),
        "t1_f2": np.random.randn(20),
        "dt_f1": np.random.randn(20),
        "dt_f2": np.random.randn(20),
        "pet_f1": np.random.randn(20),
        "pet_f2": np.random.randn(20),
        "age": np.random.randn(20)
    })
    
    # Rename for DTI naming tests
    df_pymm = df.rename(columns={
        "t1_f1": "T1Hier_Left-Caudate_vol",
        "t1_f2": "T1Hier_Right-Caudate_vol",
        "dt_f1": "DTI_mean_fa_Left",
        "dt_f2": "DTI_mean_fa_Right",
        "pet_f1": "PET_suvr_Left",
        "pet_f2": "PET_suvr_Right"
    })
    
    # 2. Run initial nnh_embed
    t1_cols = ["T1Hier_Left-Caudate_vol", "T1Hier_Right-Caudate_vol"]
    dti_cols = ["DTI_mean_fa_Left", "DTI_mean_fa_Right"]
    
    mats = {
        "t1": df_pymm[t1_cols].values,
        "dt": df_pymm[dti_cols].values
    }
    
    # Run a simple SIMLR
    from pysimlr.simlr import simlr
    initial_fit = simlr(
        [torch.tensor(mats["t1"]).float(), torch.tensor(mats["dt"]).float()],
        k=2,
        iterations=5,
        verbose=False
    )
    
    # Wrap in dict like nnh_embed output
    simlr_result = {
        "u": initial_fit["u"],
        "v": {
            "t1": pd.DataFrame(initial_fit["v"][0].numpy(), index=t1_cols, columns=["PC1", "PC2"]),
            "dt": pd.DataFrame(initial_fit["v"][1].numpy(), index=dti_cols, columns=["PC1", "PC2"])
        },
        "modality_names": ["t1", "dt"],
        "feature_names": {
            "t1": t1_cols,
            "dt": dti_cols
        }
    }
    
    # New modalities
    new_modalities = {
        "pet": ["PET_suvr_Left", "PET_suvr_Right"]
    }
    
    # 3. Test extend_simlr_embedding_with_new_modalities using 'simlr'
    ext_res = extend_simlr_embedding_with_new_modalities(
        pymm=df_pymm,
        simlr_result=simlr_result,
        new_modalities=new_modalities,
        mode="concatenate",
        k_new=2,
        min_k=2,
        method="simlr",
        iterations=5,
        verbose=False
    )
    
    assert "updated_simlr_result" in ext_res
    assert "simlr_fit" in ext_res
    assert "blocks" in ext_res
    assert "pet" in ext_res["blocks"]
    assert "sim" in ext_res["blocks"]
    
    updated_result = ext_res["updated_simlr_result"]
    assert "pet" in updated_result["v"]
    assert isinstance(updated_result["v"]["pet"], pd.DataFrame)
    assert updated_result["v"]["pet"].shape == (2, 2)
    
    pd.testing.assert_frame_equal(updated_result["v"]["t1"], simlr_result["v"]["t1"])
    pd.testing.assert_frame_equal(updated_result["v"]["dt"], simlr_result["v"]["dt"])
    
    # 4. Test with method='flow_simr_v'
    ext_res_flow = extend_simlr_embedding_with_new_modalities(
        pymm=df_pymm,
        simlr_result=simlr_result,
        new_modalities=new_modalities,
        mode="concatenate",
        k_new=2,
        min_k=2,
        method="flow_simr_v",
        iterations=5,
        verbose=False
    )
    assert ext_res_flow["simlr_fit"]["model_type"] == "flow_simr_v"
    assert "pet" in ext_res_flow["updated_simlr_result"]["v"]
