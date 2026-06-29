import os
import shutil
import pytest
import numpy as np
import pandas as pd
import torch
from pysimlr import write_simlr, read_simlr

def test_pysimlr_write_read_json_only(tmp_path):
    # Setup test data
    np.random.seed(42)
    
    simlr_object = {
        "u": np.random.randn(10, 3),
        "v": {
            "t1": pd.DataFrame(np.random.randn(5, 3), index=[f"t1_f{i}" for i in range(5)], columns=["PC1", "PC2", "PC3"]),
            "dt": pd.DataFrame(np.random.randn(4, 3), index=[f"dt_f{i}" for i in range(4)], columns=["PC1", "PC2", "PC3"])
        },
        "modality_names": ["t1", "dt"],
        "feature_names": {
            "t1": [f"t1_f{i}" for i in range(5)],
            "dt": [f"dt_f{i}" for i in range(4)]
        },
        "energyPath": pd.DataFrame(np.random.randn(10, 2), columns=["recon", "sim"]),
        "null_val": None,
        "list_val": [np.random.randn(2, 2), np.random.randn(3, 3)]
    }
    
    prefix = os.path.join(tmp_path, "test_run")
    
    # Save Python-only (no R conversion)
    write_simlr(simlr_object, file_prefix=prefix, clear_dir=True, use_r=False)
    
    outdir = f"{prefix}_simlr"
    assert os.path.exists(outdir)
    assert os.path.exists(os.path.join(outdir, "manifest.json"))
    assert not os.path.exists(os.path.join(outdir, "manifest.rds"))
    
    # Read back
    restored = read_simlr(outdir, use_r=False)
    
    # Assert correctness
    np.testing.assert_array_almost_equal(restored["u"], simlr_object["u"])
    pd.testing.assert_frame_equal(restored["v"]["t1"], simlr_object["v"]["t1"])
    pd.testing.assert_frame_equal(restored["v"]["dt"], simlr_object["v"]["dt"])
    assert restored["modality_names"] == simlr_object["modality_names"]
    assert restored["feature_names"] == simlr_object["feature_names"]
    pd.testing.assert_frame_equal(restored["energyPath"], simlr_object["energyPath"])
    assert restored["null_val"] is None
    np.testing.assert_array_almost_equal(restored["list_val"][0], simlr_object["list_val"][0])
    np.testing.assert_array_almost_equal(restored["list_val"][1], simlr_object["list_val"][1])

def test_pysimlr_write_read_r_compat(tmp_path):
    # Setup test data
    np.random.seed(42)
    
    simlr_object = {
        "u": np.random.randn(5, 2),
        "v": [
            np.random.randn(4, 2),
            np.random.randn(3, 2)
        ],
        "modality_names": ["m1", "m2"],
        "energyPath": np.random.randn(10, 1)
    }
    
    prefix = os.path.join(tmp_path, "test_r_run")
    
    # Save with R RDS conversion
    write_simlr(simlr_object, file_prefix=prefix, clear_dir=True, use_r=True)
    
    outdir = f"{prefix}_simlr"
    assert os.path.exists(outdir)
    assert os.path.exists(os.path.join(outdir, "manifest.rds"))
    
    # Read back using R deserialization path
    restored = read_simlr(outdir, use_r=True)
    
    # Assert correctness
    np.testing.assert_array_almost_equal(restored["u"], simlr_object["u"])
    np.testing.assert_array_almost_equal(restored["v"][0], simlr_object["v"][0])
    np.testing.assert_array_almost_equal(restored["v"][1], simlr_object["v"][1])
    assert restored["modality_names"] == simlr_object["modality_names"]
    np.testing.assert_array_almost_equal(restored["energyPath"], simlr_object["energyPath"])
