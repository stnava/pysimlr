"""D37: the ADNI sex covariate is built from PTGENDER in either coding and never silently constant."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]

from adni_covariates import ptgender_indicator

ADNI_CSV = os.path.join(REPO, "data", "adni", "adni_mci_cognitive_merged.csv")


def test_numeric_coding_passes_through():
    out = ptgender_indicator(pd.Series([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(out, [1.0, 0.0, 1.0])


def test_string_coding_maps_male_to_one():
    out = ptgender_indicator(pd.Series(["Male", "female", " M ", "F"]))
    np.testing.assert_array_equal(out, [1.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("col", [
    pd.Series([0.0, 0.0, 0.0]),            # constant numeric
    pd.Series(["Female", "Female"]),       # constant string
    pd.Series([0.0, 1.0, 2.0]),            # not 0/1
    pd.Series(["Male", "Other"]),          # unknown string
    pd.Series([0.0, 1.0, np.nan]),         # missing, not allowed
])
def test_bad_columns_raise(col):
    with pytest.raises(ValueError):
        ptgender_indicator(col)


def test_missing_kept_as_nan_when_allowed():
    out = ptgender_indicator(pd.Series(["Male", None, "Female"]), allow_missing=True)
    assert out[0] == 1.0 and np.isnan(out[1]) and out[2] == 0.0


@pytest.mark.skipif(not os.path.exists(ADNI_CSV), reason="ADNI-MCI export not present")
def test_protocol_b_sex_column_is_ptgender():
    import paper_adni_ablations as A

    df = pd.read_csv(ADNI_CSV)
    d = A.protocol_b(42)
    n = len(df)
    perm = np.random.default_rng(42).permutation(n)
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
    ptg = df["PTGENDER"].to_numpy(float)
    for cov, idx in ((d["cov_tr"], tr), (d["cov_te"], te)):
        assert np.unique(cov[:, 2]).size == 2
        np.testing.assert_array_equal(cov[:, 2], ptg[idx])
