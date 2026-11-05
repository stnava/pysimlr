"""Sex covariate from ADNI ``PTGENDER``, in either of the codings the repo's tables use.

The cached ADNI exports (``data/adni/adni_mci_cognitive_merged.csv``,
``adni_all_bl_cognitive_merged.csv``) store ``PTGENDER`` as 0/1, written by
``scripts/build_adni_*_cognitive_cache.py`` as ``PTGENDER == "Male"``. The raw
ADNIMERGE-style tables store the strings "Male"/"Female". The indicator returned
here is 1 = male, 0 = female in both cases, matching the cache builders.

A string test such as ``str.startswith("F")`` on the 0/1 column returns a constant
zero, which then drops out of a covariate regression silently (D37). This helper
raises instead of guessing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MALE = {"MALE", "M"}
_FEMALE = {"FEMALE", "F"}


def ptgender_indicator(col: pd.Series, allow_missing: bool = False) -> np.ndarray:
    """Return ``PTGENDER`` as a float 0/1 indicator (1 = male).

    Numeric columns must hold only 0 and 1. String columns must hold only
    Male/Female/M/F (any case, surrounding whitespace ignored). Missing values
    raise unless ``allow_missing`` is true, in which case they stay NaN for the
    caller's complete-case filter. A column with a single observed level raises:
    as a covariate it is collinear with the intercept.
    """
    s = pd.Series(col)
    miss = s.isna()
    if miss.any() and not allow_missing:
        raise ValueError(f"PTGENDER has {int(miss.sum())} missing values")
    obs = s[~miss]
    if pd.api.types.is_numeric_dtype(s):
        bad = sorted(set(obs.unique()) - {0, 1})
        if bad:
            raise ValueError(f"numeric PTGENDER must be coded 0/1; found {bad[:5]}")
        out = s.astype(float)
    else:
        up = obs.astype(str).str.strip().str.upper()
        bad = sorted(set(up.unique()) - _MALE - _FEMALE)
        if bad:
            raise ValueError(f"PTGENDER values not recognised as Male/Female/M/F: {bad[:5]}")
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out[~miss] = up.isin(_MALE).astype(float).to_numpy()
    if out[~miss].nunique() < 2:
        raise ValueError("PTGENDER is constant over the observed rows; it cannot serve as a covariate")
    return out.to_numpy(dtype=float)
