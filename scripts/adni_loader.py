"""Real, combined ADNI+PPMI multi-modal neuroimaging cohort.

Source: an ANTsPyMM-derived export from the user's own prior work
(`~/Library/Mobile Documents/com~apple~CloudDocs/code/multidisorder/data/
ppmiadni_filtered.csv`), not part of this repo -- read directly from that
path rather than copied in, since it is a large (2547 x 20150) external data
file. This is genuinely multi-view: T1Hier (regional structural volumes/
thickness), DTI (diffusion tractography/microstructure), and rsfMRI
(resting-state functional connectivity) are three separate MRI acquisitions
and processing pipelines on the same subjects, not one table split in half --
the same bar `tcga_loader.py` uses for TCGA-BRCA. T2Flair is dropped: 100% NaN
in the DX-labeled subset.

All three views use a fixed, anatomically-meaningful column subset rather
than a variance cap: `DTI_mean_fa.*` (one FA value per JHU white-matter
tract), `rsfMRI_fcnxpro122_*_mean` (one summary value per functional network,
pro122 atlas, excluding the per-seed-point `...PointNNN...` columns), and for
T1Hier, `vol*`/`thk*` columns (regional volume or thickness) plus
`midbrain_pons_ratio` -- excluding `area*` (redundant with volume, noisier as
a cortical-surface measure) and `T1Hier`'s ~30 QC/processing columns
(`RandBasisProj*`, `loop_outlier_probability`, `mhdist`, `templateL1`,
`evratio`, `resnetGrade*`, `u_hier_id*`), which are not brain measurements.
All three loaders (DX and CDRSB) use the same filters so every task is on
comparable features.
"""
import os
import re
import numpy as np
import pandas as pd
import torch

SOURCE = ("/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/"
         "code/multidisorder/data/ppmiadni_filtered.csv")

VIEWS = ["T1Hier", "DTI", "rsfMRI"]

COVARIATES = ["AGE", "PTGENDER", "PTEDUCAT"]


def _clean_columns(m: pd.DataFrame, max_features=None) -> pd.DataFrame:
    m = m.loc[:, m.notna().mean() >= 0.8]
    m = m.fillna(m.median(numeric_only=True))
    m = m.loc[:, m.std() > 0]
    if max_features is not None and m.shape[1] > max_features:
        keep = m.std().sort_values(ascending=False).index[:max_features]
        m = m[sorted(keep)]
    return m


def _view_matrix(df: pd.DataFrame, prefix: str, max_features=None) -> pd.DataFrame:
    """`T1Hier`/`DTI`/`rsfMRI` picked by variance cap (see `get_adni_case`);
    `DTI_fa` and `rsfMRI_pro122` pick a fixed anatomical/network subset
    instead (see module docstring).
    """
    if prefix == "DTI_fa":
        cols = [c for c in df.columns if c.startswith("DTI_mean_fa.")]
    elif prefix == "rsfMRI_pro122":
        cols = [c for c in df.columns
                if c.startswith("rsfMRI_fcnxpro122_") and c.endswith("_mean")
                and "Point" not in c]
    elif prefix == "T1Hier":
        # `T1Hier_*` also carries 30 QC/processing columns that are not brain
        # measurements at all: `RandBasisProj*` (a QC random-projection
        # basis), `loop_outlier_probability`, `mhdist`, `templateL1`,
        # `evratio`, `resnetGrade*` (registration/image-quality scores), and
        # `u_hier_id*` (row identifiers). Anatomical columns are `vol*` or
        # `thk*` (volume/thickness of a brain region, lobe or tissue class --
        # covers both `vol_frontal_ldktlobes` and the underscore-less
        # `volAsymcrus_icerebellum` asymmetry/L-R-average variants) plus
        # `midbrain_pons_ratio`, a genuine anatomical ratio. `area*` is
        # excluded on request -- redundant with volume for most regions and
        # noisier as a cortical-surface measure.
        cols = [c for c in df.columns
                if c.startswith("T1Hier_")
                and (re.match(r"^T1Hier_(vol|thk)", c) or "midbrain_pons_ratio" in c)]
    else:
        cols = [c for c in df.columns if c.startswith(prefix + "_")]
    m = df[cols].select_dtypes(include=[np.number])
    return _clean_columns(m, max_features)


def get_adni_case(max_features: int = 300):
    """Three real MRI-modality views plus a 3-class diagnostic label (`DX`).

    T1Hier is first restricted to its ~1560 genuinely anatomical columns
    (see module docstring), then capped to the `max_features` highest-
    variance ones among those (the same speed cap `tcga_loader.get_tcga_case`
    uses) -- not a variance cap over the raw, QC-column-mixed `T1Hier_*` set.
    DTI and rsfMRI use the fixed `mean_fa` / `pro122`-network subsets, no cap.
    """
    df = pd.read_csv(SOURCE, low_memory=False)
    df = df[df["DX"].notna()].copy()

    mats = [_view_matrix(df, "T1Hier", max_features=max_features),
            _view_matrix(df, "DTI_fa"),
            _view_matrix(df, "rsfMRI_pro122")]

    # Views were filtered independently (different NaN/constant columns per
    # view), so re-intersect on row index before stacking.
    shared = mats[0].index
    for m in mats[1:]:
        shared = shared.intersection(m.index)
    mats = [m.loc[shared].to_numpy(dtype=float) for m in mats]
    y = df.loc[shared, "DX"].to_numpy()

    return {
        "views": [torch.tensor(m).float() for m in mats],
        "y": y,
        "view_names": ["T1Hier", "DTI_fa", "rsfMRI_pro122"],
        "view_dims": [m.shape[1] for m in mats],
    }


def get_adni_cdrsb_case(max_t1hier_features: int = 300, dx_filter=None, target: str = "CDRSB"):
    """Three real MRI-modality views (T1Hier anatomical, DTI mean_fa,
    rsfMRI pro122-network) plus age/sex/education covariates, predicting a
    continuous clinical score (default `CDRSB`).

    dx_filter : str or list of str, optional
        Restrict to these `DX` values (e.g. `"MCI"` or `["MCI", "Dementia"]`).
        `None` keeps every subject with a non-missing target regardless of
        DX (the default, unfiltered cohort).
    target : str, default="CDRSB"
        Which column to predict, e.g. `"CDRSB"` or `"MMSE"`. Despite the
        function name (kept for backward compatibility -- existing callers
        pass no `target` and get CDRSB), any continuous ADNI column works.
    """
    df = pd.read_csv(SOURCE, low_memory=False)
    df = df[df[target].notna()].copy()
    if dx_filter is not None:
        keep = {dx_filter} if isinstance(dx_filter, str) else set(dx_filter)
        df = df[df["DX"].isin(keep)].copy()

    mats = [_view_matrix(df, "T1Hier", max_features=max_t1hier_features),
            _view_matrix(df, "DTI_fa"),
            _view_matrix(df, "rsfMRI_pro122")]

    shared = mats[0].index
    for m in mats[1:]:
        shared = shared.intersection(m.index)
    shared = shared[df.loc[shared, COVARIATES].notna().all(axis=1)]

    mats = [m.loc[shared].to_numpy(dtype=float) for m in mats]
    y = df.loc[shared, target].to_numpy(dtype=float)
    cov = df.loc[shared, COVARIATES].copy()
    cov["PTGENDER"] = (cov["PTGENDER"] == "Male").astype(float)
    cov = cov.to_numpy(dtype=float)

    return {
        "views": [torch.tensor(m).float() for m in mats],
        "y": y,
        "target": target,
        "covariates": cov,
        "covariate_names": COVARIATES,
        "view_names": ["T1Hier", "DTI_fa", "rsfMRI_pro122"],
        "view_dims": [m.shape[1] for m in mats],
    }


def get_ppmi_tau_abeta_case(max_t1hier_features: int = 300):
    """Three real MRI-modality views plus age/sex/education covariates,
    predicting the CSF tau/abeta ratio.

    A different sub-cohort from `get_adni_case`/`get_adni_cdrsb_case`: `tau`/
    `abeta` are only non-missing for 241 rows, all PPMI (they carry a
    `COHORT` code and `PATNO`, and `DX` -- the ADNI diagnostic label -- is
    NaN for every one of them; ADNI's own `ABETA`/`TAU` columns have only 5
    non-missing rows total, unusably few). PPMI's covariate columns use
    different names than ADNI's: `age`/`EDUCYRS` instead of `AGE`/
    `PTEDUCAT`, and no `PTGENDER` at all -- `commonSex` ("Male"/"Female")
    is used instead.
    """
    df = pd.read_csv(SOURCE, low_memory=False)
    df = df.copy()
    df["tau_abeta_ratio"] = df["tau"] / df["abeta"]
    df = df[df["tau_abeta_ratio"].notna()].copy()

    mats = [_view_matrix(df, "T1Hier", max_features=max_t1hier_features),
            _view_matrix(df, "DTI_fa"),
            _view_matrix(df, "rsfMRI_pro122")]

    ppmi_covariates = ["age", "commonSex", "EDUCYRS"]
    shared = mats[0].index
    for m in mats[1:]:
        shared = shared.intersection(m.index)
    shared = shared[df.loc[shared, ppmi_covariates].notna().all(axis=1)]

    mats = [m.loc[shared].to_numpy(dtype=float) for m in mats]
    y = df.loc[shared, "tau_abeta_ratio"].to_numpy(dtype=float)
    cov = df.loc[shared, ppmi_covariates].copy()
    cov["commonSex"] = (cov["commonSex"] == "Male").astype(float)
    cov = cov.to_numpy(dtype=float)

    return {
        "views": [torch.tensor(m).float() for m in mats],
        "y": y,
        "target": "tau_abeta_ratio",
        "covariates": cov,
        "covariate_names": ppmi_covariates,
        "view_names": ["T1Hier", "DTI_fa", "rsfMRI_pro122"],
        "view_dims": [m.shape[1] for m in mats],
    }


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "adni")
TAU_AMY_CACHE_PATH = os.path.join(CACHE_DIR, "adni_tau_amy_merged.csv")


def get_adni_tau_amy_case(target: str = "tau_amy_ratio"):
    """Real ADNI tau/amyloid-beta cohort, from the cached merge built by
    `scripts/build_adni_tau_amy_cache.py` (run that script first, or if the
    cache is missing this raises rather than silently rebuilding it inline --
    the merge takes a real join against an external longitudinal CSF file
    and should be an explicit, visible step, not a side effect of loading).

    N=47: only 47 of this repo's ~447 usable ADNI imaging subjects have a
    matching CSF `amy`/`tau` measurement in the available export (see the
    build script's docstring). The CSF measurement is the visit temporally
    closest to the imaging date, not necessarily concurrent -- the cached
    file's `days_between_scans` column records the actual gap (median 712
    days in the current cache); check it before treating a result here as
    more than a rough signal check.

    target : str, default="tau_amy_ratio"
        `"tau_amy_ratio"`, `"tau"`, `"amy"`, or `"ptau"`.
    """
    if not os.path.exists(TAU_AMY_CACHE_PATH):
        raise FileNotFoundError(
            f"{TAU_AMY_CACHE_PATH} not found. Run "
            f"`python scripts/build_adni_tau_amy_cache.py` first to build it."
        )
    df = pd.read_csv(TAU_AMY_CACHE_PATH)
    df = df[df[target].notna()].copy()

    view_prefixes = {"T1Hier": "T1Hier__", "DTI_fa": "DTI_fa__", "rsfMRI_pro122": "rsfMRI_pro122__"}
    mats = []
    for name, prefix in view_prefixes.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        mats.append(df[cols].to_numpy(dtype=float))

    y = df[target].to_numpy(dtype=float)
    cov = df[COVARIATES].copy()
    cov["PTGENDER"] = (cov["PTGENDER"] == "Male").astype(float)
    cov = cov.to_numpy(dtype=float)

    return {
        "views": [torch.tensor(m).float() for m in mats],
        "y": y,
        "target": target,
        "covariates": cov,
        "covariate_names": COVARIATES,
        "view_names": list(view_prefixes.keys()),
        "view_dims": [m.shape[1] for m in mats],
        "days_between_scans": df["days_between_scans"].to_numpy(dtype=float),
    }


T1_DTI_BIOMARKER_CACHE_PATH = os.path.join(CACHE_DIR, "adni_t1_dti_biomarker_merged.csv")


def get_adni_t1_dti_biomarker_case(target: str = "tau_abeta_ratio"):
    """Real ADNI T1Hier+DTI cohort with CSF biomarkers, from the cache built
    by `scripts/build_adni_t1_dti_biomarker_cache.py` (run that script first;
    raises rather than silently rebuilding, same reasoning as
    `get_adni_tau_amy_case`).

    N=193 -- built from `ADNIMERGE_10Feb2024_antspymm_v1.4.0.csv`, a better
    ADNIMERGE+ANTsPyMM merge than `get_adni_tau_amy_case`'s source (covers
    ADNI3, not just through 2016), about 4x its N=47. Only 2 views (T1Hier,
    DTI) -- rsfMRI is excluded because rsfMRI and DTI are almost never
    acquired at the same visit in this cohort (only 6 subjects would have
    all three), so a 3-view version of this dataset is not usable.

    target : str, default="tau_abeta_ratio"
        `"tau_abeta_ratio"`, `"TAU"`, `"ABETA"`, or `"PTAU"`.
    """
    if not os.path.exists(T1_DTI_BIOMARKER_CACHE_PATH):
        raise FileNotFoundError(
            f"{T1_DTI_BIOMARKER_CACHE_PATH} not found. Run "
            f"`python scripts/build_adni_t1_dti_biomarker_cache.py` first to build it."
        )
    df = pd.read_csv(T1_DTI_BIOMARKER_CACHE_PATH)
    df = df[df[target].notna()].copy()

    view_prefixes = {"T1Hier": "T1Hier__", "DTI_fa": "DTI_fa__"}
    mats = [df[[c for c in df.columns if c.startswith(prefix)]].to_numpy(dtype=float)
           for prefix in view_prefixes.values()]

    y = df[target].to_numpy(dtype=float)
    cov = df[["AGE", "PTGENDER", "PTEDUCAT"]].copy()
    cov["PTGENDER"] = (cov["PTGENDER"] == "Male").astype(float)
    cov = cov.to_numpy(dtype=float)

    return {
        "views": [torch.tensor(m).float() for m in mats],
        "y": y,
        "target": target,
        "covariates": cov,
        "covariate_names": ["AGE", "PTGENDER", "PTEDUCAT"],
        "view_names": list(view_prefixes.keys()),
        "view_dims": [m.shape[1] for m in mats],
        "viscode": df["VISCODE"].to_numpy(),
    }


if __name__ == "__main__":
    case = get_adni_case()
    print("DX case: N =", len(case["y"]))
    print("view dims:", dict(zip(case["view_names"], case["view_dims"])))
    print("label counts:", pd.Series(case["y"]).value_counts().to_dict())

    cdrsb = get_adni_cdrsb_case()
    print("\nCDRSB case: N =", len(cdrsb["y"]))
    print("view dims:", dict(zip(cdrsb["view_names"], cdrsb["view_dims"])))
    print("CDRSB stats:", pd.Series(cdrsb["y"]).describe().to_dict())
