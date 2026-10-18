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
from typing import Optional, List, Dict, Union, Any
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


MCI_COGNITIVE_CACHE_PATH = os.path.join(CACHE_DIR, "adni_mci_cognitive_merged.csv")


def get_adni_mci_cognitive_case(cohort: str = "t1_cog", target: str = "CDRSB",
                                max_t1_features: int = 300) -> dict:
    """Real ADNI baseline Mild Cognitive Impairment (MCI) cohort with dense
    cognitive batteries and neuroimaging, predicting CDRSB and MMSE.

    Parameters
    ----------
    cohort : {"t1_cog", "multimodal"}, default="t1_cog"
        - "t1_cog": Structural T1 MRI (regional volumes + cortical thickness)
          and dense Cognitive Battery view across N=1084 baseline MCI subjects.
        - "multimodal": Structural T1 MRI, DTI FA white matter microstructure,
          and dense Cognitive Battery view across N=229 baseline MCI subjects with DTI.
    target : {"CDRSB", "MMSE"}, default="CDRSB"
        Primary target outcome returned under key 'y'. Both targets are always
        available under 'y_cdrsb' and 'y_mmse'.
    max_t1_features : int, default=300
        Maximum number of high-variance T1 features to include.

    Returns
    -------
    dict with keys:
        'views': list of torch.FloatTensor
        'y': target array specified by `target`
        'y_cdrsb': CDRSB continuous score
        'y_mmse': MMSE continuous score
        'covariates': age, sex (1=Male, 0=Female), education
        'covariate_names': ["AGE", "PTGENDER", "PTEDUCAT"]
        'view_names': list of str
        'view_dims': list of int
        'ptid': subject IDs
    """
    if not os.path.exists(MCI_COGNITIVE_CACHE_PATH):
        raise FileNotFoundError(
            f"{MCI_COGNITIVE_CACHE_PATH} not found. Run "
            f"`python scripts/build_adni_mci_cognitive_cache.py` first."
        )
    df = pd.read_csv(MCI_COGNITIVE_CACHE_PATH)
    if cohort == "multimodal":
        df = df[df["has_dti"] == 1].copy()

    if cohort == "4view":
        vol_cols = [c for c in df.columns if c.startswith("T1Hier_vol__")]
        thk_cols = [c for c in df.columns if c.startswith("T1Hier_thk__")]
        t1w_cols = [c for c in df.columns if c.startswith("T1w__")]
        cog_cols = [c for c in df.columns if c.startswith("Cog__")]
        mats = [
            df[vol_cols].to_numpy(dtype=float),
            df[thk_cols].to_numpy(dtype=float),
            df[t1w_cols].to_numpy(dtype=float),
            df[cog_cols].to_numpy(dtype=float),
        ]
        view_names = ["T1Hier_Volume", "T1Hier_Thickness", "T1w_Thickness", "Cognitive_Battery"]
    elif cohort == "multimodal":
        t1_cols = [c for c in df.columns if c.startswith("T1Hier__")][:max_t1_features]
        dti_cols = [c for c in df.columns if c.startswith("DTI_fa__")]
        cog_cols = [c for c in df.columns if c.startswith("Cog__")]
        mats = [
            df[t1_cols].to_numpy(dtype=float),
            df[dti_cols].to_numpy(dtype=float),
            df[cog_cols].to_numpy(dtype=float),
        ]
        view_names = ["T1Hier", "DTI_fa", "Cognitive_Battery"]
    else:
        t1_cols = [c for c in df.columns if c.startswith("T1Hier__")][:max_t1_features]
        cog_cols = [c for c in df.columns if c.startswith("Cog__")]
        mats = [
            df[t1_cols].to_numpy(dtype=float),
            df[cog_cols].to_numpy(dtype=float),
        ]
        view_names = ["T1Hier", "Cognitive_Battery"]

    cov = df[COVARIATES[:3]].to_numpy(dtype=float)
    y_cdrsb = df["CDRSB"].to_numpy(dtype=float)
    y_mmse = df["MMSE"].to_numpy(dtype=float)
    y_target = y_cdrsb if target.upper() == "CDRSB" else y_mmse

    return {
        "views": [torch.tensor(m).float() for m in mats],
        "y": y_target,
        "y_cdrsb": y_cdrsb,
        "y_mmse": y_mmse,
        "target": target,
        "covariates": cov,
        "covariate_names": ["AGE", "PTGENDER", "PTEDUCAT"],
        "view_names": view_names,
        "view_dims": [m.shape[1] for m in mats],
        "ptid": df["PTID"].to_numpy(),
    }


ALL_BL_COGNITIVE_CACHE_PATH = os.path.join(CACHE_DIR, "adni_all_bl_cognitive_merged.csv")


def get_adni_all_bl_cognitive_case(cohort: str = "4view", target: str = "CDRSB",
                                   min_amyloid_pet_suvr: Optional[float] = None,
                                   amyloid_tracer: str = "any",
                                   dx_filter: Optional[str] = None):
    """Complete ADNI Year 0 baseline cohort across all diagnostic groups.

    cohort : str, default="4view"
        "4view" uses T1Hier_Volume (160), T1Hier_Thickness (160),
        T1w_Thickness (62), and Cognitive_Battery (13).
        "multimodal" uses T1Hier_Volume, T1Hier_Thickness, DTI_fa, and Cognitive_Battery.
    target : str, default="CDRSB"
        "CDRSB" or "MMSE"
    min_amyloid_pet_suvr : float, optional
        If set (e.g. 1.0), filters for subjects with amyloid PET SUVR > threshold.
    amyloid_tracer : str, default="any"
        "any" uses composite AMYLOID_PET_SUVR (AV45 Florbetapir or FBB Florbetaben).
        "av45" strictly requires Florbetapir PET (AV45 > threshold).
    dx_filter : str, optional
        If set (e.g. "MCI", "CN", "Dementia"), filters for subjects with specific diagnosis.

    Returns
    -------
    dict with:
        'views': list of torch.Tensors
        'y': continuous target array
        'y_cdrsb': CDRSB array
        'y_mmse': MMSE array
        'dx': categorical diagnosis string array (CN, MCI, Dementia)
        'covariates': age, sex, education
        'covariate_names': ["AGE", "PTGENDER", "PTEDUCAT"]
        'view_names': list of str
        'view_dims': list of int
        'ptid': subject IDs
        'amyloid_pet_suvr': PET SUVR array
    """
    if not os.path.exists(ALL_BL_COGNITIVE_CACHE_PATH):
        raise FileNotFoundError(
            f"{ALL_BL_COGNITIVE_CACHE_PATH} not found. Run "
            f"`python scripts/build_adni_all_bl_cognitive_cache.py` first."
        )
    df = pd.read_csv(ALL_BL_COGNITIVE_CACHE_PATH)
    if cohort == "multimodal":
        df = df[df["has_dti"] == 1].copy()

    if dx_filter is not None:
        if isinstance(dx_filter, (list, tuple, set)):
            df = df[df["DX"].isin(dx_filter)].copy()
        else:
            df = df[df["DX"] == dx_filter].copy()

    if min_amyloid_pet_suvr is not None:
        if amyloid_tracer.lower() == "av45":
            df = df[df["AV45"] > min_amyloid_pet_suvr].copy()
        else:
            df = df[df["AMYLOID_PET_SUVR"] > min_amyloid_pet_suvr].copy()


    # Strictly exclude any macro-aggregate lobe or tissue predictors
    vol_cols = [c for c in df.columns if c.startswith("T1Hier_vol__") and
                "lobe" not in c.lower() and "tissue" not in c.lower()]
    thk_cols = [c for c in df.columns if c.startswith("T1Hier_thk__") and
                "lobe" not in c.lower() and "tissue" not in c.lower()]
    t1w_cols = [c for c in df.columns if c.startswith("T1w__") and
                "lobe" not in c.lower() and "tissue" not in c.lower()]
    cog_cols = [c for c in df.columns if c.startswith("Cog__")]

    if cohort == "multimodal":
        dti_cols = [c for c in df.columns if c.startswith("DTI_fa__")]
        mats = [
            df[vol_cols].to_numpy(dtype=float),
            df[thk_cols].to_numpy(dtype=float),
            df[dti_cols].to_numpy(dtype=float),
            df[cog_cols].to_numpy(dtype=float),
        ]
        view_names = ["T1Hier_Volume", "T1Hier_Thickness", "DTI_fa", "Cognitive_Battery"]
    else:
        mats = [
            df[vol_cols].to_numpy(dtype=float),
            df[thk_cols].to_numpy(dtype=float),
            df[t1w_cols].to_numpy(dtype=float),
            df[cog_cols].to_numpy(dtype=float),
        ]
        view_names = ["T1Hier_Volume", "T1Hier_Thickness", "T1w_Thickness", "Cognitive_Battery"]

    cov = df[COVARIATES[:3]].to_numpy(dtype=float)
    y_cdrsb = df["CDRSB"].to_numpy(dtype=float)
    y_mmse = df["MMSE"].to_numpy(dtype=float)
    y_target = y_cdrsb if target.upper() == "CDRSB" else y_mmse

    return {
        "views": [torch.tensor(m).float() for m in mats],
        "y": y_target,
        "y_cdrsb": y_cdrsb,
        "y_mmse": y_mmse,
        "dx": df["DX"].to_numpy(),
        "target": target,
        "covariates": cov,
        "covariate_names": ["AGE", "PTGENDER", "PTEDUCAT"],
        "view_names": view_names,
        "view_dims": [m.shape[1] for m in mats],
        "ptid": df["PTID"].to_numpy(),
        "amyloid_pet_suvr": df["AMYLOID_PET_SUVR"].to_numpy() if "AMYLOID_PET_SUVR" in df.columns else None,
        "av45": df["AV45"].to_numpy() if "AV45" in df.columns else None,
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

    all_bl = get_adni_all_bl_cognitive_case()
    print("\nAll baseline cohort: N =", len(all_bl["y"]))
    print("view dims:", dict(zip(all_bl["view_names"], all_bl["view_dims"])))
    print("DX breakdown:", pd.Series(all_bl["dx"]).value_counts().to_dict())
