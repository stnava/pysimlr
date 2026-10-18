"""Build and cache a dedicated ADNI MCI multi-view dataset with dense cognitive batteries.

This script extracts baseline Mild Cognitive Impairment (MCI) participants from
ADNIMERGE+ANTsPyMM, combining:
1. Structural MRI (T1Hier regional volumes and cortical thickness)
2. Diffusion MRI (DTI tract fractional anisotropy, for available sub-cohort)
3. Dense Cognitive/Clinical Batteries (ADAS11, ADAS13, ADASQ4, FAQ, RAVLT subscales,
   LDELTOTAL, TRABSCOR, MOCA, EcogPtTotal, EcogSPTotal)
4. Targets: CDRSB (clinical dementia rating sum of boxes) and MMSE (mini-mental state examination)
5. Covariates: AGE, PTGENDER, PTEDUCAT, APOE4

Strictly excludes target variables (CDRSB, MMSE) and algebraically composite measures
(e.g., mPACCdigit, mPACCtrailsB) from the cognitive view to prevent data leakage.

Output: data/adni/adni_mci_cognitive_merged.csv
"""
import os
import re
import numpy as np
import pandas as pd

SOURCE = ("/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/"
          "code/multidisorder/data/ppmi_pym_data/"
          "ADNIMERGE_10Feb2024_antspymm_v1.4.0.csv")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "adni")
OUT_PATH = os.path.join(OUT_DIR, "adni_mci_cognitive_merged.csv")

COGNITIVE_COLS = [
    "ADAS11", "ADAS13", "ADASQ4", "FAQ",
    "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
    "LDELTOTAL", "TRABSCOR", "MOCA", "EcogPtTotal", "EcogSPTotal"
]

COVARIATES = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4"]
TARGETS = ["CDRSB", "MMSE"]


def _clean_columns(m: pd.DataFrame, max_features=None) -> pd.DataFrame:
    m = m.loc[:, m.notna().mean() >= 0.8]
    m = m.fillna(m.median(numeric_only=True))
    m = m.loc[:, m.std() > 0]
    if max_features is not None and m.shape[1] > max_features:
        keep = m.std().sort_values(ascending=False).index[:max_features]
        m = m[sorted(keep)]
    return m


def build(max_t1hier_features: int = 300):
    print(f"Loading ADNIMERGE+ANTsPyMM source from {SOURCE}...", flush=True)
    df = pd.read_csv(SOURCE, low_memory=False)

    print("Filtering for baseline MCI cohort (DX == 'MCI', VISCODE == 'bl')...", flush=True)
    mci = df[(df["DX"] == "MCI") & (df["VISCODE"] == "bl")].copy()
    print(f"Found {len(mci)} baseline MCI rows across {mci['PTID'].nunique()} unique subjects", flush=True)

    # Require both targets to be non-null
    mci = mci[mci["CDRSB"].notna() & mci["MMSE"].notna()].copy()
    print(f"{len(mci)} subjects with both CDRSB and MMSE present", flush=True)

    # 1. T1 Volume Features
    vol_cols = [c for c in mci.columns if c.startswith("T1Hier_") and
                (re.match(r"^T1Hier_vol", c) or "midbrain_pons_ratio" in c)]
    vol_df = mci[vol_cols].apply(pd.to_numeric, errors="coerce")
    valid_vol_rows = vol_df.notna().sum(axis=1) > (len(vol_cols) * 0.5)
    mci = mci[valid_vol_rows].copy()
    vol_df = vol_df.loc[mci.index]
    vol_clean = _clean_columns(vol_df, max_features=160)
    print(f"T1 Volume extracted: {vol_clean.shape[1]} features across {len(mci)} subjects", flush=True)

    # 2. T1 Cortical Thickness Features
    thk_cols = [c for c in mci.columns if c.startswith("T1Hier_") and re.match(r"^T1Hier_thk", c)]
    thk_df = mci[thk_cols].apply(pd.to_numeric, errors="coerce").loc[mci.index]
    thk_clean = _clean_columns(thk_df, max_features=160)
    print(f"T1 Thickness extracted: {thk_clean.shape[1]} features across {len(mci)} subjects", flush=True)

    # 3. T1w Tissue Intensity Features (T1w_mean_*)
    t1w_cols = [c for c in mci.columns if c.startswith("T1w_mean_") and "background" not in c]
    t1w_df = mci[t1w_cols].apply(pd.to_numeric, errors="coerce").loc[mci.index]
    t1w_clean = _clean_columns(t1w_df)
    print(f"T1w Intensity extracted: {t1w_clean.shape[1]} features across {len(mci)} subjects", flush=True)

    # 4. Cognitive Battery Features (Strictly excluding CDRSB, MMSE, mPACC)
    cog_df = mci[COGNITIVE_COLS].apply(pd.to_numeric, errors="coerce").loc[mci.index]
    cog_clean = cog_df.fillna(cog_df.median(numeric_only=True))
    print(f"Cognitive Battery extracted: {cog_clean.shape[1]} features across {len(mci)} subjects", flush=True)

    # 5. DTI Microstructure Features (available for sub-cohort)
    dti_cols = [c for c in mci.columns if c.startswith("DTI_mean_fa.")]
    dti_df = mci[dti_cols].apply(pd.to_numeric, errors="coerce").loc[mci.index]
    has_dti = (dti_df.notna().sum(axis=1) > (len(dti_cols) * 0.8)).astype(int)
    dti_sub = dti_df[has_dti == 1]
    valid_dti_cols = [c for c in dti_cols if dti_sub[c].notna().mean() >= 0.8 and dti_sub[c].std() > 0]
    dti_clean = dti_df[valid_dti_cols].fillna(dti_sub[valid_dti_cols].median(numeric_only=True))
    print(f"DTI extracted: {dti_clean.shape[1]} features (complete in {has_dti.sum()} subjects)", flush=True)

    # 6. Covariates & Targets
    out = pd.DataFrame({
        "PTID": mci["PTID"].to_numpy(),
        "VISCODE": mci["VISCODE"].to_numpy(),
        "CDRSB": pd.to_numeric(mci["CDRSB"], errors="coerce").to_numpy(),
        "MMSE": pd.to_numeric(mci["MMSE"], errors="coerce").to_numpy(),
        "AGE": pd.to_numeric(mci["AGE"], errors="coerce").to_numpy(),
        "PTGENDER": (mci["PTGENDER"].astype(str) == "Male").astype(float).to_numpy(),
        "PTEDUCAT": pd.to_numeric(mci["PTEDUCAT"], errors="coerce").to_numpy(),
        "APOE4": pd.to_numeric(mci["APOE4"], errors="coerce").fillna(0.0).to_numpy(),
        "has_dti": has_dti.to_numpy(),
    }, index=mci.index)

    # Rename columns with prefix and concatenate in one shot
    vol_renamed = vol_clean.rename(columns=lambda c: f"T1Hier_vol__{c}")
    thk_renamed = thk_clean.rename(columns=lambda c: f"T1Hier_thk__{c}")
    t1w_renamed = t1w_clean.rename(columns=lambda c: f"T1w__{c}")
    cog_renamed = cog_clean.rename(columns=lambda c: f"Cog__{c}")
    dti_renamed = dti_clean.rename(columns=lambda c: f"DTI_fa__{c}")

    # Also keep combined T1Hier prefix for backward compatibility
    t1_comb = pd.concat([vol_clean, thk_clean], axis=1).rename(columns=lambda c: f"T1Hier__{c}")

    out = pd.concat([out, vol_renamed, thk_renamed, t1w_renamed, cog_renamed, dti_renamed, t1_comb], axis=1)

    # Drop any remaining row with NaN in covariates
    cov_valid = out[["AGE", "PTGENDER", "PTEDUCAT"]].notna().all(axis=1)
    out = out[cov_valid].copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSuccessfully wrote {len(out)} rows and {out.shape[1]} columns to {OUT_PATH}", flush=True)
    print(f"Sample breakdown: {len(out)} MCI subjects, {out['has_dti'].sum()} with complete DTI", flush=True)
    print(f"CDRSB summary: mean={out['CDRSB'].mean():.2f}, std={out['CDRSB'].std():.2f}, range=[{out['CDRSB'].min():.1f}, {out['CDRSB'].max():.1f}]", flush=True)
    print(f"MMSE summary: mean={out['MMSE'].mean():.2f}, std={out['MMSE'].std():.2f}, range=[{out['MMSE'].min():.1f}, {out['MMSE'].max():.1f}]", flush=True)


if __name__ == "__main__":
    build()
