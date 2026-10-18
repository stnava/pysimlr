"""Build and cache the complete ADNI baseline cohort across all diagnostic groups.

Combines all baseline (VISCODE == 'bl', Year 0) ADNI subjects independent of DX:
- Cognitively Normal (CN)
- Mild Cognitive Impairment (MCI)
- Dementia / Alzheimer's Disease (AD)

Views:
1. Macrostructural Regional Volumetrics (T1Hier_vol, 160 features)
2. Dual-Pipeline Cortical Thickness (T1Hier_thk, 160 features)
3. Native T1w Intensity / Morphology (T1w_mean, 62 features)
4. Dense Multi-domain Cognitive Battery (13 features: ADAS, RAVLT, Logical Memory, Trails B, MoCA, ECog)

Targets: CDRSB and MMSE
Covariates: AGE, PTGENDER, PTEDUCAT, APOE4

Output: data/adni/adni_all_bl_cognitive_merged.csv
"""

import os
import re
import numpy as np
import pandas as pd

SOURCE = ("/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/"
          "code/multidisorder/data/ppmi_pym_data/"
          "ADNIMERGE_10Feb2024_antspymm_v1.4.0.csv")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "adni")
OUT_PATH = os.path.join(OUT_DIR, "adni_all_bl_cognitive_merged.csv")

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


def build():
    print(f"Loading ADNIMERGE+ANTsPyMM source from {SOURCE}...", flush=True)
    df = pd.read_csv(SOURCE, low_memory=False)

    print("Filtering for ALL baseline subjects (VISCODE == 'bl') independent of DX...", flush=True)
    bl = df[df["VISCODE"] == "bl"].copy()
    print(f"Found {len(bl)} baseline rows across {bl['PTID'].nunique()} unique subjects", flush=True)

    # Require both targets to be non-null
    bl = bl[bl["CDRSB"].notna() & bl["MMSE"].notna()].copy()
    print(f"{len(bl)} subjects with both CDRSB and MMSE present", flush=True)
    print(f"DX breakdown:\n{bl['DX'].value_counts(dropna=False)}", flush=True)

    # 1. T1 Volume Features (excluding lobe and tissue macro-aggregates)
    vol_cols = [c for c in bl.columns if c.startswith("T1Hier_") and
                (re.match(r"^T1Hier_vol", c) or "midbrain_pons_ratio" in c) and
                "lobe" not in c.lower() and "tissue" not in c.lower()]
    vol_df = bl[vol_cols].apply(pd.to_numeric, errors="coerce")
    valid_vol_rows = vol_df.notna().sum(axis=1) > (len(vol_cols) * 0.5)
    bl = bl[valid_vol_rows].copy()
    vol_df = vol_df.loc[bl.index]
    vol_clean = _clean_columns(vol_df, max_features=160)
    print(f"T1 Volume extracted (clean, non-lobe/tissue): {vol_clean.shape[1]} features across {len(bl)} subjects", flush=True)

    # 2. T1 Cortical Thickness Features (excluding lobe and tissue macro-aggregates)
    thk_cols = [c for c in bl.columns if c.startswith("T1Hier_") and re.match(r"^T1Hier_thk", c) and
                "lobe" not in c.lower() and "tissue" not in c.lower()]
    thk_df = bl[thk_cols].apply(pd.to_numeric, errors="coerce").loc[bl.index]
    thk_clean = _clean_columns(thk_df, max_features=160)
    print(f"T1 Thickness extracted (clean, non-lobe/tissue): {thk_clean.shape[1]} features across {len(bl)} subjects", flush=True)

    # 3. T1w Tissue Intensity Features (T1w_mean_*)
    t1w_cols = [c for c in bl.columns if c.startswith("T1w_mean_") and "background" not in c and
                "lobe" not in c.lower() and "tissue" not in c.lower()]
    t1w_df = bl[t1w_cols].apply(pd.to_numeric, errors="coerce").loc[bl.index]
    t1w_clean = _clean_columns(t1w_df)
    print(f"T1w Intensity extracted: {t1w_clean.shape[1]} features across {len(bl)} subjects", flush=True)

    # 4. Cognitive Battery Features (Strictly excluding CDRSB, MMSE, mPACC)
    cog_df = bl[COGNITIVE_COLS].apply(pd.to_numeric, errors="coerce").loc[bl.index]
    cog_clean = cog_df.fillna(cog_df.median(numeric_only=True))
    print(f"Cognitive Battery extracted: {cog_clean.shape[1]} features across {len(bl)} subjects", flush=True)

    # 5. DTI Microstructure Features (available for sub-cohort)
    dti_cols = [c for c in bl.columns if c.startswith("DTI_mean_fa.")]
    dti_df = bl[dti_cols].apply(pd.to_numeric, errors="coerce").loc[bl.index]
    has_dti = (dti_df.notna().sum(axis=1) > (len(dti_cols) * 0.8)).astype(int)
    dti_sub = dti_df[has_dti == 1]
    valid_dti_cols = [c for c in dti_cols if dti_sub[c].notna().mean() >= 0.8 and dti_sub[c].std() > 0]
    dti_clean = dti_df[valid_dti_cols].fillna(dti_sub[valid_dti_cols].median(numeric_only=True))
    print(f"DTI extracted: {dti_clean.shape[1]} features (complete in {has_dti.sum()} subjects)", flush=True)

    # 6. Amyloid PET Biomarkers (AV45 Florbetapir, FBB Florbetaben)
    av45_vals = pd.to_numeric(bl["AV45"], errors="coerce")
    fbb_vals = pd.to_numeric(bl["FBB"], errors="coerce")
    pet_suvr = av45_vals.combine_first(fbb_vals)

    # 7. Covariates & Targets
    out = pd.DataFrame({
        "PTID": bl["PTID"].to_numpy(),
        "VISCODE": bl["VISCODE"].to_numpy(),
        "DX": bl["DX"].fillna("Unknown").to_numpy(),
        "CDRSB": pd.to_numeric(bl["CDRSB"], errors="coerce").to_numpy(),
        "MMSE": pd.to_numeric(bl["MMSE"], errors="coerce").to_numpy(),
        "AGE": pd.to_numeric(bl["AGE"], errors="coerce").to_numpy(),
        "PTGENDER": (bl["PTGENDER"].astype(str) == "Male").astype(float).to_numpy(),
        "PTEDUCAT": pd.to_numeric(bl["PTEDUCAT"], errors="coerce").to_numpy(),
        "APOE4": pd.to_numeric(bl["APOE4"], errors="coerce").fillna(0.0).to_numpy(),
        "AV45": av45_vals.to_numpy(),
        "FBB": fbb_vals.to_numpy(),
        "AMYLOID_PET_SUVR": pet_suvr.to_numpy(),
        "has_dti": has_dti.to_numpy(),
    }, index=bl.index)

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
    print(f"Sample breakdown: {len(out)} total baseline subjects independent of DX", flush=True)
    print(f"DX counts:\n{out['DX'].value_counts()}", flush=True)
    print(f"CDRSB summary: mean={out['CDRSB'].mean():.2f}, std={out['CDRSB'].std():.2f}, range=[{out['CDRSB'].min():.1f}, {out['CDRSB'].max():.1f}]", flush=True)
    print(f"MMSE summary: mean={out['MMSE'].mean():.2f}, std={out['MMSE'].std():.2f}, range=[{out['MMSE'].min():.1f}, {out['MMSE'].max():.1f}]", flush=True)


if __name__ == "__main__":
    build()
