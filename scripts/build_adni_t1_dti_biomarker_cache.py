"""Build and cache a real, adequately-sized ADNI T1Hier+DTI / tau+amyloid
dataset, using a better-merged source than `build_adni_tau_amy_cache.py`'s.

`ADNIMERGE_10Feb2024_antspymm_v1.4.0.csv` (from the user's prior work,
`~/Library/Mobile Documents/com~apple~CloudDocs/code/multidisorder/data/
ppmi_pym_data/`) already merges ADNIMERGE (which carries `ABETA`/`TAU`/
`PTAU`) with ANTsPyMM imaging per subject-visit, current through Feb 2024 --
unlike `adni_elecsys_csf_nfl.csv` (stops in 2016, ADNI1/GO/ADNI2 only), this
one actually covers the ADNI3-era imaging.

Even so, DTI and rsfMRI are almost never acquired at the same visit in this
cohort (only 6 subjects have both plus a biomarker) -- so this uses T1Hier +
DTI only, which gives 193 subjects with `ABETA` and `TAU` both present,
about 4x `build_adni_tau_amy_cache.py`'s N=47. (T1Hier + rsfMRI is the other
viable pair, N=175 -- see that script's investigation for the numbers on all
four view combinations.)

One row per subject: prefers `VISCODE == "bl"`, falling back to the
earliest `EXAMDATE` available for that subject when no baseline row has
both biomarkers and both views.

Run with: python scripts/build_adni_t1_dti_biomarker_cache.py
Output: data/adni/adni_t1_dti_biomarker_merged.csv
"""
import os
import re

import numpy as np
import pandas as pd

from adni_loader import _clean_columns

SOURCE = ("/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/"
         "code/multidisorder/data/ppmi_pym_data/"
         "ADNIMERGE_10Feb2024_antspymm_v1.4.0.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "adni")
OUT_PATH = os.path.join(OUT_DIR, "adni_t1_dti_biomarker_merged.csv")

COVARIATES = ["AGE", "PTGENDER", "PTEDUCAT"]


def build(max_t1hier_features: int = 300):
    print("Loading ADNIMERGE+ANTsPyMM export (large file)...", flush=True)
    df = pd.read_csv(SOURCE, low_memory=False)

    t1_cols = [c for c in df.columns if re.match(r"^T1Hier_(vol|thk)", c)]
    dti_cols = [c for c in df.columns if c.startswith("DTI_mean_fa.")]

    has_biomarker = df["ABETA"].notna() & df["TAU"].notna()
    has_t1 = df[t1_cols].notna().all(axis=1)
    has_dti = df[dti_cols].notna().all(axis=1)
    usable = df[has_biomarker & has_t1 & has_dti].copy()
    print(f"{len(usable)} usable rows across {usable['PTID'].nunique()} subjects", flush=True)

    usable["EXAMDATE"] = pd.to_datetime(usable["EXAMDATE"], errors="coerce")
    usable["_is_bl"] = (usable["VISCODE"] == "bl").astype(int)
    # One row per subject: baseline first, then earliest EXAMDATE.
    usable = usable.sort_values(["PTID", "_is_bl", "EXAMDATE"], ascending=[True, False, True])
    one_per_subject = usable.groupby("PTID", as_index=False).first()
    print(f"{len(one_per_subject)} subjects after one-row-per-subject dedup", flush=True)

    t1 = _clean_columns(one_per_subject[t1_cols].apply(pd.to_numeric, errors="coerce"),
                        max_features=max_t1hier_features)
    dti = _clean_columns(one_per_subject[dti_cols].apply(pd.to_numeric, errors="coerce"))

    out = pd.DataFrame({
        "PTID": one_per_subject["PTID"].to_numpy(),
        "VISCODE": one_per_subject["VISCODE"].to_numpy(),
        "ABETA": pd.to_numeric(one_per_subject["ABETA"], errors="coerce").to_numpy(),
        "TAU": pd.to_numeric(one_per_subject["TAU"], errors="coerce").to_numpy(),
        "PTAU": pd.to_numeric(one_per_subject["PTAU"], errors="coerce").to_numpy(),
        "AGE": pd.to_numeric(one_per_subject["AGE"], errors="coerce").to_numpy(),
        "PTGENDER": one_per_subject["PTGENDER"].to_numpy(),
        "PTEDUCAT": pd.to_numeric(one_per_subject["PTEDUCAT"], errors="coerce").to_numpy(),
    })
    for c in t1.columns:
        out[f"T1Hier__{c}"] = t1[c].to_numpy()
    for c in dti.columns:
        out[f"DTI_fa__{c}"] = dti[c].to_numpy()
    out["tau_abeta_ratio"] = out["TAU"] / out["ABETA"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows, {out.shape[1]} columns to {OUT_PATH}", flush=True)
    print(f"VISCODE breakdown: {out['VISCODE'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    build()
