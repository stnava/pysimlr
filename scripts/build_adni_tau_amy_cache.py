"""Build and cache a cleaned ADNI tau/amyloid-beta multi-view dataset.

Merges two sources that don't otherwise overlap in this repo's pipeline:
  - The ANTsPyMM multi-modal imaging export (`adni_loader.SOURCE`):
    T1Hier/DTI/rsfMRI views, subject IDs in ADNI's `SITE_S_SUBJECT` format.
  - `~/Documents/writing/nick/Papers/CrossLong/Data/adni_elecsys_csf_nfl.csv`
    (ADNIMERGE-derived, confirmed genuine ADNI via its `PTID` format): CSF
    `amy` (amyloid-beta42, Elecsys assay) and `tau`, longitudinal (1213
    unique subjects, multiple visits each).

Only 47 subjects are in both sources (447 unique ADNI subjects have usable
imaging here; 1213 have CSF; the intersection is small because the two
exports were assembled for different studies/timepoints) -- small, but
real, and worth caching rather than recomputing the join and the ~1590-column
T1Hier filter on every run.

Per subject, the CSF visit closest to that subject's imaging date is kept
(preferring an exact-date match, falling back to the nearest by absolute day
difference) rather than always using baseline, so the two measurements are
temporally aligned rather than just "some CSF value this subject ever had."

Run with: python scripts/build_adni_tau_amy_cache.py
Output: data/adni/adni_tau_amy_merged.csv (one row per merged subject; T1Hier/
DTI_fa/rsfMRI_pro122 feature columns, amy/tau/tau_amy_ratio, AGE/PTGENDER/
PTEDUCAT, subjectID, days_between_scans).
"""
import os

import numpy as np
import pandas as pd

from adni_loader import SOURCE, _view_matrix

CSF_SOURCE = ("/Users/stnava/Documents/writing/nick/Papers/CrossLong/Data/"
             "adni_elecsys_csf_nfl.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "adni")
OUT_PATH = os.path.join(OUT_DIR, "adni_tau_amy_merged.csv")


def _nearest_csf_row(csf_subject: pd.DataFrame, target_date: pd.Timestamp) -> pd.Series:
    diffs = (csf_subject["EXAMDATE"] - target_date).abs()
    return csf_subject.loc[diffs.idxmin()]


def build():
    print("Loading imaging export...", flush=True)
    img = pd.read_csv(SOURCE, low_memory=False)
    img = img[img["subjectID"].astype(str).str.contains("_S_", na=False)].copy()
    img["date"] = pd.to_datetime(img["date"], format="%Y%m%d", errors="coerce")

    print("Loading CSF export...", flush=True)
    csf = pd.read_csv(CSF_SOURCE, low_memory=False)
    csf = csf[csf["amy"].notna() & csf["tau"].notna()].copy()
    csf["EXAMDATE"] = pd.to_datetime(csf["EXAMDATE"], errors="coerce")

    common_subjects = sorted(set(img["subjectID"]) & set(csf["PTID"]))
    print(f"{len(common_subjects)} subjects with both imaging and CSF", flush=True)

    t1 = _view_matrix(img, "T1Hier", max_features=300)
    dti = _view_matrix(img, "DTI_fa")
    rsf = _view_matrix(img, "rsfMRI_pro122")

    rows = []
    for subj in common_subjects:
        img_rows = img[img["subjectID"] == subj]
        # One imaging row per subject: if several, keep the one with the
        # most complete view data (fewest dropped features across the three
        # per-view filters already applied above).
        candidate_idx = [i for i in img_rows.index if i in t1.index and i in dti.index and i in rsf.index]
        if not candidate_idx:
            continue
        img_idx = candidate_idx[0]
        img_date = img.loc[img_idx, "date"]

        csf_subject = csf[csf["PTID"] == subj]
        if img_date is pd.NaT or csf_subject["EXAMDATE"].isna().all():
            csf_row = csf_subject.iloc[0]
            days_between = np.nan
        else:
            csf_row = _nearest_csf_row(csf_subject, img_date)
            days_between = abs((csf_row["EXAMDATE"] - img_date).days)

        row = {"subjectID": subj, "days_between_scans": days_between,
              "amy": csf_row["amy"], "tau": csf_row["tau"],
              "ptau": csf_row.get("ptau", np.nan),
              "AGE": csf_row.get("AGE", np.nan),
              "PTGENDER": csf_row.get("PTGENDER", np.nan),
              "PTEDUCAT": csf_row.get("PTEDUCAT", np.nan)}
        row.update({f"T1Hier__{c}": t1.loc[img_idx, c] for c in t1.columns})
        row.update({f"DTI_fa__{c}": dti.loc[img_idx, c] for c in dti.columns})
        row.update({f"rsfMRI_pro122__{c}": rsf.loc[img_idx, c] for c in rsf.columns})
        rows.append(row)

    out = pd.DataFrame(rows)
    out["tau_amy_ratio"] = out["tau"] / out["amy"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows, {out.shape[1]} columns to {OUT_PATH}", flush=True)
    print(f"days_between_scans: median={out['days_between_scans'].median():.0f}, "
         f"max={out['days_between_scans'].max():.0f}", flush=True)


if __name__ == "__main__":
    build()
