"""TCGA multi-omic cohorts from the UCSC Xena hub.

TCGA-BRCA was the only cohort in this benchmark whose views are genuinely
separate assays rather than one feature table cut in half, which made it the
only place the multi-view claims could be tested honestly -- and a single
cohort cannot distinguish "this method works" from "this dataset suits it".
These are the same three assays on other tumour types, so a conclusion that
holds on one and not the others is visible as such.

Source: the Xena TCGA hub, which serves per-cancer matrices over plain HTTPS
with no credentials. cBioPortal's datahub and Xena's GDC hub both answer 403
from here; the pan-cancer toil file is reachable but is 720 MB of RNA alone,
which is one view and so no use for this.

Roughly 40 MB per cancer, cached under ``data/tcga`` after the first call.
"""
from __future__ import annotations

import gzip
import os
import urllib.request

import numpy as np
import pandas as pd
import torch

HUB = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "tcga")

#: assay -> Xena filename template. The three matching BRCA's rs_/cn_/pp_.
ASSAYS = {
    "rna": "TCGA.{c}.sampleMap%2FHiSeqV2.gz",
    "cnv": "TCGA.{c}.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    "rppa": "TCGA.{c}.sampleMap%2FRPPA.gz",
}
CLINICAL = "TCGA.{c}.sampleMap%2F{c}_clinicalMatrix"


def fetch(cancer: str, tag: str, timeout: int = 300) -> str:
    """Download one matrix, cached. Returns the local path."""
    os.makedirs(CACHE, exist_ok=True)
    remote = (CLINICAL if tag == "clinical" else ASSAYS[tag]).format(c=cancer)
    local = os.path.join(CACHE, f"{cancer}_{tag}" + (".gz" if remote.endswith(".gz") else ".tsv"))
    if not os.path.exists(local):
        req = urllib.request.Request(HUB + remote, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(local, "wb") as w:
            w.write(r.read())
    return local


def _matrix(path: str) -> pd.DataFrame:
    """Xena matrices are features x samples; return samples x features."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0, low_memory=False)
    return df.T


def _binary_outcome(clin: pd.DataFrame) -> tuple:
    """A label that exists across solid tumours, with its name.

    Stage first: it is the closest thing to a disease axis the shared assays
    should track. `vital_status` is the fallback, and is noisier -- survival
    depends on treatment and comorbidity, not only on tumour biology.
    """
    if "pathologic_stage" in clin:
        s = clin["pathologic_stage"].astype(str)
        late = s.str.contains("III|IV", regex=True, na=False)
        early = s.str.contains("Stage I[AB]?$|Stage II[AB]?$", regex=True, na=False)
        ok = late | early
        if ok.sum() >= 100 and 0.1 < late[ok].mean() < 0.9:
            return late[ok].astype(float), ok, "stage (late vs early)"
    if "vital_status" in clin:
        v = clin["vital_status"].astype(str)
        ok = v.isin(["LIVING", "DECEASED"])
        if ok.sum() >= 100 and 0.1 < (v[ok] == "DECEASED").mean() < 0.9:
            return (v[ok] == "DECEASED").astype(float), ok, "vital status"
    raise ValueError("no usable balanced outcome in this clinical matrix")


def get_tcga_case(cancer: str, seed: int = 42, k: int = 3,
                  max_features: int = 1000) -> dict:
    """One TCGA cohort as three assay views with a clinical label.

    ``max_features`` keeps the highest-variance columns per view. It is a
    speed cap, not a modelling choice: at 1000 features and k=3 the smallest
    view still has p/k = 333, far above the point where the projection is
    forced to disjoint supports whatever the constraint says (measured: the
    structural question cannot be asked below about p/k = 5).
    """
    clin = pd.read_csv(fetch(cancer, "clinical"), sep="\t", index_col=0,
                       low_memory=False)
    y_all, ok, label = _binary_outcome(clin)
    views = {t: _matrix(fetch(cancer, t)) for t in ASSAYS}

    shared = set(clin.index[ok])
    for df in views.values():
        shared &= set(df.index)
    shared = sorted(shared)
    if len(shared) < 60:
        raise ValueError(f"{cancer}: only {len(shared)} samples shared across assays")

    mats = []
    for t in ("rna", "cnv", "rppa"):
        m = views[t].loc[shared]
        m = m.loc[:, m.notna().mean() >= 0.8]            # drop mostly-missing features
        m = m.fillna(m.median(numeric_only=True))
        m = m.loc[:, m.std() > 0]                        # a constant column divides by zero
        if m.shape[1] > max_features:
            keep = m.std().sort_values(ascending=False).index[:max_features]
            m = m[sorted(keep)]
        mats.append(m.to_numpy(dtype=float))

    y = y_all.loc[shared].to_numpy(dtype=float)
    return {"kind": f"tcga_{cancer.lower()}",
            "data": [torch.tensor(m).float() for m in mats],
            "outcome": torch.tensor(y).float(),
            "true_u": torch.zeros(len(shared), k),
            "true_v": [np.zeros((m.shape[1], k)) for m in mats],
            "shared_k": k, "needs_scaling": True, "is_classification": True,
            "outcome_label": label, "views": ["rna", "cnv", "rppa"]}
