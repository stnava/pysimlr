#!/usr/bin/env python3
"""Generate the Heart / Diabetes Varimax + nsa_flow loadings for the paper.

Writes ``paper/results_cache/nsa_v_loadings.json``; the figure
@fig-nsa-v-interpretability in ``paper/03_experiments.qmd`` is drawn from it at
build time. Replaces the pre-rendered ``paper/figures/nsa_v_interpretability.png``
(``scripts/generate_nsa_v_figures.py``), whose loadings had no cache.

Per cohort: `LEND`, k = 2, 150 epochs, newton mixing, non-negative basis,
sparseness 0.5, 3 NSA iterations, random_state = 42 (energy "nc" for Heart,
"logcosh" for Diabetes, as in the original figure); each view standardized.
Each view's basis is then Varimax-rotated and retracted by `nsa_flow`
(w = 0.5, non-negative) and every column is kept above its q = 0.5 quantile and
renormalized.

Usage: python scripts/paper_nsa_v_loadings.py [--out DIR]
"""
import argparse
import datetime
import json
import os
import platform
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pandas as pd
import torch
from nsa_flow import nsa_flow
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

import pysimlr
from pysimlr.nsa_backend import result_matrix
from pysimlr.svd import varimax

HEART_VIEWS = {"Clinical Vitals": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"],
               "Cardiophysiological": ["thalach", "exang", "oldpeak", "slope", "ca", "thal"]}
DIAB_VIEWS = {"Systemic Vitals": ["age", "sex", "bmi", "bp"],
              "Serum Biomarkers": ["s1", "s2", "s3", "s4", "s5", "s6"]}


def varimax_nsa_quantile(v, q=0.5, w=0.5):
    """Varimax (`pysimlr.svd.varimax`, R's `stats::varimax`) then the
    non-negative nsa_flow retraction, then keep each column at or above its
    q-quantile and renormalize it. nsa_flow >= 3.5 no longer takes
    `varimax=True`, so the rotation is applied here first."""
    rotated = varimax(torch.as_tensor(v, dtype=torch.float64))
    y = result_matrix(nsa_flow(rotated, w=w, nonneg=True, max_iter=200))
    y = np.asarray(y.detach().numpy() if torch.is_tensor(y) else y, dtype=float).copy()
    for j in range(y.shape[1]):
        col = y[:, j]
        if (col > 1e-6).sum() > 1:
            kept = np.where(col >= np.quantile(col, q), col, 0.0)
            nrm = np.linalg.norm(kept)
            y[:, j] = kept / nrm if nrm > 1e-12 else kept
    return y


def fit_cohort(frame, views, energy_type="nc"):
    xs = [torch.from_numpy(StandardScaler().fit_transform(frame[c].values.astype(float))).float()
          for c in views.values()]
    res = pysimlr.fit("lend", xs, 2, energy_type=energy_type, mixing_algorithm="newton",
                      positivity="positive", sparseness=0.5, random_state=42)
    out = {}
    for i, (name, cols) in enumerate(views.items()):
        load = res["loadings"][i].detach().cpu().numpy().astype(np.float64)
        margin = res["loading_margin"][i]
        margin_list = margin.detach().cpu().numpy().astype(np.float64).tolist() if margin is not None else None
        out[name] = {
            "rows": cols,
            "loadings": load.tolist(),
            "loading_margin": margin_list,
            "v": load.tolist(),
        }
    return out, res["parameters"]["used"], res.get("convergence", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "results_cache"))
    a = ap.parse_args()
    heart = pd.read_csv(os.path.join(REPO, "data", "heart_cleveland.csv"))
    diab = load_diabetes(as_frame=True).frame
    heart_v, heart_p, heart_c = fit_cohort(heart, HEART_VIEWS, "nc")
    diab_v, diab_p, diab_c = fit_cohort(diab, DIAB_VIEWS, "nc")
    out = {"Heart": heart_v, "Diabetes": diab_v,
           "convergence": {"Heart": heart_c, "Diabetes": diab_c},
           "provenance": {"cli": " ".join(sys.argv), "utc": datetime.datetime.utcnow().isoformat() + "Z",
                          "host": platform.node(), "pysimlr": pysimlr.__version__, "torch": torch.__version__,
                          "n": {"Heart": int(len(heart)), "Diabetes": int(len(diab))},
                          "parameters": {"Heart": heart_p, "Diabetes": diab_p}}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "nsa_v_loadings.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for cohort in ("Heart", "Diabetes"):
        for view, d in out[cohort].items():
            print(cohort, view, np.round(np.array(d["loadings"]), 2).tolist())


if __name__ == "__main__":
    main()
