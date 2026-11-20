#!/usr/bin/env python3
"""Generate the Heart / Diabetes LEND loadings and latent reliability cache for the paper.

Writes ``paper/results_cache/nsa_v_loadings.json``; @fig-diabetes-top-features,
@fig-heart-audit, and @fig-nsa-v-interpretability in ``paper/03_experiments.qmd``
are drawn from it at build time.

Per cohort: `LEND`, k = 2, package default iterations, newton mixing,
positivity="positive", sparseness=0.5, energy_type="nc", random_state=42,
each view standardized.

Usage: python scripts/paper_nsa_v_loadings.py [--out DIR] [--smoke]
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

import pysimlr

HEART_VIEWS = {"Clinical Vitals": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"],
               "Cardiophysiological": ["thalach", "exang", "oldpeak", "slope", "ca", "thal"]}
DIAB_VIEWS = {"Systemic Vitals": ["age", "sex", "bmi", "bp"],
              "Serum Biomarkers": ["s1", "s2", "s3", "s4", "s5", "s6"]}


def _arr(x, nd=6):
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.round(a.astype(np.float64), nd).tolist() if a.dtype.kind == "f" else a.tolist()


def fit_cohort(frame, views, energy_type="nc", *, n_subsamples=50, n_permutations=20,
               iterations=None, n_jobs=8):
    xs = [torch.from_numpy(StandardScaler().fit_transform(frame[c].values.astype(float))).float()
          for c in views.values()]
    kw = dict(
        n_subsamples=n_subsamples, n_permutations=n_permutations,
        energy_type=energy_type, mixing_algorithm="newton",
        positivity="positive", sparseness=0.5, random_state=42, n_jobs=n_jobs,
    )
    if iterations is not None:
        kw["iterations"] = iterations
    inf = pysimlr.loading_inference("lend", xs, 2, **kw)
    res = inf["result"]
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
    bundle = {
        "view_names": list(views.keys()),
        "feature_names": list(views.values()),
        "res": {
            "method": res.get("method", "lend"),
            "convergence": res.get("convergence", {}),
            "loading_rotation": _arr(res["loading_rotation"]),
            "u": _arr(res["u"]),
            "latents": [_arr(z) for z in res["latents"]],
            "structure": [_arr(c) for c in res["structure"]],
            "loadings": [_arr(l) for l in res["loadings"]],
            "loading_margin": [_arr(m) if m is not None else None for m in res["loading_margin"]],
            "v": [_arr(v) for v in res["v"]],
            "effective_rank": _arr(res["effective_rank"]),
        },
        "inf": {
            "ci_low": [_arr(c) for c in inf["ci_low"]],
            "ci_high": [_arr(c) for c in inf["ci_high"]],
            "selected": [_arr(s) for s in inf["selected"]],
            "selection_frequency": [_arr(f) for f in inf["selection_frequency"]],
            "latent_reliability": _arr(inf["latent_reliability"]),
            "latent_reliability_averaged": _arr(inf["latent_reliability_averaged"]),
            "latent_reliability_multivariate": float(inf["latent_reliability_multivariate"]),
            "cross_view_coupling": (_arr(inf["cross_view_coupling"])
                                    if inf.get("cross_view_coupling") is not None else None),
            "feature_agreement": _arr(inf["feature_agreement"]),
            "feature_agreement_by_view": [_arr(v) for v in inf["feature_agreement_by_view"]],
            "null_max_abs": float(inf["null_max_abs"]) if inf.get("null_max_abs") is not None else None,
        },
    }
    return out, res["parameters"]["used"], res.get("convergence", {}), bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="Output directory (defaults to paper/results_cache, or temp dir under --smoke)")
    ap.add_argument("--n-subsamples", type=int, default=50)
    ap.add_argument("--n-permutations", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="Fast <10s smoke check (counts=2, non-destructive default out)")
    a = ap.parse_args()

    n_sub = 2 if a.smoke else a.n_subsamples
    n_perm = 2 if a.smoke else a.n_permutations
    iters = 2 if a.smoke else a.iterations
    n_jobs = 1 if a.smoke else a.n_jobs
    out_dir = a.out or (
        os.path.join(tempfile.gettempdir(), "pysimlr_smoke_nsa_v_loadings")
        if a.smoke else os.path.join(REPO, "paper", "results_cache")
    )

    heart = pd.read_csv(os.path.join(REPO, "data", "heart_cleveland.csv"))
    diab = load_diabetes(as_frame=True).frame
    heart_v, heart_p, heart_c, heart_b = fit_cohort(
        heart, HEART_VIEWS, "nc",
        n_subsamples=n_sub, n_permutations=n_perm, iterations=iters, n_jobs=n_jobs,
    )
    diab_v, diab_p, diab_c, diab_b = fit_cohort(
        diab, DIAB_VIEWS, "nc",
        n_subsamples=n_sub, n_permutations=n_perm, iterations=iters, n_jobs=n_jobs,
    )
    try:
        commit = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cli_rel = " ".join([os.path.relpath(os.path.abspath(sys.argv[0]), REPO)] + sys.argv[1:])
    out = {"Heart": heart_v, "Diabetes": diab_v,
           "plot_bundles": {"Heart": heart_b, "Diabetes": diab_b},
           "convergence": {"Heart": heart_c, "Diabetes": diab_c},
           "provenance": {"cli": cli_rel,
                          "utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                          "platform": f"{platform.system()} {platform.machine()}",
                          "smoke": bool(a.smoke),
                          "git_commit": commit,
                          "pysimlr": pysimlr.__version__, "torch": torch.__version__,
                          "n": {"Heart": int(len(heart)), "Diabetes": int(len(diab))},
                          "parameters": {"Heart": heart_p, "Diabetes": diab_p}}}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nsa_v_loadings.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"Wrote {out_path} (smoke={bool(a.smoke)})")
    for cohort in ("Heart", "Diabetes"):
        for view, d in out[cohort].items():
            print(cohort, view, np.round(np.array(d["loadings"]), 2).tolist())


if __name__ == "__main__":
    main()
