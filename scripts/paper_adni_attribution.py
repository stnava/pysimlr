#!/usr/bin/env python3
"""ADNI-MCI attribution for the manuscript (Figure 3.9): what drives held-out CogPC1, and how stable it is.

A thin caller of `pysimlr.component_attribution`. Protocol B (scripts/paper_adni_ablations.py):
simlr, k=3, positivity="positive", sparseness=0.85, graph topology, seed 42. Views: label volume
(160 hierarchical labels), volume / surface area of the same labels, DiReCT cortical thickness
(62 DKT regions). The 20 half-sample refits use the rows `default_rng(1000 + s).choice(n, n // 2)`,
s = 0..19, with and without the collapse penalty.

Writes paper/results_cache/adni_attribution.json with two things:
- `attribution`, the `component_attribution` result, which the figure draws;
- the keys `paper_numbers` reads (`full_r2`, `components`, `dkt_*`, `consensus_vs_global`,
  `halves`, ...). They are unchanged from the pre-refactor script, and `component_attribution`
  reproduces them exactly.

Usage: python scripts/paper_adni_attribution.py [--out paper/results_cache/adni_attribution.json]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import warnings

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
VIEWS = ["volume", "vol_area", "direct_thickness"]


def run():
    import paper_adni_ablations as A

    import pysimlr
    col = A.protocol_b(42)
    res, _ = A.fit_adni(col, "simlr", positivity="positive", sparseness=0.85, random_state=42)
    n = col["x_tr"][0].shape[0]
    rows = [np.random.default_rng(1000 + s).choice(n, n // 2, replace=False) for s in range(20)]
    # The same refits with the collapse penalty off: on DiReCT thickness (58% one global factor)
    # any two non-negative region sums correlate above the 0.3 threshold, so the penalty forces
    # single-region columns (docs/audit/evidence/adni_global_factor_check.csv).
    return pysimlr.component_attribution(
        res, list(col["x_tr"]) + [col["y_tr"]], list(col["x_te"]) + [None], col["pc1_tr"], col["pc1_te"],
        views=(0, 1, 2), view_names=VIEWS + ["cognition"], feature_names=list(col["names"]) + [None],
        refit_rows=rows, variants={"collapse_weight=0": {"collapse_weight": 0.0}})


def legacy(att):
    """The pre-refactor keys, read by paper/paper_numbers.py."""
    s = att["spectra"]["2"]
    pick = lambda h: {m: {kk: h[m][kk] for kk in ("support", "r2", "nnz")} for m in ("1", "2")}
    return {"full_r2": att["full_r2"],
            "components": [{"view": VIEWS[c["view"]], "component": c["component"], "nnz": c["nnz"], "alone": c["alone"],
                            "drop": c["unique"]} for c in att["components"]],
            "dkt_eigen_share": s["eigen_share"], "dkt_mean_corr": s["mean_corr"], "consensus_vs_global": s["consensus_vs_mean"],
            "dkt_loadings_counts": s["hard_assignment"], "names": {m: att["feature_names"][m] for m in ("1", "2")},
            "full_support": {m: att["full_support"][m] for m in ("1", "2")},
            "halves": [pick(h) for h in att["refits"]["default"]],
            "halves_no_collapse": [pick(h) for h in att["refits"]["collapse_weight=0"]]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "results_cache", "adni_attribution.json"))
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    import pysimlr
    att = run()
    out = dict(legacy(att), attribution=att)
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    out["provenance"] = {"cli": " ".join(sys.argv), "pysimlr": pysimlr.__version__, "git_commit": commit,
                         "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print("full R2", round(att["full_r2"], 3), "| half-sample predictive R2 (DKT):",
          np.round([h["2"]["r2"] for h in att["refits"]["default"]], 3))
