#!/usr/bin/env python3
"""Generate the TCGA-BRCA ER-aligned loadings cache for the paper's Appendix A.

Writes ``paper/results_cache/brca_er_loadings.json``: for each view (mRNA, CNV,
RPPA) the `LEND` component whose scores correlate most with ER status, that
correlation, and the 15 largest |V| loadings with feature names; plus the ER+
prevalence. Replaces the pre-rendered ``paper/figures/fig_brca_trimodal_waterfall.png``
(``scripts/generate_brca_waterfall.py``), whose numbers had no cache.

Descriptive fit on all 822 tumours (no held-out score is reported from it):
`LEND`, k = 3, 45 epochs, package-default energy, mixing and w, signed basis,
2 NSA iterations, random_state = 42; each view standardized.

Usage: python scripts/paper_brca_loadings.py [--out DIR]
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
import torch

import pysimlr


def feature_names(cached):
    """Column names of the cached matrices, rebuilt with the selection in
    `scripts/run_deep_benchmark.get_multiomics_case` from the local TCGA
    files; the cache stores only the values. Raises unless the rebuilt
    matrices equal the cached ones, so the names provably match the columns."""
    import pandas as pd
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from tcga_loader import fetch, _matrix, ASSAYS
    clin = pd.read_csv(fetch("BRCA", "clinical"), sep="\t", index_col=0, low_memory=False)
    s = clin["breast_carcinoma_estrogen_receptor_status"].astype(str)
    ok = s.isin(["Positive", "Negative"])
    views = {t: _matrix(fetch("BRCA", t)) for t in ASSAYS}
    shared = sorted(set(clin.index[ok]) & set(views["rna"].index) & set(views["cnv"].index)
                    & set(views["rppa"].index))
    names = {}
    for t, max_f in [("rna", 200), ("cnv", 150), ("rppa", 150)]:
        m = views[t].loc[shared]
        m = m.loc[:, m.notna().mean() >= 0.98].fillna(m.median(numeric_only=True))
        m = m.loc[:, m.std() > 0]
        if m.shape[1] > max_f:
            m = m[sorted(m.std().sort_values(ascending=False).index[:max_f])]
        if not np.allclose(m.to_numpy(dtype=np.float32), cached[t], equal_nan=True):
            raise RuntimeError(f"rebuilt {t} matrix does not match the cache; names would be wrong")
        names[t] = [str(c) for c in m.columns]
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "results_cache"))
    a = ap.parse_args()
    d = np.load(os.path.join(REPO, "data", "tcga", "brca_er_cached.npz"), allow_pickle=True)
    y = d["y"].astype(float)
    col_names = feature_names(d)
    assays = ["rna", "cnv", "rppa"]
    xs = []
    for t in assays:
        x = d[t].astype(np.float64)
        sd = x.std(0, ddof=1)
        xs.append(torch.tensor((x - x.mean(0)) / np.where(sd > 0, sd, 1.0)).float())
    res = pysimlr.fit("lend", xs, 3, iterations=45,
                      positivity="either", nsa_iterations=2, random_state=42)
    out = {"n": int(len(y)), "er_positive_fraction": float(y.mean()), "views": {}}
    for i, t in enumerate(assays):
        v = res["v"][i].detach().numpy().astype(np.float64)
        z = xs[i].numpy() @ v
        corrs = [abs(float(np.corrcoef(z[:, c], y)[0, 1])) for c in range(v.shape[1])]
        c = int(np.argmax(corrs))
        w = np.abs(v[:, c])
        top = np.argsort(-w)[:15]
        names = col_names[t]
        out["views"][t] = {"component": c + 1, "abs_corr_with_er": corrs[c], "p": int(v.shape[0]),
                           "top": [[names[j], float(w[j])] for j in top]}
    out["provenance"] = {"cli": " ".join(sys.argv), "utc": datetime.datetime.utcnow().isoformat() + "Z",
                         "host": platform.node(), "pysimlr": pysimlr.__version__, "torch": torch.__version__,
                         "parameters": res["parameters"]["used"]}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "brca_er_loadings.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for t in assays:
        vv = out["views"][t]
        print(t, "component", vv["component"], f"|r|={vv['abs_corr_with_er']:.2f}", [n for n, _ in vv["top"][:6]])


if __name__ == "__main__":
    main()
