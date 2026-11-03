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

    inf = pysimlr.loading_inference(
        "lend", xs, 3,
        positivity="either", nsa_iterations=2, random_state=42,
        fraction=0.5, threshold=0.8, expected_false=1.0,
        n_permutations=20, n_jobs=8
    )
    res = inf["result"]
    u = res["u"].double()
    rot = res["loading_rotation"].double()
    u_rot = (u @ rot).numpy()

    corrs = [abs(float(np.corrcoef(u_rot[:, c], y)[0, 1])) for c in range(3)]
    best_c = int(np.argmax(corrs))

    def _arr(x, nd=6):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return np.round(a.astype(np.float64), nd).tolist() if a.dtype.kind == "f" else a.tolist()

    # Held-out 70/30 stratified split (Seed 42) for view-radar / coupling diagnostics
    from sklearn.model_selection import StratifiedShuffleSplit
    tr_idx, te_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42).split(np.zeros(len(y)), y))
    xs_tr = [torch.tensor((d[t][tr_idx].astype(np.float64) - d[t][tr_idx].mean(0)) /
                          np.where(d[t][tr_idx].std(0, ddof=1) > 0, d[t][tr_idx].std(0, ddof=1), 1.0)).float()
             for t in assays]
    xs_te = [torch.tensor((d[t][te_idx].astype(np.float64) - d[t][tr_idx].mean(0)) /
                          np.where(d[t][tr_idx].std(0, ddof=1) > 0, d[t][tr_idx].std(0, ddof=1), 1.0)).float()
             for t in assays]
    res_split = pysimlr.fit("lend", xs_tr, 3, positivity="either", nsa_iterations=2, random_state=42)
    pred_tr = pysimlr.predict(res_split, xs_tr)
    pred_te = pysimlr.predict(res_split, xs_te)

    out = {
        "n": int(len(y)),
        "er_positive_fraction": float(y.mean()),
        "er_component": best_c + 1,
        "er_abs_corr": corrs[best_c],
        "convergence": res.get("convergence", {}),
        "null_max_abs": float(inf["null_max_abs"]) if inf.get("null_max_abs") is not None else None,
        "plot_bundle": {
            "feature_names": [col_names[t] for t in assays],
            "res": {
                "method": res.get("method", "lend"),
                "convergence": res.get("convergence", {}),
                "loading_rotation": _arr(res["loading_rotation"]),
                "u": _arr(res["u"]),
                "latents": [_arr(z) for z in res["latents"]],
                "structure": [_arr(c) for c in res["structure"]],
                "loadings": [_arr(l) for l in res["loadings"]],
                "v": [_arr(v) for v in res["v"]],
                "effective_rank": _arr(res["effective_rank"]),
            },
            "inf": {
                "ci_low": [_arr(c) for c in inf["ci_low"]],
                "ci_high": [_arr(c) for c in inf["ci_high"]],
                "selected": [_arr(s) for s in inf["selected"]],
                "selection_frequency": [_arr(f) for f in inf["selection_frequency"]],
                "component_stability": inf["component_stability"],
                "null_max_abs": float(inf["null_max_abs"]) if inf.get("null_max_abs") is not None else None,
            },
            "split": {
                "res": {
                    "method": res_split.get("method", "lend"),
                    "convergence": res_split.get("convergence", {}),
                    "loading_rotation": _arr(res_split["loading_rotation"]),
                    "u": _arr(res_split["u"]),
                    "latents": [_arr(z) for z in res_split["latents"]],
                    "structure": [_arr(c) for c in res_split["structure"]],
                    "loadings": [_arr(l) for l in res_split["loadings"]],
                    "v": [_arr(v) for v in res_split["v"]],
                    "effective_rank": _arr(res_split["effective_rank"]),
                },
                "train_r2": [float(v) for v in pred_tr["r2"]],
                "pred": {
                    "u": _arr(pred_te["u"]),
                    "latents": [_arr(z) for z in pred_te["latents"]],
                    "r2": [float(v) for v in pred_te["r2"]],
                },
            },
        },
        "views": {}
    }

    for i, t in enumerate(assays):
        st = inf["structure"][i].detach().cpu().numpy().astype(np.float64)
        ci_lo = inf["ci_low"][i].detach().cpu().numpy().astype(np.float64)
        ci_hi = inf["ci_high"][i].detach().cpu().numpy().astype(np.float64)
        sel = inf["selected"][i].detach().cpu().numpy()
        freq = inf["selection_frequency"][i].detach().cpu().numpy()
        stab = inf["component_stability"][i]

        st_c = st[:, best_c]
        lo_c = ci_lo[:, best_c]
        hi_c = ci_hi[:, best_c]
        excl = (lo_c > 0) | (hi_c < 0)

        names = col_names[t]
        top = np.argsort(-np.abs(st_c))[:15]

        out["views"][t] = {
            "component": best_c + 1,
            "abs_corr_with_er": corrs[best_c],
            "p": int(st.shape[0]),
            "component_stability": [float(s) for s in stab],
            "stability_selected_count": int(sel.sum()),
            "selected_features": [names[j] for j in range(len(names)) if sel[j]],
            "q": int(inf["q"][i]),
            "expected_false_selections": float(inf["expected_false_selections"][i]),
            "top": [
                [names[j], float(st_c[j]), float(lo_c[j]), float(hi_c[j]), bool(excl[j])]
                for j in top
            ]
        }

    out["provenance"] = {
        "cli": " ".join(sys.argv),
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "host": platform.node(),
        "pysimlr": pysimlr.__version__,
        "torch": torch.__version__,
        "parameters": res["parameters"]["used"]
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "brca_er_loadings.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for t in assays:
        vv = out["views"][t]
        print(t, "component", vv["component"], f"|r|={vv['abs_corr_with_er']:.2f}",
              [n for n, _, _, _, _ in vv["top"][:6]])


if __name__ == "__main__":
    main()
