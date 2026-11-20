#!/usr/bin/env python3
"""Generate the TCGA-BRCA ER-aligned loadings cache for the paper's Appendix A.

Writes ``paper/results_cache/brca_er_loadings.json``: for each view (mRNA, CNV,
RPPA) the `LEND` component whose scores correlate most with ER status, that
correlation, and the 15 largest |structure| loadings with feature names; plus
the ER+ prevalence and `loading_inference` stability/reliability summaries.

Descriptive fit on all 822 tumours (no held-out score is reported from it):
`LEND`, k = 3, package-default iterations, energy, mixing and w, signed basis,
2 NSA iterations, random_state = 42; each view standardized.

Usage: python scripts/paper_brca_loadings.py [--out DIR] [--smoke]
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
import torch

import pysimlr


def feature_names(cached, *, smoke=False):
    """Column names of the cached matrices, rebuilt with the selection in
    `scripts/run_deep_benchmark.get_multiomics_case` from the local TCGA
    files; caches verified names into gitignored ``data/tcga/brca_er_feature_names.json``
    so ``--smoke`` avoids the ~17s ``BRCA_rna.gz`` parse."""
    sidecar = os.path.join(REPO, "data", "tcga", "brca_er_feature_names.json")
    assays = ["rna", "cnv", "rppa"]
    if smoke:
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                names = json.load(f)
            if all(len(names.get(t, [])) == cached[t].shape[1] for t in assays):
                return names
        return {t: [f"{t}_{j}" for j in range(cached[t].shape[1])] for t in assays}

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
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    with open(sidecar, "w") as f:
        json.dump(names, f)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="Output directory (defaults to paper/results_cache, or temp dir under --smoke)")
    ap.add_argument("--n-subsamples", type=int, default=100)
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
        os.path.join(tempfile.gettempdir(), "pysimlr_smoke_brca_loadings")
        if a.smoke else os.path.join(REPO, "paper", "results_cache")
    )

    d = np.load(os.path.join(REPO, "data", "tcga", "brca_er_cached.npz"), allow_pickle=True)
    y = d["y"].astype(float)
    col_names = feature_names(d, smoke=bool(a.smoke))
    assays = ["rna", "cnv", "rppa"]
    xs = []
    for t in assays:
        x = d[t].astype(np.float64)
        sd = x.std(0, ddof=1)
        xs.append(torch.tensor((x - x.mean(0)) / np.where(sd > 0, sd, 1.0)).float())

    inf_kw = dict(
        positivity="either", nsa_iterations=2, random_state=42,
        fraction=0.5, threshold=0.8, expected_false=1.0,
        n_subsamples=n_sub, n_permutations=n_perm, n_jobs=n_jobs,
    )
    if iters is not None:
        inf_kw["iterations"] = iters
    inf = pysimlr.loading_inference("lend", xs, 3, **inf_kw)
    res = inf["result"]
    u = res["u"].double()
    rot = res["loading_rotation"].double()
    u_rot = (u @ rot).numpy()

    corrs = [abs(float(np.corrcoef(u_rot[:, c], y)[0, 1])) for c in range(3)]
    best_c = int(np.argmax(corrs))

    def _arr(x, nd=6):
        a_x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return np.round(a_x.astype(np.float64), nd).tolist() if a_x.dtype.kind == "f" else a_x.tolist()

    # Held-out 70/30 stratified split (Seed 42) for view-radar / coupling diagnostics
    from sklearn.model_selection import StratifiedShuffleSplit
    tr_idx, te_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42).split(np.zeros(len(y)), y))
    xs_tr = [torch.tensor((d[t][tr_idx].astype(np.float64) - d[t][tr_idx].mean(0)) /
                          np.where(d[t][tr_idx].std(0, ddof=1) > 0, d[t][tr_idx].std(0, ddof=1), 1.0)).float()
             for t in assays]
    xs_te = [torch.tensor((d[t][te_idx].astype(np.float64) - d[t][tr_idx].mean(0)) /
                          np.where(d[t][tr_idx].std(0, ddof=1) > 0, d[t][tr_idx].std(0, ddof=1), 1.0)).float()
             for t in assays]
    split_kw = dict(positivity="either", nsa_iterations=2, random_state=42)
    if iters is not None:
        split_kw["iterations"] = iters
    res_split = pysimlr.fit("lend", xs_tr, 3, **split_kw)
    pred_tr = pysimlr.predict(res_split, xs_tr)
    pred_te = pysimlr.predict(res_split, xs_te)

    inf_bundle = {
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
    }

    out = {
        "n": int(len(y)),
        "er_positive_fraction": float(y.mean()),
        "er_component": best_c + 1,
        "er_abs_corr": corrs[best_c],
        "convergence": res.get("convergence", {}),
        "latent_reliability": inf_bundle["latent_reliability"],
        "latent_reliability_multivariate": inf_bundle["latent_reliability_multivariate"],
        "cross_view_coupling": inf_bundle["cross_view_coupling"],
        "feature_agreement": inf_bundle["feature_agreement"],
        "null_max_abs": inf_bundle["null_max_abs"],
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
            "inf": inf_bundle,
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
        feat_agr = inf["feature_agreement_by_view"][i]

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
            "feature_agreement": [float(s) for s in feat_agr],
            "stability_selected_count": int(sel.sum()),
            "selected_features": [names[j] for j in range(len(names)) if sel[j]],
            "q": int(inf["q"][i]),
            "expected_false_selections": float(inf["expected_false_selections"][i]),
            "top": [
                [names[j], float(st_c[j]), float(lo_c[j]), float(hi_c[j]), bool(excl[j])]
                for j in top
            ]
        }

    try:
        commit = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cli_rel = " ".join([os.path.relpath(os.path.abspath(sys.argv[0]), REPO)] + sys.argv[1:])
    out["provenance"] = {
        "cli": cli_rel,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": f"{platform.system()} {platform.machine()}",
        "smoke": bool(a.smoke),
        "git_commit": commit,
        "pysimlr": pysimlr.__version__,
        "torch": torch.__version__,
        "parameters": res["parameters"]["used"]
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "brca_er_loadings.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"Wrote {out_path} (smoke={bool(a.smoke)})")
    for t in assays:
        vv = out["views"][t]
        print(t, "component", vv["component"], f"|r|={vv['abs_corr_with_er']:.2f}",
              [n for n, _, _, _, _ in vv["top"][:6]])


if __name__ == "__main__":
    main()
