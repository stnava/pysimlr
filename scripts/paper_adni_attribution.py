#!/usr/bin/env python3
"""ADNI-MCI attribution for the manuscript: what drives held-out CogPC1, and how stable it is.

Protocol B (scripts/paper_adni_ablations.py); simlr, k=3, positivity="positive",
sparseness=0.85, graph topology, seed 42. Views: label volume (160 hierarchical labels),
volume / surface area of the same labels, DiReCT cortical thickness (62 DKT regions).
Writes paper/results_cache/adni_attribution.json:
  components   per view and component: features, held-out CogPC1 R^2 alone, unique contribution
  dkt_eigen_share, dkt_mean_corr, consensus_vs_global, dkt_loadings_counts
  halves       20 half-sample refits: per view, the most predictive column's support and R^2
  halves_no_collapse  the same refits with collapse_weight = 0
  provenance   pysimlr version, git commit, UTC time

Usage: python scripts/paper_adni_attribution.py [--out paper/results_cache/adni_attribution.json]
"""
import argparse, datetime, json, os, subprocess, sys, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
VIEWS = ["volume", "vol_area", "direct_thickness"]


def r2_alone(A, Ztr, ytr, Zte, yte):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    return float(r2_score(yte, LinearRegression().fit(Ztr, ytr).predict(Zte)))


def half(s, collapse_weight=None):
    warnings.simplefilter("ignore")
    import torch; torch.set_num_threads(1)
    import paper_adni_ablations as A
    col = A.protocol_b(42); n = col["x_tr"][0].shape[0]
    idx = np.random.default_rng(1000 + s).choice(n, n // 2, replace=False)
    sub = {**col, "x_tr": [x[idx] for x in col["x_tr"]], "y_tr": col["y_tr"][idx], "pc1_tr": col["pc1_tr"][idx]}
    extra = {} if collapse_weight is None else {"collapse_weight": collapse_weight}
    res, _ = A.fit_adni(sub, "simlr", positivity="positive", sparseness=0.85, random_state=42, **extra)
    vs = A.mri_basis(res); out = {}
    for m in (1, 2):
        V = vs[m]
        r2 = [r2_alone(A, sub["x_tr"][m] @ V[:, [j]], sub["pc1_tr"], col["x_te"][m] @ V[:, [j]], col["pc1_te"]) for j in range(3)]
        j = int(np.argmax(r2))
        out[m] = {"support": (V[:, j] > 1e-6).astype(int).tolist(), "r2": r2[j], "nnz": (V > 1e-6).sum(0).tolist()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "results_cache", "adni_attribution.json"))
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    import pysimlr
    import paper_adni_ablations as A
    col = A.protocol_b(42)
    res, _ = A.fit_adni(col, "simlr", positivity="positive", sparseness=0.85, random_state=42)
    vs = A.mri_basis(res)
    Ztr = np.hstack([x @ v for x, v in zip(col["x_tr"], vs)]); Zte = np.hstack([x @ v for x, v in zip(col["x_te"], vs)])
    full = r2_alone(A, Ztr, col["pc1_tr"], Zte, col["pc1_te"])
    comp = []
    for c in range(9):
        m, j = divmod(c, 3); keep = [i for i in range(9) if i != c]
        comp.append({"view": VIEWS[m], "component": j + 1, "nnz": int((vs[m][:, j] > 1e-6).sum()),
                     "alone": r2_alone(A, Ztr[:, [c]], col["pc1_tr"], Zte[:, [c]], col["pc1_te"]),
                     "drop": full - r2_alone(A, Ztr[:, keep], col["pc1_tr"], Zte[:, keep], col["pc1_te"])})
    X = col["x_tr"][2]; gm = X.mean(1)
    R = np.asarray(res["loading_rotation"], float); U = np.asarray(res["u"], float) @ R
    ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
    C = np.corrcoef(X.T); offd = float((C.sum() - 62) / (62 * 61))
    loads = np.asarray(res["loadings"][2], float)
    with ProcessPoolExecutor(10) as ex:
        halves = list(ex.map(half, range(20)))
        # The same refits with the collapse penalty off: on DiReCT thickness (58% one global
        # factor) any two non-negative region sums correlate above the 0.3 threshold, so the
        # penalty forces single-region columns (docs/audit/evidence/adni_global_factor_check.csv).
        halves_nc = list(ex.map(half, range(20), [0.0] * 20))
    out = {"full_r2": full, "components": comp,
           "dkt_eigen_share": (ev / ev.sum()).tolist(), "dkt_mean_corr": offd,
           "consensus_vs_global": [float(np.corrcoef(U[:, j], gm)[0, 1]) for j in range(3)],
           "dkt_loadings_counts": (loads > 0).sum(0).tolist(),
           "names": {str(m): col["names"][m] for m in (1, 2)},
           "full_support": {str(m): [int(x) for x in (vs[m][:, int(np.argmax([c["alone"] for c in comp[3 * m:3 * m + 3]]))] > 1e-6)] for m in (1, 2)},
           "halves": halves, "halves_no_collapse": halves_nc}
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    out["provenance"] = {"cli": " ".join(sys.argv), "pysimlr": pysimlr.__version__, "git_commit": commit,
                         "written_utc": datetime.datetime.utcnow().isoformat() + "Z"}
    json.dump(out, open(args.out, "w"))
    print("full R2", round(full, 3), "| half-sample predictive R2 (DKT):", np.round([h[2]["r2"] for h in halves], 3))
