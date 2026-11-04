#!/usr/bin/env python3
"""Does removing the global-thickness factor let non-negative SiMLR localise cortical regions?

ADNI-MCI, protocol B (scripts/paper_adni_ablations.py), simlr k=3, positivity="positive",
sparseness=0.85, graph topology, seed 42. The DiReCT cortical-thickness view (T1w_mean, 62 DKT
regions) is 58% one global factor; non-negative sparse orthogonal columns cannot all draw on it,
and the fit collapses to single-region columns.

Arms:
  current              protocol B as is
  global_removed       each DiReCT region regressed on the subject's mean regional thickness
                       (coefficients fitted on training rows, applied to held-out rows); other
                       views and the outcome untouched
  no_collapse          collapse_weight = 0
  global_removed+no_collapse
Per arm: the full fit, and 10 half-sample refits. Recorded per refit: regions per DiReCT column,
the most predictive DiReCT column (held-out CogPC1 R^2 alone) and its support; per arm the
held-out CogPC1 R^2 of all 9 MRI columns plus every view's mean (so global thickness is
available to every arm).

Rule (fixed before running): global-factor removal is adopted for the paper's regional
display if, in its arm, (1) the predictive DiReCT column has >= 5 regions in >= 8 of 10 refits,
(2) >= 3 regions are selected in >= 50% of refits, and (3) held-out CogPC1 R^2 (with the view
means) is >= the current arm's - 0.01.

Usage: python scripts/adni_global_factor_check.py [--out docs/audit/evidence/adni_global_factor_check.csv]
"""
import argparse, json, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
import numpy as np

ARMS = {"current": (False, {}), "global_removed": (True, {}),
        "no_collapse": (False, {"collapse_weight": 0.0}),
        "global_removed+no_collapse": (True, {"collapse_weight": 0.0})}


def adjust(col, remove_global):
    """Copy of the protocol-B data with the DiReCT view's global factor regressed out."""
    col = {**col, "x_tr": list(col["x_tr"]), "x_te": list(col["x_te"])}
    if not remove_global:
        return col
    a, c = col["x_tr"][2], col["x_te"][2]
    ga, gc = a.mean(1, keepdims=True), c.mean(1, keepdims=True)
    beta = (ga * a).sum(0) / (ga ** 2).sum()                      # per-region slope on training rows
    ra, rc = a - ga * beta, c - gc * beta
    mu, sd = ra.mean(0), ra.std(0, ddof=1); sd = np.where(sd > 0, sd, 1.0)
    col["x_tr"][2], col["x_te"][2] = (ra - mu) / sd, (rc - mu) / sd
    return col


def predictive_column(A, sub, col, V):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    r2 = [float(r2_score(col["pc1_te"], LinearRegression().fit(sub["x_tr"][2] @ V[:, [j]], sub["pc1_tr"])
                         .predict(col["x_te"][2] @ V[:, [j]]))) for j in range(V.shape[1])]
    j = int(np.argmax(r2)); return j, r2[j]


def run(task):
    arm, s = task
    warnings.simplefilter("ignore")
    import torch; torch.set_num_threads(1)
    import paper_adni_ablations as A
    remove, extra = ARMS[arm]
    col = adjust(A.protocol_b(42), remove)
    if s < 0:
        sub = col
    else:
        n = col["x_tr"][0].shape[0]; idx = np.random.default_rng(1000 + s).choice(n, n // 2, replace=False)
        sub = {**col, "x_tr": [x[idx] for x in col["x_tr"]], "y_tr": col["y_tr"][idx], "pc1_tr": col["pc1_tr"][idx],
               "rowmean_tr": col["rowmean_tr"][idx]}
    res, _ = A.fit_adni(sub, "simlr", positivity="positive", sparseness=0.85, random_state=42, **extra)
    vs = A.mri_basis(res); V = vs[2]
    j, r2j = predictive_column(A, sub, col, V)
    row = {"arm": arm, "refit": s, "nnz": (V > 1e-6).sum(0).tolist(), "pred_col": j, "pred_col_r2": r2j,
           "support": (V[:, j] > 1e-6).astype(int).tolist(), "stop": res["convergence"]["stop_reason"]}
    if s < 0:
        ev = A.evaluate(col, vs); row.update(cog_pc1_r2=ev["cog_pc1_r2"], cog_pc1_r2_with_means=ev["cog_pc1_r2_with_rowmean"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "docs/audit/evidence/adni_global_factor_check.csv"))
    a = ap.parse_args()
    import pandas as pd
    import paper_adni_ablations as A
    names = A.protocol_b(42)["names"][2]
    tasks = [(arm, s) for arm in ARMS for s in [-1] + list(range(10))]
    with ProcessPoolExecutor(10) as ex:
        rows = list(ex.map(run, tasks))
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    cur = df[(df.arm == "current") & (df.refit < 0)].iloc[0]
    for arm in ARMS:
        d = df[(df.arm == arm) & (df.refit >= 0)]; f = df[(df.arm == arm) & (df.refit < 0)].iloc[0]
        nn = np.array([r[r_j] for r, r_j in zip(d.nnz, d.pred_col)])
        F = np.mean(np.stack(d.support.to_list()), 0)
        top = [(names[i], round(float(F[i]), 2)) for i in np.argsort(-F)[:8] if F[i] > 0]
        c1 = int((nn >= 5).sum()) >= 8; c2 = int((F >= 0.5).sum()) >= 3
        c3 = f.cog_pc1_r2_with_means >= cur.cog_pc1_r2_with_means - 0.01
        print(f"\n== {arm}: full-fit DiReCT nnz {f.nnz}, held-out CogPC1 R2 {f.cog_pc1_r2:.3f} "
              f"(with view means {f.cog_pc1_r2_with_means:.3f})")
        print(f"   predictive column regions per refit {nn.tolist()}; median R2 {d.pred_col_r2.median():.3f}")
        print(f"   regions selected in >= 50% of refits: {int((F >= 0.5).sum())}; top {top}")
        print(f"   rule: (1) {c1} (2) {c2} (3) {c3} -> {'PASS' if (c1 and c2 and c3) else 'fail'}")


if __name__ == "__main__":
    main()
