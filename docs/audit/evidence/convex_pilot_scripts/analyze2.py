"""Round-2 analysis: per-cell summaries, per-seed paired differences vs conical, ship rule.

usage: python analyze2.py TAG [TAG ...]   (reads pilot2_TAG.csv, bases/TAG_*.npz)
"""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
KEYS = ["tag", "dataset", "sparseness", "method", "energy", "gauge"]


def jaccard(tag, ds, arm, seeds):
    vals = []
    for a, b in itertools.combinations(seeds, 2):
        fa, fb = f"{HERE}/bases/{tag}_{ds}_{arm}_{a}.npz", f"{HERE}/bases/{tag}_{ds}_{arm}_{b}.npz"
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A, B = np.load(fa), np.load(fb)
        for key in A.files:
            sa, sb = np.abs(A[key]) > 1e-10, np.abs(B[key]) > 1e-10
            k = sa.shape[1]
            J = np.array([[(sa[:, i] & sb[:, j]).sum() / max(1, (sa[:, i] | sb[:, j]).sum()) for j in range(k)] for i in range(k)])
            r, c = linear_sum_assignment(-J)
            vals.append(J[r, c].mean())
    return float(np.mean(vals)) if vals else float("nan")


df = pd.concat([pd.read_csv(f"{HERE}/pilot2_{t}.csv") for t in sys.argv[1:]], ignore_index=True)
if "error" in df and df["error"].notna().any():
    print("ERRORS:\n", df[df["error"].notna()][KEYS + ["arm", "seed", "error"]].to_string(index=False))
ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()] if "error" in df else df
summ, paired, seeds_out = [], [], []
errs = df[df["error"].notna()] if "error" in df else df.iloc[0:0]
def n_err(key, arm):
    if errs.empty:
        return 0
    m = np.logical_and.reduce([errs[k].astype(str) == str(v) for k, v in zip(KEYS, key)]) & (errs.arm == arm)
    return int(m.sum())
for key, g in ok.groupby(KEYS, dropna=False):
    seeds = sorted(g.seed.unique())
    for arm, h in g.groupby("arm"):
        summ.append(dict(zip(KEYS, key), arm=arm, n=len(h), errors=n_err(key, arm), rf=h.rf_XV_r2.mean(), ridge=h.ridge_XV_r2.mean(),
                         xview=h.cross_view_heldout.mean(), support=h.support_mean.mean(), le2=h.le2_frac.mean(),
                         jaccard=jaccard(key[0], key[1], arm, seeds), secs=h.secs.mean(),
                         nan=int(h.nan.sum()), dead=int(h.dead_cols.sum()), neg=int(h.neg_entries.sum()),
                         colsum_err=h.colsum_err.max(), nonconv=int((h.stop != "converged").sum())))
    if "positive" not in set(g.arm):
        continue
    base = g[g.arm == "positive"].set_index("seed")
    for arm in sorted(set(g.arm) - {"positive"}):
        h = g[g.arm == arm].set_index("seed")
        common = base.index.intersection(h.index)
        for m in ("rf_XV_r2", "ridge_XV_r2", "cross_view_heldout"):
            d = h.loc[common, m] - base.loc[common, m]
            paired.append(dict(zip(KEYS, key), arm=arm, metric=m, mean=d.mean(), sd=d.std(), min=d.min(),
                               wins=int((d > 0).sum()), n=len(d), per_seed=" ".join(f"{x:+.3f}" for x in d)))
S = pd.DataFrame(summ); P = pd.DataFrame(paired)
print("== cells\n" + S.round(3).to_string(index=False))
print("\n== paired (arm - conical), per seed in seed order\n" + P.round(4).to_string(index=False))
# ship rule (pre-set): per cell, convex vs conical
rows = []
for key, g in S[S.arm.isin(["positive", "convex"])].groupby(KEYS, dropna=False):
    if set(g.arm) != {"positive", "convex"}:
        continue
    c, p = g[g.arm == "convex"].iloc[0], g[g.arm == "positive"].iloc[0]
    pr = P[(P.arm == "convex") & (P.metric == "rf_XV_r2")]
    pc = P[(P.arm == "convex") & (P.metric == "cross_view_heldout")]
    sel = lambda q: q[np.logical_and.reduce([q[k].astype(str) == str(v) for k, v in zip(KEYS, key)])].iloc[0]
    r_, x_ = sel(pr), sel(pc)
    checks = {"R2_mean>=-0.01": r_["mean"] >= -0.01, "no_seed<-0.05": r_["min"] >= -0.05,
              "xview>=-0.02": x_["mean"] >= -0.02, "jacc_drop<=0.05": (p.jaccard - c.jaccard) <= 0.05,
              "clean": (c.nan == 0) and (c.dead == 0) and (c.colsum_err <= 1e-6) and (c.errors == 0),
              "runtime<=2x": c.secs <= 2 * p.secs}
    rows.append(dict(zip(KEYS, key), **checks, PASS=all(checks.values())))
R = pd.DataFrame(rows)
print("\n== ship rule (convex vs conical)\n" + R.to_string(index=False))
tag = "_".join(sys.argv[1:])[:80]
S.to_csv(f"{HERE}/cells_{tag}.csv", index=False); P.to_csv(f"{HERE}/paired_{tag}.csv", index=False)
R.to_csv(f"{HERE}/shiprule_{tag}.csv", index=False)
