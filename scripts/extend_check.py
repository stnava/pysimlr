#!/usr/bin/env python3
"""Anchored extension against an oracle joint refit (decision rule in docs/tracking.md, D19).

Planted design: z (n = 360, k = 3, non-negative); views A (30), B (24) and C (18), each a
block-sparse non-negative mixing of z plus noise; 300 training / 60 held-out subjects.
Arms, per method (simlr, ned) and seed (0-4):
  anchored  fit on A, B; `pysimlr.extend` with C; held-out u from A, B, C
  oracle    fit on A, B, C together; held-out u
  base      fit on A, B; held-out u from A, B (what the extension adds to)
  c_only    the anchored model's held-out u from C alone
  reembed   `extend(how="reembed")`; latent R^2 of its training u (in-sample: its blocks are
            built from the training scores, so it has no held-out predictor)
Metric: latent R^2 of z from u (scripts/optimizer_benchmark_synthetic.latent_r2).
Rule (fixed before running): anchored is accepted as the default if its held-out latent R^2
is within 0.05 of the oracle on average and no seed is worse by more than 0.10.

Usage: python scripts/extend_check.py [--out docs/audit/evidence/extend_check.csv]
"""
import argparse, os, sys, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np, pandas as pd
import pysimlr
from optimizer_benchmark_synthetic import latent_r2


def study(seed, n=360, k=3):
    g = np.random.default_rng(seed); z = np.abs(g.normal(size=(n, k)))
    def view(p):
        W = np.zeros((k, p)); blk = p // k
        for j in range(k):
            W[j, j * blk:(j + 1) * blk] = 0.5 + g.random(blk)
        return (z @ W + 0.5 * g.normal(size=(n, p))).astype(np.float32)
    return z, view(30), view(24), view(18)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(REPO, "docs/audit/evidence/extend_check.csv"))
    a = ap.parse_args(); warnings.simplefilter("ignore"); rows = []
    tr, te = slice(0, 300), slice(300, None)
    for method in ("simlr", "ned"):
        kw = dict(positivity="positive") if method == "simlr" else {}
        for seed in range(5):
            z, A, B, C = study(seed)
            res = pysimlr.fit(method, [A[tr], B[tr]], 3, random_state=seed, **kw)
            ext = pysimlr.extend(res, [A[tr], B[tr]], {"C": C[tr]}, random_state=seed)
            ora = pysimlr.fit(method, [A[tr], B[tr], C[tr]], 3, random_state=seed, **kw)
            rem = pysimlr.extend(res, [A[tr], B[tr]], {"C": C[tr]}, how="reembed", method=method,
                                 random_state=seed, **kw)
            zt = z[te]
            u = {"anchored": pysimlr.predict(ext, {"view0": A[te], "view1": B[te], "C": C[te]})["u"],
                 "oracle": pysimlr.predict(ora, [A[te], B[te], C[te]])["u"],
                 "base": pysimlr.predict(res, [A[te], B[te]])["u"],
                 "c_only": pysimlr.predict(ext, {"C": C[te]})["u"]}
            row = dict(method=method, seed=seed, **{f"{k_}_r2": latent_r2(zt, v) for k_, v in u.items()},
                       reembed_train_r2=latent_r2(z[tr], rem["reembedded"]["u"]),
                       c_holdout_coupling_min=min(ext["extensions"]["C"]["holdout"]["coupling"]),
                       c_stop=ext["extensions"]["C"]["convergence"]["stop_reason"],
                       c_converged=ext["extensions"]["C"]["convergence"]["converged"],
                       c_bound_gap=ext["extensions"]["C"]["convergence"]["bound_gap"],
                       c_steps=ext["extensions"]["C"]["convergence"]["n_steps"])
            rows.append(row); print(row, flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    for m, d in df.groupby("method"):
        diff = d.anchored_r2 - d.oracle_r2
        ok = diff.mean() >= -0.05 and diff.min() >= -0.10
        print(f"{m}: anchored - oracle mean {diff.mean():+.4f}, worst {diff.min():+.4f} -> "
              f"{'ACCEPT anchored as default' if ok else 'REJECT (option only)'}")


if __name__ == "__main__":
    main()
