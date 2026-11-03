#!/usr/bin/env python3
"""MAI revision: the old mean-centred gate against the null-calibrated, dof-corrected
floor gate, on views with sub-couplings, different ranks and different widths.

Design (n = 150, k = 3 planted |N| latents):
  A (p 40), B (p 30)  coupled on all three components
  C (p 25)            sub-coupled: component 1 only, plus private variation
  D (p 4)             narrow, component 2 only
  E (p 300 > n)       pure noise
and a control with five fully coupled views. Arms on simlr(dynamic_weights=True):
  old/procrustes_r2   raw scores, gate centred on mai.mean() (the previous code)
  new/procrustes_r2   dof-corrected scores, gate on the permutation floor
  new/max_cc          the same with the first canonical correlation
Records final modality weights and latent R^2 of u on the planted latents.
Usage: python scripts/mai_calibration_check.py [--workers 10]
"""
import argparse, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
import numpy as np, pandas as pd


def views(seed, control=False, n=150):
    import torch
    rng = np.random.default_rng(seed); z = np.abs(rng.standard_normal((n, 3)))
    def block(p, comps, noise=0.5, private=0.0):
        v = np.zeros((p, 3)); blk = max(1, p // max(1, len(comps)))
        for i, c in enumerate(comps):
            v[i * blk:(i + 1) * blk, c] = 0.5 + rng.random(min(blk, p - i * blk))
        x = z @ v.T + noise * rng.standard_normal((n, p))
        if private:
            x += private * rng.standard_normal((n, 2)) @ rng.standard_normal((2, p))
        return torch.tensor(x).float()
    if control:
        return z, [block(p, [0, 1, 2]) for p in (40, 30, 25, 20, 35)]
    return z, [block(40, [0, 1, 2]), block(30, [0, 1, 2]), block(25, [0], private=1.0),
               block(4, [1], noise=0.3), torch.tensor(rng.standard_normal((n, 300))).float()]


def arm_setup(arm):
    import pysimlr.consensus as C
    if not hasattr(C, "_new"):
        C._new = (C.loo_mai_calibrated, C.mai_gated_target, C.MAI_MIN_SHARE)
    arm, _, share = arm.partition("@")
    C.MAI_MIN_SHARE = float(share) if share else C._new[2]
    if arm.startswith("old"):
        def old_scores(latents, metric, dofs=None, use_rank=False, n_perm=20, quantile=0.95, seed=0):
            return C.loo_mai_scores(latents, metric, use_rank=use_rank), None
        def old_gate(mai, steepness, floor):
            gate = __import__("torch").sigmoid(steepness * (mai - mai.mean())); raw = mai * gate
            return raw / (raw.sum() + 1e-8)
        C.loo_mai_calibrated, C.mai_gated_target = old_scores, old_gate
    else:
        C.loo_mai_calibrated, C.mai_gated_target = C._new[:2]


def task(t):
    case, seed, arm = t
    warnings.simplefilter("ignore")
    import torch; torch.set_num_threads(1)
    arm_setup(arm)
    import pysimlr
    from optimizer_benchmark_synthetic import latent_r2
    z, vs = views(seed, control=(case == "control"))
    metric = arm.split("/")[1].split("@")[0]
    r = pysimlr.fit("simlr", vs, 3, dynamic_weights=True, mai_metric=metric, positivity="positive",
                    random_state=seed)
    w = np.asarray(r["modality_weights"], float)
    row = dict(case=case, seed=seed, arm=arm, latent_r2=latent_r2(z, r["u"]),
               converged=bool(r["convergence"]["converged"]))
    row.update({f"w_{'ABCDE'[i]}": w[i] for i in range(len(w))})
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(REPO, "docs/audit/evidence/mai_calibration_check.csv"))
    a = ap.parse_args()
    tasks = [(c, s, arm) for c in ("design", "control") for s in range(5)
             for arm in ("old/procrustes_r2", "new/procrustes_r2@0", "new/procrustes_r2@0.25",
                         "new/procrustes_r2", "new/max_cc@0", "new/max_cc")]
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(list(ex.map(task, tasks)))
    df.to_csv(a.out, index=False)
    cols = [c for c in df.columns if c.startswith("w_")] + ["latent_r2", "converged"]
    print(df.groupby(["case", "arm"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
