#!/usr/bin/env python3
"""Before/after the relative floor in `similarity._std`, per energy.

`old` re-installs the absolute floor sqrt(var + 1e-6) in-process; `new` is the
package. Energies: nc (does not use _std; control), align, logcosh, exp,
kurtosis, and iva (the IVA-Laplace contrast, registered here, not in the
package). Cases: the 20 default cells (scripts/default_selection_benchmark._case)
and 5 planted |N(0,1)| disjoint-basis cases. Default budget.
Usage: python scripts/std_floor_check.py [--workers 10]
"""
import argparse, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
import numpy as np, pandas as pd

ENERGIES = ["nc", "align", "logcosh", "exp", "kurtosis", "iva"]


def _setup(floor):
    import torch, pysimlr.similarity as sim, pysimlr.simlr  # noqa: F401
    lin = sys.modules["pysimlr.simlr"]
    if not hasattr(sim, "_std_new"):
        sim._std_new = sim._std
    sim._std = sim._std_new if floor == "new" else (
        lambda z, eps=1e-6: torch.sqrt(z.var(dim=0, keepdim=True, unbiased=False) + eps))

    def energy_iva(s, u, ctx=None):
        z = sim._centre(s) / sim._std(sim._centre(s)); w = sim._centre(u) / sim._std(sim._centre(u))
        return torch.sqrt(z ** 2 + w ** 2 + 1e-6).mean() - float(np.sqrt(np.pi / 2))
    sim.SIMILARITY["iva"] = (energy_iva, False, "IVA-Laplace contrast (evaluation only)")
    sim.SCALE_FREE_ENERGIES = sim.SCALE_FREE_ENERGIES | {"iva"}
    lin.SUPPORTED_ENERGY_TYPES = lin.SUPPORTED_ENERGY_TYPES | {"iva"}


def planted(seed, n=300, k=3, dims=(60, 45), noise=0.05):
    import torch
    rng = np.random.default_rng(seed); z = np.abs(rng.standard_normal((n, k))); views, sups = [], []
    for p in dims:
        v = np.zeros((p, k)); blk = p // k
        for j in range(k):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        sups.append([set(range(j * blk, (j + 1) * blk)) for j in range(k)])
        views.append(torch.tensor(z @ v.T + noise * rng.standard_normal((n, p))).float())
    return torch.as_tensor(z), views, sups


def auc(load, sups):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import roc_auc_score
    out = []
    for L, sup in zip(load, sups):
        L = np.abs(np.asarray(L, float)); p, k = L.shape; a = np.zeros((len(sup), k))
        for j, t in enumerate(sup):
            y = np.zeros(p); y[list(t)] = 1
            for c in range(k):
                a[j, c] = roc_auc_score(y, L[:, c])
        r, c = linear_sum_assignment(-a); out.append(a[r, c].mean())
    return float(np.mean(out))


def task(args):
    case, gen, pos, seed, floor = args
    warnings.simplefilter("ignore")
    import torch; torch.set_num_threads(1)
    _setup(floor)
    import pysimlr
    from optimizer_benchmark_synthetic import latent_r2
    if case == "default":
        from default_selection_benchmark import _case
        z, views, sups = _case(gen, seed)
    else:
        z, views, sups = planted(seed)
    rows = []
    for e in ENERGIES:
        r = pysimlr.fit("simlr", views, z.shape[1], energy_type=e, positivity=pos, random_state=seed)
        rows.append(dict(case=case, generator=gen, positivity=pos, seed=seed, floor=floor, energy=e,
                         latent_r2=latent_r2(z, r["u"]),
                         loadings_auc=auc([x.detach().numpy() for x in r["loadings"]], sups),
                         converged=bool(r["convergence"]["converged"]), n_steps=int(r["convergence"]["n_steps"])))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(REPO, "docs/audit/evidence/std_floor_check.csv"))
    a = ap.parse_args()
    cells = ([("default", g, p, s) for g in ("buried", "linear") for p in ("either", "positive") for s in range(5)]
             + [("planted", "nongauss", "positive", s) for s in range(5)])
    tasks = [c + (f,) for c in cells for f in ("old", "new")]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        df = pd.DataFrame([r for rs in ex.map(task, tasks) for r in rs])
    df.to_csv(a.out, index=False)
    print(df.groupby(["case", "energy", "floor"])[["latent_r2", "loadings_auc", "converged"]].mean()
          .unstack("floor").round(3).to_string())


if __name__ == "__main__":
    main()
