#!/usr/bin/env python3
"""Joint choice of the linear defaults: objective x optimizer x mixing, under both positivity settings.

Primary measure: least-squares R^2 of the true shared latent on the fitted ``u``
(recovery of the planted signal). Two generators so a choice cannot overfit one:

  buried   `optimizer_benchmark_synthetic.planted_buried`: sparse non-negative shared
           factors under private factors at 2x (PCA start wrong; CCA recovers ~0.98)
  linear   `benchmarks.synthetic_cases.build_case("linear", private_scale=0.5)`:
           the benchmark's standard linear regime

Only objectives valid on every path (linear, deep, flow) are candidates, since the
default is global: acc, nc, logcosh, exp. Every fit uses the same budget and seed.

Writes one row per fit (energy, optimizer, mixing, positivity, generator, seed,
latent_r2, support_f1, seconds, stop, error) to --out, and prints the ranking.

Usage: python scripts/default_selection_benchmark.py [--seeds 5] [--iterations 25] [--workers 10] [--out CSV]
"""
import argparse
import itertools
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

ENERGIES = ["acc", "nc", "logcosh", "exp"]
OPTIMIZERS = ["hybrid_adam", "lars", "armijo_gradient", "adam"]
MIXINGS = ["newton", "svd", "avg", "pca"]
POSITIVITY = ["positive", "either"]
GENERATORS = ["buried", "linear"]


def _case(generator, seed):
    import numpy as np
    import torch
    from optimizer_benchmark_synthetic import planted_buried
    if generator == "buried":
        z, views, sups = planted_buried(seed)
        return z, views, sups
    from pysimlr.benchmarks.synthetic_cases import build_case
    c = build_case("linear", n_samples=300, seed=seed, private_scale=0.5)
    sups = []
    for v in c["true_v"]:
        v = np.asarray(v)
        sups.append([set(np.flatnonzero(np.abs(v[:, j]) > 1e-12).tolist()) for j in range(v.shape[1])])
    return torch.as_tensor(np.asarray(c["true_u"])), [torch.as_tensor(np.asarray(x)).float() for x in c["data"]], sups


def run_one(task):
    energy, opt, mixing, pos, gen, seed, iterations = task
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from optimizer_benchmark_synthetic import latent_r2, support_f1, loadings_support_auc
    from pysimlr.simlr import simlr
    row = dict(energy=energy, optimizer=opt, mixing=mixing, positivity=pos, generator=gen, seed=seed)
    try:
        z, views, sups = _case(gen, seed)
        k = z.shape[1]
        t = time.time()
        kwargs = dict(energy_type=energy, optimizer_type=opt,
                      mixing_algorithm=mixing, positivity=pos, random_state=seed)
        if iterations is not None:
            kwargs["iterations"] = iterations
        r = simlr(views, k=k, **kwargs)
        conv = r.get("convergence", {})
        res_budget = conv.get("max_steps", iterations if iterations is not None else 300)
        v_sup = float(np.mean([support_f1(v.detach(), s) for v, s in zip(r["v"], sups)]))
        if pos == "positive":
            l_sup = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(r["loadings"], sups)]))
        else:
            l_sup = loadings_support_auc(r["loadings"], sups)
        w = r.get("normalizing_weights")
        w_spread = float(max(w) / max(min(w), 1e-12)) if w else 1.0
        row.update(seconds=time.time() - t,
                   latent_r2=latent_r2(z, r["u"]),
                   v_support_f1=v_sup,
                   support_f1=v_sup,
                   loadings_support=l_sup,
                   budget=res_budget,
                   converged=bool(conv.get("converged", r.get("converged", False))),
                   n_steps=int(conv.get("n_steps", 0)),
                   stop_reason=conv.get("stop_reason", r.get("stop_reason")),
                   learning_rate=r.get("learning_rate"),
                   weight_spread=w_spread,
                   stop=r.get("stop_reason"),
                   error=None)
    except (RuntimeError, ValueError, FloatingPointError) as exc:  # recorded, never dropped
        row.update(error=f"{type(exc).__name__}: {str(exc)[:120]}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import pandas as pd
    tasks = [(e, o, m, p, g, s, a.iterations) for e, o, m, p, g, s in
             itertools.product(ENERGIES, OPTIMIZERS, MIXINGS, POSITIVITY, GENERATORS, range(a.seeds))]
    print(f"{len(tasks)} fits", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(run_one, tasks, chunksize=4))
    df = pd.DataFrame(rows)
    print(f"done in {time.time() - t0:.0f}s; errors: {int(df['error'].notna().sum())}", flush=True)
    if a.out:
        df.to_csv(a.out, index=False)
    ok = df[df["error"].isna()]
    combo = ["energy", "optimizer", "mixing"]
    # mean over seeds, then per (combo, positivity, generator) cell
    cell = ok.groupby(combo + ["positivity", "generator"]).latent_r2.mean().unstack(["positivity", "generator"])
    summ = pd.DataFrame({"mean": cell.mean(axis=1), "worst_cell": cell.min(axis=1)})
    print("\nTop 15 combinations by mean latent R^2 over the 4 positivity x generator cells:")
    print(summ.sort_values("mean", ascending=False).head(15).round(3).to_string())
    print("\nBest per energy (mean over cells):")
    print(summ.reset_index().sort_values("mean", ascending=False).groupby("energy").head(1).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
