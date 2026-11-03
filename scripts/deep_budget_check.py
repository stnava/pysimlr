#!/usr/bin/env python3
"""Deep default budget: 150 vs 300 epochs on the R13 grid (all deep families).

Grid: lend, ned, ned_shared_private, flow_v x {Linear, Polynomial, Sine,
Shared+Private, NonnegParts, NonnegPartsWeak} x seeds 42-44, through
run_deep_benchmark.run_experiment_task (package defaults otherwise).
Rule: adopt 300 only if every family converges on >= 90% of fits at 300, and
Predictive_Metric and CMC_Latent_U are non-inferior to 150 (mean paired diff
not below -1 paired SD; no cell worse by > 0.05).
Usage: python scripts/deep_budget_check.py [--workers 10]
"""
import argparse, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
import numpy as np, pandas as pd

MODELS = [("lend", "LEND"), ("ned", "NED"), ("ned_shared_private", "NEDPP"), ("flow_v", "Flow-SiMLR-V")]
DATA = ["Linear", "Polynomial", "Sine", "Shared+Private", "NonnegParts", "NonnegPartsWeak"]


def run(t):
    warnings.simplefilter("ignore")
    import torch; torch.set_num_threads(1)
    from run_deep_benchmark import run_experiment_task
    r = run_experiment_task(t)
    return dict(model=t[3], dataset=t[0], seed=t[4], budget=t[6], converged=bool(r.get("Solver_Converged")),
                n_steps=r.get("Solver_Iters"), predictive=r.get("Predictive_Metric"), cmc=r.get("CMC_Latent_U"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(REPO, "docs/audit/evidence/deep_budget_check.csv"))
    a = ap.parse_args()
    tasks = [(d, "synthetic", m, lab, s, None, ep, 0.5, "default")
             for m, lab in MODELS for d in DATA for s in (42, 43, 44) for ep in (150, 300)]
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(list(ex.map(run, tasks)))
    df.to_csv(a.out, index=False)
    ok = True
    for lab, g in df.groupby("model"):
        conv = g.groupby("budget").converged.mean()
        line = f"{lab:13s} converged 150: {conv[150]:.3f}  300: {conv[300]:.3f}"
        ok &= conv[300] >= 0.9
        for m in ("predictive", "cmc"):
            p = g.pivot_table(index=["dataset", "seed"], columns="budget", values=m)
            d = p[300] - p[150]; worst = d.min()
            bad = (d.mean() < -d.std(ddof=1)) or (worst < -0.05)
            ok &= not bad
            line += f" | {m} diff {d.mean():+.4f} (sd {d.std(ddof=1):.4f}, worst {worst:+.4f}){' FAIL' if bad else ''}"
        print(line)
    print("RULE:", "ADOPT 300" if ok else "KEEP 150 / investigate")


if __name__ == "__main__":
    main()
