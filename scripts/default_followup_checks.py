#!/usr/bin/env python3
"""Safety and real-data checks for candidate default (energy, optimizer, mixing) combinations.

For each combination given as ``energy:optimizer:mixing``:
  narrow_r2_<pos>  linear `simlr` on two 5-feature views (k=2, 30 sweeps, sparseness 0.5):
                   in-sample R^2 of a planted outcome from the first-layer scores. Below 0.5
                   means the basis lost the signal (the check "nc"+hybrid_adam failed at 0.30).
  ned_collapses    NED (energy, mixing) on the latent-collapse fixture, seeds 1-5: count of
                   fits whose latent off-diagonal covariance norm is >= 0.5.
  ned_budget_ok    NED at 80 and 400 epochs stop at the same step, both converged.
  adni_<pos>       ADNI-MCI held-out score through `run_deep_benchmark.run_experiment_task`
                   (linear `simlr`, seeds 42-44), mean and SD.

Usage: python scripts/default_followup_checks.py acc:hybrid_adam:newton nc:lars:svd ... [--out CSV]
"""
import argparse
import os
import sys
import warnings

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import numpy as np
import torch


def narrow_r2(energy, opt, mixing, pos):
    from pysimlr.simlr import simlr
    torch.manual_seed(0); n, k = 300, 2
    z = torch.randn(n, k); w = torch.randn(k, 1)
    x = [z @ torch.randn(k, 5) + 0.3 * torch.randn(n, 5) for _ in range(2)]; y = (z @ w).squeeze(1)
    r = simlr(x, k=k, iterations=30, positivity=pos, sparseness=0.5, energy_type=energy,
              optimizer_type=opt, mixing_algorithm=mixing, random_state=0)
    s = torch.cat([m @ v for m, v in zip(x, r["v"])], 1); s = (s - s.mean(0)) / (s.std(0) + 1e-8)
    d = torch.cat([s, torch.ones(n, 1)], 1); b = torch.linalg.lstsq(d, y.unsqueeze(1)).solution
    return 1 - float(((y - (d @ b).squeeze(1)) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ned_collapses(energy, mixing):
    from pysimlr.deep import ned_simr
    bad = 0
    for seed in range(1, 6):
        torch.manual_seed(seed); n = 200
        u = torch.randn(n, 2)
        x1 = u @ torch.randn(2, 20) + 0.1 * torch.randn(n, 20); x2 = u @ torch.randn(2, 20) + 0.1 * torch.randn(n, 20)
        nm = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-6)
        r = ned_simr([nm(x1), nm(x2)], k=2, iterations=100, batch_size=32, warmup_epochs=10,
                     energy_type=energy, mixing_algorithm=mixing, random_state=seed)
        uc = r["u"] - r["u"].mean(0); c = uc.T @ uc / (n - 1)
        bad += float(torch.norm(c - torch.diag(torch.diag(c)))) >= 0.5
    return bad


def ned_budget_ok(energy, mixing):
    from pysimlr.deep import ned_simr
    from test_parameters_are_live import _views, K
    v = _views()
    s = ned_simr(v, k=K, iterations=80, energy_type=energy, mixing_algorithm=mixing, random_state=0)
    l = ned_simr(v, k=K, iterations=400, energy_type=energy, mixing_algorithm=mixing, random_state=0)
    return bool(s["convergence"]["converged"] and l["convergence"]["converged"]
                and s["convergence"]["n_steps"] == l["convergence"]["n_steps"])


def adni(energy, opt, mixing, pos):
    import run_deep_benchmark as R
    orig = R.run_single_experiment

    def rse(*a, **k):
        k.update(energy_type=energy, optimizer_type=opt, mixing_algorithm=mixing)
        return orig(*a, **k)
    R.run_single_experiment = rse
    try:
        vals = [R.run_experiment_task(("ADNI-MCI", "real", "simlr", "SiMLR", s, 25, 45, 0.5, pos))["Predictive_Metric"]
                for s in (42, 43, 44)]
    finally:
        R.run_single_experiment = orig
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("combos", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    import pandas as pd
    rows = []
    for c in a.combos:
        energy, opt, mixing = c.split(":")
        row = dict(energy=energy, optimizer=opt, mixing=mixing)
        for pos in ("positive", "either"):
            row[f"narrow_r2_{pos}"] = round(narrow_r2(energy, opt, mixing, pos), 3)
            m, sd = adni(energy, opt, mixing, pos)
            row[f"adni_{pos}"] = f"{m:.4f}+-{sd:.4f}"
        row["ned_collapses"] = ned_collapses(energy, mixing)
        row["ned_budget_ok"] = ned_budget_ok(energy, mixing)
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
