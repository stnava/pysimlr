"""Calibrate a stationarity residual for `simlr` before it is reported.

The removed residual, ||P(V - ||V||^2 grad E) - V|| / ||V|| with P the full
`simlr_sparseness` projection (quantile sparsity, NSA-Flow retraction), did not
vanish at stationary points: over 300 sweeps it fell only to 0.24-1.0 of its
first value (docs/tracking.md D8).

Candidate measured here: the same scale-invariant gradient mapping with P the
convex projection onto the iterate's current face, P_S(z) = z * 1[supp V]
(clamped at 0 under non-negative positivity). It is zero exactly where the
support-restricted problem is stationary. Also recorded: the off-support ratio,
max_j off support of t*(-grad E)_j (only increasing moves under positivity)
over min |v| on support, per column; > 1 means one gradient step would admit a
coordinate larger than the smallest kept one.

(V, grad E) are recorded at every `step` call by wrapping the optimizer; the
package is not modified. Each fit runs `--iterations` sweeps with the plateau
rule disabled (tol=0, patience=iterations).

Usage: python scripts/stationarity_residual_calibration.py [--iterations 300] [--out CSV]
"""
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

OPTIMIZERS = ["hybrid_adam", "lars", "armijo_gradient", "adam"]
FIXTURES = ["live_views", "certificate_views", "buried", "adni"]


def face_residual(v, g, nonneg):
    import torch
    t = float(v.norm() ** 2)
    mask = (v.abs() > 1e-12).to(v.dtype)
    z = (v - t * g) * mask
    if nonneg:
        z = z.clamp_min(0.0)
    on = float((z - v).norm() / v.norm().clamp_min(1e-300))
    step = -t * g
    cand = step.clamp_min(0.0) if nonneg else step.abs()
    off = 0.0
    for j in range(v.shape[1]):
        s = mask[:, j] > 0
        if bool(s.all()) or not bool(s.any()):
            continue
        off = max(off, float(cand[~s, j].max() / v[s, j].abs().min().clamp_min(1e-300)))
    return on, off


def _fixture(name, seed=0):
    import torch
    if name == "live_views":
        import test_parameters_are_live as t
        return t._views(seed), t.K
    if name == "certificate_views":
        import test_solver_certificate as t
        return t._views(seed), 3
    if name == "buried":
        from optimizer_benchmark_synthetic import planted_buried
        z, views, _ = planted_buried(seed)
        return views, z.shape[1]
    from run_deep_benchmark import get_adni_mci_case
    c = get_adni_mci_case(42)
    return [torch.as_tensor(x).float() for x in c["data"][:3]], 3


def run_one(task):
    fixture, opt, pos, iterations = task
    warnings.simplefilter("ignore")
    import torch
    torch.set_num_threads(1)
    import importlib
    S = importlib.import_module("pysimlr.simlr")
    real = S.create_optimizer
    rec = []

    class Recorder:
        def __init__(self, inner):
            self.inner = inner
            self.gradient_function = None
            self.smooth_energy_function = None

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def step(self, i, v, d, f=None):
            if self.gradient_function is not None:
                g = -self.gradient_function(v).detach().double()
                rec.append((i, *face_residual(v.detach().double(), g, pos == "positive")))
            if hasattr(self.inner, "gradient_function"):
                self.inner.gradient_function = self.gradient_function
                self.inner.smooth_energy_function = self.smooth_energy_function
            return self.inner.step(i, v, d, f)

    S.create_optimizer = lambda *a, **k: Recorder(real(*a, **k))
    views, k = _fixture(fixture)
    r = S.simlr(views, k=k, iterations=iterations, optimizer_type=opt, positivity=pos,
                tol=0.0, patience=iterations, random_state=0)
    m = len(views)
    # `simlr` probes learning rates with `step` before the first sweep; the
    # sweeps are the last len(energy) * m calls.
    n_sweeps = len(r["energy"])
    rec = rec[len(rec) - n_sweeps * m:]
    rows = []
    for s in range(n_sweeps):
        chunk = rec[s * m:(s + 1) * m]
        rows.append(dict(fixture=fixture, optimizer=opt, positivity=pos, sweep=s,
                         face=max(c[1] for c in chunk), off=max(c[2] for c in chunk),
                         energy=r["energy"][s]))
    return rows


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "audit", "evidence",
                                                   "stationarity_residual_calibration.csv"))
    a = ap.parse_args()
    tasks = [(f, o, p, a.iterations) for f in FIXTURES for o in OPTIMIZERS
             for p in ("positive", "either")]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run_one, tasks):
            rows.extend(r)
            print(f"done {r[0]['fixture']} {r[0]['optimizer']} {r[0]['positivity']} "
                  f"({len(r)} sweeps)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    import numpy as np
    summ = df.groupby(["fixture", "optimizer", "positivity"]).agg(
        sweeps=("sweep", "max"), face_first=("face", "first"), face_last=("face", "last"),
        face_min=("face", "min"), off_last=("off", "last"))
    summ["decades"] = np.log10(summ.face_first / summ.face_last.clip(lower=1e-300))
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
