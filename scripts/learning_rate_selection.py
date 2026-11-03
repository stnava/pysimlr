#!/usr/bin/env python3
"""Learning-rate default for linear `simlr`: probe ("auto") against fixed rates.

Arms (rule fixed before running; see docs/tracking.md):
  R7    learning_rate="auto" over optimizers.LR_PROBE_GRID (7 rates, 3-sweep probes)
  R3    "auto" over (1.0, 0.1, 0.01)
  F0.3  learning_rate=0.3, with the in-run no-descent recovery (lr / 10 retries)
  F1.0  learning_rate=1.0, same

Cells, per optimizer: the default-selection generators (buried, linear) x positivity x
5 seeds -- not the cells the fixed candidates were picked on -- plus three hard designs
(p >> n; low SNR; k=8 with M=5) x positivity x 3 seeds. Default energy and mixing, default
budget. ``--adni`` adds the ADNI-MCI held-out score (seeds 42-44) per arm for the default
optimizer.

Rule, per optimizer, against R7: the cheapest arm with mean dR2 >= -0.002, every cell
>= -0.02, converged fraction not lower, support F1 mean d >= -0.01 (where defined), and
(default optimizer) ADNI held-out mean within one R7 seed SD.

Outcome (2026-11-03, docs/audit/evidence/learning_rate_selection.csv, 2,000 rows, 0 errors):
the rule selects fixed 0.3 for hybrid_adam (mean dR2 +0.0013, worst -0.0049, 0.77x the
time, ADNI within one SD), lookahead and bidirectional_lookahead; fixed 1.0 for the armijo
optimizers (their probe chose 1.0 in 38/38 cells); "auto" over (1, 0.1, 0.01) for seven
others; the full grid for rmsprop. NOT ADOPTED: the fixed defaults fail existing contracts
(Stiefel near-orthogonality 0.0032 > 0.002; an NSA-Flow failure on a k = p = 5 view;
short-budget energy progress), and the 3-rate grid fails three short-budget tests while
saving ~5% for hybrid_adam. The default stays "auto" over all 7 rates.

Usage: python scripts/learning_rate_selection.py --out docs/audit/evidence/learning_rate_selection.csv [--adni]
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

ARMS = {"R7": ("auto", None), "R3": ("auto", (1.0, 0.1, 0.01)), "F0.3": (0.3, None), "F1.0": (1.0, None)}
COST_ORDER = ["F0.3", "F1.0", "R3", "R7"]          # cheapest first (fixed rates tie; 0.3 first)
POSITIVITY = ["positive", "either"]
HARD = ["wide", "lowsnr", "k8m5"]


def _hard_case(design, seed):
    import numpy as np
    import torch
    g = np.random.default_rng(1000 + seed)
    n, ps, k, noise = {"wide": (80, (2000, 1500, 1000), 3, 0.5),
                       "lowsnr": (300, (200, 150, 100), 3, 3.0),
                       "k8m5": (400, (300, 250, 200, 150, 100), 8, 0.7)}[design]
    z = g.normal(size=(n, k))
    views = [torch.tensor(z @ g.normal(size=(k, p)) + noise * g.normal(size=(n, p)), dtype=torch.float32)
             for p in ps]
    return torch.tensor(z), views, None


def run_one(task):
    opt, arm, gen, pos, seed = task
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import pysimlr.optimizers as O
    from optimizer_benchmark_synthetic import latent_r2, support_f1
    from default_selection_benchmark import _case
    from pysimlr.simlr import simlr
    lr, grid = ARMS[arm]
    if grid is not None:
        O.LR_PROBE_GRID = grid
    row = dict(optimizer=opt, arm=arm, generator=gen, positivity=pos, seed=seed)
    try:
        z, views, sups = _hard_case(gen, seed) if gen in HARD else _case(gen, seed)
        t = time.time()
        r = simlr(views, k=z.shape[1], optimizer_type=opt, positivity=pos, random_state=seed,
                  learning_rate=lr)
        conv = r["convergence"]
        row.update(seconds=time.time() - t, latent_r2=latent_r2(z, r["u"]),
                   support_f1=(float(np.mean([support_f1(v.detach(), s) for v, s in zip(r["v"], sups)]))
                               if sups is not None else float("nan")),
                   converged=bool(conv["converged"]), n_steps=int(conv["n_steps"]),
                   stop_reason=conv["stop_reason"], learning_rate=r.get("learning_rate", lr),
                   lr_reductions=int(conv.get("learning_rate_reductions") or 0),
                   learning_rate_final=conv.get("learning_rate_final"),
                   error=None)
    except (RuntimeError, ValueError, FloatingPointError) as exc:   # recorded, never dropped
        row.update(error=f"{type(exc).__name__}: {str(exc)[:120]}")
    return row


def adni_rows():
    import numpy as np
    import pysimlr.optimizers as O
    import run_deep_benchmark as R
    orig, grid0, rows = R.run_single_experiment, O.LR_PROBE_GRID, []
    for arm, (lr, grid) in ARMS.items():
        O.LR_PROBE_GRID = grid if grid is not None else grid0

        def rse(*a, _lr=lr, **k):
            k.update(learning_rate=_lr)
            return orig(*a, **k)
        R.run_single_experiment = rse
        try:
            for pos in POSITIVITY:
                for s in (42, 43, 44):
                    t = time.time()
                    v = R.run_experiment_task(("ADNI-MCI", "real", "simlr", "SiMLR", s, 300, 45, 0.5, pos))
                    rows.append(dict(optimizer="hybrid_adam", arm=arm, generator="adni_mci", positivity=pos,
                                     seed=s, seconds=time.time() - t,
                                     adni_heldout=float(v["Predictive_Metric"]), error=None))
        finally:
            R.run_single_experiment = orig
            O.LR_PROBE_GRID = grid0
    return rows


def main():
    import pandas as pd
    from default_selection_benchmark import GENERATORS, OPTIMIZERS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only-optimizer", default=None)
    ap.add_argument("--adni", action="store_true")
    a = ap.parse_args()
    opts = a.only_optimizer.split(",") if a.only_optimizer else OPTIMIZERS
    tasks = ([(o, arm, g, p, s) for o, arm, g, p, s in
              itertools.product(opts, ARMS, GENERATORS, POSITIVITY, range(5))]
             + [(o, arm, g, p, s) for o, arm, g, p, s in
                itertools.product(opts, ARMS, HARD, POSITIVITY, range(3))])
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_one, tasks, chunksize=4))
    # Written before the ADNI phase, so a failure there cannot lose the synthetic rows.
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} synthetic rows to {a.out}", flush=True)
    if a.adni:
        rows += adni_rows()
        df = pd.DataFrame(rows)
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df)} rows to {a.out}; errors: {int(df['error'].notna().sum())}")


if __name__ == "__main__":
    main()
