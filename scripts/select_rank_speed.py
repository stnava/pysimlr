#!/usr/bin/env python3
"""select_rank speed: serial against parallel refits (n_jobs), with the stability check.

Configurations on each case (same seeds, same subsample indices):
  serial  n_jobs=1 (the 0.6.0 behaviour)
  jobs    n_jobs=-1 (all cores; one worker pool shared across candidates)
A rejected third option, reusing the reference fit's tuned learning rate in every refit, was
measured on the synthetic case before this script was reduced to two configurations:
24.5 s -> 18.9 s, but stability at k=2 moved 0.913 -> 0.735 (rule R16: must stay within 0.03).
Records wall time and per-component stability (median over views) per k; parallel output
must equal serial exactly.

Usage: python scripts/select_rank_speed.py [--out CSV]
"""
import argparse, os, sys, time, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
import numpy as np, pandas as pd, torch
import pysimlr


def cases():
    rng = np.random.default_rng(0); n = 300; z = rng.standard_normal((n, 3))
    yield "synthetic", [z @ rng.standard_normal((3, p)) + rng.standard_normal((n, p)) for p in (200, 150)], {}
    from run_deep_benchmark import get_adni_mci_case
    c = get_adni_mci_case(42)
    yield "adni_mci", c["data"], dict(positivity="positive", topology="graph", path_graph=c["path_graph"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(REPO, "docs/audit/evidence/select_rank_speed.csv"))
    a = ap.parse_args(); warnings.simplefilter("ignore")
    rows = []
    for name, views, kw in cases():
        for cfg, extra in (("serial", {}), ("jobs", dict(n_jobs=-1))):
            t = time.perf_counter()
            sr = pysimlr.select_rank("simlr", views, [2, 3], n_subsamples=20, random_state=42, **kw, **extra)
            sec = time.perf_counter() - t
            for r in sr["table"]:
                rows.append(dict(case=name, config=cfg, k=r["k"], seconds=sec, selected_k=sr["k"],
                                 component_stability=" ".join(f"{x:.4f}" for x in r["component_stability"])))
            print(f"{name:9s} {cfg:10s} {sec:7.1f}s  k={sr['k']}  " +
                  "  ".join(f"k{r['k']}:[{' '.join(f'{x:.3f}' for x in r['component_stability'])}]" for r in sr["table"]), flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    for (case, k), g in df.groupby(["case", "k"]):
        a_, b_ = (g[g.config == c].component_stability.iloc[0] for c in ("serial", "jobs"))
        print(f"{case} k={k}: parallel stability identical to serial: {a_ == b_}")
    for case, g in df.groupby("case"):
        t = g.groupby("config").seconds.first()
        print(f"{case}: speed-up {t['serial'] / t['jobs']:.1f}x ({t['serial']:.1f}s -> {t['jobs']:.1f}s)")

if __name__ == "__main__":
    main()
