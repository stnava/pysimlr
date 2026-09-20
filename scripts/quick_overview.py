#!/usr/bin/env python
"""
One-minute overview across every real cohort.

The design sweep answers "which energy / mixing / w / MAI", and pays for it in
wall clock. This answers the prior question -- "what does each problem look
like at all" -- and is built to finish in about a minute so it can be run
between edits.

Where the 20x comes from, in order of size:

  subsample     n <= 400 rows and p <= 250 columns per view (top variance).
                METABRIC 1764x2000 and Golub 72x7129 dominate everything else;
                capping p is the single biggest win and costs little, because
                the questions here are about *relative* method behaviour.
  budget        iterations 8, epochs 8 rather than 30 and 60.
  grid          one seed, one mixing, two energies -- not five seeds, two
                mixings and four energies.
  parallel      one process per fit.

It is therefore an overview, not a measurement: numbers are noisier and lower
than the full sweep's, and must not be quoted as results. Use `--full` budgets
via the design sweep for anything that goes in a table.
"""
from __future__ import annotations

import os
import multiprocessing

for _v in ("OMP", "MKL", "OPENBLAS", "VECLIB_MAXIMUM", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from pysimlr.benchmarks.runner import run_single_experiment  # noqa: E402
import real_data_design_sweep as RS                          # noqa: E402


def shrink(case: dict, max_n: int, max_p: int, seed: int = 0) -> dict:
    """Subsample rows and keep the highest-variance columns per view."""
    rng = np.random.default_rng(seed)
    mats = [m.numpy() for m in case["data"]]
    n = mats[0].shape[0]
    rows = np.arange(n)
    if n > max_n:                       # stratify so a class cannot vanish
        y = case["outcome"].numpy()
        rows = np.concatenate([
            rng.permutation(np.flatnonzero(y == c))[:max(2, int(max_n * (y == c).mean()))]
            for c in np.unique(y)])
        rows = np.sort(rows)
    out = []
    for m in mats:
        m = m[rows]
        if m.shape[1] > max_p:
            keep = np.sort(np.argsort(m.var(axis=0))[::-1][:max_p])
            m = m[:, keep]
        out.append(torch.tensor(m).float())
    k = case["shared_k"]
    new = dict(case)
    new.update(data=out, outcome=case["outcome"][rows],
               true_u=torch.zeros(len(rows), k),
               true_v=[np.zeros((m.shape[1], k)) for m in out])
    return new


def run_one(task):
    torch.set_num_threads(1)
    ds, model_type, label, energy, mixing, w, seed, max_n, max_p, iters, epochs = task
    case = shrink(RS.CASES[ds](seed=seed), max_n, max_p, seed)
    is_cls = bool(case.get("is_classification"))
    t = time.perf_counter()
    try:
        out = run_single_experiment(
            model_type, case, seed=seed, iterations=iters, epochs=epochs,
            energy_type=energy, mixing_algorithm=mixing, use_nsa=True,
            positivity="positive", nsa_w=w, constraint=f"orthox{w}x1",
            sparseness_quantile=0.0, consolidate=True)
        m = out["metrics"]
        g = lambda a, b: m.get(a, np.nan) if is_cls else m.get(b, np.nan)
        return dict(Dataset=ds, Model=label, Energy=energy, Mixing=mixing, W=w,
                    Pred=float(g("test_accuracy", "test_r2")),
                    # Chance-corrected, so an imbalanced cohort cannot read as
                    # a tie when every model is predicting the majority class.
                    Bal=float(m.get("balanced_test", np.nan)),
                    Base=float(m.get("majority_rate", np.nan)),
                    Axis=float(g("first_layer_axis_test_accuracy",
                                 "first_layer_axis_test_r2")),
                    Spars=float(m.get("sparsity_ratio", np.nan)),
                    Secs=time.perf_counter() - t, Err="")
    except Exception as exc:
        return dict(Dataset=ds, Model=label, Energy=energy, Mixing=mixing, W=w,
                    Pred=np.nan, Bal=np.nan, Base=np.nan, Axis=np.nan,
                    Spars=np.nan, Secs=time.perf_counter() - t,
                    Err=f"{type(exc).__name__}: {str(exc)[:60]}")


def main():
    ap = argparse.ArgumentParser(description="One-minute cohort overview")
    ap.add_argument("--datasets", nargs="*", default=list(RS.CASES))
    ap.add_argument("--energies", nargs="*", default=["regression", "acc"])
    ap.add_argument("--mixings", nargs="*", default=["newton"])
    ap.add_argument("--ws", nargs="*", type=float, default=[0.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max-n", type=int, default=400)
    ap.add_argument("--max-p", type=int, default=250)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    # `recon` is a linear-path term (it needs X and V); the deep models carry
    # their own decoder reconstruction and cannot evaluate it as a similarity.
    _DEEP = {"LEND", "NED", "NEDPP", "Flow-SiMLR-V"}
    tasks = [(ds, mt, lbl, e, mx, w, sd, a.max_n, a.max_p, a.iterations, a.epochs)
             for ds in a.datasets for mt, lbl in RS.ALL_MODELS
             for e in a.energies for mx in a.mixings for w in a.ws
             for sd in range(a.seed, a.seed + a.seeds)
             if not (e == "recon" and lbl in _DEEP)]
    print(f"{len(tasks)} fits | n<={a.max_n} p<={a.max_p} "
          f"iters={a.iterations} epochs={a.epochs} | {a.workers} workers", flush=True)

    t0, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(run_one, t) for t in tasks]):
            rows.append(f.result())
    d = pd.DataFrame(rows)
    el = time.perf_counter() - t0

    pd.set_option("display.width", 200)
    for metric, title in (("Bal", "BALANCED ACCURACY (chance = 1/n_classes)"),
                          ("Pred", "RAW PREDICTIVE (accuracy / R2)"),
                          ("Spars", "SPARSITY")):
        print(f"\n=== {title} (rows=dataset, cols=model; best energy per cell) ===")
        piv = d.pivot_table(index="Dataset", columns="Model", values=metric,
                            aggfunc="max" if metric == "Pred" else "mean")
        print(piv.round(3).to_string())
    print("\n=== best config per dataset ===")
    score = "Bal"
    sub = d.dropna(subset=[score]).copy()
    if sub.empty:
        sub, score = d.dropna(subset=["Pred"]).copy(), "Pred"
    best = (sub.sort_values(score, ascending=False)
               .groupby("Dataset", as_index=False).head(1)
               .sort_values("Dataset"))
    print(best[["Dataset", "Model", "Energy", "Pred", "Bal", "Base", "Spars"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== DOES ANY MULTI-VIEW METHOD BEAT THE UNCONSTRAINED BASELINE? ===")
    # PCA on the concatenated views is the no-multi-view control: it ignores the
    # block structure entirely. A multi-view method earns its complexity only by
    # beating it, and only by more than the seed-to-seed noise.
    mv = [m for _t, m in RS.ALL_MODELS if m != "PCA"]
    lines = []
    for ds, grp in d.dropna(subset=["Pred"]).groupby("Dataset"):
        pca = grp[grp.Model == "PCA"]["Pred"]
        if pca.empty:
            continue
        pca_best = float(pca.max())
        noise = float(grp.groupby(["Model", "Energy", "Mixing", "W"])["Pred"]
                         .std().median()) if d["Pred"].notna().sum() else np.nan
        noise = 0.0 if not np.isfinite(noise) else noise
        best = grp[grp.Model.isin(mv)].sort_values("Pred", ascending=False).head(1)
        if best.empty:
            continue
        b = best.iloc[0]
        delta = float(b["Pred"]) - pca_best
        verdict = ("BEATS PCA" if delta > max(0.01, noise) else
                   "ties PCA" if delta > -max(0.01, noise) else "loses to PCA")
        lines.append(f"  {ds:9s} best MV = {b['Model']:16s} {b['Pred']:.3f}"
                     f"   PCA = {pca_best:.3f}   delta = {delta:+.3f}"
                     f"   seed-noise ~ {noise:.3f}   -> {verdict}")
    print("\n".join(lines) if lines else "  (no PCA rows)")

    print("\n=== cohorts where NO model beats the majority-class baseline ===")
    triv = []
    for ds, grp in d.dropna(subset=["Base"]).groupby("Dataset"):
        base = float(grp["Base"].iloc[0])
        if float(grp["Pred"].max()) <= base + 0.01:
            triv.append(f"  {ds}: best raw {grp['Pred'].max():.3f} vs baseline "
                        f"{base:.3f} -- accuracy is uninformative here")
    print("\n".join(triv) if triv else "  none")
    bad = d[d.Err != ""]
    print(f"\nfailures: {len(bad)}")
    if len(bad):
        print(bad.groupby(["Model", "Err"]).size().to_string())
    print(f"\nwall clock {el:.1f}s  (slowest fit {d.Secs.max():.1f}s)")
    out = os.path.join(os.path.dirname(__file__), "..", "paper",
                       "results_cache", "quick_overview.csv")
    d.to_csv(out, index=False)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
