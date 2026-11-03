#!/usr/bin/env python3
"""Fast synthetic benchmark for comparing simlr optimizers.

Generator (`planted_buried`): k shared factors with sparse, non-negative,
disjoint loadings (each shared factor loads on its own `support_frac` of each
view's features) plus k private factors per view at `private_scale` times the
shared scale, plus noise. The private factors dominate each view's leading
singular vectors, so the PCA start is demonstrably wrong while the shared
signal is recoverable (two-view CCA reference).

The objective's own optimum does not recover the truth (the "acc" optimum stays
near latent R^2 0.48 however long it runs, against 0.98 for CCA: covariance, not
correlation, and the private factors dominate covariance). Recovery therefore
measures the objective, not the optimizer. The primary metric is the
optimality gap at a fixed budget,
    gap = (E_best - E_ref) / (E_start - E_ref),
the fraction of the achievable decrease not achieved, where E_start is the
energy of the common PCA start and E_ref the lowest energy any optimizer
reaches with a long budget (`--ref-iterations`) on that seed.

Reported per optimizer (same settings for all; `nsa_flow` is excluded because
it is gradient descent plus a second projection, not an optimizer):
  gap         optimality gap (above); 0 = reached the best known optimum
  latent_r2   least-squares R^2 of the true shared latent on the fitted u
  support_f1  F1 of the fitted non-zero rows against the true support, best
              column matching, averaged over views
  best_energy the objective at the returned iterate
  seconds     fit time
and, per metric, a blocked (seed = block) repeated-measures F test and a
Friedman test across optimizers: small p means the benchmark separates
optimizers beyond seed noise.

Usage: python scripts/optimizer_benchmark_synthetic.py [--seeds 5] [--iterations 25] [--out CSV]
"""
import argparse
import itertools
import os
import sys
import time
import warnings

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pandas as pd
import torch

from pysimlr.simlr import simlr

OPTIMIZERS = ["armijo_gradient", "bidirectional_armijo_gradient", "lookahead", "bidirectional_lookahead",
              "lars", "hybrid_adam", "adam", "nadam", "rmsprop", "gd",
              "torch_adamw", "torch_adagrad", "torch_nadam"]


def planted_buried(seed, n=200, k=2, p=(30, 24, 18), support_frac=0.3, private_scale=2.0, noise=0.5):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g)
    views, supports = [], []
    for pm in p:
        v = torch.zeros(k, pm)
        width = max(2, int(round(support_frac * pm)))
        perm = torch.randperm(pm, generator=g)
        sup = []
        for c in range(k):
            idx = perm[c * width:(c + 1) * width]
            v[c, idx] = 0.5 + torch.rand(len(idx), generator=g)
            sup.append(set(idx.tolist()))
        priv = torch.randn(n, k, generator=g) @ torch.randn(k, pm, generator=g)
        views.append(z @ v + private_scale * priv + noise * torch.randn(n, pm, generator=g))
        supports.append(sup)
    return z, views, supports


def latent_r2(z, u):
    z = np.asarray(z, float); u = np.asarray(u, float)
    z = z - z.mean(0); u = u - u.mean(0)
    beta, *_ = np.linalg.lstsq(u, z, rcond=None)
    return 1.0 - np.sum((z - u @ beta) ** 2) / np.sum(z ** 2)


def support_f1(v, true_sup):
    v = np.abs(np.asarray(v, float)); k = v.shape[1]
    best = 0.0
    for perm in itertools.permutations(range(k)):
        f1s = []
        for c, t in zip(perm, true_sup):
            est = set(np.flatnonzero(v[:, c] > 1e-8).tolist())
            tp = len(est & t)
            f1s.append(0.0 if tp == 0 else 2 * tp / (len(est) + len(t)))
        best = max(best, float(np.mean(f1s)))
    return best


def blocked_tests(df, metric):
    """Repeated-measures F (seeds as blocks) and Friedman p across optimizers."""
    from scipy.stats import f as f_dist, friedmanchisquare
    d = df.copy()
    d[metric] = d[metric] - d.groupby("seed")[metric].transform("mean")
    k, b = d.optimizer.nunique(), d.seed.nunique()
    means = d.groupby("optimizer")[metric].mean()
    ss_t = b * ((means - means.mean()) ** 2).sum()
    ss_e = ((d[metric] - d.optimizer.map(means)) ** 2).sum()
    F = (ss_t / (k - 1)) / (ss_e / ((k - 1) * (b - 1)))
    piv = df.pivot(index="seed", columns="optimizer", values=metric)
    return dict(F=round(float(F), 2), p_F=float(f"{1 - f_dist.cdf(F, k - 1, (k - 1) * (b - 1)):.2g}"),
                p_friedman=float(f"{friedmanchisquare(*[piv[c] for c in piv]).pvalue:.2g}"))


def run_config(energy="acc", mixing="newton", positivity="positive", optimizers=OPTIMIZERS,
               seeds=5, iterations=25, ref_iterations=300, ref_optimizers=("lars", "hybrid_adam")):
    """One objective / mixing / positivity setting: every optimizer on every seed,
    with the reference optimum E_ref estimated per seed for this setting."""
    from sklearn.cross_decomposition import CCA
    fit = dict(energy_type=energy, mixing_algorithm=mixing, positivity=positivity)
    rows, ref = [], []
    for seed in range(seeds):
        z, views, sups = planted_buried(seed)
        x = [((v - v.mean(0)) / v.std(0)).numpy() for v in views]
        c0, c1 = CCA(n_components=2, max_iter=2000).fit(x[0], x[1]).transform(x[0], x[1])
        init = simlr(views, k=2, iterations=0, random_state=seed, **fit)
        ref.append(dict(seed=seed, cca=latent_r2(z, (c0 + c1) / 2), pca_start=latent_r2(z, init["u"]),
                        e_start=float(init["best_energy"])))
        for opt in optimizers:
            t = time.time()
            try:
                r = simlr(views, k=2, iterations=iterations, optimizer_type=opt, random_state=seed, **fit)
            except (RuntimeError, ValueError) as exc:  # recorded, never dropped
                rows.append(dict(optimizer=opt, seed=seed, error=f"{type(exc).__name__}: {str(exc)[:80]}"))
                continue
            rows.append(dict(optimizer=opt, seed=seed, seconds=time.time() - t,
                             latent_r2=latent_r2(z, r["u"]),
                             support_f1=float(np.mean([support_f1(v.detach(), s) for v, s in zip(r["v"], sups)])),
                             best_energy=float(r["best_energy"]), stop=r["stop_reason"]))
        e_refs = []
        for opt in ref_optimizers:
            try:
                e_refs.append(float(simlr(views, k=2, iterations=ref_iterations, optimizer_type=opt,
                                          random_state=seed, **fit)["best_energy"]))
            except (RuntimeError, ValueError):
                pass
        ref[-1]["e_ref"] = min(e_refs) if e_refs else np.nan
    df = pd.DataFrame(rows); refd = pd.DataFrame(ref)
    df = df.merge(refd[["seed", "e_start", "e_ref"]], on="seed")
    if "best_energy" in df:
        df["e_ref"] = np.fmin(df["e_ref"], df.groupby("seed")["best_energy"].transform("min"))
        df["gap"] = (df["best_energy"] - df["e_ref"]) / (df["e_start"] - df["e_ref"])
    for k, v in fit.items():
        df[k] = v
    return df, refd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=25)
    ap.add_argument("--ref-iterations", type=int, default=300)
    ap.add_argument("--ref-optimizers", nargs="*", default=["lars", "hybrid_adam"],
                    help="optimizers run for --ref-iterations to estimate E_ref (the two that reached "
                         "the lowest energies); the budget runs also count")
    ap.add_argument("--optimizers", nargs="*", default=OPTIMIZERS)
    ap.add_argument("--energies", nargs="*", default=["acc"])
    ap.add_argument("--mixings", nargs="*", default=["newton"])
    ap.add_argument("--positivity", nargs="*", default=["positive"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    frames = []
    for energy, mixing, pos in itertools.product(a.energies, a.mixings, a.positivity):
        df, refd = run_config(energy, mixing, pos, a.optimizers, a.seeds, a.iterations,
                              a.ref_iterations, a.ref_optimizers)
        frames.append(df)
        print(f"\n== energy={energy} mixing={mixing} positivity={pos}: CCA {refd.cca.mean():.3f}, "
              f"PCA start {refd.pca_start.mean():.3f}", flush=True)
        if "error" in df and df["error"].notna().any():
            print("errors:", df.loc[df["error"].notna(), ["optimizer", "error"]].drop_duplicates().to_string(index=False))
        ok = df.dropna(subset=["gap"]) if "gap" in df else df.iloc[0:0]
        if len(ok):
            summ = ok.groupby("optimizer").agg(gap=("gap", "mean"), gap_sd=("gap", "std"),
                                               latent_r2=("latent_r2", "mean"), seconds=("seconds", "mean"))
            print(summ.sort_values("gap").round(3).to_string(), flush=True)
            if ok.optimizer.nunique() > 2 and ok.seed.nunique() > 1 and ok.groupby("seed").size().nunique() == 1:
                print(f"discrimination gap: {blocked_tests(ok, 'gap')}", flush=True)
    if a.out:
        pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
