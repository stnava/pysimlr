#!/usr/bin/env python3
"""Does simlr's iteration improve on its SVD initialization?

Open question from CORRECTNESS_AUDIT.md ("The largest open question"). The
paper's generators (X_m = Z A_m + noise) put the per-view SVD basis at the
optimum, so they cannot show optimizer value. This script uses a generator in
which the per-view SVD basis is demonstrably wrong: each view also carries k
private factors with `private_scale` times the shared variance, so the leading
singular vectors of each view are private.

Arms, all at k = 3, energy --energy (default: the package default), default constraint, sparseness and optimizer:

- init      : the consensus of the PCA initialization (iteration 0).
- simlr     : full alternation (V-steps, then consensus recomputed by
              `mixing_algorithm`) for `--iterations` sweeps.
- random    : same fit from initialization_type="random".
- oracle    : two-view CCA (scikit-learn) on the scaled views, an upper
              reference for what a cross-view objective can recover.

Reported per arm: least-squares R^2 of the true shared Z on the recovered
latent (column-space recovery), and the acc energy at the returned basis.

Usage: python scripts/experiment_iteration_vs_init.py [--seeds 8] [--iterations 40]
"""
import argparse
import json
import os
import sys
import warnings

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch

from pysimlr.consensus import compute_shared_consensus
from pysimlr.defaults import DEFAULT_ENERGY_TYPE, DEFAULT_MIXING_ALGORITHM
from pysimlr.simlr import calculate_simlr_energy, initialize_simlr, simlr
from pysimlr.utils import preprocess_data


def latent_regression_r2(z, u):
    """Fraction of Z's variance explained by least-squares regression on U."""
    z = np.asarray(z, float); u = np.asarray(u, float)
    z = z - z.mean(0); u = u - u.mean(0)
    beta, *_ = np.linalg.lstsq(u, z, rcond=None)
    return 1.0 - np.sum((z - u @ beta) ** 2) / np.sum(z ** 2)


def buried(seed, n=300, k=3, p=(40, 30), private_scale=2.0, noise=0.5):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g)
    views = []
    for pm in p:
        priv = torch.randn(n, k, generator=g)
        views.append(z @ torch.randn(k, pm, generator=g)
                     + private_scale * priv @ torch.randn(k, pm, generator=g)
                     + noise * torch.randn(n, pm, generator=g))
    return z, views


def mean_energy(scaled, vs, u, energy):
    return float(np.mean([float(calculate_simlr_energy(v.to(x.dtype), x, u.to(x.dtype), energy))
                          for x, v in zip(scaled, vs)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--private-scale", type=float, default=2.0)
    ap.add_argument("--mixing", default=DEFAULT_MIXING_ALGORITHM)
    # The DEFAULT_TOPOLOGY evidence in pysimlr/defaults.py was measured with
    # --energy acc, the default at the time.
    ap.add_argument("--energy", default=DEFAULT_ENERGY_TYPE)
    ap.add_argument("--topology", default="star", help="star (each view scored against a consensus that includes it) or loo (leave-one-out)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from sklearn.cross_decomposition import CCA
    warnings.simplefilter("ignore")
    rows = []
    for seed in range(a.seeds):
        z, x = buried(seed, private_scale=a.private_scale)
        scaled = [preprocess_data(m, ["centerAndScale", "np"])[0] for m in x]
        v0 = initialize_simlr(scaled, 3)
        u0 = compute_shared_consensus([m @ v for m, v in zip(scaled, v0)],
                                      mixing_algorithm=a.mixing, k=3, topology="star")
        row = {"seed": seed, "init_r2": latent_regression_r2(z, u0),
               "init_energy": mean_energy(scaled, v0, u0, a.energy)}
        for arm, init in (("simlr", "pca"), ("random", "random")):
            res = simlr(x, k=3, iterations=a.iterations, energy_type=a.energy,
                        mixing_algorithm=a.mixing, initialization_type=init, topology=a.topology,
                        random_state=seed)
            u = res["u"][0] if isinstance(res["u"], list) else res["u"]
            row[f"{arm}_r2"] = latent_regression_r2(z, u)
            row[f"{arm}_energy"] = mean_energy(scaled, res["v"], u, a.energy)
            row[f"{arm}_best_iteration"] = int(res["best_iteration"])
        cca = CCA(n_components=3, max_iter=2000).fit(scaled[0].numpy(), scaled[1].numpy())
        c0, c1 = cca.transform(scaled[0].numpy(), scaled[1].numpy())
        row["oracle_cca_r2"] = latent_regression_r2(z, (c0 + c1) / 2)
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    keys = [k for k in rows[0] if k != "seed"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["gain_r2_mean"] = summary["simlr_r2"] - summary["init_r2"]
    summary["gain_r2_min"] = float(min(r["simlr_r2"] - r["init_r2"] for r in rows))
    summary["seeds_improved"] = int(sum(r["simlr_r2"] > r["init_r2"] + 0.01 for r in rows))
    print("SUMMARY", json.dumps({k: round(v, 4) for k, v in summary.items()}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "rows": rows, "summary": summary}, f, indent=1)


if __name__ == "__main__":
    main()
