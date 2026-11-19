"""Regenerate ``golden_init_v040.npz``: starting bases and short fits of pysimlr 0.40.0.

This is a fixture generator, not a study: run it ONLY from a checkout of commit
2c3af72 (v0.40.0), never from current code, because the point of the file is to pin
what 0.40.0 produced.

    git worktree add --detach /tmp/wt040 2c3af72
    PYTHONPATH=/tmp/wt040/src python tests/data/make_golden_init_v040.py --out tests/data/golden_init_v040.npz
"""
import argparse
import numpy as np
import torch

LEGACY_INITS = ("pca", "varimax", "random", "gcca", "nndsvd", "perturbed_pca", "ica", "best_of_n", "domain")
POSITIVITIES = ("either", "positive", "convex")
FIT_POSITIVITIES = ("either", "positive")


def golden_views(n=60, seed=0):
    g = np.random.default_rng(seed)
    z = g.standard_normal((n, 2))
    return [torch.tensor(z @ g.standard_normal((2, p)) + g.standard_normal((n, p)), dtype=torch.float32)
            for p in (14, 18, 11)]


def golden_domain(views):
    g = np.random.default_rng(1)
    return [torch.tensor(g.standard_normal((3, v.shape[1])), dtype=torch.float32) for v in views]


def main():
    import pysimlr
    from pysimlr.simlr import initialize_simlr
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()
    views = golden_views()
    dom = golden_domain(views)
    out = {}
    for it in LEGACY_INITS:
        for pos in POSITIVITIES:
            gen = torch.Generator().manual_seed(0)
            b = initialize_simlr(views, args.k, initialization_type=it, positivity=pos, generator=gen,
                                 sparseness=0.5, domain_matrices=dom if it == "domain" else None)
            for m, v in enumerate(b):
                out[f"init|{it}|{pos}|v{m}"] = np.asarray(v, dtype=np.float64)
        for pos in FIT_POSITIVITIES:
            r = pysimlr.simlr(views, args.k, iterations=args.iters, initialization_type=it, random_state=0,
                              sparseness=0.5, positivity=pos,
                              domain_matrices=dom if it == "domain" else None)
            for m, v in enumerate(r["v"]):
                out[f"fit|{it}|{pos}|v{m}"] = np.asarray(v, dtype=np.float64)
            out[f"fit|{it}|{pos}|u"] = np.asarray(r["u"], dtype=np.float64)
    np.savez_compressed(args.out, **out)
    print(len(out), "arrays")


if __name__ == "__main__":
    main()
