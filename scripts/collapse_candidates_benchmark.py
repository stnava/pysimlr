"""Collapse handling without a free threshold: candidates against the 0.3 hinge.

`utils.correlation_penalty` (hinge at |r| = DEFAULT_COLLAPSE_THRESHOLD = 0.3 on
pairwise score correlations, weight 1) stopped the "nc" + newton collapse, but
its threshold is a chosen number and it pushes genuinely correlated factors
apart. Candidates, injected without modifying the package:

  C1 "none"   collapse penalty off (collapse_weight=0; deep penalty zeroed).
  C4 "hinge"  the current default.
  C2 "polar"  no penalty; "nc" scores the representation's orthogonal
              (polar) factor, newton_schulz_orthogonalize(s, 10) -- the same
              operation newton mixing applies to the consensus u -- so both
              sides of the comparison are orthogonalised alike. No threshold,
              no weight.

Generators:
  corr    planted latents with every pairwise correlation rho, two views with
          disjoint non-negative block supports and noise; rho in {0,.3,.5,.7}.
  narrow  tests/test_collapse_penalty._narrow (collapses without a fix).
  ned     the NED collapse fixture of tests/test_collapse_penalty (seeds 1-5).
  grid    scripts/default_selection_benchmark cells for nc/hybrid_adam/newton
          ({buried, linear} x {positive, either} x seeds 0-4).

Caveat (measured after the first run): the deep rows ("corr_ned", "ned") use
one start per data seed (random_state = seed). NED collapse depends on the
start: without a penalty it collapses on 6 of 20 (data seed, start seed) pairs
of the "ned" fixture, with the hinge on 0 of 20, although these rows show 0/5
for both. Read the deep rows as single-start results.

Usage: python scripts/collapse_candidates_benchmark.py [--seeds 5] [--out CSV]
"""
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

CANDIDATES = ["none", "hinge", "polar"]
RHOS = [0.0, 0.3, 0.5, 0.7]


def install(candidate):
    """Patch the package for `candidate`; returns simlr kwargs."""
    import torch
    import pysimlr.deep as deep
    from pysimlr import similarity as sim
    from pysimlr.utils import newton_schulz_orthogonalize
    if candidate in ("none", "polar"):
        deep.correlation_penalty = lambda z, *a, **k: z.sum() * 0.0
    if candidate == "polar":
        fn, needs, doc = sim.SIMILARITY["nc"]

        def nc_polar(s, u, ctx=None, _fn=fn):
            sc = s - s.mean(dim=0, keepdim=True)
            return _fn(newton_schulz_orthogonalize(sc, iterations=10), u, ctx)
        sim.SIMILARITY["nc"] = (nc_polar, needs, doc)
    return {} if candidate == "hinge" else {"collapse_weight": 0.0}


def corr_case(rho, seed, n=300, k=3, p=(40, 30), noise=0.5):
    import torch
    g = torch.Generator().manual_seed(seed)
    c = torch.full((k, k), rho) + (1 - rho) * torch.eye(k)
    z = torch.randn(n, k, generator=g) @ torch.linalg.cholesky(c).T
    views, vs = [], []
    for pm in p:
        v = torch.zeros(pm, k)
        blk = pm // k
        for j in range(k):
            v[j * blk:(j + 1) * blk, j] = 0.5 + torch.rand(blk, generator=g)
        views.append(z @ v.T + noise * torch.randn(n, pm, generator=g))
        vs.append(v)
    return z, views, vs


def _offdiag_abs_mean(s):
    import torch
    c = torch.corrcoef(torch.as_tensor(s).double().T)
    k = c.shape[0]
    return float((c - torch.diag(torch.diag(c))).abs().sum() / (k * (k - 1)))


def run_one(task):
    gen, cand, a, seed = task
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    kw = install(cand)
    from optimizer_benchmark_synthetic import latent_r2, support_f1
    from pysimlr.simlr import simlr
    from pysimlr.deep import ned_simr
    row = dict(generator=gen, candidate=cand, param=a, seed=seed)
    try:
        if gen in ("corr", "corr_ned"):
            z, views, vs = corr_case(a, seed)
            if gen == "corr":
                r = simlr(views, k=3, iterations=40, random_state=seed, **kw)
            else:
                r = ned_simr(views, k=3, iterations=100, batch_size=32, random_state=seed)
            s0 = views[0] @ torch.as_tensor(r["v"][0]).detach().float()
            row.update(latent_r2=latent_r2(z, r["u"][0] if isinstance(r["u"], list) else r["u"]),
                       score_r2=latent_r2(z, s0.detach()),
                       score_corr=_offdiag_abs_mean(s0.detach()),
                       true_corr=_offdiag_abs_mean(z))
        elif gen == "narrow":
            import test_collapse_penalty as t
            x, y = t._narrow()
            r = simlr(x, k=2, iterations=30, positivity="positive", sparseness=0.5,
                      random_state=0, **kw)
            row.update(outcome_r2=t._outcome_r2(x, y, r["v"]),
                       score_corr=_offdiag_abs_mean(x[0] @ r["v"][0]))
        elif gen == "ned":
            torch.manual_seed(seed)
            n = 200
            u = torch.randn(n, 2)
            x = [u @ torch.randn(2, 20) + 0.1 * torch.randn(n, 20) for _ in range(2)]
            x = [(m - m.mean(0)) / (m.std(0) + 1e-6) for m in x]
            r = ned_simr(x, k=2, iterations=100, batch_size=32, warmup_epochs=10,
                         energy_type="nc", mixing_algorithm="newton", random_state=seed)
            c = abs(float(torch.corrcoef(torch.as_tensor(r["u"]).T)[0, 1]))
            row.update(u_corr=c, collapsed=bool(c >= 0.5), latent_r2=latent_r2(u, r["u"]))
        else:  # grid cell: a = "generator/positivity"
            from default_selection_benchmark import _case
            g2, pos = a.split("/")
            z, views, sups = _case(g2, seed)
            r = simlr(views, k=z.shape[1], iterations=25, positivity=pos, random_state=seed, **kw)
            row.update(latent_r2=latent_r2(z, r["u"]),
                       support_f1=float(np.mean([support_f1(v.detach(), s)
                                                 for v, s in zip(r["v"], sups)])))
        row["error"] = None
    except (RuntimeError, ValueError, FloatingPointError) as exc:  # recorded, never dropped
        row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return row


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "audit", "evidence",
                                                   "collapse_candidates.csv"))
    a = ap.parse_args()
    S = range(a.seeds)
    tasks = [(g, c, rho, s) for g in ("corr", "corr_ned") for c in CANDIDATES for rho in RHOS for s in S]
    tasks += [("narrow", c, None, 0) for c in CANDIDATES]
    tasks += [("ned", c, None, s) for c in CANDIDATES for s in range(1, 6)]
    tasks += [("grid", c, f"{g}/{p}", s) for c in CANDIDATES for g in ("buried", "linear")
              for p in ("positive", "either") for s in S]
    rows = []
    # One task per process: `install` patches module state, which would
    # otherwise leak into the next task a worker runs.
    with ProcessPoolExecutor(max_workers=8, max_tasks_per_child=1) as ex:
        for i, r in enumerate(ex.map(run_one, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("errors:", int(df.error.notna().sum()))
    pd.set_option("display.width", 200)
    for gen in ("corr", "corr_ned"):
        d = df[df.generator == gen]
        print(f"\n{gen}: mean (sd) over seeds")
        print(d.groupby(["param", "candidate"])[["true_corr", "score_corr", "score_r2", "latent_r2"]]
              .agg(["mean", "std"]).round(3).to_string())
    print("\nnarrow:"); print(df[df.generator == "narrow"][["candidate", "outcome_r2", "score_corr"]].round(3).to_string())
    d = df[df.generator == "ned"]
    print("\nned collapse fixture:"); print(d.groupby("candidate")[["collapsed", "u_corr", "latent_r2"]].agg(["sum", "mean"]).round(3).to_string())
    d = df[df.generator == "grid"]
    print("\ngrid (nc/hybrid_adam/newton):"); print(d.groupby(["param", "candidate"])[["latent_r2", "support_f1"]].mean().round(3).unstack().to_string())
    print(d.groupby("candidate")[["latent_r2", "support_f1"]].agg(["mean", "min"]).round(3).to_string())


if __name__ == "__main__":
    main()
