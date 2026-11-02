"""F1: which orthogonality setting gives the best loadings without breaking the
OLEE contract? One run; writes docs/audit/evidence/constraint_selection*.csv.

Since the amigo branch the orthogonality-defect term is gone from `simlr`'s
objective, so the "w0.5_nodefect" arm equals "w0.5" when rerun; the CSV
records both as they were.

The evidence CSV was produced before the constraint-string API was removed
(the "Stiefel" arm, a hard retraction at w = 1 with a tangent-space gradient
and no defect term, needed ``constraint="Stiefel"``). The script now runs the
nsa_w arms only; that removed arm is skipped.

Arms (incumbent w = 0.5 from the NSA-Flow paper):
  w0.5            constraint="orthox0.5" (default)
  w0.5_nodefect   same retraction, orthogonality-defect term removed from the
                  objective (simlr.orthogonality_defect patched to 0, so its
                  calibrated weight is 0)
  w0.2, w0.9      brackets
  Stiefel         hard Stiefel retraction

Fixtures and metrics:
  synthetic  buried / linear generators x positive / either, data seeds 0-4,
             300-sweep budget. Contract: frame defect D(V) (runner's
             definition) and support overlap (max off-diagonal cosine of |V|
             columns). Loadings: support F1; resampling stability = subspace
             distance between the V's fitted on two random 80% subsamples
             (the default PCA start ignores random_state, so start-seed
             stability would be trivially 0). Guards: latent R^2, stop reason.
             Energy traces are saved for F2.
  narrow     tests/test_collapse_penalty._narrow, 300 sweeps: outcome R^2.
  adni       ADNI-MCI held-out metric, benchmark protocol
             (run_deep_benchmark.run_experiment_task, 25 iterations), 3 seeds.
"""
import importlib
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "scripts"), os.path.join(REPO, "tests")]
EVID = os.path.join(REPO, "docs", "audit", "evidence")

ARMS = {"w0.5": (0.5, False), "w0.5_nodefect": (0.5, True),
        "w0.2": (0.2, False), "w0.9": (0.9, False)}
ITERS = 300


class _NoDefect:
    """Patch `pysimlr.simlr.orthogonality_defect` to 0 for one fit."""
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        if self.on:
            import torch
            self.mod = importlib.import_module("pysimlr.simlr")
            self.orig = self.mod.orthogonality_defect
            self.mod.orthogonality_defect = lambda a, normalized=True: torch.zeros((), dtype=a.dtype)

    def __exit__(self, *exc):
        if self.on:
            self.mod.orthogonality_defect = self.orig


def _contract(V):
    import numpy as np
    import torch
    from pysimlr.utils import orthogonality_defect
    D = float(np.mean([float(orthogonality_defect(torch.as_tensor(v).float(), normalized=True)) for v in V]))
    ov = []
    for v in V:
        a = np.abs(np.asarray(v, float))
        a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-12)
        g = a.T @ a
        ov.append(float(np.max(g - np.diag(np.diag(g)))) if g.shape[0] > 1 else 0.0)
    return D, float(np.mean(ov))


def _gap(a, b):
    import numpy as np
    out = []
    for x, y in zip(a, b):
        qa, _ = np.linalg.qr(np.asarray(x, float)); qb, _ = np.linalg.qr(np.asarray(y, float))
        out.append(np.linalg.norm(qa @ qa.T - qb @ qb.T, 2))
    return float(max(out))


def synth(task):
    arm, gen, pos, seed = task
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from default_selection_benchmark import _case
    from optimizer_benchmark_synthetic import latent_r2, support_f1
    from pysimlr.simlr import simlr
    con, nodef = ARMS[arm]
    z, X, sups = _case(gen, seed)
    k = z.shape[1]
    t0 = time.time()
    with _NoDefect(nodef):
        r = simlr(X, k=k, iterations=ITERS, positivity=pos, nsa_w=con, random_state=seed, verbose=False)
        V = [v.detach().double().numpy() for v in r["v"]]
        n = X[0].shape[0]
        rng = np.random.default_rng(1000 + seed)
        subs = []
        for _ in range(2):
            idx = torch.as_tensor(np.sort(rng.choice(n, int(0.8 * n), replace=False)))
            rs = simlr([x[idx] for x in X], k=k, iterations=ITERS, positivity=pos, nsa_w=con,
                       random_state=seed, verbose=False)
            subs.append([v.detach().double().numpy() for v in rs["v"]])
    D, ov = _contract(V)
    return dict(fixture="synthetic", arm=arm, gen=gen, pos=pos, seed=seed,
                r2=latent_r2(z, r["u"]), f1=float(np.mean([support_f1(v, s) for v, s in zip(V, sups)])),
                frame_defect=D, support_overlap=ov, stability_gap=_gap(*subs),
                stop=r["stop_reason"], n_steps=r["convergence"]["n_steps"], seconds=time.time() - t0,
                energy=";".join(f"{e:.10g}" for e in r["energy"]))


def narrow(task):
    arm, pos = task
    warnings.simplefilter("ignore")
    import torch
    torch.set_num_threads(1)
    from test_collapse_penalty import _narrow, _outcome_r2
    from pysimlr.simlr import simlr
    con, nodef = ARMS[arm]
    x, y = _narrow()
    with _NoDefect(nodef):
        r = simlr(x, k=2, iterations=ITERS, positivity=pos, nsa_w=con, random_state=0, verbose=False)
    s = torch.cat([m @ v for m, v in zip(x, r["v"])], 1)
    D, ov = _contract([v.detach().numpy() for v in r["v"]])
    return dict(fixture="narrow", arm=arm, pos=pos, r2=_outcome_r2(x, y, r["v"]),
                score_corr=float(torch.corrcoef(s.T)[0, 1]), frame_defect=D, support_overlap=ov,
                stop=r["stop_reason"])


def adni(task):
    arm, pos, seed = task
    warnings.simplefilter("ignore")
    import run_deep_benchmark as R
    con, nodef = ARMS[arm]
    orig = R.run_single_experiment

    def rse(*a, **k):
        k.update(nsa_w=con)
        return orig(*a, **k)
    R.run_single_experiment = rse
    try:
        with _NoDefect(nodef):
            out = R.run_experiment_task(("ADNI-MCI", "real", "simlr", "SiMLR", seed, 25, 45, 0.5, pos))
    finally:
        R.run_single_experiment = orig
    return dict(fixture="adni", arm=arm, pos=pos, seed=seed, adni=out["Predictive_Metric"],
                frame_defect=out.get("Frame_Defect"))


def main():
    import pandas as pd
    st = [(a, g, p, s) for a in ARMS for g in ("buried", "linear") for p in ("positive", "either") for s in range(5)]
    nt = [(a, p) for a in ARMS for p in ("positive", "either")]
    at = [(a, p, s) for a in ARMS for p in ("positive", "either") for s in (42, 43, 44)]
    with ProcessPoolExecutor(8) as ex:
        futs = ([ex.submit(synth, t) for t in st] + [ex.submit(narrow, t) for t in nt]
                + [ex.submit(adni, t) for t in at])
        rows = []
        for i, f in enumerate(futs):
            rows.append(f.result())
            if (i + 1) % 10 == 0:
                print(f"progress {i + 1}/{len(futs)}", flush=True)
    d = pd.DataFrame(rows)
    tr = d[d.fixture == "synthetic"][["arm", "gen", "pos", "seed", "energy"]]
    tr.to_csv(os.path.join(EVID, "constraint_selection_traces.csv"), index=False)
    d.drop(columns=["energy"]).to_csv(os.path.join(EVID, "constraint_selection.csv"), index=False)
    pd.set_option("display.width", 220)
    s = d[d.fixture == "synthetic"]
    cols = ["r2", "f1", "frame_defect", "support_overlap", "stability_gap"]
    print(s.groupby("arm")[cols].agg(["mean", "std"]).round(4).reindex(list(ARMS)).to_string())
    print(s.groupby(["gen", "pos", "arm"])[["f1", "frame_defect", "stability_gap"]].mean().round(3).unstack("arm").to_string())
    print(s.groupby("arm").stop.value_counts().unstack(fill_value=0).reindex(list(ARMS)).to_string())
    print(d[d.fixture == "narrow"][["arm", "pos", "r2", "score_corr", "frame_defect", "stop"]].round(3).to_string())
    print(d[d.fixture == "adni"].groupby(["arm", "pos"]).adni.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
