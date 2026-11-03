#!/usr/bin/env python3
"""No-descent recovery: stopping on the first all-views no-descent sweep (old) against
reducing the learning rate 10x and retrying (new, `simlr.NO_DESCENT_LR_FLOOR`).

`old` sets NO_DESCENT_LR_FLOOR to infinity in-process, so the first sweep in which
every view's line search fails stops the fit, as before; `new` is the package. Default fits (learning_rate="auto"), default budget, on:
  toy          80 subjects, views of 12 and 9 features, |N| latents (seeds 0-4)
  narrow       tests/test_collapse_penalty._narrow (positive, sparseness 0.5)
  default      the 20 default-selection cells
  adni         ADNI-MCI (seed 42 case), positive and signed, graph topology
Records converged, stop_reason, n_steps, best_energy, latent R^2 (synthetic),
outcome R^2 (narrow), the chosen learning rate and seconds.
Usage: python scripts/linesearch_cap_check.py [--workers 10]
"""
import argparse, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts"), os.path.join(REPO, "tests")]
import numpy as np, pandas as pd


def _arm(arm):
    import pysimlr.simlr  # noqa: F401  (load the module; the attribute is the function)
    mod = sys.modules["pysimlr.simlr"]
    if not hasattr(mod, "_floor_new"):
        mod._floor_new = mod.NO_DESCENT_LR_FLOOR
    mod.NO_DESCENT_LR_FLOOR = mod._floor_new if arm == "new" else float("inf")


def _case(name, seed):
    import torch
    if name == "toy":
        rng = np.random.default_rng(seed); z = np.abs(rng.standard_normal((80, 2)))
        v = [torch.tensor(z @ rng.random((2, p)) + 0.1 * rng.standard_normal((80, p))).float() for p in (12, 9)]
        return z, v, {}, None
    if name == "narrow":
        from test_collapse_penalty import _narrow
        x, y = _narrow(); return None, x, dict(sparseness=0.5), y
    if name.startswith("default_"):
        from default_selection_benchmark import _case as dc
        _, gen = name.split("_"); z, v, _ = dc(gen, seed); return z, v, {}, None
    from run_deep_benchmark import get_adni_mci_case
    c = get_adni_mci_case(42); return None, c["data"], dict(topology="graph", path_graph=c["path_graph"]), None


def task(t):
    name, pos, seed, arm = t
    warnings.simplefilter("ignore")
    import torch; torch.set_num_threads(1)
    _arm(arm)
    import pysimlr
    from optimizer_benchmark_synthetic import latent_r2
    z, views, kw, y = _case(name, seed)
    k = 2 if z is None else z.shape[1]
    if name.startswith("adni"):
        k = 3
    t0 = time.perf_counter()
    r = pysimlr.fit("simlr", views, k, positivity=pos, random_state=seed, **kw)
    row = dict(case=name, positivity=pos, seed=seed, arm=arm, converged=bool(r["convergence"]["converged"]),
               stop_reason=r["stop_reason"], n_steps=int(r["convergence"]["n_steps"]),
               best_energy=float(r["best_energy"]), learning_rate=r.get("learning_rate"),
               seconds=time.perf_counter() - t0,
               latent_r2=latent_r2(z, r["u"]) if z is not None else np.nan)
    if y is not None:
        from test_collapse_penalty import _outcome_r2
        row["outcome_r2"] = float(_outcome_r2(views, y, r["v"]))
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(REPO, "docs/audit/evidence/linesearch_cap_check.csv"))
    a = ap.parse_args()
    cells = ([("toy", p, s) for p in ("positive", "either") for s in range(5)]
             + [("narrow", "positive", 0)]
             + [(f"default_{g}", p, s) for g in ("buried", "linear") for p in ("either", "positive") for s in range(5)]
             + [("adni", p, 42) for p in ("positive", "either")])
    tasks = [c + (arm,) for c in cells for arm in ("old", "new")]
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(list(ex.map(task, tasks)))
    df.to_csv(a.out, index=False)
    grp = df.case.str.replace(r"default_.*", "default", regex=True)
    print(df.assign(group=grp).groupby(["group", "arm"]).agg(
        n=("converged", "size"), converged=("converged", "mean"), no_descent=("stop_reason", lambda s: (s == "no_descent").mean()),
        steps=("n_steps", "median"), latent_r2=("latent_r2", "mean"), best_energy=("best_energy", "mean"),
        seconds=("seconds", "median")).round(4).to_string())
    if "outcome_r2" in df:
        print(df[df.case == "narrow"][["arm", "stop_reason", "n_steps", "outcome_r2", "best_energy"]].to_string(index=False))
    p = df.pivot_table(index=["case", "positivity", "seed"], columns="arm", values="best_energy")
    print("best_energy new - old: mean %+.4g, worse in %d of %d cells" % ((p.new - p.old).mean(), int(((p.new - p.old) > 1e-9).sum()), len(p)))


if __name__ == "__main__":
    main()
