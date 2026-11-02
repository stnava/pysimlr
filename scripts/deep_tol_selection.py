"""amigo A1: does the deep models' plateau tolerance (1e-6 vs 1e-4) cost recovery?

lend, ned, ned_shared_private at their default 150-epoch budget on
{buried, linear} x {either, positive} x seeds 0-2 (scripts/
default_selection_benchmark.py generators). Reports convergence, epochs used,
latent R^2 of the full fit, and the best training loss. Writes
docs/audit/evidence/deep_tol_selection.csv.
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
EVID = os.path.join(REPO, "docs", "audit", "evidence")


def one(task):
    method, tol, gen, pos, seed = task
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import pysimlr
    from default_selection_benchmark import _case
    from optimizer_benchmark_synthetic import latent_r2
    z, X, _ = _case(gen, seed)
    r = pysimlr.fit(method, X, z.shape[1], positivity=pos, tol=tol, random_state=seed)
    c = r["convergence"]
    return dict(method=method, tol=tol, gen=gen, pos=pos, seed=seed, r2=latent_r2(z, r["u"]),
                converged=bool(c["converged"]), steps=c["n_steps"], best_loss=c["best"])


def main():
    import numpy as np
    import pandas as pd
    pd.set_option("display.width", 200)
    T = [(m, t, g, p, s) for m in ("lend", "ned", "ned_shared_private") for t in (1e-6, 1e-4)
         for g in ("buried", "linear") for p in ("either", "positive") for s in range(3)]
    with ProcessPoolExecutor(8) as ex:
        d = pd.DataFrame(list(ex.map(one, T)))
    d.to_csv(os.path.join(EVID, "deep_tol_selection.csv"), index=False)
    print(d.groupby(["method", "tol"]).agg(converged=("converged", "sum"), n=("converged", "size"),
                                           steps=("steps", "mean"), r2=("r2", "mean")).round(4).to_string())
    key = ["method", "gen", "pos", "seed"]
    a = d[d.tol == 1e-6].set_index(key); b = d[d.tol == 1e-4].set_index(key)
    for m in ("lend", "ned", "ned_shared_private"):
        diff = (b.loc[m, "r2"] - a.loc[m, "r2"]).values
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"{m}: R^2 (1e-4 minus 1e-6) mean {diff.mean():+.4f}, paired t {diff.mean() / se if se > 0 else 0:+.2f}, "
              f"min {diff.min():+.4f}; epochs {a.loc[m, 'steps'].mean():.0f} -> {b.loc[m, 'steps'].mean():.0f}")


if __name__ == "__main__":
    main()
