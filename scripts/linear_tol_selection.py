"""F2: does the linear plateau tolerance cost recovery or the OLEE contract?

Default simlr (w = 0.5, defect term on) on the constraint_selection synthetic
cells (buried / linear x positive / either, data seeds 0-4) at the default
budget (100 sweeps) for tol in {1e-6, 1e-5, 1e-4}. Reference: the 300-sweep
tol = 1e-6 rows of constraint_selection.csv. Writes
docs/audit/evidence/linear_tol_selection.csv.
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "scripts")]
EVID = os.path.join(REPO, "docs", "audit", "evidence")
TOLS = (1e-6, 1e-5, 1e-4)


def one(task):
    tol, gen, pos, seed = task
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from constraint_selection import _contract
    from default_selection_benchmark import _case
    from optimizer_benchmark_synthetic import latent_r2, support_f1
    from pysimlr.simlr import simlr
    z, X, sups = _case(gen, seed)
    r = simlr(X, k=z.shape[1], positivity=pos, tol=tol, random_state=seed, verbose=False)
    V = [v.detach().double().numpy() for v in r["v"]]
    D, ov = _contract(V)
    return dict(tol=tol, gen=gen, pos=pos, seed=seed, r2=latent_r2(z, r["u"]),
                f1=float(np.mean([support_f1(v, s) for v, s in zip(V, sups)])),
                frame_defect=D, support_overlap=ov, stop=r["stop_reason"],
                n_steps=r["convergence"]["n_steps"])


def main():
    import numpy as np
    import pandas as pd
    tasks = [(t, g, p, s) for t in TOLS for g in ("buried", "linear") for p in ("positive", "either") for s in range(5)]
    with ProcessPoolExecutor(8) as ex:
        d = pd.DataFrame(list(ex.map(one, tasks)))
    d.to_csv(os.path.join(EVID, "linear_tol_selection.csv"), index=False)
    ref = pd.read_csv(os.path.join(EVID, "constraint_selection.csv"))
    ref = ref[(ref.fixture == "synthetic") & (ref.arm == "w0.5")].set_index(["gen", "pos", "seed"])
    key = ["gen", "pos", "seed"]
    for tol in TOLS:
        a = d[d.tol == tol].set_index(key)
        conv = int((a.stop == "converged").sum())
        parts = [f"tol={tol:g}: converged {conv}/20, steps {a.n_steps.mean():.0f}"]
        for m in ("r2", "f1", "frame_defect", "support_overlap"):
            diff = (a[m] - ref.loc[a.index, m]).values
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            parts.append(f"{m} {a[m].mean():.4f} (vs 300-sweep {diff.mean():+.4f}, t {diff.mean() / se if se > 0 else 0:+.1f})")
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
