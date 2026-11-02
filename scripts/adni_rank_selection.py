"""amigo A4: per-component stability of ADNI-MCI simlr fits for k = 2..5
(`pysimlr.select_rank`, 20 half-sample refits per k, both sign modes, the
benchmark's graph topology). Writes docs/audit/evidence/adni_rank_selection.csv."""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def one(task):
    pos, k = task
    warnings.simplefilter("ignore")
    import torch
    torch.set_num_threads(1)
    import pysimlr
    import run_deep_benchmark as R
    c = R.get_adni_mci_case(42)
    out = pysimlr.select_rank("simlr", c["data"], [k], n_subsamples=20, random_state=42,
                              positivity=pos, topology="graph", path_graph=c["path_graph"])
    r = out["table"][0]
    return dict(pos=pos, k=k, min_stability=r["min_stability"], all_stable=r["all_stable"],
                component_stability=" ".join(f"{v:.3f}" for v in r["component_stability"]))


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    T = [(p, k) for p in ("either", "positive") for k in (2, 3, 4, 5)]
    with ProcessPoolExecutor(8) as ex:
        d = pd.DataFrame(list(ex.map(one, T)))
    d.to_csv(os.path.join(REPO, "docs", "audit", "evidence", "adni_rank_selection.csv"), index=False)
    print(d.to_string(index=False))
    for p in ("either", "positive"):
        s = d[(d.pos == p) & d.all_stable]
        print(p, "-> largest k with every component >= 0.9:", int(s.k.max()) if len(s) else None)
