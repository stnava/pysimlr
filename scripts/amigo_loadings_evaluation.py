"""amigo: encoder V vs `structure` vs `loadings` for the default simlr fit.

Synthetic: {buried, linear} x {either, positive} x seeds 0-2 (generators of
scripts/default_selection_benchmark.py). Per cell: latent R^2 of the full fit;
for V, structure and loadings, the span instability (mean sin^2 of principal
angles) and matched-column |r| between two fits on 98% of the rows, the
support F1 against the planted support, column coherence and support overlap.
ADNI-MCI (4 views, k = 3 and 2, the benchmark's graph topology): the same
stability numbers, plus `pysimlr.loading_inference` (30 half-sample refits)
for per-component stability and stability selection.

Writes docs/audit/evidence/amigo_loadings_synthetic.csv and
amigo_loadings_adni.csv; prints the summary tables.
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
EVID = os.path.join(REPO, "docs", "audit", "evidence")
OBJECTS = ("v", "structure", "loadings")


def span_instability(a, b):
    import numpy as np
    out = []
    for x, y in zip(a, b):
        qa, _ = np.linalg.qr(np.asarray(x, float)); qb, _ = np.linalg.qr(np.asarray(y, float))
        s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)
        out.append(float((1 - s ** 2).mean()))
    return float(np.mean(out))


def matched_r(a, b):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    A = np.vstack([np.asarray(x, float) for x in a]); B = np.vstack([np.asarray(x, float) for x in b])
    k = A.shape[1]
    m = np.abs(np.nan_to_num(np.corrcoef(A.T, B.T)[:k, k:]))
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].mean())


def _blocks(res, obj):
    return [b.detach() if hasattr(b, "detach") else b for b in res[obj]]


def synthetic(task):
    gen, pos, seed = task
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import pysimlr
    from default_selection_benchmark import _case
    from optimizer_benchmark_synthetic import latent_r2, support_f1
    from pysimlr.benchmarks.metrics import column_coherence, support_overlap
    z, X, sups = _case(gen, seed)
    k = z.shape[1]; n = X[0].shape[0]
    full = pysimlr.fit("simlr", X, k, positivity=pos, random_state=seed)
    rng = np.random.default_rng(11 + seed)
    idxs = [torch.as_tensor(np.sort(rng.choice(n, int(0.98 * n), replace=False))) for _ in range(2)]
    a, b = [pysimlr.fit("simlr", [x[i] for x in X], k, positivity=pos, random_state=seed) for i in idxs]
    row = dict(gen=gen, pos=pos, seed=seed, latent_r2=latent_r2(z, full["u"]),
               converged=full["convergence"]["converged"], n_steps=full["convergence"]["n_steps"])
    for obj in OBJECTS:
        blocks = _blocks(full, obj)
        row[f"{obj}|instability"] = span_instability(_blocks(a, obj), _blocks(b, obj))
        row[f"{obj}|matched_r"] = matched_r(_blocks(a, obj), _blocks(b, obj))
        row[f"{obj}|support_f1"] = float(np.mean([support_f1(np.asarray(v, float), s) for v, s in zip(blocks, sups)]))
        row[f"{obj}|coherence"] = float(np.mean([column_coherence(v) for v in blocks]))
        row[f"{obj}|overlap"] = float(np.mean([support_overlap(v) for v in blocks]))
    return row


def adni(task):
    pos, k = task
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import pysimlr
    import run_deep_benchmark as R
    from pysimlr.benchmarks.metrics import column_coherence, support_overlap
    c = R.get_adni_mci_case(42)
    X = c["data"]; n = X[0].shape[0]
    kw = dict(topology="graph", path_graph=c["path_graph"], positivity=pos)
    rng = np.random.default_rng(7)
    idxs = [torch.as_tensor(np.sort(rng.choice(n, int(0.98 * n), replace=False))) for _ in range(2)]
    a, b = [pysimlr.fit("simlr", [x[i] for x in X], k, random_state=42, **kw) for i in idxs]
    inf = pysimlr.loading_inference("simlr", X, k, n_subsamples=30, random_state=42, **kw)
    full = inf["result"]
    row = dict(pos=pos, k=k, converged=full["convergence"]["converged"], n_steps=full["convergence"]["n_steps"])
    for obj in OBJECTS:
        row[f"{obj}|instability"] = span_instability(_blocks(a, obj), _blocks(b, obj))
        row[f"{obj}|matched_r"] = matched_r(_blocks(a, obj), _blocks(b, obj))
        row[f"{obj}|coherence"] = float(np.mean([column_coherence(v) for v in _blocks(full, obj)]))
        row[f"{obj}|overlap"] = float(np.mean([support_overlap(v) for v in _blocks(full, obj)]))
    cs = np.array(inf["component_stability"])           # views x k
    for j in range(k):
        row[f"component_{j + 1}_stability"] = float(np.median(cs[:, j]))
    row["selected_features"] = int(sum(int(s.sum()) for s in inf["selected"]))
    row["expected_false_selections"] = float(sum(inf["expected_false_selections"]))
    row["features"] = int(sum(x.shape[1] for x in X))
    return row


def main():
    import pandas as pd
    pd.set_option("display.width", 250)
    st = [(g, p, s) for g in ("buried", "linear") for p in ("either", "positive") for s in range(3)]
    at = [(p, k) for p in ("either", "positive") for k in (3, 2)]
    with ProcessPoolExecutor(8) as ex:
        fs = [ex.submit(synthetic, t) for t in st]; fa = [ex.submit(adni, t) for t in at]
        S = pd.DataFrame([f.result() for f in fs]); A = pd.DataFrame([f.result() for f in fa])
    S.to_csv(os.path.join(EVID, "amigo_loadings_synthetic.csv"), index=False)
    A.to_csv(os.path.join(EVID, "amigo_loadings_adni.csv"), index=False)
    long = S.melt(id_vars=["gen", "pos", "seed", "latent_r2", "converged", "n_steps"])
    long[["object", "metric"]] = long.variable.str.split("|", expand=True)
    print(S.groupby(["gen", "pos"])[["latent_r2", "converged", "n_steps"]].mean().round(3).to_string())
    print(long.groupby(["pos", "object", "metric"]).value.mean().round(4).unstack("metric").to_string())
    print(A.round(4).T.to_string())


if __name__ == "__main__":
    main()
