"""Convex-positivity pilot, round 2: any method family, sparseness, energy (D36 provenance).

Provenance copy. It ran from the job tmp directory beside `nhanes_robust.py` (NHANES
loader: labs/diet/body, 2017-18, glucose removed) on branch pilot/convex-positivity
(86aaa71). `--gauge on` reproduces the first pilot's unit-L2 energy gauge, a control
that does not exist in the shipped code.

Arm = positivity string passed through ('either', 'positive', 'convex'); the
optional env CONVEX_GAUGE selects the linear energy gauge under convex in
the prototype ('off' = energy sees the simplex columns [design]; 'on' = unit-L2
gauge kept [round-1 behaviour]).

usage: python pilot2.py WORKTREE --tag TAG --datasets adni,nhanes --arms either,positive,convex
                        --seeds 10 [--sparseness 0.8] [--method simlr] [--energy nc] [--gauge off]
Writes pilot2_TAG.csv and bases/TAG_<ds>_<arm>_<seed>.npz next to this file.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys, time, warnings, argparse
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SP = {"adni": 0.85, "nhanes": 0.5}
ADNI_CSV = "/Users/stnava/data/repos/pysimlr/data/adni/adni_mci_cognitive_merged.csv"  # main checkout, read-only


def std_tr(a, b):
    mu, sd = a.mean(0), a.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (a - mu) / sd, (b - mu) / sd


def rf_r2(ztr, zte, ytr, yte, seed):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import r2_score
    gs = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1),
                      {"min_samples_leaf": [1, 5, 20], "max_features": [0.33, 1.0]},
                      cv=KFold(5, shuffle=True, random_state=seed), scoring="r2", n_jobs=1)
    gs.fit(ztr, ytr)
    return float(r2_score(yte, gs.predict(zte)))


def ridge_r2(ztr, zte, ytr, yte):
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    return float(r2_score(yte, RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(ztr, ytr).predict(zte)))


def basis_stats(vs):
    nz = [(np.abs(v) > 1e-10) for v in vs]
    cols = np.concatenate([m.sum(0) for m in nz])
    return dict(support_mean=float(cols.mean()), support_min=int(cols.min()),
                le2_frac=float((cols <= 2).mean()),
                nan=int(sum(np.isnan(v).sum() for v in vs)),
                dead_cols=int((cols == 0).sum()),
                neg_entries=int(sum((v < -1e-12).sum() for v in vs)),
                colsum_err=float(max(np.abs(v.sum(0) - 1).max() for v in vs)))


def task(args):
    ds, arm, seed, wt, sp, method, energy, gauge, tag = args
    sys.path[:0] = [wt + "/src", wt + "/scripts", os.path.dirname(HERE)]
    import torch
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    import pysimlr
    import pysimlr.sparsification as sps
    if hasattr(sps, "CONVEX_GAUGE"):
        sps.CONVEX_GAUGE = gauge
    row = dict(tag=tag, dataset=ds, arm=arm, seed=seed, sparseness=sp, method=method, energy=energy or "nc", gauge=gauge)
    kw = dict(positivity=arm, sparseness=sp, random_state=seed)
    if energy:
        kw["energy_type"] = energy
    try:
        if ds == "adni":
            import paper_adni_ablations as pa
            pa.ADNI_CSV = ADNI_CSV
            data = pa.protocol_b(seed)
            t0 = time.perf_counter()
            res, _ = pa.fit_adni(data, method, k=3, **kw)
            row["secs"] = time.perf_counter() - t0
            vs = pa.mri_basis(res)
            xtr, xte, ytr, yte = data["x_tr"], data["x_te"], data["pc1_tr"], data["pc1_te"]
            hc_views = [torch.tensor(x).float() for x in xte] + [None]
        else:
            import nhanes_robust as nr
            xs, y, _ = nr.load("J", True, "HDL")
            n = len(y); p = np.random.default_rng(seed).permutation(n); te, tr = p[: n // 4], p[n // 4:]
            xtr = [nr.z(x[tr], x[tr]) for x in xs]; xte = [nr.z(x[te], x[tr]) for x in xs]
            ytr, yte = y[tr], y[te]
            t0 = time.perf_counter()
            res = pysimlr.fit(method, [torch.tensor(x).float() for x in xtr], 5, **kw)
            row["secs"] = time.perf_counter() - t0
            vs = [np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, float) for v in res["v"]]
            hc_views = [torch.tensor(x).float() for x in xte]
    except Exception as e:  # recorded, never silently dropped
        row["error"] = repr(e)[:300]
        return row
    ztr = np.hstack([x @ v for x, v in zip(xtr, vs)]); zte = np.hstack([x @ v for x, v in zip(xte, vs)])
    a, b = std_tr(ztr, zte)
    row["rf_XV_r2"] = rf_r2(a, b, ytr, yte, seed)
    row["ridge_XV_r2"] = ridge_r2(a, b, ytr, yte)
    try:
        hc = pysimlr.heldout_coupling(res, hc_views)
        row["cross_view_heldout"] = float(np.nanmean(hc["cross_view_heldout"]))
    except Exception as e:
        row["hc_error"] = repr(e)[:200]
    row.update(basis_stats(vs))
    row["iters"] = res.get("converged_iter"); row["stop"] = res.get("stop_reason")
    used = res.get("parameters", {}).get("used", {})
    row["used_positivity"] = used.get("positivity"); row["positivity_resolved"] = used.get("positivity_resolved")
    np.savez(f"{HERE}/bases/{tag}_{ds}_{arm}_{seed}.npz", *vs)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wt"); ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default="adni,nhanes"); ap.add_argument("--arms", default="either,positive,convex")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sparseness", type=float, default=None); ap.add_argument("--method", default="simlr")
    ap.add_argument("--energy", default=None); ap.add_argument("--gauge", default="off")
    a = ap.parse_args()
    os.makedirs(f"{HERE}/bases", exist_ok=True)
    grid = [(d, arm, s, a.wt, a.sparseness if a.sparseness is not None else DEFAULT_SP[d],
             a.method, a.energy, a.gauge, a.tag)
            for d in a.datasets.split(",") for s in range(a.seeds) for arm in a.arms.split(",")]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(task, grid))
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/pilot2_{a.tag}.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(f"{time.time() - t0:.0f} s")
