"""Bisection of view-order dependence over settings (synthetic). Cell = max |du| over seeds, after undoing the permutation."""
import argparse, json, warnings
import numpy as np
import pysimlr
warnings.simplefilter("ignore")
from repro import ROOT, fit_ordered, _np

def views_for(seed, n=174, widths=(65, 54, 72, 57, 196), signal=0.0, miss=False):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n); names = [f"V{i}" for i in range(len(widths))]
    views = {}
    for nm, p in zip(names, widths):
        ld = np.abs(rng.standard_normal(p)) * (rng.random(p) < .3)
        views[nm] = signal * np.outer(z, ld) + rng.standard_normal((n, p))
    views[ROOT] = (z + 0.7 * rng.standard_normal(n))[:, None]
    confs = {nm: rng.standard_normal((n, 2)) for nm in names}; confs[ROOT] = None
    if miss:
        for nm in names[:3]: views[nm][rng.choice(n, 20, replace=False)] = np.nan
    return views, confs, names

def du(r1, r2):
    u1, u2 = _np(r1["u"]), _np(r2["u"])
    a, _, bt = np.linalg.svd(u1.T @ u2); r = a @ bt          # orthogonal Procrustes; k=1 reduces to a sign
    return float(np.abs(u1 @ r - u2).max())

def cell(seeds, iters, miss=False, signal=0.0, **kw):
    out = []
    for s in seeds:
        views, confs, names = views_for(s, miss=miss, signal=signal)
        o1 = names + [ROOT]; o2 = names[:3] + [names[4], names[3], ROOT]
        try:
            r1 = fit_ordered(views, confs, o1, iterations=iters, **kw); r2 = fit_ordered(views, confs, o2, iterations=iters, **kw)
        except Exception as e:
            return f"ERR {type(e).__name__}"
        out.append(du(r1, r2))
    return max(out), int(sum(o > 0.01 for o in out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--only", default=""); a = ap.parse_args()
    ns, it = (2, 2) if a.smoke else (a.n_seeds, a.iters)
    seeds = range(ns)
    base = dict(use_conf=False)
    grid = [("baseline (graph, newton, pca, conical, sp .8, nc_abs)", {}),
            ("missing_rows=mask, 20 NaN rows in 3 views", {"miss": True}),
            ("missing_rows=impute, no NaN", {"missing": "impute"}),
            ("confounds on", {"use_conf": True}),
            ("shared signal 0.4", {"signal": 0.4}),
            ("mixing avg", {"mixing": "avg"}), ("mixing svd", {"mixing": "svd"}), ("mixing pca", {"mixing": "pca"}), ("mixing ica", {"mixing": "ica"}),
            ("init random", {"init": "random"}), ("init perturbed_pca", {"init": "perturbed_pca"}), ("init nndsvd", {"init": "nndsvd"}),
            ("init gcca", {"init": "gcca"}),
            ("energy nc", {"energy": "nc"}), ("energy recon", {"energy": "recon"}),
            ("positivity either", {"positivity": "either"}), ("sparseness None", {"sparseness": None}),
            ("positivity either + sparseness None", {"positivity": "either", "sparseness": None}),
            ("k=2", {"k": 2}), ("no root/topology args (star)", {"use_root": False}),
            ("newton + spectral sign alignment (fix)", {"sign_alignment": "spectral"}),
            ("k=2 + spectral", {"k": 2, "sign_alignment": "spectral"}), ("star + spectral", {"use_root": False, "sign_alignment": "spectral"}),
            ("star + svd mixing", {"use_root": False, "mixing": "svd"})]
    if a.only: grid = [g for g in grid if any(t in g[0] for t in a.only.split(","))]
    rows = []
    for name, kw in grid:
        k2 = {**base, **kw}
        res = {f"iters={i}": cell(seeds, i, **k2) for i in (1, it)}
        rows.append({"setting": name, **res}); print(name, res, flush=True)
    print(json.dumps({"smoke": a.smoke, "n_seeds": ns, "rows": rows}))
