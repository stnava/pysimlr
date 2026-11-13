"""Synthetic reproducer: view-order dependence of pysimlr.fit('simlr') with a root exposure view."""
import argparse, json, warnings
import numpy as np, torch, pysimlr
warnings.simplefilter("ignore")
ROOT = "Exposure"

def make_data(n=120, widths=(60, 40, 90, 30, 70), seed=0, miss=0.1, conf=True):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    names = [f"V{i}" for i in range(len(widths))]
    views = {}
    for nm, p in zip(names, widths):
        load = np.abs(rng.standard_normal(p)) * (rng.random(p) < .5)
        views[nm] = np.outer(z, load) + rng.standard_normal((n, p)) * 1.0
    if miss:
        for nm in names[:3]:
            rows = rng.choice(n, int(miss * n), replace=False)
            views[nm][rows] = np.nan
    views[ROOT] = (z + 0.7 * rng.standard_normal(n))[:, None]
    confs = {nm: (rng.standard_normal((n, 2)) if conf else None) for nm in names}
    confs[ROOT] = None
    return views, confs

def fit_ordered(views, confs, order, seed=7, iterations=300, init="pca", k=1, sparseness=0.8,
                positivity="conical", energy="nc_abs", missing="mask", use_conf=True, use_root=True, mixing=None, **extra):
    nm = list(order)
    pvp = {}
    for i, v in enumerate(nm):
        pvp[i] = ({"positivity": "conical", "sparseness": None, "energy_type": energy} if v == ROOT else
                  {"positivity": positivity, "sparseness": sparseness, "energy_type": energy})
    kw = dict(iterations=iterations, initialization_type=init, dynamic_weights=False, strict=False, missing_rows=missing, **extra)
    if use_root:
        g = pysimlr.fork_graph(ROOT, [n for n in nm if n != ROOT], exogenous_root=False)
        kw["topology"] = "graph"
        kw["path_graph"] = {nm.index(a): [nm.index(b) for b in bs] for a, bs in g.items()}
    if mixing: kw["mixing_algorithm"] = mixing
    c = None
    if use_conf and any(confs.get(v) is not None for v in nm):
        c = {i: confs[v] for i, v in enumerate(nm) if confs.get(v) is not None}
    return pysimlr.fit("simlr", [views[v] for v in nm], k=k, confounds=c, per_view_params=pvp, random_state=seed, **kw)

def _np(t): return np.asarray(t.detach().cpu().double())

def compare(r1, o1, r2, o2):
    """Compare after undoing the permutation. Sign of each component is aligned per view (abs-corr)."""
    u1, u2 = _np(r1["u"]), _np(r2["u"])
    out = {"u_corr": float(abs(np.corrcoef(u1[:, 0], u2[:, 0])[0, 1])),
           "u_maxabs_diff": float(min(np.abs(u1 - u2).max(), np.abs(u1 + u2).max()))}
    dv = 0.0
    for v in o1:
        a, b = _np(r1["v"][o1.index(v)]), _np(r2["v"][o2.index(v)])
        dv = max(dv, min(np.abs(a - b).max(), np.abs(a + b).max()))
    out["v_maxabs_diff"] = dv
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--n-seeds", type=int, default=5)
    a = ap.parse_args()
    it, ns = (2, 2) if a.smoke else (a.iters, a.n_seeds)
    views, confs = make_data()
    o1 = ["V0", "V1", "V2", "V3", "V4", ROOT]; o2 = ["V0", "V1", "V2", "V4", "V3", ROOT]
    r1 = fit_ordered(views, confs, o1, iterations=it); r2 = fit_ordered(views, confs, o2, iterations=it)
    print(json.dumps({"smoke": a.smoke, "iters": it, "order_diff": compare(r1, o1, r2, o2)}))
