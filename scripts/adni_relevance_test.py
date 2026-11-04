"""ADNI-MCI relevance test (docs/audit/evidence/relevance/*.csv). Where does the predictive component's explained variance concentrate? Four tests, two views.

Statistic: share of the explained variance of target t (relative weights of the view's features,
normalised to sum 1) for each feature, bilateral pair and DKT lobe. Target t: mean of the other
MRI views' standardised, sign-aligned scores on the predictive component j (the consensus
component most correlated with training CogPC1). Cognition is excluded from t everywhere so
training, held-out and refit analyses are comparable.

  A. training fit, target permuted against features (B_TRAIN permutations)
  B. held-out replication: the fitted model applied to the 326 test subjects (never fitted),
     shares and their permutation null recomputed there (B_TEST)
  C. refit replication: N_REFIT half-sample refits of the training rows, each with its own
     permutation test (B_REFIT); frequency of p < 0.05
  D. full-refit null: the view's training rows permuted against the other views and the whole
     model refitted (N_NULL per view); p = P(null share >= observed training share)
p-values are uncorrected: p = (1 + #{null >= observed}) / (B + 1).
"""
import os, re, sys, time, numpy as np, pandas as pd, torch
from concurrent.futures import ProcessPoolExecutor
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [W + "/src", W + "/scripts"]

B_TRAIN, B_TEST, B_REFIT, N_REFIT, N_NULL, CHUNK = 100_000, 50_000, 20_000, 20, 300, 5_000
VIEWS = {1: "vol_area", 2: "direct_thickness"}
OUT = os.path.join(W, "docs", "audit", "evidence", "relevance")


def zs(a):
    a = a - a.mean(0); s = a.std(0); return np.where(s > 0, a / np.where(s > 0, s, 1), 0)


def setup(xz):
    n = xz.shape[0]; pp, d, qt = np.linalg.svd(xz, full_matrices=False)
    return pp @ qt, (qt.T @ np.diag(d) @ qt / np.sqrt(n)) ** 2


def shares(zo, lam2, T):
    n = T.shape[0]; w = lam2 @ ((zo.T @ zs(T) / np.sqrt(n)) ** 2)
    return w / w.sum(0, keepdims=True)


def perm_counts(zo, lam2, t, obs, groups, B, rng):
    """Counts of null >= observed for features and for each group (sums of features)."""
    ge = np.zeros(len(obs)); geg = {g: 0 for g in groups}
    for s in range(0, B, CHUNK):
        k = min(CHUNK, B - s)
        nl = shares(zo, lam2, rng.permuted(np.tile(t[:, None], (1, k)), axis=0))     # k permutations at once
        ge += (nl >= obs[:, None]).sum(1)
        for g, idx in groups.items():
            geg[g] += int((nl[idx].sum(0) >= obs[idx].sum()).sum())
    return (1 + ge) / (B + 1), {g: (1 + c) / (B + 1) for g, c in geg.items()}


def base_name(n):
    n = re.sub(r"^(left|right)_", "", n)
    return re.sub(r"_(l|r)dktlobes$", "_dktlobes", n)


def lobe_of(n):
    from brain_region_maps import DKT_LOBES
    b = re.sub(r"dkt(cortex|regions)$", "", base_name(n)).strip("_")
    return next((lb for lb, regs in DKT_LOBES.items() if b in regs), None)


def groups_for(names):
    bil, lob = {}, {}
    for i, n in enumerate(names):
        if n.startswith(("left_", "right_")) or re.search(r"_(l|r)dktlobes$", n):
            bil.setdefault("bilateral " + base_name(n), []).append(i)
        lb = lobe_of(n)
        if lb:
            lob.setdefault("lobe " + lb, []).append(i)
    return {k: v for k, v in bil.items() if len(v) == 2}, lob


def fit_and_target(d, rows=None, perm_view=None, perm_seed=None):
    """Fit on training rows (optionally a subset, optionally one view's rows permuted) and return
    the fitted result, the predictive component j and per-view MRI targets on those rows."""
    import paper_adni_ablations as paa, pysimlr
    dd = dict(d)
    idx = np.arange(len(d["pc1_tr"])) if rows is None else rows
    xs = [x[idx].copy() for x in d["x_tr"]]
    if perm_view is not None:
        xs[perm_view] = xs[perm_view][np.random.default_rng(perm_seed).permutation(len(idx))]
    dd.update(x_tr=xs, y_tr=d["y_tr"][idx], pc1_tr=d["pc1_tr"][idx])
    res, _ = paa.fit_adni(dd, "simlr", k=3, positivity="positive", sparseness=0.85, random_state=42)
    return res, dd


def mri_scores(res, mri_views):
    """Per-MRI-view latent scores in the rotated frame, from MRI alone (cognition missing)."""
    import pysimlr
    R = res["loading_rotation"].double().numpy()
    pred = pysimlr.predict(res, [torch.tensor(x).float() for x in mri_views] + [None])
    from pysimlr.extension import view_names
    lat = pred["latents"]; keys = view_names(res)[:3]
    return [np.asarray(lat[k].detach().double()) @ R for k in keys], np.asarray(pred["u"].detach().double()) @ R


def analyse(res, dd, mri_views, pc1, signs=None):
    """j, sign vector and per-view (xz, t) on the given rows."""
    sc, u = mri_scores(res, mri_views)
    j = int(np.argmax([abs(np.corrcoef(u[:, c], pc1)[0, 1]) for c in range(u.shape[1])]))
    if signs is None:
        signs = [np.where(((z - z.mean(0)) * (u - u.mean(0))).sum(0) >= 0, 1.0, -1.0) for z in sc]
    sc = [zs(z * s) for z, s in zip(sc, signs)]
    from pysimlr.api import _preprocess_views
    xs = [x.double().numpy() for x in _preprocess_views([torch.tensor(x).float() for x in mri_views],
                                                        res["scale_list"][:3], res["provenance_list"][:3])[0]]
    out = {}
    for m in VIEWS:
        t = zs(sum(z for i, z in enumerate(sc) if i != m))[:, j]
        out[m] = (zs(xs[m]), t)
    return j, signs, out


def refit_task(args):
    kind, r, m = args
    torch.set_num_threads(1)
    import paper_adni_ablations as paa
    d = paa.protocol_b(42)
    n = len(d["pc1_tr"])
    if kind == "refit":
        rows = np.sort(np.random.default_rng(1000 + r).choice(n, n // 2, replace=False))
        res, dd = fit_and_target(d, rows=rows)
        j, _, per = analyse(res, dd, dd["x_tr"][:3], dd["pc1_tr"])
        out = {}
        for mv, (xz, t) in per.items():
            keep = xz.std(0) > 0; zo, lam2 = setup(xz[:, keep]); obs = shares(zo, lam2, t[:, None])[:, 0]
            names = [nm for nm, k in zip(d["names"][mv], keep) if k]
            bil, lob = groups_for(names)
            p, pg = perm_counts(zo, lam2, t, obs, {**bil, **lob}, B_REFIT, np.random.default_rng(r))
            out[mv] = (names, obs, p, pg)
        return kind, r, out
    res, dd = fit_and_target(d, perm_view=m, perm_seed=5000 + r)
    j, _, per = analyse(res, dd, dd["x_tr"][:3], dd["pc1_tr"])
    xz, t = per[m]; keep = xz.std(0) > 0; zo, lam2 = setup(xz[:, keep])
    return kind, (r, m), shares(zo, lam2, t[:, None])[:, 0]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ADNI relevance test (see module docstring)")
    ap.add_argument("--refits", type=int, default=10, help="half-sample refits for stability (0: skip)")
    ap.add_argument("--full-null", type=int, default=0,
                    help="full-refit null fits per view (0: skip; 300 reproduced the permutation results)")
    ap.add_argument("--b-train", type=int, default=B_TRAIN); ap.add_argument("--b-test", type=int, default=B_TEST)
    ap.add_argument("--out", default=OUT); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--figure", action="store_true", help="also draw docs/figures/adni_volarea_relevance.png")
    a = ap.parse_args()
    out_dir = a.out
    T0 = time.time()
    torch.set_num_threads(1)
    import paper_adni_ablations as paa
    d = paa.protocol_b(42)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(2026)
    res, dd = fit_and_target(d)
    j, signs, tr = analyse(res, dd, d["x_tr"][:3], d["pc1_tr"])
    _, _, te = analyse(res, dd, d["x_te"][:3], d["pc1_te"], signs=signs)
    rows, grows = [], []
    ref = {}
    for m, vname in VIEWS.items():
        keep = tr[m][0].std(0) > 0
        names = [nm for nm, k in zip(d["names"][m], keep) if k]
        bil, lob = groups_for(names)
        groups = {**bil, **lob}
        zo, lam2 = setup(tr[m][0][:, keep]); obs = shares(zo, lam2, tr[m][1][:, None])[:, 0]
        p_tr, pg_tr = perm_counts(zo, lam2, tr[m][1], obs, groups, a.b_train, rng)
        zo2, lam22 = setup(te[m][0][:, keep]); obs_te = shares(zo2, lam22, te[m][1][:, None])[:, 0]
        p_te, pg_te = perm_counts(zo2, lam22, te[m][1], obs_te, groups, a.b_test, rng)
        ref[m] = dict(names=names, keep=keep, obs=obs, p_tr=p_tr, obs_te=obs_te, p_te=p_te, groups=groups,
                      pg_tr=pg_tr, pg_te=pg_te)
        print(f"{vname}: training and held-out done ({time.time() - T0:.0f} s)", flush=True)
    t0 = time.time()
    tasks = [("refit", r, None) for r in range(a.refits)] + [("null", r, m) for m in VIEWS for r in range(a.full_null)]
    refit = {m: [] for m in VIEWS}; null = {m: [] for m in VIEWS}
    if tasks:
        with ProcessPoolExecutor(min(a.workers, len(tasks))) as ex:
            for kind, key, out in ex.map(refit_task, tasks, chunksize=max(1, len(tasks) // (4 * a.workers))):
                if kind == "refit":
                    for m, v in out.items():
                        refit[m].append(v)
                else:
                    null[key[1]].append(out)
    print(f"{a.refits} refits, {a.full_null} full-refit null fits per view: {time.time() - t0:.0f} s", flush=True)
    for m, vname in VIEWS.items():
        R_ = ref[m]; names = R_["names"]; fair = 1 / len(names)
        nul = np.array(null[m]) if null[m] else np.full((0, len(names)), np.nan)   # (n_null, p)
        p_full = ((1 + (nul >= R_["obs"][None, :]).sum(0)) / (len(nul) + 1) if len(nul)
                  else np.full(len(names), np.nan))                 # NaN = not run, never a number
        freq = {nm: 0 for nm in names}; gfreq = {g: 0 for g in R_["groups"]}
        for (nm_r, obs_r, p_r, pg_r) in refit[m]:
            for nm, pp in zip(nm_r, p_r):
                freq[nm] = freq.get(nm, 0) + (pp < 0.05)
            for g, pp in pg_r.items():
                gfreq[g] = gfreq.get(g, 0) + (pp < 0.05)
        for i, nm in enumerate(names):
            rows.append(dict(view=vname, feature=nm, share_train=R_["obs"][i], ratio_fair=R_["obs"][i] / fair,
                             p_train=R_["p_tr"][i], share_test=R_["obs_te"][i], p_test=R_["p_te"][i],
                             refit_p05_freq=freq[nm] / len(refit[m]) if refit[m] else np.nan,
                             p_fullrefit_null=p_full[i]))
        for g, idx in R_["groups"].items():
            o = R_["obs"][idx].sum(); nl = nul[:, idx].sum(1)
            grows.append(dict(view=vname, group=g, n=len(idx), share_train=o, ratio_fair=o / (len(idx) * fair),
                              p_train=R_["pg_tr"][g], share_test=R_["obs_te"][idx].sum(), p_test=R_["pg_te"][g],
                              refit_p05_freq=gfreq[g] / len(refit[m]) if refit[m] else np.nan,
                              p_fullrefit_null=(1 + (nl >= o).sum()) / (len(nul) + 1) if len(nul) else np.nan))
    fd, gd = pd.DataFrame(rows), pd.DataFrame(grows)
    fd.to_csv(out_dir + "/features.csv", index=False); gd.to_csv(out_dir + "/groups.csv", index=False)
    print(f"tests done: {time.time() - T0:.0f} s total", flush=True)
    if a.figure:
        import runpy
        os.environ["RELEVANCE_DIR"] = out_dir
        runpy.run_path(os.path.join(W, "scripts", "adni_relevance_figure.py"), run_name="__main__")
        print(f"figure done: {time.time() - T0:.0f} s total", flush=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    for v in VIEWS.values():
        f = fd[fd.view == v]
        print(f"\n===== {v} (component {j + 1}; train n={len(d['pc1_tr'])}, test n={len(d['pc1_te'])}) =====")
        print("counts p<0.05 | train", int((f.p_train < .05).sum()), "test", int((f.p_test < .05).sum()),
              "full-refit null", int((f.p_fullrefit_null < .05).sum()), "of", len(f), "| chance", round(.05 * len(f), 1))
        both = f[(f.p_train < .05) & (f.p_test < .05)]
        print("features with p<0.05 in training AND held-out:")
        print(both.sort_values("share_train", ascending=False).round(4).to_string(index=False))
        g = gd[gd.view == v].sort_values("p_train")
        print("groups (lobes, then bilateral pairs with train p<0.1):")
        print(pd.concat([g[g.group.str.startswith("lobe")], g[g.group.str.startswith("bilateral") & (g.p_train < .1)]])
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
