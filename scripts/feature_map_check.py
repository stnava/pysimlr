#!/usr/bin/env python3
"""Evidence for `pysimlr.feature_maps`: which alternative feature maps to recommend.

Two parts, written to docs/audit/evidence/feature_map_check.csv:

Planted (5 seeds x 2 designs, simlr k = 2, positivity="positive", 3 views, n = 300):
  shared signals s1, s2; views 1 and 2 are linear in (s1, s2) plus noise. View 0
  has 40 features; features 0-5 carry s1 and 6-11 carry s2 (the planted support).
  (a) no global factor; (b) every view-0 feature also loads a global factor g
  (loadings U(0.3, 3), drivers 0.5) correlated 0.7 with s1 + s2: the ADNI regime
  (PC1 70% of view-0 variance; ADNI DiReCT has 58%). Metric: view-0 support AUC of |map|
  (`optimizer_benchmark_synthetic.loadings_support_auc`, Hungarian-matched), and
  the map's mean view-0 component stability from loading_inference(n_subsamples=20).

ADNI-MCI (Protocol B, seed 42, simlr k = 3, non-negative, q = 0.85, as Figure 3.9):
  10 refits on random halves of the training rows. In each, the component most
  correlated with training CogPC1; per map, the top-10 DiReCT and vol/area
  features by |map|; held-out CogPC1 R^2 of RidgeCV on those features (326 test
  rows); top-10 overlap with the reference fit's top-10 (Jaccard).
  global_partial uses control="pc1" on DiReCT only (None elsewhere).

Rule (fixed before running; decides only what the usage notes recommend): a map
is recommended for views dominated by a global factor if its support AUC is
>= structure + 0.05 in >= 4/5 seeds of design (b), no worse than structure by
> 0.02 (mean) on design (a), and its mean component stability on (b) is >= 0.8.

Usage: python scripts/feature_map_check.py [--out CSV] [--workers N]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import pysimlr  # noqa: E402
from optimizer_benchmark_synthetic import loadings_support_auc  # noqa: E402

KINDS = ("structure", "partial", "global_partial", "component_partial", "relative_weight")
ALT = KINDS[1:]


def planted(seed: int, global_factor: bool, n: int = 300):
    r = np.random.default_rng(seed)
    s = r.standard_normal((n, 2))
    x0 = r.standard_normal((n, 40))
    x0[:, :6] += 0.6 * s[:, [0]]
    x0[:, 6:12] += 0.6 * s[:, [1]]
    if global_factor:
        g = 0.7 * (s.sum(1) / np.sqrt(2)) + np.sqrt(1 - 0.49) * r.standard_normal(n)
        b = r.uniform(0.3, 3.0, 40); b[:12] = 0.5
        x0 += g[:, None] * b
    x1 = s @ r.standard_normal((2, 25)) + 0.7 * r.standard_normal((n, 25))
    x2 = s @ r.standard_normal((2, 20)) + 0.7 * r.standard_normal((n, 20))
    sup = [set(range(6)), set(range(6, 12))]
    return [torch.tensor(x).float() for x in (x0, x1, x2)], sup


def planted_rows(workers: int):
    rows = []
    for design, gf in (("a_no_global", False), ("b_global", True)):
        for seed in range(5):
            xs, sup = planted(seed, gf)
            inf = pysimlr.loading_inference("simlr", xs, 2, n_subsamples=20, random_state=seed, n_jobs=workers,
                                            positivity="positive", maps=ALT)
            res = inf["result"]
            pc1 = np.linalg.svd(xs[0].numpy() - xs[0].numpy().mean(0), compute_uv=False) ** 2
            for kind in KINDS:
                m = (res["structure"][0] if kind == "structure" else inf["maps"][kind]["map"][0]).numpy()
                stab = (inf["component_stability"][0] if kind == "structure"
                        else inf["maps"][kind]["component_stability"][0])
                rows.append({"part": "planted", "design": design, "seed": seed, "map": kind,
                             "support_auc": loadings_support_auc(np.nan_to_num(m), sup),
                             "stability": float(np.nanmean(stab)), "view0_pc1_share": float(pc1[0] / pc1.sum()),
                             "converged": bool(res["convergence"].get("converged", True))})
            print(design, seed, {r_["map"]: round(r_["support_auc"], 3) for r_ in rows[-5:]}, flush=True)
    return rows


def adni_rows(n_refits: int = 10):
    import paper_adni_ablations as paa
    from sklearn.linear_model import RidgeCV
    d = paa.protocol_b(42)
    rng = np.random.default_rng(0)
    n_tr = len(d["pc1_tr"])
    names = {1: "vol_area", 2: "direct_thickness"}

    def one(idx):
        sub = dict(d, x_tr=[x[idx] for x in d["x_tr"]], y_tr=d["y_tr"][idx], pc1_tr=d["pc1_tr"][idx])
        res, _ = paa.fit_adni(sub, "simlr", k=3, positivity="positive", sparseness=0.85, random_state=42)
        views = paa.fit_views(sub)
        maps = pysimlr.feature_maps(res, views, kinds=KINDS, control=[None, None, "pc1", None])
        u = pysimlr.predict(res, views)["u"].double().numpy() @ res["loading_rotation"].double().numpy()
        cc = [abs(np.corrcoef(u[:, j], sub["pc1_tr"])[0, 1]) for j in range(u.shape[1])]
        j = int(np.argmax(cc))
        out = {}
        for m in (1, 2):
            for kind in KINDS:
                v = np.nan_to_num(np.abs(maps[kind][m][:, j].numpy()), nan=-1.0)
                top = np.argsort(-v)[:10]
                if kind == "global_partial" and m != 2:
                    out[(m, kind)] = (None, np.nan)
                    continue
                r = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(sub["x_tr"][m][:, top], sub["pc1_tr"])
                out[(m, kind)] = (set(top.tolist()), r.score(d["x_te"][m][:, top], d["pc1_te"]))
        return out, j

    ref, j_ref = one(np.arange(n_tr))
    rows = []
    for t in range(n_refits):
        idx = np.sort(rng.choice(n_tr, n_tr // 2, replace=False))
        got, j = one(idx)
        for (m, kind), (top, r2) in got.items():
            ref_top = ref[(m, kind)][0]
            jac = (len(top & ref_top) / len(top | ref_top)) if top is not None else np.nan
            rows.append({"part": "adni", "design": names[m], "seed": t, "map": kind, "heldout_r2_top10": r2,
                         "top10_jaccard_vs_full": jac, "component": j + 1})
        print("adni refit", t, flush=True)
    for (m, kind), (top, r2) in ref.items():
        rows.append({"part": "adni_full", "design": names[m], "seed": -1, "map": kind, "heldout_r2_top10": r2,
                     "component": j_ref + 1})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "audit", "evidence", "feature_map_check.csv"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = planted_rows(a.workers)
    pd.DataFrame(rows).to_csv(a.out, index=False)        # planted part saved before the ADNI part
    rows += adni_rows()
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    p = df[df.part == "planted"].pivot_table(index=["design", "seed"], columns="map", values="support_auc")
    st = df[(df.part == "planted") & (df.design == "b_global")].groupby("map").stability.mean()
    print("\nplanted support AUC (mean over seeds):\n", p.groupby(level=0).mean().round(3))
    print("\nstability on (b):\n", st.round(3))
    for kind in ALT:
        b = p.loc["b_global"]; aa = p.loc["a_no_global"]
        wins = int(((b[kind] - b["structure"]) >= 0.05).sum())
        loss = float((aa[kind] - aa["structure"]).mean())
        ok = wins >= 4 and loss >= -0.02 and st[kind] >= 0.8
        print(f"RULE {kind:18s} wins(b) {wins}/5  mean diff(a) {loss:+.3f}  stability(b) {st[kind]:.2f}  "
              f"-> {'RECOMMEND' if ok else 'available only'}")
    ad = df[df.part == "adni"].groupby(["design", "map"]).agg(r2=("heldout_r2_top10", "median"),
                                                              jaccard=("top10_jaccard_vs_full", "median"))
    print("\nADNI half-sample refits (median):\n", ad.round(3))
    print("\nanomaly scan: NaN AUC", int(df[df.part == "planted"].support_auc.isna().sum()),
          "| unconverged planted fits", int((~df[df.part == "planted"].converged.astype(bool)).sum()),
          "| exact-repeat R2 values", int(df[df.part == "adni"].heldout_r2_top10.dropna().duplicated().sum()))


if __name__ == "__main__":
    main()
