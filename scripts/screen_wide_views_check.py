"""Does `screen_wide_views` cure the wide-view overfit (D25)? Sweep of `max_width_ratio` (D30).

Data: Franzosa et al. 2019 IBD (Nat Microbiol 4:293), supplementary datasets 2 (per-subject
metabolomics; four LC-MS platforms) and 4 (metagenomic species), 220 subjects. Pass the directory
holding s4.xlsx and s6.xlsx (MOESM4 / MOESM6) with --ibd-dir.

Protocol: 5 outer 75/25 splits. The screen sees the training views only (no outcome), and its
indices are applied to the test views. Then default linear SiMLR (k = 5). Measured: mean pairwise
cross-view |corr| of matched components on training and held-out rows (the statistic of
`pysimlr.heldout_coupling`), and held-out balanced accuracy for IBD vs control from the consensus u
(SiMLR) and from 5 concatenated-PCA scores. Reference selection: top-500 training variance per view.

Rules (fixed before running):
  default : the largest ratio with mean gap <= 0.10 and SiMLR accuracy within 0.02 of the best ratio.
  merge   : (i) IBD gap reduced >= 50% vs top-500, (ii) SiMLR accuracy no more than 0.02 below
            top-500, (iii) pure-noise views (3 x 120 features, n_train = 50) gap < 0.10.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
OUT = os.path.join(HERE, "..", "docs", "audit", "evidence", "screen_wide_views_check.csv")
RATIOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75)


def load_ibd(d):
    met = pd.read_excel(os.path.join(d, "s4.xlsx"), header=1); met = met.set_index(met.columns[0])
    mic = pd.read_excel(os.path.join(d, "s6.xlsx"), header=1); mic = mic.set_index(mic.columns[0])
    meta_rows = ["Age", "Diagnosis", "Fecal.Calprotectin", "antibiotic", "immunosuppressant", "mesalamine", "steroids"]
    feat = met.drop(index=meta_rows).astype(float).T
    sp = mic.drop(index=[i for i in mic.index if i in meta_rows + ["SRA_metagenome_name"]]).astype(float).T.loc[feat.index]
    xs = []
    for plat in ("C18-neg", "C8-pos", "HILIC-neg", "HILIC-pos"):
        a = feat[[c for c in feat if c.startswith(plat)]]
        a = a.loc[:, (a > 0).mean() >= 0.5]
        xs.append(np.log(a.clip(lower=a[a > 0].min().min() / 2).to_numpy(float)))
    s = sp.loc[:, (sp > 0).mean() >= 0.1]
    xs.append(np.log(s.to_numpy(float) + 1e-5))
    y = (met.loc["Diagnosis"].astype(str).to_numpy() != "Control").astype(int)
    return xs, y


def zs(a, b):
    mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
    return (a - mu) / sd, (b - mu) / sd


def cross(lat):
    return float(np.mean([abs(np.corrcoef(lat[a][:, j], lat[b][:, j])[0, 1])
                          for a in range(len(lat)) for b in range(a + 1, len(lat)) for j in range(lat[0].shape[1])]))


def run_one(xs, y, tr, te, idx, seed):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import balanced_accuracy_score

    import pysimlr
    A = [zs(x[tr][:, i], x[te][:, i]) for x, i in zip(xs, idx)]
    xtr, xte = [a for a, _ in A], [b for _, b in A]
    T = lambda X: [torch.tensor(a).float() for a in X]
    res = pysimlr.fit("simlr", T(xtr), 5, random_state=seed)
    V = [np.asarray(v.detach(), float) for v in res["v"]]
    out = dict(cross_train=cross([a @ v for a, v in zip(xtr, V)]), cross_test=cross([a @ v for a, v in zip(xte, V)]))
    if y is not None:
        def acc(ftr, fte):
            mu, sd = ftr.mean(0), ftr.std(0); sd[sd == 0] = 1
            return balanced_accuracy_score(y[te], LogisticRegressionCV(max_iter=3000, Cs=8).fit((ftr - mu) / sd, y[tr]).predict((fte - mu) / sd))
        u = lambda X: np.asarray(pysimlr.predict(res, T(X))["u"].detach(), float)
        C, Ct = np.hstack(xtr), np.hstack(xte)
        _, _, vt = np.linalg.svd(C - C.mean(0), full_matrices=False)
        out.update(simlr_acc=acc(u(xtr), u(xte)), pca_acc=acc(C @ vt[:5].T, Ct @ vt[:5].T))
    return out


def main():
    import pysimlr
    ap = argparse.ArgumentParser(); ap.add_argument("--ibd-dir", required=True); ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    warnings.simplefilter("ignore"); torch.set_num_threads(8)
    t0 = time.time(); rows = []
    xs, y = load_ibd(a.ibd_dir)
    for seed in range(5):
        n = len(y); p = np.random.default_rng(seed).permutation(n); te, tr = p[: n // 4], p[n // 4:]
        sel = {"top-500 variance": [np.argsort(-x[tr].var(0))[:500] if x.shape[1] > 500 else np.arange(x.shape[1]) for x in xs]}
        for r in RATIOS:
            sc = pysimlr.screen_wide_views([x[tr] for x in xs], k=5, max_width_ratio=r, random_state=seed)
            sel[f"screen {r}"] = sc["indices"]
        for name, idx in sel.items():
            rows.append(dict(data="IBD", selection=name, seed=seed, n_train=len(tr), widths=str([len(i) for i in idx]),
                             **run_one(xs, y, tr, te, idx, seed)))
        print(f"IBD split {seed} [{time.time() - t0:.0f}s]", flush=True)
    for seed in range(5):                                   # pure-noise control
        g = np.random.default_rng(100 + seed)
        nx = [g.standard_normal((450, 120)) for _ in range(3)]
        tr, te = np.arange(50), np.arange(50, 450)
        sel = {"all features": [np.arange(120)] * 3}
        for r in RATIOS:
            sel[f"screen {r}"] = pysimlr.screen_wide_views([x[tr] for x in nx], k=5, max_width_ratio=r, random_state=seed)["indices"]
        for name, idx in sel.items():
            rows.append(dict(data="noise", selection=name, seed=seed, n_train=50, widths=str([len(i) for i in idx]),
                             **run_one(nx, None, tr, te, idx, seed)))
    df = pd.DataFrame(rows); df["gap"] = df.cross_train - df.cross_test
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); df.to_csv(a.out, index=False)
    s = df.groupby(["data", "selection"])[["cross_train", "cross_test", "gap", "simlr_acc", "pca_acc"]].mean()
    pd.set_option("display.width", 200); print(s.round(3).to_string())
    ibd = s.loc["IBD"]; noise = s.loc["noise"]
    scr = ibd[ibd.index.str.startswith("screen")]
    best_acc = scr.simlr_acc.max()
    ok = scr[(scr.gap <= 0.10) & (scr.simlr_acc >= best_acc - 0.02)]
    default = max(float(i.split()[1]) for i in ok.index) if len(ok) else None
    print("DEFAULT RULE:", f"max_width_ratio = {default}" if default is not None else "no ratio qualifies")
    ref = ibd.loc["top-500 variance"]
    pick = f"screen {default if default is not None else 0.75}"
    c1 = ibd.loc[pick, "gap"] <= 0.5 * ref.gap
    c2 = ibd.loc[pick, "simlr_acc"] >= ref.simlr_acc - 0.02
    c3 = noise.loc[pick, "gap"] < 0.10
    print(f"MERGE RULE at {pick}: gap halved {c1} ({ibd.loc[pick, 'gap']:.3f} vs {ref.gap:.3f}); "
          f"SiMLR acc kept {c2} ({ibd.loc[pick, 'simlr_acc']:.3f} vs {ref.simlr_acc:.3f}); noise gap < 0.10 {c3} "
          f"({noise.loc[pick, 'gap']:.3f}) -> {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
