"""Held-out benchmark of the confound sink (`pysimlr.fit(..., confounds=...)`) on real data.

Datasets (never adni_loader's T1/DTI/rsfMRI or PPMI cases):

* ``adni_mci``: the paper's ADNI-MCI Protocol B views (T1Hier volume, T1Hier
  thickness, T1w mean intensity; ``data/adni/adni_mci_cognitive_merged.csv``),
  confounds age, sex, education and head size (sum of the regional volumes),
  outcome CogPC1 (first PC of the cognitive battery, fitted on training rows).
* ``nhanes``: NHANES 2015-16 + 2017-18 adults (cleaned public release in
  ``~/Downloads/cleaned_nhanes_21743372``), views labs / body measures /
  2-day mean diet, confounds age, sex, race/ethnicity, poverty-income ratio,
  education and survey cycle, outcome HbA1c (glucose, fasting glucose, insulin
  and osmolality are removed from the labs view: they measure or leak it).

Adjustments, each fitted with ``simlr`` and with the ``pca`` baseline:

* ``unadjusted``: no confound handling.
* ``preresid_train_only``: training views residualised on C (OLS); test views
  projected raw -- what pre-residualisation gives when ``C_test`` is not used.
* ``frozen_beta``: training views residualised on C; test views residualised
  with the *training* coefficients and each test subject's own ``C_test``
  (works at N = 1, needs ``C_test``).
* ``sink_1.0`` / ``sink_0.5``: ``confounds=C_train``, ``confound_strength`` 1 / 0.5;
  prediction uses no ``C_test``.

Metrics per split (all on held-out rows): max and mean |r(u, C)| of
``predict(...)["u"]``; held-out cross-view coupling (`heldout_coupling`,
per-view methods only); random-forest R^2 of the outcome from ``u`` (fitted on
training ``u``), for the raw outcome and for the outcome adjusted for C with
training coefficients (evaluation only).

Decision rules, fixed before the runs (also written to the JSON):

R1 (decorrelation without C_test): sink_1.0's held-out max |r(u, C)| is below
    unadjusted in >= 90% of splits and its median is <= 0.15.
R2 (decorrelation vs frozen_beta, which uses C_test): median paired difference
    sink_1.0 - frozen_beta in held-out max |r(u, C)| <= 0.05.
R3 (outcome cost vs frozen_beta, adjusted outcome): mean paired difference in
    RF R^2 >= -0.02, Wilcoxon one-sided p (sink worse) > 0.05, and sink worse
    by more than 0.05 in <= 20% of splits.
Coupling is reported for every method and not ruled on.

Usage: ``python scripts/benchmark_confound_sink.py --dataset adni_mci --splits 20 --jobs 8``.
Writes ``docs/audit/evidence/confound_sink_<dataset>.{csv,json}``.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "audit" / "evidence"
ADNI_CSV = Path(os.environ.get("ADNI_CSV", str(REPO / "data" / "adni" / "adni_mci_cognitive_merged.csv")))
NHANES_DIR = Path(os.path.expanduser(os.environ.get("NHANES_DIR", "~/Downloads/cleaned_nhanes_21743372")))
ADJUSTMENTS = ("unadjusted", "preresid_train_only", "frozen_beta", "sink_1.0", "sink_0.5")
RULES = {
    "R1": "sink_1.0 held-out max|r(u,C)| < unadjusted in >= 90% of splits and median <= 0.15",
    "R2": "median paired (sink_1.0 - frozen_beta) held-out max|r(u,C)| <= 0.05",
    "R3": "adjusted-outcome RF R^2, sink_1.0 - frozen_beta: mean >= -0.02, one-sided Wilcoxon p > 0.05, "
          "worse by > 0.05 in <= 20% of splits",
}


# ----------------------------------------------------------------------------- data
def load_adni():
    df = pd.read_csv(ADNI_CSV)
    prefixes = ("T1Hier_vol__", "T1Hier_thk__", "T1w__")
    cols = [[c for c in df.columns if c.startswith(p)] for p in prefixes]
    drop = {"Cog__MOCA", "Cog__EcogPtTotal", "Cog__EcogSPTotal"}
    cog = [c for c in df.columns if c.startswith("Cog__") and c not in drop] + ["CDRSB"]
    # PTGENDER is coded 0/1 in this export (paper_adni_ablations.protocol_b's
    # string test for "F" turns it into a constant column).
    sex = pd.to_numeric(df["PTGENDER"], errors="raise").astype(float)
    head = df[cols[0]].sum(axis=1)
    conf = np.column_stack([df["AGE"], sex, df["PTEDUCAT"], head]).astype(float)
    views = [df[c].to_numpy(float) for c in cols]
    y = df[cog].to_numpy(float)
    ok = np.isfinite(conf).all(1) & np.isfinite(y).all(1) & np.all([np.isfinite(v).all(1) for v in views], axis=0)
    return {"views": [v[ok] for v in views], "C": conf[ok], "Y": y[ok], "outcome": "CogPC1",
            "view_names": ["T1Hier_vol", "T1Hier_thk", "T1w_mean"], "confound_names": ["age", "sex", "educ", "head_size"],
            "k": 3}


def load_nhanes(max_n: int = 3000, seed: int = 0):
    resp_h = pd.read_csv(NHANES_DIR / "response_clean.csv", nrows=0).columns
    leak = {"LBXGH", "LBXSGL", "LBXGLU", "LBXGLUSI", "LBXSOSSI", "LBXIN", "LBDINSI", "LBDINLC"}
    labs = [c for c in resp_h if c.startswith(("LBX", "LBD")) and not c[-1].isdigit() and c not in leak]
    body = [c for c in resp_h if c.startswith("BMX") and not c[-1].isdigit()]
    resp = pd.read_csv(NHANES_DIR / "response_clean.csv", usecols=["SEQN", "SDDSRVYR", "LBXGH"] + labs + body,
                       low_memory=False)
    resp = resp[resp.SDDSRVYR.isin([9, 10])].drop_duplicates("SEQN")
    diet_h = pd.read_csv(NHANES_DIR / "dietary_clean.csv", nrows=0).columns
    diet_cols = [c for c in diet_h if c.startswith("DRXT")]
    diet = pd.read_csv(NHANES_DIR / "dietary_clean.csv", usecols=["SEQN", "SDDSRVYR"] + diet_cols, low_memory=False)
    diet = diet[diet.SDDSRVYR.isin([9, 10])].drop(columns="SDDSRVYR").copy().groupby("SEQN").mean().reset_index()
    demo = pd.read_csv(NHANES_DIR / "demographics_clean.csv",
                       usecols=["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "SDDSRVYR"],
                       low_memory=False).drop_duplicates("SEQN")
    m = resp.merge(diet, on="SEQN").merge(demo.drop(columns="SDDSRVYR"), on="SEQN")
    m = m[m.RIDAGEYR >= 20]

    def keep(cs):
        return [c for c in cs if m[c].notna().mean() >= 0.9 and m[c].std() > 0]
    labs, body, diet_cols = keep(labs), keep(body), keep(diet_cols)
    cov = pd.get_dummies(m[["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "DMDEDUC2", "SDDSRVYR"]]
                         .astype({"RIAGENDR": "category", "RIDRETH1": "category", "DMDEDUC2": "category",
                                  "SDDSRVYR": "category"}), drop_first=True, dtype=float)
    ok = (m[labs + body + diet_cols + ["LBXGH"]].notna().all(1) & cov.notna().all(1)).to_numpy()
    idx = np.where(ok)[0]
    if len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_n, replace=False))
    views = [m[labs].to_numpy(float)[idx], m[body].to_numpy(float)[idx], m[diet_cols].to_numpy(float)[idx]]
    return {"views": views, "C": cov.to_numpy(float)[idx], "Y": m[["LBXGH"]].to_numpy(float)[idx],
            "outcome": "HbA1c", "view_names": ["labs", "body", "diet"], "confound_names": list(cov.columns),
            # The body view has 7 measures; all 14 confound columns would span it. It is
            # adjusted for age and sex only (columns 0 and 2), the other views for all.
            "view_confounds": [None, [0, 2], None],
            "k": 5, "n_complete": int(ok.sum())}


# ----------------------------------------------------------------------------- one split
def _ols_fit(c_tr, x_tr):
    d = np.column_stack([np.ones(len(c_tr)), c_tr])
    return np.linalg.lstsq(d, x_tr, rcond=None)[0]


def _ols_resid(c, x, beta):
    return x - np.column_stack([np.ones(len(c)), c]) @ beta


def _max_abs_corr(u, c):
    import pysimlr.confounds as pc
    return pc.max_abs_corr(u, c)


def _rf_r2(u_tr, y_tr, u_te, y_te, seed):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=1)
    return float(r2_score(y_te, rf.fit(u_tr, y_tr).predict(u_te)))


def run_split(args):
    data, split, family = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    from sklearn.decomposition import PCA

    import pysimlr
    rng = np.random.default_rng(1000 + split)
    n = len(data["C"])
    perm = rng.permutation(n)
    tr, te = np.sort(perm[: int(0.7 * n)]), np.sort(perm[int(0.7 * n):])
    xs_tr = [v[tr] for v in data["views"]]
    xs_te = [v[te] for v in data["views"]]
    c_tr, c_te = data["C"][tr], data["C"][te]
    # Outcome: first PC of the outcome block on training rows (CogPC1), or the single outcome.
    Y = data["Y"]
    if Y.shape[1] > 1:
        mu, sd = Y[tr].mean(0), Y[tr].std(0, ddof=1)
        pca = PCA(1).fit((Y[tr] - mu) / sd)
        y = pca.transform((Y - mu) / sd).ravel()
    else:
        y = Y[:, 0]
    y_tr, y_te = y[tr], y[te]
    by = _ols_fit(c_tr, y_tr)
    ya_tr, ya_te = _ols_resid(c_tr, y_tr, by), _ols_resid(c_te, y_te, by)
    kw = {"random_state": 0} if family == "simlr" else {}
    rows = []
    for adj in ADJUSTMENTS:
        t_tr = [torch.tensor(x).float() for x in xs_tr]
        t_te = [torch.tensor(x).float() for x in xs_te]
        # Per-view confound columns (the same sets for every adjustment).
        vc = data.get("view_confounds") or [None] * len(xs_tr)
        ctr_v = [c_tr if cols is None else c_tr[:, cols] for cols in vc]
        cte_v = [c_te if cols is None else c_te[:, cols] for cols in vc]
        if adj in ("preresid_train_only", "frozen_beta"):
            betas = [_ols_fit(c, x) for c, x in zip(ctr_v, xs_tr)]
            t_tr = [torch.tensor(_ols_resid(c, x, b)).float() for c, x, b in zip(ctr_v, xs_tr, betas)]
            if adj == "frozen_beta":
                t_te = [torch.tensor(_ols_resid(c, x, b)).float() for c, x, b in zip(cte_v, xs_te, betas)]
        conf = c_tr if data.get("view_confounds") is None else ctr_v
        sink = {"confounds": conf, "confound_strength": float(adj.split("_")[1])} if adj.startswith("sink") else {}
        res = pysimlr.fit(family, t_tr, data["k"], **kw, **sink)
        u_tr = res["u"].double().numpy()
        u_te = pysimlr.predict(res, t_te)["u"].double().numpy()
        dec = _max_abs_corr(u_te, c_te)
        try:
            coup = float(np.nanmean(pysimlr.heldout_coupling(res, t_te)["cross_view_heldout"]))
        except ValueError:
            coup = float("nan")          # joint-basis baselines have no per-view latents
        rows.append({"split": split, "family": family, "adjustment": adj,
                     "test_max_abs_corr_C": dec["max"], "test_mean_abs_corr_C": dec["mean"],
                     "train_max_abs_corr_C": _max_abs_corr(u_tr, c_tr)["max"],
                     "heldout_coupling": coup,
                     "rf_r2_outcome": _rf_r2(u_tr, y_tr, u_te, y_te, split),
                     "rf_r2_outcome_adjusted": _rf_r2(u_tr, ya_tr, u_te, ya_te, split)})
    return rows


# ----------------------------------------------------------------------------- rules
def evaluate_rules(df: pd.DataFrame) -> dict:
    from scipy.stats import wilcoxon
    out = {}
    for fam, g in df.groupby("family"):
        p = g.pivot(index="split", columns="adjustment")
        dec = p["test_max_abs_corr_C"]
        r2a = p["rf_r2_outcome_adjusted"]
        d_unadj = dec["sink_1.0"] - dec["unadjusted"]
        d_frozen = dec["sink_1.0"] - dec["frozen_beta"]
        d_r2 = r2a["sink_1.0"] - r2a["frozen_beta"]
        p_w = float(wilcoxon(d_r2, alternative="less").pvalue) if np.any(d_r2 != 0) else 1.0
        out[fam] = {
            "R1": {"frac_below_unadjusted": float((d_unadj < 0).mean()), "median_sink": float(dec["sink_1.0"].median()),
                   "pass": bool((d_unadj < 0).mean() >= 0.9 and dec["sink_1.0"].median() <= 0.15)},
            "R2": {"median_diff_vs_frozen": float(d_frozen.median()), "pass": bool(d_frozen.median() <= 0.05)},
            "R3": {"mean_diff_r2_adj_vs_frozen": float(d_r2.mean()), "wilcoxon_p_sink_worse": p_w,
                   "frac_worse_by_0.05": float((d_r2 < -0.05).mean()),
                   "pass": bool(d_r2.mean() >= -0.02 and p_w > 0.05 and (d_r2 < -0.05).mean() <= 0.2)},
            "coupling_sink_minus_unadjusted_median": float(
                (p["heldout_coupling"]["sink_1.0"] - p["heldout_coupling"]["unadjusted"]).median()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("adni_mci", "nhanes"), required=True)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--families", default="simlr,pca")
    a = ap.parse_args()
    data = load_adni() if a.dataset == "adni_mci" else load_nhanes()
    tasks = [(data, s, f) for f in a.families.split(",") for s in range(a.splits)]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for rs in ex.map(run_split, tasks) for r in rs]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"confound_sink_{a.dataset}.csv", index=False)
    summary = df.groupby(["family", "adjustment"]).agg(["mean", "std"]).drop(columns="split").round(4)
    meta = {"dataset": a.dataset, "n": len(data["C"]), "view_names": data["view_names"],
            "view_widths": [int(v.shape[1]) for v in data["views"]], "confound_names": data["confound_names"],
            "outcome": data["outcome"], "k": data["k"], "splits": a.splits, "rules": RULES,
            "view_confounds": data.get("view_confounds"), "n_complete": data.get("n_complete"),
            "rule_outcomes": evaluate_rules(df),
            "summary": {f"{f}|{adj}": {f"{c}|{s}": v for (c, s), v in row.items()}
                        for (f, adj), row in summary.iterrows()}}
    (OUT / f"confound_sink_{a.dataset}.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps({k: meta[k] for k in ("dataset", "n", "view_widths", "confound_names", "rule_outcomes")},
                     indent=1, default=float))
    print(summary.to_string())


if __name__ == "__main__":
    main()
