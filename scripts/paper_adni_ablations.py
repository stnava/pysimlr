#!/usr/bin/env python3
"""Generate the paper's ADNI-MCI ablation caches from the current code.

Writes, into ``--out`` (default ``paper/results_cache``):

    ablation_w_sweep.csv                  appendix_operational tbl/fig w-sweep
    ablation_k_sparsity_sweep.csv         appendix_operational fig-rowmean-k-ablation (B)
    ablation_quantile_sparseness.csv      appendix_operational tbl-app-quantile-grid
    ablation_rowmean_ica_positivity.csv   appendix_operational tbl-app-rowmean-ica
    adni_mci_axis_alignment_controls.csv  03_experiments ADNI-MCI controls
    adni_mci_feature_loadings.json        03_experiments fig-adni-neuroanatomy-flagship
    provenance_adni_ablations.json        CLI, UTC times, host, versions

These six caches were previously committed without any generator; this
script replaces them (2026-11-01). Its results are not the earlier numbers.

Protocol B (shared by every ADNI experiment; the split and preprocessing of
`scripts/run_adni_mci_grid_sweep.py`, parameterised by seed):
  * views: 0 = regional volume, 1 = regional thickness, 2 = T1w DKT cortical
    thickness (62 regions), 3 = 11 cognitive scales (training only);
  * 70/30 split from ``np.random.default_rng(seed).permutation``;
  * age, sex and education residualized on the training split, then each view
    standardized with training statistics;
  * a method is fitted on the four training views (graph topology: each MRI
    view linked to the cognitive view); at test time the cognitive view is
    withheld and scores are ``Z_m = X_m V_m`` for the three MRI views;
  * CogPC1 R^2: OLS from the concatenated scores to the first principal
    component of the 11 (training-standardized) scales; Mean11 R^2: RidgeCV to
    all 11 scales, uniform average.

Basis summaries (over the three MRI views unless named):
  * D(V): `utils.orthogonality_defect(normalized=True)`, mean over views;
  * eff_rank, cond_num, col_overlap: `utils.basis_rank_report`, mean over views;
  * sparsity: fraction of entries with |v| < 1e-6; exact: fraction == 0;
    _v1: column 1 only; _v2_k: columns 2..k;
  * Hoyer: (sqrt(n) - ||v||_1/||v||_2) / (sqrt(n) - 1), mean over columns;
  * DKT active regions: entries of view 2's column j with |v| >= 1e-6.

Usage:
    python scripts/paper_adni_ablations.py [--out DIR] [--only NAME ...] [--workers N]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA as SkPCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score

import pysimlr
from pysimlr.benchmarks.metrics import axis_alignment_control_metrics, support_recovery_score
from pysimlr.benchmarks.runner import run_single_experiment
from pysimlr.benchmarks.synthetic_cases import build_case
from pysimlr.simlr import initialize_simlr
from pysimlr.utils import basis_rank_report, orthogonality_defect

ADNI_CSV = os.path.join(REPO, "data", "adni", "adni_mci_cognitive_merged.csv")
K = 3
LINEAR_ITERS, DEEP_EPOCHS = None, None
TOL = 1e-6
PATH_GRAPH = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}


# ---------------------------------------------------------------------------
# Protocol B data
# ---------------------------------------------------------------------------
def protocol_b(seed: int = 42, row_center: bool = False) -> dict:
    df = pd.read_csv(ADNI_CSV)
    prefixes = ("T1Hier_vol__", "T1Hier_thk__", "T1w__")
    cols = [[c for c in df.columns if c.startswith(p)] for p in prefixes]
    # Columns read e.g. "T1w__T1w_mean_left_insula"; the region name is what
    # remains after the second, measurement-specific prefix.
    names = [[re.sub(r"^(T1Hier_(vol|thk)_|T1w_mean_)", "", c[len(p):]) for c in cs]
             for p, cs in zip(prefixes, cols)]
    drop = {"Cog__MOCA", "Cog__EcogPtTotal", "Cog__EcogSPTotal"}
    cog_cols = [c for c in df.columns if c.startswith("Cog__") and c not in drop] + ["CDRSB"]
    n = len(df)
    sex = df["PTGENDER"].astype(str).str.upper().str.startswith("F").astype(float).values
    cov = np.column_stack([np.ones(n), df["AGE"].values, sex, df["PTEDUCAT"].values])
    perm = np.random.default_rng(seed).permutation(n)
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
    pinv = np.linalg.pinv(cov[tr])

    def prep(m):
        m = m.astype(np.float64)
        if row_center:
            # Per-subject mean over the view's features: computed within each
            # row, so it uses no other subject and cannot leak across the split.
            m = m - m.mean(axis=1, keepdims=True)
        b = pinv @ m[tr]
        a, c = m[tr] - cov[tr] @ b, m[te] - cov[te] @ b
        mu, sd = a.mean(0), a.std(0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        return (a - mu) / sd, (c - mu) / sd

    raw = [df[cs].values for cs in cols]
    x_tr, x_te, rowmean_tr, rowmean_te = [], [], [], []
    for m in raw:
        a, c = prep(m)
        x_tr.append(a); x_te.append(c)
        rm = m.astype(np.float64).mean(axis=1)
        rowmean_tr.append(rm[tr]); rowmean_te.append(rm[te])
    y_tr, y_te = prep(df[cog_cols].values)
    pca_y = SkPCA(n_components=1).fit(y_tr)
    return {"x_tr": x_tr, "x_te": x_te, "y_tr": y_tr, "y_te": y_te,
            "pc1_tr": pca_y.transform(y_tr).ravel(), "pc1_te": pca_y.transform(y_te).ravel(),
            "rowmean_tr": np.column_stack(rowmean_tr), "rowmean_te": np.column_stack(rowmean_te),
            "names": names}


def fit_views(data: dict) -> list:
    return [torch.tensor(x).float() for x in data["x_tr"]] + [torch.tensor(data["y_tr"]).float()]


def mri_basis(res: dict) -> list:
    return [np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, dtype=np.float64) for v in res["v"][:3]]


def evaluate(data: dict, vs: list) -> dict:
    z_tr = np.hstack([x @ v for x, v in zip(data["x_tr"], vs)])
    z_te = np.hstack([x @ v for x, v in zip(data["x_te"], vs)])
    ols = LinearRegression().fit(z_tr, data["pc1_tr"])
    rcv = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(z_tr, data["y_tr"])
    ols_rm = LinearRegression().fit(np.hstack([z_tr, data["rowmean_tr"]]), data["pc1_tr"])
    return {
        "cog_pc1_r2": float(r2_score(data["pc1_te"], ols.predict(z_te))),
        "cog_pc1_r2_with_rowmean": float(r2_score(
            data["pc1_te"], ols_rm.predict(np.hstack([z_te, data["rowmean_te"]])))),
        "mean11_r2": float(r2_score(data["y_te"], rcv.predict(z_te), multioutput="uniform_average")),
    }


def hoyer(v: np.ndarray) -> float:
    n = v.shape[0]
    vals = []
    for j in range(v.shape[1]):
        c = np.abs(v[:, j]); l2 = np.linalg.norm(c)
        vals.append(0.0 if l2 == 0 else (np.sqrt(n) - c.sum() / l2) / (np.sqrt(n) - 1))
    return float(np.mean(vals))


def basis_summary(vs: list) -> dict:
    stack = np.vstack(vs)
    rep = basis_rank_report([torch.tensor(v).float() for v in vs], vs[0].shape[1], warn=False)
    out = {
        "defect_D": float(np.mean([float(orthogonality_defect(torch.tensor(v), normalized=True)) for v in vs])),
        "eff_rank": float(np.mean(rep["effective_rank"])),
        "cond_num": float(np.mean(rep["condition_number"])),
        "col_overlap": float(np.mean(rep["max_column_overlap"])),
        "sparsity_all": float((np.abs(stack) < TOL).mean()),
        "sparsity_v1": float((np.abs(stack[:, 0]) < TOL).mean()),
        "sparsity_v2_k": float((np.abs(stack[:, 1:]) < TOL).mean()) if stack.shape[1] > 1 else float("nan"),
        "sparsity_exact_all": float((stack == 0).mean()),
        "sparsity_exact_v1": float((stack[:, 0] == 0).mean()),
        "hoyer_all": hoyer(stack),
        "hoyer_v1": hoyer(stack[:, :1]),
    }
    dkt = vs[2]
    for j in range(min(3, dkt.shape[1])):
        out[f"dkt_v{j + 1}_sparsity"] = float((np.abs(dkt[:, j]) < TOL).mean())
        out[f"dkt_v{j + 1}_active_regions"] = int((np.abs(dkt[:, j]) >= TOL).sum())
    return out


def fit_adni(data: dict, method: str, k: int = K, **params) -> tuple:
    budget_by_unit = {"iterations": LINEAR_ITERS, "epochs": DEEP_EPOCHS}
    budget = budget_by_unit.get(pysimlr.describe_method(method)["budget_unit"])
    kw = dict(params)
    if budget is not None:
        kw.setdefault("iterations", budget)
    t0 = time.perf_counter()
    if pysimlr.resolve_method(method).family == "baseline":
        res = pysimlr.fit(method, [torch.tensor(x).float() for x in data["x_tr"]], k, **kw)
    else:
        kw.setdefault("topology", "graph"); kw.setdefault("path_graph", PATH_GRAPH)
        res = pysimlr.fit(method, fit_views(data), k, **kw)
    return res, time.perf_counter() - t0


def synthetic_f1(**params) -> dict:
    """NonnegParts (seed 42), the benchmark's case, through the benchmark runner."""
    case = build_case(seed=42, kind="nonneg_parts", private_scale=0.5)
    kw_exp = dict(params)
    if LINEAR_ITERS is not None:
        kw_exp["iterations"] = LINEAR_ITERS
    t0 = time.perf_counter()
    out = run_single_experiment("simlr", case, seed=42, **kw_exp)
    secs = time.perf_counter() - t0
    vs = [np.asarray(v.detach().cpu(), dtype=np.float64) for v in out["result"]["v"]]
    stack = np.vstack(vs)
    rep = basis_rank_report(out["result"]["v"], vs[0].shape[1], warn=False)
    return {"f1": float(support_recovery_score(out["result"]["v"], case["true_v"])),
            "r2": float(out["metrics"].get("test_r2", float("nan"))),
            "defect": float(np.mean([float(orthogonality_defect(torch.tensor(v), normalized=True)) for v in vs])),
            "eff_rank": float(np.mean(rep["effective_rank"])),
            "cond": float(np.mean(rep["condition_number"])),
            "overlap": float(np.mean(rep["max_column_overlap"])),
            "sparsity_ratio": float((np.abs(stack) < TOL).mean()), "secs": secs}


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def w_sweep() -> pd.DataFrame:
    """`simlr`, V >= 0, k = 3, no quantile threshold: w alone sets sparsity."""
    data = protocol_b(42)
    rows = []
    for w in (0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95):
        s = synthetic_f1(nsa_w=w, positivity="positive", sparseness=None)
        res, secs = fit_adni(data, "simlr", nsa_w=w, positivity="positive", sparseness=None,
                             random_state=42)
        vs = mri_basis(res); ev = evaluate(data, vs); b = basis_summary(vs)
        rows.append({"w": w, "Synth_Support_F1": s["f1"], "Synth_R2": s["r2"], "Synth_Defect_D": s["defect"],
                     "Synth_EffRank": s["eff_rank"], "Synth_CondNum": s["cond"], "Synth_ColOverlap": s["overlap"],
                     "Synth_Sparsity": s["sparsity_ratio"], "Synth_Fit_Sec": s["secs"],
                     "ADNI_CogPC1_R2": ev["cog_pc1_r2"], "ADNI_Mean11_R2": ev["mean11_r2"],
                     "ADNI_Defect_D": b["defect_D"], "ADNI_EffRank": b["eff_rank"], "ADNI_CondNum": b["cond_num"],
                     "ADNI_ColOverlap": b["col_overlap"], "ADNI_Sparsity_All": b["sparsity_all"],
                     "ADNI_Sparsity_v1": b["sparsity_v1"], "ADNI_Sparsity_v2_k": b["sparsity_v2_k"],
                     "ADNI_Fit_Sec": secs})
        print(f"  w={w:.2f} F1={s['f1']:.3f} ADNI R2={ev['cog_pc1_r2']:.4f}", flush=True)
    return pd.DataFrame(rows)


def k_sweep() -> pd.DataFrame:
    """Unthresholded non-negative `simlr` (w = 0.5) at k in {2,3,5,8,12}:
    column-centred with a PCA start (varimax off) vs row-centred with an ICA start."""
    rows = []
    for regime, row_center, init in (("Col-Centered (PCA Init)", False, "pca"),
                                     ("Row-Centered (ICA Init)", True, "ica")):
        data = protocol_b(42, row_center=row_center)
        for k in (2, 3, 5, 8, 12):
            res, secs = fit_adni(data, "simlr", k=k, positivity="positive", sparseness=None,
                                 initialization_type=init, varimax=False, random_state=42)
            vs = mri_basis(res); ev = evaluate(data, vs); b = basis_summary(vs)
            pca, _ = fit_adni(data, "pca", k=k)
            pca_r2 = evaluate(data, mri_basis(pca))["cog_pc1_r2"]
            rows.append({"k": k, "Theoretical_Disjoint_Sparsity": 1 - 1 / k, "Regime": regime,
                         "Sparsity_All": b["sparsity_all"], "Sparsity_v1": b["sparsity_v1"],
                         "Sparsity_v2_k": b["sparsity_v2_k"], "Sparsity_Exact_All": b["sparsity_exact_all"],
                         "Sparsity_Exact_v1": b["sparsity_exact_v1"], "Hoyer_All": b["hoyer_all"],
                         "Hoyer_v1": b["hoyer_v1"], "Defect_D": b["defect_D"], "EffRank": b["eff_rank"],
                         "ColOverlap": b["col_overlap"], "CogPC1_R2_XV": ev["cog_pc1_r2"],
                         "CogPC1_R2_WithRowMean": ev["cog_pc1_r2_with_rowmean"], "PCA_CogPC1_R2": pca_r2,
                         "Mean11_Ridge_R2": ev["mean11_r2"], "Fit_Sec": secs})
            print(f"  {regime} k={k} sparsity={b['sparsity_all']:.3f} R2={ev['cog_pc1_r2']:.4f}", flush=True)
    return pd.DataFrame(rows)


def quantile_sweep() -> pd.DataFrame:
    """`simlr` (w = 0.5, k = 3, default varimax) across quantile sparseness."""
    data = protocol_b(42)
    rows = []
    for pos in ("positive", "either"):
        for q in (None, 0.50, 0.70, 0.80, 0.85, 0.90):
            t0 = time.perf_counter()
            res, _ = fit_adni(data, "simlr", positivity=pos, sparseness=q, random_state=42)
            secs = time.perf_counter() - t0
            vs = mri_basis(res); ev = evaluate(data, vs); b = basis_summary(vs)
            s = synthetic_f1(positivity=pos, sparseness=q)
            rows.append({"positivity": pos, "sparseness": "None (NSA-Flow only)" if q is None else f"{q:.2f}",
                         "sparseness_val": -1.0 if q is None else q, "cog_pc1_r2": ev["cog_pc1_r2"],
                         "mean_11scale_r2": ev["mean11_r2"], "synth_support_f1": s["f1"],
                         **{kk: b[kk] for kk in ("defect_D", "eff_rank", "col_overlap", "sparsity_all",
                                                 "sparsity_v1", "sparsity_v2_k", "sparsity_exact_all",
                                                 "sparsity_exact_v1", "hoyer_all", "hoyer_v1",
                                                 "dkt_v1_sparsity", "dkt_v2_sparsity", "dkt_v3_sparsity",
                                                 "dkt_v1_active_regions", "dkt_v2_active_regions",
                                                 "dkt_v3_active_regions")},
                         "runtime_s": secs})
            print(f"  {pos} q={q} R2={ev['cog_pc1_r2']:.4f} sparsity={b['sparsity_all']:.3f}", flush=True)
    return pd.DataFrame(rows)


def rowmean_factorial() -> pd.DataFrame:
    """2 x 2 x 2 x 2: column vs row+column centring, PCA vs ICA start, signed vs
    V >= 0, the initializer alone vs the fitted `simlr` (w = 0.5, k = 3,
    varimax off, no quantile threshold)."""
    rows = []
    for prep_lbl, row_center in (("Col-Centered (Standard)", False),
                                 ("Row+Col Centered (Row-Mean Controlled)", True)):
        data = protocol_b(42, row_center=row_center)
        for init in ("pca", "ica"):
            for pos_lbl, pos in (("Signed (Unconstrained)", "either"), ("Non-Negative (V >= 0)", "positive")):
                for stage in ("Init-Only (Single-View)", "Bipartite SiMR (Fitted)"):
                    t0 = time.perf_counter()
                    if stage.startswith("Init"):
                        views = [torch.tensor(x).float() for x in data["x_tr"]]
                        vs = [np.asarray(v, dtype=np.float64) for v in initialize_simlr(
                            views, K, initialization_type=init, positivity=pos, sparseness=None,
                            varimax=False, generator=torch.Generator().manual_seed(42))]
                    else:
                        res, _ = fit_adni(data, "simlr", positivity=pos, sparseness=None,
                                          initialization_type=init, varimax=False, random_state=42)
                        vs = mri_basis(res)
                    secs = time.perf_counter() - t0
                    ev = evaluate(data, vs); b = basis_summary(vs)
                    rows.append({"Preprocessing": prep_lbl, "Init": init.upper(), "Positivity": pos_lbl,
                                 "Stage": stage, "k": K, "w": 0.5, "CogPC1_R2_XV": ev["cog_pc1_r2"],
                                 "CogPC1_R2_WithRowMean": ev["cog_pc1_r2_with_rowmean"],
                                 "Mean11_Ridge_R2": ev["mean11_r2"],
                                 **{kk: b[kk] for kk in ("defect_D", "eff_rank", "cond_num", "col_overlap",
                                                         "sparsity_all", "sparsity_v1", "sparsity_v2_k",
                                                         "sparsity_exact_all", "sparsity_exact_v1",
                                                         "hoyer_all", "hoyer_v1")},
                                 "Fit_Seconds": secs})
                    print(f"  {prep_lbl[:3]} {init} {pos} {stage[:4]} R2={ev['cog_pc1_r2']:.4f}", flush=True)
    return pd.DataFrame(rows)


CONTROL_MODELS = (("Flow-SiMR-V", "flow_v"), ("LEND", "lend"), ("NEDPP", "ned_shared_private"),
                  ("SiMLR", "simlr"), ("NSAFlow-PCA", "nsa_pipeline"),
                  ("PCA", "pca"))


def _control_task(task):
    label, method, pos, seed = task
    torch.set_num_threads(1)
    data = protocol_b(seed)
    params = {} if pysimlr.resolve_method(method).family == "baseline" else \
        {"positivity": pos, "sparseness": 0.5, "random_state": seed}
    res, _ = fit_adni(data, method, **params)
    joint = bool(res.get("basis_is_joint"))
    m = axis_alignment_control_metrics(
        [torch.tensor(x).float() for x in data["x_tr"]], [torch.tensor(x).float() for x in data["x_te"]],
        res["v"][:3], data["pc1_tr"], data["pc1_te"], is_classification=False, random_state=seed,
        basis_is_joint=joint)
    conv = res.get("convergence", {})
    return {
        **{k: float(v) for k, v in m.items()},
        "Model": label, "Positivity": pos, "Seed": seed,
        "stop_reason": conv.get("stop_reason", "converged" if conv.get("converged", True) else "max_iter"),
        "converged": conv.get("converged", True),
    }


def axis_controls(workers: int) -> pd.DataFrame:
    """Benchmark sparsity (q = 0.5) per model, both positivities (baselines
    have none), seeds 42-44; `benchmarks.metrics.axis_alignment_control_metrics`
    against CogPC1."""
    tasks = []
    for label, method in CONTROL_MODELS:
        for pos in (("either",) if pysimlr.resolve_method(method).family == "baseline" else ("either", "positive")):
            for seed in (42, 43, 44):
                tasks.append((label, method, pos, seed))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_control_task, tasks))
    return pd.DataFrame(rows)


def feature_loadings() -> dict:
    """Bases of the MRI views for the figure: joint PCA blocks, unrotated and
    varimax-rotated non-negative `simlr` (q = None), the row-centred ICA
    start, and non-negative q in {0.80, 0.85, 0.90} (seed 42)."""
    col = protocol_b(42); row = protocol_b(42, row_center=True)
    out = {f"view{i}_names": col["names"][i] for i in (0, 1, 2)}

    def store(key, data, res_or_vs, extra=False):
        if isinstance(res_or_vs, dict) and "loadings" in res_or_vs:
            vs = [np.asarray(l.detach().cpu() if hasattr(l, "detach") else l, dtype=np.float64) for l in res_or_vs["loadings"][:3]]
        elif isinstance(res_or_vs, list):
            vs = res_or_vs
        else:
            vs = mri_basis(res_or_vs)
        for i in (0, 2):
            out[f"view{i}_{key}"] = vs[i].tolist()
        return vs

    res_pca, _ = fit_adni(col, "pca")
    store("pca", col, res_pca)
    res_col, _ = fit_adni(col, "simlr", positivity="positive", sparseness=None,
                          varimax=False, random_state=42)
    store("col_pca_simr", col, res_col)
    res_vmx, _ = fit_adni(col, "simlr", positivity="positive", sparseness=None,
                          varimax=True, random_state=42)
    store("varimax_simr", col, res_vmx)
    res_ica, _ = fit_adni(row, "simlr", positivity="positive", sparseness=None,
                          initialization_type="ica", varimax=False, random_state=42)
    store("row_ica_simr", row, res_ica)

    res_q85 = None
    for q in (0.80, 0.85, 0.90):
        res, _ = fit_adni(col, "simlr", positivity="positive", sparseness=q, random_state=42)
        vs = [np.asarray(l.detach().cpu() if hasattr(l, "detach") else l, dtype=np.float64) for l in res["loadings"][:3]]
        ev = evaluate(col, mri_basis(res)); b = basis_summary(vs)
        if abs(q - 0.85) < 1e-9:
            res_q85 = res
            store("col_pca_q85_simr", col, res)
        dkt = vs[2]
        comps = []
        for j in range(dkt.shape[1]):
            order = np.argsort(-np.abs(dkt[:, j]))
            comps.append({"component": j + 1, "active_regions": int((np.abs(dkt[:, j]) >= TOL).sum()),
                          "regions": [[col["names"][2][i], float(dkt[i, j])] for i in order
                                      if abs(dkt[i, j]) >= TOL]})
        margin_dkt = res.get("loading_margin", [None, None, None])[2]
        mean_margin = float(margin_dkt.mean()) if margin_dkt is not None else float("nan")
        conv = res.get("convergence", {})
        out[f"Col-Centered + PCA + Non-Neg + Quantile (q={q:.2f})"] = {
            "r2_pc1": ev["cog_pc1_r2"], "r2_mean11": ev["mean11_r2"], "defect_D": b["defect_D"],
            "sparsity_all": b["sparsity_all"], "sparsity_v1": b["sparsity_v1"],
            "sparsity_v2_k": b["sparsity_v2_k"], "view": "T1w Cortical Thickness (62 DKT Regions)",
            "mean_loading_margin": mean_margin,
            "stop_reason": conv.get("stop_reason", "converged" if conv.get("converged", True) else "max_iter"),
            "converged": conv.get("converged", True),
            "components": comps}
    summaries = {}
    for key, data in (("pca", col), ("col_pca_simr", col), ("varimax_simr", col), ("row_ica_simr", row),
                      ("col_pca_q85_simr", col)):
        vs = [np.asarray(out[f"view{i}_{key}"]) if i in (0, 2) else None for i in (0, 1, 2)]
        dkt = np.asarray(out[f"view2_{key}"])
        summaries[key] = {"dkt_sparsity": float((np.abs(dkt) < TOL).mean()),
                          "dkt_active_regions": [int((np.abs(dkt[:, j]) >= TOL).sum()) for j in range(dkt.shape[1])]}
    out["summaries"] = summaries
    for key, data, res_obj in (("pca", col, res_pca),
                               ("col_pca_simr", col, res_col),
                               ("varimax_simr", col, res_vmx),
                               ("row_ica_simr", row, res_ica)):
        summaries[key]["cog_pc1_r2"] = evaluate(data, mri_basis(res_obj))["cog_pc1_r2"]
        conv = res_obj.get("convergence", {})
        summaries[key]["stop_reason"] = conv.get("stop_reason", "converged" if conv.get("converged", True) else "max_iter")
        summaries[key]["converged"] = conv.get("converged", True)
    summaries["col_pca_q85_simr"]["cog_pc1_r2"] = out["Col-Centered + PCA + Non-Neg + Quantile (q=0.85)"]["r2_pc1"]
    if res_q85 is not None:
        conv_q85 = res_q85.get("convergence", {})
        summaries["col_pca_q85_simr"]["stop_reason"] = conv_q85.get("stop_reason", "converged" if conv_q85.get("converged", True) else "max_iter")
        summaries["col_pca_q85_simr"]["converged"] = conv_q85.get("converged", True)
    return out


EXPERIMENTS = {
    "w": ("ablation_w_sweep.csv", w_sweep),
    "k": ("ablation_k_sparsity_sweep.csv", k_sweep),
    "quantile": ("ablation_quantile_sparseness.csv", quantile_sweep),
    "rowmean": ("ablation_rowmean_ica_positivity.csv", rowmean_factorial),
    "controls": ("adni_mci_axis_alignment_controls.csv", None),
    "loadings": ("adni_mci_feature_loadings.json", feature_loadings),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "results_cache"))
    ap.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS), default=sorted(EXPERIMENTS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    start = datetime.datetime.utcnow().isoformat() + "Z"
    timings = {}
    for name in a.only:
        fname, fn = EXPERIMENTS[name]
        print(f"[{name}] -> {fname}", flush=True)
        t0 = time.perf_counter()
        result = axis_controls(a.workers) if name == "controls" else fn()
        path = os.path.join(a.out, fname)
        if fname.endswith(".json"):
            with open(path, "w") as f:
                json.dump(result, f)
        else:
            result.to_csv(path, index=False)
        timings[name] = time.perf_counter() - t0
    prov = {"cli": " ".join(sys.argv), "start_utc": start,
            "end_utc": datetime.datetime.utcnow().isoformat() + "Z", "host": platform.node(),
            "python": sys.version.split()[0], "torch": torch.__version__, "pysimlr": pysimlr.__version__,
            "device": "cpu", "experiments": a.only, "seconds": timings}
    with open(os.path.join(a.out, "provenance_adni_ablations.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
