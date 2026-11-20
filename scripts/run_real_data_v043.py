#!/usr/bin/env python3
"""Real-data paper benchmark revisit for pysimlr v0.43.1.

Evaluates the 6 real-data datasets from pysimlr/paper/ (no synthetic datasets):
  - ADNI-MCI (N=1,084, 3 MRI views + withheld 11-scale cognitive view, FWL covars, k=3)
  - MultiOmics / TCGA-BRCA (N=822, mRNA/CNV/RPPA, ER status classification, k=3)
  - TCGA-KIRC (N=467, mRNA/CNV/RPPA, pathologic stage classification, k=3)
  - Heart (N=297, 2 views, binary classification, k=2)
  - Diabetes (N=442, 2 views, regression, k=2)
  - mfeat (N=2,000, 4 views, 10-class digit classification, k=3)

Covers:
  (a) Main 6 real-data linear `simlr` paper benchmark (exact splits/seeds/protocol of
      operating_points_05_08_benchmark.csv for Method=='simlr', sparseness in {None, 0.5, 0.8},
      positivity in {'either', 'positive', 'convex'}) vs cached paper/results_cache baselines.
  (b) New `fit_uniform_shrunk` / `project_uniform_shrunk` (automatic defaults:
      radial 'convex' & 'conical', k=1, k=2, and paper k) under identical CV splits.
  (c) Real-data paper ablation & loading spot-checks: ADNI-MCI k/quantile/rowmean ablations
      and Heart/Diabetes nsa_v_loadings vs cached paper/results_cache/.
  (d) Standalone HTML visual comparison report with embedded charts.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Guard against parent workspace cwd shadowing installed site-packages/nsa_flow
sys.path[:] = [
    _p
    for _p in sys.path
    if not (
        (Path(_p or ".").resolve() / "nsa_flow").is_dir()
        and not (Path(_p or ".").resolve() / "nsa_flow" / "__init__.py").is_file()
    )
]

REPO = str(Path(__file__).resolve().parents[1])
CACHE_DIR = os.path.join(REPO, "paper", "results_cache")
if Path.cwd().resolve() == Path(REPO).resolve().parent:
    os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import pysimlr
import run_deep_benchmark as rdb
from pysimlr.benchmarks.metrics import calculate_all_metrics, support_overlap
from pysimlr.benchmarks.runner import split_indices
from pysimlr.utils import basis_rank_report, orthogonality_defect


REAL_DATASETS = ["ADNI-MCI", "MultiOmics", "TCGA-KIRC", "Heart", "Diabetes", "mfeat"]
BIO_DATASETS = {"ADNI-MCI", "MultiOmics", "TCGA-KIRC"}


def _ensure_repo_cwd() -> None:
    if Path.cwd().resolve() == Path(REPO).resolve().parent:
        os.chdir(REPO)


def _load_real_case(dataset_name: str, seed: int = 42) -> dict[str, Any]:
    if dataset_name == "Heart":
        return rdb.get_heart_case(seed=seed)
    if dataset_name == "Diabetes":
        return rdb.get_diabetes_case(seed=seed)
    if dataset_name == "MultiOmics":
        return rdb.get_multiomics_case(seed=seed)
    if dataset_name == "TCGA-KIRC":
        return rdb.get_kirc_case(seed=seed)
    if dataset_name == "ADNI-MCI":
        return rdb.get_adni_mci_case(seed=seed)
    if dataset_name == "mfeat":
        return rdb.get_mfeat_case(seed=seed)
    raise ValueError(f"Unknown real dataset: {dataset_name}")


def _prep_split(case: dict[str, Any], seed: int) -> dict[str, Any]:
    """Exact same train/test split, FWL covariate adjustment, and in-fold scaling as runner.py."""
    data_all = case["data"]
    u_all = case["true_u"]
    y_all = case["outcome"]
    n_samples = data_all[0].shape[0]
    train_idx, test_idx = split_indices(
        n_samples,
        seed=seed,
        train_frac=0.7,
        stratify=y_all if case.get("is_classification") else None,
    )
    train_mats = [m[train_idx] for m in data_all]
    test_mats = [m[test_idx] for m in data_all]
    u_true_test = u_all[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    if case.get("covariates") is not None:
        cov_all = torch.as_tensor(case["covariates"]).double().numpy()
        cov_tr, cov_te = cov_all[train_idx.numpy()], cov_all[test_idx.numpy()]
        adj_tr, adj_te = [], []
        for tr_m, te_m in zip(train_mats, test_mats):
            tr_np = tr_m.double().numpy()
            te_np = te_m.double().numpy()
            beta_x, _, _, _ = np.linalg.lstsq(cov_tr, tr_np, rcond=None)
            adj_tr.append(torch.tensor(tr_np - cov_tr @ beta_x, dtype=torch.float32))
            adj_te.append(torch.tensor(te_np - cov_te @ beta_x, dtype=torch.float32))
        train_mats, test_mats = adj_tr, adj_te
        if not case.get("is_classification"):
            y_tr_np = y_train.double().numpy()
            y_te_np = y_test.double().numpy()
            beta_y, _, _, _ = np.linalg.lstsq(cov_tr, y_tr_np, rcond=None)
            y_train = torch.tensor(y_tr_np - cov_tr @ beta_y, dtype=torch.float32)
            y_test = torch.tensor(y_te_np - cov_te @ beta_y, dtype=torch.float32)

    if case.get("needs_scaling"):
        sc_tr, sc_te = [], []
        for tr_m, te_m in zip(train_mats, test_mats):
            sc = StandardScaler().fit(tr_m.numpy())
            sc_tr.append(torch.tensor(sc.transform(tr_m.numpy())).float())
            sc_te.append(torch.tensor(sc.transform(te_m.numpy())).float())
        train_mats, test_mats = sc_tr, sc_te
        if not case.get("is_classification") and y_train.ndim == 2 and y_train.shape[1] > 1:
            sc_y = StandardScaler().fit(y_train.numpy())
            y_train = torch.tensor(sc_y.transform(y_train.numpy())).float()
            y_test = torch.tensor(sc_y.transform(y_test.numpy())).float()

    return {
        "train_mats": train_mats,
        "test_mats": test_mats,
        "u_true_test": u_true_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _run_part_a_task(task: tuple) -> dict[str, Any]:
    """Run one linear `simlr` cell using exact runner protocol (supporting either/positive/convex)."""
    _ensure_repo_cwd()
    torch.set_num_threads(1)
    dataset_name, seed, sparsity, positivity, iterations = task
    case = _load_real_case(dataset_name, seed=seed)
    is_classif = bool(case.get("is_classification"))

    params: dict[str, Any] = {
        "positivity": positivity,
        "nsa_iterations": 2,
        "consolidate": False,
    }
    budget_by_unit = {"iterations": iterations} if iterations is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp_res = rdb.run_single_experiment(
            "simlr",
            case,
            sparsity=sparsity,
            seed=seed,
            budget_by_unit=budget_by_unit,
            **params,
        )
    metrics = exp_res["metrics"]
    res = exp_res["result"]
    if dataset_name == "ADNI-MCI":
        test_metric = metrics.get("pc1_test", metrics.get("test_r2", 0.0))
        train_metric = metrics.get("pc1_train", metrics.get("train_r2", 0.0))
        lin_test = metrics.get("first_layer_pc1_test_r2", metrics.get("first_layer_test_r2", 0.0))
        rf_test = metrics.get("first_layer_axis_pc1_test_r2", metrics.get("first_layer_axis_test_r2", 0.0))
    else:
        test_metric = metrics.get("test_accuracy", 0.0) if is_classif else metrics.get("test_r2", 0.0)
        train_metric = metrics.get("train_accuracy", 0.0) if is_classif else metrics.get("train_r2", 0.0)
        lin_test = metrics.get("first_layer_test_accuracy", 0.0) if is_classif else metrics.get("first_layer_test_r2", 0.0)
        rf_test = metrics.get("first_layer_axis_test_accuracy", 0.0) if is_classif else metrics.get("first_layer_axis_test_r2", 0.0)

    return {
        "Dataset": dataset_name,
        "Seed": int(seed),
        "Sparseness_Target": float("nan") if sparsity is None else float(sparsity),
        "Positivity": positivity,
        "Predictive_Metric": float(test_metric),
        "Train_Metric": float(train_metric),
        "Gen_Gap": float(train_metric - test_metric),
        "Strictly_Linear_Metric": float(lin_test),
        "Axis_Sensitive_Metric": float(rf_test),
        "Frame_Defect_D": float(metrics["frame_defect_v"]),
        "Sparsity_Ratio": float(metrics["sparsity_ratio_v"]),
        "Support_Overlap_V": float(metrics["support_overlap_v"]),
        "Column_Coherence_L": float(metrics["column_coherence_loadings"]),
        "Support_Overlap_L": float(metrics["support_overlap_loadings"]),
        "Fit_Seconds": float(metrics.get("fit_seconds", 0.0)),
        "Stop_Reason": str(metrics.get("stop_reason")),
        "Solver_Converged": bool(metrics.get("solver_converged")),
        "Solver_Iters": int(metrics.get("solver_iters") or 0),
        "Sparseness_Type": str(res.get("sparseness_type", "hard")),
    }


def _run_part_b_task(task: tuple) -> dict[str, Any]:
    """Run one `fit_uniform_shrunk` + `project_uniform_shrunk` cell under exact CV split."""
    _ensure_repo_cwd()
    torch.set_num_threads(1)
    dataset_name, seed, k, positivity, iterations = task
    case = _load_real_case(dataset_name, seed=seed)
    sp = _prep_split(case, seed=seed)
    train_mats = sp["train_mats"]
    test_mats = sp["test_mats"]
    y_train = sp["y_train"]
    y_test = sp["y_test"]
    u_true_test = sp["u_true_test"]
    is_classif = bool(case.get("is_classification"))

    fit_kw: dict[str, Any] = {
        "k": int(k),
        "positivity": positivity,
        "energy_type": "auto",
        "mixing_algorithm": "auto",
        "alpha": "auto",
        "alpha_mode": "ric",
        "sparseness": None,
        "random_state": int(seed),
    }
    if iterations is not None:
        fit_kw["iterations"] = int(iterations)
    if case.get("topology") is not None:
        fit_kw["topology"] = case["topology"]
    if case.get("path_graph") is not None:
        fit_kw["path_graph"] = case["path_graph"]

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit_uniform_shrunk(train_mats, **fit_kw)
    fit_seconds = time.perf_counter() - t0

    eval_idx = case.get("eval_view_indices")
    if eval_idx is not None:
        # Withhold cognitive view at test time (replace with zeros; verified zero effect on MRI view_scores)
        proj_tr_in = [train_mats[i] if i in eval_idx else torch.zeros_like(train_mats[i]) for i in range(len(train_mats))]
        proj_te_in = [test_mats[i] if i in eval_idx else torch.zeros_like(test_mats[i]) for i in range(len(test_mats))]
    else:
        proj_tr_in = train_mats
        proj_te_in = test_mats

    proj_tr = pysimlr.project_uniform_shrunk(res, proj_tr_in, train_views=proj_tr_in)
    proj_te = pysimlr.project_uniform_shrunk(res, proj_te_in, train_views=proj_tr_in)

    def _to_per_view_scores(vs_arr: np.ndarray, indices: list[int]) -> list[torch.Tensor]:
        arr = np.asarray(vs_arr, dtype=np.float32)
        if arr.ndim == 2:
            # k == 1 -> (N, n_views)
            return [torch.from_numpy(arr[:, i : i + 1]) for i in indices]
        # k >= 2 -> (N, n_views, k)
        return [torch.from_numpy(arr[:, i, :]) for i in indices]

    use_idx = list(eval_idx) if eval_idx is not None else list(range(len(train_mats)))
    fl_tr = _to_per_view_scores(proj_tr["view_scores"], use_idx)
    fl_te = _to_per_view_scores(proj_te["view_scores"], use_idx)

    if eval_idx is not None:
        # Build consensus purely from evaluated (MRI) views, matching runner.py
        if int(k) == 1:
            u_tr_t = torch.stack(fl_tr, dim=0).mean(dim=0)
            u_te_t = torch.stack(fl_te, dim=0).mean(dim=0)
        else:
            u_ref = torch.as_tensor(res["u"]).detach().cpu().float()
            u_ref_c = u_ref - u_ref.mean(dim=0, keepdim=True)
            al_tr, al_te = [], []
            for z_tr, z_te in zip(fl_tr, fl_te):
                z_tr_c = z_tr - z_tr.mean(dim=0, keepdim=True)
                z_te_c = z_te - z_tr.mean(dim=0, keepdim=True)
                u_s, _, vt_s = torch.linalg.svd(z_tr_c.T @ u_ref_c, full_matrices=False)
                om = u_s @ vt_s
                al_tr.append(z_tr_c @ om)
                al_te.append(z_te_c @ om)
            u_tr_t = torch.stack(al_tr, dim=0).mean(dim=0)
            u_te_t = torch.stack(al_te, dim=0).mean(dim=0)
    else:
        u_tr_t = torch.as_tensor(proj_tr["u"], dtype=torch.float32)
        u_te_t = torch.as_tensor(proj_te["u"], dtype=torch.float32)

    v_list = [torch.as_tensor(v).detach().cpu().float() for v in res["v"]]
    eval_vs = [v_list[i] for i in use_idx]
    recons = [fl_te[idx] @ eval_vs[idx].T for idx in range(len(use_idx))]
    te_eval_mats = [test_mats[i] for i in use_idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = calculate_all_metrics(
            u_te_t,
            u_true_test[:, : int(k)],
            y_test,
            te_eval_mats,
            recons,
            shared_latents=fl_te,
            private_latents=None,
            v_mats=eval_vs,
            u_train=u_tr_t,
            y_train=y_train,
            is_classification=is_classif,
            first_layer_scores_train=fl_tr,
            first_layer_scores_test=fl_te,
        )

    if dataset_name == "ADNI-MCI":
        test_metric = metrics.get("pc1_test", metrics.get("test_r2", 0.0))
        train_metric = metrics.get("pc1_train", metrics.get("train_r2", 0.0))
        lin_test = metrics.get("first_layer_pc1_test_r2", metrics.get("first_layer_test_r2", 0.0))
        rf_test = metrics.get("first_layer_axis_pc1_test_r2", metrics.get("first_layer_axis_test_r2", 0.0))
    else:
        test_metric = metrics.get("test_accuracy", 0.0) if is_classif else metrics.get("test_r2", 0.0)
        train_metric = metrics.get("train_accuracy", 0.0) if is_classif else metrics.get("train_r2", 0.0)
        lin_test = metrics.get("first_layer_test_accuracy", 0.0) if is_classif else metrics.get("first_layer_test_r2", 0.0)
        rf_test = metrics.get("first_layer_axis_test_accuracy", 0.0) if is_classif else metrics.get("first_layer_axis_test_r2", 0.0)

    defects = [float(orthogonality_defect(v, normalized=True)) for v in eval_vs] if int(k) > 1 else [0.0]
    sparsities = [float((v.abs() < 1e-6).float().mean().item()) for v in eval_vs]
    overlaps = [float(support_overlap(v)) for v in eval_vs] if int(k) > 1 else [0.0]
    rep = basis_rank_report(eval_vs, int(k), warn=False) if int(k) > 1 else {"max_column_overlap": [0.0]}
    us_prov = res.get("uniform_shrinkage_provenance", {})
    alphas_used = us_prov.get("alphas", proj_tr.get("alphas", []))
    flat_alphas: list[float] = []

    def _collect(obj):
        if obj is None:
            return
        if isinstance(obj, (list, tuple, np.ndarray)):
            for item in obj:
                _collect(item)
        else:
            try:
                v = float(obj)
                if np.isfinite(v):
                    flat_alphas.append(v)
            except (TypeError, ValueError):
                pass

    _collect(alphas_used)
    alpha_mean = float(np.mean(flat_alphas)) if flat_alphas else float("nan")

    return {
        "Dataset": dataset_name,
        "Seed": int(seed),
        "k": int(k),
        "Positivity": positivity,
        "Method": "fit_uniform_shrunk",
        "Predictive_Metric": float(test_metric),
        "Train_Metric": float(train_metric),
        "Gen_Gap": float(train_metric - test_metric),
        "Strictly_Linear_Metric": float(lin_test),
        "Axis_Sensitive_Metric": float(rf_test),
        "Frame_Defect_D": float(np.mean(defects)),
        "Sparsity_Ratio": float(np.mean(sparsities)),
        "Support_Overlap_V": float(np.mean(overlaps)),
        "Max_Col_Cosine": float(np.mean(rep.get("max_column_overlap", [0.0]))),
        "Alpha_Mean": alpha_mean,
        "Fit_Seconds": float(fit_seconds),
        "Solver_Converged": bool(res.get("converged", False)),
        "Stop_Reason": str(res.get("stop_reason", "")),
        "Energy_Type": str(res.get("energy_type", "")),
        "Mixing_Algorithm": str(res.get("mixing_algorithm", "")),
    }


def _execute_pool(fn, tasks: list, workers: int, label: str) -> pd.DataFrame:
    t0 = time.perf_counter()
    rows = []
    failures = []
    if workers <= 1 or len(tasks) <= 2:
        for t in tasks:
            try:
                rows.append(fn(t))
            except (RuntimeError, ValueError, OSError) as exc:
                failures.append((t, repr(exc)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fmap = {ex.submit(fn, t): t for t in tasks}
            done = 0
            for fut in as_completed(fmap):
                t = fmap[fut]
                try:
                    rows.append(fut.result())
                except (RuntimeError, ValueError, OSError) as exc:
                    failures.append((t, repr(exc)))
                done += 1
                if done % 20 == 0 or done == len(tasks):
                    print(f"  [{label}] {done}/{len(tasks)} ({time.perf_counter()-t0:.1f}s)", flush=True)
    if failures:
        raise RuntimeError(f"{label}: {len(failures)}/{len(tasks)} failed; first: {failures[0]}")
    print(f"  [{label}] completed {len(rows)} tasks in {time.perf_counter()-t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def _run_part_c(out_dir: str, smoke: bool, workers: int, reuse_adni: bool = False) -> tuple[dict[str, Any], str]:
    """Run ADNI-MCI ablations (k, quantile, rowmean) + Heart/Diabetes nsa_v_loadings."""
    is_cache_out = os.path.abspath(out_dir) == os.path.abspath(CACHE_DIR)
    abl_dir = out_dir if is_cache_out else os.path.join(out_dir, "ablations")
    os.makedirs(abl_dir, exist_ok=True)
    summary: dict[str, Any] = {}

    if smoke:
        import paper_adni_ablations as paa
        data = paa.protocol_b(42)
        res, secs = paa.fit_adni(data, "simlr", k=2, positivity="positive", sparseness=0.5, iterations=2, random_state=42)
        vs = paa.mri_basis(res)
        ev = paa.evaluate(data, vs)
        summary["adni_k_smoke"] = {"k": 2, "cog_pc1_r2": ev["cog_pc1_r2"], "secs": secs}
        summary["adni_quantile_smoke"] = {"pos": "positive", "q": 0.5, "cog_pc1_r2": ev["cog_pc1_r2"]}
        summary["adni_rowmean_smoke"] = {"cog_pc1_r2": ev["cog_pc1_r2"]}
        heart = pd.read_csv(os.path.join(REPO, "data", "heart_cleveland.csv"))
        import paper_nsa_v_loadings as pnl
        xs = [torch.from_numpy(StandardScaler().fit_transform(heart[c].values.astype(float))).float()
              for c in pnl.HEART_VIEWS.values()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inf = pysimlr.loading_inference(
                "lend", xs, 2, n_subsamples=2, n_permutations=2,
                energy_type="nc", mixing_algorithm="newton",
                positivity="positive", sparseness=0.5, iterations=2, random_state=42, n_jobs=1,
            )
        summary["nsa_v_loadings_smoke"] = {
            "Heart_converged": bool(inf["result"].get("converged", False)),
            "Heart_latent_reliability": [round(float(x), 4) for x in np.asarray(inf["latent_reliability"]).ravel()],
        }
        if not is_cache_out:
            with open(os.path.join(abl_dir, "part_c_summary.json"), "w") as fh:
                json.dump(summary, fh, indent=2)
        return summary, abl_dir

    py = sys.executable
    adni_expected = [
        os.path.join(abl_dir, f)
        for f in ("ablation_k_sparsity_sweep.csv", "ablation_quantile_sparseness.csv", "ablation_rowmean_ica_positivity.csv")
    ]
    if not (reuse_adni and all(os.path.exists(p) for p in adni_expected)):
        t0 = time.perf_counter()
        subprocess.run(
            [py, "scripts/paper_adni_ablations.py",
             "--only", "k", "quantile", "rowmean", "--workers", str(workers),
             "--out", os.path.relpath(abl_dir, REPO)],
            cwd=REPO, check=True,
        )
        summary["adni_ablations_seconds"] = round(time.perf_counter() - t0, 2)
    else:
        summary["adni_ablations_seconds"] = "reused_from_same_v043_run"

    t1 = time.perf_counter()
    import paper_nsa_v_loadings as pnl
    from sklearn.datasets import load_diabetes
    heart = pd.read_csv(os.path.join(REPO, "data", "heart_cleveland.csv"))
    diab = load_diabetes(as_frame=True).frame

    with open(os.path.join(CACHE_DIR, "nsa_v_loadings.json")) as fh:
        old_nsa = json.load(fh)

    nsa_cmp: dict[str, Any] = {}
    for cohort, frame, views in (("Heart", heart, pnl.HEART_VIEWS), ("Diabetes", diab, pnl.DIAB_VIEWS)):
        xs = [torch.from_numpy(StandardScaler().fit_transform(frame[c].values.astype(float))).float()
              for c in views.values()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inf = pysimlr.loading_inference(
                "lend", xs, 2, n_subsamples=50, n_permutations=20,
                energy_type="nc", mixing_algorithm="newton",
                positivity="positive", sparseness=0.5, random_state=42, n_jobs=workers,
            )
        res = inf["result"]
        old_b = old_nsa["plot_bundles"][cohort]
        new_loads = [l.detach().cpu().numpy().astype(np.float64) for l in res["loadings"]]
        old_loads = [np.asarray(l, dtype=np.float64) for l in old_b["res"]["loadings"]]
        max_load_diff = max(float(np.abs(a - b).max()) for a, b in zip(old_loads, new_loads))
        old_sel = [int(np.sum(s)) for s in old_b["inf"]["selected"]]
        new_sel = [int(np.sum(np.asarray(s))) for s in inf["selected"]]
        lat_rel_raw = inf.get("latent_reliability")
        lat_rel = None if lat_rel_raw is None else [round(float(x), 4) for x in np.asarray(lat_rel_raw).ravel()]
        lat_gain_raw = inf.get("latent_reliability_gain")
        lat_gain = None if lat_gain_raw is None else [round(float(x), 4) for x in np.asarray(lat_gain_raw).ravel()]
        nsa_cmp[cohort] = {
            "cached_latent_reliability": old_b["inf"].get("latent_reliability"),
            "v043_latent_reliability": lat_rel,
            "v043_latent_reliability_gain": lat_gain,
            "cached_selected_counts": old_sel,
            "v043_selected_counts": new_sel,
            "max_abs_loading_diff": round(max_load_diff, 4),
            "v043_converged": bool(res.get("convergence", {}).get("converged", False)),
            "v043_converged_fraction": round(float(inf.get("converged_fraction", float("nan"))), 4),
        }
    summary["nsa_v_loadings_seconds"] = round(time.perf_counter() - t1, 2)
    summary["nsa_v_loadings.json"] = nsa_cmp
    return summary, abl_dir


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _summarize_part_b(df_b: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    g_b = df_b.groupby(["Dataset", "Positivity", "k"]).agg(
        US_Metric=("Predictive_Metric", "mean"),
        US_Lin=("Strictly_Linear_Metric", "mean"),
        US_Sparsity=("Sparsity_Ratio", "mean"),
        US_Defect=("Frame_Defect_D", "mean"),
        US_MaxCos=("Max_Col_Cosine", "mean"),
        US_Alpha=("Alpha_Mean", "mean"),
        US_Sec=("Fit_Seconds", "mean"),
        US_Conv=("Solver_Converged", "mean"),
    ).reset_index()
    g_b.to_csv(os.path.join(out_dir, "part_b_uniform_shrunk_summary.csv"), index=False)
    return g_b


def _build_html_report(
    out_dir: str,
    abl_dir: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    part_c: dict[str, Any],
    prov: dict[str, Any],
    write_html: bool = True,
) -> str:
    """Build comparison summaries and (when non-cache out_dir) standalone HTML report."""
    cached = pd.read_csv(os.path.join(CACHE_DIR, "operating_points_05_08_benchmark.csv"))
    cached_simlr = cached[(cached["Case_Type"] == "real") & (cached["Method"] == "simlr")].copy()

    def _sp_key(s):
        return "None" if pd.isna(s) else f"{float(s):.1f}"

    cached_simlr["Sparseness_Key"] = cached_simlr["Sparseness_Target"].map(_sp_key)
    df_a_c = df_a.copy()
    df_a_c["Sparseness_Key"] = df_a_c["Sparseness_Target"].map(_sp_key)

    g_old = cached_simlr.groupby(["Dataset", "Positivity", "Sparseness_Key"]).agg(
        Cached_Metric=("Predictive_Metric", "mean"),
        Cached_Sparsity=("Sparsity_Ratio", "mean"),
        Cached_Defect=("Frame_Defect_D", "mean"),
        Cached_Overlap=("Support_Overlap_V", "mean"),
        Cached_Sec=("Fit_Seconds", "mean"),
    ).reset_index()

    g_new = df_a_c.groupby(["Dataset", "Positivity", "Sparseness_Key"]).agg(
        V043_Metric=("Predictive_Metric", "mean"),
        V043_Sparsity=("Sparsity_Ratio", "mean"),
        V043_Defect=("Frame_Defect_D", "mean"),
        V043_Overlap=("Support_Overlap_V", "mean"),
        V043_Sec=("Fit_Seconds", "mean"),
        V043_Conv=("Solver_Converged", "mean"),
    ).reset_index()

    cmp_a = g_new.merge(g_old, on=["Dataset", "Positivity", "Sparseness_Key"], how="left")
    cmp_a["Delta_Metric"] = cmp_a["V043_Metric"] - cmp_a["Cached_Metric"]
    cmp_a["Delta_Sparsity"] = cmp_a["V043_Sparsity"] - cmp_a["Cached_Sparsity"]
    cmp_a.to_csv(os.path.join(out_dir, "part_a_comparison_summary.csv"), index=False)

    g_b = _summarize_part_b(df_b, out_dir)

    if not write_html:
        return ""

    fig1, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    matched = cmp_a[cmp_a["Cached_Metric"].notna() & cmp_a["Sparseness_Key"].isin(["0.5", "0.8"])].copy()
    matched["Label"] = matched["Dataset"] + "\n" + matched["Positivity"].str[:3] + " q=" + matched["Sparseness_Key"]
    x = np.arange(len(matched))
    w = 0.38
    axes[0].bar(x - w / 2, matched["Cached_Metric"], width=w, label="Cached Baseline (pre-v0.43)", color="#64748b")
    axes[0].bar(x + w / 2, matched["V043_Metric"], width=w, label=f"pysimlr v{prov['pysimlr']} (hard default)", color="#2563eb")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(matched["Label"], rotation=45, ha="right", fontsize=7.5)
    axes[0].set_ylabel("Test Predictive Metric (R² or Accuracy)")
    axes[0].set_title(f"Part (a): Linear SiMLR — Cached vs v{prov['pysimlr']} (q=0.5, 0.8)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    ds_order = [d for d in REAL_DATASETS if d in cmp_a["Dataset"].unique()]
    best_cached = [cmp_a[cmp_a["Dataset"] == d]["Cached_Metric"].max() for d in ds_order]
    best_v043 = [cmp_a[cmp_a["Dataset"] == d]["V043_Metric"].max() for d in ds_order]
    best_us = [g_b[g_b["Dataset"] == d]["US_Metric"].max() if d in g_b["Dataset"].values else np.nan for d in ds_order]
    x2 = np.arange(len(ds_order))
    w2 = 0.26
    axes[1].bar(x2 - w2, best_cached, width=w2, label="Cached simlr (best)", color="#64748b")
    axes[1].bar(x2, best_v043, width=w2, label=f"v{prov['pysimlr']} simlr (best)", color="#2563eb")
    axes[1].bar(x2 + w2, best_us, width=w2, label=f"v{prov['pysimlr']} fit_uniform_shrunk (best)", color="#059669")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(ds_order, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Best Test Predictive Metric")
    axes[1].set_title("Best Operating Point Across All 6 Real Datasets")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    img1 = _fig_to_b64(fig1)

    fig2, ax2 = plt.subplots(figsize=(11.5, 4.4))
    piv = g_b.pivot_table(index=["Dataset", "k"], columns="Positivity", values="US_Metric").reset_index()
    labels2 = [f"{r.Dataset}\nk={int(r.k)}" for _, r in piv.iterrows()]
    x3 = np.arange(len(piv))
    if "convex" in piv.columns:
        ax2.bar(x3 - 0.19, piv["convex"], width=0.36, label="fit_uniform_shrunk (convex / radial)", color="#059669")
    if "conical" in piv.columns:
        ax2.bar(x3 + 0.19, piv["conical"], width=0.36, label="fit_uniform_shrunk (conical)", color="#d97706")
    ax2.set_xticks(x3)
    ax2.set_xticklabels(labels2, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Test Predictive Metric (R² or Accuracy)")
    ax2.set_title(f"Part (b): fit_uniform_shrunk v{prov['pysimlr']} — Radial Convex vs Conical Across k")
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    img2 = _fig_to_b64(fig2)

    img3 = ""
    q_new_path = os.path.join(abl_dir, "ablation_quantile_sparseness.csv")
    if os.path.exists(q_new_path):
        q_old = pd.read_csv(os.path.join(CACHE_DIR, "ablation_quantile_sparseness.csv"))
        q_new = pd.read_csv(q_new_path)
        fig3, ax3 = plt.subplots(1, 2, figsize=(12.0, 4.2))
        for idx, pos in enumerate(("positive", "either")):
            so = q_old[q_old["positivity"] == pos].sort_values("sparseness_val")
            sn = q_new[q_new["positivity"] == pos].sort_values("sparseness_val")
            xs = np.arange(len(so))
            ax3[idx].plot(xs, so["cog_pc1_r2"], "o--", color="#64748b", label="Cached CogPC1 R²")
            ax3[idx].plot(xs, sn["cog_pc1_r2"], "s-", color="#2563eb", label=f"v{prov['pysimlr']} CogPC1 R²")
            ax3[idx].set_xticks(xs)
            ax3[idx].set_xticklabels(so["sparseness"], rotation=30, ha="right", fontsize=8)
            ax3[idx].set_ylabel("Test CogPC1 R²")
            ax3[idx].set_title(f"ADNI-MCI Quantile Sweep ({pos})")
            ax3[idx].legend(fontsize=8)
            ax3[idx].grid(alpha=0.3)
        fig3.tight_layout()
        img3 = _fig_to_b64(fig3)

    def _df_html(df: pd.DataFrame) -> str:
        return df.to_html(index=False, float_format=lambda v: f"{v:.4f}" if np.isfinite(v) else "—", classes="tbl", border=0)

    part_c_html = f"<pre>{json.dumps(part_c, indent=2)}</pre>"
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>pysimlr v{prov['pysimlr']} — Real-Data Paper Benchmark Revisit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 28px; color: #1e293b; background: #f8fafc; }}
h1, h2, h3 {{ color: #0f172a; }}
.card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 22px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.tbl {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin-top: 10px; }}
.tbl th, .tbl td {{ border-bottom: 1px solid #e2e8f0; padding: 6px 9px; text-align: right; }}
.tbl th {{ background: #f1f5f9; color: #334155; font-weight: 600; }}
.tbl td:nth-child(1), .tbl th:nth-child(1), .tbl td:nth-child(2), .tbl th:nth-child(2), .tbl td:nth-child(3), .tbl th:nth-child(3) {{ text-align: left; }}
img {{ max-width: 100%; border-radius: 6px; border: 1px solid #e2e8f0; }}
pre {{ background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
.meta {{ font-size: 13px; color: #475569; }}
</style></head><body>
<h1>pysimlr v{prov['pysimlr']} — Real-Data Paper Benchmark Revisit</h1>
<p class="meta"><b>Generated:</b> {prov['end_utc']} | <b>Commit:</b> <code>{prov['git_commit']}</code> | <b>Smoke:</b> {prov['smoke']} | <b>Platform:</b> {prov['platform']} | <b>Total runtime:</b> {prov['elapsed_seconds']}s</p>
<div class="card"><h2>1. Overview & Visual Comparison Across All 6 Real Datasets</h2><img src="data:image/png;base64,{img1}" alt="Part A and Best Comparison"></div>
<div class="card"><h2>2. Part (a) — Linear SiMLR Real-Data Paper Benchmark (v{prov['pysimlr']} vs Cached Baseline)</h2>{_df_html(cmp_a)}</div>
<div class="card"><h2>3. Part (b) — v{prov['pysimlr']} Uniform Shrinkage (<code>fit_uniform_shrunk</code> / <code>project_uniform_shrunk</code>)</h2><img src="data:image/png;base64,{img2}" alt="Uniform Shrinkage Comparison">{_df_html(g_b)}</div>
<div class="card"><h2>4. Part (c) — Real-Data Paper Ablation & Loading Spot-Checks</h2>{f'<img src="data:image/png;base64,{img3}" alt="ADNI Quantile Sweep">' if img3 else ''}{part_c_html}</div>
</body></html>
"""
    report_path = os.path.join(out_dir, "pysimlr_v043_real_data_report.html")
    with open(report_path, "w") as fh:
        fh.write(html)
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="pysimlr v0.43.1 real-data paper benchmark revisit")
    ap.add_argument("--out", default=None, help="Output directory (defaults to paper/results_cache, or temp dir under --smoke)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--reuse-existing", action="store_true", help="Reuse Part A/B & ADNI CSVs already computed in --out")
    ap.add_argument("--skip-part-c", action="store_true", help="Skip Part C ADNI ablations / nsa_v_loadings checks")
    ap.add_argument("--part-b-only", action="store_true", help="Run only Part B uniform shrinkage and preserve existing Part A CSVs")
    ap.add_argument("--smoke", action="store_true", help="Fast <10s smoke test exercising 100%% of stages")
    args = ap.parse_args()

    out_dir = args.out or (
        os.path.join(tempfile.gettempdir(), "pysimlr_smoke_real_data_v043")
        if args.smoke else CACHE_DIR
    )
    os.makedirs(out_dir, exist_ok=True)
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    t0 = time.perf_counter()

    if args.smoke:
        seeds = [42, 43]
        datasets_a = ["Heart", "Diabetes"]
        datasets_b = ["Heart", "Diabetes"]
        sparseness_grid = [None, 0.5]
        iters = 2
        workers = 1
    else:
        seeds = list(range(42, 42 + args.n_seeds))
        datasets_a = list(REAL_DATASETS)
        datasets_b = list(REAL_DATASETS)
        sparseness_grid = [None, 0.5, 0.8]
        iters = args.iters
        workers = args.workers

    part_b_only = bool(args.part_b_only and not args.smoke)
    tasks_a = []
    df_a = pd.DataFrame()
    if not part_b_only:
        part_a_csv = os.path.join(out_dir, "part_a_simlr_v043_raw.csv")
        for ds in datasets_a:
            pos_list = ["either", "positive", "convex"] if ds in BIO_DATASETS else ["either"]
            for pos in pos_list:
                for sp in sparseness_grid:
                    for seed in seeds:
                        tasks_a.append((ds, seed, sp, pos, iters))
        if args.reuse_existing and not args.smoke and os.path.exists(part_a_csv):
            print(f"=== Part (a): reusing {part_a_csv} ===", flush=True)
            df_a = pd.read_csv(part_a_csv)
        else:
            print(f"=== Part (a): {len(tasks_a)} linear simlr real-data tasks ===", flush=True)
            df_a = _execute_pool(_run_part_a_task, tasks_a, workers=workers, label="Part-A")
            df_a.to_csv(part_a_csv, index=False)

    part_b_csv = os.path.join(out_dir, "part_b_uniform_shrunk_raw.csv")
    tasks_b = []
    for ds in datasets_b:
        paper_k = 2 if ds in ("Heart", "Diabetes") else 3
        k_list = [1, 2] if args.smoke else ([1, 2] if paper_k == 2 else [1, 2, 3])
        for k in k_list:
            for pos in ("convex", "conical"):
                for seed in seeds:
                    tasks_b.append((ds, seed, k, pos, iters))
    if args.reuse_existing and not args.smoke and os.path.exists(part_b_csv):
        print(f"=== Part (b): reusing {part_b_csv} ===", flush=True)
        df_b = pd.read_csv(part_b_csv)
    else:
        print(f"=== Part (b): {len(tasks_b)} fit_uniform_shrunk real-data tasks ===", flush=True)
        df_b = _execute_pool(_run_part_b_task, tasks_b, workers=workers, label="Part-B")
        df_b.to_csv(part_b_csv, index=False)

    if part_b_only or (args.skip_part_c and not args.smoke):
        part_c, abl_dir = {"skipped": True}, out_dir
    else:
        print(f"=== Part (c): Real-data paper ablations & loadings (smoke={args.smoke}) ===", flush=True)
        part_c, abl_dir = _run_part_c(
            out_dir, smoke=args.smoke, workers=workers, reuse_adni=(args.reuse_existing and not args.smoke)
        )

    try:
        commit = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cli_rel = " ".join([os.path.relpath(os.path.abspath(sys.argv[0]), REPO)] + sys.argv[1:])
    prov = {
        "cli": cli_rel,
        "start_utc": start_utc,
        "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "pysimlr": pysimlr.__version__,
        "git_commit": commit,
        "device": "cpu",
        "smoke": bool(args.smoke),
        "part_b_only": part_b_only,
        "n_tasks_part_a": len(tasks_a),
        "n_tasks_part_b": len(tasks_b),
    }
    with open(os.path.join(out_dir, "provenance_v043_real_data.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    is_cache_out = os.path.abspath(out_dir) == os.path.abspath(CACHE_DIR)
    if part_b_only:
        _summarize_part_b(df_b, out_dir)
        report_path = ""
    else:
        report_path = _build_html_report(out_dir, abl_dir, df_a, df_b, part_c, prov, write_html=(not is_cache_out))
    print(f"[DONE] out={out_dir} report={report_path or 'skipped (cache out / part-b-only)'} (elapsed={prov['elapsed_seconds']}s, smoke={prov['smoke']})", flush=True)


if __name__ == "__main__":
    main()
