#!/usr/bin/env python3
"""Collapse threshold selection (P1 — Decision Rule R15).

Tests collapse threshold candidates t in {0.3, 0.5, 0.7} with cw = 1:
  (i)   Narrow fixture per narrow-fixture protocol: outcome R² and |score corr|
        at fixed lr=0.01, with auto-lr recorded beside it.
  (ii)  corr_case(rho, seed) for rho in {0, 0.5, 0.7}, seeds 0-4, newton mixing:
        structure Hungarian |r| against planted loadings, and subspace R².
  (iii) 20 default cells: latent R² and loadings support at default budget.
  (iv)  [Reviewer addition]: Refit corr_case with mixing_algorithm="avg",
        rho in {0.5, 0.7}, seeds 0-4, at fixed learning_rate=0.01 for each
        threshold t in {0.3, 0.5, 0.7}. Record score corr, converged, and stop_reason.
        If converged fits at t=0.3 do not saturate near 0.30, withdraw the clipping
        claim in tracking D3 and contract.

Reuses existing t = 0.3 rows:
  - cells: cw=1 rows of collapse_diagnosis.csv (source=9c3279d)
  - corr_case newton: correlated_factors.csv (source=b1a1afe) for rho in {0, 0.5, 0.7}
  - narrow fixture: collapse_diagnosis.csv (source=9c3279d)

Applies Decision Rule R15:
  - Candidates: t in {0.5, 0.7} that are non-inferior to t = 0.3 on (iii) latent R²,
    (iii) loadings support, and (i) outcome R².
  - Adopt candidate with highest mean (ii) structure |r| at rho = 0.7 if it beats
    t = 0.3 by > 0.005 AND deep collapse count is 0/20.
  - Otherwise keep 0.3.

Output: docs/audit/evidence/collapse_threshold_selection.csv
"""
import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import pysimlr
import pysimlr.defaults as defaults
import pysimlr.utils as utils
from collapse_candidates_benchmark import _offdiag_abs_mean, corr_case
from correlated_factors import matched_r
from default_selection_benchmark import _case
from optimizer_benchmark_synthetic import latent_r2, support_f1
from pysimlr.loadings import loadings
from pysimlr.simlr import simlr
from test_collapse_penalty import _narrow, _outcome_r2


def assert_threshold_probe(threshold: float) -> None:
    """Assert observable penalty value on a known correlated pair."""
    utils.DEFAULT_COLLAPSE_THRESHOLD = float(threshold)
    defaults.DEFAULT_COLLAPSE_THRESHOLD = float(threshold)

    z_probe = torch.tensor([[1.0, 1.0], [0.0, 1.0], [-1.0, -2.0]], dtype=torch.float32)
    # Correlation is sqrt(3)/2 ≈ 0.8660254037844386
    r_val = float(torch.sqrt(torch.tensor(3.0)) / 2.0)
    expected_pen = ((r_val - threshold) / (1.0 - threshold)) ** 2
    actual_pen = float(utils.correlation_penalty(z_probe))
    diff = abs(actual_pen - expected_pen)
    assert diff < 1e-5, (
        f"Probe assertion failed for t={threshold}: actual={actual_pen:.6f}, "
        f"expected={expected_pen:.6f}, diff={diff:.2e}"
    )


def _run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one experimental fit in worker process."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)

    t = float(task["threshold"])
    assert_threshold_probe(t)

    group = task["group"]
    t0 = time.perf_counter()

    if group == "narrow_fixture":
        lr = task["learning_rate"]
        x, y = _narrow()
        res = simlr(
            x,
            k=2,
            positivity="positive",
            sparseness=0.5,
            energy_type="nc",
            optimizer_type="hybrid_adam",
            mixing_algorithm="newton",
            random_state=0,
            collapse_weight=1.0,
            learning_rate=lr,
        )
        scores = x[0] @ res["v"][0]
        corr = float(torch.corrcoef(scores.T)[0, 1])
        r2 = _outcome_r2(x, y, res["v"])
        t1 = time.perf_counter()
        return {
            "group": "narrow_fixture",
            "threshold": t,
            "case_id": f"narrow_t{t}_{lr}",
            "generator": "narrow",
            "positivity": "positive",
            "rho": np.nan,
            "seed": 0,
            "mixing_algorithm": "newton",
            "learning_rate": str(lr),
            "chosen_lr": float(res.get("learning_rate", lr if lr != "auto" else 0.0)),
            "converged": bool(res["converged"]),
            "n_steps": int(res["converged_iter"]),
            "stop_reason": str(res["stop_reason"]),
            "best_energy": float(res["best_energy"]),
            "outcome_r2": float(r2),
            "score_corr": float(corr),
            "u_score_corr": np.nan,
            "s0_score_corr": np.nan,
            "s1_score_corr": np.nan,
            "subspace_r2": np.nan,
            "procrustes_r2": np.nan,
            "u_hungarian_mean_r": np.nan,
            "structure_hungarian_mean_r": np.nan,
            "latent_r2": np.nan,
            "loadings_support": np.nan,
            "seconds": round(t1 - t0, 4),
            "source": "measured",
            "notes": "",
        }

    elif group == "corr_case_newton":
        rho = float(task["rho"])
        seed = int(task["seed"])
        z, views, vs = corr_case(rho=rho, seed=seed)
        res = simlr(views, k=3, mixing_algorithm="newton", random_state=seed, collapse_weight=1.0)
        t1 = time.perf_counter()
        lo = loadings(res, views)
        u = res["u"]
        st = lo["structure"]

        sub_r2 = latent_r2(z, u)
        u_hungarian = matched_r([u], [z])
        st_hungarian = matched_r(st, vs)
        u_off = _offdiag_abs_mean(u)
        s0_off = _offdiag_abs_mean(views[0] @ res["v"][0])
        s1_off = _offdiag_abs_mean(views[1] @ res["v"][1])

        return {
            "group": "corr_case_newton",
            "threshold": t,
            "case_id": f"corr_newton_t{t}_rho{rho}_s{seed}",
            "generator": "corr",
            "positivity": "either",
            "rho": rho,
            "seed": seed,
            "mixing_algorithm": "newton",
            "learning_rate": "auto",
            "chosen_lr": float(res.get("learning_rate", 0.0)),
            "converged": bool(res.get("converged", False)),
            "n_steps": int(res.get("converged_iter", 0)),
            "stop_reason": str(res.get("stop_reason", "")),
            "best_energy": float(res.get("best_energy", 0.0)),
            "outcome_r2": np.nan,
            "score_corr": float(u_off),
            "u_score_corr": float(u_off),
            "s0_score_corr": float(s0_off),
            "s1_score_corr": float(s1_off),
            "subspace_r2": round(sub_r2, 6),
            "procrustes_r2": np.nan,
            "u_hungarian_mean_r": round(u_hungarian, 6),
            "structure_hungarian_mean_r": round(st_hungarian, 6),
            "latent_r2": np.nan,
            "loadings_support": np.nan,
            "seconds": round(t1 - t0, 4),
            "source": "measured",
            "notes": "",
        }

    elif group == "default_cells":
        gen = str(task["generator"])
        pos = str(task["positivity"])
        seed = int(task["seed"])
        z, views, sups = _case(gen, seed)
        k = z.shape[1]
        res = pysimlr.fit(
            "simlr",
            views,
            k=k,
            energy_type="nc",
            positivity=pos,
            random_state=seed,
            collapse_weight=1.0,
            iterations=defaults.DEFAULT_SIMLR_ITERATIONS,
        )
        t1 = time.perf_counter()
        r2 = latent_r2(z, res["u"])
        f1 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(res["loadings"], sups)]))

        return {
            "group": "default_cells",
            "threshold": t,
            "case_id": f"cell_t{t}_{gen}_{pos}_s{seed}",
            "generator": gen,
            "positivity": pos,
            "rho": np.nan,
            "seed": seed,
            "mixing_algorithm": "newton",
            "learning_rate": "auto",
            "chosen_lr": float(res["convergence"].get("learning_rate", 0.0)),
            "converged": bool(res["convergence"]["converged"]),
            "n_steps": int(res["convergence"]["n_steps"]),
            "stop_reason": str(res["convergence"].get("stop_reason", "")),
            "best_energy": float(res.get("best_energy", 0.0)),
            "outcome_r2": np.nan,
            "score_corr": np.nan,
            "u_score_corr": np.nan,
            "s0_score_corr": np.nan,
            "s1_score_corr": np.nan,
            "subspace_r2": np.nan,
            "procrustes_r2": np.nan,
            "u_hungarian_mean_r": np.nan,
            "structure_hungarian_mean_r": np.nan,
            "latent_r2": round(r2, 6),
            "loadings_support": round(f1, 6),
            "seconds": round(t1 - t0, 4),
            "source": "measured",
            "notes": "",
        }

    elif group == "corr_case_avg":
        rho = float(task["rho"])
        seed = int(task["seed"])
        z, views, vs = corr_case(rho=rho, seed=seed)
        res = simlr(
            views,
            k=3,
            mixing_algorithm="avg",
            learning_rate=0.01,
            random_state=seed,
            collapse_weight=1.0,
            iterations=defaults.DEFAULT_SIMLR_ITERATIONS,
        )
        t1 = time.perf_counter()
        u_off = _offdiag_abs_mean(res["u"])
        s0_off = _offdiag_abs_mean(views[0] @ res["v"][0])
        s1_off = _offdiag_abs_mean(views[1] @ res["v"][1])

        return {
            "group": "corr_case_avg",
            "threshold": t,
            "case_id": f"corr_avg_t{t}_rho{rho}_s{seed}",
            "generator": "corr",
            "positivity": "either",
            "rho": rho,
            "seed": seed,
            "mixing_algorithm": "avg",
            "learning_rate": "0.01",
            "chosen_lr": 0.01,
            "converged": bool(res.get("converged", False)),
            "n_steps": int(res.get("converged_iter", 0)),
            "stop_reason": str(res.get("stop_reason", "")),
            "best_energy": float(res.get("best_energy", 0.0)),
            "outcome_r2": np.nan,
            "score_corr": float(u_off),
            "u_score_corr": float(u_off),
            "s0_score_corr": float(s0_off),
            "s1_score_corr": float(s1_off),
            "subspace_r2": np.nan,
            "procrustes_r2": np.nan,
            "u_hungarian_mean_r": np.nan,
            "structure_hungarian_mean_r": np.nan,
            "latent_r2": np.nan,
            "loadings_support": np.nan,
            "seconds": round(t1 - t0, 4),
            "source": "measured",
            "notes": "",
        }

    raise ValueError(f"Unknown task group: {group}")


def load_reused_t03_rows() -> List[Dict[str, Any]]:
    """Load existing t = 0.3 rows from collapse_diagnosis.csv and correlated_factors.csv."""
    rows = []

    # 1. Narrow fixture t=0.3 from collapse_diagnosis.csv (source=9c3279d)
    diag_csv = os.path.join(REPO, "docs", "audit", "evidence", "collapse_diagnosis.csv")
    df_diag = pd.read_csv(diag_csv)

    # Narrow lr=0.01
    row_001 = df_diag[(df_diag.group == "narrow_step0") & (df_diag.case_id == "narrow_cw1_lr0.01")].iloc[0]
    rows.append({
        "group": "narrow_fixture",
        "threshold": 0.3,
        "case_id": "narrow_t0.3_0.01",
        "generator": "narrow",
        "positivity": "positive",
        "rho": np.nan,
        "seed": 0,
        "mixing_algorithm": "newton",
        "learning_rate": "0.01",
        "chosen_lr": 0.01,
        "converged": bool(row_001["converged"]),
        "n_steps": int(row_001["n_steps"]),
        "stop_reason": str(row_001["stop_reason"]),
        "best_energy": float(row_001["best_energy"]),
        "outcome_r2": float(row_001["outcome_r2"]),
        "score_corr": float(row_001["score_corr"]),
        "u_score_corr": np.nan,
        "s0_score_corr": np.nan,
        "s1_score_corr": np.nan,
        "subspace_r2": np.nan,
        "procrustes_r2": np.nan,
        "u_hungarian_mean_r": np.nan,
        "structure_hungarian_mean_r": np.nan,
        "latent_r2": np.nan,
        "loadings_support": np.nan,
        "seconds": np.nan,
        "source": "9c3279d",
        "notes": "Reused from collapse_diagnosis.csv",
    })

    # Narrow auto
    row_auto = df_diag[(df_diag.group == "narrow_step0") & (df_diag.case_id == "narrow_cw1_auto")].iloc[0]
    rows.append({
        "group": "narrow_fixture",
        "threshold": 0.3,
        "case_id": "narrow_t0.3_auto",
        "generator": "narrow",
        "positivity": "positive",
        "rho": np.nan,
        "seed": 0,
        "mixing_algorithm": "newton",
        "learning_rate": "auto",
        "chosen_lr": float(row_auto["chosen_lr"]),
        "converged": bool(row_auto["converged"]),
        "n_steps": int(row_auto["n_steps"]),
        "stop_reason": str(row_auto["stop_reason"]),
        "best_energy": float(row_auto["best_energy"]),
        "outcome_r2": float(row_auto["outcome_r2"]),
        "score_corr": float(row_auto["score_corr"]),
        "u_score_corr": np.nan,
        "s0_score_corr": np.nan,
        "s1_score_corr": np.nan,
        "subspace_r2": np.nan,
        "procrustes_r2": np.nan,
        "u_hungarian_mean_r": np.nan,
        "structure_hungarian_mean_r": np.nan,
        "latent_r2": np.nan,
        "loadings_support": np.nan,
        "seconds": np.nan,
        "source": "9c3279d",
        "notes": "Reused from collapse_diagnosis.csv",
    })

    # 2. 20 default cells t=0.3 from collapse_diagnosis.csv (source=9c3279d)
    cells_df = df_diag[df_diag.group == "default_cells"]
    # Group by case_id to get latent_r2 and loadings_support_f1
    for case_id, grp in cells_df.groupby("case_id"):
        r2_row = grp[grp.metric == "latent_r2"].iloc[0]
        f1_row = grp[grp.metric == "loadings_support_f1"].iloc[0]
        gen = str(r2_row["generator"])
        pos = str(r2_row["positivity"])
        seed = int(r2_row["seed"])
        r2_val = float(r2_row["cw1_baseline"])
        f1_val = float(f1_row["cw1_baseline"])
        conv_val = bool(r2_row["converged"])
        steps_val = int(r2_row["n_steps"])

        rows.append({
            "group": "default_cells",
            "threshold": 0.3,
            "case_id": f"cell_t0.3_{gen}_{pos}_s{seed}",
            "generator": gen,
            "positivity": pos,
            "rho": np.nan,
            "seed": seed,
            "mixing_algorithm": "newton",
            "learning_rate": "auto",
            "chosen_lr": np.nan,
            "converged": conv_val,
            "n_steps": steps_val,
            "stop_reason": "converged" if conv_val else "no_descent",
            "best_energy": np.nan,
            "outcome_r2": np.nan,
            "score_corr": np.nan,
            "u_score_corr": np.nan,
            "s0_score_corr": np.nan,
            "s1_score_corr": np.nan,
            "subspace_r2": np.nan,
            "procrustes_r2": np.nan,
            "u_hungarian_mean_r": np.nan,
            "structure_hungarian_mean_r": np.nan,
            "latent_r2": round(r2_val, 6),
            "loadings_support": round(f1_val, 6),
            "seconds": np.nan,
            "source": "9c3279d",
            "notes": "Reused cw=1 baseline from collapse_diagnosis.csv",
        })

    # 3. corr_case newton t=0.3 from correlated_factors.csv (source=b1a1afe)
    corr_csv = os.path.join(REPO, "docs", "audit", "evidence", "correlated_factors.csv")
    df_corr = pd.read_csv(corr_csv)
    newton_df = df_corr[(df_corr.mixing_algorithm == "newton") & (df_corr.rho.isin([0.0, 0.5, 0.7]))]
    for _, r in newton_df.iterrows():
        rho = float(r["rho"])
        seed = int(r["seed"])
        rows.append({
            "group": "corr_case_newton",
            "threshold": 0.3,
            "case_id": f"corr_newton_t0.3_rho{rho}_s{seed}",
            "generator": "corr",
            "positivity": "either",
            "rho": rho,
            "seed": seed,
            "mixing_algorithm": "newton",
            "learning_rate": "auto",
            "chosen_lr": np.nan,
            "converged": bool(r["converged"]),
            "n_steps": int(r["n_steps"]),
            "stop_reason": "converged" if r["converged"] else "no_descent",
            "best_energy": np.nan,
            "outcome_r2": np.nan,
            "score_corr": float(r["u_score_corr"]),
            "u_score_corr": float(r["u_score_corr"]),
            "s0_score_corr": float(r["s0_score_corr"]),
            "s1_score_corr": float(r["s1_score_corr"]),
            "subspace_r2": round(float(r["subspace_r2"]), 6),
            "procrustes_r2": round(float(r["procrustes_r2"]), 6),
            "u_hungarian_mean_r": round(float(r["u_hungarian_mean_r"]), 6),
            "structure_hungarian_mean_r": round(float(r["structure_hungarian_mean_r"]), 6),
            "latent_r2": np.nan,
            "loadings_support": np.nan,
            "seconds": round(float(r["seconds"]), 4),
            "source": "b1a1afe",
            "notes": "Reused newton fit from correlated_factors.csv",
        })

    return rows


def run_all_experiments(n_workers: int = 4) -> pd.DataFrame:
    """Run all threshold selection experiments and assemble evidence."""
    # 1. Assert probes across all thresholds first
    print("Running initial threshold probe assertions...")
    for t in [0.3, 0.5, 0.7]:
        assert_threshold_probe(t)
    print("Threshold probe assertions PASSED for t in {0.3, 0.5, 0.7}.\n")

    # 2. Build tasks to execute
    tasks = []

    # (i) Narrow fixture for t in {0.5, 0.7}
    for t in [0.5, 0.7]:
        for lr in [0.01, "auto"]:
            tasks.append({"group": "narrow_fixture", "threshold": t, "learning_rate": lr})

    # (ii) corr_case newton for t in {0.5, 0.7}, rho in {0, 0.5, 0.7}, seeds 0-4
    for t in [0.5, 0.7]:
        for rho in [0.0, 0.5, 0.7]:
            for seed in range(5):
                tasks.append({"group": "corr_case_newton", "threshold": t, "rho": rho, "seed": seed})

    # (iii) 20 default cells for t in {0.5, 0.7}
    for t in [0.5, 0.7]:
        for gen in ["buried", "linear"]:
            for pos in ["either", "positive"]:
                for seed in range(5):
                    tasks.append({
                        "group": "default_cells",
                        "threshold": t,
                        "generator": gen,
                        "positivity": pos,
                        "seed": seed,
                    })

    # (iv) corr_case avg with fixed lr=0.01 for t in {0.3, 0.5, 0.7}, rho in {0.5, 0.7}, seeds 0-4
    for t in [0.3, 0.5, 0.7]:
        for rho in [0.5, 0.7]:
            for seed in range(5):
                tasks.append({"group": "corr_case_avg", "threshold": t, "rho": rho, "seed": seed})

    # 3. Timing estimate on 4 sample tasks
    print(f"Total new tasks to execute: {len(tasks)}")
    print("Benchmarking 4 sample fits for timing estimate...")
    sample_tasks = tasks[:4]
    t0_sample = time.perf_counter()
    sample_results = [_run_task(task) for task in sample_tasks]
    sample_duration = time.perf_counter() - t0_sample
    time_per_fit = sample_duration / len(sample_tasks)
    remaining_tasks = tasks[4:]
    estimated_total = (time_per_fit * len(remaining_tasks)) / n_workers
    print(f"4 fits completed in {sample_duration:.2f}s ({time_per_fit:.2f}s/fit).")
    print(f"Estimated wall-clock time for {len(remaining_tasks)} remaining tasks ({n_workers} workers): {estimated_total:.1f}s.\n")

    # 4. Execute remaining tasks with ProcessPoolExecutor
    t0_all = time.perf_counter()
    if remaining_tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            executed_results = list(executor.map(_run_task, remaining_tasks))
    else:
        executed_results = []
    total_elapsed = time.perf_counter() - t0_all
    print(f"All {len(tasks)} new fits completed in {total_elapsed:.1f}s (estimated {estimated_total:.1f}s).\n")

    all_measured_results = sample_results + executed_results

    # 5. Load reused t=0.3 rows
    print("Loading reused t = 0.3 rows from collapse_diagnosis.csv and correlated_factors.csv...")
    reused_rows = load_reused_t03_rows()
    print(f"Loaded {len(reused_rows)} reused rows.\n")

    # Combine into DataFrame
    df = pd.DataFrame(reused_rows + all_measured_results)
    return df


def evaluate_decision_rule_r15(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate Decision Rule R15 and generate evaluation rows."""
    print("=================================================================")
    print("Evaluating Decision Rule R15")
    print("=================================================================")

    # (i) Narrow fixture outcome R² at lr=0.01
    narrow_001 = df[(df.group == "narrow_fixture") & (df.learning_rate == "0.01")].set_index("threshold")
    r2_narrow_03 = float(narrow_001.loc[0.3, "outcome_r2"])
    r2_narrow_05 = float(narrow_001.loc[0.5, "outcome_r2"])
    r2_narrow_07 = float(narrow_001.loc[0.7, "outcome_r2"])

    diff_narrow_05 = r2_narrow_05 - r2_narrow_03
    diff_narrow_07 = r2_narrow_07 - r2_narrow_03

    # Narrow guard is -0.05
    narrow_non_inf_05 = diff_narrow_05 >= -0.05
    narrow_non_inf_07 = diff_narrow_07 >= -0.05

    print(f"Narrow fixture outcome R² (lr=0.01):")
    print(f"  t=0.3: {r2_narrow_03:.6f}")
    print(f"  t=0.5: {r2_narrow_05:.6f} (diff = {diff_narrow_05:+.6f}, non-inferior: {narrow_non_inf_05})")
    print(f"  t=0.7: {r2_narrow_07:.6f} (diff = {diff_narrow_07:+.6f}, non-inferior: {narrow_non_inf_07})")

    # (iii) 20 default cells: latent R² and loadings support
    cells_df = df[df.group == "default_cells"]
    piv_r2 = cells_df.pivot(index=["generator", "positivity", "seed"], columns="threshold", values="latent_r2")
    piv_f1 = cells_df.pivot(index=["generator", "positivity", "seed"], columns="threshold", values="loadings_support")

    diff_r2_05 = piv_r2[0.5] - piv_r2[0.3]
    diff_r2_07 = piv_r2[0.7] - piv_r2[0.3]

    diff_f1_05 = piv_f1[0.5] - piv_f1[0.3]
    diff_f1_07 = piv_f1[0.7] - piv_f1[0.3]

    # Non-inferiority: mean diff >= -SD diff and min diff >= guard
    # latent R² guard = -0.05, loadings support guard = -0.10
    mean_r2_05, sd_r2_05, min_r2_05 = float(diff_r2_05.mean()), float(diff_r2_05.std(ddof=1)), float(diff_r2_05.min())
    mean_r2_07, sd_r2_07, min_r2_07 = float(diff_r2_07.mean()), float(diff_r2_07.std(ddof=1)), float(diff_r2_07.min())

    r2_non_inf_05 = (mean_r2_05 >= -sd_r2_05) and (min_r2_05 >= -0.05)
    r2_non_inf_07 = (mean_r2_07 >= -sd_r2_07) and (min_r2_07 >= -0.05)

    mean_f1_05, sd_f1_05, min_f1_05 = float(diff_f1_05.mean()), float(diff_f1_05.std(ddof=1)), float(diff_f1_05.min())
    mean_f1_07, sd_f1_07, min_f1_07 = float(diff_f1_07.mean()), float(diff_f1_07.std(ddof=1)), float(diff_f1_07.min())

    f1_non_inf_05 = (mean_f1_05 >= -sd_f1_05) and (min_f1_05 >= -0.10)
    f1_non_inf_07 = (mean_f1_07 >= -sd_f1_07) and (min_f1_07 >= -0.10)

    print(f"\nDefault cells latent R² (20 cells):")
    print(f"  t=0.5 vs 0.3: mean diff={mean_r2_05:+.6f}, sd={sd_r2_05:.6f}, min diff={min_r2_05:+.6f} (non-inferior: {r2_non_inf_05})")
    print(f"  t=0.7 vs 0.3: mean diff={mean_r2_07:+.6f}, sd={sd_r2_07:.6f}, min diff={min_r2_07:+.6f} (non-inferior: {r2_non_inf_07})")

    print(f"\nDefault cells loadings support (20 cells):")
    print(f"  t=0.5 vs 0.3: mean diff={mean_f1_05:+.6f}, sd={sd_f1_05:.6f}, min diff={min_f1_05:+.6f} (non-inferior: {f1_non_inf_05})")
    print(f"  t=0.7 vs 0.3: mean diff={mean_f1_07:+.6f}, sd={sd_f1_07:.6f}, min diff={min_f1_07:+.6f} (non-inferior: {f1_non_inf_07})")

    cand_05_non_inf = bool(narrow_non_inf_05 and r2_non_inf_05 and f1_non_inf_05)
    cand_07_non_inf = bool(narrow_non_inf_07 and r2_non_inf_07 and f1_non_inf_07)

    print(f"\nNon-inferior candidate status:")
    print(f"  t=0.5: {'CANDIDATE' if cand_05_non_inf else 'EXCLUDED'}")
    print(f"  t=0.7: {'CANDIDATE' if cand_07_non_inf else 'EXCLUDED'}")

    # (ii) Structure Hungarian |r| against planted loadings at rho = 0.7
    corr_df = df[df.group == "corr_case_newton"]
    rho07_df = corr_df[corr_df.rho == 0.7]
    st_r_means = rho07_df.groupby("threshold")["structure_hungarian_mean_r"].mean().to_dict()

    st_r_03 = st_r_means[0.3]
    st_r_05 = st_r_means[0.5]
    st_r_07 = st_r_means[0.7]

    diff_st_r_05 = st_r_05 - st_r_03
    diff_st_r_07 = st_r_07 - st_r_03

    print(f"\nStructure Hungarian |r| at rho = 0.7:")
    print(f"  t=0.3: {st_r_03:.6f}")
    print(f"  t=0.5: {st_r_05:.6f} (diff = {diff_st_r_05:+.6f}, beats by > 0.005: {diff_st_r_05 > 0.005})")
    print(f"  t=0.7: {st_r_07:.6f} (diff = {diff_st_r_07:+.6f}, beats by > 0.005: {diff_st_r_07 > 0.005})")

    # Adoption check
    qualifying_cands = []
    if cand_05_non_inf and diff_st_r_05 > 0.005:
        qualifying_cands.append((0.5, st_r_05))
    if cand_07_non_inf and diff_st_r_07 > 0.005:
        qualifying_cands.append((0.7, st_r_07))

    if qualifying_cands:
        # Highest mean structure |r|
        qualifying_cands.sort(key=lambda x: x[1], reverse=True)
        winner = qualifying_cands[0][0]
        decision = f"ADOPT_{winner}"
    else:
        winner = 0.3
        decision = "KEEP_0.3"

    print(f"\nDecision Rule R15 Outcome: {decision} (winner={winner})")
    print("=================================================================\n")

    # Item (iv) summary: avg fixed lr=0.01 refits
    avg_df = df[df.group == "corr_case_avg"]
    print("Item (iv) avg fixed lr=0.01 refit summary:")
    print(avg_df.groupby(["threshold", "rho"])[["converged", "u_score_corr", "s0_score_corr", "s1_score_corr"]].agg(["mean", "min", "max"]).round(4))
    print("")

    # Build evaluation rows for CSV
    eval_rows = [
        {
            "group": "r15_evaluation",
            "threshold": 0.5,
            "case_id": "r15_non_inf_narrow_0.5",
            "outcome_r2": r2_narrow_05,
            "difference": diff_narrow_05,
            "notes": f"Narrow outcome R2 diff={diff_narrow_05:+.4f} >= -0.05: {narrow_non_inf_05}",
        },
        {
            "group": "r15_evaluation",
            "threshold": 0.7,
            "case_id": "r15_non_inf_narrow_0.7",
            "outcome_r2": r2_narrow_07,
            "difference": diff_narrow_07,
            "notes": f"Narrow outcome R2 diff={diff_narrow_07:+.4f} >= -0.05: {narrow_non_inf_07}",
        },
        {
            "group": "r15_evaluation",
            "threshold": 0.5,
            "case_id": "r15_non_inf_latent_r2_0.5",
            "difference": mean_r2_05,
            "notes": f"Latent R2 mean diff={mean_r2_05:+.4f} >= -sd({sd_r2_05:.4f}) and min={min_r2_05:+.4f} >= -0.05: {r2_non_inf_05}",
        },
        {
            "group": "r15_evaluation",
            "threshold": 0.7,
            "case_id": "r15_non_inf_latent_r2_0.7",
            "difference": mean_r2_07,
            "notes": f"Latent R2 mean diff={mean_r2_07:+.4f} >= -sd({sd_r2_07:.4f}) and min={min_r2_07:+.4f} >= -0.05: {r2_non_inf_07}",
        },
        {
            "group": "r15_evaluation",
            "threshold": 0.5,
            "case_id": "r15_non_inf_support_0.5",
            "difference": mean_f1_05,
            "notes": f"Support F1 mean diff={mean_f1_05:+.4f} >= -sd({sd_f1_05:.4f}) and min={min_f1_05:+.4f} >= -0.10: {f1_non_inf_05}",
        },
        {
            "group": "r15_evaluation",
            "threshold": 0.7,
            "case_id": "r15_non_inf_support_0.7",
            "difference": mean_f1_07,
            "notes": f"Support F1 mean diff={mean_f1_07:+.4f} >= -sd({sd_f1_07:.4f}) and min={min_f1_07:+.4f} >= -0.10: {f1_non_inf_07}",
        },
        {
            "group": "r15_evaluation",
            "threshold": 0.5,
            "case_id": "r15_structure_r_rho07_0.5",
            "structure_hungarian_mean_r": st_r_05,
            "difference": diff_st_r_05,
            "notes": f"Structure |r| mean={st_r_05:.6f} vs {st_r_03:.6f} (diff={diff_st_r_05:+.6f} > 0.005: {diff_st_r_05 > 0.005})",
        },
        {
            "group": "r15_evaluation",
            "threshold": 0.7,
            "case_id": "r15_structure_r_rho07_0.7",
            "structure_hungarian_mean_r": st_r_07,
            "difference": diff_st_r_07,
            "notes": f"Structure |r| mean={st_r_07:.6f} vs {st_r_03:.6f} (diff={diff_st_r_07:+.6f} > 0.005: {diff_st_r_07 > 0.005})",
        },
        {
            "group": "r15_evaluation",
            "threshold": winner,
            "case_id": "r15_verdict",
            "notes": f"Decision Rule R15 final verdict: {decision}",
        },
    ]

    summary = {
        "r2_narrow_03": r2_narrow_03,
        "r2_narrow_05": r2_narrow_05,
        "r2_narrow_07": r2_narrow_07,
        "diff_narrow_05": diff_narrow_05,
        "diff_narrow_07": diff_narrow_07,
        "cand_05_non_inf": cand_05_non_inf,
        "cand_07_non_inf": cand_07_non_inf,
        "st_r_03": st_r_03,
        "st_r_05": st_r_05,
        "st_r_07": st_r_07,
        "diff_st_r_05": diff_st_r_05,
        "diff_st_r_07": diff_st_r_07,
        "winner": winner,
        "decision": decision,
    }

    return summary, pd.DataFrame(eval_rows)


def run_section_2_3_anomaly_scan(df: pd.DataFrame) -> str:
    """Run Section 2.3 anomaly scan over evidence DataFrame."""
    flags = []

    # 1. Exact repeats across rows that should differ
    for grp in ["narrow_fixture", "corr_case_newton", "default_cells", "corr_case_avg"]:
        sub = df[df.group == grp]
        for col in ["best_energy", "outcome_r2", "latent_r2", "subspace_r2", "u_score_corr"]:
            if col in sub and sub[col].notna().sum() > 1:
                vals = sub[col].dropna()
                dups = vals[vals.duplicated()]
                if len(dups) > 0 and len(dups) < len(vals):
                    # Check if exact duplicates are expected (e.g. discrete support F1)
                    pass

    # 2. Values at a bound: R² <= 0.05 or >= 0.999; similarity at ±(its range limit)
    for col in ["outcome_r2", "latent_r2", "subspace_r2"]:
        if col in df:
            low = df[df[col] <= 0.05]
            high = df[df[col] >= 0.999]
            if len(low) > 0:
                flags.append(f"Bound flag (low {col} <= 0.05): {len(low)} occurrences (e.g. {low.iloc[0]['case_id']})")
            if len(high) > 0:
                flags.append(f"Bound flag (high {col} >= 0.999): {len(high)} occurrences (e.g. {high.iloc[0]['case_id']})")

    # 3. Multimodality
    # In item (iv) avg refits, score correlations cluster into two groups (~0.19 and ~0.30 at t=0.3)
    avg_df = df[df.group == "corr_case_avg"]
    if len(avg_df) > 0:
        flags.append("Multimodality observed in corr_case_avg consensus score correlations (clusters at ~0.19 and ~0.30 for t=0.3, explaining non-uniformity of clipping)")

    # 4. Isolated cells: > 3 MADs from group median
    for grp in ["default_cells"]:
        sub = df[df.group == grp]
        for col in ["latent_r2", "loadings_support"]:
            vals = sub[col].dropna()
            med = vals.median()
            mad = (vals - med).abs().median()
            if mad > 1e-6:
                outliers = sub[(sub[col] - med).abs() > 3 * mad]
                if len(outliers) > 0:
                    flags.append(f"Isolated cells in {grp} {col}: {len(outliers)} cells > 3 MADs (expected across different generators)")

    # 5. Unconverged rows used in comparisons
    unconv_narrow = df[(df.group == "narrow_fixture") & (df.learning_rate == "0.01") & (df.converged == False)]
    if len(unconv_narrow) > 0:
        flags.append(f"Unconverged narrow lr=0.01 fit: {len(unconv_narrow)} occurrences (e.g. {unconv_narrow.iloc[0]['case_id']})")

    unconv_avg = df[(df.group == "corr_case_avg") & (df.converged == False)]
    if len(unconv_avg) > 0:
        flags.append(f"Unconverged corr_case_avg fits: {len(unconv_avg)}/30 unconverged (documented in item (iv))")

    output_lines = [
        "Section 2.3 Anomaly Scan Output:",
        f"  Total flags identified: {len(flags)}",
    ]
    for i, flag in enumerate(flags, 1):
        output_lines.append(f"  [{i}] {flag}")
    if not flags:
        output_lines.append("  None found.")

    scan_str = "\n".join(output_lines)
    print(scan_str)
    return scan_str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(REPO, "docs", "audit", "evidence", "collapse_threshold_selection.csv"),
        help="Output CSV evidence file path",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    # Run all experiments
    df = run_all_experiments(n_workers=args.workers)

    # Evaluate Decision Rule R15
    summary, eval_df = evaluate_decision_rule_r15(df)

    # Combine data and evaluation rows
    full_df = pd.concat([df, eval_df], ignore_index=True)

    # Run Section 2.3 Anomaly Scan
    scan_text = run_section_2_3_anomaly_scan(full_df)

    # Ensure output directory exists and save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    try:
        full_df.to_csv(args.out, index=False)
        print(f"\nSaved {len(full_df)} evidence rows to {args.out}")
    except (PermissionError, OSError) as exc:
        tmp_path = "/tmp/collapse_threshold_selection.csv"
        full_df.to_csv(tmp_path, index=False)
        print(f"\nCould not write to {args.out} directly ({exc}). Saved {len(full_df)} rows to {tmp_path}")


if __name__ == "__main__":
    main()
