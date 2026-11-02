#!/usr/bin/env python3
"""Milestone 1 (WP-1 & WP-1b): View weighting selection benchmark.

Compares `calibrated` (current init-relative) vs `equal` (1/M for scale-free
energies like 'nc') or `candidate` (data-relative scale for 'acc') on default
simlr fits (`pysimlr.fit("simlr", ...)`).

Measurements:
  (a) Default cells: {buried, linear} x {either, positive} x seeds 0-4 (20 cells)
      - latent R^2
      - convergence (converged boolean and n_steps)
      - weight spread (max(w) / min(w))
      - loadings and v support F1
      - loadings and v stability (identical 98% row subsamples across views)
  (b) Init-invariance:
      - mean sin^2 distance between PCA start and random starts (seeds 1 and 2)
        on u and on structure blocks.
  (c) Narrow-view collapse fixture:
      - outcome R^2 and score correlation with collapse_weight at 1.0 and 0.0.
  (d) NonnegParts support recovery:
      - seeds 42-44, Support_Recovery_L.
  (e) ADNI-MCI:
      - held-out metric (seeds 42-44, pos in {either, positive})
      - select_rank per-component stability at k=2 and k=3.

Applies Decision Rule R1 (for 'nc') and Decision Rule R9 (for 'acc').
Writes CSV to --out (default docs/audit/evidence/view_weighting_selection.csv).

Usage:
  python scripts/view_weighting_selection.py --energy nc --out docs/audit/evidence/view_weighting_selection.csv
  python scripts/view_weighting_selection.py --energy acc --out docs/audit/evidence/view_weighting_acc.csv
"""
import argparse
import itertools
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import numpy as np
import pandas as pd
import torch

import pysimlr
import pysimlr.defaults
import pysimlr.similarity
from amigo_loadings_evaluation import span_instability
from default_selection_benchmark import _case
from optimizer_benchmark_synthetic import latent_r2, support_f1
from run_deep_benchmark import get_adni_mci_case, run_experiment_task
from test_collapse_penalty import _narrow, _outcome_r2

_BASE_SELF_NORMALISED = frozenset(pysimlr.similarity.SELF_NORMALISED)
_BASE_DEFAULT_ENERGY = pysimlr.defaults.DEFAULT_ENERGY_TYPE


def set_weighting_mode(energy: str, mode: str):
    """Set weighting mode in the current process with strict state management."""
    # Step 1: ALWAYS restore baseline state first (Rule 6)
    pysimlr.similarity.SELF_NORMALISED = _BASE_SELF_NORMALISED
    pysimlr.defaults.DEFAULT_ENERGY_TYPE = _BASE_DEFAULT_ENERGY

    simlr_mod = sys.modules.get("pysimlr.simlr")
    if simlr_mod is not None and hasattr(simlr_mod, "_orig_calculate_simlr_energy"):
        simlr_mod.calculate_simlr_energy = simlr_mod._orig_calculate_simlr_energy

    if energy == "nc":
        if mode in ("equal", "candidate"):
            pysimlr.similarity.SELF_NORMALISED = _BASE_SELF_NORMALISED | {"nc"}
        elif mode == "calibrated":
            pysimlr.similarity.SELF_NORMALISED = _BASE_SELF_NORMALISED - {"nc"}
    elif energy == "acc":
        pysimlr.defaults.DEFAULT_ENERGY_TYPE = "acc"
        if mode in ("equal", "candidate"):
            if simlr_mod is not None:
                if not hasattr(simlr_mod, "_orig_calculate_simlr_energy"):
                    simlr_mod._orig_calculate_simlr_energy = simlr_mod.calculate_simlr_energy
                orig_fn = simlr_mod._orig_calculate_simlr_energy

                def candidate_calc(v, x, u, energy_type="nc", lambda_val=0.0, prior_matrix=None):
                    caller = sys._getframe(1).f_code.co_name
                    if caller == "_calibrate_weights" and energy_type == "acc":
                        n = x.shape[0]
                        s_m = (torch.linalg.norm(x, 2) * torch.linalg.norm(u, "fro") / (n - 1)).clamp_min(1e-10)
                        return s_m
                    return orig_fn(v, x, u, energy_type=energy_type, lambda_val=lambda_val, prior_matrix=prior_matrix)

                simlr_mod.calculate_simlr_energy = candidate_calc
        elif mode == "calibrated":
            pass


def _blocks(res, obj):
    return [b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b) for b in res[obj]]


def run_default_cell_task(task_args: tuple) -> list:
    """Run default cell and init-invariance for both calibrated and alternative modes."""
    generator, pos, seed, energy = task_args
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)

    alt_name = "equal" if energy == "nc" else "candidate"
    z, views, sups = _case(generator, seed)
    n = views[0].shape[0]
    k = z.shape[1]

    # Generate row subsamples ONCE per cell with identical rows across all views (Rule 6)
    rng = np.random.default_rng(11 + seed)
    sub_n = int(0.98 * n)
    idx1 = torch.as_tensor(np.sort(rng.choice(n, sub_n, replace=False)))
    idx2 = torch.as_tensor(np.sort(rng.choice(n, sub_n, replace=False)))
    views_sub1 = [x[idx1] for x in views]
    views_sub2 = [x[idx2] for x in views]

    results_by_mode = {}
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            # Full fit (PCA init default)
            full = pysimlr.fit("simlr", views, k, iterations=25, energy_type=energy,
                               positivity=pos, random_state=seed)

            # Sanity assertion for scale-free equal weights
            if mode == "equal" and energy == "nc":
                expected_w = 1.0 / len(full["normalizing_weights"])
                assert all(abs(w - expected_w) < 1e-6 for w in full["normalizing_weights"]), \
                    f"Weight leakage: {full['normalizing_weights']} != {expected_w}"

            lat_r2 = latent_r2(z, full["u"])
            conv = bool(full["convergence"]["converged"])
            steps = int(full["convergence"]["n_steps"])
            w = full["normalizing_weights"]
            spread = float(max(w) / max(min(w), 1e-12))
            v_sup_f1 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(full["v"], sups)]))
            l_sup_f1 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(full["loadings"], sups)]))

            # Subsample fits for loadings/V stability
            fit_s1 = pysimlr.fit("simlr", views_sub1, k, iterations=25, energy_type=energy,
                                 positivity=pos, random_state=seed)
            fit_s2 = pysimlr.fit("simlr", views_sub2, k, iterations=25, energy_type=energy,
                                 positivity=pos, random_state=seed)
            l_instab = span_instability(_blocks(fit_s1, "loadings"), _blocks(fit_s2, "loadings"))
            v_instab = span_instability(_blocks(fit_s1, "v"), _blocks(fit_s2, "v"))

            # Init-invariance (b): random starts 1 and 2 vs PCA
            r_fit1 = pysimlr.fit("simlr", views, k, iterations=25, energy_type=energy,
                                 positivity=pos, initialization_type="random", random_state=1)
            r_fit2 = pysimlr.fit("simlr", views, k, iterations=25, energy_type=energy,
                                 positivity=pos, initialization_type="random", random_state=2)
            u_init_inv = 0.5 * (span_instability([full["u"]], [r_fit1["u"]]) +
                                span_instability([full["u"]], [r_fit2["u"]]))
            struct_init_inv = 0.5 * (span_instability(_blocks(full, "structure"), _blocks(r_fit1, "structure")) +
                                     span_instability(_blocks(full, "structure"), _blocks(r_fit2, "structure")))

            results_by_mode[mode] = {
                "latent_r2": float(lat_r2),
                "converged": float(conv),
                "n_steps": float(steps),
                "weight_spread": float(spread),
                "v_support_f1": float(v_sup_f1),
                "loadings_support_f1": float(l_sup_f1),
                "loadings_instability": float(l_instab),
                "v_instability": float(v_instab),
                "u_init_invariance": float(u_init_inv),
                "structure_init_invariance": float(struct_init_inv),
            }
        finally:
            set_weighting_mode(energy, "calibrated")

    # Format paired output rows
    case_id = f"{generator}_{pos}_s{seed}"
    records = []
    for metric_name in results_by_mode["calibrated"]:
        c_val = results_by_mode["calibrated"][metric_name]
        a_val = results_by_mode[alt_name][metric_name]
        grp = "init_invariance" if "init_invariance" in metric_name else "default_cells"
        records.append({
            "group": grp,
            "case_id": case_id,
            "generator": generator,
            "positivity": pos,
            "seed": seed,
            "k": k,
            "metric": metric_name,
            "calibrated": c_val,
            "alternative": a_val,
            "difference": a_val - c_val,
        })
    return records


def run_narrow_fixture(energy: str) -> list:
    """Run narrow collapse fixture (c) for calibrated and alternative modes."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    alt_name = "equal" if energy == "nc" else "candidate"

    x, y = _narrow()
    results = {}
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            # cw = 1.0
            r_cw1 = pysimlr.fit("simlr", x, k=2, iterations=30, positivity="positive", sparseness=0.5,
                                energy_type=energy, random_state=0, collapse_weight=1.0)
            r2_cw1 = _outcome_r2(x, y, r_cw1["v"])
            sc1 = x[0] @ r_cw1["v"][0]
            corr_cw1 = abs(float(torch.corrcoef(sc1.T)[0, 1]))

            # cw = 0.0
            r_cw0 = pysimlr.fit("simlr", x, k=2, iterations=30, positivity="positive", sparseness=0.5,
                                energy_type=energy, random_state=0, collapse_weight=0.0)
            r2_cw0 = _outcome_r2(x, y, r_cw0["v"])
            sc0 = x[0] @ r_cw0["v"][0]
            corr_cw0 = abs(float(torch.corrcoef(sc0.T)[0, 1]))

            results[mode] = {
                "outcome_r2_cw1": float(r2_cw1),
                "score_corr_cw1": float(corr_cw1),
                "outcome_r2_cw0": float(r2_cw0),
                "score_corr_cw0": float(corr_cw0),
            }
        finally:
            set_weighting_mode(energy, "calibrated")

    records = []
    for metric_name in results["calibrated"]:
        c_val = results["calibrated"][metric_name]
        a_val = results[alt_name][metric_name]
        records.append({
            "group": "narrow_fixture",
            "case_id": "narrow_k2",
            "generator": "narrow",
            "positivity": "positive",
            "seed": 0,
            "k": 2,
            "metric": metric_name,
            "calibrated": c_val,
            "alternative": a_val,
            "difference": a_val - c_val,
        })
    return records


def run_nonneg_task(task_args: tuple) -> list:
    """Run NonnegParts support recovery (d) for calibrated and alternative modes."""
    seed, energy = task_args
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    alt_name = "equal" if energy == "nc" else "candidate"

    results = {}
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            task = ("NonnegParts", "synthetic", "simlr", "SiMLR", seed, 25, 45, 0.5, "positive")
            row = run_experiment_task(task)
            results[mode] = {
                "support_recovery_loadings": float(row.get("Support_Recovery_L", float("nan"))),
                "support_recovery_v": float(row.get("Support_Recovery_V", float("nan"))),
            }
        finally:
            set_weighting_mode(energy, "calibrated")

    records = []
    for metric_name in results["calibrated"]:
        c_val = results["calibrated"][metric_name]
        a_val = results[alt_name][metric_name]
        records.append({
            "group": "nonneg_parts",
            "case_id": f"nonneg_s{seed}",
            "generator": "nonneg_parts",
            "positivity": "positive",
            "seed": seed,
            "k": 3,
            "metric": metric_name,
            "calibrated": c_val,
            "alternative": a_val,
            "difference": a_val - c_val,
        })
    return records


def run_adni_heldout_task(task_args: tuple) -> list:
    """Run ADNI-MCI held-out evaluation for calibrated and alternative modes."""
    seed, pos, energy = task_args
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    alt_name = "equal" if energy == "nc" else "candidate"

    results = {}
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            task = ("ADNI-MCI", "real", "simlr", "SiMLR", seed, 25, 45, 0.5, pos)
            row = run_experiment_task(task)
            results[mode] = {
                "adni_predictive_metric": float(row.get("Predictive_Metric", float("nan"))),
            }
        finally:
            set_weighting_mode(energy, "calibrated")

    records = []
    for metric_name in results["calibrated"]:
        c_val = results["calibrated"][metric_name]
        a_val = results[alt_name][metric_name]
        records.append({
            "group": "adni_heldout",
            "case_id": f"adni_{pos}_s{seed}",
            "generator": "adni_mci",
            "positivity": pos,
            "seed": seed,
            "k": 3,
            "metric": metric_name,
            "calibrated": c_val,
            "alternative": a_val,
            "difference": a_val - c_val,
        })
    return records


def run_adni_stability_task(task_args: tuple) -> list:
    """Run ADNI-MCI select_rank component stability for calibrated and alternative modes."""
    pos, energy = task_args
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    alt_name = "equal" if energy == "nc" else "candidate"

    case = get_adni_mci_case(42)
    results = {}
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            sr = pysimlr.select_rank("simlr", case["data"], [2, 3], n_subsamples=20,
                                     random_state=42, positivity=pos, topology="graph",
                                     path_graph=case["path_graph"], energy_type=energy)
            table = {r["k"]: r for r in sr["table"]}
            results[mode] = {
                "min_component_stability_k2": float(table[2]["min_stability"]),
                "min_component_stability_k3": float(table[3]["min_stability"]),
            }
        finally:
            set_weighting_mode(energy, "calibrated")

    records = []
    for metric_name in results["calibrated"]:
        k_val = 2 if "k2" in metric_name else 3
        c_val = results["calibrated"][metric_name]
        a_val = results[alt_name][metric_name]
        records.append({
            "group": "adni_stability",
            "case_id": f"adni_stab_{pos}_k{k_val}",
            "generator": "adni_mci",
            "positivity": pos,
            "seed": 42,
            "k": k_val,
            "metric": metric_name,
            "calibrated": c_val,
            "alternative": a_val,
            "difference": a_val - c_val,
        })
    return records


def evaluate_decision_rules(df: pd.DataFrame, energy: str) -> tuple:
    """Evaluate Decision Rule R1 (for 'nc') or Rule R9 (for 'acc') from paired data."""
    alt_label = "equal" if energy == "nc" else "candidate"
    print("\n" + "=" * 80)
    print(f"EVALUATING DECISION RULE FOR ENERGY '{energy}' (Calibrated vs. {alt_label})")
    print("=" * 80)

    # Directionality: higher is better vs lower is better
    higher_is_better = {
        "latent_r2", "v_support_f1", "loadings_support_f1", "outcome_r2_cw1",
        "support_recovery_loadings", "support_recovery_v",
        "adni_predictive_metric", "min_component_stability_k2", "min_component_stability_k3",
    }
    lower_is_better = {
        "loadings_instability", "v_instability", "weight_spread", "n_steps",
        "u_init_invariance", "structure_init_invariance",
    }

    summary_rows = []
    all_non_inferior = True

    metrics = df["metric"].unique()
    for metric in sorted(metrics):
        sub = df[df["metric"] == metric]
        diffs = sub["difference"].values
        n = len(diffs)
        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        seed_sd = float(np.std(sub["calibrated"].values, ddof=1)) if n > 1 else 0.0
        threshold = max(sd_d, seed_sd)
        t_stat = float(mean_d / (sd_d / math.sqrt(n))) if (sd_d > 1e-12 and n > 1) else 0.0

        if metric in higher_is_better:
            direction = "higher"
            # Worse beyond noise if mean_d < 0 and |mean_d| > 1.0 * threshold (paired or seed SD)
            worse_beyond_noise = (mean_d < 0) and (abs(mean_d) > threshold)
            wins = int(np.sum(diffs > 1e-6))
            losses = int(np.sum(diffs < -1e-6))
            ties = n - wins - losses
        elif metric in lower_is_better:
            direction = "lower"
            # Worse beyond noise if mean_d > 0 and mean_d > 1.0 * threshold (paired or seed SD)
            worse_beyond_noise = (mean_d > 0) and (mean_d > threshold)
            wins = int(np.sum(diffs < -1e-6))
            losses = int(np.sum(diffs > 1e-6))
            ties = n - wins - losses
        else:
            direction = "neutral"
            worse_beyond_noise = False
            wins = int(np.sum(np.abs(diffs) > 1e-6))
            losses = 0
            ties = n - wins

        status = "FAIL (Worse beyond noise)" if worse_beyond_noise else "PASS (Non-inferior)"
        if worse_beyond_noise:
            all_non_inferior = False

        summary_rows.append({
            "metric": metric,
            "direction": direction,
            "n_pairs": n,
            "calibrated_mean": float(sub["calibrated"].mean()),
            "alt_mean": float(sub["alternative"].mean()),
            "mean_diff": mean_d,
            "sd_diff": sd_d,
            "paired_t": t_stat,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "status": status,
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("-" * 80)

    if energy == "nc":
        print(f"DECISION RULE R1 OUTCOME: {'ADOPT EQUAL 1/M WEIGHTS' if all_non_inferior else 'RETAIN CALIBRATED WEIGHTS'}")
        print(f"Criteria met: All metrics non-inferior = {all_non_inferior}")
    elif energy == "acc":
        print(f"DECISION RULE R9 OUTCOME: {'ADOPT CANDIDATE DATA-RELATIVE SCALE' if all_non_inferior else 'RETAIN CALIBRATED WEIGHTS'}")
        print(f"Criteria met: Candidate scale non-inferior = {all_non_inferior}")
    print("=" * 80 + "\n")
    return all_non_inferior, summary_df


def main():
    parser = argparse.ArgumentParser(description="View weighting selection experiment.")
    parser.add_argument("--energy", choices=["nc", "acc"], default="nc",
                        help="Energy type to benchmark (default: nc).")
    parser.add_argument("--out", default=None,
                        help="Output path for the generated evidence CSV.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker count for ProcessPoolExecutor.")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(REPO, "docs", "audit", "evidence",
                                f"view_weighting_{'selection' if args.energy == 'nc' else 'acc'}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.perf_counter()
    print(f"Starting view weighting selection experiment for energy='{args.energy}'...")
    print(f"Destination CSV: {args.out}")
    print(f"Parallel workers: {args.workers}")

    all_records = []

    # 1. Default-selection cells (a) & init-invariance (b): 20 cells
    default_tasks = [
        (gen, pos, seed, args.energy)
        for gen in ("buried", "linear")
        for pos in ("either", "positive")
        for seed in range(5)
    ]
    print(f"\n[1/5] Running {len(default_tasks)} default cells across {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_default_cell_task, t) for t in default_tasks]
        for f in as_completed(futures):
            all_records.extend(f.result())
    print("Default cells complete.")

    # 2. Narrow-view collapse fixture (c): single fixture
    print("\n[2/5] Running narrow-view collapse fixture...")
    all_records.extend(run_narrow_fixture(args.energy))
    print("Narrow fixture complete.")

    # 3. NonnegParts support recovery (d): seeds 42-44
    nonneg_tasks = [(s, args.energy) for s in (42, 43, 44)]
    print(f"\n[3/5] Running {len(nonneg_tasks)} NonnegParts tasks...")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(nonneg_tasks))) as executor:
        futures = [executor.submit(run_nonneg_task, t) for t in nonneg_tasks]
        for f in as_completed(futures):
            all_records.extend(f.result())
    print("NonnegParts tasks complete.")

    # 4. ADNI-MCI held-out (e): 3 seeds x 2 pos = 6 tasks
    adni_heldout_tasks = [
        (s, pos, args.energy)
        for s in (42, 43, 44)
        for pos in ("either", "positive")
    ]
    print(f"\n[4/5] Running {len(adni_heldout_tasks)} ADNI-MCI held-out tasks...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_adni_heldout_task, t) for t in adni_heldout_tasks]
        for f in as_completed(futures):
            all_records.extend(f.result())
    print("ADNI-MCI held-out complete.")

    # 5. ADNI-MCI select_rank stability (e): 2 pos = 2 tasks
    adni_stab_tasks = [(pos, args.energy) for pos in ("either", "positive")]
    print(f"\n[5/5] Running {len(adni_stab_tasks)} ADNI-MCI select_rank stability tasks...")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(adni_stab_tasks))) as executor:
        futures = [executor.submit(run_adni_stability_task, t) for t in adni_stab_tasks]
        for f in as_completed(futures):
            all_records.extend(f.result())
    print("ADNI-MCI select_rank complete.")

    df = pd.DataFrame(all_records)
    # Reorder columns nicely
    cols = ["group", "case_id", "generator", "positivity", "seed", "k", "metric",
            "calibrated", "alternative", "difference"]
    df = df[cols]
    df.to_csv(args.out, index=False)
    elapsed = time.perf_counter() - t0
    print(f"\nWrote {len(df)} paired measurements to {args.out} in {elapsed:.1f} s.")

    passed, summary_df = evaluate_decision_rules(df, args.energy)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
