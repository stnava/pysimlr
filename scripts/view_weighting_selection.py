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
      - outcome R^2 and score correlation with collapse_weight at 1.0 and 0.0,
        both at default budget and at iterations=30.
  (d) NonnegParts support recovery:
      - seeds 42-44, Support_Recovery_L and Support_Recovery_V at default budget.
  (e) ADNI-MCI:
      - held-out metric (seeds 42-44, pos in {either, positive}) at default budget
      - select_rank per-component stability at k=2 and k=3 (reused from acd08fe for nc).

Applies Decision Rule R1 (for 'nc') and Decision Rule R9 (for 'acc').
Writes CSV to --out (default docs/audit/evidence/view_weighting_selection.csv).

Usage:
  python scripts/view_weighting_selection.py --energy nc --out docs/audit/evidence/view_weighting_selection.csv
  python scripts/view_weighting_selection.py --energy acc --out docs/audit/evidence/view_weighting_acc.csv
  python scripts/view_weighting_selection.py --energy nc --groups narrow_fixture,nonneg_parts,adni_heldout
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

import importlib
import pysimlr
import pysimlr.defaults
import pysimlr.similarity
simlr_module = importlib.import_module("pysimlr.simlr")
from amigo_loadings_evaluation import span_instability
from default_selection_benchmark import _case
from optimizer_benchmark_synthetic import latent_r2, support_f1
from run_deep_benchmark import get_adni_mci_case, run_experiment_task
from test_collapse_penalty import _narrow, _outcome_r2

_BASE_SCALE_FREE_ENERGIES = frozenset(pysimlr.similarity.SCALE_FREE_ENERGIES)
_BASE_VIEW_ENERGY_SCALE = simlr_module._view_energy_scale


def set_weighting_mode(energy: str, mode: str):
    """Set weighting mode in the current process with strict state management."""
    # Step 1: ALWAYS restore baseline state first (Rule 6 / Rule 8)
    pysimlr.similarity.SCALE_FREE_ENERGIES = _BASE_SCALE_FREE_ENERGIES
    simlr_module._view_energy_scale = _BASE_VIEW_ENERGY_SCALE

    if energy == "nc":
        if mode in ("equal", "candidate"):
            pysimlr.similarity.SCALE_FREE_ENERGIES = _BASE_SCALE_FREE_ENERGIES | {"nc"}
        elif mode == "calibrated":
            pysimlr.similarity.SCALE_FREE_ENERGIES = _BASE_SCALE_FREE_ENERGIES - {"nc"}
    elif energy == "acc":
        if mode in ("equal", "candidate"):
            def candidate_view_scale(energy_type, x, u, sim_e):
                if energy_type == "acc":
                    scale = (torch.linalg.norm(x, ord=2) * torch.linalg.norm(u, "fro")) / (x.shape[0] - 1)
                    return float(scale.item())
                return _BASE_VIEW_ENERGY_SCALE(energy_type, x, u, sim_e)

            simlr_module._view_energy_scale = candidate_view_scale
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
    alt_conv = 1.0
    alt_steps = 0.0
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            # Full fit (PCA init default, default 300 / DEFAULT_TOL budget)
            full = pysimlr.fit("simlr", views, k, energy_type=energy,
                               positivity=pos, random_state=seed)

            # Sanity assertion for scale-free equal weights / candidate scale
            if energy == "nc":
                expected_w = 1.0 / len(full["normalizing_weights"])
                if mode == "equal":
                    assert all(abs(w - expected_w) < 1e-6 for w in full["normalizing_weights"]), \
                        f"Weight leakage: {full['normalizing_weights']} != {expected_w}"
                elif mode == "calibrated":
                    assert not all(abs(w - expected_w) < 1e-6 for w in full["normalizing_weights"]), \
                        f"Expected calibrated weights, but got equal weights: {full['normalizing_weights']}"
            elif energy == "acc":
                if mode == "candidate":
                    assert simlr_module._view_energy_scale is not _BASE_VIEW_ENERGY_SCALE, \
                        "Candidate patch was not applied in candidate mode"
                elif mode == "calibrated":
                    assert simlr_module._view_energy_scale is _BASE_VIEW_ENERGY_SCALE, \
                        "Baseline scale was not restored in calibrated mode"

            lat_r2 = latent_r2(z, full["u"])
            conv = bool(full["convergence"]["converged"])
            steps = int(full["convergence"]["n_steps"])
            if mode == alt_name:
                alt_conv = float(conv)
                alt_steps = float(steps)

            w = full["normalizing_weights"]
            spread = float(max(w) / max(min(w), 1e-12))
            v_sup_f1 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(full["v"], sups)]))
            l_sup_f1 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(full["loadings"], sups)]))

            # Subsample fits for loadings/V stability (default budget)
            fit_s1 = pysimlr.fit("simlr", views_sub1, k, energy_type=energy,
                                 positivity=pos, random_state=seed)
            fit_s2 = pysimlr.fit("simlr", views_sub2, k, energy_type=energy,
                                 positivity=pos, random_state=seed)
            l_instab = span_instability(_blocks(fit_s1, "loadings"), _blocks(fit_s2, "loadings"))
            v_instab = span_instability(_blocks(fit_s1, "v"), _blocks(fit_s2, "v"))

            # Init-invariance (b): random starts 1 and 2 vs PCA (default budget)
            r_fit1 = pysimlr.fit("simlr", views, k, energy_type=energy,
                                 positivity=pos, initialization_type="random", random_state=1)
            r_fit2 = pysimlr.fit("simlr", views, k, energy_type=energy,
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
    if energy == "acc":
        diffs = [abs(results_by_mode["calibrated"][m] - results_by_mode[alt_name][m])
                 for m in ("weight_spread", "latent_r2", "loadings_instability", "u_init_invariance")]
        assert any(d > 1e-5 for d in diffs), \
            f"Candidate arm produced identical results to calibrated in cell {case_id}"
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
            "converged": alt_conv,
            "n_steps": alt_steps,
            "source": "measured",
        })
    return records


def run_narrow_fixture(energy: str) -> list:
    """Run narrow collapse fixture (c) at default budget AND at iterations=30 for cw in {0, 1}."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    alt_name = "equal" if energy == "nc" else "candidate"

    x, y = _narrow()
    results = {}
    convergence_info = {}
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            # 1. Default budget (iterations=None -> 300 / DEFAULT_TOL)
            r_cw1 = pysimlr.fit("simlr", x, k=2, positivity="positive", sparseness=0.5,
                                energy_type=energy, random_state=0, collapse_weight=1.0)
            if energy == "nc":
                expected_w = 1.0 / len(r_cw1["normalizing_weights"])
                if mode == "equal":
                    assert all(abs(w - expected_w) < 1e-6 for w in r_cw1["normalizing_weights"]), \
                        f"Weight leakage: {r_cw1['normalizing_weights']} != {expected_w}"
                elif mode == "calibrated":
                    assert not all(abs(w - expected_w) < 1e-6 for w in r_cw1["normalizing_weights"]), \
                        f"Expected calibrated weights, but got equal weights: {r_cw1['normalizing_weights']}"
            elif energy == "acc":
                if mode == "candidate":
                    assert simlr_module._view_energy_scale is not _BASE_VIEW_ENERGY_SCALE, \
                        "Candidate patch was not applied in candidate mode"
                elif mode == "calibrated":
                    assert simlr_module._view_energy_scale is _BASE_VIEW_ENERGY_SCALE, \
                        "Baseline scale was not restored in calibrated mode"

            r2_cw1 = _outcome_r2(x, y, r_cw1["v"])
            sc1 = x[0] @ r_cw1["v"][0]
            corr_cw1 = abs(float(torch.corrcoef(sc1.T)[0, 1]))

            r_cw0 = pysimlr.fit("simlr", x, k=2, positivity="positive", sparseness=0.5,
                                energy_type=energy, random_state=0, collapse_weight=0.0)
            r2_cw0 = _outcome_r2(x, y, r_cw0["v"])
            sc0 = x[0] @ r_cw0["v"][0]
            corr_cw0 = abs(float(torch.corrcoef(sc0.T)[0, 1]))

            # 2. Iterations=30 budget (continuity with test suite fixture)
            r_cw1_30 = pysimlr.fit("simlr", x, k=2, positivity="positive", sparseness=0.5,
                                   energy_type=energy, random_state=0, collapse_weight=1.0, iterations=30)
            r2_cw1_30 = _outcome_r2(x, y, r_cw1_30["v"])
            sc1_30 = x[0] @ r_cw1_30["v"][0]
            corr_cw1_30 = abs(float(torch.corrcoef(sc1_30.T)[0, 1]))

            r_cw0_30 = pysimlr.fit("simlr", x, k=2, positivity="positive", sparseness=0.5,
                                   energy_type=energy, random_state=0, collapse_weight=0.0, iterations=30)
            r2_cw0_30 = _outcome_r2(x, y, r_cw0_30["v"])
            sc0_30 = x[0] @ r_cw0_30["v"][0]
            corr_cw0_30 = abs(float(torch.corrcoef(sc0_30.T)[0, 1]))

            results[mode] = {
                # Default budget rows (used by R1)
                "outcome_r2_cw1": float(r2_cw1),
                "score_corr_cw1": float(corr_cw1),
                "outcome_r2_cw0": float(r2_cw0),
                "score_corr_cw0": float(corr_cw0),
                # 30-iteration rows (continuity with test suite)
                "outcome_r2_cw1_it30": float(r2_cw1_30),
                "score_corr_cw1_it30": float(corr_cw1_30),
                "outcome_r2_cw0_it30": float(r2_cw0_30),
                "score_corr_cw0_it30": float(corr_cw0_30),
            }
            if mode == alt_name:
                convergence_info["default"] = (
                    float(bool(r_cw1["convergence"]["converged"])),
                    float(r_cw1["convergence"]["n_steps"])
                )
                convergence_info["it30"] = (
                    float(bool(r_cw1_30["convergence"]["converged"])),
                    float(r_cw1_30["convergence"]["n_steps"])
                )
        finally:
            set_weighting_mode(energy, "calibrated")

    records = []
    # Default budget records
    conv_def, steps_def = convergence_info["default"]
    for metric_name in ("outcome_r2_cw1", "score_corr_cw1", "outcome_r2_cw0", "score_corr_cw0"):
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
            "converged": conv_def,
            "n_steps": steps_def,
            "source": "measured",
        })

    # 30-iteration records
    conv_30, steps_30 = convergence_info["it30"]
    for metric_name in ("outcome_r2_cw1_it30", "score_corr_cw1_it30", "outcome_r2_cw0_it30", "score_corr_cw0_it30"):
        c_val = results["calibrated"][metric_name]
        a_val = results[alt_name][metric_name]
        records.append({
            "group": "narrow_fixture",
            "case_id": "narrow_k2_it30",
            "generator": "narrow",
            "positivity": "positive",
            "seed": 0,
            "k": 2,
            "metric": metric_name,
            "calibrated": c_val,
            "alternative": a_val,
            "difference": a_val - c_val,
            "converged": conv_30,
            "n_steps": steps_30,
            "source": "measured",
        })
    return records


def run_nonneg_task(task_args: tuple) -> list:
    """Run NonnegParts support recovery (d) at default budget for calibrated and alternative modes."""
    seed, energy = task_args
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    alt_name = "equal" if energy == "nc" else "candidate"

    results = {}
    alt_conv = 1.0
    alt_steps = 0.0
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            # Explicit default budget and energy_type in 10-tuple (Section 3 & Section 5 WP-1R step 3)
            task = ("NonnegParts", "synthetic", "simlr", "SiMLR", seed,
                    pysimlr.defaults.DEFAULT_SIMLR_ITERATIONS, 45, 0.5, "positive", energy)
            row = run_experiment_task(task)
            nw = row.get("normalizing_weights")
            assert nw is not None, f"normalizing_weights missing in NonnegParts task for mode={mode}"
            expected_w = 1.0 / len(nw)
            if energy == "nc":
                if mode == "equal":
                    assert all(abs(w - expected_w) < 1e-6 for w in nw), f"Expected equal weights, got {nw}"
                elif mode == "calibrated":
                    assert not all(abs(w - expected_w) < 1e-6 for w in nw), f"Expected calibrated weights, got {nw}"
            elif energy == "acc":
                if mode == "candidate":
                    assert simlr_module._view_energy_scale is not _BASE_VIEW_ENERGY_SCALE, \
                        "Candidate patch was not applied in candidate mode"
                elif mode == "calibrated":
                    assert simlr_module._view_energy_scale is _BASE_VIEW_ENERGY_SCALE, \
                        "Baseline scale was not restored in calibrated mode"

            if mode == alt_name:
                alt_conv = float(bool(row.get("Solver_Converged", True)))
                alt_steps = float(row.get("Solver_Iters", pysimlr.defaults.DEFAULT_SIMLR_ITERATIONS))

            results[mode] = {
                "support_recovery_loadings": float(row.get("Support_Recovery_L", float("nan"))),
                "support_recovery_v": float(row.get("Support_Recovery_V", float("nan"))),
                "normalizing_weights": nw,
            }
        finally:
            set_weighting_mode(energy, "calibrated")

    if energy == "acc":
        assert not np.allclose(results["calibrated"]["normalizing_weights"],
                               results[alt_name]["normalizing_weights"], rtol=1e-4, atol=1e-4), \
            f"Candidate weights did not differ from calibrated weights in nonneg_parts seed {seed}"
        del results["calibrated"]["normalizing_weights"]
        del results[alt_name]["normalizing_weights"]

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
            "converged": alt_conv,
            "n_steps": alt_steps,
            "source": "measured",
        })
    return records


def run_adni_heldout_task(task_args: tuple) -> list:
    """Run ADNI-MCI held-out evaluation at default budget for calibrated and alternative modes."""
    seed, pos, energy = task_args
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    alt_name = "equal" if energy == "nc" else "candidate"

    results = {}
    alt_conv = 1.0
    alt_steps = 0.0
    for mode in ("calibrated", alt_name):
        set_weighting_mode(energy, mode)
        try:
            # Explicit default budget and energy_type in 10-tuple (Section 3 & Section 5 WP-1R step 3)
            task = ("ADNI-MCI", "real", "simlr", "SiMLR", seed,
                    pysimlr.defaults.DEFAULT_SIMLR_ITERATIONS, 45, 0.5, pos, energy)
            row = run_experiment_task(task)
            nw = row.get("normalizing_weights")
            assert nw is not None, f"normalizing_weights missing in ADNI held-out task for mode={mode}"
            expected_w = 1.0 / len(nw)
            if energy == "nc":
                if mode == "equal":
                    assert all(abs(w - expected_w) < 1e-6 for w in nw), f"Expected equal weights, got {nw}"
                elif mode == "calibrated":
                    assert not all(abs(w - expected_w) < 1e-6 for w in nw), f"Expected calibrated weights, got {nw}"
            elif energy == "acc":
                if mode == "candidate":
                    assert simlr_module._view_energy_scale is not _BASE_VIEW_ENERGY_SCALE, \
                        "Candidate patch was not applied in candidate mode"
                elif mode == "calibrated":
                    assert simlr_module._view_energy_scale is _BASE_VIEW_ENERGY_SCALE, \
                        "Baseline scale was not restored in calibrated mode"

            if mode == alt_name:
                alt_conv = float(bool(row.get("Solver_Converged", True)))
                alt_steps = float(row.get("Solver_Iters", pysimlr.defaults.DEFAULT_SIMLR_ITERATIONS))

            results[mode] = {
                "adni_predictive_metric": float(row.get("Predictive_Metric", float("nan"))),
                "normalizing_weights": nw,
            }
        finally:
            set_weighting_mode(energy, "calibrated")

    if energy == "acc":
        assert not np.allclose(results["calibrated"]["normalizing_weights"],
                               results[alt_name]["normalizing_weights"], rtol=1e-4, atol=1e-4), \
            f"Candidate weights did not differ from calibrated weights in ADNI held-out seed {seed} pos {pos}"
        del results["calibrated"]["normalizing_weights"]
        del results[alt_name]["normalizing_weights"]

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
            "converged": alt_conv,
            "n_steps": alt_steps,
            "source": "measured",
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
            if energy == "acc":
                if mode == "candidate":
                    assert simlr_module._view_energy_scale is not _BASE_VIEW_ENERGY_SCALE, \
                        "Candidate patch was not applied in candidate mode"
                elif mode == "calibrated":
                    assert simlr_module._view_energy_scale is _BASE_VIEW_ENERGY_SCALE, \
                        "Baseline scale was not restored in calibrated mode"
            sr = pysimlr.select_rank("simlr", case["data"], [2, 3], n_subsamples=20,
                                     random_state=42, positivity=pos, topology="graph",
                                     path_graph=case["path_graph"], energy_type=energy)
            table = {r["k"]: r for r in sr["table"]}
            results[mode] = {
                2: float(table[2]["min_stability"]),
                3: float(table[3]["min_stability"]),
            }
        finally:
            set_weighting_mode(energy, "calibrated")

    records = []
    for k_val in (2, 3):
        c_val = results["calibrated"][k_val]
        a_val = results[alt_name][k_val]
        records.append({
            "group": "adni_stability",
            "case_id": f"adni_stab_{pos}_k{k_val}",
            "generator": "adni_mci",
            "positivity": pos,
            "seed": 42,
            "k": k_val,
            "metric": "min_component_stability",
            "calibrated": c_val,
            "alternative": a_val,
            "difference": a_val - c_val,
            "converged": np.nan,
            "n_steps": np.nan,
            "source": "measured",
        })
    return records


def get_adni_stability_rows(energy: str) -> list:
    """Retrieve ADNI-MCI select_rank component stability records.

    Per Section 5 WP-1R step 3:
    'Reuse, don't re-run, (e)-stability (the adni_stability rows: select_rank at k = 2 and 3).
     They already ran at the default budget, and both arms differed, so the patch took effect.
     Copy them with source=acd08fe.'
    """
    if energy == "nc":
        # Verified valid rows measured under acd08fe at default budget
        return [
            {
                "group": "adni_stability",
                "case_id": "adni_stab_positive_k2",
                "generator": "adni_mci",
                "positivity": "positive",
                "seed": 42,
                "k": 2,
                "metric": "min_component_stability",
                "calibrated": 0.9651146278635524,
                "alternative": 0.9609386872452806,
                "difference": -0.004175940618271756,
                "converged": np.nan,
                "n_steps": np.nan,
                "source": "acd08fe",
            },
            {
                "group": "adni_stability",
                "case_id": "adni_stab_positive_k3",
                "generator": "adni_mci",
                "positivity": "positive",
                "seed": 42,
                "k": 3,
                "metric": "min_component_stability",
                "calibrated": 0.7912613063164405,
                "alternative": 0.8030343486680626,
                "difference": 0.01177304235162202,
                "converged": np.nan,
                "n_steps": np.nan,
                "source": "acd08fe",
            },
            {
                "group": "adni_stability",
                "case_id": "adni_stab_either_k2",
                "generator": "adni_mci",
                "positivity": "either",
                "seed": 42,
                "k": 2,
                "metric": "min_component_stability",
                "calibrated": 0.770528404660616,
                "alternative": 0.7515670033495488,
                "difference": -0.01896140131106716,
                "converged": np.nan,
                "n_steps": np.nan,
                "source": "acd08fe",
            },
            {
                "group": "adni_stability",
                "case_id": "adni_stab_either_k3",
                "generator": "adni_mci",
                "positivity": "either",
                "seed": 42,
                "k": 3,
                "metric": "min_component_stability",
                "calibrated": 0.4172311888307556,
                "alternative": 0.5680600131542479,
                "difference": 0.15082882432349232,
                "converged": np.nan,
                "n_steps": np.nan,
                "source": "acd08fe",
            },
        ]
    else:
        # For non-nc energies, run fresh
        adni_stab_tasks = [(pos, energy) for pos in ("either", "positive")]
        recs = []
        for t in adni_stab_tasks:
            recs.extend(run_adni_stability_task(t))
        return recs


def evaluate_decision_rules(df: pd.DataFrame, energy: str) -> tuple:
    """Evaluate Decision Rule R1 (for 'nc') or Rule R9 (for 'acc') per Section 3 specifications.

    Section 3 Definitions:
      - Paired difference: alternative minus baseline, oriented so positive = better.
      - Worse beyond noise:
        * (mean test): n >= 2 and mean oriented paired difference < -SD of paired differences (ddof=1).
          Seed SD is NOT used.
        * (per-cell guard): any single cell is worse by more than the metric's guard:
            latent R^2, outcome R^2, held-out metric: 0.05
            support F1: 0.10
            span instability / sin^2 init-invariance: 0.05
            component stability: 0.10
            convergence rate (aggregate): 0.10
        * When n = 1, only the guard applies.
      - Non-inferior: not worse beyond noise on every listed metric.
    """
    alt_label = "equal" if energy == "nc" else "candidate"
    print("\n" + "=" * 80)
    print(f"EVALUATING DECISION RULE FOR ENERGY '{energy}' (Calibrated vs. {alt_label})")
    print("=" * 80)

    # Core representation metrics evaluated under R1 / R9
    core_metrics = {
        "latent_r2", "loadings_support_f1", "loadings_instability", "converged",
        "u_init_invariance", "structure_init_invariance",
        "outcome_r2_cw1", "support_recovery_loadings", "support_recovery_v",
        "adni_predictive_metric", "min_component_stability",
        "min_component_stability_k2", "min_component_stability_k3",
    }
    higher_is_better = {
        "latent_r2", "loadings_support_f1", "outcome_r2_cw1",
        "support_recovery_loadings", "support_recovery_v",
        "adni_predictive_metric", "min_component_stability",
        "min_component_stability_k2", "min_component_stability_k3",
        "converged",
    }
    lower_is_better = {
        "loadings_instability", "u_init_invariance", "structure_init_invariance",
    }
    diagnostic_metrics = {
        "n_steps", "weight_spread", "v_instability", "v_support_f1",
        "outcome_r2_cw0", "score_corr_cw0", "score_corr_cw1",
        "outcome_r2_cw1_it30", "score_corr_cw1_it30", "outcome_r2_cw0_it30", "score_corr_cw0_it30",
    }

    # Section 3 Per-cell guard thresholds
    guard_thresholds = {
        "latent_r2": 0.05,
        "outcome_r2_cw1": 0.05,
        "adni_predictive_metric": 0.05,
        "loadings_support_f1": 0.10,
        "support_recovery_loadings": 0.10,
        "support_recovery_v": 0.10,
        "loadings_instability": 0.05,
        "u_init_invariance": 0.05,
        "structure_init_invariance": 0.05,
        "min_component_stability": 0.10,
        "min_component_stability_k2": 0.10,
        "min_component_stability_k3": 0.10,
        "converged": 0.10,  # Aggregate guard
    }

    summary_rows = []
    all_non_inferior = True
    guard_failures = []

    metrics = df["metric"].unique()
    for metric in sorted(metrics):
        sub = df[df["metric"] == metric]
        raw_diffs = sub["difference"].values
        n = len(raw_diffs)
        mean_raw_diff = float(np.mean(raw_diffs))
        sd_diff = float(np.std(raw_diffs, ddof=1)) if n > 1 else 0.0

        if metric in diagnostic_metrics:
            direction = "diagnostic"
            status = "DIAGNOSTIC"
            mean_test_status = "N/A"
            guard_status = "N/A"
            summary_rows.append({
                "metric": metric,
                "direction": direction,
                "n_pairs": n,
                "calibrated_mean": float(sub["calibrated"].mean()),
                "alt_mean": float(sub["alternative"].mean()),
                "mean_diff": mean_raw_diff,
                "sd_diff": sd_diff,
                "mean_test": mean_test_status,
                "guard": guard_status,
                "status": status,
            })
            continue

        # Orient differences: positive = better
        if metric in higher_is_better:
            direction = "higher"
            oriented_diffs = raw_diffs
        elif metric in lower_is_better:
            direction = "lower"
            oriented_diffs = -raw_diffs
        else:
            direction = "neutral"
            oriented_diffs = raw_diffs

        mean_oriented = float(np.mean(oriented_diffs))
        sd_oriented = float(np.std(oriented_diffs, ddof=1)) if n > 1 else 0.0

        # Mean test: n >= 2 and mean oriented paired difference < -SD of paired differences (ddof=1)
        mean_test_failed = False
        if n >= 2 and sd_oriented > 1e-12:
            if mean_oriented < -sd_oriented:
                mean_test_failed = True
                mean_test_status = f"FAIL (mean={mean_oriented:+.4f} < -SD={-sd_oriented:.4f})"
            else:
                mean_test_status = f"PASS (mean={mean_oriented:+.4f} >= -SD={-sd_oriented:.4f})"
        else:
            mean_test_status = "N/A (n=1 or zero SD)"

        # Per-cell guard
        guard_limit = guard_thresholds.get(metric, None)
        per_cell_failed = False
        worst_loss = 0.0
        worst_cell = "N/A"
        if guard_limit is not None:
            if metric == "converged":
                # Aggregate guard for convergence rate: mean loss must not exceed guard
                worst_loss = -mean_oriented
                if mean_oriented < -guard_limit:
                    per_cell_failed = True
                    guard_status = f"FAIL (mean loss {worst_loss:.4f} > {guard_limit})"
                    guard_failures.append(f"{metric}: mean loss {worst_loss:.4f} > {guard_limit}")
                else:
                    guard_status = f"PASS (loss {worst_loss:.4f} <= {guard_limit})"
            else:
                # Per-cell check: worst single cell oriented difference < -guard_limit
                min_oriented_idx = np.argmin(oriented_diffs)
                worst_oriented = oriented_diffs[min_oriented_idx]
                worst_loss = -worst_oriented
                worst_cell = str(sub["case_id"].iloc[min_oriented_idx])
                if worst_oriented < -guard_limit:
                    per_cell_failed = True
                    guard_status = f"FAIL (cell {worst_cell} lost {worst_loss:.4f} > {guard_limit})"
                    guard_failures.append(f"{metric} cell {worst_cell}: lost {worst_loss:.4f} > {guard_limit}")
                else:
                    guard_status = f"PASS (worst loss {worst_loss:.4f} <= {guard_limit})"
        else:
            guard_status = "N/A"

        # Worse beyond noise = mean test failed OR per-cell guard failed
        worse_beyond_noise = mean_test_failed or per_cell_failed
        if worse_beyond_noise and metric in core_metrics:
            all_non_inferior = False
            status = "FAIL (Worse beyond noise)"
        else:
            status = "PASS (Non-inferior)"

        summary_rows.append({
            "metric": metric,
            "direction": direction,
            "n_pairs": n,
            "calibrated_mean": float(sub["calibrated"].mean()),
            "alt_mean": float(sub["alternative"].mean()),
            "mean_diff": mean_raw_diff,
            "sd_diff": sd_diff,
            "mean_test": mean_test_status,
            "guard": guard_status,
            "status": status,
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("-" * 80)

    if guard_failures:
        print("PER-CELL GUARD FAILURES:")
        for gf in guard_failures:
            print(f"  [X] {gf}")
    else:
        print("PER-CELL GUARDS: ALL PASSED (no cell breached metric guards)")
    print("-" * 80)

    passed_rule = all_non_inferior
    if energy == "nc":
        print(f"DECISION RULE R1 OUTCOME: {'ADOPT EQUAL 1/M WEIGHTS' if passed_rule else 'RETAIN CALIBRATED WEIGHTS'}")
        print(f"Summary: Non-inferior across all core representation metrics = {all_non_inferior}")
    elif energy == "acc":
        print(f"DECISION RULE R9 OUTCOME: {'ADOPT CANDIDATE DATA-RELATIVE SCALE' if passed_rule else 'RETAIN CALIBRATED WEIGHTS'}")
        print(f"Summary: Candidate scale non-inferior = {all_non_inferior}")
    print("=" * 80 + "\n")
    return passed_rule, summary_df


def main():
    parser = argparse.ArgumentParser(description="View weighting selection experiment.")
    parser.add_argument("--energy", choices=["nc", "acc"], default="nc",
                        help="Energy type to benchmark (default: nc).")
    parser.add_argument("--out", default=None,
                        help="Output path for the generated evidence CSV.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker count for ProcessPoolExecutor.")
    parser.add_argument("--groups", default=None,
                        help="Comma-separated groups to run (default: run all). Options: default_cells,narrow_fixture,nonneg_parts,adni_heldout,adni_stability.")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(REPO, "docs", "audit", "evidence",
                                f"view_weighting_{'selection' if args.energy == 'nc' else 'acc'}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.perf_counter()
    print(f"Starting view weighting selection experiment for energy='{args.energy}'...")
    print(f"Destination CSV: {args.out}")
    print(f"Parallel workers: {args.workers}")

    # Determine groups to run
    all_group_names = {"default_cells", "narrow_fixture", "nonneg_parts", "adni_heldout", "adni_stability"}
    if args.groups:
        target_groups = set(g.strip() for g in args.groups.split(","))
        invalid_groups = target_groups - all_group_names
        if invalid_groups:
            raise ValueError(f"Unknown group(s): {invalid_groups}. Allowed: {all_group_names}")
    else:
        target_groups = all_group_names

    print(f"Selected groups to run: {sorted(target_groups)}")

    new_records = []

    # 1. Default-selection cells (a) & init-invariance (b): 20 cells
    if "default_cells" in target_groups:
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
                new_records.extend(f.result())
        print("Default cells complete.")

    # 2. Narrow-view collapse fixture (c): single fixture
    if "narrow_fixture" in target_groups:
        print("\n[2/5] Running narrow-view collapse fixture (default budget & iterations=30)...")
        new_records.extend(run_narrow_fixture(args.energy))
        print("Narrow fixture complete.")

    # 3. NonnegParts support recovery (d): seeds 42-44
    if "nonneg_parts" in target_groups:
        nonneg_tasks = [(s, args.energy) for s in (42, 43, 44)]
        print(f"\n[3/5] Running {len(nonneg_tasks)} NonnegParts tasks at default budget...")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(nonneg_tasks))) as executor:
            futures = [executor.submit(run_nonneg_task, t) for t in nonneg_tasks]
            for f in as_completed(futures):
                new_records.extend(f.result())
        print("NonnegParts tasks complete.")

    # 4. ADNI-MCI held-out (e): 3 seeds x 2 pos = 6 tasks
    if "adni_heldout" in target_groups:
        adni_heldout_tasks = [
            (s, pos, args.energy)
            for s in (42, 43, 44)
            for pos in ("either", "positive")
        ]
        print(f"\n[4/5] Running {len(adni_heldout_tasks)} ADNI-MCI held-out tasks at default budget...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_adni_heldout_task, t) for t in adni_heldout_tasks]
            for f in as_completed(futures):
                new_records.extend(f.result())
        print("ADNI-MCI held-out complete.")

    # 5. ADNI-MCI select_rank stability (e):
    if "adni_stability" in target_groups:
        print("\n[5/5] Retrieving ADNI-MCI select_rank stability records...")
        new_records.extend(get_adni_stability_rows(args.energy))
        print("ADNI-MCI select_rank records added.")

    new_df = pd.DataFrame(new_records)

    # Merge with existing CSV if selective running
    if args.groups and os.path.exists(args.out):
        print(f"Merging {len(new_df)} new rows into existing {args.out}...")
        existing_df = pd.read_csv(args.out)
        # Drop rows belonging to the updated groups (or init_invariance if default_cells updated)
        drop_groups = set(target_groups)
        if "default_cells" in drop_groups:
            drop_groups.add("init_invariance")
        retained_df = existing_df[~existing_df["group"].isin(drop_groups)]
        final_df = pd.concat([retained_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    cols = ["group", "case_id", "generator", "positivity", "seed", "k", "metric",
            "calibrated", "alternative", "difference", "converged", "n_steps", "source"]
    final_df = final_df[cols]
    elapsed = time.perf_counter() - t0
    try:
        final_df.to_csv(args.out, index=False)
        print(f"\nWrote {len(final_df)} paired measurements to {args.out} in {elapsed:.1f} s.")
    except PermissionError:
        tmp_out = f"/tmp/{os.path.basename(args.out)}"
        final_df.to_csv(tmp_out, index=False)
        print(f"\nWrote {len(final_df)} paired measurements to {tmp_out} (PermissionError writing to {args.out}) in {elapsed:.1f} s.")

    passed, summary_df = evaluate_decision_rules(final_df, args.energy)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
