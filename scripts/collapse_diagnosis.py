#!/usr/bin/env python3
"""WP-2 (D2): Linear narrow-view collapse diagnosis on corrected objective.

Executes Step 0, Steps 1-4 (planted solution and energy evaluation),
Step 5 (20 default cells), and applies Decision Rule R2.

Writes evidence to docs/audit/evidence/collapse_diagnosis.csv.
"""
import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import pysimlr
import pysimlr.defaults
from default_selection_benchmark import _case
from optimizer_benchmark_synthetic import latent_r2, support_f1
from pysimlr.consensus import compute_shared_consensus
from pysimlr.simlr import _prepare_fit_inputs, calculate_simlr_energy, simlr
from pysimlr.sparsification import simlr_sparseness
from test_collapse_penalty import _narrow, _outcome_r2


def replay_narrow_fixture():
    """Replay _narrow's RNG exactly to extract planted W_m and data matrices."""
    torch.manual_seed(0)
    n, k = 300, 2
    z = torch.randn(n, k)
    w = torch.randn(k, 1)
    W_list, x = [], []
    for _ in range(2):
        W_m = torch.randn(k, 5)
        noise_m = torch.randn(n, 5)
        W_list.append(W_m)
        x.append(z @ W_m + 0.3 * noise_m)
    y = (z @ w).squeeze(1)
    return x, y, z, W_list


def step0_narrow_lr_sweeps(x, y):
    """Step 0: Fit narrow fixture at fixed learning rates and auto for cw in {0, 1}."""
    results = {}
    records = []
    lrs = [0.3, 0.1, 0.03, 0.01]

    for cw in [0.0, 1.0]:
        # Also fit auto to capture endpoint behavior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_auto = simlr(x, k=2, positivity="positive", sparseness=0.5,
                             energy_type="nc", optimizer_type="hybrid_adam",
                             mixing_algorithm="newton", random_state=0,
                             collapse_weight=cw, learning_rate="auto")
        scores_auto = x[0] @ res_auto["v"][0]
        corr_auto = float(torch.corrcoef(scores_auto.T)[0, 1])
        r2_auto = _outcome_r2(x, y, res_auto["v"])
        auto_chosen_lr = res_auto["learning_rate"]
        records.append({
            "group": "narrow_step0",
            "case_id": f"narrow_cw{int(cw)}_auto",
            "cw": cw,
            "learning_rate": "auto",
            "chosen_lr": auto_chosen_lr,
            "stop_reason": res_auto["stop_reason"],
            "converged": bool(res_auto["converged"]),
            "n_steps": int(res_auto["converged_iter"]),
            "best_energy": float(res_auto["best_energy"]),
            "outcome_r2": float(r2_auto),
            "score_corr": float(corr_auto),
        })

        best_converged_fit = None
        best_converged_lr = None
        lowest_energy_fit = None
        lowest_energy = float("inf")

        for lr in lrs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = simlr(x, k=2, positivity="positive", sparseness=0.5,
                            energy_type="nc", optimizer_type="hybrid_adam",
                            mixing_algorithm="newton", random_state=0,
                            collapse_weight=cw, learning_rate=lr)

            scores = x[0] @ res["v"][0]
            corr = float(torch.corrcoef(scores.T)[0, 1])
            r2 = _outcome_r2(x, y, res["v"])
            conv = bool(res["converged"])
            energy = float(res["best_energy"])

            records.append({
                "group": "narrow_step0",
                "case_id": f"narrow_cw{int(cw)}_lr{lr}",
                "cw": cw,
                "learning_rate": lr,
                "chosen_lr": lr,
                "stop_reason": res["stop_reason"],
                "converged": conv,
                "n_steps": int(res["converged_iter"]),
                "best_energy": energy,
                "outcome_r2": float(r2),
                "score_corr": float(corr),
            })

            if energy < lowest_energy:
                lowest_energy = energy
                lowest_energy_fit = (lr, res, r2, corr)

            if conv and best_converged_fit is None:
                # lrs is sorted descending: [0.3, 0.1, 0.03, 0.01], so first converged is largest
                best_converged_fit = (lr, res, r2, corr)
                best_converged_lr = lr

        if best_converged_fit is not None:
            results[cw] = {
                "fit": best_converged_fit[1],
                "lr": best_converged_fit[0],
                "converged": True,
                "r2": best_converged_fit[2],
                "score_corr": best_converged_fit[3],
                "auto_chosen_lr": auto_chosen_lr,
            }
        else:
            results[cw] = {
                "fit": lowest_energy_fit[1],
                "lr": lowest_energy_fit[0],
                "converged": False,
                "r2": lowest_energy_fit[2],
                "score_corr": lowest_energy_fit[3],
                "auto_chosen_lr": auto_chosen_lr,
            }

    return results, records


def evaluate_pure_similarity(torch_mats, v_candidate):
    """Evaluate sum over views of calculate_simlr_energy(v, x, u_m, 'nc') / M with loo newton u."""
    projections = [m @ b for m, b in zip(torch_mats, v_candidate)]
    u_loo = compute_shared_consensus(projections, mixing_algorithm="newton", k=2, topology="loo")
    M = len(torch_mats)
    return sum(calculate_simlr_energy(v_candidate[i], torch_mats[i], u_loo[i], "nc").item() / M for i in range(M))


def steps1_4_energy_evaluation(x, y, z, W_list, step0_results):
    """Steps 1-4: Planted solution, energy evaluation at V(cw=0), P(V_true), V(cw=1), and sanity check."""
    V_true = [torch.linalg.pinv(W_m) for W_m in W_list]

    torch_mats, provenance_list, _ = _prepare_fit_inputs(
        x, k=2, scale_list=["centerAndScale", "np"], positivity="positive",
        initialization_type="pca", random_state=0, domain_matrices=None,
        sparseness=0.5, varimax=None)

    # Map V_true into preprocessed space via provenance:
    # x_prep = (x - mean) / (std * np_factor) -> V_prep = diag(std * np_factor) @ V_true
    V_prep = [prov["cas_std"].unsqueeze(1) * prov["np_factor"] * v_t
              for prov, v_t in zip(provenance_list, V_true)]

    # Project with sparsification.simlr_sparseness at the fit's settings
    P_V_true = [simlr_sparseness(v, constraint_type="ortho", positivity="positive",
                                 sparseness=0.5, constraint_weight=0.5,
                                 energy_type="nc", modality_index=i)
                for i, v in enumerate(V_prep)]

    fit_cw0 = step0_results[0.0]["fit"]
    fit_cw1 = step0_results[1.0]["fit"]

    e_cw0 = evaluate_pure_similarity(torch_mats, fit_cw0["v"])
    e_planted = evaluate_pure_similarity(torch_mats, P_V_true)
    e_cw1 = evaluate_pure_similarity(torch_mats, fit_cw1["v"])

    # Sanity residual at V(cw=0)
    best_e_cw0 = float(fit_cw0["best_energy"])
    sanity_residual = abs(e_cw0 - best_e_cw0)

    # Also compute outcome R^2 and score corr for P(V_true)
    scores_planted = x[0] @ P_V_true[0]
    corr_planted = float(torch.corrcoef(scores_planted.T)[0, 1])
    r2_planted = _outcome_r2(x, y, P_V_true)

    records = [
        {
            "group": "planted_solution",
            "case_id": "V_cw0",
            "point": "V(cw=0)",
            "energy": e_cw0,
            "res_best_energy": best_e_cw0,
            "sanity_residual": sanity_residual,
            "outcome_r2": float(step0_results[0.0]["r2"]),
            "score_corr": float(step0_results[0.0]["score_corr"]),
            "lr_used": step0_results[0.0]["lr"],
            "converged": step0_results[0.0]["converged"],
        },
        {
            "group": "planted_solution",
            "case_id": "P_V_true",
            "point": "P(V_true)",
            "energy": e_planted,
            "res_best_energy": np.nan,
            "sanity_residual": np.nan,
            "outcome_r2": float(r2_planted),
            "score_corr": float(corr_planted),
            "lr_used": np.nan,
            "converged": np.nan,
        },
        {
            "group": "planted_solution",
            "case_id": "V_cw1",
            "point": "V(cw=1)",
            "energy": e_cw1,
            "res_best_energy": float(fit_cw1["best_energy"]),
            "sanity_residual": abs(e_cw1 - float(fit_cw1["best_energy"])),
            "outcome_r2": float(step0_results[1.0]["r2"]),
            "score_corr": float(step0_results[1.0]["score_corr"]),
            "lr_used": step0_results[1.0]["lr"],
            "converged": step0_results[1.0]["converged"],
        },
    ]

    return {
        "e_cw0": e_cw0,
        "e_planted": e_planted,
        "e_cw1": e_cw1,
        "sanity_residual": sanity_residual,
        "r2_cw0": step0_results[0.0]["r2"],
        "r2_cw1": step0_results[1.0]["r2"],
        "corr_cw0": step0_results[0.0]["score_corr"],
        "corr_cw1": step0_results[1.0]["score_corr"],
    }, records


def _run_single_cell(cell_args):
    """Worker task for running one cell with cw=0 and cw=1."""
    gen, pos, seed = cell_args
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)

    z, views, sups = _case(gen, seed)
    k = z.shape[1]

    # cw = 0
    res0 = pysimlr.fit("simlr", views, k=k, energy_type="nc", positivity=pos,
                       random_state=seed, collapse_weight=0.0)
    r2_0 = latent_r2(z, res0["u"])
    f1_0 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(res0["loadings"], sups)]))
    conv0 = float(res0["convergence"]["converged"])
    steps0 = float(res0["convergence"]["n_steps"])

    # cw = 1
    res1 = pysimlr.fit("simlr", views, k=k, energy_type="nc", positivity=pos,
                       random_state=seed, collapse_weight=1.0)
    r2_1 = latent_r2(z, res1["u"])
    f1_1 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(res1["loadings"], sups)]))
    conv1 = float(res1["convergence"]["converged"])
    steps1 = float(res1["convergence"]["n_steps"])

    return {
        "gen": gen,
        "pos": pos,
        "seed": seed,
        "case_id": f"{gen}_{pos}_s{seed}",
        "r2_cw0": r2_0,
        "r2_cw1": r2_1,
        "diff_r2": r2_0 - r2_1,
        "f1_cw0": f1_0,
        "f1_cw1": f1_1,
        "diff_f1": f1_0 - f1_1,
        "conv0": conv0,
        "conv1": conv1,
        "steps0": steps0,
        "steps1": steps1,
    }


def step5_default_cells(n_workers=4):
    """Step 5: Run 20 default cells with cw=0 against cw=1 at default budget."""
    generators = ["buried", "linear"]
    positivities = ["either", "positive"]
    seeds = range(5)
    tasks = [(g, p, s) for g in generators for p in positivities for s in seeds]

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            cell_results = list(pool.map(_run_single_cell, tasks))
    else:
        cell_results = [_run_single_cell(t) for t in tasks]

    records = []
    diff_r2 = []
    diff_f1 = []

    for cr in cell_results:
        diff_r2.append(cr["diff_r2"])
        diff_f1.append(cr["diff_f1"])

        records.append({
            "group": "default_cells",
            "case_id": cr["case_id"],
            "generator": cr["gen"],
            "positivity": cr["pos"],
            "seed": cr["seed"],
            "k": 2,
            "metric": "latent_r2",
            "cw1_baseline": cr["r2_cw1"],
            "cw0_alternative": cr["r2_cw0"],
            "difference": cr["diff_r2"],
            "converged": cr["conv0"],
            "n_steps": cr["steps0"],
        })
        records.append({
            "group": "default_cells",
            "case_id": cr["case_id"],
            "generator": cr["gen"],
            "positivity": cr["pos"],
            "seed": cr["seed"],
            "k": 2,
            "metric": "loadings_support_f1",
            "cw1_baseline": cr["f1_cw1"],
            "cw0_alternative": cr["f1_cw0"],
            "difference": cr["diff_f1"],
            "converged": cr["conv0"],
            "n_steps": cr["steps0"],
        })

    diff_r2 = np.array(diff_r2)
    diff_f1 = np.array(diff_f1)

    mean_r2, sd_r2 = float(np.mean(diff_r2)), float(np.std(diff_r2, ddof=1))
    worst_r2 = float(np.min(diff_r2))
    non_inf_r2 = (mean_r2 >= -sd_r2) and (worst_r2 >= -0.05)

    mean_f1, sd_f1 = float(np.mean(diff_f1)), float(np.std(diff_f1, ddof=1))
    worst_f1 = float(np.min(diff_f1))
    non_inf_f1 = (mean_f1 >= -sd_f1) and (worst_f1 >= -0.10)

    summary = {
        "mean_r2": mean_r2,
        "sd_r2": sd_r2,
        "worst_r2": worst_r2,
        "non_inf_r2": non_inf_r2,
        "mean_f1": mean_f1,
        "sd_f1": sd_f1,
        "worst_f1": worst_f1,
        "non_inf_f1": non_inf_f1,
        "non_inferior": non_inf_r2 and non_inf_f1,
    }

    return summary, records


def apply_decision_rule_r2(energy_diag, cell_summary):
    """Apply Decision Rule R2 to measured inputs."""
    # Condition 1: cw = 0 is non-inferior to cw = 1 on narrow outcome R^2 (guard 0.05)
    diff_narrow_r2 = energy_diag["r2_cw0"] - energy_diag["r2_cw1"]
    cond1 = diff_narrow_r2 >= -0.05

    # Condition 2: cw = 0 is non-inferior on cell latent R^2 and loadings F1
    cond2 = cell_summary["non_inferior"]

    # Condition 3: narrow |score corr| at cw = 0 is < 0.5
    abs_score_corr = abs(energy_diag["corr_cw0"])
    cond3 = abs_score_corr < 0.5

    all_hold = cond1 and cond2 and cond3
    outcome = "SET_DEFAULT_LINEAR_COLLAPSE_WEIGHT_0" if all_hold else "KEEP_DEFAULT_COLLAPSE_WEIGHT_1"

    summary_records = [
        {
            "group": "decision_rule_r2",
            "case_id": "r2_cond1_narrow_outcome_r2",
            "metric": "narrow_outcome_r2",
            "cw1_baseline": energy_diag["r2_cw1"],
            "cw0_alternative": energy_diag["r2_cw0"],
            "difference": diff_narrow_r2,
            "threshold_or_guard": -0.05,
            "passed": cond1,
        },
        {
            "group": "decision_rule_r2",
            "case_id": "r2_cond2_cell_latent_r2",
            "metric": "cell_latent_r2_worst",
            "cw1_baseline": np.nan,
            "cw0_alternative": np.nan,
            "difference": cell_summary["worst_r2"],
            "threshold_or_guard": -0.05,
            "passed": cell_summary["non_inf_r2"],
        },
        {
            "group": "decision_rule_r2",
            "case_id": "r2_cond2_cell_loadings_f1",
            "metric": "cell_loadings_f1_worst",
            "cw1_baseline": np.nan,
            "cw0_alternative": np.nan,
            "difference": cell_summary["worst_f1"],
            "threshold_or_guard": -0.10,
            "passed": cell_summary["non_inf_f1"],
        },
        {
            "group": "decision_rule_r2",
            "case_id": "r2_cond3_narrow_score_corr",
            "metric": "abs_narrow_score_corr",
            "cw1_baseline": abs(energy_diag["corr_cw1"]),
            "cw0_alternative": abs_score_corr,
            "difference": abs_score_corr,
            "threshold_or_guard": 0.5,
            "passed": cond3,
        },
        {
            "group": "decision_rule_r2",
            "case_id": "r2_overall_verdict",
            "metric": "decision_rule_r2_outcome",
            "cw1_baseline": np.nan,
            "cw0_alternative": np.nan,
            "difference": np.nan,
            "threshold_or_guard": np.nan,
            "passed": all_hold,
            "outcome": outcome,
        },
    ]

    return {
        "cond1": cond1,
        "diff_narrow_r2": diff_narrow_r2,
        "cond2": cond2,
        "cond3": cond3,
        "abs_score_corr": abs_score_corr,
        "all_hold": all_hold,
        "outcome": outcome,
    }, summary_records


def main():
    parser = argparse.ArgumentParser(description="WP-2 (D2) Linear collapse diagnosis")
    parser.add_argument("--out", default=os.path.join(REPO, "docs/audit/evidence/collapse_diagnosis.csv"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("=== WP-2 (D2) Collapse Diagnosis ===")
    t_start = time.time()

    # Step 0
    print("\n--- Step 0: Obtain Converged Narrow Fits ---")
    x, y, z, W_list = replay_narrow_fixture()
    step0_results, step0_records = step0_narrow_lr_sweeps(x, y)

    for cw in [0.0, 1.0]:
        res_info = step0_results[cw]
        print(f"cw={cw}: chosen_lr={res_info['lr']} (converged={res_info['converged']}), "
              f"r2={res_info['r2']:.4f}, score_corr={res_info['score_corr']:.4f}, "
              f"auto_chosen_lr={res_info['auto_chosen_lr']}")

    # Steps 1-4
    print("\n--- Steps 1-4: Planted Solution & Pure Similarity Objective ---")
    energy_diag, energy_records = steps1_4_energy_evaluation(x, y, z, W_list, step0_results)
    print(f"Energy at V(cw=0): {energy_diag['e_cw0']:.6f}")
    print(f"Energy at P(V_true): {energy_diag['e_planted']:.6f}")
    print(f"Energy at V(cw=1): {energy_diag['e_cw1']:.6f}")
    print(f"Sanity residual at V(cw=0): {energy_diag['sanity_residual']:.2e}")

    # Step 5
    print("\n--- Step 5: Measure 20 Default Cells ---")
    cell_summary, cell_records = step5_default_cells(n_workers=args.workers)
    print(f"Latent R2: mean={cell_summary['mean_r2']:.6f}, sd={cell_summary['sd_r2']:.6f}, "
          f"worst={cell_summary['worst_r2']:.6f}, non_inferior={cell_summary['non_inf_r2']}")
    print(f"Loadings F1: mean={cell_summary['mean_f1']:.6f}, sd={cell_summary['sd_f1']:.6f}, "
          f"worst={cell_summary['worst_f1']:.6f}, non_inferior={cell_summary['non_inf_f1']}")

    # Apply Decision Rule R2
    print("\n--- Decision Rule R2 ---")
    r2_verdict, r2_records = apply_decision_rule_r2(energy_diag, cell_summary)
    print(f"Condition 1 (Narrow R2 diff >= -0.05): {r2_verdict['cond1']} (diff = {r2_verdict['diff_narrow_r2']:.4f})")
    print(f"Condition 2 (Cell latent R2 & loadings F1 non-inferior): {r2_verdict['cond2']}")
    print(f"Condition 3 (Narrow |score corr| < 0.5): {r2_verdict['cond3']} (|corr| = {r2_verdict['abs_score_corr']:.4f})")
    print(f"Verdict: {r2_verdict['outcome']}")

    # Write CSV
    all_records = step0_records + energy_records + cell_records + r2_records
    df = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
