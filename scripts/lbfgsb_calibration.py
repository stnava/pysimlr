#!/usr/bin/env python3
"""WP-7 (D5): NSALBFGSB Proximal-Quartic Calibration and Decision Rule R5 Evaluation.

Grid definition:
  - quad_coeff in {1.0, 2.5, 5.0}
  - quartic_coeff in {16.0, 64.0, 256.0}
  Total: 9 combinations of (quad_coeff, quartic_coeff).

Evaluation cells (12 cells):
  - {buried, linear} x {either, positive} x seeds 0..2
  - Budget: Default budget (DEFAULT_SIMLR_ITERATIONS = 300, DEFAULT_TOL = 1e-4)
  Total fits: 9 x 12 = 108 fits.

Baseline:
  - Default simlr (hybrid_adam) under equal-arm view weighting.
  - Reused from docs/audit/evidence/view_weighting_selection.csv (or recomputed on demand).

Measurements per fit:
  1. latent_r2: least-squares linear subspace R^2 of true latent z onto fitted u.
  2. loadings_support_f1: Hungarian/permutation-matched maximum F1 of loadings against true support.
  3. seconds: wall-clock runtime for the fit.
  4. perturbation_gap: D9 reproducibility metric (max subspace gap under 1e-8 input perturbation).

Outputs:
  - CSV:  docs/audit/evidence/lbfgsb_calibration.csv (108 rows)
  - HTML: optional via --html (default "" writes no HTML)

Decision Rule R5:
  - If no grid cell is non-inferior to hybrid_adam on BOTH latent R^2 and loadings support F1:
      Drop simlr_lbfgsb:
      * Remove MethodSpec from api.py, with tests, docs, and scripts;
      * Remove nsa_lbfgsb optimizer unless other code uses it (grep);
      * Add "simlr_lbfgsb" to removed names raising ValueError pointing to simlr;
      * Ensure grep -rn "simlr_lbfgsb" returns only removal entry and tracking history.
  - Otherwise:
      Set constants to the non-inferior cell with best mean R^2, breaking ties by seconds.
"""
import argparse
import itertools
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from default_selection_benchmark import _case
from optimizer_benchmark_synthetic import latent_r2, support_f1
import pysimlr
import pysimlr.optimizers

# Grid definition
QUAD_COEFFS = [1.0, 2.5, 5.0]
QUARTIC_COEFFS = [16.0, 64.0, 256.0]
GENERATORS = ["buried", "linear"]
POSITIVITIES = ["either", "positive"]
SEEDS = [0, 1, 2]

# Baseline constants in codebase before calibration
BASE_QUAD_COEFF = 2.5
BASE_QUARTIC_COEFF = 64.0

# Non-inferiority guards (from /tmp/amigo_handoff_v2.md Section 3)
GUARD_LATENT_R2 = 0.05
GUARD_LOADINGS_F1 = 0.10
GATE_PERTURBATION_GAP = 1e-4


def load_or_compute_baseline(
    baseline_csv: str = os.path.join(REPO, "docs/audit/evidence/view_weighting_selection.csv"),
    recompute: bool = False,
) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """Load equal-arm baseline hybrid_adam measurements for the 12 cells."""
    if not recompute and os.path.exists(baseline_csv):
        df = pd.read_csv(baseline_csv)
        cells = df[
            (df["group"] == "default_cells")
            & (df["generator"].isin(GENERATORS))
            & (df["positivity"].isin(POSITIVITIES))
            & (df["seed"].isin(SEEDS))
        ]
        piv = cells.pivot_table(
            index=["generator", "positivity", "seed"],
            columns="metric",
            values="alternative",
        )
        baseline = {}
        for (gen, pos, seed), row in piv.iterrows():
            baseline[(gen, pos, seed)] = {
                "latent_r2": float(row["latent_r2"]),
                "loadings_support_f1": float(row["loadings_support_f1"]),
                "converged": float(row.get("converged", 1.0)),
                "n_steps": float(row.get("n_steps", 0.0)),
            }
        if len(baseline) == 12:
            return baseline

    # Self-contained computation if CSV missing or recompute requested
    print("Computing baseline hybrid_adam fits for 12 cells...")
    baseline = {}
    for gen in GENERATORS:
        for pos in POSITIVITIES:
            for seed in SEEDS:
                z, views, sups = _case(gen, seed)
                k = z.shape[1]
                res = pysimlr.fit("simlr", views, k=k, positivity=pos, random_state=seed)
                lr2 = float(latent_r2(z, res["u"]))
                lf1 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(res["loadings"], sups)]))
                baseline[(gen, pos, seed)] = {
                    "latent_r2": lr2,
                    "loadings_support_f1": lf1,
                    "converged": float(res["convergence"]["converged"]),
                    "n_steps": float(res["convergence"]["n_steps"]),
                }
    return baseline


def _subspace_projection_gap(v_base: List[Any], v_pert: List[Any]) -> float:
    """Compute maximum subspace projector distance ||Qa Qa^T - Qb Qb^T||_2 over views (D9 metric)."""
    view_gaps = []
    for va, vb in zip(v_base, v_pert):
        qa, _ = torch.linalg.qr(torch.as_tensor(va).double())
        qb, _ = torch.linalg.qr(torch.as_tensor(vb).double())
        gap = float(torch.linalg.matrix_norm(qa @ qa.T - qb @ qb.T, ord=2))
        view_gaps.append(gap)
    return float(max(view_gaps)) if view_gaps else float("nan")


def _run_single_calibration_task(task_args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one calibration fit task in worker process with Rule 8 monkeypatching."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)

    c2 = float(task_args["quad_coeff"])
    c4 = float(task_args["quartic_coeff"])
    gen = str(task_args["generator"])
    pos = str(task_args["positivity"])
    seed = int(task_args["seed"])
    base_lr2 = float(task_args["baseline_latent_r2"])
    base_lf1 = float(task_args["baseline_loadings_support_f1"])

    # Rule 8 Step 1: ALWAYS restore baseline state first
    pysimlr.optimizers.NSALBFGSB._PROX_QUAD_COEFF = BASE_QUAD_COEFF
    pysimlr.optimizers.NSALBFGSB._PROX_QUARTIC_COEFF = BASE_QUARTIC_COEFF
    setattr(pysimlr.optimizers, "_PROX_QUAD_COEFF", BASE_QUAD_COEFF)
    setattr(pysimlr.optimizers, "_PROX_QUARTIC_COEFF", BASE_QUARTIC_COEFF)

    # Rule 8 Step 2: Apply patch to the exact attributes read at call time
    pysimlr.optimizers.NSALBFGSB._PROX_QUAD_COEFF = c2
    pysimlr.optimizers.NSALBFGSB._PROX_QUARTIC_COEFF = c4
    setattr(pysimlr.optimizers, "_PROX_QUAD_COEFF", c2)
    setattr(pysimlr.optimizers, "_PROX_QUARTIC_COEFF", c4)

    # Rule 8 Step 3: Assert patch took effect in the worker process
    assert pysimlr.optimizers.NSALBFGSB._PROX_QUAD_COEFF == c2, "Quad coeff patch failed"
    assert pysimlr.optimizers.NSALBFGSB._PROX_QUARTIC_COEFF == c4, "Quartic coeff patch failed"

    cell_id = f"{gen}_{pos}_s{seed}"
    t0 = time.perf_counter()
    try:
        z, views, sups = _case(gen, seed)
        k = z.shape[1]

        # 1. Unperturbed primary fit (default budget: iterations=None -> 300 / DEFAULT_TOL)
        res = pysimlr.fit("simlr_lbfgsb", views, k=k, positivity=pos, random_state=seed)
        t1 = time.perf_counter()
        fit_seconds = t1 - t0

        lr2 = float(latent_r2(z, res["u"]))
        lf1 = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(res["loadings"], sups)]))
        conv = bool(res["convergence"]["converged"])
        steps = int(res["convergence"]["n_steps"])

        # 2. Perturbation gap evaluation (D9 metric: 1e-8 float32 perturbation)
        g = torch.Generator().manual_seed(100 + seed)
        pert_views = [v + 1e-8 * torch.randn(v.shape, generator=g, dtype=v.dtype) for v in views]
        res_pert = pysimlr.fit("simlr_lbfgsb", pert_views, k=k, positivity=pos, random_state=seed)
        pert_gap = _subspace_projection_gap(res["v"], res_pert["v"])

        return {
            "quad_coeff": c2,
            "quartic_coeff": c4,
            "cell_id": cell_id,
            "generator": gen,
            "positivity": pos,
            "seed": seed,
            "k": k,
            "latent_r2": round(lr2, 6),
            "baseline_latent_r2": round(base_lr2, 6),
            "diff_latent_r2": round(lr2 - base_lr2, 6),
            "loadings_support_f1": round(lf1, 6),
            "baseline_loadings_support_f1": round(base_lf1, 6),
            "diff_loadings_support_f1": round(lf1 - base_lf1, 6),
            "seconds": round(fit_seconds, 4),
            "perturbation_gap": float(f"{pert_gap:.3e}"),
            "converged": conv,
            "n_steps": steps,
            "source": "measured",
            "notes": "",
        }
    except Exception as exc:
        t1 = time.perf_counter()
        return {
            "quad_coeff": c2,
            "quartic_coeff": c4,
            "cell_id": cell_id,
            "generator": gen,
            "positivity": pos,
            "seed": seed,
            "k": 0,
            "latent_r2": float("nan"),
            "baseline_latent_r2": round(base_lr2, 6),
            "diff_latent_r2": float("nan"),
            "loadings_support_f1": float("nan"),
            "baseline_loadings_support_f1": round(base_lf1, 6),
            "diff_loadings_support_f1": float("nan"),
            "seconds": round(t1 - t0, 4),
            "perturbation_gap": float("nan"),
            "converged": False,
            "n_steps": 0,
            "source": "measured",
            "notes": f"ERROR: {type(exc).__name__}: {str(exc)[:160]}",
        }
    finally:
        # Rule 8 cleanup
        pysimlr.optimizers.NSALBFGSB._PROX_QUAD_COEFF = BASE_QUAD_COEFF
        pysimlr.optimizers.NSALBFGSB._PROX_QUARTIC_COEFF = BASE_QUARTIC_COEFF


def evaluate_rule_r5(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate Decision Rule R5 non-inferiority criteria across the 9 grid points."""
    summary_rows = []
    best_candidate = None
    best_candidate_r2 = -float("inf")
    best_candidate_sec = float("inf")

    for (c2, c4), group in df.groupby(["quad_coeff", "quartic_coeff"]):
        n_cells = len(group)
        # Latent R2 metrics
        d_r2 = group["diff_latent_r2"].values
        mean_d_r2 = float(np.mean(d_r2))
        sd_d_r2 = float(np.std(d_r2, ddof=1)) if n_cells > 1 else 0.0
        min_d_r2 = float(np.min(d_r2))
        r2_mean_pass = bool(mean_d_r2 >= -sd_d_r2)
        r2_guard_pass = bool(min_d_r2 >= -GUARD_LATENT_R2)
        r2_non_inferior = bool(r2_mean_pass and r2_guard_pass)

        # Loadings Support F1 metrics
        d_f1 = group["diff_loadings_support_f1"].values
        mean_d_f1 = float(np.mean(d_f1))
        sd_d_f1 = float(np.std(d_f1, ddof=1)) if n_cells > 1 else 0.0
        min_d_f1 = float(np.min(d_f1))
        f1_mean_pass = bool(mean_d_f1 >= -sd_d_f1)
        f1_guard_pass = bool(min_d_f1 >= -GUARD_LOADINGS_F1)
        f1_non_inferior = bool(f1_mean_pass and f1_guard_pass)

        # Perturbation gap gate
        max_pert_gap = float(group["perturbation_gap"].max())
        gap_pass = bool(max_pert_gap < GATE_PERTURBATION_GAP)

        # Overall non-inferiority
        is_non_inferior = bool(r2_non_inferior and f1_non_inferior and gap_pass)

        mean_r2 = float(group["latent_r2"].mean())
        mean_sec = float(group["seconds"].mean())

        summary_rows.append({
            "quad_coeff": c2,
            "quartic_coeff": c4,
            "mean_r2": round(mean_r2, 4),
            "mean_d_r2": round(mean_d_r2, 4),
            "sd_d_r2": round(sd_d_r2, 4),
            "min_d_r2": round(min_d_r2, 4),
            "r2_mean_pass": r2_mean_pass,
            "r2_guard_pass": r2_guard_pass,
            "r2_non_inferior": r2_non_inferior,
            "mean_d_f1": round(mean_d_f1, 4),
            "sd_d_f1": round(sd_d_f1, 4),
            "min_d_f1": round(min_d_f1, 4),
            "f1_mean_pass": f1_mean_pass,
            "f1_guard_pass": f1_guard_pass,
            "f1_non_inferior": f1_non_inferior,
            "max_pert_gap": max_pert_gap,
            "gap_pass": gap_pass,
            "mean_seconds": round(mean_sec, 2),
            "non_inferior": is_non_inferior,
        })

        if is_non_inferior:
            if (mean_r2 > best_candidate_r2) or (abs(mean_r2 - best_candidate_r2) < 1e-5 and mean_sec < best_candidate_sec):
                best_candidate = (c2, c4)
                best_candidate_r2 = mean_r2
                best_candidate_sec = mean_sec

    summary_df = pd.DataFrame(summary_rows)

    if best_candidate is not None:
        decision = {
            "branch": "CALIBRATE",
            "recommended_quad_coeff": best_candidate[0],
            "recommended_quartic_coeff": best_candidate[1],
            "mean_latent_r2": best_candidate_r2,
            "mean_seconds": best_candidate_sec,
            "action": (
                f"Set _PROX_QUAD_COEFF={best_candidate[0]} and "
                f"_PROX_QUARTIC_COEFF={best_candidate[1]} in src/pysimlr/optimizers.py."
            ),
            "commit_msg": "feat(optimizers): calibrate NSALBFGSB prox constants (R5)",
        }
    else:
        decision = {
            "branch": "DROP",
            "recommended_quad_coeff": None,
            "recommended_quartic_coeff": None,
            "action": (
                "Drop simlr_lbfgsb: remove MethodSpec from src/pysimlr/api.py, "
                "add to REMOVED_METHODS, remove nsa_lbfgsb from optimizers.py, "
                "migrate tests, and update docs."
            ),
            "commit_msg": "refactor(api,optimizers): drop simlr_lbfgsb (R5)",
        }

    return summary_df, decision


def main():
    parser = argparse.ArgumentParser(description="WP-7 NSALBFGSB Calibration Benchmark (Rule R5)")
    parser.add_argument("--workers", type=int, default=6, help="ProcessPoolExecutor max_workers (default: 6)")
    parser.add_argument("--out", type=str, default=os.path.join(REPO, "docs/audit/evidence/lbfgsb_calibration.csv"),
                        help="Output evidence CSV path")
    parser.add_argument("--recompute-baseline", action="store_true", help="Recompute baseline hybrid_adam fits")
    parser.add_argument("--html", type=str, default="", help="Optional HTML report output path (default empty, writes no HTML)")
    args = parser.parse_args()

    print(f"=== WP-7 NSALBFGSB Calibration Benchmark (Rule R5) ===")
    print(f"Workers: {args.workers}")
    print(f"Grid: 9 combinations of (quad_coeff, quartic_coeff) in {QUAD_COEFFS} x {QUARTIC_COEFFS}")
    print(f"Cells: 12 cells ({GENERATORS} x {POSITIVITIES} x seeds 0..2)")
    print(f"Total fits: 108 fits")

    # 1. Baseline loading / computation
    baseline = load_or_compute_baseline(recompute=args.recompute_baseline)
    print(f"Baseline loaded for {len(baseline)} cells.")

    # 2. Build task list
    tasks = []
    for c2 in QUAD_COEFFS:
        for c4 in QUARTIC_COEFFS:
            for gen in GENERATORS:
                for pos in POSITIVITIES:
                    for seed in SEEDS:
                        base_entry = baseline[(gen, pos, seed)]
                        tasks.append({
                            "quad_coeff": c2,
                            "quartic_coeff": c4,
                            "generator": gen,
                            "positivity": pos,
                            "seed": seed,
                            "baseline_latent_r2": base_entry["latent_r2"],
                            "baseline_loadings_support_f1": base_entry["loadings_support_f1"],
                        })

    print(f"Launching {len(tasks)} tasks on {args.workers} workers...")
    t_start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_single_calibration_task, t): t for t in tasks}
        completed_count = 0
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            completed_count += 1
            if completed_count % 12 == 0 or completed_count == len(tasks):
                elapsed = time.perf_counter() - t_start
                print(f"Progress: {completed_count}/{len(tasks)} fits completed ({elapsed:.1f}s elapsed)...")

    t_total = time.perf_counter() - t_start
    print(f"All {len(tasks)} fits completed in {t_total:.1f}s ({t_total/60:.2f}m).")

    df = pd.DataFrame(results)
    # Sort deterministically
    df.sort_values(by=["quad_coeff", "quartic_coeff", "generator", "positivity", "seed"], inplace=True)

    # Save evidence CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Evidence CSV written to: {args.out} ({len(df)} rows)")

    # 3. Rule R5 Non-inferiority Evaluation
    summary_df, decision = evaluate_rule_r5(df)
    print("\n=== Decision Rule R5 Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nDecision Branch: {decision['branch']}")
    print(f"Action: {decision['action']}")
    print(f"Recommended Commit Message: {decision['commit_msg']}")

    # 4. Optional HTML report
    if args.html:
        os.makedirs(os.path.dirname(os.path.abspath(args.html)), exist_ok=True)
        html_content = f"""<!DOCTYPE html>
<html>
<head><title>WP-7 NSALBFGSB Calibration Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
th {{ background-color: #f2f2f2; }}
.badge-pass {{ color: green; font-weight: bold; }}
.badge-fail {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
<h1>WP-7 NSALBFGSB Calibration Benchmark Report</h1>
<p>Decision Branch: <strong>{decision['branch']}</strong></p>
<p>Action: {decision['action']}</p>
<h2>Summary across Grid Combinations</h2>
{summary_df.to_html(classes="table", index=False)}
</body>
</html>
"""
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report written to: {args.html}")


if __name__ == "__main__":
    main()
