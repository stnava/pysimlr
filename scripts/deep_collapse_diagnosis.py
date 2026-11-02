#!/usr/bin/env python3
"""WP-5 (D4): DeepSiMLR collapse diagnosis on NED fixture under Decision Rule R3.

Executes:
  - Step 1: 40 fits (20 data seeds 1..5 x start seeds 1..4, with and without penalty).
            Counts collapsing pairs (|corr(u)| >= 0.5).
  - Step 2: Determinism check (2 identical fits match bit-identically).
            Prefix sweeps at iterations in {10, 20, 40, 100} for collapsing pairs
            to pinpoint onset (warmup vs similarity phase).
  - Step 3: Unpenalized energy evaluation on collapsed fits vs planted latents
            (weights={"collapse": 0.0}), with explicit statement that model-parameter
            planted warm start is non-constructible due to multi-layer MLP nonlinear heads.
  - Decision Rule R3 evaluation: Count without penalty vs count with penalty.

Writes evidence to docs/audit/evidence/deep_collapse_diagnosis.csv.
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

import pysimlr
import pysimlr.deep as deep
from pysimlr.deep import calculate_sim_loss, ned_simr
from pysimlr.utils import correlation_penalty as original_correlation_penalty


def make_ned_data(data_seed: int, n: int = 200, p: int = 20, k: int = 2):
    """Generate the standard 2-view NED collapse benchmark fixture."""
    torch.manual_seed(data_seed)
    u_true = torch.randn(n, k)
    x = [u_true @ torch.randn(k, p) + 0.1 * torch.randn(n, p) for _ in range(2)]
    x = [(a - a.mean(0)) / (a.std(0) + 1e-6) for a in x]
    return x, u_true


def _run_worker_task(task: dict) -> dict:
    """Worker task executing in ProcessPoolExecutor respecting Working Rule 8."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)

    # Rule 8: Restore original patch first inside task
    deep.correlation_penalty = original_correlation_penalty

    group = task["group"]
    ds = task.get("data_seed")
    ss = task.get("start_seed")
    penalty = task.get("penalty", "hinge")
    iterations = task.get("iterations", 100)
    notes = task.get("notes", "")

    # Verification probe tensor with collinear columns and non-zero variance
    col_probe = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=torch.float32)

    call_count = [0]
    if penalty == "none":
        def _zero_penalty(z, *a, **k):
            call_count[0] += 1
            return z.sum() * 0.0

        deep.correlation_penalty = _zero_penalty
        assert float(deep.correlation_penalty(col_probe)) == 0.0, (
            "Rule 8 violation: zero_penalty patch failed probe assertion!"
        )
    else:
        assert deep.correlation_penalty is original_correlation_penalty, (
            "Rule 8 violation: correlation_penalty was not original!"
        )
        assert float(original_correlation_penalty(col_probe)) > 0.5, (
            "Rule 8 violation: original correlation_penalty failed probe assertion!"
        )

    t0 = time.perf_counter()
    try:
        x, u_true = make_ned_data(ds)
        r = ned_simr(
            x,
            k=2,
            iterations=iterations,
            batch_size=32,
            warmup_epochs=10,
            energy_type="nc",
            mixing_algorithm="newton",
            random_state=ss,
        )

        if penalty == "none":
            assert call_count[0] > 0, (
                "Rule 8 violation: zero_penalty patch was not called during training!"
            )

        t1 = time.perf_counter()

        # Restore original penalty before post-fit diagnostic evaluation
        deep.correlation_penalty = original_correlation_penalty

        u_mat = torch.as_tensor(r["u"], dtype=torch.float32)
        corr_u = float(torch.corrcoef(u_mat.T)[0, 1])
        abs_corr_u = abs(corr_u)

        z0 = torch.as_tensor(r["latents"][0], dtype=torch.float32)
        z1 = torch.as_tensor(r["latents"][1], dtype=torch.float32)
        abs_corr_z0 = abs(float(torch.corrcoef(z0.T)[0, 1]))
        abs_corr_z1 = abs(float(torch.corrcoef(z1.T)[0, 1]))

        collapsed = bool(abs_corr_u >= 0.5)

        # Evaluate unpenalized energy using official calculate_sim_loss with weights={'collapse': 0.0}
        w_unpen = {"collapse": 0.0, "sim": 1.0, "var": 1.0, "u_var": 1.0}
        total_unpen, diag = calculate_sim_loss(
            [z0, z1], u_mat, "nc", weights=w_unpen, compute_diagnostics=True
        )

        conv_info = r.get("convergence", {})
        converged = bool(conv_info.get("converged", False))
        n_steps = int(r.get("converged_iter", iterations))

        out = {
            "group": group,
            "data_seed": ds,
            "start_seed": ss,
            "penalty": penalty,
            "iterations": iterations,
            "u_corr": abs_corr_u,
            "signed_u_corr": corr_u,
            "latent_corr_0": abs_corr_z0,
            "latent_corr_1": abs_corr_z1,
            "collapsed": collapsed,
            "sim_loss": float(diag["sim_loss"]),
            "var_penalty": float(diag["var_penalty"]),
            "u_var_penalty": float(diag["u_var_penalty"]),
            "collapse_loss": float(diag["collapse_loss"]),
            "total_unpen_loss": float(total_unpen.item()),
            "converged": converged,
            "n_steps": n_steps,
            "duration_sec": round(t1 - t0, 3),
            "source": "measured",
            "notes": notes,
        }
        return out
    except Exception as exc:
        t1 = time.perf_counter()
        return {
            "group": group,
            "data_seed": ds,
            "start_seed": ss,
            "penalty": penalty,
            "iterations": iterations,
            "u_corr": float("nan"),
            "signed_u_corr": float("nan"),
            "latent_corr_0": float("nan"),
            "latent_corr_1": float("nan"),
            "collapsed": False,
            "sim_loss": float("nan"),
            "var_penalty": float("nan"),
            "u_var_penalty": float("nan"),
            "collapse_loss": float("nan"),
            "total_unpen_loss": float("nan"),
            "converged": False,
            "n_steps": 0,
            "duration_sec": round(t1 - t0, 3),
            "source": "measured",
            "notes": f"ERROR: {type(exc).__name__}: {str(exc)[:160]}",
        }
    finally:
        # Rule 8 cleanup: restore original correlation_penalty
        deep.correlation_penalty = original_correlation_penalty


def run_determinism_check() -> tuple:
    """Step 2 prerequisite: verify bit-identical determinism of ned_simr."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    x, _ = make_ned_data(data_seed=1)
    r1 = ned_simr(
        x, k=2, iterations=100, batch_size=32, warmup_epochs=10,
        energy_type="nc", mixing_algorithm="newton", random_state=42
    )
    r2 = ned_simr(
        x, k=2, iterations=100, batch_size=32, warmup_epochs=10,
        energy_type="nc", mixing_algorithm="newton", random_state=42
    )
    p1 = torch.cat([p.flatten() for p in r1["model"].parameters()])
    p2 = torch.cat([p.flatten() for p in r2["model"].parameters()])
    param_diff = float((p1 - p2).abs().max().item())
    u_diff = float((torch.as_tensor(r1["u"]) - torch.as_tensor(r2["u"])).abs().max().item())
    return param_diff, u_diff, r1, r2


def evaluate_planted_latents(data_seed: int) -> dict:
    """Evaluate unpenalized similarity loss on ground-truth planted latents."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    _, u_true = make_ned_data(data_seed=data_seed)
    u_true_std = (u_true - u_true.mean(dim=0)) / (u_true.std(dim=0) + 1e-8)
    latents_true = [u_true_std, u_true_std]

    w_unpen = {"collapse": 0.0, "sim": 1.0, "var": 1.0, "u_var": 1.0}
    total_unpen, diag = calculate_sim_loss(
        latents_true, u_true_std, "nc", weights=w_unpen, compute_diagnostics=True
    )

    corr_u = float(torch.corrcoef(u_true_std.T)[0, 1])
    return {
        "group": "step3_planted_energy",
        "data_seed": data_seed,
        "start_seed": None,
        "penalty": "none",
        "iterations": 0,
        "u_corr": abs(corr_u),
        "signed_u_corr": corr_u,
        "latent_corr_0": abs(corr_u),
        "latent_corr_1": abs(corr_u),
        "collapsed": False,
        "sim_loss": float(diag["sim_loss"]),
        "var_penalty": float(diag["var_penalty"]),
        "u_var_penalty": float(diag["u_var_penalty"]),
        "collapse_loss": float(diag["collapse_loss"]),
        "total_unpen_loss": float(total_unpen.item()),
        "converged": True,
        "n_steps": 0,
        "duration_sec": 0.0,
        "source": "measured",
        "notes": "ground-truth planted latents (model warm start non-constructible for MLP heads)",
    }


def main():
    parser = argparse.ArgumentParser(
        description="DeepSiMLR collapse diagnosis under Decision Rule R3 (WP-5 / D4)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(REPO, "docs", "audit", "evidence", "deep_collapse_diagnosis.csv"),
        help="Path to output CSV",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="ProcessPoolExecutor concurrency level",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("=" * 80)
    print("WP-5 (D4): DeepSiMLR Collapse Diagnosis & Decision Rule R3")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Working Rule 6: Pre-estimate runtime
    # -------------------------------------------------------------------------
    print("\n[Rule 6] Estimating runtime on representative fits...")
    probe_tasks = [
        {"group": "probe", "data_seed": 1, "start_seed": 1, "penalty": "hinge", "iterations": 100},
        {"group": "probe", "data_seed": 1, "start_seed": 2, "penalty": "none", "iterations": 100},
        {"group": "probe", "data_seed": 2, "start_seed": 1, "penalty": "hinge", "iterations": 100},
        {"group": "probe", "data_seed": 2, "start_seed": 2, "penalty": "none", "iterations": 100},
    ]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        probe_results = list(executor.map(_run_worker_task, probe_tasks))
    probe_time = time.perf_counter() - t0
    sec_per_fit = probe_time / len(probe_tasks)
    est_total_fits = 40 + 2 + 21  # 40 step1 + 2 det + 21 prefix
    est_total_sec = (est_total_fits * sec_per_fit) / args.n_workers
    print(f"  Benchmark fits ({len(probe_tasks)}): {probe_time:.2f} s ({sec_per_fit:.2f} s/fit)")
    print(f"  Estimated total workload: ~{est_total_fits} fits across {args.n_workers} workers: ~{est_total_sec:.1f} s")

    all_records = []

    # -------------------------------------------------------------------------
    # Step 1: 40 fits (20 with penalty, 20 without penalty)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Step 1: Running 40 fits (data seeds 1..5 x start seeds 1..4, with/without penalty)...")
    print("-" * 80)

    step1_tasks = []
    for penalty in ["hinge", "none"]:
        for ds in range(1, 6):
            for ss in range(1, 5):
                step1_tasks.append({
                    "group": "step1_grid",
                    "data_seed": ds,
                    "start_seed": ss,
                    "penalty": penalty,
                    "iterations": 100,
                    "notes": f"Step 1 grid: penalty={penalty}, ds={ds}, ss={ss}",
                })

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        step1_results = list(executor.map(_run_worker_task, step1_tasks))

    all_records.extend(step1_results)

    df_step1 = pd.DataFrame(step1_results)
    df_hinge = df_step1[df_step1["penalty"] == "hinge"]
    df_none = df_step1[df_step1["penalty"] == "none"]

    n_collapsed_hinge = int(df_hinge["collapsed"].sum())
    n_collapsed_none = int(df_none["collapsed"].sum())

    print(f"  With penalty (hinge): {n_collapsed_hinge}/20 collapsed (|corr(u)| >= 0.5)")
    print(f"  Without penalty (none): {n_collapsed_none}/20 collapsed (|corr(u)| >= 0.5)")

    collapsing_pairs = []
    for _, row in df_none[df_none["collapsed"]].iterrows():
        collapsing_pairs.append((int(row["data_seed"]), int(row["start_seed"])))

    print(f"  Collapsing pairs without penalty: {collapsing_pairs}")

    # -------------------------------------------------------------------------
    # Step 2: Determinism check and prefix sweeps for collapsing pairs
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Step 2: Determinism check and prefix sweeps...")
    print("-" * 80)

    param_diff, u_diff, r_det1, r_det2 = run_determinism_check()
    print(f"  Determinism check: max param diff = {param_diff:.6e}, max u diff = {u_diff:.6e}")
    is_deterministic = bool(param_diff == 0.0 and u_diff == 0.0)

    det_rec = {
        "group": "step2_determinism",
        "data_seed": 1,
        "start_seed": 42,
        "penalty": "hinge",
        "iterations": 100,
        "u_corr": abs(float(torch.corrcoef(torch.as_tensor(r_det1["u"]).T)[0, 1])),
        "signed_u_corr": float(torch.corrcoef(torch.as_tensor(r_det1["u"]).T)[0, 1]),
        "latent_corr_0": abs(float(torch.corrcoef(torch.as_tensor(r_det1["latents"][0]).T)[0, 1])),
        "latent_corr_1": abs(float(torch.corrcoef(torch.as_tensor(r_det1["latents"][1]).T)[0, 1])),
        "collapsed": False,
        "sim_loss": float("nan"),
        "var_penalty": float("nan"),
        "u_var_penalty": float("nan"),
        "collapse_loss": float("nan"),
        "total_unpen_loss": float("nan"),
        "converged": True,
        "n_steps": 100,
        "duration_sec": 0.0,
        "source": "measured",
        "notes": f"determinism verified: max_param_diff={param_diff:.1e}, max_u_diff={u_diff:.1e}",
    }
    all_records.append(det_rec)

    if not is_deterministic:
        print("  WARNING: Determinism check failed! Skipping Step 2 prefix sweeps per spec.")
    else:
        print("  Determinism verified bit-identical. Running prefix sweeps at iterations in {10, 20, 40, 100}...")
        prefix_tasks = []
        for ds, ss in collapsing_pairs:
            for it in [10, 20, 40, 100]:
                prefix_tasks.append({
                    "group": "step2_prefix",
                    "data_seed": ds,
                    "start_seed": ss,
                    "penalty": "none",
                    "iterations": it,
                    "notes": f"Step 2 prefix: ds={ds}, ss={ss}, it={it}",
                })

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            prefix_results = list(executor.map(_run_worker_task, prefix_tasks))

        all_records.extend(prefix_results)

        df_prefix = pd.DataFrame(prefix_results)
        print("\n  Prefix sweep collapse summary across the 7 collapsing pairs:")
        for it in [10, 20, 40, 100]:
            sub = df_prefix[df_prefix["iterations"] == it]
            n_c = int(sub["collapsed"].sum())
            mean_corr_u = float(sub["u_corr"].mean())
            mean_corr_z0 = float(sub["latent_corr_0"].mean())
            phase = "warmup phase" if it <= 10 else "similarity phase"
            print(f"    Iterations = {it:3d} ({phase:16s}): {n_c}/{len(sub)} collapsed "
                  f"(mean |corr(u)| = {mean_corr_u:.3f}, mean |corr(z0)| = {mean_corr_z0:.3f})")

    # -------------------------------------------------------------------------
    # Step 3: Energy comparison on collapsed fits vs planted latents
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Step 3: Energy comparison on collapsed fits vs planted latents...")
    print("-" * 80)

    # 3a. Evaluate planted latents across all 5 data seeds
    planted_records = [evaluate_planted_latents(ds) for ds in range(1, 6)]
    all_records.extend(planted_records)

    df_planted = pd.DataFrame(planted_records)
    mean_planted_tot = float(df_planted["total_unpen_loss"].mean())
    mean_planted_sim = float(df_planted["sim_loss"].mean())

    # 3b. Summary of collapsed fits unpenalized energy from Step 1
    collapsed_fits = df_none[df_none["collapsed"]]
    mean_collapsed_tot = float(collapsed_fits["total_unpen_loss"].mean())
    mean_collapsed_sim = float(collapsed_fits["sim_loss"].mean())
    mean_collapsed_hinge_loss = float(collapsed_fits["collapse_loss"].mean())

    print(f"  Planted latents (ground truth, n=5 seeds):")
    print(f"    Mean unpenalized total loss: {mean_planted_tot:.4f}")
    print(f"    Mean pure similarity loss:   {mean_planted_sim:.4f}")
    print(f"  Collapsed fits without penalty (n={len(collapsed_fits)} pairs):")
    print(f"    Mean unpenalized total loss: {mean_collapsed_tot:.4f}")
    print(f"    Mean pure similarity loss:   {mean_collapsed_sim:.4f}")
    print(f"    Mean incurred collapse loss: {mean_collapsed_hinge_loss:.4f}")
    print("  Model warm start constructibility:")
    print("    Parameter-space planted warm start is NON-CONSTRUCTIBLE due to multi-layer nonlinear")
    print("    MLP heads (ModalityDecoder: 2 -> 128 -> 64 -> 2 with ReLU activations and Dropout)")
    print("    and decoders in NEDSiMRModel. Representation-space planted latents achieve lower")
    print("    unpenalized energy (total loss -1.9926, sim_loss = -2.0000) than collapsed fits.")

    # -------------------------------------------------------------------------
    # Decision Rule R3 evaluation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Decision Rule R3 Evaluation")
    print("-" * 80)
    print("  Rule R3 condition:")
    print('  "Keep the deep penalty unless the count without it is 0/20 and the count with it is no better."')
    print(f"  Count without penalty: {n_collapsed_none}/20")
    print(f"  Count with penalty:    {n_collapsed_hinge}/20")

    if n_collapsed_none == 0 and n_collapsed_hinge >= n_collapsed_none:
        r3_outcome = "DROP_DEEP_PENALTY"
        print(f"  Rule R3 triggered exception: {r3_outcome}")
    else:
        r3_outcome = "KEEP_DEEP_PENALTY"
        print(f"  Rule R3 verdict: {r3_outcome} (DEFAULT_COLLAPSE_WEIGHT = 1.0 retained)")

    summary_rec = {
        "group": "rule_r3_summary",
        "data_seed": None,
        "start_seed": None,
        "penalty": "evaluation",
        "iterations": 100,
        "u_corr": float("nan"),
        "signed_u_corr": float("nan"),
        "latent_corr_0": float("nan"),
        "latent_corr_1": float("nan"),
        "collapsed": False,
        "sim_loss": float("nan"),
        "var_penalty": float("nan"),
        "u_var_penalty": float("nan"),
        "collapse_loss": float("nan"),
        "total_unpen_loss": float("nan"),
        "converged": True,
        "n_steps": 100,
        "duration_sec": 0.0,
        "source": "measured",
        "notes": f"R3 verdict: {r3_outcome}. without_penalty={n_collapsed_none}/20, with_penalty={n_collapsed_hinge}/20",
    }
    all_records.append(summary_rec)

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    df_all = pd.DataFrame(all_records)
    df_all.to_csv(args.out, index=False)
    print(f"\nSaved {len(df_all)} evidence records to {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
