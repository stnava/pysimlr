#!/usr/bin/env python3
"""WP-5 (D4): DeepSiMLR collapse diagnosis on NED fixture under Decision Rule R3.

Executes:
  - Step 1: 40 fits (20 data seeds 1..5 x start seeds 1..4, with and without penalty).
            Counts collapsing pairs (|corr(u)| >= 0.5).
  - Step 2: Determinism check (2 identical fits match bit-identically).
            Prefix sweeps at iterations in {10, 20, 40, 100} for collapsing pairs
            to pinpoint onset (warmup vs similarity phase).
  - Step 3: Unpenalized energy evaluation using the model's returned training
            consensus (u_per_modality under topology='loo', else u), asserting
            1e-5 agreement with the trained model's final similarity term on
            penalised converged fits. A planted warm start is not constructible
            for the MLP model.
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

        conv_info = r.get("convergence", {})
        converged = bool(conv_info.get("converged", False))
        n_steps = int(r.get("converged_iter", iterations))

        compute_energy = bool(task.get("compute_energy", False))
        if compute_energy:
            # Evaluate against the model's returned training consensus:
            # under topology="loo", training aligns each view to r["u_per_modality"]
            # rather than the star aggregate r["u"] (whose anchor signs are unaligned).
            u_target = (
                [torch.as_tensor(u, dtype=torch.float32) for u in r["u_per_modality"]]
                if r.get("u_per_modality") is not None
                else u_mat
            )
            w_unpen = {"collapse": 0.0, "sim": 1.0, "var": 1.0, "u_var": 1.0}
            total_unpen, diag = calculate_sim_loss(
                [z0, z1], u_target, "nc", weights=w_unpen, compute_diagnostics=True
            )
            recomputed_sim = float(diag["sim_loss"])

            # Assert agreement to 1e-5 with the trained model's final training similarity term
            xm, _ = deep._standardize_deep(x)
            model_eval = r["model"].eval()
            with torch.no_grad():
                lat_m, _, u_m = model_eval(xm)
            _, diag_m = calculate_sim_loss(lat_m, u_m, "nc", compute_diagnostics=False)
            model_final_sim = float(diag_m["sim_loss"])
            if penalty == "hinge" and converged:
                sim_diff = abs(recomputed_sim - model_final_sim)
                assert sim_diff <= 1e-5, (
                    f"Step 3 evaluator mismatch on penalised converged fit "
                    f"(ds={ds}, ss={ss}): recomputed={recomputed_sim:.7f}, "
                    f"model_final={model_final_sim:.7f}, diff={sim_diff:.2e}"
                )

            sim_loss_val = recomputed_sim
            var_pen_val = float(diag["var_penalty"])
            u_var_pen_val = float(diag["u_var_penalty"])
            collapse_loss_val = float(diag["collapse_loss"])
            total_unpen_val = float(total_unpen.item())
        else:
            sim_loss_val = float("nan")
            var_pen_val = float("nan")
            u_var_pen_val = float("nan")
            collapse_loss_val = float("nan")
            total_unpen_val = float("nan")

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
            "sim_loss": sim_loss_val,
            "var_penalty": var_pen_val,
            "u_var_penalty": u_var_pen_val,
            "collapse_loss": collapse_loss_val,
            "total_unpen_loss": total_unpen_val,
            "converged": converged,
            "n_steps": n_steps,
            "duration_sec": round(t1 - t0, 3),
            "source": "measured",
            "notes": notes,
        }
        return out
    except AssertionError:
        raise
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


def _parse_groups(raw_groups: list) -> set:
    parts = []
    for item in raw_groups:
        parts.extend(p.strip() for p in item.split(",") if p.strip())
    valid = {"step1", "step2", "step3", "all"}
    unknown = set(parts) - valid
    if unknown:
        raise ValueError(f"Unknown --groups {sorted(unknown)}; choose from {sorted(valid)}")
    if "all" in parts:
        return {"step1", "step2", "step3"}
    return set(parts)


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
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        help="Groups to execute: step1, step2, step3, or all (merges into --out when subset)",
    )
    args = parser.parse_args()
    groups = _parse_groups(args.groups)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("=" * 80)
    print(f"WP-5 (D4): DeepSiMLR Collapse Diagnosis & Decision Rule R3 (groups={sorted(groups)})")
    print("=" * 80)

    # Pre-estimate runtime on 4 fits
    print("\n[Estimate] Timing 4 fits...")
    probe_tasks = [
        {"group": "probe", "data_seed": 1, "start_seed": 1, "penalty": "hinge", "iterations": 100, "compute_energy": ("step3" in groups)},
        {"group": "probe", "data_seed": 1, "start_seed": 2, "penalty": "none", "iterations": 100, "compute_energy": ("step3" in groups)},
        {"group": "probe", "data_seed": 2, "start_seed": 1, "penalty": "hinge", "iterations": 100, "compute_energy": ("step3" in groups)},
        {"group": "probe", "data_seed": 2, "start_seed": 2, "penalty": "none", "iterations": 100, "compute_energy": ("step3" in groups)},
    ]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(4, args.n_workers)) as executor:
        list(executor.map(_run_worker_task, probe_tasks))
    probe_time = time.perf_counter() - t0
    sec_per_fit = probe_time / len(probe_tasks)
    est_total_fits = (
        (40 if "step1" in groups else 0)
        + (30 if "step2" in groups else 0)
        + (40 if ("step3" in groups and "step1" not in groups) else 0)
    )
    est_total_sec = (est_total_fits * sec_per_fit) / max(1, args.n_workers)
    print(f"  4 probe fits: {probe_time:.2f} s ({sec_per_fit:.2f} s/fit)")
    print(f"  Estimated workload: ~{est_total_fits} fits across {args.n_workers} workers: ~{est_total_sec:.1f} s")

    existing_df = pd.read_csv(args.out) if os.path.exists(args.out) else pd.DataFrame()
    new_records = []
    collapsing_pairs = []

    # -------------------------------------------------------------------------
    # Step 1: 40 fits (20 with penalty, 20 without penalty)
    # -------------------------------------------------------------------------
    if "step1" in groups:
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
                        "compute_energy": False,
                        "notes": f"Step 1 grid: penalty={penalty}, ds={ds}, ss={ss}",
                    })

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            step1_results = list(executor.map(_run_worker_task, step1_tasks))

        new_records.extend(step1_results)
        df_step1 = pd.DataFrame(step1_results)
        df_hinge = df_step1[df_step1["penalty"] == "hinge"]
        df_none = df_step1[df_step1["penalty"] == "none"]
        n_collapsed_hinge = int(df_hinge["collapsed"].sum())
        n_collapsed_none = int(df_none["collapsed"].sum())
        print(f"  With penalty (hinge): {n_collapsed_hinge}/20 collapsed (|corr(u)| >= 0.5)")
        print(f"  Without penalty (none): {n_collapsed_none}/20 collapsed (|corr(u)| >= 0.5)")
        for _, row in df_none[df_none["collapsed"]].iterrows():
            collapsing_pairs.append((int(row["data_seed"]), int(row["start_seed"])))

        r3_outcome = (
            "DROP_DEEP_PENALTY"
            if (n_collapsed_none == 0 and n_collapsed_hinge >= n_collapsed_none)
            else "KEEP_DEEP_PENALTY"
        )
        new_records.append({
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
        })
    elif not existing_df.empty and "step1_grid" in existing_df["group"].values:
        s1_existing = existing_df[existing_df["group"] == "step1_grid"]
        for _, row in s1_existing[(s1_existing["penalty"] == "none") & (s1_existing["collapsed"].astype(bool))].iterrows():
            collapsing_pairs.append((int(row["data_seed"]), int(row["start_seed"])))

    # -------------------------------------------------------------------------
    # Step 2: Determinism check and prefix sweeps for collapsing pairs
    # -------------------------------------------------------------------------
    if "step2" in groups:
        print("\n" + "-" * 80)
        print("Step 2: Determinism check and prefix sweeps...")
        print("-" * 80)

        param_diff, u_diff, r_det1, _ = run_determinism_check()
        print(f"  Determinism check: max param diff = {param_diff:.6e}, max u diff = {u_diff:.6e}")
        is_deterministic = bool(param_diff == 0.0 and u_diff == 0.0)

        new_records.append({
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
        })

        if is_deterministic and collapsing_pairs:
            prefix_tasks = []
            for ds, ss in collapsing_pairs:
                for it in [10, 20, 40, 100]:
                    prefix_tasks.append({
                        "group": "step2_prefix",
                        "data_seed": ds,
                        "start_seed": ss,
                        "penalty": "none",
                        "iterations": it,
                        "compute_energy": False,
                        "notes": f"Step 2 prefix: ds={ds}, ss={ss}, it={it}",
                    })
            with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                prefix_results = list(executor.map(_run_worker_task, prefix_tasks))
            new_records.extend(prefix_results)

    # -------------------------------------------------------------------------
    # Step 3: Energy evaluation with model's returned training consensus
    # -------------------------------------------------------------------------
    if "step3" in groups:
        print("\n" + "-" * 80)
        print("Step 3: Evaluating similarity & unpenalized loss on 40 fits...")
        print("-" * 80)

        step3_tasks = []
        for penalty in ["hinge", "none"]:
            for ds in range(1, 6):
                for ss in range(1, 5):
                    step3_tasks.append({
                        "group": "step3_energy",
                        "data_seed": ds,
                        "start_seed": ss,
                        "penalty": penalty,
                        "iterations": 100,
                        "compute_energy": True,
                        "notes": f"Step 3 energy: penalty={penalty}, ds={ds}, ss={ss}",
                    })

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            step3_results = list(executor.map(_run_worker_task, step3_tasks))

        new_records.extend(step3_results)
        new_records.append({
            "group": "step3_constructibility",
            "data_seed": None,
            "start_seed": None,
            "penalty": "none",
            "iterations": 0,
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
            "duration_sec": 0.0,
            "source": "measured",
            "notes": "A planted warm start is not constructible for the MLP model.",
        })

        df_s3 = pd.DataFrame(step3_results)
        hinge_s3 = df_s3[df_s3["penalty"] == "hinge"]
        none_coll_s3 = df_s3[(df_s3["penalty"] == "none") & (df_s3["collapsed"])]
        none_uncoll_s3 = df_s3[(df_s3["penalty"] == "none") & (~df_s3["collapsed"])]
        print(f"  Penalised fits (n={len(hinge_s3)}): mean sim_loss={hinge_s3['sim_loss'].mean():.4f} "
              f"(min={hinge_s3['sim_loss'].min():.4f}, max={hinge_s3['sim_loss'].max():.4f}), "
              f"mean total_unpen={hinge_s3['total_unpen_loss'].mean():.4f}")
        print(f"  Unpenalised uncollapsed (n={len(none_uncoll_s3)}): mean sim_loss={none_uncoll_s3['sim_loss'].mean():.4f} "
              f"(min={none_uncoll_s3['sim_loss'].min():.4f}, max={none_uncoll_s3['sim_loss'].max():.4f}), "
              f"mean total_unpen={none_uncoll_s3['total_unpen_loss'].mean():.4f}")
        print(f"  Unpenalised collapsed (n={len(none_coll_s3)}): mean sim_loss={none_coll_s3['sim_loss'].mean():.4f} "
              f"(min={none_coll_s3['sim_loss'].min():.4f}, max={none_coll_s3['sim_loss'].max():.4f}), "
              f"mean collapse_loss={none_coll_s3['collapse_loss'].mean():.4f}, "
              f"mean total_unpen={none_coll_s3['total_unpen_loss'].mean():.4f}")
        print("  Warm start: A planted warm start is not constructible for the MLP model.")

    # -------------------------------------------------------------------------
    # Merge with preserved groups and save CSV
    # -------------------------------------------------------------------------
    replaced_groups = set()
    if "step1" in groups:
        replaced_groups.update({"step1_grid", "rule_r3_summary"})
    if "step2" in groups:
        replaced_groups.update({"step2_determinism", "step2_prefix"})
    if "step3" in groups:
        replaced_groups.update({"step3_planted_energy", "step3_energy", "step3_constructibility"})

    kept_frames = []
    if not existing_df.empty and groups != {"step1", "step2", "step3"}:
        kept = existing_df[~existing_df["group"].isin(replaced_groups)].copy()
        if not kept.empty:
            kept.loc[kept["source"] == "measured", "source"] = "0d98ff2"
            if "step3" in groups:
                stale_mask = kept["group"].isin(["step1_grid", "step2_prefix"])
                for col in ["sim_loss", "var_penalty", "u_var_penalty", "collapse_loss", "total_unpen_loss"]:
                    if col in kept.columns:
                        kept.loc[stale_mask, col] = float("nan")
            kept_frames.append(kept)

    df_new = pd.DataFrame(new_records)
    df_all = pd.concat(kept_frames + [df_new], ignore_index=True) if kept_frames else df_new
    group_order = {
        "step1_grid": 0,
        "step2_determinism": 1,
        "step2_prefix": 2,
        "step3_energy": 3,
        "step3_constructibility": 4,
        "rule_r3_summary": 5,
    }
    df_all["_ord"] = df_all["group"].map(lambda g: group_order.get(g, 99))
    df_all = df_all.sort_values("_ord", kind="mergesort").drop(columns=["_ord"]).reset_index(drop=True)
    df_all.to_csv(args.out, index=False)
    print(f"\nSaved {len(df_all)} evidence records to {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
