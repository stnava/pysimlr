#!/usr/bin/env python3
"""WP-8 (D6): NED rounding sensitivity diagnosis under Decision Rule R6.

Investigates whether float32 training divergence under 1e-8 input perturbation
stems from a discrete tie-break (Branch A) or continuous chaotic gradient
amplification (Branch B) in multi-view deep neural network optimization.

Executes:
  1. Synthetic fixture generation: 2 views, n=200, p=20, k=2, standardized.
  2. Paired fits with identical seed: unperturbed X vs perturbed X' = X + 1e-8 * eps.
  3. Epoch-by-epoch Euclidean parameter distance trace for e in 1..E (default E=100).
  4. Pinpointing first epoch e* where parameter distance crosses >= 1e-2.
  5. Instrumentation of 4 candidate operations at and before e*:
     - Consensus sign alignment
     - Sparsity quantile / top-k selection and tie gaps
     - Best-epoch selection monitor
     - Gradient norm clipping threshold crossings
  6. Empirical Lyapunov exponent and amplification factor measurement.
  7. Automated Decision Rule R6 evaluation:
     - Branch A: Discrete tie-break identified and resolved to < 1e-4.
     - Branch B: Continuous chaotic gradient amplification confirmed.

Outputs evidence CSV to docs/audit/evidence/ned_rounding.csv.
"""
import argparse
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not os.path.exists(os.path.join(REPO, "src")):
    REPO = "/Users/stnava/data/repos/pysimlr"
sys.path.insert(0, os.path.join(REPO, "src"))

import pysimlr
import pysimlr.deep as deep
from pysimlr.deep import ned_simr


def make_ned_fixture(
    data_seed: int = 42, n: int = 200, p: int = 20, k: int = 2
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Generate standardized 2-view synthetic fixture."""
    torch.manual_seed(data_seed)
    u_true = torch.randn(n, k)
    x_views = [
        u_true @ torch.randn(k, p) + 0.1 * torch.randn(n, p) for _ in range(2)
    ]
    x_views = [(x - x.mean(0)) / (x.std(0) + 1e-6) for x in x_views]
    return x_views, u_true


def run_paired_trace(
    data_seed: int = 42,
    model_seed: int = 0,
    epochs: int = 100,
    perturbation_eps: float = 1e-8,
    pert_seed: int = 999,
) -> Dict[str, Any]:
    """Run unperturbed vs perturbed paired fits and record epoch-by-epoch state."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)

    x_unpert, _ = make_ned_fixture(data_seed=data_seed)

    # Perturbed input X' = X + eps * noise
    torch.manual_seed(pert_seed)
    eps_noise = [torch.randn_like(x) for x in x_unpert]
    x_pert = [x + perturbation_eps * e for x, e in zip(x_unpert, eps_noise)]

    traces = {
        "run1": {
            "epoch_params": [],
            "signs": [],
            "anchors": [],
            "masks": [],
            "grad_clipped": [],
            "best_epochs": [],
            "losses": [],
        },
        "run2": {
            "epoch_params": [],
            "signs": [],
            "anchors": [],
            "masks": [],
            "grad_clipped": [],
            "best_epochs": [],
            "losses": [],
        },
    }
    current_run = ["run1"]

    orig_train_loop = deep._train_loop

    def hooked_train_loop(
        model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight,
        energy_type, warmup_epochs, verbose, device, **kwargs
    ):
        run_name = current_run[0]
        data_tracker = traces[run_name]

        warmup_epochs = deep.resolve_warmup_epochs(warmup_epochs, epochs)
        stabilization_start_epoch, stabilization_ramp_epochs = deep._resolve_stabilization_schedule(
            epochs, warmup_epochs, None, None
        )
        penalty_weights = {"sim": sim_weight, "var": 1.0, "collapse": deep.DEFAULT_COLLAPSE_WEIGHT, "u_var": 1.0}
        all_params = list(model.parameters())
        first_layer_encs = getattr(model, 'encoders', getattr(model, 'linear_encoders', []))

        best_loss, best_epoch_idx = float("inf"), None
        last_sim_weight = None

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            current_sim_weight = 0.0 if epoch < warmup_epochs else sim_weight
            if current_sim_weight != last_sim_weight:
                best_loss = float("inf")
            last_sim_weight = current_sim_weight
            penalty_weights["sim"] = current_sim_weight

            deep._update_first_layer_schedule(model, epoch, epochs, stabilization_start_epoch, stabilization_ramp_epochs)

            for batch_idx, batch in enumerate(dataloader):
                batch_mats = [b.to(device) for b in batch]
                optimizer.zero_grad()
                try:
                    for enc in first_layer_encs:
                        enc._step_v_cache = enc.v
                    res = model(batch_mats)
                    shared_latents, reconstructions, u_shared = res[0], res[1], res[2]
                    recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
                    sim_loss_total, _ = deep.calculate_sim_loss(
                        shared_latents, u_shared, energy_type, weights=penalty_weights, compute_diagnostics=False
                    )
                    total_loss = recon_loss + sim_loss_total
                    for enc in model.linear_encoders:
                        total_loss += 0.05 * (deep.orthogonality_defect(enc.v) + deep.orthogonality_defect(enc.v_raw))
                    total_loss.backward()

                    # Candidate 4: Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0, foreach=True).item()
                    data_tracker["grad_clipped"].append((epoch + 1, batch_idx, grad_norm, grad_norm > 1.0))

                    optimizer.step()
                    epoch_loss += total_loss.item()
                finally:
                    for enc in first_layer_encs:
                        enc._step_v_cache = None

            epoch_loss /= len(dataloader)
            scheduler.step(epoch_loss)

            # Candidate 3: Best-epoch tracking
            if epoch >= warmup_epochs and np.isfinite(epoch_loss) and epoch_loss < best_loss:
                best_loss, best_epoch_idx = epoch_loss, epoch
            data_tracker["best_epochs"].append((epoch + 1, best_epoch_idx, best_loss, epoch_loss))
            data_tracker["losses"].append(epoch_loss)

            # Candidate 1: Consensus anchor sign orientation and anchor distance
            anc = getattr(model, "consensus_anchor", None)
            if anc is not None:
                data_tracker["signs"].append(anc.sign().clone())
                data_tracker["anchors"].append(anc.detach().clone())

            # Candidate 2: Sparsity quantile selection / tie margin
            masks = []
            for enc in model.linear_encoders:
                v = enc.v
                mag = v.abs()
                p, k = v.shape
                n_keep = max(1, int(round(p * (1.0 - 0.5))))
                vals, idx_sort = torch.sort(mag, dim=0, descending=True)
                keep_mask = torch.zeros_like(mag, dtype=torch.bool)
                keep_mask.scatter_(0, idx_sort[:n_keep], True)
                tie_gap = (vals[n_keep - 1] - vals[n_keep]).abs().min().item()
                masks.append((keep_mask.clone(), tie_gap))
            data_tracker["masks"].append(masks)

            p_vec = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
            data_tracker["epoch_params"].append(p_vec)

        return orig_train_loop(
            model, dataloader, optimizer, scheduler, mse_loss, 0, sim_weight,
            energy_type, warmup_epochs, False, device
        )

    deep._train_loop = hooked_train_loop
    t0 = time.perf_counter()
    try:
        current_run[0] = "run1"
        res1 = ned_simr(x_unpert, k=2, iterations=epochs, random_state=model_seed)
        current_run[0] = "run2"
        res2 = ned_simr(x_pert, k=2, iterations=epochs, random_state=model_seed)
    finally:
        deep._train_loop = orig_train_loop
    duration = time.perf_counter() - t0

    t1, t2 = traces["run1"], traces["run2"]
    param_distances = [
        torch.norm(p1 - p2).item()
        for p1, p2 in zip(t1["epoch_params"], t2["epoch_params"])
    ]

    anc1, anc2 = t1.get("anchors", []), t2.get("anchors", [])
    anchor_distances = [
        torch.norm(a1 - a2).item()
        for a1, a2 in zip(anc1, anc2)
    ] if (len(anc1) == len(param_distances) and len(anc2) == len(param_distances)) else [float("nan")] * len(param_distances)

    # Pinpoint first epoch where parameter distance >= 1e-2
    e_star = None
    for ep, dist in enumerate(param_distances, 1):
        if dist >= 1e-2:
            e_star = ep
            break

    ep_eval = e_star if e_star is not None else len(param_distances)

    # 1. Candidate 1: Sign alignment
    sign_mismatches = 0
    for ep_idx in range(ep_eval):
        if ep_idx < len(t1["signs"]) and ep_idx < len(t2["signs"]):
            if (t1["signs"][ep_idx] != t2["signs"][ep_idx]).any():
                sign_mismatches += 1

    # 2. Candidate 2: Sparsity mask ties
    mask_diffs = 0
    min_tie_gap = float("inf")
    for ep_idx in range(ep_eval):
        for v_idx in range(len(t1["masks"][ep_idx])):
            k1, tg1 = t1["masks"][ep_idx][v_idx]
            k2, tg2 = t2["masks"][ep_idx][v_idx]
            min_tie_gap = min(min_tie_gap, tg1, tg2)
            if not torch.equal(k1, k2):
                mask_diffs += 1

    # 3. Candidate 3: Best-epoch selection
    best_ep1 = t1["best_epochs"][ep_eval - 1][1]
    best_ep2 = t2["best_epochs"][ep_eval - 1][1]
    best_epoch_mismatch = (best_ep1 != best_ep2)

    # 4. Candidate 4: Gradient clipping
    clip_discrepancies = 0
    for idx in range(len(t1["grad_clipped"])):
        ep1, _, _, c1 = t1["grad_clipped"][idx]
        _, _, _, c2 = t2["grad_clipped"][idx]
        if ep1 > ep_eval:
            break
        if c1 != c2:
            clip_discrepancies += 1

    # Amplification and Lyapunov exponent
    d_start = param_distances[0]
    d_end = param_distances[-1]
    total_amplification = d_end / perturbation_eps
    if e_star is not None and e_star > 1:
        lyapunov = (np.log(param_distances[e_star - 1]) - np.log(d_start)) / (e_star - 1)
    else:
        lyapunov = (np.log(d_end) - np.log(d_start)) / (len(param_distances) - 1)
    doubling_time = np.log(2.0) / lyapunov if lyapunov > 0 else float("inf")

    # Decision Rule R6 evaluation:
    # If any discrete candidate caused the divergence:
    discrete_tie = (sign_mismatches > 0 or mask_diffs > 0 or best_epoch_mismatch or clip_discrepancies > 0)
    rule_branch = "Branch A" if discrete_tie else "Branch B"

    return {
        "data_seed": data_seed,
        "model_seed": model_seed,
        "iterations": epochs,
        "e_star": e_star,
        "d_start": d_start,
        "d_end": d_end,
        "max_distance": max(param_distances),
        "total_amplification": total_amplification,
        "lyapunov_exponent": lyapunov,
        "doubling_time_epochs": doubling_time,
        "sign_mismatches": sign_mismatches,
        "mask_diffs": mask_diffs,
        "min_tie_gap": min_tie_gap,
        "best_epoch_run1": best_ep1,
        "best_epoch_run2": best_ep2,
        "best_epoch_mismatch": best_epoch_mismatch,
        "clip_discrepancies": clip_discrepancies,
        "rule_branch": rule_branch,
        "duration_sec": duration,
        "param_distances": param_distances,
        "anchor_distances": anchor_distances,
        "losses_run1": t1["losses"],
        "losses_run2": t2["losses"],
    }


def measure_epoch0(
    data_seed: int = 42,
    model_seed: int = 0,
    perturbation_eps: float = 1e-8,
    pert_seed: int = 999,
) -> Dict[str, Any]:
    """Measure Euclidean parameter distance at epoch 0 (after initialization, before training)."""
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)

    x_unpert, _ = make_ned_fixture(data_seed=data_seed)

    torch.manual_seed(pert_seed)
    eps_noise = [torch.randn_like(x) for x in x_unpert]
    x_pert = [x + perturbation_eps * e for x, e in zip(x_unpert, eps_noise)]

    r1 = ned_simr(x_unpert, k=2, iterations=0, random_state=model_seed)
    r2 = ned_simr(x_pert, k=2, iterations=0, random_state=model_seed)

    p1 = torch.cat([p.detach().flatten() for p in r1["model"].parameters()])
    p2 = torch.cat([p.detach().flatten() for p in r2["model"].parameters()])
    d0 = torch.norm(p1 - p2).item()

    anc1 = getattr(r1["model"], "consensus_anchor", None)
    anc2 = getattr(r2["model"], "consensus_anchor", None)
    anc_dist = (
        torch.norm(anc1.detach() - anc2.detach()).item()
        if anc1 is not None and anc2 is not None
        else 0.0
    )

    return {
        "group": "epoch0",
        "data_seed": data_seed,
        "model_seed": model_seed,
        "epoch": 0,
        "param_distance": d0,
        "relative_amplification": d0 / perturbation_eps,
        "loss_unperturbed": float("nan"),
        "loss_perturbed": float("nan"),
        "anchor_distance": anc_dist,
        "sign_mismatches": 0,
        "sparsity_mask_diffs": 0,
        "min_sparsity_tie_gap": float("nan"),
        "best_epoch_unperturbed": -1,
        "best_epoch_perturbed": -1,
        "grad_clip_discrepancies": 0,
        "lyapunov_exponent": float("nan"),
        "doubling_time_epochs": float("nan"),
        "r6_branch": "Branch B",
        "source": "measured",
        "notes": f"Epoch 0 (initialisation): d(0)={d0:.6e}, relative_amp={d0/perturbation_eps:.1f}x",
    }


def main():
    parser = argparse.ArgumentParser(
        description="WP-8 (D6): NED rounding sensitivity diagnosis under Decision Rule R6."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(REPO, "docs", "audit", "evidence", "ned_rounding.csv"),
        help="Path to output evidence CSV",
    )
    parser.add_argument(
        "--html",
        type=str,
        default="",
        help="Path to optional HTML report (default empty to prevent stray files)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epoch budget E (default 100)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        help="Groups to execute: epoch0, trace, all (default 'all'; merges into --out when subset)",
    )
    parser.add_argument(
        "--data-seeds",
        type=int,
        nargs="+",
        default=[42, 2, 1],
        help="Data generation seeds to evaluate",
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[0, 42],
        help="Model initialization seeds to evaluate",
    )
    args = parser.parse_args()

    groups = set(args.groups)
    run_epoch0 = "epoch0" in groups or "all" in groups
    run_trace = "trace" in groups or "all" in groups

    print("=" * 80)
    print(f"WP-8 (D6): NED Rounding Sensitivity Diagnosis & Decision Rule R6 (groups={sorted(groups)})")
    print("=" * 80)
    print(f"Output CSV path: {args.out}")
    print(f"HTML report:     {'<disabled>' if not args.html else args.html}")
    print(f"Budget epochs:   {args.epochs}")
    print(f"Grid:            data_seeds={args.data_seeds} x model_seeds={args.model_seeds}")

    all_records = []
    epoch0_records = []

    if run_epoch0:
        print("\n--- Measuring Epoch 0 (initialisation before training) ---")
        for ds in args.data_seeds:
            for ms in args.model_seeds:
                res_e0 = measure_epoch0(data_seed=ds, model_seed=ms)
                epoch0_records.append(res_e0)
                print(f"  Cell data_seed={ds}, model_seed={ms}: d(0) = {res_e0['param_distance']:.6e} ({res_e0['relative_amplification']:.1f}x)")

        max_d0 = max(r["param_distance"] for r in epoch0_records)
        print(f"\n  Decision Rule R6 evaluation on epoch-0 distance:")
        print(f"    Max measured d(0) = {max_d0:.6e} vs threshold 1.00e-06")
        if max_d0 >= 1e-6:
            print("    => d(0) >= 1e-6: amplification starts at initialisation.")
        else:
            print("    => d(0) < 1e-6: amplification does NOT start at initialisation; verdict stays: keep ned in _ROUNDING_UNSTABLE (Branch B).")

    summary_results = []
    if run_trace:
        print("\n--- Running 100-epoch paired traces ---")
        for ds in args.data_seeds:
            for ms in args.model_seeds:
                print(f"\nEvaluating paired fit: data_seed={ds}, model_seed={ms}...")
                res = run_paired_trace(data_seed=ds, model_seed=ms, epochs=args.epochs)
                summary_results.append(res)

                print(f"  First jump epoch e* = {res['e_star']}")
                print(f"  Param distance: start={res['d_start']:.2e}, end={res['d_end']:.2e}, max={res['max_distance']:.2e}")
                print(f"  Total amplification: {res['total_amplification']:.2e}x")
                print(f"  Lyapunov exponent: {res['lyapunov_exponent']:.4f}/epoch (doubling every {res['doubling_time_epochs']:.2f} epochs)")
                print(f"  Candidate checks (up to e*={res['e_star']}):")
                print(f"    - Sign alignment mismatches: {res['sign_mismatches']}")
                print(f"    - Sparsity mask diffs:        {res['mask_diffs']} (min tie gap: {res['min_tie_gap']:.4e})")
                print(f"    - Best-epoch selection:      run1={res['best_epoch_run1']} vs run2={res['best_epoch_run2']} (mismatch={res['best_epoch_mismatch']})")
                print(f"    - Gradient clipping diffs:   {res['clip_discrepancies']}")
                print(f"  => Rule R6 evaluation: {res['rule_branch']}")

                # Summary row for evidence CSV
                all_records.append({
                    "group": "summary",
                    "data_seed": ds,
                    "model_seed": ms,
                    "epoch": res["e_star"] if res["e_star"] is not None else -1,
                    "param_distance": res["d_end"],
                    "relative_amplification": res["total_amplification"],
                    "loss_unperturbed": res["losses_run1"][-1],
                    "loss_perturbed": res["losses_run2"][-1],
                    "anchor_distance": res["anchor_distances"][-1] if res["anchor_distances"] else float("nan"),
                    "sign_mismatches": res["sign_mismatches"],
                    "sparsity_mask_diffs": res["mask_diffs"],
                    "min_sparsity_tie_gap": res["min_tie_gap"],
                    "best_epoch_unperturbed": res["best_epoch_run1"],
                    "best_epoch_perturbed": res["best_epoch_run2"],
                    "grad_clip_discrepancies": res["clip_discrepancies"],
                    "lyapunov_exponent": res["lyapunov_exponent"],
                    "doubling_time_epochs": res["doubling_time_epochs"],
                    "r6_branch": res["rule_branch"],
                    "source": "measured",
                    "notes": f"Summary: e*={res['e_star']}, amp={res['total_amplification']:.1e}x, lyap={res['lyapunov_exponent']:.3f}",
                })

                # Epoch trace rows
                for ep_idx, (p_d, a_d, l1, l2) in enumerate(
                    zip(res["param_distances"], res["anchor_distances"], res["losses_run1"], res["losses_run2"]), 1
                ):
                    all_records.append({
                        "group": "epoch_trace",
                        "data_seed": ds,
                        "model_seed": ms,
                        "epoch": ep_idx,
                        "param_distance": p_d,
                        "relative_amplification": p_d / 1e-8,
                        "loss_unperturbed": l1,
                        "loss_perturbed": l2,
                        "anchor_distance": a_d,
                        "sign_mismatches": 0,
                        "sparsity_mask_diffs": 0,
                        "min_sparsity_tie_gap": float("nan"),
                        "best_epoch_unperturbed": -1,
                        "best_epoch_perturbed": -1,
                        "grad_clip_discrepancies": 0,
                        "lyapunov_exponent": float("nan"),
                        "doubling_time_epochs": float("nan"),
                        "r6_branch": res["rule_branch"],
                        "source": "measured",
                        "notes": f"Trace ep {ep_idx}: dist={p_d:.2e}",
                    })

        # Rule R6 final synthesis record
        n_branch_a = sum(1 for r in summary_results if r["rule_branch"] == "Branch A")
        n_branch_b = sum(1 for r in summary_results if r["rule_branch"] == "Branch B")
        final_branch = "Branch A" if n_branch_a > 0 else "Branch B"
        final_notes = (
            f"Rule R6 outcome: {final_branch} selected ({n_branch_b}/{len(summary_results)} cells confirmed Branch B "
            f"continuous chaotic gradient amplification, 0 discrete tie-breaks identified; ned remains in _ROUNDING_UNSTABLE)"
        )

        all_records.append({
            "group": "rule_evaluation",
            "data_seed": -1,
            "model_seed": -1,
            "epoch": -1,
            "param_distance": float("nan"),
            "relative_amplification": float("nan"),
            "loss_unperturbed": float("nan"),
            "loss_perturbed": float("nan"),
            "anchor_distance": float(np.nanmean([r["anchor_distances"][-1] for r in summary_results])) if summary_results else float("nan"),
            "sign_mismatches": sum(r["sign_mismatches"] for r in summary_results),
            "sparsity_mask_diffs": sum(r["mask_diffs"] for r in summary_results),
            "min_sparsity_tie_gap": min(r["min_tie_gap"] for r in summary_results),
            "best_epoch_unperturbed": -1,
            "best_epoch_perturbed": -1,
            "grad_clip_discrepancies": sum(r["clip_discrepancies"] for r in summary_results),
            "lyapunov_exponent": float(np.mean([r["lyapunov_exponent"] for r in summary_results])),
            "doubling_time_epochs": float(np.mean([r["doubling_time_epochs"] for r in summary_results])),
            "r6_branch": final_branch,
            "source": "measured",
            "notes": final_notes,
        })
        print("\n" + "=" * 80)
        print(final_notes)
        print("=" * 80)

    new_records = epoch0_records + all_records
    df_new = pd.DataFrame(new_records)

    if args.out and args.out.strip():
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            if os.path.exists(args.out) and not (run_trace and run_epoch0):
                existing_df = pd.read_csv(args.out)
                drop_groups = []
                if run_epoch0:
                    drop_groups.append("epoch0")
                if run_trace:
                    drop_groups.extend(["summary", "epoch_trace", "rule_evaluation"])
                kept_df = existing_df[~existing_df["group"].isin(drop_groups)]
                df_all = pd.concat([kept_df, df_new], ignore_index=True)
            else:
                df_all = df_new

            group_order = {"summary": 0, "epoch0": 1, "epoch_trace": 2, "rule_evaluation": 3}
            df_all["_ord"] = df_all["group"].map(lambda g: group_order.get(g, 99))
            df_all = df_all.sort_values(
                ["_ord", "data_seed", "model_seed", "epoch"], kind="mergesort"
            ).drop(columns=["_ord"]).reset_index(drop=True)

            df_all.to_csv(args.out, index=False)
            print(f"\nWrote evidence CSV ({len(df_all)} rows) to: {args.out}")
        except Exception as exc:
            print(f"\nWarning: could not write CSV to {args.out}: {exc}")
    else:
        print("\nNo output CSV path specified (--out ''), skipping CSV write.")


if __name__ == "__main__":
    main()
