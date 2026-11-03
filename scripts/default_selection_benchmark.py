#!/usr/bin/env python3
"""Joint choice of the linear defaults: objective x optimizer x mixing, under both positivity settings.

Primary measure: least-squares R^2 of the true shared latent on the fitted ``u``
(recovery of the planted signal). Two generators so a choice cannot overfit one:

  buried   `optimizer_benchmark_synthetic.planted_buried`: sparse non-negative shared
           factors under private factors at 2x (PCA start wrong; CCA recovers ~0.98)
  linear   `benchmarks.synthetic_cases.build_case("linear", private_scale=0.5)`:
           the benchmark's standard linear regime

Only objectives valid on every path (linear, deep, flow) are candidates, since the
default is global: acc, nc, logcosh, exp. Every fit uses the same budget and seed.

Writes one row per fit (energy, optimizer, mixing, positivity, generator, seed,
latent_r2, v_support_f1, loadings_support, budget, converged, n_steps, stop_reason,
learning_rate, weight_spread, seconds, stop, error) to --out.

Usage:
  python scripts/default_selection_benchmark.py --stage1 [--out docs/audit/evidence/default_selection_stage1.csv]
  python scripts/default_selection_benchmark.py --stage2 [--out docs/audit/evidence/default_selection_stage2.csv]
"""
import argparse
import itertools
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

ENERGIES = ["acc", "nc", "logcosh", "exp"]
OPTIMIZERS = [
    "hybrid_adam", "adam", "nadam", "rmsprop", "gd",
    "armijo_gradient", "bidirectional_armijo_gradient",
    "lookahead", "bidirectional_lookahead",
    "torch_adamw", "torch_adagrad", "torch_nadam", "lars"
]
MIXINGS = ["newton", "svd", "avg", "pca"]
POSITIVITY = ["positive", "either"]
GENERATORS = ["buried", "linear"]


def _case(generator, seed):
    import numpy as np
    import torch
    from optimizer_benchmark_synthetic import planted_buried
    if generator == "buried":
        z, views, sups = planted_buried(seed)
        return z, views, sups
    from pysimlr.benchmarks.synthetic_cases import build_case
    c = build_case("linear", n_samples=300, seed=seed, private_scale=0.5)
    sups = []
    for v in c["true_v"]:
        v = np.asarray(v)
        sups.append([set(np.flatnonzero(np.abs(v[:, j]) > 1e-12).tolist()) for j in range(v.shape[1])])
    return torch.as_tensor(np.asarray(c["true_u"])), [torch.as_tensor(np.asarray(x)).float() for x in c["data"]], sups


def run_one(task):
    if len(task) == 7:
        energy, opt, mixing, pos, gen, seed, iterations = task
        probe_grid = None
    elif len(task) == 8:
        energy, opt, mixing, pos, gen, seed, iterations, probe_grid = task
    else:
        raise ValueError(f"Unexpected task length: {len(task)}")
    warnings.simplefilter("ignore")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    if probe_grid is not None:
        import pysimlr.optimizers as _opt_mod
        _opt_mod.LR_PROBE_GRID = probe_grid
    from optimizer_benchmark_synthetic import latent_r2, support_f1, loadings_support_auc
    from pysimlr.simlr import simlr
    row = dict(energy=energy, optimizer=opt, mixing=mixing, positivity=pos, generator=gen, seed=seed)
    try:
        z, views, sups = _case(gen, seed)
        k = z.shape[1]
        t = time.time()
        kwargs = dict(energy_type=energy, optimizer_type=opt,
                      mixing_algorithm=mixing, positivity=pos, random_state=seed)
        if iterations is not None:
            kwargs["iterations"] = iterations
        r = simlr(views, k=k, **kwargs)
        if "loadings" not in r:
            from pysimlr.loadings import loadings as _loadings
            lo = _loadings(r, views)
            r["loadings"] = lo["loadings"]
        conv = r.get("convergence", {})
        res_budget = conv.get("max_steps", iterations if iterations is not None else 300)
        v_sup = float(np.mean([support_f1(v.detach(), s) for v, s in zip(r["v"], sups)]))
        if pos == "positive":
            l_sup = float(np.mean([support_f1(np.asarray(b, float), s) for b, s in zip(r["loadings"], sups)]))
        else:
            l_sup = loadings_support_auc(r["loadings"], sups)
        w = r.get("normalizing_weights")
        w_spread = float(max(w) / max(min(w), 1e-12)) if w else 1.0
        row.update(seconds=time.time() - t,
                   latent_r2=latent_r2(z, r["u"]),
                   v_support_f1=v_sup,
                   support_f1=v_sup,
                   loadings_support=l_sup,
                   budget=res_budget,
                   converged=bool(conv.get("converged", r.get("converged", False))),
                   n_steps=int(conv.get("n_steps", 0)),
                   stop_reason=conv.get("stop_reason", r.get("stop_reason")),
                   learning_rate=r.get("learning_rate"),
                   weight_spread=w_spread,
                   stop=r.get("stop_reason"),
                   error=None)
    except (RuntimeError, ValueError, FloatingPointError) as exc:  # recorded, never dropped
        row.update(error=f"{type(exc).__name__}: {str(exc)[:120]}")
    return row


def check_and_apply_r11(df, workers=10, iterations=None):
    """Rule R11: For any (optimizer, energy) with >= 25% of fits at a LR_PROBE_GRID endpoint,
    extend grid one half-decade beyond it, at most twice per direction. Re-run affected rows.
    """
    import pandas as pd
    from pysimlr.optimizers import LR_PROBE_GRID, extend_probe_grid

    ok = df[df["error"].isna()].copy()
    excluded_pairs = []
    extended_info = []

    pairs = ok.groupby(["optimizer", "energy"])
    rerun_tasks = []

    for (opt, energy), grp in pairs:
        n_fits = len(grp)
        upper_endpoint = LR_PROBE_GRID[0]  # 1.0
        lower_endpoint = LR_PROBE_GRID[-1] # 0.001

        upper_hits = (grp["learning_rate"] == upper_endpoint).sum()
        lower_hits = (grp["learning_rate"] == lower_endpoint).sum()

        upper_ratio = upper_hits / n_fits
        lower_ratio = lower_hits / n_fits

        upper_steps = 0
        lower_steps = 0

        if upper_ratio >= 0.25:
            upper_steps = 1
        if lower_ratio >= 0.25:
            lower_steps = 1

        if upper_steps > 0 or lower_steps > 0:
            print(f"[R11 Trigger] ({opt}, {energy}): upper={upper_ratio:.2f}, lower={lower_ratio:.2f}")
            current_grid = extend_probe_grid(upper_steps=upper_steps, lower_steps=lower_steps)
            # Re-run the rows for this pair
            pair_tasks = []
            for _, r in grp.iterrows():
                pair_tasks.append((energy, opt, r["mixing"], r["positivity"], r["generator"], int(r["seed"]), iterations, current_grid))
            
            with ProcessPoolExecutor(max_workers=workers) as ex:
                new_rows = list(ex.map(run_one, pair_tasks, chunksize=4))
            new_df = pd.DataFrame(new_rows)
            
            # Check if second extension is needed
            new_upper_end = current_grid[0]
            new_lower_end = current_grid[-1]
            new_upper_ratio = (new_df["learning_rate"] == new_upper_end).sum() / len(new_df)
            new_lower_ratio = (new_df["learning_rate"] == new_lower_end).sum() / len(new_df)

            if (new_upper_ratio >= 0.25 and upper_steps < 2) or (new_lower_ratio >= 0.25 and lower_steps < 2):
                if new_upper_ratio >= 0.25 and upper_steps < 2:
                    upper_steps = 2
                if new_lower_ratio >= 0.25 and lower_steps < 2:
                    lower_steps = 2
                current_grid = extend_probe_grid(upper_steps=upper_steps, lower_steps=lower_steps)
                pair_tasks = []
                for _, r in grp.iterrows():
                    pair_tasks.append((energy, opt, r["mixing"], r["positivity"], r["generator"], int(r["seed"]), iterations, current_grid))
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    new_rows = list(ex.map(run_one, pair_tasks, chunksize=4))
                new_df = pd.DataFrame(new_rows)
                new_upper_end = current_grid[0]
                new_lower_end = current_grid[-1]
                new_upper_ratio = (new_df["learning_rate"] == new_upper_end).sum() / len(new_df)
                new_lower_ratio = (new_df["learning_rate"] == new_lower_end).sum() / len(new_df)

            # Check exclusion if still at boundary after max extensions
            still_upper = new_upper_ratio >= 0.25 and upper_steps == 2
            still_lower = new_lower_ratio >= 0.25 and lower_steps == 2
            if still_upper or still_lower:
                print(f"[R11 Exclusion] ({opt}, {energy}) still at boundary: upper={new_upper_ratio:.2f}, lower={new_lower_ratio:.2f}")
                excluded_pairs.append((opt, energy))

            extended_info.append({
                "optimizer": opt,
                "energy": energy,
                "upper_steps": upper_steps,
                "lower_steps": lower_steps,
                "grid": current_grid,
                "still_boundary": still_upper or still_lower
            })

            # Replace rows in df
            mask = (df["optimizer"] == opt) & (df["energy"] == energy)
            df = df[~mask].copy()
            df = pd.concat([df, new_df], ignore_index=True)

    return df, excluded_pairs, extended_info


def evaluate_r12(df_stage2):
    """Evaluate Rule R12 on Stage 2 dataframe."""
    import pandas as pd
    from pysimlr.similarity import SCALE_FREE_ENERGIES

    ok = df_stage2[df_stage2["error"].isna()].copy()
    incumbent = ("nc", "hybrid_adam", "newton")

    inc_grp = ok[(ok["energy"] == incumbent[0]) & (ok["optimizer"] == incumbent[1]) & (ok["mixing"] == incumbent[2])]
    inc_indexed = inc_grp.set_index(["positivity", "generator", "seed"])
    inc_r2 = inc_indexed["latent_r2"]
    inc_loadings = inc_indexed["loadings_support"]
    inc_cells = inc_grp.groupby(["positivity", "generator"])["latent_r2"].mean()

    combos = ok.groupby(["energy", "optimizer", "mixing"])
    summary = []

    for (energy, opt, mixing), grp in combos:
        conv_rate = float(grp["converged"].mean())
        mean_r2 = float(grp["latent_r2"].mean())
        cells = grp.groupby(["positivity", "generator"])["latent_r2"].mean()
        worst_cell = float(cells.min())
        mean_loadings = float(grp["loadings_support"].mean())
        med_sec = float(grp["seconds"].median())

        grp_indexed = grp.set_index(["positivity", "generator", "seed"])
        common = inc_indexed.index.intersection(grp_indexed.index)
        cand_r2 = grp_indexed.loc[common, "latent_r2"]
        cand_loadings = grp_indexed.loc[common, "loadings_support"]

        diff_r2 = cand_r2 - inc_r2.loc[common]
        paired_diff_mean = float(diff_r2.mean())
        paired_diff_sd = float(diff_r2.std(ddof=1)) if len(diff_r2) >= 2 else 0.0
        better_beyond_noise = (len(diff_r2) >= 2) and (paired_diff_mean > paired_diff_sd)

        cell_diffs = cells - inc_cells
        max_cell_diff = float(cell_diffs.max())

        diff_loadings = cand_loadings - inc_loadings.loc[common]
        mean_loadings_diff = float(diff_loadings.mean())
        sd_loadings_diff = float(diff_loadings.std(ddof=1)) if len(diff_loadings) >= 2 else 0.0
        loadings_worse = (mean_loadings_diff < -sd_loadings_diff) or (float(diff_loadings.min()) < -0.10)
        loadings_non_inferior = not loadings_worse

        summary.append({
            "energy": energy,
            "optimizer": opt,
            "mixing": mixing,
            "conv_rate": conv_rate,
            "mean_r2": mean_r2,
            "worst_cell": worst_cell,
            "loadings_support": mean_loadings,
            "paired_diff_mean": paired_diff_mean,
            "paired_diff_sd": paired_diff_sd,
            "better_beyond_noise": better_beyond_noise,
            "max_cell_diff": max_cell_diff,
            "loadings_non_inferior": loadings_non_inferior,
            "median_seconds": med_sec,
            "is_incumbent": (energy, opt, mixing) == incumbent,
        })

    s_df = pd.DataFrame(summary)
    inc_row = s_df[s_df["is_incumbent"]].iloc[0]

    print("\n=== Literal Section 3 R12 Stage 2 Table ===")
    cols_display = ["energy", "optimizer", "mixing", "conv_rate", "mean_r2", "worst_cell",
                    "paired_diff_mean", "paired_diff_sd", "max_cell_diff", "loadings_support", "median_seconds"]
    print(s_df.sort_values("mean_r2", ascending=False)[cols_display].to_string(index=False))

    inc_conv_ok = inc_row["conv_rate"] >= 0.9
    better_cands = s_df[~s_df["is_incumbent"] & s_df["better_beyond_noise"]]
    cell_guard_fails = s_df[~s_df["is_incumbent"] & (s_df["max_cell_diff"] > 0.05)]

    print(f"\nIncumbent convergence: {inc_row['conv_rate']:.3f} (>= 0.9: {inc_conv_ok})")
    print(f"Candidates better beyond noise (mean_diff > SD_diff): {len(better_cands)}")
    print(f"Candidates where incumbent is > 0.05 below on any cell: {len(cell_guard_fails)}")

    if inc_conv_ok and better_cands.empty and cell_guard_fails.empty:
        print("\n=> R12 DECISION: Incumbent STAYS.")
        return incumbent, "incumbent_retained", s_df
    else:
        print("\n=> Incumbent criteria failed; picking alternative candidate.")
        # Filter candidates:
        # - non-inferior on loadings support: loadings >= inc_loadings - 0.10
        # - worst cell >= inc_worst_cell - 0.02
        candidates = s_df[~s_df["is_incumbent"]].copy()
        valid = candidates[
            (candidates["loadings_support"] >= inc_row["loadings_support"] - 0.10) &
            (candidates["worst_cell"] >= inc_row["worst_cell"] - 0.02)
        ].copy()
        
        if valid.empty:
            print("No candidate meets non-inferiority constraints; incumbent STAYS.")
            return incumbent, "incumbent_retained_no_valid_alternative", s_df

        # Sort by mean_r2 desc, ties within 0.005 by conv_rate desc, then median_seconds asc
        valid = valid.sort_values(["mean_r2", "conv_rate", "median_seconds"], ascending=[False, False, True])
        top_cand = valid.iloc[0]
        winner = (top_cand["energy"], top_cand["optimizer"], top_cand["mixing"])
        
        # Energy constraint: only change if winner energy in SCALE_FREE_ENERGIES
        if winner[0] not in SCALE_FREE_ENERGIES:
            print(f"Winner energy {winner[0]!r} not in SCALE_FREE_ENERGIES; falling back to best 'nc' candidate.")
            nc_valid = valid[valid["energy"] == "nc"]
            if not nc_valid.empty:
                winner = (nc_valid.iloc[0]["energy"], nc_valid.iloc[0]["optimizer"], nc_valid.iloc[0]["mixing"])
            else:
                winner = incumbent
        print(f"\n=> R12 DECISION: WINNER is {winner}")
        return winner, "alternative_selected", s_df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage1", action="store_true", help="Run Stage 1 benchmark (seeds 0-1 across full grid)")
    ap.add_argument("--stage2", action="store_true", help="Run Stage 2 benchmark (seeds 0-4 for top 8 + incumbent)")
    ap.add_argument("--seeds", type=int, default=None, help="Number of seeds (default: 2 for stage1, 5 for stage2)")
    ap.add_argument("--iterations", type=int, default=None, help="Budget (default: package default 300)")
    ap.add_argument("--workers", type=int, default=10, help="Number of workers")
    ap.add_argument("--only-optimizer", default=None, help="Comma-separated optimizers to run")
    ap.add_argument("--only-energy", default=None, help="Comma-separated energies to run")
    ap.add_argument("--out", default=None, help="Output CSV path")
    a = ap.parse_args()

    import pandas as pd

    opts = a.only_optimizer.split(",") if a.only_optimizer else OPTIMIZERS
    energies = a.only_energy.split(",") if a.only_energy else ENERGIES

    if a.stage1:
        n_seeds = a.seeds if a.seeds is not None else 2
        tasks = [(e, o, m, p, g, s, a.iterations) for e, o, m, p, g, s in
                 itertools.product(energies, opts, MIXINGS, POSITIVITY, GENERATORS, range(n_seeds))]
        print(f"Running Stage 1: {len(tasks)} fits...", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(run_one, tasks, chunksize=4))
        df = pd.DataFrame(rows)
        print(f"Stage 1 done in {time.time() - t0:.1f}s; errors: {int(df['error'].notna().sum())}", flush=True)

        # Check and apply R11
        df, excluded_pairs, extended_info = check_and_apply_r11(df, workers=a.workers, iterations=a.iterations)

        out_path = a.out or "docs/audit/evidence/default_selection_stage1.csv"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved Stage 1 results to {out_path} ({len(df)} rows)")

    elif a.stage2:
        stage1_path = "docs/audit/evidence/default_selection_stage1.csv"
        if not os.path.exists(stage1_path):
            raise FileNotFoundError(f"Stage 1 CSV not found at {stage1_path}. Run --stage1 first.")
        df_s1 = pd.read_csv(stage1_path)
        
        # Drop combinations with convergence rate < 0.9 and R11 boundary excluded pairs
        combo_cols = ["energy", "optimizer", "mixing"]
        ok_s1 = df_s1[df_s1["error"].isna()].copy()

        # Identify R11 excluded pairs (>= 25% boundary fits at 1.0 or 0.001)
        excluded_pairs = []
        for (opt, energy), grp in ok_s1.groupby(["optimizer", "energy"]):
            n = len(grp)
            up = (grp["learning_rate"] == 1.0).sum() / n
            low = (grp["learning_rate"] == 0.001).sum() / n
            if up >= 0.25 or low >= 0.25:
                excluded_pairs.append((opt, energy))

        cand_pool = ok_s1.copy()
        for opt, energy in excluded_pairs:
            cand_pool = cand_pool[~((cand_pool["optimizer"] == opt) & (cand_pool["energy"] == energy))]

        conv_rates = cand_pool.groupby(combo_cols)["converged"].mean()
        valid_combos = conv_rates[conv_rates >= 0.9].index.tolist()

        # Compute mean latent R^2 across cells for valid non-excluded combinations
        cell_means = cand_pool.groupby(combo_cols + ["positivity", "generator"])["latent_r2"].mean().unstack(["positivity", "generator"])
        mean_over_cells = cell_means.mean(axis=1)
        mean_over_cells_valid = mean_over_cells.loc[mean_over_cells.index.isin(valid_combos)]
        top8 = mean_over_cells_valid.sort_values(ascending=False).head(8).index.tolist()

        incumbent = ("nc", "hybrid_adam", "newton")
        if incumbent not in top8:
            stage2_combos = top8 + [incumbent]
        else:
            stage2_combos = top8

        print(f"Stage 2 combinations ({len(stage2_combos)}):")
        for c in stage2_combos:
            print(f"  {c}")

        n_seeds = a.seeds if a.seeds is not None else 5
        # Re-use seeds 0 and 1 from Stage 1 for these combinations
        s1_reused = df_s1[df_s1.set_index(combo_cols).index.isin(stage2_combos) & (df_s1["seed"] < 2)].copy()
        
        # Run seeds 2 to n_seeds-1
        new_tasks = []
        for (e, o, m) in stage2_combos:
            for p, g, s in itertools.product(POSITIVITY, GENERATORS, range(2, n_seeds)):
                new_tasks.append((e, o, m, p, g, s, a.iterations))

        print(f"Running Stage 2 new fits: {len(new_tasks)} fits...", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            new_rows = list(ex.map(run_one, new_tasks, chunksize=4))
        df_new = pd.DataFrame(new_rows)
        print(f"Stage 2 new fits done in {time.time() - t0:.1f}s; errors: {int(df_new['error'].notna().sum())}", flush=True)

        df_stage2 = pd.concat([s1_reused, df_new], ignore_index=True)
        out_path = a.out or "docs/audit/evidence/default_selection_stage2.csv"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        df_stage2.to_csv(out_path, index=False)
        print(f"Saved Stage 2 results to {out_path} ({len(df_stage2)} rows)")

        # Evaluate R12
        evaluate_r12(df_stage2)

    else:
        # Default behavior: run all tasks per CLI arguments
        n_seeds = a.seeds if a.seeds is not None else 5
        tasks = [(e, o, m, p, g, s, a.iterations) for e, o, m, p, g, s in
                 itertools.product(energies, opts, MIXINGS, POSITIVITY, GENERATORS, range(n_seeds))]
        print(f"{len(tasks)} fits", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(run_one, tasks, chunksize=4))
        df = pd.DataFrame(rows)
        print(f"done in {time.time() - t0:.0f}s; errors: {int(df['error'].notna().sum())}", flush=True)
        if a.out:
            df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
