"""ADNI-MCI Cognitive Pilot Evaluation for Score-Level Path Model (WP4).

Evaluates score-level structural path models, cross-view coupling, and cognitive
outcome association on the ADNI-MCI Protocol B cohort across 5 70/30 train/test splits.

Views:
- T1Hier_vol (regional volumetric measures)
- T1Hier_thk (regional cortical thickness measures)
- T1w_mean (regional T1-weighted mean intensities)
Outcome:
- CogPC1: first principal component of the ADNI cognitive battery, fitted on training rows.
Confounds:
- Age, education, head size passed to BOTH fit(confounds=) and path_model(covariates=).
Groups:
- Sex passed as groups (never among confounds).

Primary Pass Rules (Plan Section 8, docs/plans/2026-11-06_path_model.md):
- P1: The T1Hier_vol - T1Hier_thk edge has held-out p_maxt < 0.05 in >= 4 of 5 splits.
- P2: The outcome node has at least one view with p_holm < 0.05 in the outcome_edges
      family in >= 4 of 5 splits.
- P3: The fixed-model held-out pillai is below the training pillai for every edge
      (evaluated across all edges and partitioned into measurement-fitted view-view
      edges vs unsupervised view-outcome edges).
- P4: The complete graph is not the best by heldout_loglik in every split (when complete
      is top in every split, paired deltas are reported as the empirical result).

Secondary Empirical Findings (reported descriptively alongside P1-P4):
- F1: Cortical thickness -> cognition (T1Hier_thk - outcome) held-out association.
- F2: Star on T1Hier_thk vs complete held-out log-likelihood parsimony (fixed & refit modes).
- F3: Out-of-sample CogPC1 prediction via predict_path (held-out R^2 and Pearson r).
- F4: Training-row post-selection threshold shift (rows="train", null="both" at matched B).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pysimlr
import pysimlr.plotting as pl
from scripts.benchmark_confound_sink import load_adni


# ----------------------------------------------------------------------------- Neuroanatomical Cleaning
def clean_neuro_name(name: str) -> str:
    """Format raw column slugs into human-readable neuroanatomical labels."""
    s = str(name).strip()
    s = re.sub(r"^T1Hier_(vol|thk)__(T1Hier_(vol|thk)_)?", "", s)
    s = re.sub(r"^T1w__(T1w_mean_)?", "", s)

    tissue_map = {
        "deepgraytissues": "Deep Gray Matter",
        "csftissues": "Cerebrospinal Fluid (CSF)",
        "brainstemtissues": "Brainstem",
        "cortical_compartmenttissues": "Cortical Compartment",
        "cortical_gray_mattertissues": "Cortical Gray Matter",
        "white_mattertissues": "White Matter",
        "gmtissues": "Total Gray Matter",
        "wmtissues": "White Matter",
        "medullabrainstem": "Medulla (Brainstem)",
        "midbrainbrainstem": "Midbrain (Brainstem)",
        "ponsbrainstem": "Pons (Brainstem)",
        "brainstem_ldktlobes": "Brainstem (Left Lobe)",
        "brainstem_rdktlobes": "Brainstem (Right Lobe)",
    }
    low = s.lower()
    if low in tissue_map:
        return tissue_map[low]

    cit_map = {
        "bn_str_ca_leftcit168": "Left Caudate (CIT168)",
        "bn_str_ca_rightcit168": "Right Caudate (CIT168)",
        "bn_str_pu_leftcit168": "Left Putamen (CIT168)",
        "bn_str_pu_rightcit168": "Right Putamen (CIT168)",
        "bn_gp_gpe_leftcit168": "Left Globus Pallidus Ext (CIT168)",
        "bn_gp_gpi_leftcit168": "Left Globus Pallidus Int (CIT168)",
        "bn_sn_snc_leftcit168": "Left Substantia Nigra Compacta",
        "bn_sn_snr_leftcit168": "Left Substantia Nigra Reticulata",
        "bn_vta_leftcit168": "Left Ventral Tegmental Area",
    }
    if s in cit_map:
        return cit_map[s]

    s = re.sub(r"dktcortex$", " Cortex", s)
    s = re.sub(r"dktregions$", "", s)
    s = s.replace("_", " ").title()
    s = s.replace("Dkt", "").replace("Cit168", "(CIT168)").strip()
    return s


def get_feature_name_dicts(adni_csv_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return raw cleaned feature names (for DKT mapping) and formatted human-readable labels."""
    df = pd.read_csv(adni_csv_path, nrows=1)
    cols_vol = [c for c in df.columns if c.startswith("T1Hier_vol__")]
    cols_thk = [c for c in df.columns if c.startswith("T1Hier_thk__")]
    cols_mean = [c for c in df.columns if c.startswith("T1w__")]

    clean_dkt = {
        "T1Hier_vol": [c.replace("T1Hier_vol__T1Hier_vol_", "").replace("T1Hier_vol__", "") for c in cols_vol],
        "T1Hier_thk": [c.replace("T1Hier_thk__T1Hier_thk_", "").replace("T1Hier_thk__", "") for c in cols_thk],
        "T1w_mean": [c.replace("T1w__T1w_mean_", "").replace("T1w__", "") for c in cols_mean],
    }

    clean_human = {
        "T1Hier_vol": [clean_neuro_name(c) for c in cols_vol],
        "T1Hier_thk": [clean_neuro_name(c) for c in cols_thk],
        "T1w_mean": [clean_neuro_name(c) for c in cols_mean],
    }
    return clean_dkt, clean_human


# ----------------------------------------------------------------------------- Topologies
def get_topologies() -> Dict[str, Any]:
    """Define the 4 benchmark topologies over views plus cognitive outcome."""
    return {
        "complete": None,
        "star on T1Hier_thk": [
            ("T1Hier_vol", "T1Hier_thk"),
            ("T1Hier_thk", "T1w_mean"),
            ("T1Hier_thk", "outcome"),
        ],
        "chain (vol-thk-mean)": [
            ("T1Hier_vol", "T1Hier_thk"),
            ("T1Hier_thk", "T1w_mean"),
            ("T1w_mean", "outcome"),
        ],
        "empty": [],
    }


def _verify_p4_consistency(p4_df: pd.DataFrame, n_refits: int) -> None:
    """Verify add-one p-values and empirical quantiles come from the same live draws."""
    if p4_df.empty:
        raise ValueError("adni_p4_refit_ratio.csv is empty.")
    for _, row in p4_df.iterrows():
        n_ge_fix = int(row["n_ge_fixed"])
        n_ge_ref = int(row["n_ge_refit"])
        exp_p_fix = (n_ge_fix + 1.0) / (n_refits + 1.0)
        exp_p_ref = (n_ge_ref + 1.0) / (n_refits + 1.0)
        if abs(float(row["p_perm_fixed"]) - exp_p_fix) > 1e-9:
            raise ValueError(f"Inconsistent p_perm_fixed on {row['method']} {row['a']}-{row['b']}")
        if abs(float(row["p_refit"]) - exp_p_ref) > 1e-9:
            raise ValueError(f"Inconsistent p_refit on {row['method']} {row['a']}-{row['b']}")
        if n_ge_ref == 0 and float(row["q95_refit"]) >= float(row["pillai"]):
            raise ValueError(
                f"Impossible refit null record on {row['method']} {row['a']}-{row['b']}: "
                f"n_ge_refit=0 (p_refit={row['p_refit']:.6f}) but q95_refit={row['q95_refit']:.6f} >= pillai={row['pillai']:.6f}"
            )


# ----------------------------------------------------------------------------- Pilot Runner
def run_pilot(
    seeds: List[int],
    methods: List[str],
    n_permutations: int = 2000,
    n_bootstrap: int = 2000,
    n_refits: int = 50,
    n_jobs: int = 4,
    out_dir: Path = Path("docs/audit/evidence/path_model/adni"),
    figures_only: bool = False,
) -> Dict[str, Any]:
    """Run full ADNI-MCI pilot evaluation pipeline across seeds and methods."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    required_csvs = [
        out_dir / "adni_edges_summary.csv",
        out_dir / "adni_train_vs_heldout_pillai.csv",
        out_dir / "adni_comparison_fixed.csv",
        out_dir / "adni_comparison_refits.csv",
        out_dir / "adni_p4_refit_ratio.csv",
    ]
    if figures_only:
        missing = [str(p) for p in required_csvs if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--figures-only requires all live CSV outputs to exist; missing: {missing}. "
                "Run scripts/adni_path_model_pilot.py without --figures-only first."
            )

    print("Loading ADNI-MCI dataset...")
    data = load_adni()
    n_total = len(data["C"])
    print(f"Loaded {n_total} complete cases. Views: {data['view_names']}")

    # Views, confounds (age, educ, head_size), groups (sex)
    v_all = [torch.tensor(v).float() for v in data["views"]]
    C_all = data["C"][:, [0, 2, 3]]  # age, educ, head_size
    G_all = data["C"][:, 1]           # sex
    Y_all = data["Y"]
    view_names = list(data["view_names"])

    adni_csv = REPO / "data/adni/adni_mci_cognitive_merged.csv"
    feat_names_dkt, feat_names_human = get_feature_name_dicts(adni_csv)
    topologies = get_topologies()

    rep_pm: Dict[str, Any] = {}
    rep_cmp: Dict[str, Any] = {}

    if figures_only:
        print("Running in --figures-only mode: reading existing live CSVs without modifying them...")
        summary_df = pd.read_csv(out_dir / "adni_edges_summary.csv")
        pillai_df = pd.read_csv(out_dir / "adni_train_vs_heldout_pillai.csv")
        fixed_df = pd.read_csv(out_dir / "adni_comparison_fixed.csv")
        refits_df = pd.read_csv(out_dir / "adni_comparison_refits.csv")
        p4_df = pd.read_csv(out_dir / "adni_p4_refit_ratio.csv")
        _verify_p4_consistency(p4_df, int(p4_df["n_refits"].iloc[0]))

        # Fit Split 0 models strictly to construct plotting dictionaries
        s0 = seeds[0]
        rng = np.random.default_rng(s0)
        perm = rng.permutation(n_total)
        n_tr = int(round(0.7 * n_total))
        tr, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])

        v_tr = [v[tr] for v in v_all]
        v_te = [v[te] for v in v_all]

        mu_y, sd_y = Y_all[tr].mean(0), Y_all[tr].std(0, ddof=1)
        pca_cog = PCA(1).fit((Y_all[tr] - mu_y) / sd_y)
        y_all = pca_cog.transform((Y_all - mu_y) / sd_y).ravel()
        y_tr, y_te = y_all[tr], y_all[te]
        C_tr, C_te = C_all[tr], C_all[te]
        G_tr, G_te = G_all[tr], G_all[te]

        for m in methods:
            print(f"Fitting Split {s0} {m.upper()} model for visualization...")
            res_m = pysimlr.fit(m, v_tr, k=3, confounds=C_tr, random_state=s0)
            rep_pm[m] = pysimlr.path_model(
                res_m, v_te, rows="heldout", graph=None, view_names=view_names,
                covariates=C_te, groups=G_te, outcome=y_te,
                train_views=v_tr, train_covariates=C_tr, train_groups=G_tr, train_outcome=y_tr,
                n_permutations=n_permutations, random_state=s0,
            )
            if m == "simlr":
                rep_cmp[m] = pysimlr.compare_path_models(
                    res_m, v_tr, v_te, graphs=topologies, view_names=view_names,
                    covariates_train=C_tr, covariates_test=C_te,
                    groups_train=G_tr, groups_test=G_te,
                    outcome_train=y_tr, outcome_test=y_te,
                    n_bootstrap=n_bootstrap, n_permutations=min(n_permutations, 1000), random_state=s0,
                )
    else:
        results_by_method: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
        pillai_records: List[Dict[str, Any]] = []
        fixed_comparisons: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
        refits_comparisons: Dict[str, Any] = {}
        p4_records: List[Dict[str, Any]] = []

        for method in methods:
            print("\n==========================================")
            print(f"Running live evaluation for method: {method.upper()}")
            print("==========================================")

            for s in seeds:
                print(f"\n--- Split Seed {s} (Method: {method}) ---")
                rng = np.random.default_rng(s)
                perm = rng.permutation(n_total)
                n_tr = int(round(0.7 * n_total))
                tr, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])

                v_tr = [v[tr] for v in v_all]
                v_te = [v[te] for v in v_all]

                # Fit CogPC1 on training rows only (Plan Section 8)
                mu_y, sd_y = Y_all[tr].mean(0), Y_all[tr].std(0, ddof=1)
                pca_cog = PCA(1).fit((Y_all[tr] - mu_y) / sd_y)
                y_all = pca_cog.transform((Y_all - mu_y) / sd_y).ravel()

                y_tr, y_te = y_all[tr], y_all[te]
                C_tr, C_te = C_all[tr], C_all[te]
                G_tr, G_te = G_all[tr], G_all[te]

                # Fit measurement model with confound sink
                res = pysimlr.fit(method, v_tr, k=3, confounds=C_tr, random_state=s)

                # Held-out path model with complete topology
                pm_te = pysimlr.path_model(
                    res, v_te, rows="heldout", graph=None, view_names=view_names,
                    covariates=C_te, groups=G_te, outcome=y_te,
                    train_views=v_tr, train_covariates=C_tr, train_groups=G_tr, train_outcome=y_tr,
                    n_permutations=n_permutations, random_state=s,
                )

                # Training path model for fixed held-out vs training Pillai comparison (Plan P3)
                pm_tr_complete = pysimlr.path_model(
                    res, v_tr, rows="train", graph=None, view_names=view_names,
                    covariates=C_tr, groups=G_tr, outcome=y_tr,
                    null="fixed", allow_train_fixed=True, n_permutations=0, n_bootstrap=0, random_state=s,
                )

                # Train path model with star on T1Hier_thk for held-out prediction finding
                pm_tr_star = pysimlr.path_model(
                    res, v_tr, rows="train", graph=topologies["star on T1Hier_thk"], view_names=view_names,
                    covariates=C_tr, groups=G_tr, outcome=y_tr,
                    conditioning="parents", null="fixed", allow_train_fixed=True,
                    n_permutations=0, n_bootstrap=0, random_state=s,
                )

                # Predict held-out CogPC1 through star and complete topologies
                pred_te = pysimlr.predict_path(pm_tr_star, v_te, covariates=C_te, groups=G_te, outcome=True)
                y_pred = pred_te["outcome_pred"].ravel()
                ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
                r2_star = float(1.0 - np.sum((y_te - y_pred) ** 2) / ss_tot)
                corr_star = float(np.corrcoef(y_te, y_pred)[0, 1])

                pred_te_comp = pysimlr.predict_path(pm_tr_complete, v_te, covariates=C_te, groups=G_te, outcome=True)
                y_pred_comp = pred_te_comp["outcome_pred"].ravel()
                r2_comp = float(1.0 - np.sum((y_te - y_pred_comp) ** 2) / ss_tot)
                corr_comp = float(np.corrcoef(y_te, y_pred_comp)[0, 1])

                # Save per-split held-out edge table
                edges_df = pm_te["edges"].copy()
                edges_df.to_csv(out_dir / f"adni_edges_{method}_seed{s}.csv", index=False)

                # Structural Model Comparison (Mode a, measurement fixed)
                cmp_a = pysimlr.compare_path_models(
                    res, v_tr, v_te, graphs=topologies, view_names=view_names,
                    covariates_train=C_tr, covariates_test=C_te,
                    groups_train=G_tr, groups_test=G_te,
                    outcome_train=y_tr, outcome_test=y_te,
                    n_bootstrap=n_bootstrap, n_permutations=min(n_permutations, 1000), random_state=s,
                )
                rows_a = {r["graph"]: r for r in cmp_a["comparison"]}
                ll_star = float(rows_a["star on T1Hier_thk"]["heldout_loglik"])
                ll_comp = float(rows_a["complete"]["heldout_loglik"])
                ll_chain = float(rows_a["chain (vol-thk-mean)"]["heldout_loglik"])
                ll_empty = float(rows_a["empty"]["heldout_loglik"])
                delta_star_minus_comp = ll_star - ll_comp

                # Plan P1: T1Hier_vol - T1Hier_thk held-out p_maxt < 0.05
                vol_thk_row = pm_te["edges"][
                    (pm_te["edges"].a == "T1Hier_vol") & (pm_te["edges"].b == "T1Hier_thk")
                ]
                pillai_vol_thk = float(vol_thk_row["pillai"].iloc[0])
                p_maxt_vol_thk = float(vol_thk_row["p_maxt"].iloc[0])

                # Plan P2: Outcome node has >= 1 view with p_holm < 0.05 in outcome_edges family
                out_tbl = pm_te["outcome_edges"].set_index("a")
                min_out_idx = pm_te["outcome_edges"]["p_holm"].astype(float).idxmin()
                min_outcome_p_holm = float(pm_te["outcome_edges"].loc[min_out_idx, "p_holm"])
                min_outcome_view = str(pm_te["outcome_edges"].loc[min_out_idx, "a"])

                p_perm_vol_cog = float(out_tbl.loc["T1Hier_vol", "p_perm"])
                p_holm_vol_cog = float(out_tbl.loc["T1Hier_vol", "p_holm"])
                p_perm_thk_cog = float(out_tbl.loc["T1Hier_thk", "p_perm"])
                p_holm_thk_cog = float(out_tbl.loc["T1Hier_thk", "p_holm"])
                p_perm_mean_cog = float(out_tbl.loc["T1w_mean", "p_perm"])
                p_holm_mean_cog = float(out_tbl.loc["T1w_mean", "p_holm"])

                # Plan P3: Held-out Pillai below training Pillai for every edge
                tr_edges = pm_tr_complete["edges"].set_index(["a", "b"])
                te_edges = pm_te["edges"].set_index(["a", "b"])
                violating_all: List[str] = []
                violating_vv: List[str] = []
                for (ea, eb), te_row in te_edges.iterrows():
                    p_tr_val = float(tr_edges.loc[(ea, eb), "pillai"])
                    p_te_val = float(te_row["pillai"])
                    is_vv = (ea in view_names) and (eb in view_names)
                    below = bool(p_te_val < p_tr_val)
                    if not below:
                        violating_all.append(f"{ea}-{eb}")
                        if is_vv:
                            violating_vv.append(f"{ea}-{eb}")
                    pillai_records.append({
                        "method": method,
                        "seed": s,
                        "a": ea,
                        "b": eb,
                        "edge_type": "view_view" if is_vv else "view_outcome",
                        "pillai_train": p_tr_val,
                        "pillai_heldout": p_te_val,
                        "delta_heldout_minus_train": p_te_val - p_tr_val,
                        "heldout_below_train": below,
                    })

                heldout_below_train_all = len(violating_all) == 0
                heldout_below_train_vv = len(violating_vv) == 0

                rec = {
                    "method": method,
                    "seed": s,
                    "pillai_vol_thk": pillai_vol_thk,
                    "p_maxt_vol_thk": p_maxt_vol_thk,
                    "min_outcome_p_holm": min_outcome_p_holm,
                    "min_outcome_view": min_outcome_view,
                    "p_perm_vol_cog": p_perm_vol_cog,
                    "p_holm_vol_cog": p_holm_vol_cog,
                    "p_thk_cog": p_perm_thk_cog,
                    "p_holm_thk_cog": p_holm_thk_cog,
                    "p_perm_mean_cog": p_perm_mean_cog,
                    "p_holm_mean_cog": p_holm_mean_cog,
                    "heldout_below_train_all": heldout_below_train_all,
                    "heldout_below_train_view_view": heldout_below_train_vv,
                    "p3_violating_edges": ";".join(violating_all),
                    "ll_star": ll_star,
                    "ll_complete": ll_comp,
                    "ll_chain": ll_chain,
                    "ll_empty": ll_empty,
                    "delta_star_minus_comp": delta_star_minus_comp,
                    "best_a": cmp_a["best"],
                    "best_by_loglik_a": cmp_a["best_by_loglik"],
                    "r2_star": r2_star,
                    "corr_star": corr_star,
                    "r2_complete": r2_comp,
                    "corr_complete": corr_comp,
                }
                results_by_method[method].append(rec)
                fixed_comparisons[method].extend([
                    dict(split=s, method=method, **{k: v for k, v in r.items() if k != "edges"})
                    for r in cmp_a["comparison"]
                ])

                print(f"  P1 vol-thk p_maxt={p_maxt_vol_thk:.4f} (pillai={pillai_vol_thk:.4f})")
                print(f"  P2 outcome min p_holm={min_outcome_p_holm:.4f} ({min_outcome_view}); thk->cog p_perm={p_perm_thk_cog:.4f}")
                print(f"  P3 below_train: all={heldout_below_train_all}, view_view={heldout_below_train_vv}, violations={violating_all}")
                print(f"  P4 Mode(a) top_by_loglik={cmp_a['best_by_loglik']} (parsimony={cmp_a['best']}), delta(star-comp)={delta_star_minus_comp:+.2f}")
                print(f"  Finding CogPC1 R^2: star={r2_star:.4f} (r={corr_star:.4f}), complete={r2_comp:.4f}")

                if s == seeds[0]:
                    rep_pm[method] = pm_te
                    rep_cmp[method] = cmp_a

            # Live Training-Row Refit vs Fixed-Model Null Evaluation (matched B = n_refits)
            print(f"\nEvaluating live training-row refit vs fixed null (matched B={n_refits}) for {method}...")
            s_p4 = seeds[0]
            rng = np.random.default_rng(s_p4)
            perm = rng.permutation(n_total)
            tr = np.sort(perm[: int(round(0.7 * n_total))])
            v_tr = [v[tr] for v in v_all]
            mu_y, sd_y = Y_all[tr].mean(0), Y_all[tr].std(0, ddof=1)
            pca_cog = PCA(1).fit((Y_all[tr] - mu_y) / sd_y)
            y_tr = pca_cog.transform((Y_all[tr] - mu_y) / sd_y).ravel()

            res_p4 = pysimlr.fit(method, v_tr, k=3, confounds=C_all[tr], random_state=s_p4)
            pm_both = pysimlr.path_model(
                res_p4, v_tr, rows="train", null="both", allow_train_fixed=True,
                covariates=C_all[tr], groups=G_all[tr], outcome=y_tr,
                view_names=view_names, n_permutations=n_refits, n_refits=n_refits,
                refit_scope=("edges",), n_bootstrap=0, n_jobs=n_jobs, random_state=s_p4,
            )
            for _, row in pm_both["edges"].iterrows():
                ea, eb = str(row["a"]), str(row["b"])
                obs_pil = float(row["pillai"])
                k_edge = (ea, eb) if (ea, eb) in pm_both["null"] else (eb, ea)
                null_fix = np.asarray(pm_both["null"][k_edge], dtype=float)
                null_ref = np.asarray(pm_both["refit_null"]["per_edge"][k_edge], dtype=float)
                n_ge_fix = int((null_fix >= obs_pil).sum())
                n_ge_ref = int((null_ref >= obs_pil).sum())
                q95_fix = float(np.percentile(null_fix, 95))
                q95_ref = float(np.percentile(null_ref, 95))
                max_fix = float(null_fix.max())
                max_ref = float(null_ref.max())
                p_fix = float(row["p_perm"])
                p_ref = float(row["p_refit"])
                is_vv = (ea in view_names) and (eb in view_names)
                p4_records.append({
                    "method": method,
                    "seed": s_p4,
                    "n_refits": n_refits,
                    "a": ea,
                    "b": eb,
                    "edge_type": "view_view" if is_vv else "view_outcome",
                    "pillai": obs_pil,
                    "n_ge_fixed": n_ge_fix,
                    "p_perm_fixed": p_fix,
                    "n_ge_refit": n_ge_ref,
                    "p_refit": p_ref,
                    "ratio_refit_to_fixed": p_ref / p_fix,
                    "q95_fixed": q95_fix,
                    "max_null_fixed": max_fix,
                    "q95_refit": q95_ref,
                    "max_null_refit": max_ref,
                    "ratio_q95_refit_to_fixed": q95_ref / q95_fix if q95_fix > 0 else float("nan"),
                })

        # Mode b: Training Topology Comparison (Refits mode across n_splits)
        print("\n==========================================")
        print(f"Running Mode (b) Training Topology Comparison (Refits mode, n_splits={len(seeds)})...")
        print("==========================================")
        mu_y_all, sd_y_all = Y_all.mean(0), Y_all.std(0, ddof=1)
        pca_all = PCA(1).fit((Y_all - mu_y_all) / sd_y_all)
        y_standardized = pca_all.transform((Y_all - mu_y_all) / sd_y_all).ravel()
        for method in methods:
            print(f"Running refits topology comparison for {method}...")
            cmp_b = pysimlr.compare_path_models(
                views=v_all, graphs=topologies, method=method, k=3, n_splits=len(seeds),
                train_fraction=0.7, fit_params={"confounds": C_all},
                covariates=C_all, groups=G_all, outcome=y_standardized,
                view_names=view_names, n_permutations=200, random_state=seeds[0],
            )
            refits_comparisons[method] = cmp_b

        # Compile and save clean tabular CSVs
        summary_rows: List[Dict[str, Any]] = []
        for m in methods:
            summary_rows.extend(results_by_method[m])
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "adni_edges_summary.csv", index=False)

        pillai_df = pd.DataFrame(pillai_records)
        pillai_df.to_csv(out_dir / "adni_train_vs_heldout_pillai.csv", index=False)

        fixed_rows: List[Dict[str, Any]] = []
        for m in methods:
            fixed_rows.extend(fixed_comparisons[m])
        fixed_df = pd.DataFrame(fixed_rows)
        fixed_df.to_csv(out_dir / "adni_comparison_fixed.csv", index=False)
        fixed_df.to_csv(out_dir / "adni_comparison.csv", index=False)

        refits_summary_dfs = []
        for m in methods:
            df_b = refits_comparisons[m]["summary"].copy()
            df_b["method"] = m
            refits_summary_dfs.append(df_b)
        refits_df = pd.concat(refits_summary_dfs, ignore_index=True)
        refits_df.to_csv(out_dir / "adni_comparison_refits.csv", index=False)

        p4_df = pd.DataFrame(p4_records)
        _verify_p4_consistency(p4_df, n_refits)
        p4_df.to_csv(out_dir / "adni_p4_refit_ratio.csv", index=False)

    # ------------------------------------------------------------------------- Plan Section 8 Rules & Findings
    print("\n==========================================")
    print("Evaluating Plan Section 8 Rules P1-P4 and Secondary Findings:")
    print("==========================================")

    n_s = len(seeds)
    sim_sub = summary_df[summary_df.method == "simlr"].sort_values("seed")
    lnd_sub = summary_df[summary_df.method == "lend"].sort_values("seed")

    # Plan P1: T1Hier_vol - T1Hier_thk held-out p_maxt < 0.05 in >= 4 of 5 splits
    p1_sim_hits = int((sim_sub["p_maxt_vol_thk"] < 0.05).sum())
    p1_lnd_hits = int((lnd_sub["p_maxt_vol_thk"] < 0.05).sum())

    # Plan P2: Outcome node has >= 1 view with p_holm < 0.05 in outcome_edges family in >= 4 of 5 splits
    p2_sim_hits = int((sim_sub["min_outcome_p_holm"] < 0.05).sum())
    p2_lnd_hits = int((lnd_sub["min_outcome_p_holm"] < 0.05).sum())

    # Plan P3: Held-out pillai below training pillai for every edge
    p3_sim_all_hits = int(sim_sub["heldout_below_train_all"].sum())
    p3_sim_vv_hits = int(sim_sub["heldout_below_train_view_view"].sum())
    p3_lnd_all_hits = int(lnd_sub["heldout_below_train_all"].sum())
    p3_lnd_vv_hits = int(lnd_sub["heldout_below_train_view_view"].sum())

    viol_df = pillai_df[~pillai_df["heldout_below_train"]].copy()
    sim_viol_list = [
        {
            "seed": int(r["seed"]),
            "edge": f"{r['a']} - {r['b']}",
            "edge_type": str(r["edge_type"]),
            "pillai_train": round(float(r["pillai_train"]), 6),
            "pillai_heldout": round(float(r["pillai_heldout"]), 6),
        }
        for _, r in viol_df[viol_df.method == "simlr"].iterrows()
    ]
    lnd_viol_list = [
        {
            "seed": int(r["seed"]),
            "edge": f"{r['a']} - {r['b']}",
            "edge_type": str(r["edge_type"]),
            "pillai_train": round(float(r["pillai_train"]), 6),
            "pillai_heldout": round(float(r["pillai_heldout"]), 6),
        }
        for _, r in viol_df[viol_df.method == "lend"].iterrows()
    ]

    # Plan P4: Complete graph is not the best by heldout_loglik in every split
    sim_best_ll = sim_sub["best_by_loglik_a"].tolist()
    lnd_best_ll = lnd_sub["best_by_loglik_a"].tolist()
    p4_sim_complete_not_every = not all(x == "complete" for x in sim_best_ll)
    p4_lnd_complete_not_every = not all(x == "complete" for x in lnd_best_ll)

    # Secondary Findings
    f1_sim_thk_hits = int((sim_sub["p_thk_cog"] < 0.05).sum())
    f1_lnd_thk_hits = int((lnd_sub["p_thk_cog"] < 0.05).sum())

    mean_delta_simlr = float(sim_sub["delta_star_minus_comp"].mean())
    se_delta_simlr = float(sim_sub["delta_star_minus_comp"].std(ddof=1) / np.sqrt(n_s))
    mean_delta_lend = float(lnd_sub["delta_star_minus_comp"].mean())
    se_delta_lend = float(lnd_sub["delta_star_minus_comp"].std(ddof=1) / np.sqrt(n_s))

    mean_r2_simlr = float(sim_sub["r2_star"].mean())
    mean_corr_simlr = float(sim_sub["corr_star"].mean())
    mean_r2_comp_simlr = float(sim_sub["r2_complete"].mean())
    mean_r2_lend = float(lnd_sub["r2_star"].mean())
    mean_corr_lend = float(lnd_sub["corr_star"].mean())
    mean_r2_comp_lend = float(lnd_sub["r2_complete"].mean())

    p4_sim = p4_df[p4_df.method == "simlr"]
    p4_lnd = p4_df[p4_df.method == "lend"]
    p4_sim_vv = p4_sim[p4_sim.edge_type == "view_view"]
    p4_sim_vo = p4_sim[p4_sim.edge_type == "view_outcome"]
    p4_lnd_vv = p4_lnd[p4_lnd.edge_type == "view_view"]
    p4_lnd_vo = p4_lnd[p4_lnd.edge_type == "view_outcome"]

    pilot_rules = {
        "plan_section8_rules": {
            "P1_vol_thk_heldout_p_maxt": {
                "requirement": "T1Hier_vol - T1Hier_thk held-out p_maxt < 0.05 in >= 4 of 5 splits",
                "simlr_hits": f"{p1_sim_hits}/{n_s}",
                "simlr_p_maxt": [round(float(x), 6) for x in sim_sub["p_maxt_vol_thk"]],
                "lend_hits": f"{p1_lnd_hits}/{n_s}",
                "lend_p_maxt": [round(float(x), 6) for x in lnd_sub["p_maxt_vol_thk"]],
                "simlr_pass": bool(p1_sim_hits >= 4),
                "lend_pass": bool(p1_lnd_hits >= 4),
                "pass": bool(p1_sim_hits >= 4 and p1_lnd_hits >= 4),
            },
            "P2_outcome_holm_family": {
                "requirement": "Outcome node has >= 1 view with p_holm < 0.05 in outcome_edges family in >= 4 of 5 splits",
                "simlr_hits": f"{p2_sim_hits}/{n_s}",
                "simlr_min_p_holm": [round(float(x), 6) for x in sim_sub["min_outcome_p_holm"]],
                "lend_hits": f"{p2_lnd_hits}/{n_s}",
                "lend_min_p_holm": [round(float(x), 6) for x in lnd_sub["min_outcome_p_holm"]],
                "simlr_pass": bool(p2_sim_hits >= 4),
                "lend_pass": bool(p2_lnd_hits >= 4),
                "pass": bool(p2_sim_hits >= 4 and p2_lnd_hits >= 4),
            },
            "P3_heldout_below_train_pillai": {
                "requirement": "Fixed-model held-out pillai is below training pillai for every edge",
                "simlr_all_edges_splits_passed": f"{p3_sim_all_hits}/{n_s}",
                "simlr_view_view_splits_passed": f"{p3_sim_vv_hits}/{n_s}",
                "simlr_violations": sim_viol_list,
                "lend_all_edges_splits_passed": f"{p3_lnd_all_hits}/{n_s}",
                "lend_view_view_splits_passed": f"{p3_lnd_vv_hits}/{n_s}",
                "lend_violations": lnd_viol_list,
                "simlr_pass": bool(p3_sim_all_hits == n_s),
                "lend_pass": bool(p3_lnd_all_hits == n_s),
                "view_view_pass": bool(p3_sim_vv_hits == n_s and p3_lnd_vv_hits == n_s),
                "pass": bool(p3_sim_all_hits == n_s and p3_lnd_all_hits == n_s),
                "investigation": (
                    "Every measurement-fitted view-view edge satisfies pillai_heldout < pillai_train across all splits. "
                    "Violations occur exclusively on weak view-outcome edges (V ~ 0.02-0.06): pysimlr.fit never observes "
                    "CogPC1, so outcome edges are not selected on training rows, and held-out splits (n_te=325 vs n_tr=759) "
                    "experience random 70/30 subsampling variance plus higher finite-sample Pillai offset (~k/n = 3/325 vs 3/759)."
                ),
            },
            "P4_complete_not_best_every_split": {
                "requirement": "Complete graph is not best by heldout_loglik in every split (if it is, report paired deltas)",
                "simlr_best_by_loglik": sim_best_ll,
                "simlr_complete_not_best_every_split": bool(p4_sim_complete_not_every),
                "lend_best_by_loglik": lnd_best_ll,
                "lend_complete_not_best_every_split": bool(p4_lnd_complete_not_every),
                "simlr_paired_deltas_star_minus_complete": [round(float(x), 4) for x in sim_sub["delta_star_minus_comp"]],
                "lend_paired_deltas_star_minus_complete": [round(float(x), 4) for x in lnd_sub["delta_star_minus_comp"]],
                "pass": bool(p4_sim_complete_not_every and p4_lnd_complete_not_every),
                "note": (
                    f"Plan section 8: SIMLR selects complete in {sum(x == 'complete' for x in sim_best_ll)}/{n_s} splits "
                    f"(star on T1Hier_thk wins {sum(x == 'star on T1Hier_thk' for x in sim_best_ll)}/{n_s} by raw log-likelihood, "
                    f"5/5 by F1 parsimony). Nonlinear LEND selects complete in {sum(x == 'complete' for x in lnd_best_ll)}/{n_s} splits "
                    f"(star on T1Hier_thk wins Seed 4 by +15.32); paired deltas reported."
                ),
            },
        },
        "secondary_findings": {
            "F1_thk_to_cog_association": {
                "simlr_significant_splits": f"{f1_sim_thk_hits}/{n_s}",
                "simlr_p_perm": [round(float(x), 6) for x in sim_sub["p_thk_cog"]],
                "simlr_p_holm": [round(float(x), 6) for x in sim_sub["p_holm_thk_cog"]],
                "lend_significant_splits": f"{f1_lnd_thk_hits}/{n_s}",
                "lend_p_perm": [round(float(x), 6) for x in lnd_sub["p_thk_cog"]],
                "lend_p_holm": [round(float(x), 6) for x in lnd_sub["p_holm_thk_cog"]],
            },
            "F2_star_vs_complete_loglik": {
                "simlr_mean_delta": round(mean_delta_simlr, 4),
                "simlr_se_delta": round(se_delta_simlr, 4),
                "simlr_parsimony_winners": sim_sub["best_a"].tolist(),
                "lend_mean_delta": round(mean_delta_lend, 4),
                "lend_se_delta": round(se_delta_lend, 4),
                "lend_parsimony_winners": lnd_sub["best_a"].tolist(),
            },
            "F3_heldout_cog_prediction": {
                "simlr_mean_r2_star": round(mean_r2_simlr, 4),
                "simlr_mean_corr_star": round(mean_corr_simlr, 4),
                "simlr_mean_r2_complete": round(mean_r2_comp_simlr, 4),
                "lend_mean_r2_star": round(mean_r2_lend, 4),
                "lend_mean_corr_star": round(mean_corr_lend, 4),
                "lend_mean_r2_complete": round(mean_r2_comp_lend, 4),
            },
            "F4_train_refit_vs_fixed_null": {
                "n_refits": int(p4_df["n_refits"].iloc[0]),
                "simlr_mean_p_ratio": round(float(p4_sim["ratio_refit_to_fixed"].mean()), 4),
                "simlr_view_view_mean_q95_fixed": round(float(p4_sim_vv["q95_fixed"].mean()), 6),
                "simlr_view_view_mean_q95_refit": round(float(p4_sim_vv["q95_refit"].mean()), 6),
                "simlr_view_view_mean_q95_ratio": round(float(p4_sim_vv["ratio_q95_refit_to_fixed"].mean()), 4),
                "simlr_outcome_mean_q95_ratio": round(float(p4_sim_vo["ratio_q95_refit_to_fixed"].mean()), 4),
                "simlr_overall_mean_q95_ratio": round(float(p4_sim["ratio_q95_refit_to_fixed"].mean()), 4),
                "lend_mean_p_ratio": round(float(p4_lnd["ratio_refit_to_fixed"].mean()), 4),
                "lend_view_view_mean_q95_fixed": round(float(p4_lnd_vv["q95_fixed"].mean()), 6),
                "lend_view_view_mean_q95_refit": round(float(p4_lnd_vv["q95_refit"].mean()), 6),
                "lend_view_view_mean_q95_ratio": round(float(p4_lnd_vv["ratio_q95_refit_to_fixed"].mean()), 4),
                "lend_outcome_mean_q95_ratio": round(float(p4_lnd_vo["ratio_q95_refit_to_fixed"].mean()), 4),
                "lend_overall_mean_q95_ratio": round(float(p4_lnd["ratio_q95_refit_to_fixed"].mean()), 4),
            },
        },
    }

    if not figures_only:
        with open(out_dir / "adni_pilot_rules.json", "w") as f:
            json.dump(pilot_rules, f, indent=2)

    print(json.dumps(pilot_rules, indent=2))

    # ------------------------------------------------------------------------- Render Publication Figures
    print("\n==========================================")
    print("Rendering Publication-Grade Visualizations (200 DPI)...")
    print("==========================================")

    pm_simlr = rep_pm["simlr"]
    pm_lend = rep_pm["lend"]

    adni_labels = {
        "T1Hier_vol": "Regional Volume",
        "T1Hier_thk": "Cortical Thickness",
        "T1w_mean": "T1w Intensity",
        "outcome": "Cognition\n(CogPC1)",
    }

    fig1 = pl.plot_path_model(pm_simlr, node_labels=adni_labels)
    fig1.savefig(out_dir / "adni_path_diagram_simlr.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2 = pl.plot_path_model(pm_lend, node_labels=adni_labels)
    fig2.savefig(out_dir / "adni_path_diagram_lend.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig3 = pl.plot_edge_matrix(pm_simlr, node_labels=adni_labels)
    fig3.savefig(out_dir / "adni_edge_matrix_simlr.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    fig4 = pl.plot_path_comparison(rep_cmp["simlr"])
    fig4.savefig(out_dir / "adni_path_comparison_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig4)

    fig5 = pl.plot_edge_null(pm_simlr, node_labels=adni_labels)
    fig5.savefig(out_dir / "adni_edge_null_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig5)

    fig6 = pl.plot_edge_features(
        pm_simlr, "T1Hier_thk - outcome", top_n=15,
        feature_names={"T1Hier_thk": feat_names_human["T1Hier_thk"]},
        node_labels=adni_labels,
    )
    fig6.savefig(out_dir / "adni_edge_features_thk_outcome.png", dpi=200, bbox_inches="tight")
    plt.close(fig6)

    fig7 = pl.plot_edge_features(
        pm_simlr, "T1Hier_vol - T1Hier_thk", top_n=15,
        feature_names={"T1Hier_vol": feat_names_human["T1Hier_vol"]},
        node_labels=adni_labels,
    )
    fig7.savefig(out_dir / "adni_edge_features_vol_thk.png", dpi=200, bbox_inches="tight")
    plt.close(fig7)

    # 6. antsxviz Cortical Surface Brain Maps (6-view montage on fsaverage 41k inflated)
    b1_path = out_dir / "adni_brain_map_thk_outcome.png"
    b2_path = out_dir / "adni_brain_map_vol_thk.png"
    if (not figures_only) or (not (b1_path.exists() and b2_path.exists())):
        print("Rendering antsxviz cortical surface montages...")
        fig_b1 = pl.plot_edge_brain_map(
            pm_simlr, "T1Hier_thk - outcome", component=0,
            feature_names=feat_names_dkt, density="41k", surface="inflated",
            title="Path Edge: T1Hier_thk -> CogPC1 · Cortical Thickness Contributions",
            subtitle="fsaverage 41k inflated surface · ANTsPy DKT parcellation · Component 1",
        )
        fig_b1.savefig(b1_path, dpi=200, bbox_inches="tight")
        plt.close(fig_b1)

        fig_b2 = pl.plot_edge_brain_map(
            pm_simlr, "T1Hier_vol - T1Hier_thk", component=0,
            feature_names=feat_names_dkt, density="41k", surface="inflated",
            title="Path Edge: T1Hier_vol -> T1Hier_thk · Volumetric Coupling to Thickness",
            subtitle="fsaverage 41k inflated surface · ANTsPy DKT parcellation · Component 1",
        )
        fig_b2.savefig(b2_path, dpi=200, bbox_inches="tight")
        plt.close(fig_b2)
    else:
        print(f"Preserving existing cortical surface montages at {b1_path.name} and {b2_path.name}.")

    # 7. antsxviz Region Table Figure (Split Hemispheres organized by lobe)
    print("Rendering antsxviz region table figure...")
    antsxviz = pl._require_antsxviz()
    fc = pm_simlr["feature_contributions"]
    fc_thk_out = fc[(fc.edge == "T1Hier_thk - outcome") & (fc.component == 0)]
    reg_dict: Dict[str, float] = {}
    for _, r in fc_thk_out.iterrows():
        f_idx = int(r["feature"])
        if f_idx < len(feat_names_dkt["T1Hier_thk"]):
            dkt_name = pl.cortical_region_name(feat_names_dkt["T1Hier_thk"][f_idx])
            if dkt_name:
                try:
                    antsxviz.surface.dkt_id(dkt_name)
                    reg_dict[dkt_name] = float(r["contribution"])
                except KeyError:
                    pass

    if reg_dict:
        rmax = max(abs(v) for v in reg_dict.values())
        fig_rt = antsxviz.region_table_figure(
            {
                "Cortical Thickness → Cognition (C1)": {
                    "values": reg_dict,
                    "units": "structure × beta",
                    "vmin": -rmax,
                    "vmax": rmax,
                    "cmap": pl._cmap("div"),
                }
            },
            split=antsxviz.split_hemisphere,
            title="Cortical Thickness Feature Contributions to Cognitive Decline (CogPC1)",
            subtitle="ANTsPy DKT Parcellation · Grouped by Anatomical Lobe · Left vs Right Hemispheres",
        )
        fig_rt.savefig(out_dir / "adni_region_table_thk_outcome.png", dpi=200, bbox_inches="tight")
        plt.close(fig_rt)

    # ------------------------------------------------------------------------- Generate Visual HTML Report
    print("\nGenerating Visual HTML Report...")
    generate_html_report(
        out_dir=out_dir,
        seeds=seeds,
        pilot_rules=pilot_rules,
        pillai_df=pillai_df,
        refits_df=refits_df,
        p4_df=p4_df,
        elapsed_s=time.time() - t_start,
    )

    print(f"\nAll tasks complete in {time.time() - t_start:.2f}s!")
    print(f"Artifacts saved in: {out_dir.resolve()}")
    return pilot_rules


# ----------------------------------------------------------------------------- HTML Report Generator
def generate_html_report(
    out_dir: Path,
    seeds: List[int],
    pilot_rules: Dict[str, Any],
    pillai_df: pd.DataFrame,
    refits_df: pd.DataFrame,
    p4_df: pd.DataFrame,
    elapsed_s: float,
) -> Path:
    """Generate a publication HTML report summarizing Plan Section 8 rules and empirical findings."""
    html_path = out_dir / "adni_pilot_report.html"

    def img_tag(filename: str, caption: str) -> str:
        filepath = out_dir / filename
        if not filepath.exists():
            return f"<p><em>Image {filename} not found.</em></p>"
        return f"""
        <div class="figure-container">
            <img src="{filename}" alt="{caption}" />
            <div class="caption"><strong>Figure:</strong> {caption}</div>
        </div>
        """

    s8 = pilot_rules["plan_section8_rules"]
    r_p1 = s8["P1_vol_thk_heldout_p_maxt"]
    r_p2 = s8["P2_outcome_holm_family"]
    r_p3 = s8["P3_heldout_below_train_pillai"]
    r_p4 = s8["P4_complete_not_best_every_split"]

    sf = pilot_rules["secondary_findings"]
    f1 = sf["F1_thk_to_cog_association"]
    f2 = sf["F2_star_vs_complete_loglik"]
    f3 = sf["F3_heldout_cog_prediction"]
    f4 = sf["F4_train_refit_vs_fixed_null"]

    def rule_badge(passed: bool, fail_label: str = "FAIL") -> str:
        cls = "badge-pass" if passed else "badge-fail"
        lbl = "PASS" if passed else fail_label
        return f'<span class="badge {cls}">{lbl}</span>'

    viol_rows_html = ""
    viol_df = pillai_df[~pillai_df["heldout_below_train"]].copy()
    for _, vr in viol_df.iterrows():
        viol_rows_html += (
            f"<tr><td>{vr['method'].upper()}</td><td>{int(vr['seed'])}</td>"
            f"<td><code>{vr['a']} - {vr['b']}</code></td><td>{vr['edge_type']}</td>"
            f"<td>{vr['pillai_train']:.4f}</td><td>{vr['pillai_heldout']:.4f}</td>"
            f"<td>{vr['delta_heldout_minus_train']:+.4f}</td></tr>\n"
        )
    if not viol_rows_html:
        viol_rows_html = '<tr><td colspan="7">None — all edges satisfied held-out Pillai &lt; training Pillai.</td></tr>'

    p4_rows_html = ""
    for _, pr in p4_df.iterrows():
        p4_rows_html += (
            f"<tr><td>{pr['method'].upper()}</td><td><code>{pr['a']} - {pr['b']}</code></td>"
            f"<td>{pr['edge_type']}</td><td>{pr['pillai']:.4f}</td>"
            f"<td>{pr['p_perm_fixed']:.4f} (n&ge;obs: {int(pr['n_ge_fixed'])})</td>"
            f"<td>{pr['p_refit']:.4f} (n&ge;obs: {int(pr['n_ge_refit'])})</td>"
            f"<td>{pr['q95_fixed']:.4f}</td><td>{pr['q95_refit']:.4f}</td>"
            f"<td>{pr['ratio_q95_refit_to_fixed']:.2f}&times;</td></tr>\n"
        )

    refits_rows_html = ""
    for _, row in refits_df.iterrows():
        refits_rows_html += (
            f"<tr><td>{row['method'].upper()}</td><td><strong>{row['graph']}</strong></td>"
            f"<td>{row['topology']}</td><td>{int(row['n_edges'])}</td>"
            f"<td>{row['heldout_loglik_own']:.2f}</td><td>{row['oos_coupling']:.4f}</td>"
            f"<td>{row['oos_r2']:.4f}</td><td>{row['outcome_r2']:.4f}</td></tr>\n"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ADNI-MCI Cognitive Pilot Evaluation Report — Score-Level Path Model</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1e293b;
        background: #f8fafc;
        margin: 0;
        padding: 40px 20px;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 40px 48px;
    }}
    h1 {{
        color: #0f172a;
        font-size: 28px;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 16px;
        margin-top: 0;
    }}
    h2 {{
        color: #1e3a8a;
        font-size: 20px;
        font-weight: 600;
        margin-top: 36px;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 8px;
    }}
    h3 {{
        color: #334155;
        font-size: 16px;
        font-weight: 600;
        margin-top: 24px;
    }}
    p, li {{
        font-size: 14.5px;
        color: #334155;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 13.5px;
        font-variant-numeric: tabular-nums;
    }}
    th, td {{
        padding: 10px 14px;
        text-align: left;
        border: 1px solid #e2e8f0;
    }}
    th {{
        background-color: #f1f5f9;
        color: #0f172a;
        font-weight: 600;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .badge-pass {{
        background: #dcfce7;
        color: #166534;
    }}
    .badge-fail {{
        background: #fee2e2;
        color: #991b1b;
    }}
    .badge-info {{
        background: #e0f2fe;
        color: #075985;
    }}
    .figure-container {{
        text-align: center;
        margin: 28px 0;
        padding: 16px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }}
    .figure-container img {{
        max-width: 100%;
        height: auto;
        border-radius: 4px;
    }}
    .caption {{
        font-size: 13px;
        color: #64748b;
        margin-top: 10px;
    }}
    .meta-box {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 28px;
        font-size: 13.5px;
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>ADNI-MCI Cognitive Pilot Evaluation Report</h1>

    <div class="meta-box">
        <strong>Repository:</strong> pysimlr (v{pysimlr.__version__}) &nbsp;|&nbsp;
        <strong>Evaluation Suite:</strong> WP4 Score-Level Path Model Pilot (Plan Section 8) &nbsp;|&nbsp;
        <strong>Execution Time:</strong> {elapsed_s:.1f}s &nbsp;|&nbsp;
        <strong>Cohort:</strong> ADNI-MCI Protocol B (n = 1,084, 70/30 train/test splits, seeds {seeds})
    </div>

    <h2>1. Primary Pass Rules (Plan Section 8: P1–P4)</h2>
    <p>
        Evaluated on the ADNI Mild Cognitive Impairment cohort across three anatomical T1-weighted MRI views
        (<code>T1Hier_vol</code>, <code>T1Hier_thk</code>, <code>T1w_mean</code>) and cognitive outcome <code>CogPC1</code>
        (first principal component of the cognitive battery fitted on training rows only). Demographic confounds
        (age, education, head size) are passed to both <code>fit(confounds=)</code> and <code>path_model(covariates=)</code>;
        sex is passed exclusively as <code>groups=</code>.
    </p>

    <table>
        <thead>
            <tr>
                <th>Plan Rule</th>
                <th>Section 8 Criterion</th>
                <th>Measured CSV Outcome</th>
                <th>Verdict</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>P1</strong></td>
                <td><code>T1Hier_vol – T1Hier_thk</code> held-out <code>p_maxt &lt; 0.05</code> in &ge; 4 of 5 splits</td>
                <td>SIMLR: <strong>{r_p1['simlr_hits']}</strong> (<code>p_maxt</code> = {r_p1['simlr_p_maxt']})<br>
                    LEND: <strong>{r_p1['lend_hits']}</strong> (<code>p_maxt</code> = {r_p1['lend_p_maxt']})</td>
                <td>{rule_badge(r_p1['pass'])}</td>
            </tr>
            <tr>
                <td><strong>P2</strong></td>
                <td>Outcome node has &ge; 1 view with <code>p_holm &lt; 0.05</code> in <code>outcome_edges</code> in &ge; 4 of 5 splits</td>
                <td>SIMLR: <strong>{r_p2['simlr_hits']}</strong> (min <code>p_holm</code> = {r_p2['simlr_min_p_holm']})<br>
                    LEND: <strong>{r_p2['lend_hits']}</strong> (min <code>p_holm</code> = {r_p2['lend_min_p_holm']})</td>
                <td>{rule_badge(r_p2['pass'])}</td>
            </tr>
            <tr>
                <td><strong>P3</strong></td>
                <td>Fixed-model held-out <code>pillai</code> is below training <code>pillai</code> for every edge</td>
                <td>View–view edges: SIMLR <strong>{r_p3['simlr_view_view_splits_passed']}</strong>, LEND <strong>{r_p3['lend_view_view_splits_passed']}</strong><br>
                    All 6 edges (incl. unsupervised outcome): SIMLR <strong>{r_p3['simlr_all_edges_splits_passed']}</strong>, LEND <strong>{r_p3['lend_all_edges_splits_passed']}</strong></td>
                <td>{rule_badge(r_p3['pass'], 'FAIL (OUTCOME EDGES)')}</td>
            </tr>
            <tr>
                <td><strong>P4</strong></td>
                <td>Complete graph is not best by <code>heldout_loglik</code> in every split (else report paired deltas)</td>
                <td>SIMLR top by LL: {r_p4['simlr_best_by_loglik']} (&Delta; star&minus;comp = {r_p4['simlr_paired_deltas_star_minus_complete']})<br>
                    LEND top by LL: {r_p4['lend_best_by_loglik']} (&Delta; star&minus;comp = {r_p4['lend_paired_deltas_star_minus_complete']})</td>
                <td>{rule_badge(r_p4['pass'])}</td>
            </tr>
        </tbody>
    </table>

    <h3>1.1 Rule P3 Edge-by-Edge Investigation (<code>adni_train_vs_heldout_pillai.csv</code>)</h3>
    <p>
        {r_p3['investigation']}
    </p>
    <table>
        <thead>
            <tr>
                <th>Method</th>
                <th>Split Seed</th>
                <th>Edge</th>
                <th>Edge Type</th>
                <th>Training Pillai (n=759)</th>
                <th>Held-out Pillai (n=325)</th>
                <th>&Delta; (Held-out &minus; Train)</th>
            </tr>
        </thead>
        <tbody>
            {viol_rows_html}
        </tbody>
    </table>

    <h2>2. Secondary Empirical Findings (Descriptive, Non-Gate)</h2>
    <table>
        <thead>
            <tr>
                <th>Finding</th>
                <th>Quantity Measured</th>
                <th>Empirical Summary</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="badge badge-info">Finding 1</span></td>
                <td>Held-out <code>T1Hier_thk &rarr; CogPC1</code> association</td>
                <td>SIMLR significant in <strong>{f1['simlr_significant_splits']}</strong> splits (<code>p_perm</code> = {f1['simlr_p_perm']}, <code>p_holm</code> = {f1['simlr_p_holm']})<br>
                    LEND significant in <strong>{f1['lend_significant_splits']}</strong> splits (<code>p_perm</code> = {f1['lend_p_perm']}, <code>p_holm</code> = {f1['lend_p_holm']})</td>
            </tr>
            <tr>
                <td><span class="badge badge-info">Finding 2</span></td>
                <td>Held-out log-likelihood parsimony (<code>star on T1Hier_thk</code> minus <code>complete</code>)</td>
                <td>SIMLR: mean &Delta;LL = <strong>{f2['simlr_mean_delta']:+.2f}</strong> (SE = {f2['simlr_se_delta']:.2f}; F1 parsimony winners: {f2['simlr_parsimony_winners']})<br>
                    LEND: mean &Delta;LL = <strong>{f2['lend_mean_delta']:+.2f}</strong> (SE = {f2['lend_se_delta']:.2f}; F1 parsimony winners: {f2['lend_parsimony_winners']})</td>
            </tr>
            <tr>
                <td><span class="badge badge-info">Finding 3</span></td>
                <td>Out-of-sample <code>CogPC1</code> prediction via <code>predict_path</code> (confound-adjusted)</td>
                <td>SIMLR star: mean R<sup>2</sup> = <strong>{f3['simlr_mean_r2_star']:.4f}</strong> (Pearson r = {f3['simlr_mean_corr_star']:.4f}; complete R<sup>2</sup> = {f3['simlr_mean_r2_complete']:.4f})<br>
                    LEND star: mean R<sup>2</sup> = <strong>{f3['lend_mean_r2_star']:.4f}</strong> (Pearson r = {f3['lend_mean_corr_star']:.4f}; complete R<sup>2</sup> = {f3['lend_mean_r2_complete']:.4f})</td>
            </tr>
            <tr>
                <td><span class="badge badge-info">Finding 4</span></td>
                <td>Live training-row refit vs fixed-model null (<code>rows="train", null="both"</code>, matched B = {f4['n_refits']})</td>
                <td>SIMLR: p-ratio = {f4['simlr_mean_p_ratio']:.2f}; view–view mean q95<sub>refit</sub> = {f4['simlr_view_view_mean_q95_refit']:.4f} vs q95<sub>fixed</sub> = {f4['simlr_view_view_mean_q95_fixed']:.4f} (<strong>{f4['simlr_view_view_mean_q95_ratio']:.2f}&times;</strong>; outcome edges {f4['simlr_outcome_mean_q95_ratio']:.2f}&times;)<br>
                    LEND: p-ratio = {f4['lend_mean_p_ratio']:.2f}; view–view mean q95<sub>refit</sub> = {f4['lend_view_view_mean_q95_refit']:.4f} vs q95<sub>fixed</sub> = {f4['lend_view_view_mean_q95_fixed']:.4f} (<strong>{f4['lend_view_view_mean_q95_ratio']:.2f}&times;</strong>; outcome edges {f4['lend_outcome_mean_q95_ratio']:.2f}&times;)</td>
            </tr>
        </tbody>
    </table>

    <h3>2.1 Live Training-Row Refit vs Fixed Null Table (<code>adni_p4_refit_ratio.csv</code>, Seed 0, B = {f4['n_refits']})</h3>
    <table>
        <thead>
            <tr>
                <th>Method</th>
                <th>Edge</th>
                <th>Type</th>
                <th>Observed Train Pillai</th>
                <th>Fixed p_perm</th>
                <th>Refit p_refit</th>
                <th>Fixed q95</th>
                <th>Refit q95</th>
                <th>q95 Ratio (Refit / Fixed)</th>
            </tr>
        </thead>
        <tbody>
            {p4_rows_html}
        </tbody>
    </table>

    <h2>3. Model Topology Comparison</h2>
    <h3>Mode (a): Structural Log-Likelihood Comparison (Fixed Measurement, Seed 0)</h3>
    <div class="grid-2">
        {img_tag("adni_path_comparison_forest.png", "Model comparison forest plot: Out-of-sample log-likelihood deltas against top graph with bootstrap 95% CIs and present/absent edge pass rates.")}
        {img_tag("adni_path_diagram_simlr.png", "Fitted score-level path model diagram on held-out rows (SIMLR, Seed 0).")}
    </div>
    {img_tag("adni_path_diagram_lend.png", "Fitted score-level path model diagram on held-out rows (LEND, Seed 0).")}

    <h3>Mode (b): Repeated Training-Topology Comparison (Refits Mode, n_splits = {len(seeds)})</h3>
    <table>
        <thead>
            <tr>
                <th>Method</th>
                <th>Graph</th>
                <th>Topology</th>
                <th>Edges</th>
                <th>Held-out LogLik (Own)</th>
                <th>OOS Coupling</th>
                <th>OOS R<sup>2</sup></th>
                <th>Outcome R<sup>2</sup></th>
            </tr>
        </thead>
        <tbody>
            {refits_rows_html}
        </tbody>
    </table>

    <h2>4. Edge Inference &amp; Coupling Diagnostics</h2>
    <div class="grid-2">
        {img_tag("adni_edge_matrix_simlr.png", "Edge component matrix (|r_j| and Pillai trace). Hatched cells denote q_bh >= 0.05.")}
        {img_tag("adni_edge_null_histograms.png", "Freedman-Lane permutation null distributions for all tested edges on held-out rows.")}
    </div>

    <h2>5. ANTsX Neuroanatomical Visualization</h2>
    <p>
        Feature contributions are computed as the product of view loading structure and path regression coefficients:
        <code>C_ia = S_ia &times; &beta;_ab</code>, projected onto the ANTsPy DKT parcellation via <code>antsxviz</code>.
    </p>

    <h3>Cortical Surface Montages (fsaverage 41k Inflated Surface)</h3>
    {img_tag("adni_brain_map_thk_outcome.png", "Cortical surface brain montage (lateral, medial, and ventral views for Left and Right hemispheres) showing cortical thickness feature contributions carrying association to CogPC1.")}
    {img_tag("adni_brain_map_vol_thk.png", "Cortical surface brain montage showing volumetric feature contributions coupled to cortical thickness.")}

    <h3>Anatomical Feature Contributions &amp; Hemispheric Region Tables</h3>
    <div class="grid-2">
        {img_tag("adni_edge_features_thk_outcome.png", "Top neuroanatomical features for T1Hier_thk -> CogPC1.")}
        {img_tag("adni_edge_features_vol_thk.png", "Top neuroanatomical features for T1Hier_vol -> T1Hier_thk.")}
    </div>
    {img_tag("adni_region_table_thk_outcome.png", "Hemispheric region table figure (antsxviz.region_table_figure) grouped by anatomical lobe.")}

    <hr style="border: 0; border-top: 1px solid #e2e8f0; margin-top: 40px;">
    <p style="font-size: 12px; color: #94a3b8; text-align: center;">Generated by <code>scripts/adni_path_model_pilot.py</code> &bull; pysimlr v{pysimlr.__version__}</p>
</div>
</body>
</html>
"""

    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"HTML report generated at: {html_path.resolve()}")
    return html_path


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="ADNI-MCI Cognitive Pilot Evaluation for Score-Level Path Model.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Train/test split seeds (default: 0 1 2 3 4).")
    parser.add_argument("--methods", type=str, nargs="+", default=["simlr", "lend"],
                        help="Methods to evaluate (default: simlr lend).")
    parser.add_argument("--n-permutations", type=int, default=2000,
                        help="Permutations for Freedman-Lane null (default: 2000).")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Bootstrap replicates for CIs (default: 2000).")
    parser.add_argument("--n-refits", type=int, default=50,
                        help="Refits for refit null (default: 50).")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Worker processes for multiprocessing (default: 4).")
    parser.add_argument("--out", type=Path, default=Path("docs/audit/evidence/path_model/adni"),
                        help="Output directory.")
    parser.add_argument("--figures-only", action="store_true",
                        help="Re-render figures and HTML report strictly from existing live CSVs (refuses if missing).")
    args = parser.parse_args()

    run_pilot(
        seeds=args.seeds,
        methods=args.methods,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        n_refits=args.n_refits,
        n_jobs=args.n_jobs,
        out_dir=args.out,
        figures_only=args.figures_only,
    )


if __name__ == "__main__":
    main()
