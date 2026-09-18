#!/usr/bin/env python3
"""
Adversarial Statistical & Numerical Verification Script for Milestone MR-M1.
Recomputes all statistics from paper/results_cache/deep_ranking_benchmark.csv
from raw mathematical definitions and verifies every number in paper/03_experiments.qmd,
paper/04_discussion.qmd, and paper/appendix_reproducibility.qmd.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats

def main():
    csv_path = "paper/results_cache/deep_ranking_benchmark.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} with shape {df.shape}")
    print(f"Datasets ({df['Dataset'].nunique()}): {sorted(df['Dataset'].unique())}")
    print(f"Models ({df['Model'].nunique()}): {sorted(df['Model'].unique())}")
    print(f"Seeds ({df['Seed'].nunique()}): {sorted(df['Seed'].unique())}")

    # ==========================================
    # 1. PIVOT TABLE & REPLICATE INTEGRITY
    # ==========================================
    piv = df.pivot_table(index=["Dataset", "Seed"], columns="Model", values="Predictive_Metric")
    print(f"\nPivot table shape: {piv.shape} (Expect 70 rows x 7 columns)")
    assert piv.shape == (70, 7), f"Expected (70, 7), got {piv.shape}"

    # ==========================================
    # 2. INDEPENDENT FRIEDMAN & IMAN-DAVENPORT
    # ==========================================
    mat = piv.values # shape (70, 7)
    n_datasets, k_models = mat.shape

    # Rank each row (higher is better -> rank 1 is highest score)
    # rankdata with -row assigns 1 to largest value
    raw_ranks = np.zeros_like(mat)
    for i in range(n_datasets):
        raw_ranks[i] = stats.rankdata(-mat[i])

    mean_ranks = np.mean(raw_ranks, axis=0)
    model_ranks = dict(zip(piv.columns, mean_ranks))

    print("\n--- Independent Mean Ranks (from first principles) ---")
    for m, r in sorted(model_ranks.items(), key=lambda x: x[1]):
        print(f"  {m:20s}: {r:.6f} (rounded: {r:.4f}, 2dp: {r:.2f})")

    # Friedman chi2 from formula:
    # chi2_F = [12 * N / (k * (k + 1))] * sum(R_j^2) - 3 * N * (k + 1)
    ss_ranks = np.sum(mean_ranks ** 2)
    chi2_stat = (12.0 * n_datasets / (k_models * (k_models + 1))) * ss_ranks - 3.0 * n_datasets * (k_models + 1)
    df_chi2 = k_models - 1
    p_chi2 = float(stats.chi2.sf(chi2_stat, df=df_chi2))

    # Iman-Davenport F statistic from formula:
    # F_F = (N - 1) * chi2_F / (N * (k - 1) - chi2_F)
    # df1 = k - 1, df2 = (k - 1) * (N - 1)
    df1 = k_models - 1
    df2 = (k_models - 1) * (n_datasets - 1)
    f_stat = ((n_datasets - 1) * chi2_stat) / (n_datasets * (k_models - 1) - chi2_stat)
    p_f = float(stats.f.sf(f_stat, dfn=df1, dfd=df2))

    print(f"\n--- Friedman & Iman-Davenport Tests ---")
    print(f"Friedman chi2: {chi2_stat:.6f} (reported 69.39)")
    print(f"Friedman p-value: {p_chi2:.6e} (reported 5.44e-13)")
    print(f"Iman-Davenport F: {f_stat:.6f} (reported 13.66)")
    print(f"Iman-Davenport df1: {df1}, df2: {df2} (reported df1=6, df2=414)")
    print(f"Iman-Davenport p-value: {p_f:.6e} (reported 3.63e-14)")

    # ==========================================
    # 3. NEMENYI CRITICAL DIFFERENCE
    # ==========================================
    # CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
    # For alpha=0.05, k=7, df=inf:
    # studentized_range quantile divided by sqrt(2)
    q_alpha = stats.studentized_range.ppf(1.0 - 0.05, k_models, np.inf) / np.sqrt(2.0)
    cd_calc = q_alpha * np.sqrt((k_models * (k_models + 1)) / (6.0 * n_datasets))
    print(f"\n--- Nemenyi Critical Difference (alpha=0.05) ---")
    print(f"q_alpha: {q_alpha:.6f}")
    print(f"CD: {cd_calc:.6f} (reported 1.077)")

    # Check difference between Flow-SiMLR-V and next models
    flow_rank = model_ranks["Flow-SiMLR-V"]
    lend_rank = model_ranks["LEND"]
    lbfgs_rank = model_ranks["SiMLR-LBFGS"]
    simlr_rank = model_ranks["SiMLR"]
    ned_rank = model_ranks["NED"]
    nedpp_rank = model_ranks["NEDPP"]
    nsa_rank = model_ranks["NSAFlow-Turnkey"]

    delta_flow_lend = lend_rank - flow_rank
    print(f"Rank gap Flow-SiMLR-V to LEND/SiMLR-LBFGS: {delta_flow_lend:.6f} (reported 1.23 > CD 1.077)")

    # ==========================================
    # 4. PAIRWISE WILCOXON & PERMUTATION TESTS
    # ==========================================
    # Load pysimlr deep_ranking engine for comparison
    sys.path.insert(0, "src")
    from pysimlr.benchmarks.deep_ranking import pairwise_wilcoxon_holm

    pairwise_res = pairwise_wilcoxon_holm(df, metric_col="Predictive_Metric")
    print("\n--- Pairwise Tests (Across all 70 blocks) ---")
    print(pairwise_res[["Model_A", "Model_B", "Mean_Diff", "P_Wilcoxon_Raw", "P_Wilcoxon_Holm", "P_Permutation", "Significant"]].to_string())

    # ==========================================
    # 5. OVERALL SUMMARY METRICS PER ARCHITECTURE
    # ==========================================
    print("\n--- Summary Metrics per Architecture Across all 70 Tasks ---")
    summary = df.groupby("Model").agg({
        "Predictive_Metric": ["mean", "std", "max"],
        "Strictly_Linear_Metric": ["mean", "max"],
        "Axis_Sensitive_Metric": ["mean", "max"],
        "CMC_Latent_U": ["mean", "max"],
        "Feature_Recovery_V": ["mean", "max"],
        "Frame_Defect_D": ["mean", "max"],
        "Lobe_Crosstalk": ["mean", "max"],
        "Sparsity_Ratio": "mean",
        "Fit_Seconds": ["mean", "std", "sum"]
    })
    print(summary.to_string())

    # ==========================================
    # 6. REGIME-SPECIFIC CMC & PREDICTIVE CHECKS
    # ==========================================
    print("\n--- CMC Latent U by Regime ---")
    cmc_piv = df.groupby(["Dataset", "Model"])["CMC_Latent_U"].mean().unstack()
    print(cmc_piv.to_string())

    print("\n--- Predictive Metric by Regime ---")
    pred_piv = df.groupby(["Dataset", "Model"])["Predictive_Metric"].mean().unstack()
    print(pred_piv.to_string())

    print("\n--- Frame Defect D by Regime ---")
    fd_piv = df.groupby(["Dataset", "Model"])["Frame_Defect_D"].mean().unstack()
    print(fd_piv.to_string())

    print("\n--- Lobe Crosstalk by Regime ---")
    lc_piv = df.groupby(["Dataset", "Model"])["Lobe_Crosstalk"].mean().unstack()
    print(lc_piv.to_string())

    # ==========================================
    # 7. EXPLICIT TEST OF CLAIMS IN 03_EXPERIMENTS & 04_DISCUSSION
    # ==========================================
    print("\n=======================================================")
    print("VERIFICATION OF SPECIFIC NUMERICAL CLAIMS IN MANUSCRIPT")
    print("=======================================================")

    claims = [
        ("Friedman chi2", 69.39, chi2_stat, 1e-2),
        ("Friedman p-value", 5.44e-13, p_chi2, 1e-14),
        ("Iman-Davenport F", 13.66, f_stat, 1e-2),
        ("Iman-Davenport p-value", 3.63e-14, p_f, 1e-15),
        ("Nemenyi CD", 1.077, cd_calc, 1e-3),
        ("Flow-SiMLR-V mean rank", 2.4714, model_ranks["Flow-SiMLR-V"], 1e-3),
        ("SiMLR-LBFGS mean rank", 3.7000, model_ranks["SiMLR-LBFGS"], 1e-3),
        ("LEND mean rank", 3.7000, model_ranks["LEND"], 1e-3),
        ("SiMLR mean rank", 3.7143, model_ranks["SiMLR"], 1e-3),
        ("NED mean rank", 4.6786, model_ranks["NED"], 1e-3),
        ("NEDPP mean rank", 4.6929, model_ranks["NEDPP"], 1e-3),
        ("NSAFlow-Turnkey mean rank", 5.0429, model_ranks["NSAFlow-Turnkey"], 1e-3),
    ]

    for name, reported, computed, tol in claims:
        err = abs(reported - computed)
        status = "PASS" if err <= tol else "FAIL"
        print(f"[{status}] {name:30s}: reported={reported}, computed={computed:.6f}, abs_err={err:.6e}")

    # Check discussion claims:
    # Flow-SiMLR-V predictive: overall 0.6312, Multi-Omics 0.9055, Heart 0.7722
    flow_overall_pred = df[df["Model"] == "Flow-SiMLR-V"]["Predictive_Metric"].mean()
    flow_mo_pred = df[(df["Model"] == "Flow-SiMLR-V") & (df["Dataset"] == "MultiOmics")]["Predictive_Metric"].mean()
    flow_heart_pred = df[(df["Model"] == "Flow-SiMLR-V") & (df["Dataset"] == "Heart")]["Predictive_Metric"].mean()
    flow_diab_pred = df[(df["Model"] == "Flow-SiMLR-V") & (df["Dataset"] == "Diabetes")]["Predictive_Metric"].mean()

    print(f"\nFlow-SiMLR-V Predictive claims:")
    print(f"  Overall: reported=0.6312, computed={flow_overall_pred:.4f}")
    print(f"  MultiOmics: reported=0.9055, computed={flow_mo_pred:.4f}")
    print(f"  Heart: reported=0.7722, computed={flow_heart_pred:.4f}")

    # Check NSAFlow-Turnkey MultiOmics CMC 0.9880, Linear CMC 0.8484
    nsa_mo_cmc = df[(df["Model"] == "NSAFlow-Turnkey") & (df["Dataset"] == "MultiOmics")]["CMC_Latent_U"].mean()
    nsa_lin_cmc = df[(df["Model"] == "NSAFlow-Turnkey") & (df["Dataset"] == "Linear")]["CMC_Latent_U"].mean()
    flow_mo_cmc = df[(df["Model"] == "Flow-SiMLR-V") & (df["Dataset"] == "MultiOmics")]["CMC_Latent_U"].mean()
    lbfgs_mo_cmc = df[(df["Model"] == "SiMLR-LBFGS") & (df["Dataset"] == "MultiOmics")]["CMC_Latent_U"].mean()
    simlr_mo_cmc = df[(df["Model"] == "SiMLR") & (df["Dataset"] == "MultiOmics")]["CMC_Latent_U"].mean()
    flow_sp_cmc = df[(df["Model"] == "Flow-SiMLR-V") & (df["Dataset"] == "Shared+Private")]["CMC_Latent_U"].mean()

    print(f"\nCMC Latent Recovery claims:")
    print(f"  NSAFlow Multi-Omics CMC: reported=0.9880, computed={nsa_mo_cmc:.4f}")
    print(f"  Flow-SiMLR-V Multi-Omics CMC: reported=0.9274, computed={flow_mo_cmc:.4f}")
    print(f"  SiMLR-LBFGS Multi-Omics CMC: reported=0.8933, computed={lbfgs_mo_cmc:.4f}")
    print(f"  SiMLR Multi-Omics CMC: reported=0.7823, computed={simlr_mo_cmc:.4f}")
    print(f"  NSAFlow Linear CMC: reported=0.8484, computed={nsa_lin_cmc:.4f}")
    print(f"  Flow-SiMLR-V Shared+Private CMC: reported=0.7137, computed={flow_sp_cmc:.4f}")

    # Check Frame defect claims:
    # SiMLR-LBFGS Diabetes D = 0.0000, Heart D = 6.56e-11
    # SiMLR Heart D = 3.61e-8
    # Flow-SiMLR-V Heart D = 0.0145
    # LEND Heart D = 0.0010
    # NED Heart D = 0.0009
    # NEDPP Heart D = 0.0009
    lbfgs_diab_d = df[(df["Model"] == "SiMLR-LBFGS") & (df["Dataset"] == "Diabetes")]["Frame_Defect_D"].mean()
    lbfgs_heart_d = df[(df["Model"] == "SiMLR-LBFGS") & (df["Dataset"] == "Heart")]["Frame_Defect_D"].mean()
    simlr_heart_d = df[(df["Model"] == "SiMLR") & (df["Dataset"] == "Heart")]["Frame_Defect_D"].mean()
    flow_heart_d = df[(df["Model"] == "Flow-SiMLR-V") & (df["Dataset"] == "Heart")]["Frame_Defect_D"].mean()
    lend_heart_d = df[(df["Model"] == "LEND") & (df["Dataset"] == "Heart")]["Frame_Defect_D"].mean()
    ned_heart_d = df[(df["Model"] == "NED") & (df["Dataset"] == "Heart")]["Frame_Defect_D"].mean()
    nedpp_heart_d = df[(df["Model"] == "NEDPP") & (df["Dataset"] == "Heart")]["Frame_Defect_D"].mean()

    print(f"\nFrame Defect D claims:")
    print(f"  SiMLR-LBFGS Diabetes D: reported=0.0000, computed={lbfgs_diab_d:.6e}")
    print(f"  SiMLR-LBFGS Heart D: reported=6.56e-11, computed={lbfgs_heart_d:.6e}")
    print(f"  SiMLR Heart D: reported=3.61e-8, computed={simlr_heart_d:.6e}")
    print(f"  Flow-SiMLR-V Heart D: reported=0.0145, computed={flow_heart_d:.4f}")
    print(f"  LEND Heart D: reported=0.0010, computed={lend_heart_d:.4f}")
    print(f"  NED Heart D: reported=0.0009, computed={ned_heart_d:.4f}")
    print(f"  NEDPP Heart D: reported=0.0009, computed={nedpp_heart_d:.4f}")

    # Check Fit times:
    simlr_time = df[df["Model"] == "SiMLR"]["Fit_Seconds"].mean()
    lbfgs_time = df[df["Model"] == "SiMLR-LBFGS"]["Fit_Seconds"].mean()
    flow_time = df[df["Model"] == "Flow-SiMLR-V"]["Fit_Seconds"].mean()
    nsa_time = df[df["Model"] == "NSAFlow-Turnkey"]["Fit_Seconds"].mean()
    lend_time = df[df["Model"] == "LEND"]["Fit_Seconds"].mean()
    ned_time = df[df["Model"] == "NED"]["Fit_Seconds"].mean()
    nedpp_time = df[df["Model"] == "NEDPP"]["Fit_Seconds"].mean()

    print(f"\nFit Time claims:")
    print(f"  SiMLR time: reported=0.11 s (107 ms), computed={simlr_time:.4f} s")
    print(f"  SiMLR-LBFGS time: reported=1.50 s, computed={lbfgs_time:.4f} s")
    print(f"  Flow-SiMLR-V time: reported=4.13 s, computed={flow_time:.4f} s")
    print(f"  NSAFlow-Turnkey time: reported=4.64 s, computed={nsa_time:.4f} s")
    print(f"  LEND time: reported=1.65 s, computed={lend_time:.4f} s")
    print(f"  NED time: reported=2.05 s, computed={ned_time:.4f} s")
    print(f"  NEDPP time: reported=2.37 s, computed={nedpp_time:.4f} s")

    # Check Lobe crosstalk:
    print(f"\nLobe Crosstalk across all 490 runs:")
    max_crosstalk = df["Lobe_Crosstalk"].max()
    print(f"  Max Lobe Crosstalk: {max_crosstalk} (strictly zero: {max_crosstalk == 0.0})")

    # ==========================================
    # 8. PERMUTATION RESOLUTION FLOOR INVESTIGATION
    # ==========================================
    print("\n=======================================================")
    print("INVESTIGATION OF PERMUTATION RESOLUTION FLOOR CLAIMS")
    print("=======================================================")
    # Notice in 04_discussion.qmd line 15:
    # "and exact sign-flip permutation p <= 0.00007"
    # And line 19:
    # "Inference is conducted at the level of the independent replicate (N_seeds = 10).
    # The exact sign-flip permutation resolution floor is 2 / 2^10 ~= 0.00195.
    # All reported p-values respect this combinatorial resolution floor."
    # Let's check what P_Permutation was computed across the 70 tasks vs per dataset (10 seeds).

    for ma in ["Flow-SiMLR-V"]:
        for mb in ["LEND", "NED", "NEDPP", "SiMLR", "SiMLR-LBFGS", "NSAFlow-Turnkey"]:
            sub = pairwise_res[((pairwise_res["Model_A"] == ma) & (pairwise_res["Model_B"] == mb)) |
                               ((pairwise_res["Model_A"] == mb) & (pairwise_res["Model_B"] == ma))]
            if len(sub) > 0:
                row = sub.iloc[0]
                p_p = row["P_Permutation"]
                print(f"  {ma} vs {mb}: Wilcoxon-Holm p={row['P_Wilcoxon_Holm']:.4e}, Permutation p (70 blocks)={p_p:.6f}")

    # Now let's calculate exact permutation tests within each dataset (N=10 seeds):
    print("\nWithin each dataset (N=10 seeds), pairwise Flow-SiMLR-V vs others:")
    for ds in sorted(df["Dataset"].unique()):
        sub_ds = df[df["Dataset"] == ds].pivot_table(index="Seed", columns="Model", values="Predictive_Metric")
        for mb in ["LEND", "NED", "NEDPP", "SiMLR", "SiMLR-LBFGS", "NSAFlow-Turnkey"]:
            diffs = sub_ds["Flow-SiMLR-V"].values - sub_ds[mb].values
            from pysimlr.benchmarks.deep_ranking import exact_sign_flip_permutation
            p_perm_10 = exact_sign_flip_permutation(diffs)
            print(f"    [{ds:15s}] Flow-SiMLR-V vs {mb:16s}: diff={diffs.mean():+.4f}, p_perm={p_perm_10:.6f} (min possible 2/1024 = 0.001953)")

if __name__ == "__main__":
    main()
