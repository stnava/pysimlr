import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pysimlr import flow_simr_v
from pysimlr.utils import set_all_seeds, adjusted_rvcoef
from scipy.stats import t

def run_interaction_analysis():
    set_all_seeds(42)
    print("=================================================================")
    print("STEP 1: PERFORM PERFUSION IMPUTATION FOR sub-EAS111EAS111 FIRST")
    print("=================================================================")
    
    fpath = "../extern/ExpArt/data/expartdf_power_analysis.csv"
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Cannot find raw EXPART data at: {fpath}")
    
    df = pd.read_csv(fpath, low_memory=False)
    
    subject_id = "sub-EAS111EAS111"
    sub_row = df[df['subjectID'] == subject_id]
    other_rows = df[df['subjectID'] != subject_id]
    
    t1_cols = [c for c in df.columns if c.startswith('T1Hier') or c.startswith('T1w')]
    dti_cols = [c for c in df.columns if c.startswith('DTI')]
    rsf_cols = [c for c in df.columns if c.startswith('rsfMRI')]
    perf_cols = [c for c in df.columns if c.startswith('perf')]
    
    t1_cols_num = [c for c in t1_cols if df[c].dtype in [np.float64, np.int64]]
    dti_cols_num = [c for c in dti_cols if df[c].dtype in [np.float64, np.int64]]
    rsf_cols_num = [c for c in rsf_cols if df[c].dtype in [np.float64, np.int64]]
    perf_cols_num = [c for c in perf_cols if df[c].dtype in [np.float64, np.int64]]
    
    scaler_t1 = StandardScaler()
    scaler_dti = StandardScaler()
    scaler_rsf = StandardScaler()
    scaler_perf = StandardScaler()
    
    other_t1 = other_rows[t1_cols_num].fillna(other_rows[t1_cols_num].mean())
    other_dti = other_rows[dti_cols_num].fillna(other_rows[dti_cols_num].mean())
    other_rsf = other_rows[rsf_cols_num].fillna(other_rows[rsf_cols_num].mean())
    other_perf = other_rows[perf_cols_num].fillna(other_rows[perf_cols_num].mean())
    
    pca_t1 = PCA(n_components=9).fit(scaler_t1.fit_transform(other_t1))
    pca_dti = PCA(n_components=9).fit(scaler_dti.fit_transform(other_dti))
    pca_rsf = PCA(n_components=9).fit(scaler_rsf.fit_transform(other_rsf))
    pca_perf = PCA(n_components=9).fit(scaler_perf.fit_transform(other_perf))
    
    other_t1_pcs = pca_t1.transform(scaler_t1.transform(other_t1))
    other_dti_pcs = pca_dti.transform(scaler_dti.transform(other_dti))
    other_rsf_pcs = pca_rsf.transform(scaler_rsf.transform(other_rsf))
    other_perf_pcs = pca_perf.transform(scaler_perf.transform(other_perf))
    
    matrices_perf = [other_t1_pcs, other_dti_pcs, other_rsf_pcs, other_perf_pcs]
    scalers_perf = [StandardScaler() for _ in range(4)]
    scaled_perf_mats = [s.fit_transform(m) for s, m in zip(scalers_perf, matrices_perf)]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(device)
    
    res_perf_impute = flow_simr_v(
        scaled_perf_mats,
        k=2,
        epochs=1500,
        batch_size=128,
        positivity='positive',
        use_nsa=True,
        nsa_w=0.5,
        dynamic_weights=True,
        dynamic_weights_start=300,
        energy_type='regression',
        mixing_algorithm='newton',
        device=device,
        verbose=False,
        use_rank_mai=False
    )
    
    model_perf = res_perf_impute['model'].to(device_obj)
    
    sub_t1_raw = sub_row[t1_cols_num].fillna(other_rows[t1_cols_num].mean())
    sub_dti_raw = sub_row[dti_cols_num].fillna(other_rows[dti_cols_num].mean())
    sub_rsf_raw = sub_row[rsf_cols_num].fillna(other_rows[rsf_cols_num].mean())
    
    sub_t1_pcs = pca_t1.transform(scaler_t1.transform(sub_t1_raw))
    sub_dti_pcs = pca_dti.transform(scaler_dti.transform(sub_dti_raw))
    sub_rsf_pcs = pca_rsf.transform(scaler_rsf.transform(sub_rsf_raw))
    
    sub_t1_scaled = scalers_perf[0].transform(sub_t1_pcs)
    sub_dti_scaled = scalers_perf[1].transform(sub_dti_pcs)
    sub_rsf_scaled = scalers_perf[2].transform(sub_rsf_pcs)
    
    with torch.no_grad():
        z_t1 = torch.as_tensor(sub_t1_scaled).float().to(device_obj)
        u_t1 = model_perf.encoders[0](model_perf.linear_encoders[0](z_t1)).cpu().numpy()
        
        z_dti = torch.as_tensor(sub_dti_scaled).float().to(device_obj)
        u_dti = model_perf.encoders[1](model_perf.linear_encoders[1](z_dti)).cpu().numpy()
        
        z_rsf = torch.as_tensor(sub_rsf_scaled).float().to(device_obj)
        u_rsf = model_perf.encoders[2](model_perf.linear_encoders[2](z_rsf)).cpu().numpy()
        
    joint_u = res_perf_impute['latents']
    u_t1_cohort = joint_u[0].detach().cpu().numpy()[:, :2]
    u_dti_cohort = joint_u[1].detach().cpu().numpy()[:, :2]
    u_rsf_cohort = joint_u[2].detach().cpu().numpy()[:, :2]
    u_perf_cohort = joint_u[3].detach().cpu().numpy()[:, :2]
    
    other_u_cohort = np.hstack([u_t1_cohort, u_dti_cohort, u_rsf_cohort])
    sub_u_obs = np.hstack([u_t1, u_dti, u_rsf])
    
    mu_target = np.mean(u_perf_cohort, axis=0)
    mu_obs = np.mean(other_u_cohort, axis=0)
    
    cov_target_target = np.cov(u_perf_cohort, rowvar=False)
    cov_obs_obs = np.cov(other_u_cohort, rowvar=False)
    cov_target_obs = np.cov(u_perf_cohort, other_u_cohort, rowvar=False)[:2, 2:]
    
    inv_cov_obs = np.linalg.inv(cov_obs_obs + np.eye(6) * 1e-6)
    pred_u_perf = mu_target + cov_target_obs @ inv_cov_obs @ (sub_u_obs - mu_obs).T
    pred_u_perf = pred_u_perf.T
    
    with torch.no_grad():
        pred_u_perf_t = torch.as_tensor(pred_u_perf).float().to(device_obj)
        pred_flow_z = model_perf.flows[3].inverse(pred_u_perf_t)
        pred_perf_scaled = (pred_flow_z @ model_perf.linear_encoders[3].v.t()).cpu().numpy()
        
    imputed_perf_pcs = scalers_perf[3].inverse_transform(pred_perf_scaled)[0]
    
    cohort_t1_pcs = pca_t1.transform(scaler_t1.transform(df[t1_cols_num].fillna(other_rows[t1_cols_num].mean())))
    cohort_dti_pcs = pca_dti.transform(scaler_dti.transform(df[dti_cols_num].fillna(other_rows[dti_cols_num].mean())))
    cohort_rsf_pcs = pca_rsf.transform(scaler_rsf.transform(df[rsf_cols_num].fillna(other_rows[rsf_cols_num].mean())))
    
    cohort_perf_pcs = np.zeros((98, 9))
    target_idx = df[df['subjectID'] == subject_id].index[0]
    raw_perf_all = df[perf_cols_num].fillna(other_rows[perf_cols_num].mean())
    cohort_perf_pcs = pca_perf.transform(scaler_perf.transform(raw_perf_all))
    cohort_perf_pcs[target_idx] = imputed_perf_pcs
    
    mr_matrix = np.hstack([cohort_t1_pcs, cohort_dti_pcs, cohort_rsf_pcs, cohort_perf_pcs])
    exp_cols = ['X1_BEC', 'X2_BEC', 'X3_BEC', 'X4_BEC', 'X5_BEC']
    
    bl_matrix_raw = np.sqrt(df[exp_cols].fillna(df[exp_cols].mean()).values)
    mr_matrix_raw = mr_matrix
    
    df['Sex_num'] = (df['Sex'] == 'Male').astype(float)
    df['hink'] = df['hink'].fillna(0.0)
    cov_cols = ['hink', 'Subject_Age', 'Sex_num', 'BV', 'Highest_Edu']
    covariates = df[cov_cols].fillna(df[cov_cols].mean()).values
    
    # -------------------------------------------------------------
    # STEP 2: FIT FLOW-SIMLR-V EXPLORATION MODEL (GATED)
    # -------------------------------------------------------------
    print("\nFitting Flow-SiMLR-V Gated model to extract latent representations...")
    scaled_bl = StandardScaler().fit_transform(bl_matrix_raw)
    scaled_mr = StandardScaler().fit_transform(mr_matrix_raw)
    
    res = flow_simr_v(
        [scaled_bl, scaled_mr],
        k=2,
        epochs=1000,
        batch_size=128,
        positivity='positive',
        use_nsa=True,
        nsa_w=0.5,
        dynamic_weights=True,
        dynamic_weights_start=200,
        energy_type='regression',
        mixing_algorithm='newton',
        device=device,
        verbose=False,
        use_rank_mai=False
    )
    
    v_bl = res['v'][0].cpu().numpy()
    v_mr = res['v'][1].cpu().numpy()
    
    proj_bl = scaled_bl @ v_bl
    proj_mr = scaled_mr @ v_mr
    
    # Define exposure score and MRI score
    exposure_score = proj_bl[:, 0]
    mri_score = proj_mr[:, 0]
    
    # Sum of raw counts for thresholding
    total_bl = bl_matrix_raw.sum(axis=1)
    
    # -------------------------------------------------------------
    # STEP 3: RUN INTERACTION REGRESSION SWEEP (N=98 AT EVERY STEP)
    # -------------------------------------------------------------
    print("\nRunning Interaction Regression Sweep across 18 percentiles...")
    percentiles = np.round(np.linspace(10, 80, 15), 1)  # Restrict slightly to ensure both groups are large enough
    interaction_records = []
    
    for p in percentiles:
        raw_thresh = np.percentile(total_bl, p)
        
        # Binary indicator D: 1 if Exposed (> threshold), 0 if Control (<= threshold)
        D = (total_bl > raw_thresh).astype(float)
        
        # Interaction variable: Blast_score * D
        interaction_term = exposure_score * D
        
        # Fit OLS: MRI_score = beta_0 + beta_1 * Blast_score + beta_2 * D + beta_3 * Interaction + Covariates
        Z = np.hstack([
            np.ones((len(mri_score), 1)),
            exposure_score.reshape(-1, 1),
            D.reshape(-1, 1),
            interaction_term.reshape(-1, 1),
            covariates
        ])
        
        # Solve OLS
        beta = np.linalg.pinv(Z.T @ Z) @ Z.T @ mri_score
        resids = mri_score - Z @ beta
        
        df_error = len(mri_score) - Z.shape[1]
        s2 = np.sum(resids**2) / df_error
        Sigma = s2 * np.linalg.pinv(Z.T @ Z)
        
        # beta_1: Slope in Controls
        slope_ctrl = beta[1]
        se_ctrl = np.sqrt(Sigma[1, 1])
        t_ctrl = slope_ctrl / se_ctrl
        p_ctrl = 2 * (1 - t.cdf(abs(t_ctrl), df=df_error))
        
        # beta_3: Interaction term (Difference in slopes)
        slope_diff = beta[3]
        se_diff = np.sqrt(Sigma[3, 3])
        t_diff = slope_diff / se_diff
        p_diff = 2 * (1 - t.cdf(abs(t_diff), df=df_error))
        
        # beta_1 + beta_3: Slope in Exposed
        slope_exp = beta[1] + beta[3]
        se_exp = np.sqrt(Sigma[1, 1] + Sigma[3, 3] + 2 * Sigma[1, 3])
        t_exp = slope_exp / se_exp
        p_exp = 2 * (1 - t.cdf(abs(t_exp), df=df_error))
        
        n_ctrl = np.sum(D == 0)
        n_exp = np.sum(D == 1)
        
        interaction_records.append({
            "Percentile": f"{p}%",
            "Threshold_Count": raw_thresh,
            "N_Controls": n_ctrl,
            "N_Exposed": n_exp,
            "Slope_Controls": slope_ctrl,
            "P_Value_Controls": p_ctrl,
            "Slope_Exposed": slope_exp,
            "P_Value_Exposed": p_exp,
            "Slope_Difference": slope_diff,
            "P_Value_Difference": p_diff
        })
        
    df_int = pd.DataFrame(interaction_records)
    df_int.to_csv("expart_interaction_threshold_analysis.csv", index=False)
    print("Saved interaction segmented analysis results to: expart_interaction_threshold_analysis.csv")
    print("\n=== Interaction Regression Results (N=98 constantly) ===")
    print(df_int[["Percentile", "N_Controls", "N_Exposed", "Slope_Controls", "Slope_Exposed", "Slope_Difference", "P_Value_Difference"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # Plotting Slopes and Interaction P-value curves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Plot Slopes
    ax = axes[0]
    ax.plot(percentiles, df_int["Slope_Exposed"], marker='o', color='#d62728', lw=2.5, label="Slope in Exposed group (X > T)")
    ax.plot(percentiles, df_int["Slope_Controls"], marker='x', linestyle='--', color='#1f77b4', lw=2, label="Slope in Controls group (X <= T)")
    ax.set_title("Blast -> MRI Slope in Controls vs. Exposed (N=98)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("Linear Slope Coefficient")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot Interaction P-value
    ax = axes[1]
    ax.plot(percentiles, df_int["P_Value_Difference"], marker='s', color='purple', lw=2.5, label="P-value of Slope Difference (Interaction)")
    ax.axhline(0.05, color='black', linestyle=':', label="Significance Alpha = 0.05")
    ax.set_title("P-value of Slope Difference between Groups (N=98 constantly)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("P-value (Interaction term)")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/interaction_threshold_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Interaction Segmented analysis plots to: expart_report_figures/interaction_threshold_analysis.png")

if __name__ == "__main__":
    run_interaction_analysis()
