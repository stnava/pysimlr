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

def run_hinge_analysis():
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
    # Standardize covariates and raw views
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
    
    # Extract OLEE weights and project
    v_bl = res['v'][0].cpu().numpy()
    v_mr = res['v'][1].cpu().numpy()
    
    proj_bl = scaled_bl @ v_bl
    proj_mr = scaled_mr @ v_mr
    
    # Define exposure score as the first latent component of Blast
    exposure_score = proj_bl[:, 0]
    mri_score = proj_mr[:, 0]
    
    # Calculate sum of raw blast counts for threshold definitions
    total_bl = bl_matrix_raw.sum(axis=1)
    
    # -------------------------------------------------------------
    # STEP 3: RUN SEGMENTED HINGE REGRESSION SWEEP (N=98 AT EVERY STEP)
    # -------------------------------------------------------------
    print("\nRunning Segmented Hinge Regression Sweep across 18 percentiles...")
    percentiles = np.round(np.linspace(0, 85, 18), 1)
    hinge_records = []
    
    for p in percentiles:
        # Define threshold value on raw blast counts
        raw_thresh = np.percentile(total_bl, p)
        
        # Hinge variable: (Blast exposure score - threshold)_+
        # We find the exposure score corresponding to this percentile
        score_thresh = np.percentile(exposure_score, p)
        X_hinge = np.maximum(0, exposure_score - score_thresh)
        
        # Fit OLS: MRI_score = beta_0 + beta_1 * Blast_score + beta_2 * Hinge_score + Covariates
        # Design matrix: Z = [Intercept, Blast_score, Hinge_score, Covariates]
        Z = np.hstack([
            np.ones((len(mri_score), 1)),
            exposure_score.reshape(-1, 1),
            X_hinge.reshape(-1, 1),
            covariates
        ])
        
        # Solve OLS
        beta = np.linalg.pinv(Z.T @ Z) @ Z.T @ mri_score
        resids = mri_score - Z @ beta
        
        # Residual variance and covariance
        df_error = len(mri_score) - Z.shape[1]
        s2 = np.sum(resids**2) / df_error
        Sigma = s2 * np.linalg.pinv(Z.T @ Z)
        
        # Combined slope above threshold: beta_1 + beta_2
        slope_above = beta[1] + beta[2]
        se_above = np.sqrt(Sigma[1, 1] + Sigma[2, 2] + 2 * Sigma[1, 2])
        
        t_stat = slope_above / se_above
        p_val = 2 * (1 - t.cdf(abs(t_stat), df=df_error))
        
        hinge_records.append({
            "Percentile": f"{p}%",
            "Raw_Threshold": raw_thresh,
            "N": len(mri_score), # Retains full N=98
            "Slope_Below": beta[1],
            "Slope_Above": slope_above,
            "Hinge_Slope_Shift": beta[2],
            "SE_Above": se_above,
            "T_Statistic_Above": t_stat,
            "P_Value_Above": p_val
        })
        
    df_hinge = pd.DataFrame(hinge_records)
    df_hinge.to_csv("expart_hinge_threshold_analysis.csv", index=False)
    print("Saved hinge segmented analysis results to: expart_hinge_threshold_analysis.csv")
    print("\n=== Hinge Segmented Regression Results (N=98 constantly) ===")
    print(df_hinge[["Percentile", "Raw_Threshold", "Slope_Below", "Slope_Above", "T_Statistic_Above", "P_Value_Above"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # Plotting t-statistic and P-value curves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Plot Slope Above Threshold
    ax = axes[0]
    ax.plot(percentiles, df_hinge["Slope_Above"], marker='o', color='#1f77b4', lw=2.5, label="Slope Above Threshold")
    ax.plot(percentiles, df_hinge["Slope_Below"], marker='x', linestyle='--', color='gray', label="Baseline Slope Below Threshold")
    ax.set_title("Blast -> MRI Slope Above Threshold (N=98)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold of Exposure")
    ax.set_ylabel("Linear Slope Coefficient")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot P-value and significance threshold
    ax = axes[1]
    ax.plot(percentiles, df_hinge["P_Value_Above"], marker='s', color='#d62728', lw=2.5, label="P-value of Slope Above")
    ax.axhline(0.05, color='black', linestyle=':', label="Significance Alpha = 0.05")
    ax.set_title("P-value of Slope Above Threshold (N=98 constantly)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold of Exposure")
    ax.set_ylabel("P-value")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/hinge_threshold_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Hinge Segmented analysis plots to: expart_report_figures/hinge_threshold_analysis.png")

if __name__ == "__main__":
    run_hinge_analysis()
