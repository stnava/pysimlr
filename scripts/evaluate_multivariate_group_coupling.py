import torch
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pysimlr import flow_simr_v
from pysimlr.utils import set_all_seeds, adjusted_rvcoef

def run_multivariate_group_coupling(n_perms=1000):
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
    
    # Impute perfusion PCs first (retaining cohort logic)
    res_perf_impute = flow_simr_v(
        scaled_perf_mats,
        k=2,
        epochs=100, # fast for helper script
        batch_size=128,
        positivity='positive',
        use_nsa=True,
        nsa_w=0.5,
        dynamic_weights=True,
        dynamic_weights_start=30,
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
    
    # Freedman-Lane Residualization on covariates
    def fit_covariate_models(m, covs):
        X = np.hstack([np.ones((m.shape[0], 1)), covs])
        beta = np.linalg.pinv(X.T @ X) @ X.T @ m
        preds = X @ beta
        resids = m - preds
        return resids, preds, beta
        
    bl_res, bl_pred, _ = fit_covariate_models(bl_matrix_raw, covariates)
    mr_res, mr_pred, _ = fit_covariate_models(mr_matrix_raw, covariates)
    
    # Calculate sum of raw blast counts for threshold definitions
    total_bl = bl_matrix_raw.sum(axis=1)
    
    # -------------------------------------------------------------
    # MULTIVARIATE GROUP COUPLING SWEEP
    # -------------------------------------------------------------
    print("\nRunning Multivariate Group Coupling Sweep across 18 percentiles...")
    percentiles = np.round(np.linspace(10, 80, 15), 1)  # Restrict to ensure stable group sizes
    coupling_records = []
    rng = np.random.default_rng(42)
    
    # Convert residual matrices to PyTorch tensors for adjusted_rvcoef
    bl_res_t = torch.as_tensor(bl_res).float()
    mr_res_t = torch.as_tensor(mr_res).float()
    
    for p in percentiles:
        raw_thresh = np.percentile(total_bl, p)
        
        # Define binary group labels
        exposed_mask = total_bl > raw_thresh
        ctrl_mask = total_bl <= raw_thresh
        
        n_ctrl = np.sum(ctrl_mask)
        n_exp = np.sum(exposed_mask)
        
        # Compute observed Adjusted RV for both groups
        rv_ctrl = adjusted_rvcoef(bl_res_t[ctrl_mask], mr_res_t[ctrl_mask])
        rv_exp = adjusted_rvcoef(bl_res_t[exposed_mask], mr_res_t[exposed_mask])
        delta_rv_obs = rv_exp - rv_ctrl
        
        # Permutation test on group labels
        perm_deltas = []
        for _ in range(n_perms):
            # Shuffle the binary exposed indicator
            perm_mask = rng.permutation(exposed_mask)
            perm_ctrl = ~perm_mask
            
            rv_ctrl_p = adjusted_rvcoef(bl_res_t[perm_ctrl], mr_res_t[perm_ctrl])
            rv_exp_p = adjusted_rvcoef(bl_res_t[perm_mask], mr_res_t[perm_mask])
            perm_deltas.append(rv_exp_p - rv_ctrl_p)
            
        perm_deltas = np.array(perm_deltas)
        p_val = (1.0 + np.sum(perm_deltas >= delta_rv_obs)) / (n_perms + 1.0)
        
        coupling_records.append({
            "Percentile": f"{p}%",
            "Threshold_Count": raw_thresh,
            "N_Controls": n_ctrl,
            "N_Exposed": n_exp,
            "RV_Controls": rv_ctrl,
            "RV_Exposed": rv_exp,
            "RV_Difference": delta_rv_obs,
            "Permutation_P_Value": p_val
        })
        
    df_coup = pd.DataFrame(coupling_records)
    df_coup.to_csv("expart_multivariate_group_coupling_analysis.csv", index=False)
    print("Saved multivariate group coupling analysis results to: expart_multivariate_group_coupling_analysis.csv")
    print("\n=== Multivariate Group Coupling Results (N=98 constantly) ===")
    print(df_coup[["Percentile", "N_Controls", "N_Exposed", "RV_Controls", "RV_Exposed", "RV_Difference", "Permutation_P_Value"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # Plotting Curves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Plot RV Coefficients
    ax = axes[0]
    ax.plot(percentiles, df_coup["RV_Exposed"], marker='o', color='#d62728', lw=2.5, label="Multivariate Coupling in Exposed (X > T)")
    ax.plot(percentiles, df_coup["RV_Controls"], marker='x', linestyle='--', color='#1f77b4', lw=2, label="Multivariate Coupling in Controls (X <= T)")
    ax.set_title("Blast <-> MRI Multivariate Coupling (Adjusted RV)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("Adjusted RV Coefficient")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot Permutation P-value
    ax = axes[1]
    ax.plot(percentiles, df_coup["Permutation_P_Value"], marker='s', color='darkgreen', lw=2.5, label="Permutation P-value of Coupling Difference")
    ax.axhline(0.05, color='black', linestyle=':', label="Significance Alpha = 0.05")
    ax.set_title("Significance of Coupling Difference between Groups", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("P-value")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/multivariate_group_coupling_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Multivariate Group Coupling plots to: expart_report_figures/multivariate_group_coupling_analysis.png")

if __name__ == "__main__":
    run_multivariate_group_coupling()
