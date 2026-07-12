import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pysimlr import flow_simr_v
from pysimlr.utils import set_all_seeds

def run_fused_ridge(target_var='npsy_TMT.B_TotalTime_Raw', lambda1=0.1, lambda2_vals=[0.0, 0.5, 5.0]):
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
        epochs=100,
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
    
    exp_cols = ['X1_BEC', 'X2_BEC', 'X3_BEC', 'X4_BEC', 'X5_BEC']
    bl_matrix_raw = np.sqrt(df[exp_cols].fillna(df[exp_cols].mean()).values)
    
    mri_matrix_raw = np.hstack([cohort_t1_pcs, cohort_dti_pcs, cohort_rsf_pcs, cohort_perf_pcs])
    y_target_raw = df[target_var].fillna(df[target_var].mean()).values
    
    df['Sex_num'] = (df['Sex'] == 'Male').astype(float)
    df['hink'] = df['hink'].fillna(0.0)
    cov_cols = ['hink', 'Subject_Age', 'Sex_num', 'BV', 'Highest_Edu']
    covariates = df[cov_cols].fillna(df[cov_cols].mean()).values
    
    def fit_covariate_models(m, covs):
        X = np.hstack([np.ones((m.shape[0], 1)), covs])
        beta = np.linalg.pinv(X.T @ X) @ X.T @ m
        preds = X @ beta
        resids = m - preds
        return resids, preds, beta
        
    mri_res, _, _ = fit_covariate_models(mri_matrix_raw, covariates)
    y_res, _, _ = fit_covariate_models(y_target_raw.reshape(-1, 1), covariates)
    y_res = y_res.flatten()
    
    total_bl = bl_matrix_raw.sum(axis=1)
    percentiles = np.round(np.linspace(10, 80, 15), 1)
    
    scaler_mri = StandardScaler()
    mri_scaled = scaler_mri.fit_transform(mri_res)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_res.reshape(-1, 1)).flatten()
    
    M = len(percentiles)
    D = mri_scaled.shape[1]
    
    results = {}
    
    for l2 in lambda2_vals:
        print(f"\nSolving Fused Ridge Regression with lambda1={lambda1}, lambda2={l2}...")
        A = np.zeros((M * D, M * D))
        b = np.zeros(M * D)
        
        for j in range(M):
            raw_thresh = np.percentile(total_bl, percentiles[j])
            exposed_mask = total_bl > raw_thresh
            
            Z_j = mri_scaled[exposed_mask]
            y_j = y_scaled[exposed_mask]
            N_j = Z_j.shape[0]
            
            diag_block = (Z_j.T @ Z_j) / N_j + lambda1 * np.eye(D)
            
            if j > 0:
                diag_block += l2 * np.eye(D)
            if j < M - 1:
                diag_block += l2 * np.eye(D)
                
            A[j*D : (j+1)*D, j*D : (j+1)*D] = diag_block
            
            if j > 0:
                A[j*D : (j+1)*D, (j-1)*D : j*D] = -l2 * np.eye(D)
            if j < M - 1:
                A[j*D : (j+1)*D, (j+1)*D : (j+2)*D] = -l2 * np.eye(D)
                
            b[j*D : (j+1)*D] = (Z_j.T @ y_j) / N_j
            
        w_sol = np.linalg.solve(A, b)
        w_reshaped = w_sol.reshape((M, D))
        results[l2] = w_reshaped
        
    fig, axes = plt.subplots(1, len(lambda2_vals), figsize=(18, 5.5), sharey=True)
    modalities = {
        "T1 (Volume)": slice(0, 9),
        "DTI (Diffusion)": slice(9, 18),
        "rsfMRI (Functional)": slice(18, 27),
        "Perfusion (Blood Flow)": slice(27, 36)
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    df_records = []
    
    for idx, l2 in enumerate(lambda2_vals):
        w_mat = results[l2]
        ax = axes[idx]
        
        for (mod_name, slc), col in zip(modalities.items(), colors):
            coef_norms = np.linalg.norm(w_mat[:, slc], axis=1)
            ax.plot(percentiles, coef_norms, marker='o', lw=2.5, color=col, label=mod_name)
            
            for j in range(M):
                df_records.append({
                    "Lambda2": l2,
                    "Percentile": f"{percentiles[j]}%",
                    "Threshold_Count": np.percentile(total_bl, percentiles[j]),
                    "Brain_Modality": mod_name,
                    "Coefficient_L2_Norm": coef_norms[j]
                })
                
        ax.set_title(f"Fused Ridge ($\lambda_2 = {l2}$)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Percentile Threshold defining Group")
        if idx == 0:
            ax.set_ylabel("Predictive Coefficient L2 Norm")
        ax.set_xticks(percentiles[::2])
        ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/fused_ridge_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Fused Ridge Regression plots to: expart_report_figures/fused_ridge_analysis.png")
    
    df_fused = pd.DataFrame(df_records)
    df_fused.to_csv("expart_fused_ridge_analysis.csv", index=False)
    print("Saved fused ridge analysis data to: expart_fused_ridge_analysis.csv")

if __name__ == "__main__":
    run_fused_ridge()
