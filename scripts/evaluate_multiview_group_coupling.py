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

def run_multiview_group_coupling(n_perms=1000):
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
    
    # Define variables and matrices for multiview
    exp_cols = ['X1_BEC', 'X2_BEC', 'X3_BEC', 'X4_BEC', 'X5_BEC']
    bl_matrix_raw = np.sqrt(df[exp_cols].fillna(df[exp_cols].mean()).values)
    
    # 4 distinct brain views
    t1_matrix = cohort_t1_pcs
    dti_matrix = cohort_dti_pcs
    rsf_matrix = cohort_rsf_pcs
    perf_matrix = cohort_perf_pcs
    
    # Combine brain views into complete MRI matrix
    mri_matrix_raw = np.hstack([t1_matrix, dti_matrix, rsf_matrix, perf_matrix])
    
    # Clinical and cognitive views
    npsy_mh_cols = ['npsy_BDI_Total', 'npsy_BSI.18_TotalRaw', 'npsy_IRI_Total', 'npsy_PCL5Total']
    npsy_pe_cols = ['npsy_CVLTShortDelayFreeRecall_Raw', 'npsy_CVLTLongDelayFreeRecall_Raw', 
                     'npsy_DKEFS_Total_Raw_Score', 'npsy_NIH_Fluid_Composite_Score', 
                     'npsy_TMT.A_TotalTime_Raw', 'npsy_TMT.B_TotalTime_Raw', 'npsy_WASIComposite']
    
    mh_matrix_raw = df[npsy_mh_cols].fillna(df[npsy_mh_cols].mean()).values
    pe_matrix_raw = df[npsy_pe_cols].fillna(df[npsy_pe_cols].mean()).values
    
    # Covariates
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
        
    bl_res, _, _ = fit_covariate_models(bl_matrix_raw, covariates)
    t1_res, _, _ = fit_covariate_models(t1_matrix, covariates)
    dti_res, _, _ = fit_covariate_models(dti_matrix, covariates)
    rsf_res, _, _ = fit_covariate_models(rsf_matrix, covariates)
    perf_res, _, _ = fit_covariate_models(perf_matrix, covariates)
    mri_res, _, _ = fit_covariate_models(mri_matrix_raw, covariates)
    mh_res, _, _ = fit_covariate_models(mh_matrix_raw, covariates)
    pe_res, _, _ = fit_covariate_models(pe_matrix_raw, covariates)
    
    # Calculate sum of raw blast counts for threshold definitions
    total_bl = bl_matrix_raw.sum(axis=1)
    
    # -------------------------------------------------------------
    # 1. VIEW-SPECIFIC BRAIN IMAGING COUPLING SWEEP
    # -------------------------------------------------------------
    print("\nRunning View-Specific Brain Imaging Coupling Sweep...")
    percentiles = np.round(np.linspace(10, 80, 15), 1)
    
    # Convert residual matrices to PyTorch tensors
    bl_t = torch.as_tensor(bl_res).float()
    t1_t = torch.as_tensor(t1_res).float()
    dti_t = torch.as_tensor(dti_res).float()
    rsf_t = torch.as_tensor(rsf_res).float()
    perf_t = torch.as_tensor(perf_res).float()
    mri_t = torch.as_tensor(mri_res).float()
    mh_t = torch.as_tensor(mh_res).float()
    pe_t = torch.as_tensor(pe_res).float()
    
    view_tensors = {
        "T1 (Volume)": t1_t,
        "DTI (Diffusion)": dti_t,
        "rsfMRI (Functional)": rsf_t,
        "Perfusion (Blood Flow)": perf_t
    }
    
    rng = np.random.default_rng(42)
    view_records = []
    
    for p in percentiles:
        raw_thresh = np.percentile(total_bl, p)
        exposed_mask = total_bl > raw_thresh
        ctrl_mask = total_bl <= raw_thresh
        
        n_ctrl = np.sum(ctrl_mask)
        n_exp = np.sum(exposed_mask)
        
        row_dict = {
            "Percentile": f"{p}%",
            "Threshold_Count": raw_thresh,
            "N_Controls": n_ctrl,
            "N_Exposed": n_exp
        }
        
        for view_name, view_t in view_tensors.items():
            rv_ctrl = adjusted_rvcoef(bl_t[ctrl_mask], view_t[ctrl_mask])
            rv_exp = adjusted_rvcoef(bl_t[exposed_mask], view_t[exposed_mask])
            delta_rv = rv_exp - rv_ctrl
            
            # Permutation test
            perm_deltas = []
            for _ in range(n_perms):
                perm_mask = rng.permutation(exposed_mask)
                perm_ctrl = ~perm_mask
                
                rv_ctrl_p = adjusted_rvcoef(bl_t[perm_ctrl], view_t[perm_ctrl])
                rv_exp_p = adjusted_rvcoef(bl_t[perm_mask], view_t[perm_mask])
                perm_deltas.append(rv_exp_p - rv_ctrl_p)
                
            perm_deltas = np.array(perm_deltas)
            p_val = (1.0 + np.sum(perm_deltas >= delta_rv)) / (n_perms + 1.0)
            
            row_dict[f"RV_{view_name}_Controls"] = rv_ctrl
            row_dict[f"RV_{view_name}_Exposed"] = rv_exp
            row_dict[f"RV_{view_name}_Difference"] = delta_rv
            row_dict[f"P_Value_{view_name}"] = p_val
            
        view_records.append(row_dict)
        
    df_views = pd.DataFrame(view_records)
    df_views.to_csv("expart_multiview_brain_coupling_analysis.csv", index=False)
    print("Saved view-specific brain coupling analysis to expart_multiview_brain_coupling_analysis.csv")
    
    # -------------------------------------------------------------
    # 2. CLINICAL CASCADE COUPLING SWEEP
    # -------------------------------------------------------------
    print("\nRunning Clinical-Cognitive Cascade Coupling Sweep...")
    cascade_pairs = {
        "Brain_vs_MH": (mri_t, mh_t),
        "MH_vs_PE": (mh_t, pe_t),
        "Brain_vs_PE": (mri_t, pe_t)
    }
    
    cascade_records = []
    
    for p in percentiles:
        raw_thresh = np.percentile(total_bl, p)
        exposed_mask = total_bl > raw_thresh
        ctrl_mask = total_bl <= raw_thresh
        
        n_ctrl = np.sum(ctrl_mask)
        n_exp = np.sum(exposed_mask)
        
        row_dict = {
            "Percentile": f"{p}%",
            "Threshold_Count": raw_thresh,
            "N_Controls": n_ctrl,
            "N_Exposed": n_exp
        }
        
        for pair_name, (t_x, t_y) in cascade_pairs.items():
            rv_ctrl = adjusted_rvcoef(t_x[ctrl_mask], t_y[ctrl_mask])
            rv_exp = adjusted_rvcoef(t_x[exposed_mask], t_y[exposed_mask])
            delta_rv = rv_exp - rv_ctrl
            
            # Permutation test
            perm_deltas = []
            for _ in range(n_perms):
                perm_mask = rng.permutation(exposed_mask)
                perm_ctrl = ~perm_mask
                
                rv_ctrl_p = adjusted_rvcoef(t_x[perm_ctrl], t_y[perm_ctrl])
                rv_exp_p = adjusted_rvcoef(t_x[perm_mask], t_y[perm_mask])
                perm_deltas.append(rv_exp_p - rv_ctrl_p)
                
            perm_deltas = np.array(perm_deltas)
            p_val = (1.0 + np.sum(perm_deltas >= delta_rv)) / (n_perms + 1.0)
            
            row_dict[f"RV_{pair_name}_Controls"] = rv_ctrl
            row_dict[f"RV_{pair_name}_Exposed"] = rv_exp
            row_dict[f"RV_{pair_name}_Difference"] = delta_rv
            row_dict[f"P_Value_{pair_name}"] = p_val
            
        cascade_records.append(row_dict)
        
    df_cascade = pd.DataFrame(cascade_records)
    df_cascade.to_csv("expart_clinical_cascade_coupling_analysis.csv", index=False)
    print("Saved clinical cascade coupling analysis to expart_clinical_cascade_coupling_analysis.csv")
    
    # -------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------
    os.makedirs("expart_report_figures", exist_ok=True)
    
    # 1. View-Specific Brain Coupling Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    views = ["T1 (Volume)", "DTI (Diffusion)", "rsfMRI (Functional)", "Perfusion (Blood Flow)"]
    
    # Exposed Group RV
    ax = axes[0]
    for view_name, col in zip(views, colors):
        ax.plot(percentiles, df_views[f"RV_{view_name}_Exposed"], marker='o', lw=2.5, color=col, label=view_name)
    ax.set_title("Exposed Group Coupling (Adjusted RV)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("Adjusted RV with Blast")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Controls Group RV
    ax = axes[1]
    for view_name, col in zip(views, colors):
        ax.plot(percentiles, df_views[f"RV_{view_name}_Controls"], marker='x', linestyle='--', lw=2, color=col, label=view_name)
    ax.set_title("Controls Group Coupling (Adjusted RV)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("Adjusted RV with Blast")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Permutation P-values
    ax = axes[2]
    for view_name, col in zip(views, colors):
        ax.plot(percentiles, df_views[f"P_Value_{view_name}"], marker='s', lw=2.5, color=col, label=view_name)
    ax.axhline(0.05, color='black', linestyle=':', label="Alpha = 0.05")
    ax.set_title("Significance (Permutation P-value)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("P-value")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("expart_report_figures/multiview_brain_coupling_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Multiview Brain Coupling plots to: expart_report_figures/multiview_brain_coupling_analysis.png")
    
    # 2. Clinical Cascade Coupling Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors_c = ["#9467bd", "#8c564b", "#e377c2"]
    pairs = ["Brain_vs_MH", "MH_vs_PE", "Brain_vs_PE"]
    labels = ["Brain MRI <-> Mental Health", "Mental Health <-> Cognitive Performance", "Brain MRI <-> Cognitive Performance"]
    
    # Exposed Group RV
    ax = axes[0]
    for p_name, col, lbl in zip(pairs, colors_c, labels):
        ax.plot(percentiles, df_cascade[f"RV_{p_name}_Exposed"], marker='o', lw=2.5, color=col, label=lbl)
    ax.set_title("Exposed Group Coupling (Adjusted RV)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("Adjusted RV")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Controls Group RV
    ax = axes[1]
    for p_name, col, lbl in zip(pairs, colors_c, labels):
        ax.plot(percentiles, df_cascade[f"RV_{p_name}_Controls"], marker='x', linestyle='--', lw=2, color=col, label=lbl)
    ax.set_title("Controls Group Coupling (Adjusted RV)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("Adjusted RV")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Permutation P-values
    ax = axes[2]
    for p_name, col, lbl in zip(pairs, colors_c, labels):
        ax.plot(percentiles, df_cascade[f"P_Value_{p_name}"], marker='s', lw=2.5, color=col, label=lbl)
    ax.axhline(0.05, color='black', linestyle=':', label="Alpha = 0.05")
    ax.set_title("Significance (Permutation P-value)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("P-value")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("expart_report_figures/clinical_cascade_coupling_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Clinical Cascade Coupling plots to: expart_report_figures/clinical_cascade_coupling_analysis.png")

if __name__ == "__main__":
    run_multiview_group_coupling()
