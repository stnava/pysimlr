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
from scipy.stats import pearsonr

def run_rigorous_flow_significance(n_perms=20):
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
    
    npsy_mh_cols = ['npsy_BDI_Total', 'npsy_BSI.18_TotalRaw', 'npsy_IRI_Total', 'npsy_PCL5Total']
    npsy_pe_cols = ['npsy_CVLTShortDelayFreeRecall_Raw', 'npsy_CVLTLongDelayFreeRecall_Raw', 
                     'npsy_DKEFS_Total_Raw_Score', 'npsy_NIH_Fluid_Composite_Score', 
                     'npsy_TMT.A_TotalTime_Raw', 'npsy_TMT.B_TotalTime_Raw', 'npsy_WASIComposite']
    exp_cols = ['X1_BEC', 'X2_BEC', 'X3_BEC', 'X4_BEC', 'X5_BEC']
    
    bl_matrix_raw = np.sqrt(df[exp_cols].fillna(df[exp_cols].mean()).values)
    mh_matrix_raw = df[npsy_mh_cols].fillna(df[npsy_mh_cols].mean()).values
    mr_matrix_raw = mr_matrix
    pe_matrix_raw = df[npsy_pe_cols].fillna(df[npsy_pe_cols].mean()).values
    
    # Setup Covariates
    df['Sex_num'] = (df['Sex'] == 'Male').astype(float)
    df['hink'] = df['hink'].fillna(0.0)
    cov_cols = ['hink', 'Subject_Age', 'Sex_num', 'BV', 'Highest_Edu']
    covariates = df[cov_cols].fillna(df[cov_cols].mean()).values
    
    # Freedman-Lane Residualization
    def fit_covariate_models(m, covs):
        X = np.hstack([np.ones((m.shape[0], 1)), covs])
        beta = np.linalg.pinv(X.T @ X) @ X.T @ m
        preds = X @ beta
        resids = m - preds
        return resids, preds, beta
        
    bl_res, bl_pred, bl_beta = fit_covariate_models(bl_matrix_raw, covariates)
    mh_res, mh_pred, mh_beta = fit_covariate_models(mh_matrix_raw, covariates)
    mr_res, mr_pred, mr_beta = fit_covariate_models(mr_matrix_raw, covariates)
    pe_res, pe_pred, pe_beta = fit_covariate_models(pe_matrix_raw, covariates)
    
    scaled_bl_obs = StandardScaler().fit_transform(bl_res)
    scaled_mh_obs = StandardScaler().fit_transform(mh_res)
    scaled_mr_obs = StandardScaler().fit_transform(mr_res)
    scaled_pe_obs = StandardScaler().fit_transform(pe_res)
    
    scaled_mats_obs = [scaled_bl_obs, scaled_mh_obs, scaled_mr_obs, scaled_pe_obs]
    
    print("\n=================================================================")
    print("STEP 2: FIT OBSERVED FLOW-SIMLR-V MODEL WITH POSITIVITY")
    print("=================================================================")
    
    k = 2
    epochs_flow = 800  # Streamlined for faster refitting
    
    res_obs = flow_simr_v(
        scaled_mats_obs,
        k=k,
        epochs=epochs_flow,
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
    
    # Calculate observed statistics
    v_bl_obs = res_obs['v'][0].cpu().numpy()
    v_mr_obs = res_obs['v'][2].cpu().numpy()
    proj_bl_obs = scaled_bl_obs @ v_bl_obs
    proj_mr_obs = scaled_mr_obs @ v_mr_obs
    obs_coupling = adjusted_rvcoef(torch.as_tensor(proj_bl_obs).float(), torch.as_tensor(proj_mr_obs).float())
    print(f"Observed Flow Blast <-> MRI coupling (Adjusted RV): {obs_coupling:.4f}")
    
    # Observed non-linear correlation for Blast Latent Dim 1 with CVLTLongDelay
    model_obs = res_obs['model'].to(device_obj)
    with torch.no_grad():
        bl_t = torch.as_tensor(scaled_bl_obs).float().to(device_obj)
        latent_bl_obs = model_obs.encoders[0](model_obs.linear_encoders[0](bl_t))[:, 0].cpu().numpy()
        
    target_pe_idx = npsy_pe_cols.index('npsy_CVLTLongDelayFreeRecall_Raw')
    obs_corr, obs_p = pearsonr(latent_bl_obs, pe_res[:, target_pe_idx])
    print(f"Observed Flow Blast Latent Dim 1 <-> CVLTLongDelay correlation: r={obs_corr:.4f} (p={obs_p:.4f})")
    
    # -----------------------------------------------------------------
    # Post-hoc Permutation significance (without refitting, current implementation)
    # -----------------------------------------------------------------
    print("\n--- Running Post-hoc Permutation Significance Tests (no model refitting) ---")
    posthoc_rv_nulls = []
    posthoc_corr_nulls = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_perms):
        # Shuffle projection and latent vectors directly
        proj_bl_perm = rng.permutation(proj_bl_obs, axis=0)
        posthoc_rv_nulls.append(adjusted_rvcoef(torch.as_tensor(proj_bl_perm).float(), torch.as_tensor(proj_mr_obs).float()))
        
        latent_bl_perm = rng.permutation(latent_bl_obs)
        posthoc_corr_nulls.append(pearsonr(latent_bl_perm, pe_res[:, target_pe_idx])[0])
        
    posthoc_rv_p = (1.0 + np.sum(np.array(posthoc_rv_nulls) >= obs_coupling)) / (n_perms + 1.0)
    posthoc_corr_p = (1.0 + np.sum(np.abs(np.array(posthoc_corr_nulls)) >= np.abs(obs_corr))) / (n_perms + 1.0)
    
    print(f"  Post-hoc Coupling p-value: {posthoc_rv_p:.4f}")
    print(f"  Post-hoc Correlation p-value: {posthoc_corr_p:.4f}")
    
    # -----------------------------------------------------------------
    # Rigorous Refit-under-Permutation significance (Full Flow-SiMLR-V)
    # -----------------------------------------------------------------
    print(f"\n--- Running Rigorous Refit-under-Permutation Tests ({n_perms} perms, refitting full flow-simlr-v each time) ---")
    refit_rv_nulls = []
    refit_corr_nulls = []
    
    X_cov = np.hstack([np.ones((covariates.shape[0], 1)), covariates])
    
    for b in range(n_perms):
        print(f"  Running permutation refit {b+1}/{n_perms}...")
        
        # 1. Permute the residuals of the target view (Blast)
        bl_res_perm = rng.permutation(bl_res, axis=0)
        
        # 2. Reconstruct under the null
        bl_null = bl_pred + bl_res_perm
        
        # 3. Re-residualize
        beta_bl_null = np.linalg.pinv(X_cov.T @ X_cov) @ X_cov.T @ bl_null
        bl_adj_null = bl_null - X_cov @ beta_bl_null
        
        # 4. Standardize all views
        scaled_bl_null = StandardScaler().fit_transform(bl_adj_null)
        scaled_mh = StandardScaler().fit_transform(mh_res)
        scaled_mr = StandardScaler().fit_transform(mr_res)
        scaled_pe = StandardScaler().fit_transform(pe_res)
        
        scaled_mats_null = [scaled_bl_null, scaled_mh, scaled_mr, scaled_pe]
        
        # 5. Refit full Flow-SiMLR-V model with positivity constraints
        res_null = flow_simr_v(
            scaled_mats_null,
            k=k,
            epochs=epochs_flow,
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
        
        # 6. Extract the projections and latents from the null model
        v_bl_null = res_null['v'][0].cpu().numpy()
        v_mr_null = res_null['v'][2].cpu().numpy()
        proj_bl_null = scaled_bl_null @ v_bl_null
        proj_mr_null = scaled_mr @ v_mr_null
        
        # 7. Compute null statistics
        rv_null = adjusted_rvcoef(torch.as_tensor(proj_bl_null).float(), torch.as_tensor(proj_mr_null).float())
        refit_rv_nulls.append(rv_null)
        
        model_null = res_null['model'].to(device_obj)
        with torch.no_grad():
            bl_t_null = torch.as_tensor(scaled_bl_null).float().to(device_obj)
            latent_bl_null = model_null.encoders[0](model_null.linear_encoders[0](bl_t_null))[:, 0].cpu().numpy()
            
        r_null, _ = pearsonr(latent_bl_null, pe_res[:, target_pe_idx])
        refit_corr_nulls.append(r_null)
        
    refit_rv_p = (1.0 + np.sum(np.array(refit_rv_nulls) >= obs_coupling)) / (n_perms + 1.0)
    refit_corr_p = (1.0 + np.sum(np.abs(np.array(refit_corr_nulls)) >= np.abs(obs_corr))) / (n_perms + 1.0)
    
    print(f"\n  Refit Coupling p-value: {refit_rv_p:.4f}")
    print(f"  Refit Correlation p-value: {refit_corr_p:.4f}")
    
    # Save statistics and generate comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Coupling RV Coefficient Null Distributions
    ax = axes[0]
    ax.hist(posthoc_rv_nulls, bins=10, alpha=0.5, label="Post-hoc Null (No Refit)", color='blue', edgecolor='k')
    ax.hist(refit_rv_nulls, bins=10, alpha=0.5, label="Rigorous Null (Refit Flow)", color='green', edgecolor='k')
    ax.axvline(obs_coupling, color='red', linestyle='--', lw=2, label=f"Observed RV = {obs_coupling:.4f}")
    ax.set_title(f"Modality Coupling Null (Full Flow-SiMLR-V)\nPost-hoc p = {posthoc_rv_p:.3f} | Refit p = {refit_rv_p:.3f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Adjusted RV Coefficient")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot Latent Dim 1 Correlation Null Distributions
    ax = axes[1]
    ax.hist(posthoc_corr_nulls, bins=10, alpha=0.5, label="Post-hoc Null (No Refit)", color='blue', edgecolor='k')
    ax.hist(refit_corr_nulls, bins=10, alpha=0.5, label="Rigorous Null (Refit Flow)", color='green', edgecolor='k')
    ax.axvline(obs_corr, color='red', linestyle='--', lw=2, label=f"Observed r = {obs_corr:.4f}")
    ax.set_title(f"Latent Dim 1 Correlation Null (Full Flow-SiMLR-V)\nPost-hoc p = {posthoc_corr_p:.3f} | Refit p = {refit_corr_p:.3f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Pearson Correlation r")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/flow_significance_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("\nSaved comparison figure: expart_report_figures/flow_significance_comparison.png")
    
    summary_df = pd.DataFrame({
        "Metric": ["Blast <-> MRI Coupling (RV)", "Blast Dim 1 <-> Memory Correlation (r)"],
        "Observed": [obs_coupling, obs_corr],
        "Post_hoc_P": [posthoc_rv_p, posthoc_corr_p],
        "Refit_P": [refit_rv_p, refit_corr_p]
    })
    summary_df.to_csv("expart_flow_rigorous_significance_results.csv", index=False)
    print("Saved comparison summary to: expart_flow_rigorous_significance_results.csv")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_rigorous_flow_significance(n_perms=20)
