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
from sklearn.model_selection import KFold

def run_pipeline(use_gbev=False):
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
    
    # Raw imaging columns
    t1_cols = [c for c in df.columns if c.startswith('T1Hier') or c.startswith('T1w')]
    dti_cols = [c for c in df.columns if c.startswith('DTI')]
    rsf_cols = [c for c in df.columns if c.startswith('rsfMRI')]
    perf_cols = [c for c in df.columns if c.startswith('perf')]
    
    # Filter for numeric columns
    t1_cols_num = [c for c in t1_cols if df[c].dtype in [np.float64, np.int64]]
    dti_cols_num = [c for c in dti_cols if df[c].dtype in [np.float64, np.int64]]
    rsf_cols_num = [c for c in rsf_cols if df[c].dtype in [np.float64, np.int64]]
    perf_cols_num = [c for c in perf_cols if df[c].dtype in [np.float64, np.int64]]
    
    # PCA and standardizers fitted on the 97 complete subjects
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
    
    # PCA coordinates for cohort
    other_t1_pcs = pca_t1.transform(scaler_t1.transform(other_t1))
    other_dti_pcs = pca_dti.transform(scaler_dti.transform(other_dti))
    other_rsf_pcs = pca_rsf.transform(scaler_rsf.transform(other_rsf))
    other_perf_pcs = pca_perf.transform(scaler_perf.transform(other_perf))
    
    # Flow-SiMLR-V for perfusion imputation on 97 subjects
    matrices_perf = [other_t1_pcs, other_dti_pcs, other_rsf_pcs, other_perf_pcs]
    scalers_perf = [StandardScaler() for _ in range(4)]
    scaled_perf_mats = [s.fit_transform(m) for s, m in zip(scalers_perf, matrices_perf)]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(device)
    
    # Perfusion Imputation training loop (1500 epochs, full-batch)
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
    
    # Transform raw target subject features to PCA space
    sub_t1_raw = sub_row[t1_cols_num].fillna(other_rows[t1_cols_num].mean())
    sub_dti_raw = sub_row[dti_cols_num].fillna(other_rows[dti_cols_num].mean())
    sub_rsf_raw = sub_row[rsf_cols_num].fillna(other_rows[rsf_cols_num].mean())
    
    sub_t1_pcs = pca_t1.transform(scaler_t1.transform(sub_t1_raw))
    sub_dti_pcs = pca_dti.transform(scaler_dti.transform(sub_dti_raw))
    sub_rsf_pcs = pca_rsf.transform(scaler_rsf.transform(sub_rsf_raw))
    
    # Standardize target subject PCA features
    sub_t1_scaled = scalers_perf[0].transform(sub_t1_pcs)
    sub_dti_scaled = scalers_perf[1].transform(sub_dti_pcs)
    sub_rsf_scaled = scalers_perf[2].transform(sub_rsf_pcs)
    
    # Compute observed latents for target subject
    with torch.no_grad():
        z_t1 = torch.as_tensor(sub_t1_scaled).float().to(device_obj)
        u_t1 = model_perf.encoders[0](model_perf.linear_encoders[0](z_t1)).cpu().numpy()
        
        z_dti = torch.as_tensor(sub_dti_scaled).float().to(device_obj)
        u_dti = model_perf.encoders[1](model_perf.linear_encoders[1](z_dti)).cpu().numpy()
        
        z_rsf = torch.as_tensor(sub_rsf_scaled).float().to(device_obj)
        u_rsf = model_perf.encoders[2](model_perf.linear_encoders[2](z_rsf)).cpu().numpy()
        
    # Schur complement parameters
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
    
    # Decode predicted perfusion latent to Perfusion PC features
    with torch.no_grad():
        pred_u_perf_t = torch.as_tensor(pred_u_perf).float().to(device_obj)
        pred_flow_z = model_perf.flows[3].inverse(pred_u_perf_t)
        pred_perf_scaled = (pred_flow_z @ model_perf.linear_encoders[3].v.t()).cpu().numpy()
        
    imputed_perf_pcs = scalers_perf[3].inverse_transform(pred_perf_scaled)[0]
    
    print("\nImputed Perfusion PC values for sub-EAS111EAS111:")
    for i, val in enumerate(imputed_perf_pcs):
        print(f"  perfPC{i+1}: {val:.4f}")
        
    # Write to summary file
    os.makedirs("clinical_results", exist_ok=True)
    with open("clinical_results/sub_EAS111_perfusion_imputation.txt", "w") as f:
        f.write("Subject: sub-EAS111EAS111\n")
        f.write("Imputation type: t1PC* dtPC* rsfPC* -> perfPC*\n")
        f.write("Imputed Perfusion PCs:\n")
        for i, val in enumerate(imputed_perf_pcs):
            f.write(f"  perfPC{i+1}: {val:.6f}\n")
            
    # -----------------------------------------------------------------
    # Reconstruct the completed Brain Imaging PC matrix (98 subjects)
    # -----------------------------------------------------------------
    cohort_t1_pcs = pca_t1.transform(scaler_t1.transform(df[t1_cols_num].fillna(other_rows[t1_cols_num].mean())))
    cohort_dti_pcs = pca_dti.transform(scaler_dti.transform(df[dti_cols_num].fillna(other_rows[dti_cols_num].mean())))
    cohort_rsf_pcs = pca_rsf.transform(scaler_rsf.transform(df[rsf_cols_num].fillna(other_rows[rsf_cols_num].mean())))
    
    cohort_perf_pcs = np.zeros((98, 9))
    target_idx = df[df['subjectID'] == subject_id].index[0]
    
    raw_perf_all = df[perf_cols_num].fillna(other_rows[perf_cols_num].mean())
    cohort_perf_pcs = pca_perf.transform(scaler_perf.transform(raw_perf_all))
    cohort_perf_pcs[target_idx] = imputed_perf_pcs
    
    mr_matrix = np.hstack([cohort_t1_pcs, cohort_dti_pcs, cohort_rsf_pcs, cohort_perf_pcs])
    mr_headers = [f"t1PC{i}" for i in range(1, 10)] + \
                 [f"dtPC{i}" for i in range(1, 10)] + \
                 [f"rsfPC{i}" for i in range(1, 10)] + \
                 [f"perfPC{i}" for i in range(1, 10)]
    
    # Clinical and exposure modalities
    npsy_mh_cols = ['npsy_BDI_Total', 'npsy_BSI.18_TotalRaw', 'npsy_IRI_Total', 'npsy_PCL5Total']
    npsy_pe_cols = ['npsy_CVLTShortDelayFreeRecall_Raw', 'npsy_CVLTLongDelayFreeRecall_Raw', 
                     'npsy_DKEFS_Total_Raw_Score', 'npsy_NIH_Fluid_Composite_Score', 
                     'npsy_TMT.A_TotalTime_Raw', 'npsy_TMT.B_TotalTime_Raw', 'npsy_WASIComposite']
    
    # Optional GBEV Score inclusion
    if use_gbev:
        exp_cols = ['X1_BEC', 'X2_BEC', 'X3_BEC', 'X4_BEC', 'X5_BEC', 'BETSGBEV.Score']
    else:
        exp_cols = ['X1_BEC', 'X2_BEC', 'X3_BEC', 'X4_BEC', 'X5_BEC']
        
    print(f"\nBlast exposure variables: {exp_cols}")
    
    # Sqrt transformation to moderate outlier leverage in skewed count data
    bl_matrix_raw = np.sqrt(df[exp_cols].fillna(df[exp_cols].mean()).values)
    mh_matrix_raw = df[npsy_mh_cols].fillna(df[npsy_mh_cols].mean()).values
    mr_matrix_raw = mr_matrix
    pe_matrix_raw = df[npsy_pe_cols].fillna(df[npsy_pe_cols].mean()).values
    
    # -----------------------------------------------------------------
    # Covariate Adjustment (hink, Subject_Age, Sex_num, BV, Highest_Edu)
    # -----------------------------------------------------------------
    df['Sex_num'] = (df['Sex'] == 'Male').astype(float)
    df['hink'] = df['hink'].fillna(0.0) # Impute hink to zero if missing
    cov_cols = ['hink', 'Subject_Age', 'Sex_num', 'BV', 'Highest_Edu']
    print(f"Controlling for covariates: {cov_cols}")
    covariates = df[cov_cols].fillna(df[cov_cols].mean()).values
    
    def residualize_matrix(m, covs):
        X = np.hstack([np.ones((m.shape[0], 1)), covs])
        beta = np.linalg.pinv(X.T @ X) @ X.T @ m
        return m - X @ beta
        
    bl_matrix_adj = residualize_matrix(bl_matrix_raw, covariates)
    mh_matrix_adj = residualize_matrix(mh_matrix_raw, covariates)
    mr_matrix_adj = residualize_matrix(mr_matrix_raw, covariates)
    pe_matrix_adj = residualize_matrix(pe_matrix_raw, covariates)
    
    matrices = [bl_matrix_adj, mh_matrix_adj, mr_matrix_adj, pe_matrix_adj]
    raw_matrices = [bl_matrix_raw, mh_matrix_raw, mr_matrix_raw, pe_matrix_raw]
    names = ["Blast", "Mental_Health", "Brain_Imaging", "Cognitive_Performance"]
    headers = [exp_cols, npsy_mh_cols, mr_headers, npsy_pe_cols]
    
    # Standardize matrices over the 98 subjects
    scaled_mats = [StandardScaler().fit_transform(m) for m in matrices]
    
    print("\n=================================================================")
    print("STEP 2: RUN FLOW-SIMLR-V EXPLORATION (GATED VS. UNIFORM)")
    print("=================================================================")
    
    k = 2
    epochs_exploration = 2000 # Full-batch true convergence
    dynamic_weights_start = 500
    
    results_compare = {}
    models_to_test = [
        ("Gated", True),
        ("Uniform", False)
    ]
    
    for label, use_dynamic in models_to_test:
        print(f"\nFitting Flow-SiMLR-V [{label}] for {epochs_exploration} epochs (full-batch)...")
        res = flow_simr_v(
            scaled_mats,
            k=k,
            epochs=epochs_exploration,
            batch_size=128, # Full-batch
            positivity='positive',
            use_nsa=True,
            nsa_w=0.5,
            dynamic_weights=use_dynamic,
            dynamic_weights_start=dynamic_weights_start,
            energy_type='regression',
            mixing_algorithm='newton',
            device=device,
            verbose=False,
            use_rank_mai=False
        )
        results_compare[label] = res
        
        print(f"  [{label}] Modality Weights: {res['modality_weights'].tolist()}")
        
        # Attributions for Blast
        v_blast = res['v'][0].cpu().numpy()
        print(f"  [{label}] Raw Blast Exposure Attributions:")
        for idx, col_name in enumerate(headers[0]):
             print(f"    {col_name:20s} -> Latent Dim 1: {v_blast[idx, 0]:.4f}, Latent Dim 2: {v_blast[idx, 1]:.4f}")
             
        # Modality coupling
        proj = [torch.as_tensor(scaled_mats[i] @ res['v'][i].cpu().numpy()).float() for i in range(4)]
        print(f"  [{label}] Modality Coupling Strengths (Adjusted RV):")
        for i in range(4):
            for j in range(i+1, 4):
                rv = adjusted_rvcoef(proj[i], proj[j])
                print(f"    {names[i]} <-> {names[j]} -> Adjusted RV = {rv:.4f}")
                
        # Save Gating History plot (if gating was used)
        if use_dynamic:
            out_dir = "expart_report_figures"
            os.makedirs(out_dir, exist_ok=True)
            weight_history = res['weight_history']
            if len(weight_history) > 0:
                plt.figure(figsize=(8, 5))
                w_hist_np = np.array(weight_history)
                for idx, name in enumerate(names):
                    plt.plot(w_hist_np[:, idx], label=name, lw=2)
                plt.axvline(dynamic_weights_start, color='red', linestyle='--', label=f'Gating Start (Warmup={dynamic_weights_start})')
                plt.title(f"Flow-SiMLR-V Gating History (2000 Epochs, N=98)")
                plt.xlabel("Epochs")
                plt.ylabel("Modality Weight")
                plt.legend()
                plt.grid(True, alpha=0.3)
                fname = "gated_weights_98.png"
                plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  Saved: {out_dir}/{fname}")

    # Output weights and RV coefficients comparison to files for parsing in report
    weights_df = pd.DataFrame({
        "Modality": names,
        "Gated": results_compare["Gated"]["modality_weights"].tolist(),
        "Uniform": results_compare["Uniform"]["modality_weights"].tolist()
    })
    weights_df.to_csv("expart_flow_weights.csv", index=False)
    
    # Compute RV coefficients table for output
    rv_records = []
    for i in range(4):
        for j in range(i+1, 4):
            rv_std = adjusted_rvcoef(
                torch.as_tensor(scaled_mats[i] @ results_compare["Gated"]['v'][i].cpu().numpy()).float(),
                torch.as_tensor(scaled_mats[j] @ results_compare["Gated"]['v'][j].cpu().numpy()).float()
            )
            rv_unif = adjusted_rvcoef(
                torch.as_tensor(scaled_mats[i] @ results_compare["Uniform"]['v'][i].cpu().numpy()).float(),
                torch.as_tensor(scaled_mats[j] @ results_compare["Uniform"]['v'][j].cpu().numpy()).float()
            )
            
            rv_records.append({
                "Pair": f"{names[i]} <-> {names[j]}",
                "Gated": rv_std,
                "Uniform": rv_unif
            })
    pd.DataFrame(rv_records).to_csv("expart_flow_rvcoefs.csv", index=False)
    
    # -------------------------------------------------------------
    # NEW: Plot Convergence Curves for Flow-SiMLR-V Gated Model
    # -------------------------------------------------------------
    res_g = results_compare["Gated"]
    loss_h = res_g['loss_history']
    recon_h = res_g['recon_history']
    sim_h = res_g['sim_history']
    
    plt.figure(figsize=(8, 5))
    plt.plot(loss_h, label="Total Loss", color='purple', lw=2)
    plt.plot(recon_h, label="NLL/Flow Reconstruction Loss (Recon)", color='blue', alpha=0.7)
    plt.plot(sim_h, label="Similarity Alignment Loss (Sim)", color='orange', alpha=0.7)
    plt.title("Flow-SiMLR-V Convergence History (2000 Epochs, N=98)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("expart_report_figures/flow_convergence.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved convergence history plot: expart_report_figures/flow_convergence.png")
    
    # -------------------------------------------------------------
    # NEW: Creative Threshold Analysis (Blast vs. MRI relationship across percentiles)
    # -------------------------------------------------------------
    # Calculate sum of raw blast counts
    total_bl = bl_matrix_raw.sum(axis=1)
    
    # Extract fitted projections
    v_bl_g = res_g['v'][0].cpu().numpy()
    v_mr_g = res_g['v'][2].cpu().numpy()
    
    proj_bl_g = scaled_mats[0] @ v_bl_g
    proj_mr_g = scaled_mats[2] @ v_mr_g
    
    percentiles = np.round(np.linspace(0, 85, 18), 1)
    thresh_records = []
    
    for p in percentiles:
        thresh = np.percentile(total_bl, p)
        sub_idx = np.where(total_bl >= thresh)[0]
        
        # Extract subgroups
        sub_proj_bl = torch.as_tensor(proj_bl_g[sub_idx]).float()
        sub_proj_mr = torch.as_tensor(proj_mr_g[sub_idx]).float()
        
        # Subgroup Adjusted RV coefficient
        sub_rv = adjusted_rvcoef(sub_proj_bl, sub_proj_mr)
        
        # Pearson correlation of Dimension 1
        sub_r, sub_p = pearsonr(proj_bl_g[sub_idx, 0], proj_mr_g[sub_idx, 0])
        
        thresh_records.append({
            "Percentile": f"{p}%",
            "Threshold": thresh,
            "N": len(sub_idx),
            "Adjusted_RV": sub_rv,
            "Latent_Dim1_r": sub_r,
            "Latent_Dim1_p": sub_p
        })
        
    df_thresh = pd.DataFrame(thresh_records)
    df_thresh.to_csv("expart_threshold_analysis.csv", index=False)
    print("Saved threshold analysis metrics to expart_threshold_analysis.csv")
    
    # Plot Threshold coupling curve
    plt.figure(figsize=(9, 5))
    plt.plot(percentiles, df_thresh["Adjusted_RV"], marker='o', color='#d62728', lw=2.5, markersize=6)
    plt.title("Blast <-> Brain Imaging Coupling Strength across 18 Severity Thresholds", fontsize=12, fontweight='bold')
    plt.xlabel("Cohort Subgroup (Total Blast Exposure Percentile Threshold)")
    plt.ylabel("Adjusted RV Coefficient between Latents")
    plt.xticks(percentiles[::2], [f"{int(p)}%" for p in percentiles[::2]])
    plt.grid(True, alpha=0.3)
    plt.savefig("expart_report_figures/blast_mri_threshold_coupling.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved threshold analysis curve plot: expart_report_figures/blast_mri_threshold_coupling.png")
    
    # -------------------------------------------------------------
    # NEW: Beautiful Visualization of Blast relationships to other views
    # -------------------------------------------------------------
    v_mh_g = res_g['v'][1].cpu().numpy()
    v_pe_g = res_g['v'][3].cpu().numpy()
    
    proj_mh_g = scaled_mats[1] @ v_mh_g
    proj_pe_g = scaled_mats[3] @ v_pe_g
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    targets = [
        ("Brain Imaging (MRI)", proj_mr_g, '#1f77b4'), 
        ("Mental Health (MH)", proj_mh_g, '#ff7f0e'), 
        ("Cognitive Performance (PE)", proj_pe_g, '#2ca02c')
    ]
    
    for idx, (title, proj_t, color) in enumerate(targets):
        ax = axes[idx]
        x_val = proj_bl_g[:, 0]
        y_val = proj_t[:, 0]
        r_val, p_val = pearsonr(x_val, y_val)
        
        ax.scatter(x_val, y_val, color=color, edgecolors='k', alpha=0.8, s=60)
        # Fit regression line
        m, b = np.polyfit(x_val, y_val, 1)
        ax.plot(x_val, m*x_val + b, 'k--', lw=1.5)
        
        ax.set_title(f"Blast vs. {title}\n(Pearson r = {r_val:.3f}, p = {p_val:.2e})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Blast Latent Coordinate 1")
        ax.set_ylabel(f"{title} Latent Coordinate 1")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("expart_report_figures/blast_relationships.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Blast relationship multi-panel plot: expart_report_figures/blast_relationships.png")
    
    # Compute In-Sample prediction correlations to explain the user's question
    print("\n--- Computing In-Sample Cross-View Prediction Correlations (to contrast with out-of-sample CV) ---")
    in_sample_results = []
    for label, use_dynamic in models_to_test:
        res = results_compare[label]
        model = res['model'].to(device_obj)
        cond_inf = res['cond_inference']
        lat = res['latents']
        
        # MRI -> MH and PE
        latent_img = lat[2].to(device_obj)[:, :k]
        pred_lat_mh = cond_inf.predict_conditional(observed_idx=2, target_idx=1, observed_z=latent_img.cpu())
        pred_lat_pe = cond_inf.predict_conditional(observed_idx=2, target_idx=3, observed_z=latent_img.cpu())
        
        with torch.no_grad():
            flow_mh = model.flows[1].inverse(pred_lat_mh.to(device_obj))
            scaled_mh = (flow_mh @ model.linear_encoders[1].v.t()).cpu().numpy()
            pred_mh_raw = StandardScaler().fit(mh_matrix_adj).inverse_transform(scaled_mh)
            
            flow_pe = model.flows[3].inverse(pred_lat_pe.to(device_obj))
            scaled_pe = (flow_pe @ model.linear_encoders[3].v.t()).cpu().numpy()
            pred_pe_raw = StandardScaler().fit(pe_matrix_adj).inverse_transform(scaled_pe)
            
        mh_names = [n.replace("npsy_", "") for n in headers[1]]
        for idx, col_name in enumerate(mh_names):
            r, p = pearsonr(mh_matrix_adj[:, idx], pred_mh_raw[:, idx])
            in_sample_results.append({
                "Model": label,
                "Target_Modality": "Mental_Health",
                "Variable": col_name,
                "Source": "Brain_Imaging",
                "In_Sample_Correlation": r,
                "In_Sample_P_Value": p
            })
            
        pe_names = [n.replace("npsy_", "") for n in headers[3]]
        for idx, col_name in enumerate(pe_names):
            r, p = pearsonr(pe_matrix_adj[:, idx], pred_pe_raw[:, idx])
            in_sample_results.append({
                "Model": label,
                "Target_Modality": "Cognitive_Performance",
                "Variable": col_name,
                "Source": "Brain_Imaging",
                "In_Sample_Correlation": r,
                "In_Sample_P_Value": p
            })
            
    df_in_sample = pd.DataFrame(in_sample_results)
    df_in_sample.to_csv("expart_in_sample_results.csv", index=False)
    print("Saved in-sample prediction correlations to expart_in_sample_results.csv")
    
    # -----------------------------------------------------------------
    # Freedman-Lane Permutation Significance testing
    # -----------------------------------------------------------------
    print("\nRunning Freedman-Lane Permutation Tests for Latent Dim 1 correlations...")
    # Get Gated model latents for Freedman-Lane
    gated_res = results_compare["Gated"]
    gated_model = gated_res['model'].to(device_obj)
    with torch.no_grad():
        bl_t_gated = torch.as_tensor(scaled_mats[0]).float().to(device_obj)
        latent_bl_gated = gated_model.encoders[0](gated_model.linear_encoders[0](bl_t_gated))[:, 0].cpu().numpy()
        
    def freedman_lane_perm_test(X, Y, covs, n_perms=2000):
        covs_design = np.hstack([np.ones((len(Y), 1)), covs])
        beta_Y = np.linalg.pinv(covs_design.T @ covs_design) @ covs_design.T @ Y
        R_Y = Y - covs_design @ beta_Y
        beta_X = np.linalg.pinv(covs_design.T @ covs_design) @ covs_design.T @ X
        R_X = X - covs_design @ beta_X
        r_obs, p_param = pearsonr(R_X, R_Y)
        perm_rs = []
        rng = np.random.default_rng(42)
        for _ in range(n_perms):
            R_Y_perm = rng.permutation(R_Y)
            r_perm, _ = pearsonr(R_X, R_Y_perm)
            perm_rs.append(r_perm)
        perm_rs = np.array(perm_rs)
        p_perm = (1.0 + np.sum(np.abs(perm_rs) >= np.abs(r_obs))) / (n_perms + 1.0)
        return r_obs, p_param, p_perm

    fl_results = []
    for idx, col in enumerate(mh_names):
        r_obs, p_param, p_perm = freedman_lane_perm_test(latent_bl_gated, mh_matrix_raw[:, idx], covariates)
        fl_results.append({"Variable": col, "Modality": "Mental_Health", "Observed_Partial_r": r_obs, "Parametric_p": p_param, "Freedman_Lane_p": p_perm})

    for idx, col in enumerate(pe_names):
        r_obs, p_param, p_perm = freedman_lane_perm_test(latent_bl_gated, pe_matrix_raw[:, idx], covariates)
        fl_results.append({"Variable": col, "Modality": "Cognitive_Performance", "Observed_Partial_r": r_obs, "Parametric_p": p_param, "Freedman_Lane_p": p_perm})

    df_fl = pd.DataFrame(fl_results)
    df_fl.to_csv("expart_freedman_lane_results.csv", index=False)
    print("\n=== Freedman-Lane Permutation Significance of Latent Dim 1 ===")
    print(df_fl.to_string(index=False))

    def rv_freedman_lane_perm_test(A_res, B_res, n_perms=500):
        A_t = torch.as_tensor(A_res).float()
        B_t = torch.as_tensor(B_res).float()
        obs_rv = adjusted_rvcoef(A_t, B_t)
        perm_rvs = []
        rng = np.random.default_rng(42)
        for _ in range(n_perms):
            A_res_perm = rng.permutation(A_res, axis=0)
            A_t_perm = torch.as_tensor(A_res_perm).float()
            perm_rvs.append(adjusted_rvcoef(A_t_perm, B_t))
        perm_rvs = np.array(perm_rvs)
        p_perm = (1.0 + np.sum(perm_rvs >= obs_rv)) / (n_perms + 1.0)
        return obs_rv, p_perm

    print("\nRunning Freedman-Lane Permutation Tests for Modality Coupling...")
    coupling_pairs = [
        ("Blast", "Mental_Health", bl_matrix_adj, mh_matrix_adj),
        ("Blast", "Brain_Imaging", bl_matrix_adj, mr_matrix_adj),
        ("Blast", "Cognitive_Performance", bl_matrix_adj, pe_matrix_adj),
        ("Brain_Imaging", "Mental_Health", mr_matrix_adj, mh_matrix_adj),
        ("Mental_Health", "Cognitive_Performance", mh_matrix_adj, pe_matrix_adj),
        ("Brain_Imaging", "Cognitive_Performance", mr_matrix_adj, pe_matrix_adj)
    ]
    coupling_results = []
    for name1, name2, mat1, mat2 in coupling_pairs:
        rv, p_perm = rv_freedman_lane_perm_test(mat1, mat2)
        coupling_results.append({"Modality_1": name1, "Modality_2": name2, "Adjusted_RV_Residuals": rv, "Freedman_Lane_p": p_perm})

    df_coupling = pd.DataFrame(coupling_results)
    df_coupling.to_csv("expart_freedman_lane_coupling.csv", index=False)
    print("\n=== Freedman-Lane Permutation Significance of Modality Coupling ===")
    print(df_coupling.to_string(index=False))
    
    print("\n=================================================================")
    print("STEP 3: RUN RIGOROUS 5-FOLD CROSS-VALIDATION FOR CLINICAL IMPUTATION")
    print("=================================================================")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    epochs_cv = 1200 # Sufficient for full-batch convergence
    cv_weights_start = 300
    
    cv_results = []
    
    for label, use_dynamic in models_to_test:
        print(f"\n--- Running 5-Fold CV sweep for {label} ---")
        
        # Predictions storage
        # Source: Brain_Imaging (index 2) -> target: MH (index 1), PE (index 3)
        cv_pred_mh_from_mr = np.zeros_like(mh_matrix_adj)
        cv_pred_pe_from_mr = np.zeros_like(pe_matrix_adj)
        
        # Source: Blast (index 0) -> target: MH (index 1), PE (index 3)
        cv_pred_mh_from_bl = np.zeros_like(mh_matrix_adj)
        cv_pred_pe_from_bl = np.zeros_like(pe_matrix_adj)
        
        fold_idx = 1
        for train_idx, test_idx in kf.split(mr_matrix):
            print(f"[{label}] Training Fold {fold_idx}/5...")
            
            # Scale train/test matrices with covariate adjustment
            train_mats_scaled = []
            test_mats_scaled = []
            train_scalers = []
            
            train_covs = covariates[train_idx]
            test_covs = covariates[test_idx]
            
            for view_idx, m in enumerate(raw_matrices):
                train_v = m[train_idx]
                test_v = m[test_idx]
                
                # Residualize train and test views using train covariates design
                X_train = np.hstack([np.ones((train_v.shape[0], 1)), train_covs])
                beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ train_v
                train_v_adj = train_v - X_train @ beta
                
                X_test = np.hstack([np.ones((test_v.shape[0], 1)), test_covs])
                test_v_adj = test_v - X_test @ beta
                
                s = StandardScaler()
                train_mats_scaled.append(s.fit_transform(train_v_adj))
                test_mats_scaled.append(s.transform(test_v_adj))
                train_scalers.append(s)
                
            res_cv = flow_simr_v(
                train_mats_scaled,
                k=k,
                epochs=epochs_cv,
                batch_size=128, # Full-batch
                positivity='positive',
                use_nsa=True,
                nsa_w=0.5,
                dynamic_weights=use_dynamic,
                dynamic_weights_start=cv_weights_start,
                energy_type='regression',
                mixing_algorithm='newton',
                device=device,
                verbose=False,
                use_rank_mai=False
            )
            
            model_cv = res_cv['model'].to(device_obj)
            cond_inference_cv = res_cv['cond_inference']
            
            # Test features as tensors
            test_bl_t = torch.as_tensor(test_mats_scaled[0]).float().to(device_obj)
            test_mr_t = torch.as_tensor(test_mats_scaled[2]).float().to(device_obj)
            
            with torch.no_grad():
                # Extract latents
                test_latent_bl = model_cv.encoders[0](model_cv.linear_encoders[0](test_bl_t))[:, :k]
                test_latent_mr = model_cv.encoders[2](model_cv.linear_encoders[2](test_mr_t))[:, :k]
                
                # Predictions from MR (Brain Imaging)
                pred_lat_mh_mr = cond_inference_cv.predict_conditional(observed_idx=2, target_idx=1, observed_z=test_latent_mr.cpu())
                pred_lat_pe_mr = cond_inference_cv.predict_conditional(observed_idx=2, target_idx=3, observed_z=test_latent_mr.cpu())
                
                # Predictions from BL (Blast)
                pred_lat_mh_bl = cond_inference_cv.predict_conditional(observed_idx=0, target_idx=1, observed_z=test_latent_bl.cpu())
                pred_lat_pe_bl = cond_inference_cv.predict_conditional(observed_idx=0, target_idx=3, observed_z=test_latent_bl.cpu())
                
                # Decode predictions
                # MH from MR
                flow_mh_mr = model_cv.flows[1].inverse(pred_lat_mh_mr.to(device_obj))
                scaled_mh_mr = (flow_mh_mr @ model_cv.linear_encoders[1].v.t()).cpu().numpy()
                cv_pred_mh_from_mr[test_idx] = train_scalers[1].inverse_transform(scaled_mh_mr)
                
                # PE from MR
                flow_pe_mr = model_cv.flows[3].inverse(pred_lat_pe_mr.to(device_obj))
                scaled_pe_mr = (flow_pe_mr @ model_cv.linear_encoders[3].v.t()).cpu().numpy()
                cv_pred_pe_from_mr[test_idx] = train_scalers[3].inverse_transform(scaled_pe_mr)
                
                # MH from BL
                flow_mh_bl = model_cv.flows[1].inverse(pred_lat_mh_bl.to(device_obj))
                scaled_mh_bl = (flow_mh_bl @ model_cv.linear_encoders[1].v.t()).cpu().numpy()
                cv_pred_mh_from_bl[test_idx] = train_scalers[1].inverse_transform(scaled_mh_bl)
                
                # PE from BL
                flow_pe_bl = model_cv.flows[3].inverse(pred_lat_pe_bl.to(device_obj))
                scaled_pe_bl = (flow_pe_bl @ model_cv.linear_encoders[3].v.t()).cpu().numpy()
                cv_pred_pe_from_bl[test_idx] = train_scalers[3].inverse_transform(scaled_pe_bl)
                
            fold_idx += 1
            
        # Calculate CV correlations and save to list
        mh_names = [n.replace("npsy_", "") for n in headers[1]]
        for idx, col_name in enumerate(mh_names):
            r_mr, p_mr = pearsonr(mh_matrix_adj[:, idx], cv_pred_mh_from_mr[:, idx])
            r_bl, p_bl = pearsonr(mh_matrix_adj[:, idx], cv_pred_mh_from_bl[:, idx])
            
            cv_results.append({
                "Model": label,
                "Target_Modality": "Mental_Health",
                "Variable": col_name,
                "Source": "Brain_Imaging",
                "Correlation": r_mr,
                "P_Value": p_mr
            })
            cv_results.append({
                "Model": label,
                "Target_Modality": "Mental_Health",
                "Variable": col_name,
                "Source": "Blast",
                "Correlation": r_bl,
                "P_Value": p_bl
            })
            
        pe_names = [n.replace("npsy_", "") for n in headers[3]]
        for idx, col_name in enumerate(pe_names):
            r_mr, p_mr = pearsonr(pe_matrix_adj[:, idx], cv_pred_pe_from_mr[:, idx])
            r_bl, p_bl = pearsonr(pe_matrix_adj[:, idx], cv_pred_pe_from_bl[:, idx])
            
            cv_results.append({
                "Model": label,
                "Target_Modality": "Cognitive_Performance",
                "Variable": col_name,
                "Source": "Brain_Imaging",
                "Correlation": r_mr,
                "P_Value": p_mr
            })
            cv_results.append({
                "Model": label,
                "Target_Modality": "Cognitive_Performance",
                "Variable": col_name,
                "Source": "Blast",
                "Correlation": r_bl,
                "P_Value": p_bl
            })
            
    # Save CV results to CSV
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv("expart_imputation_cv_results.csv", index=False)
    print("\nSaved comprehensive CV results to expart_imputation_cv_results.csv")
    
    # Print top out-of-sample predictors
    print("\n=== Top Out-of-Sample CV Predictions (r >= 0.1) ===")
    top_preds = cv_df[cv_df['Correlation'] >= 0.1].sort_values(by="Correlation", ascending=False)
    for _, row in top_preds.iterrows():
        print(f"  [{row['Model']}] {row['Source']} -> {row['Variable']} ({row['Target_Modality']}): r={row['Correlation']:.3f} (p={row['P_Value']:.2e})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gbev", action="store_true", help="Include GBEV Score in Blast exposures modality")
    args = parser.parse_args()
    
    run_pipeline(use_gbev=args.use_gbev)
