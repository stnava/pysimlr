import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pysimlr import flow_simr_v
from pysimlr.utils import set_all_seeds

def run_elastic_net_fused_ridge(n_perms=200, lambda1=0.1, alpha=0.5, lambda2=1.0, sparsity_thresh=1e-4):
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
    y_target_raw = df['npsy_TMT.B_TotalTime_Raw'].fillna(df['npsy_TMT.B_TotalTime_Raw'].mean()).values
    total_bl = bl_matrix_raw.sum(axis=1)
    
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
    
    percentiles = np.round(np.linspace(10, 80, 15), 1)
    M = len(percentiles)
    N = mri_res.shape[0]
    D = mri_res.shape[1]
    
    scaler_mri = StandardScaler()
    mri_scaled = scaler_mri.fit_transform(mri_res)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_res.reshape(-1, 1)).flatten()
    
    X_t = torch.as_tensor(mri_scaled).float()
    Y_t = torch.as_tensor(y_scaled).float()
    
    exposed_masks = []
    subgroup_Zs = []
    subgroup_ys = []
    for j in range(M):
        thresh = np.percentile(total_bl, percentiles[j])
        mask = total_bl > thresh
        exposed_masks.append(mask)
        subgroup_Zs.append(X_t[mask])
        subgroup_ys.append(Y_t[mask])
        
    # -------------------------------------------------------------
    # OPTIMIZATION FUNCTION (ELASTIC NET FUSED RIDGE SOLVER)
    # -------------------------------------------------------------
    def solve_elastic_net_fused(y_input_t):
        W = nn.Parameter(torch.zeros(M, D))
        optimizer = optim.LBFGS([W], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
        sub_ys = [y_input_t[mask] for mask in exposed_masks]
        
        def closure():
            optimizer.zero_grad()
            mse_loss = 0.0
            for j in range(M):
                Z_j = subgroup_Zs[j]
                y_j = sub_ys[j]
                N_j = Z_j.shape[0]
                preds = Z_j @ W[j]
                mse_loss += torch.sum((y_j - preds) ** 2) / N_j
                
            l2_pen = (1.0 - alpha) * lambda1 * torch.sum(W ** 2)
            l1_pen = alpha * lambda1 * torch.sum(torch.sqrt(W ** 2 + 1e-5))
            fused_pen = 0.0
            if M > 1:
                fused_pen = lambda2 * torch.sum((W[1:] - W[:-1]) ** 2)
                
            loss = mse_loss + l2_pen + l1_pen + fused_pen
            loss.backward()
            return loss
            
        optimizer.step(closure)
        W_out = W.detach().numpy()
        
        if alpha > 0.0:
            W_out[np.abs(W_out) < sparsity_thresh] = 0.0
            
        return W_out
        
    # Fit Observed Model
    print(f"Fitting Observed Elastic Net Fused Ridge model (lambda1={lambda1}, alpha={alpha}, lambda2={lambda2})...")
    W_obs = solve_elastic_net_fused(Y_t)
    
    # Evaluate model (returns full array of MSEs across thresholds)
    def evaluate_model(W_mat, y_input_np):
        mses = []
        r2s = []
        for j in range(M):
            mask = exposed_masks[j]
            Z_j = mri_scaled[mask]
            y_j = y_input_np[mask]
            preds = Z_j @ W_mat[j]
            mse = np.mean((y_j - preds) ** 2)
            var_y = np.var(y_j)
            r2 = 1.0 - (mse / var_y) if var_y > 1e-6 else 0.0
            mses.append(mse)
            r2s.append(r2)
        return np.array(mses), np.array(r2s)
        
    subgroup_ns = np.array([np.sum(mask.numpy() if hasattr(mask, 'numpy') else mask) for mask in exposed_masks])
    obs_mses, obs_r2s = evaluate_model(W_obs, y_scaled)
    obs_global_mse = np.mean(obs_mses)
    obs_global_r2 = np.mean(obs_r2s)
    obs_global_dev = np.sum(obs_mses * subgroup_ns)
    print(f"Observed Global MSE: {obs_global_mse:.4f}")
    print(f"Observed Global R2: {obs_global_r2:.4f}")
    print(f"Observed Global Deviance (SSE): {obs_global_dev:.4f}")
    
    modalities = {
        "T1 (Volume)": slice(0, 9),
        "DTI (Diffusion)": slice(9, 18),
        "rsfMRI (Functional)": slice(18, 27),
        "Perfusion (Blood Flow)": slice(27, 36)
    }
    
    obs_norms = {m: np.linalg.norm(W_obs[:, slc], axis=1) for m, slc in modalities.items()}
    # Trajectory overall statistic (mean norm across thresholds)
    obs_mean_norms = {m: np.mean(obs_norms[m]) for m in modalities}
    
    # -------------------------------------------------------------
    # RUN PERMUTATIONS (200 shuffles)
    # -------------------------------------------------------------
    print(f"\nRunning {n_perms} Permutations for Global & Modality Significance...")
    rng = np.random.default_rng(42)
    
    perm_global_mses = []
    perm_global_devs = []
    perm_local_mses = []
    perm_local_norms = {m: [] for m in modalities}
    perm_mean_norms = {m: [] for m in modalities}
    
    for b_idx in range(n_perms):
        if (b_idx + 1) % 50 == 0:
            print(f"  Permutation {b_idx+1}/{n_perms}...")
            
        shuf_idx = rng.permutation(N)
        y_perm_scaled = y_scaled[shuf_idx]
        y_perm_t = torch.as_tensor(y_perm_scaled).float()
        
        W_perm = solve_elastic_net_fused(y_perm_t)
        
        p_mses, p_r2s = evaluate_model(W_perm, y_perm_scaled)
        perm_global_mses.append(np.mean(p_mses))
        perm_global_devs.append(np.sum(p_mses * subgroup_ns))
        perm_local_mses.append(p_mses)
        
        for m, slc in modalities.items():
            norms_perm = np.linalg.norm(W_perm[:, slc], axis=1)
            perm_local_norms[m].append(norms_perm)
            perm_mean_norms[m].append(np.mean(norms_perm))
            
    perm_global_mses = np.array(perm_global_mses)
    perm_global_devs = np.array(perm_global_devs)
    perm_local_mses = np.array(perm_local_mses) # (n_perms, M)
    
    # Calculate P-values
    # 1. Global Model P-value (using sum of deviances)
    global_p_val = (1.0 + np.sum(perm_global_devs <= obs_global_dev)) / (n_perms + 1.0)
    
    # 2. Local Threshold P-values (Prediction significance at each threshold)
    local_threshold_p_vals = []
    for j in range(M):
        p_val_j = (1.0 + np.sum(perm_local_mses[:, j] <= obs_mses[j])) / (n_perms + 1.0)
        local_threshold_p_vals.append(p_val_j)
    local_threshold_p_vals = np.array(local_threshold_p_vals)
    
    # 3. Global and Local Modality P-values
    global_modality_p_vals = {}
    local_modality_p_vals = {}
    
    for m in modalities:
        perm_m_mean = np.array(perm_mean_norms[m])
        # Global trajectory P-value for modality
        global_modality_p_vals[m] = (1.0 + np.sum(perm_m_mean >= obs_mean_norms[m])) / (n_perms + 1.0)
        
        # Local P-values for modality at each threshold
        perm_m_local = np.array(perm_local_norms[m]) # (n_perms, M)
        local_p_vals_m = []
        for j in range(M):
            p_val_mj = (1.0 + np.sum(perm_m_local[:, j] >= obs_norms[m][j])) / (n_perms + 1.0)
            local_p_vals_m.append(p_val_mj)
        local_modality_p_vals[m] = np.array(local_p_vals_m)
        
    print(f"\n=== GLOBAL SIGNIFICANCE EVALUATION ===")
    print(f"  Observed Global MSE: {obs_global_mse:.6f}")
    print(f"  Observed Global Deviance (SSE): {obs_global_dev:.6f}")
    print(f"  Null Deviance (Mean ± SD): {np.mean(perm_global_devs):.6f} ± {np.std(perm_global_devs):.6f}")
    print(f"  Global Permutation P-value (Deviance): {global_p_val:.6f}")
    print("\n=== GLOBAL MODALITY TRAJECTORY P-VALUES ===")
    for m in modalities:
        print(f"  {m}: Observed Mean Norm = {obs_mean_norms[m]:.4f}, Global P-value = {global_modality_p_vals[m]:.6f}")
        
    # -------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # Panel 1: Modality Norms Trajectories
    ax = axes[0]
    for m, col in zip(modalities, colors):
        ax.plot(percentiles, obs_norms[m], marker='o', lw=2.5, color=col, label=f"{m} (Global p={global_modality_p_vals[m]:.3f})")
    ax.set_title("Elastic Net Fused Ridge Modality Norms", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold")
    ax.set_ylabel("Predictive Coefficient L2 Norm")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Panel 2: Modality Local P-values
    ax = axes[1]
    for m, col in zip(modalities, colors):
        ax.plot(percentiles, local_modality_p_vals[m], marker='s', lw=2, color=col, label=f"{m} local p")
    ax.axhline(0.05, color='black', linestyle=':', label="Alpha = 0.05")
    ax.set_title("Modality Threshold-Specific P-values", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold")
    ax.set_ylabel("Local P-value")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Panel 3: Local Threshold Model P-values (MSE significance)
    ax = axes[2]
    ax.plot(percentiles, local_threshold_p_vals, marker='^', lw=2.5, color='purple', label=f"Model SSE (Global p={global_p_val:.3f})")
    ax.axhline(0.05, color='black', linestyle=':', label="Alpha = 0.05")
    ax.set_title("Model Prediction P-values by Threshold", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold")
    ax.set_ylabel("Local Threshold P-value")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/sparse_fused_ridge_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Elastic Net Fused Ridge plots to: expart_report_figures/sparse_fused_ridge_analysis.png")
    
    # Save CSV results
    df_records = []
    for j in range(M):
        row = {
            "Percentile": f"{percentiles[j]}%",
            "Threshold_Count": np.percentile(total_bl, percentiles[j]),
            "Subgroup_N": subgroup_ns[j],
            "Obs_MSE": obs_mses[j],
            "Obs_Deviance": obs_mses[j] * subgroup_ns[j],
            "Obs_R2": obs_r2s[j],
            "Global_Deviance_Obs": obs_global_dev,
            "Global_PValue": global_p_val,
            "PVal_Threshold": local_threshold_p_vals[j]
        }
        for m in modalities:
            row[f"Norm_{m}"] = obs_norms[m][j]
            row[f"PVal_{m}_Local"] = local_modality_p_vals[m][j]
            row[f"PVal_{m}_Global"] = global_modality_p_vals[m]
        df_records.append(row)
    df_out = pd.DataFrame(df_records)
    df_out.to_csv("expart_sparse_fused_ridge_results.csv", index=False)
    print("Saved results to expart_sparse_fused_ridge_results.csv")

if __name__ == "__main__":
    run_elastic_net_fused_ridge(n_perms=200, lambda1=0.25, alpha=0.8, lambda2=1.0, sparsity_thresh=1e-3)
