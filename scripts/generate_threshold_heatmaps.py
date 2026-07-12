import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pysimlr import flow_simr_v
from pysimlr.utils import set_all_seeds

def generate_heatmaps(lambda1=0.1, lambda2=5.0):
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
    D = mri_res.shape[1] # 36
    N = mri_res.shape[0] # 98
    
    scaler_mri = StandardScaler()
    mri_scaled = scaler_mri.fit_transform(mri_res)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_res.reshape(-1, 1)).flatten()
    
    # -------------------------------------------------------------
    # Solve Fused Ridge Regression (lambda2 = 5.0)
    # -------------------------------------------------------------
    print("\nSolving Fused Ridge...")
    A_ridge = np.zeros((M * D, M * D))
    b_ridge = np.zeros(M * D)
    
    for j in range(M):
        raw_thresh = np.percentile(total_bl, percentiles[j])
        exposed_mask = total_bl > raw_thresh
        
        Z_j = mri_scaled[exposed_mask]
        y_j = y_scaled[exposed_mask]
        N_j = Z_j.shape[0]
        
        diag_block = (Z_j.T @ Z_j) / N_j + lambda1 * np.eye(D)
        
        if j > 0:
            diag_block += lambda2 * np.eye(D)
        if j < M - 1:
            diag_block += lambda2 * np.eye(D)
            
        A_ridge[j*D : (j+1)*D, j*D : (j+1)*D] = diag_block
        
        if j > 0:
            A_ridge[j*D : (j+1)*D, (j-1)*D : j*D] = -lambda2 * np.eye(D)
        if j < M - 1:
            A_ridge[j*D : (j+1)*D, (j+1)*D : (j+2)*D] = -lambda2 * np.eye(D)
            
        b_ridge[j*D : (j+1)*D] = (Z_j.T @ y_j) / N_j
        
    w_ridge = np.linalg.solve(A_ridge, b_ridge).reshape((M, D))
    
    # -------------------------------------------------------------
    # Solve Fused Logistic Regression (lambda2 = 5.0)
    # -------------------------------------------------------------
    print("\nSolving Fused Logistic...")
    y_targets = np.zeros((N, M))
    for j in range(M):
        thresh = np.percentile(total_bl, percentiles[j])
        y_targets[:, j] = (total_bl > thresh).astype(float)
        
    X_t = torch.as_tensor(mri_scaled).float()
    Y_t = torch.as_tensor(y_targets).float()
    
    W_t = nn.Parameter(torch.zeros(M, D))
    bias_t = nn.Parameter(torch.zeros(M))
    optimizer = optim.LBFGS([W_t, bias_t], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        logits = X_t @ W_t.t() + bias_t
        bce_loss = 0.0
        for j in range(M):
            bce_loss += nn.functional.binary_cross_entropy_with_logits(logits[:, j], Y_t[:, j], reduction='mean')
        l1_pen = lambda1 * torch.sum(W_t ** 2)
        l2_pen = lambda2 * torch.sum((W_t[1:] - W_t[:-1]) ** 2)
        loss = bce_loss + l1_pen + l2_pen
        loss.backward()
        return loss
        
    optimizer.step(closure)
    w_logistic = W_t.detach().numpy()
    
    # -------------------------------------------------------------
    # HEATMAP PLOTTING
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    yticklabels = [f"{int(p)}% (N={np.sum(total_bl > np.percentile(total_bl, p))})" for p in percentiles]
    xticklabels = [f"T1_{i}" for i in range(1, 10)] + \
                  [f"DTI_{i}" for i in range(1, 10)] + \
                  [f"rsf_{i}" for i in range(1, 10)] + \
                  [f"perf_{i}" for i in range(1, 10)]
                  
    # Plot Ridge Heatmap
    ax = axes[0]
    sns.heatmap(w_ridge, cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Predictive Weight (std)'},
                yticklabels=yticklabels, xticklabels=xticklabels)
    ax.set_title("Fused Ridge Regression Coefficients (Predicting TMT.B)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Severity Subgroup Percentile Threshold", fontsize=11)
    ax.set_xlabel("Brain Imaging PCA Features", fontsize=11)
    # Add vertical divider lines between modalities
    ax.axvline(9, color="black", linestyle="--", lw=1.5)
    ax.axvline(18, color="black", linestyle="--", lw=1.5)
    ax.axvline(27, color="black", linestyle="--", lw=1.5)
    # Add text labels for modalities
    ax.text(4.5, -0.5, "T1 (Volume)", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
    ax.text(13.5, -0.5, "DTI (Diffusion)", ha="center", fontsize=10, fontweight="bold", color="#ff7f0e")
    ax.text(22.5, -0.5, "rsfMRI (Functional)", ha="center", fontsize=10, fontweight="bold", color="#2ca02c")
    ax.text(31.5, -0.5, "Perfusion (Blood Flow)", ha="center", fontsize=10, fontweight="bold", color="#d62728")
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # Plot Logistic Heatmap
    ax = axes[1]
    sns.heatmap(w_logistic, cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Classification Weight (std)'},
                yticklabels=yticklabels, xticklabels=xticklabels)
    ax.set_title("Fused Logistic Regression Coefficients (Predicting Exposure)", fontsize=13, fontweight='bold')
    ax.set_ylabel("")
    ax.set_xlabel("Brain Imaging PCA Features", fontsize=11)
    ax.axvline(9, color="black", linestyle="--", lw=1.5)
    ax.axvline(18, color="black", linestyle="--", lw=1.5)
    ax.axvline(27, color="black", linestyle="--", lw=1.5)
    ax.text(4.5, -0.5, "T1 (Volume)", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
    ax.text(13.5, -0.5, "DTI (Diffusion)", ha="center", fontsize=10, fontweight="bold", color="#ff7f0e")
    ax.text(22.5, -0.5, "rsfMRI (Functional)", ha="center", fontsize=10, fontweight="bold", color="#2ca02c")
    ax.text(31.5, -0.5, "Perfusion (Blood Flow)", ha="center", fontsize=10, fontweight="bold", color="#d62728")
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/threshold_coefficient_heatmaps.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Threshold Coefficient Heatmaps to: expart_report_figures/threshold_coefficient_heatmaps.png")

if __name__ == "__main__":
    generate_heatmaps()
