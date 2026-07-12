import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pysimlr import flow_simr_v
from pysimlr.utils import set_all_seeds

def run_multiview_importance(n_perms=200, lambda1=0.1, lambda2=1.0):
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
    
    # Brain view matrices
    t1_matrix = cohort_t1_pcs
    dti_matrix = cohort_dti_pcs
    rsf_matrix = cohort_rsf_pcs
    perf_matrix = cohort_perf_pcs
    
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
        
    t1_res, _, _ = fit_covariate_models(t1_matrix, covariates)
    dti_res, _, _ = fit_covariate_models(dti_matrix, covariates)
    rsf_res, _, _ = fit_covariate_models(rsf_matrix, covariates)
    perf_res, _, _ = fit_covariate_models(perf_matrix, covariates)
    mri_res = np.hstack([t1_res, dti_res, rsf_res, perf_res])
    
    percentiles = np.round(np.linspace(10, 80, 15), 1)
    M = len(percentiles)
    N = mri_res.shape[0] # 98
    
    scaler_t1_res = StandardScaler()
    scaler_dti_res = StandardScaler()
    scaler_rsf_res = StandardScaler()
    scaler_perf_res = StandardScaler()
    scaler_mri_res = StandardScaler()
    
    t1_scaled = scaler_t1_res.fit_transform(t1_res)
    dti_scaled = scaler_dti_res.fit_transform(dti_res)
    rsf_scaled = scaler_rsf_res.fit_transform(rsf_res)
    perf_scaled = scaler_perf_res.fit_transform(perf_res)
    mri_scaled = scaler_mri_res.fit_transform(mri_res)
    
    # Build binary targets
    y_targets = np.zeros((N, M))
    for j in range(M):
        thresh = np.percentile(total_bl, percentiles[j])
        y_targets[:, j] = (total_bl > thresh).astype(float)
        
    Y_t = torch.as_tensor(y_targets).float()
    
    # -------------------------------------------------------------
    # PART 1: SINGLE-VIEW VS. MULTI-VIEW FUSED LOGISTIC AUC
    # -------------------------------------------------------------
    view_data = {
        "T1 (Volume)": torch.as_tensor(t1_scaled).float(),
        "DTI (Diffusion)": torch.as_tensor(dti_scaled).float(),
        "rsfMRI (Functional)": torch.as_tensor(rsf_scaled).float(),
        "Perfusion (Blood Flow)": torch.as_tensor(perf_scaled).float(),
        "Combined Multiview": torch.as_tensor(mri_scaled).float()
    }
    
    auc_results = {}
    
    for view_name, X_t in view_data.items():
        print(f"Fitting Fused Logistic for {view_name}...")
        D_v = X_t.shape[1]
        W = nn.Parameter(torch.zeros(M, D_v))
        bias = nn.Parameter(torch.zeros(M))
        optimizer = optim.LBFGS([W, bias], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            logits = X_t @ W.t() + bias
            bce_loss = 0.0
            for j in range(M):
                bce_loss += nn.functional.binary_cross_entropy_with_logits(logits[:, j], Y_t[:, j], reduction='mean')
            l1_pen = lambda1 * torch.sum(W ** 2)
            l2_pen = lambda2 * torch.sum((W[1:] - W[:-1]) ** 2)
            loss = bce_loss + l1_pen + l2_pen
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        with torch.no_grad():
            probs = torch.sigmoid(X_t @ W.t() + bias).numpy()
            
        auc_scores = []
        for j in range(M):
            auc_scores.append(roc_auc_score(y_targets[:, j], probs[:, j]))
        auc_results[view_name] = np.array(auc_scores)
        
    # -------------------------------------------------------------
    # PART 2: PERMUTATION TEST ON COMBINED MULTIVIEW COEFFICIENTS
    # -------------------------------------------------------------
    print("\nRunning Permutations on Combined Multiview Coefficients...")
    X_comb_t = view_data["Combined Multiview"]
    D = X_comb_t.shape[1] # 36
    
    # Fit observed combined model
    W_obs = nn.Parameter(torch.zeros(M, D))
    bias_obs = nn.Parameter(torch.zeros(M))
    optimizer_obs = optim.LBFGS([W_obs, bias_obs], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
    
    def closure_obs():
        optimizer_obs.zero_grad()
        logits = X_comb_t @ W_obs.t() + bias_obs
        bce_loss = 0.0
        for j in range(M):
            bce_loss += nn.functional.binary_cross_entropy_with_logits(logits[:, j], Y_t[:, j], reduction='mean')
        l1_pen = lambda1 * torch.sum(W_obs ** 2)
        l2_pen = lambda2 * torch.sum((W_obs[1:] - W_obs[:-1]) ** 2)
        loss = bce_loss + l1_pen + l2_pen
        loss.backward()
        return loss
        
    optimizer_obs.step(closure_obs)
    W_obs_np = W_obs.detach().numpy()
    
    # Modality slice mappings in combined weight matrix
    modalities = {
        "T1 (Volume)": slice(0, 9),
        "DTI (Diffusion)": slice(9, 18),
        "rsfMRI (Functional)": slice(18, 27),
        "Perfusion (Blood Flow)": slice(27, 36)
    }
    
    # Observed norms for each modality at each threshold
    obs_norms = {m: np.linalg.norm(W_obs_np[:, slc], axis=1) for m, slc in modalities.items()}
    
    # Count of how many times permuted norm >= observed norm
    perm_counts = {m: np.zeros(M) for m in modalities}
    
    rng = np.random.default_rng(42)
    
    for b_idx in range(n_perms):
        if (b_idx + 1) % 50 == 0:
            print(f"  Permutation {b_idx+1}/{n_perms}...")
            
        # Shuffle targets row-wise to preserve correlation across thresholds
        shuf_idx = rng.permutation(N)
        Y_perm_t = Y_t[shuf_idx]
        
        W_perm = nn.Parameter(torch.zeros(M, D))
        bias_perm = nn.Parameter(torch.zeros(M))
        optimizer_perm = optim.LBFGS([W_perm, bias_perm], lr=1.0, max_iter=150, line_search_fn='strong_wolfe')
        
        def closure_perm():
            optimizer_perm.zero_grad()
            logits = X_comb_t @ W_perm.t() + bias_perm
            bce_loss = 0.0
            for j in range(M):
                bce_loss += nn.functional.binary_cross_entropy_with_logits(logits[:, j], Y_perm_t[:, j], reduction='mean')
            l1_pen = lambda1 * torch.sum(W_perm ** 2)
            l2_pen = lambda2 * torch.sum((W_perm[1:] - W_perm[:-1]) ** 2)
            loss = bce_loss + l1_pen + l2_pen
            loss.backward()
            return loss
            
        optimizer_perm.step(closure_perm)
        W_perm_np = W_perm.detach().numpy()
        
        for m, slc in modalities.items():
            norms_perm = np.linalg.norm(W_perm_np[:, slc], axis=1)
            perm_counts[m] += (norms_perm >= obs_norms[m]).astype(float)
            
    # Calculate p-values
    p_values = {}
    for m in modalities:
        p_values[m] = (1.0 + perm_counts[m]) / (n_perms + 1.0)
        
    # -------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: AUC Comparison Single-view vs Multi-view
    ax = axes[0]
    colors_auc = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "black"]
    styles_auc = ["--", "--", "--", "--", "-"]
    markers_auc = ["x", "^", "s", "o", "D"]
    lws_auc = [1.5, 1.5, 1.5, 1.5, 3.0]
    
    for i, (name, auc_scores) in enumerate(auc_results.items()):
        ax.plot(percentiles, auc_scores, color=colors_auc[i], linestyle=styles_auc[i],
                marker=markers_auc[i], lw=lws_auc[i], label=name)
        
    ax.axhline(0.5, color='gray', linestyle=':')
    ax.set_title("Single-View vs. Combined Multiview Predictability", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("Classification AUC of Exposure Status")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Panel 2: Modality Permutation P-values
    ax = axes[1]
    colors_p = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, m in enumerate(modalities):
        ax.plot(percentiles, p_values[m], color=colors_p[i], marker='s', lw=2, label=f"{m} p-value")
        
    ax.axhline(0.05, color='black', linestyle=':', label="Significance Alpha = 0.05")
    ax.set_title("Permutation Significance of Modality Contributions", fontsize=11, fontweight='bold')
    ax.set_xlabel("Percentile Threshold defining Groups")
    ax.set_ylabel("P-value")
    ax.set_xticks(percentiles[::2])
    ax.set_xticklabels([f"{int(p)}%" for p in percentiles[::2]])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/multiview_importance_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved Multiview Importance plots to: expart_report_figures/multiview_importance_comparison.png")
    
    # Save CSV results
    df_auc_records = []
    for j in range(M):
        row = {"Percentile": f"{percentiles[j]}%"}
        for name in auc_results:
            row[f"AUC_{name}"] = auc_results[name][j]
        df_auc_records.append(row)
    df_auc = pd.DataFrame(df_auc_records)
    df_auc.to_csv("expart_multiview_auc_comparison.csv", index=False)
    
    df_perm_records = []
    for j in range(M):
        row = {"Percentile": f"{percentiles[j]}%"}
        for m in modalities:
            row[f"Norm_{m}"] = obs_norms[m][j]
            row[f"PVal_{m}"] = p_values[m][j]
        df_perm_records.append(row)
    df_perm = pd.DataFrame(df_perm_records)
    df_perm.to_csv("expart_fused_logistic_permutations.csv", index=False)
    
    # Print quantitative proof
    print("\n=== Multi-View vs. Single-View AUC Comparison (N=98) ===")
    print(df_auc[["Percentile", "AUC_Combined Multiview", "AUC_Perfusion (Blood Flow)", "AUC_DTI (Diffusion)", "AUC_T1 (Volume)", "AUC_rsfMRI (Functional)"]].to_string(index=False))
    
    print("\n=== Permutation P-Values of Modality Contributions ===")
    print(df_perm[["Percentile", "PVal_Perfusion (Blood Flow)", "PVal_DTI (Diffusion)", "PVal_T1 (Volume)", "PVal_rsfMRI (Functional)"]].to_string(index=False))

if __name__ == "__main__":
    run_multiview_importance()
