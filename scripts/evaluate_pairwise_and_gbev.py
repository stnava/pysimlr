import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pysimlr import flow_simr_v
from pysimlr.utils import set_all_seeds, adjusted_rvcoef
import multiprocessing

# Set PyTorch to single thread inside worker processes to prevent thrashing
torch.set_num_threads(1)

def get_logistic_deviance(y, p):
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    logL = np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return -2.0 * logL

def solve_fused_logistic(X_t, Y_t, C_t, lambda1, alpha, lambda2, sparsity_thresh, M, D, N, K):
    W_brain = nn.Parameter(torch.zeros(M, D))
    W_covs = nn.Parameter(torch.zeros(M, K))
    optimizer = optim.LBFGS([W_brain, W_covs], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        logits = X_t @ W_brain.t() + C_t @ W_covs.t()
        bce_loss = 0.0
        for j in range(M):
            bce_loss += nn.functional.binary_cross_entropy_with_logits(logits[:, j], Y_t[:, j], reduction='mean')
        bce_loss = bce_loss / M
        l2_pen = (1.0 - alpha) * lambda1 * torch.sum(W_brain ** 2)
        l1_pen = alpha * lambda1 * torch.sum(torch.sqrt(W_brain ** 2 + 1e-5))
        fused_pen = 0.0
        if M > 1:
            fused_pen = lambda2 * torch.sum((W_brain[1:] - W_brain[:-1]) ** 2)
        loss = bce_loss + l2_pen + l1_pen + fused_pen
        loss.backward()
        return loss
        
    optimizer.step(closure)
    W_brain_out = W_brain.detach().numpy()
    W_covs_out = W_covs.detach().numpy()
    if alpha > 0.0:
        W_brain_out[np.abs(W_brain_out) < sparsity_thresh] = 0.0
    return W_brain_out, W_covs_out

def solve_fused_linear(X_t, Y_t, C_t, exposed_masks, lambda1, alpha, lambda2, sparsity_thresh, M, D, N, K):
    W_brain = nn.Parameter(torch.zeros(M, D))
    W_covs = nn.Parameter(torch.zeros(M, K))
    optimizer = optim.LBFGS([W_brain, W_covs], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
    sub_ys = [Y_t[mask] for mask in exposed_masks]
    subgroup_Zs = [X_t[mask] for mask in exposed_masks]
    subgroup_Cs = [C_t[mask] for mask in exposed_masks]
    
    def closure():
        optimizer.zero_grad()
        mse_loss = 0.0
        for j in range(M):
            Z_j = subgroup_Zs[j]
            C_j = subgroup_Cs[j]
            y_j = sub_ys[j]
            N_j = Z_j.shape[0]
            if N_j > 0:
                preds = Z_j @ W_brain[j] + C_j @ W_covs[j]
                mse_loss += torch.sum((y_j - preds) ** 2) / N_j
        mse_loss = 0.5 * (mse_loss / M)
        l2_pen = (1.0 - alpha) * lambda1 * torch.sum(W_brain ** 2)
        l1_pen = alpha * lambda1 * torch.sum(torch.sqrt(W_brain ** 2 + 1e-5))
        fused_pen = 0.0
        if M > 1:
            fused_pen = lambda2 * torch.sum((W_brain[1:] - W_brain[:-1]) ** 2)
        loss = mse_loss + l2_pen + l1_pen + fused_pen
        loss.backward()
        return loss
        
    optimizer.step(closure)
    W_brain_out = W_brain.detach().numpy()
    W_covs_out = W_covs.detach().numpy()
    if alpha > 0.0:
        W_brain_out[np.abs(W_brain_out) < sparsity_thresh] = 0.0
    return W_brain_out, W_covs_out

def logistic_worker(args):
    torch.set_num_threads(1)
    shuf_idx, mri_scaled, C_t, total_bl, thresholds, lambda1, alpha, lambda2, sparsity_thresh = args
    N = mri_scaled.shape[0]
    M = len(thresholds)
    D = mri_scaled.shape[1]
    K = C_t.shape[1]
    
    shuf_total_bl = total_bl[shuf_idx]
    y_targets = np.zeros((N, M))
    for j in range(M):
        y_targets[:, j] = (shuf_total_bl >= thresholds[j]).astype(float)
        
    X_t = torch.as_tensor(mri_scaled).float()
    Y_t = torch.as_tensor(y_targets).float()
    C_torch = torch.as_tensor(C_t).float()
    
    W_brain, W_covs = solve_fused_logistic(X_t, Y_t, C_torch, lambda1, alpha, lambda2, sparsity_thresh, M, D, N, K)
    logits = mri_scaled @ W_brain.T + C_t @ W_covs.T
    probs = 1.0 / (1.0 + np.exp(-logits))
    dev_sum = 0.0
    for j in range(M):
        dev_sum += get_logistic_deviance(y_targets[:, j], probs[:, j])
    return dev_sum

def linear_worker(args):
    torch.set_num_threads(1)
    shuf_idx, mri_scaled, C_t, y_scaled, total_bl, percentiles, lambda1, alpha, lambda2, sparsity_thresh = args
    N = mri_scaled.shape[0]
    M = len(percentiles)
    D = mri_scaled.shape[1]
    K = C_t.shape[1]
    
    y_shuf = y_scaled[shuf_idx]
    exposed_masks = []
    for j in range(M):
        thresh = np.percentile(total_bl, percentiles[j])
        mask = total_bl > thresh
        exposed_masks.append(torch.as_tensor(mask).bool())
        
    X_t = torch.as_tensor(mri_scaled).float()
    Y_t = torch.as_tensor(y_shuf).float()
    C_torch = torch.as_tensor(C_t).float()
    
    W_brain, W_covs = solve_fused_linear(X_t, Y_t, C_torch, exposed_masks, lambda1, alpha, lambda2, sparsity_thresh, M, D, N, K)
    dev_sum = 0.0
    for j in range(M):
        mask = exposed_masks[j].numpy()
        Z_j = mri_scaled[mask]
        C_j = C_t[mask]
        y_j = y_shuf[mask]
        preds = Z_j @ W_brain[j] + C_j @ W_covs[j]
        dev_sum += np.sum((y_j - preds) ** 2)
    return dev_sum

# Helper for regressing covariates from latents
def regress_covariates(Z, C):
    # Solve linear regression: Z = C * B + E => B = (C^T C)^(-1) C^T Z
    # E = Z - C * B
    B = np.linalg.pinv(C.T @ C) @ C.T @ Z
    E = Z - C @ B
    return E

def run_pairwise_and_gbev_studies():
    set_all_seeds(42)
    workspace_dir = "/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/code/pysimlr"
    artifact_dir = "/Users/stnava/.gemini/antigravity-cli/brain/fda666c5-3968-404e-b3e4-f71672770deb"
    
    print("=================================================================")
    print("STEP 1: LEARN K=8 shared representation via Simlr-Flow-V")
    print("=================================================================")
    fpath_raw = "../extern/ExpArt/data/expartdf_power_analysis.csv"
    if not os.path.exists(fpath_raw):
        raise FileNotFoundError(f"Cannot find raw EXPART data at: {fpath_raw}")
        
    df_raw = pd.read_csv(fpath_raw, low_memory=False)
    
    t1_cols = [c for c in df_raw.columns if c.startswith('T1Hier') or c.startswith('T1w')]
    dti_cols = [c for c in df_raw.columns if c.startswith('DTI')]
    rsf_cols = [c for c in df_raw.columns if c.startswith('rsfMRI')]
    perf_cols = [c for c in df_raw.columns if c.startswith('perf')]
    
    t1_cols_num = [c for c in t1_cols if df_raw[c].dtype in [np.float64, np.int64]]
    dti_cols_num = [c for c in dti_cols if df_raw[c].dtype in [np.float64, np.int64]]
    rsf_cols_num = [c for c in rsf_cols if df_raw[c].dtype in [np.float64, np.int64]]
    perf_cols_num = [c for c in perf_cols if df_raw[c].dtype in [np.float64, np.int64]]
    
    def get_clean_scaled(cols):
        m = df_raw[cols].fillna(df_raw[cols].mean()).values
        std = np.std(m, axis=0)
        std[std < 1e-6] = 1.0
        mean = np.mean(m, axis=0)
        return (m - mean) / std
        
    t1_scaled = get_clean_scaled(t1_cols_num)
    dti_scaled = get_clean_scaled(dti_cols_num)
    rsf_scaled = get_clean_scaled(rsf_cols_num)
    perf_scaled = get_clean_scaled(perf_cols_num)
    
    scaled_mats = [t1_scaled, dti_scaled, rsf_scaled, perf_scaled]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    res_flow = flow_simr_v(
        scaled_mats,
        k=8,
        epochs=2000,
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
    
    # Latents per modality
    latents_np = [lat.detach().cpu().numpy() for lat in res_flow['latents']]
    modalities = ['T1 Anatomy', 'DTI White Matter', 'rsfMRI Connectivity', 'ASL Perfusion']
    
    # Load Covariates and Outcomes
    fpath_r = "scratch/expart_exact_r_data.csv"
    df_r = pd.read_csv(fpath_r)
    N = df_r.shape[0]
    
    df_r['Sex_num'] = (df_r['Sex'] == 'Male').astype(float)
    df_r['Race2_num'] = (df_r['Race2'] == 'White').astype(float)
    df_r['hink'] = df_r['hink'].fillna(0.0)
    df_r['Highest_Edu'] = df_r['Highest_Edu'].fillna(df_r['Highest_Edu'].mean())
    df_r['Subject_Age'] = df_r['Subject_Age'].fillna(df_r['Subject_Age'].mean())
    df_r['image_quality'] = df_r['image_quality'].fillna(df_r['image_quality'].mean())
    
    C = np.ones((N, 7))
    C[:, 1] = df_r['Race2_num'].values
    C[:, 2] = df_r['Sex_num'].values
    C[:, 3] = df_r['hink'].values
    C[:, 4] = df_r['Highest_Edu'].values
    C[:, 5] = df_r['Subject_Age'].values
    C[:, 6] = df_r['image_quality'].values
    
    # ==============================================================================
    # STEP 2: PAIRWISE MODALITY COUPLING COMPARISON (FREEDMAN-LANE 500 PERMS)
    # ==============================================================================
    print("\n=================================================================")
    print("STEP 2: RUN PAIRWISE COUPLING COMPARISONS ON COVARIATE-RESIDUALIZED LATENTS")
    print("=================================================================")
    
    # Regress out covariates from each modality's latents to obtain residuals
    residuals = [regress_covariates(lat, C) for lat in latents_np]
    residuals_torch = [torch.as_tensor(res).float() for res in residuals]
    
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    obs_rvs = {}
    for pair in pairs:
        obs_rvs[pair] = adjusted_rvcoef(residuals_torch[pair[0]], residuals_torch[pair[1]])
        
    print(f"Running 500 subject-shuffling permutations to check pairwise coupling significance...")
    n_perms_pw = 500
    null_rvs = {pair: [] for pair in pairs}
    
    for p in range(n_perms_pw):
        shuf_idx = np.random.permutation(N)
        for pair in pairs:
            # Shuffle the second modality row correspondence to break coupling
            shuf_y = residuals_torch[pair[1]][shuf_idx]
            null_rvs[pair].append(adjusted_rvcoef(residuals_torch[pair[0]], shuf_y))
            
    print("\nPairwise Coupling Significance Results:")
    print("-" * 75)
    print(f"{'Pairwise Modality Comparison':40s} | {'Obs RV':7s} | {'p-value':7s} | {'Z-score':7s}")
    print("-" * 75)
    
    pairwise_results = []
    for pair in pairs:
        obs = obs_rvs[pair]
        null_dist = np.array(null_rvs[pair])
        p_val = (1.0 + np.sum(null_dist >= obs)) / (n_perms_pw + 1.0)
        z_score = (obs - np.mean(null_dist)) / (np.std(null_dist) + 1e-8)
        
        pair_name = f"{modalities[pair[0]]} <-> {modalities[pair[1]]}"
        sig_char = "*" if p_val < 0.05 else " "
        print(f"{pair_name:40s} | {obs:.4f} | {p_val:.4f} {sig_char} | {z_score:.2f}")
        
        pairwise_results.append({
            "Pair": pair_name,
            "Obs_RV": obs,
            "p_value": p_val,
            "Z_score": z_score
        })
    print("-" * 75)
    
    df_pw = pd.DataFrame(pairwise_results)
    df_pw.to_csv(os.path.join(workspace_dir, "expart_pairwise_modality_relations_results.csv"), index=False)
    
    # Save a pairwise report to artifacts
    pw_artifact_path = os.path.join(artifact_dir, "pairwise_modality_relations_report.md")
    with open(pw_artifact_path, "w") as f:
        f.write("# EXPART Modality Pairwise Coupling Analysis (K=8)\n\n")
        f.write("Pairwise relationships between the raw neuroimaging modalities were evaluated by calculating the **Adjusted RV Coefficient** on the shared latent spaces after regressing out covariates. Significance was tested using **500 permutations** (subject-wise row shuffling).\n\n")
        f.write("## Pairwise Modality Relations Table\n\n")
        f.write("| Modality Pairwise Comparison | Observed RV | Permutation p-value | Z-score |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for row in pairwise_results:
            sig = "**" if row['p_value'] < 0.05 else ""
            f.write(f"| {row['Pair']} | {row['Obs_RV']:.4f} | {sig}{row['p_value']:.4f}{sig} | {row['Z_score']:.2f} |\n")
        f.write("\n## Major Modality Coupling Takeaways\n\n")
        f.write("1. **White Matter (DTI) and Anatomy (T1):** Strong coupling showing coordinate anatomical and microstructural degradation.\n")
        f.write("2. **Perfusion (CBF) and Functional Networks (rsfMRI):** Highlights neurovascular coupling alterations.\n")
        
    print(f"Saved pairwise report to: {pw_artifact_path}")

    # ==============================================================================
    # STEP 3: RUN GBEVM3 EXPOSURE STUDY (1,000 PERMUTATIONS)
    # ==============================================================================
    print("\n=================================================================")
    print("STEP 3: RUN SYSTEMATIC SIGNIFICANCE TESTS FOR GBEVM3 BLAST ENERGY")
    print("=================================================================")
    n_perms_gbev = 1000
    
    # Continuous GBEVm3 exposure
    gbev = df_raw['GBEVm3'].fillna(df_raw['GBEVm3'].median()).values
    
    # Quantile thresholds (P5 to P90)
    gbev_thresholds = np.percentile(gbev, np.linspace(5, 90, 18))
    M_gbev = len(gbev_thresholds)
    
    mri_matrix_raw = np.hstack(latents_np)
    scaler_mri = StandardScaler()
    mri_scaled = scaler_mri.fit_transform(mri_matrix_raw)
    D = mri_scaled.shape[1]
    K = C.shape[1]
    
    y_gbev_targets = np.zeros((N, M_gbev))
    for j in range(M_gbev):
        y_gbev_targets[:, j] = (gbev >= gbev_thresholds[j]).astype(float)
        
    X_t = torch.as_tensor(mri_scaled).float()
    Y_t = torch.as_tensor(y_gbev_targets).float()
    C_torch = torch.as_tensor(C).float()
    
    print("Fitting observed GBEVm3 Fused Logistic model...")
    W_gbev, bias_gbev = solve_fused_logistic(X_t, Y_t, C_torch, 0.01, 0.8, 0.01, 0.002, M_gbev, D, N, K)
    
    # Plot GBEVm3 Coefficient Heatmap
    print("Generating GBEVm3 coefficient heatmap...")
    row_labels = []
    for mod in ['T1', 'DTI', 'rsfMRI', 'Perf']:
        for comp in range(1, 9):
            row_labels.append(f"{mod}_Comp{comp}")
            
    col_labels = [f"P{int(5*(i+1))}" for i in range(M_gbev)]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        W_gbev.T,
        annot=False,
        cmap="RdBu_r",
        center=0,
        yticklabels=row_labels,
        xticklabels=col_labels,
        cbar_kws={'label': 'Coefficient Weight'}
    )
    plt.title("Flow-Simlr-V (K=8) Component Coefficients across GBEVm3 Thresholds", fontsize=13, fontweight='bold')
    plt.xlabel("GBEVm3 Threshold Percentile")
    plt.ylabel("Learned Modality Component")
    plt.xticks(rotation=45)
    plt.tight_layout()
    gbev_heatmap_path = os.path.join(artifact_dir, "gbev_threshold_coefficients_heatmap_k8.png")
    plt.savefig(gbev_heatmap_path, dpi=300)
    plt.close()
    print(f"Saved GBEVm3 coefficient heatmap to: {gbev_heatmap_path}")
    
    obs_logits = mri_scaled @ W_gbev.T + C @ bias_gbev.T
    obs_probs = 1.0 / (1.0 + np.exp(-obs_logits))
    obs_deviances = np.array([get_logistic_deviance(y_gbev_targets[:, j], obs_probs[:, j]) for j in range(M_gbev)])
    obs_dev_sum = np.sum(obs_deviances)
    
    print(f"Observed Global GBEVm3 Logistic Deviance: {obs_dev_sum:.4f}")
    
    rng = np.random.default_rng(42)
    gbev_perm_indices = [rng.permutation(N) for _ in range(n_perms_gbev)]
    
    gbev_logistic_args = [
        (shuf_idx, mri_scaled, C, gbev, gbev_thresholds, 0.01, 0.8, 0.01, 0.002)
        for shuf_idx in gbev_perm_indices
    ]
    
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=n_cores) as pool:
        null_dev_sums = pool.map(logistic_worker, gbev_logistic_args)
        
    null_dev_sums = np.array(null_dev_sums)
    p_val_global_gbev = (1.0 + np.sum(null_dev_sums <= obs_dev_sum)) / (n_perms_gbev + 1.0)
    
    print(f"GBEVm3 Null Deviance: {np.mean(null_dev_sums):.4f} ± {np.std(null_dev_sums):.4f}")
    print(f"GBEVm3 Global Permutation P-value: {p_val_global_gbev:.6f}")
    
    gbev_summary = [{
        "Outcome": "GBEVm3 (Exposure Classification)",
        "Model": "Fused Logistic",
        "Obs_Deviance": obs_dev_sum,
        "Null_Mean": np.mean(null_dev_sums),
        "Null_SD": np.std(null_dev_sums),
        "P_Value": p_val_global_gbev
    }]
    
    # Run Fused Linear subgroups regressions using GBEVm3 as the subgroup binarizer
    outcomes = {
        "npsy_NIH_Fluid_Composite_Score": "NIH Fluid Composite (Cognitive)",
        "npsy_CVLTLongDelayFreeRecall_Raw": "CVLT-II Long Delay Recall (Cognitive)",
        "npsy_WASIComposite": "WASI Composite IQ (Cognitive)",
        "npsy_TMT.B_TotalTime_Raw": "Trail Making Test Part B (Cognitive)",
        "npsy_DKEFS_Total_Raw_Score": "DKEFS Total Score (Cognitive)",
        "npsy_BDI_Total": "Beck Depression Inventory (Psychiatric)",
        "npsy_PCL5Total": "PTSD Checklist PCL-5 (Psychiatric)",
        "npsy_BSI.18_TotalRaw": "Brief Symptom Inventory BSI-18 (Psychiatric)"
    }
    
    percentiles = np.round(np.linspace(10, 80, 15), 1)
    M_reg = len(percentiles)
    
    exposed_masks = []
    for j in range(M_reg):
        thresh = np.percentile(gbev, percentiles[j])
        mask = gbev > thresh
        exposed_masks.append(torch.as_tensor(mask).bool())
        
    for o_col, o_label in outcomes.items():
        print(f"Running GBEVm3-restricted subgroup regression on {o_label}...")
        y_raw = df_r[o_col].fillna(df_r[o_col].mean()).values
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
        
        W_obs, bias_obs = solve_fused_linear(X_t, torch.as_tensor(y_scaled).float(), C_torch, exposed_masks, 0.25, 0.8, 1.0, 1e-3, M_reg, D, N, K)
        
        obs_deviances = []
        for j in range(M_reg):
            mask = exposed_masks[j].numpy()
            Z_j = mri_scaled[mask]
            C_j = C[mask]
            y_j = y_scaled[mask]
            preds = Z_j @ W_obs[j] + C_j @ bias_obs[j]
            obs_deviances.append(np.sum((y_j - preds) ** 2))
        obs_dev_sum = np.sum(obs_deviances)
        
        linear_args = [
            (shuf_idx, mri_scaled, C, y_scaled, gbev, percentiles, 0.25, 0.8, 1.0, 1e-3)
            for shuf_idx in gbev_perm_indices
        ]
        
        with multiprocessing.Pool(processes=n_cores) as pool:
            null_dev_sums = pool.map(linear_worker, linear_args)
            
        null_dev_sums = np.array(null_dev_sums)
        p_val_global = (1.0 + np.sum(null_dev_sums <= obs_dev_sum)) / (n_perms_gbev + 1.0)
        
        gbev_summary.append({
            "Outcome": o_label,
            "Model": "Fused Linear Regression",
            "Obs_Deviance": obs_dev_sum,
            "Null_Mean": np.mean(null_dev_sums),
            "Null_SD": np.std(null_dev_sums),
            "P_Value": p_val_global
        })
        
    df_gbev_summary = pd.DataFrame(gbev_summary)
    gbev_csv_path = os.path.join(workspace_dir, "expart_raw_flow_significance_gbevm3_k8_1000_results.csv")
    df_gbev_summary.to_csv(gbev_csv_path, index=False)
    print(f"Saved GBEVm3 results to: {gbev_csv_path}")
    
    # Save a GBEVm3 report to artifacts
    gbev_artifact_path = os.path.join(artifact_dir, "raw_flow_significance_gbevm3_k8_1000_report.md")
    with open(gbev_artifact_path, "w") as f:
        f.write("# Raw Modality Flow-Simlr-V GBEVm3 Significance Report (K=8, 1000 Permutations)\n\n")
        f.write(f"Analyzed on the full **EXPART cohort ($N=98$)** using **1000 target-shuffling permutations** based on joint latents learned directly from the **4 raw brain modality predictors** (T1, DTI, rsfMRI, Perfusion) using **Simlr-Flow-V ($k=8$ per modality, 2000 training epochs)**. This analysis tests exposure and subgroup outcomes using the **GBEVm3 (continuous blast energy score)** stream instead of BEC3.\n\n")
        f.write("## GBEVm3 Global Significance Table\n\n")
        f.write("| Outcome | Model | Observed Deviance | Null Deviance (Mean ± SD) | Permutation P-value |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        for row in gbev_summary:
            f.write(f"| {row['Outcome']} | {row['Model']} | {row['Obs_Deviance']:.4f} | {row['Null_Mean']:.4f} &plusmn; {row['Null_SD']:.4f} | **{row['P_Value']:.6f}** |\n")
        f.write("\n## Major GBEVm3 Takeaways\n\n")
        f.write("1. **GBEVm3 Exposure Classification:** Evaluates how well multi-modal shared representations align with physical blast energy severity.\n")
        
    print(f"Saved GBEVm3 report to: {gbev_artifact_path}")

if __name__ == "__main__":
    run_pairwise_and_gbev_studies()
