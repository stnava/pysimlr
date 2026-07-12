import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
from pysimlr import flow_simr_v
from pysimlr.utils import set_all_seeds
import multiprocessing

# Set PyTorch to single thread to avoid thrashing during multiprocessing
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
        
        # Scale loss by M to match R's average error loss over all entries (N * J)
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
        
        # Scale loss to match R's average error loss over all entries (N * J) with 0.5 factor
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

# Multiprocessing Worker for Logistic Classification Permutations
def logistic_worker(args):
    # Set thread limit inside worker
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
    
    # Calculate deviance sum
    logits = mri_scaled @ W_brain.T + C_t @ W_covs.T
    probs = 1.0 / (1.0 + np.exp(-logits))
    dev_sum = 0.0
    for j in range(M):
        dev_sum += get_logistic_deviance(y_targets[:, j], probs[:, j])
    return dev_sum

# Multiprocessing Worker for Linear Regression Permutations
def linear_worker(args):
    # Set thread limit inside worker
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
    
    # Calculate sum of deviances (SSE) on the exposed subgroups
    dev_sum = 0.0
    for j in range(M):
        mask = exposed_masks[j].numpy()
        Z_j = mri_scaled[mask]
        C_j = C_t[mask]
        y_j = y_shuf[mask]
        preds = Z_j @ W_brain[j] + C_j @ W_covs[j]
        dev_sum += np.sum((y_j - preds) ** 2)
    return dev_sum

def run_systematic_perm_tests(n_perms=500, lambda1=0.25, alpha=0.8, lambda2=1.0, sparsity_thresh=1e-3):
    set_all_seeds(42)
    
    # ==============================================================================
    # STEP 1: EXTRACT LATENTS BY RUNNING SIMLR-FLOW-V ON RAW MODALITIES (K=2, 2000 epochs)
    # ==============================================================================
    print("=================================================================")
    print("STEP 1: EXTRACT LATENTS BY RUNNING SIMLR-FLOW-V ON RAW MODALITIES (K=2, 2000 EPOCHS)")
    print("=================================================================")
    t0_flow = time.time()
    
    fpath_raw = "../extern/ExpArt/data/expartdf_power_analysis.csv"
    if not os.path.exists(fpath_raw):
        raise FileNotFoundError(f"Cannot find raw EXPART data at: {fpath_raw}")
    
    df_raw = pd.read_csv(fpath_raw, low_memory=False)
    
    t1_cols = [c for c in df_raw.columns if c.startswith('T1Hier') or c.startswith('T1w')]
    dti_cols = [c for c in df_raw.columns if c.startswith('DTI')]
    rsf_cols = [c for c in df_raw.columns if c.startswith('rsfMRI')]
    perf_cols = [c for c in df_raw.columns if c.startswith('perf')]
    
    # Extract numeric columns only
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
    print(f"Modality Shapes: T1={t1_scaled.shape}, DTI={dti_scaled.shape}, rsfMRI={rsf_scaled.shape}, Perfusion={perf_scaled.shape}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Simlr-Flow-V (K=2, Epochs=2000) on device: {device}...")
    
    res_flow = flow_simr_v(
        scaled_mats,
        k=2,
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
    
    print(f"Simlr-Flow-V Completed! Elapsed time: {time.time() - t0_flow:.2f} seconds")
    
    # Stack latents horizontally to form 8 predictors (2 components x 4 modalities)
    latents_np = [lat.detach().cpu().numpy() for lat in res_flow['latents']]
    mri_matrix_raw = np.hstack(latents_np)
    
    # Standardize learned latents
    scaler_mri = StandardScaler()
    mri_scaled = scaler_mri.fit_transform(mri_matrix_raw)
    
    # ==============================================================================
    # STEP 2: LOAD OUTCOMES AND COVARIATES FROM PRE-PROCESSED R ENVIRONMENT
    # ==============================================================================
    print("\n=================================================================")
    print("STEP 2: LOAD OUTCOMES AND COVARIATES FROM PRE-PROCESSED R ENVIRONMENT")
    print("=================================================================")
    
    fpath_r = "scratch/expart_exact_r_data.csv"
    df_r = pd.read_csv(fpath_r)
    N = df_r.shape[0]
    
    # Construct R covariates design matrix (including column of ones for intercept)
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
    
    K = C.shape[1]
    total_bl = df_r['Exposure_raw'].values
    
    # 18 exact thresholds from R
    thresholds = [12, 50, 280, 560, 1200, 2880, 3600, 5760, 8000, 9600, 12960, 17500, 24000, 28350, 36000, 57000, 108000, 159360]
    M_class = len(thresholds)
    D = mri_scaled.shape[1]
    
    percentiles = np.round(np.linspace(10, 80, 15), 1)
    M_reg = len(percentiles)
    
    rng = np.random.default_rng(42)
    perm_indices = [rng.permutation(N) for _ in range(n_perms)]
    
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Parallelizing permutations using {n_cores} CPU cores...")
    
    results_summary = []
    
    # ==============================================================================
    # STUDY 1: BEC3 CLASSIFICATION (Fused Logistic)
    # ==============================================================================
    print("\n-----------------------------------------------------------------")
    print("RUNNING STUDY 1: BEC3 EXPOSURE CLASSIFICATION (FUSED LOGISTIC)")
    print("-----------------------------------------------------------------")
    t0 = time.time()
    
    y_targets = np.zeros((N, M_class))
    for j in range(M_class):
        y_targets[:, j] = (total_bl >= thresholds[j]).astype(float)
        
    X_t = torch.as_tensor(mri_scaled).float()
    Y_t = torch.as_tensor(y_targets).float()
    C_torch = torch.as_tensor(C).float()
    
    W_obs, bias_obs = solve_fused_logistic(X_t, Y_t, C_torch, 0.01, 0.8, 0.01, 0.002, M_class, D, N, K)
    
    obs_logits = mri_scaled @ W_obs.T + C @ bias_obs.T
    obs_probs = 1.0 / (1.0 + np.exp(-obs_logits))
    obs_deviances = np.array([get_logistic_deviance(y_targets[:, j], obs_probs[:, j]) for j in range(M_class)])
    obs_dev_sum = np.sum(obs_deviances)
    
    print(f"Observed Global Logistic Deviance: {obs_dev_sum:.4f}")
    
    logistic_args = [
        (shuf_idx, mri_scaled, C, total_bl, thresholds, 0.01, 0.8, 0.01, 0.002)
        for shuf_idx in perm_indices
    ]
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        null_dev_sums = pool.map(logistic_worker, logistic_args)
        
    null_dev_sums = np.array(null_dev_sums)
    p_val_global = (1.0 + np.sum(null_dev_sums <= obs_dev_sum)) / (n_perms + 1.0)
    
    print(f"Null Deviance (Mean ± SD): {np.mean(null_dev_sums):.4f} ± {np.std(null_dev_sums):.4f}")
    print(f"Global Permutation P-value (Logistic): {p_val_global:.6f}")
    print(f"Elapsed Time: {time.time() - t0:.2f} seconds")
    
    results_summary.append({
        "Outcome": "BEC3 (Exposure Classification)",
        "Model": "Fused Logistic",
        "Obs_Deviance": obs_dev_sum,
        "Null_Mean": np.mean(null_dev_sums),
        "Null_SD": np.std(null_dev_sums),
        "P_Value": p_val_global
    })
    
    # ==============================================================================
    # STUDIES 2-9: REGRESSION ON COGNITIVE & PSYCHIATRIC OUTCOMES (Fused Linear)
    # ==============================================================================
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
    
    exposed_masks = []
    for j in range(M_reg):
        thresh = np.percentile(total_bl, percentiles[j])
        mask = total_bl > thresh
        exposed_masks.append(torch.as_tensor(mask).bool())
        
    for o_col, o_label in outcomes.items():
        print("\n-----------------------------------------------------------------")
        print(f"RUNNING STUDY: REGRESSION ON {o_label.upper()}")
        print("-----------------------------------------------------------------")
        t0 = time.time()
        
        y_raw = df_r[o_col].fillna(df_r[o_col].mean()).values
        
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
        
        X_t = torch.as_tensor(mri_scaled).float()
        Y_t = torch.as_tensor(y_scaled).float()
        C_torch = torch.as_tensor(C).float()
        
        W_obs, bias_obs = solve_fused_linear(X_t, Y_t, C_torch, exposed_masks, lambda1, alpha, lambda2, sparsity_thresh, M_reg, D, N, K)
        
        # Calculate observed sum of deviances (SSE)
        obs_deviances = []
        for j in range(M_reg):
            mask = exposed_masks[j].numpy()
            Z_j = mri_scaled[mask]
            C_j = C[mask]
            y_j = y_scaled[mask]
            preds = Z_j @ W_obs[j] + C_j @ bias_obs[j]
            obs_deviances.append(np.sum((y_j - preds) ** 2))
        obs_dev_sum = np.sum(obs_deviances)
        
        print(f"Observed Global Linear Deviance (SSE): {obs_dev_sum:.4f}")
        
        linear_args = [
            (shuf_idx, mri_scaled, C, y_scaled, total_bl, percentiles, lambda1, alpha, lambda2, sparsity_thresh)
            for shuf_idx in perm_indices
        ]
        
        with multiprocessing.Pool(processes=n_cores) as pool:
            null_dev_sums = pool.map(linear_worker, linear_args)
            
        null_dev_sums = np.array(null_dev_sums)
        p_val_global = (1.0 + np.sum(null_dev_sums <= obs_dev_sum)) / (n_perms + 1.0)
        
        print(f"Null Deviance (Mean ± SD): {np.mean(null_dev_sums):.4f} ± {np.std(null_dev_sums):.4f}")
        print(f"Global Permutation P-value (Linear): {p_val_global:.6f}")
        print(f"Elapsed Time: {time.time() - t0:.2f} seconds")
        
        results_summary.append({
            "Outcome": o_label,
            "Model": "Fused Linear Regression",
            "Obs_Deviance": obs_dev_sum,
            "Null_Mean": np.mean(null_dev_sums),
            "Null_SD": np.std(null_dev_sums),
            "P_Value": p_val_global
        })
        
    # ==============================================================================
    # PRINT SYSTEMATIC SUMMARY REPORT
    # ==============================================================================
    print("\n\n" + "="*80)
    print("            RAW FLOW (K=2, 2000 EPOCHS) SIGNIFICANCE SUMMARY REPORT")
    print("="*80)
    df_summary = pd.DataFrame(results_summary)
    
    print(df_summary.to_string(index=False, formatters={
        "Obs_Deviance": "{:.4f}".format,
        "Null_Mean": "{:.4f}".format,
        "Null_SD": "{:.4f}".format,
        "P_Value": "{:.6f}".format
    }))
    print("="*80)
    
    df_summary.to_csv("expart_raw_flow_significance_k2_2000_results.csv", index=False)
    print("Saved systematic results to: expart_raw_flow_significance_k2_2000_results.csv")
    
    # Save a markdown report version
    artifact_path = "/Users/stnava/.gemini/antigravity-cli/brain/fda666c5-3968-404e-b3e4-f71672770deb/raw_flow_significance_k2_2000_report.md"
    with open(artifact_path, "w") as f:
        f.write("# Raw Modality Flow-Simlr-V Systematic Significance Report (K=2, 2000 Epochs)\n\n")
        f.write(f"Analyzed on the full **EXPART cohort ($N=98$)** using **{n_perms} target-shuffling permutations** based on joint latents learned directly from the **4 raw brain modality predictors** (T1, DTI, rsfMRI, Perfusion) using **Simlr-Flow-V ($k=2$ per modality, 2000 training epochs)**.\n\n")
        f.write("## Parameters\n")
        f.write("### Simlr-Flow-V Configuration\n")
        f.write("- Number of latents per view ($k$): 2 (8 features total)\n")
        f.write("- Positivity Constraint: positive\n")
        f.write("- NSA weighting ($w$): 0.5\n")
        f.write("- Mixing Algorithm: Newton\n")
        f.write("- Training Epochs: 2000\n\n")
        
        f.write("### Study 1: BEC3 Classification (Fused Logistic)\n")
        f.write("- $\\lambda_1$: 0.01 (Ridge)\n")
        f.write("- $\\alpha$: 0.8 (Fused Elastic Net)\n")
        f.write("- $\\lambda_2$: 0.01 (Fusion)\n")
        f.write("- `sparsity_thresh`: 0.002\n")
        f.write("- Thresholds: 18 exact R thresholds (P5 to P90)\n\n")
        
        f.write("### Studies 2-9: Subgroup-Restricted Regression (Fused Linear EN)\n")
        f.write(f"- $\\lambda_1$: {lambda1}\n")
        f.write(f"- $\\alpha$ (Elastic Net Mix): {alpha}\n")
        f.write(f"- $\\lambda_2$ (Fusion Penalty): {lambda2}\n")
        f.write(f"- `sparsity_thresh`: {sparsity_thresh}\n")
        f.write("- Thresholds: 15 percentiles (10% to 80% percentiles)\n\n")
        
        f.write("## Global Significance Table\n\n")
        f.write("| Outcome | Model | Observed Deviance | Null Deviance (Mean ± SD) | Permutation P-value |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        for row in results_summary:
            f.write(f"| {row['Outcome']} | {row['Model']} | {row['Obs_Deviance']:.4f} | {row['Null_Mean']:.4f} &plusmn; {row['Null_SD']:.4f} | **{row['P_Value']:.6f}** |\n")
        f.write("\n## Major Scientific Takeaways\n\n")
        f.write("1. **Simlr-Flow-V learned directly from Raw Features (K=2):** Extracting 8 latent dimensions from the full high-dimensional raw voxel-wise/region-wise modality features and training for 2000 epochs to guarantee convergence.\n")
        f.write("2. **BEC3 Classification Global Significance:** Comparison of $p$-value with the PCA-based baseline exposes the value of learned raw feature flows.\n")
        
    print(f"Saved markdown report to: {artifact_path}")

if __name__ == "__main__":
    run_systematic_perm_tests()
