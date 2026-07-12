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
import multiprocessing

# Global worker function for multiprocessing
def run_single_permutation(args):
    perm_idx, bl_res, bl_pred, mh_res, mh_pred, mr_res, mr_pred, pe_res, pe_pred, X_cov, epochs_flow = args
    import torch
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from pysimlr import flow_simr_v
    from pysimlr.utils import adjusted_rvcoef
    
    # Restrict PyTorch to a single thread to prevent CPU thread oversubscription
    torch.set_num_threads(1)
    
    # Shuffle residuals independently (global null for edges)
    rng = np.random.default_rng(42 + perm_idx)
    bl_res_p = rng.permutation(bl_res, axis=0)
    mh_res_p = rng.permutation(mh_res, axis=0)
    mr_res_p = rng.permutation(mr_res, axis=0)
    pe_res_p = rng.permutation(pe_res, axis=0)
    
    # Reconstruct views
    bl_null = bl_pred + bl_res_p
    mh_null = mh_pred + mh_res_p
    mr_null = mr_pred + mr_res_p
    pe_null = pe_pred + pe_res_p
    
    # Re-residualize
    def re_residualize(y, X):
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        return y - X @ beta
        
    bl_adj = re_residualize(bl_null, X_cov)
    mh_adj = re_residualize(mh_null, X_cov)
    mr_adj = re_residualize(mr_null, X_cov)
    pe_adj = re_residualize(pe_null, X_cov)
    
    # Standardize
    scaled_bl = StandardScaler().fit_transform(bl_adj)
    scaled_mh = StandardScaler().fit_transform(mh_adj)
    scaled_mr = StandardScaler().fit_transform(mr_adj)
    scaled_pe = StandardScaler().fit_transform(pe_adj)
    
    scaled_mats = [scaled_bl, scaled_mh, scaled_mr, scaled_pe]
    
    # Fit Flow-SiMLR-V (force CPU device for safety in multiprocessing)
    res = flow_simr_v(
        scaled_mats,
        k=2,
        epochs=epochs_flow,
        batch_size=128,
        positivity='positive',
        use_nsa=True,
        nsa_w=0.5,
        dynamic_weights=True,
        dynamic_weights_start=200,
        energy_type='regression',
        mixing_algorithm='newton',
        device='cpu',
        verbose=False,
        use_rank_mai=False
    )
    
    # Calculate RVs of linear projections
    proj = [scaled_mats[i] @ res['v'][i].numpy() for i in range(4)]
    
    rv_results = {}
    names = ["Blast", "Mental_Health", "Brain_Imaging", "Cognitive_Performance"]
    for i in range(4):
        for j in range(i+1, 4):
            rv = adjusted_rvcoef(torch.as_tensor(proj[i]).float(), torch.as_tensor(proj[j]).float())
            rv_results[f"{names[i]}__{names[j]}"] = rv
            
    return rv_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perms", type=int, default=100, help="Number of permutations to run")
    args = parser.parse_args()
    n_perms = args.perms
    
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
    epochs_flow = 800
    
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
    
    # Calculate observed coupling for each edge
    proj_obs = [scaled_mats_obs[i] @ res_obs['v'][i].cpu().numpy() for i in range(4)]
    
    names = ["Blast", "Mental_Health", "Brain_Imaging", "Cognitive_Performance"]
    obs_rvs = {}
    for i in range(4):
        for j in range(i+1, 4):
            key = f"{names[i]}__{names[j]}"
            rv = adjusted_rvcoef(torch.as_tensor(proj_obs[i]).float(), torch.as_tensor(proj_obs[j]).float())
            obs_rvs[key] = rv
            print(f"Observed {names[i]} <-> {names[j]} Adjusted RV: {rv:.4f}")
            
    # -----------------------------------------------------------------
    # Post-hoc Permutation Edge Significance (no refitting, N=1000)
    # -----------------------------------------------------------------
    print("\n--- Running Post-hoc Permutation Edge Significance Tests (N=1000, no refitting) ---")
    posthoc_nulls = {key: [] for key in obs_rvs.keys()}
    rng = np.random.default_rng(42)
    
    for _ in range(1000):
        # Independently shuffle all 4 projection matrices
        proj_shuffled = [rng.permutation(p, axis=0) for p in proj_obs]
        for i in range(4):
            for j in range(i+1, 4):
                key = f"{names[i]}__{names[j]}"
                rv_p = adjusted_rvcoef(torch.as_tensor(proj_shuffled[i]).float(), torch.as_tensor(proj_shuffled[j]).float())
                posthoc_nulls[key].append(rv_p)
                
    posthoc_ps = {}
    for key, val in obs_rvs.items():
        nulls = np.array(posthoc_nulls[key])
        p_val = (1.0 + np.sum(nulls >= val)) / 1001.0
        posthoc_ps[key] = p_val
        print(f"  Post-hoc {key.replace('__', ' <-> ')}: p = {p_val:.4f}")
        
    # -----------------------------------------------------------------
    # Rigorous Refit-under-Permutation Edge Significance (Full Flow-SiMLR-V)
    # -----------------------------------------------------------------
    print(f"\n--- Running Parallelized Permutation Edge Tests ({n_perms} perms, refitting flow each time) ---")
    
    X_cov = np.hstack([np.ones((covariates.shape[0], 1)), covariates])
    
    # Prepare arguments for multiprocessing
    args_list = []
    for b in range(n_perms):
        args_list.append((
            b, bl_res, bl_pred, mh_res, mh_pred, mr_res, mr_pred, pe_res, pe_pred, X_cov, epochs_flow
        ))
        
    n_cores = min(6, multiprocessing.cpu_count())
    print(f"Spawning Pool with {n_cores} parallel CPU processes...")
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        # Map tasks across processes
        results = pool.map(run_single_permutation, args_list)
        
    print("Permutations complete. Analyzing distributions...")
    
    refit_nulls = {key: [] for key in obs_rvs.keys()}
    for res_dict in results:
        for key in obs_rvs.keys():
            refit_nulls[key].append(res_dict[key])
            
    refit_ps = {}
    for key, val in obs_rvs.items():
        nulls = np.array(refit_nulls[key])
        p_val = (1.0 + np.sum(nulls >= val)) / (n_perms + 1.0)
        refit_ps[key] = p_val
        print(f"  Refit-Flow {key.replace('__', ' <-> ')}: p = {p_val:.4f}")
        
    # -----------------------------------------------------------------
    # Save statistics and generate detailed 2x3 panel plot
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    keys = list(obs_rvs.keys())
    for idx, key in enumerate(keys):
        ax = axes[idx]
        obs_val = obs_rvs[key]
        
        post_n = posthoc_nulls[key]
        ref_n = refit_nulls[key]
        
        ax.hist(post_n, bins=20, alpha=0.5, label="Post-hoc Null (No Refit)", color='blue', edgecolor='k')
        ax.hist(ref_n, bins=10, alpha=0.5, label="Rigorous Null (Refit Flow)", color='green', edgecolor='k')
        ax.axvline(obs_val, color='red', linestyle='--', lw=2.5, label=f"Observed RV = {obs_val:.4f}")
        
        edge_name = key.replace("__", " <-> ")
        ax.set_title(f"{edge_name}\nPost-hoc p = {posthoc_ps[key]:.3f} | Refit p = {refit_ps[key]:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Adjusted RV Coefficient")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    os.makedirs("expart_report_figures", exist_ok=True)
    plt.savefig("expart_report_figures/flow_edge_significance.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("\nSaved detailed edge comparison figure: expart_report_figures/flow_edge_significance.png")
    
    # Save results to a CSV file
    summary_records = []
    for key in obs_rvs.keys():
        summary_records.append({
            "Modality_Edge": key.replace("__", " <-> "),
            "Observed_Adjusted_RV": obs_rvs[key],
            "Post_hoc_P_Value": posthoc_ps[key],
            "Refit_P_Value": refit_ps[key]
        })
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv("expart_flow_edge_significance_results.csv", index=False)
    print("Saved comparison summary to: expart_flow_edge_significance_results.csv")
    print("\n=== Edge Significance Table ===")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
