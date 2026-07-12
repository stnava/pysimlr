import torch
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pysimlr import flow_simr_v
from pysimlr.flows import FlowConditionalInference
from pysimlr.utils import set_all_seeds

def run_subject_perfusion_imputation():
    set_all_seeds(42)
    print("Loading raw 98-subject cohort from expartdf_power_analysis.csv...")
    fpath = "../extern/ExpArt/data/expartdf_power_analysis.csv"
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Cannot find raw EXPART data at: {fpath}")
    
    df = pd.read_csv(fpath, low_memory=False)
    
    # Identify subject missing perfusion
    subject_id = "sub-EAS111EAS111"
    sub_row = df[df['subjectID'] == subject_id]
    other_rows = df[df['subjectID'] != subject_id]
    
    print(f"Target subject: {subject_id}")
    print(f"Other subjects: {len(other_rows)}")
    
    # Select raw imaging columns
    t1_cols = [c for c in df.columns if c.startswith('T1Hier') or c.startswith('T1w')]
    dti_cols = [c for c in df.columns if c.startswith('DTI')]
    rsf_cols = [c for c in df.columns if c.startswith('rsfMRI')]
    perf_cols = [c for c in df.columns if c.startswith('perf')]
    
    # Filter for numeric columns
    t1_cols_num = [c for c in t1_cols if df[c].dtype in [np.float64, np.int64]]
    dti_cols_num = [c for c in dti_cols if df[c].dtype in [np.float64, np.int64]]
    rsf_cols_num = [c for c in rsf_cols if df[c].dtype in [np.float64, np.int64]]
    perf_cols_num = [c for c in perf_cols if df[c].dtype in [np.float64, np.int64]]
    
    # -----------------------------------------------------------------
    # Fit PCA and Scalers on the 97 subjects (other_rows)
    # -----------------------------------------------------------------
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
    
    # Transform other subjects to get PCA features
    other_t1_pcs = pca_t1.transform(scaler_t1.transform(other_t1))
    other_dti_pcs = pca_dti.transform(scaler_dti.transform(other_dti))
    other_rsf_pcs = pca_rsf.transform(scaler_rsf.transform(other_rsf))
    other_perf_pcs = pca_perf.transform(scaler_perf.transform(other_perf))
    
    # We define the 4 views for the Flow-SiMLR-V model: T1, DTI, rsfMRI, and Perfusion
    matrices = [other_t1_pcs, other_dti_pcs, other_rsf_pcs, other_perf_pcs]
    names = ["T1w_MRI", "DTI_MRI", "rsfMRI", "Perfusion"]
    
    # Standardize PCA features
    scalers = [StandardScaler() for _ in range(4)]
    scaled_mats = [s.fit_transform(m) for s, m in zip(scalers, matrices)]
    
    k = 2
    epochs = 1500
    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(device)
    
    print("\nFitting Flow-SiMLR-V on T1, DTI, rsfMRI, and Perfusion views for 97 subjects...")
    res = flow_simr_v(
        scaled_mats,
        k=k,
        epochs=epochs,
        batch_size=batch_size,
        positivity='positive',
        use_nsa=True,
        nsa_w=0.5,
        dynamic_weights=True,
        dynamic_weights_start=300,
        energy_type='regression',
        mixing_algorithm='newton',
        device=device,
        verbose=False
    )
    
    model = res['model'].to(device_obj)
    
    # -----------------------------------------------------------------
    # Extract observed PC features for target subject EAS111
    # -----------------------------------------------------------------
    sub_t1_raw = sub_row[t1_cols_num].fillna(other_rows[t1_cols_num].mean())
    sub_dti_raw = sub_row[dti_cols_num].fillna(other_rows[dti_cols_num].mean())
    sub_rsf_raw = sub_row[rsf_cols_num].fillna(other_rows[rsf_cols_num].mean())
    
    # Transform raw features to PC space using fitted PCA models
    sub_t1_pcs = pca_t1.transform(scaler_t1.transform(sub_t1_raw))
    sub_dti_pcs = pca_dti.transform(scaler_dti.transform(sub_dti_raw))
    sub_rsf_pcs = pca_rsf.transform(scaler_rsf.transform(sub_rsf_raw))
    
    # Standardize target subject PC features using scalers fitted on the 97 subjects
    sub_t1_scaled = scalers[0].transform(sub_t1_pcs)
    sub_dti_scaled = scalers[1].transform(sub_dti_pcs)
    sub_rsf_scaled = scalers[2].transform(sub_rsf_pcs)
    
    # -----------------------------------------------------------------
    # Multi-view Imputation using Schur complement
    # We predict the latent space of Perfusion (View 3)
    # given the observed latents of T1 (View 0), DTI (View 1), and rsfMRI (View 2)
    # -----------------------------------------------------------------
    print(f"\nPerforming conditional inference to predict Perfusion for {subject_id}...")
    
    # Compute observed latent coordinates for target subject
    with torch.no_grad():
        z_t1 = torch.as_tensor(sub_t1_scaled).float().to(device_obj)
        u_t1 = model.encoders[0](model.linear_encoders[0](z_t1)).cpu().numpy()
        
        z_dti = torch.as_tensor(sub_dti_scaled).float().to(device_obj)
        u_dti = model.encoders[1](model.linear_encoders[1](z_dti)).cpu().numpy()
        
        z_rsf = torch.as_tensor(sub_rsf_scaled).float().to(device_obj)
        u_rsf = model.encoders[2](model.linear_encoders[2](z_rsf)).cpu().numpy()
        
    # Extract the shared joint latents for the 97 subjects
    joint_u = res['latents']
    u_t1_cohort = joint_u[0].detach().cpu().numpy()[:, :k]
    u_dti_cohort = joint_u[1].detach().cpu().numpy()[:, :k]
    u_rsf_cohort = joint_u[2].detach().cpu().numpy()[:, :k]
    u_perf_cohort = joint_u[3].detach().cpu().numpy()[:, :k]
    
    # Construct observed joint latent matrix for cohort
    other_u_cohort = np.hstack([u_t1_cohort, u_dti_cohort, u_rsf_cohort]) # Shape: 97, 6
    sub_u_obs = np.hstack([u_t1, u_dti, u_rsf]) # Shape: 1, 6
    
    # Compute mean and covariance parameters for Schur complement
    mu_target = np.mean(u_perf_cohort, axis=0)
    mu_obs = np.mean(other_u_cohort, axis=0)
    
    cov_target_target = np.cov(u_perf_cohort, rowvar=False)
    cov_obs_obs = np.cov(other_u_cohort, rowvar=False)
    cov_target_obs = np.cov(u_perf_cohort, other_u_cohort, rowvar=False)[:k, k:]
    
    # Solve Schur complement system
    print("Solving Schur complement system...")
    inv_cov_obs = np.linalg.inv(cov_obs_obs + np.eye(6) * 1e-6)
    pred_u_perf = mu_target + cov_target_obs @ inv_cov_obs @ (sub_u_obs - mu_obs).T
    pred_u_perf = pred_u_perf.T # Shape: 1, 2
    
    # Decode predicted perfusion latent back to Perfusion PC space
    print("Decoding the predicted latent coordinates back to Perfusion PCs...")
    with torch.no_grad():
        pred_u_perf_t = torch.as_tensor(pred_u_perf).float().to(device_obj)
        pred_flow_z = model.flows[3].inverse(pred_u_perf_t)
        pred_perf_scaled = (pred_flow_z @ model.linear_encoders[3].v.t()).cpu().numpy()
        
    # Inverse standardize back to the PCA space
    imputed_perf_pcs = scalers[3].inverse_transform(pred_perf_scaled)[0]
    
    # Print results
    print(f"\n=== Imputation Results for {subject_id} (T1 + DTI + rsfMRI -> Perfusion) ===")
    print("Imputed Perfusion PC values:")
    for i, val in enumerate(imputed_perf_pcs):
        print(f"  perfPC{i+1}: {val:.4f}")
        
    # Compare with cohort average
    cohort_mean_perf = np.mean(other_perf_pcs, axis=0)
    cohort_std_perf = np.std(other_perf_pcs, axis=0)
    print("\nCohort Perfusion PC distributions (Mean ± SD):")
    for i in range(9):
        print(f"  perfPC{i+1}: {cohort_mean_perf[i]:.4f} ± {cohort_std_perf[i]:.4f}")
        
    # Save results to a text summary
    os.makedirs("clinical_results", exist_ok=True)
    with open("clinical_results/sub_EAS111_perfusion_imputation.txt", "w") as f:
        f.write(f"Subject: {subject_id}\n")
        f.write("Imputation type: t1PC* dtPC* rsfPC* -> perfPC*\n")
        f.write("Imputed Perfusion PCs:\n")
        for i, val in enumerate(imputed_perf_pcs):
            f.write(f"  perfPC{i+1}: {val:.6f}\n")
        f.write("\nCohort Mean:\n")
        for i in range(9):
            f.write(f"  perfPC{i+1}: {cohort_mean_perf[i]:.6f}\n")
    print("\nSaved imputation summary to: clinical_results/sub_EAS111_perfusion_imputation.txt")

if __name__ == "__main__":
    run_subject_perfusion_imputation()
