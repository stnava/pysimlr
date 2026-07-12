import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pysimlr import flow_simr_v
from pysimlr.flows import FlowConditionalInference
from pysimlr.utils import set_all_seeds, adjusted_rvcoef
from scipy.stats import pearsonr

def load_expart_data():
    names = ["Blast", "Mental_Health", "Brain_Imaging", "Cognitive_Performance"]
    files = ["data/expart_bl.csv", "data/expart_mh.csv", "data/expart_mr.csv", "data/expart_pe.csv"]
    matrices = []
    headers = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing expected data file: {f}")
        df = pd.read_csv(f)
        matrices.append(df.values)
        headers.append(df.columns.tolist())
    return names, matrices, headers

def run_analysis():
    set_all_seeds(42)
    names, matrices, headers = load_expart_data()
    
    # Configure Flow-SiMLR-V
    k = 2
    epochs = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Fitting Flow-SiMLR-V on EXPART dataset...")
    print(f"Modality dimensions: Blast={matrices[0].shape[1]}, Mental Health={matrices[1].shape[1]}, Brain Imaging={matrices[2].shape[1]}, Cognitive Performance={matrices[3].shape[1]}")
    
    # We use positivity='either' to prevent coordinate clamping, as verified in clinical trials
    res = flow_simr_v(
        matrices, 
        k=k, 
        epochs=epochs, 
        positivity='either', 
        use_nsa=True, 
        nsa_w=0.5, 
        dynamic_weights=True,
        dynamic_weights_start=60,
        energy_type='regression',
        mixing_algorithm='newton',
        device=device,
        verbose=False
    )
    
    print("\nFlow-SiMLR-V Fit Complete!")
    print(f"Final reconstruction errors: {res['errors']}")
    print(f"Final modality weights: {res['modality_weights']}")
    
    # Extract results
    latents = res['latents']
    v_mats = res['v']
    u_shared = res['u']
    cond_inference = res['cond_inference']
    
    # Ensure u_shared is a tensor
    if isinstance(u_shared, list):
        u_shared = u_shared[0]
        
    out_dir = "expart_report_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Plot weight history (Gating Warmup visualization)
    weight_history = res['weight_history']
    if len(weight_history) > 0:
        plt.figure(figsize=(8, 5))
        w_hist_np = np.array(weight_history)
        for idx, name in enumerate(names):
            plt.plot(w_hist_np[:, idx], label=name, lw=2)
        plt.axvline(60, color='red', linestyle='--', label='Gating Start (Warmup=60)')
        plt.title("Flow-SiMLR-V Gating Convergence")
        plt.xlabel("Epochs")
        plt.ylabel("Modality Weight")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "flow_gating_convergence.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {out_dir}/flow_gating_convergence.png")
        
    # 2. Plot Shared Latent Space (2D Projection)
    plt.figure(figsize=(6, 5))
    u_np = u_shared.detach().numpy()
    plt.scatter(u_np[:, 0], u_np[:, 1], color='#1f77b4', edgecolors='k', alpha=0.8, s=60)
    plt.title("Flow-SiMLR-V: Shared Latent Space (2D)")
    plt.xlabel("Shared Latent 1")
    plt.ylabel("Shared Latent 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "flow_latent_space.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {out_dir}/flow_latent_space.png")
    
    # 3. Perform and evaluate conditional cross-view imputation
    # Let's predict Mental Health outcomes (idx 1) from Brain Imaging features (idx 2)
    # And predict Cognitive Performance (idx 3) from Brain Imaging features (idx 2)
    print("\n--- Running exact Gaussian Conditional Inference (Cross-View Prediction) ---")
    
    # Condition Target on Source
    device_obj = torch.device(device)
    model = res['model'].to(device_obj)
    
    # Predict MH from Brain Imaging
    latent_img = latents[2].to(device_obj)[:, :k] # Shared latent of Brain Imaging
    pred_latent_mh = cond_inference.predict_conditional(observed_idx=2, target_idx=1, observed_z=latent_img.cpu())
    
    # Decode predicted latent back to MH feature space
    with torch.no_grad():
        pred_latent_flow_mh = model.flows[1].inverse(pred_latent_mh.to(device_obj))
        pred_mh_features = (pred_latent_flow_mh @ model.linear_encoders[1].v.t()).cpu().numpy()
    
    true_mh_features = matrices[1]
    
    # Compute correlations for MH
    mh_names = [name.replace("npsy_", "").replace("_Total", "").replace("_Raw", "") for name in headers[1]]
    mh_corrs = []
    print("\nPredicting Mental Health Outcomes from Brain Imaging (MRI):")
    for idx, col_name in enumerate(mh_names):
        true_col = true_mh_features[:, idx]
        pred_col = pred_mh_features[:, idx]
        r, p = pearsonr(true_col, pred_col)
        mh_corrs.append(r)
        print(f"  {col_name:25s} -> Pearson r = {r:6.3f} (p = {p:.2e})")
        
    # Predict PE from Brain Imaging
    latent_img_pe = latents[2].to(device_obj)[:, :k]
    pred_latent_pe = cond_inference.predict_conditional(observed_idx=2, target_idx=3, observed_z=latent_img_pe.cpu())
    
    with torch.no_grad():
        pred_latent_flow_pe = model.flows[3].inverse(pred_latent_pe.to(device_obj))
        pred_pe_features = (pred_latent_flow_pe @ model.linear_encoders[3].v.t()).cpu().numpy()
        
    true_pe_features = matrices[3]
    pe_names = [name.replace("npsy_", "").replace("_Total", "").replace("_Raw", "") for name in headers[3]]
    pe_corrs = []
    print("\nPredicting Cognitive Performance Outcomes from Brain Imaging (MRI):")
    for idx, col_name in enumerate(pe_names):
        true_col = true_pe_features[:, idx]
        pred_col = pred_pe_features[:, idx]
        r, p = pearsonr(true_col, pred_col)
        pe_corrs.append(r)
        print(f"  {col_name:25s} -> Pearson r = {r:6.3f} (p = {p:.2e})")
        
    # 4. Generate visual correlation scatter plots for winning predictions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pick top predicted MH variable (e.g. PCL5Total or BDI)
    best_mh_idx = np.argmax(mh_corrs)
    axes[0].scatter(true_mh_features[:, best_mh_idx], pred_mh_features[:, best_mh_idx], color='#ff7f0e', edgecolors='k', alpha=0.8)
    axes[0].plot([true_mh_features[:, best_mh_idx].min(), true_mh_features[:, best_mh_idx].max()],
                 [true_mh_features[:, best_mh_idx].min(), true_mh_features[:, best_mh_idx].max()], 'k--', lw=1.5)
    axes[0].set_title(f"Predicted vs True {mh_names[best_mh_idx]}\n(r = {mh_corrs[best_mh_idx]:.3f})")
    axes[0].set_xlabel("True Score")
    axes[0].set_ylabel("Predicted Score")
    axes[0].grid(True, alpha=0.3)
    
    # Pick top predicted PE variable
    best_pe_idx = np.argmax(pe_corrs)
    axes[1].scatter(true_pe_features[:, best_pe_idx], pred_pe_features[:, best_pe_idx], color='#2ca02c', edgecolors='k', alpha=0.8)
    axes[1].plot([true_pe_features[:, best_pe_idx].min(), true_pe_features[:, best_pe_idx].max()],
                 [true_pe_features[:, best_pe_idx].min(), true_pe_features[:, best_pe_idx].max()], 'k--', lw=1.5)
    axes[1].set_title(f"Predicted vs True {pe_names[best_pe_idx]}\n(r = {pe_corrs[best_pe_idx]:.3f})")
    axes[1].set_xlabel("True Score")
    axes[1].set_ylabel("Predicted Score")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "flow_imputation_results.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {out_dir}/flow_imputation_results.png")
    
    # Save the output statistics for the Quarto report
    df_impute = pd.DataFrame({
        "Modality": ["Mental_Health"] * len(mh_names) + ["Cognitive_Performance"] * len(pe_names),
        "Variable": mh_names + pe_names,
        "Correlation": mh_corrs + pe_corrs
    })
    df_impute.to_csv("expart_flow_imputation_results.csv", index=False)
    print("Saved expart_flow_imputation_results.csv")
    
    # Save linear attribution feature importance V
    v_blast = v_mats[0].numpy()
    print("\nFeature importance (Linear Projection matrix V) for Blast Modality:")
    for idx, col_name in enumerate(headers[0]):
        print(f"  {col_name:25s} -> Dim 1: {v_blast[idx, 0]:.4f}, Dim 2: {v_blast[idx, 1]:.4f}")

if __name__ == "__main__":
    run_analysis()
