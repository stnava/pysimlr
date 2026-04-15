import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pysimlr.utils import invariant_orthogonality_defect
from pysimlr.benchmarks.synthetic_cases import build_case
from sklearn.preprocessing import StandardScaler

def run_rigorous_nsa_experiment(n_runs=10):
    # 1. Setup Data Case (Nonlinear Polynomial)
    print("Building Nonlinear (Polynomial) case...")
    case = build_case("nonlinear", regime="polynomial", n_samples=800, shared_k=3)
    # We take the first modality for simplicity in weight analysis
    X_raw = case["data"][0].numpy()
    
    # 2. Train/Test Split (70/30)
    n_samples = X_raw.shape[0]
    n_train = int(0.7 * n_samples)
    
    alphas = np.linspace(0, 1, 20)
    all_v_defects = np.zeros((n_runs, len(alphas)))
    all_z_defects = np.zeros((n_runs, len(alphas)))
    
    print(f"Running {n_runs} seeds to quantify impact...")
    for run in range(n_runs):
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)
        
        # Shuffle and split
        indices = np.random.permutation(n_samples)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        # Scale based on train, apply to test
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_idx])
        X_test = scaler.transform(X_raw[test_idx])
        
        n_features = X_train.shape[1]
        n_latent = case["shared_k"]
        
        # 3. Setup Weights (Raw vs Projected)
        V_raw = np.random.randn(n_features, n_latent)
        U, S, Vh = np.linalg.svd(V_raw, full_matrices=False)
        V_proj = U @ Vh # Perfect Stiefel manifold
        
        for i, alpha in enumerate(alphas):
            # Mechanical Contract Interpolation
            V_active = (1 - alpha) * V_raw + alpha * V_proj
            
            # Metric 1: Weight Defect (V)
            v_defect = invariant_orthogonality_defect(V_active).item()
            all_v_defects[run, i] = v_defect
            
            # Metric 2: Latent Defect (XV Test)
            Z_test = X_test @ V_active
            z_defect = invariant_orthogonality_defect(Z_test).item()
            all_z_defects[run, i] = z_defect

    # 4. Statistical Aggregation
    v_mean = np.mean(all_v_defects, axis=0)
    v_std = np.std(all_v_defects, axis=0)
    z_mean = np.mean(all_z_defects, axis=0)
    z_std = np.std(all_z_defects, axis=0)

    # 5. Plotting
    os.makedirs('output_figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Plot Weight Defect
    plt.plot(alphas, v_mean, color='darkblue', lw=2, label='Weight Defect (V)')
    plt.fill_between(alphas, v_mean - v_std, v_mean + v_std, color='darkblue', alpha=0.1)
    
    # Plot Latent Defect (Test)
    plt.plot(alphas, z_mean, color='darkred', lw=2, label='Latent Defect (XV Test)')
    plt.fill_between(alphas, z_mean - z_std, z_mean + z_std, color='darkred', alpha=0.1)
    
    # Tight y-axis as requested
    max_val = max(np.max(v_mean + v_std), np.max(z_mean + z_std))
    plt.ylim(-0.02 * max_val, max_val * 1.1)
    
    plt.xlabel('Mixing Alpha (NSA-Flow Adherence Weight)')
    plt.ylabel('Invariant Orthogonality Defect (Mean ± SD)')
    plt.title('Rigorous Manifold Adherence Audit: V vs XV (Test Data)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.savefig('output_figures/nsa_weight_impact.png')
    plt.close()
    
    print("\nRigorous figure (Mean ± SD, Nonlinear Case) saved to output_figures/nsa_weight_impact.png")
    print(f"Final Alpha=1.0 Weights Defect: {v_mean[-1]:.2e} ± {v_std[-1]:.2e}")
    print(f"Final Alpha=1.0 Test Latent Defect: {z_mean[-1]:.2e} ± {z_std[-1]:.2e}")

if __name__ == "__main__":
    run_rigorous_nsa_experiment()
