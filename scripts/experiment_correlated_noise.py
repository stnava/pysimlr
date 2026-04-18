import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pysimlr.deep import lend_simr
from pysimlr.utils import procrustes_r2

def build_correlated_noise_case(n_samples=500, k=2, m=5, rho=0.0, sigma=2.0):
    """
    m modalities with shared signal U and correlated noise.
    Using higher sigma to see the floor.
    """
    u_true = torch.randn(n_samples, k)
    
    # Shared noise component
    shared_noise = torch.randn(n_samples, 100) # common feature noise
    
    data_matrices = []
    for i in range(m):
        v = torch.randn(k, 100)
        # Indep noise
        indep_noise = torch.randn(n_samples, 100)
        
        # Total noise = sqrt(1-rho)*indep + sqrt(rho)*shared
        noise = (np.sqrt(1-rho) * indep_noise + np.sqrt(rho) * shared_noise) * sigma
        
        x = u_true @ v + noise
        data_matrices.append(x)
        
    return data_matrices, u_true

def experiment_correlated_noise():
    rhos = [0.0, 0.2, 0.5, 0.8]
    ms = [2, 5, 10, 15, 20]
    results = []
    
    for rho in rhos:
        for m in ms:
            data, u_true = build_correlated_noise_case(m=m, rho=rho, sigma=2.0)
            
            # Simple SiMLR is faster for this check
            from pysimlr import simlr
            res = simlr(data, k=2, iterations=100, verbose=False)
            u_est = res['u']
            
            r2 = procrustes_r2(u_true, torch.as_tensor(u_est))
            mse = 1.0 - r2
            
            results.append({"rho": rho, "m": m, "MSE": mse})
            print(f"rho={rho}, m={m}, MSE={mse:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("paper/results_cache/correlated_noise_results.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="m", y="MSE", hue="rho", marker='o')
    plt.title("Latent Recovery Floor under Correlated Noise (High Noise Regime)", fontweight='bold')
    plt.xlabel("Number of Modalities (M)")
    plt.ylabel("Recovery Error (1 - R2_U)")
    plt.grid(True)
    plt.savefig("paper/figures/correlated_noise_plot.png")
    plt.close()

if __name__ == "__main__":
    experiment_correlated_noise()
