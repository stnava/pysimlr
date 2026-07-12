import torch
import pandas as pd
import numpy as np
from pysimlr import lend_simr, create_path_graph
from scipy.stats import norm

def pathwise_permutation_test(data, k, graph, n_perms=20, **kwargs):
    # 1. Observed Fit
    print("    Fitting observed model...")
    obs_res = lend_simr(data, k=k, topology="graph", path_graph=graph, **kwargs)
    obs_sim_loss = np.mean(obs_res["sim_history"][-5:])
    obs_total_loss = np.mean(obs_res["loss_history"][-5:])
    
    # 2. Permutations
    null_sim_losses = []
    for p in range(n_perms):
        if (p+1) % 5 == 0 or p == 0:
            print(f"    Permutation {p+1}/{n_perms}...")
        
        # Shuffle each modality independently to break cross-modality structure
        perm_data = [m[np.random.permutation(m.shape[0])] for m in data]
        perm_res = lend_simr(perm_data, k=k, topology="graph", path_graph=graph, **kwargs)
        
        # We record the final alignment loss. (Lower is better)
        null_sim_losses.append(np.mean(perm_res["sim_history"][-5:]))
        
    # 3. Stats
    null_dist = np.array(null_sim_losses)
    # P-value: probability that random data aligns better (has lower loss) than observed
    p_val = np.mean(null_dist <= obs_sim_loss)
    # Z-score: How many standard deviations *below* the null mean is our observed loss
    z_score = (np.mean(null_dist) - obs_sim_loss) / (np.std(null_dist) + 1e-8)
    
    return {
        "obs_sim_loss": obs_sim_loss,
        "obs_total_loss": obs_total_loss,
        "p_val": p_val,
        "z_score": z_score,
        "null_mean": np.mean(null_dist),
        "null_std": np.std(null_dist)
    }

def run_pathwise_analysis():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    data = [bl, mh, mr, pe]
    
    # Modalities: 0:BL, 1:MH, 2:MR, 3:PE
    models = {
        "User Hypothesis (BL->MR->PE/MH)": create_path_graph([(0, 2), (2, 3), (2, 1)], 4),
        "Sequential Chain (BL->MR->PE->MH)": create_path_graph([(0, 2), (2, 3), (3, 1)], 4)
    }
    
    results = {}
    for name, graph in models.items():
        print(f"\nEvaluating Path Model: {name}")
        stats = pathwise_permutation_test(
            data, k=3, graph=graph, n_perms=30, # Increased permutations for more robust z-scores
            epochs=100, energy_type="acc", mixing_algorithm="svd", 
            sparseness_quantile=0.5, sim_weight=2.0, verbose=False
        )
        results[name] = stats
        
    print("\n" + "="*85)
    print("GLOBAL PATHWISE SIGNIFICANCE REPORT (LEND ARCHITECTURE)")
    print("="*85)
    print(f"{'Path Model':35} | {'Sim Loss':9} | {'Null Mean':9} | {'Z-score':7} | {'p-value':7}")
    print("-" * 85)
    for name, s in results.items():
        sig = "*" if s["p_val"] < 0.05 else " "
        print(f"{name:35} | {s['obs_sim_loss']:9.4f} | {s['null_mean']:9.4f} | {s['z_score']:7.2f} | {s['p_val']:7.4f} {sig}")
    print("="*85)

if __name__ == "__main__":
    run_pathwise_analysis()
