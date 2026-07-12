import torch
import pandas as pd
import numpy as np
from pysimlr import lend_simr, create_path_graph
import os

def pathwise_permutation_test(data, k, topology="graph", path_graph=None, n_perms=50, **kwargs):
    # 1. Observed Fit
    obs_res = lend_simr(data, k=k, topology=topology, path_graph=path_graph, **kwargs)
    obs_sim_loss = np.mean(obs_res["sim_history"][-5:])
    
    # 2. Permutations
    null_sim_losses = []
    for p in range(n_perms):
        perm_data = [m[np.random.permutation(m.shape[0])] for m in data]
        perm_res = lend_simr(perm_data, k=k, topology=topology, path_graph=path_graph, **kwargs)
        null_sim_losses.append(np.mean(perm_res["sim_history"][-5:]))
        if (p+1) % 10 == 0: print(f"  Completed {p+1}/50...")
        
    # 3. Stats
    null_dist = np.array(null_sim_losses)
    p_val = np.mean(null_dist <= obs_sim_loss)
    z_score = (np.mean(null_dist) - obs_sim_loss) / (np.std(null_dist) + 1e-8)
    
    return {
        "obs_sim_loss": obs_sim_loss,
        "p_val": p_val,
        "z_score": z_score,
        "null_mean": np.mean(null_dist)
    }

def run_fresh_sweep():
    torch.manual_seed(42)
    np.random.seed(42)

    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    data = [bl, mh, mr, pe]
    
    models = {
        "Sequential Chain (BL->MR->PE->MH)": {"topology": "graph", "path_graph": create_path_graph([(0, 2), (2, 3), (3, 1)], 4)},
        "User Hypothesis (BL->MR->PE/MH)": {"topology": "graph", "path_graph": create_path_graph([(0, 2), (2, 3), (2, 1)], 4)},
        "LOO (Leave-One-Out)": {"topology": "loo", "path_graph": None},
        "Star (Full Connectivity)": {"topology": "star", "path_graph": None}
    }
    
    results = []
    for name, config in models.items():
        print(f"Sweep: Evaluating {name}...")
        stats = pathwise_permutation_test(
            data, k=2, topology=config["topology"], path_graph=config["path_graph"], 
            n_perms=50, epochs=100, energy_type="acc", mixing_algorithm="svd", 
            sparseness_quantile=0.5, sim_weight=2.0, verbose=False
        )
        stats["Model"] = name
        results.append(stats)
        
    df = pd.DataFrame(results)
    df.to_csv("expart_structural_comparison.csv", index=False)
    print("\nSweep Complete. Results saved to expart_structural_comparison.csv")

if __name__ == "__main__":
    run_fresh_sweep()
