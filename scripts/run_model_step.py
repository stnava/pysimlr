import torch
import pandas as pd
import numpy as np
from pysimlr import lend_simr, create_path_graph
import os
import sys

def run_single_model_test(model_name, config, n_perms=50):
    torch.manual_seed(42)
    np.random.seed(42)

    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    data = [bl, mh, mr, pe]
    
    k = 2
    # OPTIMIZED "WINNER" PARAMETERS
    kwargs = {
        "epochs": 200, 
        "nsa_iterations": 10,
        "energy_type": "acc", 
        "mixing_algorithm": "newton", 
        "sparseness_quantile": 0.5, 
        "sim_weight": 2.0, 
        "verbose": False,
        "optimizer_type": "larslow",
        "dynamic_weights": False,
        "warmup_epochs": 20
    }
    
    # Observed Fit
    print(f"Fitting observed model: {model_name}...")
    obs_res = lend_simr(data, k=k, topology=config["topology"], path_graph=config["path_graph"], **kwargs)
    obs_sim_loss = np.mean(obs_res["sim_history"][-5:])
    
    # Permutations
    null_sim_losses = []
    for p in range(n_perms):
        perm_data = [m[np.random.permutation(m.shape[0])] for m in data]
        perm_res = lend_simr(perm_data, k=k, topology=config["topology"], path_graph=config["path_graph"], **kwargs)
        null_sim_losses.append(np.mean(perm_res["sim_history"][-5:]))
        if (p+1) % 10 == 0: print(f"  {model_name}: {p+1}/{n_perms} permutations complete...")
        
    null_dist = np.array(null_sim_losses)
    p_val = np.mean(null_dist <= obs_sim_loss)
    z_score = (np.mean(null_dist) - obs_sim_loss) / (np.std(null_dist) + 1e-8)
    
    return {
        "Model": model_name,
        "obs_sim_loss": obs_sim_loss,
        "p_val": p_val,
        "z_score": z_score,
        "null_mean": np.mean(null_dist)
    }

def main():
    if len(sys.argv) < 2: return
    name = sys.argv[1]
    
    # 0:BL, 1:MH, 2:MR, 3:PE
    models = {
        "Undirected Chain": {
            "topology": "graph", 
            "path_graph": create_path_graph([(0, 2), (2, 3), (3, 1)], 4)
        },
        "Forward Chain": {
            "topology": "graph", 
            "path_graph": {0:[], 2:[0], 3:[2], 1:[3]}
        },
        "Reverse Chain": {
            "topology": "graph", 
            "path_graph": {1:[], 3:[1], 2:[3], 0:[2]}
        },
        "User Hypothesis": {
            "topology": "graph", 
            "path_graph": create_path_graph([(0, 2), (2, 3), (2, 1)], 4)
        },
        "Star": {"topology": "star", "path_graph": None}
    }
    
    res = run_single_model_test(name, models[name])
    df = pd.DataFrame([res])
    header = not os.path.exists("expart_structural_comparison.csv")
    df.to_csv("expart_structural_comparison.csv", mode='a', index=False, header=header)
    print(f"Results for {name} saved.")

if __name__ == "__main__":
    main()
