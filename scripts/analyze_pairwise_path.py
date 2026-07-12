import torch
import pandas as pd
import numpy as np
from pysimlr import lend_simr, create_path_graph
from pysimlr.utils import adjusted_rvcoef

def run_pairwise_significance(n_perms=100):
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load Data
    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    data = [bl, mh, mr, pe]
    modality_names = ["BL", "MH", "MR", "PE"]
    
    # 2. Sequential Chain Config
    graph_chain = create_path_graph([(0, 2), (2, 3), (3, 1)], 4)
    k = 2
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
    
    # 3. Observed Pairwise RVs
    print("Fitting observed Sequential Chain model...")
    obs_res = lend_simr(data, k=k, topology="graph", path_graph=graph_chain, **kwargs)
    obs_latents = [l.detach().cpu() for l in obs_res["latents"]]
    
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    obs_rvs = {pair: adjusted_rvcoef(obs_latents[pair[0]], obs_latents[pair[1]]) for pair in pairs}
    
    # 4. Permutations
    print(f"Running {n_perms} permutations for pairwise significance...")
    null_rvs = {pair: [] for pair in pairs}
    
    for p in range(n_perms):
        perm_data = [m[np.random.permutation(m.shape[0])] for m in data]
        perm_res = lend_simr(perm_data, k=k, topology="graph", path_graph=graph_chain, **kwargs)
        perm_latents = [l.detach().cpu() for l in perm_res["latents"]]
        
        for pair in pairs:
            null_rvs[pair].append(adjusted_rvcoef(perm_latents[pair[0]], perm_latents[pair[1]]))
            
        if (p+1) % 10 == 0:
            print(f"  Completed {p+1}/{n_perms} permutations...")
            
    # 5. Compute P-values
    print("\n" + "="*50)
    print("PAIRWISE SIGNIFICANCE WITHIN SEQUENTIAL CHAIN")
    print("="*50)
    print(f"{'Pair':12} | {'Obs RV':7} | {'p-value':7} | {'Z-score':7}")
    print("-" * 50)
    
    results = []
    for pair in pairs:
        obs = obs_rvs[pair]
        null_dist = np.array(null_rvs[pair])
        p_val = np.mean(null_dist >= obs)
        z_score = (obs - np.mean(null_dist)) / (np.std(null_dist) + 1e-8)
        
        pair_name = f"{modality_names[pair[0]]} <-> {modality_names[pair[1]]}"
        sig = "*" if p_val < 0.05 else " "
        print(f"{pair_name:12} | {obs:.4f} | {p_val:.4f} {sig} | {z_score:.2f}")
        
        results.append({
            "Pair": pair_name,
            "Obs_RV": obs,
            "p_value": p_val,
            "Z_score": z_score
        })
    print("="*50)
    
    # Save to CSV
    pd.DataFrame(results).to_csv("expart_pairwise_path_significance.csv", index=False)

if __name__ == "__main__":
    run_pairwise_significance(n_perms=100)
