import torch
import pandas as pd
import numpy as np
from pysimlr import simlr, create_path_graph
from pysimlr.utils import adjusted_rvcoef
from scipy.stats import ttest_1samp

def structural_permutation_test_linear(data, k, graph, n_perms=100, **simlr_params):
    # 1. Observed Result
    obs_res = simlr(data, k=k, topology="graph", path_graph=graph, **simlr_params)
    
    obs_edge_sims = {}
    # Ensure everything is torch tensors for adjusted_rvcoef
    data_t = [torch.as_tensor(m).float() for m in data]
    v_t = [v.to(data_t[0].device) for v in obs_res["v"]]
    
    latents = [data_t[i] @ v_t[i] for i in range(len(data))]
    for u, neighbors in graph.items():
        for v in neighbors:
            if u < v:
                obs_edge_sims[(u, v)] = adjusted_rvcoef(latents[u], latents[v])
    
    # 2. Null Distribution
    null_edge_sims = {edge: [] for edge in obs_edge_sims.keys()}
    for p in range(n_perms):
        perm_data = [m[np.random.permutation(m.shape[0])] for m in data]
        perm_data_t = [torch.as_tensor(m).float() for m in perm_data]
        perm_res = simlr(perm_data, k=k, topology="graph", path_graph=graph, **simlr_params)
        
        perm_v_t = [v.to(perm_data_t[0].device) for v in perm_res["v"]]
        perm_latents = [perm_data_t[i] @ perm_v_t[i] for i in range(len(data))]
        for u, v in obs_edge_sims.keys():
            null_edge_sims[(u, v)].append(adjusted_rvcoef(perm_latents[u], perm_latents[v]))
            
    # 3. Stats
    stats = {}
    modality_names = ["BL", "MH", "MR", "PE"]
    for edge, obs in obs_edge_sims.items():
        null_dist = np.array(null_edge_sims[edge])
        p_val = np.mean(null_dist >= obs)
        t_stat = (obs - np.mean(null_dist)) / (np.std(null_dist) + 1e-8)
        edge_name = f"{modality_names[edge[0]]} <-> {modality_names[edge[1]]}"
        stats[edge_name] = {"obs": obs, "p": p_val, "z": t_stat}
    return stats

def run_linear_structural_analysis():
    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    data = [bl, mh, mr, pe]
    
    # Hypothesized Chain based on linear sigs:
    # Segment 1: BL (0) <-> MH (1)
    # Segment 2: MH (1) <-> PE (3)
    # Segment 3: PE (3) <-> MR (2)
    graph_linear_best = create_path_graph([(0, 1), (1, 3), (3, 2)], n_modalities=4)
    
    print("Running Linear Structural Significance Analysis (N=62)...")
    stats = structural_permutation_test_linear(
        data, k=3, graph=graph_linear_best, n_perms=100, 
        energy_type="acc", sparseness_quantile=0.5, iterations=50
    )
    
    print("\n" + "="*50)
    print("LINEAR STRUCTURAL SIGNIFICANCE REPORT")
    print("="*50)
    print(f"{'Path Segment':18} | {'Obs RV':7} | {'p-value':7} | {'Z-score':7}")
    print("-" * 50)
    for edge, s in stats.items():
        sig = "*" if s["p"] < 0.05 else " "
        print(f"{edge:18} | {s['obs']:.4f} | {s['p']:.4f} {sig} | {s['z']:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_linear_structural_analysis()
