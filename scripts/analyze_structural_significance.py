import torch
import pandas as pd
import numpy as np
from pysimlr import lend_simr, create_path_graph
from pysimlr.utils import adjusted_rvcoef
from scipy.stats import ttest_1samp

def structural_permutation_test(data, k, graph, n_perms=50, **simlr_params):
    # 1. Observed Result
    print("Fitting observed model...")
    obs_res = lend_simr(data, k=k, topology="graph", path_graph=graph, **simlr_params)
    
    # Calculate Observed Edge Similarities
    obs_edge_sims = {}
    latents = obs_res["latents"]
    for u, neighbors in graph.items():
        for v in neighbors:
            if u < v: # Avoid double counting
                obs_edge_sims[(u, v)] = adjusted_rvcoef(latents[u], latents[v])
    
    # 2. Null Distribution via Permutation
    null_edge_sims = {edge: [] for edge in obs_edge_sims.keys()}
    
    for p in range(n_perms):
        print(f"Permutation {p+1}/{n_perms}...")
        # Shuffle each modality independently
        perm_data = [m[np.random.permutation(m.shape[0])] for m in data]
        perm_res = lend_simr(perm_data, k=k, topology="graph", path_graph=graph, **simlr_params)
        
        perm_latents = perm_res["latents"]
        for u, v in obs_edge_sims.keys():
            null_edge_sims[(u, v)].append(adjusted_rvcoef(perm_latents[u], perm_latents[v]))
            
    # 3. Calculate p-values
    stats = {}
    modality_names = ["BL", "MH", "MR", "PE"]
    for edge, obs in obs_edge_sims.items():
        null_dist = np.array(null_edge_sims[edge])
        p_val = np.mean(null_dist >= obs)
        # Z-score style t-stat for effect size
        t_stat = (obs - np.mean(null_dist)) / (np.std(null_dist) + 1e-8)
        
        edge_name = f"{modality_names[edge[0]]} <-> {modality_names[edge[1]]}"
        stats[edge_name] = {
            "observed": obs,
            "p_value": p_val,
            "t_stat": t_stat,
            "null_mean": np.mean(null_dist)
        }
        
    return stats

def run_significance_analysis():
    # Load Data
    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    data = [bl, mh, mr, pe]
    
    # Best Model: Chain (0 -> 2 -> 3 -> 1) i.e. BL -> MR -> PE -> MH
    graph_chain = create_path_graph([(0, 2), (2, 3), (3, 1)], n_modalities=4)
    
    print("Running Significance Analysis for Sequential Chain Model (BL -> MR -> PE -> MH)...")
    stats = structural_permutation_test(
        data, 
        k=3, 
        graph=graph_chain, 
        n_perms=20, # 20 permutations for statistical stability in this demo
        epochs=100,
        energy_type="acc",
        mixing_algorithm="svd",
        sparseness_quantile=0.5,
        verbose=False
    )
    
    print("\n" + "="*50)
    print("EXPART STRUCTURAL SIGNIFICANCE REPORT")
    print("="*50)
    print(f"{'Path Segment':18} | {'Obs RV':7} | {'p-value':7} | {'Z-score':7}")
    print("-" * 50)
    for edge, s in stats.items():
        sig = "*" if s["p_value"] < 0.05 else " "
        print(f"{edge:18} | {s['observed']:.4f} | {s['p_value']:.4f} {sig} | {s['t_stat']:.2f}")
    print("="*50)
    print("(*) indicates p < 0.05 (One-tailed permutation test)")

if __name__ == "__main__":
    run_significance_analysis()
