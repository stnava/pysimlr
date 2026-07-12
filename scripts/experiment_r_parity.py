import torch
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from pysimlr.simlr import simlr
from pysimlr.utils import set_all_seeds, adjusted_rvcoef
import os
import sys
import time

def load_expart_data():
    names = ["bl", "mh", "mr", "pe"]
    files = [f"data/expart_{n}.csv" for n in names]
    matrices = []
    for f in files:
        df = pd.read_csv(f)
        matrices.append(torch.as_tensor(df.values).float())
    return names, matrices

def fit_and_score(mats, k=2):
    # R-Parity Config
    res = simlr(
        mats, 
        k=k, 
        optimizer_type="armijo_gradient", 
        iterations=50, 
        energy_type="regression", 
        mixing_algorithm="ica", 
        sparseness_quantile=0.5,
        constraint="nsaflow_nsx0.5x10",
        positivity="positive",
        scale_list=["centerAndScale", "np"],
        verbose=False
    )
    projections = [m @ v for m, v in zip(mats, res['v'])]
    
    # Pairwise RVCoeff
    metrics = {
        "bl_mh": adjusted_rvcoef(projections[0], projections[1]),
        "bl_mr": adjusted_rvcoef(projections[0], projections[2]),
        "bl_pe": adjusted_rvcoef(projections[0], projections[3])
    }
    return metrics

def main():
    set_all_seeds(42)
    names, matrices = load_expart_data()
    n_perms = 20
    
    print("Running R-Parity Experiment (Armijo, ICA, 50% Sparsity, NSAFlow Constraint, K=2)...")
    
    # 1. Observed
    obs_metrics = fit_and_score(matrices)
    
    # 2. Null
    null_metrics_list = {m: [] for m in obs_metrics}
    for i in range(n_perms):
        if (i+1) % 5 == 0: print(f"  Permutation {i+1}/{n_perms}...")
        perm_mats = [m[torch.randperm(m.shape[0])] for m in matrices]
        perm_metrics = fit_and_score(perm_mats)
        for m in obs_metrics:
            null_metrics_list[m].append(perm_metrics[m])
            
    # 3. Stats
    print("\n--- R-Parity Results (Highest Min T-Stat Logic) ---")
    t_stats = []
    for m in obs_metrics:
        obs_val = obs_metrics[m]
        null_vals = np.array(null_metrics_list[m])
        t_stat, _ = ttest_1samp(null_vals, obs_val, alternative='less')
        t_stats.append(-t_stat)
        print(f"  Pair {m.upper()}: RV={obs_val:.4f}, T-Stat={-t_stat:.2f}")
    
    print(f"\n  Minimum T-Stat: {min(t_stats):.2f}")
    print(f"  Mean T-Stat:    {np.mean(t_stats):.2f}")

if __name__ == "__main__":
    main()
