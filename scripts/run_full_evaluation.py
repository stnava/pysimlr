import torch
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from pysimlr.simlr import simlr
from pysimlr.deep import deep_simr
from pysimlr.utils import set_all_seeds, adjusted_rvcoef
import os
import itertools
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

def fit_and_score(mats, arch, opt, mai, energy, mixing, k):
    # Standard preprocessing: centerAndScale and np
    sl = ["centerAndScale", "np"]
    
    if arch == "linear":
        res = simlr(mats, k=k, optimizer_type=opt, iterations=30, 
                    energy_type=energy, mixing_algorithm=mixing, 
                    scale_list=sl, verbose=False)
        projections = [m @ v for m, v in zip(mats, res['v'])]
    else:
        # Deep Models
        res = deep_simr(
            mats, k=k, model_type=arch, optimizer_type=opt,
            dynamic_weights=mai, mai_metric="procrustes_r2",
            energy_type=energy, mixing_algorithm=mixing,
            epochs=30, learning_rate=1e-3, 
            scale_list=sl,
            verbose=False, device="cpu"
        )
        projections = res['first_layer_scores']
    
    # Compute all-pairs RVCoeff between projections
    n = len(projections)
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append(adjusted_rvcoef(projections[i], projections[j]))
    
    metrics = {
        "omnibus": np.mean(all_pairs) if all_pairs else 0.0,
        "bl_mh": adjusted_rvcoef(projections[0], projections[1]),
        "bl_mr": adjusted_rvcoef(projections[0], projections[2]),
        "bl_pe": adjusted_rvcoef(projections[0], projections[3])
    }
    return metrics, res

def main():
    set_all_seeds(42)
    names, matrices = load_expart_data()
    k = 2
    n_perms = 20
    
    architectures = ["linear", "lend", "ned", "ned_shared_private"]
    optimizers = ["adam", "lars"]
    energies = ["acc", "regression", "nc"]
    mixing_algs = ["ica", "svd", "newton"]
    mai_options = [False, True]
    
    configs = []
    for arch in architectures:
        for opt in optimizers:
            for energy in energies:
                for mixing in mixing_algs:
                    # MAI only for deep models
                    options = mai_options if arch != "linear" else [False]
                    for mai in options:
                        configs.append({
                            "arch": arch, "opt": opt, "mai": mai, 
                            "energy": energy, "mixing": mixing
                        })
    
    total_configs = len(configs)
    all_results = []
    best_avg_t_stat = -float('inf')
    best_config = None
    
    print(f"Starting Full Evaluation: {total_configs} configurations, {n_perms} permutations each.")
    print("Preprocessing: centerAndScale + np")
    print("Optimization Target: Mean of Pairwise T-Statistics (BL=>MH, BL=>MR, BL=>PE)")
    
    start_time = time.time()
    
    for i, cfg in enumerate(configs):
        # 1. Observed Scores
        obs_metrics, _ = fit_and_score(matrices, cfg['arch'], cfg['opt'], cfg['mai'], 
                                        cfg['energy'], cfg['mixing'], k)
        
        # 2. Null Distributions
        null_metrics_list = {m: [] for m in obs_metrics}
        for _ in range(n_perms):
            perm_mats = [m[torch.randperm(m.shape[0])] for m in matrices]
            perm_metrics, _ = fit_and_score(perm_mats, cfg['arch'], cfg['opt'], cfg['mai'], 
                                            cfg['energy'], cfg['mixing'], k)
            for m in obs_metrics:
                null_metrics_list[m].append(perm_metrics[m])
        
        # 3. Stats for each metric
        res = {
            "arch": cfg['arch'], "optimizer": cfg['opt'], "mai": cfg['mai'],
            "energy": cfg['energy'], "mixing": cfg['mixing'],
        }
        
        for m in obs_metrics:
            obs_val = obs_metrics[m]
            null_vals = np.array(null_metrics_list[m])
            # 1-sided t-test: H1: obs > null
            t_stat, p_t = ttest_1samp(null_vals, obs_val, alternative='less')
            
            res[f"{m}_rvcoef"] = obs_val
            res[f"{m}_t_stat"] = -t_stat 
            res[f"{m}_p_t"] = p_t
            res[f"{m}_p_perm"] = np.mean(null_vals >= obs_val)

        # 4. Determine "Best" based on mean of pairwise T-stats (BL=>MH, BL=>MR, BL=>PE)
        avg_t = (res["bl_mh_t_stat"] + res["bl_mr_t_stat"] + res["bl_pe_t_stat"]) / 3
        res["avg_pairwise_t"] = avg_t
        all_results.append(res)
        
        if avg_t > best_avg_t_stat:
            best_avg_t_stat = avg_t
            best_config = res
            
        # Report progress every 3 configurations
        if (i + 1) % 3 == 0 or (i + 1) == total_configs:
            progress = (i + 1) / total_configs * 100
            elapsed = time.time() - start_time
            print(f"\n--- Progress: {progress:.1f}% ({i+1}/{total_configs}) ---")
            print(f"Elapsed Time: {elapsed/60:.1f} min")
            print(f"Best Current Result (Mean Pairwise T): Arch={best_config['arch']}, Opt={best_config['optimizer']}, Energy={best_config['energy']}, Mixing={best_config['mixing']}, MAI={best_config['mai']}")
            print(f"Mean Pairwise T-stat: {best_config['avg_pairwise_t']:.2f}")
            print(f"Omnibus: RV={best_config['omnibus_rvcoef']:.4f}, T-stat={best_config['omnibus_t_stat']:.2f}, P(perm)={best_config['omnibus_p_perm']:.3f}")
            print(f"Pair-wise Stats for Best Method:")
            print(f"  BL=>MH: RV={best_config['bl_mh_rvcoef']:.4f}, T-Stat={best_config['bl_mh_t_stat']:.4f}")
            print(f"  BL=>MR: RV={best_config['bl_mr_rvcoef']:.4f}, T-Stat={best_config['bl_mr_t_stat']:.4f}")
            print(f"  BL=>PE: RV={best_config['bl_pe_rvcoef']:.4f}, T-Stat={best_config['bl_pe_t_stat']:.4f}")
            sys.stdout.flush()

    df = pd.DataFrame(all_results)
    df.to_csv("expart_full_evaluation_comprehensive.csv", index=False)
    print("\nEvaluation Complete. Full results saved to expart_full_evaluation_comprehensive.csv")

if __name__ == "__main__":
    main()
