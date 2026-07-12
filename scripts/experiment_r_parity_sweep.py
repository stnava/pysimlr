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

def fit_and_score(mats, arch, opt, mai, energy, mixing, k, sparseness):
    sl = ["centerAndScale", "np"]
    constraint = "nsaflow_nsx0.5x10"
    
    if arch == "linear":
        res = simlr(mats, k=k, optimizer_type=opt, iterations=30, 
                    energy_type=energy, mixing_algorithm=mixing, 
                    sparseness_quantile=sparseness,
                    constraint=constraint,
                    positivity="positive",
                    scale_list=sl, verbose=False)
        projections = [m @ v for m, v in zip(mats, res['v'])]
    else:
        res = deep_simr(
            mats, k=k, model_type=arch, optimizer_type=opt,
            dynamic_weights=mai, mai_metric="procrustes_r2",
            energy_type=energy, mixing_algorithm=mixing,
            sparseness_quantile=sparseness,
            epochs=30, learning_rate=1e-3, 
            scale_list=sl,
            verbose=False, device="cpu",
            retraction_type="soft_ns"
        )
        projections = res['first_layer_scores']
    
    metrics = {
        "bl_mh": adjusted_rvcoef(projections[0], projections[1]),
        "bl_mr": adjusted_rvcoef(projections[0], projections[2]),
        "bl_pe": adjusted_rvcoef(projections[0], projections[3])
    }
    return metrics

def main():
    set_all_seeds(42)
    names, matrices = load_expart_data()
    k = 2
    n_perms = 10 # Reduced for quick experiment
    
    # Focused Grid based on R-Parity and User Feedback
    architectures = ["linear", "lend", "ned"]
    optimizers = ["armijo_gradient", "adam", "lars"]
    energies = ["regression"]
    mixing_algs = ["ica", "svd"]
    sparseness_options = [0.0, 0.5]
    mai_options = [False, True]
    
    configs = []
    for arch in architectures:
        for opt in optimizers:
            for energy in energies:
                for mixing in mixing_algs:
                    for sp in sparseness_options:
                        options = mai_options if arch != "linear" else [False]
                        for mai in options:
                            configs.append({
                                "arch": arch, "opt": opt, "mai": mai, 
                                "energy": energy, "mixing": mixing, "sp": sp
                            })
    
    total_configs = len(configs)
    all_results = []
    best_min_t = -float('inf')
    best_config = None
    
    print(f"Starting Quick R-Parity Sweep: {total_configs} configurations, {n_perms} permutations each.")
    print("Criterion: Highest Minimum T-Statistic across (BL=>MH, BL=>MR, BL=>PE)")
    
    start_time = time.time()
    
    for i, cfg in enumerate(configs):
        # 1. Observed
        obs_metrics = fit_and_score(matrices, cfg['arch'], cfg['opt'], cfg['mai'], 
                                     cfg['energy'], cfg['mixing'], k, cfg['sp'])
        
        # 2. Null
        null_metrics_list = {m: [] for m in obs_metrics}
        for _ in range(n_perms):
            perm_mats = [m[torch.randperm(m.shape[0])] for m in matrices]
            perm_metrics = fit_and_score(perm_mats, cfg['arch'], cfg['opt'], cfg['mai'], 
                                          cfg['energy'], cfg['mixing'], k, cfg['sp'])
            for m in obs_metrics:
                null_metrics_list[m].append(perm_metrics[m])
        
        # 3. Stats
        res = {k: v for k, v in cfg.items()}
        pair_ts = []
        for m in obs_metrics:
            obs_val = obs_metrics[m]
            null_vals = np.array(null_metrics_list[m])
            t_stat, _ = ttest_1samp(null_vals, obs_val, alternative='less')
            t_stat = -t_stat
            res[f"{m}_rvcoef"] = obs_val
            res[f"{m}_t_stat"] = t_stat
            pair_ts.append(t_stat)
            
        min_t = min(pair_ts)
        res["min_t_stat"] = min_t
        all_results.append(res)
        
        if min_t > best_min_t:
            best_min_t = min_t
            best_config = res
            
        if (i + 1) % 3 == 0 or (i + 1) == total_configs:
            progress = (i + 1) / total_configs * 100
            print(f"\n--- Progress: {progress:.1f}% ({i+1}/{total_configs}) ---")
            print(f"Best Current Result (Highest Min T): Arch={best_config['arch']}, Opt={best_config['opt']}, "
                  f"Energy={best_config['energy']}, Mixing={best_config['mixing']}, SP={best_config['sp']}, MAI={best_config['mai']}")
            print(f"Min T-Stat: {best_config['min_t_stat']:.2f}")
            print(f"Pair-wise (RV / T):")
            print(f"  BL=>MH: {best_config['bl_mh_rvcoef']:.4f} / {best_config['bl_mh_t_stat']:.2f}")
            print(f"  BL=>MR: {best_config['bl_mr_rvcoef']:.4f} / {best_config['bl_mr_t_stat']:.2f}")
            print(f"  BL=>PE: {best_config['bl_pe_rvcoef']:.4f} / {best_config['bl_pe_t_stat']:.2f}")
            sys.stdout.flush()

    df = pd.DataFrame(all_results)
    df.to_csv("expart_r_parity_sweep.csv", index=False)
    print("\nSweep Complete.")

if __name__ == "__main__":
    main()
