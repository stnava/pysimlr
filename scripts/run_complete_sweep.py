import torch
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from pysimlr.simlr import simlr
from pysimlr.deep import deep_simr
from pysimlr.utils import set_all_seeds, adjusted_rvcoef
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    # Ensure orthogonality constraint is present for linear models when possible
    constraint = "nsaflow_nsx0.5x10" if mixing != "ica" else "none"
    
    if arch == "linear":
        res = simlr(mats, k=k, optimizer_type=opt, iterations=500, 
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
            epochs=500, learning_rate=1e-3, 
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

def run_one_config(args):
    cfg, matrices, k, n_perms = args
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
    res = {k_v: v_v for k_v, v_v in cfg.items()}
    pair_ts = []
    for m in obs_metrics:
        obs_val = obs_metrics[m]
        null_vals = np.array(null_metrics_list[m])
        t_stat, _ = ttest_1samp(null_vals, obs_val, alternative='less')
        t_stat = -t_stat
        res[f"{m}_rvcoef"] = obs_val
        res[f"{m}_t_stat"] = t_stat
        pair_ts.append(t_stat)
        
    res["min_t_stat"] = min(pair_ts)
    return res

def main():
    set_all_seeds(42)
    names, matrices = load_expart_data()
    k = 2
    n_perms = 10
    output_path = "expart_complete_sweep_results.csv"
    
    architectures = ["linear", "lend", "ned", "ned_shared_private"]
    optimizers = ["armijo_gradient", "adam", "lars"]
    energies = ["acc", "regression", "nc"]
    mixing_algs = ["ica", "svd", "newton"]
    sparseness_options = [0.0, 0.5]
    mai_options = [False, True]
    
    all_configs = []
    for arch in architectures:
        for opt in optimizers:
            for energy in energies:
                for mixing in mixing_algs:
                    for sp in sparseness_options:
                        options = mai_options if arch != "linear" else [False]
                        for mai in options:
                            all_configs.append({
                                "arch": arch, "opt": opt, "mai": mai, 
                                "energy": energy, "mixing": mixing, "sp": sp
                            })
    
    total_configs = len(all_configs)
    results = []
    
    if os.path.exists(output_path):
        results = pd.read_csv(output_path).to_dict('records')
        print(f"Resuming from {len(results)} existing results.")
    
    completed_keys = set()
    for r in results:
        key = (r['arch'], r['opt'], bool(r['mai']), r['energy'], r['mixing'], float(r['sp']))
        completed_keys.add(key)
    
    todo_configs = []
    for cfg in all_configs:
        key = (cfg['arch'], cfg['opt'], bool(cfg['mai']), cfg['energy'], cfg['mixing'], float(cfg['sp']))
        if key not in completed_keys:
            todo_configs.append(cfg)
    
    if not todo_configs:
        print("All configurations already completed.")
        return

    best_min_t = -float('inf')
    best_config = None
    if results:
        best_min_t = max([r['min_t_stat'] for r in results])
        best_config = next(r for r in results if r['min_t_stat'] == best_min_t)

    print(f"Starting/Resuming Sweep: {len(todo_configs)} configs remaining out of {total_configs}.")
    sys.stdout.flush()

    # Use multiprocessing to speed up
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_one_config, (cfg, matrices, k, n_perms)): cfg for cfg in todo_configs}
        
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                
                if not np.isnan(res['min_t_stat']) and res['min_t_stat'] > best_min_t:
                    best_min_t = res['min_t_stat']
                    best_config = res
                
                # Progress and summary
                progress = len(results) / total_configs * 100
                if best_config is not None:
                    print(f"[{progress:4.1f}%] BEST: Arch={best_config['arch']}, Opt={best_config['opt']}, Eng={best_config['energy']}, Mix={best_config['mixing']}, SP={best_config['sp']}, MAI={best_config['mai']}")
                    print(f"       MinT={best_config['min_t_stat']:5.2f} | BL=>MH:{best_config['bl_mh_t_stat']:5.2f}, BL=>MR:{best_config['bl_mr_t_stat']:5.2f}, BL=>PE:{best_config['bl_pe_t_stat']:5.2f}")
                else:
                    print(f"[{progress:4.1f}%] Config completed: Arch={res['arch']}, Opt={res['opt']}, Eng={res['energy']}, Mix={res['mixing']}, SP={res['sp']}, MAI={res['mai']} (MinT={res['min_t_stat']})")
                sys.stdout.flush()
                
                # Save after each completion
                pd.DataFrame(results).to_csv(output_path, index=False)
            except Exception as e:
                cfg = futures[future]
                print(f"Error running config {cfg}: {e}")
                import traceback
                traceback.print_exc()

    print("\nComplete Sweep Finished.")

if __name__ == "__main__":
    main()
