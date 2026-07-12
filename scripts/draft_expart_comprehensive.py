import torch
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from pysimlr.simlr import simlr
from pysimlr.deep import LENDSiMRModel, NEDSiMRModel, NEDSharedPrivateSiMRModel, deep_simr
from pysimlr.utils import set_all_seeds, adjusted_rvcoef
import os
import itertools

def load_expart_data():
    names = ["bl", "mh", "mr", "pe"]
    files = [f"data/expart_{n}.csv" for n in names]
    matrices = []
    for f in files:
        df = pd.read_csv(f)
        matrices.append(torch.as_tensor(df.values).float())
    return names, matrices

def run_experiment(matrices, model_type, opt, use_mai, n_perms=20):
    """
    Run omnibus and pairwise significance for a given configuration.
    """
    k = 2
    n_modalities = len(matrices)
    input_dims = [m.shape[1] for m in matrices]
    
    # Combinations: Omnibus (all) + Pairwise
    combos = [list(range(n_modalities))] # Omnibus
    combos.extend(list(itertools.combinations(range(n_modalities), 2)))
    
    results = []
    
    for combo in combos:
        combo_indices = list(combo)
        sub_mats = [matrices[i] for i in combo_indices]
        sub_dims = [input_dims[i] for i in combo_indices]
        
        # 1. Observed Score
        obs_score, _ = fit_and_score(sub_mats, model_type, opt, use_mai, k)
        
        # 2. Null Distribution
        null_scores = []
        for _ in range(n_perms):
            perm_mats = [m[torch.randperm(m.shape[0])] for m in sub_mats]
            score, _ = fit_and_score(perm_mats, model_type, opt, use_mai, k)
            null_scores.append(score)
        
        # 3. Statistics
        p_perm = np.mean(np.array(null_scores) >= obs_score)
        # 1-sided t-test: H1: obs > null
        t_stat, p_t = ttest_1samp(null_scores, obs_score, alternative='less')
        
        results.append({
            "combo": [["bl", "mh", "mr", "pe"][i] for i in combo_indices],
            "obs_rvcoef": obs_score,
            "p_perm": p_perm,
            "p_t_test": p_t,
            "null_mean": np.mean(null_scores),
            "null_std": np.std(null_scores)
        })
        
    return results

def fit_and_score(mats, model_type, opt, use_mai, k):
    if model_type == "linear":
        res = simlr(mats, k=k, optimizer_type=opt, iterations=50, verbose=False)
        projections = [m @ v for m, v in zip(mats, res['v'])]
    else:
        res = deep_simr(
            mats, k=k, model_type=model_type, optimizer_type=opt,
            dynamic_weights=use_mai, mai_metric="procrustes_r2",
            epochs=100, learning_rate=1e-3, verbose=False, device="cpu"
        )
        # For deep models, we use first_layer_scores which are X_i @ V_i (interpretable projection)
        projections = res['first_layer_scores']
    
    # Compute all-pairs RVCoeff between projections
    n = len(projections)
    pair_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_scores.append(adjusted_rvcoef(projections[i], projections[j]))
    
    # Return mean pairwise similarity as the omnibus score
    return np.mean(pair_scores) if pair_scores else 0.0, res

def main():
    set_all_seeds(42)
    names, matrices = load_expart_data()
    
    architectures = ["linear", "lend", "ned", "ned_shared_private"]
    optimizers = ["adam", "lars"]
    mai_options = [False, True]
    
    all_results = []
    print("Starting Full Evaluation (10 permutations per config)...")
    for arch in architectures:
        for opt in optimizers:
            # MAI only for deep models
            options = mai_options if arch != "linear" else [False]
            for mai in options:
                print(f"Running: Arch={arch}, Opt={opt}, MAI={mai}")
                results = run_experiment(matrices, arch, opt, mai, n_perms=10)
                for r in results:
                    r.update({"arch": arch, "optimizer": opt, "mai": mai})
                    all_results.append(r)
    
    df = pd.DataFrame(all_results)
    df.to_csv("expart_comprehensive_draft.csv", index=False)
    print("\nResults saved to expart_comprehensive_draft.csv")
    
    # Summary of Omnibus results
    omnibus = df[df['combo'].apply(len) == 4]
    print("\nOmnibus Significance Summary:")
    print(omnibus[["arch", "optimizer", "mai", "obs_rvcoef", "p_perm", "p_t_test"]])

if __name__ == "__main__":
    main()
