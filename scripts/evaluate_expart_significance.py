import torch
import numpy as np
import pandas as pd
from pysimlr.paths import permutation_test
from pysimlr.utils import set_all_seeds
import os

def load_expart_data():
    files = ["data/expart_bl.csv", "data/expart_mh.csv", "data/expart_mr.csv", "data/expart_pe.csv"]
    matrices = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing expected data file: {f}")
        df = pd.read_csv(f)
        # Convert to numpy and then torch handles it
        matrices.append(df.values)
    return matrices

def run_evaluation():
    set_all_seeds(42)
    try:
        matrices = load_expart_data()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Define hyperparameter grid for methods evaluation
    ks = [2, 3, 5]
    optimizers = ["hybrid_adam", "nsa_flow"]
    energies = ["regression", "acc"]
    
    results = []
    
    # Reduced permutations for faster execution in this environment
    # but still enough to get a sense of significance
    n_perms = 50 
    
    total_runs = len(ks) * len(optimizers) * len(energies)
    count = 0
    
    print(f"Starting methods evaluation on expart data ({total_runs} configurations)...")
    print(f"Each run uses {n_perms} permutations.")
    
    for k in ks:
        for opt in optimizers:
            for energy in energies:
                count += 1
                print(f"[{count}/{total_runs}] Running: k={k}, opt={opt}, energy={energy}")
                
                try:
                    # permutation_test handles the observed run + N null runs
                    res = permutation_test(
                        matrices, 
                        k=k, 
                        n_permutations=n_perms,
                        optimizer_type=opt,
                        energy_type=energy,
                        verbose=False,
                        iterations=50 # Number of SiMLR iterations
                    )
                    
                    results.append({
                        "k": k,
                        "optimizer": opt,
                        "energy": energy,
                        "observed_similarity": res["observed_similarity"],
                        "p_value": res["p_value"],
                        "null_mean": np.mean(res["null_similarities"]),
                        "null_std": np.std(res["null_similarities"])
                    })
                except Exception as e:
                    print(f"Error in configuration k={k}, opt={opt}, energy={energy}: {e}")
    
    if not results:
        print("No results collected due to errors.")
        return

    df_results = pd.DataFrame(results)
    output_path = "expart_methods_evaluation.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nEvaluation complete. Results saved to {output_path}")
    
    print("\nSummary Results (Sorted by P-Value):")
    summary = df_results.sort_values("p_value")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_evaluation()
