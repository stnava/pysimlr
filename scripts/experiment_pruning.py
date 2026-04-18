import torch
import numpy as np
import pandas as pd
from pysimlr.deep import lend_simr
from pysimlr.benchmarks.synthetic_cases import build_case

def experiment_pruning():
    k_true = 3
    k_model = 10
    n = 1000
    
    # Build a linear case with k_true=3 and 2 modalities
    case = build_case("linear", n_samples=n, shared_k=k_true, p_list=[100, 100])
    x1, x2 = [torch.as_tensor(m).float() for m in case["data"]]
    
    # Train LEND with k=10
    res = lend_simr([x1, x2], k=k_model, epochs=200, nsa_w=0.1, verbose=False)
    
    # Calculate the variance captured by each component in the consensus U
    # U is orthogonal, so each column has norm 1. 
    # But we can look at the average correlation of each component with the original views.
    
    # Let's look at the "Signal Energy" per component: ||X_m V_m[:, j]||^2
    v1 = res['v'][0]
    v2 = res['v'][1]
    
    energy1 = torch.norm(x1 @ v1, dim=0)**2 / n
    energy2 = torch.norm(x2 @ v2, dim=0)**2 / n
    
    df = pd.DataFrame({
        "Component": range(1, k_model + 1),
        "Energy_Mod1": energy1.detach().numpy(),
        "Energy_Mod2": energy2.detach().numpy(),
    })
    
    # Sort by Energy
    df = df.sort_values(by="Energy_Mod1", ascending=False)
    
    print("Pruning Evidence (k_true=3, k_model=10):")
    print(df.to_string(index=False))
    
    df.to_csv("paper/results_cache/pruning_evidence.csv", index=False)

if __name__ == "__main__":
    experiment_pruning()
