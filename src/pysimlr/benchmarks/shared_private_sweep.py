import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from .runner import run_single_experiment
from .synthetic_cases import build_case

def tune_shared_private(case_kind: str = "shared_plus_private",
                         n_samples: int = 1000,
                         n_seeds: int = 1,
                         seed: int = 42):
    """Specific tuning grid for NED Shared/Private model."""
    
    # Define grid
    ortho_weights = [0.0, 0.01, 0.05, 0.1]
    var_weights = [0.0, 0.05, 0.1, 0.2]
    
    results = []
    
    case = build_case(kind=case_kind, n_samples=n_samples, seed=seed)
    
    for ow in ortho_weights:
        for vw in var_weights:
            print(f"Tuning SP: ortho_weight={ow}, var_weight={vw}...")
            
            exp = run_single_experiment(
                "shared_private", 
                case, 
                seed=seed,
                epochs=100,
                warmup_epochs=20,
                private_orthogonality_weight=ow,
                private_variance_weight=vw
            )
            
            metrics = exp["metrics"]
            metrics.update({
                "ortho_weight": ow,
                "var_weight": vw
            })
            results.append(metrics)
            
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    df_tune = tune_shared_private(n_samples=500)
    print("Best Tuning Results:")
    print(df_tune.sort_values("recovery", ascending=False).head(5))
    df_tune.to_csv("shared_private_tuning_study.csv", index=False)
