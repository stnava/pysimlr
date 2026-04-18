import torch
import numpy as np
import pandas as pd
from pysimlr.deep import ned_simr_shared_private
from pysimlr.benchmarks.synthetic_cases import build_case
from pysimlr.utils import procrustes_r2, adjusted_rvcoef, invariant_orthogonality_defect
import os
import argparse

def evaluate_nedpp(kind, regime=None, n_seeds=10):
    results = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}...")
        if regime:
            case = build_case(kind, regime=regime, seed=seed)
        else:
            case = build_case(kind, seed=seed)
            
        mats = case["data"]
        true_u = case["true_u"]
        true_v = case["true_v"]
        
        # Fit NEDPP
        res = ned_simr_shared_private(mats, k=case["shared_k"], private_k=2, epochs=150, verbose=False)
        
        # Latent Recovery (CMC)
        u_est = res["u"]
        # Standardized CMC: Procrustes R2
        cmc = procrustes_r2(true_u, u_est)
        
        # Subspace Recovery Error (SRE)
        # SRE = 1 - mean(procrustes_r2(v_true, v_est))
        v_ests = res["v"]
        v_errors = []
        for v_est, v_tr in zip(v_ests, true_v):
            # v_est is from shared part of encoder
            # In NEDPP, res['v'] contains the shared basis for each modality
            # Standardized SRE: 1 - Procrustes R2
            err = 1.0 - procrustes_r2(v_tr, v_est)
            v_errors.append(err)
        sre = np.mean(v_errors)
        
        # Legacy column names
        latent_recovery = cmc
        feature_recovery = 1.0 - sre
        
        results.append({
            "Regime": regime if regime else kind,
            "Seed": seed,
            "CMC": cmc,
            "SRE": sre,
            "Latent Recovery (U)": latent_recovery,
            "Feature Recovery (V)": feature_recovery
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NEDPP synthetic results")
    parser.add_argument("--smoke-test", action="store_true", help="Run with fewer seeds for testing")
    args = parser.parse_args()
    
    n_seeds = 1 if args.smoke_test else 10

    print("Generating NEDPP results for Linear...")
    linear_res = evaluate_nedpp("linear", n_seeds=n_seeds)

    print("Generating NEDPP results for Polynomial...")
    poly_res = evaluate_nedpp("nonlinear", regime="polynomial", n_seeds=n_seeds)

    print("Generating NEDPP results for Sinusoidal...")
    sin_res = evaluate_nedpp("nonlinear", regime="sinusoidal", n_seeds=n_seeds)

    print("Generating NEDPP results for Shared+Private...")
    private_res = evaluate_nedpp("shared_plus_private", n_seeds=n_seeds)

    all_nedpp = linear_res + poly_res + sin_res + private_res
    df = pd.DataFrame(all_nedpp)
    df.to_csv("paper/results_cache/nedpp_synthetic_v10.csv", index=False)
    print("Done. Saved to paper/results_cache/nedpp_synthetic_v10.csv")
