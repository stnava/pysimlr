import torch
import numpy as np
from scripts.benchmark_nonlinear_private_noise_simlr_vs_nedpp import run_shared_private_benchmark

def test_nedpp_advantage_contract():
    # Run with fixed protocol
    df = run_shared_private_benchmark(n_samples=1000, n_seeds=3, noise_level=0.1)
    
    r2_medians = df.groupby("model")["heldout_outcome_r2"].median()
    simlr_r2 = r2_medians["linear"]
    ned_r2 = r2_medians["ned"]
    nedpp_r2 = r2_medians["nedpp"]
    
    print(f"Median Held-out R2 - SIMLR: {simlr_r2:.4f}, NED: {ned_r2:.4f}, NEDPP: {nedpp_r2:.4f}")
    
    # 1. NEDPP should beat SIMLR on held-out shared-target metrics
    assert nedpp_r2 >= simlr_r2 + 0.05, f"NEDPP R2 advantage over SIMLR too small: {nedpp_r2 - simlr_r2:.4f}"
    
    # 2. NEDPP matches or exceeds plain NED on shared-target metrics
    assert nedpp_r2 >= ned_r2 - 0.02, f"NEDPP underperforming relative to NED: {nedpp_r2 - ned_r2:.4f}"
    
    # 3. Recovery check
    rec_medians = df.groupby("model")["latent_recovery_corr"].median()
    ned_rec = rec_medians["ned"]
    nedpp_rec = rec_medians["nedpp"]
    print(f"Median Recovery - NED: {ned_rec:.4f}, NEDPP: {nedpp_rec:.4f}")
    
    assert nedpp_rec >= ned_rec - 0.05, "NEDPP recovery significantly worse than NED."

if __name__ == "__main__":
    test_nedpp_advantage_contract()
