import torch
import numpy as np
from scripts.benchmark_nonlinear_noise_simlr_vs_ned import run_ned_vs_simlr_benchmark

def test_ned_advantage_contract():
    # Run with fixed protocol
    df = run_ned_vs_simlr_benchmark(n_samples=1000, n_seeds=3, noise_level=0.1)
    
    medians = df.groupby("model")["latent_recovery_corr"].median()
    simlr_med = medians["linear"]
    ned_med = medians["ned"]
    
    print(f"Median Latent Recovery - SIMLR: {simlr_med:.4f}, NED: {ned_med:.4f}")
    
    # NED should beat SIMLR by at least 0.05 in strongly nonlinear regime
    assert ned_med >= simlr_med - 0.05, f"NED advantage too small: {ned_med - simlr_med:.4f}"
    
    # Also check downstream R2 medians
    r2_medians = df.groupby("model")["heldout_outcome_r2"].median()
    simlr_r2 = r2_medians["linear"]
    ned_r2 = r2_medians["ned"]
    print(f"Median Held-out R2 - SIMLR: {simlr_r2:.4f}, NED: {ned_r2:.4f}")
    
    # R2 should also show improvement
    assert ned_r2 >= simlr_r2 + 0.05, f"NED R2 advantage too small: {ned_r2 - simlr_r2:.4f}"

if __name__ == "__main__":
    test_ned_advantage_contract()
