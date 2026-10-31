import torch
import numpy as np
import pytest
from scripts.benchmark_nonlinear_private_noise_simlr_vs_nedpp import run_shared_private_benchmark

@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "NEDPP's held-out R^2 advantage over SiMLR is below the 0.05 this "
        "contract claims. It held at 85ad3e2 by a thin margin (0.7032 vs "
        "0.6388, +0.064) and fell below at f4e53a6 (support-masked quantile "
        "sparseness and sign-gauge fix): 0.6596 vs 0.6355 (+0.024). Measured "
        "over 8 seeds (42-49, n=300): held-out R^2 0.664 (NEDPP) vs 0.619 "
        "(SiMLR, +0.045) vs 0.663 (NED, +0.001); NEDPP leads SiMLR on 6 of 8 "
        "seeds. Latent recovery does separate the models -- 0.552 (NEDPP) vs "
        "0.434 (SiMLR) vs 0.424 (NED) -- so the shared/private split helps "
        "recovery more than it helps a linear outcome read-out. Marked xfail "
        "rather than relaxed: do not lower the margin to make it pass; flip "
        "it only when an 8-seed run clears +0.05."
    ),
)
def test_nedpp_advantage_contract():
    # Run with fixed protocol (n_samples=300, n_seeds=2 provides fast, deterministic contract verification)
    df = run_shared_private_benchmark(n_samples=300, n_seeds=2, noise_level=0.1)
    
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
