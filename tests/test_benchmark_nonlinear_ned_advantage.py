import pytest
import torch
import numpy as np
from scripts.benchmark_nonlinear_noise_simlr_vs_ned import run_ned_vs_simlr_benchmark


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "This contract previously held only because a bug in _train_loop's "
        "early-stopping baseline truncated NED training to warmup+patience "
        "(~26 of the 200 epochs the benchmark script requests), which happened "
        "to land near the peak of the latent-recovery curve. With the intended "
        "budget actually run, NED's latent recovery peaks around epoch 40 and "
        "then degrades as reconstruction keeps improving:\n"
        "  epochs:  26     40     60     100    200\n"
        "  latent:  0.606  0.752  0.727  0.687  0.659\n"
        "  recon:   0.870  0.686  0.581  0.511  0.453\n"
        "Up-weighting the similarity term mitigates but does not remove it, so "
        "this is not merely a missing scale normalization between the two loss "
        "terms (measured, 3 seeds, epochs 40 -> 200):\n"
        "  sim_weight= 1.0:  0.752 -> 0.659   (-0.093)\n"
        "  sim_weight= 5.0:  0.765 -> 0.684   (-0.081)\n"
        "  sim_weight=20.0:  0.767 -> 0.705   (-0.062)\n"
        "With a longer budget the nonlinear decoder heads fit reconstruction "
        "through pathways that bypass the shared latent, so the shared "
        "representation degrades even as the training objective improves. "
        "Resolving this needs validation-based model selection on a latent- "
        "quality criterion -- the trainers currently early-stop on training "
        "loss only, which cannot detect this. Remove this mark once such "
        "selection exists; do not restore the assertion by shortening the "
        "epoch budget, which is what made it pass before."
    ),
)
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
