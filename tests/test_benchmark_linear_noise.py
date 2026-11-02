import functools
import pytest
from scripts.benchmark_linear_noise_simlr_vs_lend import run_linear_benchmark


@functools.lru_cache(maxsize=4)
def _cached_linear_benchmark(n_samples: int, n_seeds: int, noise_level: float):
    return run_linear_benchmark(
        n_samples=n_samples, n_seeds=n_seeds, noise_level=noise_level
    )


@pytest.mark.slow
@pytest.mark.xdist_group("linear_noise_bench")
def test_linear_parity_contract():
    # Run with fixed protocol
    df = _cached_linear_benchmark(n_samples=1000, n_seeds=3, noise_level=0.1).copy()
    
    medians = df.groupby("model")["latent_recovery_corr"].median()
    simlr_rec = medians["linear"]
    lend_rec = medians["lend"]
    
    print(f"Median Recovery - SIMLR: {simlr_rec:.4f}, LEND: {lend_rec:.4f}")
    
    # 1. Close performance on latent recovery
    assert (lend_rec - simlr_rec) > -0.05, f"Recovery gap too large negative: {(simlr_rec - lend_rec):.4f}"
    
    # 2. Close performance on outcome R2
    r2_medians = df.groupby("model")["heldout_outcome_r2"].median()
    simlr_r2 = r2_medians["linear"]
    lend_r2 = r2_medians["lend"]
    print(f"Median Held-out R2 - SIMLR: {simlr_r2:.4f}, LEND: {lend_r2:.4f}")
    
    assert (lend_r2 - simlr_r2) > -0.05, f"R2 gap too large negative: {(simlr_r2 - lend_r2):.4f}"
    
    # 3. Minimum usefulness floor
    assert simlr_rec > 0.5, "SIMLR performing below floor on linear data."
    assert lend_rec > 0.5, "LEND performing below floor on linear data."

if __name__ == "__main__":
    test_linear_parity_contract()


@pytest.mark.slow
@pytest.mark.xdist_group("linear_noise_bench")
def test_linear_model_heldout_regression_not_broken():
    df = _cached_linear_benchmark(n_samples=1000, n_seeds=3, noise_level=0.1).copy()
    simlr = df[df["model"] == "linear"]

    assert simlr["heldout_outcome_r2"].median() > 0.5
    assert simlr["reconstruction_mse"].median() < 0.5
