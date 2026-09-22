import pytest
import torch
import numpy as np
from pysimlr.deep import ned_simr


@pytest.mark.xfail(strict=True, reason=(
    "NED has a bistable failure mode: the two latents sometimes collapse onto "
    "one direction, giving an off-diagonal covariance norm near 1.0 against "
    "the 0.5 threshold while the per-dimension standard deviations stay at "
    "1.0. It is redundancy rather than collapse, and it is a basin the "
    "optimisation sometimes falls into rather than a property of any one "
    "setting.\n\n"
    "Re-measured over seeds 0-9 plus 42 after the latent gauge went into "
    "`calculate_sim_loss`: 3 of 11 seeds land in the basin under the default "
    "positivity='positive' (median off-diagonal 0.0023) and 4 of 11 under "
    "positivity='either' (median 0.2509). The gauge moved *which* seeds fall "
    "in -- seed 42, this test's original fixture, now comes out at 0.0032 and "
    "the test xpassed -- but it did not remove the basin, so the fixture is "
    "seed 1 (off-diagonal 0.9159), which is in it. Before the gauge the rates "
    "were 1 of 11 and 6 of 11.\n\n"
    "It is not intrinsic to non-negativity: rectifying oracle least-squares "
    "loadings on the same data separates the latents to a correlation of "
    "0.003. Nor is it the orthogonality penalty -- switching from the "
    "orthonormality defect D to the angle defect C moves the off-diagonal from "
    "0.0025 to 0.0024 and recovery from 0.8718 to 0.8723 over four seeds. The "
    "cause is in the consensus path, not the basis.\n\n"
    "Marked xfail rather than relaxed so it flips to a failure once NED's "
    "optimisation stops admitting this basin. A flip means re-running the "
    "seed sweep before celebrating: a single seed leaving the basin is not "
    "the basin closing. Real-data accuracy is unaffected."))
def test_latent_collapse():
    # 1. Create data with k=2 real signal
    n_samples = 200
    d1, d2 = 20, 20
    k = 2
    
    torch.manual_seed(1)
    u_true = torch.randn(n_samples, k)
    x1 = u_true @ torch.randn(k, d1) + 0.1 * torch.randn(n_samples, d1)
    x2 = u_true @ torch.randn(k, d2) + 0.1 * torch.randn(n_samples, d2)
    
    def norm(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
    x1, x2 = norm(x1), norm(x2)
    
    # 2. Fit NED with small batch to stress stability
    # Use 100 epochs to ensure training progress
    res = ned_simr([x1, x2], k=k, epochs=100, batch_size=32, warmup_epochs=10, verbose=False)
    
    u_pred = res['u']
    
    # Check for NaNs
    assert not torch.isnan(u_pred).any()
    
    # Check variance per dimension (should be above floor)
    u_std = torch.std(u_pred, dim=0)
    print(f"Latent stds: {u_std.tolist()}")
    # With normalization and penalties, std should be around 1.0 or at least not 0.
    assert torch.all(u_std > 0.1), f"Latent collapse detected: stds={u_std.tolist()}"
    
    # Check off-diagonal covariance (should be small due to collapse penalty)
    u_c = u_pred - u_pred.mean(dim=0)
    cov = (u_c.T @ u_c) / (n_samples - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    off_diag_norm = torch.norm(off_diag).item()
    print(f"Off-diagonal covariance norm: {off_diag_norm:.6f}")
    assert off_diag_norm < 0.5

if __name__ == "__main__":
    test_latent_collapse()
