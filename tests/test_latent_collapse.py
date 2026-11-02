import torch
from pysimlr.deep import ned_simr


# History: NED had a bistable failure mode in which the two latents collapsed
# onto one direction (off-diagonal covariance norm near 1.0 against the 0.5
# threshold). Over seeds 0-9 plus 42 it caught 3/11 seeds under
# positivity='positive' and 4/11 under 'either' with energy_type='regression'
# and nsa_w=0.1, so this was a strict xfail. With the package defaults unified
# in 0.3.0 (energy_type='acc', nsa_w=0.5; `pysimlr.defaults`) the same sweep
# gives 0/11 under both positivities (median off-diagonal 0.0015 'either',
# 0.0025 'positive'; this fixture, seed 1: 0.0071), measured 2026-11-01.
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
    res = ned_simr([x1, x2], k=k, iterations=100, batch_size=32, warmup_epochs=10, verbose=False)
    
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
