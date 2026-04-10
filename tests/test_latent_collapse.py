import torch
import numpy as np
from pysimlr.deep import ned_simr

def test_latent_collapse():
    # 1. Create data with k=2 real signal
    n_samples = 200
    d1, d2 = 20, 20
    k = 2
    
    torch.manual_seed(42)
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
