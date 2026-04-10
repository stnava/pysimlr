import torch
import numpy as np
from pysimlr.deep import LENDNSAEncoder
from pysimlr.utils import invariant_orthogonality_defect

def test_lend_nsa_encoder_demo_logic():
    # Case 1: Mostly Linear
    n_samples = 300
    k = 1
    d = 10
    
    torch.manual_seed(42)
    u_true = torch.randn(n_samples, k)
    v_true = torch.randn(d, k)
    x = u_true @ v_true.t() + 0.1 * torch.randn(n_samples, d)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    
    n_train = 200
    x_train, x_test = x[:n_train], x[n_train:]
    u_train, u_test = u_true[:n_train], u_true[n_train:]
    
    # Fit
    enc = LENDNSAEncoder(d, latent_dim=k)
    optimizer = torch.optim.Adam(enc.parameters(), lr=0.01)
    
    for _ in range(50):
        optimizer.zero_grad()
        z = enc(x_train)
        # Trivial single-modality sim loss
        loss = torch.mean((z - u_train)**2) + 0.1 * invariant_orthogonality_defect(enc.v_raw)
        loss.backward()
        optimizer.step()
        
    # Eval
    enc.eval()
    with torch.no_grad():
        z_test = enc(x_test)
        v_eval = enc.v
        
    # Assertions
    correlation = np.abs(np.corrcoef(z_test.numpy().flatten(), u_test.numpy().flatten())[0, 1])
    print(f"Test Correlation: {correlation:.4f}")
    assert correlation > 0.8
    
    ortho_defect = invariant_orthogonality_defect(v_eval).item()
    print(f"Ortho Defect: {ortho_defect:.2e}")
    assert ortho_defect < 1e-5

def test_lend_nsa_encoder_sparsity():
    d, k = 20, 5
    q = 0.5
    enc = LENDNSAEncoder(d, latent_dim=k, sparseness_quantile=q)
    enc.eval()
    v = enc.v
    sparsity = (v == 0).float().mean().item()
    print(f"Sparsity: {sparsity:.2f}")
    # Quantile 0.5 should give at least 40% sparsity allowing for some noise
    assert sparsity > 0.4

if __name__ == "__main__":
    test_lend_nsa_encoder_demo_logic()
    test_lend_nsa_encoder_sparsity()
