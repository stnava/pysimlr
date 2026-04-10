import torch
import numpy as np
from pysimlr.deep import ned_simr_shared_private, predict_deep
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_shared_private_non_starvation():
    # 1. Setup synthetic data
    # u_shared drives target y
    # p1, p2 are nuisance modality-specific latents
    n_samples = 400
    k_shared = 2
    k_private = 2
    
    torch.manual_seed(42)
    u_shared = torch.randn(n_samples, k_shared)
    p1 = torch.randn(n_samples, k_private)
    p2 = torch.randn(n_samples, k_private)
    
    # Outcomes depends only on shared signal
    y = u_shared @ torch.randn(k_shared, 1) + 0.05 * torch.randn(n_samples, 1)
    
    # Modalities are non-linear mixtures of shared and private
    def map_fn(s, p, d):
        z = torch.cat([s, p], dim=1)
        w = torch.randn(k_shared + k_private, d)
        return torch.tanh(z @ w)
        
    x1 = map_fn(u_shared, p1, 20)
    x2 = map_fn(u_shared, p2, 20)
    
    def norm(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
    x1, x2 = norm(x1), norm(x2)
    
    # 2. Fit Shared/Private model
    res = ned_simr_shared_private([x1, x2], k=k_shared, private_k=k_private, 
                                  epochs=100, batch_size=64, verbose=False)
    
    pred = predict_deep([x1, x2], res)
    u_pred = pred['u'].numpy()
    p_preds = [p.numpy() for p in pred['private_latents']]
    y_np = y.numpy()
    
    # 3. Assertions
    
    # A. Shared branch should predict y
    reg_shared = LinearRegression().fit(u_pred, y_np)
    r2_shared = r2_score(y_np, reg_shared.predict(u_pred))
    print(f"R2 from Shared latent: {r2_shared:.4f}")
    
    # B. Private branches alone should NOT predict y well (starvation check)
    # They should focus on nuisance.
    r2_privates = []
    for p_pred in p_preds:
        reg_p = LinearRegression().fit(p_pred, y_np)
        r2_p = r2_score(y_np, reg_p.predict(p_pred))
        r2_privates.append(r2_p)
        print(f"R2 from Private latent: {r2_p:.4f}")
        
    assert r2_shared > 0.5, "Shared branch failed to capture shared signal."
    assert r2_shared > max(r2_privates), "Private branch capturing more shared signal than shared branch."
    
    # C. Cross-covariance check (separation)
    from pysimlr.benchmarks.metrics import shared_private_diagnostics
    diag = shared_private_diagnostics(pred['latents'], pred['private_latents'])
    print(f"Diagnostics: {diag}")
    for i in range(2):
        assert diag[f"mod{i}_cross_cov"] < 0.2, f"Modality {i} shared and private latents are highly entangled."

if __name__ == "__main__":
    test_shared_private_non_starvation()
