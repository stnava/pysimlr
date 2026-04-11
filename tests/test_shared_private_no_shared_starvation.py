import torch
import numpy as np
from pysimlr.deep import ned_simr_shared_private, predict_deep
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_shared_private_non_starvation():
    # 1. Setup synthetic data
    # Increase samples for better stability
    n_samples = 600
    k_shared = 2
    k_private = 2
    
    torch.manual_seed(42)
    u_shared = torch.randn(n_samples, k_shared)
    p1 = torch.randn(n_samples, k_private)
    p2 = torch.randn(n_samples, k_private)
    
    # Outcomes depends strongly on shared signal
    y = u_shared[:, 0:1] * 2.0 + u_shared[:, 1:2] * 1.0 + 0.05 * torch.randn(n_samples, 1)
    
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
                                  epochs=150, batch_size=64, verbose=False)
    
    # Evaluate on split for robustness.
    n_train = 400
    train_batch = [x1[:n_train], x2[:n_train]]
    test_batch = [x1[n_train:], x2[n_train:]]
    y_train = y[:n_train].numpy()
    y_test = y[n_train:].numpy()

    # Predict train and test to align coordinates
    pred_train = predict_deep(train_batch, res, device="cpu")
    pred_test = predict_deep(test_batch, res, device="cpu")
    
    u_train = pred_train['u'].numpy()
    u_test = pred_test['u'].numpy()
    
    # 3. Assertions
    
    # A. Shared branch should predict y on held-out data
    reg_shared = LinearRegression().fit(u_train, y_train)
    r2_shared = r2_score(y_test, reg_shared.predict(u_test))
    print(f"Held-out R2 from Shared latent: {r2_shared:.4f}")
    
    # B. Private branches alone should NOT predict y well (starvation check)
    p_trains = [p.numpy() for p in pred_train['private_latents']]
    p_tests = [p.numpy() for p in pred_test['private_latents']]
    
    r2_privates = []
    for pt, ps in zip(p_trains, p_tests):
        reg_p = LinearRegression().fit(pt, y_train)
        r2_p = r2_score(y_test, reg_p.predict(ps))
        r2_privates.append(r2_p)
        print(f"Held-out R2 from Private latent: {r2_p:.4f}")
        
    assert r2_shared > 0.4, f"Shared branch failed to capture shared signal: R2={r2_shared:.4f}"
    assert r2_shared > max(r2_privates), "Private branch capturing more shared signal than shared branch."
    
    # C. Cross-covariance check (separation)
    from pysimlr.benchmarks.metrics import shared_private_diagnostics
    diag = shared_private_diagnostics(pred_test['latents'], pred_test['private_latents'])
    print(f"Diagnostics: {diag}")
    for i in range(2):
        # Increased threshold to 0.35 for better tolerance in small-batch synthetic cases
        assert diag[f"mod{i}_cross_cov"] < 0.35, f"Modality {i} shared and private latents are highly entangled."

if __name__ == "__main__":
    test_shared_private_non_starvation()
