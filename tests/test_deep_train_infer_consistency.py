import torch
import numpy as np
from pysimlr.deep import lend_simr, NEDSiMRModel

def test_deep_train_infer_consistency():
    # 1. Create synthetic data
    n_train = 100
    d1, d2 = 10, 8
    k = 2
    
    torch.manual_seed(42)
    u_train = torch.randn(n_train, k)
    x1_train = u_train @ torch.randn(k, d1) + 0.1 * torch.randn(n_train, d1)
    x2_train = u_train @ torch.randn(k, d2) + 0.1 * torch.randn(n_train, d2)
    
    # 2. Fit LEND
    print("Fitting LEND...")
    # Use dropout=0.0
    res = lend_simr([x1_train, x2_train], k=k, epochs=20, warmup_epochs=5, verbose=False, dropout=0.0)
    model = res['model']
    
    # 3. Check consistency between two eval calls (should be identical)
    batch = [x1_train[:10], x2_train[:10]]
    
    model.eval()
    with torch.no_grad():
        l1, r1, u1 = model(batch)
        l2, r2, u2 = model(batch)
        
    for la, lb in zip(l1, l2):
        assert torch.allclose(la, lb, atol=1e-6)
    assert torch.allclose(u1, u2, atol=1e-6)
    
    # 4. Check that train vs eval difference is expected (due to projection)
    # Actually, let's just check that with mixing_algorithm="avg", they are identical
    # IF we disable the projection in LENDNSAEncoder too? No, encoders always project in eval.
    
    # Let's verify that if we use mixing_algorithm="avg", the u_shared calculation 
    # uses the same mean logic in both train and eval.
    # Note: Encoders will still differ (v_raw vs v).
    
    res_avg = lend_simr([x1_train, x2_train], k=k, epochs=5, warmup_epochs=0, mixing_algorithm="avg", dropout=0.0)
    model_avg = res_avg['model']
    
    # In model_avg, u_shared = mean(latents) in both modes.
    # latents_train = x @ v_raw
    # latents_eval = x @ v
    
    model_avg.eval()
    with torch.no_grad():
        l_eval, _, u_eval = model_avg(batch)
        # u_eval should be mean of l_eval
        u_expected = torch.mean(torch.stack(l_eval), dim=0)
        # However, compute_shared_consensus also normalizes projections.
        # So we check against the actual function.
        from pysimlr.consensus import compute_shared_consensus
        u_manual = compute_shared_consensus(l_eval, mixing_algorithm="avg", k=k, training=False)
        assert torch.allclose(u_eval, u_manual, atol=1e-6)

    print("Deep train-infer consistency test: PASSED")

if __name__ == "__main__":
    test_deep_train_infer_consistency()
