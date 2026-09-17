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
    # forward() returns one consensus per modality under the default "loo"
    # topology, so compare element-wise rather than assuming a single tensor.
    assert type(u1) is type(u2)
    if isinstance(u1, list):
        assert len(u1) == len(u2)
        for ua, ub in zip(u1, u2):
            assert torch.allclose(ua, ub, atol=1e-6)
    else:
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
        from pysimlr.consensus import compute_shared_consensus
        # Reproduce the model's own consensus call, including its topology --
        # the default is "loo", which yields one consensus per modality.
        u_manual = compute_shared_consensus(
            l_eval, mixing_algorithm="avg", k=k, training=False,
            topology=model_avg.topology,
        )
        assert type(u_eval) is type(u_manual)
        if isinstance(u_eval, list):
            for ua, ub in zip(u_eval, u_manual):
                assert torch.allclose(ua, ub, atol=1e-6)
        else:
            assert torch.allclose(u_eval, u_manual, atol=1e-6)

    print("Deep train-infer consistency test: PASSED")

if __name__ == "__main__":
    test_deep_train_infer_consistency()


def test_train_and_eval_agree_for_every_mixing_algorithm():
    """The case the test above deliberately sidesteps.

    With the SVD/PCA/ICA mixing algorithms, eval() takes the anchored path
    while train() recomputes the basis. Those used to disagree on scale: the
    anchor branch returned before the shared standardization block, so eval
    latents came back roughly half the size of the training ones.
    """
    torch.manual_seed(7)
    n, d1, d2, k = 60, 10, 8, 3
    z = torch.randn(n, k)
    x = [
        z @ torch.randn(k, d1) + 0.1 * torch.randn(n, d1),
        z @ torch.randn(k, d2) + 0.1 * torch.randn(n, d2),
    ]

    for alg in ["svd", "pca", "avg", "newton"]:
        res = lend_simr(
            x, k=k, epochs=8, warmup_epochs=2, mixing_algorithm=alg,
            topology="star", dropout=0.0, verbose=False,
        )
        model = res["model"]

        model.train()
        with torch.no_grad():
            _, _, u_tr = model(x)
        model.eval()
        with torch.no_grad():
            _, _, u_ev = model(x)

        assert not isinstance(u_tr, list) and not isinstance(u_ev, list)
        tr_std = u_tr.std(dim=0)
        ev_std = u_ev.std(dim=0)
        ratio = (ev_std / (tr_std + 1e-8))
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=0.05), (
            f"{alg}: eval/train latent scale ratio {ratio.tolist()} -- the "
            f"anchored prediction path is not standardized like training"
        )
