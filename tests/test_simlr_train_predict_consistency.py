import torch
import numpy as np
from pysimlr.simlr import simlr, predict_simlr

def test_train_predict_consistency():
    # 1. Create synthetic data
    n_train = 200
    d1, d2 = 20, 15
    k = 3
    
    torch.manual_seed(42)
    u_train = torch.randn(n_train, k)
    w1 = torch.randn(k, d1)
    w2 = torch.randn(k, d2)
    
    def norm_mat(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-6)

    x1_train = norm_mat(u_train @ w1 + 0.1 * torch.randn(n_train, d1))
    x2_train = norm_mat(u_train @ w2 + 0.1 * torch.randn(n_train, d2))
    
    # 2. Fit SIMLR
    print("Fitting SIMLR...")
    res = simlr([x1_train, x2_train], k=k, iterations=50, energy_type="acc")
    print("Fit complete.")
    
    # 3. Predict on the same train data
    print("Predicting on train data...")
    pred_train = predict_simlr([x1_train, x2_train], res)
    
    # Compare pred_train['u'] with res['u']
    # They should be very close, but may differ by a sign flip per column
    u_res = res['u']
    u_pred = pred_train['u']
    
    for j in range(k):
        corr = np.abs(np.corrcoef(u_res[:, j], u_pred[:, j])[0, 1])
        print(f"Latent {j} correlation: {corr:.6f}")
        assert corr > 0.99
        
    # Compare reconstructions
    for i in range(2):
        x_orig_recon = res['u'] @ res['w'][i]
        x_pred_recon = pred_train['reconstructions'][i]
        diff = torch.norm(x_orig_recon - x_pred_recon) / torch.norm(x_orig_recon)
        print(f"Modality {i} reconstruction diff: {diff:.6e}")
        assert diff < 1e-2

    print("Train-Predict consistency test: PASSED")

if __name__ == "__main__":
    test_train_predict_consistency()
