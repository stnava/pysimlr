import torch
import numpy as np
from pysimlr.simlr import simlr, predict_simlr

def test_simlr_prediction_consistency():
    # 1. Create synthetic data
    n_total = 1000 # More samples
    n_train = 800
    n_test = 200
    d1, d2 = 50, 40 # Higher dimension
    k = 5
    
    torch.manual_seed(42)
    u_total = torch.randn(n_total, k)
    w1 = torch.randn(k, d1)
    w2 = torch.randn(k, d2)
    
    def norm_mat(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-6)

    x1_total = norm_mat(u_total @ w1 + 0.1 * torch.randn(n_total, d1))
    x2_total = norm_mat(u_total @ w2 + 0.1 * torch.randn(n_total, d2))
    
    x1_train = x1_total[:n_train]
    x2_train = x2_total[:n_train]
    x1_test = x1_total[n_train:]
    x2_test = x2_total[n_train:]
    
    # 2. Fit SIMLR - using "acc" which is more stable
    print("Fitting SIMLR with acc...")
    res = simlr([x1_train, x2_train], k=k, iterations=100, energy_type="acc")
    print("Fit complete.")
    
    # 3. Predict on test data
    print("Predicting on test data...")
    pred_test = predict_simlr([x1_test, x2_test], res)
    
    avg_error = np.mean(pred_test['errors'])
    print(f"Average reconstruction error on test data: {avg_error:.4f}")
    
    # 4. Compare with in-sample error
    pred_train = predict_simlr([x1_train, x2_train], res)
    avg_train_error = np.mean(pred_train['errors'])
    print(f"Average reconstruction error on train data: {avg_train_error:.4f}")

if __name__ == "__main__":
    test_simlr_prediction_consistency()
