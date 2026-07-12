import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from pysimlr.deep import lend_simr
from pysimlr.deep_opt import lend_simr_optimized
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import os

# Setup Data
data = load_diabetes()
X = StandardScaler().fit_transform(data.data)
y = data.target
train_size = int(len(X) * 0.7)

# Split into two views
X_train_v1 = X[:train_size, :5]
X_train_v2 = X[:train_size, 5:]
X_test_v1 = X[train_size:, :5]
X_test_v2 = X[train_size:, 5:]
y_train = y[:train_size]
y_test = y[train_size:]

def get_r2(u_train, u_test, y_train, y_test):
    # Standard Ridge probe used in our benchmarks
    reg = Ridge(alpha=1.0).fit(u_train, y_train)
    y_pred = reg.predict(u_test)
    return r2_score(y_test, y_pred)

print("--- Head-to-Head: Baseline vs Optimized (Diabetes) ---")

# 1. Baseline
torch.manual_seed(42)
res_base = lend_simr([X_train_v1, X_train_v2], k=2, epochs=100, positivity='positive', sparseness_quantile=0.5, nsa_iterations=3, verbose=False)
u_train_base = res_base['u'].detach().numpy()

from pysimlr.deep import predict_deep
pred_base = predict_deep([X_test_v1, X_test_v2], res_base)
u_test_base = pred_base['u'].detach().numpy()

r2_base = get_r2(u_train_base, u_test_base, y_train, y_test)
print(f"Baseline LEND Test R2: {r2_base:.4f}")

# 2. Optimized (BatchNorm + LeakyReLU)
torch.manual_seed(42)
res_opt = lend_simr_optimized([X_train_v1, X_train_v2], k=2, epochs=100, positivity='positive', sparseness_quantile=0.5, nsa_iterations=3)
u_train_opt = res_opt['u'].detach().numpy()

model_opt = res_opt['model']
model_opt.eval()
with torch.no_grad():
    _, _, u_test_opt_tensor = model_opt([torch.tensor(X_test_v1).float(), torch.tensor(X_test_v2).float()])
u_test_opt = u_test_opt_tensor.detach().numpy()

r2_opt = get_r2(u_train_opt, u_test_opt, y_train, y_test)
print(f"Optimized LEND Test R2: {r2_opt:.4f}")

if r2_opt > r2_base:
    print(f"\nSUCCESS: Optimized architecture beat baseline by {r2_opt - r2_base:.4f}")
else:
    print(f"\nFAILURE: Baseline was more robust for this dataset.")
