"""Helpers shared by the analysis scripts in this directory.

Each was copied verbatim into two to five scripts; the copies are replaced by
imports from here (`from script_shared import ...`), which works for any
script run as ``python scripts/<name>.py``.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


# --- fused-penalty significance workers (evaluate_*significance*.py, evaluate_pairwise_and_gbev.py)
def get_logistic_deviance(y, p):
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    logL = np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return -2.0 * logL

def solve_fused_linear(X_t, Y_t, C_t, exposed_masks, lambda1, alpha, lambda2, sparsity_thresh, M, D, N, K):
    W_brain = nn.Parameter(torch.zeros(M, D))
    W_covs = nn.Parameter(torch.zeros(M, K))
    optimizer = optim.LBFGS([W_brain, W_covs], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
    sub_ys = [Y_t[mask] for mask in exposed_masks]
    subgroup_Zs = [X_t[mask] for mask in exposed_masks]
    subgroup_Cs = [C_t[mask] for mask in exposed_masks]
    
    def closure():
        optimizer.zero_grad()
        mse_loss = 0.0
        for j in range(M):
            Z_j = subgroup_Zs[j]
            C_j = subgroup_Cs[j]
            y_j = sub_ys[j]
            N_j = Z_j.shape[0]
            if N_j > 0:
                preds = Z_j @ W_brain[j] + C_j @ W_covs[j]
                mse_loss += torch.sum((y_j - preds) ** 2) / N_j
        mse_loss = 0.5 * (mse_loss / M)
        l2_pen = (1.0 - alpha) * lambda1 * torch.sum(W_brain ** 2)
        l1_pen = alpha * lambda1 * torch.sum(torch.sqrt(W_brain ** 2 + 1e-5))
        fused_pen = 0.0
        if M > 1:
            fused_pen = lambda2 * torch.sum((W_brain[1:] - W_brain[:-1]) ** 2)
        loss = mse_loss + l2_pen + l1_pen + fused_pen
        loss.backward()
        return loss
        
    optimizer.step(closure)
    W_brain_out = W_brain.detach().numpy()
    W_covs_out = W_covs.detach().numpy()
    if alpha > 0.0:
        W_brain_out[np.abs(W_brain_out) < sparsity_thresh] = 0.0
    return W_brain_out, W_covs_out

def solve_fused_logistic(X_t, Y_t, C_t, lambda1, alpha, lambda2, sparsity_thresh, M, D, N, K):
    W_brain = nn.Parameter(torch.zeros(M, D))
    W_covs = nn.Parameter(torch.zeros(M, K))
    optimizer = optim.LBFGS([W_brain, W_covs], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        logits = X_t @ W_brain.t() + C_t @ W_covs.t()
        bce_loss = 0.0
        for j in range(M):
            bce_loss += nn.functional.binary_cross_entropy_with_logits(logits[:, j], Y_t[:, j], reduction='mean')
        bce_loss = bce_loss / M
        l2_pen = (1.0 - alpha) * lambda1 * torch.sum(W_brain ** 2)
        l1_pen = alpha * lambda1 * torch.sum(torch.sqrt(W_brain ** 2 + 1e-5))
        fused_pen = 0.0
        if M > 1:
            fused_pen = lambda2 * torch.sum((W_brain[1:] - W_brain[:-1]) ** 2)
        loss = bce_loss + l2_pen + l1_pen + fused_pen
        loss.backward()
        return loss
        
    optimizer.step(closure)
    W_brain_out = W_brain.detach().numpy()
    W_covs_out = W_covs.detach().numpy()
    if alpha > 0.0:
        W_brain_out[np.abs(W_brain_out) < sparsity_thresh] = 0.0
    return W_brain_out, W_covs_out

def linear_worker(args):
    torch.set_num_threads(1)
    shuf_idx, mri_scaled, C_t, y_scaled, total_bl, percentiles, lambda1, alpha, lambda2, sparsity_thresh = args
    N = mri_scaled.shape[0]
    M = len(percentiles)
    D = mri_scaled.shape[1]
    K = C_t.shape[1]
    
    y_shuf = y_scaled[shuf_idx]
    exposed_masks = []
    for j in range(M):
        thresh = np.percentile(total_bl, percentiles[j])
        mask = total_bl > thresh
        exposed_masks.append(torch.as_tensor(mask).bool())
        
    X_t = torch.as_tensor(mri_scaled).float()
    Y_t = torch.as_tensor(y_shuf).float()
    C_torch = torch.as_tensor(C_t).float()
    
    W_brain, W_covs = solve_fused_linear(X_t, Y_t, C_torch, exposed_masks, lambda1, alpha, lambda2, sparsity_thresh, M, D, N, K)
    dev_sum = 0.0
    for j in range(M):
        mask = exposed_masks[j].numpy()
        Z_j = mri_scaled[mask]
        C_j = C_t[mask]
        y_j = y_shuf[mask]
        preds = Z_j @ W_brain[j] + C_j @ W_covs[j]
        dev_sum += np.sum((y_j - preds) ** 2)
    return dev_sum

def logistic_worker(args):
    torch.set_num_threads(1)
    shuf_idx, mri_scaled, C_t, total_bl, thresholds, lambda1, alpha, lambda2, sparsity_thresh = args
    N = mri_scaled.shape[0]
    M = len(thresholds)
    D = mri_scaled.shape[1]
    K = C_t.shape[1]
    
    shuf_total_bl = total_bl[shuf_idx]
    y_targets = np.zeros((N, M))
    for j in range(M):
        y_targets[:, j] = (shuf_total_bl >= thresholds[j]).astype(float)
        
    X_t = torch.as_tensor(mri_scaled).float()
    Y_t = torch.as_tensor(y_targets).float()
    C_torch = torch.as_tensor(C_t).float()
    
    W_brain, W_covs = solve_fused_logistic(X_t, Y_t, C_torch, lambda1, alpha, lambda2, sparsity_thresh, M, D, N, K)
    logits = mri_scaled @ W_brain.T + C_t @ W_covs.T
    probs = 1.0 / (1.0 + np.exp(-logits))
    dev_sum = 0.0
    for j in range(M):
        dev_sum += get_logistic_deviance(y_targets[:, j], probs[:, j])
    return dev_sum


# --- tabular view preprocessing (benchmark_adni_*.py, benchmark_ppmi_*.py)
def preprocess_view(df_sub, return_cols=False):
    """Numeric columns (categoricals with < 10 levels one-hot), < 90% missing,
    mean-imputed, near-constant columns dropped, standardized. With
    `return_cols`, also the kept column names."""
    df_num = pd.DataFrame(index=df_sub.index)
    for col in df_sub.columns:
        s = pd.to_numeric(df_sub[col], errors='coerce')
        if not s.isna().all():
            df_num[col] = s
        else:
            if df_sub[col].nunique() < 10:
                dummies = pd.get_dummies(df_sub[col], prefix=col)
                df_num = pd.concat([df_num, dummies], axis=1)
    
    valid_cols = [c for c in df_num.columns if df_num[c].isna().mean() < 0.90]
    if not valid_cols: return (None, None) if return_cols else None
    
    df_valid = df_num[valid_cols].astype(float)
    X_imp = SimpleImputer(strategy='mean').fit_transform(df_valid.values)
    
    stds = np.std(X_imp, axis=0)
    valid_indices = stds > 1e-6
    if not np.any(valid_indices): return (None, None) if return_cols else None
    
    final_cols = [valid_cols[i] for i in range(len(valid_cols)) if valid_indices[i]]
    X = X_imp[:, valid_indices]
    X_sc = StandardScaler().fit_transform(X)
    
    if return_cols:
        return X_sc, final_cols
    return X_sc

def preprocess_view_clipped(df_sub, return_cols=False):
    """Numeric columns only, < 90% missing, mean-imputed, standardized,
    clipped to [-10, 10] and standardized again (the "refined" variant)."""
    df_num = df_sub.apply(pd.to_numeric, errors='coerce')
    valid_cols = [c for c in df_num.columns if df_num[c].isna().mean() < 0.90]
    if not valid_cols: return (None, None) if return_cols else None
    
    df_valid = df_num[valid_cols].astype(float)
    X_imp = SimpleImputer(strategy='mean').fit_transform(df_valid.values)
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    X_clipped = np.clip(X_sc, -10, 10)
    X_final = scaler.fit_transform(X_clipped)
    
    if return_cols:
        return X_final, valid_cols
    return X_final


# --- data loaders
def load_expart_data():
    names = ["bl", "mh", "mr", "pe"]
    files = [f"data/expart_{n}.csv" for n in names]
    matrices = []
    for f in files:
        df = pd.read_csv(f)
        matrices.append(torch.as_tensor(df.values).float())
    return names, matrices

def load_brca_subtypes():
    df = pd.read_csv('data/BRCA/brca_data_w_subtypes.csv', on_bad_lines='skip')
    
    # Use ER.Status as target
    target_col = 'ER.Status'
    df = df[df[target_col].isin(['Positive', 'Negative'])].copy()
    
    rs_cols = [c for c in df.columns if c.startswith('rs_')]
    cn_cols = [c for c in df.columns if c.startswith('cn_')]
    pp_cols = [c for c in df.columns if c.startswith('pp_')]
    
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    
    Xs = [df[rs_cols].values.astype(float), 
          df[cn_cols].values.astype(float),
          df[pp_cols].values.astype(float)]
    
    return Xs, y, [rs_cols, cn_cols, pp_cols], le.classes_

def get_top_features_per_component(v_mat, feature_names, component=0, top_n=10):
    importance = np.abs(v_mat[:, component])
    top_indices = np.argsort(importance)[::-1][:top_n]
    return [feature_names[i] for i in top_indices], importance[top_indices]
