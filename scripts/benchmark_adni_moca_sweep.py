import pandas as pd
import numpy as np
import os
import time
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private, predict_deep

def preprocess_view(df_sub):
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
    if not valid_cols: return None
    df_valid = df_num[valid_cols].astype(float)
    X_imp = SimpleImputer(strategy='mean').fit_transform(df_valid.values)
    stds = np.std(X_imp, axis=0)
    valid_indices = stds > 1e-6
    if not np.any(valid_indices): return None
    X = X_imp[:, valid_indices]
    return StandardScaler().fit_transform(X)

def run_moca_model_sweep(file_path):
    print("=== ADNI MOCA Model Sweep (LEND vs NED vs NEDPP | MAI on/off) ===")
    
    df = pd.read_csv(file_path, low_memory=False)
    df = df[df['studyName'] == 'ADNI'].copy()
    
    # Target
    target_col = 'MOCA_bl'
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].values
    
    # Imaging Modalities
    t1_cols = [c for c in df.columns if c.startswith('T1Hier')]
    dti_cols = [c for c in df.columns if c.startswith('DTI_mean_fa')]
    rsf_cols = [c for c in df.columns if c.startswith('rsfMRI_fcnxpro129_') and '_2_' in c]

    views = {
        'T1': preprocess_view(df[t1_cols]),
        'DTI': preprocess_view(df[dti_cols]),
        'rsfMRI': preprocess_view(df[rsf_cols])
    }
    views = {k: v for k, v in views.items() if v is not None}
    Xs = list(views.values())
    
    # 75/25 Split
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.25, random_state=42)
    Xs_train = [x[idx_train] for x in Xs]
    Xs_test = [x[idx_test] for x in Xs]
    y_train, y_test = y[idx_train], y[idx_test]

    model_fns = {
        'LEND': lend_simr,
        'NED': ned_simr,
        'NEDPP': lambda *args, **kwargs: ned_simr_shared_private(*args, private_k=2, **kwargs)
    }
    mai_options = [False, True]
    
    results = []

    for m_name, m_fn in model_fns.items():
        for mai in mai_options:
            print(f"Testing: {m_name:5s} | MAI: {str(mai):5s} ...", end=" ", flush=True)
            try:
                start_t = time.time()
                res = m_fn(
                    Xs_train, k=10, epochs=100, 
                    hidden_dims=[64, 32],
                    dynamic_weights=mai,
                    energy_type='regression', 
                    mixing_algorithm='newton',
                    verbose=False
                )
                
                test_proj = predict_deep(Xs_test, res)
                U_train = res.get('u', res.get('U')).numpy()
                U_test = test_proj.get('u', test_proj.get('U')).numpy()
                
                reg = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(U_train, y_train)
                y_pred = reg.predict(U_test)
                
                r2 = r2_score(y_test, y_pred)
                corr, _ = pearsonr(y_test, y_pred)
                elapsed = time.time() - start_t
                
                print(f"DONE. R2: {r2:.4f}, Corr: {corr:.4f}")
                results.append({
                    'Model': m_name,
                    'MAI': mai,
                    'R2': r2,
                    'Corr': corr,
                    'Time': elapsed
                })
            except Exception as e:
                print(f"FAILED: {e}")

    print("\n" + "="*70)
    print("MOCA PREDICTION LEADERBOARD (ADNI Imaging Only)")
    print("="*70)
    res_df = pd.DataFrame(results).sort_values(by='Corr', ascending=False)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    run_moca_model_sweep(path)
