import pandas as pd
import numpy as np
import os
import time
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from pysimlr.deep import lend_simr, predict_deep

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

def run_adni_imaging_cdrsb_study(file_path):
    print("=== ADNI Imaging-Only Study: Predicting CDRSB_bl ===")
    
    df = pd.read_csv(file_path, low_memory=False)
    df = df[df['studyName'] == 'ADNI'].copy()
    
    # Target
    df = df.dropna(subset=['CDRSB_bl']).copy()
    y = df['CDRSB_bl'].values
    
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
    
    print(f"Sample Size: {len(y)}")
    print(f"Imaging Views: {list(views.keys())}")
    for k, v in views.items():
        print(f"  {k:10s}: {v.shape}")

    # 75/25 Split
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.25, random_state=42)
    
    Xs_train = [x[idx_train] for x in Xs]
    Xs_test = [x[idx_test] for x in Xs]
    y_train, y_test = y[idx_train], y[idx_test]

    print("\nFitting LEND-SiMLR on neuroimaging data...")
    res = lend_simr(
        Xs_train, k=12, epochs=200, 
        hidden_dims=[64, 32],
        energy_type='regression', 
        mixing_algorithm='newton',
        tol=1e-7, patience=20,
        verbose=False
    )
    
    # Project
    test_proj = predict_deep(Xs_test, res)
    U_train = res.get('u', res.get('U')).numpy()
    U_test = test_proj.get('u', test_proj.get('U')).numpy()
    
    # Predict CDRSB
    print("\n--- Regression Results (Neuro-Imaging Latents -> CDRSB_bl) ---")
    reg = RidgeCV(alphas=np.logspace(-3, 3, 10))
    reg.fit(U_train, y_train)
    
    y_pred = reg.predict(U_test)
    # Clip negative predictions as CDRSB cannot be < 0
    y_pred = np.maximum(0, y_pred)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    corr, pval = pearsonr(y_test, y_pred)
    
    print(f"  R^2 Score: {r2:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  Correlation (rho): {corr:.4f} (p={pval:.4e})")

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    run_adni_imaging_cdrsb_study(path)
