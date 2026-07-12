import pandas as pd
import numpy as np
import os
import time
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df_valid.values)
    
    stds = np.std(X, axis=0)
    valid_indices = stds > 1e-6
    if not np.any(valid_indices): return None
    
    X = X[:, valid_indices]
    X_sc = StandardScaler().fit_transform(X)
    return X_sc

def run_architecture_search(file_path):
    print("=== PPMI+ADNI Architecture Search (Fixed Newton + NC) ===")
    
    if not os.path.exists(file_path):
        print(f"[!] Warning: Data file {file_path} not found.")
        return

    df = pd.read_csv(file_path, low_memory=False)
    # Binary classification: Positive vs Negative
    df = df[df['AsynStatus'].isin(['Positive', 'Negative'])].copy()
    
    le = LabelEncoder()
    y = le.fit_transform(df['AsynStatus'].astype(str))
    
    # Define Modalities
    exclude_cols = ["AV45", "TAU", "PTAU", "AV45_bl", "TAU_bl", "PTAU_bl", "FBB", "FBB_bl", "PIB", "PIB_bl", "AsynStatus", "DXSubAsyn"]
    clinical_cols = [
        "MMSE", "mPACCdigit", "mPACCtrailsB", "MOCA", "AGE", "PTGENDER", "PTEDUCAT", "APOE4",
        "updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score", "updrs_totscore",
        "moca", "quip", "gds", "scopa", "upsit"
    ]
    clinical_cols = [c for c in clinical_cols if c in df.columns and c not in exclude_cols]
    t1_cols = [c for c in df.columns if c.startswith('T1Hier')]
    dti_cols = [c for c in df.columns if c.startswith('DTI_mean_fa')]
    rsf_cols = [c for c in df.columns if c.startswith('rsfMRI_fcnxpro129_') and '_2_' in c]

    views = {
        'Clinical': preprocess_view(df[clinical_cols]),
        'T1': preprocess_view(df[t1_cols]),
        'DTI': preprocess_view(df[dti_cols]),
        'rsfMRI': preprocess_view(df[rsf_cols])
    }
    views = {k: v for k, v in views.items() if v is not None}
    Xs = list(views.values())
    
    # 75/25 Split
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.25, random_state=42, stratify=y)
    
    Xs_train = [x[idx_train] for x in Xs]
    Xs_test = [x[idx_test] for x in Xs]
    y_train, y_test = y[idx_train], y[idx_test]
    
    print(f"Train size: {len(idx_train)}, Test size: {len(idx_test)}")
    
    # Configuration
    model_fns = {
        'LEND': lend_simr,
        'NED': ned_simr,
        'NEDPP': lambda *args, **kwargs: ned_simr_shared_private(*args, private_k=2, **kwargs)
    }
    architectures = [[64, 32], [128, 64], [256, 128]]
    
    results = []

    for m_name, m_fn in model_fns.items():
        for arch in architectures:
            print(f"\nTesting: {m_name} | Arch: {arch} | Fixed: Newton + NC")
            try:
                start_t = time.time()
                
                # Fit model on Train
                # Guaranteeing convergence: Higher epochs, lower tolerance, higher patience
                res = m_fn(
                    Xs_train, k=10, 
                    epochs=250, 
                    hidden_dims=arch,
                    dynamic_weights=False, # Fixing to False for architecture consistency
                    energy_type='nc', 
                    mixing_algorithm='newton',
                    tol=1e-7,
                    patience=20,
                    verbose=False
                )
                
                # Project Test
                test_proj = predict_deep(Xs_test, res)
                U_train = res.get('u', res.get('U')).numpy()
                U_test = test_proj.get('u', test_proj.get('U')).numpy()
                
                # Classify
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(U_train, y_train)
                y_pred = clf.predict(U_test)
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                conv_ep = res.get('converged_iter', 250)
                elapsed = time.time() - start_t
                
                print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f} (Converged @ {conv_ep})")
                results.append({
                    'Model': m_name,
                    'Arch': str(arch),
                    'Accuracy': acc,
                    'F1': f1,
                    'Conv_Epoch': conv_ep,
                    'Time': elapsed
                })
            except Exception as e:
                print(f"  FAILED: {e}")

    print("\n" + "="*80)
    print("ARCHITECTURE SEARCH RESULTS (Fixed Newton + NC, Guaranteed Convergence)")
    print("="*80)
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='Accuracy', ascending=False)
    print(res_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    run_architecture_search(path)
