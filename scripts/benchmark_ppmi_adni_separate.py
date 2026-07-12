import pandas as pd
import numpy as np
import os
import time
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
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

def run_separate_study(file_path, study_name):
    print(f"\n{'='*60}")
    print(f"RUNNING STUDY: {study_name} (LEND | regression | newton)")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path, low_memory=False)
    df = df[df['studyName'] == study_name].copy()
    df = df[df['AsynStatus'].isin(['Positive', 'Negative'])].copy()
    
    if len(df) < 20:
        print(f"Too few samples for {study_name} ({len(df)})")
        return None

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
    
    print(f"Samples: Total={len(df)}, Train={len(idx_train)}, Test={len(idx_test)}")
    print(f"Modality Coverage: {list(views.keys())}")
    
    start_t = time.time()
    res = lend_simr(
        Xs_train, k=10, epochs=150, 
        hidden_dims=[64, 32],
        energy_type='regression', 
        mixing_algorithm='newton',
        tol=1e-7, patience=20,
        verbose=False
    )
    
    test_proj = predict_deep(Xs_test, res)
    U_train = res.get('u', res.get('U')).numpy()
    U_test = test_proj.get('u', test_proj.get('U')).numpy()
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(U_train, y_train)
    y_pred = clf.predict(U_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    elapsed = time.time() - start_t
    
    print(f"\nResults for {study_name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return {'Study': study_name, 'Accuracy': acc, 'F1': f1}

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    
    adni_res = run_separate_study(path, 'ADNI')
    ppmi_res = run_separate_study(path, 'PPMI')
    
    print("\n" + "="*60)
    print("FINAL COMPARISON: ADNI vs PPMI")
    print("="*60)
    results = [r for r in [adni_res, ppmi_res] if r is not None]
    print(pd.DataFrame(results).to_string(index=False))
