import pandas as pd
import numpy as np
import os
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from pysimlr.deep import lend_simr

def analyze_ppmi_adni_at_status(file_path):
    print("=== ADNI Multi-modal Study: A/T Status Prediction ===")
    if not os.path.exists(file_path):
        print(f"[!] Warning: Data file {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    df = df[df['studyName'] == 'ADNI'].copy()
    print(f"ADNI subset shape: {df.shape}")

    # 2. Prepare A/T Positivity Target
    amy = df['AV45_bl'].combine_first(df['FBB_bl'])
    tau = df['PTAU_bl'].combine_first(df['TAU_bl'])
    
    def get_at_status(a, t):
        if pd.isna(a) or pd.isna(t):
            return np.nan
        a_pos = a > 1.11
        t_pos = t > 19.2
        a_str = "+" if a_pos else "-"
        t_str = "+" if t_pos else "-"
        return f"A{a_str}/T{t_str}"

    df['AT_Status'] = [get_at_status(a, t) for a, t in zip(amy, tau)]
    df = df.dropna(subset=['AT_Status']).copy()
    print(f"Shape after filtering for A/T status: {df.shape}")
    
    if len(df) < 5:
        print("Too few samples with both Amyloid and Tau data.")
        return

    le = LabelEncoder()
    y = le.fit_transform(df['AT_Status'])
    target_names = le.classes_
    print(f"Target classes: {target_names}")
    print(f"Class distribution:\n{df['AT_Status'].value_counts()}")

    # 3. Define Modalities
    views_dict = {}
    
    exclude_cols = ["AV45", "TAU", "PTAU", "AV45_bl", "TAU_bl", "PTAU_bl", "FBB", "FBB_bl", "PIB", "PIB_bl"]
    cognition_cols = [
        "MMSE", "mPACCdigit", "mPACCtrailsB", "MOCA", "AGE",
        "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY", "APOE4",
        "ABETA", "CDRSB", "ADAS11", "ADAS13", "ADASQ4",
        "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
        "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ",
        "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogPtTotal",
        "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "EcogSPTotal"
    ]
    cognition_cols = [c for c in cognition_cols if c in df.columns and c not in exclude_cols]
    t1_cols = [c for c in df.columns if c.startswith('T1Hier')]
    dti_cols = [c for c in df.columns if c.startswith('DTI_mean_fa')]
    rsf_cols = [c for c in df.columns if c.startswith('rsfMRI_fcnxpro129_') and '_2_' in c]

    def preprocess_view(df_sub):
        df_num = pd.DataFrame(index=df_sub.index)
        for col in df_sub.columns:
            if pd.api.types.is_numeric_dtype(df_sub[col]):
                df_num[col] = df_sub[col]
            else:
                if df_sub[col].nunique() < 10:
                    dummies = pd.get_dummies(df_sub[col], prefix=col)
                    df_num = pd.concat([df_num, dummies], axis=1)
        
        valid_cols = [c for c in df_num.columns if df_num[c].isna().mean() < 0.90]
        if not valid_cols: return None
        X = df_num[valid_cols].values
        X_imp = SimpleImputer(strategy='mean').fit_transform(X)
        X_sc = StandardScaler().fit_transform(X_imp)
        return X_sc

    views_dict['Cognition'] = preprocess_view(df[cognition_cols])
    views_dict['T1'] = preprocess_view(df[t1_cols])
    views_dict['DTI'] = preprocess_view(df[dti_cols])
    views_dict['rsfMRI'] = preprocess_view(df[rsf_cols])

    views_dict = {k: v for k, v in views_dict.items() if v is not None}
    print(f"Views used: {list(views_dict.keys())}")
    for k, v in views_dict.items():
        print(f"  {k:10s}: {v.shape}")

    Xs = list(views_dict.values())
    
    k_latents = min(8, len(df) // 2)
    print(f"\nFitting LEND-SiMLR (k={k_latents})...")
    
    res = lend_simr(
        Xs, k=k_latents, epochs=150, 
        energy_type='logcosh', 
        mixing_algorithm='newton',
        positivity='positive', 
        sparseness_quantile=0.5,
        nsa_iterations=3,
        verbose=False
    )
    
    U = res.get('u', res.get('U')).numpy()
    
    print("\n--- Evaluating Classification (A/T Status) ---")
    
    if len(df) < 20:
        print("Dataset too small for reliable classification.")
    else:
        n_splits = min(3, len(df) // 10)
        if n_splits < 2:
            X_train, X_test, y_train, y_test = train_test_split(U, y, test_size=0.3, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=[str(c) for c in target_names], zero_division=0))
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(clf, U, y, cv=cv)
            print(f"CV Accuracy ({n_splits}-fold): {np.mean(scores):.4f}")
            
            X_train, X_test, y_train, y_test = train_test_split(U, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("\nClassification Report (Sample Split):")
            print(classification_report(y_test, y_pred, target_names=[str(c) for c in target_names], zero_division=0))

    return res

if __name__ == "__main__":
    ppmi_adni_path = "../multidisorder/data/ppmiadni_filtered.csv"
    analyze_ppmi_adni_at_status(ppmi_adni_path)
