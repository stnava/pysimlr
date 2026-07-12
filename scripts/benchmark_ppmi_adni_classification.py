import pandas as pd
import numpy as np
import os
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pysimlr.deep import lend_simr

def analyze_ppmi_adni(file_path):
    print("=== PPMI-ADNI Multi-modal Study (ADNI Subset) ===")
    if not os.path.exists(file_path):
        print(f"[!] Warning: Data file {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    # Loading carefully due to potential CSV inconsistencies seen earlier
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"Initial shape: {df.shape}")

    # 1. Filter for ADNI study
    if 'studyName' not in df.columns:
        print("Error: 'studyName' column not found.")
        return
    
    df = df[df['studyName'] == 'ADNI'].copy()
    print(f"ADNI subset shape: {df.shape}")

    # 2. Prepare Target (DXSubAsyn)
    target_col = 'DXSubAsyn'
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return
        
    # Drop rows with missing target
    df = df.dropna(subset=[target_col]).copy()
    print(f"Shape after dropping missing targets: {df.shape}")
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    target_names = le.classes_
    print(f"Target classes: {target_names}")
    print(f"Class distribution: {df[target_col].value_counts().to_dict()}")

    # 3. Define Modalities
    views_dict = {}
    
    # Modality 1: Cognition & Biomarkers
    cognition_cols = [
        "MMSE", "mPACCdigit", "mPACCtrailsB", "MOCA", "AGE",
        "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY", "APOE4",
        "FDG", "PIB", "AV45", "FBB", "ABETA", "TAU", "PTAU", 
        "CDRSB", "ADAS11", "ADAS13", "ADASQ4",
        "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
        "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ",
        "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogPtTotal",
        "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "EcogSPTotal"
    ]
    # Filter to existing columns
    cognition_cols = [c for c in cognition_cols if c in df.columns]
    print(f"Found {len(cognition_cols)} of {len(list(set(cognition_cols)))} requested cognition/biomarker columns.")
    
    # Modality 2: T1 (prefix T1Hier)
    t1_cols = [c for c in df.columns if c.startswith('T1Hier')]
    print(f"Found {len(t1_cols)} T1Hier columns.")
    
    # Modality 3: DTI (prefix DTI_mean_fa)
    dti_cols = [c for c in df.columns if c.startswith('DTI_mean_fa')]
    print(f"Found {len(dti_cols)} DTI_mean_fa columns.")
    
    # Modality 4: rsfMRI (prefix rsfMRI_fcnxpro129_ and contains _2_)
    rsf_cols = [c for c in df.columns if c.startswith('rsfMRI_fcnxpro129_') and '_2_' in c]
    print(f"Found {len(rsf_cols)} rsfMRI columns.")

    # 4. Preprocess and Scale
    def preprocess_view(df_sub, name):
        # Convert to numeric, handle categoricals with dummies
        df_num = pd.DataFrame(index=df_sub.index)
        for col in df_sub.columns:
            if pd.api.types.is_numeric_dtype(df_sub[col]):
                df_num[col] = df_sub[col]
            else:
                # Basic one-hot for small categoricals
                if df_sub[col].nunique() < 10:
                    dummies = pd.get_dummies(df_sub[col], prefix=col)
                    df_num = pd.concat([df_num, dummies], axis=1)
        
        # Impute and Scale
        if df_num.empty:
            return None
            
        # Keep columns with < 90% missing
        valid_cols = [c for c in df_num.columns if df_num[c].isna().mean() < 0.90]
        if not valid_cols:
            return None
            
        X = df_num[valid_cols].values
        X_imp = SimpleImputer(strategy='mean').fit_transform(X)
        X_sc = StandardScaler().fit_transform(X_imp)
        return X_sc

    views_dict['Cognition'] = preprocess_view(df[cognition_cols], 'Cognition')
    views_dict['T1'] = preprocess_view(df[t1_cols], 'T1')
    views_dict['DTI'] = preprocess_view(df[dti_cols], 'DTI')
    views_dict['rsfMRI'] = preprocess_view(df[rsf_cols], 'rsfMRI')

    # Filter out empty views
    views_dict = {k: v for k, v in views_dict.items() if v is not None}
    print(f"Views used: {list(views_dict.keys())}")
    for k, v in views_dict.items():
        print(f"  {k:10s}: {v.shape}")

    if not views_dict:
        print("No valid views found.")
        return

    Xs = list(views_dict.values())
    
    # 5. Extract Shared Latent Space
    print("\nFitting LEND-SiMLR...")
    start_time = time.time()
    
    res = lend_simr(
        Xs, k=15, epochs=150, 
        energy_type='logcosh', 
        mixing_algorithm='newton',
        positivity='positive', 
        sparseness_quantile=0.5,
        nsa_iterations=3,
        verbose=True
    )
    
    end_time = time.time()
    print(f"Latent extraction completed in {end_time - start_time:.2f}s.")
    
    U = res.get('u', res.get('U')).numpy()
    
    # 6. Classification Performance
    print("\n--- Evaluating Classification (DXSubAsyn) from Shared Latents ---")
    X_train, X_test, y_train, y_test = train_test_split(U, y, test_size=0.3, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in target_names]))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return res, clf

if __name__ == "__main__":
    ppmi_adni_path = "../multidisorder/data/ppmiadni_filtered.csv"
    analyze_ppmi_adni(ppmi_adni_path)
