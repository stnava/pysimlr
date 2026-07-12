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

def run_multimodal_study(file_path, target_type='AsynStatus', study_name='Both'):
    """
    Run a multimodal study using LEND-SiMLR latents to predict a target outcome.
    
    target_type: 'AT_Status', 'AsynStatus', or 'DXSubAsyn'
    study_name: 'ADNI', 'PPMI', or 'Both'
    """
    print(f"=== Study: {study_name} | Target: {target_type} ===")
    
    if not os.path.exists(file_path):
        print(f"[!] Warning: Data file {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    # 1. Filter by Study
    if study_name != 'Both':
        df = df[df['studyName'] == study_name].copy()
    print(f"Subset shape after study filter: {df.shape}")

    # 2. Prepare Target
    if target_type == 'AT_Status':
        # Amyloid positive: AV45 > 1.11 or FBB > 1.08
        # Tau positive: PTAU > 19.2
        amy = df['AV45_bl'].combine_first(df['FBB_bl'])
        tau = df['PTAU_bl'].combine_first(df['TAU_bl'])
        
        def get_at_status(a, t):
            if pd.isna(a) or pd.isna(t): return np.nan
            a_pos, t_pos = a > 1.11, t > 19.2
            return f"A{'+' if a_pos else '-'}/T{'+' if t_pos else '-'}"

        df['Target'] = [get_at_status(a, t) for a, t in zip(amy, tau)]
    elif target_type == 'AsynStatus':
        df['Target'] = df['AsynStatus']
        # Filter out PosMSA
        df = df[df['Target'] != 'PosMSA'].copy()
        print("Filtered out 'PosMSA' class from AsynStatus.")
    elif target_type == 'DXSubAsyn':
        df['Target'] = df['DXSubAsyn']
    else:
        print(f"Unknown target_type: {target_type}")
        return

    # Drop rows with missing target
    df = df.dropna(subset=['Target']).copy()
    print(f"Shape after filtering for target: {df.shape}")
    
    if len(df) < 5:
        print("Too few samples for analysis.")
        return

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['Target'].astype(str))
    target_names = le.classes_
    print(f"Target classes: {target_names}")
    print(f"Class distribution:\n{df['Target'].value_counts()}")

    # 3. Define Modalities
    views_dict = {}
    
    # Cognitive / Clinical / Biomarkers
    exclude_cols = ["AV45", "TAU", "PTAU", "AV45_bl", "TAU_bl", "PTAU_bl", "FBB", "FBB_bl", "PIB", "PIB_bl", "AsynStatus", "DXSubAsyn", "Target"]
    clinical_cols = [
        # ADNI-centric
        "MMSE", "mPACCdigit", "mPACCtrailsB", "MOCA", "AGE",
        "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY", "APOE4",
        "ABETA", "CDRSB", "ADAS11", "ADAS13", "ADASQ4",
        "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
        "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ",
        "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogPtTotal",
        "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "EcogSPTotal",
        # PPMI-centric
        "updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score", "updrs_totscore",
        "moca", "quip", "gds", "scopa", "upsit", "upsit_pctl"
    ]
    clinical_cols = [c for c in clinical_cols if c in df.columns and c not in exclude_cols]
    
    # Brain Structure & Function
    t1_cols = [c for c in df.columns if c.startswith('T1Hier')]
    dti_cols = [c for c in df.columns if c.startswith('DTI_mean_fa')]
    rsf_cols = [c for c in df.columns if c.startswith('rsfMRI_fcnxpro129_') and '_2_' in c]

    def preprocess_view(df_sub):
        df_num = pd.DataFrame(index=df_sub.index)
        for col in df_sub.columns:
            # Force numeric conversion
            s = pd.to_numeric(df_sub[col], errors='coerce')
            if not s.isna().all():
                df_num[col] = s
            else:
                # Handle categoricals
                if df_sub[col].nunique() < 10:
                    dummies = pd.get_dummies(df_sub[col], prefix=col)
                    df_num = pd.concat([df_num, dummies], axis=1)
        
        # Filter columns with too much missing data
        valid_cols = [c for c in df_num.columns if df_num[c].isna().mean() < 0.90]
        if not valid_cols: return None
        df_valid = df_num[valid_cols].astype(float)
        
        # Impute
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(df_valid.values)
        
        # Scale (and filter constant columns)
        stds = np.std(X, axis=0)
        valid_indices = stds > 1e-6
        if not np.any(valid_indices): return None
        
        X = X[:, valid_indices]
        X_sc = StandardScaler().fit_transform(X)
        return X_sc

    views_dict['Clinical'] = preprocess_view(df[clinical_cols])
    views_dict['T1'] = preprocess_view(df[t1_cols])
    views_dict['DTI'] = preprocess_view(df[dti_cols])
    views_dict['rsfMRI'] = preprocess_view(df[rsf_cols])

    views_dict = {k: v for k, v in views_dict.items() if v is not None}
    print(f"Views used: {list(views_dict.keys())}")
    for k, v in views_dict.items():
        print(f"  {k:10s}: {v.shape}")

    if not views_dict:
        print("No valid views found.")
        return

    Xs = list(views_dict.values())
    
    # 4. Extract Latents
    k_latents = min(15, len(df) // 4)
    if k_latents < 2: k_latents = 2
    
    print(f"\nFitting LEND-SiMLR (k={k_latents})...")
    
    res = lend_simr(
        Xs, k=k_latents, epochs=100, 
        energy_type='logcosh', 
        mixing_algorithm='newton',
        positivity='positive', 
        sparseness_quantile=0.5,
        nsa_iterations=3,
        verbose=False
    )
    
    U = res.get('u', res.get('U')).numpy()
    
    # 5. Evaluate
    print(f"\n--- Prediction Results for {target_type} ---")
    if len(df) < 20:
        print("Dataset too small for reliable classification.")
    else:
        if len(df) < 50:
            cv_folds = 3
        else:
            cv_folds = 5
            
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(clf, U, y, cv=cv)
        print(f"CV Accuracy ({cv_folds}-fold): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        # Show report on a single split
        X_train, X_test, y_train, y_test = train_test_split(U, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("\nClassification Report (Sample Split):")
        print(classification_report(y_test, y_pred, target_names=[str(c) for c in target_names], zero_division=0))

    return res

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    
    # Run for AsynStatus in both ADNI and PPMI combined, excluding PosMSA
    run_multimodal_study(path, target_type='AsynStatus', study_name='Both')
