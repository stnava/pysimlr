import pandas as pd
import numpy as np
import os
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pysimlr.deep import lend_simr

def analyze_adni_predictive_mri_csf(file_path):
    print(f"=== ADNI Predictive Study: DEMO+MRI+COG -> CSF (Using MRI-Available Subset) ===")
    if not os.path.exists(file_path):
        print(f"[!] Warning: Data file {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Target Modality: CSF
    target_modality = 'CSF'
    target_cols = [c for c in df.columns if str(c).startswith(f"{target_modality}.")]
    valid_target_cols = [c for c in target_cols if df[c].notna().any() and pd.api.types.is_numeric_dtype(df[c])]
    
    # Input Modalities: DEMO, MRI, COG
    mri_cols = [c for c in df.columns if c.startswith('MRI.')]
    
    # Filter to rows that have BOTH MRI and CSF data
    df_subset = df.dropna(subset=mri_cols + valid_target_cols, how='any').copy()
    print(f"Rows with BOTH MRI and CSF data: {len(df_subset)}")
    
    if len(df_subset) < 20:
        print("Too few samples for a meaningful study. Try a different subset.")
        return

    views_dict = {}
    
    # 1. DEMO View
    print("\nPreparing DEMO view...")
    demo_cols = ['AGE', 'SEX', 'EDUC', 'APOE']
    df_demo = df_subset[demo_cols].copy()
    df_demo['AGE'] = df_demo['AGE'].fillna(df_demo['AGE'].mean())
    df_demo['EDUC'] = df_demo['EDUC'].fillna(df_demo['EDUC'].mean())
    df_demo['SEX'] = df_demo['SEX'].fillna('Unknown')
    df_demo['APOE'] = df_demo['APOE'].fillna('Unknown')
    demo_encoded = pd.get_dummies(df_demo, columns=['SEX', 'APOE'], drop_first=True).astype(float)
    views_dict['DEMO'] = StandardScaler().fit_transform(demo_encoded.values)
    print(f"  DEMO: {views_dict['DEMO'].shape}")

    # 2. MRI and COG Views
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    for p in ['MRI.', 'COG.']:
        cols = [c for c in df.columns if str(c).startswith(p)]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        # Keep columns with < 50% missing in this specific subset
        valid_cols = [c for c in numeric_cols if df_subset[c].isna().mean() < 0.50]
        
        if not valid_cols:
            print(f"  Skipping {p:5s}")
            continue

        view_data = df_subset[valid_cols].values
        print(f"  {p:5s}: {len(valid_cols)} valid numeric cols, NaN ratio: {np.isnan(view_data).mean():.2%}")
        
        view_data_imp = imputer.fit_transform(view_data)
        views_dict[p] = scaler.fit_transform(view_data_imp)

    Xs = list(views_dict.values())
    view_names = list(views_dict.keys())
    
    # 3. Extract Latent Features using LEND-SiMLR
    print(f"\nFitting LEND-SiMLR on {view_names} to extract latents...")
    k_latents = 8
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
    
    # 4. Predictive Modeling
    print(f"\n--- Predicting {target_modality} from Latent Features U ---")
    Y_raw = df_subset[valid_target_cols].values
    Y_imp = imputer.fit_transform(Y_raw)
    Y_sc = scaler.fit_transform(Y_imp)
    
    X_train, X_test, Y_train, Y_test = train_test_split(U, Y_sc, test_size=0.2, random_state=42)
    
    print(f"Training Ridge Regressor on {X_train.shape[0]} samples...")
    model = RidgeCV(alphas=np.logspace(-2, 4, 10))
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    r2_scores = r2_score(Y_test, Y_pred, multioutput='raw_values')
    
    print("\nPrediction Results (across CSF features):")
    print(f"  Mean R^2: {np.mean(r2_scores):.4f} (Max: {np.max(r2_scores):.4f})")
    
    best_target_idx = np.argmax(r2_scores)
    print(f"  Best predicted CSF feature: {valid_target_cols[best_target_idx]} (R^2 = {r2_scores[best_target_idx]:.4f})")
    
    return res, model, r2_scores

if __name__ == "__main__":
    adni_path = "/Users/stnava/Downloads/ADNIMERGE2/ADNI_Master_Full_Prefix.csv"
    analyze_adni_predictive_mri_csf(adni_path)
