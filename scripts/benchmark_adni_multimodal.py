import pandas as pd
import numpy as np
import os
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pysimlr.deep import lend_simr

def analyze_adni(file_path):
    print("=== ADNI Multi-view Benchmark ===")
    if not os.path.exists(file_path):
        print(f"[!] Warning: Data file {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Initial shape: {df.shape}")

    if 'DX' in df.columns:
        y = df['DX']
        print("\nDX label distribution:")
        print(y.value_counts(dropna=False))
    else:
        y = None

    prefixes = ['COG.', 'CSF.', 'MRI.', 'PET.', 'DTI.', 'AMY.', 'TAU.', 'PATH.']
    views_dict = {}
    
    # Preprocessing: Impute missing values and standardize
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    for p in prefixes:
        cols = [c for c in df.columns if str(c).startswith(p)]
        if cols:
            # Keep columns with at least 10% non-NaN data
            valid_cols = [c for c in cols if df[c].isna().mean() < 0.90]
            if not valid_cols:
                print(f"  Skipping {p:5s} (all columns have >90% missing data)")
                continue
                
            # Filter to numeric columns only
            numeric_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(df[c])]
            
            if not numeric_cols:
                print(f"  Skipping {p:5s} (no numeric columns after filtering)")
                continue

            view_data = df[numeric_cols].values
            
            nan_ratio = np.isnan(view_data).mean()
            print(f"View {p:5s}: {len(numeric_cols)} valid numeric columns (originally {len(cols)}), NaN ratio: {nan_ratio:.2%}")
            
            # Impute and Scale
            view_data_imp = imputer.fit_transform(view_data)
            view_data_sc = scaler.fit_transform(view_data_imp)
            
            views_dict[p] = view_data_sc

    if not views_dict:
        print("No views loaded.")
        return

    Xs = list(views_dict.values())
    view_names = list(views_dict.keys())
    
    print("\nPrepared Views for LEND-SiMLR:")
    for name, x in zip(view_names, Xs):
        print(f"  {name:5s}: {x.shape}")

    print("\nFitting LEND-Newton-LogCosh Gold Standard configuration...")
    start_time = time.time()
    
    # We use k=5 latent dimensions as a default
    res = lend_simr(
        Xs, k=5, epochs=100, 
        energy_type='logcosh', 
        mixing_algorithm='newton',
        positivity='positive', 
        sparseness_quantile=0.5,
        nsa_iterations=3,
        verbose=True
    )
    
    end_time = time.time()
    print(f"\nADNI Analysis Complete in {end_time - start_time:.2f}s.")
    
    # Optional downstream evaluation could go here
    if y is not None:
        print("Labels (DX) are available for downstream evaluation.")

    return res

if __name__ == "__main__":
    adni_path = "/Users/stnava/Downloads/ADNIMERGE2/ADNI_Master_Full_Prefix.csv"
    analyze_adni(adni_path)
