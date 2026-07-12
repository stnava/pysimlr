import pandas as pd
import numpy as np
import os
import time
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private, predict_deep

def preprocess_view(df_sub):
    df_num = df_sub.apply(pd.to_numeric, errors='coerce')
    valid_cols = [c for c in df_num.columns if df_num[c].isna().mean() < 0.90]
    if not valid_cols: return None
    df_valid = df_num[valid_cols].astype(float)
    X_imp = SimpleImputer(strategy='mean').fit_transform(df_valid.values)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    X_clipped = np.clip(X_sc, -10, 10)
    return scaler.fit_transform(X_clipped)

def run_final_sweep(file_path):
    print("=== FINAL REFINED SWEEP: PPMI+ADNI MOCA Prediction ===")
    
    df = pd.read_csv(file_path, low_memory=False)
    df['target'] = df['MOCA'].combine_first(df['moca'])
    df = df.dropna(subset=['target']).copy()
    y = df['target'].values
    
    forbidden = ['Rand', 'ackground', 'lassified']
    def get_cols(prefix):
        cols = [c for c in df.columns if c.startswith(prefix)]
        cols = [c for c in cols if 'adjusted' in c]
        cols = [c for c in cols if not any(f in c for f in forbidden)]
        return cols

    t1_cols = get_cols('T1Hier')
    dti_cols = get_cols('DTI_mean_fa')
    rsf_cols = [c for c in get_cols('rsfMRI_fcnxpro129_') if '_2_' in c]

    Xs = [preprocess_view(df[t1_cols]), preprocess_view(df[dti_cols]), preprocess_view(df[rsf_cols])]
    Xs = [x for x in Xs if x is not None]
    
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42)
    Xs_train = [x[idx_train] for x in Xs]
    Xs_test = [x[idx_test] for x in Xs]
    y_train, y_test = y[idx_train], y[idx_test]

    model_options = {
        'LEND': lend_simr,
        'NED': ned_simr,
        'NEDPP': lambda *args, **kwargs: ned_simr_shared_private(*args, private_k=2, **kwargs)
    }
    arch_options = [[64, 32], [128, 64]]
    mai_options = [False, True]
    energy_options = ['nc', 'regression']
    
    results = []

    for m_name, m_fn in model_options.items():
        for arch in arch_options:
            for mai in mai_options:
                for energy in energy_options:
                    print(f"Testing: {m_name:5s} | Arch: {str(arch):10s} | MAI: {str(mai):5s} | Energy: {energy:10s} ...", end=" ", flush=True)
                    try:
                        start_t = time.time()
                        res = m_fn(
                            Xs_train, k=10, epochs=100, 
                            hidden_dims=arch,
                            dynamic_weights=mai,
                            energy_type=energy, 
                            mixing_algorithm='newton',
                            tol=1e-7, patience=15,
                            verbose=False
                        )
                        
                        test_proj = predict_deep(Xs_test, res)
                        U_train = res.get('u', res.get('U')).numpy()
                        U_test = test_proj.get('u', test_proj.get('U')).numpy()
                        
                        reg = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(U_train, y_train)
                        y_pred = reg.predict(U_test)
                        corr, _ = pearsonr(y_test, y_pred)
                        elapsed = time.time() - start_t
                        
                        print(f"DONE. Corr: {corr:.4f}")
                        results.append({
                            'Model': m_name, 'Arch': str(arch), 'MAI': mai, 'Energy': energy, 'Corr': corr
                        })
                    except Exception as e:
                        print(f"FAILED: {e}")

    print("\n" + "="*80)
    print("FINAL REFINED LEADERBOARD")
    print("="*80)
    res_df = pd.DataFrame(results).sort_values(by='Corr', ascending=False)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    run_final_sweep(path)
