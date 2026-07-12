import pandas as pd
import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from pysimlr.deep import lend_simr, predict_deep

def preprocess_view(df_sub, return_cols=False):
    df_num = df_sub.apply(pd.to_numeric, errors='coerce')
    valid_cols = [c for c in df_num.columns if df_num[c].isna().mean() < 0.90]
    if not valid_cols: return (None, None) if return_cols else None
    
    df_valid = df_num[valid_cols].astype(float)
    X_imp = SimpleImputer(strategy='mean').fit_transform(df_valid.values)
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    X_clipped = np.clip(X_sc, -10, 10)
    X_final = scaler.fit_transform(X_clipped)
    
    if return_cols:
        return X_final, valid_cols
    return X_final

def run_updrs_study(file_path):
    print("=== PPMI UPDRS Total Score Study (Imaging -> Motor/Clinical) ===")
    
    df = pd.read_csv(file_path, low_memory=False)
    # UPDRS is PPMI-only
    df = df[df['studyName'] == 'PPMI'].copy()
    
    target_col = 'updrs_totscore_BL' # Use Baseline UPDRS
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].values
    
    forbidden = ['Rand', 'ackground', 'lassified']
    
    # We use non-adjusted features as they performed better previously, 
    # but we map them from the adjusted list to ensure clinical quality.
    def get_raw_cols(prefix):
        adj_cols = [c for c in df.columns if c.startswith(prefix) and 'adjusted' in c]
        adj_cols = [c for c in adj_cols if not any(f in c for f in forbidden)]
        mapped_cols = []
        for c in adj_cols:
            raw_c = c.replace('_adjusted', '').replace('.adjusted', '')
            if raw_c in df.columns:
                mapped_cols.append(raw_c)
        return mapped_cols

    t1_cols = get_raw_cols('T1Hier')
    dti_cols = get_raw_cols('DTI_mean_fa')
    rsf_cols = [c for c in get_raw_cols('rsfMRI_fcnxpro129_') if '_2_' in c]

    print(f"Features Identified (Raw mapped):")
    print(f"  T1: {len(t1_cols)}, DTI: {len(dti_cols)}, rsfMRI: {len(rsf_cols)}")

    X_t1, cols_t1 = preprocess_view(df[t1_cols], return_cols=True)
    X_dti, cols_dti = preprocess_view(df[dti_cols], return_cols=True)
    X_rsf, cols_rsf = preprocess_view(df[rsf_cols], return_cols=True)

    views = {'T1': X_t1, 'DTI': X_dti, 'rsfMRI': X_rsf}
    views_cols = {'T1': cols_t1, 'DTI': cols_dti, 'rsfMRI': cols_rsf}
    views = {k: v for k, v in views.items() if v is not None}
    Xs = list(views.values())
    view_names = list(views.keys())
    
    print(f"\nStudy Cohort Size: {len(y)}")

    # 1. Prediction Benchmark (75/25)
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.25, random_state=42)
    Xs_train = [x[idx_train] for x in Xs]
    Xs_test = [x[idx_test] for x in Xs]
    y_train, y_test = y[idx_train], y[idx_test]

    print("\nFitting LEND-SiMLR (k=12, MAI=True)...")
    res_final = lend_simr(
        Xs_train, k=12, epochs=150, 
        hidden_dims=[128, 64],
        dynamic_weights=True,
        energy_type='regression', mixing_algorithm='newton',
        verbose=False
    )
    
    test_proj = predict_deep(Xs_test, res_final)
    U_train = res_final.get('u', res_final.get('U')).numpy()
    U_test = test_proj.get('u', test_proj.get('U')).numpy()
    
    # Regression
    reg = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(U_train, y_train)
    y_pred = reg.predict(U_test)
    r2 = r2_score(y_test, y_pred)
    corr, pval = pearsonr(y_test, y_pred)
    
    print(f"\nUPDRS Results (Test Set):")
    print(f"  R^2: {r2:.4f}")
    print(f"  Correlation: {corr:.4f} (p={pval:.4e})")

    # 2. Re-fit and Visualization
    print("\nGenerating final clinician interpretation plots...")
    res_viz = lend_simr(Xs, k=12, epochs=150, hidden_dims=[128, 64], dynamic_weights=True, energy_type='regression', mixing_algorithm='newton', verbose=False)
    U = res_viz.get('u', res_viz.get('U')).numpy()
    V_mats = res_viz.get('v')

    corrs = [pearsonr(U[:, i], y)[0] for i in range(U.shape[1])]
    best_idx = np.argmax(np.abs(corrs))
    
    output_dir = 'clinical_results_updrs'
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.regplot(x=U[:, best_idx], y=y, scatter_kws={'alpha':0.3, 'color':'#2c3e50'}, line_kws={'color':'#e74c3c'})
    plt.title(f"PPMI Analysis: Imaging vs. UPDRS Total\nCorrelation: {corrs[best_idx]:.3f}", fontsize=14)
    plt.xlabel(f"Shared Imaging Latent (Dim {best_idx})", fontsize=12)
    plt.ylabel("UPDRS Total Baseline", fontsize=12)
    plt.savefig(f'{output_dir}/updrs_vs_latent.png', dpi=300)

    for i, (name, v_mat) in enumerate(zip(view_names, V_mats)):
        v_weights = v_mat[:, best_idx].numpy()
        feat_names = views_cols[name]
        top_idx = np.argsort(np.abs(v_weights))[-15:]
        top_labels = [feat_names[j].replace(name+'Hier_', '').replace(name+'_mean_fa.', '').replace('rsfMRI_fcnxpro129_', '') for j in top_idx]

        plt.figure(figsize=(10, 8))
        sns.barplot(x=v_weights[top_idx], y=top_labels, hue=top_labels, palette='mako', legend=False)
        plt.title(f"Top {name} Drivers of UPDRS Signal", fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_features_{name}_updrs.png', dpi=300)

    print(f"\nAnalysis complete. Visuals in '{output_dir}/'.")

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    run_updrs_study(path)
