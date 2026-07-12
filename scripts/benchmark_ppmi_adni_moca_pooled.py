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
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private, predict_deep

def preprocess_view(df_sub, return_cols=False):
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
    if not valid_cols: return (None, None) if return_cols else None
    
    df_valid = df_num[valid_cols].astype(float)
    X_imp = SimpleImputer(strategy='mean').fit_transform(df_valid.values)
    
    stds = np.std(X_imp, axis=0)
    valid_indices = stds > 1e-6
    if not np.any(valid_indices): return (None, None) if return_cols else None
    
    final_cols = [valid_cols[i] for i in range(len(valid_cols)) if valid_indices[i]]
    X = X_imp[:, valid_indices]
    X_sc = StandardScaler().fit_transform(X)
    
    if return_cols:
        return X_sc, final_cols
    return X_sc

def run_pooled_moca_sweep(file_path):
    print("=== PPMI+ADNI Pooled MOCA Study: Imaging -> Cognition ===")
    
    df = pd.read_csv(file_path, low_memory=False)
    # Combine MOCA sources
    df['target'] = df['MOCA'].combine_first(df['moca'])
    df = df.dropna(subset=['target']).copy()
    y = df['target'].values
    
    print(f"Total Pooled Sample Size: {len(df)}")
    print(df.groupby('studyName')['target'].count())
    
    # Imaging Modalities
    t1_cols_raw = [c for c in df.columns if c.startswith('T1Hier')]
    dti_cols_raw = [c for c in df.columns if c.startswith('DTI_mean_fa')]
    rsf_cols_raw = [c for c in df.columns if c.startswith('rsfMRI_fcnxpro129_') and '_2_' in c]

    X_t1, cols_t1 = preprocess_view(df[t1_cols_raw], return_cols=True)
    X_dti, cols_dti = preprocess_view(df[dti_cols_raw], return_cols=True)
    X_rsf, cols_rsf = preprocess_view(df[rsf_cols_raw], return_cols=True)

    views = {'T1': X_t1, 'DTI': X_dti, 'rsfMRI': X_rsf}
    views_cols = {'T1': cols_t1, 'DTI': cols_dti, 'rsfMRI': cols_rsf}
    views = {k: v for k, v in views.items() if v is not None}
    Xs = list(views.values())
    view_names = list(views.keys())
    
    # 75/25 Split
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.25, random_state=42)
    Xs_train = [x[idx_train] for x in Xs]
    Xs_test = [x[idx_test] for x in Xs]
    y_train, y_test = y[idx_train], y[idx_test]

    model_fns = {
        'LEND': lend_simr,
        'NED': ned_simr,
        'NEDPP': lambda *args, **kwargs: ned_simr_shared_private(*args, private_k=2, **kwargs)
    }
    mai_options = [False, True]
    
    sweep_results = []

    for m_name, m_fn in model_fns.items():
        for mai in mai_options:
            print(f"\nTesting: {m_name:5s} | MAI: {str(mai):5s} ...", end=" ", flush=True)
            try:
                start_t = time.time()
                res = m_fn(
                    Xs_train, k=12, epochs=100, 
                    hidden_dims=[64, 32],
                    dynamic_weights=mai,
                    energy_type='regression', mixing_algorithm='newton',
                    verbose=False
                )
                
                test_proj = predict_deep(Xs_test, res)
                U_train = res.get('u', res.get('U')).numpy()
                U_test = test_proj.get('u', test_proj.get('U')).numpy()
                
                reg = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(U_train, y_train)
                y_pred = reg.predict(U_test)
                
                r2 = r2_score(y_test, y_pred)
                corr, _ = pearsonr(y_test, y_pred)
                elapsed = time.time() - start_t
                
                print(f"DONE. R2: {r2:.4f}, Corr: {corr:.4f}")
                sweep_results.append({
                    'Model': m_name, 'MAI': mai, 'R2': r2, 'Corr': corr, 'Time': elapsed, 'res': res
                })
            except Exception as e:
                print(f"FAILED: {e}")

    # Identify Best Model
    best_res_meta = max(sweep_results, key=lambda x: x['Corr'])
    print(f"\nWinner: {best_res_meta['Model']} (MAI={best_res_meta['MAI']}) with Corr={best_res_meta['Corr']:.4f}")

    # Re-fit Winner on All Data for Final Interpretation
    print("\nRe-fitting winning model on all data for clinician interpretation...")
    best_m_fn = model_fns[best_res_meta['Model']]
    if best_res_meta['Model'] == 'NEDPP':
         # Handle lambda closure
         res_final = ned_simr_shared_private(Xs, k=12, private_k=2, epochs=150, hidden_dims=[64, 32], dynamic_weights=best_res_meta['MAI'], energy_type='regression', mixing_algorithm='newton', verbose=False)
    else:
         res_final = best_m_fn(Xs, k=12, epochs=150, hidden_dims=[64, 32], dynamic_weights=best_res_meta['MAI'], energy_type='regression', mixing_algorithm='newton', verbose=False)

    U = res_final.get('u', res_final.get('U')).numpy()
    V_mats = res_final.get('v')

    # Clinical Latent Identification
    corrs = [pearsonr(U[:, i], y)[0] for i in range(U.shape[1])]
    best_latent_idx = np.argmax(np.abs(corrs))
    max_corr = corrs[best_latent_idx]
    
    # Generate Plots
    output_dir = 'clinical_results_pooled'
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.regplot(x=U[:, best_latent_idx], y=y, scatter_kws={'alpha':0.3, 'color':'#2c3e50'}, line_kws={'color':'#e74c3c'})
    plt.title(f"Pooled PPMI+ADNI: Shared Imaging Signal vs. MOCA\nCorrelation: {max_corr:.3f}", fontsize=14)
    plt.xlabel(f"Shared Neuroimaging Latent (Dim {best_latent_idx})", fontsize=12)
    plt.ylabel("MOCA Score", fontsize=12)
    plt.savefig(f'{output_dir}/moca_vs_latent_pooled.png', dpi=300)

    for i, (name, v_mat) in enumerate(zip(view_names, V_mats)):
        v_weights = v_mat[:, best_latent_idx].numpy()
        feat_names = views_cols[name]
        top_idx = np.argsort(np.abs(v_weights))[-15:]
        top_weights = v_weights[top_idx]
        top_labels = [feat_names[j].replace(name+'Hier_', '').replace(name+'_mean_fa.', '').replace('rsfMRI_fcnxpro129_', '') for j in top_idx]

        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_weights, y=top_labels, hue=top_labels, palette='vlag', legend=False)
        plt.title(f"Top {name} Features Driving Pooled Cognitive Signal", fontsize=14)
        plt.xlabel("Importance Weight", fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_features_{name}_pooled.png', dpi=300)

    print(f"\nInterpretation Complete. See '{output_dir}/' folder.")
    
    # Leaderboard Summary
    res_df = pd.DataFrame(sweep_results).drop(columns=['res']).sort_values(by='Corr', ascending=False)
    print("\n" + "="*70)
    print("MOCA PREDICTION LEADERBOARD (POOLED Cohort)")
    print("="*70)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    run_pooled_moca_sweep(path)
