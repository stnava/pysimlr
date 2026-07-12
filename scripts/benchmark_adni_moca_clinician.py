import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from pysimlr.deep import lend_simr, predict_deep

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

def run_adni_moca_interpretation(file_path):
    print("=== ADNI Clinician-Facing Study: Imaging -> MOCA_bl ===")
    
    df = pd.read_csv(file_path, low_memory=False)
    df = df[df['studyName'] == 'ADNI'].copy()
    
    # Target
    target_col = 'MOCA_bl'
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].values
    
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
    
    print(f"Sample Size: {len(y)}")
    for k, v in views.items():
        print(f"  {k:10s}: {v.shape[1]} features")

    # 1. Prediction Benchmark (75/25)
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.25, random_state=42)
    Xs_train = [x[idx_train] for x in Xs]
    Xs_test = [x[idx_test] for x in Xs]
    y_train, y_test = y[idx_train], y[idx_test]

    print("\nBenchmarking predictive power...")
    res_bench = lend_simr(
        Xs_train, k=12, epochs=150, 
        hidden_dims=[64, 32],
        energy_type='regression', mixing_algorithm='newton',
        verbose=False
    )
    test_proj = predict_deep(Xs_test, res_bench)
    U_train = res_bench.get('u', res_bench.get('U')).numpy()
    U_test = test_proj.get('u', test_proj.get('U')).numpy()
    
    reg = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(U_train, y_train)
    y_pred = reg.predict(U_test)
    r2 = r2_score(y_test, y_pred)
    corr, _ = pearsonr(y_test, y_pred)
    print(f"  Test R^2: {r2:.4f}, Correlation: {corr:.4f}")

    # 2. Final Fit on All Data for Interpretation
    print("\nRe-fitting model on all data for clinician interpretation...")
    res_final = lend_simr(
        Xs, k=12, epochs=200, 
        hidden_dims=[64, 32],
        energy_type='regression', mixing_algorithm='newton',
        verbose=False
    )
    U = res_final.get('u', res_final.get('U')).numpy()
    V_mats = res_final.get('v') # List of basis matrices (input_dim x k)

    # 3. Identify the "Clinical Latent"
    corrs = [pearsonr(U[:, i], y)[0] for i in range(U.shape[1])]
    best_latent_idx = np.argmax(np.abs(corrs))
    max_corr = corrs[best_latent_idx]
    print(f"  Most Clinically Relevant Latent: Dimension {best_latent_idx} (rho = {max_corr:.4f})")

    # 4. Generate Visualizations
    os.makedirs('clinical_results', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Plot 1: Latent vs Clinical Outcome
    plt.figure(figsize=(8, 6))
    sns.regplot(x=U[:, best_latent_idx], y=y, scatter_kws={'alpha':0.5, 'color':'#2c3e50'}, line_kws={'color':'#e74c3c'})
    plt.title(f"Clinician Summary: Shared Imaging Signal vs. Cognitive Score (MOCA)\nCorrelation: {max_corr:.3f}", fontsize=14)
    plt.xlabel(f"Shared Neuroimaging Latent (Dim {best_latent_idx})", fontsize=12)
    plt.ylabel("Clinical MOCA Score", fontsize=12)
    plt.savefig('clinical_results/moca_vs_latent.png', dpi=300)
    print("  Saved: clinical_results/moca_vs_latent.png")

    # Plot 2-4: Top contributing regions for each modality
    for i, (name, v_mat) in enumerate(zip(view_names, V_mats)):
        v_weights = v_mat[:, best_latent_idx].numpy()
        feat_names = views_cols[name]
        
        # Get top 15 by absolute weight
        top_idx = np.argsort(np.abs(v_weights))[-15:]
        top_weights = v_weights[top_idx]
        top_labels = [feat_names[j] for j in top_idx]
        
        # Clean up labels for clinician readability (remove prefixes)
        top_labels = [l.replace(name+'Hier_', '').replace(name+'_mean_fa.', '').replace('rsfMRI_fcnxpro129_', '') for l in top_labels]

        plt.figure(figsize=(10, 8))
        colors = ['#e74c3c' if w < 0 else '#2ecc71' for w in top_weights]
        sns.barplot(x=top_weights, y=top_labels, palette=colors)
        plt.title(f"Top {name} Structural Features Driving Cognitive Signal", fontsize=14)
        plt.xlabel("Importance Weight", fontsize=12)
        plt.tight_layout()
        plt.savefig(f'clinical_results/top_features_{name}.png', dpi=300)
        print(f"  Saved: clinical_results/top_features_{name}.png")

    print("\nInterpretation Complete. See 'clinical_results/' folder for output.")

if __name__ == "__main__":
    path = "../multidisorder/data/ppmiadni_filtered.csv"
    run_adni_moca_interpretation(path)
