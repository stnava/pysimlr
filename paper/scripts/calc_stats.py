import pandas as pd
import numpy as np
from scipy import stats
import os

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2: return np.nan
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pool_sd == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / pool_sd

def mean_ci_diff(x, y, confidence=0.95):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan, np.nan
    mean_diff = np.mean(x) - np.mean(y)
    se_diff = np.sqrt(np.var(x, ddof=1)/nx + np.var(y, ddof=1)/ny)
    if se_diff == 0: return mean_diff, mean_diff
    t_val = stats.t.ppf((1 + confidence) / 2., nx + ny - 2)
    margin = t_val * se_diff
    return mean_diff - margin, mean_diff + margin

def analyze_dataset(df, dataset_col, metric, title):
    # Map Model to Group
    df = df.copy()
    model_map = {'linear': 'Linear', 'lend': 'Linear', 'ned': 'Deep', 'shared_private': 'Deep',
                 'SiMLR': 'Linear', 'LEND': 'Linear', 'NED': 'Deep', 'NEDPP': 'Deep'}
    df['Arch_Group'] = df['Model'].map(model_map)
    
    # Map Consensus to Group
    cons_map = {'newton': 'Newton/ICA', 'ica': 'Newton/ICA', 'svd': 'SVD/PCA', 'pca': 'SVD/PCA'}
    df['Cons_Group'] = df['Consensus'].str.lower().map(cons_map)
    
    results = []
    
    # Iterate over Dataset/Regime and Consensus Group
    for ds in df[dataset_col].unique():
        for cg in df['Cons_Group'].dropna().unique():
            sub = df[(df[dataset_col] == ds) & (df['Cons_Group'] == cg)].dropna(subset=[metric])
            
            deep_vals = sub[sub['Arch_Group'] == 'Deep'][metric].values
            lin_vals = sub[sub['Arch_Group'] == 'Linear'][metric].values
            
            if len(deep_vals) > 1 and len(lin_vals) > 1:
                t_stat, p_val = stats.ttest_ind(deep_vals, lin_vals, equal_var=False)
                cd = cohen_d(deep_vals, lin_vals)
                ci_low, ci_high = mean_ci_diff(deep_vals, lin_vals)
                
                # N
                n_deep, n_lin = len(deep_vals), len(lin_vals)
                
                results.append({
                    'Dataset/Regime': ds,
                    'Consensus': cg,
                    'Metric': metric,
                    'N (Deep, Lin)': f"({n_deep}, {n_lin})",
                    'Deep Mean': np.mean(deep_vals),
                    'Lin Mean': np.mean(lin_vals),
                    'Diff (Deep-Lin)': np.mean(deep_vals) - np.mean(lin_vals),
                    '95% CI': f"[{ci_low:.3f}, {ci_high:.3f}]",
                    'Cohen d': cd,
                    'p-value': p_val
                })
    
    res_df = pd.DataFrame(results)
    if res_df.empty: return "No data."
    
    res_df['Deep Mean'] = res_df['Deep Mean'].round(3)
    res_df['Lin Mean'] = res_df['Lin Mean'].round(3)
    res_df['Diff (Deep-Lin)'] = res_df['Diff (Deep-Lin)'].round(3)
    res_df['Cohen d'] = res_df['Cohen d'].round(3)
    
    # Format p-value carefully
    def fmt_p(p):
        if p < 0.0001: return "< 0.0001"
        return f"{p:.4f}"
    res_df['p-value'] = res_df['p-value'].apply(fmt_p)
    
    return res_df.to_markdown(index=False)

synth_df = pd.read_csv('/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/code/pysimlr/paper/results_cache/unified_synthetic_v21.csv')
real_df = pd.read_csv('/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/code/pysimlr/paper/results_cache/unified_real_v21.csv')

print("### Synthetic Results (Predictive Accuracy)\n")
print(analyze_dataset(synth_df, 'Regime', 'Predictive Accuracy (Y)', 'Synthetic Benchmark'))
print("\n### Synthetic Results (CMC)\n")
print(analyze_dataset(synth_df, 'Regime', 'CMC', 'Synthetic Benchmark CMC'))
print("\n### Real Results (Predictive Accuracy)\n")
print(analyze_dataset(real_df, 'Dataset', 'Predictive Accuracy (Y)', 'Real Benchmark'))

