import pandas as pd
import numpy as np
from scipy import stats

def analyze_synthetic(file_path):
    print(f"\n{'='*20} SYNTHETIC DATA ANALYSIS (v21) {'='*20}")
    df = pd.read_csv(file_path)
    
    # Nonlinear regime analysis (Sine)
    sine_df = df[df['Regime'] == 'Sine']
    
    metrics = ['Predictive Accuracy (Y)', 'CMC', 'Feature Recovery (V)', 'Latent Recovery (U)']
    
    for metric in metrics:
        print(f"\n--- Metric: {metric} (Regime: Sine) ---")
        summary = sine_df.groupby('Model')[metric].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        # Calculate 95% CI
        summary['ci95_hi'] = summary['mean'] + 1.96 * summary['std'] / np.sqrt(summary['count'])
        summary['ci95_lo'] = summary['mean'] - 1.96 * summary['std'] / np.sqrt(summary['count'])
        print(summary)
        
        # Compare each deep model to SiMLR
        simlr_vals = sine_df[sine_df['Model'] == 'SiMLR'][metric]
        for model in ['LEND', 'NED', 'NEDPP']:
            model_vals = sine_df[sine_df['Model'] == model][metric]
            if not model_vals.empty and not simlr_vals.empty:
                t_stat, p_val = stats.ttest_ind(model_vals, simlr_vals, equal_var=False)
                print(f"  {model} vs SiMLR: p-value = {p_val:.4g} (t = {t_stat:.2f})")

def analyze_real(file_path):
    print(f"\n{'='*20} REAL DATA ANALYSIS (v21) {'='*20}")
    df = pd.read_csv(file_path)
    
    datasets = df['Dataset'].unique()
    metrics = ['Predictive Accuracy (Y)', 'CMC']
    
    for dataset in datasets:
        print(f"\n--- Dataset: {dataset} ---")
        ds_df = df[df['Dataset'] == dataset]
        
        for metric in metrics:
            print(f"\nMetric: {metric}")
            summary = ds_df.groupby('Model')[metric].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
            summary['ci95_hi'] = summary['mean'] + 1.96 * summary['std'] / np.sqrt(summary['count'])
            summary['ci95_lo'] = summary['mean'] - 1.96 * summary['std'] / np.sqrt(summary['count'])
            print(summary)
            
            # Compare deep models to SiMLR
            simlr_vals = ds_df[ds_df['Model'] == 'SiMLR'][metric]
            for model in ['LEND', 'NED', 'NEDPP']:
                model_vals = ds_df[ds_df['Model'] == model][metric]
                if not model_vals.empty and not simlr_vals.empty:
                    t_stat, p_val = stats.ttest_ind(model_vals, simlr_vals, equal_var=False)
                    print(f"  {model} vs SiMLR: p-value = {p_val:.4g} (t = {t_stat:.2f})")

if __name__ == "__main__":
    analyze_synthetic('paper/results_cache/unified_synthetic_v21.csv')
    analyze_real('paper/results_cache/unified_real_v21.csv')
