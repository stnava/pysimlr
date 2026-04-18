import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def calculate_partial_eta_squared(anova_table):
    anova_table['eta_sq_p'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])
    return anova_table

def perform_analysis(file_path):
    df = pd.read_csv(file_path)
    n = len(df)
    print(f"Sample size: n = {n}")
    
    # Task 1: Multi-factor ANOVA
    print("\n### Task 1: Multi-factor ANOVA ###")
    
    for target in ['Predictive Accuracy (Y)', 'Strictly Linear Accuracy']:
        print(f"\nANOVA for {target}:")
        # Formula with main effects and interaction between Model and Consensus
        formula = f'Q("{target}") ~ C(Model) + C(Loss) + C(Consensus) + C(Model):C(Consensus)'
        model = ols(formula, data=df).fit()
        anova_results = anova_lm(model, typ=2)
        anova_results = calculate_partial_eta_squared(anova_results)
        print(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'eta_sq_p']].to_latex())
        
    # Task 2: Generalization Gap
    print("\n### Task 2: Generalization Gap ###")
    gen_gap_col = 'Gen Gap (Y)'
    model_stats = df.groupby('Model')[gen_gap_col].agg(['mean', 'std', 'count'])
    model_stats['ci_95'] = 1.96 * model_stats['std'] / np.sqrt(model_stats['count'])
    print("\nGeneralization Gap by Model (Mean +/- 95% CI):")
    print(model_stats)
    
    # Statistical test: LEND vs Others (Pooled NED/NEDPP/SiMLR)
    lend_gap = df[df['Model'] == 'LEND'][gen_gap_col]
    others_gap = df[df['Model'] != 'LEND'][gen_gap_col]
    t_stat, p_val = stats.ttest_ind(lend_gap, others_gap, equal_var=False)
    print(f"\nT-test LEND vs Others Gap: t={t_stat:.4f}, p={p_val:.4f}")

    # Task 3: Transparency Inversion
    print("\n### Task 3: Transparency Inversion ###")
    df['Inversion_Diff'] = df['Strictly Linear Accuracy'] - df['Predictive Accuracy (Y)']
    
    inversion_stats = df.groupby('Model')['Inversion_Diff'].agg(['mean', 'std', 'count'])
    inversion_stats['ci_95'] = 1.96 * inversion_stats['std'] / np.sqrt(inversion_stats['count'])
    print("\nTransparency Inversion (Linear - Pred Y) by Model:")
    print(inversion_stats)
    
    t_stat, p_val = stats.ttest_1samp(df['Inversion_Diff'], 0)
    print(f"\nOverall Transparency Inversion One-sample t-test: t={t_stat:.4f}, p={p_val:.4g}")
    
    # Task 4: Recommendation
    print("\n### Task 4: Clinical Recommendation Logic ###")
    # Best configuration based on Predictive Accuracy and Generalization Gap
    best_configs = df.groupby(['Model', 'Loss', 'Consensus'])['Predictive Accuracy (Y)'].mean().sort_values(ascending=False).head(10)
    print("\nTop 10 Configurations (Mean Predictive Accuracy):")
    print(best_configs)

if __name__ == "__main__":
    perform_analysis('paper/results_cache/unified_real_v17.csv')
