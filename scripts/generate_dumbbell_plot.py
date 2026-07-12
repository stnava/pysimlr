import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = {'Linear': '#1f77b4', 'Deep-Aligned': '#2ca02c'}

# Load data
df_syn = pd.read_csv('paper/results_cache/unified_synthetic_v21.csv')
df_real = pd.read_csv('paper/results_cache/unified_real_v21.csv')

# Preprocess Synthetic
df_syn['Model_Type'] = df_syn['Model'].apply(lambda x: 'Linear' if x == 'SiMLR' else 'Deep-Aligned' if x == 'LEND' else None)
df_syn = df_syn.dropna(subset=['Model_Type'])
# Use iterative consensus only for stability
df_syn = df_syn[df_syn['Consensus'].isin(['newton', 'ica'])]

synth_agg = df_syn.groupby(['Regime', 'Model_Type'])['Strictly Linear Accuracy'].mean().unstack()

# Preprocess Real
df_real['Model_Type'] = df_real['Model'].apply(lambda x: 'Linear' if x == 'SiMLR' else 'Deep-Aligned' if x == 'LEND' else None)
df_real = df_real.dropna(subset=['Model_Type'])
df_real = df_real[df_real['Consensus'].isin(['newton', 'ica'])]

real_agg = df_real.groupby(['Dataset', 'Model_Type'])['Strictly Linear Accuracy'].mean().unstack()
real_agg.index.name = 'Regime'

# Combine
agg = pd.concat([synth_agg, real_agg])
agg = agg.reset_index()

# Plotting
plt.figure(figsize=(10, 7))

# Draw the line
plt.hlines(y=agg['Regime'], xmin=agg['Linear'], xmax=agg['Deep-Aligned'], color='gray', alpha=0.5, linewidth=3)

# Draw the dots
plt.scatter(agg['Linear'], agg['Regime'], color=palette['Linear'], label='SiMLR (Baseline)', s=150, zorder=3)
plt.scatter(agg['Deep-Aligned'], agg['Regime'], color=palette['Deep-Aligned'], label='LEND (Deep-Aligned)', s=150, zorder=3)

# Annotations (Deltas)
for i, row in agg.iterrows():
    delta = row['Deep-Aligned'] - row['Linear']
    plt.text((row['Linear'] + row['Deep-Aligned'])/2, i + 0.1, f"Δ={delta:+.3f}", 
             ha='center', fontsize=12, fontweight='bold', color='darkgreen')

plt.title("The Deep-to-Linear Benefit: Regularization Gain in Feature Discovery", fontsize=16, pad=20)
plt.xlabel("Strictly Linear Accuracy (XV)", fontsize=14)
plt.ylabel("Data Regime / Clinical Dataset", fontsize=14)
plt.legend(loc='lower right', frameon=True)
plt.grid(axis='x', linestyle='--', alpha=0.7)

os.makedirs('paper/figures', exist_ok=True)
plt.tight_layout()
plt.savefig('paper/figures/fig_v21_dumbbell_benefit.png', dpi=300)
print("Saved dumbbell plot to paper/figures/fig_v21_dumbbell_benefit.png")
