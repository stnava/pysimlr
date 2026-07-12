
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = {
    'SiMLR': '#1f77b4',
    'LEND': '#2ca02c',
    'NED': '#ff7f0e',
    'NEDPP': '#9467bd',
    'Flow-SiMLR-V': '#d62728'
}

# Ensure figures directory exists
os.makedirs('paper/figures', exist_ok=True)

# Load data
df_syn = pd.read_csv('paper/results_cache/unified_synthetic_v21.csv')
df_real = pd.read_csv('paper/results_cache/unified_real_v21.csv')

model_map = {'linear': 'SiMLR', 'lend': 'LEND', 'ned': 'NED', 'shared_private': 'NEDPP'}
df_syn['Model'] = df_syn['Model'].map(model_map).fillna(df_syn['Model'])
df_syn['Regime'] = df_syn['Regime'].str.replace('Private', 'Private Noise')

df_real['Model'] = df_real['Model'].map(model_map).fillna(df_real['Model'])


# --- Figure 1: Global Performance Landscape (Synthetic) ---
# Faceted bar plot of Predictive Accuracy (Y) across regimes
plt.figure(figsize=(14, 6))
g = sns.catplot(
    data=df_syn, x='Model', y='Strictly Linear Accuracy', col='Regime',
    kind='bar', palette=palette, height=5, aspect=0.8,
    errorbar='sd', capsize=.1
)
g.set_axis_labels("Model", "Strictly Linear Accuracy (XV R2)")
g.set_titles("{col_name} Regime")
plt.tight_layout()
plt.savefig('paper/figures/fig_v21_global_landscape.png', dpi=300)
plt.close()

# Educational Comment: 
# The faceted bar plot allows for direct comparison of model performance across different data topologies. 
# By visualizing the mean and standard deviation, we highlight not just peak performance but also 
# architectural stability across diverse generative regimes.

# --- Figure 2: Consensus Stability Plot (U vs XV) (Synthetic) ---
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_syn, x='Latent Recovery (U)', y='Strictly Linear Accuracy',
    hue='Model', palette=palette, alpha=0.6, s=60
)
# Add a parity line if it makes sense, or just a trend line
sns.regplot(
    data=df_syn, x='Latent Recovery (U)', y='Strictly Linear Accuracy',
    scatter=False, color='gray', line_kws={'linestyle': '--'}
)
plt.title("Consensus Stability: Latent Recovery vs. Linear Accuracy (Synthetic)")
plt.xlabel("Latent Recovery (CMC/U)")
plt.ylabel("Strictly Linear Accuracy (XV)")
plt.tight_layout()
plt.savefig('paper/figures/fig_v21_consensus_stability.png', dpi=300)
plt.close()

# Educational Comment:
# This scatter plot visualizes the alignment between deep consensus (U) and the first-layer linear basis (XV).
# High correlation indicates that the model's non-linear heads are effectively building upon a 
# robust linear foundation, validating the First-Layer Contract.

# --- Figure 3: Linear-to-Deep Performance Parity (Real-world) ---
# Filter out negative or zero performance to focus on informative regions
df_real_pos = df_real[(df_real['Predictive Accuracy (Y)'] > 0) & (df_real['Strictly Linear Accuracy'] > 0)].copy()

plt.figure(figsize=(8, 8))
sns.scatterplot(
    data=df_real_pos, x='Predictive Accuracy (Y)', y='Strictly Linear Accuracy',
    hue='Model', style='Dataset', palette=palette, alpha=0.85, s=150, edgecolor='w', linewidth=1.5
)

# Calculate dynamic limits for a clean square plot
min_val = min(df_real_pos['Predictive Accuracy (Y)'].min(), df_real_pos['Strictly Linear Accuracy'].min()) * 0.95
max_val = max(df_real_pos['Predictive Accuracy (Y)'].max(), df_real_pos['Strictly Linear Accuracy'].max()) * 1.05

# Add Parity Line (y=x)
plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', alpha=0.7, zorder=0, label='Parity (y=x)')

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.title("Linear-to-Deep Performance Parity (Real-World)", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Deep Consensus Accuracy ($R^2_U$ or $Acc_U$)", fontsize=13, fontweight='bold')
plt.ylabel("Strictly Linear Accuracy ($R^2_{XV}$ or $Acc_{XV}$)", fontsize=13, fontweight='bold')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True, borderpad=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('paper/figures/fig_v21_blessing_interpretability.png', dpi=300, bbox_inches='tight')
plt.close()

# Educational Comment:
# The 'Blessing of Interpretability' is evidenced when points cluster near the parity line. 
# This indicates that the strictly linear features (XV) capture nearly all the predictive power 
# of the deep model, allowing for transparent clinical discovery without sacrificing performance.

# --- Figure 4: Consensus x Loss Interaction Heatmap (LEND) ---
# We'll use LEND as the representative deep architecture for this heatmap
df_lend = df_syn[df_syn['Model'] == 'LEND']
pivot_table = df_lend.groupby(['Loss', 'Consensus'])['Predictive Accuracy (Y)'].mean().unstack()

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': 'Mean Pred Accuracy (Y)'})
plt.title("Interaction: Loss Function x Consensus Method (LEND)")
plt.tight_layout()
plt.savefig('paper/figures/fig_v21_interaction_heatmap.png', dpi=300)
plt.close()

# Educational Comment:
# Heatmaps are ideal for visualizing combinatorial hyperparameter spaces. 
# This plot identifies the 'Gold Standard' configurations by revealing which 
# pairings of Loss functions and Consensus methods yield the most robust latent discovery.
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = {
    'SiMLR': '#1f77b4',
    'LEND': '#2ca02c',
    'NED': '#ff7f0e',
    'NEDPP': '#9467bd',
    'Flow-SiMLR-V': '#d62728'
}

# --- Figure 5: LOO vs STAR Generalization Comparison ---
df_loo = pd.read_csv('paper/results_cache/unified_real_loo.csv')
df_star = pd.read_csv('paper/results_cache/unified_real_v21.csv')

model_map = {'linear': 'SiMLR', 'lend': 'LEND', 'ned': 'NED', 'shared_private': 'NEDPP'}
df_loo['Model'] = df_loo['Model'].map(model_map).fillna(df_loo['Model'])
df_star['Model'] = df_star['Model'].map(model_map).fillna(df_star['Model'])

df_loo['Evaluation'] = 'LOO (Leave-One-Out)'
df_star['Evaluation'] = 'STAR (K-Fold)'

df_combined = pd.concat([df_loo, df_star])

# We'll plot Strictly Linear Accuracy comparison
plt.figure(figsize=(12, 6))
g = sns.barplot(
    data=df_combined, x='Model', y='Strictly Linear Accuracy', hue='Evaluation', 
    palette={'LOO (Leave-One-Out)': '#8c564b', 'STAR (K-Fold)': '#e377c2'},
    errorbar='sd', capsize=0.1
)
plt.title("Generalization Robustness: LOO vs. STAR Architecture (Real-world)")
plt.ylabel("Strictly Linear Accuracy (XV)")
plt.xlabel("Model Architecture")
plt.legend(title="Validation Strategy")
plt.tight_layout()
plt.savefig('paper/figures/fig_v21_loo_vs_star.png', dpi=300)
plt.close()

# Let's also create a Gen Gap comparison which is very relevant for LOO vs K-Fold
plt.figure(figsize=(12, 6))
g = sns.boxplot(
    data=df_combined, x='Model', y='Strictly Linear Gap', hue='Evaluation', 
    palette={'LOO (Leave-One-Out)': '#8c564b', 'STAR (K-Fold)': '#e377c2'}
)
plt.title("Generalization Gap: LOO vs. STAR Architecture (Real-world)")
plt.ylabel("Strictly Linear Gen Gap (Train - Test)")
plt.xlabel("Model Architecture")
plt.legend(title="Validation Strategy")
plt.tight_layout()
plt.savefig('paper/figures/fig_v21_loo_vs_star_gap.png', dpi=300)
plt.close()
