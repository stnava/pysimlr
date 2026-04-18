import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pysimlr.deep import lend_simr

def generate_heart_waterfall():
    url_heart = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols_heart = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df_heart = pd.read_csv(url_heart, names=cols_heart).replace('?', np.nan).dropna()
    y_heart = (df_heart['num'].values.astype(int) > 0).astype(int)

    v1_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg']
    v2_cols = ['thalach', 'exang', 'oldpeak', 'slope']
    X1_h = StandardScaler().fit_transform(df_heart[v1_cols].values.astype(float))
    X2_h = StandardScaler().fit_transform(df_heart[v2_cols].values.astype(float))

    # Train LEND
    res = lend_simr([torch.from_numpy(X1_h).float(), torch.from_numpy(X2_h).float()], k=2, epochs=200, verbose=False)
    
    u = res['u'].detach().numpy()
    v1 = res['v'][0].detach().numpy()
    
    # Pick a high-risk patient
    patient_idx = np.argmax(u[:, 0])
    x_patient = X1_h[patient_idx]
    
    contributions = x_patient * v1[:, 0]
    
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if c > 0 else '#1f77b4' for c in contributions]
    plt.barh(v1_cols, contributions, color=colors, edgecolor='black', alpha=0.8)
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"Heart Disease Patient Audit: Patient #{patient_idx}", fontweight='bold')
    plt.xlabel("Contribution to Cardiovascular Risk Factor (Shared Latent 1)")
    plt.ylabel("Clinical Feature")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("paper/figures/heart_waterfall.png")
    plt.close()

if __name__ == "__main__":
    generate_heart_waterfall()
