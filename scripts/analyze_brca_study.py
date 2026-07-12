import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from pysimlr.deep import lend_simr, predict_deep, ned_simr
from pysimlr.simlr import simlr
import os

def load_and_clean_brca():
    # Load data, stopping before the junk at the end
    # Based on the head, it's a CSV.
    df = pd.read_csv('data/BRCA/data.csv', on_bad_lines='skip')
    
    # Identify modalities
    rs_cols = [c for c in df.columns if c.startswith('rs_')]
    cn_cols = [c for c in df.columns if c.startswith('cn_')]
    
    # Target
    target_col = 'vital.status'
    if target_col not in df.columns:
        # Fallback to last column if name varies
        target_col = df.columns[-1]
        
    df = df.dropna(subset=[target_col])
    
    # Encode target (0/1)
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))
    
    X_rs = df[rs_cols].values.astype(float)
    X_cn = df[cn_cols].values.astype(float)
    
    return [X_rs, X_cn], y, rs_cols, cn_cols, le.classes_

def run_brca_study():
    print("=== BRCA Multi-omics Study ===")
    try:
        Xs, y, rs_names, cn_names, classes = load_and_clean_brca()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Samples: {len(y)}, Classes: {classes}")
    print(f"Views: RNA-Seq ({Xs[0].shape[1]}), CNV ({Xs[1].shape[1]})")

    # Split
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.3, random_state=42)
    
    X_train = [X[idx_train] for X in Xs]
    X_test = [X[idx_test] for X in Xs]
    y_train = y[idx_train]
    y_test = y[idx_test]
    
    # Scale
    scalers = [StandardScaler() for _ in Xs]
    X_train_scaled = [s.fit_transform(x) for s, x in zip(scalers, X_train)]
    X_test_scaled = [s.transform(x) for s, x in zip(scalers, X_test)]

    # 1. Baseline: SiMLR (Pure Linear)
    print("\nFitting SiMLR (Linear Baseline)...")
    res_simlr = simlr([torch.tensor(x).float() for x in X_train_scaled], k=5, energy_type="acc", mixing_algorithm="newton")
    # Evaluate via XV
    v_simlr = res_simlr['v']
    u_test_simlr = np.concatenate([X_test_scaled[i] @ v_simlr[i].numpy() for i in range(2)], axis=1)
    u_train_simlr = np.concatenate([X_train_scaled[i] @ v_simlr[i].numpy() for i in range(2)], axis=1)
    
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000).fit(u_train_simlr, y_train)
    acc_simlr = accuracy_score(y_test, clf.predict(u_test_simlr))
    print(f"SiMLR XV Accuracy: {acc_simlr:.4f}")

    # 2. LEND (Deep-Aligned Linear)
    print("\nFitting LEND (Gold Standard)...")
    res_lend = lend_simr([torch.tensor(x).float() for x in X_train_scaled], k=5, epochs=100, 
                         energy_type="nc", mixing_algorithm="newton", positivity="positive", 
                         sparseness_quantile=0.5, nsa_iterations=3, verbose=False)
    
    # Evaluate XV
    v_lend = [v.detach().numpy() for v in res_lend['v']]
    u_test_lend = np.concatenate([X_test_scaled[i] @ v_lend[i] for i in range(2)], axis=1)
    u_train_lend = np.concatenate([X_train_scaled[i] @ v_lend[i] for i in range(2)], axis=1)
    
    clf_lend = LogisticRegression(max_iter=1000).fit(u_train_lend, y_train)
    acc_lend = accuracy_score(y_test, clf_lend.predict(u_test_lend))
    print(f"LEND XV Accuracy: {acc_lend:.4f}")

    # 3. NED (Fully Deep Consensus)
    print("\nFitting NED...")
    res_ned = ned_simr([torch.tensor(x).float() for x in X_train_scaled], k=5, epochs=100,
                       energy_type="logcosh", mixing_algorithm="newton", verbose=False)
    
    # Evaluate XV
    v_ned = [v.detach().numpy() for v in res_ned['v']]
    u_test_ned = np.concatenate([X_test_scaled[i] @ v_ned[i] for i in range(2)], axis=1)
    u_train_ned = np.concatenate([X_train_scaled[i] @ v_ned[i] for i in range(2)], axis=1)
    
    clf_ned = LogisticRegression(max_iter=1000).fit(u_train_ned, y_train)
    acc_ned = accuracy_score(y_test, clf_ned.predict(u_test_ned))
    print(f"NED XV Accuracy: {acc_ned:.4f}")

    print("\n=== Top Biomarkers (LEND) ===")
    # Extract top features from V
    for i, (name, feats) in enumerate([("RNA", rs_names), ("CNV", cn_names)]):
        weights = np.abs(v_lend[i]).sum(axis=1)
        top_idx = np.argsort(weights)[-5:][::-1]
        print(f"Top {name} Features:")
        for idx in top_idx:
            print(f"  - {feats[idx]} (Weight: {weights[idx]:.4f})")

if __name__ == "__main__":
    run_brca_study()
