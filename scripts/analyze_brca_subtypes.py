import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from pysimlr.deep import lend_simr, predict_deep, ned_simr, ned_simr_shared_private
from pysimlr.simlr import simlr
import os
from sklearn.linear_model import LogisticRegression

def load_brca_subtypes():
    df = pd.read_csv('data/BRCA/brca_data_w_subtypes.csv', on_bad_lines='skip')
    
    # Use ER.Status as target
    target_col = 'ER.Status'
    df = df[df[target_col].isin(['Positive', 'Negative'])].copy()
    
    rs_cols = [c for c in df.columns if c.startswith('rs_')]
    cn_cols = [c for c in df.columns if c.startswith('cn_')]
    pp_cols = [c for c in df.columns if c.startswith('pp_')]
    
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    
    Xs = [df[rs_cols].values.astype(float), 
          df[cn_cols].values.astype(float),
          df[pp_cols].values.astype(float)]
    
    return Xs, y, [rs_cols, cn_cols, pp_cols], le.classes_

def run_study():
    print("=== BRCA ER Status Multi-omics Study (5-fold CV) ===")
    Xs, y, feature_names, classes = load_brca_subtypes()
    print(f"Samples: {len(y)}, Class Distribution: {np.bincount(y)}")
    print(f"Views: RNA-Seq ({Xs[0].shape[1]}), CNV ({Xs[1].shape[1]}), Protein ({Xs[2].shape[1]})")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (idx_train, idx_test) in enumerate(kf.split(Xs[0], y)):
        print(f"\nFold {fold+1}/5...")
        
        X_train = [X[idx_train] for X in Xs]
        X_test = [X[idx_test] for X in Xs]
        y_train, y_test = y[idx_train], y[idx_test]
        
        # Scale
        scalers = [StandardScaler() for _ in Xs]
        X_train_s = [s.fit_transform(x) for s, x in zip(scalers, X_train)]
        X_test_s = [s.transform(x) for s, x in zip(scalers, X_test)]
        
        # 1. SiMLR
        res_simlr = simlr([torch.tensor(x).float() for x in X_train_s], k=5, energy_type="acc", mixing_algorithm="newton")
        u_train_simlr = np.concatenate([X_train_s[i] @ res_simlr['v'][i].numpy() for i in range(3)], axis=1)
        u_test_simlr = np.concatenate([X_test_s[i] @ res_simlr['v'][i].numpy() for i in range(3)], axis=1)
        acc_simlr = accuracy_score(y_test, LogisticRegression().fit(u_train_simlr, y_train).predict(u_test_simlr))
        
        # 2. LEND
        res_lend = lend_simr([torch.tensor(x).float() for x in X_train_s], k=5, epochs=100, energy_type="nc", mixing_algorithm="newton", positivity="positive", sparseness_quantile=0.5, nsa_iterations=3, verbose=False)
        v_lend = [v.detach().numpy() for v in res_lend['v']]
        u_train_lend = np.concatenate([X_train_s[i] @ v_lend[i] for i in range(3)], axis=1)
        u_test_lend = np.concatenate([X_test_s[i] @ v_lend[i] for i in range(3)], axis=1)
        acc_lend = accuracy_score(y_test, LogisticRegression().fit(u_train_lend, y_train).predict(u_test_lend))

        # 3. NED
        res_ned = ned_simr([torch.tensor(x).float() for x in X_train_s], k=5, epochs=100, energy_type="logcosh", mixing_algorithm="newton", verbose=False)
        v_ned = [v.detach().numpy() for v in res_ned['v']]
        u_train_ned = np.concatenate([X_train_s[i] @ v_ned[i] for i in range(3)], axis=1)
        u_test_ned = np.concatenate([X_test_s[i] @ v_ned[i] for i in range(3)], axis=1)
        acc_ned = accuracy_score(y_test, LogisticRegression().fit(u_train_ned, y_train).predict(u_test_ned))

        # 4. NEDPP
        res_nedpp = ned_simr_shared_private([torch.tensor(x).float() for x in X_train_s], k=5, epochs=100, energy_type="logcosh", mixing_algorithm="newton", verbose=False)
        v_nedpp = [v.detach().numpy() for v in res_nedpp['v']]
        u_train_nedpp = np.concatenate([X_train_s[i] @ v_nedpp[i] for i in range(3)], axis=1)
        u_test_nedpp = np.concatenate([X_test_s[i] @ v_nedpp[i] for i in range(3)], axis=1)
        acc_nedpp = accuracy_score(y_test, LogisticRegression().fit(u_train_nedpp, y_train).predict(u_test_nedpp))
        
        results.append({'SiMLR': acc_simlr, 'LEND': acc_lend, 'NED': acc_ned, 'NEDPP': acc_nedpp})

    df_res = pd.DataFrame(results)
    print("\n=== Final Results (Accuracy Mean ± SD) ===")
    print(df_res.agg(['mean', 'std']).T)

if __name__ == "__main__":
    run_study()