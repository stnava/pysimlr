import os
import multiprocessing
from sklearn.model_selection import KFold

# SET ENVIRONMENT VARIABLES BEFORE ANY IMPORTS TO PREVENT OMP HANGS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import traceback
from scipy import stats

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))
from pysimlr.simlr import simlr, predict_simlr
from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private, predict_deep
from pysimlr.benchmarks.metrics import calculate_all_metrics

def filter_kwargs(func, kwargs):
    import inspect
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def get_diabetes_case():
    data = load_diabetes(); X_scaled = StandardScaler().fit_transform(data.data); y = data.target
    # SPLIT INTO 3 VIEWS to see LOO benefit
    mats = [torch.tensor(X_scaled[:, :3]).float(), 
            torch.tensor(X_scaled[:, 3:6]).float(),
            torch.tensor(X_scaled[:, 6:]).float()]; k = 2
    return {"data": mats, "outcome": torch.tensor(y).float(), "shared_k": k}

def get_heart_case():
    url_heart = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    try: df_h = pd.read_csv(url_heart, names=cols).replace('?', np.nan).dropna().apply(pd.to_numeric)
    except:
        X_dummy = np.random.randn(300, 13); y_dummy = np.random.randint(0, 2, 300); df_h = pd.DataFrame(X_dummy); df_h['num'] = y_dummy
    X_h = StandardScaler().fit_transform(df_h.drop('num', axis=1)); y = df_h['num'].values.astype(int); 
    # SPLIT INTO 3 VIEWS
    mats = [torch.tensor(X_h[:, :4]).float(), 
            torch.tensor(X_h[:, 4:8]).float(),
            torch.tensor(X_h[:, 8:]).float()]; k = 2
    return {"data": mats, "outcome": torch.tensor(y).float(), "shared_k": k}

def run_cv_experiment(model_type, dataset_name, topology, loss, consensus, seed=42):
    torch.set_num_threads(1)
    case = get_diabetes_case() if dataset_name == "Diabetes" else get_heart_case()
    data_all = case["data"]
    y_all = case["outcome"]
    k = case["shared_k"]
    n_samples = data_all[0].shape[0]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_metrics = []
    
    params = {
        "iterations": 100, "epochs": 100, "energy_type": loss, "mixing_algorithm": consensus, 
        "use_nsa": True, "positivity": "positive", "nsa_w": 0.5, "sparseness_quantile": 0.5,
        "nsa_iterations": 3, "topology": topology, "verbose": False
    }

    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        train_mats = [m[train_idx] for m in data_all]
        test_mats = [m[test_idx] for m in data_all]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        torch.manual_seed(seed) 
        
        if model_type == "linear":
            f_params = filter_kwargs(simlr, params)
            res = simlr(train_mats, k=k, **f_params)
        elif model_type == "lend":
            f_params = filter_kwargs(lend_simr, params)
            res = lend_simr(train_mats, k=k, **f_params)
        elif model_type == "ned":
            f_params = filter_kwargs(ned_simr, params)
            res = ned_simr(train_mats, k=k, **f_params)
        elif model_type == "shared_private":
            f_params = filter_kwargs(ned_simr_shared_private, params)
            res = ned_simr_shared_private(train_mats, k=k, **f_params)

        if "model" in res:
            pred_test = predict_deep(test_mats, res, device="cpu")
            pred_train = predict_deep(train_mats, res, device="cpu")
        else:
            pred_test = predict_simlr(test_mats, res)
            pred_train = predict_simlr(train_mats, res)

        fl_scores_train = pred_train.get("first_layer_scores")
        fl_scores_test = pred_test.get("first_layer_scores")
        if fl_scores_train is None:
            fl_scores_train, fl_scores_test = pred_train.get("latents"), pred_test.get("latents")

        m = calculate_all_metrics(
            pred_test['u'], torch.zeros_like(pred_test['u']), y_test, test_mats, pred_test['reconstructions'],
            u_train=pred_train['u'], y_train=y_train,
            first_layer_scores_train=fl_scores_train, first_layer_scores_test=fl_scores_test,
        )
        is_h = (dataset_name == "Heart")
        fold_metrics.append(m.get("first_layer_test_accuracy") if is_h else m.get("first_layer_test_r2"))
    
    return {
        "Dataset": dataset_name, "Model": model_type, "Topology": topology, 
        "Loss": loss, "Consensus": consensus, "Seed": seed,
        "Mean Strictly Linear Accuracy": np.mean(fold_metrics)
    }

def run_full_comparison():
    datasets = ["Heart", "Diabetes"]
    model_configs = [("linear", "SiMLR"), ("lend", "LEND"), ("ned", "NED"), ("shared_private", "NEDPP")]
    losses = ["regression", "acc", "logcosh", "nc"]
    mixing_methods = ["newton", "svd", "pca", "ica"]
    topologies = ["star", "loo"]
    seeds = [42, 43, 44] # Multiple seeds for better paired t-test
    
    tasks = []
    for ds in datasets:
        for m_type, m_label in model_configs:
            for loss in losses:
                for mix in mixing_methods:
                    for topo in topologies:
                        for seed in seeds:
                            tasks.append((m_type, ds, topo, loss, mix, seed))
    
    print(f"Starting 3-VIEW 5-Fold CV Comparison with {len(tasks)} tasks...")
    sys.stdout.flush()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_cv_experiment, *t): t for t in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
                if len(results) % 10 == 0:
                    print(f"  Progress: {len(results)}/{len(tasks)}")
                    sys.stdout.flush()
            except Exception:
                traceback.print_exc()
                sys.stdout.flush()

    df = pd.DataFrame(results)
    df.to_csv("paper/results_cache/topology_cv_results_3view.csv", index=False)
    
    # Statistical Analysis
    comparison = []
    model_map = {'linear': 'SiMLR', 'lend': 'LEND', 'ned': 'NED', 'shared_private': 'NEDPP'}
    df['Model'] = df['Model'].map(model_map).fillna(df['Model'])
    
    for (ds, model), group in df.groupby(['Dataset', 'Model']):
        star = group[group['Topology'] == 'star'].sort_values(['Loss', 'Consensus', 'Seed'])
        loo = group[group['Topology'] == 'loo'].sort_values(['Loss', 'Consensus', 'Seed'])
        
        x = loo['Mean Strictly Linear Accuracy'].values
        y = star['Mean Strictly Linear Accuracy'].values
        
        if len(x) > 1 and len(y) > 1:
            t, p = stats.ttest_rel(x, y)
            d = (np.mean(x) - np.mean(y)) / np.std(x - y, ddof=1) if np.std(x - y, ddof=1) != 0 else 0
            comparison.append({'Dataset': ds, 'Model': model, 'Star CV Mean': np.mean(y), 'LOO CV Mean': np.mean(x), 'p-value': p, 'Cohen s d': d})
            
    res_df = pd.DataFrame(comparison).round(4)
    print("\n--- 5-Fold CV Paired T-Test: LOO vs STAR (3 Views) ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    run_full_comparison()
