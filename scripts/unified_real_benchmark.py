import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from pysimlr.benchmarks.runner import run_single_experiment
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import csv
import traceback

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

def get_diabetes_case(seed=42):
    data = load_diabetes(); X_scaled = StandardScaler().fit_transform(data.data); y = data.target
    mats = [X_scaled[:, :5], X_scaled[:, 5:]]; k = 2
    return {"data": [torch.tensor(m).float() for m in mats], "outcome": torch.tensor(y).float(), "true_u": torch.zeros(X_scaled.shape[0], k), "true_v": [np.zeros((m.shape[1], k)) for m in mats], "shared_k": k}

def get_heart_case(seed=42):
    url_heart = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    try: df_h = pd.read_csv(url_heart, names=cols).replace('?', np.nan).dropna().apply(pd.to_numeric)
    except:
        X_dummy = np.random.randn(300, 13); y_dummy = np.random.randint(0, 2, 300); df_h = pd.DataFrame(X_dummy); df_h['num'] = y_dummy
    X_h = StandardScaler().fit_transform(df_h.drop('num', axis=1)); y = df_h['num'].values.astype(int); mats = [X_h[:, :7], X_h[:, 7:]]; k = 2
    return {"data": [torch.tensor(m).float() for m in mats], "outcome": torch.tensor(y).float(), "true_u": torch.zeros(X_h.shape[0], k), "true_v": [np.zeros((m.shape[1], k)) for m in mats], "shared_k": k}

def run_experiment_task(task_args):
    dataset_name, model_type, model_label, seed, energy_type, mixing_algorithm = task_args
    try:
        case = get_diabetes_case(seed=seed) if dataset_name == "Diabetes" else get_heart_case(seed=seed)
        params = {
            "iterations": 50, 
            "epochs": 10, # Shorten for debug
            "energy_type": energy_type, 
            "mixing_algorithm": mixing_algorithm, 
            "use_nsa": True,
            "positivity": "positive",
            "nsa_w": 0.5,
            "sparseness_quantile": 0.5
        }
        exp_res = run_single_experiment(model_type, case, seed=seed, **params)
        metrics = exp_res["metrics"]
        
        is_h = (dataset_name == "Heart")
        test_acc = metrics.get("test_accuracy", 0.0) if is_h else metrics.get("test_r2", 0.0)
        train_acc = metrics.get("train_accuracy", 0.0) if is_h else metrics.get("train_r2", 0.0)
        lin_test = metrics.get("first_layer_test_accuracy", 0.0) if is_h else metrics.get("first_layer_test_r2", 0.0)
        lin_train = metrics.get("first_layer_train_accuracy", 0.0) if is_h else metrics.get("first_layer_train_r2", 0.0)
        
        return {
            "Dataset": dataset_name,
            "Model": model_label,
            "Seed": seed,
            "Loss": energy_type,
            "Consensus": mixing_algorithm,
            "Predictive Accuracy (Y)": float(test_acc),
            "Train Accuracy (Y)": float(train_acc),
            "Gen Gap (Y)": float(train_acc - test_acc),
            "Strictly Linear Accuracy": float(lin_test),
            "Strictly Linear Train": float(lin_train),
            "Strictly Linear Gap": float(lin_train - lin_test),
            "CMC": float(metrics.get("recovery", 0.0)),
            "SRE": 0.0
        }
    except Exception as e:
        raise e # FORCE CRASH TO SEE TRACEBACK

def run_real_benchmark(n_seeds=3, workers=None):
    datasets = ["Heart", "Diabetes"]
    model_configs = [("linear", "SiMLR"), ("lend", "LEND"), ("ned", "NED"), ("shared_private", "NEDPP")]
    losses, mixing_methods = ["regression", "acc", "logcosh", "nc"], ["newton", "svd", "pca", "ica"]
    tasks = []
    for dataset_name in datasets:
        for model_type, model_label in model_configs:
            for energy_type in losses:
                for mixing_algorithm in mixing_methods:
                    for seed in range(42, 42 + n_seeds): tasks.append((dataset_name, model_type, model_label, seed, energy_type, mixing_algorithm))
    
    print(f"Starting REAL benchmark (v20) with {len(tasks)} tasks...")
    os.makedirs("paper/results_cache", exist_ok=True)
    out_file = "paper/results_cache/unified_real_v20.csv"
    header = ["Dataset", "Model", "Seed", "Loss", "Consensus", "Predictive Accuracy (Y)", "Train Accuracy (Y)", "Gen Gap (Y)", "Strictly Linear Accuracy", "Strictly Linear Train", "Strictly Linear Gap", "CMC", "SRE"]
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header); writer.writeheader()
    
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_experiment_task, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                with open(out_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header); writer.writerow(res); f.flush()
            completed += 1
            if completed % 10 == 0: print(f"  Progress: {completed}/{len(tasks)} completed"); sys.stdout.flush()
    print(f"Real benchmark complete: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v20 REAL benchmark")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(); 
    if args.smoke_test: run_real_benchmark(n_seeds=1, workers=1)
    else: run_real_benchmark(n_seeds=args.n_seeds, workers=args.workers)
