import os
import multiprocessing

# SET ENVIRONMENT VARIABLES BEFORE ANY IMPORTS TO PREVENT OMP HANGS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
from pysimlr.benchmarks.runner import run_single_experiment
from pysimlr.benchmarks.synthetic_cases import build_case
from pysimlr.utils import procrustes_r2
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import traceback

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

def calculate_v_recovery(res, true_v):
    v_est = res.get('v')
    if v_est is None:
        return 0.0
    r2s = []
    for ve, vt in zip(v_est, true_v):
        ve_t = torch.as_tensor(ve).float()
        vt_t = torch.as_tensor(vt).float()
        r2s.append(procrustes_r2(vt_t, ve_t))
    return np.mean(r2s)

def run_experiment_task(task_args):
    # Ensure each worker runs strictly single-threaded
    torch.set_num_threads(1)
    
    regime_name, case_params, model_type, model_label, seed, energy_type, mixing_algorithm, iterations, epochs, use_nsa = task_args
    case = build_case(seed=seed, **case_params)
    params = {
        "iterations": iterations, 
        "epochs": epochs, 
        "energy_type": energy_type, 
        "mixing_algorithm": mixing_algorithm, 
        "use_nsa": use_nsa, 
        "positivity": "positive", 
        "nsa_w": 0.5, 
        "sparseness_quantile": 0.5
    }
    if model_type == "shared_private" and "private_k" in case: 
        params["private_k"] = case["private_k"]
    
    # Run the experiment. Uncaught exceptions here will be passed to future.result()
    exp_res = run_single_experiment(model_type, case, seed=seed, **params)
    metrics = exp_res["metrics"]
    res = exp_res["result"]
    v_rec = calculate_v_recovery(res, case["true_v"])
    
    return {
        "Regime": regime_name,
        "Model": model_label,
        "Seed": seed,
        "Loss": energy_type,
        "Consensus": mixing_algorithm,
        "CMC": float(metrics.get("recovery", 0.0)),
        "SRE": 1.0 - v_rec,
        "Predictive Accuracy (Y)": float(metrics.get("test_r2", 0.0)),
        "Train Accuracy (Y)": float(metrics.get("train_r2", 0.0)),
        "Gen Gap (Y)": float(metrics.get("gen_gap", 0.0)),
        "Strictly Linear Accuracy": float(metrics.get("first_layer_test_r2", 0.0)),
        "Strictly Linear Train": float(metrics.get("first_layer_train_r2", 0.0)),
        "Strictly Linear Gap": float(metrics.get("first_layer_gen_gap", 0.0)),
        "Latent Recovery (U)": float(metrics.get("recovery", 0.0)),
        "Feature Recovery (V)": v_rec
    }

def run_unified_benchmark(n_seeds=5, iterations=50, epochs=150, use_nsa=True, workers=None):
    regimes = [
        ("Linear", {"kind": "linear"}), 
        ("Polynomial", {"kind": "nonlinear", "regime": "polynomial"}), 
        ("Sine", {"kind": "nonlinear", "regime": "sinusoidal"}), 
        ("Private", {"kind": "shared_plus_private"})
    ]
    model_configs = [("linear", "SiMLR"), ("lend", "LEND"), ("ned", "NED"), ("shared_private", "NEDPP")]
    losses = ["regression", "acc", "logcosh", "nc"]
    mixing_methods = ["newton", "svd", "pca", "ica"]
    
    tasks = []
    for regime_name, case_params in regimes:
        for model_type, model_label in model_configs:
            for energy_type in losses:
                for mixing_algorithm in mixing_methods:
                    for seed in range(42, 42 + n_seeds):
                        tasks.append((regime_name, case_params, model_type, model_label, seed, energy_type, mixing_algorithm, iterations, epochs, use_nsa))
    
    print(f"Starting FULL SYNTHETIC benchmark (v21) with {len(tasks)} tasks...")
    os.makedirs("paper/results_cache", exist_ok=True)
    out_file = "paper/results_cache/unified_synthetic_v21.csv"
    header = ["Regime", "Model", "Seed", "Loss", "Consensus", "CMC", "SRE", "Predictive Accuracy (Y)", "Train Accuracy (Y)", "Gen Gap (Y)", "Strictly Linear Accuracy", "Strictly Linear Train", "Strictly Linear Gap", "Latent Recovery (U)", "Feature Recovery (V)"]
    
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header); writer.writeheader()
    
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_experiment_task, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    with open(out_file, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=header)
                        writer.writerow(res)
                        f.flush() # Forces immediate write to disk
            except Exception as e:
                task_args = futures[future]
                print(f"\n[!] ERROR in task {task_args[0]} | {task_args[2]} | seed {task_args[4]}:")
                traceback.print_exc()
                sys.stdout.flush()
                
            completed += 1
            if completed % 10 == 0: 
                print(f"  Progress: {completed}/{len(tasks)} completed")
                sys.stdout.flush()
                
    print(f"Benchmark complete: {out_file}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Run v21 FULL synthetic benchmark")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--no-nsa", action="store_false", dest="use_nsa")
    parser.set_defaults(use_nsa=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_unified_benchmark(n_seeds=args.n_seeds, iterations=args.iterations, epochs=args.epochs, use_nsa=args.use_nsa, workers=args.workers)
