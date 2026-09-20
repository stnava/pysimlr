#!/usr/bin/env python
"""
Deep Method Ranking Benchmark Suite for PySIMLR.

Executes a statistically powered benchmark across 7 model architectures over:
  1. 4 Synthetic Generative Regimes with known ground-truth U* and V*
     (Linear, Polynomial, Sine, Shared+Private).
  2. 3 Real Clinical and Multimodal Datasets
     (Heart Disease, Diabetes Progression, Multi-Omics 3-View).
  3. N_seeds >= 10 independent replications per task, establishing a
     permutation resolution floor of 2 / 2^10 = 0.00195.

Captures a complete evaluation battery:
  - Downstream predictive capacity (R2 / Accuracy / Generalization Gap)
  - Latent coordinate recovery (CMC on U)
  - Feature projection recovery (SRE / Procrustes R2 on V)
  - Axis sensitivity (Random Forest XV vs Strictly Linear XV)
  - Invariant fidelity (Frame Defect D, Lobe Crosstalk, Sparsity)
  - Computational efficiency (Wall-clock fit time in seconds)

Saves consolidated results to paper/results_cache/deep_ranking_benchmark.csv.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Ensure single-threading per process to prevent thread contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pysimlr.benchmarks.runner import run_single_experiment
from pysimlr.benchmarks.synthetic_cases import build_case
from pysimlr.utils import procrustes_r2
from pysimlr.benchmarks.metrics import support_recovery_score


def get_diabetes_case(seed: int = 42) -> dict:
    data = load_diabetes()
    # Unscaled on purpose. Standardising here fits the mean and variance on
    # every row, including the rows that become the test fold, so the test
    # features are centred with information drawn from themselves. The scaler
    # is fitted on the training split instead -- see `needs_scaling` and
    # `run_single_experiment`.
    X_scaled = data.data
    y = data.target
    mats = [X_scaled[:, :5], X_scaled[:, 5:]]
    k = 2
    return {
        "kind": "diabetes_real",
        "data": [torch.tensor(m).float() for m in mats],
        "outcome": torch.tensor(y).float(),
        "true_u": torch.zeros(X_scaled.shape[0], k),
        "true_v": [np.zeros((m.shape[1], k)) for m in mats],
        "shared_k": k,
        "is_classification": False,
        "needs_scaling": True,
    }


def get_heart_case(seed: int = 42) -> dict:
    url_heart = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    try:
        df_h = pd.read_csv(url_heart, names=cols).replace('?', np.nan).dropna().apply(pd.to_numeric)
    except Exception:
        np.random.seed(seed)
        X_dummy = np.random.randn(300, 13)
        y_dummy = np.random.randint(0, 2, 300)
        df_h = pd.DataFrame(X_dummy)
        df_h['num'] = y_dummy
    X_h = df_h.drop('num', axis=1).values.astype(float)  # scaled per-fold
    y = (df_h['num'].values > 0).astype(int)
    mats = [X_h[:, :7], X_h[:, 7:]]
    k = 2
    return {
        "kind": "heart_real",
        "data": [torch.tensor(m).float() for m in mats],
        "outcome": torch.tensor(y).float(),
        "true_u": torch.zeros(X_h.shape[0], k),
        "true_v": [np.zeros((m.shape[1], k)) for m in mats],
        "shared_k": k,
        "is_classification": True,
        "needs_scaling": True,
    }


def get_multiomics_case(n_samples: int = 400, seed: int = 42) -> dict:
    """
    3-view multi-omics benchmark (mRNA, CNV, Protein) with known biological latents.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    k = 3
    # Latent biological drivers (e.g. subtype, immune infiltration, proliferation)
    u = rng.standard_normal((n_samples, k))
    
    # View 1: mRNA (60 features)
    v1 = rng.standard_normal((60, k))
    x1 = u @ v1.T + 0.3 * rng.standard_normal((n_samples, 60))
    
    # View 2: CNV (40 features)
    v2 = rng.standard_normal((40, k))
    x2 = u @ v2.T + 0.4 * rng.standard_normal((n_samples, 40))
    
    # View 3: Proteomics (30 features)
    v3 = rng.standard_normal((30, k))
    x3 = u @ v3.T + 0.3 * rng.standard_normal((n_samples, 30))
    
    # Phenotype outcome: continuous survival index influenced by latents
    outcome = 1.5 * u[:, 0] - 1.2 * u[:, 1] + 0.8 * u[:, 2] + 0.2 * rng.standard_normal(n_samples)
    
    mats = [x1, x2, x3]  # scaled per-fold, not across the whole cohort
    return {
        "kind": "multiomics_3view",
        "data": [torch.tensor(m).float() for m in mats],
        "outcome": torch.tensor(outcome).float(),
        "true_u": torch.tensor(u).float(),
        "true_v": [v1, v2, v3],
        "shared_k": k,
        "is_classification": False,
        "needs_scaling": True,
    }


def calculate_v_recovery(res: dict, true_v: list) -> float:
    v_est = res.get('v')
    if v_est is None or true_v is None:
        return 0.0
    r2s = []
    for ve, vt in zip(v_est, true_v):
        if vt is None:
            continue
        vt_t = torch.as_tensor(vt).float()
        if bool((vt_t == 0).all()):
            continue
        ve_t = torch.as_tensor(ve).float()
        r2s.append(procrustes_r2(vt_t, ve_t))
    return float(np.mean(r2s)) if r2s else 0.0


def run_experiment_task(task_args: tuple) -> dict:
    torch.set_num_threads(1)
    dataset_name, case_type, model_type, model_label, seed, iterations, epochs = task_args

    if case_type == "synthetic":
        if dataset_name == "Linear":
            case = build_case(seed=seed, kind="linear")
        elif dataset_name == "Polynomial":
            case = build_case(seed=seed, kind="nonlinear", regime="polynomial")
        elif dataset_name == "Sine":
            case = build_case(seed=seed, kind="nonlinear", regime="sinusoidal")
        elif dataset_name in ("Shared+Private", "Private"):
            case = build_case(seed=seed, kind="shared_plus_private")
        elif dataset_name == "NonnegParts":
            case = build_case(seed=seed, kind="nonneg_parts")
        elif dataset_name == "NonnegPartsWeak":
            case = build_case(seed=seed, kind="nonneg_parts_weak")
        else:
            case = build_case(seed=seed, kind="linear")
        is_classif = False
    elif dataset_name == "Heart":
        case = get_heart_case(seed=seed)
        is_classif = True
    elif dataset_name == "Diabetes":
        case = get_diabetes_case(seed=seed)
        is_classif = False
    elif dataset_name == "MultiOmics":
        case = get_multiomics_case(n_samples=400, seed=seed)
        is_classif = False
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    energy_type = "acc" if is_classif else "regression"
    params = {
        "iterations": iterations,
        "epochs": epochs,
        "energy_type": energy_type,
        "mixing_algorithm": "newton",
        "use_nsa": True,
        "positivity": "positive",
        "nsa_w": 0.5,
        # sparseness_quantile is ignored: sparsity is a consequence of w.
        "sparseness_quantile": 0.0,
        "nsa_iterations": 3,
        "consolidate": True,
    }
    if model_type == "shared_private" and "private_k" in case:
        params["private_k"] = case["private_k"]

    exp_res = run_single_experiment(model_type, case, seed=seed, **params)
    metrics = exp_res["metrics"]
    res = exp_res["result"]

    v_rec = calculate_v_recovery(res, case.get("true_v"))
    # `v_rec` is Procrustes-aligned and therefore rotation invariant: it scores
    # the subspace, not the axes. `v_sup` scores which features load on which
    # component, which is what sparsity and disjoint support actually claim.
    v_sup = support_recovery_score(res.get("v") or [], case.get("true_v") or [])
    test_metric = metrics.get("test_accuracy", 0.0) if is_classif else metrics.get("test_r2", 0.0)
    train_metric = metrics.get("train_accuracy", 0.0) if is_classif else metrics.get("train_r2", 0.0)
    lin_test = metrics.get("first_layer_test_accuracy", 0.0) if is_classif else metrics.get("first_layer_test_r2", 0.0)
    rf_test = metrics.get("first_layer_axis_test_accuracy", 0.0) if is_classif else metrics.get("first_layer_axis_test_r2", 0.0)

    return {
        "Dataset": dataset_name,
        "Case_Type": case_type,
        "Model": model_label,
        "Seed": seed,
        "Predictive_Metric": float(test_metric),
        "Train_Metric": float(train_metric),
        "Gen_Gap": float(train_metric - test_metric),
        "Strictly_Linear_Metric": float(lin_test),
        "Axis_Sensitive_Metric": float(rf_test),
        "CMC_Latent_U": float(metrics.get("recovery", 0.0)),
        "Feature_Recovery_V": float(v_rec),
        "Support_Recovery_V": float(v_sup),
        "Frame_Defect_D": float(metrics.get("frame_defect", 0.0)),
        "Lobe_Crosstalk": float(metrics.get("lobe_crosstalk", 0.0)),
        "Sparsity_Ratio": float(metrics.get("sparsity_ratio", 0.0)),
        "Fit_Seconds": float(metrics.get("fit_seconds", 0.0)),
        "Stop_Reason": metrics.get("stop_reason"),
        "Solver_Converged": metrics.get("solver_converged"),
        "Grad_Map": metrics.get("grad_map"),
        "Energy_Reduction": metrics.get("energy_reduction"),
        "Solver_Iters": metrics.get("solver_iters"),
    }


def run_deep_benchmark(n_seeds: int = 10, workers: int = 4, iterations: int = 40, epochs: int = 80) -> pd.DataFrame:
    # `NonnegParts` and `NonnegPartsWeak` have non-negative, sparse, disjoint
    # ground-truth loadings. Every other synthetic regime draws `V` with
    # `torch.randn`, so the truth is signed and dense and a non-negative method
    # is being scored on a basis it cannot represent. Without these two there
    # is no regime in the suite where the structural hypothesis holds.
    synthetic_regimes = ["Linear", "Polynomial", "Sine", "Shared+Private",
                         "NonnegParts", "NonnegPartsWeak"]
    real_regimes = ["Heart", "Diabetes", "MultiOmics"]

    model_configs = [
        ("pca", "PCA"),
        ("linear", "SiMLR"),
        ("simlr_lbfgs", "SiMLR-LBFGS"),
        ("simlr_lbfgsb", "SiMLR-LBFGSB"),
        ("nsa_pipeline", "NSAFlow-Turnkey"),
        ("lend", "LEND"),
        ("ned", "NED"),
        ("flow_v", "Flow-SiMLR-V"),
        ("shared_private", "NEDPP"),
    ]

    tasks = []
    # Synthetic regimes
    for reg in synthetic_regimes:
        for model_type, model_label in model_configs:
            for seed in range(42, 42 + n_seeds):
                tasks.append((reg, "synthetic", model_type, model_label, seed, iterations, epochs))

    # Real / Multimodal regimes
    for dset in real_regimes:
        for model_type, model_label in model_configs:
            for seed in range(42, 42 + n_seeds):
                tasks.append((dset, "real", model_type, model_label, seed, iterations, epochs))

    total_tasks = len(tasks)
    print(f"Starting Deep Benchmark with {total_tasks} tasks across {n_seeds} seeds and {len(model_configs)} models...")
    print(f"Workers: {workers} parallel processes (single-threaded PyTorch).")

    results = []
    t_start = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(run_experiment_task, t): t for t in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                rec = future.result()
                results.append(rec)
            except Exception as exc:
                print(f"[ERROR] Task {task} failed with exception: {exc}")
            completed += 1
            if completed % 25 == 0 or completed == total_tasks:
                elapsed = time.perf_counter() - t_start
                rate = completed / elapsed
                rem = (total_tasks - completed) / max(0.01, rate)
                print(f"[{completed}/{total_tasks}] ({completed/total_tasks*100:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {rem:.1f}s | Rate: {rate:.2f} tasks/s")

    df = pd.DataFrame(results)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "paper", "results_cache")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "deep_ranking_benchmark.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved {len(df)} records to {out_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Deep Method Ranking Benchmark")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--workers", type=int, default=6, help="Worker processes (default: 6)")
    parser.add_argument("--iterations", type=int, default=30, help="SiMLR iterations (default: 30)")
    parser.add_argument("--epochs", type=int, default=60, help="Deep epochs (default: 60)")
    args = parser.parse_args()

    run_deep_benchmark(
        n_seeds=args.seeds,
        workers=args.workers,
        iterations=args.iterations,
        epochs=args.epochs
    )
