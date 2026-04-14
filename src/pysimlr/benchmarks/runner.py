import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import inspect
import json
from typing import List, Dict, Any, Optional, Union
from .synthetic_cases import build_case
from .metrics import calculate_all_metrics
from pysimlr.simlr import simlr, predict_simlr, predict_shared_latent
from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private, predict_deep

def filter_kwargs(func, kwargs):
    """
    Filter a dictionary of keyword arguments to only those accepted by a function.

    Uses reflection (inspect) to match provided keys against the function's 
    signature.

    Parameters
    ----------
    func : Callable
        The target function to inspect.
    kwargs : Dict[str, Any]
        A dictionary of potential keyword arguments.

    Returns
    -------
    Dict[str, Any]
        A subset of `kwargs` that can be safely passed to `func`.
    """
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def run_single_experiment(model_type: str, 
                          case: Dict[str, Any], 
                          sparsity: float = 0.0, 
                          seed: int = 42,
                          **params) -> Dict[str, Any]:
    """
    Run a single experiment for a given model architecture and sparsity level.

    Handles training/testing split, model initialization, fitting, and 
    comprehensive evaluation using `calculate_all_metrics`.

    Parameters
    ----------
    model_type : str
        The type of model to run ('linear', 'lend', 'ned', or 'shared_private').
    case : Dict[str, Any]
        The dataset case dictionary (from `build_case`).
    sparsity : float, default=0.0
        The sparsity quantile to apply to the first layer.
    seed : int, default=42
        Random seed for reproducibility.
    **params : Dict[str, Any]
        Additional hyperparameters passed to the model fitting function.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - `metrics`: Dictionary of performance metrics.
        - `result`: The raw result from the model fitting.
        - `model_type`: The name of the model.
        - `sparsity`: The applied sparsity level.
        - `seed`: The random seed used.
    """
    data_all = case["data"]
    u_all = case["true_u"]
    y_all = case["outcome"]
    k = case["shared_k"]
    
    # 70/30 Train/Test split
    n_samples = data_all[0].shape[0]
    train_size = int(n_samples * 0.7)
    
    train_mats = [m[:train_size] for m in data_all]
    test_mats = [m[train_size:] for m in data_all]
    u_true_test = u_all[train_size:]
    y_test = y_all[train_size:]
    y_train = y_all[:train_size]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if model_type == "linear":
        f_params = filter_kwargs(simlr, params)
        res = simlr(train_mats, k=k, sparseness_quantile=sparsity, **f_params)
    elif model_type == "lend":
        f_params = filter_kwargs(lend_simr, params)
        res = lend_simr(train_mats, k=k, sparseness_quantile=sparsity, **f_params)
    elif model_type == "ned":
        f_params = filter_kwargs(ned_simr, params)
        res = ned_simr(train_mats, k=k, sparseness_quantile=sparsity, **f_params)
    elif model_type == "shared_private":
        f_params = filter_kwargs(ned_simr_shared_private, params)
        res = ned_simr_shared_private(train_mats, k=k, sparseness_quantile=sparsity, **f_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Standardize u_train and u_test by passing them both through the prediction path.
    # This ensures coordinate alignment (signs/rotation) for outcome R2.
    if "model" in res: # Deep model
        pred_test = predict_deep(test_mats, res, device="cpu")
        pred_train = predict_deep(train_mats, res, device="cpu")
    else: # Linear SIMLR
        pred_test = predict_simlr(test_mats, res)
        pred_train = predict_simlr(train_mats, res)

    shared_l = pred_test.get("latents")
    private_l = pred_test.get("private_latents")
    
    metrics = calculate_all_metrics(
        pred_test['u'],
        u_true_test,
        y_test,
        test_mats,
        pred_test['reconstructions'],
        shared_latents=shared_l,
        private_latents=private_l,
        v_mats=res.get("v"),
        u_train=pred_train['u'],
        y_train=y_train,
        first_layer=pred_test.get("first_layer") or res.get("first_layer"),
        interpretability=pred_test.get("interpretability") or res.get("interpretability"),
    )
    
    metrics.update({
        "model": model_type,
        "sparsity": sparsity,
        "seed": seed
    })
    
    return {"metrics": metrics, "result": res}

def run_seeded_benchmark(model_type: str, 
                         case_kind: str = "nonlinear_shared",
                         n_samples: int = 1000,
                         n_seeds: int = 3,
                         sparsity: float = 0.0,
                         noise_level: float = 0.1,
                         **model_params) -> pd.DataFrame:
    """
    Run a full benchmark across multiple random seeds for a specific model.

    Automates data generation (via `build_case`) and experiment execution 
    (via `run_single_experiment`) for a specified number of iterations.

    Parameters
    ----------
    model_type : str
        The type of model to benchmark (e.g., `LEND`, `NED`).
    case_kind : str, default="nonlinear_shared"
        The kind of synthetic case to generate.
    n_samples : int, default=1000
        Number of samples in each dataset.
    n_seeds : int, default=3
        Number of independent runs.
    sparsity : float, default=0.0
        The sparsity level to apply.
    noise_level : float, default=0.1
        The noise level for data generation.
    **model_params : Dict[str, Any]
        Additional hyperparameters for the model.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing metrics from all seeds.
    """
    all_metrics = []
    for i in range(n_seeds):
        seed = 42 + i
        case = build_case(kind=case_kind, n_samples=n_samples, seed=seed, noise_level=noise_level)
        exp_res = run_single_experiment(model_type, case, sparsity=sparsity, seed=seed, **model_params)
        all_metrics.append(exp_res["metrics"])
    return pd.DataFrame(all_metrics)

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw benchmark results by model and sparsity level.

    Calculates the mean, standard deviation, and 95% confidence intervals 
    for all metrics across experimental seeds.

    Parameters
    ----------
    df : pd.DataFrame
        The raw results DataFrame from a benchmark sweep.

    Returns
    -------
    pd.DataFrame
        An aggregated DataFrame with summary statistics for each unique 
        model and sparsity configuration.
    """
    group_cols = ["model", "sparsity"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["seed"]]
    
    means = df.groupby(group_cols)[metric_cols].mean().reset_index()
    stds = df.groupby(group_cols)[metric_cols].std().reset_index()
    counts = df.groupby(group_cols)[metric_cols[0]].count().reset_index().rename(columns={metric_cols[0]: "n"})
    
    res = pd.merge(means, stds, on=group_cols, suffixes=('', '_sd'))
    res = pd.merge(res, counts, on=group_cols)
    
    for col in metric_cols:
        res[f"{col}_ci95"] = 1.96 * (res[f"{col}_sd"] / np.sqrt(res["n"]))
    return res.fillna(0.0)

def get_best_per_model(df: pd.DataFrame, metric: str = "test_r2") -> pd.DataFrame:
    """
    Find the best performing configuration for each unique model architecture.

    Groups the results by model and selects the row with the maximum value 
    for the specified metric.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated benchmark results.
    metric : str, default="test_r2"
        The metric to use for determining the "best" configuration.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the best rows for each model.
    """
    if "model" not in df.columns: return pd.DataFrame()
    best_idx = df.groupby("model")[metric].idxmax()
    return df.loc[best_idx].sort_values(metric, ascending=False)

def sweep_benchmark(model_types: List[str],
                    case_kind: str = "nonlinear_shared",
                    n_samples: int = 1000,
                    sparsities: List[float] = [0.0, 0.5, 0.8],
                    n_seeds: int = 3,
                    save_prefix: Optional[str] = None,
                    noise_level: float = 0.1,
                    **common_params) -> Dict[str, pd.DataFrame]:
    """
    Run full benchmark sweep across multiple model architectures and sparsity levels.

    This is the primary entry point for systematic SiMLR benchmarking. It 
    iterates through all model and sparsity combinations, aggregates 
    results across multiple seeds, and identifies the best configuration 
    for each model type.

    Parameters
    ----------
    model_types : List[str]
        List of models to benchmark (e.g., ['linear', 'lend', 'ned']).
    case_kind : str, default="nonlinear_shared"
        The synthetic case to generate.
    n_samples : int, default=1000
        Number of samples in the generated dataset.
    sparsities : List[float], default=[0.0, 0.5, 0.8]
        List of sparsity levels to test.
    n_seeds : int, default=3
        Number of independent runs per configuration.
    save_prefix : Optional[str], default=None
        If provided, saves results to CSV files with this prefix.
    noise_level : float, default=0.1
        The noise level for data generation.
    **common_params : Dict[str, Any]
        Hyperparameters shared across all model runs.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary containing:
        - `raw`: All individual seed results.
        - `summary`: Aggregated results (mean/SD/CI95) per model-sparsity.
        - `best`: The best-performing configuration for each model.
    """
    all_raw = []
    for m_type in model_types:
        for spar in sparsities:
            print(f"Benchmarking {m_type} at sparsity {spar} (Seeds: {n_seeds})...")
            df_seeds = run_seeded_benchmark(m_type, case_kind, n_samples, n_seeds, sparsity=spar, noise_level=noise_level, **common_params)
            all_raw.append(df_seeds)
            
    df_raw = pd.concat(all_raw, ignore_index=True)
    df_summary = aggregate_results(df_raw)
    df_best = get_best_per_model(df_summary)
    
    if save_prefix:
        df_raw.to_csv(f"{save_prefix}_raw.csv", index=False)
        df_summary.to_csv(f"{save_prefix}_summary.csv", index=False)
        df_best.to_csv(f"{save_prefix}_best.csv", index=False)
        manifest = {"model_types": model_types, "case_kind": case_kind, "n_samples": n_samples, "sparsities": sparsities, "n_seeds": n_seeds, "noise_level": noise_level, "common_params": common_params}
        with open(f"{save_prefix}_manifest.json", "w") as f: json.dump(manifest, f, indent=4)
            
    return {"raw": df_raw, "summary": df_summary, "best": df_best}

def main():
    parser = argparse.ArgumentParser(description="SiMLR Benchmark Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    model_types = config.get("model_types", ["linear", "ned"])
    sparsities = config.get("sparsities", [0.0, 0.5])
    n_seeds = config.get("n_seeds", 1); n_samples = config.get("n_samples", 1000)
    case_kind = config.get("case_kind", "nonlinear_shared"); save_prefix = config.get("save_prefix", "benchmark_results")
    noise_level = config.get("noise_level", 0.1)
    reserved_keys = ["model_types", "sparsities", "n_seeds", "n_samples", "case_kind", "save_prefix", "noise_level"]
    common_params = {k: v for k, v in config.items() if k not in reserved_keys}
    sweep_benchmark(model_types=model_types, case_kind=case_kind, n_samples=n_samples, sparsities=sparsities, n_seeds=n_seeds, save_prefix=save_prefix, noise_level=noise_level, **common_params)

if __name__ == "__main__":
    main()
