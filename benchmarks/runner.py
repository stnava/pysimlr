import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import inspect
from typing import List, Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
from pysimlr import simlr, lend_simr, ned_simr, ned_simr_shared_private, predict_simlr
from .synthetic_cases import build_case
from .metrics import calculate_all_metrics

def filter_params(func, params):
    """Filter params dict to only include keys accepted by func."""
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters}

def run_single_experiment(model_type: str, 
                          case: Dict[str, Any], 
                          test_size: float = 0.2,
                          sparsity: float = 0.0,
                          seed: int = 42,
                          **model_params) -> Dict[str, Any]:
    """Train and evaluate a single model variant."""
    data = case["data"]
    n_samples = data[0].shape[0]
    indices = np.arange(n_samples)
    
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    
    train_mats = [m[train_idx] for m in data]
    test_mats = [m[test_idx] for m in data]
    y_test = case["outcome"][test_idx].numpy()
    u_true_test = case["true_u"][test_idx]
    
    k = case["shared_k"]
    
    if model_type == "linear":
        params = filter_params(simlr, model_params)
        res = simlr(train_mats, k=k, sparseness_quantile=sparsity, **params)
    elif model_type == "lend":
        params = filter_params(lend_simr, model_params)
        res = lend_simr(train_mats, k=k, sparseness_quantile=sparsity, **params)
    elif model_type == "ned":
        params = filter_params(ned_simr, model_params)
        res = ned_simr(train_mats, k=k, sparseness_quantile=sparsity, **params)
    elif model_type == "shared_private":
        params = filter_params(ned_simr_shared_private, model_params)
        res = ned_simr_shared_private(train_mats, k=k, sparseness_quantile=sparsity, **params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    pred = predict_simlr(test_mats, res)
    shared_l = pred.get("latents")
    private_l = pred.get("private_latents")
    if model_type == "shared_private" and shared_l is None and "latents" in pred:
         shared_l = pred["latents"]

    metrics = calculate_all_metrics(
        pred['u'], u_true_test, y_test, test_mats, pred['reconstructions'],
        shared_latents=shared_l,
        private_latents=private_l
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
                         **model_params) -> pd.DataFrame:
    """Run benchmark across multiple random seeds."""
    all_metrics = []
    for i in range(n_seeds):
        seed = 42 + i
        case = build_case(kind=case_kind, n_samples=n_samples, seed=seed)
        exp_res = run_single_experiment(model_type, case, sparsity=sparsity, seed=seed, **model_params)
        all_metrics.append(exp_res["metrics"])
    return pd.DataFrame(all_metrics)

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw results by model and sparsity."""
    group_cols = ["model", "sparsity"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["seed"]]
    
    means = df.groupby(group_cols)[metric_cols].mean().reset_index()
    stds = df.groupby(group_cols)[metric_cols].std().reset_index()
    stds = stds.rename(columns={c: f"{c}_sd" for c in metric_cols})
    
    return pd.merge(means, stds, on=group_cols)

def get_best_per_model(df: pd.DataFrame, metric: str = "test_r2") -> pd.DataFrame:
    """Choose the best configuration for each model."""
    if "model" not in df.columns: return pd.DataFrame()
    best_idx = df.groupby("model")[metric].idxmax()
    return df.loc[best_idx].sort_values(metric, ascending=False)

def sweep_benchmark(model_types: List[str],
                    case_kind: str = "nonlinear_shared",
                    n_samples: int = 1000,
                    sparsities: List[float] = [0.0, 0.5, 0.8],
                    n_seeds: int = 3,
                    save_prefix: Optional[str] = None,
                    **common_params) -> Dict[str, pd.DataFrame]:
    """Run full sweep across models and sparsities with multi-seed aggregation."""
    all_raw = []
    for m_type in model_types:
        for spar in sparsities:
            print(f"Benchmarking {m_type} at sparsity {spar} (Seeds: {n_seeds})...")
            df_seeds = run_seeded_benchmark(m_type, case_kind, n_samples, n_seeds, sparsity=spar, **common_params)
            all_raw.append(df_seeds)
            
    df_raw = pd.concat(all_raw, ignore_index=True)
    df_summary = aggregate_results(df_raw)
    df_best = get_best_per_model(df_summary)
    
    if save_prefix:
        df_raw.to_csv(f"{save_prefix}_raw.csv", index=False)
        df_summary.to_csv(f"{save_prefix}_summary.csv", index=False)
        df_best.to_csv(f"{save_prefix}_best.csv", index=False)
        
    return {"raw": df_raw, "summary": df_summary, "best": df_best}

def main():
    parser = argparse.ArgumentParser(description="SiMLR Benchmark Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    model_types = config.get("model_types", ["linear", "ned"])
    sparsities = config.get("sparsities", [0.0, 0.5])
    n_seeds = config.get("n_seeds", 1)
    n_samples = config.get("n_samples", 1000)
    case_kind = config.get("case_kind", "nonlinear_shared")
    save_prefix = config.get("save_prefix", "benchmark_results")
    
    reserved = ["model_types", "sparsities", "n_seeds", "n_samples", "case_kind", "save_prefix"]
    model_params = {k: v for k, v in config.items() if k not in reserved}
    
    results = sweep_benchmark(
        model_types=model_types,
        case_kind=case_kind,
        n_samples=n_samples,
        sparsities=sparsities,
        n_seeds=n_seeds,
        save_prefix=save_prefix,
        **model_params
    )
    
    print(f"Results saved with prefix: {save_prefix}")

if __name__ == "__main__":
    main()
