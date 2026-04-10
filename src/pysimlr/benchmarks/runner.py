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
from pysimlr.simlr import simlr, predict_simlr
from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private

def filter_kwargs(func, kwargs):
    """Filter kwargs to only include those accepted by the function."""
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def run_single_experiment(model_type: str, 
                          case: Dict[str, Any], 
                          sparsity: float = 0.0, 
                          seed: int = 42,
                          **params) -> Dict[str, Any]:
    """Run a single experiment for a given model and sparsity."""
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
        
    # Get u_train for metric calculation
    # We need to run prediction on train data to get the latents for fitting downstream model
    from pysimlr.simlr import predict_shared_latent
    if model_type in ["linear", "lend", "ned", "shared_private"]:
        # Standard way to get u from trained model
        u_train = res.get("u")
    else:
        u_train = predict_shared_latent(train_mats, res)

    pred = predict_simlr(test_mats, res)
    shared_l = pred.get("latents")
    private_l = pred.get("private_latents")
    if model_type == "shared_private" and shared_l is None and "latents" in pred:
         shared_l = pred["latents"]

    v_mats = res.get("v")

    metrics = calculate_all_metrics(
        pred['u'], u_true_test, y_test, test_mats, pred['reconstructions'],
        shared_latents=shared_l,
        private_latents=private_l,
        v_mats=v_mats,
        u_train=u_train,
        y_train=y_train
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
    """Run benchmark across multiple random seeds."""
    all_metrics = []
    for i in range(n_seeds):
        seed = 42 + i
        case = build_case(kind=case_kind, n_samples=n_samples, seed=seed, noise_level=noise_level)
        exp_res = run_single_experiment(model_type, case, sparsity=sparsity, seed=seed, **model_params)
        all_metrics.append(exp_res["metrics"])
    return pd.DataFrame(all_metrics)

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw results by model and sparsity with 95% CI."""
    group_cols = ["model", "sparsity"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["seed"]]
    
    # Calculate counts for SEM calculation
    counts = df.groupby(group_cols)[metric_cols[0]].count().reset_index()
    counts = counts.rename(columns={metric_cols[0]: "n"})
    
    means = df.groupby(group_cols)[metric_cols].mean().reset_index()
    stds = df.groupby(group_cols)[metric_cols].std().reset_index()
    
    res = pd.merge(means, stds, on=group_cols, suffixes=('', '_sd'))
    res = pd.merge(res, counts, on=group_cols)
    
    # Calculate 95% CI: 1.96 * (std / sqrt(n))
    for col in metric_cols:
        res[f"{col}_ci95"] = 1.96 * (res[f"{col}_sd"] / np.sqrt(res["n"]))
        
    # Handle single-seed case (n=1) where std and CI95 are NaN
    res = res.fillna(0.0)
        
    return res

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
                    noise_level: float = 0.1,
                    **common_params) -> Dict[str, pd.DataFrame]:
    """Run full sweep across models and sparsities with multi-seed aggregation."""
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
        
        # Save config manifest for provenance
        manifest = {
            "model_types": model_types,
            "case_kind": case_kind,
            "n_samples": n_samples,
            "sparsities": sparsities,
            "n_seeds": n_seeds,
            "noise_level": noise_level,
            "common_params": common_params
        }
        with open(f"{save_prefix}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)
            
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
    noise_level = config.get("noise_level", 0.1)
    
    # Extract all other parameters to pass to model functions
    # This includes epochs, iterations, learning_rate, hidden_dims, etc.
    reserved_keys = ["model_types", "sparsities", "n_seeds", "n_samples", "case_kind", "save_prefix", "noise_level"]
    common_params = {k: v for k, v in config.items() if k not in reserved_keys}
    
    # Run the sweep
    sweep_benchmark(
        model_types=model_types,
        case_kind=case_kind,
        n_samples=n_samples,
        sparsities=sparsities,
        n_seeds=n_seeds,
        save_prefix=save_prefix,
        noise_level=noise_level,
        **common_params
    )

if __name__ == "__main__":
    main()
