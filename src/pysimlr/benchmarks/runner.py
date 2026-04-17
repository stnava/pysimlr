import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import inspect
import json
from typing import List, Dict, Any, Optional, Union, Callable
from .synthetic_cases import build_case
from .metrics import calculate_all_metrics
from pysimlr.simlr import simlr, predict_simlr, predict_shared_latent
from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private, predict_deep

def filter_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter a dictionary of keyword arguments to only those accepted by a function.
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
    """
    data_all = case["data"]
    u_all = case["true_u"]
    y_all = case["outcome"]
    k = case["shared_k"]
    
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
        
    if "model" in res:
        pred_test = predict_deep(test_mats, res, device="cpu")
        pred_train = predict_deep(train_mats, res, device="cpu")
    else:
        pred_test = predict_simlr(test_mats, res)
        pred_train = predict_simlr(train_mats, res)

    shared_l = pred_test.get("latents")
    private_l = pred_test.get("private_latents")
    
    fl_scores_train = pred_train.get("first_layer_scores")
    fl_scores_test = pred_test.get("first_layer_scores")
    if fl_scores_train is None and model_type == "linear":
        fl_scores_train = pred_train.get("latents")
        fl_scores_test = pred_test.get("latents")

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
        first_layer_scores_train=fl_scores_train,
        first_layer_scores_test=fl_scores_test,
    )
    
    metrics.update({"model": model_type, "sparsity": sparsity, "seed": seed})
    return {"metrics": metrics, "result": res}

def run_seeded_benchmark(model_type: str, 
                         case_kind: str = "nonlinear_shared",
                         n_samples: int = 1000,
                         n_seeds: int = 3,
                         sparsity: float = 0.0,
                         noise_level: float = 0.1,
                         **model_params) -> pd.DataFrame:
    all_metrics = []
    for seed in range(42, 42 + n_seeds):
        case = build_case(n_samples=n_samples, kind=case_kind, seed=seed, noise_scale=noise_level)
        res = run_single_experiment(model_type, case, sparsity=sparsity, seed=seed, **model_params)
        all_metrics.append(res["metrics"])
    return pd.DataFrame(all_metrics)

def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "sparsity"]
    if "energy_type" in results_df.columns: group_cols.append("energy_type")
    if "mixing_algorithm" in results_df.columns: group_cols.append("mixing_algorithm")
    
    agg_df = results_df.groupby(group_cols).agg(['mean', 'std']).reset_index()
    return agg_df

def get_best_per_model(results_df: pd.DataFrame, metric: str = "recovery") -> pd.DataFrame:
    means = results_df.groupby(["model", "sparsity"]).agg({metric: "mean"}).reset_index()
    idx = means.groupby("model")[metric].transform(max) == means[metric]
    return means[idx]

def sweep_benchmark(model_types: List[str] = ["linear", "lend", "ned"],
                    case_kind: str = "nonlinear_shared",
                    n_samples: int = 1000,
                    sparsities: List[float] = [0.0],
                    n_seeds: int = 3,
                    save_prefix: str = "benchmark",
                    noise_level: float = 0.1,
                    **common_params) -> pd.DataFrame:
    results = []
    for m_type in model_types:
        for spar in sparsities:
            print(f"Running sweep: Model={m_type}, Sparsity={spar}")
            df_seeds = run_seeded_benchmark(m_type, case_kind, n_samples, n_seeds, sparsity=spar, noise_level=noise_level, **common_params)
            results.append(df_seeds)
            
    full_df = pd.concat(results, ignore_index=True)
    full_df.to_csv(f"{save_prefix}_results.csv", index=False)
    return full_df

def main():
    parser = argparse.ArgumentParser(description="SiMLR Benchmark Suite")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f: config = yaml.safe_load(f)
    else:
        config = {"model_types": ["linear", "lend", "ned"], "case_kind": "nonlinear_shared", "n_samples": 500, "n_seeds": 3, "sparsities": [0.0], "save_prefix": "smoke_test"}
    
    m_types = config.pop("model_types")
    c_kind = config.pop("case_kind")
    n_s = config.pop("n_samples")
    n_seeds = config.pop("n_seeds")
    spars = config.pop("sparsities")
    pref = config.pop("save_prefix")
    nl = config.pop("noise_level", 0.1)
    sweep_benchmark(m_types, c_kind, n_s, spars, n_seeds, pref, nl, **config)

if __name__ == "__main__":
    main()
