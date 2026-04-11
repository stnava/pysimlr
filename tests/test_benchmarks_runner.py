import torch
import numpy as np
import pandas as pd
import pytest
from pysimlr.benchmarks.runner import (
    filter_kwargs,
    run_single_experiment,
    run_seeded_benchmark,
    aggregate_results,
    get_best_per_model,
    sweep_benchmark
)
from pysimlr.benchmarks.synthetic_cases import build_case

def test_filter_kwargs():
    def dummy_func(a, b, c=None):
        return a + b
    
    kwargs = {"a": 1, "b": 2, "d": 4}
    filtered = filter_kwargs(dummy_func, kwargs)
    assert filtered == {"a": 1, "b": 2}
    assert "d" not in filtered

def test_run_single_experiment():
    case = build_case(kind="nonlinear_shared", n_samples=100, seed=42)
    # Test linear model
    res_linear = run_single_experiment("linear", case, sparsity=0.0, seed=42, iterations=2)
    assert "metrics" in res_linear
    assert res_linear["metrics"]["model"] == "linear"
    
    # Test lend model
    res_lend = run_single_experiment("lend", case, sparsity=0.0, seed=42, epochs=1)
    assert "metrics" in res_lend
    assert res_lend["metrics"]["model"] == "lend"

    # Test ned model
    res_ned = run_single_experiment("ned", case, sparsity=0.0, seed=42, epochs=1)
    assert "metrics" in res_ned
    assert res_ned["metrics"]["model"] == "ned"

    # Test shared_private model
    res_sp = run_single_experiment("shared_private", case, sparsity=0.0, seed=42, epochs=1)
    assert "metrics" in res_sp
    assert res_sp["metrics"]["model"] == "shared_private"

def test_run_seeded_benchmark():
    df = run_seeded_benchmark("linear", case_kind="nonlinear_shared", n_samples=50, n_seeds=2, iterations=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "recovery" in df.columns

def test_aggregate_results():
    data = {
        "model": ["linear", "linear", "ned", "ned"],
        "sparsity": [0.0, 0.0, 0.0, 0.0],
        "seed": [42, 43, 42, 43],
        "recovery": [0.8, 0.9, 0.7, 0.8],
        "test_r2": [0.5, 0.6, 0.4, 0.5]
    }
    df = pd.DataFrame(data)
    summary = aggregate_results(df)
    assert len(summary) == 2
    assert "recovery" in summary.columns
    assert "recovery_sd" in summary.columns
    assert "recovery_ci95" in summary.columns

def test_get_best_per_model():
    data = {
        "model": ["linear", "linear", "ned", "ned"],
        "sparsity": [0.0, 0.5, 0.0, 0.5],
        "test_r2": [0.8, 0.7, 0.6, 0.9]
    }
    df = pd.DataFrame(data)
    best = get_best_per_model(df, metric="test_r2")
    assert len(best) == 2
    assert best.loc[best["model"] == "linear", "test_r2"].values[0] == 0.8
    assert best.loc[best["model"] == "ned", "test_r2"].values[0] == 0.9

def test_sweep_benchmark():
    # Use very small parameters for speed
    res = sweep_benchmark(
        model_types=["linear"],
        case_kind="nonlinear_shared",
        n_samples=50,
        sparsities=[0.0],
        n_seeds=1,
        iterations=1
    )
    assert "raw" in res
    assert "summary" in res
    assert "best" in res
    assert len(res["raw"]) == 1

def test_run_single_experiment_error():
    case = build_case(kind="nonlinear_shared", n_samples=50)
    with pytest.raises(ValueError, match="Unknown model type"):
        run_single_experiment("unknown", case)


def test_run_single_experiment_pr4_metrics_present():
    case = build_case(kind="nonlinear_shared", n_samples=60, seed=42)
    res = run_single_experiment("lend", case, sparsity=0.0, seed=42, epochs=1)
    metrics = res["metrics"]
    assert "first_layer_density_mean" in metrics
    assert "first_layer_alignment_r2_mean" in metrics
    assert "shared_to_first_layer_r2_mean" in metrics
    assert "first_layer_prediction_preservation" in metrics
