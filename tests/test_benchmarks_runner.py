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


# --- Regression tests for the evaluation-harness asymmetries ---------------
# Before these, PCA and nsa_pipeline were scored by their own sklearn
# Pipeline (a regularised head on their own scaler) while every other model
# was scored by `cross_val_metrics`, and their invariants were read off the
# joint basis while everyone else's came from the per-view blocks.

def test_split_indices_depend_on_seed():
    """A fixed prefix split made ten seeds of a fixed dataset identical."""
    from pysimlr.benchmarks.runner import split_indices
    a, _ = split_indices(100, seed=42)
    b, _ = split_indices(100, seed=43)
    assert not torch.equal(a, b)
    again, _ = split_indices(100, seed=42)
    assert torch.equal(a, again), "same seed must reproduce the same split"


def test_split_indices_partition_all_rows():
    from pysimlr.benchmarks.runner import split_indices
    tr, te = split_indices(100, seed=7, train_frac=0.7)
    assert len(tr) == 70 and len(te) == 30
    assert sorted(torch.cat([tr, te]).tolist()) == list(range(100))


def test_split_indices_stratify_keeps_every_class_on_both_sides():
    from pysimlr.benchmarks.runner import split_indices
    y = torch.tensor([0] * 90 + [1] * 10)
    for seed in range(10):
        tr, te = split_indices(100, seed=seed, train_frac=0.7, stratify=y)
        assert set(y[tr].tolist()) == {0, 1}
        assert set(y[te].tolist()) == {0, 1}


def test_pipeline_models_do_not_override_the_shared_metric():
    """PCA must be scored by the same estimator as every other model."""
    from pysimlr.benchmarks.runner import run_single_experiment
    from pysimlr.benchmarks.synthetic_cases import build_case
    case = build_case(kind="linear", n_samples=200, seed=42)
    from pysimlr.benchmarks.metrics import cross_val_metrics
    from pysimlr.benchmarks.runner import split_indices

    out = run_single_experiment("pca", case, seed=42, iterations=5)
    metrics = out["metrics"]

    # The pipeline's own (regularised) score is kept, but under its own key
    # so it can never stand in for the shared metric.
    assert "pipeline_test_score" in metrics

    # `test_r2` must be exactly what the shared estimator gives on the latents.
    tr_idx, te_idx = split_indices(
        case["data"][0].shape[0], seed=42, train_frac=0.7, stratify=None
    )
    y = case["outcome"]
    expected = cross_val_metrics(
        out["result"]["custom_pred_train"]["u"], y[tr_idx].numpy(),
        out["result"]["custom_pred_test"]["u"], y[te_idx].numpy(),
        is_classification=False,
    )
    assert metrics["test_r2"] == pytest.approx(expected["test"], abs=1e-9)


def test_invariants_use_per_view_blocks_for_every_model():
    """PCA's D=0.0000 was an artefact of measuring the joint basis instead."""
    from pysimlr.benchmarks.runner import run_single_experiment
    from pysimlr.benchmarks.synthetic_cases import build_case
    case = build_case(kind="linear", n_samples=200, seed=42)
    pca = run_single_experiment("pca", case, seed=42, iterations=5)["metrics"]
    # Per-view restrictions of a joint orthonormal basis are not orthonormal,
    # so an honest per-view frame defect for PCA is strictly positive.
    assert pca["frame_defect"] > 1e-3, (
        "PCA scoring frame_defect 0 means the joint basis is still being used"
    )


def test_every_model_reports_the_axis_sensitive_metric():
    """A hardcoded model allowlist silently zeroed this for new models."""
    from pysimlr.benchmarks.runner import run_single_experiment
    from pysimlr.benchmarks.synthetic_cases import build_case
    case = build_case(kind="linear", n_samples=160, seed=42)
    for model_type in ("linear", "simlr_lbfgs", "pca"):
        m = run_single_experiment(model_type, case, seed=42, iterations=3)["metrics"]
        assert "first_layer_axis_test_r2" in m, (
            f"{model_type} reports no axis-sensitive metric, so the benchmark "
            f"column defaults to 0.0 and the model looks infinitely bad"
        )
        assert m["first_layer_axis_test_r2"] != 0.0
