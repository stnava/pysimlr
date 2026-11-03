"""Unit test for package-default linear budget policy in benchmark harnesses (B1)."""
import sys
import pytest
import torch
from unittest.mock import MagicMock

import pysimlr.api as api
from pysimlr.benchmarks.runner import run_single_experiment
from pysimlr.benchmarks.synthetic_cases import build_case
import scripts.default_selection_benchmark as dsb


def test_none_budget_is_omitted_in_run_single_experiment(monkeypatch):
    """When budget_by_unit has no entry for a unit (e.g. iterations is None), iterations kwarg is omitted."""
    case = build_case("linear", n_samples=30, seed=0)
    captured_kwargs = {}

    def mock_fit(model_type, views, k, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "method": model_type,
            "u": torch.randn(30, k),
            "v": [torch.randn(v.shape[1], k) for v in views],
            "loadings": [torch.randn(v.shape[1], k) for v in views],
            "parameters": {"ignored": set(), "resolved": {}, "defaults": {}},
            "convergence": {"converged": True, "n_steps": 5, "max_steps": 300, "stop_reason": "converged"},
        }

    k = case["true_u"].shape[1]

    def mock_predict(result, views):
        return {
            "latents": [torch.randn(v.shape[0], k) for v in views],
            "u": torch.randn(views[0].shape[0], k),
            "first_layer_scores": [torch.randn(v.shape[0], k) for v in views],
            "reconstructions": [torch.randn(v.shape[0], v.shape[1]) for v in views],
        }

    monkeypatch.setattr(api, "fit", mock_fit)
    monkeypatch.setattr(api, "predict", mock_predict)

    # 1. budget_by_unit has only non-None entries (e.g. epochs only, iterations was None)
    run_single_experiment("simlr", case, sparsity=0.0, seed=0,
                           budget_by_unit={"epochs": 45})
    assert "iterations" not in captured_kwargs, "iterations should be omitted when not in budget_by_unit"

    # 2. budget_by_unit specifies iterations explicitly
    captured_kwargs.clear()
    run_single_experiment("simlr", case, sparsity=0.0, seed=0,
                           budget_by_unit={"iterations": 25, "epochs": 45})
    assert captured_kwargs.get("iterations") == 25


def test_none_budget_is_omitted_in_default_selection_run_one(monkeypatch):
    """In default_selection_benchmark.run_one, iterations=None omits iterations= from simlr call."""
    captured_kwargs = {}

    def mock_simlr(views, k=2, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "u": torch.randn(views[0].shape[0], k),
            "v": [torch.randn(v.shape[1], k) for v in views],
            "loadings": [torch.randn(v.shape[1], k) for v in views],
            "normalizing_weights": [1.0 / len(views)] * len(views),
            "stop_reason": "converged",
            "learning_rate": 0.01,
            "convergence": {"converged": True, "n_steps": 10, "max_steps": 300, "stop_reason": "converged"},
        }

    simlr_module = sys.modules["pysimlr.simlr"]
    monkeypatch.setattr(simlr_module, "simlr", mock_simlr)

    task_none = ("nc", "hybrid_adam", "newton", "positive", "linear", 0, None)
    res_none = dsb.run_one(task_none)
    assert "iterations" not in captured_kwargs, "iterations should be omitted when task has None"
    assert res_none["budget"] == 300
    assert res_none["converged"] is True
    assert res_none["n_steps"] == 10

    captured_kwargs.clear()
    task_25 = ("nc", "hybrid_adam", "newton", "positive", "linear", 0, 25)
    res_25 = dsb.run_one(task_25)
    assert captured_kwargs.get("iterations") == 25
    assert res_25["budget"] == 300 or res_25["budget"] == 25
