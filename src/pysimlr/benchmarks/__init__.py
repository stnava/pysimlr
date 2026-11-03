
import importlib
from typing import Any

from .metrics import (
    latent_recovery_score,
    outcome_r2_score,
    reconstruction_mse,
    latent_variance_diagnostics,
    shared_private_diagnostics,
    calculate_v_orthogonality,
    calculate_all_metrics,
    first_layer_sparsity_metrics,
    alignment_metrics_from_report,
    shared_attribution_metrics_from_report,
    prediction_preservation_metrics_from_report,
)


_RUNNER_ATTRS = frozenset({
    "run_single_experiment",
    "run_seeded_benchmark",
    "aggregate_results",
    "get_best_per_model",
    "sweep_benchmark",
})

__all__ = [
    "latent_recovery_score",
    "outcome_r2_score",
    "reconstruction_mse",
    "latent_variance_diagnostics",
    "shared_private_diagnostics",
    "calculate_v_orthogonality",
    "calculate_all_metrics",
    "first_layer_sparsity_metrics",
    "alignment_metrics_from_report",
    "shared_attribution_metrics_from_report",
    "prediction_preservation_metrics_from_report",
    "run_single_experiment",
    "run_seeded_benchmark",
    "aggregate_results",
    "get_best_per_model",
    "sweep_benchmark",
]


def __getattr__(name: str) -> Any:
    if name in _RUNNER_ATTRS:
        mod = importlib.import_module(".runner", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
