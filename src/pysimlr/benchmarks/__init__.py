
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

_PLOTTING_ATTRS = frozenset({
    "plot_pareto_recovery_vs_r2",
    "plot_sparsity_sensitivity",
    "plot_stability_diagnostics",
    "plot_sparsity_vs_orthogonality",
    "plot_reconstruction_tradeoff",
    "plot_v_heatmaps",
    "plot_latent_correlation",
    "plot_first_layer_alignment_heatmap",
    "plot_first_layer_feature_importance",
    "plot_interpretability_tradeoff",
})

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
    "plot_pareto_recovery_vs_r2",
    "plot_sparsity_sensitivity",
    "plot_stability_diagnostics",
    "plot_sparsity_vs_orthogonality",
    "plot_reconstruction_tradeoff",
    "plot_v_heatmaps",
    "plot_latent_correlation",
    "plot_first_layer_alignment_heatmap",
    "plot_first_layer_feature_importance",
    "plot_interpretability_tradeoff",
    "run_single_experiment",
    "run_seeded_benchmark",
    "aggregate_results",
    "get_best_per_model",
    "sweep_benchmark",
]


def __getattr__(name: str) -> Any:
    if name in _PLOTTING_ATTRS:
        mod = importlib.import_module(".plotting", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    if name in _RUNNER_ATTRS:
        mod = importlib.import_module(".runner", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
