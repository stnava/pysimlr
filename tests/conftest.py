"""Pytest configuration for contention-free parallel execution.

Pins each worker process to a single OpenMP / PyTorch intra-op thread
(``torch.set_num_threads(1)``). Without this, ``pytest -n auto`` spawns
``N_workers * 8`` OpenMP threads that thrash on small ``(30..1000) x (6..50)``
SVD and BLAS kernels; even in a single process, ``num_threads=1`` is ~25%
faster on test-sized problems.

Orders multi-second tests/groups first (Longest-Processing-Time-First with
``--no-loadscope-reorder``), schedules medium integration files second, and
batches the millisecond unit-test tail into 25-item xdist groups to avoid
per-item IPC round-trips and quadratic ``assigned_work`` scans in ``xdist``.
"""
import os

import pytest

_LPT_PRIORITY = {
    # Slow-marked statistical / multi-seed contracts (for `pytest -m ""`)
    "test_ned_advantage_contract": 0,
    "test_null_p_values_are_calibrated_at_the_nominal_level": 1,
    "test_linear_parity_contract": 2,
    "test_nedpp_advantage_contract": 3,
    "test_shared_private_non_starvation": 4,
    "test_slow_deep_knobs_change_the_fit": 5,
    "test_null_calibration_fixed_model": 6,
    "test_null_calibration_refit": 7,
    # Heavy default-suite tests and cached xdist_groups (> 1.0s)
    "test_the_quick_start_runs": 10,
    "test_loo_iteration_improves_on_a_wrong_initialization": 11,
    "test_ned_nc_newton_does_not_collapse": 12,
    "test_lbfgsb_is_reproducible_under_float32_input_perturbations": 13,
    "test_the_evaluation_loop_runs_with_no_per_method_branch": 14,
    "test_latent_collapse": 15,
    "test_every_method_returns_its_best_iterate_not_its_last": 16,
    "test_heavy_optional_modules_are_not_imported_eagerly": 17,
    "test_a_shared_parameter_changes_the_fit": 18,
    "test_changing_a_parameter_changes_the_fit": 19,
    "test_deep_knobs_change_the_fit": 20,
    "test_better_optimisation_does_not_silently_improve_support": 21,
    "test_lbfgsb_never_returns_a_rank_deficient_basis": 22,
    "test_p_value_is_in_range_and_respects_the_add_one_floor": 23,
    "test_simlr_perm_reports_permutation_fields": 24,
    "test_null_p_values_are_not_concentrated_at_the_floor": 25,
    "test_paths_permutation_test_never_returns_zero": 26,
    "test_scenario1_multimodal_omics_integration": 27,
    "test_nedpp_shared_private_disentanglement": 28,
    "test_flow_simr_multi_view_training": 29,
    "test_no_rank_loss_at_any_constraint_weight": 30,
    "test_every_constraint_family_is_capped_below_w_equals_one": 31,
    "test_workload_clinical_classification_pipeline": 32,
    "test_entry_points_forward_varimax_and_change_the_fit": 33,
}

_MEDIUM_FILES = {
    "test_alternative_initializations.py",
    "test_api_registry.py",
    "test_benchmark_reporting.py",
    "test_plotting.py",
    "test_benchmarks_runner.py",
    "test_benchmarks_sp_sweep.py",
    "test_collapse_penalty.py",
    "test_comprehensive.py",
    "test_consensus_anchor_prediction.py",
    "test_coverage_deep_simlr_ext.py",
    "test_coverage_ext.py",
    "test_deep_early_stopping.py",
    "test_deep_first_layer_contract.py",
    "test_deep_initialization_and_positivity.py",
    "test_deep_train_infer_consistency.py",
    "test_dynamic_weights.py",
    "test_e2e_opaque_box.py",
    "test_flow_adversarial.py",
    "test_flow_mai.py",
    "test_flow_new_features.py",
    "test_flows.py",
    "test_knobs_are_live.py",
    "test_learning_rate_tuning.py",
    "test_mai_metrics.py",
    "test_milestone1_adversarial_stress.py",
    "test_milestone3_adversarial_structural.py",
    "test_nnh.py",
    "test_nsa_flow_hardening.py",
    "test_optimizer_benchmark_synthetic.py",
    "test_optimizer_correctness.py",
    "test_parameters_are_live.py",
    "test_random_state.py",
    "test_retraction_validation.py",
    "test_sklearn_estimators.py",
    "test_solver_certificate.py",
    "test_structural_models.py",
    "test_support_identifiability.py",
    "test_tier1_features.py",
    "test_tier2_boundaries.py",
    "test_tier3_combinations.py",
    "test_tier4_scenarios.py",
}


def pytest_configure(config):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch

    torch.set_num_threads(1)


def _item_tier(item) -> int:
    prio = _LPT_PRIORITY.get(item.originalname)
    if prio is not None:
        return prio
    if item.path.name in _MEDIUM_FILES:
        return 50
    return 100


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    items.sort(key=_item_tier)
    tail_idx = 0
    for item in items:
        if _item_tier(item) == 100 and item.get_closest_marker("xdist_group") is None:
            item.add_marker(pytest.mark.xdist_group(f"_unit_tail_{tail_idx // 25}"))
            tail_idx += 1
