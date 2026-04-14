# Documentation Audit: pysimlr

Tracking progress of docstring updates to NumPy standards and paper consistency.

## Target APIs

| Module | Symbol | Implementation | Numpy Validity | Correctness | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **benchmarks.metrics** | `alignment_metrics_from_report` | ✅ | ✅ | ✅ | Complete |
| **** | `calculate_all_metrics` | ✅ | ✅ | ✅ | Complete |
| **** | `calculate_v_orthogonality` | ✅ | ✅ | ✅ | Complete |
| **** | `first_layer_sparsity_metrics` | ✅ | ✅ | ✅ | Complete |
| **** | `heldout_outcome_mse` | ✅ | ✅ | ✅ | Complete |
| **** | `heldout_outcome_r2_score` | ✅ | ✅ | ✅ | Complete |
| **** | `in_sample_latent_linear_fit_r2` | ✅ | ✅ | ✅ | Complete |
| **** | `latent_recovery_score` | ✅ | ✅ | ✅ | Complete |
| **** | `latent_variance_diagnostics` | ✅ | ✅ | ✅ | Complete |
| **** | `outcome_r2_score` | ✅ | ✅ | ✅ | Complete |
| **** | `prediction_preservation_metrics_from_report` | ✅ | ✅ | ✅ | Complete |
| **** | `reconstruction_mse` | ✅ | ✅ | ✅ | Complete |
| **** | `reconstruction_mse_summary` | ✅ | ✅ | ✅ | Complete |
| **** | `shared_attribution_metrics_from_report` | ✅ | ✅ | ✅ | Complete |
| **** | `shared_private_diagnostics` | ✅ | ✅ | ✅ | Complete |
| **benchmarks.plotting** | `plot_first_layer_alignment_heatmap` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_first_layer_feature_importance` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_interpretability_tradeoff` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_latent_correlation` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_pareto_recovery_vs_r2` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_reconstruction_tradeoff` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_sparsity_sensitivity` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_sparsity_vs_orthogonality` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_stability_diagnostics` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_v_heatmaps` | ✅ | ✅ | ✅ | Complete |
| **benchmarks.protocol** | `BenchmarkProtocol` | ✅ | ✅ | ✅ | Complete |
| **** | `run_repeated_benchmark` | ✅ | ✅ | ✅ | Complete |
| **benchmarks.runner** | `aggregate_results` | ✅ | ✅ | ✅ | Complete |
| **** | `filter_kwargs` | ✅ | ✅ | ✅ | Complete |
| **** | `get_best_per_model` | ✅ | ✅ | ✅ | Complete |
| **** | `main` | ✅ | ✅ | ✅ | Complete |
| **** | `run_seeded_benchmark` | ✅ | ✅ | ✅ | Complete |
| **** | `run_single_experiment` | ✅ | ✅ | ✅ | Complete |
| **** | `sweep_benchmark` | ✅ | ✅ | ✅ | Complete |
| **benchmarks.shared_private_sweep** | `tune_shared_private` | ✅ | ✅ | ✅ | Complete |
| **benchmarks.synthetic_cases** | `build_case` | ✅ | ✅ | ✅ | Complete |
| **** | `build_linear_footprint_case` | ✅ | ✅ | ✅ | Complete |
| **** | `build_nonlinear_shared_case` | ✅ | ✅ | ✅ | Complete |
| **** | `build_shared_plus_private_case` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_case_generative_shape` | ✅ | ✅ | ✅ | Complete |
| **consensus** | `compute_shared_consensus` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **deep** | `LENDNSAEncoder` | ✅ | ✅ | ✅ | Complete |
| **** | `LENDSiMRModel` | ✅ | ✅ | ✅ | Complete |
| **** | `ModalityDecoder` | ✅ | ✅ | ✅ | Complete |
| **** | `ModalityEncoder` | ✅ | ✅ | ✅ | Complete |
| **** | `NEDSharedPrivateSiMRModel` | ✅ | ✅ | ✅ | Complete |
| **** | `NEDSiMRModel` | ✅ | ✅ | ✅ | Complete |
| **** | `calculate_sim_loss` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `deep_simr` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `lend_simr` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `ned_simr` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `ned_simr_shared_private` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `predict_deep` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **interpretability** | `analyze_first_layer_alignment` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `attribute_prediction_to_features` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `attribute_shared_to_first_layer` | ✅ | ✅ | ✅ | Complete |
| **** | `build_first_layer_contract` | ✅ | ✅ | ✅ | Complete |
| **** | `build_interpretability_report` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `extract_first_layer_factors` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `summarize_basis_matrix` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **nnh** | `antspymm_predictors` | ✅ | ✅ | ✅ | Complete |
| **** | `nnh_embed` | ✅ | ✅ | ✅ | Complete |
| **** | `nnh_update_residuals` | ✅ | ✅ | ✅ | Complete |
| **optimizers** | `Adam` | ✅ | ✅ | ✅ | Complete |
| **** | `ArmijoGradient` | ✅ | ✅ | ✅ | Complete |
| **** | `BidirectionalArmijoGradient` | ✅ | ✅ | ✅ | Complete |
| **** | `BidirectionalLookahead` | ✅ | ✅ | ✅ | Complete |
| **** | `HybridAdam` | ✅ | ✅ | ✅ | Complete |
| **** | `LARS` | ✅ | ✅ | ✅ | Complete |
| **** | `Lookahead` | ✅ | ✅ | ✅ | Complete |
| **** | `NSAFlowOptimizer` | ✅ | ✅ | ✅ | Complete |
| **** | `Nadam` | ✅ | ✅ | ✅ | Complete |
| **** | `RMSProp` | ✅ | ✅ | ✅ | Complete |
| **** | `SGD` | ✅ | ✅ | ✅ | Complete |
| **** | `SimlrOptimizer` | ✅ | ✅ | ✅ | Complete |
| **** | `TorchNativeOptimizer` | ✅ | ✅ | ✅ | Complete |
| **** | `backtracking_linesearch` | ✅ | ✅ | ✅ | Complete |
| **** | `bidirectional_linesearch` | ✅ | ✅ | ✅ | Complete |
| **** | `create_optimizer` | ✅ | ✅ | ✅ | Complete |
| **paths** | `permutation_test` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `simlr_path` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **regression** | `smooth_matrix_prediction` | ✅ | ✅ | ✅ | Complete |
| **** | `smooth_regression` | ✅ | ✅ | ✅ | Complete |
| **simlr** | `calculate_ica_energy` | ✅ | ✅ | ✅ | Complete |
| **** | `calculate_ica_gradient` | ✅ | ✅ | ✅ | Complete |
| **** | `calculate_simlr_energy` | ✅ | ✅ | ✅ | Complete |
| **** | `calculate_simlr_gradient` | ✅ | ✅ | ✅ | Complete |
| **** | `calculate_u` | ✅ | ✅ | ✅ | Complete |
| **** | `decompose_energy` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `estimate_rank` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `initialize_simlr` | ✅ | ✅ | ✅ | Complete |
| **** | `pairwise_matrix_similarity` | ✅ | ✅ | ✅ | Complete |
| **** | `parse_constraint` | ✅ | ✅ | ✅ | Complete |
| **** | `predict_shared_latent` | ✅ | ✅ | ✅ | Complete |
| **** | `predict_simlr` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `project_gradient` | ✅ | ✅ | ✅ | Complete |
| **** | `reconstruct_from_learned_maps` | ✅ | ✅ | ✅ | Complete |
| **** | `simlr` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `simlr_perm` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **sparse** | `sparse_distance_matrix` | ✅ | ✅ | ✅ | Complete |
| **** | `sparse_distance_matrix_xy` | ✅ | ✅ | ✅ | Complete |
| **sparsification** | `indicator_opt_both_ways` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `optimize_indicator_matrix` | ✅ | ✅ | ✅ | Complete |
| **** | `orthogonalize_and_q_sparsify` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `project_to_orthonormal_nonnegative` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `project_to_partially_orthonormal_nonnegative` | ✅ | ✅ | ✅ | Complete |
| **** | `rank_based_matrix_segmentation` | ✅ | ✅ | ✅ | Complete |
| **** | `simlr_sparseness` | ✅ | ✅ | ✅ | Complete |
| **svd** | `ba_svd` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `multiscale_svd` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `safe_pca` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `whiten_matrix` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **utils** | `adjusted_rvcoef` | ✅ | ✅ | ✅ | Complete |
| **** | `get_names_from_dataframe` | ✅ | ✅ | ✅ | Complete |
| **** | `gradient_invariant_orthogonality_defect` | ✅ | ✅ | ✅ | Complete |
| **** | `gradient_mean_orthogonality_defect` | ✅ | ✅ | ✅ | Complete |
| **** | `invariant_orthogonality_defect` | ✅ | ✅ | ✅ | Complete |
| **** | `l1_normalize_features` | ✅ | ✅ | ✅ | Complete |
| **** | `map_asym_var` | ✅ | ✅ | ✅ | Complete |
| **** | `map_lr_average_var` | ✅ | ✅ | ✅ | Complete |
| **** | `mean_orthogonality_defect` | ✅ | ✅ | ✅ | Complete |
| **** | `multigrep` | ✅ | ✅ | ✅ | Complete |
| **** | `orthogonality_summary` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `preprocess_data` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `procrustes_mse` | ✅ | ✅ | ✅ | Complete |
| **** | `procrustes_r2` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `rvcoef` | ✅ | ✅ | ✅ | Complete (Sandbox) |
| **** | `rvcoef_components` | ✅ | ✅ | ✅ | Complete |
| **** | `rvcoef_gram_impl` | ✅ | ✅ | ✅ | Complete |
| **** | `rvcoef_trace_impl` | ✅ | ✅ | ✅ | Complete |
| **** | `safe_svd` | ✅ | ✅ | ✅ | Complete |
| **** | `set_all_seeds` | ✅ | ✅ | ✅ | Complete |
| **** | `set_seed_based_on_time` | ✅ | ✅ | ✅ | Complete |
| **** | `stiefel_defect` | ✅ | ✅ | ✅ | Complete |
| **visualization** | `generate_all_architecture_graphs` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_energy` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_latent_2d` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_lend_simr_architecture` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_ned_shared_private_architecture` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_ned_simr_architecture` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_nsa_flow_architecture` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_v_matrix` | ✅ | ✅ | ✅ | Complete |
| **viz** | `plot_convergence_dynamics` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_feature_signatures` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_latent_consensus` | ✅ | ✅ | ✅ | Complete |
| **** | `plot_view_correlations` | ✅ | ✅ | ✅ | Complete |

## Bugs Found

(Log any logic discrepancies here)

- `simlr.calculate_simlr_gradient` had a double `return torch.zeros_like(v)`. Fixed.
- `regression.smooth_matrix_prediction` had an `UnboundLocalError` and was returning the wrong shape. Fixed.
- `benchmarks.metrics` had several naming mismatches with `protocol.py` and tests. Fixed by adding aliases and supporting flexible report schemas.
- `visualization.py` architecture plotting functions were returning `None`, causing test failures. Fixed to return the `fig` object.
- `viz.plot_feature_signatures` was missing the `top_n` argument. Fixed.
- `viz.plot_convergence_dynamics` failed when history was empty. Fixed.
- `tests/test_comprehensive.py` had an incorrect assertion for `smooth_matrix_prediction` output shape. Fixed.
