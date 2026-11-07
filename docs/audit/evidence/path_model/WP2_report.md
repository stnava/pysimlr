# WP2 report: refit null, training topologies, and verification

Executor: Agent A (WP2 Close-out and Verification Worker), 2026-11-06.
Plan: `docs/plans/2026-11-06_path_model.md` sections 1b, 3.2, 4.2, 5, 6, 7; handoff guide `docs/plans/2026-11-06_path_model_handoff.md` section 3.
Baseline commit: `d0d35bd` (feat(sem): path model WP2 (interim)). Dead-code cleanup in `tests/e2e/test_tier4_scenarios.py`.

## What was built

Code in commit `d0d35bd`:
- `sem._refit_null` / `_refit_scores_task` (plan 3.2): refit null grouped by permuted target node $b$. For each view $b$ ending a tested edge, $B$ refits with $b$'s training rows and its confound rows permuted (`confounds.permute_view_confounds`) serve all tested edges into $b$. Freedman-Lane null for outcome edges without refits. Omnibus refits permute all views but the first.
- `null="refit"` and `null="both"` (plan 3.2): when `rows="train"`, defaults to `"refit"`. Reports `p_refit`, `p_refit_holm`, and stores `null_pillai_refit`.
- `compare_path_models` refit mode (plan 4.2): repeated train/test splits (`n_splits`), fits each graph topology on training rows (`method=`, `k=`), evaluates held-out log-likelihood, computes paired deltas against the top graph, and selects `best` by the parsimony rule (F1/F4).
- `predict_path` (plan 5): predicts missing views using neighbor scores across significant directed edges (`pm["edges"]`); returns standardized scores, reconstructed views, and in-sample/out-of-sample $R^2$.
- `extension.predict_views`: fixed joint-basis handling when a subset of views is present (`views[m] is None`).
- `loadings.feature_maps`: score extraction folded into `sem.view_scores` with signs aligned against $u$.
- `paths.permutation_test`: refactored through `pysimlr.fit` + `path_model(rows="train", null="refit", refit_scope=("omnibus",))`, preserving legacy keys (`observed_similarity`, `null_similarities`, `p_value`, `n_permutations`, `n_exceeding`) and add-one floor.
- Removals: deprecated `fit_structural_models` and `diagrams.plot_path_model` removed; dependent tests rewritten to `compare_path_models`.
- Exports: `path_model`, `compare_path_models`, `predict_path`, `view_scores` registered in `__all__` and `_LAZY_ATTRS`.

## Verification commands and verbatim summary lines

- Fast suite:
  `python -m pytest -q -p no:cacheprovider` -> `1998 passed, 1 skipped, 2 xfailed, 942 warnings in 97.44s (0:01:37)`
- Slow SEM suite (including both calibration tests):
  `python -m pytest -q -m "" -p no:cacheprovider tests/test_sem.py` -> `31 passed, 5 warnings in 507.57s (0:08:27)`
- Implementation standards:
  `python -m pytest -q tests/test_implementation_standards.py` -> `7 passed in 3.46s`
- Lint:
  `python -m antsxqc.lint src/pysimlr` -> `0 finding(s)`
- Demo script (`python scripts/path_model_demo.py --out docs/audit/evidence/path_model/wp2`):
  - Smoke results of plan 1b fully reproduced:
    - Present edges (view0-view1, view1-view2, view2-outcome): `p_perm` = 0.000200 (floor 1/5001).
    - Absent edges given parents: view0-view2 0.595881, view0-outcome 0.252350, view1-outcome 0.065587.
    - Graph fit: `present_pass` 1.0, `absent_pass` 1.0, `max_absent_p` 0.121776, `rejected` False.
    - Comparison: `best chain + outcome; best_by_loglik chain + outcome`; chain `best_fraction` 0.9275, complete delta -10.685 (CI -24.047 to 3.643).
    - Prediction: missing-view $R^2$ 0.683; outcome $R^2$ 0.734.
    - Artifacts: `docs/audit/evidence/path_model/wp2/reference_report.html` and 4 PNG figures generated.
- Calibration numbers:
  - Fixed-model null (`test_null_calibration_fixed_model`, 200 datasets, n = 120, B = 500 perms):
    per-edge `p_perm < 0.05` rejection rates = 0.045, 0.050, 0.035; max-T family rate = 0.050.
  - Refit null (`test_null_calibration_refit`, 100 datasets, n = 100, B = 100 refits, simlr 5 iterations):
    per-edge `p_refit < 0.05` rejection rates = [0.070, 0.030, 0.020]; mean = 0.040, max = 0.070.
    Rejection rate strictly satisfies the required [0.02, 0.09] calibration band (plan test 6).
- Permutation test legacy contract & floor:
  `python -m pytest -q tests/test_permutation_pvalues.py -k test_paths_permutation_test_never_returns_zero` -> `1 passed, 1 warning in 3.48s`.
- Confound permutation verification:
  Confirmed in `sem._refit_null` (`_refit_scores_task` line 843): each view's confound rows follow the view's permutation via `confounds.permute_view_confounds`.
  `test_refit_null_permutes_confounds_with_the_view` explicitly asserts that the permuted confounds argument arriving at `api.fit` matches `C[perm]` for each view.
- Dead-code check:
  `grep -rn "fit_structural_models\|holm_adjust\|_bh_qvalues\|_view_scores" src tests docs/usage_notes.md docs/api_contract.md README.md`
  Verified clean: zero active references. All hits are either deprecation docstrings (`structural.py`, `test_structural_models.py`, `test_tier4_scenarios.py`), `test_stats.py` assertions confirming removal (`screening._bh_qvalues`), docstrings referencing removed oracles (`loadings._view_scores`), or the `deep_ranking.holm_adjust` compatibility alias pointing to `pysimlr.stats.holm`.

## Deviations from the plan, with reasons

1. **`refit_scope` parameterization:**
   `sem._refit_null` accepts `refit_scope=("edges", "omnibus")` or subsets. This allows omnibus refits (`permutation_test`) or edge-only refits (`test_null_calibration_refit`) without running unused sweeps.
2. **Backward-compatible return signature for `permutation_test`:**
   Preserved all 5 legacy dictionary keys (`observed_similarity`, `null_similarities`, `p_value`, `n_permutations`, `n_exceeding`) while attaching new rich outputs (`edges`, `omnibus`, `result`).
3. **Multiprocessing macOS sandbox compatibility:**
   Full test suites require `BypassSandbox=True` due to PyTorch IPC shared memory restrictions (`torch.storage._share_filename_cpu_`) and external checkout access (`/Users/stnava/Documents/code/ANTsPy`).

## Open items and WP3 carry-over

- **WP3 carry-over item:** In `plot_path_comparison`, markers for `present_pass` and `absent_pass` overlap when both equal 1.0 (or identical values); Agent B should dodge them vertically.
- `predict_views` with joint basis fits now cleanly handles missing views; full E2E prediction pipeline tested.
