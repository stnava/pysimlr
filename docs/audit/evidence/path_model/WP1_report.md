# WP1 report: scores and statistics (path model)

Executor: WP1 agent, 2026-11-06. Plan: `docs/plans/2026-11-06_path_model.md` sections 2, 3, 7, 10.
Nothing committed. New files intent-added (`git add -N`): `src/pysimlr/sem.py`, `tests/test_sem.py`,
`scripts/path_model_demo.py`. `paper/` untouched. Files belonging to the concurrent stats session
(`stats.py`, `test_stats.py`, the `deep_ranking.py` / `screening.py` / `pytest.ini` / `__init__.py` diffs,
D40 in `tracking.md`) were not edited.

## What was built

`src/pysimlr/sem.py` (993 lines; the reference was 538)
- `view_scores(result, views, *, train_views=None, standardize=True, rotate=True, signs=None)`: the single
  score extractor. Handles missing views (`None` / dict by name) for per-view-latent fits, joint-basis
  fits through each view's block of `v` (`source="joint_block"`), training-row signs (`sign_rows`), and
  returns `train_scores` / `train_consensus`.
- Edge statistics from one batched kernel `_stats_from_perm`: with `A = [Q, Qa]` and `T = A^T pi(Rb)`,
  pillai = `trace((Rb*^T Rb*)^{-1} T_a^T T_a) / min(ka, kb)`, `r_j`, and moderation F. One QR per edge;
  each sweep is matrix products plus a batch of k x k solves (no per-sweep QR). The identity permutation gives
  the observed values, so the observed value and its null come from the same code. `beta_j` and `rv` are reported.
- Moderation: per component, F for adding `Ra_j * G` to [S, Ra_j]. The edge statistic is the sum over components.
  The Freedman-Lane null permutes the residual of the reduced model [S, Ra_j], using the same node permutation
  as the other statistics in that sweep.
- Joint sweeps in chunks of 500, one permutation per node per sweep. Families `edges`, `edge_components`,
  `moderation` (on the edge sums) and `outcome_edges`, each with `p_perm`, `p_maxt` (that family's own sweep
  maximum), `p_holm` and `q_bh`. Holm and BH come from `pysimlr.stats.holm` and `bh_fdr`. p-values are add-one.
- Omnibus (`max_pillai`, `mean_pillai`). Graph fit (3.5): `present_pass`, `absent_pass`, `absent_p`
  (unadjusted, per F2), and `max_absent_pillai` with a max-T p over the absent edges only (the reference
  took the max over all edges), plus `rejected`.
- Subject bootstrap (`n_bootstrap=2000`): percentile CIs for pillai, r and beta. Ridge residualisation in each replicate.
- `inference=` hook: `_oob_metrics(..., return_scores=True)` returns the out-of-bag `view_scores`.
  `loading_inference` stores them as `oob_scores` (float32, with `rows`) and `n_rows`. `path_model` computes
  each edge's pillai per refit and reports `ci_refit_low` / `ci_refit_high` without extra fits.
- Wide-view refusal (D-6). `feature_contributions` (structure_a[:, j] * beta_j, plan 5).
  `compare_path_models` with F1 (`best` by the parsimony rule, `best_by_loglik`, `best_rule`) and F4
  (`_n_params` counts each undirected edge once). `predict_path` now goes through `view_scores(signs=pm["signs"])`.
- Outputs are DataFrames: `edges`, `edge_components`, `moderation`, `outcome_edges`, `feature_contributions`.

Refactors
- `loadings._oob_metrics` now uses `view_scores`; its old extraction is gone. `_fit_structure` and
  `loading_inference` carry the out-of-bag scores.
- `utils.heldout_coupling(result, views, *, train_views=None)` uses `view_scores(rotate=False,
  standardize=False)`. Joint fits now work: they need `train_views` for the training statistics, otherwise
  those statistics are `None` with a `train_reason`. New key `scores_source`.
- `benchmarks/synthetic_cases.build_path_graph_case`: BFS spanning forest, alpha = 0.7 chain latents,
  `covariate_effect`, `group_effect` on `moderated_edge`, one private RNG stream. `_sparse_signed_basis`
  gained an optional `generator=`.
- `scripts/path_model_demo.py` now uses the generator and the DataFrame outputs, and still fits with
  `confounds=C`. It writes to `--out` (WP1 run: `docs/audit/evidence/path_model/wp1/`; the supervisor's
  `reference_*` files were not overwritten).
- Tests: `tests/test_sem.py` (21 tests), `tests/test_heldout_coupling.py` (joint case rewritten),
  `tests/conftest.py` (`test_null_calibration_fixed_model` added to `_LPT_PRIORITY`).

## Commands and verbatim summary lines

- `python -m pytest -n auto -q` -> `1993 passed, 1 skipped, 2 xfailed, 961 warnings in 84.96s (0:01:24)`
- `python -m pytest -q -m "" -k "test_sem" tests/test_sem.py` -> `21 passed in 53.79s`
- `python -m pytest tests/test_implementation_standards.py -q` -> `7 passed in 3.75s`
- `python -m antsxqc.lint src/pysimlr` -> `0 finding(s)`
- `python scripts/path_model_demo.py --out docs/audit/evidence/path_model/wp1` (chain, simlr, sink, B = 5000):
  - present edges view0-view1, view1-view2 and view2-outcome: `p_perm` 0.000200 (floor 1/5001).
  - absent edges given parents: view0-view2 0.596, view0-outcome 0.252, view1-outcome 0.066.
  - fit: present_pass 1.0, absent_pass 1.0, max_absent_p 0.122, rejected False.
  - comparison: `best chain + outcome; best_by_loglik chain + outcome`. Chain best_fraction 0.9275, complete
    delta -10.7 (CI -24.0 to 3.6).
  - missing-view R2 0.683, outcome R2 0.734.
  - Smoke results of plan 1b all reproduced. In the supervisor's reference run, view0-outcome was 0.036.
- Calibration numbers (scratch scripts mirroring tests 3 and 5, same generator and seeds):
  - test 5 setting (200 datasets, B = 500, n = 120): per-edge `p_perm < 0.05` rates 0.045 / 0.05 / 0.035; max-T family rate 0.05.
  - test 3 setting over 100 seeds: per-edge rates 0.03 / 0.06 / 0.06; max-T 0.07; p quantiles (0.1, 0.25, 0.5, 0.75) = 0.095, 0.244, 0.477, 0.701.
- `test_fixed_null_cost` (M = 4, k = 5, n = 300, B = 10 000, complete graph): 0.81 s (limit 30 s; plan target 10 s).

Tests, by plan section 7 number: 1 `test_view_scores_matches_oob_metrics_frame[simlr|pca]` (bitwise, `np.array_equal`,
against the old `_oob_metrics` code kept as an oracle), 2, 3, 4, 5 (slow), 7 (simlr, lend, ned, flow_v, pca), 8, 9, 15, 17.
Additional tests:
- `test_sink_fit_and_covariates_compose` (coordinator note 3: `fit(confounds=C)` plus `covariates=C`; present
  edges at the floor and absent edges p > 0.05 in >= 8/10 seeds; passes).
- add-one floor and stored nulls; bootstrap coverage; the inference hook; the F4 parameter count; the F1 parsimony rule.

## Deviations from the plan, with reasons

1. **Test 3 criterion.** "Every edge `p_perm > 0.05` in at least 9 of 10 seeds" fails for a calibrated test.
   - With 3 null edges, P(all pass) = 0.857 per seed and P(>= 9 of 10 seeds) = 0.55.
   - Seeds 0-9 gave 6/10 on the per-edge criterion and 7/10 on max-T, yet 100 seeds show nominal rates (numbers above).
   - Following the rule "an implausible result is a defect to find": I checked calibration first and found no defect.
   - The test now runs 30 seeds. It asserts inflation without covariates in all 30, a pooled per-edge rejection rate
     <= 0.10 (nominal 0.05) and a max-T family rate <= 0.15.
2. **Test 4** uses the F2 tolerance: 8 of 10 seeds, n = 500, noise 0.3, held-out 150 rows.
3. **`null="refit"` / `"both"` raise `NotImplementedError`** (refit null is WP2). Consequence: `rows="train"`
   with the default null raises until WP2. `n_refits` and `n_jobs` are accepted and recorded but unused.
4. **Added parameters and keys:**
   - Parameters: `train_outcome=` (the inference hook needs the outcome on loading_inference's OOB rows);
     `view_scores` `standardize`, `rotate`, `signs`.
   - Result keys: `diagnostic` column, `families`, `outcome_edges` table, `sign_rows`, `wide_views`, `p_reason`;
     `edge_components` also has `p_holm`, `beta_ci_low` and `beta_ci_high`.
   - The `outcome_edges` family is a separate table, so `edges.p_holm` stays the all-edges family.
5. **Consensus node** (`include_consensus=True`): its edges are tested (`p_perm`) but kept out of every family,
   the omnibus, the graph fit and every conditioning set.
6. **Wide views** are computed for every family as `p_m >= n_train` from `result["v"]`. Before this, only `simlr`
   results carried `wide_views`.
7. **Residualisation.**
   - Main path: rank-revealing QR (tolerance 1e-10), as in the reference. This is not the plan's "ridge 1e-8".
     The two are equivalent at full rank, and QR detects collinearity: an edge whose scores are collinear with S
     raises `ValueError`.
   - Bootstrap: ridge 1e-8 * trace / cols. Both choices are recorded in `parameters["residualization"]`.
8. **`compare_path_models` (F1).** The column `delta_vs_best` became `delta_vs_top` (measured against
   `best_by_loglik`), because `best` is no longer the top graph. The demo plot was updated to match.
   `best_fraction` is the share of replicates in which a graph is the top.
9. **The graph-fit test** is reported only under `conditioning="parents"`. Under `"pairwise"` its values are
   `None` with a `reason`; the reference computed it regardless.
10. **`heldout_coupling` stays in the method's native frame** (`rotate=False`). Its statistics are absolute
    correlations, and the existing contract (`coupling_train == result["coupling_train"]`) is in that frame.
11. **Moderation statistic**: the edge-level test uses the sum of per-component F, as plan section 2 states.
    Per-component rows carry `p_perm` only.
12. **Random streams**: permutations and bootstrap come from `SeedSequence(random_state).spawn(2)`, so p-values
    differ in the last digits from the reference run's stream.

## Open questions for the supervisor

- `predict`/`predict_views` raises `IndexError` for joint-basis fits with a missing view (pre-existing; seen
  with `pca`). `view_scores` now refuses that case with a clear `ValueError`. Fix it in `extension.predict_views`
  (WP2)?
- `loadings.feature_maps` still has its own score extraction (`loadings._view_scores`, line 524). Its signs are
  learned on the evaluation rows against a given `u`, which differs from `view_scores`. It is outside the WP1 list.
  Fold it in during WP2?
- `loading_inference` now always stores `oob_scores`: float32, n_subsamples x n_oob x M x k, about 4 MB at
  n = 1000, M = 4, k = 5, 100 subsamples. Should that be opt-in for very large n?
- `__init__` exports (`path_model`, `view_scores`, ...) are left for WP2 as the plan schedules. Tests import
  `pysimlr.sem` directly.
