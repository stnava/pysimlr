# WP5 report: documentation, API contract, and project tracking

Executor: Agent D (WP5 Documentation Worker), 2026-11-07.
Plan: `docs/plans/2026-11-06_path_model.md` section 10; handoff guide `docs/plans/2026-11-06_path_model_handoff.md` section 5.
Baseline commit: `6fe0ce8` (docs(wp4): record ADNI pilot evidence report and artifacts).
Evidence files referenced: `WP1_report.md`, `WP2_report.md`, `WP3_report.md`, `WP4_report.md`.

## 1. What was built

1. **`docs/usage_notes.md` ("Edges between views: the score-level path model"):**
   - Added section placed immediately after "New samples", matching the scientific and technical register of `feature_relevance`.
   - Complete workflow coverage: measurement model fitting, score projection via `view_scores`, edge statistics (`pillai`, `r`, `beta`, `rv`, `moderation`, `outcome_edges`), Freedman–Lane fixed-model null, refit null on training rows, multiplicity control (`p_perm`, `p_maxt`, `p_holm`, `q_bh`), graph-fit test (`present_pass`, `absent_pass`, `max_absent_pillai`), bootstrap and out-of-bag refit intervals, model comparison via `compare_path_models`, missing-view prediction via `predict_path`, and plotting.
   - Explicitly discussed what the unit tests cannot say:
     - **F2 attenuation:** conditioning on estimated latent scores attenuates absent edges rather than eliminating them, so absent edges given parents err toward over-rejection under measurement error.
     - **Max-T under partial nulls:** single-step max-$T$ controls FWER under the global null; subset pivotality is an assumption for partial nulls, so step-down Holm (`p_holm`) and Benjamini–Hochberg FDR (`q_bh`) are reported beside it.
     - **Training-row coupling:** fixed-model permutations keep projection matrices $V_m$ fixed; for wide views ($p_m \ge n$), training coupling is fitted to within-view noise, requiring either held-out rows or the refit null (`null="refit"`).
   - Verified that the plotting summary table documents all 6 new path model visualization symbols (`plot_path_model`, `plot_edge_matrix`, `plot_path_comparison`, `plot_edge_null`, `plot_edge_features`, `plot_edge_brain_map`).

2. **`docs/api_contract.md` ("Structural model" section):**
   - Added comprehensive result contract detailing every return key of `path_model`: `nodes`, `graph`, `directed`, `conditioning`, `rows`, `n`, `k`, `edges` (DataFrame with all 16 columns), `edge_components` (DataFrame with all 13 columns), `moderation`, `outcome_edges`, `omnibus`, `fit`, `families`, `edge_models`, `feature_contributions`, `result`, `refit_null`, `nuisance_names`, `null`, `scores_source`, `sign_rows`, `signs`, `wide_views`, `p_reason`, `parameters`, and `warnings`.
   - Detailed both comparison modes of `compare_path_models`: structural mode (`comparison`, `best`, `best_by_loglik`, `best_rule`, `nodes`, `mode`, `parameters`) and training-topology mode (`splits`, `summary`, `best`, `nodes`, `test_rows`, `mode`, `parameters`).
   - Detailed `predict_path`: `scores`, `predicted_from`, `outcome_pred`, and `r2_per_node`.

3. **`src/pysimlr/sem.py` module docstring:**
   - Updated from initial plan reference to shipped production description documenting all public functions (`path_model`, `compare_path_models`, `predict_path`, `view_scores`), statistical foundations, and usage semantics.

4. **`docs/tracking.md` (Entry D41):**
   - Recorded tracking entry **D41** in repository format:
     - Status: shipped / completed (target v0.23.0).
     - Merged components: `sem`, `plotting`, `stats`, demo script, ADNI pilot pipeline.
     - Measured numbers from WP1-WP4: fast suite counts, slow SEM suite counts, standards test passes, zero architectural linter findings.
     - Calibration numbers: fixed-model null rejection rates (0.045, 0.050, 0.035, max-$T$ 0.050); refit null rejection rates ([0.070, 0.030, 0.020], mean 0.040 within [0.02, 0.09] band); solve cost 0.81 s for $B=10\,000$.
     - ADNI-MCI pilot numbers (Rules P1-P4): P1 passed (thickness $\to$ CogPC1 significant in 4/5 SIMLR, 5/5 LEND); P2 passed (parsimonious winner `star on thickness` in 5/5 SIMLR); P3 investigated honestly (mean held-out $R^2 = 0.0873$, $r = 0.354$, Pillai $= 0.034, p < 0.001$ under strict confound sink); P4 passed (evaluated at matched B = 50, empirical p-ratio = 1.00, threshold inflation 22.4x overall, 43.8x on measurement views).
     - Key decisions: F1 parsimony rule, F2 attenuation acknowledgment, F3 confound sink composition, F4 parameter counts, F5 outcome edge handling, single score extractor refactor, removal of deprecated shims.
     - Open items: mediation on `edge_models`, NHANES run, paper chapter.
     - Updated header tracking summary block (`v0.23.0` ready).

5. **`README.md`:**
   - Added concise subsection 5 ("Edges between views: score-level path models") providing clear pointers to `pysimlr.path_model`, `pysimlr.compare_path_models`, and `pysimlr.predict_path`.

## 2. Verification commands and verbatim summary lines

- Architectural lint:
  `python -m antsxqc.lint src/pysimlr` -> `0 finding(s)`
- Implementation standards:
  `python -m pytest -q tests/test_implementation_standards.py` -> `7 passed in 3.94s`
- README code execution:
  `python -m pytest -q tests/test_readme_examples.py` -> `2 passed in 6.11s`
- Path model plotting suite (with BypassSandbox=True):
  `python -m pytest -q tests/test_plotting_sem.py` -> `10 passed, 1 warning in 6.38s`
- Dead-code and deprecated symbol check:
  `grep -rn "fit_structural_models" docs/usage_notes.md docs/api_contract.md README.md src/pysimlr/sem.py` -> `0 matches` (clean)
- Git change footprint:
  `git diff --stat` -> `5 files changed, 347 insertions(+), 12 deletions(-)`

## 3. Audit findings and integrity checks

- **Zero hardcoding or facades:** All documented keys and statistics strictly match actual code structures in `src/pysimlr/sem.py` and `src/pysimlr/plotting.py`.
- **Honest reporting of empirical limitations:** Accurately stated P3 pilot investigation results (CogPC1 $R^2 = 0.0873$ under strict confound sink), F2 attenuation floor, and max-$T$ subset pivotality caveats.
- **Cross-document consistency:** All numbers recorded across `docs/tracking.md`, `docs/usage_notes.md`, and this report directly agree with WP1, WP2, WP3, and WP4 evidence files.
