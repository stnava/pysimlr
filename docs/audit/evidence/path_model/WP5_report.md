# WP5 report: documentation, API contract, and project tracking

Date: 2026-11-07.
Plan: `docs/plans/2026-11-06_path_model.md` section 10; handoff guide `docs/plans/2026-11-06_path_model_handoff.md` section 5.
Evidence files referenced: `WP1_report.md`, `WP2_report.md`, `WP3_report.md`, `WP4_report.md`.

## 1. What was built

1. **`docs/usage_notes.md` ("Edges between views: the score-level path model"):**
   - Added section placed immediately after "New samples", matching the technical register of `feature_relevance`.
   - Workflow coverage: measurement model fitting, score projection via `view_scores`, edge statistics (`pillai`, `r`, `beta`, `rv`, `moderation`, `outcome_edges`), Freedman-Lane fixed-model null, refit null on training rows, multiplicity control (`p_perm`, `p_maxt`, `p_holm`, `q_bh`), graph-fit test (`present_pass`, `absent_pass`, `max_absent_pillai`), bootstrap and out-of-bag refit intervals, model comparison via `compare_path_models`, missing-view prediction via `predict_path`, and plotting.
   - Explicitly stated what the tests cannot say:
     - **F2 attenuation:** conditioning on estimated latent scores attenuates absent edges rather than eliminating them, so absent edges given parents err toward over-rejection under measurement error.
     - **Max-T under partial nulls:** single-step max-$T$ controls FWER under the global null; subset pivotality is an assumption for partial nulls, so step-down Holm (`p_holm`) and Benjamini-Hochberg FDR (`q_bh`) are reported beside it.
     - **Training-row coupling:** fixed-model permutations keep projection matrices $V_m$ fixed; for wide views ($p_m \ge n$), training coupling is fitted to within-view noise, requiring either held-out rows or the refit null (`null="refit"`).
   - Updated the plotting summary table with all 6 path model visualization symbols (`plot_path_model`, `plot_edge_matrix`, `plot_path_comparison`, `plot_edge_null`, `plot_edge_features`, `plot_edge_brain_map`).

2. **`docs/api_contract.md` ("Structural model" section):**
   - Documented every return key of `path_model`: `nodes`, `graph`, `directed`, `conditioning`, `rows`, `n`, `k`, `edges` (DataFrame with all 16 columns), `edge_components` (DataFrame with all 13 columns), `moderation`, `outcome_edges`, `omnibus`, `fit`, `families`, `edge_models`, `feature_contributions`, `result`, `refit_null`, `nuisance_names`, `null`, `scores_source`, `sign_rows`, `signs`, `wide_views`, `p_reason`, `parameters`, and `warnings`.
   - Documented both comparison modes of `compare_path_models`: structural mode (`comparison`, `best`, `best_by_loglik`, `best_rule`, `nodes`, `mode`, `parameters`) and training-topology mode (`splits`, `summary`, `best`, `nodes`, `test_rows`, `mode`, `parameters`).
   - Documented `predict_path`: `scores`, `predicted_from`, `outcome_pred`, and `r2_per_node`.

3. **`src/pysimlr/sem.py` module docstring:**
   - Updated from initial plan reference to shipped module documentation covering `path_model`, `compare_path_models`, `predict_path`, and `view_scores`.

4. **`docs/tracking.md` (Entries D41 and D42):**
   - Recorded tracking entry **D41** (`scripts/adni_path_model_pilot.py`, package palette ramps `VIEW_COLORS`/`_BLUE`/`_RED`/`_WARN`, Plan Section 8 primary pilot rules P1-P4 with CSV numbers and P3 view-view vs outcome-edge breakdown, live matched-$B=50$ training refit-vs-fixed thresholds, calibration rates, decisions F1-F5, and open items) and **D42** (`v0.23.1` post-audit verification and evidence tracking).

5. **`README.md`:**
   - Added subsection 5 ("Edges between views: score-level path models") pointing to `pysimlr.path_model`, `pysimlr.compare_path_models`, and `pysimlr.predict_path`.

## 2. Verification commands and verbatim summary lines

- `python -m antsxqc.lint src/pysimlr` -> `0 finding(s)`
- `python -m pytest -q tests/test_implementation_standards.py` -> `7 passed`
- `python -m pytest -q tests/test_readme_examples.py` -> `2 passed`
- `python -m pytest -q tests/test_plotting_sem.py` -> `10 passed`
- `grep -rn "fit_structural_models" docs/usage_notes.md docs/api_contract.md README.md src/pysimlr/sem.py` -> `0 matches`
