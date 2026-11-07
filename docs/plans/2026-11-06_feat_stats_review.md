# Review & Pre-Merge Recommendations: `feat/stats` (`D40` `pysimlr.stats` + `D41` `pysimlr.sem`)

Date: 2026-11-06  
Reviewed tip: `7086a7ebfa4eb946147540d3d99374b0662b7c45` (`origin/feat/stats`, 7 commits ahead of `main` `12c51db`)  
Target release: `pysimlr v0.23.0`

---

## 1. Executive Verdict

The core library implementations in `D40` (`src/pysimlr/stats.py`, `285f238`) and `D41` (`src/pysimlr/sem.py` + `src/pysimlr/plotting.py`, `80616ec`..`7086a7e`) are **architecturally sound, statistically principled, and safe with respect to downstream callers**. Before merging `feat/stats` into `main` and tagging `v0.23.0`, **four concrete close-out items** (below in Section 3) should be resolved as part of `WP5` and release hygiene.

---

## 2. Verified Strengths (Keep As-Is)

1. **`D40` — Consolidated Inferential Core (`src/pysimlr/stats.py`):**
   - Replaces five divergent private implementations across `antsxbrainmapping`, `antsxbrainmappingprivate`, and `pysimlr` (`bh_fdr`, `holm`, `jzs_bf10_correlation`, `jzs_bf10_one_sample_t`, `permutation_correlations`, `freedman_lane_partial_p`, `pooled_correlation_fisher_z`, `stouffer_z`, `icc_3`).
   - Fixes the silent `log BF10 <= 50` ceiling (`5.2e21` cap) and `8e-4` quadrature error in the legacy 400-point log-grid by using `scipy.integrate.quad` (`7e-11` accuracy vs. 2,000,001-point reference), while retaining `jzs_bf10_correlation_grid` for exact historical reproduction.
   - Keeps `import pysimlr` free of eager `scipy` imports and restricts `_configure_matplotlib_backend()` in `src/pysimlr/__init__.py` strictly to `_PLOTTING_MODULES = ("diagrams", "plotting")`.

2. **`D41` — Method-Agnostic Score-Level Structural / Path Model (`src/pysimlr/sem.py`, `src/pysimlr/plotting.py`):**
   - Cleanly retires the deep/flow-only `fit_structural_models` (verified `0` external callers across `antsxbrainmapping`, `antsxbrainmappingprivate`, and `antsxmm`) in favor of `view_scores`, `path_model`, `compare_path_models`, `predict_path`, and six `plotting` / `antsxviz` functions (`plot_path_model`, `plot_edge_matrix`, `plot_path_comparison`, `plot_edge_null`, `plot_edge_features`, `plot_edge_brain_map`).
   - **Single canonical score extractor (`view_scores`):** Centralizes `loading_rotation`, training-`u` sign alignment, and `latents` vs. `joint_block` handling across `sem.py`, `loadings.py`, and `utils.py`.
   - **Rotation-invariant multi-block edge inference:** Uses `pillai` (mean squared canonical correlation) as the primary multi-component edge statistic alongside reference-frame partial `r`/`beta`, group interaction moderation (`Ra_j * G`), joint-sweep Freedman-Lane `p_maxt` + `p_holm` + `q_bh`, and an SEM $\chi^2$-analogue graph-fit test (`max_absent_pillai` over omitted conditional edges).
   - **First-class `D35` Confound-Sink Coupling:** `sem._refit_scores_task` permutes each target node's confound rows together with its view rows via `confounds.permute_view_confounds`, `_compare_training_topologies` slices `confounds` per split via `confounds.slice_confounds`, and training-row fixed-model p-values are refused by default (`allow_train_fixed=False`) so wide views ($p_m \ge n$) cannot manufacture significance.

---

## 3. Pre-Merge Recommendations (`WP5` & Release Checklist)

### R1. Fix Rule `P4` Interpretation & Matched-`B` Evaluation in `WP4` / `WP5`
- **Finding:** `docs/audit/evidence/path_model/WP4_report.md` and `docs/audit/evidence/path_model/adni/adni_pilot_rules.json` report `SIMLR mean_ratio = 39.235294117647065` (`LEND = 34.88`) for `p_refit / p_fixed` as empirical proof that the refit null is conservative relative to the fixed-model null.
- **Root cause:** On strong `N = 1,084` ADNI edges (`n_exceeding = 0` under both nulls), both p-values hit their add-one Monte Carlo floors (`1 / (B + 1)`):
  $$\frac{1 / (50 + 1)}{1 / (2000 + 1)} = \frac{2001}{51} = 39.235294117647065.$$
  Thus `39.24` reflects the unequal permutation budgets (`n_refits = 50` vs. `n_permutations = 2000`), **not** true null conservativeness.
- **Recommendation:**
  1. In `scripts/adni_path_model_pilot.py` (or `WP5`'s summary in `docs/tracking.md`), either evaluate `p_refit / p_fixed` at **matched `B`** (e.g., subsampling/setting `B_fixed = n_refits = 50`, where saturated edges yield `ratio = 1.0` rather than `39.24`) or report the **ratio of null standard deviations / null 95th-percentile `pillai` thresholds** ($\text{q95}(\text{pillai}_{\text{refit}}) / \text{q95}(\text{pillai}_{\text{fixed}})$), which measures true null widening without add-one floor clamping.
  2. Never cite `39.24` in `docs/tracking.md` (`D41`) or `docs/usage_notes.md` as an empirical conservativeness factor.

### R2. Complete `WP5` Documentation, API Contract & `D41` Tracking Before Tagging `v0.23.0`
`origin/feat/stats` currently stops after `WP4` (`7086a7e`). Complete `WP5` per `docs/plans/2026-11-06_path_model_handoff.md` Section 5:
1. **`docs/tracking.md`:** Add entry **`D41`** documenting `src/pysimlr/sem.py`, the 6 new `plotting` functions, the removal of `fit_structural_models` and `diagrams.plot_path_model`, verified refit-null Type-I calibration rates (`[0.070, 0.030, 0.020]` at $\alpha = 0.05$), and honest ADNI pilot metrics (`P1`–`P4` with R1's correction). Also remove the stale bullet in `D40` saying *"sem.py (untracked, in flight in another session) still calls deep_ranking.holm_adjust"* since `sem.py` now imports `bh_fdr, holm` from `.stats`.
2. **`docs/api_contract.md`:** Add the **"Structural model"** section specifying inputs, return keys, shapes, and refusal rules for `view_scores`, `path_model`, `compare_path_models`, and `predict_path`.
3. **`docs/usage_notes.md`:** Expand beyond the `+6` plotting-table lines to add **"Edges between views: the score-level path model"**, explicitly covering:
   - What fixed-model vs. refit nulls test (and why `allow_train_fixed=False` by default);
   - Measurement-error attenuation on conditional chains ($a - b - c$ under `conditioning="parents"`);
   - **Linear vs. Deep Conditional-Graph Behavior (`WP4` `P2`):** Explain why linear `simlr` favored the parsimonious `star` graph (`ΔLL = +7.40`) on ADNI while nonlinear `lend` strongly favored `complete` (`ΔLL = -57.51` across 5/5 splits) — nonlinear encoders induce higher-order cross-view score coupling that is not d-separated by linear partial residualization on intermediate scores.
4. **Release Bump:** Bump `pyproject.toml` `0.22.2 -> 0.23.0` only after full `pytest -m ""`, `tests/test_implementation_standards.py`, and `python -m antsxqc.lint .` (`0` findings) pass.

### R3. Prune Redundant Interim Binary Artifacts Under `docs/audit/evidence/path_model/` (`5.98 MB`)
- **Finding:** `docs/audit/evidence/path_model/` currently tracks `38` files totaling **`5.98 MB`**, including **four near-duplicate copies** of the planted-chain reference PNGs and base64-embedded HTML reports (`reference_*`, `wp1/`, `wp2/`, `wp3/`) plus `adni/` PNGs/HTML.
- **Recommendation:** Keep the lightweight, auditable markdown reports (`WP1_report.md`..`WP5_report.md`), `adni_pilot_rules.json` / CSV tables, and at most **one** canonical set of final reference figures (removing the duplicate interim `wp1/`, `wp2/`, `wp3/` PNGs and base64 `.html` bundles before merging to `main` so `pysimlr` git history stays lean).

### R4. Downstream Adoption Plan (`antsxbrainmapping` & `antsxbrainmappingprivate`)
Once `pysimlr v0.23.0` is released:
1. Replace private copies of BH/Holm, JZS `BF10` quadrature/grid, Freedman-Lane partial permutation, Fisher-$z$ / Stouffer pooling, and `ICC(3,1)` in `antsxbrainmapping` (`discovery_transport`, `multiview_simlr`, `triangulation_variance`, `cognitive_networks`) and `antsxbrainmappingprivate` with `pysimlr.stats`.
2. Add optional scalar Spearman $\rho$ / directional permutation helper if needed to retire `triangulation_variance._univariate_perm_boot_stats`.
3. Use `pysimlr.path_model` + `compare_path_models` + `plot_path_model` / `plot_edge_brain_map` in multi-cohort multimodal studies (e.g., testing directed biophysical cascades across structural strain, white-matter tracts, perfusion/neuroinflammation, and functional network segregation on held-out cohorts with coupled `D35` confound sinks).
