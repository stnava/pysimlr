# WP4 report: ADNI-MCI cognitive pilot evaluation

Date: 2026-11-07. Specification: `docs/plans/2026-11-06_path_model.md` section 8; `docs/plans/2026-11-06_path_model_handoff.md` section 5.
Regeneration command (live, no cached shortcuts):
`python scripts/adni_path_model_pilot.py` -> `All tasks complete in 563.15s!`
Outputs tracked under `docs/audit/evidence/path_model/adni/` (`adni_edges_<method>_seed<s>.csv`, `adni_edges_summary.csv`, `adni_train_vs_heldout_pillai.csv`, `adni_comparison.csv`, `adni_comparison_fixed.csv`, `adni_comparison_refits.csv`, `adni_p4_refit_ratio.csv`, `adni_pilot_rules.json`, 10 PNGs, `adni_pilot_report.html`).

## 1. Dataset and Experimental Setup
- Cohort (`scripts/benchmark_confound_sink.load_adni`, `data/adni/adni_mci_cognitive_merged.csv`): `n = 1,084` complete cases (`633` male, `451` female; 70/30 splits: `n_train = 759`, `n_test = 325`, seeds `0..4`).
- Views (`k = 3`): `T1Hier_vol` (`p = 160`), `T1Hier_thk` (`p = 160`), `T1w_mean` (`p = 62`).
- Outcome: `CogPC1`, first principal component of the standardized cognitive battery fitted strictly on training rows per split.
- Confounds & Groups: Age, education, and head size passed to **both** `pysimlr.fit(..., confounds=C_tr)` and `pysimlr.path_model(..., covariates=C_te)`; `groups=sex` passed to `path_model(groups=)` (never among confounds).
- Methods: `simlr` and `lend` (`random_state=s` per split; `n_permutations = 2000`, `n_bootstrap = 2000`).
- Candidate graphs: `complete` (6 edges), `star on T1Hier_thk` (3 edges), `chain (vol-thk-mean)` (3 edges), `empty` (0 edges), evaluated in Mode (a) fixed measurement and Mode (b) training-topology refits (`n_splits = 5`).

## 2. Primary Pass Rules (Plan Section 8: P1–P4)

| Rule | Plan Section 8 Criterion | SIMLR (`seeds 0..4`) | LEND (`seeds 0..4`) | Verdict |
|---|---|---|---|---|
| **P1** | `T1Hier_vol - T1Hier_thk` held-out `p_maxt < 0.05` in >= 4/5 splits | **5/5** (`p_maxt = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005]`; `pillai = [0.788, 0.771, 0.723, 0.760, 0.807]`) | **5/5** (`p_maxt = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005]`; `pillai = [0.805, 0.660, 0.717, 0.723, 0.686]`) | **PASS** |
| **P2** | Outcome node has >= 1 view with `p_holm < 0.05` in `outcome_edges` in >= 4/5 splits | **4/5** (`min p_holm = [0.0030, 0.0015, 0.0015, 0.4003, 0.0165]`) | **4/5** (`min p_holm = [0.0030, 0.0060, 0.0015, 0.0165, 0.4228]`) | **PASS** |
| **P3** | Fixed-model held-out `pillai` below training `pillai` for every edge | All 6 edges: **2/5** (seeds 3, 4). View-view edges: **5/5** (`15/15` edges). | All 6 edges: **2/5** (seeds 3, 4). View-view edges: **5/5** (`15/15` edges). | **FAIL (outcome edges only; view-view 5/5 PASS)** |
| **P4** | Complete graph is not best by `heldout_loglik` in every split (else report paired deltas) | Top by LL: `[star, star, star, star, complete]` (`complete` wins **1/5**; `star - complete` = `[+14.90, +20.77, +0.60, +10.67, -9.96]`) | Top by LL: `[complete, complete, complete, complete, star]` (`complete` wins **4/5**; `star - complete` = `[-33.48, -23.07, -68.86, -41.97, +15.32]`) | **PASS** |

### Investigation of Plan Rule P3 (`adni_train_vs_heldout_pillai.csv`)
Across all 60 `(method, seed, edge)` evaluations:
- **Measurement-fitted view-view edges (`30/30` across SIMLR and LEND)**: Held-out `pillai` is **strictly below** training `pillai` on every split (`delta_heldout_minus_train` ranges from `-0.0445` to `-0.1331` in SIMLR and `-0.0144` to `-0.1104` in LEND). Score extraction and train/test isolation have no leakage defect.
- **Unsupervised view-outcome edges (`14/30` have `pillai_heldout >= pillai_train`)**:
  - SIMLR violations (seeds 0, 1, 2): Seed 0 `vol-outcome` (`0.0340` vs `0.0286`), `thk-outcome` (`0.0518` vs `0.0340`), `mean-outcome` (`0.0548` vs `0.0311`); Seed 1 `thk-outcome` (`0.0544` vs `0.0520`), `mean-outcome` (`0.0479` vs `0.0416`); Seed 2 `vol-outcome` (`0.0538` vs `0.0472`), `thk-outcome` (`0.0682` vs `0.0434`), `mean-outcome` (`0.0295` vs `0.0256`).
  - LEND violations (seeds 0, 1, 2): Seed 0 `vol-outcome` (`0.0470` vs `0.0265`), `thk-outcome` (`0.0384` vs `0.0316`), `mean-outcome` (`0.0227` vs `0.0220`); Seed 1 `mean-outcome` (`0.0441` vs `0.0236`); Seed 2 `vol-outcome` (`0.0711` vs `0.0623`), `thk-outcome` (`0.0932` vs `0.0772`).
  - **Cause**: `pysimlr.fit` never observes `CogPC1`, so `view - outcome` edges undergo zero training-row optimization. On weak unsupervised associations (`V ~ 0.02-0.09`), held-out splits (`n_te = 325`) have both larger positive finite-sample Pillai expectation under the null (`~k/n = 3/325 = 0.0092` vs `3/759 = 0.0040` on train) and ordinary 70/30 subsampling variance (in seeds 3 and 4 the sampling fluctuation goes the opposite way and all 6 edges satisfy `pillai_heldout < pillai_train`).

## 3. Secondary Empirical Findings (Descriptive, Non-Gate)

1. **Cortical thickness to cognition (`T1Hier_thk -> CogPC1`, `adni_edges_summary.csv`)**:
   - Held-out `p_perm < 0.05` in `4/5` splits for SIMLR (`p_perm = [0.0010, 0.0005, 0.0005, 0.2239, 0.0055]`, `p_holm = [0.0030, 0.0015, 0.0015, 0.4478, 0.0165]`) and `4/5` splits for seeded LEND (`p_perm = [0.0050, 0.0020, 0.0005, 0.0055, 0.1444]`, `p_holm = [0.0100, 0.0060, 0.0015, 0.0165, 0.4228]`).
2. **Structural parsimony (`adni_comparison_fixed.csv`, `adni_comparison_refits.csv`)**:
   - Mode (a) fixed measurement: SIMLR mean `star - complete` held-out log-likelihood delta is `+7.40` (`SE = 5.45`), and F1 parsimony selects `star on T1Hier_thk` in `5/5` splits. Nonlinear LEND mean `star - complete` delta is `-30.41` (`SE = 13.72`; F1 winners: `complete` in seeds 0, 2, 3; `chain` in seed 1; `star` in seed 4).
   - Mode (b) training-topology refits (`n_splits = 5`): SIMLR `star`/`chain` achieve higher held-out reconstruction (`oos_r2 = 0.1354`, `outcome_r2 = 0.0507`) than `complete` (`oos_r2 = 0.1137`, `outcome_r2 = 0.0279`), while `complete` has higher own-graph Gaussian log-likelihood (`-2133.09` vs `-2521.70`). In LEND, `star on T1Hier_thk` leads own-graph held-out log-likelihood (`-1983.64` vs `-2444.65` for `complete`).
3. **Out-of-sample `CogPC1` prediction (`predict_path` on confound-residualized scores)**:
   - SIMLR `star on T1Hier_thk`: mean held-out `R^2 = 0.0873` (Pearson `r = 0.3501`; `complete` `R^2 = 0.0926`).
   - LEND `star on T1Hier_thk`: mean held-out `R^2 = 0.0894` (Pearson `r = 0.3543`; `complete` `R^2 = 0.0918`).
4. **Live training-row refit vs fixed-model null (`rows="train", null="both"`, Seed 0, `n_permutations = n_refits = 50`, `adni_p4_refit_ratio.csv`)**:
   - All 12 edges have `n_ge_fixed = 0` and `n_ge_refit = 0` (`p_perm = p_refit = 1/51 = 0.019608`, matched-`B` p-ratio `1.00`), and every `max_null_refit` lies strictly below observed `pillai`.
   - SIMLR view-view edges: `q95_refit` = `[0.3323, 0.2821, 0.2847]` (mean `0.2997`) vs `q95_fixed` = `[0.0074, 0.0067, 0.0068]` (mean `0.0070`) -> **`43.03x`** mean threshold shift (`44.99x`, `42.27x`, `41.82x`); outcome edges mean ratio `1.14x` (both nulls use Freedman-Lane without refits); overall 6-edge mean `22.08x`.
   - LEND view-view edges: `q95_refit` = `[0.2226, 0.0715, 0.0780]` (mean `0.1240`) vs `q95_fixed` = `[0.0078, 0.0072, 0.0083]` (mean `0.0078`) -> **`15.99x`** mean threshold shift (`28.63x`, `9.95x`, `9.38x`); outcome edges mean ratio `1.17x`; overall 6-edge mean `8.58x`.

## 4. Rendered Figures and Report (`docs/audit/evidence/path_model/adni/`)
- Path diagrams, matrix, forest comparison, null histograms, feature lollipops (200 dpi): `adni_path_diagram_simlr.png`, `adni_path_diagram_lend.png`, `adni_edge_matrix_simlr.png`, `adni_path_comparison_forest.png`, `adni_edge_null_histograms.png`, `adni_edge_features_thk_outcome.png`, `adni_edge_features_vol_thk.png`.
- `antsxviz` DKT cortical surface and hemispheric lobe tables: `adni_brain_map_thk_outcome.png`, `adni_brain_map_vol_thk.png`, `adni_region_table_thk_outcome.png`.
- HTML summary report: `adni_pilot_report.html`.
