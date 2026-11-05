# Tracking Summary
current phase: `v0.19.0` released (`D33`: `component_attribution` / `plot_component_attribution`, paper Fig. 3.9 as reusable code; `D32`: coarse-to-fine split-scale `probe_views` / `probe_test_views`; `D31`: seeded orthogonal deflation ICA initialization).
next WP: pin nsa_flow 3.6.0 once released; open finding in D17 (NSA-Flow failure on k = p narrow views at lr 0.3).
blockers: none
active work: D36 (`positivity` = "linear" | "conical" | "convex", convex in every family) on branch `feat/convex-positivity`, awaiting merge; open: relevance on real wide data (IBD) untested; rendered book not yet re-rendered for the refreshed caches; adni_mci_1214_grid_sweep.csv stale for dynamic-weight rows; the owner's paper and manuscript edits from `amigo` are set aside in `git stash` (stash@{0}), not applied
device: Apple Silicon (macOS arm64), CPU for fits (MPS runs the deep models but is 1.5-5x slower at the measured sizes)
last updated: 2026-11-05

---

Entries are newest first. `D<n>` are implementer deliverables; `R<n>` is reserved for reviewer notes.
This file was created at v0.2.19; earlier history lives in git (`git log --oneline`) and in the
release commit bodies.

## D36 (positivity = "linear" | "conical" | "convex"; convex inside every family; positivity validation, 2026-11-05)
status: done on branch `feat/convex-positivity` (not merged, not released)

- **Vocabulary (`src/pysimlr/positivity.py`).** `resolve_positivity`, `is_nonneg` and `is_convex`.
  - `"linear"` and `"conical"` are exact aliases of `"either"` and `"positive"` (bit-identical fits).
  - `True`/`False` keep their meaning.
  - **Bug fixed:** unknown strings now raise `ValueError` (non-strings raise `TypeError`).
    Before, every membership test let them fall through as `"either"`, so
    `positivity="conical"` (or a typo) silently fitted a signed basis.
  - The ten duplicated tuples of non-negative values now read one `NONNEG` set.
  - `parameters["used"]["positivity"]` keeps the caller's spelling, because refits replay
    `used`. `parameters["positivity_resolved"]` holds the canonical value.
- **`"convex"`: each column satisfies `v >= 0` and `sum(v) = 1`, enforced in the optimisation
  of every family that has a basis.**
  - *Linear `simlr`, all energies.* After the conical projection (NSA retraction, quantile
    sparsity), each column goes through the Euclidean simplex projection (Duchi et al. 2008)
    restricted to its current support: idempotent, and never densifies a sparse column.
    The start point is the conical initialization rescaled to unit column sums.
  - *Deep models (LEND / NED / NEDPP) and flows* (which share the first layer through
    `_linear_encoders`). Inside `LENDNSAEncoder.v` the order is clamp → quantile →
    column floor → the same projection. It is vectorised and piecewise linear, like
    sparsemax, so gradients pass. `_finalize_bases` uses the same operator.
  - *`fit_anchored_encoder` / `extend`.* Start on the simplex, then the same projection after
    every step.
  - *Predict, missing-view predict, `feature_relevance` and `loading_inference` refits.* These
    use the stored or refitted `V` with no separate path.
  - *Refusals.* Baselines (`pca`, `nsa_pipeline`) refuse `"convex"`; `consolidate=True`
    refuses it.
  - There is no rescaling fallback at the end of a fit.
- **What convex changes mathematically.** Every energy except `recon` used to score
  `V / ||V||_2` column by column. Under `"convex"`, the simplex replaces that unit-length
  scaling: the energy sees `V` itself. Each latent is then a weighted feature average with
  variance `vᵀΣv`, which is large when the selected features move together.
  - That scale enters `nc` (normalized over the whole matrix, not per column), `acc`,
    `procrustes`, the k × k contrasts, and `recon_r2`.
  - For energies invariant column by column (`nc_abs`, `align`), convex changes the
    projected-gradient path, not the objective.
  - A unit test checks that a convex `nc` fit is not a column rescaling of the conical fit.
- **Measured behaviour, reported as behaviour, not as a gate** (the owner ships convex for all
  families).
  - Protocol:
    - ADNI-MCI Protocol B: k=3, CogPC1, random forest tuned within training rows, on the 9
      standardized `X·V` read-out features.
    - NHANES 2017-18 HDL: labs/diet/body, glucose removed, k=5, the same random-forest
      read-out.
    - Held-out cross-view coupling via `heldout_coupling`.
    - Support Jaccard: matched columns (Hungarian), all seed pairs.
    - Paired against conical on the same splits.
  - Evidence: `docs/audit/evidence/convex_pilot_{cells,paired}.csv`; scripts in
    `docs/audit/evidence/convex_pilot_scripts/`.

  Linear `simlr`, `nc`, 10 seeds. Columns: random-forest R², conical → convex (mean
  difference, seeds where convex was better) | held-out cross-view coupling difference |
  support per column | support Jaccard | share of columns with ≤ 2 features.

  | data, sparseness | RF R² | xview Δ | support | Jaccard | ≤2-feature cols |
  |---|---|---|---|---|---|
  | ADNI 0.85 | .214 → .207 (−.007, 5/10) | −.021 | 7.9 → 5.8 | .36 → .31 | .28 → .31 |
  | ADNI 0.80 | .225 → .179 (−.046, 0/10) | +.011 | 8.1 → 6.2 | .33 → .36 | .31 → .34 |
  | NHANES 0.50 (default) | .225 → .240 (+.015, 6/10) | −.004 | 6.3 → 5.7 | .64 → .63 | .37 → .35 |
  | NHANES 0.80 | .241 → .258 (+.017, 8/10) | +.010 | 5.2 → 4.7 | .66 → .63 | .35 → .36 |

  Per-seed random-forest differences, convex − conical, seeds 0–9:
  - ADNI 0.85: −.034 +.009 +.006 +.025 −.055 +.003 −.023 −.007 +.024 −.021
  - ADNI 0.80: −.032 −.034 −.014 −.023 −.076 −.031 −.043 −.045 −.097 −.069
  - NHANES 0.5: −.008 +.040 +.050 +.019 +.001 −.017 −.007 +.054 +.017 −.004
  - NHANES 0.8: −.023 +.006 +.047 +.037 +.010 +.028 −.015 +.047 +.028 +.006

  Linear (signed) against conical, for reference:
  - Sparseness 0.8: ADNI −.013 (3/10), NHANES +.007 (6/10).
  - Default sparseness: ADNI −.002, NHANES +.058 (10/10).

  Other energies, ADNI 0.85, 5 seeds, random-forest difference against conical:
  - `nc_abs`: −.001 (per seed −.006 +.004 −.018 +.043 −.027); Jaccard .32 → .25.
  - `recon_r2`: +.005 (+.004 −.029 +.024 +.003 +.021); coupling −.027; Jaccard .48 → .40.

  Deep and flow, 3 seeds, default sparseness: random-forest difference against conical (per
  seed) | coupling difference | Jaccard conical → convex.

  | model | ADNI RF Δ | ADNI xview Δ | ADNI Jaccard | NHANES RF Δ | NHANES xview Δ | NHANES Jaccard |
  |---|---|---|---|---|---|---|
  | LEND | −.030 (−.035 −.015 −.041) | +.008 | .70 → .47 | −.037 (−.009 −.064 −.038) | +.000 | .72 → .62 |
  | NED | +.002 (+.001 −.003 +.006) | −.114 | .74 → .39 | +.004 (+.025 −.003 −.011) | +.003 | .72 → .60 |
  | NEDPP | +.008 (−.020 +.008 +.035) | −.072 | .68 → .30 | +.012 (−.051 +.073 +.013) | +.016 | .73 → .59 |
  | flow_v | −.009 (−.027 +.007 −.007) | +.004 | .56 → .41 | −.009 (−.033 −.007 +.015) | −.000 | .53 → .59 |

  - **Robustness.** Every convex fit had columns ≥ 0 summing to 1 within 1e-6, with no NaN or
    dead columns.
  - **Crash found and fixed.** Before the deep column floor was added, 3 of 24 deep/flow convex
    fits raised `ValueError` because a clamp left a column with no positive entry (LEND NHANES
    seed 0; flow_v ADNI and NHANES seed 2). Reviving the column with `enforce_column_floor`
    before the projection fixed all three. Seeds that had not crashed were unchanged.
  - **Control that did not ship** (keeping the unit-length scaling inside the energy, i.e. the
    first pilot's design). Random-forest difference against conical: ADNI .85 −.002,
    NHANES .5 +.015, ADNI .8 −.038, NHANES .8 −.008. Jaccard was lower than with the shipped
    design (ADNI .28, NHANES .58 at default sparseness).
  - **Limits.** Two datasets, small seed counts (3 for deep/flow), `nc` only for deep. These are
    measurements, not a claim about other data.
- **Open (not changed here).** On the deep path, `"nonnegative"` / `"nonneg"` match no branch
  in `LENDNSAEncoder.v`: no clamp, no normalisation. This predates D36.
- **Verification:**
  - `tests/test_convex_positivity.py`: 37 tests covering the projection properties, gradcheck,
    the resolver, aliases bit-identical (linear and deep), every family on the simplex, 8 linear
    energies, predict and missing-view predict, `extend`, the `feature_relevance` and
    `loading_inference` refits, the baseline refusal, the `consolidate` refusal, and the nsa_w
    cap.
  - Full suite: 1844 passed, 2 xfailed.
  - `antsxqc.lint src tests`: 0 findings. Ruff: no new findings in changed files.

## D33 (component_attribution: paper Fig. 3.9 as reusable code, 2026-11-05)
status: done (released in v0.19.0)
- **New:** `pysimlr.component_attribution(result, train_views, test_views, outcome, outcome_test, views,
  view_names, feature_names, n_refits, refit_fraction, refit_rows, variants, refit_params, random_state,
  n_jobs)` in `src/pysimlr/attribution.py`.
  - Held-out R² of each component alone, and its unique contribution.
  - Per view: spectrum, mean correlation, consensus-vs-mean, hard assignment.
  - Refit stability of the predictive column's support, with parameter-override variants.
  - JSON-ready output.
- **New:** `plotting.plot_component_attribution(att, spectrum_view, selection_views, compare, label,
  view_labels, view_colors, titles, outcome_label, variant_labels, feature_word, top_n, ...)`.
  - It replaces about 90 lines of plotting code that were written directly in
    `paper/03_experiments.qmd`; the chunk is now a single call.
- **Paper script:** `scripts/paper_adni_attribution.py` is a thin caller. It writes `attribution` plus
  the unchanged keys `paper_numbers` reads.
  - The regenerated cache is identical to the pre-refactor one on every legacy key (difference 0.0,
    provenance aside).
  - The exactness check `component_attribution` vs the old cache gives 0.0 on full R², each
    component's R² alone and unique, the spectrum, consensus correlations, hard counts, full support,
    and all 20 + 20 refits.
  - The figure was rendered and compared side by side with the old one: same content.
- **Tests:** `tests/test_component_attribution.py`, 7 tests, plus a `test_plotting` registry entry. They
  cover: schema and JSON; R² alone and unique against a direct computation; global-factor spectrum;
  signal views; reproducible `refit_rows`; missing test views and errors (a bare `StopIteration`, caught
  by the tests, is fixed); the plot.
  - Full suite: 1800 passed, 2 xfailed.

## D32 (coarse-to-fine split-scale readout via `probe_views` / `probe_test_views`, 2026-11-04)
status: done (released in v0.18.0)
- **Problem.** `feature_maps`, `loading_inference`, `feature_relevance`, and `plotting.feature_table`
  previously required every attributed matrix to have the exact columns fitted by `result["v"][m]`.
  Multi-scale studies (`1,000+` connectome edges or white-matter tracts on `n < 150` subjects) had
  to choose between fitting directly on `p >> n` (`D25` encoder noise-fitting) or running
  `screen_wide_views` (`D30`, where broad cross-view normative factors crowd out focal pathology
  drivers under the width cap).
- **Fix.**
  - `feature_maps(..., probe_views=None)`: computes latent scores from the fitted macro `views` and
    evaluates any non-`None` `probe_views[m]` (`preprocess_data` with `result["scale_list"]`)
    strictly against the *other* views' scores (`target="other_views"` enforced).
  - `loading_inference(..., map_options={"probe_views": ...})`: fits and Procrustes-aligns each
    subsample on the well-conditioned macro `views` while computing aligned subsample CIs,
    stabilities and calibrated `q_m` selection frequencies on the matching subsample rows of
    `probe_views`.
  - `feature_relevance(..., probe_views=None, probe_test_views=None)`: computes targets, half-sample
    refits and row-permuted null refits on `train_views` / `test_views` while evaluating shrunk
    relative-weight shares, training/held-out permutation nulls and replication tiers on
    `probe_views` / `probe_test_views` (replaying training probe provenance onto held-out probe rows).
  - `plotting.feature_table`: adapts per-view feature rows automatically when a probed map's width
    differs from `result["structure"][m]`, so `plot_brain_map`, `plot_feature_map_grid`,
    `plot_feature_map_shift` and `plot_feature_map_comparison` work unchanged on fine probe views.
- **Evidence:** `tests/test_coarse_to_fine_probe_views.py` (5 tests: bit-identical `0.0` diff when
  `probe_views == train_views`; strict non-circularity under encoder perturbation; subsample
  `loading_inference` + plotting tables; wide-probe `feature_relevance` with refits, null refits and
  missing test views; and planted `44`-node -> `946`-edge connectome benchmark where coarse-to-fine
  achieves `focal_edge_auc = 1.000, r_focal = 0.953, gap = 0.060` vs `screen_wide_views`
  `auc = 0.48` and unscreened wide `gap = 0.225`).

## D31 (ICA initialization: deflation from a seeded orthogonal start, 2026-11-04)
status: done (released in v0.18.0)
- **Defect.** `initialization_type="ica"` ran FastICA (parallel algorithm) from `w_init = I_k`. For an
  orthonormal SVD basis the identity is a stationary point of FastICA's update, so it stalled or
  diverged.
  - ADNI-MCI rederivation: 63 of 383 ICA fits raised `ValueError: slice 0 has a NaN entry`, all at
    k ≥ 4, all on the 160-feature regional-volume view. Earlier I had guessed it was the 11-feature
    cognitive view; that was wrong, and the cognitive view never failed.
  - Over 360 ADNI view × split × k cases: 22 errors and 68 non-convergences.
- **Fix.** `_ica_basis_for_view` uses the deflation algorithm, starting from the QR factor of a
  seeded Gaussian, so it stays deterministic. That setting converged in all 360 cases.
  - Of the alternatives, deflation from the identity gave 82 errors; parallel from an orthogonal
    start, 75 non-convergences; `fun="exp"`, 22 errors.
  - A non-finite result now raises a `RuntimeError` naming the view.
- **Behaviour change:** ICA starts differ for every user of `initialization_type="ica"`.
- **Verification:**
  - The four ADNI configurations that failed now converge (inner CogPC1 R² 0.168–0.237).
  - `tests/test_ica_initialization.py`, 7 tests: five synthetic heavy-tailed cases that break the old
    setting; determinism; a fit; a 20-seed × 4-k sweep with no non-convergence.
  - Full suite: 1791 passed, 2 xfailed.

## D30 (D27 Part B screen: pooled BH gate; measured, opt-in, not a cure, 2026-11-04)
status: done (released in v0.17.0)
- **Gate.** The per-feature cutoff is Benjamini–Hochberg over a pooled permutation null.
  - The pooled null has B × p_live draws, so p resolves to 1 / (B·p). Per-feature counts stop at
    1 / (B + 1) and could never pass BH over thousands of features.
  - `summary` reports `n_pass_bh` and `n_pass_gate` beside `p_out`, so a binding width cap is
    visible. The output gains `q_bh`.
- **Evidence:** `scripts/screen_wide_views_check.py` → `docs/audit/evidence/screen_wide_views_check.csv`.
  Rules were fixed before running.
  - **IBD** (5 splits, n_train = 165). The gate passes almost every feature, because the platforms
    share common factors, so the cap selects.
    - Gap (training − held-out cross-view coupling): 0.40 / 0.33 / 0.19 / 0.13 / 0.02 at
      `max_width_ratio` 0.5 / 0.3 / 0.2 / 0.1 / 0.05, against 0.39 for top-500 variance.
    - SiMLR balanced accuracy: best 0.759 (ratio 0.1), against 0.732 for top-500.
  - **Noise** (3 × 120, n_train = 50). The gate keeps the floor of 7 features per view, yet the gap
    is 0.35 (0.85 unscreened).
  - **Default rule:** no ratio qualifies. **Merge rule:** FAIL on all three criteria.
- **Decision (per the approved plan):**
  - the screen stays opt-in, its default unchanged (0.75), with the limitation in the module
    docstring;
  - `WideViewWarning` is not pointed at it;
  - judge fits with `heldout_coupling`.
- **Finding:** part of the overfit is fitting k components to few rows, not view width alone. A
  7-feature noise view still overfits at n = 50, k = 5.
- **Rerun after merging the branch's view-level mean/tail/max omnibus with Holm** (same script and CSV):
  - IBD gap 0.39 at the default 0.75, against 0.39 for top-500; SiMLR accuracy 0.736 against 0.732;
  - noise gap 0.34;
  - the default and merge rules still fail, so the screen stays opt-in.
- **Docs:** usage notes, "Wide-view workflow: opt-in screening before `fit`, shrinkage after `fit`".

## D29 (review fixes to D27 Part A: shrinkage on record, two named LW definitions, 2026-11-04)
status: done (released in v0.17.0)
- **λ recorded.**
  - `feature_relevance` returns `shrinkage_used` and `shrinkage_used_test`, plus `summary.shrinkage`
    and `summary.shrinkage_test`.
  - `feature_maps(..., return_shrinkage=True)` returns λ per target and view; the default return is
    unchanged.
  - `loading_inference` stores `maps["relative_weight"]["rw_shrinkage"]`.
  - `"auto"` therefore never switches estimator off the record.
- **Two definitions, named apart.**
  - `"ledoit_wolf"` gives λ from X alone (relevance; the permutation null is exact, because λ cannot
    move when t is permuted).
  - `"ledoit_wolf_joint"` gives λ_j from [X, t_j] (feature maps; exact shrunk Schur R²).
  - `"auto"` resolves to the first in relevance and the second in feature maps. The joint form is
    refused by the relevance basis.
- **Held-out uses the training λ.** It falls back to `"auto"` only where the classical estimator is
  undefined on the held-out rows (p ≥ n_test), and that fallback is recorded.
- **Gates:**
  - full suite: 1778 passed, 2 xfailed;
  - ADNI relevance equivalence: 222/222 features and 93/93 groups, tiers identical (max |Δ| 2e-10);
  - 3 new tests (λ is target-independent; the λ record from `feature_maps`; the recorded held-out
    fallback) and assertions added to the wide-view relevance test.
- **Not changed:** Part B (`screening.py`). On IBD its default cap of 0.75·n binds every view at 123
  features, and the gap falls only from 0.39 to 0.30. Part B is revised separately (D30).

## D28 (heldout_coupling: cross-view agreement on held-out rows, 2026-11-04)
status: done (released in v0.17.0)
- `pysimlr.heldout_coupling(result, views)` gives, for training and held-out rows over the available
  views:
  - per view and component, |corr(z, u)|;
  - per component, the mean pairwise cross-view |corr|, and the training − held-out gap.
- Joint baselines raise.
- `WideViewWarning` now names it.
- Validated:
  - IBD 1.00 / 0.58;
  - noise views with p ≥ n 0.88–0.91 / 0.04–0.05;
  - genuine signal gap ≤ 0.03.
- Tests: `tests/test_heldout_coupling.py`, 6 tests.
## D27 (shrunk Johnson relative weights for p >= n and outcome-blind wide-view predictor gating, 2026-11-04)
status: done (released in v0.17.0)
- **Problem.**
  - Classical Johnson (2000) `relative_weight` in `loadings.py` and `relevance.py` required full-rank
    `p < n` (and `p < n_sub` during half-sample `loading_inference` / `feature_relevance` refits),
    emitting NaNs via `rcond=1e-10` eigenvalue gating whenever `p >= n` or `p >= n_sub`.
  - Simultaneously, `D25` (`WideViewWarning`) showed that feeding unconstrained wide views (`p >= n`)
    directly into `simlr` lets the solver couple within-view noise to consensus (`coupling_train` up to
    0.88–1.00 vs held-out < 0.20), creating a need for strictly outcome-blind cross-view predictor gating.
- **Part A — Joint Ledoit–Wolf Schur R² relative weights (`loadings.py`, `relevance.py`).**
  - Added `_resolve_rw_shrinkage` and `_basis_from_svd` in `src/pysimlr/loadings.py`, upgrading
    `_relative_weight_map`, `_relative_weight_basis`, `_relative_weight_shares`, and `feature_maps(...,
    rw_shrinkage=0.0)` (`"auto"`, `"ledoit_wolf"`, float in `[0, 1)`, or per-view sequence), and wired
    `loading_inference` (`map_options={"rw_shrinkage": "auto"}`) plus `feature_relevance(...,
    shrinkage="auto")` in `src/pysimlr/relevance.py`.
  - Shrinks the joint `[X, t_j]` correlation matrix with a single Ledoit–Wolf intensity `lambda_j` to `I`,
    so `sum_f w_f = (1 - lambda_j)^2 r_Xt^T R_lambda^-1 r_Xt = 1 - 1 / (S_joint,lambda^-1)_tt` lies
    strictly in `[0, 1)` to machine precision (`~1e-16`) via thin SVD (`O(min(n, p)^2 p)`), while keeping
    `rw_shrinkage=0.0` bit-identical (`0.0` diff) on full-rank `p < n` views.
- **Part B — Outcome-blind cross-view predictor gating (`screening.py`, `__init__.py`).**
  - Added `pysimlr.screen_wide_views` and `pysimlr.apply_view_screen` in `src/pysimlr/screening.py`
    (with no outcome parameter in the signature, preventing target leakage by construction).
  - Combines row-permuted cross-view subspace communality (`p_perm`), exact finite-sample Westfall–Young
    min-`p` omnibus view calibration across mean, sparse-tail and max communality with Holm–Bonferroni
    step-down across views (`p_view`), half-sample stability selection (`selection_frequency`), greedy
    within-view collinearity deduplication (`max_collinearity`), and sample-adaptive width control
    (`max_width_ratio`).
- **Tests:** `tests/test_shrunk_relative_weight_and_screening.py` (5 tests: exact Schur `R²_LW` identity
  to `1e-12`, bit-identical `0.0` equivalence on `p < n`, wide-view `feature_maps` + half-sample
  `loading_inference` with `p > n_sub`, wide-view `feature_relevance` at `p = 200 > n_train = 80`, and
  `screen_wide_views` Type I suppression on pure-noise `D25` views + sparse driver recovery + collinear
  pruning + `apply_view_screen`). Verified alongside `test_feature_maps.py`, `test_feature_relevance.py`,
  `test_wide_view_warning.py`, `test_latent_scale.py`, and `test_implementation_standards.py` (47 passed,
  0 lint findings).
- **Docs:** usage notes, "Wide-view workflow: screening before `fit`, shrinkage after `fit`".

## D26 (linear SiMLR latents at the scale of u, 2026-11-04)
status: done (released in v0.16.0; SiMLR cache rows refreshed in 4110375)
- **Defect.** `fit` and `predict` returned linear-SiMLR `latents` at the preprocessed-view scale.
  - That scale is about 1e-3 the size of `u` (SD about 4–6e-4 on a 2-component example), and it
    shrinks as the view grows.
  - A downstream estimator with a fixed penalty shrank them to nothing. A default `RidgeCV` read-out
    gave R² ≈ 0 on every NHANES outcome until its features were standardised.
- **Fix.** `simlr._latent_scales` computes one divisor per view, stored as `result["latent_scale"]`
  and applied by `predict_simlr`. Each view then has unit average column variance on the training
  rows, keeping the relative size of its components.
  - It is one constant per view, not per column, so every internal use is unchanged: correlations,
    z-scoring, and the GCV ridge in `extension.consensus_maps` (penalty relative to the data's own
    scale).
  - Subset prediction gives the same `u` to 1e-6 (test).
  - Results without `latent_scale` keep the old scale.
  - Deep and flow latents were already of order 1 (0.7–2.2) and are untouched.
- **Effect on the paper cache** (`deep_ranking_benchmark.csv`, seed 42, rerun at this commit).
  - SiMLR rows whose metric fits a penalised estimator to latents moved:
    - ADNI-MCI predictive 0.1254 → 0.1325;
    - TCGA-KIRC strictly-linear 0.6170 → 0.6241;
    - Linear strictly-linear 0.9727 → 0.9724;
    - NonnegParts and NonnegPartsWeak ≤ 1e-5.
  - Every other model, and SiMLR's Heart and mfeat rows, are bit-identical.
  - The cached SiMLR first-layer numbers were biased low by up to about 0.007. Refresh the SiMLR
    rows before quoting them.
- **Cache refresh.** `scripts/refresh_simlr_cache_rows.py` reran the 135 linear-SiMLR tasks of
  `deep_ranking_benchmark{,_q05,_q_none,_q08}.csv` and `operating_points_05_08_benchmark.csv` with
  their exact original task tuples, and spliced the rows in place.
  - Provenance, including the largest change per metric and file, is in
    `paper/results_cache/provenance_simlr_refresh_D26.json`.
  - Attribution: the pre-D26 commit `a51aa66` reproduces the old rows exactly (difference 0) on the
    three largest movers: ADNI-MCI q-none seed 42, ADNI-MCI q0.8 seed 44, TCGA-KIRC q-none seed 43.
    So every change is D26's.
  - Size of the changes:
    - strictly-linear read-outs moved ≤ 0.014;
    - ADNI-MCI, the only case whose `u` is built from aligned latents: predictive −0.015 to +0.027,
      training +0.05 to +0.15.
  - Of the 465 manuscript numbers in `paper_numbers`, 25 changed, all computed inline (no hard-coded
    copies in the prose). Examples:
    - `adni_SiMLR_u` 0.143 → 0.146;
    - Friedman linear-probe p 1.80e-2 → 1.50e-2;
    - SiMLR linear rank 4.04 → 4.21;
    - `sp_best_model` SiMLR → Flow-SiMR-V, a near-tie at an unchanged value.
- **Not caused by D26 (open):** `adni_mci_1214_grid_sweep.csv` (written 2026-11-03) is stale for
  dynamic-weight rows.
  - SiMLR-Lookahead with MAI `cca`/`rvcoef` reruns deterministically to a different value
    (≤ 0.012, including the `v`-based RidgeCV and CogPC1 columns).
  - The likely cause is `a32ad62` (MAI weight bounds, v0.13.0).
  - Static-weight rows reproduce exactly.
  - The axis-alignment controls read `res["v"]` and are unaffected.
- **Tests:** `tests/test_latent_scale.py`, 4 tests: unit average variance and fit == predict;
  divisor round-trip; subset `u` unchanged; old results keep their scale. Full suite: 1765 passed,
  2 xfailed (before the new tests).

## D25 (wide views: training coupling reported and warned, 2026-11-04)
status: done (released in v0.16.0)
- **Defect.** A view with p ≥ n can couple to noise during training, and nothing in the result said
  so. Found while screening public data.
  - IBD metabolome + microbiome (p = 500 per platform, n = 165 training subjects): training
    cross-view coupling was 1.00 against 0.58 held-out.
  - Raising sparseness to 0.95 still left 0.83 against 0.61: the support is chosen from all p
    features.
  - The exact rank condition (active features ≥ n − 1) is too narrow; it fired only at
    sparseness 0.5.
- **Fix.**
  - New `utils.coupling_capacity_report`, which `simlr` now calls: `coupling_train`,
    `view_width_ratio`, `wide_views` in every result.
  - New `WideViewWarning`, which names each wide view with its p and training coupling and points
    to the held-out check.
  - `simlr` warns once per user-facing fit; learning-rate probes don't warn.
  - Nothing changes the fit.
- **Tests:** `tests/test_wide_view_warning.py`, 5 tests.
  - Narrow views don't warn.
  - The wide view is named.
  - `coupling_train` equals the score–consensus correlation.
  - Noise views with p ≥ n: training coupling > 0.7 (measured 0.88–0.89 on 4 seeds) and held-out
    < 0.2 (measured 0.03–0.05).
  - The class is separate from `ConvergenceWarning`.
- **Docs:** usage notes, "Wide views: training coupling is not evidence".

## D24 (feature_relevance: the relevance cutoff as a pysimlr feature, 2026-11-04)
status: done (released in v0.15.0; merge 2db6c71)
- **New:** `pysimlr.feature_relevance(result, train_views, test_views, outcome, outcome_test,
  component, views, feature_names, view_names, groups, min_group_size, alpha, n_permutations,
  n_permutations_test, n_refits, refit_permutations, refit_fraction, n_null_refits, random_state,
  n_jobs)` in `src/pysimlr/relevance.py`.
  - It covers the relative-weight share test against the fair share, held-out replication, refit
    stability, the optional whole-model refit null, group tests and tiers 0 / 1 / 2.
  - `loadings._relative_weight_map` is split into `_relative_weight_basis` and
    `_relative_weight_shares`, which are shared with the permutation loop. The 15 feature-map
    tests are unchanged and pass.
  - A test that was not run is NaN, never a number.
- **Plots:**
  - `plotting.plot_feature_relevance(rel, view, label, separate, grouping, grouping_title,
    brain, region_names, meshes, show_all, r_label, ...)` shows the forest, the group panel, the
    separate panel and an optional brain row.
  - `plot_brain_map(..., tiers=rel)` paints the tiers with categorical colourbar ticks.
  - The region mapping is factored into `_region_index`, shared with `brain_rows`.
- **ADNI:** the two scripts are thin callers.
  - Equivalence with the original scripts: 222/222 features and 93/93 groups (max |diff| ~1e-16),
    identical tiers.
  - The figure regenerates in 18.3 s, including the fit.
  - The forest now takes the view colour (orange for vol/area), as every other pysimlr plot does.
- **Finding:** the share test measures concentration, not strength.
  - The null share is about Beta(½, (p − 1)/2), so p is bounded by how evenly a signal spreads.
    Four equal drivers among 24 features reach p ≈ 0.01 however strong.
  - This is documented in the module docstring and in the usage notes.
- **Tests:** `tests/test_feature_relevance.py`, 11 tests:
  - vectorised p equals a loop;
  - planted drivers reach tier 2 (checked on 7 seeds);
  - a null view gives about α false positives;
  - groups, no test views, missing view at test time, reproducibility of refits and null refits;
  - errors, plots and tier rows.
- **Change request:** antsxviz `surface_figure` has no colourbar ticks key; pysimlr relabels the
  tier bar afterwards (`docs/change_requests/antsxviz.md`).

## D23 (antsxviz integration: brain maps via the ANTsX house renderer, 2026-11-04)
status: done (branch `feat/antsxviz-integration`)
- **New:** `pysimlr.plot_brain_map(res, inf, maps, component, view, kinds, feature_names, view_names,
  region_names, title, subtitle, notes, meshes, density, surface)`. It draws the fsaverage
  cortex through antsxviz 0.1.0 (`load_fsaverage`, `surface_figure`).
  - Optional extra: `pip install 'pysimlr[viz]'`. antsxviz is imported only when called; if it's
    missing, the ImportError names the extra. There is no fallback renderer.
  - `brain_rows` adapts `feature_map_values` to antsxviz rows. `cortical_region_name` maps
    `..._dktcortex` / `..._dktregions` names to DKT regions; the dktcortex variant wins.
    Non-cortical features are returned and listed in the subtitle, never silently dropped.
  - Kept in pysimlr: `feature_map_values` and `map_colormap`.
- **`plot_feature_map_grid` moved to antsxviz** (owner: "if it is general enough, it belongs in
  antsxviz").
  - It was general enough: antsxviz's `region_table_figure` passes non-DKT feature names through
    unchanged and takes `split=None` and arbitrary `groups`.
  - It produced the same table as pysimlr's on the gallery's synthetic cortex: values, colours,
    rings, lobe headers and stability.
  - pysimlr's grid is now a 15-line wrapper: `feature_map_values` → `rows_from_encoding` →
    `region_table_figure`. 77 lines of drawing code were removed. It needs the `viz` extra and
    raises an ImportError naming it.
  - This resolves the TODO(human) in antsxviz's `docs/change_requests/pysimlr.md`.
  - Palettes were checked identical: VIEW_COLORS and all four `map_colormap` ramps, max diff 0.0.
- **Scripts:**
  - `scripts/brain_region_maps.py` is now a thin antsxviz caller. About 370 lines of GIFTI,
    label-transfer, mesh and render code were deleted.
  - `adni_relevance_figure.py` and `adni_relevance_test.py` use antsxviz (`render_surface`,
    `dkt_id`, `DKT_LOBES`). Lobe assignment is unchanged for all 222 ADNI features.
- **antsxviz's audit settles the L/R question:**
  - only `CIT168_..._dkt.nii.gz` is mirrored; the CIT168 T1 isn't (SyN CC 0.802 vs 0.798
    mirrored);
  - per-subject ADNI DKT labels are correct.

  The relevance caption now says so.
- **Found and reported, not fixed (external):** antsxviz's fsaverage annotation leaves lateral
  parietal holes (1055 / 520 vertices; `max_hole=400`), which render grey in lateral views. See
  `docs/change_requests/antsxviz.md`.
- **Release v0.14.0:** the `viz` extra requires `antsxviz>=0.1.1`. That version fills the lateral
  parietal holes (91.6% of vertices labelled) and fetches its annotation through antsxdata
  v0.4.0 (Figshare 10.6084/m9.figshare.34917999). `test_palette_matches_antsxviz` guards
  pysimlr's palette against drift. ADNI figures were re-rendered and inspected; the holes are
  gone. Full suite: 1774 passed, 3 xfailed; `antsxqc.lint` 0 findings.
- **Tests:** `tests/test_brain_maps.py`, 4 tests: name mapping; the adapter against a stand-in
  antsxviz; the missing-extra ImportError; and an end-to-end render, skipped without
  antsxviz/assets, 3.4 s.
- **Figures regenerated and inspected:** `adni_dkt_feature_maps.png`, `_grid.png` and
  `adni_volarea_relevance.png`.

## D22 (Positive-sparse polarity coupling & DOF/OOF-calibrated subset consensus, 2026-11-04)
status: done on branch `fix/pos-sparse-calibrated-consensus`
- **Column-sign-invariant correlation energy `energy_nc_abs` & one-time `"nc"` + positive warning (`similarity.py`):**
  - Added smooth, degree-0 scale-free, centring-invariant, column-sign-invariant `energy_nc_abs` (`-mean_j(sqrt(r_j^2 + tau^2) - tau)`) and `SIGN_INVARIANT_ENERGIES` (`acc`, `exp`, `kurtosis`, `logcosh`, `nc_abs`, `negentropy`, `recon_r2`).
  - Kept `DEFAULT_ENERGY_TYPE = "nc"` (explicit opt-in for `"nc_abs"`), emitting a one-time `UserWarning` via `check_fit_energy` when signed `"nc"` is combined with non-negative `positivity` (`"positive"`, etc.).
- **Polarity-oriented OLEE loadings (`loadings.py`, `api.py`):**
  - Added `view_coupling_signs` (`result["coupling_signs"]`, `+1.0` or `-1.0` per view and component). Under non-negative `positivity` and `SIGN_INVARIANT_ENERGIES`, structure columns are oriented by active-feature third-moment sign before `olee_loadings`, preserving `loadings >= 0` with disjoint column supports when a view couples negatively to a shared axis; signed energies remain bit-identical (`+1.0`).
- **DOF/OOF-calibrated per-component `subset_consensus` & pole-oriented anchored extension (`extension.py`):**
  - Replaced raw OLS and uncalibrated scalar `R^2 / (1 - R^2)` weights in `_affine_map` / `subset_consensus` with GCV-stabilized ridge maps `_gcv_ridge_solve` and default per-component `(k,)` precision weights driven by `consensus.mai_dof_adjust` (with smooth `(n - 1) / (n - 1 + dof)` shrinkage when `dof` approaches or exceeds `n - 1`) or extension fold `oof_r2`.
  - Updated `fit_anchored_encoder` step-0 initialization under non-negative `positivity` + `SIGN_INVARIANT_ENERGIES` to select the higher-variance `(X^T t)_+` vs `(-X^T t)_+` pole, and added stratified 5-fold OOF (`oof_r2`, `coupling`) in `_anchored_holdout` when `15 <= n < 60`.
- **Verification:** `tests/test_pos_sparse_calibrated_consensus.py` (4 tests) + `tests/test_similarity_registry.py` (`nc_abs` + sign-invariance registry check) + `tests/test_extend.py` + `tests/test_olee_contract.py` + `tests/test_implementation_standards.py`; `antsxqc.lint src tests` 0 findings; 4-cohort blast benchmark confirmed `ExpArt` `ASL_Perfusion` consensus share rose `1.0% -> ~48%` (`DTI + rsfMRI_Conn` `98.4% -> <40%`) and zero-shot `Career Breachers (N=31)` `Consensus U` AUC rose to `0.78-0.80` (`U + Z_m` up to `0.874`).
- Renumbered D20 -> D22 on merge into `main` (D20/D21 were taken by v0.13.0 work).

## D21 (alternative feature maps: partial / inverse correlation, relative weights, 2026-11-04)
status: done (branch `feat/feature-maps`)
- **New:** `pysimlr.feature_maps(res, views, kinds, target="other_views", shrinkage="ledoit_wolf",
  control="pc1")`, with four maps beside `structure`, all in the fit's rotated frame. Nothing
  existing changes.
  - `partial`: partial correlation given the other features, from the inverse of the Ledoit–Wolf
    correlation matrix.
  - `relative_weight`: Johnson 2000. Shares of R² that sum to R². NaN with a warning when p ≥ n.
  - `global_partial`: structure after removing PC1, the row mean or a covariate; `control` is set
    per view.
  - `component_partial`: the Haufe forward pattern; equals `structure` when components are
    uncorrelated.
- **Target.** By default the maps are computed against the *other* views' scores. Regressing a
  linear fit's own score on X would return the unstable encoder.
- **Inference.** `loading_inference(maps=..., map_options=...)` computes the maps on every
  subsample refit at no extra fits.
  - Refits are matched by the signed permutation nearest the structure Procrustes rotation;
    `relative_weight` takes no sign.
  - Output: per-map intervals, selection frequency and stability.
- **Display.**
  - `feature_table(maps=)`.
  - `plot_top_features(kind=...)`: relative weights in percent; filled means selected in ≥ 80% of
    refits.
  - `plot_feature_map_comparison`: per-feature selection fill; it drops component_partial when
    that equals structure.
  - `plot_feature_map_shift`: rank change (gallery figure 10).
  - `plot_feature_map_grid`: the tabular twin of a brain map (gallery figure 11).
  - All of these share `feature_map_values` and `map_colormap`.
- **Brain rendering:** `scripts/brain_region_maps.py`, on the fsaverage cortex.
  - **Surface:** TemplateFlow 41k, inflated by Taubin smoothing of the white surface, with
    sulcal-depth shading. GIFTI is read by a small XML parser, not nibabel. RAS → LPS
    conversion verified: 100% of mid-thickness vertices fall inside the brain mask.
  - **Labels:** CIT168 DKT registered to the fsaverage T1 with `syntx.registration` (SyN,
    335 s; cached in ~/.cache/pysimlr). Each vertex takes the nearest label within 3 mm; then
    small holes are filled and only the largest piece of each region is kept.
    - Coverage: 89–91% of vertices; the rest is the medial wall.
    - Centroids are anatomically placed (entorhinal: medial-inferior temporal; precentral:
      central strip).
  - **Replaced the first version,** which meshed the CIT168 label volume. The owner judged it
    "aesthetically displeasing and anatomically incorrect". Defects fixed on the way: the
    template's x reversal, the open medial wall, scalar interpolation, the cerebellum blob,
    unlabelled sulci, and a long note shrinking its panel.
  - **Open (upstream, not fixed):** in `CIT168_T1w_700um_pad_adni_dkt.nii.gz` every "left"
    DKT label (1xxx) lies at physical −x, which is anatomical right in LPS. After registration
    to fsaverage, all 1xxx labels land on fsaverage's right hemisphere.
    - Either the template header or its labels are mirrored. That is not resolved.
    - Nor is it checked whether per-subject ADNI DKT values inherit the problem.
    - The renderer takes the hemisphere from the surface file, not from the label code.
    - Owner: ANTsPyT1w.
- **Relevance cutoff (owner: "there is no hard cutoff, we cannot tell what is relevant").**
  `scripts/adni_relevance_test.py` → `docs/audit/evidence/relevance/{features,groups}.csv`;
  figure in `scripts/adni_relevance_figure.py` → `docs/figures/adni_volarea_relevance.png`.
  - **Statistic:** each feature's (and each bilateral pair's and DKT lobe's) share of the
    explained variance of the predictive component's MRI-only target, from relative weights.
    The fair share is 1/p.
  - **Four tests:**
    - training permutations (100k);
    - held-out replication on 326 subjects (50k permutations);
    - 20 half-sample refits (20k each);
    - a full-refit null: the view permuted against the others and the model refitted
      (300 per view, so p ≥ 0.0033).
  - **Vol/area:** 24 of 160 features at training p < 0.05, 24 held-out, 22 under the
    full-refit null, against 8 by chance; 18 replicate.
    - Lobes: temporal 1.88× (p < 0.001 / 0.007 / 0.003), parietal 1.79× (0.003 / 0.002 /
      0.007), frontal 1.56×, occipital 0.64×.
    - Bilateral superior temporal: 2.8×, p 0.001 / 0.001.
    - Entorhinal has the largest share, but is borderline held-out (p 0.063) and passes in
      30–50% of refits.
  - **DiReCT:** 0 of 62 in training, 1 held-out (chance 3.1); no lobe passes. Its signal is
    the global thickness factor.
  - **Figure rule:** tiers, dark = training and held-out p < 0.05; pale = training only; grey
    otherwise. L/R labels provisional pending the CIT168 investigation.
  - **Fast path:** `python scripts/adni_relevance_test.py --figure` runs tests and figure in
    25 s, against about 400 s before.
    - Where the time went: profiling showed it was all model refits. Mesh build took 2.9 s, each
      render 0.5 s, and permutations were negligible.
    - Defaults are now `--refits 10 --full-null 0`.
    - The full-refit null (300 per view) reproduced the permutation results (22 vs 24 features;
      the same lobe conclusions) and doesn't enter the tiers, so it is opt-in.
    - Permutations are vectorised with `Generator.permuted`. Verified: this draws the same
      shuffles as the per-column loop, so p-values and all 222 tiers are bit-identical to the
      full run.
    - A null that wasn't run is reported as NaN and left off the figure, never shown as a
      number.
- **Owner review:** "the plots should be the same whether on brain or not." The grid and brain
  views now share one encoding.
- **Rule** (pre-set: AUC ≥ structure + 0.05 in ≥ 4/5 global-factor seeds; ≤ 0.02 worse without a
  global factor; stability ≥ 0.8). Global-factor design PC1 = 70%; support AUC against
  structure's 0.79:
  - `relative_weight`: 0.97 (4/5 wins, −0.007, stability 0.92), **RECOMMEND**;
  - `partial`: 0.96 (4/5, −0.007, 0.88), **RECOMMEND**;
  - `global_partial`: 0.95, but −0.10 without a global factor, so available only;
  - `component_partial`: 0.81, available only.
- **ADNI** (Figure 3.9 fit, 20 half-sample refits; component 2 stability):
  - **DiReCT:** `relative_weight` 0.86 (L entorhinal, L inferior parietal, R entorhinal,
    L fusiform) against structure 0.93.
  - **Vol/area:** `partial` 0.55 and `relative_weight` 0.71 against structure 0.99. These maps
    rank CSF volume first.
  - **Held-out R² of the top 10:** the alternatives gain on DiReCT (0.17 vs 0.15 in half-sample
    refits) and on the full vol/area fit (0.24 vs 0.17), but not in vol/area half-sample refits
    (0.16 vs 0.17).
- **Pilot first, at the owner's request:** one fit, one split (`pilot_maps.png`); then the
  20-refit evidence.
- **Anomaly scan:** no NaN AUCs and no unconverged fits. Two exact-repeat R² values come from
  structure and component_partial choosing identical top-10 sets, which is expected.
- **Verification:** see the commit body.

## D20 (MAI weight bounds, ADNI Figure 3.9 corrected, open items closed, 2026-11-04)
status: done (`a32ad62` code, `e9bf886` paper and docs, `8a5d631` LaTeX source; released in v0.13.0)
- **MAI weights bounded to [0.01, 0.5]** (`consensus.MAI_WEIGHT_BOUNDS`, capped-simplex
  projection; replaces `MAI_MIN_SHARE`). No view can take more than half the consensus, and a
  trash view keeps 0.01. A `RuntimeWarning` fires when fewer than two views clear their null floor.
  - Rule, all four criteria pass: planted R² 0.908 (≥ 0.879); max weight ≤ 0.5; noise view at
    0.010 on 4/5 seeds (0.068 on the seed where its MAI cleared the floor); ADNI failing class
    324/324 ok, paired CogPC1 difference −0.0008 (`mai_calibration_check.csv`,
    `adni_mci_mai_regression_check.csv`).
- **Figure 3.9 was wrong.** It showed hard-assigned (argmax) loadings of DiReCT thickness:
  - DiReCT thickness has one global factor (PC1 58% of variance; mean inter-region r 0.55);
  - consensus C2 tracks it (r 0.87 with mean thickness), so argmax put 2 / 60 / 0 regions on
    C1 / C2 / C3;
  - the text's "about 9 regions per component" was false.
  - **The new figure** (`scripts/paper_adni_attribution.py` → `adni_attribution.json`):
    - C2 carries the held-out CogPC1 in every MRI view; vol/area C2 has R² 0.211 alone and
      unique R² +0.054;
    - its features replicate across 20 half-samples: entorhinal, temporal lobe, inferior
      parietal at 0.75.
- **Item 4: collapse penalty vs global-factor views**
  (`scripts/adni_global_factor_check.py` → `adni_global_factor_check.csv`, 10 half-samples).
  - **`collapse_weight=0` passes:**
    - the predictive DiReCT column kept ≥ 5 regions in 9/10 refits (default: 6/10);
    - 3 regions were selected in ≥ 50% of refits (default: 1);
    - CogPC1 R² was 0.224 vs 0.210.
  - **Regressing out mean thickness fails:** the signal goes with the global factor.
  - **Package default unchanged** (evidence from one dataset). Usage notes say to set
    `collapse_weight=0` before reading regional loadings on such views.
- **Paper:**
  - the ADNI views are renamed (vol/area is volume ÷ surface area, not thickness; DiReCT is
    cortical thickness);
  - abstract, intro, Chapter 3 item 2, discussion and appendix rewritten from the attribution
    numbers (`attr_*` in `paper_numbers.py`);
  - section title no longer claims Braak mapping;
  - sensorimotor-weight sentence removed (it read the global factor).
- **Row-mean ablation bug:** `protocol_b(row_center=True)` also row-centred the 11 cognitive
  scales, removing overall severity from the outcome.
  - Fixed: the MRI views only.
  - Rerun: column centring 0.196, row centring 0.162 (PCA start) and 0.193 (ICA start).
  - All ablation caches regenerated at `a32ad62`; JSON caches carry provenance, including the
    git commit.
- **Extended views in plots:** `extend` stores training latents, so the chord and radar plots
  show the new views (`test_plots_show_extended_views`).
- **Decisions on open items:**
  - **Deep-model extend default stays `anchored`,** with the D19 caveat in the usage notes. The
    ned rule failure came from a weak frozen model, and that is the user's choice to refit.
  - **`learning_rate="auto"` stays** (D17: fixed 0.3 matched it only on planted cells).
  - **nsa_flow 3.6.0** is installed but not on PyPI, so it can't be pinned.
  - **MPS polar factor** (`scripts/mps_polar_timing.py` → `mps_polar_timing.csv`, n = 2000,
    20 epochs, 2 repeats). `nsa_flow.polar_factor` in place of the SVD round-trip:
    - speedups ned 6%, lend 0%;
    - ned latents agree to min |r| 0.927;
    - rule (≥ 15% faster and ≥ 0.99 agreement): **KEEP SVD.**
- **Verification:** 1744 passed, 3 xfailed; `antsxqc.lint` 0 findings; the manuscript is
  re-rendered with `make paper`.

## D19 (pysimlr.extend: adding unseen modalities to a fitted model, 2026-11-03)
status: done (`2582c7c`; released in v0.12.0)
- **`pysimlr.extend(result, views, new_views, how="anchored" | "reembed")`,** for any method.
  - **anchored:**
    - the model is frozen; each new view gets a linear encoder fitted to the model's consensus
      under the fit's energy and constraints;
    - its `structure`, `loadings` and `loading_margin` are in the original frame;
    - a held-out coupling warning (Fisher-z bar), and a certificate against the least-squares
      bound (`BOUND_GAP_TOL` 0.01).
  - **reembed:** ANTsR's algorithm with every option; existing scores come from `predict`, so
    deep and flow models are handled.
- **`predict` from any subset of views** (dict by name, or `None` entries), extended or not:
  precision-weighted per-view affine maps to the training consensus. With all original views
  present, the output is bit-identical to before.
- **Rule** (anchored within 0.05 of a full refit on average, no seed worse by more than 0.10):
  - simlr: **ACCEPT**, mean −0.0022, worst −0.0027.
  - ned: **REJECT** as a default for deep models, mean −0.028, worst −0.249. On that seed the
    frozen A + B model was weak (0.668) and the refit reached 0.929. Anchoring inherits the
    frozen model's quality.
  - `how="anchored"` stays the general default; the default for deep models is the owner's call.
- **Anomaly found and fixed:** every encoder fit stopped `no_descent` after 4–16 steps. The NSA
  retraction is not a convex projection, so a stall certifies nothing (the gradient map does not
  shrink), and counting it as converged was wrong. Endpoints are certified against the
  least-squares bound instead:
  - simlr gaps 0.002–0.004, certified;
  - ned gaps 0.011–0.013, reported not converged.
- **Not done:**
  - **ANTsR parity:** ANTsR 0.6.7's `extend_simlr_embedding_with_new_modalities` errors inside
    its own fit on a clean two-block input ("arguments imply differing number of rows: 1, 12,
    7"); its projection step works. External repo, not edited.
  - **`nnh.extend_simlr_embedding_with_new_modalities`** (the ANTsPyMM data-frame port) is left
    as it was rather than rerouted through `reembed`.
  - **No gallery figure:** extended views are not appended to `latents`, so the chord and radar
    plots don't show them yet.
- **Verification:** `tests/test_extend.py` (10 tests); full suite 1742 passed, 3 xfailed; `antsxqc.lint` 0 findings. Module: `src/pysimlr/extension.py` (`extend.py` would shadow the `extend` function).


## D18 (v0.11.0 release: plotting redesign, 2026-11-03)
status: done (`3474e9d`, `9d354fb`; this release commit)
- **`pysimlr.plotting` redesigned.** The owner rejected the v0.9.0 gallery; this version was
  accepted.
  - **`theme()`:** one visual system, with a type hierarchy, a header band with a reading line
    and a status line, hairline axes and ringed markers.
  - **`VIEW_COLORS`:** fixed categorical slots, validated for colour-vision deficiency
    (adjacent dE >= 9.1). One blue ramp for magnitude; blue-neutral-red for signed values.
  - **No second y-axis anywhere.**
- **New figures:**
  - `plot_view_chord`: per-pair slots filled in proportion to r, so weak coupling shows as
    empty arc;
  - `plot_component_radar`;
  - `plot_view_radar`;
  - `plot_feature_chord`.
- **Layout changes callers must expect:**
  - `plot_latent_coupling` has two panels;
  - `plot_view_correlations` draws a chord, or a matrix when given `ax`;
  - axes titles are left-aligned (read them with `get_title("left")`).
- **Gallery:** `docs/plotting_gallery.html`, a planted four-view study with sub-coupling,
  captions computed from the fit, 30 figures.
- **Also in this release (`9d354fb`):** the learning-rate default study (D17, not adopted);
  every fit reports `learning_rate`.
- **Verification:** 1731 passed, 3 xfailed; `antsxqc.lint` 0 findings; no `.antsxqc-allow`.

## D17 (learning-rate default, per-component MAI, device-sync audit: three null results, 2026-11-03)
status: done (this commit); no default changed
- **Learning-rate default** (`scripts/learning_rate_selection.py`,
  `docs/audit/evidence/learning_rate_selection.csv`).
  - **Run:** 13 optimizers × 4 arms × 38 fresh cells, plus ADNI-MCI held-out; 2,000 rows,
    0 errors.
  - **What the rule selected:**
    - fixed 0.3 for `hybrid_adam`: mean ΔR² +0.0013, worst −0.0049, 0.77× the time, ADNI
      within one SD;
    - fixed 0.3 for the lookahead pair;
    - fixed 1.0 for the armijo pair (their probe chose 1.0 in 38/38 cells);
    - a 3-rate probe grid for seven optimizers;
    - all 7 rates for rmsprop.
  - **Not adopted.** The fixed defaults fail existing contract tests:
    - Stiefel near-orthogonality: 0.0032 > 0.002;
    - an NSA-Flow "unusable projection" on a k = p = 5 positive view;
    - short-budget energy progress for bidirectional_armijo.

    The 3-rate grid fails three short-budget tests and saves only ~5% for `hybrid_adam`.
    `"auto"` over 7 rates stays.
  - **Landed:** every fit now reports its starting `learning_rate`, not only `"auto"` fits.
  - **Open:** an explicit `learning_rate=0.3` makes NSA-Flow fail on 1 of 5 seeds of the
    k = p = 5 narrow-view fixture.
- **Per-component MAI weights** (`mai_mode="component"`).
  - **Construction:** canonical directions of each view against its LOO consensus, with
    k × k operators. Gauge-equivariant, and reduces to today's per-view weights.
  - **Planted design, 5 seeds:**
    - R² +0.0085 vs view mode (worst −0.010);
    - the sub-coupled view keeps its coupled direction (g ≈ 0.32);
    - the control stays uniform.
  - **Not adopted under the pre-registered rule:**
    - the noise view's directions cannot be 0: it is not assessable at p ≥ n − 1;
    - the cost is 2.5× view mode against a 1.5× budget.
  - The gain is smaller than the no-binary floor costs (0.906 → 0.879). Code not committed.
- **Inner-loop device syncs.**
  - **Made sync-free:** Newton–Schulz (device-side stopping masks) and the
    `_apply_quantile_sparseness` guards. Values are bit-identical, gradients within float32
    rounding, and CPU fits bit-identical.
  - **No gain on MPS:** ned n=2000, 20 epochs: 29.2 and 30.3 s before, 29.5 and 50.4 s after.
    The cost is the CPU round-trip of the SVD in each first-layer basis access (MPS has no
    SVD); removing it needs an approximate polar step, which changes results. This waits on
    the nsa_flow device option. Not committed.
  - **Kept:** the one `isnan(total_loss)` sync per step; a device-side skip would change
    Adam's state.
- **Verification:** 1723 passed, 3 xfailed; `antsxqc.lint` 0 findings.

## D16 (v0.10.0 release: MAI calibration and regression fix, efficiency review, 2026-11-03)
status: done (`7da65e7`, `9a9b336`, `f037094`, `010b6c2`, `87af391`, `25a3c49`, `0be8943`; this release commit)
- **MAI (`7da65e7`, fixed in `87af391`):**
  - **Calibration:** scores are dof-corrected, with dof counted against the rows the bases were
    fitted on (`n_fit`); a view with dof >= n - 1 is not assessable and keeps 1/M.
  - **Gate:** on a permutation null floor (95th percentile, 20 permutations, every 10 updates).
  - **No binary weights:** every view keeps at least half its uniform share
    (`MAI_MIN_SHARE = 0.5`, owner's choice).
  - **New metric `max_cc`.**
- **Regression, reported by the amigo-paper agent:** `7da65e7` failed 159 of 1,025 cells of
  the ADNI-MCI sweep (bipartite graph; deep/flow; svd/pca/ica) with NaN. Causes:
  - batch-n dof correction (MRI views zeroed);
  - a binary [0, 0, 0, 1] gate;
  - NaN from zero-weight neighbourhoods and FastICA on zero columns.

  After `87af391`, the failing class runs 324/324 ok
  (`docs/audit/evidence/adni_mci_mai_regression_check.csv`). Forced [0, 0, 0, 1] weights are
  finite in 36/36 model × mixing × topology cells.
- **MAI evidence** (`mai_calibration_check.csv`, planted design, 5 seeds):
  - `procrustes_r2` with floor 0.5: noise view 0.107, sub-coupled 0.150, narrow 0.143; latent
    R² 0.879.
  - No floor gave R² 0.906; the old mean gate 0.908, but gave the noise view 0.30 and the
    sub-coupled and narrow views 0.
  - **Correction:** the `max_cc` 0.000 for the noise view reported at `7da65e7` came from the
    dof >= n - 1 rule, not from evidence.
- **select_rank extensions (`9a9b336`, owner):** subspace, out-of-bag R²/coupling, basis
  diagnostics.
- **Efficiency review** (CPU unless stated):
  - **`fused_ridge_cv`:** 38.6 s -> 5.0 s with a dense Cholesky Newton solve below 4000
    unknowns. Coefficients equal to 4e-17 (`f037094`).
  - **Deep/flow on MPS:** they failed outright; now run (`010b6c2`):
    - clip foreach;
    - float64 steps moved to the CPU;
    - fused Adam on MPS/CUDA;
    - per-step loss syncs reduced to one per epoch.

    CPU results are bit-identical. MPS is still 1.5-5x slower than the CPU at n 500-5000,
    p up to 20000: the steps are launch-bound with k = 3-5.
  - **Truncated init SVD:** ARPACK for min(n, p) >= 1000, 10x faster at 2000 x 2000.
    Equivalence: mean ΔR² −0.0002, worst −0.0056, converged 32/32 (`25a3c49`).
  - **Lazy pandas/scipy:** import 1.0 s -> 0.6 s (`0be8943`).
- **Rejected in the review:**
  - early-stopping the learning-rate probe grid: it changed the pick in 5 of 32 cells;
  - running the probes in parallel: slower, 9.1 s vs 7.9 s.
- **Open:**
  - **The learning-rate default.** On 64 planted cells, fixed 0.3 matched `"auto"` (mean ΔR²
    +0.0006, worst −0.0034) at about 0.67x the time. This must be confirmed on fresh cells
    and every optimizer before any change; the 3-sweep probe scores are not unimodal.
  - **Per-component MAI weights.**
  - **An inner-loop device-sync audit:** `_apply_quantile_sparseness` `bool()` guards,
    `isnan(total_loss)`, MAI scoring.
- **Verification:** 1722 passed, 3 xfailed; `antsxqc.lint` 0 findings; no `.antsxqc-allow`.

## D15 (v0.9.0 release: visualization suite, 2026-11-03)
status: done (`1f40318`; this release commit)
- **`pysimlr.plotting`** implements the visualization spec (P0-P2) plus interpretable-feature
  facilities:
  - `feature_table` (data);
  - `plot_top_features` and `plot_feature_groups`;
  - `plot_latent_coupling` (training vs held-out);
  - `plot_convergence` (deep components);
  - `plot_basis_diagnostics` and `plot_rank_selection`;
  - `plot_modality_weights` and `plot_flow_warping`;
  - `plot_variance_explained` and `plot_fit_summary`;
  - `plot_benchmark_uncertainty`.
- **One rotation rule** (`_aligned_frame`) for every plot, the Okabe-Ito palette, and an orange
  "NOT converged" banner on unconverged fits.
- **Removed:** `plot_structure_intervals`, `plot_component_stability` and `plot_view_agreement`
  (replaced by the above).
- **Gallery:** `docs/plotting_gallery.html`, 31 reviewed figures in 7 sections from
  `scripts/plotting_gallery.py`.
- **Verification:** see the release commit body.

## D14 (v0.8.0 release, 2026-11-03)
status: done (this release commit; `main` at 88ecbf8 before it)
- **Breaking:**
  - `energy_type="negentropy"` is refused (use `negentropy_weight`);
  - `fused_ridge` is one multi-task GLM (`groups=` strata or `thresholds=`, the ANTsR model;
    gaussian, binomial and poisson; covariates; `top_k`);
  - `pysimlr.viz`, `pysimlr.visualization` and `pysimlr.benchmarks.plotting` are removed in
    favour of `pysimlr.plotting` (interpretation plots, none of `v`), with the diagrams in
    `pysimlr.diagrams`.
- **Fixes:**
  - `similarity._std` now has a relative floor; the absolute 1e-6 floor had broken `align`,
    `logcosh`, `exp`, `kurtosis` and `negentropy` at simlr's internal scale (`18db280`);
  - `structure` is exactly corr(X, u R) under every mixing (`ae5eb07`);
  - a sweep in which every view's line search fails now reduces the learning rate 10x and
    retries instead of stopping with `no_descent`. On an 80-subject case 3/10 fits converged
    before and 10/10 after; default cells and ADNI are unchanged (`60ba439`).
- **Defaults:** `DEFAULT_DEEP_ITERATIONS = 300` for every deep and flow fit (`80f9536`);
  `collapse_threshold` is a fit parameter (v0.7.0).
- **Convergence:** `pysimlr.ConvergenceWarning` on every unconverged fit (`d357e2f`).
- **From `amigo`:**
  - B2 re-run at the shipped probe grid and after the `_std` fix: the defaults stay
    `nc`/`hybrid_adam`/`newton` (mean latent R^2 0.988, 100% converged);
  - the R10 whitener evidence;
  - the benchmark readiness statement: READY for linear and deep;
  - opt-in `nc_ridge`, and the D1 tracking.
- **Verification:** see the release commit body.

## D13 (v0.7.0 release: parallel refits and `collapse_threshold`, 2026-11-02)
status: done (`4a29d74`, `364dc67`; this release commit)
- **`n_jobs`** on `loading_inference` and `select_rank` (`4a29d74`):
  - refits and permutation fits run in worker processes, one torch thread each;
  - every index is drawn in the parent, so the output is identical for every `n_jobs`;
  - `select_rank` shares one worker pool across candidates;
  - measured, ks [2, 3] with 20 subsamples on 12 cores: ADNI-MCI 97.8 s -> 17.2 s, synthetic
    24.2 s -> 6.4 s (`docs/audit/evidence/select_rank_speed.csv`);
  - rejected: reusing the reference fit's tuned learning rate. It was 1.3x faster, but it moved
    an under-identified component's stability from 0.913 to 0.735.
- **`collapse_threshold`** on `simlr`, `lend`, `ned`, `ned_shared_private` and `flow_v`
  (`364dc67`):
  - default 0.3, range [0, 1);
  - the trade-off: converged `avg` fits keep score correlation up to about the threshold (0.31,
    0.51 and 0.69 at 0.3, 0.5 and 0.7), while a high threshold lets narrow views collapse (at
    0.7 the score correlation was 0.89 and outcome R^2 fell 0.10)
    (`docs/audit/evidence/collapse_threshold_selection.csv`).
- **Verification:** see the release commit body.

## D12 (v0.6.0 release: `amigo` WP-2..WP-8 and phase C corrections, 2026-11-02)
status: done (merge of `amigo` at `03886cb` into `main`; this release commit)
- **Removed:** the `simlr_lbfgsb` method and the `nsa_lbfgsb` optimizer, along with their
  oracle plumbing. They were not competitive with `hybrid_adam`: on the 12-cell grid, no prox
  setting was non-inferior (rule R5, `f8db84b`, `75077c5`). Both names raise errors naming the
  replacement and the release.
- **Diagnostics with evidence** (in `docs/tracking_amigo.md`; scripts under `scripts/`, CSVs
  under `docs/audit/evidence/`):
  - **Linear narrow-view collapse:** a step-size basin. The lowest-energy unpenalised fit is not
    collapsed. Collapse weight 1.0 kept, on the score-correlation criterion (R2; `9c3279d`,
    corrected `8a6d595`).
  - **Deep collapse:** 7/20 collapse without the penalty, 0/20 with it, onset after warmup.
    Collapsed fits have worse similarity. Penalty kept (R3; `0d98ff2`, evaluator fixed
    `489e0fd`).
  - **Correlated factors:** newton decorrelates consensus scores; `avg` keeps correlation,
    saturating near 0.30 on partly unconverged fits. D3 open (R4; `b1a1afe`, `992b143`,
    `03886cb`).
  - **NED rounding:** continuous float32 amplification. No discrete tie-break found, so `ned`
    stays rounding-unstable (R6; `ae61cf0`).
- **Release fixes:**
  - the removal messages give the correct release (0.6.0 for `nsa_lbfgsb`) and no internal
    rule IDs;
  - the README optimizer list is corrected;
  - the usage notes are updated (removals; the correlated-factor caveat).
- **Verification:** see the release commit body.

## D11 (v0.5.0 release: `amigo` milestones 0-1 and usage notes, 2026-11-02)
status: done (`main` fast-forwarded to `793ab7f`; usage notes `a5732bd`; this release commit)
- API:
  - one stop vocabulary (`api.STOP_REASONS`);
  - `nsa_w` is the single orthogonality weight, capped at 0.99 (`defaults.MAX_NSA_W`);
  - `constraint` and `parse_constraint` removed;
  - leave-one-out consensus by default;
  - `grad_map` reported by every family (`3a88f5f`, `9f2a5d4`).
- Objective:
  - the similarity energy is minimised subject to the NSA-Flow projection; the defect penalty is
    removed (`d3e5590`);
  - `DEFAULT_TOL = 1e-4` for every method, with 300 sweeps (`5cc49d8`);
  - scale-free energies (`nc`, `align`, `procrustes`, `recon_r2`, ICA contrasts) weight views
    equally, 1/M. The old init-relative weights spread up to 279x on synthetic fits and 70x on
    ADNI. `acc` and `recon` keep the 1/(|E0| M) calibration (`acd08fe`, `708c9c6`, `394144c`;
    rules R1 and R9 in `docs/tracking_amigo.md`).
- Interpretation:
  - every fit carries `structure` (structure coefficients), two-stage `loadings` (varimax,
    sign-anchored; hard assignment under positivity) and `loading_margin`;
  - `loading_inference` gives half-sample intervals, component stability and stability selection;
  - `select_rank` picks k by per-component stability (`580bb91`, `792e9a6`, `9cdf093`).
- Benchmarks: `lobe_crosstalk` (identically 0) replaced by column coherence and support overlap
  (`5c77d60`).
- Docs: `docs/usage_notes.md`, covering defaults, convergence checks, interpretation and known
  limitations.
- Verification at `a5732bd`: full suite 1638 passed, 3 xfailed; `antsxqc.lint` 0 findings.
- Not in this release: the later `amigo` commits, including the `simlr_lbfgsb`/`nsa_lbfgsb`
  removal and the WP-2..WP-8 diagnostics. Their write-ups are being corrected (phase C).

## D10 (Library hot-path and test-suite speedup on `perf/fast-test-suite`, 2026-11-02)
status: committed on branch `perf/fast-test-suite` (off `main` `33e05f8`)
- Root causes fixed in library (`src/pysimlr/`):
  - `flows.py`: replaced eager `import antstorch` (which pulled ITK/ANTsPy/matplotlib at `~2.58s` per
    worker) with direct submodule loader `_load_antstorch_real_nvp_builder()` (`~0.008s`); bound
    `_ShuffledTensorBatches` to target device; added per-step `_step_v_cache`; hoisted `_all_params` and
    replaced 80-tensor per-batch `isfinite()` scans with `clip_grad_norm_(..., foreach=True)` `total_norm`
    check; passed `compute_diagnostics=False` inside `_train_flow_loop`; replaced `copy.deepcopy(state_dict)`
    with tensor `.detach().clone()`.
  - `deep.py`: replaced `DataLoader(TensorDataset, ...)` per-row Python collation with
    `_ShuffledTensorBatches` (`index_select`, bit-identical `torch.randperm` RNG sequence); added per-step
    `_step_v_cache` on `LENDNSAEncoder` and early `alpha <= 0.0` guard before `self.v`; enabled
    `optim.Adam(..., foreach=True)`; added backward-compatible `compute_diagnostics: bool = True` to
    `calculate_sim_loss` (`False` inside `_train_loop`); hoisted `all_params`, `clip_grad_norm_(..., foreach=True)`,
    and clone-based best-state snapshots; wrapped `first_layer_diagnostics` in `torch.no_grad()`.
  - `simlr.py`: shared standardized views and `initialize_simlr` basis across `learning_rate="auto"` probes
    and final fit via `_INTERNAL_PROBE_TOKEN`; skipped post-fit SVD/pinv/interpretability/`basis_rank_report`
    and per-sweep `stationarity_residual` on discarded probes (`_probe_only=True`); removed dead inner-loop
    `_capture_consensus_anchor`; hoisted float64 view/domain tensors and added 1-slot `_last_proj` reuse in
    `_view_oracles`.
  - `consensus.py`, `utils.py`, `sparsification.py`, `benchmarks/__init__.py`: skipped unused `pinv(mean_u)`
    `local_anchor` on LOO subsets (`not need_anchor`) and fast-pathed `len(proj_list) == 1`; cached
    `_EYE_OVER_K_CACHE` (`orthogonality_defect`) and `_THREE_EYE_CACHE` (`newton_schulz_orthogonalize`) and
    skipped provably impossible `step_idx < 4` norm syncs for `scaled_unit, k >= 2, tol <= 1e-4`; vectorized
    `_apply_quantile_sparseness` across all `k` columns and searched MRU-first in `_NSA_RETRACT_CACHE` /
    `_SMOOTHING_CACHE`; converted `pysimlr.benchmarks` to PEP 562 lazy exports.
- Test-runner and suite hygiene (`pyproject.toml`, `pytest.ini`, `tests/conftest.py`, `tests/`):
  - Pinned `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MPLBACKEND=Agg`, and `torch.set_num_threads(1)` per
    worker; configured `-n 10 --dist loadgroup --no-loadscope-reorder -m "not slow"` and disabled unused
    autoloaded plugins (`playwright`, `jaxtyping`, `asyncio`, `anyio`, `base_url`).
  - Implemented `@pytest.hookimpl(tryfirst=True)` 3-tier LPT scheduling + 25-item xdist batching for `<10ms`
    unit tests; deferred heavy `sklearn`/`matplotlib`/`seaborn` imports from 10-worker collection to
    `scope="module"` fixtures; deduplicated redundant identical setup fits via module `lru_cache` without
    loosening any threshold, sample count, iteration count, or `strict=True` xfail.
- Measured results (`n = 3` isolated sequential repeats on Apple Silicon arm64, report:
  `docs/test_suite_speedup_report.html`):
  - Default suite (`pytest`, `1566 passed, 2 xfailed, 0 failed`): `136.40s` baseline (`50.60s` naive xdist)
    -> **`19.75s / 19.77s / 19.91s`** min / median / max (`6.90x` speedup, strictly `< 20.0s` on all 3 runs).
  - Full suite (`pytest -m ""`, `1588 passed, 4 xfailed, 0 failed`): `199.86s` -> **`32.80s`** (`6.09x` speedup).
verification: pytest (n=3) 1566 passed, 2 xfailed, 0 failed in 19.75s/19.77s/19.91s; pytest -m "" 1588 passed, 4 xfailed, 0 failed in 32.80s; test_implementation_standards.py 7 passed; antsxqc.lint 0 findings.

## D9 (NSALBFGSB rounding-instability root cause and proximal-quartic fix, 2026-11-02)
status: done in the working tree, uncommitted (on top of D8; paper unchanged per owner decision)
- Root cause (verified vs refuted): instrumented the exact 3-view reproduction (`test_api_registry._views(n=60)`,
  `k=3`, `iterations=20`, `1e-8` float32 input perturbations, 3 seeds) and the +20 constant-shift test.
  Candidates refuted as primary cause: outer quantile projection (`sparseness=None` still gave gaps
  `5.82e-02` positive / `9.94e-01` either; `sparseness=None, constraint="none"` gave `2.26e-02` /
  `3.24e-01`; support XOR was 0 on the sweeps where divergence grew by `1e4x`), dead-column step
  rejection (`last_rejected` fired 0 times across all 57-63 step calls), inner-solve exception handler
  (0 exceptions/warnings), C++ `_native` extension (`_native=None` gave identical `5.82e-02` /
  `8.31e-01`), and missing-callable fallback (0 runtime triggers). Verified cause: `_smooth_objective(V)`
  with consensus `u` fixed is indefinite around the incoming iterate `V_0` (finite-difference Hessian
  eigenvalues: 7/16 to 18/30 negative on `_views`, down to `-9.22`; 30/60 negative on `buried`, down to
  `-32.44`), combining the exact 0-homogeneous radial-tangential saddle `+-||grad_{v_j} E||_2 / ||v_j||_2`
  of column normalisation with `4 W_orth <V_0^T V_0 - I, D^T D>` and `correlation_penalty` non-convexity.
  Even when zero discrete line-search or active-set branches differed (`alpha=(1.0, 1.0)`, `fix_xor=0`),
  running 20 unanchored quasi-Newton steps collapsed L-BFGS scaling `theta = y'y / s'y` from `32.1` to
  `0.00344`, drifted column norms `1.0 -> 7.42`, and amplified a `7.69e-07` difference to `2.13`
  (`2.77e6x`) in a single inner solve. Even smooth `logcosh` diverged to `0.653` without an anchor.
- Fix (`src/pysimlr/optimizers.py`, `NSALBFGSB.step`): replaced the unanchored inner objective with the
  column-scale-invariant proximal-quartic alternating-minimisation subproblem
  `Phi(V; V_0) = E(V) + (mu_2 / 2)||V - V_0||_F^2 + (mu_4 / 4)||V - V_0||_F^4`, where
  `mu_2 = 2.5 / s_0^2`, `mu_4 = 64.0 / (s_0^2 ||V_0||_F^2)`, and `s_0^2 = ||V_0||_F^2 / k`. Both value
  and gradient vanish at `V = V_0`, so fixed-point stationarity, the outer energy, and package defaults
  are unchanged (`hybrid_adam` bit-identical: `[9.57e-08, 2.60e-07, 6.29e-07]` positive,
  `[3.07e-06, 5.15e-07, 7.53e-07]` either). `mu_2` convexifies local negative curvature and pins the
  column-norm gauge; `mu_4` bounds single-sweep excursions against the frozen consensus near
  `||V - V_0||_F <= 0.25 ||V_0||_F` (purely quadratic `mu_2` alone still allowed `||V - V_0||_F = 1.13`
  on `buried` into `-7.49` curvature). Also replaced the silent plain-step fallback on missing
  energy/gradient callables with `ValueError`.
- Measured results (report: `docs/simlr_lbfgsb_d9_report.html`): `nsa_lbfgsb` perturbation gaps drop
  from `[5.82e-02, 8.49e-02, 9.47e-02]` to `[7.46e-07, 7.46e-07, 3.23e-10]` (`positive`) and from
  `[4.29e-01, 4.22e-01, 8.31e-01]` to `[1.39e-06, 2.83e-06, 1.05e-06]` (`either`). All 7 supported
  energies (`nc`, `recon_r2`, `align`, `procrustes`, `exp`, `logcosh`, `acc`) have max gap `<= 5.42e-06`
  (down from `7.44e-03 .. 7.67e-01`). `+20` constant-shift max `|dR^2|` drops `1.43e-02 -> 6.45e-08`
  (`hybrid_adam` `5.83e-08`). Isolated fit timings (`n=5`, `k=3, iter=20`): `nsa_lbfgsb`
  `818.8 / 825.5 / 831.1 ms` min/median/max (`hybrid_adam` `412.9 / 414.3 / 418.3 ms`).
- Tests: `"simlr_lbfgsb"` removed from `tests/test_api_registry.py::_ROUNDING_UNSTABLE`; added
  `test_lbfgsb_refuses_missing_energy_or_gradient_callable` and
  `test_lbfgsb_is_reproducible_under_float32_input_perturbations[{positive,either}]` in
  `tests/test_solver_certificate.py`. `SiMLR-LBFGSB` remains out of `paper/` per D8 owner decision.
verification: pytest -m "" 1578 passed, 4 xfailed, 0 failed; antsxqc.lint 0 findings.

## D8 (Default objective, collapse penalty, stopping rule, SiMLR-LBFGSB dropped, 2026-11-02)
status: done in the working tree, uncommitted (on top of e254f05)
owner decisions: remove the ANTsTorch whitener; choose energy, optimizer and mixing jointly under
both positivity settings; "nc with newton", collapses to be fixed; DEFAULT_ENERGY_TYPE = "nc"; settle
the stopping tolerance; fix or drop SiMLR-LBFGSB; deal with the oscillating alternation.
- Default selection (scripts/default_selection_benchmark.py; evidence
  docs/audit/evidence/default_selection_grid.csv): 4 energies x 5 optimizers x 4 mixings x
  {positive, either} x {buried-private, linear} generators, 5 seeds, latent R^2. Every top cell is
  "nc": nsa_lbfgsb/newton 0.983, nsa_lbfgsb/avg 0.976, hybrid_adam/avg 0.971, hybrid_adam/newton
  0.969 (worst cell 0.915); best logcosh 0.860, acc 0.750, exp 0.569. Follow-up checks (scripts/default_followup_checks.py;
  docs/audit/evidence/default_followup_checks.csv): nc newton combos 0/5 NED collapses, ADNI held-out
  ~0.16-0.17 against acc ~0.153; nc/hybrid_adam/avg had 1 collapse, so newton is kept. Defaults:
  energy "nc", optimizer hybrid_adam, mixing newton.
- Collapse: newton/svd mixing orthogonalises the consensus u, so a collapse in the per-view scores
  XV was invisible to a u-based penalty, and "nc" is rank-blind. New `utils.correlation_penalty`, a
  hinge on pairwise score correlations, sum over pairs of ((|r|-t)_+/(1-t))^2 with t = 0.3
  (DEFAULT_COLLAPSE_THRESHOLD), weight 1.0 (DEFAULT_COLLAPSE_WEIGHT, `simlr(collapse_weight=)`).
  Applied to each view's scores (linear: in the oracle and in every reported energy) and to the
  latents (deep; it replaces the u-covariance term). Variants measured: mean over pairs diluted at
  k=5 (rank 4); squared |r| cut nonneg_parts support recovery 1.000 -> 0.917; power 4 fixed support but
  raised "unusable projection" at k=p=5; power 8 gave rank 4 and narrow R^2 0.823; hinge t=0.5 gave rank
  4; hinge t=0.3 passed every check (support 1.000). It also reduces NED's rounding sensitivity: a
  1e-6 input perturbation moves R^2 by 0.0036 with it, 0.084 without.
- Oscillation (issue 5): resolved by the two items above. The stress tests in
  test_milestone1_adversarial_stress.py pass and their strict xfails are removed. Consensus damping
  (u <- (1-b)u + b u_new) was measured and refuted: at b = 1, 0.5, 0.25, 0.1 the stress range stayed
  0.004-0.006, the median residual rose 0.54 -> ~0.72 and recovery fell 0.969 -> 0.903. Not added.
- Stopping rule (issue 2): the prox-gradient residual with a fixed step ||V||^2 is not a valid
  certificate for the non-convex quantile projection, so `tol` has one meaning, the relative-
  improvement plateau. A plateau stop reports certificate="energy_plateau", converged=True;
  no_descent and max_iter stop uncertified. The residual is no longer computed: `grad_map`,
  `grad_map_history` and the per-view `grad_map_E` diagnostics are removed (owner instruction), and
  simlr no longer needs nsa_flow's gradient_mapping (nsa_lbfgsb still uses it inside its own solve).
- Scripts use the package defaults: run_deep_benchmark.py, paper_brca_loadings.py, the three
  benchmark_*_simlr_vs_* scripts and real_data_method_comparison.py no longer pass energy "acc" or
  mixing "svd"; experiment_iteration_vs_init.py takes --energy (default the package default; the
  DEFAULT_TOPOLOGY evidence was measured with acc). New `defaults.DEFAULT_MIXING_ALGORITHM = "newton"`
  replaces the literal in the simlr/deep/flow signatures (no behaviour change).
  `consensus.compute_shared_consensus` keeps its own "svd" default: switching it to newton failed 4
  direct-call tests (newton with k=None returns the full width instead of the inferred rank; with
  training=True it returns no anchor; the graph topology mismatches shapes). Package callers always
  pass k and the mixing, so fits are unaffected; the newton branch's gaps are open.
  run_adni_mci_grid_sweep.py keeps its explicit design grid (it crosses energies and mixings on
  purpose), but its MAI arms use only acc/align/procrustes with svd mixing, so they exclude the default.
- SiMLR-LBFGSB (issue 3): not reproducible under "nc" either, with subspace gaps 0.04-0.49 for inputs
  1e-8 apart (hybrid_adam ~1e-8). Dropped from the paper (model lists, bar charts, label maps, prose,
  paper_numbers keys, scripts/paper_adni_ablations.py, run_adni_mci_grid_sweep.py,
  real_data_design_sweep.py, run_deep_benchmark.py). It stays in the library, recorded in
  test_api_registry._ROUNDING_UNSTABLE. The methods text now names hybrid_adam, not L-BFGS-B.
- Whitener removed (`FlowWhitener`, `flow_whiten_matrix`, `_lamnr_flows_function`; tests deleted).
  ANTsTorch apply_lamnr_flows_whitener.py:161 solves with L^T instead of L in the forward map; the
  inverse (line 171) is correct. Not edited (external repo); noted in docs/api_contract.md.
tests: new tests/test_collapse_penalty.py; test_tol_controls_early_stopping rewritten for the plateau
rule; test_degradation_tracks_rank_loss replaced by test_no_rank_loss_at_any_constraint_weight
(effective rank > 2.8 at orthox 0, 0.1, 0.5); the collapse_weight liveness fixture is unconstrained
with a random start (the default fixture's scores never reach |r| = 0.3). New strict xfails:
`ned` in the +20-shift R^2 test (rounding, not shift: a 1e-6 perturbation gives the same 0.0036);
test_ned_advantage_contract (both models on package defaults: SiMLR latent recovery 0.745 against
NED 0.675, NED 0.070 below, the clause allows 0.05; NED held-out R^2 still leads, 0.882 against
0.710). Owner decision pending.
Follow-up (owner review of D8, same day):
- Newton/avg consensus (`consensus.compute_shared_consensus`): averaging mixings now raise a clear
  ValueError for unequal widths or a width other than k (they ignored k and failed in torch.stack),
  and return an exact linear anchor (replay error <= 5e-7), so anchored prediction is batch-invariant.
  Default mixing is DEFAULT_MIXING_ALGORITHM. Real predict paths were already batch-invariant via the
  frozen map (|u_full - u_half| <= 9e-8); trained bases bit-identical before/after for simlr, ned,
  lend, ned_shared_private, flow_v. Tests: tests/test_consensus_newton.py; reduction tests pass
  mixing_algorithm="svd" explicitly.
- Stationarity residual restored as a diagnostic with a valid definition,
  `utils.stationarity_residual`: gradient mapping with P the convex projection onto the iterate's
  current face (support mask, clamp under positivity) plus an off-support ratio. Calibration
  (scripts/stationarity_residual_calibration.py, 32 runs x 300 sweeps; evidence
  docs/audit/evidence/stationarity_residual_calibration.csv): >= 2 decades of decrease in 7 of 8 runs
  whose energy settled (1.74 in the eighth); stays high where the energy keeps moving (lars either).
  `grad_map`, `grad_map_history`, per-view `grad_map_E`/`off_support` are back; the benchmark runner
  uses the same helper for every model (its "acc" energy fallback is now the package default).
- Collapse threshold (owner: "a magic number"): candidates measured
  (scripts/collapse_candidates_benchmark.py; docs/audit/evidence/collapse_candidates.csv).
  C2, scoring "nc" on the polar factor of the scores, failed (narrow-view R^2 0.523, grid min 0.250).
  Linear: without a penalty narrow views collapse (R^2 0.301); the 0.3 hinge fixes it (0.947) but
  lowers buried/either recovery (0.878 vs 0.933). Correlated planted factors (rho 0-0.7): linear
  recovery identical with and without the hinge, and recovered score correlation is ~0.03-0.15 for
  every candidate, so the orthogonal newton consensus, not the hinge, decorrelates the scores.
  Deep: without any penalty NED collapses on 6 of 20 (data seed, start seed) pairs, with the hinge on
  0 of 20. A first measurement with one start per data seed (0/5 either way) wrongly suggested the
  old u-covariance loss was the cause and the deep penalty could go; it was removed and restored the
  same day. OPEN: root-cause the linear narrow-view collapse (owner decision: root cause first);
  the hinge stays until then.
verification: pytest -m "" 1578 passed, 4 xfailed, 0 failed (includes D9); antsxqc.lint 0 findings.

## D7 (Defect fixes, shim removal, housekeeping, 2026-11-01)
status: done, committed e254f05
owner decisions: fix the gd anomaly, the unseeded start, the stationarity certificate and the
late-bound callbacks; remove the "nsa_flow" optimizer; remove compatibility shims and do all
housekeeping "brutally".
fixes (each with a regression test):
- gd anomaly: `recon_r2` (and `regression`, which resolved to it) profiles out the column scale
  -- E(v) = E(3v) = E(v diag(1,10,0.1)) = 0.99147 -- yet was in GAUGE_FREE_ENERGIES, so its column
  norms drifted (1 -> 2-10 in two gd steps) and fits returned bases of arbitrary scale. Now only
  the unprofiled `recon` is gauge-free. The learning-rate test measures displacement from a
  near-zero-rate fit (linear in lr in the stable range: 5.9e-5 at 1e-4, 5.9e-4 at 1e-3).
- Unset random_state: one seed is drawn up front (shared by the learning-rate probes and the
  fit), recorded, and handed to every initializer; it was None, so randomised starts ("gcca") used
  the global stream after an optimizer-dependent number of probe draws. Seeded fits leave the
  global stream untouched; unseeded fits consume exactly one draw (the deep/flow contract). The
  varimax test's spy held the initializer's list, which simlr mutates -- it measured the fit.
- Stationarity certificate: the prox-gradient residual ||P(V - ||V||^2 grad E) - V|| / ||V|| of
  the current iterate under the projection actually applied; it used the step-size residual (a
  tiny line-search step read as stationary), the descent direction as the gradient (an uphill
  probe) and a bare clamp for P. A probe the projection refuses certifies nothing (+inf, noted).
  Fits now mostly stop "plateau, uncertified" (residuals 0.005-0.03 on the test fixture).
- Per-view callbacks: energy, gradient, objective and projection are built by `_view_oracles(i,
  u_i, sparseness)`; defined in the loop they read the loop index and data when called.
removed: optimizer "nsa_flow" (gd plus a second projection, with a silent SVD fallback) and its
`nsa_w` optimizer key; method-name aliases; `deep_simr`, `flow_simlr`, `flow_simlr_v`,
`FlowSiMLR*Model`, `flow_whitener`; energy aliases `regression`/`gauss`/`normalized_correlation`
(raise naming `recon_r2`/`align`, `exp`, `nc`), `recon` as a linear objective; the constraint
string's third field (format `<type>x<weight>`); `retraction_type` (bit-identical fits for any
value); legacy nsa_flow constructor keywords and NSAFlowLayer fallback; `predict_simlr(
allow_legacy_refit)`, prediction from pre-0.3.0 results, `reconstruct_from_learned_maps` returning
[] without `w`; `invariant_orthogonality_defect` (+gradient), `calculate_u`, `_newton_step_ortho`,
`load_nsa_flow_orth`, `_retraction_candidate`, `_backend_has_sign_blind_fidelity`,
`rank_effect_size`, `plot_case_generative_shape`, `procrustes_mse`; regression-result alias keys,
`top_n`, unused `iterations`/`max_iter`/`tol`/`constraint_iterations`; generator aliases
`noise_level`, `nonlinear_shared`, `linear_footprint`. Listed in docs/api_contract.md.
kept on evidence: the deep/flow orthogonality penalty on the raw first-layer weights, documented
as a test-compatibility shim, is a regulariser -- without it NEDPP's shared R^2 fell to 0.384 (< 0.4).
housekeeping: 101 scripts deleted (92 referenced by nothing; 9 stale or superseded), 20 remain;
22 unreferenced paper figures deleted (one pre-rendered result figure remains in use:
paper/figures/heart_waterfall.png); unused imports and locals removed (ruff F401/F841/F541);
retraction/smoothing caches bounded to 64 keys.
verification: pytest -m "" 1575 passed, 6 xfailed, 0 failed; antsxqc.lint 0 findings.

## D6 (Open items closed: audit A1-A4, iteration vs SVD start, topology default, duplicate code, 2026-11-01)
status: code released as v0.4.0 (minor: topology default changes results); paper caches regenerating, paper not yet rebuilt
owner decisions (2026-11-01): "loo should be the default topology"; "eliminate duplicate code,
maximize code reuse".
what changed:
- Audit A1-A4 (docs/audit/AUDIT_2026_09.md): all four were already fixed (d2e86b7, 1b6790a,
  56549a3) with no recorded resolution. `tests/test_audit_2026_09.py` (17 tests) reproduces each
  measurement; gradients are checked against central finite differences, not autograd.
- Iteration vs SVD start (CORRECTNESS_AUDIT.md "Answer"): the limiter is the topology.
  `scripts/experiment_iteration_vs_init.py`, private factors at twice the shared scale, 8 seeds,
  40 sweeps: star gain -0.002 (3/8 seeds), loo +0.217 (8/8); two-view CCA 0.9945. Evidence in
  docs/audit/evidence/; test `test_loo_iteration_improves_on_a_wrong_initialization`.
- `defaults.DEFAULT_TOPOLOGY = "loo"` for every entry point. `simlr` trains each view against
  the other views but reports `u`, `w` and the prediction map from the all-view consensus (as
  the deep and flow paths do); per-view targets are `u_per_modality`.
- Defect found and fixed: a zero line-search step (Armijo found no decrease) gave a fixed-point
  residual of exactly 0 and was certified "stationary". `simlr` now stops with
  `stop_reason="no_descent"` and no certificate. On the test fixture LARS reaches energy 0.0180
  against Armijo's 0.0197 stopping point, so those earlier "converged" Armijo fits were stalls.
- Defect found and fixed: the synthetic generators replaced an explicit `p_list` / `shared_k` /
  `noise_scale` with regime constants whenever `private_scale` was set; default outputs are
  bit-identical to HEAD (9 calls checked). Test: tests/test_synthetic_explicit_params.py.
- Duplicate code removed (behaviour preserved, suite green after each step): MAI scoring
  (`consensus.loo_mai_scores`, `mai_gated_target`, `check_mai_metric`; was 6 copies); model
  mixins `_ModalityWeightingMixin` / `_MultiViewModelMixin` (update_mai, consensus state and
  buffers, `_training_consensus`, projection schedule, diagnostics, initialize_v); `_mlp`,
  `_linear_encoders`; `_fit_first_layer_model` (lend/ned bodies); Nadam(Adam),
  BidirectionalLookahead(Lookahead), BidirectionalArmijoGradient(ArmijoGradient);
  `simlr._check_views` / `_scale_like_training`; `_nsa_data_transform` on
  `nsa_backend.result_matrix`; `sparse._keep_neighbours` / `_finish_laplacian`;
  `utils._angle_gram` / `_svd_on_cpu`; `interpretability._fit_with_importance`;
  `nnh._map_left_right` / `_unit_columns`; `flows._lamnr_flows_function`;
  `synthetic_cases._planted_latent`. Scripts: 29 scripts import from the new
  `scripts/script_shared.py` (copies verified AST-identical before replacement). Duplicate scan:
  no identical or >= 0.85-similar functions left in src/pysimlr.
- `sparseness_quantile` renamed `sparseness` in every module, helper and test (no alias).
- `optimizer_type="torch_lbfgs"` and the `simlr_lbfgs` method removed (owner decision, after
  the v0.4.0 commit a39ac19). Measured (40 iterations, acc, 3 fixtures x 2 topologies): it was the
  weakest optimizer in 5 of 6 runs and on one never left its start while being certified
  "stationary". Cause, from reading: its closure scored the projected basis but received the
  tangent-projected gradient, and its curvature history spanned changing consensus targets.
  `create_optimizer("torch_lbfgs")` raises naming `nsa_lbfgsb`; the NSA transforms default to
  NSA-Flow's own optimizer (nsa_flow deprecated torch_lbfgs); `SiMLR-LBFGS` left the benchmark
  scripts and the paper, whose model count is now computed (`N.n_models_word`).
- fig-nsa-v-interpretability is built from code: `scripts/paper_nsa_v_loadings.py` ->
  results_cache/nsa_v_loadings.json (nsa_flow 3.5 dropped `varimax=True`, so the old script no
  longer ran; the rotation is `pysimlr.svd.varimax`).
- `scripts/run_adni_mci_grid_sweep.py`: no absolute paths; writes its provenance; exits non-zero
  if any configuration fails (the 224 NameError rows of f4e53a6 went unnoticed).
- Optimizer benchmark: `scripts/optimizer_benchmark_synthetic.py` (67 s, 5 seeds, 14 optimizers;
  `nsa_flow` excluded: it is gd plus a second projection). Generator: sparse non-negative shared
  factors buried under private factors at 2x; PCA start latent R^2 0.17, two-view CCA 0.978.
  The "acc" optimum itself stays near 0.48 however long it runs, so recovery measures the
  objective; the primary metric is the optimality gap (E - E_ref)/(E_start - E_ref) at 25
  iterations. Separates optimizers: blocked F 3.15, p 0.0016 (Friedman p 0.017); support F1
  does not (p 0.93; quantile sparsity fixes the support size). Gap: lars 0.10, nsa_lbfgsb 0.22,
  hybrid_adam 0.32, ... armijo_gradient (default) 0.41, gd 0.44, bidirectional_lookahead 0.56.
  Evidence: docs/audit/evidence/optimizer_benchmark_synthetic.csv.
- Default linear optimizer is `hybrid_adam` (`defaults.DEFAULT_LINEAR_OPTIMIZER`; owner decision).
  Grid (docs/audit/evidence/optimizer_grid_synthetic.csv; energies acc/nc/logcosh/recon_r2 x
  mixing newton/svd/avg x positivity, 6 optimizers, 3 seeds): hybrid_adam best mean rank and in the
  top three in 11 of the first 12 settings; armijo_gradient worst. Held-out ADNI-MCI unchanged
  within seed noise across optimizers.
- Default objective NOT changed to "nc" (owner asked; reverted on evidence, decision pending):
  "nc" with hybrid_adam fails the narrow-view linear check (R^2 0.301 vs 0.973 for acc), collapses
  NED latents on 2/5 seeds of the latent-collapse fixture (off-diagonal 1.24/1.34 vs <=0.002), and
  breaks NED budget independence. "logcosh", "exp" and "kurtosis" pass all three checks
  (narrow R^2 0.982 / 0.979 / 0.951, 0/5 collapses, budget-independent); in the grid "logcosh"
  recovers the shared latent better than "acc" in every mixing/positivity cell.
- Open defects recorded as xfail: gd travel is non-monotone in learning_rate (relative distance
  5.22 at lr 0.1, 1.2e-2 at 0.5; unexplained); simlr passes generator=None to initialize_simlr
  when random_state is None, so the start depends on how much global RNG the learning-rate probes
  consumed (optimizer-dependent).
verification: fast suite 1598 passed / 2 xfailed; `pytest -m ""` 1622 passed / 3 xfailed / 0
failed (200.7 s, 6 workers); `python -m antsxqc.lint src/pysimlr` 0 findings.
open:
- Fixed (owner: "fix this bug in any impacted optimizer"): the optimizer oracle. `simlr`
  scored the projected point in float32 while handing a tangent-projected gradient taken there
  (slopes 2-14x off finite differences; energy moving in float32 spacings, 4.657e-10 at
  E = -6.024e-3). Now every optimizer gets two float64 oracles: the objective E(P(v)) for line
  searches, and the smooth part E(v) (column-normalised for gauge energies) with its exact
  autograd gradient for L-BFGS-B, which also runs in float64. Inputs 1e-8 apart, max|V1 - V2|
  after 20 iterations: armijo 3.4e-5, lars 6.0e-7, hybrid_adam 2.3e-6, nsa_flow 6.0e-7 (were up
  to 1.4). Test: test_the_oracle_simlr_hands_its_optimizer_is_consistent (slope = FD to 1e-5).
- Open: `nsa_lbfgsb` stays sensitive on "acc"/"nc" (0.16 after one sweep): L-BFGS-B on an
  objective with absolute-value kinks; its Wolfe line search fails at a kink and the paths
  branch. Smooth energies are far less sensitive (recon_r2 3e-2 after 20). Strict xfail in the
  shift test with this reason; no per-optimizer settings. Not an nsa_flow bug.
- Open: with the exact direction, armijo on the ill-conditioned stress case reaches a better
  objective (best -1.629 vs -1.116 at 40 iterations) but oscillates (-1.29 to -1.63) before
  settling; the two settling stress tests are strict xfail. Damping the alternation is not done.
- ADNI-MCI, scripts/run_deep_benchmark.run_experiment_task, seeds 42-44, before (a39ac19) ->
  after: SiMLR Predictive_Metric 0.1550/0.1513/0.1585 -> 0.1671/0.1556/0.1578 (mean 0.1549 ->
  0.1602); SiMLR-LBFGSB 0.1730/0.1469/0.1556 -> 0.1564/0.1461/0.1648 (mean 0.1585 -> 0.1558);
  LBFGSB fit time 2.2-2.9 s -> 1.0-1.1 s. Three seeds; not tested for significance.
- Fixed: `_NSA_RETRACT_CACHE` / `_SMOOTHING_CACHE` keyed on `id()`, which Python reuses after
  an object is freed; a new backend or smoothing matrix could receive another object's cached
  result (the intermittent test_backend_is_invoked_for_every_soft_ortho_constraint failure).
  Entries now hold the object and are matched by identity (tests/test_cache_identity.py).
- Armijo stalls on the projected energy (line search through a piecewise retraction); LARS gets
  lower. Not fixed here.
- Remaining compatibility aliases (`flow_simlr`, `FlowSiMLRModel`, `flow_whitener`) and
  `predict_simlr(allow_legacy_refit=)` were left; they are shims, not duplicates.

## D5 (Manuscript built from code, 2026-11-01)
status: built for owner review, uncommitted
owner decisions (2026-11-01): adopt the v0.3.0 caches; write generators for the six unbacked
caches; Wilcoxon over 12 dataset blocks with Holm; inline computed values; edit on top of the
owner's manuscript edits; "build the paper as it is ... a paper built from code as much as is
possible".
what changed:
- Every result number in the prose is a Quarto inline value from `paper/paper_numbers.py`
  (Markdown-valued, so math and code spans render); captions no longer carry typed numbers.
- Caches: main benchmark + ADNI sweep regenerated with v0.3.0; six previously unbacked caches
  regenerated by `scripts/paper_adni_ablations.py`; new `brca_er_loadings.json` from
  `scripts/paper_brca_loadings.py` (feature names rebuilt and verified equal to the cached data).
- Figures now drawn from caches: ADNI neuroanatomy (titles, panel B = data-driven leading
  regions), row-mean bars, BRCA waterfall. Still a pre-rendered PNG: fig-nsa-v-interpretability.
- Chunk fixes: pooled means exclude the paired positive reruns (`op_default`); the ADNI table no
  longer fills a missing sweep value with 0.095; the Pareto envelope is computed.
conclusions that changed (owner to review):
- ADNI-MCI best model: NED 0.236 (+55% over PCA), not LEND 0.252 (+65.8%).
- Neuroanatomy at q = 0.85: components lead with parietal/frontal/occipital, fusiform/temporal,
  and parietal/supramarginal regions; Braak-stage labels removed; left precentral loads 0.325
  (sensorimotor cortex is not spared).
- Support-shuffle control lowers single-axis R^2 for all six models tested; PCA rotation for 3/6.
- Shared+Private PCA 0.291 vs best multi-view 0.586 (was 0.618 vs 0.956).
- Sweep: best linear-family config 0.226 (SiMLR-LARS), not 0.241; mean-11 best 0.119 vs PCA 0.081.
- Discussion limitation 2 (row-mean) and 3 (extreme w) restated from the regenerated ablations.

## D4 (Exception audit, package defaults, prediction invariance, reviewer R1-R9, paper audit, 2026-11-01)
status: done
release: v0.3.0 (tag `v0.3.0`)
review input: /tmp/agy.md (R1-R9), each claim re-measured before acting

**Broad exception handlers: 37 -> 0.** Removed where they substituted a different algorithm or
value without a warning: FastICA consensus/initializer, encoder SVD/polar fallbacks, nsa_flow
retraction/polar/initializer fallbacks, zero W on pinv failure, certificate -> None, consensus
anchor/stats -> None (prediction then silently became batch dependent), Hungarian -> greedy,
line search initial energy -> step 0. Narrowed with a warning where the substitute is stated:
MAI view score (6 duplicated copies merged into `consensus.mai_score`; LinAlgError -> 0 with a
RuntimeWarning), line-search candidate rejection, lr tuner, ridge solve -> pinv, diagnostic SVD
-> NaN, nnh rank SVD, matplotlib guard, antstorch ImportError, Rscript (now warns).
`tests/test_implementation_standards.py` allows zero.

**Defaults in one place.** `src/pysimlr/defaults.py`: DEFAULT_ENERGY_TYPE="acc",
DEFAULT_NSA_W=0.5, used by every entry point, model and helper (deep/flow were "regression"
and 0.1 with no recorded reason). NED latent-collapse sweep (seeds 0-9, 42): 0/11 seeds
collapse under both positivities (was 4/11 and 3/11); the strict xfail was removed.

**Reviewer findings (verified first):**
- R2: `score`/R^2 was 1-NRMSE and the baselines scored in raw units (pca 0.97 vs simlr 0.64 at a
  +20 shift). `predict` now returns per-view `r2` and `mse` in each method's training-standardized
  space; `score` is mean `r2`. The benchmark's `reconstruction_mse` compared raw data with
  standardized reconstructions; now standardized for every method.
- Batch dependence (worse than R2 said): lend latents moved up to 2.2 and simlr(loo) 0.58
  between batch sizes. Fixed by freezing the training batch's consensus as an affine map
  (`consensus.capture_consensus_map`, residual ~6e-7); every method now batch invariant to 1e-6
  incl. N=1. The strict xfail in test_consensus_anchor_standardization was resolved.
- R3: inverse_transform returns input units and input shape (`utils.invert_preprocessing`),
  supports the baselines; `scale` added; build_simlr_pipeline no longer double-standardizes.
- R4: baselines use `utils.preprocess_data`; provenance replays; PCA records a seed only when
  its randomized solver ran.
- R5: `private_recon_weight` was never read; removed.
- R6: fit_structural_models routed through the registry (unknown names raised instead of
  silently running ned_simr).
- R7: presets excluded from `accepts`; L-BFGS presets accept only the keys they read;
  warn-once set replaced by Python's per-call-site dedupe.
- R1: ~25 script call sites migrated; plan gate 1 is a test.
- R8: AST test that every parameter of a registered fit is read; per-method batch-invariance,
  shift-invariance and provenance tests.

**Found along the way.**
- `learning_rate="auto"` ran 7 probe fits for torch_lbfgs/nsa_lbfgsb, which never read the
  learning rate (all candidates scored identically); skipped now. Median fit 3.55 s -> 2.00 s
  (LBFGS) and 3.70 s -> 2.68 s (LBFGSB) in the regenerated benchmark, identical results.
- `nsa_flow` column-quantile sparseness kept the same row in every column on exact magnitude
  ties: an orthonormal 2x2 (the PCA start of two standardized features) became rank 1 at
  sparseness=0.5 (35/200 nnh_embed draws, also at v0.2.19). Fixed upstream in nsa_flow 3.5.1
  (333a064); verified here: the pinned case passes, all 8 signed 2x2 Hadamard bases stay rank 2,
  nnh_embed 0/200 failures. The strict xfail became a regression test over the 8 bases;
  pyproject.toml now requires nsa_flow>=3.5.1.
- tests/test_coverage_deep_simlr_ext.py::test_imports_fallback deleted: it re-imported
  pysimlr.simlr, rebinding the package attribute to the module for every later test in the
  worker, and asserted attributes that never existed.
- scripts/run_deep_benchmark._execute_tasks returned surviving rows (an empty frame when every
  task failed) and the caller wrote them as the benchmark; it now raises after the sweep.
- L-BFGS-B at 4 iterations turns a 2e-6 input difference into a 0.15 R^2 change; agrees by 20.

**Paper.** Built with `make paper` from the existing caches (see the audit report for
build output). Audit report: reports/paper_audit_2026-11-01.html. Headline results:
- ~25 verified mismatches between prose and caches, several changing a conclusion (abstract
  LEND ADNI-MCI 0.252 vs cache 0.242 / table ranks Flow first; Shared+Private PCA 0.618 vs 0.291;
  BRCA, KIRC, Heart, F1 claims; NEDPP control row absent from its CSV; panel A3 plots the
  row-ICA solution because its "varimax" key does not exist).
- The "1,214-configuration sweep" has 1,184 rows of which 224 failed (every linear simr
  variant with mixing="svd": NameError align_column_signs) -- 960 configurations.
- Six caches have no generating script in the repository.
- The sweep cache (f4e53a6) does not reproduce at HEAD v0.2.19: PCA mean-11 R^2 0.0728 vs cached
  0.0804; the paper's best linear config (Lookahead, nc, ica) 0.187 vs 0.2407. v0.3.0 matches
  HEAD. Regenerated v0.3.0 sweep: 1214/1214 configurations ok; best CogPC1 NED 0.262, LEND 0.261,
  Flow 0.246, NEDPP 0.235, SiMLR 0.209, PCA 0.178 (single seed).
- scripts/run_adni_mci_grid_sweep.py hard-codes /Users/stnava/data/repos/pysimlr/src on sys.path.
- Appendix Wilcoxon tests use 36 dataset-seed pairs as independent with no Holm correction; at
  the 12-block level only SiMLR-LBFGS/LBFGSB stay below 0.05 vs PCA.
- Regenerated with v0.3.0 into the scratchpad (not committed): Friedman linear chi2 30.5
  (p 1.7e-4) vs 31.5; LEND still first by mean rank (3.71); ADNI-MCI deep values moved
  (NED 0.236, Flow 0.229, LEND 0.220); linear and baseline values identical.

**Verification.** `pytest -m ""`: 1591 passed, 4 xfailed, 0 failed (before the ADNI sweep
regeneration and the paper build). `antsxqc.lint src/pysimlr`: 0 findings.

**Open (owner decisions).**
- Adopt the v0.3.0 caches for the paper, and correct the prose either way (D5).
- Commit the six missing cache generators, or remove the claims they back.

## D3 (Central method interface and correctness pass, 2026-11-01)
status: done
release: v0.3.0 (tag `v0.3.0`; breaking API)
plan: docs/plans/2026-11-01_central_interface.md
contract: docs/api_contract.md
files added: src/pysimlr/api.py, docs/api_contract.md, docs/plans/2026-11-01_central_interface.md, tests/conftest.py, tests/test_api_registry.py, tests/test_random_state.py, tests/test_implementation_standards.py, tests/test_readme_examples.py
files changed (src): simlr.py, deep.py, flows.py, optimizers.py, sklearn.py (rewritten), nnh.py, sparsification.py, nsa_backend.py, visualization.py, __init__.py, benchmarks/runner.py, benchmarks/deep_ranking.py, benchmarks/shared_private_sweep.py, benchmarks/__init__.py
files changed (other): README.md (Quick Start rewritten), configs/benchmark_{smoke,full}.yaml, scripts/run_deep_benchmark.py, scripts/generate_nsa_report.py, paper/build_paper_experiments.py, docs/comparison_simlr_lend{,_basic}.qmd, 251 keyword renames across tests/, scripts/, docs/*.qmd by an AST rewriter (calls to the nine public entry points only)

**What and why.** Five call sites translated between the entry points' vocabularies by hand
(benchmark runner, `nnh_embed`, `extend_simlr_embedding_with_new_modalities`, the four sklearn
wrappers, `scripts/run_deep_benchmark.py`), and each had drifted. Now:
- `api.METHODS` registers `simlr`, `simlr_lbfgs`, `simlr_lbfgsb`, `lend`, `ned`,
  `ned_shared_private`, `flow_v`, `pca` (joint), `nsa_pipeline`; accepted parameters are derived
  from each native signature. `fit` records `requested` / `used` / `ignored` (with reason) /
  `defaults` / `random_state` / `budget_unit`; a parameter for another method warns once
  (`strict=True` raises); a name no method has raises.
- One name per concept at the native entry points: `iterations` (deep/flow `epochs` removed),
  `sparseness`, `random_state` (`init_seed` removed). Optimizer keys that collided with other
  parameters renamed: `k` -> `lookahead_k`, `tol` -> `lbfgsb_tol`, `beta` -> `rmsprop_beta`.
  Deep/flow `**kwargs` catch-alls removed; unknown keywords raise everywhere.
- `random_state` on deep/flow runs the whole fit on a forked, seeded RNG stream (standardization
  jitter, weight init, dropout, DataLoader shuffle); the caller's global state is restored.
- `MultiViewSiMR` replaces `SiMLREstimator`/`SiMLR`/`SiMLRTransformer`/`LENDTransformer`/
  `NEDTransformer`/`FlowSiMLRTransformer`; `view_sizes` makes multi-view input usable in
  scikit-learn CV; `GridSearchCV` over `method` works.

**Defects fixed (verified by execution or by reading the cited code):**
- README Quick Start: 2 of 3 snippets wrong (`simlr(optimizer=, random_state=)` ran the default
  optimizer unseeded; `SiMLR(optimizer=, sparseness=)` raised `TypeError`).
- `simlr` never forwarded `positivity` to the optimizer: `nsa_lbfgsb` under
  `positivity="positive"` ran with no lower bound (spy: optimizer saw `positivity='either'`,
  `nonneg=False`). Changes `simlr_lbfgsb` benchmark results under positive mode.
- Single-view `SiMLREstimator.transform` projected unscaled X (max abs diff 30.0 vs the library
  predictor). LEND/NED wrappers' `epochs` argument never reached training.
- `init_seed` switched `initialization_type="pca"` to `"random"`; `random_state` does not.
- `deep_simr` dropped `optimizer_type`; unknown deep `optimizer_type` silently became Adam;
  `lars` without `torch_optimizer` silently became `larslow` (now warns).
- `simlr(patience=...)` drew a spurious unknown-parameter warning; `flow_simr_v` accepted `tol`
  and `patience` and ignored them (now wired to the training loop).
- `flow_simr` discarded its convergence report; it now returns it (still no `v`; not registered).
- `nnh_embed`: `constraint` was never forwarded to `simlr` (default now `None` = `simlr`'s
  default, so default behaviour is unchanged); the flow path replaced `energy="acc"` with
  `"regression"` (now passed as given -- changes `nnh_embed(use_flow=True)` results); the
  pinv-failure zero fallback for `w` removed.
- Benchmark runner: `filter_kwargs` removed (it dropped valid aliases); two `except Exception`
  sites removed; `convergence_report` certified deep/flow bases against the runner's
  `energy_type` default ("acc") rather than the model's own, and always projected onto V >= 0
  even for signed bases -- both now read from the fit's parameter record.
- `deep_ranking.critical_difference` substituted 2.948 / 2.569 on any scipy failure (the
  9-model value is 3.102); Wilcoxon failures became p = 1. Both now propagate.
- `paper/build_paper_experiments.py` called `ned_simr_shared_private(k_shared=, k_private=)`.
- Duplicate `generate_all_architecture_graphs` in visualization.py; three `get('V') or get('Y')`
  tensor-truthiness chains (latent: current backends return only `Y`); mutable defaults on the
  deep entry points and models.

**Not a defect (survey claim checked and rejected):** `consensus.py`'s
`warnings.filterwarnings` is inside `warnings.catch_warnings()`.

**Behaviour changes to note for results.** Deep/flow fits are now seeded per fit, so numbers
under a given global seed differ from 0.2.19. `run_single_experiment` passes `random_state=seed`
to every method. The baselines' supervised `pipeline_test_score` is gone (they are scored by the
shared estimator like every method). The `iterations` liveness test needed an explicit random
start: from the PCA start, 1 and 30 iterations return the same best iterate (consistent with
the open "iteration does not improve on SVD init" item in CORRECTNESS_AUDIT.md).

**Verification.** Fast suite 1405 passed / 6 xfailed before; after: 1526 passed / 6 xfailed.
`pytest -m ""`: 1557 passed, 7 xfailed, 0 failed (136.7 s, 6 workers). `python -m antsxqc.lint
src/pysimlr`: 0 findings. `python -m pysimlr.benchmarks.runner --config
configs/benchmark_smoke.yaml` completed (run in a scratch directory). One
`run_deep_benchmark.run_experiment_task` per model on the Linear synthetic case completed.
Not run: the full `scripts/run_deep_benchmark.py` sweep and the paper build; the ~100 other
scripts were renamed mechanically but not executed.

**Open items.**
- D4: 37 broad exception handlers (per-file counts pinned in test_implementation_standards).
- TODO(human): method defaults differ (`energy_type` acc vs regression, `nsa_w` 0.5 vs 0.1,
  `topology` star vs loo); recorded per fit, not aligned.
- TODO(human): `MultiViewSiMR.score` = held-out reconstruction R^2; confirm as the default
  model-selection criterion.
- The paper protocol dict sends `nsa_iterations` to linear methods and `consolidate` to deep
  ones; the new `Ignored_Params` column shows neither is applied (it never was).
- `nn.Module` constructors and sparsification helpers keep `sparseness_quantile`.
- `scripts/unified_real_cv_comparison.py` keeps its own `filter_kwargs` dispatch.
- Duplicated helpers (MAI update x6, `set_projection_schedule` x5, polar projection x3).
- Audit items A1-A4 in docs/audit/AUDIT_2026_09.md have no recorded resolution.

## D2 (Remove the silent backend fallback in `svd.varimax` / `svd.varimax_rotation`, 2026-10-31)
status: done
release: v0.2.19 (tag `v0.2.19`, the commit that adds this file)
files touched: src/pysimlr/svd.py, src/pysimlr/nsa_backend.py, tests/test_varimax.py, docs/tracking.md, pyproject.toml, src/pysimlr/__init__.py
tests added: tests/test_varimax.py (8 dispatch tests; one existing test repaired)

**What and why.** Both functions wrapped the nsa_flow call in `except Exception: pass` and fell
through to the PyTorch implementation. Measured on nsa_flow 3.5.0, that catch was silently
rescuing two inputs the backend rejects -- NumPy arrays (`TypeError` inside `torch.eye`) and
batched 3-D input to `varimax` (`RuntimeError`) -- and would equally have hidden any genuine
backend failure. The two paths also disagreed on degenerate shapes (`k < 2` or `p < 2`).

Now:
- `_as_loading_matrix` converts input with `torch.as_tensor` and refuses non-2-D input with
  `ValueError` (the PyTorch path's convergence test summed the criterion across a batch, so
  batching was never supported by either path).
- Degenerate shapes return the identity rotation / unrotated input on every path.
- The PyTorch implementations are explicit private functions, `_varimax_rotation_torch` and
  `_varimax_torch`, used only when the loader resolves no backend.
- `nsa_backend._resolve_backend_function` checks with `inspect.signature(...).bind` that the
  backend accepts the keywords pysimlr passes. An incompatible backend is refused once, with a
  `RuntimeWarning` and an entry in `INCOMPATIBLE_BACKEND_FUNCTIONS`, reported by
  `backend_report(extended=True)["incompatible_functions"]`.
- Backend errors propagate.
- Same file, same pattern: `load_gradient_mapping` / `load_lbfgsb` caught `Exception` around a
  submodule import; narrowed to `ImportError`.

**User-caught gap.** The v0.2.18 report flagged the catch as "worth tightening later"; the owner
asked for it fixed now. Found while doing so:
`test_pysimlr_varimax_matches_nsa_flow_linalg_varimax_exactly` claimed to test the PyTorch
fallback but called the public `varimax_rotation`, which dispatches to nsa_flow when installed --
it compared nsa_flow with itself. It now calls `_varimax_rotation_torch` / `_varimax_torch`
directly (agreement with nsa_flow to 1e-10).

**Verification.** `pytest -m ""` (all tests including slow) and `python -m antsxqc.lint .` --
counts in the v0.2.19 commit body. Not run: a mutation check of the propagate test against the
v0.2.18 code (the new tests import private functions that did not exist there); the test fails
against a catch-and-fallback by construction.

**Open items.**
- 46 `except Exception` sites remain in `src/pysimlr` (deep 11, simlr 8, flows 6,
  sparsification 5, optimizers 5, utils 3, nnh 2, benchmarks/runner 2, benchmarks/deep_ranking 2,
  consensus 1, interpretability 1). Not audited; some may be legitimate boundary handlers that
  record provenance, others silent fallbacks. Examples seen in passing: `_ica_basis_for_view`
  falls back to the SVD basis if FastICA raises; `nnh_embed` substitutes a zero `w` if `pinv`
  raises.
- nsa_flow's `varimax_rotation` emits a torch `UserWarning` (`float()` on a tensor requiring
  grad) when given a grad-tracking input; upstream in nsa_flow, not fixed here.

## D1 (Consistent `varimax` across linear, deep and flow entry points, 2026-10-31)
status: done
release: v0.2.18 -- commit 91cd48d, tag `v0.2.18` (spec committed as a temporary note in 259c4ff, deleted in 91cd48d)
files touched: src/pysimlr/simlr.py, deep.py, flows.py, nnh.py, README.md, docs/theory/SIMLR_THEORY.md (section 9.4), 7 test files
tests added: tests/test_varimax_consistency.py (48 tests)

**What and why.** `simlr` exposed `varimax` (added in 0.2.17, 5cc5fcd / 1768d3b), but the deep
and flow entry points initialized through `initialize_deep_encoders`, which had no such argument,
and their `**kwargs` dropped it silently. Under `positivity="either"` the flag was also inert for
`pca` and `gcca` starts.

- Every entry point takes `varimax: Optional[bool] = None` and forwards it to
  `initialize_simlr`. Deep/flow entry points now warn on unrecognised keyword arguments.
- `resolve_varimax`: `None` rotates only a non-negative basis; explicit `True`/`False` wins.
- `gcca`: one rotation `R = varimax_rotation(shared_u)` of the joint MAXVAR subspace before the
  per-view regressions, so views are column-aligned by construction. `pca`/`perturbed_pca`:
  per-view Varimax then `_align_multiview_basis_columns`.
- `initialization_type="varimax"` with explicit `varimax=False` raises; mismatched sample counts
  raise `ValueError("Size mismatch ...")` before fitting.
- No post-hoc rotation of trained deep/flow bases: the decoders and RealNVP couplings are not
  O(k)-equivariant.

**Decision (2026-10-31): signed rotation is opt-in.** The spec asked for `varimax=True` as the
default for signed bases too. Measured consequences of that default led the owner to choose the
positivity-dependent default instead:
- NED latent-collapse sweep (`test_latent_collapse` generator, seeds 0-9 and 42, off-diagonal
  covariance norm >= 0.5 counts as collapsed): signed 4/11 unrotated vs 6/11 rotated;
  non-negative 5/11 unrotated vs 4/11 rotated.
- Linear `simlr` (`_coupled` fixture, 6 seeds, lars, 20 iterations): the rotated start was never
  worse (it fixed seed 3, latent recovery 0.504 -> 0.990), but at `orthox0.9x1` one seed's fit
  fell from 0.986 to 0.755.

**Pre-existing failures repaired in the same release** (all failed before Varimax existed, at
f5484c7):
- `test_basis_metric_invariance` (2): float32 fixtures put ~1e-7 roundoff into `u @ m` before the
  metric, against a 1e-9 assertion; fixtures are now float64 (worst difference 2.2e-16). Casting
  inside `cross_val_metrics` was tried and does not fix it.
- `test_positive_constraint_does_not_cripple_latent_recovery` (3, slow): the default consensus
  moved to newton and this test never pinned it; pinned to `svd` as its fast sibling is.
- `test_ned_advantage_contract` (slow): strict xfail that now passes; over 8 seeds NED latent
  recovery 0.700 vs SiMLR 0.578, held-out R^2 0.827 vs 0.725. Mark removed. The trainers still
  select the best state by training loss, not a latent-quality criterion.
- `test_nedpp_advantage_contract` (slow): R^2 margin over SiMLR +0.045 over 8 seeds against the
  claimed +0.05 (it held at 85ad3e2 by +0.064, fell to +0.024 at f4e53a6); marked strict xfail
  with these numbers rather than relaxed. NEDPP's latent-recovery lead is real (0.552 vs 0.434).

**Verification.** `pytest -m ""`: 1418 passed, 7 xfailed, 0 failed; `antsxqc.lint`: 0 findings
(recorded in the 91cd48d commit body). Mutation check: removing the deep-path forwarding fails
24 of the new tests; removing the gcca rotation fails 11.

**Open items.**
- Model selection on a latent-quality validation criterion for the deep trainers (blocks a
  durable NED/NEDPP advantage contract).
- The `orthox0.9x1` seed-1 drop after a rotated signed start is unexplained.
