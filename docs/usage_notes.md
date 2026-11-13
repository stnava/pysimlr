# Usage notes

These are practical notes for fitting and interpreting `pysimlr` models. They describe the
package as it is on `main`. The result contract is in `docs/api_contract.md`, and every default
is set in `src/pysimlr/defaults.py`, which also records the evidence behind it.

## The default linear fit

```python
import numpy as np
import pysimlr

rng = np.random.default_rng(0)
z = rng.standard_normal((120, 3))
views = [z @ rng.standard_normal((3, 40)) + 0.5 * rng.standard_normal((120, 40)),
         z @ rng.standard_normal((3, 30)) + 0.5 * rng.standard_normal((120, 30))]

res = pysimlr.fit("simlr", views, k=3, random_state=0)
assert res["convergence"]["converged"], res["stop_reason"]
```

The defaults are:

| setting | default |
|---|---|
| energy | `nc` (a bounded correlation) |
| consensus | `newton` mixing, leave-one-out topology |
| optimizer | `hybrid_adam`, with `learning_rate="auto"` |
| budget | 300 sweeps, stopping at a relative plateau of `tol = 1e-4` |
| orthogonality weight | `nsa_w = 0.5` |
| sparseness | 0.5 |
| collapse penalty | weight 1.0 |

Leave `iterations` and `tol` unset unless you have a reason. A short budget (for example 25)
stops most fits before convergence. Results compared at such a budget compare optimizer
trajectories, not solutions.

## Always check convergence

`res["convergence"]["converged"]` and `res["stop_reason"]` tell you how the fit ended. The stop
reason is one of:

| `stop_reason` | meaning |
|---|---|
| `converged` | the energy reached its plateau |
| `max_iter` | the budget ran out first |
| `no_descent` | the step stopped reducing the energy |
| `non_finite` | a NaN or infinity appeared |
| `closed_form` | the method has no iterations (for example `pca`) |

**Known case:** `learning_rate="auto"` probes the grid 1.0 … 0.001 with 3-sweep trials. On small
or narrow views it can choose 1.0, and the fit then stops with `no_descent` after a few sweeps.
- `res["learning_rate"]` shows the chosen value.
- If a fit is not `converged`, refit with a fixed, smaller step, for example
  `learning_rate=0.01`.
- Among the converged fits, keep the one with the lowest `res["best_energy"]`.

## Wide views: training coupling is not evidence

When a view has at least as many features as training samples (p ≥ n), its scores can be fitted to
the other views' noise.
- `simlr` reports `res["coupling_train"]`, |corr(view score, consensus)| per view and component;
  `res["view_width_ratio"]` (p / n); and `res["wide_views"]`.
- It warns (`pysimlr.WideViewWarning`) when any view is wide.

Sparsity doesn't remove the problem, because the support is *chosen* from all p features. On IBD
stool metabolomics (four 500-feature platforms plus 156 species, n = 165 training subjects, k = 5):

| sparseness | training coupling | held-out coupling |
|---|---|---|
| 0.5 | 1.00 | 0.58 |
| 0.8 | 0.99 | 0.57 |
| 0.9 | 0.92 | 0.56 |
| 0.95 | 0.83 | 0.61 |

Independent noise views (three 120-feature views, n = 50) reach a training coupling of 0.88 and a
held-out coupling of 0.03.

For a wide view, judge coupling on held-out rows with `pysimlr.heldout_coupling(res, held_out_views)`
(`None` for a missing view). It returns, for training and held-out rows side by side:
- per view and component, |corr(latent, u)|;
- per component, the mean pairwise cross-view |corr|, and `cross_view_gap` (training − held-out).

Measured values:
- IBD: 1.00 in training against 0.58 held-out;
- noise views with p ≥ n: 0.88–0.91 against 0.04–0.05;
- a genuine shared signal: gap ≤ 0.03.

Alternatively, reduce the features on the training split before fitting (below), and still check
`heldout_coupling` afterwards.

## Wide-view workflow: coarse-to-fine split-scale readout (`probe_views`), shrinkage, and opt-in screening

Wide multi-view studies (`p_m >= n`, such as `1,000+` connectome edges or white-matter tract metrics
on `n < 150` subjects) face two distinct problems:

1. **Consensus encoder overfitting (`fit`):** feeding `p_m >= n` columns directly into `fit` lets
   `V_m` fit within-view noise (`D25`), inflating training coupling while degrading held-out
   cross-view agreement (`heldout_coupling`).
2. **Unsupervised screening bias (`screen_wide_views`):** pruning raw wide matrices before `fit` via
   within-view singular subspaces or variance caps retains broad normative factors (age, brain size,
   scanner contrast) that span hundreds of features while discarding focal cross-view pathology edges
   before `fit` ever runs (`D30`).

### 1. Preferred pattern: coarse-to-fine split-scale readout (`probe_views` / `probe_test_views`)

Whenever a wide modality admits a natural physical or anatomical summary at `p_macro < n` (for
example `44`–`70` unilateral DKT regional node strengths, lobar/tract summaries, or network hubs),
separate **consensus learning** from **fine-grained attribution**:

- **Stage 1 (Consensus fit on `p_macro < n`):** run `fit` / `loading_inference` on the
  well-conditioned macro views (`wide_views == []`), keeping unilateral hemisphere resolution so
  lateralised signal is preserved.
- **Stage 2 (Frozen fine-grained readout via `probe_views`):** pass the full-width matrices
  (`p_fine >> n`, or `None` for views kept at macro resolution) as `probe_views` to `feature_maps`,
  `loading_inference(..., map_options={"rw_shrinkage": "auto", "probe_views": ...})`, and
  `feature_relevance(..., probe_views=..., probe_test_views=...)`. Every fine feature is scored
  strictly against the **other** views' macro latent scores (`target="other_views"`) using
  Ledoit–Wolf shrunk relative weights (`rw_shrinkage="auto"` / `shrinkage="auto"`), with zero
  circularity and zero pre-fit feature loss.

**Measured multi-scale comparison** (`tests/test_coarse_to_fine_probe_views.py`: `n_train = n_test =
93`, `44`-node / `946`-edge connectome with `15` focal cross-view edges plus two strong `240`-edge
normative factors, 5 seeds):

| workflow | `cross_view_gap` | held-out focal `|r|` | focal edge ROC AUC (all `946`) |
|---|---|---|---|
| **Coarse-to-fine (`44` nodes → `946` edges via `probe_views`)** | **0.060** | **0.953** | **1.000** (`100%` retained) |
| `screen_wide_views` (`max_width_ratio=0.75`) | 0.081 | 0.863 | 0.477 (`2.7%` retained) |
| `screen_wide_views` (`max_width_ratio=0.25`) | 0.068 | 0.121 | 0.488 (`0.0%` retained) |
| Unscreened wide `fit` (`p = 946` directly) | 0.225 | 0.946 | 1.000 (`3.8×` overfit gap) |

```python
# Stage 1 + Stage 2 in one pass: fit & align subsamples on well-conditioned macro views (p_m < n),
# while computing shrunk relative-weight and structure maps + CIs on arbitrary-width fine views.
# Pass None in probe_views for any modality already at its desired resolution.
inf = pysimlr.loading_inference(
    "simlr",
    macro_train_views,
    k=5,
    fraction=0.5,
    n_subsamples=50,
    maps=("structure", "relative_weight"),
    map_options={"rw_shrinkage": "auto", "probe_views": fine_train_views},
    positivity="positive",
    random_state=0,
    n_jobs=-1,
)
res = inf["result"]
assert res["wide_views"] == []

# Check cross-view generalisation on the held-out macro views
hc = pysimlr.heldout_coupling(res, macro_test_views)

# Fine-grained training + held-out tier replication across all p_fine features (λ recorded)
rel = pysimlr.feature_relevance(
    res,
    macro_train_views,
    macro_test_views,
    probe_views=fine_train_views,
    probe_test_views=fine_test_views,   # None entry when a modality is absent in the held-out cohort
    outcome=y_train,
    outcome_test=y_test,
    feature_names=fine_feature_names,
    view_names=view_names,
    shrinkage="auto",
    refit_fraction=0.5,
    random_state=0,
)

# Brain surfaces, shift plots, and relevance forests work directly on the probed fine features
pysimlr.plot_brain_map(res, inf=inf, component=rel["component"], view=0, feature_names=fine_feature_names)
pysimlr.plot_feature_relevance(rel, view=0)
```

### 2. Shrunk relative weights (`rw_shrinkage="auto"`, `shrinkage="auto"`)

Even without `probe_views`, classical Johnson relative weights invert `R_XX` and emit `NaN` whenever
`p >= n` — including half-sample `loading_inference` refits (`n_sub = 0.5 * n` when `0.5 * n <= p <
n`) and smaller external test cohorts (`n_test <= p`).
- `"auto"` is bit-identical to classical Johnson (`λ = 0`) when full-rank `p < n`, switches to
  Ledoit–Wolf shrinkage otherwise (`"ledoit_wolf"` from `X` alone in `feature_relevance` so the
  permutation null is exact; `"ledoit_wolf_joint"` on `[X, t_j]` in `feature_maps`), and records the
  `λ` used (`rel["shrinkage_used"]`, `inf["maps"]["relative_weight"]["rw_shrinkage"]`).
- Direct `feature_maps` calls keep `rw_shrinkage=0.0` by default so rank deficiency surfaces
  explicitly (`NaN` + warning) unless `rw_shrinkage="auto"` is requested.

### 3. Fallback when no domain summary exists: opt-in `screen_wide_views`

When a modality is an unstructured flat assay with no physical macro grouping (for example untargeted
metabolomics peaks), `pysimlr.screen_wide_views` and `pysimlr.apply_view_screen` provide an opt-in,
strictly outcome-blind pre-fit gate on the training rows (`q_bh < alpha` against a pooled
row-permutation null, Westfall–Young omnibus view test with Holm step-down, half-sample stability,
collinearity pruning, and width cap `floor(max_width_ratio * n)`):

```python
gate = pysimlr.screen_wide_views(
    train_views, k=4, max_width_ratio=0.3, max_collinearity=0.92,
    feature_names=feature_names, view_names=view_names, random_state=0,
)
train_gated = gate["views"]
test_gated = pysimlr.apply_view_screen(gate, test_views)
```

**Practical rules.**
- **Prefer domain macro-views + `probe_views` over `screen_wide_views` whenever anatomy or ontology
  exists.** Do not collapse left/right hemispheres into bilateral averages if lateralised signal
  matters; keep unilateral macro ROIs (`p < n`) and let `rw_shrinkage="auto"` handle `p >= 0.5 * n`
  during half-sample refits.
- **Screen inside every CV fold, never on pooled `train + test` rows.** If you use
  `screen_wide_views`, fit it on the fold's training rows and slice held-out arrays with
  `apply_view_screen(gate, test_views)`, then inspect `gate["summary"]` (`n_pass_bh` vs `p_out`,
  `view_coupled`) beside `pysimlr.heldout_coupling`.
- **Separate physical global amplitude before standardisation when targeting regional topology.**
  When a modality is scaled by a subject-wide multiplier (mean grey-matter CBF in ASL, reference
  SUVR in PET, intracranial volume in morphometry), divide regional columns by that physical scalar
  before z-scoring (and optionally hold global scalars in a compact view) rather than regressing
  out post-z-score row means.

## What to interpret: structure and loadings, not `v`

`res["v"]` is the encoder. It is weakly identified in a view's low-variance directions, so two
fits that agree on `u` can have different `v`. For interpretation, use the objects that are
stable:

- **`res["structure"]`:** one matrix per view (features × k) of correlations between each
  feature and each shared component (structure coefficients).
- **`res["loadings"]`:** the two-stage loadings, the OLEE summary of `structure`.
- **Both are in a varimax-rotated, sign-anchored frame.** The rotation is
  `R = res["loading_rotation"]`, and `structure` is exactly corr(X, `u @ R`). `res["u"]`,
  `res["latents"]` and `predict(...)["u"]` are *not* rotated. To pair component j of the loadings
  with subject scores, use `res["u"] @ R` (or `predict(res, new)["u"] @ R`). Do not rotate
  `structure` again.
  - With `positivity="positive"` each feature is hard-assigned to one component, so supports are
    disjoint, and `res["loading_margin"]` gives the assignment margin.
  - With signed fits (`positivity="either"`, the default) the loadings are dense, and
    `loading_margin` is `None`. Do not threshold signed loadings by eye; use the intervals below.

## Alternative feature maps (`pysimlr.feature_maps`)

`structure` measures association, so on a view dominated by one global factor it ranks
features by how much they share that factor, not by what drives the component. Cortical
thickness, where every region thins together, is the typical case.

`pysimlr.feature_maps(res, views)` returns four alternatives beside it. All are in the same
rotated frame as `structure`, and all are measured against the *other* views' scores
(`target="other_views"`). That target matters: a linear fit's own score is exactly X·V, so
regressing it on X would only return the unstable encoder.

| map | what it measures | use it when |
|---|---|---|
| `partial` | partial correlation of x with the component given every other feature, from the inverse of the (Ledoit–Wolf shrunk) correlation matrix | you want each feature's unique contribution; works with p ≥ n |
| `relative_weight` | Johnson's relative weights: each feature's non-negative share of R²(component ~ X); the shares sum to that R² | features are collinear and you want an importance split that stays positive; default `rw_shrinkage=0.0` needs p < n (otherwise NaN, with a warning), or pass `rw_shrinkage="auto"` for p ≥ n and half-sample refits |
| `global_partial` | `structure` after regressing out the view's PC1, its row mean, or a covariate you pass (`control=`, settable per view) | you know the view has a global size factor |
| `component_partial` | Haufe et al.'s forward pattern, corr(X, T)·corr(T)⁻¹ | components are correlated (`mixing_algorithm="avg"`); otherwise it equals `structure` |

**Unobserved rows (`missing_rows="mask"`).** Each map is computed on the rows where the view is observed and its
other-views target exists; the target at a row is the mean of the standardized scores of the other views observed
there. A row observed in one view only enters no map; a partially observed row enters with a target from fewer
(noisier) views. Pass `min_rows` (default 10) to set the smallest usable row count: below it, or for a constant
feature or latent column, the entries are NaN with a `MissingRowsWarning`, never 0. `feature_maps(..., return_info=True)`
returns the rows behind each view and feature (`n_eff`, `n_eff_feature`) and the reason for any NaN (`reasons`);
`loading_inference` reports the same under `inf["maps"][kind]`, plus `oos_coupling_n`. Before v0.31.1 this setting
returned all-zero maps and `oos_coupling == 0.0`; recompute any such output. `feature_relevance` and
`component_attribution` still need complete rows and now raise on all-NaN rows.

**Use them with inference.** `loading_inference(..., maps=(...), map_options={"rw_shrinkage": "auto"})`
refits them with the same subsamples and reports intervals, selection frequency, `rw_shrinkage`,
and each map's own component stability (`inf["maps"][kind]`).

**Displays.** All of them read one shared encoding, `feature_map_values`:
- **Values:** |map|, or for `relative_weight`, percent of the variance the view explains.
- **Scale:** each map gets its own scale from 0 to its maximum, in `map_colormap(view)`.
- **Selection:** ≥ 80% of refits.

That way a table and a brain show the same thing:
- `plot_feature_map_shift(res, inf=inf, view=m)`: rank change from structure to relative weight.
  Read it first, because it answers "what moves once the global factor is shared out?"
- `plot_feature_map_grid(res, inf=inf, view=m, split=..., groups=...)`: the tabular twin of a
  brain map. `split` pairs hemispheres and `groups` orders by lobe. It works for any features and
  is drawn by antsxviz (`region_table_figure`), so it needs the `viz` extra.
- `plot_brain_map(res, inf=inf, component=j, view=m)`: the same encoding painted on the
  fsaverage cortex (inflated, sulcal shading; lateral, medial and ventral views; selected regions
  outlined).
  - Drawn by antsxviz, the ANTsX house renderer. Install it with `pip install 'pysimlr[viz]'`;
    without it the function raises an ImportError naming the extra.
  - Feature names map to DKT cortical regions through `cortical_region_name`, or `region_names=`
    if you pass one. Features that are not cortical regions (tissue totals, lobes) are listed in
    the subtitle, never silently dropped.
  - `brain_rows` returns the antsxviz rows for custom compositions.
- `plot_brain_map(res, view=m, atlas="hoa" | "jhu" | "tracts" | "tract_sheets")`: the same encoding on deep
  grey matter (Harvard-Oxford), white-matter regions (JHU) or 20 tracts (tubes or medial sheets) as lit 3D
  structures inside a glass brain (antsxviz scenes), one row per map.
  - Feature names are antsxmm wide-table columns as they are (`FA_HOA_<region>`, `vol_<region>hoa_by_region`,
    `FA_JHU_labels_edited_<region>`, `troi_fa_mean_<tract>`, `bundle_<metric>_<tract>`; a `DTI+` prefix is fine);
    `region_names={feature: atlas feature name}` renames others. Features not drawn are listed in the subtitle.
  - MNI by default; pass a subject's own `labels=`, `brain_mask=`, `tracts=`/`sheets=` (antsxdwi bundles) to draw
    in that subject's space.
  - Colour limits: min..max of each map by default, `scale="zero_max"` for 0..max, `clim=` to set them; the same
    options apply to the DKT surface and to `plot_feature_map_grid`, so twins stay identical.
- `plot_brain_map(res, view=m, atlas="yeo")`: Yeo-17 homotopic parcel values (`rsf_yeo_alff_ROI<n>_*`,
  `rsf_yeo_falff_*`; `rsf_yeo_homotopic_*` paints both parcels of a pair) on the fsaverage cortex, same views and
  colour-limit options as the DKT surface (antsxviz >= 0.3.0).
- `plot_connectome(res, view=m)`: a component's map over network-pair features (`rsf_yeo_netconn_{a}_2_{b}`) as a
  Yeo-network connectome (3D, chord and block matrix).
  - ADNI output: `scripts/brain_region_maps.py` → `docs/figures/adni_dkt_feature_maps.png` and
    `_grid.png`.
- `plot_top_features(kind=...)` and `plot_feature_map_comparison` show one map or all of them at
  once.

**Evidence** (`scripts/feature_map_check.py` → `docs/audit/evidence/feature_map_check.csv`):

- **Planted.** View 0 has 12 driver features; non-drivers load a global factor that correlates
  0.7 with the signal (PC1 = 70% of the variance).

  | map | support AUC | stability | no-global design |
  |---|---|---|---|
  | `structure` | 0.79 | — | — |
  | `relative_weight` | 0.97 | 0.92 | −0.007 |
  | `partial` | 0.96 | 0.88 | −0.007 |
  | `global_partial` | 0.95 | — | −0.10 |
  | `component_partial` | 0.81 | — | — |

  The last column is the AUC difference from `structure` on the design without a global factor.
  - The pre-set rule passed for `relative_weight` and `partial`.
  - `global_partial` failed it: removing PC1 harms a view that has no global factor.
- **ADNI-MCI** (Figure 3.9 fit; 20 half-sample refits).
  - **DiReCT thickness.** `relative_weight` ranks L entorhinal, L inferior parietal, R
    entorhinal and L fusiform first, with stability 0.86 against `structure`'s 0.93. Only
    L entorhinal is selected in ≥ 80% of refits.
  - **Volume / surface area (160 labels, p/n ≈ 0.4 at half samples).** The alternatives are much
    less stable than `structure`: `partial` 0.55, `relative_weight` 0.71, `structure` 0.99.
    They rank CSF volume first.
  - **Held-out CogPC1 R² of each map's top 10 features.**
    - On the full training set: DiReCT 0.18 (`partial`) and 0.17 (`relative_weight`) against
      0.16 (`structure`); vol/area 0.24 against 0.17.
    - In half-sample refits (median): DiReCT 0.17 against 0.15; vol/area 0.16 against 0.17.

**Recommendation.**
- On a view with a global factor, read `relative_weight` (or `partial`) beside `structure`, and
  trust a feature only if `loading_inference` selects it.
- On wide or weakly identified views, `structure` remains the most stable map.
- Nothing here changes `structure`, `loadings` or any default.

## Which features are relevant: a hard cutoff (`pysimlr.feature_relevance`)

A feature map is continuous and doesn't say which features matter. `feature_relevance` turns a
component into a decision. The rule is stated before looking:

```python
rel = pysimlr.feature_relevance(res, train_views, test_views,      # None for a view missing at test
                                outcome=y_train, outcome_test=y_test,  # picks the component; adds held-out r
                                views=(1, 2), feature_names=names,
                                groups={"lobe": lobe_of, "bilateral": pair_of})
rel["features"], rel["groups"], rel["summary"]      # pandas frames
fig = pysimlr.plotting.plot_feature_relevance(rel, view=1, grouping="lobe", brain=True,
                                              separate=lambda n: n.endswith("tissues"))
fig = pysimlr.plotting.plot_brain_map(None, view=1, tiers=rel)   # tiers on the cortex only
```

- **Statistic.** For view m on component j:
  - `feature_relevance` routes score extraction through `view_scores`, so joint-basis fits (`pca`, `nsa_pipeline`) and partial `test_views` (`None` for missing views) work identically to per-view fits.
- The statistic is each feature's share of the variance the view explains in a target (relative
  weights, normalised to sum to 1).
- `shrinkage="auto"` (the default) uses classical Johnson shares when full-rank `p < n` and
  Ledoit–Wolf shrunk shares when `p >= n`, collinear, or `p >= n_sub` / `n_test`.
- The target is the mean of the other views' sign-aligned scores on j.
- The fair share is 1/p.

**Tests.**
- **Training:** the target is permuted against the features (`n_permutations`, default 100 000).
- **Held-out:** the fitted model is applied to the test subjects and the test is rerun
  (`n_permutations_test`, default 50 000).
- **Refits:** half-sample refits, each with its own test (`n_refits`, default 10). They give the
  fraction of refits with p < alpha.
- **Optional refit null:** `n_null_refits` refits the whole model with the view's rows shuffled.
  It is slow, and it is reported beside the tiers, not in them.
- **Groups:** tested on the sum of their shares.
- **p-values:** uncorrected. `summary` sets the counts against chance (alpha × p).

**Tiers.**
- 2 = p < alpha in training **and** held-out;
- 1 = training only;
- 0 = neither.

**Wide views (p ≥ n).** The default `shrinkage="auto"` keeps the classical estimator (λ = 0) on a
full-rank view with p < n, and switches to Ledoit–Wolf shrinkage otherwise.
- **Two Ledoit–Wolf definitions:**
  - `"ledoit_wolf"` takes λ from the view alone, so the permutation null is exact. `feature_relevance`
    uses this one.
  - `"ledoit_wolf_joint"` (in `feature_maps`) takes λ per target from [X, t], so the weights sum
    exactly to the shrunk R².
- **The λ used is always on record:**
  - `rel["shrinkage_used"]` and the `summary` column `shrinkage`;
  - `feature_maps(..., return_shrinkage=True)`;
  - `loading_inference(...)["maps"]["relative_weight"]["rw_shrinkage"]`.
- Held-out shares reuse the training λ. Where that estimator is undefined on the held-out rows, λ is
  re-resolved and recorded in `shrinkage_used_test`.

**How to read it.**
- The test measures **concentration, not strength**. A single feature's null share is about
  Beta(½, (p − 1)/2), whatever the signal.
  - With p = 160, a feature needs a share of about 2.4% (3.8× fair) for p < 0.05.
  - When k features share a signal equally, each gets about 1/k. On the planted test, 4 strong
    drivers among 24 features reach p ≈ 0.01–0.025 and no lower.
  - More permutations sharpen p; they don't add power.
- A diffuse signal passes no single feature. Use group tests for it (lobes, networks).
- Read `r_outcome_test` for strength. CSF on ADNI has a large share but r = +0.17: it is a
  non-redundant slice of the component, not a strong marker.

**Evidence.**
- **ADNI, exact equivalence.** On the Figure 3.9 fit, `feature_relevance` reproduces the original
  scripts exactly: 222/222 features and 93/93 groups, max difference ~1e-16, identical tiers.
  `scripts/adni_relevance_test.py` and `scripts/adni_relevance_figure.py` are now thin callers.
  The figure takes about 18 s on 8 workers, including the fit.
- **Planted.** In `tests/test_feature_relevance.py`, every driver reaches tier 2 and no noise
  feature does, over 7 seeds. A pure-noise view of 60 features gives 1–4 at p < 0.05 in training
  (chance is 3).


## Which components carry a prediction (`pysimlr.component_attribution`, paper Fig. 3.9)

```python
att = pysimlr.component_attribution(res, train_views, test_views, y_train, y_test,
                                    variants={"collapse_weight=0": {"collapse_weight": 0.0}})
fig = pysimlr.plotting.plot_component_attribution(att, compare={2: "collapse_weight=0"})
```

It answers three questions about any fit with first-layer bases `v`.

**Which components predict.** Each view's `X_m V_m` is scored on held-out rows with a linear
regression fitted on the training rows. Every component gets its held-out R² alone, and its
*unique* contribution: the loss in R² when it is dropped from all of them.

**What each view's variance looks like.** For each view it reports:
- the eigenvalue spectrum of the training correlation matrix (one dominant factor means a global
  axis, as on DiReCT thickness);
- the mean inter-feature correlation;
- each consensus component's correlation with the view's subject mean;
- with OLEE loadings, how many features are hard-assigned to each component.

**Whether the predictive support replicates.**
- Refits on random subsets of the training rows (`n_refits`, `refit_fraction`, or exact
  `refit_rows`) reuse the fit's own parameters.
- In each refit and view, the component that predicts best alone is the predictive column, and
  its support is recorded.
- `variants` repeats the refits with parameter overrides.

**Output and figure.**
- The output is JSON-ready, so it can be cached as is.
- **Panels:** A (components); B (the spectrum of `spectrum_view`, by default the view with the
  largest first eigenvalue); then one selection-frequency panel per view in `selection_views`.
- **Comparison:** `compare={view: variant}` draws that view as default (hollow) against the
  variant (filled).
- **Overrides:** labels, colours, titles, `variant_labels` and `feature_word`.
- Save inside `with pysimlr.plotting.theme():` to keep the house font.

**Paper figure.** `scripts/paper_adni_attribution.py` is a thin caller. It writes
`paper/results_cache/adni_attribution.json`: the result under `attribution`, plus the keys
`paper_numbers` reads. The manuscript chunk calls `plot_component_attribution` with the paper's
labels and titles. On the Figure 3.9 fit the function reproduces the pre-refactor script exactly
(every number and all 40 refits, difference 0.0).

## Uncertainty and the choice of k

```python
inf = pysimlr.loading_inference("simlr", views, 3, n_subsamples=100, random_state=0)
inf["component_stability"]   # per view: one value per component
inf["ci_low"], inf["ci_high"]   # 95% percentile intervals per view
inf["selected"]              # stability-selected features (threshold 0.8, expected_false 1)

sel = pysimlr.select_rank("simlr", views, [2, 3, 4], n_subsamples=20, random_state=0)
sel["k"]          # selected k under criterion (default "component"), or None
sel["criteria"]   # all five rules from the same refits: component, subspace, oos_r2, oos_r2_1se, oos_coupling
sel["table"]      # per-k stability, subspace cosines, out-of-bag R^2 and coupling, effective rank
```

- **How the refits work.**
  - `loading_inference` refits the whole pipeline on random subsamples (`fraction=0.5` by
    default), aligns each refit by Procrustes, and reports percentile intervals.
  - Its per-component stability is the median correlation between each aligned refit component
    and the full-data component.
  - Every subsample refit also evaluates `predict` on its unused `1 - fraction` rows (zero extra
    fits), giving held-out reconstruction `oos_r2` and held-out leave-one-view-out score
    correlation `oos_coupling`.
  - The default of 100 refits is costly on large data; `select_rank` uses 20 refits per
    candidate k.
  - Pass `n_jobs=-1` (all cores) or `n_jobs=4` to run the refits in parallel. Results are
    identical to serial. On ADNI-MCI, `select_rank` over k ∈ {2, 3} took 17 s instead of 98 s.
  - Subsample refits warn (`RuntimeWarning`) when `converged_fraction < 0.5` naming the iteration budget, record `median_converged_iter`, `oos_coupling_splits` and `n_effective_refits`, and omit unconverged refits when `exclude_unconverged=True` (closed-form `pca` stays included).
- **Which `criterion` to read.** One `select_rank` call populates `sel["criteria"]` with all six:
  - `"component"` (default): every varimax-rotated axis has stability ≥ `min_stability`. Use this
    when naming and interpreting individual components `C1..Ck`.
  - `"subspace"`: the smallest Grassmannian principal cosine of the stacked structure span is
    ≥ `min_stability`. Stays high when real components share near-equal singular values and rotate
    within a stable `k`-dimensional subspace.
  - `"oos_r2_1se"` and `"oos_r2"`: smallest `k` within 1 SE of the peak out-of-bag held-out
    reconstruction R² (and the peak itself, when > 0). Use these when choosing a subspace rank for
    downstream prediction (for example ridge or `fused_ridge` on `u @ R`).
  - `"oos_coupling"`: largest `k` whose every component generalizes across views out of sample
    above `min_oos_coupling` (default: one-sided α = 0.01 Fisher-z null bar).
  - `"heldout_coupling"`: when `heldout_views=` is passed as a diagnostic, largest `k` whose every component clears the Fisher-z 0.01 bar on `heldout_coupling`'s `cross_view_heldout` (`None` when omitted).
  - `pysimlr.select_model(views, candidates, ks, ...)` wraps `select_rank` over `(candidate, k)` cells on one paired out-of-bag split design, gates by health (`converged_fraction >= min_converged_fraction` and not `basis_degenerate`) and the Fisher-z `oos_coupling_threshold`, and selects the most parsimonious `(complexity_tier, k, -sparseness, -min_subspace_stability)` cell within 1 paired SE of the highest `S = sum_j max(median_coupling_j, 0)^2`.
  - Check `basis_degenerate` (`min_effective_rank < k - 0.5`) in `sel["table"]`: when `True` at
    larger `k`, basis columns are merging and `nsa_w` should be raised (for example `0.75`).

#### Bottleneck bound on k

`pysimlr.bottleneck_rank(views, confounds=C, bound="min" | "max" | "stacked")` bounds shared rank across views (`"min"` = rank shared by every view; `"max"` = rank a consensus can carry from any view; `"stacked"` = permutation rank of the pooled between-view operator `M = g_all @ g_all - sum_i G_i @ G_i`), always returning `k_max_min`, `k_max_max`, `k_max_stacked` and per-view `sigma1_over_tau95`, `tau95_rel_se` (bootstrap relative SE of the null threshold `threshold_first` only; sampling variance of `sigma1` is not included) and `borderline` (`|ratio - 1| < 2 * tau95_rel_se`).
For each view it takes the singular values of the cross-covariance between the standardised view and
the other views stacked, and counts the leading ones above the 95th percentile of the same values under
row permutations of that view (`cross_rank`); `k_max` is the minimum over views (under default `bound="min"`) and `bottleneck_view`
the view attaining it. Private structure in a view does not enter a cross-covariance, and a shared
latent counts however small its within-view footprint: on the planted chain with `k = 3`,
`p = (20, 25, 6)`, `n = 400`, the cross-view rank of the 6-column view is 3 on every seed, while
Horn's parallel analysis on that view alone (`within_rank`, also reported; `mode="within"` makes it set
`k_max`) gives 2 on three of four seeds because a random 3-by-6 loading basis has one weak direction.
Pass `confounds`: a covariate leaking into every view is a shared dimension and is counted until it is
residualised out. The bound is a guide for the `ks` grid, not a selector; the refit criteria above still
decide. `select_rank(method, views, "auto")` uses `bound="max"` by default (`range(1, k_max + 1)`, raising a `ValueError` with the worst view's `sigma1/tau95` when `k_max == 0`); an explicit `ks` above `k_max`
raises a `RuntimeWarning`, and the dict is returned as `sel["bottleneck"]`.

- **Stability is a graded measure, not a pass/fail test.**
  - Each refit sees half the rows, so stability describes reproducibility at about n/2. That is
    pessimistic when power is low: a real component in a small or noisy sample can sit around
    0.8.
  - The `select_rank` default `min_stability=0.9` is a conservative choice, not a derived
    threshold. In lower-power settings 0.8 is a reasonable bar: pass it explicitly
    (`select_rank(..., min_stability=0.8)`).
  - Whichever threshold you use, state it, and report every component's stability value.
  - Interpret components well below the threshold cautiously, rather than treating them as
    established.
- **Changing `fraction`.** A larger fraction raises the stability values, but partly because each
  refit then shares more rows with the full-data fit. So the numbers become more optimistic, not
  more informative. Keep the default unless you have a reason, and report the fraction with the
  results.
- **Stability selection needs uniform half-samples.** The `selected` features and
  `expected_false_selections` rely on the Meinshausen–Bühlmann bound, which holds only for
  half-samples (and `threshold > 0.5`). Since v0.32.0 the bound is NaN, with
  `expected_false_selections_reason`, for any other design (`fraction != 0.5`, `subsample="bootstrap"`,
  `strata=`); treat `selected` there as a ranking without error control.
- **Signed displays.** Show only features whose interval excludes 0, and name features from
  `selected`.
- **`select_rank` can return `None`.** No candidate k met the requested criterion. Inspect
  `sel["table"]` and `sel["criteria"]`:
  - try a smaller k, or a lower `min_stability` if the setting is low-power (state it);
  - or try `positivity="positive"`.

## Grouped analyses, excluded views, per-refit outputs, resampling schemes, orientation (v0.32.0)

Everything here is opt-in; with the new arguments left at their defaults every output is unchanged
(tested against values generated on v0.31.1).

**Within-group analysis (`groups=`).** Sites, cohorts and scanners shift feature means, and a latent that
separates the groups then correlates with every feature that shifts. Pass `groups=` (length-n labels) to
`structure_coefficients`, `loadings`, `sem.view_scores`, `feature_maps` and `loading_inference`: features,
latent columns and the other-views target are centered within group before any correlation or map, so the
result is a pooled within-group association and a group mean difference cannot produce one.
`scale_within_group=False` (default) scales globally after centering; `True` also divides each group by its
own SD (use when the groups differ in variance and one should not dominate). A group with fewer than 3
usable rows for a view is left out of that view with a `MissingRowsWarning`; below `min_rows` the maps are NaN
with a reason (the D52 rules), and it composes with `missing_rows="mask"`. `loadings(..., groups=g)` also
derives the varimax rotation within group; `feature_maps` uses the result's `loading_rotation`, so for
consistent within-group maps pass `{**res, **loadings(res, views, groups=g)}` (`loading_inference` does
this for the reference and every refit, and its `out["result"]` then holds the within-group keys).
`stats.permutation_correlations(..., groups=g)` centers within group, permutes rows only within group (the
unrestricted permutation of data that differ in group means is not a valid null for a within-group claim:
in the tests it called a pure group shift significant 90+% of the time), bootstraps within group, and uses
the effective sample size `n - G + 1`. In `loading_inference` the permutation null permutes within group
whenever `groups` is given (no separate switch: grouped structure with an ungrouped null would not be a
null for the same statistic). Resampling is not stratified by `groups` unless you also pass
`strata=groups`; with small groups do. The out-of-bag `oos_coupling` is computed on within-group scores;
`oos_r2` (reconstruction) is not grouped.

**Excluded views (`feature_maps(..., exclude_views=[...])`).** The listed views (indices or names) are left
out of the `target="other_views"` target of every view. They still receive maps, against the target built from
the non-excluded views; ignore them if unwanted (a separate `skip_maps_for` would only save time). Forward it
through `loading_inference(map_options={"exclude_views": [...]})`. **This changes the target only.** The
fitted consensus and each view's encoder were estimated with the excluded view in the model, so the other
views' scores still carry information pulled from it. For a strictly non-circular test when the excluded
view is the outcome ("root"), refit without it, `fit(method, [v for v in views if v is not root], k)`, take
latents and maps from that fit, and relate the root to them (`pysimlr.extend` can add it afterwards). Use
`exclude_views` as a screen (which features track the latent that the other views support) and the refit for
the claim.

**Per-refit outputs (`loading_inference(..., return_refits=(...))`).** Default: nothing is kept. Request any of
`"weights"`, `"maps"`, `"latents"`, `"rows"` to receive `out["refits"]` with, for each kept refit, the aligned
structure, per-view weights, maps, the consensus latent of every row (in-bag from the refit, out-of-bag
predicted, flagged by `in_bag`) and the row indices, all in the reference frame of the stability summaries,
plus `alignment["rotation"]` so a caller can undo it (`u_refit = latent @ rotation[b].T`). Stability
recomputed from the returned arrays reproduces `component_stability` (tested). Memory is about
`4 B (structure + weights + maps + n k)` bytes for `B` refits; a `ValueError` is raised before fitting when it
would exceed `max_refit_bytes` (default 2e9). Use it for refit-level questions the summaries do not answer:
which subjects drive the latent, how an outcome relates to the out-of-bag latent per refit, or a bespoke
stability measure.

**Resampling schemes (`subsample=`, `fraction=`, `strata=`).** `"half"` (default; `round(fraction n)` rows
without replacement, half-samples at the default fraction; draws identical to earlier versions),
`"fraction"` (same draw, explicit for `fraction != 0.5`) and `"bootstrap"` (`n` rows with replacement,
out-of-bag = never drawn, about 37%; refits see duplicated rows). `strata=` (length-n labels) makes the draws
proportional within strata (for the bootstrap, each stratum to its own size). To stratify by the
view-observation pattern under `missing_rows="mask"` build labels from it and pass them as `strata`; it is not
automatic. Guarantees: only uniform half-sampling carries the stability-selection bound; elsewhere
`expected_false_selections` is NaN with `expected_false_selections_reason` and `out["resampling"]["bound_valid"]`
is False. Intervals, component stability and selection frequency remain descriptive resampling summaries without a
coverage or error-rate guarantee (bootstrap intervals are percentile intervals; large fractions understate
variability). Stratified draws are for balance (every refit sees every site), not for a stronger guarantee.

**Orientation across fits (`pysimlr.orient_latents`).** Latents, weights and maps of separate fits agree only up to
signs, component order (and rotation). `orient_latents(latents, weights=, maps=, mode="sign" | "signed_permutation" |
"procrustes")` puts a list of fits in one frame by a rule that does not depend on their order: the reference
starts at the medoid (largest summed pairwise absolute-cosine agreement), is replaced by the mean of the oriented
fits until the transforms stop changing, and the final frame orders components by mean column norm and gives
each a non-negative sum of cubes. Permuting the input list permutes the outputs and changes nothing else
(tested for all three modes). It returns the transforms, so weights and maps are carried along (use `abs` of a
transform for share-valued maps). `loading_inference(return_refits=...)` does not call it: its refits are already
aligned to the full-data fit, which is order-free by construction. `orient=` was not added to `fit` or
`view_scores`: a single fit has no peers to orient to, and its own convention (varimax columns with non-negative sum
of cubes) is unchanged.

**Worked example: multi-site data, root outcome view, missing rows.**

```python
import warnings, numpy as np, torch, pysimlr
rng = np.random.default_rng(0)
site = np.repeat([0, 1, 2], 50)                      # three sites, n = 150
z = rng.standard_normal((150, 1))                    # shared latent
shift = np.array([0.0, 2.0, -2.0])[site][:, None]    # site offsets in every view
def view(p, load):
    w = np.zeros((1, p)); w[0, :load] = 1.0
    return torch.tensor(z @ w * 1.5 + rng.standard_normal((150, p)) + shift, dtype=torch.float32)
views = [view(12, 4), view(10, 3), view(8, 3), view(3, 3)]     # the last view is the outcome "root"
views[1][:10] = float("nan"); views[2][100:110] = float("nan") # rows missing in views 1 and 2

res = pysimlr.fit("simlr", views, 1, missing_rows="mask", random_state=0, iterations=100)
res_w = {**res, **pysimlr.loadings(res, views, groups=site)}   # within-site structure and rotation
maps, info = pysimlr.feature_maps(res_w, views, kinds=["structure"], groups=site,
                                  exclude_views=[3], return_info=True)
# info["n_eff"] == [150, 140, 140, 150]; view 0's planted features have map values ~0.8 within site.

inf = pysimlr.loading_inference(
    "simlr", views, 1, missing_rows="mask", iterations=100, n_subsamples=100, n_permutations=100,
    groups=site, strata=site, maps=("structure",), map_options={"exclude_views": [3]},
    return_refits=("latents", "rows"))
# inf["null_max_abs"]: within-site permutation null; inf["expected_false_selections"] is NaN (stratified draws),
# inf["refits"]["latents"]: (100, 150, 1) latents in the reference frame, inf["refits"]["in_bag"] flags the rows.
# For the non-circular claim about the root: refit on views[:3], then relate views[3] to that fit's latents.
```

## New samples

```python
pred = pysimlr.predict(res, views)
pred["u"], pred["r2"], pred["mse"]
```

Prediction freezes the consensus as an affine map, so it does not depend on the batch. `r2` is
per view, in the training standardisation.

## Edges between views: the score-level path model

Multi-view representation learning (`pysimlr.fit`) embeds each modality into an aligned latent
coordinate space. But shared latents alone do not reveal conditional dependencies: does cortical
thinning drive cognitive decline directly, or is its effect mediated through regional volume loss?
Does functional connectivity link to anatomy once global structural atrophy is accounted for?

`pysimlr.path_model` evaluates dependency graphs over view scores and clinical outcomes post-fit.
It treats the fitted model (`result`) as a frozen measurement model, projects subjects into a
standardized, sign-aligned coordinate frame, and tests edges via Freedman–Lane permutation sweeps
with joint family-wise multiplicity control, omnibus tests, graph-fit tests, bootstrap effect-size
intervals, and refit calibration. Competing topologies are compared out-of-sample via
`pysimlr.compare_path_models`, and missing views are reconstructed along significant paths via
`pysimlr.predict_path`.

```python
import pysimlr

# 1. Fit measurement model with confounds routed through the sink
res = pysimlr.fit("simlr", train_views, k=3, confounds=C_train, random_state=42)

# 2. Test a hypothesized path graph on held-out subjects
# graph: None (complete), edge list [("vol", "thk"), ("thk", "outcome")], or adjacency dict
pm = pysimlr.path_model(
    res, test_views, rows="heldout",
    graph=[("vol", "thk"), ("thk", "outcome")],
    covariates=C_test, groups=sex_test, outcome=y_test,
    conditioning="parents",       # "pairwise" or "parents"
    null="fixed",                 # "fixed" (Freedman-Lane) or "refit" (on rows="train")
    n_permutations=10_000, n_bootstrap=2000, random_state=42,
)

# Inspect edge statistics, multiplicity-adjusted p-values, and omnibus fit
pm["edges"]            # DataFrame: a, b, pillai, rv, p_perm, p_maxt, p_holm, q_bh, CIs
pm["edge_components"] # DataFrame: per-component partial r and slope beta with CIs
pm["fit"]              # present_pass, absent_pass, max_absent_p, rejected
pm["omnibus"]          # max_pillai, mean_pillai, p_max, p_mean

# 3. Compare alternative topologies out-of-sample (structural mode)
graphs = {
    "complete": None,
    "chain": [("vol", "thk"), ("thk", "outcome")],
    "star": [("thk", "vol"), ("thk", "outcome")],
    "empty": [],
}
cmp = pysimlr.compare_path_models(
    res, views_train=train_views, views_test=test_views, graphs=graphs,
    covariates_train=C_train, covariates_test=C_test,
    outcome_train=y_train, outcome_test=y_test,
    n_bootstrap=2000, random_state=42,
)
print("Parsimonious best:", cmp["best"], "by rule:", cmp["best_rule"])

# 4. Predict missing views across path edges
pred_path = pysimlr.predict_path(pm, views_with_missing, covariates=C_test, groups=sex_test)
pred_path["scores"], pred_path["predicted_from"], pred_path["outcome_pred"]

# 5. Visualize with publication-ready figures
pysimlr.plot_path_model(pm)
pysimlr.plot_edge_matrix(pm, what="r")
pysimlr.plot_path_comparison(cmp)
pysimlr.plot_edge_null(pm)
pysimlr.plot_edge_features(pm, edge=("vol", "thk"))
```

**Score extraction (`view_scores`).**
All score extraction across `pysimlr` is unified in `pysimlr.view_scores`:
- Each view is projected into the reference frame of `result["structure"]` (rotated by
  `result["loading_rotation"]`).
- Column signs are aligned so each latent covaries non-negatively with `u` on the training rows
  (`sign_rows="training"`).
- Scores are column-standardized on evaluation rows (`standardize=True`).
- Linear and deep models score through their latents (`source="latents"`), whereas joint-basis fits
  (`pca`, `nsa_pipeline`, `gcca`) score through their block of `v` (`source="joint_block"`).
- Missing views (`None`) are accepted; available views are scored independently.

**Scores from any source (`path_model_from_scores`).** The inference layer does not need a
`pysimlr` fit. `pysimlr.path_model_from_scores({"vol": Z_vol, "thk": Z_thk, ...}, outcome=y,
covariates=C, ...)` takes a mapping of node name to `(n, k_m)` score block (embeddings, factor
scores, another package's latents; blocks of different width are tested through `pillai` only),
standardizes each block column-wise as `view_scores` does, and runs the identical fixed-model
inference, families, omnibus, graph fit, bootstrap intervals and node regressions. On
`dict(zip(names, pysimlr.view_scores(res, views)["scores"]))` the result equals `path_model`'s
(`test_path_model_from_scores_matches_path_model`). Without a measurement model there is no refit
null (`null` must be `"fixed"`), no `inference=` intervals, no consensus node and no
`feature_contributions` (`plot_edge_features` and `plot_edge_brain_map` raise a `ValueError` on scores-only results); `predict_path` on the result takes a mapping of the same names with
`None` for a missing block, and `pysimlr.compare_path_models_from_scores(scores_train, scores_test, graphs, ...)` runs structural graph comparison directly on score blocks (supporting missing blocks in `scores_test`, with `standardize=True`/`"per_split"` z-scoring each split on its own rows as `view_scores` does, `"train"` standardizing `scores_test` with `scores_train`'s mean and SD, and `False` leaving blocks unscaled). `rows="train"` declares that the scores were fitted on these rows and
follows the training-row policy below. On `rows="heldout"`, both `path_model` and structural `compare_path_models` accept `None` for views absent on evaluation rows (dropping absent views with a `RuntimeWarning` unless explicitly referenced in `graph`).

**Edge statistics.**
For an edge $(a, b)$ with conditioning set $S$ (nuisance design $N$: intercept, standardized
covariates, group dummies; under `conditioning="parents"`, additionally the scores of other adjacent
nodes):
- Both score blocks are residualized on $S$ via rank-revealing QR decomposition (tolerance $10^{-10}$):
  $R_a, R_b$.
- **`pillai`**: Mean squared canonical correlation between $R_a$ and $R_b$,
  $\operatorname{tr}((R_b^{*\top} R_b^*)^{-1} T_a^\top T_a) / \min(k_a, k_b)$. In $[0, 1]$, invariant
  to orthogonal rotations of either block. The primary invariant edge statistic.
- **`r` and `beta` per component**: Partial correlation $r_j$ and regression slope $\beta_j$
  ($z_{bj}$ on $z_{aj}$ given $S$) in the rotated reference frame. Conditioned on the reference
  rotation, as every per-component claim in `pysimlr` is.
- **`rv`**: Adjusted RV coefficient between $R_a$ and $R_b$; reported for orientation, not tested.
- **`moderation`** (with `groups`): Sum of per-component $F$-statistics for adding group interaction
  terms $R_{a,j} \cdot G$ to $[S, R_{a,j}]$ predicting $R_{b,j}$. Tests whether edge strength differs
  across diagnostic or demographic cohorts.
- **`outcome_edges`**: Edges into an external clinical outcome ($Y$). Tested as a separate hypothesis
  family (ANCOVA question).

**Inference and multiplicity.**
- **Fixed-model null (`null="fixed"`, Freedman & Lane 1983):** In each permutation sweep
  $b \in \{1, \dots, B\}$, one permutation per node is drawn and shared across all its columns. Scores
  are reconstructed as $Z_b^* = P_S Z_b + \pi_b(R_b)$ (only the part of $Z_b$ unexplained by $S$ is
  permuted). All edges are evaluated in the same sweep.
- **Multiplicity correction:**
  - `p_perm`: Add-one permutation $p$-value, $(1 + \#\{\text{null} \ge \text{obs}\}) / (1 + B)$, floor
    $1 / (B + 1)$; never 0.
  - `p_maxt`: Single-step max-$T$ across the entire family (evaluated against the empirical distribution
    of maximum statistics across sweeps). Controls FWER under the global null.
  - `p_holm` and `q_bh`: Step-down Holm and Benjamini–Hochberg FDR adjustments from `pysimlr.stats`.
  - Families: `edges` (all tested edges, Pillai), `edge_components` (all components, $|r_j|$),
    `moderation` (sum-$F$), and `outcome_edges` (edges into outcome).
- **Refit null (`null="refit"`, `rows="train"`):** For each view node $b$ ending a tested edge,
  $B$ refits of `pysimlr.fit` are executed with $b$'s training rows and its confound rows jointly
  permuted (`confounds.permute_view_confounds`). Reports `p_refit` and `p_refit_holm`. Eliminates
  training-row optimization coupling.
- **Effect-size uncertainty:**
  - Percentile 95% bootstrap CIs (`ci_low`, `ci_high`) for Pillai, $r_j$, and $\beta_j$ via
    $n_{\text{boot}} = 2000$ subject resamples with the measurement model fixed.
  - Out-of-bag refit intervals (`ci_refit_low`, `ci_refit_high`) when `inference=` (`loading_inference`)
    is passed, evaluating Pillai across subsample refits without additional fitting cost.
- **Graph fit:**
  - Under `conditioning="parents"`, `present_pass` is the fraction of hypothesized edges with
    $p_{\text{holm}} < \alpha$.
  - `absent_pass` is the fraction of absent edges with unadjusted $p_{\text{perm}} \ge \alpha$.
  - `max_absent_pillai` with max-$T$ $p$-value across absent edges tests conditional independence (the
    permutation analogue of the SEM $\chi^2$ omnibus test; graph is rejected if $p < \alpha$).

**Model comparison (`compare_path_models` and `compare_path_models_from_scores`).**
- **Structural mode (`result` provided, measurement model fixed):**
  - Node regressions fitted on training scores using Generalized Cross-Validation ridge (Golub, Heath & Wahba 1979, *Technometrics* 21:215–223) on nuisance-orthogonalized parent blocks, PRESS leave-one-out hat-matrix residual variance $r_i / (1 - h_{ii})$ (Allen 1974, *Technometrics* 16:125–127), and measurement-basis degrees-of-freedom shrinkage $\gamma = (n - 1) / (n - 1 + \text{dof}_{\text{eff}})$ from `consensus.view_dofs` (`dof_eff = max({dof(b)} | {dof(a) : a in pa(b)})`; Ezekiel 1930; Wherry 1931, *Ann. Math. Stat.* 2:440–457; matching `extension._affine_map`); evaluated by Gaussian log-likelihood on held-out scores (`heldout_loglik`). `compare_path_models_from_scores` accepts an optional `dofs=` mapping or sequence for the same basis-DOF calibration.
  - Paired subject bootstrap over held-out rows computes `delta_vs_top` (log-likelihood difference vs top
    model) with 95% CIs and `best_fraction` (replicate win rate).
  - Parsimony selection (`best`): The graph with the fewest edges whose delta CI against the top model
    contains 0 (`best_rule`). Raw likelihood top reported beside it as `best_by_loglik`.
- **Training-topology mode (`method` and `k` provided, refits across splits):**
  - Repeated train/test splits ($n_{\text{splits}} = 5$).
  - Evaluates models fitted with `topology="graph"` under their own graph against the complete graph,
    reporting held-out log-likelihood, OOS coupling, OOS reconstruction $R^2$, and outcome $R^2$.

**Prediction along path edges (`predict_path`).**
Given a fitted `path_model` result and new subject data with missing views (`None`):
- Iteratively predicts missing view scores from available neighbors using fitted ridge node regressions
  $Z_b \sim [N, Z_{\text{pa}(b)}]$.
- Predicts continuous outcomes ($Y$) from parent latent scores.
- Evaluates out-of-sample score reconstruction $R^2$ per node against ground truth when `views_true` is
  supplied.

**What the tests can and cannot say.**
1. **F2 attenuation (measurement error in the conditioning set):**
   Conditioning on *estimated* latent scores attenuates an absent edge rather than removing it
   completely. In a true chain $X_0 \to X_1 \to X_2$, the scores $\hat{Z}_1$ estimate the true latent
   with finite-sample measurement noise. Partialling out $\hat{Z}_1$ leaves residual covariance between
   $X_0$ and $X_2$. Consequently, permutation tests for absent edges given parents err toward
   over-rejection ($p < \alpha$), especially with small $k$, noisy features, or small $n$ (on a planted
   chain with $n = 500$, noise 0.3, the absent edge $X_0 - X_2$ yielded $p_{\text{perm}} = 0.027$ and
   $0.072$ on two draws). The graph-fit test accommodates this by requiring 8 of 10 seeds to pass in
   synthetic benchmarks and reporting unadjusted $p$-values (`absent_p`) so borderline edges remain
   visible. A rejected graph fit indicates substantial unmodeled covariance, but an isolated marginal
   rejection on an absent edge should be interpreted cautiously alongside the attenuation floor.
2. **Max-$T$ under partial nulls (subset pivotality):**
   Single-step max-$T$ (`p_maxt`) controls the family-wise error rate exactly under the *global null*
   (where all edges in the family have zero effect). Under a *partial null* (where some edges carry
   strong signal and others are null), max-$T$ assumes subset pivotality—namely, that the joint
   distribution of the null statistics does not depend on the non-zero parameters. In complex multivariate
   networks with correlated nuisance designs, subset pivotality is an approximation. Therefore, `pysimlr`
   always reports step-down Holm adjusted $p$-values (`p_holm`) and Benjamini–Hochberg FDR $q$-values
   (`q_bh`) alongside `p_maxt`. Holm provides strong FWER control without assuming subset pivotality.
3. **Training-row coupling vs held-out evaluation:**
   Fixed-model permutation tests (`null="fixed"`) permute subjects while keeping the projection matrices
   $V_m$ fixed. When evaluated on *training rows*, this test is severely inflated if any view is wide
   ($p_m \ge n$): the encoder $V_m$ was optimized to couple within-view noise to other views (see D25,
   `WideViewWarning`). The resulting training scores exhibit strong coupling even under complete
   independence. Even without a wide view, the fixed-model null on training rows ignores that the model
   was fitted to those rows. So on `rows="train"` fixed-model p-values are not reported for any view
   unless `allow_train_fixed=True` or a non-empty reason string `allow_train_fixed="reason"` (stored in `parameters["allow_train_fixed_reason"]` and echoed in the warning; a warning is issued and `p_reason` says why; the refit null is the
   default null there). Every returned `edges`, `edge_components` and `outcome_edges` frame carries a `null_rows` column (`heldout-fixed`, `train-refit`, `train-fixed`, `train-fixed-unreported`, or `train-both`), mirrored in `fit["evidence"]` and the `plot_path_model` subtitle. To test dependency on training rows, use the refit null (`null="refit"`), which
   permutes view rows prior to refitting the measurement model ($B$ refits with permuted confound rows).
   Otherwise, evaluate the fixed-model null strictly on held-out rows (`rows="heldout"`), where score
   projections are genuine out-of-sample representations.
4. **Worker processes (`n_jobs > 1`):** the refits run in spawned worker processes. Call `path_model`
   (and `compare_path_models` with `method=`) with `n_jobs > 1` from a script under
   `if __name__ == "__main__":` or from an importable module. From the interactive prompt or a notebook
   cell the workers cannot import the caller and the pool fails with `BrokenProcessPool`; use
   `n_jobs=1` there. Results do not depend on `n_jobs`.

**Evidence and calibration.**
- **Null calibration (fast and slow suites):**
  - Fixed-model null (`test_null_calibration_fixed_model`, 200 datasets, $n = 120$, $B = 500$): per-edge
    rejection rates at $\alpha = 0.05$ were $0.045, 0.050, 0.035$ (family max-$T$ rate $0.050$).
  - Refit null (`test_null_calibration_refit`, 100 datasets, $n = 100$, $B = 100$, 5 iterations): per-edge
    rejection rates were $[0.070, 0.030, 0.020]$ (mean $0.040$, max $0.070$), strictly within the nominal
    $[0.02, 0.09]$ calibration band.
  - Matrix solve cost: $B = 10\,000$ fixed-model sweeps on $M = 4, k = 5, n = 300$ executes in $0.81$ s
    (well within the 10 s plan target).
- **ADNI-MCI cognitive pilot ($n = 1084$, 3 views: `T1Hier_vol` $p=160$, `T1Hier_thk` $p=160$, `T1w_mean` $p=62$, outcome `CogPC1`, 5 splits):**
  - Plan P1 (`T1Hier_vol`–`T1Hier_thk` held-out `p_maxt < 0.05`): passed in 5/5 splits for both `simlr` and `lend` (`p_maxt = 0.0005` in all splits).
  - Plan P2 (`outcome_edges` family has $\ge 1$ view with `p_holm < 0.05`): passed in 4/5 splits for both `simlr` and `lend` (driven by cortical thickness and regional volume).
  - Plan P3 (held-out `pillai` below training `pillai` per edge): passed in 5/5 splits across all measurement-fitted view–view edges (`15/15` in `simlr`, `15/15` in `lend`), while weak unsupervised `view - outcome` edges (`CogPC1` is unseen by `fit`) exceeded training `pillai` in 3/5 splits due to 70/30 split sampling variance and finite-sample $k/n$ Pillai offset.
  - Plan P4 (`complete` not best by `heldout_loglik` in every split): passed (`simlr` favored `star on T1Hier_thk` over `complete` in 4/5 splits by raw log-likelihood, mean $\Delta = +7.40$, $\text{SE} = 5.45$, and 5/5 by F1 parsimony; `lend` favored `complete` in 4/5 splits and `star` in 1/5, mean $\Delta = -30.41$, $\text{SE} = 13.72$).
  - Additional findings: `predict_path` reconstructed held-out `CogPC1` at mean $R^2 = 0.0873$ ($r = 0.350$) for `simlr` and $R^2 = 0.0894$ ($r = 0.354$) for `lend` after demographic confound residualization. On training rows (`rows="train", null="both"`, matched $B = 50$), the live refit-null 95th-percentile Pillai threshold exceeded the fixed-model null threshold by $43.0\times$ on `simlr` view–view edges (`0.300` vs `0.0070`) and $16.0\times$ on `lend` view–view edges (`0.124` vs `0.0078`), versus $1.14\times$–$1.17\times$ on unsupervised outcome edges.

### Linear vs. Deep Conditional-Graph Behavior

In the ADNI-MCI pilot evaluation, linear `simlr` selected the parsimonious star graph (mediating volume through thickness to cognition, mean $\Delta\text{LL} = +7.40$ vs `complete`), whereas nonlinear `lend` favored `complete` in 4 of 5 splits (mean $\Delta\text{LL} = -30.41$). Linear encoders yield score projections where residual variance is linearly separable, allowing linear partial residualization on intermediate scores to attenuate indirect edges. Nonlinear multi-layer perceptron encoders induce higher-order cross-view score couplings that are not completely d-separated by linear partial regression, leaving residual cross-view predictive signal for the complete graph.

## Choosing an energy

- **`nc` (the default)** and the other scale-free energies (`align`, `procrustes`, `recon_r2`,
  and the contrasts `logcosh`, `exp`, `kurtosis`) weight views equally, 1/M. For `nc` the
  total energy is bounded below by −1.
- **`logcosh`, `exp` and `kurtosis` are not negentropy.** They apply a contrast to the k × k
  cross-correlation between a view's scores and the consensus. They are coupling energies, like
  `nc`.
- **`negentropy` is not a fit objective.** It has no dependence on the consensus, so as an
  `energy_type` each view would run its own ICA with no coupling, and pysimlr refuses it. Use it
  as an auxiliary term (`negentropy_weight > 0`) beside a similarity energy.
- **`acc` and `recon` are not scale-free.** Each view is weighted by 1/(|E0|·M), where E0 is its
  energy at the start of the fit, so their relative view weights depend on the initialisation.
  Prefer `nc` unless you need these specifically.

## Constraining the basis (`positivity`)

`positivity` constrains each basis column `v` (features × 1) on every method that has a basis:
`simlr`, `lend`, `ned`, `ned_shared_private`, `flow_v`, and the anchored encoders of
`pysimlr.extend`. The three names below form a hierarchy over how the features are combined.

| value | aliases | constraint on each column | each latent `X v` is |
|---|---|---|---|
| `"linear"` | `"either"`, `False` (the default) | any real weights | a signed linear combination |
| `"conical"` | `"positive"`, `"hard"`, `True` | `v ≥ 0` | a non-negative combination |
| `"convex"` | — | `v ≥ 0` and `sum(v) = 1` | a weighted average of the (standardized) features |

Other values that are still accepted: `"nonnegative"`/`"nonneg"` (the conical constraint with
no reflection fallback) and `"softplus"` (deep only). `"negative"` is accepted on the linear
path. Any other string raises `ValueError`.

- **The aliases are exact.** `"linear"` and `"conical"` give fits bit-identical to `"either"`
  and `"positive"`. `res["parameters"]["used"]["positivity"]` keeps the spelling you passed;
  `res["parameters"]["positivity_resolved"]` is the canonical value.
- **`"convex"` is enforced inside the optimisation, not as a rescaling at the end.**
  - Linear path: every step ends with the Euclidean projection onto the probability simplex,
    restricted to the column's current support, so a sparse column is never densified.
  - Deep and flow models: the same projection is applied to the first-layer basis on every
    forward pass. It is piecewise linear, like sparsemax, so gradients pass through it.
- **The simplex replaces the unit-length column scaling.** The energy sees `V` with columns
  summing to 1, so a latent's scale (`vᵀΣv`, large when the selected features move together)
  enters every energy that is not invariant column by column: `nc` (normalized over the whole
  matrix), `acc`, `procrustes`, the k × k contrasts, and `recon_r2`. A convex fit is therefore
  not a rescaled conical fit.
- **Refusals:**
  - `pca` and `nsa_pipeline` have no basis constraint, so they refuse `"convex"`.
  - `consolidate=True` is refused under `"convex"`.
- **Measured behaviour (D36, ADNI-MCI and NHANES, reported, not a recommendation):**
  - Held-out R² against conical moved by −0.05 to +0.02 depending on dataset, model family and
    sparseness.
  - Supports were somewhat smaller, and support stability across resamples was usually lower,
    most markedly for the deep models.
  - See `docs/tracking.md` D36 for the per-family tables.

## View order and `sign_alignment`

Since v0.31.0 the averaging mixings (`newton`, the default, and `avg`) align the views' latent
signs with a rule that is symmetric in the views (`sign_alignment="spectral"`), so a fit does not
depend on the order of the views. Before, the rule was greedy from the first view and, for three or
more views whose latents disagreed in sign at the start, reordering the views could change the fit
(first-sweep consensus differences up to 3.6 on weakly coupled synthetic views; the difference is
often forgotten after convergence, but not always). Results from v0.30.0 and earlier for such fits
shift when refitted. To reproduce an old result pass `sign_alignment="greedy"` (every entry point
that has the option, including `pysimlr.fit(..., sign_alignment="greedy")`). Two-view fits are
unaffected up to the global sign of a `u` column. For `k > 1`, signs are not the whole story:
components can be swapped or rotated between views, which neither rule addresses. Fits with
`random`/`perturbed_pca` initialization and nonlinear deep fits keep their own order dependence.

## View weights (`dynamic_weights=True`, MAI)

Off by default. When on, each view's weight in the consensus follows its modality alignment
index (MAI): the leave-one-out agreement of the view's latent with the other views' consensus.
- **The score is corrected for degrees of freedom.** A view with many non-zero loadings can
  agree with anything by fitting. Each score is adjusted like an R² for the view's non-zero
  loadings per column, against the rows the bases were fitted on. The deep and flow models
  score on a mini-batch but fit on the whole training set, so they correct with the
  training-set size.
- **A view that can fit anything is not assessable.** That is a view with dof ≥ n − 1. It
  keeps the uniform share 1/M and is neither favoured nor suppressed. MAI cannot tell signal
  from noise in such a view in-sample.
- **Weights are gated against a null floor, not the mean MAI.** The floor is the 95th
  percentile of the same score with the consensus rows permuted (20 permutations, recomputed
  every 10 updates). If no view clears its floor, the weights stay uniform with a
  `RuntimeWarning`.
- **Weights are bounded to [0.01, 0.5]** (`consensus.MAI_WEIGHT_BOUNDS`). They are the nearest
  weights to the gated target that sum to 1 and stay within those bounds.
  - **No view supplies more than half the consensus,** so one view cannot take it over on a
    noisy score.
  - **A view with no evidence of coupling falls to 1%,** which effectively removes a trash
    modality without ever reaching exactly 0.
  - **With two views, the bounds force 0.5 / 0.5,** so MAI cannot reweight two views.
  - **If only one view clears its null floor,** the consensus rests on that view (capped at
    0.5) and a `RuntimeWarning` says so: the rest of the weight goes to views with no evidence.
  - **If a neighbourhood's weights sum to 0,** its consensus uses the views unweighted, with a
    warning.
- **The ramp from uniform to target weights takes 20 updates.**
- **Choosing `mai_metric`:**
  - `procrustes_r2` (the default) scores agreement across all k components. It gives the best
    latent recovery.
  - `max_cc` is the first canonical correlation between the view's latent and the consensus,
    on rank-adaptive orthonormal bases. It keeps views coupled through only some components
    (sub-coupled views) at full weight. Its dof is the union support, so a wide view is
    more often not assessable.
- **Evidence:** a planted design with two fully coupled views (A, B), one sub-coupled view
  (C), one narrow view (D, p=4) and a p > n noise view (E), at n=150, 5 seeds
  (`scripts/mai_calibration_check.py`, `docs/audit/evidence/mai_calibration_check.csv`). Mean
  final weights:

  | arm | A | B | C | D | E | latent R² |
  |---|---|---|---|---|---|---|
  | previous (raw score, mean gate) | 0.35 | 0.35 | 0.00 | 0.00 | 0.30 | 0.908 |
  | `procrustes_r2`, no floor | 0.41 | 0.41 | 0.09 | 0.08 | 0.01 | 0.906 |
  | `procrustes_r2`, floor 0.25 | 0.35 | 0.35 | 0.12 | 0.11 | 0.06 | 0.896 |
  | `procrustes_r2`, floor 0.5 (v0.10–v0.12) | 0.30 | 0.30 | 0.15 | 0.14 | 0.11 | 0.879 |
  | **`procrustes_r2`, bounds [0.01, 0.5] (default)** | 0.40 | 0.40 | 0.09 | 0.08 | 0.02 | 0.908 |
  | `max_cc`, bounds [0.01, 0.5] | 0.22 | 0.22 | 0.22 | 0.15 | 0.20 | 0.856 |

  - **The bounds recover the latent R² the half-share floor gave up** (0.908 against 0.879).
    The noise view is at the 0.01 bound on 4 of 5 seeds. On the fifth its MAI clears its null
    floor, and it gets 0.068.
  - **Under `max_cc`, E is not assessable** (union support ≥ n − 1), so it keeps its uniform
    share.
  - **In the all-coupled control, every arm is uniform** (within 0.01).
- **Cost:** about 10 ms per update with floors at n=1000 and M=6, against 1–2 ms without.
  Floors run on 1 update in 10.

## Adding a modality to a fitted model (`pysimlr.extend`)

New modalities often arrive after a model is fitted. `pysimlr.extend(res, views, new_views)`
adds them without refitting, for any method. It returns a new result and never changes `res`.
`views` are the original views for the subjects that have the new modality, as a list or a dict
by view name; `new_views` is `{"pet": X_pet, ...}`.

- **`how="anchored"` (the default)** freezes the model.
  - **How it works:** each new view gets a linear encoder fitted to the model's own consensus,
    under the fit's energy, positivity, sparseness and NSA-Flow weight.
  - **Same component frame:** the new view's `structure` and `loadings` use the model's rotation,
    so components keep their numbers and meaning.
  - **Checks:** a held-out coupling check warns when the new view does not align with the model.
  - **Unchanged predictions:** predictions from the original views are unchanged, bit for bit.
- **`how="reembed"`** is ANTsR's algorithm (`extend_simlr_embedding_with_new_modalities`), with
  every option.
  - **How it works:** it fits a new joint embedding of the existing views' scores and the new
    views' features.
  - **Output:** the result is a new model (`reembedded`). The original model does not gain the
    new view.
- **`predict` from any subset of views:**
  - **Input:** `pysimlr.predict(res, {"view0": X0, "pet": X_pet})`, or a list with `None` for a
    missing view. This works for every result, extended or not.
  - **The consensus:** a precision-weighted average of each available view's map to the training
    consensus.
  - **All original views present:** the original, exact consensus is used.

**Evidence** (`scripts/extend_check.py`, `docs/audit/evidence/extend_check.csv`): a planted
design with three views, 5 seeds; anchored extension with C compared with a full refit on all
three views.

| method | latent R², A + B | A + B + C, anchored | full refit | C alone |
|---|---|---|---|---|
| `simlr` | 0.969 | 0.974 | 0.976 | 0.908 |
| `ned` | 0.795 | 0.830 | 0.858 | 0.863 |

- **Linear `simlr`:** anchored is within 0.002–0.003 of the refit on every seed.
- **Deep models:** an anchored extension is only as good as the frozen model. On one seed the
  frozen `ned` model was weak (0.67), and the refit gained 0.25. Extending can't repair a weak
  model; refit when the original model is weak.
- **The new view's encoder is linear,** even for deep and flow models.

## Confound sink (`confounds`): nuisance removal for every method, no test-time confounds

`pysimlr.fit(method, views, k, confounds=C)` works for every registered method (`simlr`, `lend`,
`ned`, `ned_shared_private`, `flow_v`, `pca`, `nsa_pipeline`). Before the method sees the data,
each view is cleaned by a fixed affine map fitted on the training rows. The map is stored in the
view's preprocessing provenance, so everything that replays preprocessing applies the same map:
`predict`, missing-view `predict`, `loadings`, `feature_maps`, `heldout_coupling`,
`feature_relevance`, and the refits of `loading_inference`, `select_rank` and
`component_attribution`. Prediction needs no confound values.

```python
res = pysimlr.fit("simlr", train_views, k=4, confounds=C_train)    # (n, q) array / DataFrame / Series,
                                                                   # per-view list, or {view: C_m} dict
pred = pysimlr.predict(res, test_views)                            # no C_test; N = 1, partial lists, or any Mapping work
pysimlr.confounds.confound_decorrelation(res, test_views, C_test)  # held-out max / mean |r(u, C)| (full, partial list, or Mapping)
```

### What the map is

For view `m`, the confounds are standardised and reduced to an orthonormal basis `Q_m` (`n x k_c`).
The rank is capped at `confound_rank` or `n // 4`; a cap that binds warns. Constant columns are
dropped and missing entries are mean-filled; both are recorded. `H_m` is an orthonormal basis of
`X_c^T Q_m`: the feature directions along which the view covaries with the confounds. The cleaned
view is

    X_m - gamma (X_m - mu_m) H_m H_m^T,    gamma = confound_strength in [0, 1].

At `gamma = 1`, every linear score of a cleaned training view has zero sample covariance with
every confound column. The confound's feature pattern is removed from new rows without their
confound values. `result["u"]` equals `predict(result, training_views)["u"]`. The decorrelation
of that predict path is `result["confound_sink"]["train_max_abs_corr"]`. A view whose features the
confounds span entirely (`p_m <= rank`) raises an error; give it fewer confounds with a per-view
list. `predict` also returns `latents_sink` (per view: `(X - mu) H M`, an image-derived estimate
of the confound basis) and `u_sink` (their mean over the views given, for shared confounds only).

### What it costs: measured on ADNI-MCI

`scripts/benchmark_confound_sink.py` → `docs/audit/evidence/confound_sink_adni_mci.{csv,json}`.
Setup: 20 random 70/30 splits; confounds age, sex, education and head size; held-out rows only;
outcome CogPC1 with a random-forest read-out. The rules were fixed before the run.

| `simlr` | max \|r(u, C)\| | adjusted CogPC1 R² | cross-view coupling |
|---|---|---|---|
| unadjusted | 0.524 | 0.031 | 0.902 |
| pre-residualised training rows only | 0.557 | -0.167 | 0.773 |
| frozen training betas + `C_test` | 0.121 | 0.093 | 0.851 |
| `confounds=`, strength 1 | **0.115** | **-0.009** | 0.843 |
| `confounds=`, strength 0.5 | 0.491 | 0.036 | 0.878 |

The `pca` baseline gives the same picture: max |r| 0.115 vs 0.128 frozen vs 0.823 unadjusted;
adjusted R² -0.017 vs 0.106.

Without `C_test`, the sink decorrelates as well as frozen betas with `C_test` (rules R1 and R2
pass). It loses the confound-adjusted outcome signal (rule R3 fails: mean paired -0.10, worse by
more than 0.05 in 18 of 20 splits). The reason is that the cognition-relevant atrophy lies along
the same feature directions as age and head size, and the projection removes those directions.
When per-subject confounds are available at deployment (they usually are: age, sex, site, motion
and head size are measured per scan), residualise with frozen training betas instead. Use the sink
when confound values will not be available for new subjects, and accept the loss of the biology
that shares the confounds' feature directions. Strength 0.5 does not decorrelate (it only halves
the variance along those directions).

NHANES 2015-18 (labs / body / diet; confounds age, sex, race, poverty-income ratio, education and
cycle; HbA1c; `confound_sink_nhanes.{csv,json}`) gives the same pattern:

- held-out max |r(u, C)| is 0.142 for the sink, 0.161 for frozen betas and 0.685 unadjusted;
- adjusted HbA1c R² is -0.008 for the sink vs 0.016 for frozen betas (R3 fails);
- held-out cross-view coupling falls from 0.273 to 0.065, because the shared axis of these views
  is mostly demographic.


## Fused ridge (multi-task regression with fused coefficients)

`fit_fused_ridge` fits one coefficient vector per *task* and pulls neighbouring tasks'
coefficients together (penalty λ₁Σ‖w_t‖² + λ₂Σ_{neighbours}‖w_a − w_b‖²). The tasks are built
in one of two ways:

- **`thresholds=` (the `ANTsR::fusedRidge` model):** one binary task per outcome threshold,
  1[y ≥ c_t], on all rows (`family="binomial"` by default). As λ₂ grows, neighbouring
  thresholds share slopes.
- **`groups=`:** one task per stratum, site or exposure level. Pass a label per row, or
  boolean masks, which may overlap. `family="gaussian"` by default.

```python
cv = pysimlr.fused_ridge_cv(X, y, thresholds=np.quantile(y, [.25, .5, .75]),
                            lambda1s=(0.01, 0.1), lambda2s=(0, 0.5, 5), top_ks=(None, 10, 20))
fit = cv["result"]                       # refit at the CV-chosen (lambda1, lambda2, top_k)
p = pysimlr.predict_fused_ridge(fit, X_new, type="response")   # P(y >= c_t), one column per t
pysimlr.plot_coefficient_paths(fit); pysimlr.plot_fused_coefficients(fit); pysimlr.plot_fused_ridge_cv(cv)
```

- **Families:** `gaussian`, `binomial` and `poisson`.
- **Covariates:** `covariates=` are unpenalised, and every task has its own intercept.
- **Scale:** features are standardised with the sample sd, as in R's `scale`. `coef` is on the
  original scale; `coef_std` is the standardised one.
- **`top_k`, as ANTsR's `topK`:**
  - in the fit, it takes each task's K largest standardised |coefficients|, takes the union,
    and refits on those features;
  - in `predict_fused_ridge(..., top_k=K)`, it keeps each task's K largest coefficients
    without refitting;
  - in `fused_ridge_cv`, selection happens inside each training fold.
- **Agreement with ANTsR:** the objective and solution match `ANTsR::fusedRidgeDirect` (to
  about 1e-3, tested), and `top_k` selects the same features as `ANTsR::fusedRidge(topK=)`.
  ANTsR's two backends scale the penalty differently, so their coefficients differ from each
  other at the same λ.

## Plotting

All plots live in `pysimlr.plotting` (also importable from `pysimlr` directly), and they follow
one convention:
- each takes a result object;
- it accepts `ax=None` and returns the Figure (no `show`, no saving);
- matplotlib only;
- it plots interpretable quantities, with signed values on a symmetric diverging scale;
- it is drawn in one visual system, `pysimlr.plotting.theme()`:
  - views take fixed, colour-vision-safe categorical colours (`VIEW_COLORS`) and are always
    labelled as well as coloured;
  - magnitude uses one blue ramp, and signed values a blue–red scale around a neutral midpoint;
  - figures have hairline axes and a left-aligned title with a one-line reading.

  Wrap your own figures in `with pysimlr.plotting.theme():` to match.

The gallery `docs/plotting_gallery.html` (`scripts/plotting_gallery.py`) shows every figure on
a planted four-view study.

| function | shows | input |
|---|---|---|
| `feature_table(res, inf=)` | **data**, one row per view × feature × component: structure, loading, CI, `excludes_zero`, `selected`, `selection_frequency`, stability | `fit` (+ `loading_inference`) |
| `plot_top_features(res, inf=, component=)` | a component's top features, one panel per view: CIs, permutation null, selected in bold, stability badge | `fit` (+ inference) |
| `plot_feature_groups(res, groups, inf=)` | groups (lobes, networks, pathways) × components: mean or max \|structure\|, or counts of significant (CI excludes 0) or stability-selected features | `fit` (+ inference) |
| `plot_loadings(res, view)` | loadings, features × components (rotated frame) | `fit` |
| `plot_latent_coupling(res, pred=)` | view-vs-consensus r per component, training and held-out (with the change), and view-vs-view | `fit` (+ `predict`) |
| `plot_view_chord(res, pred=)` | one chord diagram per component: which views share it (ribbon fills its slot in proportion to r) | `fit` (+ `predict`) |
| `plot_component_radar(res, pred=)` | each view's coupling across components, one radar per view; a sub-coupled view is a spike | `fit` (+ `predict`) |
| `plot_view_radar(res, views, pred=)` | a fingerprint per view: coupling and R², training and held-out, rank, sparsity, weight | `fit` (+ `predict`) |
| `plot_feature_chord(res, inf=)` | features of every view joined to the components they build (width = \|structure\|) | `fit` (+ inference) |
| `plot_convergence(res)` | the objective history (deep and flow: total, sim and recon); flags unconverged fits | `fit` |
| `plot_basis_diagnostics(res)` | effective rank against k, column overlap, conditioning, sparsity and margin | `fit` |
| `plot_rank_selection(sel)` | every component's stability at each k; the weakest-component curve; the 0.9 and 0.8 bars; the selected k | `select_rank` |
| `plot_variance_explained(res, views, pred=)` | reconstruction R², training against held-out; structure captured per component | `fit` (+ `predict`) |
| `plot_modality_weights(res)` | view weights against 1/M, and MAI, with flow_v trajectories | `simlr` / `flow_v` fit |
| `plot_flow_warping(res, views)` | the linear first layer against the latent the model uses (Spearman), per view and component | deep or flow `fit` |
| `plot_scores(res, new_views=)` | subject scores `u @ R`, with held-out subjects projected | `fit` |
| `plot_view_correlations(views)` | adjusted RV between views, as a chord (a matrix when drawn into `ax`) | views |
| `plot_latent_recovery(res, z_true)` | \|corr\| of components with planted latents | `fit` |
| `plot_fit_summary(res, views, pred=, inf=)` | a one-call diagnostic sheet | `fit` (+ `predict`, inference) |
| `plot_benchmark(df, x, y)` | a metric against a setting, per method (mean ± SE) | results DataFrame |
| `plot_benchmark_uncertainty(df, baseline=)` | per-method score with a bootstrap CI over splits; paired difference against a baseline | long DataFrame (method, split, score) |
| `plot_fused_coefficients`, `plot_coefficient_paths`, `plot_fused_ridge_cv` | fused ridge | `fit_fused_ridge` / `fused_ridge_cv` |
| `plot_predictions(y, yhat)` | predicted against observed | arrays |
| `plot_path_model(pm, component=)` | path model diagram: views on a circle, edge widths by Pillai (or \|r\| when `component=j` with sign color), significance, graph-fit status | `path_model` |
| `plot_edge_matrix(pm, what="r")` | edge × component heatmap with BH q-value hatching and Pillai column | `path_model` |
| `plot_path_comparison(cmp)` | forest plot of held-out log-likelihood deltas vs top graph, with fraction passing panel | `compare_path_models` |
| `plot_edge_null(pm, edges=)` | observed edge statistic against its permutation null distribution | `path_model` |
| `plot_edge_features(pm, edge, top_n=)` | feature contributions to an edge (structure × beta), one panel per component | `path_model` |
| `plot_edge_brain_map(pm, edge, ...)` | feature contributions to an edge painted on cortical surfaces / atlases via antsxviz | `path_model` |
| `plot_dose_recruitment(rec, ...)` | continuous `Dose × View` recruitment heatmap, adaptive-$N_{\text{eff}}$ local linear curves with simultaneous `sup-t` bands, local slopes, and $2\sigma$ bandwidth spans | `modality_dose_recruitment` |

Multi-view plots colour views with the Okabe–Ito palette. A plot of an unconverged fit is
titled in orange with its stop reason. Stability badges are black at ≥ 0.9, amber at ≥ 0.8 and
red below; change them with `bars=`.

Nothing plots the encoder `v`. The architecture diagrams are in `pysimlr.diagrams`.

## Continuous dose-response and stage-dependent view recruitment

Discrete quartile or cutpoint bar charts lose within-bin ordering and depend on arbitrary thresholds. `pysimlr.adaptive_local_linear_curve`, `pysimlr.modality_dose_recruitment`, and `pysimlr.plot_dose_recruitment` estimate continuous displacement $\Delta(x_0) = \hat{\mu}(x_0) - \hat{\mu}_0$ and local derivative $\partial\hat{\mu}/\partial x$ across one or more continuous exposure, disease-duration, or biomarker-burden axes:

```python
rec = pysimlr.modality_dose_recruitment(
    scores={"Consensus": oof_u, "Perfusion": oof_asl, "Morphometry": oof_t1},
    doses={"Primary_Dose": raw_counts, "Co_Exposure": secondary_counts},
    covariates={"Primary_Dose": np.log10(secondary_counts + 1.0), "Co_Exposure": np.log10(raw_counts + 1.0)},
    standardize_scores=True,
    target_neff=32.0,
    min_sigma=0.28,
    n_boot=1000,
    seed=42,
)
fig = pysimlr.plot_dose_recruitment(rec, dose_units="events")
```

- **Adaptive-$N_{\text{eff}}$ bandwidths and resolution transparency:** At each evaluation coordinate $x_0$, Gaussian kernel width $\sigma(x_0) \in [\sigma_{\min}, \sigma_{\max}]$ is solved by bisection so Kish effective sample size $(\sum_i w_i)^2 / \sum_i w_i^2$ matches `target_neff` (typically $30\text{–}35$). Always inspect `median_span_2sigma` and Row B/C2's $2\sigma(x_0)$ caliper so smooth kernel interpolation is read at its actual smoothing span rather than mistaken for single-event precision.
- **0-mass boundary anchoring:** When at least `zero_mass_min` observations sit at `dose == 0`, they enter positive-tail local linear regressions as a single consolidated boundary anchor at $x = 0$ (weight capped at `zero_anchor_cap`), preventing zero-inflation from collapsing bandwidths while keeping $\Delta(0) = 0$ continuous.
- **Zero extrapolation outside observed support:** Grid coordinates without nearby observations (`in_support == False`) stay `NaN` in `delta` and render cross-hatched in `plot_dose_recruitment`, so cohorts lacking a modality at high or low doses never extrapolate silently.
- **Pointwise vs simultaneous `sup-t` bands:** `onset_pointwise` marks the first sustained grid crossing where the pointwise interval excludes $0$; `onset_simultaneous` uses the studentised maximum-modulus (`sup-t`) bootstrap critical value $c_{1-\alpha}$ across the entire in-support curve (refitting covariate partial residuals inside every bootstrap replicate).

## Known limitations (being worked on)

- **Correlated factors are not represented as such.**
  - With the default `newton` mixing, consensus scores are decorrelated. The shared *subspace* is
    recovered, but correlated factor axes are not.
  - `mixing_algorithm="avg"` keeps factor correlation, but only up to about the collapse
    threshold, `collapse_threshold` (default 0.3). On converged fits with true factor
    correlation 0.5-0.7, the recovered score correlation peaked at 0.31 with threshold 0.3, 0.51
    with 0.5, and 0.69 with 0.7.
  - If you expect correlated factors, pass a higher `collapse_threshold` (for example 0.5-0.7)
    with `mixing_algorithm="avg"`. The cost: on narrow views a higher threshold lets components
    collapse. At 0.7, a narrow-view test case's score correlation rose to 0.89 and its outcome
    R² fell by 0.10. Check score correlations and stability when you raise it.
- **The collapse penalty.** It prevents components collapsing onto each other on narrow views,
  but on such views it also costs similarity. Its threshold is the trade-off described above;
  its weight (`collapse_weight`, linear only) stays at 1.0.
- **Non-negative bases on a view dominated by one global factor.** Examples are cortical
  thickness, where every region thins together, or any panel whose features are all positively
  correlated.
  - **Why the penalty breaks localisation here.** Any two non-negative region sums on such a
    view correlate above the 0.3 threshold, so the penalty shrinks columns to one or two
    features.
  - **ADNI-MCI DiReCT thickness** (62 DKT regions, PC1 = 58% of variance;
    `scripts/adni_global_factor_check.py` → `docs/audit/evidence/adni_global_factor_check.csv`,
    10 half-samples). With the default, the predictive column kept ≥ 5 regions in 6/10
    refits, and only one region was selected in ≥ 50% of them. With `collapse_weight=0` it kept
    ≥ 5 regions in 9/10, three regions replicated in ≥ 50% (L entorhinal, L supramarginal, L superior frontal), and held-out CogPC1 R² was 0.224
    against 0.210.
  - **Regressing out the subject's mean thickness doesn't help.** The columns spread out, but the
    predictive signal goes with the global factor (thickness-column R² ≈ 0).
  - **What to do.** Before reading regional loadings on such a view, refit with
    `collapse_weight=0` and check stability across refits.
  - **What not to do.** Don't read hard-assigned (argmax) loadings on such a view. Structure
    coefficients measure association, so nearly every region lands on the component that tracks
    the global factor.
  - **Why the default stays at 1.0.** This is evidence from one dataset.
- **The default combination** of energy, optimizer and mixing is being re-confirmed at the full
  iteration budget. The current choice comes from a 25-iteration benchmark.
- **Removed in 0.6.0:** the `simlr_lbfgsb` method and the `nsa_lbfgsb` optimizer. They were not
  competitive with `hybrid_adam`, and using either name raises an error pointing to `simlr`.
- **NED (`ned`) is sensitive to input rounding.** Inputs differing by 1e-8 give measurably
  different fits, because float32 training amplifies the difference. Compare NED fits across
  seeds, not across tiny input changes.
- **Never set `nsa_w` to 1.** It is unstable, and the API rejects values above 0.99.
