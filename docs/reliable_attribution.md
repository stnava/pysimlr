# Attributing a target to features (`pysimlr.attribute_target`)

```python
out = pysimlr.attribute_target(views, target, covariates=sex, groups=site, n_null=5)
out["maps"]            # per view, signed; positive = the feature rises with the target
out["score"]           # out-of-fold correlation of the latent with the target, and a flag
out["generic"]         # how PC1-like the map and the latent are, and what a null target gives
```

## What the map means

Each entry of `maps` is the correlation of one feature with the **cross-validated latent** that a one-component
target-rooted `simlr` fit builds from all views (`method="structure"`). Covariates and group means are removed first.
It is an association: not a causal effect, and not a unique contribution. A feature that only tracks a driver scores as
high as the driver. Positive means the feature rises with the target (the sign is fixed per view by that view's own
coupling to the latent, so a view that couples negatively is not reported with the wrong sign).

`method="univariate"` returns the covariate-adjusted correlation of each feature with the target instead (no latent).
With `n_null` permutations it adds `p_perm` and `q_fdr` per view.

## When to trust it

Agreement (Pearson) of the map from a small sample with the reference map (sex-adjusted univariate partial r at
n = 20,000; median of 20 draws; UK Biobank, 624 features in three views; aggregates only):

| target (reference out-of-fold r) | map | n = 30 | n = 90 | n = 200 |
|---|---|---|---|---|
| age (0.77) | single fit (`structure`) | 0.84 | 0.94 | 0.97 |
| | univariate | 0.84 | 0.94 | 0.97 |
| | frozen reference (`reference=`) | 0.89 | 0.96 | 0.97 |
| fluid intelligence (0.20) | single fit | 0.43 | 0.61 | 0.79 |
| | univariate | 0.27 | 0.33 | 0.64 |
| | frozen reference | 0.78 | 0.88 | 0.91 |
| fluid intelligence given age and sex (0.20) | single fit | 0.33 | 0.54 | 0.71 |
| | univariate | 0.19 | 0.29 | 0.44 |
| | frozen reference | 0.74 | 0.86 | 0.89 |
| planted, PC1-orthogonal (0.44) | single fit | 0.37 | 0.48 | 0.58 |
| | univariate | 0.36 | 0.49 | 0.65 |
| | frozen reference | 0.43 | 0.55 | 0.63 |
| planted, PC1-orthogonal (0.19) | single fit | 0.12 | 0.23 | 0.39 |
| | univariate | 0.18 | 0.29 | 0.33 |
| | frozen reference | 0.41 | 0.51 | 0.59 |

Read it as follows.

* **The single-fit map is not better than a univariate map where the truth is not aligned with each view's leading
  principal component (PC1).** On age, the truth map correlates 0.85 / 0.84 with the T1 / DTI PC1 loading maps and the
  single fit agrees at 0.97 at n = 200, but so does the univariate map, and after removing the PC1 component from map and
  truth the agreement falls to 0.87 for both. On the planted targets, whose truth is nearly orthogonal to PC1
  (|r| 0.0 to 0.3), the single fit is within 0.03 to 0.07 of the univariate map and sometimes below it. Its advantage on
  fluid intelligence (0.43 against 0.27 at n = 30) is not present once PC1 is removed (0.12 against 0.10).
* **A fit with no signal returns PC1.** On a permuted target at n = 20,000 the single-fit map correlates 0.79 to 0.90
  with the PC1 loading map of T1 and DTI; at n = 30 to 200 the mean |r| over permuted targets is 0.3 to 0.6. This is the
  generic component. `generic` measures it for your data.
* **The frozen reference is the only map that helped across targets** (planted 0.19: 0.41 against 0.12 to 0.18 at
  n = 30). It needs a large sample from the same population and describes that sample's latent, not a refit of yours.
* **Targets with a weak relation (out-of-fold r near 0.2) need n >= 200** for any method to agree above 0.4 with the
  reference on PC1-orthogonal features; at n = 30 treat the map as a ranking hint.
* Post-hoc adjustments were tried and rejected: subtracting the permuted-target generic direction, subtracting its mean
  |map|, and partialling the view's first one or three PCs when computing the structure coefficient. They lowered
  agreement on the PC1-aligned targets (age 0.97 to 0.49, 0.96, 0.34, 0.17 at n = 200) and did not raise it on the
  PC1-orthogonal ones. None is offered.
* Bagging (`n_bag`) gave the same agreement as the single fit at about 60 times the cost, so it is off by default. Use it
  for intervals and for selection with a stated expected number of false selections, not for accuracy.
* Sparseness 0.2, 0.5, 0.8 made no difference.

## Diagnostics

* `score["r"]`, `score["predictive"]`: out-of-fold correlation of the latent score with the target against the
  one-sided p = 0.01 value for that n. When false the map describes a latent that does not predict the target (a
  `UserWarning` is raised). In reference mode `r` is the correlation of the projected score with the target in the new
  sample (the reference never saw these rows).
* `generic["views"][m]`: `map_pc1_absr` (|corr| of the map with the view's PC1 loading map), `latent_pc1_absr` (|corr| of
  the latent with the view's PC1 score). With `n_null > 0`, `null_*` are the means over permuted-target refits,
  `map_excess = observed - null`, and `generic` is true when the map is at least 0.5 aligned with PC1 and exceeds the null
  by less than `generic_tol` (0.1, a heuristic). A true flag means the map is as PC1-like as a target with no signal
  produces; do not read the PC1 pattern as specific to the target.
* `uncertainty` (with `n_bag > 0`): bagged map, delete-d rescaled standard error `sqrt(m/d) * sd` (`m` rows per refit,
  `d = n - m`) and 95% interval, bootstrap-ratio `ratio`. With `n_null > 0` each permuted target is bagged too, and the
  selected set is the largest top-`|ratio|` set whose permuted expectation of exceedances is at most `fdr` times its size;
  `expected_false_selections` is that expectation at the cut. Cost: `n_bag * (1 + n_null)` fits.
* Reference mode returns Fisher-z 95% intervals of each correlation.

## Model families (`model=`)

`attribute_target(..., model=...)` builds the latent with any registered fit method that takes a root view. Each is
fitted as `pysimlr.fit(model, views + [target], 1, **target_root_fit_params(...))`: the target is the last view and the
only neighbour of every feature view (`topology="graph"`). The latent for training rows is `result["u"]`; held-out rows
(and the small sample in reference mode) go through `pysimlr.predict(result, views + [None])`, the model's own encoder
path with the target view absent. The target sign is chosen on the training rows (the correlation of `u` with the
target); for the deep models the root view's own latent can point the other way, so it is not used.

| model | family | root view / target graph | out-of-fold score | frozen reference | unobserved (all-NaN) rows | per-view settings | default call (s) |
|---|---|---|---|---|---|---|---|
| `simlr` | linear | yes | yes | yes | yes (`missing_rows="mask"`) | yes (`per_view_params`) | 3.4 |
| `lend` | deep | yes | yes | yes | no (`ValueError`) | no (fit-wide) | 5.1 |
| `ned` | deep | yes | yes | yes | no | no | 5.5 |
| `ned_shared_private` | deep | yes | yes | yes | no | no | 12.1 |
| `flow_v` | flow | yes | yes | yes | no | no | 11.1 |
| `pca`, `nsa_pipeline` | baseline | no (one joint basis, no per-view encoder) | refused (`ValueError`) | refused | | | |

`flow_simr` is not registered (no linear first layer, no per-view basis) and is not accepted. The return dict records
`model` and `root_mode` (`"target_root"` for a fit; `"frozen_projection"` with `reference=`); no model needed the
fallback of fitting without the target and orienting the consensus afterwards.

Timings: one default call (`n_folds=5`, so six fits, default budgets: 300 sweeps / 300 epochs), n = 120, three views of
12 / 14 / 10 features, one CPU process, Apple Silicon, wall-clock seconds from `time_models.py`. Reference-mode calls
take about 0.002 to 0.015 s. Cost grows with n and features; the deep models are slower per epoch than the linear fit.
With a raw `fit` result as `reference=` the latent's direction is chosen on the sample being projected
(`score["orientation"] == "small_sample"`), so `score["r"]` is then non-negative by construction and `predictive`
should be read as a two-sided test.

**Evidence covers linear SiMLR only.** Every number in the table of agreement above comes from `model="simlr"`.
The deep and flow models are tested on synthetic data for correct sign, recovery of a planted effect, null flagging and
the frozen reference; nothing about their agreement with a large-sample truth on real data is established beyond the
reduced check below.

The `view_coupling_signs` defect fixed in this release has no analogue for these models: they take `positivity` and
`energy_type` fit-wide (`deep.py` `lend_simr` / `ned_simr` / `ned_simr_shared_private`, `flows.py` `flow_simr_v` have no
`per_view_params`), which is what `loadings` reads. A negatively coupled conical `nc_abs` view gets `-1` in every
family (`test_deep_and_flow_negative_coupling_sign_has_no_per_view_defect`).

### Reduced check with a nonlinear encoder (`model="lend"`)

One reduced run, 10 draws per cell, `lend` at 100 epochs, the frozen reference fitted on 2,000 rows (a weak reference:
out-of-fold r 0.75 for age, 0.07 for the planted target), truth the sex-adjusted univariate partial r on 20,000 rows,
UK Biobank, aggregates only (`deep_reduced.py`; participant caches deleted). Pearson agreement with truth; in
brackets, after removing each view's PC1 component from map and truth:

| target | n | single fit (`lend`) | univariate | frozen reference |
|---|---|---|---|---|
| age | 90 | 0.94 (0.76) | 0.94 (0.74) | 0.95 (0.69) |
| age | 200 | 0.97 (0.83) | 0.96 (0.84) | 0.96 (0.75) |
| planted, PC1-orthogonal | 90 | 0.36 (0.13) | 0.33 (0.28) | 0.39 (0.25) |
| planted, PC1-orthogonal | 200 | 0.34 (0.12) | 0.32 (0.30) | 0.39 (0.23) |

The pattern of the linear benchmark repeats: the single-fit map equals the univariate map on age, and on the weak
off-axis target its agreement is mostly the PC1 component (0.12 against 0.30 once PC1 is removed). On that target the
`lend` latent did not predict out of fold (median out-of-fold r 0.04 and -0.06), so `score["predictive"]` is false
for it. The frozen reference was not better than the univariate map here, unlike the linear case, but this reference
was weak. One model, 100 epochs, 10 draws: indicative only.

## Reuse

`out["result"]` and `out["frame"]` (covariate model and scaling) are what `reference=` needs: fit once on the large
sample, then `attribute_target(small_views, small_target, reference=big)`. A raw `pysimlr.fit` result with the target
as its last view is also accepted when no covariates are used. `pysimlr.target_root_fit_params(n_views)` returns the
fit settings used, for callers who want to fit themselves.

## Behaviour change in the same release: `view_coupling_signs`

`loadings(result, views)["coupling_signs"]` (and `result["coupling_signs"]`) read positivity and energy from the
fit-level parameters only. A fit that set them through `per_view_params` (for example conical weights with `nc_abs`)
was treated as signed, so a view that couples negatively to the consensus still got `+1`. Per-view settings are now used.
Fits with fit-level `positivity` / `energy_type` are unchanged.
