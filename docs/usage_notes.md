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
- **Which `criterion` to read.** One `select_rank` call populates `sel["criteria"]` with all five:
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
  - Check `basis_degenerate` (`min_effective_rank < k - 0.5`) in `sel["table"]`: when `True` at
    larger `k`, basis columns are merging and `nsa_w` should be raised (for example `0.75`).
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
- **Stability selection needs `fraction=0.5`.** The `selected` features and
  `expected_false_selections` rely on the Meinshausen–Bühlmann bound, which holds only for
  half-samples (and `threshold > 0.5`). With any other fraction, treat `selected` as a ranking
  without error control.
- **Signed displays.** Show only features whose interval excludes 0, and name features from
  `selected`.
- **`select_rank` can return `None`.** No candidate k met the requested criterion. Inspect
  `sel["table"]` and `sel["criteria"]`:
  - try a smaller k, or a lower `min_stability` if the setting is low-power (state it);
  - or try `positivity="positive"`.

## New samples

```python
pred = pysimlr.predict(res, views)
pred["u"], pred["r2"], pred["mse"]
```

Prediction freezes the consensus as an affine map, so it does not depend on the batch. `r2` is
per view, in the training standardisation.

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
- **No binary weights.** Every view keeps at least half its uniform share:
  w = 0.5/M + 0.5 · target (`consensus.MAI_MIN_SHARE`). With 4 views the weights stay in
  [0.125, 0.625], so one view cannot take the consensus on a noisy score, and a view's graph
  neighbourhood is never empty. If a neighbourhood's weights do sum to 0, its consensus uses
  the views unweighted, with a warning.
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
  | `procrustes_r2`, floor 0.5 (default) | 0.30 | 0.30 | 0.15 | 0.14 | 0.11 | 0.879 |
  | `max_cc`, floor 0.5 | 0.21 | 0.21 | 0.21 | 0.18 | 0.20 | 0.853 |

  - **The floor trades about 0.03 latent R² on this design** for weights that cannot go
    binary.
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

Multi-view plots colour views with the Okabe–Ito palette. A plot of an unconverged fit is
titled in orange with its stop reason. Stability badges are black at ≥ 0.9, amber at ≥ 0.8 and
red below; change them with `bars=`.

Nothing plots the encoder `v`. The architecture diagrams are in `pysimlr.diagrams`.

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
- **The default combination** of energy, optimizer and mixing is being re-confirmed at the full
  iteration budget. The current choice comes from a 25-iteration benchmark.
- **Removed in 0.6.0:** the `simlr_lbfgsb` method and the `nsa_lbfgsb` optimizer. They were not
  competitive with `hybrid_adam`, and using either name raises an error pointing to `simlr`.
- **NED (`ned`) is sensitive to input rounding.** Inputs differing by 1e-8 give measurably
  different fits, because float32 training amplifies the difference. Compare NED fits across
  seeds, not across tiny input changes.
- **Never set `nsa_w` to 1.** It is unstable, and the API rejects values above 0.99.
