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
sel["k"]       # largest k whose every component has stability >= min_stability (default 0.9), or None
sel["table"]   # per-k component_stability, min_stability, all_stable
```

- **How the refits work.**
  - `loading_inference` refits the whole pipeline on random subsamples (`fraction=0.5` by
    default), aligns each refit by Procrustes, and reports percentile intervals.
  - Its per-component stability is the median correlation between each aligned refit component
    and the full-data component.
  - The default of 100 refits is costly on large data; `select_rank` uses 20 refits per
    candidate k.
  - Pass `n_jobs=-1` (all cores) or `n_jobs=4` to run the refits in parallel. Results are
    identical to serial. On ADNI-MCI, `select_rank` over k ∈ {2, 3} took 17 s instead of 98 s.
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
- **`select_rank` can return `None`.** No candidate k met `min_stability`. Inspect `sel["table"]`:
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
- it plots interpretable quantities, with signed values on a symmetric diverging scale.

| plot | shows | input |
|---|---|---|
| `plot_loadings(res, view)` | loadings, features × components (rotated frame) | `fit` result |
| `plot_structure_intervals(inf, view, component)` | structure coefficients with 95% intervals; selected features in bold | `loading_inference` result |
| `plot_component_stability(rank)` | per-component stability per k, with reference lines at 0.9 and 0.8 | `select_rank` result |
| `plot_scores(res, new_views=)` | subject scores `u @ R`, with held-out subjects projected | `fit` result |
| `plot_view_agreement(res, component)` | each view's scores against the consensus, with r | `fit` result |
| `plot_convergence(res)` | the objective history, with the stop reason; flags unconverged fits | `fit` result |
| `plot_view_correlations(views)` | adjusted RV between views | views |
| `plot_latent_recovery(res, z_true)` | \|corr\| of components with planted latents | `fit` result |
| `plot_fused_coefficients`, `plot_coefficient_paths`, `plot_fused_ridge_cv` | fused ridge | `fit_fused_ridge` / `fused_ridge_cv` results |
| `plot_benchmark(df, x, y, by, kind)` | one benchmark metric against a setting, per method | results DataFrame |
| `plot_predictions(y, yhat)` | predicted against observed, with r | arrays |

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
