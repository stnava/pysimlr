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
- **`res["loadings"]`:** the two-stage loadings. These are the structure coefficients in a
  varimax-rotated, sign-anchored basis.
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
sel["k"]       # largest k whose every component has stability >= 0.9, or None
sel["table"]   # per-k component_stability, min_stability, all_stable
```

- **How the refits work.** `loading_inference` refits the whole pipeline on half-samples, aligns
  each refit by Procrustes, and reports percentile intervals and per-component stability. The
  default of 100 refits is costly on large data; `select_rank` uses 20 refits per candidate k.
- **Report only stable components.** A component with stability below 0.9 is not identified by
  the data: report it as such, or drop it.
- **Signed displays.** Show only features whose interval excludes 0, and name features from
  `selected`.
- **`select_rank` can return `None`.** That means no candidate k is fully stable: try a smaller
  k, or `positivity="positive"`.

## New samples

```python
pred = pysimlr.predict(res, views)
pred["u"], pred["r2"], pred["mse"]
```

Prediction freezes the consensus as an affine map, so it does not depend on the batch. `r2` is
per view, in the training standardisation.

## Choosing an energy

- **`nc` (the default)** and the other scale-free energies (`align`, `procrustes`, `recon_r2`,
  and the ICA contrasts `logcosh`, `exp`, `kurtosis`, `negentropy`) weight views equally, 1/M.
  For `nc` the total energy is bounded below by −1.
- **`acc` and `recon` are not scale-free.** Each view is weighted by 1/(|E0|·M), where E0 is its
  energy at the start of the fit, so their relative view weights depend on the initialisation.
  Prefer `nc` unless you need these specifically.

## Known limitations (being worked on)

- **Correlated factors are not represented as such.**
  - With the default `newton` mixing, consensus scores are decorrelated. The shared *subspace* is
    recovered, but correlated factor axes are not.
  - `mixing_algorithm="avg"` keeps factor correlation, but only up to about 0.30: the collapse
    penalty's threshold (`DEFAULT_COLLAPSE_THRESHOLD = 0.3`) clips stronger correlation. Treat
    recovered score correlations above that as censored.
- **The collapse penalty.** It prevents components collapsing onto each other on narrow views,
  but on such views it also costs similarity. Whether its threshold is right is under evaluation.
- **The default combination** of energy, optimizer and mixing is being re-confirmed at the full
  iteration budget. The current choice comes from a 25-iteration benchmark.
- **The `simlr_lbfgsb` method and the `nsa_lbfgsb` optimizer** are scheduled for removal: they
  were not competitive with `hybrid_adam`. Do not build on them.
- **NED (`ned`) is sensitive to input rounding.** Inputs differing by 1e-8 give measurably
  different fits, because float32 training amplifies the difference. Compare NED fits across
  seeds, not across tiny input changes.
- **Never set `nsa_w` to 1.** It is unstable, and the API rejects values above 0.99.
