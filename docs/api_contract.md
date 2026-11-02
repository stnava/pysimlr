# pysimlr method interface (v0.3.0)

One registry, one `fit`, one `predict`, one record of which parameters each method used.
Implemented in `src/pysimlr/api.py`; scikit-learn layer in `src/pysimlr/sklearn.py`.

## Registered methods

| name | family | native fit | budget unit |
|---|---|---|---|
| `simlr` | linear | `simlr` | iterations (sweeps) |
| `simlr_lbfgsb` | linear | `simlr`, `optimizer_type="nsa_lbfgsb"` fixed; accepts only the keys L-BFGS-B reads | iterations |
| `lend` | deep | `lend_simr` | epochs |
| `ned` | deep | `ned_simr` | epochs |
| `ned_shared_private` | deep | `ned_simr_shared_private` | epochs |
| `flow_v` | flow | `flow_simr_v` | epochs |
| `pca` | baseline | `api.pca_joint` | none |
| `nsa_pipeline` | baseline | `api.nsa_pipeline` | none |

`pysimlr.method_names(family=None)` lists them; `pysimlr.describe_method(name)` returns the family,
budget unit, presets and every accepted parameter with its default.

Not registered:
- `flow_simr` has no linear first layer and returns no per-view basis `v`.
- Per-view PCA (`MultiViewPCA`) has no shared k-dimensional latent.

Both remain available directly.

## Calling

```python
res = pysimlr.fit(method, views, k, *, strict=False, **params)
pred = pysimlr.predict(res, new_views)
```

`views` is one matrix or a list of matrices (tensor, ndarray or DataFrame) with matched rows.

### Parameters

A method accepts exactly the named parameters of its native fit. The registry derives them
from the signature, so the two cannot drift. `simlr` also accepts its optimizer
hyperparameters (`SIMLR_OPTIMIZER_KEYS`).

| Passed parameter | Result |
|---|---|
| Accepted by this method | Forwarded; listed in `parameters["used"]`. |
| Accepted only by another method | Not forwarded; listed in `parameters["ignored"]` with the reason; a `RuntimeWarning` deduplicated per call site by Python's default warning filter; `strict=True` raises `TypeError`. |
| Fixed by a preset (`simlr_lbfgsb`'s `optimizer_type`) | As above, with reason "fixed by method". |
| Accepted by no method | `TypeError`: it can only be a misspelling. |

The native entry points raise `TypeError` on any unknown keyword. Renamed `simlr` keywords
raise with the new name.

### One name per concept

| concept | name everywhere | replaced |
|---|---|---|
| latent dimension | `k` (`n_components` in sklearn) | |
| budget | `iterations` (unit per method in `describe_method`) | `epochs` on deep/flow entry points |
| quantile sparsity | `sparseness` | `sparseness_quantile`, `sparsity` (entry points) |
| seed | `random_state` | `init_seed`; sklearn's global seeding |
| RMSProp decay | `rmsprop_beta` | optimizer `beta`, which collided with the flow models' NLL weight `beta` |
| Lookahead period | `lookahead_k` | optimizer `k`, which collided with the rank `k` |
| L-BFGS-B tolerance | `lbfgsb_tol` | optimizer `tol`, which collided with the convergence `tol` |

Schedule offsets keep their `_epoch(s)` suffix: `warmup_epochs`, `stabilization_start_epoch`,
`stabilization_ramp_epochs` and `shared_warmup_epochs`. They are measured in the method's
budget unit.

`energy_type` and `nsa_w` take their defaults from one place, `src/pysimlr/defaults.py`:
`DEFAULT_ENERGY_TYPE = "acc"` and `DEFAULT_NSA_W = 0.5`, for every method. Before
v0.3.0 the deep and flow models defaulted to `"regression"` and 0.1 with no recorded
reason. Other defaults still differ by method, for example `topology` (`star` for
`simlr`, `loo` for deep and flow) and `mixing_algorithm`. `parameters["defaults"]`
records the value each fit used.

### `random_state`

- **Linear (`simlr`).** Seeds the initializers that draw random numbers (`random`,
  `perturbed_pca`, `best_of_n`) through a local generator. Under the deterministic `pca`
  start it has no effect, and `result["random_state"]` is `None`. It no longer switches
  `pca` to `random`, which the old `init_seed` did.
- **Deep and flow.** The whole fit runs on a private RNG stream seeded with it: the
  near-constant-column jitter, weight initialization, dropout and batch order. The caller's
  global RNG state is restored afterwards. With `None`, one seed is drawn from the global
  stream and returned in `result["random_state"]`. Two unseeded calls therefore differ, and
  each is reproducible from the seed it reports.
- **Baselines.** `pca` passes it to scikit-learn's PCA, which uses it only for the
  randomized solver. `nsa_pipeline` is deterministic and does not accept it.

## Result contract

Every `fit` result has the following keys, checked by `validate_result`:

| key | content |
|---|---|
| `method`, `family` | registered name and family |
| `parameters` | `requested`, `used`, `ignored` (name -> reason), `defaults`, `random_state`, `budget_unit`, `preset` |
| `u` | `(n, k)` tensor. For linear `loo`/`graph` topologies it is the first view's consensus; all views' consensus is in `u_per_modality`. |
| `v` | one `(p_m, k)` block per view |
| `latents` | per-view first-layer projections (`[u]` for the joint baselines) |
| `convergence` | `converged`, `stop_reason`, `n_steps`, `max_steps`, `tol`, `patience`, `best`, `final` (closed-form baselines: `stop_reason="closed_form"`; `simlr` converges on the energy plateau rule (`tol`, `patience`; `certificate="energy_plateau"`), stops uncertified with `stop_reason="no_descent"` when no view's line search finds a decrease; it computes no stationarity residual -- the `grad_map`/`grad_map_history` keys were removed because the prox-gradient residual is not a valid certificate for the non-convex sparsity projection) |
| rank diagnostics | `effective_rank`, `numerical_rank`, `condition_number`, `max_column_overlap` per view; for the joint baselines `joint_*` with `basis_is_joint=True` |
| `sparseness`, `scale_list`, `provenance_list`, `random_state` | |

`predict` returns `u`, `latents`, `reconstructions` and `errors`, plus anything the
method's own predictor adds. It also returns per-view `r2` and `mse`.

All of these are computed in each method's preprocessed space: every view standardized
with its training mean and standard deviation (`utils.preprocess_data`, `centerAndScale`).
`simlr` additionally divides by a constant, which R² does not see. The two baselines use
the same preprocessing.

- `errors` is `||X - X̂||_F / ||X||_F` per view.
- `r2` is `1 - ||X - X̂||² / ||X||²`: held-out variance explained relative to the
  training mean, on one per-feature scale for every method.

### Prediction does not depend on the batch

After fitting, each model freezes its consensus as an affine map, fitted by least squares
on the training batch (`consensus.capture_consensus_map`). Every mixing rule applied at
prediction time is affine for a fixed batch: RMS scaling, sign alignment, the SVD, PCA or
ICA anchor, Newton–Schulz orthogonalization, and column centring and scaling. The map's
fit residual is reported, about 6e-7 in practice. `predict` replays the map row by row.

A test row's latent is therefore the same whichever other rows share its batch, including
a batch of one. Before v0.3.0 these quantities were re-derived from each test batch:
- `lend`'s latents moved by up to 2.2;
- `simlr` with `topology="loo"` moved by 0.58.

Deep and flow results report the fit residual as `consensus_map_residual`. `simlr`
results carry the map in `consensus_map`.

`parameter_summary(result)` flattens the record for a results table. The benchmark runner
writes `method` and `ignored_params` into every metrics row.

## scikit-learn

`MultiViewSiMR(method="simlr", n_components=2, ...)` is one estimator for every registered
method. All parameters are explicit, so `clone`, `get_params` and `GridSearchCV` work.

- **Parameter defaults.** `None` means the selected method's default. Anything not listed
  goes in `extra_params`.
- **Cross-validation.** scikit-learn slices rows of one array, so concatenate the views and
  pass their column counts in `view_sizes`. A list of views works outside CV.
- **`transform`.** Goes through `pysimlr.predict`: the same preprocessing and consensus as
  `fit`.
- **`score`.** Held-out reconstruction R² averaged over views: the mean of `predict`'s
  `r2`. Chosen by the owner as the default model-selection criterion.
- **`scale`.** Maps to `scale_list` for `simlr` and to `scale` for the baselines. The deep
  and flow models always standardize, so for them it is recorded as ignored.
  `build_simlr_pipeline` no longer adds an outer `StandardScaler`, which used to
  standardize twice.
- **`inverse_transform`.** Linear methods and the joint baselines. Returns input units,
  undoing the training preprocessing with `utils.invert_preprocessing`. Returns one array
  when the input was one array (a single view or a `view_sizes` concatenation), and a list
  for list input.
- **Fitted attributes.** `result_`, `loadings_`, `components_`, `parameters_used_`,
  `parameters_ignored_`, `family_`, `budget_unit_`, `n_features_in_`.

`MultiViewPCA` (per-view PCA, with `view_sizes`) and `build_simlr_pipeline` remain.
`SiMLREstimator`, `SiMLR`, `SiMLRTransformer`, `LENDTransformer`, `NEDTransformer` and
`FlowSiMLRTransformer` were removed in 0.3.0.

## Removed in 0.4.0 (no aliases; each raises or is gone)

- Method-name aliases (`linear`, `shared_private`, `nedpp`, `flow_simr_v`, `nsaflow`, ...):
  use the registered names above.
- `deep_simr`, `flow_simlr`, `flow_simlr_v`, `FlowSiMLRModel`, `FlowSiMLRVModel`: use
  `lend_simr`, `flow_simr`, `flow_simr_v`, `FlowSiMRModel`, `FlowSiMRVModel`.
- `FlowWhitener`, `flow_whiten_matrix` and `flow_whitener`: the ANTsTorch LAMNr-Flows whitener
  wrapper. Nothing in pysimlr used it, and its "whitened_full" forward map is not white upstream
  (`apply_lamnr_flows_whitener.py:161` solves with L^T where L belongs; output covariance off the
  identity by 3.2 in simulation). Use ANTsTorch directly once that is fixed.
- Energies `regression`, `gauss`, `normalized_correlation` raise naming the replacement
  (`recon_r2` linear / `align` deep, `exp`, `nc`); `recon` is refused as a linear objective.
- Optimizers `torch_lbfgs` and `nsa_flow` and the `simlr_lbfgs` method.
- The constraint string's third field (`"orthox0.5x1"`): the format is `<type>x<weight>`.
- `retraction_type` (it had no effect with nsa_flow>=3.5.1), `simlr_sparseness(constraint_iterations=)`,
  `predict_simlr(allow_legacy_refit=)`, prediction from pre-0.3.0 results (no frozen consensus),
  `invariant_orthogonality_defect` and its gradient, `calculate_u`, `load_nsa_flow_orth`.

## Benchmark loop

```python
for name in pysimlr.method_names():
    res = pysimlr.fit(name, train_views, k, iterations=50, random_state=seed,
                      energy_type="acc", nsa_w=0.5)
    pred = pysimlr.predict(res, test_views)
```

`benchmarks.runner.run_single_experiment(model_type, case, ..., budget_by_unit=None,
**params)` does this for one split and scores it. `budget_by_unit={"iterations": 25,
"epochs": 45}` gives sweeps and epochs separate budgets in one sweep.

## Enforcement

| file | what it checks |
|---|---|
| `tests/test_api_registry.py` | Every method fits and predicts through one call shape. `accepts` matches each signature. The parameter record behaves as described. Every parameter shared by two or more families is either shown to change the fit or exempted with a reason. |
| `tests/test_interchangeable_api.py` | The documented loop runs with `strict=True` over every linear, deep and flow method. |
| `tests/test_random_state.py` | Seeding contract on every stochastic entry point. |
| `tests/test_sklearn_estimators.py` | `MultiViewSiMR`, `GridSearchCV` over `method`, single-view `transform` agreeing with `predict`. |
| `tests/test_implementation_standards.py` | No broad excepts at all (the D4 audit removed or narrowed all 37). No new global seeding, unscoped warning filters or mutable defaults. No `**kwargs` on a registered fit. Every parameter of a registered fit is read. Scripts, paper and docs use the current parameter names. |
| `tests/test_readme_examples.py` | The README Quick Start runs. |
