# pysimlr method interface (v0.3.0)

One registry, one `fit`, one `predict`, one record of which parameters each method used.
Implemented in `src/pysimlr/api.py`; scikit-learn layer in `src/pysimlr/sklearn.py`.

## Registered methods

| name | family | native fit | budget unit |
|---|---|---|---|
| `simlr` | linear | `simlr` | iterations (sweeps) |
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
| Fixed by a preset | As above, with reason "fixed by method". |
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

Schedule offsets keep their `_epoch(s)` suffix: `warmup_epochs`, `stabilization_start_epoch`,
`stabilization_ramp_epochs` and `shared_warmup_epochs`. They are measured in the method's
budget unit.

`energy_type`, `nsa_w`, `mixing_algorithm` and `topology` take their defaults from one
place, `src/pysimlr/defaults.py`, for every method (`"nc"`, 0.5, `"newton"`, `"loo"`);
`compute_shared_consensus` defaults to the same topology. `nsa_w` is the one orthogonality
weight (NSA-Flow retraction plus the calibrated defect term; 0 = no orthogonality
constraint, sparsity still applied). The remaining differences, from
`scripts/method_settings_table.py` (`docs/audit/evidence/method_settings.csv`):

| setting | linear (`simlr`) | deep / flow | kind |
|---|---|---|---|
| budget `iterations` | 300 sweeps (`DEFAULT_SIMLR_ITERATIONS`) | 150 epochs | intrinsic unit |
| `tol` | 1e-4 (`DEFAULT_TOL`) | 1e-4 (`DEFAULT_TOL`) | one constant |
| `learning_rate` | `"auto"` | 0.0005 | intrinsic (different optimizers) |
| `optimizer_type` | `hybrid_adam` | `adam` | intrinsic |
| `collapse_weight` | parameter (1.0) | fixed at 1.0 | open (D4) |
| `batch_size`, `warmup_epochs` | -- | 64, auto | intrinsic (stochastic training) |

`parameters["defaults"]` records the value each fit used.

### Interpretation and inference

`structure` and `loadings` are point estimates. `u` was fitted on the same rows, which inflates
`|structure|` (with the cross-view link destroyed by permutation, in-sample `max |C|` reached
0.4-0.5 where ~0.07 is expected). Feature-level claims use `pysimlr.loading_inference(method,
views, k, n_subsamples=100, fraction=0.5, threshold=0.8, n_permutations=...)`: whole-pipeline
refits on half-samples, Procrustes-aligned to the reference; percentile intervals per feature,
per-component stability (claims per component need it high, e.g. >= 0.9), Meinshausen-Buhlmann
stability selection with its bound `q^2 / ((2*threshold - 1) p)`, and a permutation null for
`max |structure|`.

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
| `u` | `(n, k)` tensor. For `simlr` it is the star consensus of the fitted views under every topology; the per-view targets of `loo`/`graph` are in `u_per_modality`. |
| `v` | one `(p_m, k)` block per view: the encoder (`u ~ X v`) that scores new subjects. Weakly identified in each view's low-variance directions, so not the object to read feature by feature. |
| `structure` | one `(p_m, k)` block per view: `corr(x_f, u_j)` in the varimax basis of all views stacked, column signs set so dominant loadings are positive (`pysimlr.loadings`). Association, not unique contribution. |
| `loadings` | one `(p_m, k)` block per view: the OLEE summary of `structure`, with orthogonal columns. `positivity="positive"`: each feature assigned to the component it correlates with most positively (disjoint supports); otherwise the polar factor of `structure` at its column norms. |
| `loading_rotation`, `loading_margin` | the `k x k` rotation `R` (the columns of `structure`/`loadings` describe `u @ R`); per feature, top minus second positive correlation (positive only, else `None`). |
| `latents` | per-view first-layer projections (`[u]` for the joint baselines) |
| `convergence` | `converged`, `stop_reason`, `certificate`, `grad_map`, `n_steps`, `max_steps`, `tol`, `patience`, `best`, `final`. `stop_reason` is one of `api.STOP_REASONS` for every method: `converged` (the energy plateau rule fired, `certificate="energy_plateau"`; NSA-Flow: its own certificate), `max_iter` (budget exhausted), `no_descent` (`simlr`: no view's line search found a decrease), `non_finite`, `closed_form` (PCA). `converged` is `stop_reason == "converged"`. `grad_map` is the face residual `utils.stationarity_residual` of the basis, a diagnostic not a stop rule: per sweep for `simlr`, post hoc on the returned `V` and `u` for deep/flow (`api._basis_grad_map`), 0 for PCA |
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

### View weighting

Multi-view similarity terms are combined using per-view weights (`res["normalizing_weights"]`).
Scale-free terms (degree-0 homogeneous or bounded, defined in `similarity.SCALE_FREE_ENERGIES`:
`nc`, `align`, `procrustes`, `recon_r2`, `logcosh`, `exp`, `kurtosis`, and `negentropy`) receive
equal weight `1/M` across the `M` views (`res["normalizing_weights"] == [1/M]*M`). This eliminates
arbitrary initialization-dependent view weight spreads (previously 1.05–279× on synthetic default
fits, up to ~3200× in WP-1 cells, and 70× on ADNI positive, which could cause total energy to fall
below -1) and aligns linear SiMLR with deep models which sum view terms unweighted. For
`energy_type="nc"`, total energy is bounded below: `min(res["energy"]) >= -1 - 1e-6`.
`acc` and `recon` weight each view by 1/(|E0|·M), where E0 is the view's energy at the start of
the fit, so their relative view weights depend on the initialization.

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

### Limitations: Correlated factors and consensus representation

When generative factors are correlated, newton decorrelates consensus scores; avg retains correlation, which in one benchmark saturated near the collapse threshold (0.30), measured on fits that were partly unconverged; this is under evaluation (docs/tracking_amigo.md, D3).

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

## Removed since 0.3.0 (in 0.4.0, 0.5.0 or 0.6.0; no aliases; each raises or is gone)

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
- Optimizers `torch_lbfgs`, `nsa_flow`, and `nsa_lbfgsb`, and methods `simlr_lbfgs` and `simlr_lbfgsb` (`nsa_lbfgsb` and `simlr_lbfgsb` in 0.6.0: not competitive with `hybrid_adam`; use `'simlr'` with the default optimizer `'hybrid_adam'`).
- `simlr(constraint=...)`, `parse_constraint`, `project_gradient` and the 13 constraint type
  strings (`Stiefel`, `Grassmann`, `NewtonSchulz`, `ortho*`, `nsaflow*`): every non-`none` type
  ran the same NSA-Flow retraction. Use `nsa_w` (0 = no orthogonality constraint).
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
