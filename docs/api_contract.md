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

`pysimlr.loading_inference` also records `median_converged_iter`, `oos_coupling_splits` (per-split `(k,)` list or `None`) and `n_effective_refits`, warns when `converged_fraction < 0.5`, and omits unconverged refits when `exclude_unconverged=True` (`select_rank` deduplicates the warning to one per `k`). `select_rank` additionally accepts `heldout_views=` (`RANK_CRITERIA` includes `"heldout_coupling"`), and `pysimlr.select_model(views, candidates, ks, ...)` evaluates `(candidate, k)` cells on one paired out-of-bag split design under the health gate, Fisher-z `oos_coupling_threshold` signal gate, and paired 1-SE parsimony rule.

**Unobserved rows (`missing_rows="mask"`).** `pysimlr.feature_maps(result, views, kinds=..., ..., min_rows=10, return_info=False)` computes every map for view `m` on the rows where view `m` is observed and its target exists (`target="other_views"`: the mean of the other views' standardized scores observed at that row; undefined, so dropped, where no other view is observed). With no unobserved rows the result is the complete-data result; when the views share their observed rows (any two views, or rows absent from every view) it equals the complete-case computation. A view with fewer than `min_rows` usable rows, a constant feature, or a constant latent column gives NaN (never 0) for the affected entries and a `MissingRowsWarning`; `return_info=True` returns `(maps, info)` with `n_eff`, `n_eff_feature`, `n_constant_features`, `constant_latent_columns`, `reasons[kind][view]` and `relative_weight`. `loading_inference` computes `oos_coupling` per split on rows with the needed views observed (NaN where undefined) and adds `oos_coupling_n` and, per map kind, `n_eff`, `n_eff_feature`, `reasons`. `feature_relevance` and `component_attribution` raise `ValueError` on all-NaN rows. Before v0.31.1 `feature_maps` returned all zeros and `oos_coupling` read 0.0 in this setting.

**Opt-in interpretation options (v0.32.0; all default to the previous behavior).** `groups=` (length-n labels) and `scale_within_group=False` on `structure_coefficients`, `loadings`, `sem.view_scores`, `feature_maps` and `loading_inference`: features, latent columns and the other-views target are centered within group (scaled globally, or within group when `scale_within_group=True`); groups with fewer than 3 usable rows for a view are left out with a `MissingRowsWarning`; `feature_maps(..., return_info=True)["groups"]` records `n_groups` and `left_out_groups` per view; composes with `missing_rows="mask"`. `stats.permutation_correlations(..., groups=)` centers, permutes and bootstraps within group and uses `n - G + 1` as effective n. `feature_maps(..., exclude_views=[...])` (indices or names; `target="other_views"` only; also via `loading_inference(map_options={"exclude_views": ...})`) leaves views out of every view's other-views target; the fitted consensus still contains them (refit without them for a non-circular test); a view with no non-excluded other view gets NaN maps with a reason. `loading_inference(..., groups=, scale_within_group=False, subsample="half"|"fraction"|"bootstrap", strata=None, return_refits=(), max_refit_bytes=2e9)`: with `groups` the permutation null permutes within group and `out["result"]` carries within-group loadings; `subsample="half"` (default) draws `round(fraction n)` rows without replacement exactly as before, `"bootstrap"` draws `n` with replacement (out-of-bag = never drawn), `strata` makes draws proportional within strata; `out["resampling"]` records the design, and `expected_false_selections` is NaN with `expected_false_selections_reason` unless the design is uniform half-sampling at `fraction=0.5`; `return_refits` (any of `"weights"`, `"maps"`, `"latents"`, `"rows"`) adds `out["refits"]` with `index`, `structure` `(B, sum p, k)`, `view_sizes`, `alignment` (`rotation` `(B, k, k)`: `u_refit @ rotation[b]` is the latent in the reference frame; `procrustes`, `loading_rotation`, `signed_permutation`), `weights` (per view `(B, p_m, k)`), `maps[kind]` (per view `(B, p_m, k)`), `latents` `(B, n, k)`, `in_bag` / `n_draws` `(B, n)`, `rows` / `oob_rows`; a `ValueError` is raised before fitting when the arrays would exceed `max_refit_bytes`. `pysimlr.orient_latents(latents, *, weights=None, maps=None, mode="signed_permutation", reference=None, max_iter=20)` returns `latents`, `transforms` `(F, k, k)`, `reference`, `reference_index`, `agreement`, `n_iter`, `converged` (and `weights` / `maps`); the reference rule (medoid by summed absolute agreement, mean refinement, components by mean norm, sign by non-negative sum of cubes) does not depend on the order of the fits.

`pysimlr.bottleneck_rank(views, *, confounds=None, mode="cross", bound="min", n_permutations=100, quantile=0.95, random_state=0)`
returns `k_max` (int, set by `bound` in `{"min", "max", "stacked"}`), `k_max_min`, `k_max_max`, `k_max_stacked`, `bottleneck_view` (int or `None` for `"stacked"`),
`per_view` (DataFrame: `view`, `n`, `p`, `cross_rank`, `within_rank`, `first_singular_value`,
`threshold_first`, `sigma1_over_tau95`, `tau95_rel_se`, `borderline`), `singular_values` and `thresholds` (cross-view, one array per view), `eigenvalues`
and `eigen_thresholds` (within-view parallel analysis), and `parameters` (`mode`, `bound`, `n_permutations`,
`quantile`, `random_state`, `confounds`). `confounds` takes the same shared, per-view or dict forms as
`fit`. `select_rank` accepts `ks="auto"` with default `bound="max"` (grid `1..k_max`, raising a `ValueError` with the worst view's `sigma1/tau95` when `k_max == 0`), warns when an explicit grid exceeds `k_max`,
and returns this dict as `result["bottleneck"]` (`None` for a single view). `predict` and `confound_decorrelation` accept any `Mapping` (including `MappingProxyType`) or partial view sequence with `None`. `feature_relevance` routes score extraction through `sem.view_scores` for joint-basis and per-view fits alike.

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
| `latents` | per-view first-layer projections (`[u]` for the joint baselines). For `simlr` each view's `X_m V_m` is divided by one constant, `latent_scale[m]`, fitted on the training rows. That puts the view at unit average column variance, the scale of `u`, and `predict` applies the same constant. Without it the preprocessing left them about 1e-3 the size of `u`, and a fixed-penalty estimator shrank them to nothing. Results saved before v0.16 have no `latent_scale` and keep the old scale. |
| `coupling_train`, `view_width_ratio`, `wide_views` | `simlr`: per view and component, |corr(latent, u)| on the training rows; p_m / n; views with p_m ≥ n, whose training coupling can be fitted to noise (`WideViewWarning`). |
| `convergence` | `converged`, `stop_reason`, `certificate`, `grad_map`, `n_steps`, `max_steps`, `tol`, `patience`, `best`, `final`. `stop_reason` is one of `api.STOP_REASONS` for every method: `converged` (the energy plateau rule fired, `certificate="energy_plateau"`; NSA-Flow: its own certificate), `max_iter` (budget exhausted), `no_descent` (`simlr`: no view's line search found a decrease), `non_finite`, `closed_form` (PCA). `converged` is `stop_reason == "converged"`. `grad_map` is the face residual `utils.stationarity_residual` of the basis, a diagnostic only, not a stop rule, and does not certify convergence (on its old calibration fixture it falls 0.86 decades, against 2.8 before): per sweep for `simlr`, post hoc on the returned `V` and `u` for deep/flow (`api._basis_grad_map`), 0 for PCA |
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

### View order and sign alignment (`sign_alignment`, default `"spectral"` since v0.31.0)

The averaging mixings (`mixing_algorithm="avg"` and the default `"newton"`) average the views'
latent columns after putting them on one sign convention, because a latent is defined only up to a
sign. `sign_alignment` chooses the rule:

- `"spectral"` (default): each column is flipped by the sign of the leading eigenvector of the
  `M x M` Gram matrix of the views' columns. It is symmetric in the views, so the consensus does not
  depend on the order of `data_matrices`. It equals the legacy rule whenever the views already agree
  in sign.
- `"greedy"` (legacy, v0.30.0 and earlier): the first view is the reference and each next view is
  flipped to agree with the running sum. With three or more views whose latents disagree in sign
  (typical at a PCA start with weakly coupled views) the result depends on view order; measured
  first-sweep consensus differences up to 3.6 (unit-SD `u`).

For two views both rules give each column the same relative sign (they can differ only by the
global sign of a column of `u`). The option is on `simlr`, `compute_shared_consensus`, and the
deep and flow entry points (`lend`, `ned`, `ned_shared_private`, `flow_v`, `flow`), and is recorded
as `result["sign_alignment"]` for `simlr`. The `svd`, `pca` and `ica` mixings ignore it. The
leave-one-out agreement scores behind `dynamic_weights` always use the spectral rule. Sign alignment
does not resolve component swaps or rotations between views when `k > 1`, and training order
effects that come from the optimizer (nonlinear `ned` fits are chaotic) or from random
initialization drawn in view order (`random`, `perturbed_pca`, deep weight initialization) remain.
See `docs/view_order_dependence_report.md`.

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

When generative factors are correlated, newton decorrelates consensus scores. avg keeps
correlation only up to about the collapse threshold, `collapse_threshold` (default 0.3; converged
fits peaked at 0.31, 0.51 and 0.69 for thresholds 0.3, 0.5 and 0.7). Raising it lets correlated
factors keep more correlation, but it weakens the protection against collapse on narrow views
(docs/tracking_amigo.md, D3; docs/audit/evidence/collapse_threshold_selection.csv).

## Structural model

`pysimlr.path_model`, `pysimlr.compare_path_models`, and `pysimlr.predict_path` provide
score-level dependency graph estimation, model comparison, and path prediction post-fit.
`pysimlr.path_model_from_scores(scores, ...)` runs the same inference on a mapping of node name
to `(n, k_m)` score block from any source (standardized column-wise unless `standardize=False`);
it returns the `path_model` contract below with `result=None`, `scores_source="given"`,
`signs=None`, `wide_views=[]`, an empty `feature_contributions`, and `parameters["scores_given"]
= True`. It has no refit null (`null="fixed"` only), no `inference=`, and no consensus node (`plot_edge_features` and `plot_edge_brain_map` raise `ValueError` when `pm["result"] is None`). `pysimlr.compare_path_models_from_scores(scores_train, scores_test, graphs, *, standardize=True, ...)` runs structural graph comparison directly on score mappings (`scores_source="given"`, where `standardize=True`/`"per_split"` z-scores each split independently to match `compare_path_models`, `standardize="train"` scales `scores_test` with `scores_train` statistics, and `standardize=False` leaves scores unscaled). On `rows="heldout"`, `path_model`, structural `compare_path_models`, and `compare_path_models_from_scores` accept `None` for views absent on evaluation rows (dropping absent views with a `RuntimeWarning` and storing `orig_view_indices` on `pm`, while raising `ValueError` if `graph` references an absent view).

Training-row policy: with `rows="train"` and the fixed-model null, `p_perm`, `p_maxt`, `p_holm`,
`q_bh`, the omnibus p-values and the graph-fit p-values are `None` / NaN for every view unless
`allow_train_fixed=True` or a non-empty reason string (recorded in `parameters["allow_train_fixed_reason"]` and echoed in `warnings`); `p_reason` says why. Every returned `edges`, `edge_components` and `outcome_edges` frame carries `null_rows` (`heldout-fixed`, `train-refit`, `train-fixed`, `train-fixed-unreported`, or `train-both`), mirrored in `fit["evidence"]`. `n_jobs > 1` spawns worker processes and must be
called under a `__main__` guard or from an importable module.

### `path_model` result contract

`path_model` returns a dictionary with the following return keys:

| key | type | content |
|---|---|---|
| `nodes` | `List[str]` | Node names (views, optional `"outcome"`, optional `"consensus"`). |
| `graph` | `Dict[str, List[str]]` | Adjacency dictionary of the hypothesized graph over `nodes`. |
| `directed` | `bool` | Whether edges in `graph` are treated as directed. |
| `conditioning` | `str` | Conditioning strategy: `"pairwise"` (bivariate given nuisance) or `"parents"` (conditioned on adjacent graph neighbors). |
| `rows` | `str` | `"heldout"` or `"train"`. |
| `n` | `int` | Number of subjects in evaluation rows. |
| `k` | `int` | Latent component dimensionality. |
| `edges` | `pd.DataFrame` | Edge statistics across all tested pairs: `a`, `b`, `present` (hypothesized), `diagnostic` (e.g. consensus edge), `pillai` (mean squared canonical correlation in $[0, 1]$), `rv` (adjusted RV), `p_perm` (Freedman–Lane add-one permutation p-value, floor $1/(B+1)$), `p_maxt` (family-wise max-$T$ p-value), `p_holm` (Holm step-down), `q_bh` (Benjamini–Hochberg FDR), `p_refit` (refit null p-value or NaN), `p_refit_holm`, `ci_low`, `ci_high` (95% bootstrap CI), `ci_refit_low`, `ci_refit_high` (out-of-bag refit intervals from `inference=`). |
| `edge_components` | `pd.DataFrame` | Per-component statistics in rotated reference frame: `a`, `b`, `component` ($j \in \{0, \dots, k-1\}$), `r` (partial correlation), `beta` (regression slope), `p_perm`, `p_maxt`, `p_holm`, `q_bh`, `ci_low`, `ci_high`, `beta_ci_low`, `beta_ci_high`. |
| `moderation` | `Optional[pd.DataFrame]` | Group moderation tests (`None` if `groups` not provided): `a`, `b`, `sum_f` (sum over components of $F$ for group interaction $R_{a,j} \cdot G$), `p_perm`, `p_maxt`, `p_holm`, `q_bh`. |
| `outcome_edges` | `Optional[pd.DataFrame]` | Edges into an external clinical outcome (`None` if `outcome` not provided): `a`, `b`, `pillai`, `p_perm`, `p_maxt`, `p_holm`, `q_bh`, `ci_low`, `ci_high`. |
| `omnibus` | `Dict[str, Any]` | Whole-graph statistics: `max_pillai`, `mean_pillai`, `p_max`, `p_mean`, `null_mean_pillai` (list of null means), `n_edges`, `p_mean_refit`, `p_max_refit`, `null_mean_pillai_refit`. |
| `fit` | `Dict[str, Any]` | Graph fit evaluation (`conditioning="parents"`): `conditioning`, `null`, `present_pass` (fraction of hypothesized edges with $p_{\text{holm}} < \alpha$), `absent_pass` (fraction of absent edges with unadjusted $p_{\text{perm}} \ge \alpha$), `absent_p` (list of unadjusted absent p-values), `max_absent_pillai`, `max_absent_p` (max-$T$ p over absent edges, SEM $\chi^2$ analogue), `rejected` (bool: `max_absent_p < alpha` or `present_pass < 1.0`), `reason`. |
| `families` | `Dict[str, Dict[str, Any]]` | Declaration of hypothesis families (`edges`, `edge_components`, `moderation`, `outcome_edges`), each with `n_tests` and `statistic`. |
| `edge_models` | `Dict[str, Dict[str, Any]]` | Fitted ridge node regressions per node: `W`, `parents`, `parent_cols`, `n_nuisance`, `ridge_lambda`, `res_var`. Used by `predict_path`. |
| `feature_contributions` | `pd.DataFrame` | Feature-level contributions to edges: `a`, `b`, `feature`, `feature_idx`, `component`, `structure`, `beta`, `contribution` ($\text{structure}_{a,j} \cdot \beta_{ab,j}$). |
| `result` | `Optional[Dict[str, Any]]` | The underlying `pysimlr.fit` measurement model result; `None` from `path_model_from_scores`. |
| `refit_null` | `Optional[Dict[str, Any]]` | Dictionary of per-edge refit null distributions (`per_edge`, `outcome_edges`) or `None`. |
| `nuisance_names` | `List[str]` | Column names of the nuisance design matrix $N$ (intercept, covariates, group dummies). |
| `null` | `Dict[Tuple[str, str], List[float]]` | Stored permutation null Pillai draws (up to 2000) per edge for null distribution visualization. |
| `scores_source` | `str` | `"latents"` (linear and deep models), `"joint_block"` (joint-basis fits like PCA) or `"given"` (`path_model_from_scores`). |
| `sign_rows` | `str` | `"training"` or `"evaluation"`, indicating which rows determined component sign alignment. |
| `signs` | `List[Optional[np.ndarray]]` | Sign vectors per view ($\pm 1$) aligned to consensus. |
| `wide_views` | `List[bool]` | Whether each view is wide ($p_m \ge n_{\text{train}}$). |
| `p_reason` | `Optional[str]` | Explanation when permutation p-values are not reported (e.g. $B=0$ or wide training rows). |
| `parameters` | `Dict[str, Any]` | Resolved call parameters: `rows`, `null`, `n_permutations`, `n_refits`, `refit_scope`, `n_bootstrap`, `alpha`, `random_state`, `n_jobs`, `allow_train_fixed`, `conditioning`, `include_consensus`, `directed`, `inference`, `n_refit_pillai`, `residualization`, `chunk`, `standardize`, `scores_given`. |
| `warnings` | `List[str]` | List of runtime warnings issued. |

### `compare_path_models` result contract

`compare_path_models` operates in one of two modes:

#### 1. Structural comparison mode (`result` provided, measurement model fixed)
Returns a dictionary with:
- `comparison`: List of dictionaries (one per evaluated graph in `graphs`), containing:
  - `graph`: Graph name.
  - `n_edges`: Number of edges (undirected edges counted once).
  - `heldout_loglik`: Sum of Gaussian log-likelihoods of held-out scores under node regressions (fitted with GCV ridge [Golub, Heath & Wahba 1979], PRESS leave-one-out hat-matrix residual variance [Allen 1974], and `consensus.view_dofs` shrinkage $\gamma = (n - 1) / (n - 1 + \text{dof}_{\text{eff}})$ [Ezekiel 1930; Wherry 1931]).
  - `n_params`: Parameter count ($k \times k$ per edge plus nuisance block per node).
  - `bic`: In-sample BIC ($n_{\text{params}} \ln(n_{\text{train}}) - 2 \ln L_{\text{train}}$).
  - `present_pass`: Fraction of present edges significant on held-out scores.
  - `absent_pass`: Fraction of absent edges independent on held-out scores.
  - `max_absent_p`: Max-$T$ p-value across absent edges.
  - `edges`: Edge statistics DataFrame from `path_model`.
  - `delta_vs_top`: Held-out log-likelihood difference relative to `best_by_loglik`.
  - `ci_low`, `ci_high`: 95% bootstrap confidence interval of `delta_vs_top`.
  - `best_fraction`: Fraction of bootstrap replicates in which this graph achieved top log-likelihood.
- `best`: Parsimonious winner: graph with fewest edges whose 95% delta CI includes 0 (ties broken by log-likelihood).
- `best_by_loglik`: Graph with highest raw held-out log-likelihood.
- `best_rule`: Description of selection rule used.
- `nodes`: Evaluated node names.
- `mode`: `"structural"`.
- `parameters`: `n_bootstrap`, `n_permutations`, `random_state`, `alpha`.

#### 2. Training-topology comparison mode (`method` and `k` provided, refits across splits)
Returns a dictionary with:
- `splits`: DataFrame with one row per `(split, graph)` recording:
  - `split`: Split index ($0 \dots n_{\text{splits}}-1$).
  - `graph`, `topology`: Graph name and topology (`"star"` or `"graph"`).
  - `n_edges`, `n_test`: Edge count and test sample size.
  - `heldout_loglik_own`: Held-out log-likelihood under own graph fit.
  - `heldout_loglik_complete`: Held-out log-likelihood under complete graph fit.
  - `present_pass`, `absent_pass`: Graph fit pass rates on test split.
  - `oos_coupling`: Mean held-out leave-one-view-out consensus coupling.
  - `oos_r2`: Mean adjusted held-out reconstruction $R^2$.
  - `outcome_r2`: Held-out $R^2$ of linear outcome prediction from consensus (if outcome passed).
- `summary`: DataFrame aggregating criteria across splits (mean values, `delta_<crit>` vs top model, min/max deltas, and split win counts `wins_<crit>`).
- `best`: Dictionary mapping each criterion to its top-performing graph name.
- `nodes`: Evaluated node names.
- `test_rows`: List of held-out test index arrays per split.
- `mode`: `"training_topology"`.
- `parameters`: `method`, `k`, `n_splits`, `train_fraction`, `n_permutations`, `random_state`, `alpha`.

### `predict_path` result contract

`predict_path` reconstructs missing views and outcomes across path edges and returns the keys
below. For a `path_model_from_scores` result, `views` is a mapping of the model's node names to
score blocks (`None` or absent for a missing block), standardized as the model was.

| key | type | content |
|---|---|---|
| `scores` | `Dict[str, np.ndarray]` | Dictionary mapping each node name to its $(n, k)$ standardized score array (observed scores for present views, ridge predictions for missing views). |
| `predicted_from` | `Dict[str, List[str]]` | Dictionary mapping each predicted node name to the list of available neighbor nodes used to predict it. |
| `outcome_pred` | `Optional[np.ndarray]` | Predicted continuous outcome array $(n, r)$ if `outcome=True` and model includes an outcome node, else `None`. |
| `r2_per_node` | `Optional[Dict[str, float]]` | Out-of-sample reconstruction $R^2$ ($1 - \text{SSE} / \text{SST}$) against true scores for each predicted view (requires `views_true`), else `None`. |

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

## Target attribution (`attribute_target`)

`pysimlr.attribute_target(views, target, *, covariates=None, groups=None, method="structure", reference=None,
n_null=0, n_bag=0, sparseness=0.8, n_folds=5, bag_fraction=0.8, fdr=0.1, generic_tol=0.1, random_state=0,
n_jobs=1, fit_options=None)` fits the target as the root view of a one-component target-rooted `simlr` model
(`pysimlr.target_root_fit_params`) and returns signed per-view `maps` (positive = the feature rises with the target),
the out-of-fold `score` with a `predictive` flag, a `generic` PC1 diagnostic, optional `uncertainty` and `inference`,
and `result` / `frame` for reuse as `reference=`. Meaning, evidence and limits: `docs/reliable_attribution.md`.
Additive; no existing function changed except as below.

Also added: `feature_maps(..., return_target=True)` (the target each view was mapped against, in `extra["target"]`),
`loading_inference(..., draw_random_state=None)` (seeds only the row draws; `None` keeps `random_state`), and
`stats.permutation_correlations` drops masked rows (all-NaN features, non-finite outcome) with a `MissingRowsWarning`
instead of raising.

**Behaviour change.** `loadings(...)["coupling_signs"]` and `view_coupling_signs` now use the positivity and energy
each view actually used (`per_view_params` overrides); a conical `nc_abs` view fitted through `per_view_params` that
couples negatively to the consensus now gets `-1` instead of `+1`.

## Removed since 0.3.0 (in 0.4.0, 0.5.0 or 0.6.0; no aliases; each raises or is gone)

- Method-name aliases (`linear`, `shared_private`, `nedpp`, `flow_simr_v`, `nsaflow`, ...):
  use the registered names above.
- `deep_simr`, `flow_simlr`, `flow_simlr_v`, `FlowSiMLRModel`, `FlowSiMLRVModel`: use
  `lend_simr`, `flow_simr`, `flow_simr_v`, `FlowSiMRModel`, `FlowSiMRVModel`.
- `FlowWhitener`, `flow_whiten_matrix` and `flow_whitener`: the ANTsTorch LAMNr-Flows whitener
  wrapper. Nothing in pysimlr used it. The upstream whitener defect in ANTsTorch was fixed in
  commit `ed39d99` (`apply_lamnr_flows_whitener.py:161` solves with L); on an 8D Gaussian (n = 2000,
  condition ~ 50), output covariance reaches ‖cov − I‖_F = 0.1767 < 0.25 and max |off-diagonal| = 0.0329 < 0.1
  (`scripts/antstorch_whitener_check.py`, `docs/audit/evidence/antstorch_whitener_check.csv`, Rule R10). Call ANTsTorch directly (`lamnr_flows_whitener`).
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
