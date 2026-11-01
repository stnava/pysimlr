# pysimlr: correctness review and one central method interface

Repo: `/Users/stnava/data/repos/pysimlr` (v0.2.19, main). Baseline before any change:
`pytest` (fast suite) 1405 passed, 6 xfailed, 0 failed, 167 s. Working tree carries
uncommitted paper/manuscript edits (22 files) that this work does not touch.

## Context

pysimlr has five multi-view fitting entry points (`simlr`, `lend_simr`, `ned_simr`,
`ned_simr_shared_private`, `flow_simr_v`; thin aliases `deep_simr`, `flow_simr`, `flow_simlr`,
`flow_simlr_v`) and two baselines (`pca`, `nsa_pipeline`) that exist only as branches inside the
benchmark runner. Each entry point has its own parameter vocabulary, and the translation between
vocabularies is re-implemented by hand at five places:

- `src/pysimlr/benchmarks/runner.py:run_single_experiment` (if/elif over 9 `model_type` strings, lines 286-441)
- `src/pysimlr/nnh.py` (`nnh_embed` `use_flow` branch, lines 431-484; `extend_simlr_embedding_with_new_modalities` `method` branch, lines 1155-1190)
- `src/pysimlr/sklearn.py` (four near-identical estimator classes)
- `scripts/run_deep_benchmark.py` (own `iterations` vs `epochs` table, lines 348-352, duplicated label table at 446-456 and 506-516)
- `tests/test_interchangeable_api.py` (its own `ALGORITHMS` / `COMMON_PARAMS` / `REQUIRED_KEYS`)

Every one of those translation layers has drifted from the entry points it wraps (findings below).
The owner wants one central interface whose parameters map consistently onto every sub-algorithm,
whose common output records which parameters were used and which were not, and which makes
"loop over the methods" a one-liner in benchmarks, with a scikit-learn layer on top. Backward
compatibility is explicitly not a constraint: shims are removed, not added.

Owner decisions taken during planning (2026-11-01):
1. A parameter a method cannot use is **recorded and warned once** (`strict=True` raises); the
   native entry points raise `TypeError` on unknown keywords.
2. Names are **unified at the native entry points** (one name per concept everywhere), not only
   translated centrally.
3. One sklearn class, **`MultiViewSiMR(method=...)`**, with every canonical parameter explicit;
   the four existing estimator classes are removed.
4. Verified bugs are fixed in the same pass; the rest of the correctness findings become tracked
   follow-up work packages.

## Correctness review: findings

"Verified" = reproduced by execution or confirmed by reading the cited lines during planning.

### A. Interface drift and silent parameter loss (verified)

| # | Finding | Where |
|---|---|---|
| A1 | README quick start is wrong in 2 of 3 snippets. `simlr(optimizer="lbfgsb", random_state=42)` runs the default `armijo_gradient` with no seed (both keys fall into `**opt_params`, 8 repeated `UserWarning`s). `SiMLR(optimizer=..., sparseness=...)` raises `TypeError`. `lend_simr(random_state=42)` warns and ignores the seed. | `README.md:33-113`, `optimizers.py:1157` |
| A2 | Unknown keywords never raise at any entry point: `simlr` forwards to the optimizer (warn), deep entry points `_warn_unused_kwargs` (warn), `flow_simr` raises, `nnh_embed` flow path drops them silently via `inspect.signature` filtering. Four behaviours for one mistake. | `simlr.py:1684`, `deep.py:42-55`, `nnh.py:462-465` |
| A3 | Single-view `SiMLREstimator.transform` projects raw X onto `loadings_` although `fit` standardized X (`scale_list=["centerAndScale","np"]`); measured max abs difference 30.0 against `predict_shared_latent` on a scaled input (latent magnitude 10.2 vs 0.81). `fit_transform` therefore disagrees with `result_["u"]`. | `sklearn.py:224-227` |
| A4 | `LENDTransformer`/`NEDTransformer` pass both `iterations=` and `epochs=`; the callee runs `epochs = kwargs.pop("iterations", epochs)`, so `epochs` (default 30) is dead and training runs `iterations` (default 10). `LENDTransformer.consolidate` and `SiMLREstimator.use_nsa` are stored and never forwarded. `random_state` seeds global torch/numpy RNG instead of being passed as a seed. | `sklearn.py:155-156, 346-357, 420-431` |
| A5 | `deep_simr` does not forward `optimizer_type`; it is a dead parameter. | `deep.py:2408` |
| A6 | Unknown `optimizer_type` on the deep/flow path silently becomes Adam. | `deep.py:58-70` |
| A7 | `simlr` pops `patience` after `create_optimizer`, so passing `patience` draws a spurious "unrecognised optimizer parameter" warning although it is used. | `simlr.py:1684` vs `1743` |
| A8 | `init_seed` with the default `initialization_type="pca"` silently switches the initializer to `"random"`. A benchmark that sets a seed on every method therefore changes the linear method's algorithm. | `simlr.py:1655-1658` |
| A9 | Benchmark `filter_kwargs` drops valid aliases (`iterations` for deep, `epochs`/`sparseness_quantile` for `simlr`) and lets optimizer keys through to deep functions that then warn. `pca`/`nsa_pipeline` branches ignore every parameter without recording it. | `runner.py:18-49, 322-439` |
| A10 | `paper/build_paper_experiments.py:234` calls `ned_simr_shared_private(k_shared=..., k_private=3)`; neither name exists, so the generated experiment raises `TypeError`. | `paper/build_paper_experiments.py:234` |
| A11 | `nnh_embed` does not forward `constraint` on the `simlr` path. | `nnh.py:493-500` |
| A12 | `flow_simr` returns no `v`, `convergence`, `sparseness` or rank keys (its convergence report is computed and discarded at `flows.py:701`), so it cannot be scored by the shared contract. | `flows.py:713-731` |

### B. Software defects (verified by reading)

| # | Finding | Where |
|---|---|---|
| B1 | `compute_shared_consensus` calls `warnings.filterwarnings("ignore", ConvergenceWarning)` at call time, permanently changing process-wide filters. | `consensus.py:136` |
| B2 | Tensor truthiness in `or` chains: `result.get('V') or result.get('Y')` raises on a multi-element tensor. Three sites, two outside any `try`. | `sparsification.py:215-217`, `simlr.py:948-950`, `optimizers.py:803` |
| B3 | `except Exception: pass` / `continue` in the benchmark runner (rank report, per-view grad-map term) silently drop reported numbers. | `runner.py:190, 538` |
| B4 | `deep_ranking` substitutes hard-coded critical-range constants (`2.948`/`2.569`) when `studentized_range.ppf` fails, and `p_wilc = 1.0` when Wilcoxon fails. | `benchmarks/deep_ranking.py:227, 338` |
| B5 | 42 broad handlers in `src/pysimlr`; 37 substitute a value and continue without warning (full table in the planning survey; e.g. `pinv` failure -> zero `W` at `simlr.py:2110` and `nnh.py:485`, FastICA failure -> SVD basis at `simlr.py:568`, MAI failure -> `0.0` at six copies). | tracked as D3 follow-up |
| B6 | Mutable default arguments (`hidden_dims=[128, 64]` x5, `scale_list=[...]`, `covariates=[]`, `weights={...}`, runner/synthetic-case lists). None mutated in place today. | `deep.py:686,737,987,1506,1557,1926,2063,2197`, `simlr.py:1314`, `nnh.py:305,942` |
| B7 | Global RNG seeding inside library code: the four sklearn `fit` methods, `runner.run_single_experiment:274`, `protocol.py:204`, every `synthetic_cases.build_*`. | as listed |
| B8 | `generate_all_architecture_graphs` is defined twice in `visualization.py` (296, 393); the second shadows the first with a different default. | `visualization.py` |
| B9 | No `tests/test_implementation_standards.py`; nothing enforces the absence of new broad excepts, global seeding, or mutable defaults. | tests/ |
| B10 | Duplicated helpers: list-to-tensor conversion (11 sites), MAI update (6 copies), `set_projection_schedule` (5), polar projection (3), sign alignment (4). | survey table |

### C. Open scientific items (not touched by this plan; recorded for tracking)

`docs/audit/AUDIT_2026_09.md` A1-A4 have no resolution status (regression gradient vs energy,
zero-gradient energies, inconsistent defect functionals, energy/gradient orthogonality mismatch);
`CORRECTNESS_AUDIT.md` leaves anchored prediction batch-dependent (strict xfail) and the
"iteration does not improve on SVD init" question open. `DOCS_AUDIT.md` marks every symbol's
correctness `TBD`.

## Design: one registry, one `fit`, one `predict`, one record of parameters

New module `src/pysimlr/api.py`, exported from `pysimlr/__init__.py` (eager; it imports only
the modules already imported there).

### Canonical parameter vocabulary

One name per concept at every level (native entry point, `api.fit`, sklearn):

| canonical | replaces | notes |
|---|---|---|
| `k` | `n_components` (sklearn keeps `n_components` because sklearn requires it) | |
| `iterations` | `epochs` | unit per method is recorded as `budget_unit` ("iterations" linear, "epochs" deep/flow, "none" closed-form). `epochs` is deleted from deep/flow signatures. |
| `sparseness` | `sparseness_quantile`, `sparsity` | entry points and sklearn only; the `nn.Module` constructors keep `sparseness_quantile` (internal, follow-up WP) |
| `random_state` | `init_seed`, sklearn global seeding | seeds a local `torch.Generator` for initialization. Deep/flow fits run inside `torch.random.fork_rng(devices=[])` (CPU; pass the device list when a CUDA device is used) opened **before `_standardize_deep`** (`deep.py:2027`, `flows.py:1177`; it draws `randn_like` noise on near-constant columns at `deep.py:176`) and closed after `_finalize_bases` and the final eval forward pass, with `torch.manual_seed(seed)` inside and `generator=torch.Generator().manual_seed(seed)` passed to every `DataLoader(shuffle=True)` (`deep.py:2041, 2175, 2328`, `flows.py:1209`). When `random_state is None`, one seed is drawn from the global RNG *before* the fork (`int(torch.randint(2**31, (1,)))`) and recorded as `parameters.used.random_state`, so two unseeded calls still differ (the trap `nsa_backend._rng_preserving` documents at `nsa_backend.py:62-70`). **Does not change `initialization_type`** (fixes A8). |
| schedule offsets | (unchanged) | `warmup_epochs`, `stabilization_start_epoch`, `stabilization_ramp_epochs`, `shared_warmup_epochs` and the `_train_loop(epochs=...)` local keep their names: they are offsets measured in the method's budget unit, and `budget_unit` says what an epoch is. "One name per concept" applies to the budget parameter only. |
| `energy_type`, `mixing_algorithm`, `positivity`, `nsa_w`, `topology`, `path_graph`, `varimax`, `initialization_type`, `domain_matrices`, `dynamic_weights`, `mai_metric`, `use_rank_mai`, `optimizer_type`, `tol`, `patience`, `verbose`, `device` | already shared | `optimizer_type` keeps its name but valid values are per method and listed by `describe_method`; unknown values raise everywhere (fixes A6). |
| method-specific | `constraint`, `consolidate`, `scale_list`, `smoothing_matrices`, `negentropy_weight`, `nonneg_u`, `orthogonalize_u`, `learning_rate`, optimizer hyperparameters (linear); `hidden_dims`, `dropout`, `batch_size`, `weight_decay`, `sim_weight`, `warmup_epochs`, `use_nsa`, `first_layer_mode`, `nsa_iterations`, `retraction_type`, `prune_threshold`, stabilization knobs (deep); `private_k`, `private_*_weight`, `shared_warmup_epochs` (nedpp); `num_layers`, `hidden_dim`, `beta`, `scale_bound`, `gamma`, `force_fallback` (flow) | unchanged names |

Defaults that differ between methods (e.g. `energy_type`: `"acc"` linear vs `"regression"`
deep/flow, `topology`: `"star"` vs `"loo"`) are **not** changed here: that is a scientific
semantics change. `api.fit` records the resolved default per method so a benchmark table can show
it. `TODO(human)`: decide whether to align these defaults in a later release.

### Registry

```python
@dataclass(frozen=True, slots=True)
class MethodSpec:
    name: str                 # "simlr" | "lend" | "ned" | "ned_shared_private" | "flow_v" | "flow" | "pca" | "nsa_pipeline"
    family: str               # "linear" | "deep" | "flow" | "baseline"
    fit: Callable[..., dict]  # native fit: (views, k, **native_kwargs) -> dict
    predict: Callable[..., dict]
    accepts: frozenset[str]   # canonical parameter names this method consumes (derived from inspect.signature of fit, plus optimizer keys for simlr)
    budget_unit: str
    aliases: tuple[str, ...]  # "linear", "simlr_lbfgs" (preset), "flow_simr_v", "shared_private", ...
    presets: Mapping[str, Any]  # fixed native kwargs for alias methods, e.g. simlr_lbfgs -> optimizer_type="torch_lbfgs"
    description: str

METHODS: dict[str, MethodSpec]          # single source of truth, like similarity.SIMILARITY
def method_names(family: str | None = None) -> list[str]
def resolve_method(name: str) -> MethodSpec   # aliases, presets; unknown -> ValueError listing names
def describe_method(name: str) -> dict        # accepted params with defaults, budget unit, family
```

`accepts` is derived from `inspect.signature(spec.fit)` at registry build time (not hand-typed),
so a native signature change cannot drift from the registry; a test asserts the derived set
equals the documented set. For `simlr` the set is
`named ∪ (SIMLR_OPTIMIZER_DEFAULTS − {k, tol, positivity, nsa_w, use_nsa}) ∪ {patience}`:
`SIMLR_OPTIMIZER_DEFAULTS` (`optimizers.py:15-37`) carries keys that collide with named `simlr`
parameters and can never reach the optimizer (`k` is the Lookahead period, `tol` the LBFGS-B
tolerance). WP1 renames them `lookahead_k` and `lbfgsb_tol` and deletes `use_nsa` from the
defaults so the collision disappears; `simlr(use_nsa=...)` then raises `TypeError` naming the
replacement (`constraint='none'`), consistent with "no shims".

### `fit` and `predict`

```python
def fit(method: str, views, k: int, *, strict: bool = False, **params) -> dict
def predict(result: dict, views) -> dict   # {"u", "latents", "reconstructions", "errors"}, dispatched on result["method"]
```

`fit`:
1. `resolve_method`, convert `views` once with one shared helper (`api._as_views`, replacing the
   11 ad-hoc conversions over time; this pass uses it in api/sklearn/runner only).
2. Partition `params` into `used` (in `accepts`) and `ignored` (not). Ignored: one
   `RuntimeWarning` per `(method, param)` per process (same once-only pattern as
   `similarity._DEPRECATION_ANNOUNCED`), or `TypeError` if `strict`.
3. Apply presets, call `spec.fit`, then attach:
   - `result["method"]`, `result["family"]`
   - `result["parameters"] = {"requested": {...}, "used": {...}, "defaults": {...}, "ignored": {name: reason}, "budget_unit": ..., "preset": {...}}`
4. `validate_result(result)` asserts the common contract (below) and raises on a missing key, so a
   method cannot quietly drop part of the contract (replaces `test_the_shared_contract_has_not_shrunk`
   as a runtime check, test kept).

Common result contract (every method, including baselines): `method`, `family`, `parameters`,
`u` (n x k tensor; per-view list stays under `u_per_modality`), `v` (list of p_m x k tensors or
blocks of a joint basis with `basis_is_joint`), `latents`, `convergence` (the existing eight-field
record), `effective_rank`/`numerical_rank`/`condition_number`/`max_column_overlap` (per view, or
`joint_*` when the basis is joint), `sparseness`, `scale_list`, `provenance_list`.

Baselines move out of the runner into `api.py` as **unsupervised** engines with the same
signature as every other method, `fit(views, k, *, random_state=None, ...)`: `pca_joint`
(concatenated-view PCA, blocked `v`, `basis_is_joint=True`), `pca` (per-view, the
`MultiViewPCA` engine) and `nsa_pipeline` (NSAFlow on the concatenation; `nsa_w`, `consolidate`,
`sparseness`). `y` never enters `api.fit`; the supervised head and `pipeline_test_score` that the
runner's `pca`/`nsa_pipeline` branches computed are dropped in favour of the one downstream
estimator `calculate_all_metrics` already applies to every method (the runner comment at
`runner.py:552-557` says this is the comparable number). The `eval_view_indices` subset becomes a
runner pre-step applied to every method before `api.fit`, not a baseline-only argument.

### Native entry-point changes (horizontal pass over `simlr.py`, `deep.py`, `flows.py`)

- Signatures: `sparseness_quantile` -> `sparseness`; `init_seed` -> `random_state`; `epochs` ->
  `iterations`; deep/flow `**kwargs` catch-alls removed (`tol`, `patience`, `retraction_type`
  become named parameters on lend/ned/flow_v as they already are on nedpp); `_warn_unused_kwargs`
  deleted. `simlr` keeps `**opt_params` but validates keys against
  `SIMLR_OPTIMIZER_DEFAULTS | {"patience"}` up front and raises `TypeError` on anything else
  (fixes A2, A7); `patience` is popped before `create_optimizer`.
- `deep_simr` forwards `optimizer_type` (A5); `_get_optimizer` raises on unknown names (A6).
- `flow_simr` returns the full contract (A12) or, if the owner prefers, is removed; default in
  this plan: return the contract (its convergence report already exists).
- `random_state` no longer flips `initialization_type` (A8); when the initializer is
  deterministic the parameter record says `used` with `effect: "none (deterministic initialization)"`.

Blast radius: 413 occurrences of `init_seed|sparseness_quantile|epochs=` across tests, scripts,
docs qmd and paper qmd. Migration is mechanical (sed with word boundaries, then the suite);
occurrences inside `_train_loop` internals (`epochs` as a local) are untouched.
Callers that pass keys through the deleted `**kwargs` and must be edited by hand (they would
otherwise raise `TypeError`):
- `sparsity=`: `tests/test_list_sparsity.py:32`, `paper/build_paper_experiments.py:94, 264`.
- `iterations=` and `epochs=` together: `tests/test_e2e_opaque_box.py:234, 253, 297, 303`, `tests/test_sklearn_estimators.py:159, 172`.
- Shared parameter dicts splatted into both `simlr` and a deep function:
  `scripts/benchmark_linear_noise_simlr_vs_lend.py:44`, `scripts/benchmark_nonlinear_noise_simlr_vs_ned.py:37`,
  `scripts/benchmark_nonlinear_private_noise_simlr_vs_nedpp.py:38,41`, `scripts/analyze_structural_significance.py:11,28`,
  `docs/benchmark_linear_noise_simlr_vs_lend.qmd`, `docs/benchmark_nonlinear_noise_simlr_vs_ned.qmd`,
  `docs/benchmark_nonlinear_private_noise_simlr_vs_nedpp.qmd` (lines 51/57/60/144): route them
  through `api.fit`, which records and warns instead of raising.
- `scripts/unified_real_cv_comparison.py:82-92` imports the runner's `filter_kwargs` (deleted in WP4).
- `flow_simr_v` names `tol`/`patience` but does not forward them to `_train_flow_loop`
  (`flows.py:1213-1217`); WP1 wires them in rather than only renaming.

Tests that change meaning (rewritten, not just renamed):
`tests/test_initialisation.py::test_simlr_init_seed_selects_random_and_leaves_the_global_rng_alone`
(A8 behaviour removed; use `initialization_type="random"` explicitly),
`tests/test_varimax_consistency.py::test_an_unknown_keyword_warns_instead_of_vanishing` and
`tests/test_parameters_are_live.py::test_an_unrecognised_optimizer_hyperparameter_warns`,
`::test_a_removed_parameter_is_named_not_silently_swallowed` (warn -> `TypeError`),
`tests/test_knobs_are_live.py:206-222` and `tests/test_benchmarks_runner.py::test_filter_kwargs`
(replaced by parameter-record tests), `tests/test_benchmarks_runner.py:164` (`custom_pred_test`
key gone), `tests/test_deep_initialization_and_positivity.py:63, 97-105` (`init_seed` ->
`random_state`; add "two unseeded runs differ"), `tests/test_interchangeable_api.py`
(`COMMON_PARAMS` gains `random_state`), all of `tests/test_sklearn_estimators.py` and
`tests/test_e2e_opaque_box.py:262-310`.

### scikit-learn layer (`src/pysimlr/sklearn.py`, rewritten)

```python
class MultiViewSiMR(BaseEstimator, TransformerMixin):
    def __init__(self, method="simlr", n_components=2, iterations=50, energy_type=None,
                 mixing_algorithm="newton", positivity="either", sparseness=0.5, nsa_w=0.1,
                 constraint=None, optimizer_type=None, learning_rate=None, topology=None,
                 varimax=None, initialization_type="pca", scale=True, hidden_dims=None,
                 dropout=0.1, batch_size=64, private_k=None, num_layers=4, hidden_dim=64,
                 device=None, tol=1e-6, verbose=False, random_state=None, strict=False)
```
- `None` means "use the method's own default" so one grid can span methods; `fit` passes only
  non-`None` values plus `scale` mapped to `scale_list`, through `api.fit`.
- Fitted attributes: `result_`, `loadings_` (list, or single matrix for single-view input),
  `components_`, `n_features_in_`, `parameters_used_`, `parameters_ignored_`, `budget_unit_`.
- `transform` always goes through `api.predict` (fixes A3); `inverse_transform` only for
  `family == "linear"` (raises `NotImplementedError` otherwise, stated in the docstring).
- `fit_transform` returns `transform(X)` for every family (removes the `u_[0]` special case).
- Multi-view input through sklearn cross-validation: `cross_val_score`/`GridSearchCV` index `X`
  by rows and reject a list of arrays (confirmed on scikit-learn 1.9.0: "inconsistent numbers of
  samples"). `MultiViewSiMR` therefore accepts either a list of views (direct use) or one
  `(n, sum p_m)` array with `view_sizes: list[int] | None` giving the column boundaries (CV use);
  `None` means single view. `MultiViewPCA` gets the same parameter.
- `GridSearchCV` needs a `score`. `score(X, y=None)` returns `1 - mean(normalised reconstruction
  error)` from `api.predict(...)["errors"]`, i.e. held-out reconstruction R² averaged over views;
  for `family == "flow"` the same quantity through the flow decoder. `TODO(human)`: confirm this
  is the model-selection criterion wanted; `scoring=` can always override it.
- `MultiViewPCA` stays. `build_simlr_pipeline` stays, built on `MultiViewSiMR`.
  `SiMLREstimator`, `SiMLRTransformer`, `SiMLR`, `LENDTransformer`, `NEDTransformer`,
  `FlowSiMLRTransformer` are deleted; `__init__._LAZY_SKLEARN` and `__all__` updated; `grep` for stragglers
  (`tests/test_e2e_opaque_box.py`, `tests/test_sklearn_estimators.py`, `scripts/generate_nsa_report.py`).
- Loop over methods: `GridSearchCV(MultiViewSiMR(), {"method": method_names()})` works because
  every parameter is explicit and `clone`-able.

### Consumers migrated to the registry

- `benchmarks/runner.py`: `run_single_experiment` body becomes `res = api.fit(model_type, train_mats, k, **params)`;
  `pred = api.predict(res, test_mats)`; `filter_kwargs` deleted (the parameter record replaces it);
  per-view "dimension-aware" sparseness list is kept as a pre-step. Post-dispatch normalisation
  (Procrustes alignment for `eval_view_indices`, joint-rank logic, invariants, `convergence_report`)
  stays but loses its `except Exception` sites (B3). `parameters.ignored` is written into the metrics
  row as `ignored_params` so a table can show which conditions were not distinct.
- `nnh.py`: both hand-written dispatches replaced by `api.fit(method, ...)`; `constraint` forwarded (A11).
- `scripts/run_deep_benchmark.py`: label tables replaced by `api.method_names()` plus a label map;
  the `iterations`/`epochs` branch deleted.
- `paper/build_paper_experiments.py:234` fixed to `k=` / `private_k=` (A10).
- `tests/test_interchangeable_api.py`: `ALGORITHMS` becomes `api.method_names(family in {"linear","deep","flow"})`
  and `run` calls `api.fit`.

### Verified-bug fixes outside the interface (same pass)

B1 (`catch_warnings` context), B2 (explicit `is None` chains, three sites), B3, B4 (raise with the
failing statistic named instead of substituting constants), B8 (delete the first definition).
B6/B7 handled where the file is already being edited (entry points, sklearn, runner); the rest
listed in tracking.

### Guard rails

- `tests/test_implementation_standards.py` (B9): grep over `src/pysimlr/**/*.py` with an
  `ALLOWED_EXCEPTIONS: dict[(relpath, line_substring)] -> reason` for: `except Exception`
  (allowlist = the 42 current sites minus the ones fixed here, each with the D3 class as reason),
  `torch.manual_seed(`/`np.random.seed(` outside `utils.set_all_seeds`, `warnings.filterwarnings(`
  outside a `catch_warnings` block, mutable defaults in `def` signatures (allowlist current ones).
- `tests/test_api_registry.py`: every registered method fits/predicts on the planted fixture
  through `api.fit`; `accepts` equals the signature-derived set; a parameter outside `accepts`
  lands in `parameters.ignored` and warns exactly once; `strict=True` raises; unknown method raises
  with the name list; `random_state` reproduces bit-identical `u` for every method and does not
  change `initialization_type`; every canonical parameter in `accepts` is live (generalises
  `test_parameters_are_live.test_every_parameter_is_classified` from `simlr` only to all methods via
  the registry); `validate_result` rejects a dict missing a contract key.
- README quick start rewritten to the new API and executed by a new `tests/test_readme_examples.py`
  (extracts the fenced python blocks under "Quick Start" and runs them with small sizes).

## Work packages (each one commit, in this order; horizontal: finish a WP on all targets before the next)

| WP | Scope | Files |
|---|---|---|
| 0 | Copy this plan into the repo as `docs/plans/2026-11-01_central_interface.md` (the "accessible location"), and add a "next WP" pointer to the `docs/tracking.md` header | `docs/plans/`, `docs/tracking.md` |
| 1 | Native unification + A5/A6/A7/A8/A12 + B1/B2/B8 | `simlr.py`, `deep.py`, `flows.py`, `consensus.py`, `sparsification.py`, `optimizers.py`, `visualization.py`, mechanical rename across `tests/`, `scripts/`, `docs/*.qmd`, `paper/*.qmd` |
| 2 | `api.py` registry, `fit`/`predict`/`validate_result`, baselines moved in, `__init__` exports, `tests/test_api_registry.py` | `src/pysimlr/api.py`, `__init__.py` |
| 3 | sklearn rewrite (`MultiViewSiMR`), delete old classes, `tests/test_sklearn_estimators.py` rewritten, A3/A4/B7 | `sklearn.py`, `__init__.py`, `tests/test_e2e_opaque_box.py`, `scripts/generate_nsa_report.py` |
| 4 | Consumers on the registry: runner (delete `filter_kwargs`, B3), nnh (A11), `run_deep_benchmark.py`, paper script (A10), `test_interchangeable_api.py`, `test_knobs_are_live.py` (its `filter_kwargs` test becomes a parameter-record test), B4 | `benchmarks/runner.py`, `nnh.py`, `scripts/run_deep_benchmark.py`, `paper/build_paper_experiments.py`, `benchmarks/deep_ranking.py` |
| 5 | Guard rails + docs: `tests/test_implementation_standards.py`, README quick start + `tests/test_readme_examples.py`, `docs/api_contract.md` (canonical vocabulary, result contract, parameter record), `docs/tracking.md` entry D3 with follow-ups (remaining except sites by class, Module-level `sparseness_quantile`, duplicated MAI/schedule helpers, default alignment TODO(human), audit A1-A4), version bump to 0.3.0 in `pyproject.toml`/`__init__.py` (breaking API) | as named |

No commits, tags or pushes without instruction; each WP is left staged-ready and import-checked.

## Verification

1. After WP1: `pytest` fast suite green; `grep -rn "sparseness_quantile\|init_seed\|epochs=\|sparsity=\|varimx\|filter_kwargs" tests scripts docs/*.qmd paper/*.qmd src/pysimlr` returns only `nn.Module` internals and training-loop locals; two `lend_simr(random_state=0)` calls give bit-identical `u` on a fixture with a near-constant column, two `random_state=None` calls differ, and the global torch RNG state is unchanged after either.
2. After WP2: `python -c "import pysimlr; print(pysimlr.method_names())"` lists 8 names; the benchmark loop below runs with no per-method branch and every result passes `validate_result`:
   ```python
   for m in pysimlr.method_names():
       r = pysimlr.fit(m, views, k=2, iterations=5, random_state=0, energy_type="acc", nsa_w=0.5)
       p = pysimlr.predict(r, views)
       print(m, r["parameters"]["ignored"], p["u"].shape)
   ```
3. After WP3: `sklearn.utils.estimator_checks` is not applicable to list input; instead run `GridSearchCV(MultiViewSiMR(), {"method": method_names(["linear","deep","flow"])}, cv=2)` on the planted fixture, and assert `transform(X)` equals `predict(result_, [X])["u"]` to 1e-6 for single-view input (the A3 regression test).
4. After WP4: `python -m pysimlr.benchmarks.runner --config configs/benchmark_smoke.yaml` completes and its metrics rows carry `ignored_params`; `python scripts/run_deep_benchmark.py --smoke` (or its smallest documented invocation) runs.
5. After WP5: `pytest -m ""` (full suite including slow) green or with only the 6 pre-existing xfails; `python -m antsxqc.lint src/pysimlr` 0 findings; README blocks execute.
6. Report: six-part report (files, API, test counts, measured numbers, out-of-ownership changes, not done), tracking entry drafted with real hashes only once the owner commits.


## Implementation notes (2026-11-01)

Done as planned, with these deviations (reasons in docs/tracking.md D3):
- `pca` in the registry is the joint (concatenated-view) PCA, as in the benchmark runner;
  per-view PCA is not registered (no shared k-dim latent). `flow_simr` is not registered (no `v`).
- A keyword no registered method accepts raises `TypeError` even when `strict=False`.
- Joint baselines still fit on `eval_view_indices` only; the multi-view methods see all views,
  as before. The plan's "subset every method" would have changed their inputs.
- `runner.run_single_experiment(budget_by_unit=...)` added so one sweep can give sweeps and
  epochs different budgets.
- Optimizer `beta` renamed `rmsprop_beta` (collided with the flow NLL weight `beta`).
- B1 (consensus warning filter) was not a defect.
