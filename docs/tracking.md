# Tracking Summary
current phase: v0.4.0 (leave-one-out default, audit items closed, duplicate code removed); v0.3.0 released -- one method interface (`pysimlr.fit`/`predict`, `MultiViewSiMR`), one vocabulary and one defaults module, no broad exception handlers, batch-invariant prediction, standardized-space reconstruction R^2; nsa_flow>=3.5.1. Manuscript correction (D5) in progress, not part of the release.
next WP: D5/D6 paper -- caches regenerating under v0.4.0 (DEFAULT_TOPOLOGY="loo"); then rebuild the paper and list the conclusions that moved for owner review. Paper sources and caches are uncommitted.
blockers: none
active work: D5 (uncommitted: paper/03_experiments.qmd, paper/appendix_operational.qmd, paper/paper_numbers.py, paper/results_cache/*, paper/_freeze, docs/manuscript; the owner's own pre-existing manuscript edits are in the same files, snapshot in the session scratchpad)
device: Apple Silicon (macOS 27 arm64), CPU only; venv /Users/stnava/venvs/ants; torch 2.13.0, numpy 2.0.1, scikit-learn 1.9.0, nsa_flow 3.5.1 (333a064, editable, /Users/stnava/data/repos/nsa_flow)
last updated: 2026-11-01

---

Entries are newest first. `D<n>` are implementer deliverables; `R<n>` is reserved for reviewer notes.
This file was created at v0.2.19; earlier history lives in git (`git log --oneline`) and in the
release commit bodies.

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
- fig-nsa-v-interpretability is built from code: `scripts/paper_nsa_v_loadings.py` ->
  results_cache/nsa_v_loadings.json (nsa_flow 3.5 dropped `varimax=True`, so the old script no
  longer ran; the rotation is `pysimlr.svd.varimax`).
- `scripts/run_adni_mci_grid_sweep.py`: no absolute paths; writes its provenance; exits non-zero
  if any configuration fails (the 224 NameError rows of f4e53a6 went unnoticed).
verification: fast suite 1598 passed / 2 xfailed; `pytest -m ""` 1622 passed / 3 xfailed / 0
failed (200.7 s, 6 workers); `python -m antsxqc.lint src/pysimlr` 0 findings.
open:
- Both L-BFGS presets under loo reach different stationary points for inputs differing by float32
  rounding (per-view R^2 up to 0.11 apart); the shift-invariance test pins them to star.
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
