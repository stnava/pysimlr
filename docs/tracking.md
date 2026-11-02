# Tracking Summary
current phase: v0.4.0 plus D7 (e254f05), D8/D9 (33e05f8 on `main`), and D10 on branch `perf/fast-test-suite` (library hot-path and test-suite speedup: default pytest 136.40s -> 19.75/19.77/19.91s min/median/max; full suite 199.86s -> 32.80s). Paper caches and _freeze cleared.
next WP: merge `perf/fast-test-suite` to `main` (on instruction); regenerate all paper caches; rebuild; list the conclusions that moved for owner review.
blockers: none
active work: D10 committed on `perf/fast-test-suite`; D5 paper sources (uncommitted; the owner's own manuscript edits are in the same files)
device: Apple Silicon (macOS 27 arm64), CPU only; venv /Users/stnava/venvs/ants; torch 2.13.0, numpy 2.0.1, scikit-learn 1.9.0, nsa_flow 3.5.1 (333a064, editable, /Users/stnava/data/repos/nsa_flow)
last updated: 2026-11-02

---

Entries are newest first. `D<n>` are implementer deliverables; `R<n>` is reserved for reviewer notes.
This file was created at v0.2.19; earlier history lives in git (`git log --oneline`) and in the
release commit bodies.

## D10 (Library hot-path and test-suite speedup on `perf/fast-test-suite`, 2026-11-02)
status: committed on branch `perf/fast-test-suite` (off `main` `33e05f8`)
- Root causes fixed in library (`src/pysimlr/`):
  - `flows.py`: replaced eager `import antstorch` (which pulled ITK/ANTsPy/matplotlib at `~2.58s` per
    worker) with direct submodule loader `_load_antstorch_real_nvp_builder()` (`~0.008s`); bound
    `_ShuffledTensorBatches` to target device; added per-step `_step_v_cache`; hoisted `_all_params` and
    replaced 80-tensor per-batch `isfinite()` scans with `clip_grad_norm_(..., foreach=True)` `total_norm`
    check; passed `compute_diagnostics=False` inside `_train_flow_loop`; replaced `copy.deepcopy(state_dict)`
    with tensor `.detach().clone()`.
  - `deep.py`: replaced `DataLoader(TensorDataset, ...)` per-row Python collation with
    `_ShuffledTensorBatches` (`index_select`, bit-identical `torch.randperm` RNG sequence); added per-step
    `_step_v_cache` on `LENDNSAEncoder` and early `alpha <= 0.0` guard before `self.v`; enabled
    `optim.Adam(..., foreach=True)`; added backward-compatible `compute_diagnostics: bool = True` to
    `calculate_sim_loss` (`False` inside `_train_loop`); hoisted `all_params`, `clip_grad_norm_(..., foreach=True)`,
    and clone-based best-state snapshots; wrapped `first_layer_diagnostics` in `torch.no_grad()`.
  - `simlr.py`: shared standardized views and `initialize_simlr` basis across `learning_rate="auto"` probes
    and final fit via `_INTERNAL_PROBE_TOKEN`; skipped post-fit SVD/pinv/interpretability/`basis_rank_report`
    and per-sweep `stationarity_residual` on discarded probes (`_probe_only=True`); removed dead inner-loop
    `_capture_consensus_anchor`; hoisted float64 view/domain tensors and added 1-slot `_last_proj` reuse in
    `_view_oracles`.
  - `consensus.py`, `utils.py`, `sparsification.py`, `benchmarks/__init__.py`: skipped unused `pinv(mean_u)`
    `local_anchor` on LOO subsets (`not need_anchor`) and fast-pathed `len(proj_list) == 1`; cached
    `_EYE_OVER_K_CACHE` (`orthogonality_defect`) and `_THREE_EYE_CACHE` (`newton_schulz_orthogonalize`) and
    skipped provably impossible `step_idx < 4` norm syncs for `scaled_unit, k >= 2, tol <= 1e-4`; vectorized
    `_apply_quantile_sparseness` across all `k` columns and searched MRU-first in `_NSA_RETRACT_CACHE` /
    `_SMOOTHING_CACHE`; converted `pysimlr.benchmarks` to PEP 562 lazy exports.
- Test-runner and suite hygiene (`pyproject.toml`, `pytest.ini`, `tests/conftest.py`, `tests/`):
  - Pinned `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MPLBACKEND=Agg`, and `torch.set_num_threads(1)` per
    worker; configured `-n 10 --dist loadgroup --no-loadscope-reorder -m "not slow"` and disabled unused
    autoloaded plugins (`playwright`, `jaxtyping`, `asyncio`, `anyio`, `base_url`).
  - Implemented `@pytest.hookimpl(tryfirst=True)` 3-tier LPT scheduling + 25-item xdist batching for `<10ms`
    unit tests; deferred heavy `sklearn`/`matplotlib`/`seaborn` imports from 10-worker collection to
    `scope="module"` fixtures; deduplicated redundant identical setup fits via module `lru_cache` without
    loosening any threshold, sample count, iteration count, or `strict=True` xfail.
- Measured results (`n = 3` isolated sequential repeats on Apple Silicon arm64, report:
  `docs/test_suite_speedup_report.html`):
  - Default suite (`pytest`, `1566 passed, 2 xfailed, 0 failed`): `136.40s` baseline (`50.60s` naive xdist)
    -> **`19.75s / 19.77s / 19.91s`** min / median / max (`6.90x` speedup, strictly `< 20.0s` on all 3 runs).
  - Full suite (`pytest -m ""`, `1588 passed, 4 xfailed, 0 failed`): `199.86s` -> **`32.80s`** (`6.09x` speedup).
verification: pytest (n=3) 1566 passed, 2 xfailed, 0 failed in 19.75s/19.77s/19.91s; pytest -m "" 1588 passed, 4 xfailed, 0 failed in 32.80s; test_implementation_standards.py 7 passed; antsxqc.lint 0 findings.

## D9 (NSALBFGSB rounding-instability root cause and proximal-quartic fix, 2026-11-02)
status: done in the working tree, uncommitted (on top of D8; paper unchanged per owner decision)
- Root cause (verified vs refuted): instrumented the exact 3-view reproduction (`test_api_registry._views(n=60)`,
  `k=3`, `iterations=20`, `1e-8` float32 input perturbations, 3 seeds) and the +20 constant-shift test.
  Candidates refuted as primary cause: outer quantile projection (`sparseness=None` still gave gaps
  `5.82e-02` positive / `9.94e-01` either; `sparseness=None, constraint="none"` gave `2.26e-02` /
  `3.24e-01`; support XOR was 0 on the sweeps where divergence grew by `1e4x`), dead-column step
  rejection (`last_rejected` fired 0 times across all 57-63 step calls), inner-solve exception handler
  (0 exceptions/warnings), C++ `_native` extension (`_native=None` gave identical `5.82e-02` /
  `8.31e-01`), and missing-callable fallback (0 runtime triggers). Verified cause: `_smooth_objective(V)`
  with consensus `u` fixed is indefinite around the incoming iterate `V_0` (finite-difference Hessian
  eigenvalues: 7/16 to 18/30 negative on `_views`, down to `-9.22`; 30/60 negative on `buried`, down to
  `-32.44`), combining the exact 0-homogeneous radial-tangential saddle `+-||grad_{v_j} E||_2 / ||v_j||_2`
  of column normalisation with `4 W_orth <V_0^T V_0 - I, D^T D>` and `correlation_penalty` non-convexity.
  Even when zero discrete line-search or active-set branches differed (`alpha=(1.0, 1.0)`, `fix_xor=0`),
  running 20 unanchored quasi-Newton steps collapsed L-BFGS scaling `theta = y'y / s'y` from `32.1` to
  `0.00344`, drifted column norms `1.0 -> 7.42`, and amplified a `7.69e-07` difference to `2.13`
  (`2.77e6x`) in a single inner solve. Even smooth `logcosh` diverged to `0.653` without an anchor.
- Fix (`src/pysimlr/optimizers.py`, `NSALBFGSB.step`): replaced the unanchored inner objective with the
  column-scale-invariant proximal-quartic alternating-minimisation subproblem
  `Phi(V; V_0) = E(V) + (mu_2 / 2)||V - V_0||_F^2 + (mu_4 / 4)||V - V_0||_F^4`, where
  `mu_2 = 2.5 / s_0^2`, `mu_4 = 64.0 / (s_0^2 ||V_0||_F^2)`, and `s_0^2 = ||V_0||_F^2 / k`. Both value
  and gradient vanish at `V = V_0`, so fixed-point stationarity, the outer energy, and package defaults
  are unchanged (`hybrid_adam` bit-identical: `[9.57e-08, 2.60e-07, 6.29e-07]` positive,
  `[3.07e-06, 5.15e-07, 7.53e-07]` either). `mu_2` convexifies local negative curvature and pins the
  column-norm gauge; `mu_4` bounds single-sweep excursions against the frozen consensus near
  `||V - V_0||_F <= 0.25 ||V_0||_F` (purely quadratic `mu_2` alone still allowed `||V - V_0||_F = 1.13`
  on `buried` into `-7.49` curvature). Also replaced the silent plain-step fallback on missing
  energy/gradient callables with `ValueError`.
- Measured results (report: `docs/simlr_lbfgsb_d9_report.html`): `nsa_lbfgsb` perturbation gaps drop
  from `[5.82e-02, 8.49e-02, 9.47e-02]` to `[7.46e-07, 7.46e-07, 3.23e-10]` (`positive`) and from
  `[4.29e-01, 4.22e-01, 8.31e-01]` to `[1.39e-06, 2.83e-06, 1.05e-06]` (`either`). All 7 supported
  energies (`nc`, `recon_r2`, `align`, `procrustes`, `exp`, `logcosh`, `acc`) have max gap `<= 5.42e-06`
  (down from `7.44e-03 .. 7.67e-01`). `+20` constant-shift max `|dR^2|` drops `1.43e-02 -> 6.45e-08`
  (`hybrid_adam` `5.83e-08`). Isolated fit timings (`n=5`, `k=3, iter=20`): `nsa_lbfgsb`
  `818.8 / 825.5 / 831.1 ms` min/median/max (`hybrid_adam` `412.9 / 414.3 / 418.3 ms`).
- Tests: `"simlr_lbfgsb"` removed from `tests/test_api_registry.py::_ROUNDING_UNSTABLE`; added
  `test_lbfgsb_refuses_missing_energy_or_gradient_callable` and
  `test_lbfgsb_is_reproducible_under_float32_input_perturbations[{positive,either}]` in
  `tests/test_solver_certificate.py`. `SiMLR-LBFGSB` remains out of `paper/` per D8 owner decision.
verification: pytest -m "" 1578 passed, 4 xfailed, 0 failed; antsxqc.lint 0 findings.

## D8 (Default objective, collapse penalty, stopping rule, SiMLR-LBFGSB dropped, 2026-11-02)
status: done in the working tree, uncommitted (on top of e254f05)
owner decisions: remove the ANTsTorch whitener; choose energy, optimizer and mixing jointly under
both positivity settings; "nc with newton", collapses to be fixed; DEFAULT_ENERGY_TYPE = "nc"; settle
the stopping tolerance; fix or drop SiMLR-LBFGSB; deal with the oscillating alternation.
- Default selection (scripts/default_selection_benchmark.py; evidence
  docs/audit/evidence/default_selection_grid.csv): 4 energies x 5 optimizers x 4 mixings x
  {positive, either} x {buried-private, linear} generators, 5 seeds, latent R^2. Every top cell is
  "nc": nsa_lbfgsb/newton 0.983, nsa_lbfgsb/avg 0.976, hybrid_adam/avg 0.971, hybrid_adam/newton
  0.969 (worst cell 0.915); best logcosh 0.860, acc 0.750, exp 0.569. Follow-up checks (scripts/default_followup_checks.py;
  docs/audit/evidence/default_followup_checks.csv): nc newton combos 0/5 NED collapses, ADNI held-out
  ~0.16-0.17 against acc ~0.153; nc/hybrid_adam/avg had 1 collapse, so newton is kept. Defaults:
  energy "nc", optimizer hybrid_adam, mixing newton.
- Collapse: newton/svd mixing orthogonalises the consensus u, so a collapse in the per-view scores
  XV was invisible to a u-based penalty, and "nc" is rank-blind. New `utils.correlation_penalty`, a
  hinge on pairwise score correlations, sum over pairs of ((|r|-t)_+/(1-t))^2 with t = 0.3
  (DEFAULT_COLLAPSE_THRESHOLD), weight 1.0 (DEFAULT_COLLAPSE_WEIGHT, `simlr(collapse_weight=)`).
  Applied to each view's scores (linear: in the oracle and in every reported energy) and to the
  latents (deep; it replaces the u-covariance term). Variants measured: mean over pairs diluted at
  k=5 (rank 4); squared |r| cut nonneg_parts support recovery 1.000 -> 0.917; power 4 fixed support but
  raised "unusable projection" at k=p=5; power 8 gave rank 4 and narrow R^2 0.823; hinge t=0.5 gave rank
  4; hinge t=0.3 passed every check (support 1.000). It also reduces NED's rounding sensitivity: a
  1e-6 input perturbation moves R^2 by 0.0036 with it, 0.084 without.
- Oscillation (issue 5): resolved by the two items above. The stress tests in
  test_milestone1_adversarial_stress.py pass and their strict xfails are removed. Consensus damping
  (u <- (1-b)u + b u_new) was measured and refuted: at b = 1, 0.5, 0.25, 0.1 the stress range stayed
  0.004-0.006, the median residual rose 0.54 -> ~0.72 and recovery fell 0.969 -> 0.903. Not added.
- Stopping rule (issue 2): the prox-gradient residual with a fixed step ||V||^2 is not a valid
  certificate for the non-convex quantile projection, so `tol` has one meaning, the relative-
  improvement plateau. A plateau stop reports certificate="energy_plateau", converged=True;
  no_descent and max_iter stop uncertified. The residual is no longer computed: `grad_map`,
  `grad_map_history` and the per-view `grad_map_E` diagnostics are removed (owner instruction), and
  simlr no longer needs nsa_flow's gradient_mapping (nsa_lbfgsb still uses it inside its own solve).
- Scripts use the package defaults: run_deep_benchmark.py, paper_brca_loadings.py, the three
  benchmark_*_simlr_vs_* scripts and real_data_method_comparison.py no longer pass energy "acc" or
  mixing "svd"; experiment_iteration_vs_init.py takes --energy (default the package default; the
  DEFAULT_TOPOLOGY evidence was measured with acc). New `defaults.DEFAULT_MIXING_ALGORITHM = "newton"`
  replaces the literal in the simlr/deep/flow signatures (no behaviour change).
  `consensus.compute_shared_consensus` keeps its own "svd" default: switching it to newton failed 4
  direct-call tests (newton with k=None returns the full width instead of the inferred rank; with
  training=True it returns no anchor; the graph topology mismatches shapes). Package callers always
  pass k and the mixing, so fits are unaffected; the newton branch's gaps are open.
  run_adni_mci_grid_sweep.py keeps its explicit design grid (it crosses energies and mixings on
  purpose), but its MAI arms use only acc/align/procrustes with svd mixing, so they exclude the default.
- SiMLR-LBFGSB (issue 3): not reproducible under "nc" either, with subspace gaps 0.04-0.49 for inputs
  1e-8 apart (hybrid_adam ~1e-8). Dropped from the paper (model lists, bar charts, label maps, prose,
  paper_numbers keys, scripts/paper_adni_ablations.py, run_adni_mci_grid_sweep.py,
  real_data_design_sweep.py, run_deep_benchmark.py). It stays in the library, recorded in
  test_api_registry._ROUNDING_UNSTABLE. The methods text now names hybrid_adam, not L-BFGS-B.
- Whitener removed (`FlowWhitener`, `flow_whiten_matrix`, `_lamnr_flows_function`; tests deleted).
  ANTsTorch apply_lamnr_flows_whitener.py:161 solves with L^T instead of L in the forward map; the
  inverse (line 171) is correct. Not edited (external repo); noted in docs/api_contract.md.
tests: new tests/test_collapse_penalty.py; test_tol_controls_early_stopping rewritten for the plateau
rule; test_degradation_tracks_rank_loss replaced by test_no_rank_loss_at_any_constraint_weight
(effective rank > 2.8 at orthox 0, 0.1, 0.5); the collapse_weight liveness fixture is unconstrained
with a random start (the default fixture's scores never reach |r| = 0.3). New strict xfails:
`ned` in the +20-shift R^2 test (rounding, not shift: a 1e-6 perturbation gives the same 0.0036);
test_ned_advantage_contract (both models on package defaults: SiMLR latent recovery 0.745 against
NED 0.675, NED 0.070 below, the clause allows 0.05; NED held-out R^2 still leads, 0.882 against
0.710). Owner decision pending.
Follow-up (owner review of D8, same day):
- Newton/avg consensus (`consensus.compute_shared_consensus`): averaging mixings now raise a clear
  ValueError for unequal widths or a width other than k (they ignored k and failed in torch.stack),
  and return an exact linear anchor (replay error <= 5e-7), so anchored prediction is batch-invariant.
  Default mixing is DEFAULT_MIXING_ALGORITHM. Real predict paths were already batch-invariant via the
  frozen map (|u_full - u_half| <= 9e-8); trained bases bit-identical before/after for simlr, ned,
  lend, ned_shared_private, flow_v. Tests: tests/test_consensus_newton.py; reduction tests pass
  mixing_algorithm="svd" explicitly.
- Stationarity residual restored as a diagnostic with a valid definition,
  `utils.stationarity_residual`: gradient mapping with P the convex projection onto the iterate's
  current face (support mask, clamp under positivity) plus an off-support ratio. Calibration
  (scripts/stationarity_residual_calibration.py, 32 runs x 300 sweeps; evidence
  docs/audit/evidence/stationarity_residual_calibration.csv): >= 2 decades of decrease in 7 of 8 runs
  whose energy settled (1.74 in the eighth); stays high where the energy keeps moving (lars either).
  `grad_map`, `grad_map_history`, per-view `grad_map_E`/`off_support` are back; the benchmark runner
  uses the same helper for every model (its "acc" energy fallback is now the package default).
- Collapse threshold (owner: "a magic number"): candidates measured
  (scripts/collapse_candidates_benchmark.py; docs/audit/evidence/collapse_candidates.csv).
  C2, scoring "nc" on the polar factor of the scores, failed (narrow-view R^2 0.523, grid min 0.250).
  Linear: without a penalty narrow views collapse (R^2 0.301); the 0.3 hinge fixes it (0.947) but
  lowers buried/either recovery (0.878 vs 0.933). Correlated planted factors (rho 0-0.7): linear
  recovery identical with and without the hinge, and recovered score correlation is ~0.03-0.15 for
  every candidate, so the orthogonal newton consensus, not the hinge, decorrelates the scores.
  Deep: without any penalty NED collapses on 6 of 20 (data seed, start seed) pairs, with the hinge on
  0 of 20. A first measurement with one start per data seed (0/5 either way) wrongly suggested the
  old u-covariance loss was the cause and the deep penalty could go; it was removed and restored the
  same day. OPEN: root-cause the linear narrow-view collapse (owner decision: root cause first);
  the hinge stays until then.
verification: pytest -m "" 1578 passed, 4 xfailed, 0 failed (includes D9); antsxqc.lint 0 findings.

## D7 (Defect fixes, shim removal, housekeeping, 2026-11-01)
status: done, committed e254f05
owner decisions: fix the gd anomaly, the unseeded start, the stationarity certificate and the
late-bound callbacks; remove the "nsa_flow" optimizer; remove compatibility shims and do all
housekeeping "brutally".
fixes (each with a regression test):
- gd anomaly: `recon_r2` (and `regression`, which resolved to it) profiles out the column scale
  -- E(v) = E(3v) = E(v diag(1,10,0.1)) = 0.99147 -- yet was in GAUGE_FREE_ENERGIES, so its column
  norms drifted (1 -> 2-10 in two gd steps) and fits returned bases of arbitrary scale. Now only
  the unprofiled `recon` is gauge-free. The learning-rate test measures displacement from a
  near-zero-rate fit (linear in lr in the stable range: 5.9e-5 at 1e-4, 5.9e-4 at 1e-3).
- Unset random_state: one seed is drawn up front (shared by the learning-rate probes and the
  fit), recorded, and handed to every initializer; it was None, so randomised starts ("gcca") used
  the global stream after an optimizer-dependent number of probe draws. Seeded fits leave the
  global stream untouched; unseeded fits consume exactly one draw (the deep/flow contract). The
  varimax test's spy held the initializer's list, which simlr mutates -- it measured the fit.
- Stationarity certificate: the prox-gradient residual ||P(V - ||V||^2 grad E) - V|| / ||V|| of
  the current iterate under the projection actually applied; it used the step-size residual (a
  tiny line-search step read as stationary), the descent direction as the gradient (an uphill
  probe) and a bare clamp for P. A probe the projection refuses certifies nothing (+inf, noted).
  Fits now mostly stop "plateau, uncertified" (residuals 0.005-0.03 on the test fixture).
- Per-view callbacks: energy, gradient, objective and projection are built by `_view_oracles(i,
  u_i, sparseness)`; defined in the loop they read the loop index and data when called.
removed: optimizer "nsa_flow" (gd plus a second projection, with a silent SVD fallback) and its
`nsa_w` optimizer key; method-name aliases; `deep_simr`, `flow_simlr`, `flow_simlr_v`,
`FlowSiMLR*Model`, `flow_whitener`; energy aliases `regression`/`gauss`/`normalized_correlation`
(raise naming `recon_r2`/`align`, `exp`, `nc`), `recon` as a linear objective; the constraint
string's third field (format `<type>x<weight>`); `retraction_type` (bit-identical fits for any
value); legacy nsa_flow constructor keywords and NSAFlowLayer fallback; `predict_simlr(
allow_legacy_refit)`, prediction from pre-0.3.0 results, `reconstruct_from_learned_maps` returning
[] without `w`; `invariant_orthogonality_defect` (+gradient), `calculate_u`, `_newton_step_ortho`,
`load_nsa_flow_orth`, `_retraction_candidate`, `_backend_has_sign_blind_fidelity`,
`rank_effect_size`, `plot_case_generative_shape`, `procrustes_mse`; regression-result alias keys,
`top_n`, unused `iterations`/`max_iter`/`tol`/`constraint_iterations`; generator aliases
`noise_level`, `nonlinear_shared`, `linear_footprint`. Listed in docs/api_contract.md.
kept on evidence: the deep/flow orthogonality penalty on the raw first-layer weights, documented
as a test-compatibility shim, is a regulariser -- without it NEDPP's shared R^2 fell to 0.384 (< 0.4).
housekeeping: 101 scripts deleted (92 referenced by nothing; 9 stale or superseded), 20 remain;
22 unreferenced paper figures deleted (one pre-rendered result figure remains in use:
paper/figures/heart_waterfall.png); unused imports and locals removed (ruff F401/F841/F541);
retraction/smoothing caches bounded to 64 keys.
verification: pytest -m "" 1575 passed, 6 xfailed, 0 failed; antsxqc.lint 0 findings.

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
- `optimizer_type="torch_lbfgs"` and the `simlr_lbfgs` method removed (owner decision, after
  the v0.4.0 commit a39ac19). Measured (40 iterations, acc, 3 fixtures x 2 topologies): it was the
  weakest optimizer in 5 of 6 runs and on one never left its start while being certified
  "stationary". Cause, from reading: its closure scored the projected basis but received the
  tangent-projected gradient, and its curvature history spanned changing consensus targets.
  `create_optimizer("torch_lbfgs")` raises naming `nsa_lbfgsb`; the NSA transforms default to
  NSA-Flow's own optimizer (nsa_flow deprecated torch_lbfgs); `SiMLR-LBFGS` left the benchmark
  scripts and the paper, whose model count is now computed (`N.n_models_word`).
- fig-nsa-v-interpretability is built from code: `scripts/paper_nsa_v_loadings.py` ->
  results_cache/nsa_v_loadings.json (nsa_flow 3.5 dropped `varimax=True`, so the old script no
  longer ran; the rotation is `pysimlr.svd.varimax`).
- `scripts/run_adni_mci_grid_sweep.py`: no absolute paths; writes its provenance; exits non-zero
  if any configuration fails (the 224 NameError rows of f4e53a6 went unnoticed).
- Optimizer benchmark: `scripts/optimizer_benchmark_synthetic.py` (67 s, 5 seeds, 14 optimizers;
  `nsa_flow` excluded: it is gd plus a second projection). Generator: sparse non-negative shared
  factors buried under private factors at 2x; PCA start latent R^2 0.17, two-view CCA 0.978.
  The "acc" optimum itself stays near 0.48 however long it runs, so recovery measures the
  objective; the primary metric is the optimality gap (E - E_ref)/(E_start - E_ref) at 25
  iterations. Separates optimizers: blocked F 3.15, p 0.0016 (Friedman p 0.017); support F1
  does not (p 0.93; quantile sparsity fixes the support size). Gap: lars 0.10, nsa_lbfgsb 0.22,
  hybrid_adam 0.32, ... armijo_gradient (default) 0.41, gd 0.44, bidirectional_lookahead 0.56.
  Evidence: docs/audit/evidence/optimizer_benchmark_synthetic.csv.
- Default linear optimizer is `hybrid_adam` (`defaults.DEFAULT_LINEAR_OPTIMIZER`; owner decision).
  Grid (docs/audit/evidence/optimizer_grid_synthetic.csv; energies acc/nc/logcosh/recon_r2 x
  mixing newton/svd/avg x positivity, 6 optimizers, 3 seeds): hybrid_adam best mean rank and in the
  top three in 11 of the first 12 settings; armijo_gradient worst. Held-out ADNI-MCI unchanged
  within seed noise across optimizers.
- Default objective NOT changed to "nc" (owner asked; reverted on evidence, decision pending):
  "nc" with hybrid_adam fails the narrow-view linear check (R^2 0.301 vs 0.973 for acc), collapses
  NED latents on 2/5 seeds of the latent-collapse fixture (off-diagonal 1.24/1.34 vs <=0.002), and
  breaks NED budget independence. "logcosh", "exp" and "kurtosis" pass all three checks
  (narrow R^2 0.982 / 0.979 / 0.951, 0/5 collapses, budget-independent); in the grid "logcosh"
  recovers the shared latent better than "acc" in every mixing/positivity cell.
- Open defects recorded as xfail: gd travel is non-monotone in learning_rate (relative distance
  5.22 at lr 0.1, 1.2e-2 at 0.5; unexplained); simlr passes generator=None to initialize_simlr
  when random_state is None, so the start depends on how much global RNG the learning-rate probes
  consumed (optimizer-dependent).
verification: fast suite 1598 passed / 2 xfailed; `pytest -m ""` 1622 passed / 3 xfailed / 0
failed (200.7 s, 6 workers); `python -m antsxqc.lint src/pysimlr` 0 findings.
open:
- Fixed (owner: "fix this bug in any impacted optimizer"): the optimizer oracle. `simlr`
  scored the projected point in float32 while handing a tangent-projected gradient taken there
  (slopes 2-14x off finite differences; energy moving in float32 spacings, 4.657e-10 at
  E = -6.024e-3). Now every optimizer gets two float64 oracles: the objective E(P(v)) for line
  searches, and the smooth part E(v) (column-normalised for gauge energies) with its exact
  autograd gradient for L-BFGS-B, which also runs in float64. Inputs 1e-8 apart, max|V1 - V2|
  after 20 iterations: armijo 3.4e-5, lars 6.0e-7, hybrid_adam 2.3e-6, nsa_flow 6.0e-7 (were up
  to 1.4). Test: test_the_oracle_simlr_hands_its_optimizer_is_consistent (slope = FD to 1e-5).
- Open: `nsa_lbfgsb` stays sensitive on "acc"/"nc" (0.16 after one sweep): L-BFGS-B on an
  objective with absolute-value kinks; its Wolfe line search fails at a kink and the paths
  branch. Smooth energies are far less sensitive (recon_r2 3e-2 after 20). Strict xfail in the
  shift test with this reason; no per-optimizer settings. Not an nsa_flow bug.
- Open: with the exact direction, armijo on the ill-conditioned stress case reaches a better
  objective (best -1.629 vs -1.116 at 40 iterations) but oscillates (-1.29 to -1.63) before
  settling; the two settling stress tests are strict xfail. Damping the alternation is not done.
- ADNI-MCI, scripts/run_deep_benchmark.run_experiment_task, seeds 42-44, before (a39ac19) ->
  after: SiMLR Predictive_Metric 0.1550/0.1513/0.1585 -> 0.1671/0.1556/0.1578 (mean 0.1549 ->
  0.1602); SiMLR-LBFGSB 0.1730/0.1469/0.1556 -> 0.1564/0.1461/0.1648 (mean 0.1585 -> 0.1558);
  LBFGSB fit time 2.2-2.9 s -> 1.0-1.1 s. Three seeds; not tested for significance.
- Fixed: `_NSA_RETRACT_CACHE` / `_SMOOTHING_CACHE` keyed on `id()`, which Python reuses after
  an object is freed; a new backend or smoothing matrix could receive another object's cached
  result (the intermittent test_backend_is_invoked_for_every_soft_ortho_constraint failure).
  Entries now hold the object and are matched by identity (tests/test_cache_identity.py).
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
