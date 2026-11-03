# Benchmark Readiness Statement

## 1. Budget Policy and Stop Rule

- **Budget Policy**:
  - Linear `simlr` defaults to 300 sweeps (`DEFAULT_SIMLR_ITERATIONS = 300`), providing headroom above typical convergence at ~73 sweeps.
  - Deep architectures (`lend`, `ned`, `flow_v`, `ned_shared_private`) default to 300 epochs (`DEFAULT_DEEP_ITERATIONS = 300`), unifying the step budget across paradigms.
  - Across all benchmark harnesses (`scripts/default_selection_benchmark.py`, `scripts/run_deep_benchmark.py`, `scripts/unified_benchmark.py`), `--iterations` and `--epochs` default to `None`, omitting arguments to inherit package defaults.
- **Stop Rule**:
  - Plateau monitoring via `ConvergenceMonitor` with relative energy tolerance `DEFAULT_TOL = 1e-4` and patience = 10, stopping converged fits early.

---

## 2. Package Defaults and Tracked Evidence Paths

| Parameter | Default Value | Evidence Path |
|---|---|---|
| Similarity Energy | "nc" | `docs/audit/evidence/default_selection_stage1.csv`, `docs/audit/evidence/default_selection_stage2.csv` |
| Linear Optimizer | "hybrid_adam" | `docs/audit/evidence/default_selection_stage2.csv`, `docs/audit/evidence/optimizer_benchmark_synthetic.csv` |
| Mixing Algorithm | "newton" | `docs/audit/evidence/default_selection_stage2.csv`, `docs/audit/evidence/default_followup_checks.csv` |
| NSA-Flow Weight | 0.5 | `docs/audit/evidence/constraint_selection.csv` |
| Consensus Topology | "loo" | `docs/audit/evidence/iteration_vs_init_loo.json`, `docs/audit/evidence/iteration_vs_init_star.json` |
| Collapse Weight | 1.0 | `docs/audit/evidence/collapse_candidates.csv`, `docs/audit/evidence/collapse_diagnosis.csv` |
| Collapse Threshold | 0.3 | `docs/audit/evidence/collapse_threshold_selection.csv` |
| Convergence Tolerance | 1e-4 | `docs/audit/evidence/linear_tol_selection.csv`, `docs/audit/evidence/deep_tol_selection.csv` |
| Deep Epoch Budget | 300 | `docs/audit/evidence/deep_budget_check.csv` |

---

## 3. Section 3 Measurement Definitions

- **Default Budget**: Calling fits without `iterations`/`epochs`/`tol` or explicitly passing package defaults. Fits record `converged`, `n_steps`, and `stop_reason`.
- **Paired Difference**: Alternative minus baseline over the same cells, oriented so positive is better.
- **Worse Beyond Noise**: Either holds:
  - (mean) n >= 2 and mean oriented difference < -SD of differences (ddof = 1).
  - (guard) any single cell is worse by more than its metric guard:
    - latent R2, outcome R2, held-out metric: 0.05
    - support F1 or AUROC: 0.10
    - instability / sin2: 0.05
    - component stability: 0.10
    - convergence rate: 0.10
- **Better Beyond Noise**: n >= 2 and mean oriented difference > +SD of differences (ddof = 1).
- **Non-Inferior**: Not worse beyond noise on every listed metric.
- **Loadings Support**: Support F1 of loadings on positive fits; `loadings_support_auc` (mean AUROC of |loadings| against true support under Hungarian matching) on signed fits.

---

## 4. Rule Outcomes (R11, R12, R13, R15)

- **Rule R11 (Probe Grid Extension)**: Evaluated on 1664 Stage-1 fits. 25 (optimizer, energy) pairs exhibited >= 25% boundary fits on the base grid (1.0 or 0.001). Upward extension was evaluated and rejected: the 3-sweep probe biases toward large steps on short horizons (at buried/positive/seed 0, lr=3.0 dropped R2 to 0.846 vs >= 0.974 for alternatives). Package `LR_PROBE_GRID` stays strictly `(1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)`.
- **Rule R12 (Linear Defaults Confirmation)**: Evaluated on 180 Stage-2 fits across top 8 valid candidates + incumbent across 5 seeds. Incumbent `("nc", "hybrid_adam", "newton")` converged at 1.0 (>= 0.9), mean latent R2 = 0.9883, worst cell = 0.9808, loadings support = 0.7740. No candidate was better beyond noise (all paired diff means <= -0.0006), and no candidate exceeded incumbent by > 0.05 on any cell (max cell diff 0.0003 < 0.05). Incumbent defaults stay.
- **Rule R13 (Deep Consistency Check)**: Evaluated across deep families (`flow_v`, `ned`, `lend`, `ned_shared_private`). At 150 epochs, 20 warmup epochs left 130 monitored steps and `lend` (16/18 = 0.889) and `ned_shared_private` (11/18 = 0.611) were still improving rather than stuck on a plateau; raising `DEFAULT_DEEP_ITERATIONS` to 300 gives `NEDPP` 18/18 (1.000), `LEND` 17/18 (0.944 >= 0.90), `NED` 18/18 (1.000), and `Flow-V` 18/18 (1.000), with metrics non-inferior in `docs/audit/evidence/deep_budget_check.csv`. Rule R13 passes across all deep families.
- **Rule R15 (Collapse Threshold Selection)**: Evaluated across 150 fits in `docs/audit/evidence/collapse_threshold_selection.csv`. Score correlation tracks t (0.302 at 0.3, 0.509 at 0.5, 0.692 at 0.7). Narrow fixture R2 degrades from 0.958 at t=0.3 to 0.856 at t=0.7 (diff -0.102 fails 0.05 guard). Default t=0.3 is retained.

---

## 5. Caveats and Limitations

1. **D3 Open**: Subspace R2 drop is 0.0207 > 0.02 at rho=0.7; `avg` score correlation is capped near collapse threshold (0.3), while `newton` decorrelates (`docs/audit/evidence/correlated_factors.csv`).
2. **Collapse Threshold (t=0.3)**: Trade-off between protecting narrow views and retaining score correlation between correlated factors under `avg`.
3. **NED Rounding Gap (C5)**: Parameter divergence across 100 epochs is continuous float32 gradient amplification with Lyapunov exponent ~0.10/epoch, not initialization gap: d(0) is 1.76e-7 to 3.16e-7 < 1e-6 (`docs/audit/evidence/ned_rounding.csv`).
4. **acc Init-Relative View Weights**: Init-relative scaling weights in non-scale-free energy `acc` depend on initial empirical energy rather than equal 1/M partition (`docs/audit/evidence/view_weighting_acc.csv`).
5. **Narrow Fixture LR Endpoints**: Step-size probe biases toward upper grid endpoint 1.0 on short horizons.
6. **Pre-18db280 Standardized Energy Results Invalid**: `similarity._std` relative floor fix (`eps * mean(var)` in commit `18db280`, `docs/audit/evidence/std_floor_check.csv`). Linear results for `align`, `logcosh`, `exp`, `kurtosis`, and `negentropy` measured before commit `18db280` are invalid.
7. **LEND Convergence at 300 Epochs**: At `DEFAULT_DEEP_ITERATIONS = 300`, `lend` achieves 17/18 (0.944) convergence, passing the >= 0.90 threshold with 1 max_iter stop on synthetic regimes.
8. **Budget Units Differ**: Linear units are coordinate descent sweeps (300) while deep/flow units are mini-batch training epochs (300).
9. **Deep collapse_weight Parameter**: Deep and flow architectures (`lend`, `ned`, `ned_shared_private`, `flow_simr`, `flow_simr_v`) evaluate `correlation_penalty(..., threshold=collapse_threshold)` at fixed weight `DEFAULT_COLLAPSE_WEIGHT = 1.0` (as verified in WP-5 / R3); they expose `collapse_threshold` but do not expose a user-configurable `collapse_weight` parameter.

---

## 6. Rule R14 Verdict

| Criterion | Requirement | Result | Status |
|---|---|---|---|
| R11 Default Pair | No boundary exclusion for default pair | Withdrawn by reviewer (probe bias, not grid defect) | **WITHDRAWN** |
| R12 Winner Convergence | Winner convergence rate >= 0.9 | Incumbent convergence = 1.000 | **PASS** |
| R13 Deep Convergence | Each deep family convergence >= 0.9 | `flow_v` = 1.000 (18/18), `ned` = 1.000 (18/18), `ned_shared_private` = 1.000 (18/18), `lend` = 0.944 (17/18) | **PASS** |
| Test Suite Gate | 100% green in clean worktree | 1663 passed, 3 xfailed | **PASS** |
| Evidence Provenance | All cited files tracked in git | Verified via `git ls-files --error-unmatch` | **PASS** |

### **Overall Verdict: READY**
- **LINEAR: READY** (R12 incumbent defaults `nc / hybrid_adam / newton` confirmed, 100% convergence rate [1.000], clean test suite, all evidence paths tracked).
- **DEEP: READY** (R13 deep convergence >= 0.90 across all 4 deep families at unified `DEFAULT_DEEP_ITERATIONS = 300`, predictive accuracy and latent recovery non-inferior in `docs/audit/evidence/deep_budget_check.csv`).
