# Benchmark Readiness Statement

## 1. Budget Policy and Stop Rule

- **Budget Policy**:
  - Linear `simlr` defaults to 300 sweeps (`DEFAULT_SIMLR_ITERATIONS = 300`), providing headroom above typical convergence at ~73 sweeps.
  - Deep architectures (`lend`, `ned`, `flow_v`, `ned_shared_private`) default to 150 epochs (`DEFAULT_DEEP_ITERATIONS = 150`).
  - Across all benchmark harnesses (`scripts/default_selection_benchmark.py`, `scripts/run_deep_benchmark.py`, `scripts/unified_benchmark.py`), `--iterations` defaults to `None`, omitting the argument to execute at package defaults.
- **Stop Rule**:
  - Plateau monitoring via `ConvergenceMonitor` with relative energy tolerance `DEFAULT_TOL = 1e-4` and patience = 10.

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

---

## 3. Section 3 Measurement Definitions

- **Default Budget**: Calling fits without `iterations`/`tol` or explicitly passing package defaults. Every run records `converged`, `n_steps`, and `stop_reason`.
- **Paired Difference**: Alternative minus baseline over the same cells, oriented so + = better.
- **Worse Beyond Noise**: Evaluated if either holds:
  - (mean) n >= 2 and the mean oriented difference < -SD of differences (ddof = 1).
  - (guard) any one cell is worse by more than its metric guard:
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

- **Rule R11 (Probe Grid Extension)**: Evaluated on 1664 Stage-1 fits across 4 energies, 13 optimizers, 4 mixings, 2 positivities, 2 generators, 2 seeds. 25 (optimizer, energy) pairs exhibited >= 25% boundary fits on the base grid (1.0 or 0.001). Upward extension was evaluated and rejected: the 3-sweep probe biases toward large steps on short horizons (at buried/positive/seed 0, lr=3.0 dropped R2 to 0.846 vs >= 0.974 for alternatives). Package `LR_PROBE_GRID` stays strictly `(1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)`.
- **Rule R12 (Linear Defaults Confirmation)**: Evaluated on 180 Stage-2 fits across top 8 valid candidates + incumbent across 5 seeds. Incumbent `("nc", "hybrid_adam", "newton")` converged at 1.0 (>= 0.9), mean latent R2 = 0.9883, worst cell = 0.9808, loadings support = 0.7740. No candidate was better beyond noise (all paired diff means <= -0.0006), and no candidate exceeded incumbent by > 0.05 on any cell (max cell diff 0.0003 < 0.05). Incumbent defaults stay.
- **Rule R13 (Deep Consistency Check)**: Evaluated across 72 synthetic fits covering deep families (`flow_v`, `ned`, `lend`, `ned_shared_private`). At the harness's 45 epochs, all 4 families had convergence rates < 0.90. Re-running at package default 150 epochs produced: `flow_v` 18/18 (1.000), `ned` 17/18 (0.944), `lend` 16/18 (0.889), and `ned_shared_private` 11/18 (0.611). Fails on `lend` and `ned_shared_private`.
- **Rule R15 (Collapse Threshold Selection)**: Evaluated across 150 fits in `collapse_threshold_selection.csv`. Score correlation tracks t (0.302 at 0.3, 0.509 at 0.5, 0.692 at 0.7). Narrow fixture R2 degrades from 0.958 at t=0.3 to 0.856 at t=0.7 (diff -0.102 fails 0.05 guard). Default t=0.3 is retained.

---

## 5. Caveats and Limitations

1. **D3 Open**: Subspace R2 drop is 0.0207 > 0.02 at rho=0.7; `avg` score correlation is capped near collapse threshold (0.3), while `newton` decorrelates.
2. **Collapse Threshold (t=0.3)**: Trade-off between protecting narrow views and retaining score correlation between correlated factors under `avg`.
3. **NED Rounding Gap (C5)**: Parameter divergence across 100 epochs is continuous float32 gradient amplification with Lyapunov exponent ~0.10/epoch, not initialization gap: d(0) is 1.76e-7 to 3.16e-7 < 1e-6.
4. **`acc` Weighting**: Init-relative scaling weights in non-scale-free energy `acc` depend on initial empirical energy rather than equal 1/M partition.
5. **Narrow Fixture LR Endpoints**: Step-size probe biases toward upper grid endpoint 1.0 on short horizons.

---

## 6. Rule R14 Verdict

| Criterion | Requirement | Result | Status |
|---|---|---|---|
| R11 Default Pair | No boundary exclusion for default pair | `(hybrid_adam, nc)` has 37.5% fits at 1.0 boundary | **FAIL** |
| R12 Winner Convergence | Winner convergence rate >= 0.9 | Incumbent convergence = 1.000 | **PASS** |
| R13 Deep Convergence | Each deep family convergence >= 0.9 | `lend` = 0.889, `ned_shared_private` = 0.611 | **FAIL** |
| Test Suite Gate | 100% green in clean worktree | 1656 passed, 3 xfailed | **PASS** |
| Evidence Provenance | All cited files tracked in git | Verified via `git ls-files --error-unmatch` | **PASS** |

### **Verdict**: **NOT READY**

**Failing Items**:
1. Rule R11: Incumbent pair `(hybrid_adam, nc)` exhibited 37.5% endpoint fits at 1.0 in Stage 1 (>= 25% boundary exclusion trigger).
2. Rule R13: Deep family convergence at 150 epochs fell below 0.90 for `lend` (0.889) and `ned_shared_private` (0.611).
