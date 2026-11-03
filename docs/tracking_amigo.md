# pysimlr: amigo branch tracking

Branch `amigo`, from `main` at `b10c993` (D10 merged fast-forward; not pushed).
Last updated 2026-11-02.
Verification at the branch tip: `pytest -m ""` 1593 passed, 7 strict xfailed, 0 failed;
`antsxqc.lint src/pysimlr` 0 findings. Each commit below was verified on its own in a clean
worktree.

The earlier history (D1-D10) is in `docs/tracking.md`. This file holds the amigo work and every
item that is in progress or on hold.

## Owner decisions taken on this branch

1. The OLEE contract is orthogonality. Under `positivity="positive"`, orthogonality implies
   disjoint supports and support overlap is part of the contract. For signed fits, overlap is
   descriptive.
2. `simlr` minimises the similarity energy subject to the NSA-Flow projection, with no
   orthogonality-defect penalty. This is the original SiMLR design. `nsa_w = 0.5` is kept,
   following the NSA-Flow paper's calibration of the projection.
3. `simlr` stopping: `tol = 1e-4`, `iterations = 300`.
4. `lobe_crosstalk` is replaced by support overlap.
5. Interpretation uses two-stage loadings computed from `U`, for every method. The encoder `V`
   is the scoring rule.

6. `nsa_w = 1` must never be run (known unstable); `MAX_NSA_W = 0.99`, enforced at every entry
   point.
7. One plateau tolerance for every method, `DEFAULT_TOL = 1e-4` (A1, measured).
8. ADNI: k = 3, with per-component stability shown. `select_rank` chooses k by stability in
   general.

## Commits

| commit | scope |
|---|---|
| `3a88f5f` | Stop vocabulary, one orthogonality weight, and the consensus topology default. Also fixes the runner's deep-convergence bug and the D10 follow-ups. |
| `580bb91` | `pysimlr.loadings` and `loading_inference`; `structure`/`loadings` on every fit; the OLEE contract on the loadings. |
| `d3e5590` | Defect penalty removed; `DEFAULT_SIMLR_TOL = 1e-4`, `DEFAULT_SIMLR_ITERATIONS = 300`. |
| `5c77d60` | Benchmark scores both the encoder (`*_v`) and the loadings (`*_loadings`); `lobe_crosstalk` removed. |
| `792e9a6` | Stability-selection `q` chosen from the error bound; amigo evaluation script and evidence. |
| `c18384a` | This document, `docs/amigo_report.html`, contract docs. |
| `9f2a5d4` | `nsa_w > 0.99` rejected everywhere; tests moved from 1.0 to 0.99. |
| `5cc49d8` | `DEFAULT_TOL = 1e-4` for every method; `scripts/deep_tol_selection.py`. |
| `9cdf093` | `pysimlr.select_rank`; `scripts/adni_rank_selection.py`. |
| `a045a81` | Resolve the three post-design-change xfails. |
| `acd08fe` | Equal weights for scale-free view energies. |
| `eabab15` | Re-measure after equal view weights. |
| `708c9c6` | View-weighting remediation (R1 at default budget). |
| `394144c` | Keep `acc` view scale under Decision Rule R9 (WP-1b). |
| `84df107` | Drop `simlr_lbfgsb` and `nsa_lbfgsb` per Decision Rule R5 (WP-7). |

## Why: feature identifiability (measured)

**Mechanism.**
- Under the correlation energy `nc`, the scores `XV` determine `V` only where `X` has variance.
- Restricted to the leading-variance directions, refits of `V` agree: mean sin² 0.014, against
  0.28 overall at 98% subsamples.
- Input noise of 1e-6 moves nothing, so the fit is continuous. The instability is weak
  identification, not chaos.
- The low-variance share of `V`'s norm (about 60%) has no chance baseline, because a random `V`
  puts 63-90% there. It is not evidence of the mechanism on its own.

**What the energy changes.**
- `acc` and `recon_r2` identify `V`.
- They fail on the buried-signal generator: latent R² 0.10-0.62, against 0.98 for `nc`.

**Remedies compared** (scratchpad `ident_fix.csv`):

| remedy | latent R² | support F1 | V instability |
|---|---|---|---|
| two-stage loadings | 0.988 (unchanged) | 0.709 | 0.059 |
| ridge `nc`, λ = 0.5 | 0.983 | 0.708 | 0.055 |

PCA restriction fails: at 90% of variance it drops the buried signal; at 99% it keeps the flat
directions.

**Three independent reviews** (scientific, statistical, ML engineering) agreed:
- report correlation-scale structure coefficients and keep `V` for scoring;
- put the OLEE contract on the loadings;
- gate claims on whole-pipeline refit inference.

The statistical review found that in-sample `max |C|` reaches 0.4-0.5 under a null where about
0.07 is expected. This is selection inflation from fitting `U` on the same rows, so plug-in
thresholds are anti-conservative.

## Results: `scripts/amigo_loadings_evaluation.py`

Evidence: `docs/audit/evidence/amigo_loadings_{synthetic,adni}.csv`.

**Synthetic data.** Default `simlr`, 4 cells × 3 seeds, two refits on 98% of the rows. Every fit
converged, in 41-78 sweeps, with latent R² 0.982-0.997.

| sign mode | object | span instability | matched-column \|r\| | support F1 | column coherence | support overlap |
|---|---|---|---|---|---|---|
| signed | V | 0.335 | 0.637 | 0.676 | 0.019 | 0.416 |
| signed | structure | 0.0023 | 0.999 | 0.558 | 0.079 | 0.307 |
| signed | loadings | 0.0023 | 0.999 | 0.558 | 0.000 | 0.303 |
| positive | V | 0.124 | 0.912 | 0.603 | 0.077 | 0.077 |
| positive | structure | 0.0013 | 0.999 | 0.558 | 0.078 | 0.303 |
| positive | loadings | 0.0009 | 0.999 | 0.676 | 0.000 | 0.000 |

**ADNI-MCI.** 4 views, p = 160/160/62/11, graph topology; `loading_inference` with 30
half-sample refits.

| sign mode, k | V instability | loadings instability | loadings matched \|r\| | component stability | selected (E[false]) |
|---|---|---|---|---|---|
| signed, 3 | 0.616 | 0.343 | 0.63 | 0.84 / 0.99 / 0.38 | 7 (≤ 3.3) |
| signed, 2 | 0.635 | 0.280 | 0.87 | 0.99 / 0.73 | 5 (≤ 3.3) |
| positive, 3 | 0.460 | 0.288 | 0.63 | 0.95 / 0.74 / 1.00 | 11 (≤ 3.3) |
| positive, 2 | 0.057 | **0.0068** | **0.995** | 0.96 / 1.00 | 13 (≤ 3.3) |

**Reading.**
- Loadings are 50-150× more stable than `V` on synthetic data and meet the contract exactly:
  coherence 0, and positive overlap 0.
- On ADNI, loadings are only as stable as `U`. At k = 3 one component is not identified in either
  sign mode. At k = 2 under positivity, everything is stable.
- Signed loadings are dense, so their support F1 (0.558) is below `V`'s (0.676). Sparse signed
  claims come from stability selection, not from the point estimate.

Measured cost of removing the defect penalty, apart from the 3 xfails: positive-mode `V` is less
stable on the planted case (span instability 0.030 → 0.161). The loadings are unaffected.

## Open items, owner decisions

- **A1. Done.** Deep fits at 1e-6 and 1e-4 are identical: 36 fits, same epochs and R². The
  per-epoch loss moves by about 1e-3 from minibatch noise.
  Evidence: `docs/audit/evidence/deep_tol_selection.csv`.
- **A2. Paper.** Several parts read objects or columns that changed:
  - `paper/appendix_flow_simr_v.qmd:235-278` reports "Signed Lobe Crosstalk" (`Lobe_Crosstalk`,
    identically 0) and `Frame_Defect_D` of `V`.
  - The ADNI and BRCA feature figures read `V`.
  - Replace these with `structure`/`loadings`, intervals from `loading_inference`, and
    per-component stability.
  - Regenerate every cache from the final code. This also covers deep-fit drift of about 1e-7
    from D10.
- **A3. Resolved** (owner decisions).
  - *Near-orthogonality at the strongest weight:* accepted. The new value, 1.26e-3, is
    re-baselined to a limit of 2e-3.
  - *Face-residual calibration test:* dropped. The `grad_map` diagnostic itself stays; see A7.
  - *`nc` support recovery:* investigated over 10 fixture seeds, with the old code in a worktree.
    - The drop comes from the objective, not `tol`: new code scores 0.885 at either `tol` and at
      300 sweeps, against 0.940 for the old code.
    - Under `hybrid_adam` the old fit kept the support (0.991) only because the tuned learning
      rate was 0.001, so V stayed at its initialisation.
    - Without the penalty V moves, `nc` improves by about 7e-4, and V's support drifts to 0.787.
      `nc` is nearly flat across supports.
    - The loadings recover the planted support exactly: 1.0000 on all 10 seeds with both
      optimizers.
    - The test now checks the loadings
      (`test_support_identifiability.py::test_nc_loadings_recover_the_planted_support`).
- **A4. ADNI rank. Owner decision: k = 3, with per-component stability shown.**
  `select_rank`, 20 half-sample refits per k (`docs/audit/evidence/adni_rank_selection.csv`):

  | sign mode | k = 2 | k = 3 | k = 4 | k = 5 |
  |---|---|---|---|---|
  | positive | 0.965 / 0.996 | 0.953 / 0.791 / 0.996 | min 0.674 | min 0.724 |
  | signed | 0.993 / 0.771 | 0.837 / 0.994 / 0.417 | min 0.593 | min 0.481 |

  By the 0.9 rule the data identify k = 2 under positivity and no k when signed. At k = 3 the
  paper should show the 0.79 (positive) and 0.42 (signed) components as not identified. On the
  planted linear case (k = 3), `select_rank` recovers 3 on two seeds: k = 4 adds a component at
  0.66-0.69, and k = 2 merges two components (one at 0.891).
- **A5. Signed displays.** The polar summary is dense. Decide whether signed figures show
  `structure` with intervals, or only stability-selected features.
- **A6. Ridge `nc` (λ ≈ 0.5)** as an optional fit setting that makes `V` itself identified. It
  needs CV for λ and an ADNI check. It is not on by default.
- **A7. `grad_map` under projection-only fits.** It no longer falls 2 decades on the calibration
  fixture. Keep it as a diagnostic, re-calibrate it, or drop it.
- **A8. `loading_inference` cost and defaults.** The default `n_subsamples = 100` costs 100 refits.
  The permutation null is opt-in. Bootstrap percentiles from half-samples are conservative.

- **A9. `lars` oscillates.** With the auto-tuned `learning_rate = 1.0`, on the support fixture
  (old and new code alike), the total energy alternates between two states every sweep: -2.08
  and -0.94. The fit reports "converged" after 11 sweeps only because the best value stops
  improving. Investigate the step rule; this is a 2-cycle, not convergence.
- **A10. `nc` per-view weights.** Adopted equal 1/M weights for scale-free energies
  (`similarity.SCALE_FREE_ENERGIES`, including `nc`, `align`, `procrustes`, `recon_r2`,
  and ICA terms) under Decision Rule R1. Total energy bounded min >= -1.
  Collapse-penalty finding: before this change, the collapse penalty was effectively off
  wherever calibrated weights were large (similarity term scaled by up to ~2000x); with equal
  weights, the penalty is effectively on for the first time. The narrow fixture under equal
  weights has R² 0.608 (cw=0) against 0.962 (cw=1), both fits unconverged (`no_descent` at sweep 5); see D2/WP-2.
- **A11. `acc` view scale (Decision Rule R9, WP-1b).** Evaluated candidate operator-norm scale
  $(\|X_m\|_2 \|u\|_F) / (n - 1)$ against $|E_0|$ calibration for non-scale-free energy `acc`
  (evidence: `docs/audit/evidence/view_weighting_acc.csv`).
  While the candidate scale satisfies view-rescaling invariance, it failed 5 per-cell representation
  guards on the WP-1R metric set:
  - `latent_r2` on `linear_positive_s3`: loss of 0.1441 (> 0.05 budget)
  - `loadings_instability` on `buried_positive_s2`: loss of 0.1312 (> 0.05 budget)
  - `loadings_support_f1` on `buried_positive_s4`: loss of 0.2037 (> 0.10 budget)
  - `structure_init_invariance` on `buried_positive_s2`: loss of 0.1552 (> 0.05 budget)
  - `u_init_invariance` on `buried_either_s4`: loss of 0.1977 (> 0.05 budget)
  Per Decision Rule R9, candidate scale is rejected; outcome: `RETAIN CALIBRATED WEIGHTS` (`|E0|` calibration).

## On hold from the earlier plan

- **D2. Linear narrow-view collapse (Decision Rule R2, WP-2).** Evidence: `docs/audit/evidence/collapse_diagnosis.csv`.
  Auto learning rate stops with `no_descent` at sweep 5 (lr 0.3 at cw=0; grid endpoint 1.0 at cw=1).

  | fit | converged | best_energy | outcome R² | \|score corr\| |
  |---|---|---|---|---|
  | cw=0, lr 0.1 | yes | −0.697 | 0.357 | 0.976 |
  | cw=0, lr 0.03 | yes | −0.697 | 0.266 | 0.979 |
  | **cw=0, lr 0.01** | **yes** | **−0.885** | **0.940** | **0.672** |
  | cw=1, lr 0.03 | yes | −0.183 | 0.962 | 0.591 |
  | cw=1, lr 0.01 | yes | −0.213 | 0.958 | 0.565 |
  | planted P(V_true) | — | −0.677 (pure similarity) | 0.988 | 0.428 |

  Collapse is a step-size basin effect. The unpenalised objective's lowest converged energy (−0.885, lr 0.01) is non-collapsed (|corr| 0.672, R² 0.940).
  R2 re-applied with the lowest-energy fits: condition 1 passes (−0.018, guard −0.05), condition 2 passes on 20 default cells (latent R² worst −0.0014, loadings F1 worst −0.0031), condition 3 fails (0.672 ≥ 0.5) → keep `DEFAULT_COLLAPSE_WEIGHT = 1.0`.
  The penalty costs pure similarity −0.885 → −0.213 for +0.018 outcome R²; see P1.
  Sanity-check deviation: post-update `u` at `V(cw=0, lr=0.1)` gave residual 6.12e-5 (> 1e-6), so the test checks iteration 0 (0.0); recomputing with the previous sweep's `u` (`best_iteration = 10`) gives residual 0.0 (< 1e-6), verifying the one-sweep `u` logging lag.
- **D4. Deep collapse (Decision Rule R3, WP-5).** Evaluated deep collapse on the NED fixture
  (n = 200, 2 views with 20 features, `ned_simr(iterations=100, batch_size=32, warmup_epochs=10)`)
  using `scripts/deep_collapse_diagnosis.py` (evidence: `docs/audit/evidence/deep_collapse_diagnosis.csv`).
  - *Step 1 (collapse rates across 20 data seeds 1..5 × start seeds 1..4)*:
    - With penalty (`hinge`, `DEFAULT_COLLAPSE_WEIGHT = 1.0`): **0/20** pairs collapsed (|corr(u)| >= 0.5, max |r| = 0.194, mean |r| = 0.081).
    - Without penalty (`none`, `pysimlr.deep.correlation_penalty` patched to zero per Rule 8): **7/20** pairs collapsed (35%, |corr(u)| between 0.765 and 0.981, mean 0.895; latent |corr(z0)| >= 0.997).
    - Collapsing pairs: (1, 4), (2, 4), (3, 2), (3, 4), (4, 3), (5, 1), (5, 2).
  - *Step 2 (determinism & onset timing)*:
    - Determinism verified bit-identical (`max param diff = 0.0`, `max u diff = 0.0`).
    - Prefix sweeps at iterations in {10, 20, 40, 100} on the 7 collapsing pairs pinpoint onset:
      - At iteration 10 (end of warmup, autoencoder reconstruction only): **0/7** collapsed (mean |corr(u)| = 0.135, range 0.011 to 0.266).
      - At iteration 20 (10 epochs after "nc" similarity activated): **7/7** collapsed (mean |corr(u)| = 0.816, range 0.600 to 0.934; latent |corr(z0)| jumped to mean 0.985).
      - Onset diagnosis: dimensional collapse originates in the similarity phase (epochs 10-20), not during the reconstruction warmup phase.
  - *Step 3 (energy evaluation & constructibility)*:
    - A planted warm start is not constructible for the MLP model.
    - Evaluated similarity against model's returned consensus (u_per_modality; recomputed matches final training similarity to <= 1e-5).
    - Penalised fits (n=20): mean sim_loss = -1.9269 (min -1.9970, max -1.7079), mean total_unpen = -1.9195.
    - Unpenalised fits: uncollapsed (n=13) mean sim_loss = -1.9355, total_unpen = -1.9282; collapsed (n=7) mean sim_loss = -1.4804, total_unpen = -1.4730 (incurring hinge collapse loss 0.9958).
  - *Decision Rule R3 evaluation*:
    - Condition: Keep the deep penalty unless the count without it is 0/20 and the count with it is no better.
    - Measured: without penalty 7/20, with penalty 0/20 -> keep 1.0.
- **D3. Correlated factors (Decision Rule R4, WP-6).** Status: **explained: the penalty caps score correlation at about t**.
  Evidence: `docs/audit/evidence/correlated_factors.csv` and `docs/audit/evidence/collapse_threshold_selection.csv`.
  Clipping is confirmed: maximum score correlation of converged `avg` fits tracks $t$: 0.302 ($\rho=0.5$) and 0.309 ($\rho=0.7$) for $t=0.3$; 0.499 ($\rho=0.5$) and 0.509 ($\rho=0.7$) for $t=0.5$; 0.647 ($\rho=0.5$) and 0.692 ($\rho=0.7$) for $t=0.7$. The penalty caps recoverable score correlation at about $t$.
  Subspace R2 drop from rho=0.0 to rho=0.7 is 0.0207 for newton and 0.0209 for avg; newton decorrelates scores while avg retains correlation.
  Trade-off: $t=0.3$ protects narrow views (at $t=0.7$ narrow |score corr| is 0.89 and outcome $R^2$ is -0.10), but caps correlated factors at about 0.3 under `avg`. Note that R15's criterion (ii) used `newton`, which cannot show threshold effect (reviewer design flaw).
- **P1. Collapse threshold selection (Decision Rule R15).** Status: **COMPLETE** (outcome: **KEEP 0.3**).
  Evidence: `docs/audit/evidence/collapse_threshold_selection.csv` (150 rows across t in {0.3, 0.5, 0.7}).
  Narrow fixture outcome R2 at lr=0.01: t=0.3 is 0.9582, t=0.5 is 0.9323 (diff -0.0260), t=0.7 is 0.8561 (diff -0.1022, fails -0.05 guard). The narrow t=0.5 fit is unconverged too (11 steps).
  Default cells: t=0.5 default-cell failure is one unconverged fit (`buried_positive_s0` stopped with `no_descent` at 25 steps, R2=0.6045; the other 19 cells match t=0.3). Candidate t=0.7 is non-inferior on cells. Scan correction: 1 low-R2 bound hit (not 3). No re-run needed: t=0.5 also fails criterion (ii).
  Structure Hungarian |r| at rho=0.7: t=0.3 is 0.9575, t=0.5 is 0.9561 (diff -0.0013), t=0.7 is 0.9573 (diff -0.0002); neither beats 0.3 by > 0.005.
  Trade-off: t=0.3 protects narrow views (at t=0.7 narrow |score corr| is 0.89 and outcome R2 is -0.10), but caps correlated factors at about 0.3 under `avg`. Note that R15's criterion (ii) used `newton`, which cannot show threshold effect (reviewer design flaw).
  Decision Rule R15: keep DEFAULT_COLLAPSE_THRESHOLD = 0.3.
- **D5. simlr_lbfgsb calibration vs drop (Decision Rule R5, WP-7).** Executed 3×3 proximal
  hyperparameter grid search using `scripts/lbfgsb_calibration.py` (evidence:
  `docs/audit/evidence/lbfgsb_calibration.csv`):
  - *Grid points*: `_PROX_QUAD_COEFF` $\in \{1.0, 2.5, 5.0\}$ × `_PROX_QUARTIC_COEFF` $\in \{16.0, 64.0, 256.0\}$.
  - *Evaluation design*: 12 cells ({buried, linear} × {either, positive} × seeds 0–2) at default budget (300 iterations), totaling 108 fits in parallel (`ProcessPoolExecutor(max_workers=6)`).
  - *Baseline*: `hybrid_adam` equal-arm baseline from `docs/audit/evidence/view_weighting_selection.csv` (mean latent R² 0.9859 on buried, 0.9840 on linear; support F1 0.7088 on buried, 0.7303 on linear).
  - *Measured calibration results*:
    - Across all 9 grid points, `simlr_lbfgsb` suffers severe collapse on the buried-signal cells (`buried_either`, `buried_positive`): mean latent R² collapses to $0.0108 - 0.0153$ (individual cell latent R² $\approx 0.009 - 0.021$).
    - Non-inferiority cell guard: paired cell $\Delta \text{latent\_r2}$ on buried structure reaches $-0.9635$ to $-0.9684 \ll -0.05$ guard across all 9 grid points.
    - Non-inferiority mean difference: mean $\Delta \text{latent\_r2}$ across all 12 cells ranges from $-0.485$ to $-0.488 \ll -0.05$.
    - Perturbation reproducibility (D9 metric, 1e-8 input noise, QR projector diff norm): 5 of 9 grid points fail the $10^{-4}$ gate (mean perturbation gap up to $0.0048$, max $0.0416$).
  - *Decision Rule R5 evaluation*:
    - Condition: Retain `simlr_lbfgsb` if at least one grid point achieves non-inferiority to `hybrid_adam` (mean $\Delta \text{latent\_r2} \ge -0.05$, mean $\Delta \text{support\_f1} \ge -0.10$, and per-cell guards $\ge -0.05$ and $-0.10$ respectively); otherwise drop `simlr_lbfgsb` and `nsa_lbfgsb`.
    - Result: 0 of 9 grid points meet non-inferiority (all 9 fail the per-cell $\Delta \text{latent\_r2}$ guard by $\approx -0.968$).
    - The R5 guard failed on one cell (`buried_either_s0`, R² 0.015 against 0.979, in all 9 constant settings); not isolated, because the method was dropped.
    - *Verdict*: **DROP** `simlr_lbfgsb` and `nsa_lbfgsb` per Decision Rule R5.
  - *Action taken*:
    - `src/pysimlr/api.py`: Removed `simlr_lbfgsb` from registered specs and presets. Added to `REMOVED_METHODS` raising `ValueError` directing users to `'simlr'` with default optimizer `'hybrid_adam'`.
    - `src/pysimlr/optimizers.py` & `src/pysimlr/simlr.py`: Removed `NSALBFGSB` class and `nsa_lbfgsb` from optimizer registry. Added to `REMOVED_OPTIMIZERS`. Removed `LEARNING_RATE_FREE_OPTIMIZERS`, `gradient_function` hook, and `smooth_energy_fn`.
    - Migrated tests and documentation across test suite, scripts, and contract docs.
- **D6. NED rounding sensitivity (Decision Rule R6, WP-8).** Evaluated float32 training divergence under 1e-8 Gaussian input noise across data seeds {42, 2, 1} and model seeds {0, 42} using scripts/ned_rounding.py (evidence: docs/audit/evidence/ned_rounding.csv).
  - Epoch-0 parameter distance d(0) measures 1.76e-7 (ds 1), 2.56e-7 (ds 42), and 3.16e-7 (ds 2), all strictly below the 1e-6 initialisation threshold.
  - Across 100 epochs, parameter distance grows continuously (e* = 27 to 75, max amplification 2.57e7x, mean Lyapunov exponent 0.1004/epoch) with 0 discrete tie-breaks across 4 candidate operations.
  - Decision Rule R6: because d(0) < 1e-6 and no discrete tie-breaks occurred, amplification is continuous chaotic gradient amplification (Branch B).
  - Verdict: Branch B confirmed; ned remains listed in _ROUNDING_UNSTABLE in tests/test_api_registry.py.
- **T2. Change requested in ANTsTorch** (a different repository, not edited from here):
  - Request to ANTsTorch maintainers, precise enough to apply blind:
    In `antstorch/__init__.py`, provide a PEP 562 `__getattr__` for `create_real_nvp_normalizing_flow_model`, imported from `lamnr_flows/architectures/create_normalizing_flow_model.py`:
    ```python
    def __getattr__(name: str):
        if name == "create_real_nvp_normalizing_flow_model":
            from .lamnr_flows.architectures.create_normalizing_flow_model import (
                create_real_nvp_normalizing_flow_model,
            )
            return create_real_nvp_normalizing_flow_model
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ```
  - Benefit: about 2.6 s faster import time by avoiding eager imports of ITK, ANTsPy, and matplotlib when loading normalizing flow models.
  - pysimlr follow-up once it lands: delete `flows._load_antstorch_real_nvp_builder` and `tests/test_flow_antstorch_loader.py`, reverting `src/pysimlr/flows.py` to `from antstorch import create_real_nvp_normalizing_flow_model`.
  - Invariant: do NOT edit ANTsTorch from pysimlr.
- **Release.** v0.5.0 after the paper phase, on instruction.

## Caveats

- Synthetic results use few seeds on the `default_selection_benchmark` generators.
- ADNI checks are not covariate-adjusted.
- `true_v` in the linear generator is the latent-to-data mixing, which is the object the loadings
  estimate. `V` maps data to latent, so its support F1 is against a different object.
