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
| (next) | `pysimlr.select_rank`; `scripts/adni_rank_selection.py`. |

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
  weights has cw=0 R² 0.608 vs 0.348 (with cw=1 R² 0.962 vs 0.937). This finding carries
  directly into WP-2 / D2.

## On hold from the earlier plan

- **D2/D4. Collapse: objective or optimizer?**
  - Linear narrow-view fixture: under calibrated weights R² was 0.301 without penalty; under equal weights cw=0 R² is 0.608 (vs 0.962 with cw=1).
  - Deep: 6/20 (data seed, start seed) pairs collapse without the penalty.
  - Planned test: compare the energy at the planted solution with the energy at the collapsed
    one. This decides the 0.3 hinge threshold.
- **D3. Correlated factors.** The newton consensus decorrelates the scores; true ρ = 0.7 comes
  back as recovered ρ 0.03-0.15. Is this a gauge freedom or a defect? Report subspace R² against
  column recovery.
- **D5. D9 proximal constants** (2.5 / 64) are uncalibrated: run a 3×3 grid, or drop
  `simlr_lbfgsb` if it never beats `hybrid_adam`.
- **D6. NED rounding sensitivity** (`_ROUNDING_UNSTABLE`): find the first epoch where the
  trajectories diverge, and the operation that causes it.
- **T2. Change requested in ANTsTorch** (a different repository, not edited from here):
  - Make `create_real_nvp_normalizing_flow_model` a lazy (PEP 562) export of
    `antstorch/__init__.py`, so importing it does not load ITK/ANTsPy/matplotlib (~2.6 s).
  - pysimlr then returns to `from antstorch import ...` and deletes
    `flows._load_antstorch_real_nvp_builder` and `tests/test_flow_antstorch_loader.py`.
- **Release.** v0.5.0 after the paper phase, on instruction.

## Caveats

- Synthetic results use few seeds on the `default_selection_benchmark` generators.
- ADNI checks are not covariate-adjusted.
- `true_v` in the linear generator is the latent-to-data mixing, which is the object the loadings
  estimate. `V` maps data to latent, so its support F1 is against a different object.
