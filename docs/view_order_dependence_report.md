# View-order dependence of `pysimlr.fit("simlr", ...)` with averaging mixings

Branch `debug/view-order-dependence` (worktree of pysimlr at v0.30.0, c8cb44c). Nothing pushed, main untouched.

## Summary

1. The cause is the greedy sign alignment `align_column_signs` (`src/pysimlr/utils.py`, running sum at ~L2369-2381: first matrix is the reference, each next matrix is flipped to agree with the running sum). The default mixing is `newton`, an averaging mixing, and `compute_shared_consensus` calls it at `consensus.py` L205 (and L309 for pruning). When the views' latents are not mutually sign-consistent, which is the case at the PCA start (cosines of 0.0-0.2 with mixed signs), the flips depend on the order the views are visited. Measured on real fold data: the same two orders flip different views and the first-sweep consensus differs by up to 3.2 (u has unit SD).
2. It is a defect in the sense that the model is documented as symmetric in views and this step is not. It is not the whole story: the objective is also very flat/multi-modal here. A 1e-7 relative jitter of the data, with no change of order, already moves out-of-fold scores (RMS 0.008, max 0.03); fixing the sign step removes the structural part but not that sensitivity.
3. The layers above pysimlr (private fold loop, antsxbrainmapping helpers) are not a source of order dependence: with the same permutation applied by hand to a recorded fold's inputs (views, per-view params, confounds, graph), pysimlr gives the same differences; the pipeline only feeds views in the given order. Seeds are not a source: with `initialization_type="pca"` the fit is deterministic and `random_state` is unused (seed 1 vs 2 vs 42 give identical u).

## What is and is not order dependent in pysimlr (read, then measured)

* Updates are Jacobi, not Gauss-Seidel: `simlr.py` computes the consensus once per sweep (`compute_shared_consensus`, ~L2176) and then each view takes its step from that fixed `u`; no later view sees an earlier view's update. Order of the view loop therefore does not matter.
* Per-view parameters, confounds, `path_graph`: matched by index and permuted consistently in the reproducer (and by `api.fit` for names); no mismatch found.
* Root: found by the caller (`fit_names.index`); the graph is keyed by index. Not "view 0".
* Random draws: PCA init draws none. `random`, `perturbed_pca` consume one generator sequentially across views (`simlr.py` `initialize_simlr`, ~L352), so position decides each view's draw. That is order dependence by design and is not what the study uses.
* The root's consensus concatenates/averages its neighbours in `path_graph[root]` list order; with `svd`/`pca`/`ica` this only permutes columns (float-level differences), with `avg`/`newton` it goes through the greedy alignment above.

## Reproducer (synthetic)

`docs/view_order_dependence/repro.py` (`--smoke` supported) builds views of different widths with a shared latent, exposure root, k=1, same settings as the study. `docs/view_order_dependence/bisect_order.py` (`--smoke`) produces the table below. Minimal form, from `tests/test_view_order_sign_alignment.py`:

```python
views = _noise_views(seed)                     # five noise views (65,54,72,57,196 columns) + 1-column Root
o1 = ["V0","V1","V2","V3","V4","Root"];  o2 = ["V0","V1","V2","V4","V3","Root"]
fit(views, o1, iterations=1, sign_alignment="greedy") vs fit(views, o2, ...)   # u differs by ~3.6-3.8 (seeds 3,4)
same with sign_alignment="spectral"                                            # u differs by <1e-6
```

Note that on well-conditioned synthetic data with real shared signal (signal 0.4), the two orders agree to float32 rounding: the defect appears when latents are weakly coupled at initialization, as in the real folds.

## Bisection (synthetic; 8 seeds; cell = max |du| over seeds after undoing the permutation and a Procrustes/sign alignment of u, then number of seeds with |du| > 0.01)

| setting | after 1 sweep | after 300 iterations |
|---|---|---|
| baseline: graph, newton, pca init, conical, sp 0.8, nc_abs | 3.77 (2/8) | 5e-7 (0/8) |
| 20 NaN rows in 3 views, `missing_rows="mask"` | 3.25 (3/8) | 5e-7 |
| `missing_rows="impute"`, no NaN | 3.77 (2/8) | 5e-7 |
| confounds on | 3.81 (1/8) | 5e-7 |
| shared signal 0.4 | 7e-7 | 7e-7 |
| mixing avg | 3.77 (2/8) | 5e-7 |
| mixing svd / pca / ica | <2e-6 | <8e-7 |
| init random | 3.92 (8/8) | 0.99 (8/8) |
| init perturbed_pca | 0.86 (7/8) | 0.94 (8/8) |
| init nndsvd | 5e-7 | 1.26 (1/8) |
| init gcca | 7e-7 | 5e-7 |
| energy nc | 3.57 (3/8) | 5e-7 |
| positivity either | 2.17 (3/8) | 5e-7 |
| sparseness None | 5e-7 | 5e-7 |
| positivity either + sparseness None | 5e-7 | 7e-7 |
| k = 2 | 3.73 (3/8) | 1.47 (4/8) |
| no root/topology args (star) | 5.39 (3/8) | 3.12 (5/8) |
| newton + `sign_alignment="spectral"` (fix) | 7e-7 | 5e-7 |
| k = 2 + spectral | 2.09 (6/8) | 2.83 (6/8) |
| star + spectral | 7e-7 | 0.20 (3/8) |
| star + svd mixing | 2e-6 | 1.28 (4/8) |

(`energy recon` errors in this fixture and is omitted. The synthetic data have no shared signal except through the root, so the star rows after 300 iterations show multi-modality, not the sign step: svd mixing is float-level at sweep 1 and still differs at 300.)

Reading: asymmetry present after one sweep, only for averaging mixings (avg/newton) and only when latents are sign-inconsistent; gone with svd/pca/ica; independent of mask, confounds, energy, positivity and sparseness (in this fixture sparseness/positivity-off removes it only because the initial projections become sign-consistent). Where the optimizer is well conditioned (baseline, 300 iterations) the first-sweep difference is forgotten and orders agree to 5e-7: the order effect is amplified, not created, by the later optimization. k = 2 is a separate problem: signs are not enough, because averaging pairs column l of each view with column l of the others, but components of different views can be rotated or swapped (spectral alignment does not fix it).

## Real pipeline (read-only on the participant table; aggregates only)

Per-fold fits recorded through `pysimlr.fit` for the two orders (5 folds, one repeat). Folds 0 and 2 agree to float32 (du 7e-7, 2e-6). Fold 1 converges to a different solution (du 1.4; energy -0.5020 in order A vs -0.4978 in order B; 93 vs 78 steps), folds 3 and 4 differ by 0.002 and 0.013.

With `iterations=1`, applying the permutation by hand to the recorded inputs, folds 1 and 4 differ by 0.85 and 3.18 under newton or avg mixing and by 1e-6 under svd/pca mixing; with spectral alignment they agree to 0.0. The recorded cosines among the five views' first latents are all below 0.23 in absolute value with mixed signs; greedy flipped (ASL, rsfMRI) in order A and only (ASL) in order B.

Study-level effect (3 repeats, 5 folds, all four cohorts; out-of-fold score SD about 0.305; r is the within-cohort correlation with exposure):

| comparison | max abs score difference | median | RMS | r with exposure |
|---|---|---|---|---|
| order A vs B, greedy (current) | 0.30 | 0.036 | 0.078 | 0.263 vs 0.273 |
| order A vs B, spectral alignment | 0.069 | 0.0043 | 0.0135 | 0.253 vs 0.253 |
| same order, `perturbed_pca` seed 42 vs 43 | 0.52 | 0.075 | 0.125 | 0.248 vs 0.252 |
| same order, `perturbed_pca` seed 42 vs 44 | 0.32 | 0.078 | 0.118 | 0.248 vs 0.279 |
| same order, `random` init seed 42 vs 43 | 0.40 | 0.098 | 0.140 | 0.237 vs 0.255 |
| same order, data jitter 1e-7 (relative), pca init | 0.027 | 0.0044 | 0.0077 | n/a |
| same order, pca init, seed 42 vs 1 vs 2 | 0 | 0 | 0 | identical |

(Study values in the task, 5 repeats: r 0.2755 vs 0.2753 with differences up to 0.25; consistent with RMS 0.078 here: averaging over repeats shrinks neither systematic flips nor shared structure.)

So view order is another source of optimizer variability, about two thirds the size of re-seeding a perturbed start (RMS 0.078 vs 0.12-0.14), much larger than float noise (0.008), and r with exposure moves by 0.001-0.01, i.e. smaller than the seed-to-seed spread of r (0.248-0.279). Under the pca start the study has no seed variability at all, so order is currently its only (hidden) source, and it exceeds the effect of a 1e-7 data perturbation by roughly a factor of ten. After the sign fix the order effect is 0.0135 RMS, about 1.8 times the 1e-7 jitter, i.e. what remains is the flat/chaotic objective.

## Fix (prototype, behind an argument that defaults to current behavior)

* `utils.align_column_signs(mats, reference=None, method="greedy")`: new `method="spectral"` uses, per column, the leading eigenvector of the Gram matrix of the views (`_spectral_signs`); symmetric in the views, so permutation-equivariant. Global sign: majority of views keep their sign, then `sum_i s_i sum(m_i)`, then +1. Equals greedy whenever all views agree in sign.
* `sign_alignment` parameter ("greedy" default, "spectral") on `simlr(...)`, `compute_shared_consensus(...)` and the capture helpers (`simlr.py`, `consensus.py`); recorded in the result next to `orthogonalize_u`. Prediction uses the frozen least-squares map, so it needs no change. `api.fit` accepts it automatically (derived from the signature). Not threaded into `deep.py` callers (they keep greedy).
* Effect on the synthetic reproducer: first-sweep difference 3.77 -> 7e-7; the 300-iteration difference stays at float level where it was already.

## Tests

* New `tests/test_view_order_sign_alignment.py` (synthetic): greedy is order-dependent on a frustrated triple; spectral is permutation-equivariant; spectral equals greedy when views agree; unknown method raises; one-sweep fit is order free only with spectral (greedy diff > 0.1, spectral < 1e-4). 5 passed. Strictly, the two "fails before" cases are the spectral ones, which raise `TypeError`/`ValueError` on the unmodified code because the argument does not exist; the greedy-dependence assertions pass on both.
* `tests/test_parameters_are_live.py`: `sign_alignment` added to EXEMPT with a reason.
* Full suite (`pytest -q`, 97 s): 2080 passed, 3 failed, 2 skipped, 2 xfailed. The three failures are pre-existing on the unmodified checkout (verified with the changes stashed): `test_collapse_penalty::test_r2_linear_collapse_contract_and_sanity_residual` (ValueError), `test_parameters_are_live::test_every_parameter_is_classified` (unclassified `missing_rows`, `per_view_params`; my parameter is classified), `test_mai_calibration::test_simlr_ranks_a_wide_noise_view_last...` (also failing on the base).

## Decisions for the lead author

1. Make `sign_alignment="spectral"` the default (changes every averaging-mixing fit with 3+ views whose latents disagree in sign; all published numbers from newton/avg would shift by the amount above), or keep greedy as default and set it explicitly in `single_run._fit_kwargs`.
2. The study still has about 0.013 RMS (max 0.07) of residual order sensitivity, and the pca start hides all seed variability. If the claim is "robust to fit order", report the seed/jitter spread; consider multi-start averaging.
3. Mixing `svd`/`pca` is already order-free at sweep 1 but gives different models; not evaluated for study quality.
4. k > 1 needs more than sign alignment (component rotation/swap); not fixed here.
5. Sequential-seed initializations (`random`, `perturbed_pca`) are order dependent by design; seeding per view by name would change the draws for everyone.
