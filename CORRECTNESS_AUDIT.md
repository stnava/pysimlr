# Correctness audit: findings and fixes

A review of the machine-learning, statistical and software correctness of
`pysimlr`, with every claim verified by execution. Each finding below names the
defect, the evidence, and the fix. Regression tests were added for all of them;
the new test files are listed at the end.

Measurements come from a synthetic 3-factor benchmark (n=120, two views of 25
and 18 features, noise 0.4) unless stated otherwise.

---

## Critical — silently wrong results

### 1. Deep/flow models produced a different `U` in `eval()` than in `train()`

`consensus.py` — the anchored prediction branch returned *before* the
centering/standardization block the training path applies. On a trained
`LENDSiMRModel(mixing_algorithm="svd")` with identical input:

| | before | after |
|---|---|---|
| eval `u` std | `[0.587, 0.522, 0.465]` | `[1.0, 1.0, 1.0]` |
| ratio eval/train | 0.59, 0.52, 0.46 | 1.0, 1.0, 1.0 |

A second defect compounded it: the anchor is an EMA of SVD right-singular-vector
matrices that were never sign-aligned, so columns flipping sign between epochs
cancelled (anchor norm 1.352 against 1.732 for an orthonormal 3-basis).

**Fixed.** The anchor branch now falls through to the shared standardization,
and `align_anchor_columns` orients each new basis to the stored anchor before
the EMA. Anchor norm recovered to 1.706. Any model fitted on training `U` was
previously miscalibrated on held-out `U`.

*Partially open:* the anchor pins the rotation, but the centering/scaling
moments are still recomputed per batch, so anchored prediction remains
batch-composition dependent. Tracked as a `strict=True` xfail in
`tests/test_consensus_anchor_standardization.py`.

### 2. Deep training stopped before the similarity objective was optimized

`deep.py` — the similarity term switches on at `warmup_epochs`, which
discontinuously raises the objective, but `best_loss` was carried across that
boundary. No post-warmup epoch could beat it, so the patience counter
incremented every epoch and training always stopped at `warmup + patience`.

Requesting 150 epochs ran **26**, with the loss minimum at epoch 15 — during
warmup, before the similarity term was active at all. The similarity objective,
the entire point of SiMLR, received 6 epochs.

**Fixed.** The baseline resets when the objective's composition changes. The
same request now runs 51 epochs and the similarity loss falls 2.223 → 0.256.

### 3. `indicator_opt_both_ways` returned m²⊙I

`sparsification.py` — it passed the *masked values* returned by
`optimize_indicator_matrix` back in as if they were a 0/1 indicator.

```
m = [[2, -1], [0.5, 3]]   →   [[4, 0], [0, 9]]      (expected [[2, 0], [0, 3]])
```

Entries were squared and signs destroyed. Because `sum(m * I_pos) ≥ 0` always
exceeded `sum(m * (−m⊙I)) ≤ 0`, the negative-orientation branch was unreachable.
This was the only path for `rank_based_matrix_segmentation(basic=False)`, which
additionally ignored its `sparseness_quantile`, `positivity` and `transpose`
arguments.

**Fixed.** Both functions rewritten around an exact assignment solver
(`scipy.optimize.linear_sum_assignment`), which also replaces a "max_iter"
loop that recomputed an identical greedy result every pass. A regression test
covers a case where greedy scores 11 against the optimum's 17.

### 4. `gradient_invariant_orthogonality_defect` was not that gradient

`utils.py` — the chain rule through the Frobenius normalization was missing.
Against autograd on the same matrix:

```
autograd  [ 0.0366,  0.0138,  0.0093]
manual    [ 0.2189,  0.0669, -0.0408]     ← sign differs on component 3
<A, grad> = 1.664                          (must be 0: the defect is scale-invariant)
```

Not a mis-scaling — a different direction. **Fixed** to the closed form
`(4/s)(Aₚ M − ‖M‖²Aₚ)`, now matching autograd to ~1e-17 across shapes, with
`<A, grad> = 0` restored.

### 5. `whiten_matrix` did not whiten

`svd.py` — `safe_pca` already returns the orthonormal left singular vectors,
i.e. the scores with `s` divided out once; the function divided by `s` again.
Covariance diagonal on a matrix with singular values 10…0.1:

```
before: [2.4e-07, 1.0e-06, 2.4e-05, 9.8e-05, 2.3e-03]
after:  [1.0, 1.0, 1.0, 1.0, 1.0]
```

**Fixed** to `u * sqrt(n−1)`, with a `rank` field and exact zeros for null
directions.

### 6. Fitting reseeded the global RNG, collapsing every permutation null

*Introduced by the fix in "design issues" below that unified the three
inconsistent NSA-Flow import names, and caught only because a calibration
test behaved impossibly.*

`nsa_flow_orth` takes a `seed` argument (default 42) and seeds the
**process-wide** generator with it. `simlr_sparseness` calls it once per view
per iteration, so after any `simlr()` call every subsequent `torch` random draw
was reset to the same fixed state:

```
randperm after 1st simlr_sparseness: [0, 3, 2, 4, 1, 5]
randperm after 2nd simlr_sparseness: [0, 3, 2, 4, 1, 5]   <- identical
```

Every permutation replicate therefore used the *same* permutation, and each
null distribution collapsed to a single repeated value:

```
null values: [-0.0016, -0.0016, -0.0016, -0.0016, -0.0016, ...]
```

With a constant null the p-value is pinned at either the floor or 1.0 depending
only on which side the observed statistic falls — which is exactly what the
calibration sweep showed (every replicate returning 0.050, or every replicate
returning 1.000, depending on data shape).

The blast radius is wider than the p-values: `simlr_perm`,
`paths.permutation_test` and `estimate_rank` all draw their nulls with
`torch.randperm`, and any downstream train/test split, bootstrap or
augmentation performed after a fit was equally frozen. Nothing failed; the
numbers were simply wrong.

`optimizers.py` already carried a manual save/restore for this, with the
comment *"nsa_flow_orth has side effects on global seed"* — the hazard was
known at one call site and not the others. Before the import was unified,
`import nsa` failed, so the classical path never reached the backend.

**Fixed** in `nsa_backend._rng_preserving`, which saves and restores CPU and
CUDA RNG state around every backend call, covering all call sites rather than
one. The backend's own internal determinism is unaffected; only the caller's
stream is protected. Regression tests in `tests/test_rng_isolation.py` assert
that successive fits leave the RNG advancing, that a fit preserves the stream
exactly, and that a permutation null actually varies.

### 7. Fitting a flow model switched the matplotlib backend, deleting figures

Same class of defect as finding 6, and found the same way -- by chasing an
output that made no sense rather than working around it.

The optional `antstorch` dependency calls ``matplotlib.use("Agg")`` at module
import time, in both `lamnr_flows/core/train_lamnr_glow_base.py:33` and
`lamnr_glow_tool_base.py:51`. `flows.py` imports antstorch lazily inside the
flow constructors, so merely *fitting* a flow model switched the process-wide
backend to Agg.

In a notebook or Quarto render that silently breaks inline figure capture: the
cell runs to completion and reports success, but emits nothing. Three figures
in the Flow-SiMR-V appendix had been missing from the manuscript for exactly
this reason -- the prose discussed them at length, the crossrefs
(`@fig-flow-imputation`, `@fig-flow-attribution`, `@fig-flow-geodesic`) were
unresolved, and the build logged only a crossref warning with no hint of the
cause. The frozen execution cache had captured the empty output, so rebuilding
reproduced the loss rather than exposing it.

**Fixed** with `flows.preserve_matplotlib_backend`, a context manager wrapped
around every antstorch import. antstorch's own behaviour is untouched; the side
effect no longer escapes into the caller's session. All three figures now
render and the manuscript builds with zero unresolved crossrefs.

### 8. `simlr_perm` p-values had a 42% false-positive rate

`simlr.py` used `ttest_1samp(null, observed, alternative='less')`, which tests
whether the null's *mean* lies below the observation rather than where the
observation falls in its upper tail. On 24 replicates of data with **no shared
structure whatsoever**:

| | mean p | rejections at α=0.05 | smallest p |
|---|---|---|---|
| t-test (before) | 0.447 | **42%** | 2.3e-11 |
| permutation (after) | 0.431 | **4%** | 0.05 |

Re-verified after finding 6 was fixed, since that defect had invalidated the
intermediate measurements. Fraction of runs at the attainable floor under a
true null, 16 replicates per configuration:

| n | features | n_perms | floor | observed at floor | mean p |
|---|---|---|---|---|---|
| 60 | 10 / 8 | 19 | 0.050 | 0.125 | 0.591 |
| 40 | 6 / 5 | 19 | 0.050 | 0.000 | 0.566 |
| 40 | 6 / 5 | 9 | 0.100 | 0.000 | 0.581 |
| 120 | 10 / 8 | 19 | 0.050 | 0.125 | 0.578 |

At 16 replicates the standard error on a 0.05 proportion is 0.054, so the
0.125 cells are within about 1.4 standard errors and the zeros are
conservative. The estimator is calibrated to slightly conservative, which is
the safe direction.

**Fixed** to the add-one permutation estimate `(1 + #{null ≥ obs}) / (1 + n)`
(Phipson & Smyth 2010). `paths.permutation_test` had the same defect class — a
plain `mean(null ≥ obs)` that can return an unattainable p of exactly 0 — and
received the same correction.

### 9. `positivity='positive'` reflected loadings and crippled the gradient

`sparsification.py` used `torch.abs(v)`, a reflection, where the sign
constraint calls for a projection — turning a genuinely negative loading into a
positive contributor of equal size, and disagreeing with
`orthogonalize_and_q_sparsify`, which clamps. Worse, `simlr()` routed the
**gradient** through the same call, forcing every gradient entry non-negative:
the update could only ever push V one way.

**Fixed.** `apply_positivity` resolves column sign ambiguity (so an
SVD column is flipped rather than annihilated) and then clamps; the gradient is
no longer sign-forced.

Latent recovery (Procrustes R-squared, mean +- sd over 10 seeds), measured on
the final state of the code so it also carries the constraint-default and energy-gauge fixes:

| positivity | optimizer | before | after |
|---|---|---|---|
| either | hybrid_adam | 0.9832 +-0.0046 | 0.9838 +-0.0050 |
| either | lars | 0.9865 +-0.0038 | 0.9843 +-0.0053 |
| either | armijo_gradient | 0.9854 +-0.0035 | 0.9848 +-0.0047 |
| **positive** | hybrid_adam | 0.8787 +-0.0671 | **0.9822 +-0.0048** |
| **positive** | lars | 0.9213 +-0.0169 | **0.9821 +-0.0051** |
| **positive** | armijo_gradient | 0.9297 +-0.0172 | **0.9830 +-0.0042** |

The sign-constrained path gains 0.10, 0.06 and 0.05, and its seed-to-seed
spread collapses by roughly 13x (+-0.067 to +-0.005) — the old variance was
the optimizer being unable to descend along coordinates it had overshot. The
unconstrained path moves by at most 0.0022 against a standard error of about
0.0016, i.e. within noise.

---

## Serious

### 10. `orthogonalize_u=True` was a complete no-op
`compute_shared_consensus` accepted `orthogonalize` and never referenced it.
**Fixed**: off-diagonal covariance now drops from 1e-1/3e-1 to ~1e-8 for the
svd/avg/newton mixing algorithms.

### 11. `topology="loo"` used a different estimator at prediction time
Training returned one consensus per modality; prediction short-circuited to the
**star** consensus — a different estimator and a different return type. Every
deep model's reported `u` was therefore a star consensus while the model was
trained against leave-one-out. **Fixed**: the topology is honoured in both
directions, and `_aggregate_shared_consensus` makes the single reported
embedding explicit (`u`), with the per-modality list alongside
(`u_per_modality`).

### 12. `preprocess_data` silently refit on test data
Every branch was guarded by `if provenance and KEY in provenance`; a missing key
— or an empty dict, since `if {}` is False — fell through and recomputed the
statistic from the held-out matrix, with no error:

```
real provenance → test mean [5.13, 3.80]      empty dict → [9.6e-07, 4.1e-06]
```

**Fixed**: incomplete provenance now raises. A method appearing twice in
`scale_list` also collided on one key; statistics are now namespaced by position.

### 13. `safe_pca` crashed when `nc` exceeded the rank
`safe_pca(randn(5,100), nc=10)` raised `RuntimeError: shape mismatch`.
**Fixed** by zero-padding to a consistent `(n, nc)` / `(p, nc)` / `(nc,)`.

### 14. `create_graph_laplacian` auto-detection misread common inputs

| input | detected as | result |
|---|---|---|
| `(P,3)` int voxel coordinates | `mesh_faces` | 31×31 graph instead of 4×4 |
| `(16,16)` uint8 mask | `coordinates` | 16×16 instead of 175×175 |
| `(8,8,8)` uint8 mask | — | `ValueError` |
| `(3,3)` float adjacency | `coordinates` | kNN over the rows, not the adjacency |

**Fixed**: any 3-D array and any binary 2-D array is a mask whatever its dtype;
symmetry with a zero diagonal identifies an adjacency; and genuinely ambiguous
integer `(P,2)`/`(P,3)` input now raises with instructions instead of guessing.
The mask connectivity default is dimension-aware (8 in 2-D, 26 in 3-D).

### 15. `return_torch=True` output could not be fed back in
`Tensor.is_sparse` is False for the CSR layout, so the function's own output
raised `TypeError: can't convert SparseCsr layout tensor to numpy`.
**Fixed** via a layout-aware converter; COO, CSR and dense all round-trip.

### 16. `lambda_val` was not scale-free
Inverse-distance weights carry the coordinate units, so identical geometry gave
different smoothing:

```
coords ×1:     ‖Sv‖/‖v‖ = 0.406        coords ×1000:  0.998
```

**Fixed** by an explicit `scale_degree` flag (mean degree 1), on by default for
the smoothing-operator factories where `lambda_val` lives, off for the raw
Laplacian builders so they still return the textbook matrix. Now 0.9557 at ×1,
×1000 and ×0.001, and comparable across weighting schemes.

### 17. The grid Laplacian used a pure-Python triple loop
**Fixed** with vectorized index shifts — bit-identical output, ~54× faster. A
216,000-voxel 3-D mask at 26-connectivity takes 0.11 s; an 11.8M-voxel mask
takes 4.7 s where the old code would need roughly four minutes.

### 18–20. Optimizer defects
- `bidirectional_linesearch` chose its direction by comparing **step sizes**,
  which says nothing about which direction descends. **Fixed** to compare
  achieved energies.
- `filter_params` built its result solely from a `defaults` dict that omitted
  `decay_rate`, `beta`, `k` and `alpha`, so those four were discarded even when
  passed explicitly — LARS, RMSProp and Lookahead were permanently stuck on
  fallbacks. Typo'd keys vanished silently. **Fixed**; unknown keys now warn.
- `create_optimizer` silently substituted HybridAdam for an unrecognised name.
  **Fixed** to raise.
- `BidirectionalArmijoGradient` updated a momentum buffer and ignored it.
- `larslow`'s trust-ratio fallback omitted `trust_coefficient`, giving any
  zero-initialised parameter 1000× the intended learning rate.
- `HybridAdam` applied the AMSGrad maximum unconditionally, ignoring `amsgrad`.
- LBFGS's closure returned a float and never refreshed the gradient, so its
  line search ran on a stale direction.

### 21–26. Smaller correctness defects
- `domain_lambdas=1` (an int) raised `'int' object is not subscriptable`.
- The ICA energy divided by `n` while its gradient divided by `k`.
- `decompose_energy` evaluated on **raw** data while `u`/`v` were fit on scaled
  data, and called `.numpy()` without `.detach().cpu()` (raises on GPU/MPS).
- `estimate_rank` raised `IndexError` on degenerate input.
- `ba_svd` returned `torch.randn` on SVD failure, making a numerical failure
  indistinguishable from a decomposition. Now raises.
- `exp(20 · v_raw)` overflowed to `inf` in float32 for `v_raw > 3.5`; the
  resulting NaN was silently converted to an all-zero basis, killing the
  modality. Now max-shifted.

### 27. Interpretability R² was in-sample and unregularized
Every alignment and attribution figure came from an effectively unregularized
fit (`l2=1e-6`) scored on its own training data:

| case | in-sample R² | cross-validated R² |
|---|---|---|
| pure noise, n=40, p=35 | **0.924** | **−366.8** |
| real signal, n=200, p=10 | 0.998 | 0.998 |

**Fixed**: `global_r2_cv` / `r2_per_target_cv` are now reported alongside. The
rank-deficient fallback also used `torch.linalg.lstsq(..., rcond=...)`, whose
default CPU driver assumes full rank and ignores `rcond`; it now uses the
pseudo-inverse.

---

## Design issues documented rather than changed

- **`constraint="ortho"` was a silent no-op** (weight defaulted to 0.0, and
  every soft-orthogonality branch is gated on `weight > 0`). Fixed to 0.1.
  `"orthox0"` still selects no constraint explicitly.
- **The `acc` energy was unbounded below.** With the column scale left free,
  `-sum|U'XV|` could be improved indefinitely by inflating ‖V‖; observed
  magnitudes reached 1e8, making the energy trajectory and the relative-change
  convergence test meaningless. Fixed by unit-normalizing columns (which also
  repaired `unit_norm`, silently ignored whenever `sparseness_quantile == 0`).
  Maximum |energy| over a run fell from 2.9e8 (hybrid_adam) and 1.0e7
  (armijo_gradient) to about 1.2 for every optimizer, so the trajectory and the
  relative-change convergence test are meaningful quantities again.

  Ordering matters here and I got it wrong first: normalizing *before* the
  sparsity step left the soft threshold free to shrink the columns afterwards,
  so `constraint="Stiefel"` returned V with ||v_j|| = 0.96 and `V'V != I`. The
  pre-existing code passed its Stiefel test only by accident — once half the
  coefficients are zero, `quantile(|v|, 0.5)` is exactly 0 and the threshold
  becomes a no-op, so the SVD retraction's unit norms survived untouched.
  Unifying the NSA-Flow import (below) made the retraction output dense again
  and un-masked it. Normalization now runs last in every constraint branch, and
  `V'V` is the identity with sparsity preserved.
- **Alternating minimization is not monotone.** `u` is recomputed after each
  sweep, so the energy falls steeply for a few iterations and then drifts
  upward — meaning `simlr` returned a worse iterate than one it had already
  found. It now tracks and returns `best_energy` / `best_iteration`. Two tests
  that asserted step-wise monotonicity were passing only because the energy was
  diverging; they now assert bounded, settling, overall descent.
- **The default soft orthogonality constraint costs accuracy.** Latent recovery
  was better without it (`orthox0`: 0.9867) than with it (`ortho`: 0.9826), and
  degraded further as the weight rose (`orthox0.9x3`: 0.9732). Documented in
  the README rather than changed, since the default is a modelling choice.
- **`acc` and `logcosh` sum the full K×K cross-covariance**, rewarding
  off-diagonal coupling and so working against the anti-collapse penalties.
  Measured effect on latent recovery was small (0.981 vs 0.980 diagonal-only),
  so this is documented rather than changed.
- **The optional NSA-Flow dependency had three different import names** across
  three modules (`import nsa`, `import nsa_flow as nsa`,
  `from nsa_flow import nsa_flow_orth`), so which retraction ran depended on
  which name happened to resolve — and it is not declared in `pyproject.toml`.
  Unified behind `pysimlr.nsa_backend`, with `backend_report()` for provenance.
  (NSA-Flow itself is correct: `retraction="polar"` orthogonalizes to ~5e-07,
  matching SVD. It preserves a uniform non-unit column scale, which is what the
  new normalization handles.)
- **Packaging**: the optional NSA-Flow backend is now declared as a `nsa`
  extra, and package discovery switched to `packages.find` so a future
  subpackage is included automatically. (I initially recorded the explicit
  `packages = ["pysimlr"]` list as a bug that dropped `pysimlr.benchmarks` from
  the wheel; building it both ways shows it did not — 29 files with benchmarks
  present either way. The change is robustness only.)
- **17 bare `except:` clauses** also swallowed `KeyboardInterrupt` and
  `SystemExit`, so Ctrl-C during a long fit landed in a fallback path instead of
  stopping the run. All narrowed to `except Exception`. The worst of them was in
  `ba_svd` and is listed above under finding 26.
- **Import cost 2.27 s**, because `__init__` called `matplotlib.use("Agg")` at
  module level and eagerly imported `visualization` and `benchmarks` (pulling in
  seaborn, scipy.stats, ipywidgets). Now **1.18 s**, with matplotlib, seaborn,
  sklearn and nsa_flow all deferred.
- **Docstrings that promised more than the code delivers** were corrected:
  `project_to_orthonormal_nonnegative` does not return an orthonormal matrix;
  `project_to_partially_orthonormal_nonnegative`'s `ortho_strength` controls the
  rate, not the final distance; `newton_schulz_orthogonalize` scales by the
  Frobenius rather than spectral norm and is percent-level off at its default 5
  iterations; `multiscale_svd` does no neighbourhood selection at `knn=0`;
  `sparse_distance_matrix` is dense and asymmetric.
- **119 blanket "audited for functional correctness" docstring blocks** were
  removed. They were attached to, among others, the orthogonality gradient, the
  whitening function, the indicator sparsifier and the permutation test — all
  found to be wrong here. A claim that is false for some members says nothing
  about the rest.
- **Four docstring examples were wrong and nothing ran them.** `rvcoef`'s used
  an unseeded `torch.randn`, so its expected `1.0` matched only when float32
  rounding cooperated; `map_asym_var` and `map_lr_average_var` printed float32
  digits for code that casts to float64; the smoothing-operator examples
  referenced undefined names. Fixed, and `tests/test_doctests.py` now runs every
  module's examples as part of the suite.

## The largest open question: how much the iteration contributes

This surfaced only once the objective was gauge-fixed (the unbounded-energy fix), and it is
pre-existing rather than caused by any change here.

Latent recovery (Procrustes R-squared against the true factors) from the SVD
initialization alone, versus after 40-60 iterations:

| regime | constraint | init | after simlr | gain |
|---|---|---|---|---|
| well-conditioned | `orthox0` | 0.9871 | 0.9873 | +0.0002 |
| well-conditioned | `Stiefel` | 0.9871 | 0.9872 | +0.0001 |
| well-conditioned | `ortho` (default weight) | 0.9871 | 0.9845 | **−0.0027** |
| shared signal buried under private structure | `orthox0` | 0.0760 | 0.0954 | +0.0194 |
| same, sparsification disabled | `orthox0` | 0.0970 | 0.0949 | −0.0021 |
| same, `energy_type="regression"`, no sparsity | `orthox0` | 0.0970 | 0.0970 | **+0.0000** |

Read carefully, this says two different things:

- **On well-conditioned data the iteration is indistinguishable from its
  initialization.** That is not by itself a defect. For a generator of the form
  `X_i = Z A_i + noise`, the leading right singular vectors of each view *are*
  the optimal basis, so the initialization is already at the optimum and there
  is nothing to gain. It does mean the benchmark generators used in the paper
  cannot demonstrate that the optimizer contributes anything, because they are
  all of that form.
- **On harder data the gain is real but small**: +0.0194 on a base of 0.0760, a
  roughly 25% relative improvement — while leaving absolute recovery poor
  (0.095). So the iteration does something; the question is whether it earns 40+
  iterations of compute.

Before the fixes here the well-conditioned case ended **0.0040 below** its own
initialization, so this is an improvement, not a regression.

The energy trajectory is consistent with a near-stationary start. On the
ill-conditioned stress input it now runs `-1.000, -0.933, … -0.765` over 12
iterations, i.e. it rises. The pre-fix code appeared to descend monotonically
only because `-sum|U'XV|` is unbounded below when the column scale is free, so
the apparent progress was `||V||` inflating toward 2.9e8; two stress tests
asserted that monotonicity and were passing on exactly that artifact. The
exactly-zero gain in the last table row is the signature of the new
best-iterate tracking returning iteration 0 when nothing improves.

**Two hypotheses tested and falsified**, recorded so they are not re-run:

1. *The search direction is retracted onto the constraint manifold, which
   orthonormalizes it and discards its magnitude.* Skipping that step entirely
   changes latent recovery by less than 0.0004 (−0.0012 vs −0.0012 easy,
   +0.0089 vs +0.0092 buried). Not the limiter.
2. *The step size is too small for the objective's scale* — plausible, since
   `scale_list=["centerAndScale","np"]` divides by `n*p` and leaves the `acc`
   energy around 1e-3. Raising `learning_rate` from 0.001 to 0.1 changes the
   gain by 0.001 (+0.0194 to +0.0203), and 10.0 makes it worse. LARS's trust
   ratio normalizes the gradient magnitude away, so the step is set by
   `lr * ||V||` regardless. Not the limiter either.

Still untested: whether recomputing `u` by SVD after each sweep undoes the
preceding V-steps. A fixed-`u` probe showed the V-subproblem itself moving the
energy by only 0.06% over 5 steps, which points away from this and toward the
initialization simply being near-stationary for the objective.

**Caveat on the numbers in this document.** Because the initialization dominates
latent recovery on these generators, differences of ±0.003 in the tables above
are not measuring optimizer quality. The large effects (0.88 → 0.98 for the sign
constraint; 2.9e8 → 1.2 for the energy) are real; the fine distinctions are not.

## Open, tracked as strict xfails

- Anchored prediction remains batch-composition dependent (finding 1).
- `test_simlr_iteration_improves_on_initialization`, plus the two
  `test_stress_simlr_*_correlation_ascent` monotonicity tests — see the section
  above.
- `test_ned_advantage_contract`: NED's latent recovery peaks near epoch 40 and
  then degrades as reconstruction keeps improving (0.752 → 0.659 from 40 to 200
  epochs). The contract previously passed only because training was truncated
  near that peak. Up-weighting the similarity term mitigates but does not remove
  it (a 20× weight recovers a third of the loss), so this needs
  validation-based model selection on a latent-quality criterion, which the
  trainers do not have — they early-stop on training loss only.

## Statistical re-analysis of the manuscript

See `paper/STATISTICAL_REANALYSIS.md` and
`paper/scripts/calc_stats_corrected.py`. In summary: the published p-values come
from a two-sample t-test over 80 rows per cell that contain only **5 seeds**,
the rest being repeated measures on the same splits. The design's resolution
floor is `2/2⁵ = 0.0625`. The deep-over-linear advantage on the real datasets is
real and consistent (5/5 seeds), but the **Polynomial** nonlinear regime — the
paper's central claim — does not survive seed-level inference (p = 0.12–0.16,
CIs spanning zero), and on CMC in the Linear regime the deep models are *worse*
on 0/5 seeds while the pooled test reports p = 1.7e-29.

## Test suite status and runtime

```
pytest                     # fast suite, the default
pytest -m slow             # only the heavy statistical/benchmark tests
pytest -m ""               # everything
```

`pytest.ini` sets `addopts = -m "not slow"`. The default suite is **161s for
640 tests** with no single test above 12s, down from a peak of 733s.

I initially attributed the suite's slowdown to the early-stopping fix (finding
2) making 29 test call sites train their full 150-epoch budget. That was wrong
on both counts: an AST walk shows **zero** trainer calls omit an explicit
`epochs=`, and my line-based grep had simply missed `epochs=` sitting on a
continuation line. Profiling gives the real distribution:

| test | before | mine? |
|---|---|---|
| `test_null_p_values_are_calibrated_at_the_nominal_level` | **250s (34% of the suite)** | yes |
| `test_positive_constraint_does_not_cripple_latent_recovery` x3 | 83s | yes |
| `test_ned_advantage_contract` | 45s | pre-existing |
| `test_benchmark_linear_noise` x2 | 70s | pre-existing |
| `test_flow_whiteners_adversarial` | 35s | pre-existing |
| `test_nedpp_advantage_contract` | 27s | pre-existing |

So the new tests were the majority of the cost, not the source fix. Each heavy
test is now behind the `slow` marker with a fast counterpart that guards the
same invariant, and the multi-seed benchmark contracts are marked `slow`
because they are benchmarks rather than unit tests. Iteration counts in the
remaining fast tests were also halved where the assertion did not depend on
them.

One of my own screens was also wrong rather than merely slow: with
`n_perms=9` the attainable p-value floor *is* 0.1, so asserting
`sum(p <= 0.1) <= 3` counted correct minimum p-values as false positives. The
rejection count has to be taken at the floor exactly.

## New test files

`test_consensus_anchor_standardization.py`, `test_deep_early_stopping.py`,
`test_sparsification_correctness.py`, `test_orthogonality_gradients.py`,
`test_whitening.py`, `test_permutation_pvalues.py`,
`test_positivity_and_gradient_consistency.py`,
`test_preprocess_provenance_strict.py`, `test_graph_laplacian_correctness.py`,
`test_optimizer_correctness.py`, `test_interpretability_validity.py`,
`test_package_import.py`, `test_rng_isolation.py`, `test_doctests.py`.

These assert numerical properties — gradients against autograd, whitened
covariance against the identity, p-value calibration under a true null,
train/eval equivalence, scale invariance — rather than dict keys and shapes. The
pre-existing suite had 1110 assertions of which 623 were structural and only 61
checked numerical equivalence, which is how these defects survived.
