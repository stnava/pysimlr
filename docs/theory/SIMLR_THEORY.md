# SiMLR: the method, stated

A SiMLR fit is determined by **three choices**:

| | choice | what it controls |
|---|---|---|
| 1 | **energy** $E$ | what "aligned" means between a view and the shared latent |
| 2 | **mixing** $\mathcal{M}$ | how the per-view projections become one shared latent $u$ |
| 3 | **weight** $w$ | where the basis sits between "just non-negative" and "disjoint parts" |

plus the invocations every method needs: the **data** $\{X_i\}$, the **basis size** $k$,
the **initialisation**, and the **optimiser**.

Everything else that used to be a knob — `sparseness_quantile`, `sparseness_alg`,
`constraint_iterations`, `orth_weights`, and the choice among five different
orthogonality functionals — is either a consequence of $w$ or was never
well-defined. See §6.

---

## 1. The problem

Views $X_i \in \mathbb{R}^{n \times p_i}$, $i = 1\dots m$, on shared rows. Find bases
$V_i \in \mathbb{R}^{p_i \times k}$ whose projections $P_i = X_i V_i$ agree with a
single shared latent $u \in \mathbb{R}^{n \times k}$, subject to $V_i$ lying in a
structured set $\mathcal{C}_w$.

$$
\min_{\{V_i\} \subset \mathcal{C}_w} \; \sum_{i=1}^{m} \eta_i \, E(V_i;\, X_i,\, u),
\qquad u = \mathcal{M}\big(\{X_i V_i\}\big)
$$

$u$ is **not** a free variable. It is a deterministic function of the current
projections, fixed by the mixing method. So the method is block coordinate
descent in $\{V_i\}$ with $u$ recomputed between sweeps, and it is *not*
monotone in the joint objective: each $V_i$ is optimised against a fixed $u$,
then $u$ moves.

$\eta_i$ normalises the per-view energies to comparable scale at iteration 0.

---

## 2. Choice 1 — the energy

$E$ says what agreement between $X_iV_i$ and $u$ means. Each is paired with a
gradient; all are verified against finite differences at $\cos = 1.000000$.

Write $P = XV$, $a = \lVert u \rVert_F$, $b = \lVert P \rVert_F$, $s = \langle u, P\rangle$.

| energy | $E(V)$ | descent direction $-\nabla_V E$ | homog. degree | gauge |
|---|---|---|---|---|
| `regression` | $\lVert X - uV^\top\rVert_F^2$ | $2\big(X^\top u - V u^\top u\big)$ | 1.87 | **free** |
| `acc` | $-\sum_{jl}\lvert C_{jl}\rvert$, $C = \tfrac{u^\top XV}{n-1}$ | $\tfrac{1}{n-1}X^\top u\,\operatorname{sign}(C)$ | 1.00 | **fixed** |
| `nc` | $-s/(ab)$ | $X^\top\!\big[\tfrac{u}{ab} - \tfrac{sP}{ab^3}\big]$ | 0.00 | fixed (free) |
| `logcosh`/`exp`/`gauss`/`kurtosis` | negentropy contrast | FastICA-style | 1.0–4.0 | **fixed** |
| `dat` | $-\lambda\lVert DV\rVert_F^2$ | $2\lambda D^\top D V$ | 2 | regulariser only |

Three properties of this table are load-bearing.

**`regression` has a closed-form block minimiser.** Setting the gradient to
zero gives

$$V^\star = X^\top u\,(u^\top u)^{-1}$$

— ordinary least squares of $X$ on $u$. The Hessian in $\operatorname{vec}(V)$ is
$2(u^\top u)\otimes I$, so $u^\top u$ *is* the curvature of the block problem, and
the gradient step with a $(u^\top u)^{-1}$ metric is the exact Newton step.

This matters because $u^\top u \neq I$. Every mixing method returns
variance-standardised scores, so $u^\top u = (n-1)I$ — measured at $279\,I$ on a
3-view case with $n=280$. A gradient that omits it is not a rescaled descent
direction; it has deleted the curvature. (The shipped gradient did omit it, and
measured $\cos = -0.0045$ against true descent. Fixed; see
`docs/audit/AUDIT_2026_09.md` A1.)

**Gauge.** $\lvert$degree$\rvert > 0$ with $E \to -\infty$ under $V \to cV$ means the
objective is unbounded below and the column scale must be pinned, or the
optimiser "improves" forever by inflating $\lVert V\rVert$. `regression` is the
exception: it has a finite minimiser whose column norms are set by the data, so
renormalising it every sweep deletes the very scale it is solving for.
`sparsification.GAUGE_FREE_ENERGIES` encodes this; it is the only correct
default and cannot be chosen per-call by taste.

**`nc` is degree-0.** By Euler's theorem $\langle\nabla E, V\rangle = 0$: the
gradient is everywhere orthogonal to $V$, and the objective is a function on the
sphere quotient, not on $\mathbb{R}^{p\times k}$. Radial motion leaves it exactly
invariant. A reported energy reduction of 0 for `nc` is therefore not evidence
that nothing happened.

Objectives with no gradient are **refused**, not silently evaluated as zero
(`SUPPORTED_ENERGY_TYPES`).

---

## 3. Choice 2 — the mixing method

$u = \mathcal{M}(\{P_i\})$, with $\mathcal{M} \in \{$`svd`, `pca`, `ica`, `newton`,
`avg`$\}$ and an optional `topology` (`star`, path graph) controlling which views
are compared.

The mixing method **entirely determines $u$**, and $u$ is what every energy is
measured against. It therefore selects the notion of "shared": `svd`/`pca` take
the dominant common subspace of the projections, `ica` a maximally non-Gaussian
one, `avg` the plain mean, `newton` a quasi-Newton consensus.

Two consequences that are easy to miss:

- Because $u$ enters the energy but is not differentiated through, changing
  $\mathcal{M}$ changes the objective the block step is solving even when the
  gradient formula is unchanged. Comparisons across mixing methods are
  comparisons across *different objectives*, not across solvers.
- All mixings return $u$ with standardised columns, hence $u^\top u = (n-1)I$.
  Anything derived assuming orthonormal $u$ is wrong by $n-1$.

---

## 4. Choice 3 — the weight $w$

$\mathcal{C}_w$ is defined by NSA-Flow, and $w$ is a genuine convex weight:

$$
\operatorname{proj}_w(V_0) \;=\; \arg\min_{Y \ge 0}\;
(1-w)\frac{\lVert Y - V_0\rVert_F^2}{\lVert V_0\rVert_F^2} \;+\; w\,\tilde{D}(Y)
$$

- $w = 0$ gives $\max(0, V_0)$ — non-negativity and nothing else.
- $w = 1$ gives orthogonal columns which, under non-negativity, means
  **disjoint supports**: a hard clustering of the $p$ features.
- Intermediate $w$ interpolates. Sparsity is a *consequence* of $w$, measured
  monotone: $w = 0.1 \to 0.356$, $w = 0.5 \to 0.467$, $w = 0.9 \to 0.578$ zero
  fraction.

This is why there is no separate sparsity knob. A quantile threshold applied
after the projection pulls against the orthogonality the projection just
produced, and nothing reconciles them; one weight cannot disagree with itself.

`simlr_sparseness` is exactly this call, plus two pieces of input preparation:
the optional smoothing prior $S V$ (a prior on the basis, not part of
$\mathcal{C}_w$), and the choice of what candidate to hand the solver — the
signed iterate carries sign information the sign-blind subspace fidelity is
built to consume.

**There is no fallback.** A missing backend raises `ImportError`; an unusable
projection raises `RuntimeError`. Substituting an SVD polar factor projects onto
a *different* set, which made the same call run a different algorithm depending
on the install.

---

## 5. The algorithm

```
V_i  <- initialise(X_i, k)                    # per-view truncated SVD, or
                                              # fitted from data when non-negative
repeat:
    P_i <- X_i V_i
    u   <- M({P_i})                           # choice 2
    for each view i:
        g   <- -grad E(V_i; X_i, u)           # choice 1
        V_i <- proj_w( V_i + t g )            # choice 3
    certificate <- max_i gradmap(V_i, g_i)
until certificate <= tol, or plateau, or max_iter
```

This is **proximal gradient**: $E$ is differentiated, $\Omega$ never is —
$\Omega$ is the function whose proximal operator is $\operatorname{proj}_w$.
That is what makes the pair consistent by construction. Two obligations follow:

1. **The certificate must use the same operator as the step**, i.e. the
   prox-gradient mapping $\tfrac{1}{t}\lVert V - \operatorname{proj}_w(V - t\nabla E)\rVert$.
   Scoring with a plain non-negative projection certifies a different problem.
2. **The reported energy is the composite** $F = E + \lambda(w)\Omega$. Proximal
   gradient decreases $F$, not $E$; reporting $E$ alone tracks a quantity
   nobody is minimising.

Because $\operatorname{proj}_w$ normalises by $\lVert V_0\rVert^2$, the effective
$\lambda$ depends on the point, making it an *inexact / variable-metric* prox.
Standard prox-gradient rates do not apply; stationarity claims should stay
modest.

---

## 6. What is no longer a choice

| removed | why |
|---|---|
| `sparseness_quantile`, `sparseness_alg` | sparsity is a consequence of $w$ |
| `constraint_iterations` | the solver owns its own iteration budget |
| 5 orthogonality functionals in `utils` | they disagreed by 35.47 on a perfect disjoint non-negative basis; $\Omega$ is now whichever one `proj_w` uses |
| `orth_weights` penalty term | $\Omega$ is in the projection, not the energy |
| SVD-polar fallback | a different projection is a different method |
| `cca`/`pca`/`ica` as `energy_type` | never had an energy or a gradient |

**Recommended $\Omega$: $C_g$ (energy-weighted angle defect), not $D$.** $D$
charges $\sum_i (G_{ii} - 1/k)^2$, penalising components with unequal explained
variance — it scores a *perfect* disjoint non-negative basis at 35.47 where
$C_g$ scores 0.000. For multi-omic parts with heterogeneous variance that term
fights the science.

---

## 7. Status

Implemented: corrected gradients (all $\cos = 1.0$), refusal of undefined
energies, energy/gradient consistency, energy-dependent gauge, single projection
via NSA-Flow, degeneracy rejection, stationarity certificate and
`energy_reduction` reporting.

Not yet done: autograd for $E$ (currently hand-written but verified);
prox-gradient certificate using $\operatorname{proj}_w$ rather than
$\operatorname{proj}_{\ge 0}$; reporting the composite $F$ rather than $E$;
selecting $C_g$ as the default $\Omega$.

---

## 8. Is the retraction a prox? (analysis 2026-09-19; fixed in pysimlr 0.2.12 / nsa_flow 3.1.1)

The outer loop is proximal-gradient in form: `v_updated = optimizer.step(...)`
then `simlr_sparseness(v_updated, ...)`, i.e. the projection is applied to the
**post-gradient-step point** — verified at `simlr.py:962-966`. For this to
converge to a stationary point of `E + R` on `V >= 0`, the operator must be the
Euclidean prox

    argmin_{Y>=0}  1/2 ||Y - (V - eta grad E)||^2  +  lambda Dtilde(Y)

which is NSA-Flow's anchored energy with `fidelity="anchor"`.

**It is not one, but not for the reason first proposed.** The proposed
mechanism was that pysimlr hands the solver a *signed* iterate, `fidelity="auto"`
sees negative mass above `neg_mass_tol=0.01`, and switches to the subspace
fidelity `||(I-P)Y||^2/||Y||^2`, which is invariant under `X0 -> X0 M` and so is
the prox of nothing. That is a correct description of the *intended* path.

Measured on the installed backend (nsa_flow 3.0.0), the actual path is:

    signed gradient-step point      neg_mass=0.6050   fidelity_mode=subspace
    rectified (what we pass today)  neg_mass=0.0000   fidelity_mode=anchor

pysimlr passes the **rectified** candidate and therefore gets `anchor`. The
cause is a version-detection bug: `_backend_has_sign_blind_fidelity()` tests
`'fidelity' in inspect.signature(nsa_flow).parameters`, but 3.0 moved
`fidelity` into `**kwargs`, so the check returns False and
`_retraction_candidate` silently falls back to the rectified branch. The signed
path that its own measurements motivated (0.9930 vs 0.9162 recovery) is dead
code on this install.

So the operator applied is `prox(clamp(z))`, not `prox(z)`. Clamping the prox
point changes the prox, so the conclusion stands — the composition is not a
proximal-gradient method — by the *second* footnote's mechanism rather than the
first's. Both must be fixed; fixing only the detection would move the failure
from "prox of the wrong point" to "not a prox at all".

### The unresolved part of option 1

"Solve `E_SiMLR(V_1..V_m) + sum_i w_i Dtilde(V_i)` as one problem" presumes `E`
is a function of the `V_i` alone. It is not: `E` depends on them directly *and*
through `u = M({X_i V_i})`, and the shipped gradient treats `u` as a constant.
Two distinct proposals hide here:

* **1a — joint over views, `u` frozen per outer sweep.** Well defined today.
  One `minimise` call over the concatenated `V_i` with `proj=project_nonneg`,
  `optimizer="lbfgsb"`, penalty from `ORTH`. Strictly better than the current
  per-view block step, and it yields one certificate per sweep. Still
  alternating in `u`.
* **1b — fully joint, differentiating through the mixing.** Only this gives
  "one objective, one certificate" in the strong sense. Measured
  differentiability of `compute_shared_consensus`:

      avg  svd  pca  newton   differentiable
      ica                     NOT differentiable (detaches)

  and `svd`/`pca` carry the usual unstable backward at repeated singular
  values, which is exactly the isotropic-initialisation regime. So 1b is
  unavailable for `ica` mixing and fragile for two others.

Recommendation: **1a as the default**, 1b opt-in for mixings that support it.
The three concrete defects are orthogonal to that choice and should be fixed
either way: the fidelity detection, the clamp before the prox, and emulating
`nonneg` by rectifying the input rather than passing the feasible set.

### Footnotes, checked

* The unit-RMS rescale in `_nsa_retract` **is** a no-op under anchor fidelity.
  Measured scale-equivariance of the anchored solve: relative deviation of
  `Y(cV)/c` from `Y(V)` is 8.5e-07, 0, 4.0e-16 at `c = 0.1, 1, 10`.
* `nonneg` should be the feasible set passed to the solver, never emulated by
  rectifying the input — which is exactly the clamp that breaks the prox.

### What was done (2026-09-19)

Both sides fixed; the retraction **is now the prox** `argmin_{Y>=0} (1-w)||Y-z||^2/||z||^2 + w Dtilde(Y)`
of the signed post-step point `z`.

nsa_flow 3.1.1
: `fidelity` is an explicit keyword of `nsa_flow` again (anchored mode only,
  `ValueError` otherwise) and pinned by a test on
  `inspect.signature`, so the signature probe cannot silently go False again.

pysimlr 0.2.12 (`sparsification.py`)
: * `_nsa_retract` calls `nsa_flow(z, w, mode="anchored", fidelity="anchor",
    nonneg=..., max_iter=...)`. `"auto"` is never used inside the loop: it is a
    threshold on the iterate's negative mass and can flip the operator between
    iterations, and the subspace term it selects is not a prox.
  * `simlr_sparseness` hands the **signed** iterate to the prox when
    `positivity` is non-negative; the pre-clamp is gone. `negative` still
    solves the reflected problem and negates.
  * Before the prox, the **per-column sign gauge is resolved**
    (`_resolve_column_sign_gauge`): column `j` is negated iff
    `||max(0,-z_j)|| > ||max(0,z_j)||`. Every SiMLR energy is even in each
    column of `V` (they act on `X V` through squares, covariances, or a
    regression whose coefficient absorbs the sign), so `V -> V diag(s)` is a
    symmetry of `E`; choosing `s` is a gauge choice, not a move of the iterate,
    and prox(z diag(s)) is still the proximal step -- on the quotient by that
    symmetry. It exists for the case that broke the old code's replacement,
    a 5x5 orthogonal `Q` with one entirely non-positive column: the honest
    prox zeroes that column (correctly -- but for a reason that was only a
    sign convention), and the old `apply_positivity("positive")` fell back to
    `abs(v)`, which is a symmetry of nothing. Mixed-sign columns keep their
    orientation and are projected honestly. Tested: negating any input column
    leaves the output unchanged; an all-negative column is flipped, not zeroed
    and not reflected.
  * Assumption stated: column-sign evenness of `E`. It holds for every shipped
    `energy_type`; a future energy that is odd in a column would need this
    step disabled.
  * `_usable_retraction` judges rank loss against `clamp(z)` when `nonneg`,
    i.e. against what the feasible set can support: the prox of a full-rank
    signed square matrix is legitimately rank-deficient (a random orthogonal
    5x5 clamps to rank 4), and the old check only passed because it compared
    against the pre-clamped candidate.
  * Diagnostics carry `certificate`, `n_grad`, `defect_D` (the functional
    comparable to PCA), `energy_reduction`; `_warn_if_unconverged` uses
    nsa_flow >= 3 semantics (`converged` requires a certificate).
  * `_retraction_candidate` / `_backend_has_sign_blind_fidelity` are retained
    for import compatibility and no longer consulted.
  * `tests/test_prox_contract.py` pins all of it: the output equals the
    directly-computed anchored prox of the signed `z`; `prox(z) != prox(clamp(z))`
    (so the fix is not vacuous); `fidelity_mode == "anchor"` across iterates
    with negative mass ~0.5; the certificate is present; `negative` is the
    reflected prox.

Not done
: **1a** (joint over views, `u` frozen per sweep). The per-view proximal
  step is now correct; 1a replaces it with one L-BFGS-B solve over the
  concatenated `V_i` per sweep via `nsa_flow.optim.minimise`. Its convergence
  argument is block coordinate descent with an exact inner solve on the
  `V`-block and closed-form `u`; report the `V`-block certificate plus the
  `u`-residual for an honest joint criterion. **1b** stays opt-in; for
  `svd`/`pca` mixings written as a polar factor, `nsa_flow.polar_factor`'s
  Sylvester backward (`h_i + h_j` denominators) is the stable differentiable
  path at repeated singular values; `ica` stays out.

### Consistency pass across methods (pysimlr 0.2.13)

Each method was stepped through in order of complexity and made consistent
with the shared prox of §8.

`simlr` (linear)
: The search *direction* is no longer passed through `simlr_sparseness`
  (an extra solve per view per sweep that CORRECTNESS_AUDIT.md had measured
  at < 0.0004 effect and that is not part of proximal gradient). One weight:
  `nsa_w=None` now defaults to the prox weight parsed from `constraint`, so
  the `"nsa_flow"` optimizer's intermediate retraction and the prox agree
  unless the caller separates them. `sparseness_quantile` defaults to its
  no-op value `0.0` (the old `0.5` fired the deprecation path three times per
  sweep). A view whose certificate is `<= tol` takes no step and is not
  re-projected after the first sweep.

`NSAFlowOptimizer`
: Its intermediate retraction is the same operator family in sign-free form,
  `nsa_flow(z, w, mode="anchored", fidelity="anchor", nonneg=False)`, on the
  default solver. It had been requesting the deprecated `torch_lbfgs` with a
  `max_iter=5` fallback.

`lend_simr`, `ned_simr`, `ned_simr_shared_private`, `flow_simr`
: The encoders keep their training-time surrogate (`NSAFlowLinear` blend +
  clamp + normalise; routing every access through the solver was 26x per
  access). The **returned** bases are now projected once with the shared prox
  (`_finalize_bases`, `unit_columns=True` to keep the encoders' gauge), first-
  layer scores are recomputed as `X @ V` from the projected basis so the
  contract is exact, and `result["retraction_diagnostics"]` carries the
  certificate per view. Tested: the returned bases are fixed points of the
  prox.

sklearn wrappers
: `nsa_w` defaults match the wrapped functions (`None` -> prox weight for
  `SiMLREstimator`; `0.1` for LEND/NED/Flow, which had silently been `0.5`);
  `sparseness_quantile` defaults to `0.0`.
