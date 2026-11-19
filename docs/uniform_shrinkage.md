# Uniform Shrinkage for Directional Multi-View SiMLR (`pysimlr.uniform_shrinkage`)

```python
import pysimlr

# 1. Orient views so that higher means the same thing on every feature of a view (your responsibility)
prep = pysimlr.prepare_views(raw_views, fit_rows=train_mask, signs=signs, nuisance=nuis)

# 2. Build fit keyword arguments; alpha_m = clip(p_m / (n_m + p_m), 0.05, 0.85)
params = pysimlr.uniform_shrinkage_params(prep["views"], alpha="auto", sparseness=None)
print(params.provenance)           # resolved alpha, n_obs, nsa_w, scale_list, energy_type, ...

# 3. Fit and score
fit_res = pysimlr.fit("simlr", prep["views"], k=1, random_state=0, **params)
proj = pysimlr.project_uniform_shrunk(fit_res, test_views)
u_scores, view_scores = proj["u"], proj["view_scores"]
```

## Idea

If every feature of view $m$ is oriented to a shared direction, the plain view mean $p_m^{-1}\mathbf{1}^\top z$ has zero estimation variance but no feature specificity. An unregularized non-negative weight vector $\mathbf{v}_m$ has estimation variance of order $p_m/n_m$. When $n_m \lesssim p_m$ the fit can trail the plain mean out of sample. The shrinkage step pulls each solver iterate toward the uniform direction by an amount $\alpha_m$.

## What the code does

**Fit step.** For view $m$ and a column $v$ of the current iterate, let $c = \max(v, 0)$ (or $c = \max(-v, 0)$ if $v$ has no positive entry; only with `positive_part=True`). The operator returns

$$S'_m(v) = (1-\alpha_m)\,c + \alpha_m\,\operatorname{mean}(c),$$

with the mean taken per group when `groups=` is given (labels `-1`, `None`, NaN are left unchanged). With `positive_part=False` it is the linear map $\mathbf{S}_m(\alpha_m) = (1-\alpha_m)\mathbf{I} + \alpha_m\mathbf{\Pi}_m$, $\mathbf{\Pi}_m = p_m^{-1}\mathbf{1}\mathbf{1}^\top$ (block-wise for groups).

**Schedule.** $\alpha_m = \operatorname{clip}\bigl(c\,p_m/(n_m + c\,p_m),\ 0.05,\ 0.85\bigr)$ with `prior_scale` $c = 1$ by default. $n_m$ is the number of rows of view $m$ with at least one finite feature, which is the set of rows `missing_rows="mask"` keeps (rows that are entirely NaN do not count; scattered NaNs do not reduce $n_m$).

**Projection (`project_uniform_shrunk`).** For each column of the fitted basis, $w = (1-\gamma)\,\operatorname{simplex}(\max(v,0)) + \gamma/p_m$ (flipped to $\max(-v,0)$ when $\sum v < 0$; uniform if the positive part is zero). The view score is $Z_m w_m$ (`view_scaling="natural"`); the consensus is $u = \operatorname{nanmean}_m (Z_m w_m)$. With `post_shrink="auto"`, $\gamma = \operatorname{clip}(1 - n_{\text{tr}}/(1.5\,p_{\max}),\ 0,\ 0.6)$, where $n_{\text{tr}}$ is the largest per-view count of rows with a finite feature and $p_{\max}$ the widest view. Rows with any NaN in a view get a NaN score for that view.

### Relation to the rest of pysimlr

* $\mathbf{S}_m(\alpha) = (\mathbf{I} + \lambda\mathbf{L})^{-1}$ with $\mathbf{L} = \mathbf{I} - \mathbf{\Pi}$ the Laplacian of the complete graph (block-diagonal complete graph with `groups=`), $\lambda = \alpha/(1-\alpha)$. It is the closed-form special case of `SparseGraphResolvent`, in $O(p_m k)$ time without forming a matrix. `alpha` and `lambda_val` are two parametrizations of one knob: $\alpha = \lambda/(1+\lambda)$.
* **One step, not a projector.** $\mathbf{S}$ is not idempotent: $\mathbf{S}(\mathbf{S}v) = (1-\alpha)^2 v + (1-(1-\alpha)^2)\mathbf{\Pi}v$. The operator is stateless. The solver applies it once per step: `pysimlr.sparsification._apply_smoothing` recognises the solver's own projected output (registered as its own image) and does not smooth it again. Removing the operator's former internal cache left fitted weights and out-of-sample correlations unchanged (max abs difference 0.0 on a planted $n < p$ example, `alpha="auto"` and `alpha=0.5`).
* **Limits.** $\alpha = 1$: the fitted direction is uniform and the score equals the plain view mean (true by construction; tested). $\alpha = 0$: the smoother is the identity, but the result does **not** reduce exactly to a plain `simlr` fit: `uniform_shrinkage_params` uses a different `initial_v`, `nsa_w` defaults to 0.2 when sparseness is requested, and `positive_part` clips the iterate. Fitted weights have cosine about 0.99 with the plain fit in the tested example (test threshold 0.95).
* **Sign orientation is a user contract.** Shrinking toward $+p_m^{-1}\mathbf{1}$ assumes $+1$ means the same on every feature. If a view's signs are flipped its score is negated; nothing checks this.
* **$k > 1$.** `uniform_shrinkage_params(k>1)` raises `ValueError` unless `groups=` is given for every smoothed view and each such $\alpha \le 0.15$ (`MAX_ALPHA_MULTI_COMPONENT`). Pulling several columns toward the same uniform vector made the two components of a $k=2$ fit nearly identical (correlation 0.84 in an unrestricted run).
* **Explicit defaults.** `uniform_shrinkage_params` exposes `scale_list=("none",)`, `energy_type="recon_r2"`, `mixing_algorithm="svd"`, `sparseness=0.3`, `sparseness_type="soft"`, `missing_rows="mask"` and `nsa_w="auto"` (0.2 if sparseness is requested, otherwise 0.0; pass a float to override). The returned dict carries the resolved values in its `.provenance` attribute (a dict subclass; `**params` passes only fit arguments).

## Synthetic benchmark

Planted model: one latent $z$; every feature loads positively (background 0.10, first fifth of each view 0.20), unit Gaussian noise, features standardized on the training rows, three views. Score: absolute correlation between the consensus score and the planted $z$ on 400 held-out rows, mean over 20 seeds. Both fitted arms use the same sparseness and `project_uniform_shrunk(post_shrink=0)`, so the only difference is the smoother; differences are paired by seed with a percentile bootstrap (2,000 resamples) over seeds. Produced by `python scripts/uniform_shrinkage_benchmark.py --n-seeds 20 --n-boot 2000` (about 30 s; `--smoke` runs every stage with counts of 2).

| regime | n_train | widths | sparseness | plain mean | unshrunk | shrunk | shrunk - unshrunk [95% CI] | frac seeds better | shrunk - plain mean [95% CI] |
| :-- | --: | :-- | :-- | --: | --: | --: | :-- | --: | :-- |
| `n_lt_p` | 40 | [100, 120, 100] | None | 0.901 | 0.848 | 0.899 | +0.051 [+0.046, +0.055] | 1.00 | -0.003 [-0.004, -0.001] |
| `n_lt_p_sparse0.3` | 40 | [100, 120, 100] | 0.3 | 0.901 | 0.849 | 0.842 | -0.006 [-0.011, -0.002] | 0.20 | -0.059 [-0.065, -0.053] |
| `n_gg_p` | 600 | [20, 25, 15] | None | 0.666 | 0.644 | 0.646 | +0.002 [-0.000, +0.004] | 0.70 | -0.020 [-0.027, -0.014] |

Reading the table:

* With $n < p$ and no sparseness, shrinkage recovers most of the gap between the unshrunk fit and the plain mean (+0.051), in every seed, and ends within 0.003 of the plain mean (which is close to optimal here because every feature carries signal).
* With `sparseness=0.3` the smoother gives no gain at $n < p$ (slightly worse, -0.006) and both fits trail the plain mean by about 0.06. The sparsity penalty already controls the variance, and shrinkage then adds bias. Prefer `sparseness=None` with this module, or accept no benefit.
* With $n \gg p$ ($\alpha$ near 0.05) the difference is within noise (+0.002, CI includes 0).
* These are one planted model with positive loadings on every feature. They show where the operator can help, not that it helps on any given dataset; check held-out performance against the plain mean.

## Should it be default ON in `pysimlr.simlr`?

No. It stays opt-in. It needs shared coordinate polarity (see the contract above), it is restricted to $k=1$ (or small-$\alpha$ grouped $k>1$), and it adds nothing when $n \gg p$. On unoriented features the shrinkage destroys the signal (`tests/test_uniform_shrinkage.py::test_mixed_sign_planted_signal_shows_why_default_is_off`).

| Regime | Feature polarity | $k$ | $n/p_{\max}$ | Recommendation |
| :-- | :-- | :-: | :-: | :-- |
| Oriented views | aligned | 1 | $n \lesssim 2.5\,p_{\max}$ | try it, `sparseness=None`, compare with the plain mean on held-out rows |
| Oriented views | aligned | 1 | $n > 5\,p_{\max}$ | optional ($\alpha \le 0.15$, little effect) |
| Unknown or mixed signs | mixed | any | any | keep off (`positivity="either"`, or `SparseGraphResolvent` over a graph) |
| Multi-component | either | $\ge 2$ | any | off, or `groups=` with $\alpha \le 0.15$ |
