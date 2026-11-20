# Uniform and Support-Restricted Shrinkage (`pysimlr.uniform_shrinkage`)

```python
import pysimlr

# 1. Orient views so higher means the same direction on every feature (user contract)
prep = pysimlr.prepare_views(raw_views, fit_rows=train_mask, signs=signs, nuisance=nuis)

# 2. Fit directional uniform-shrunk (k=1) or two-stage support-restricted (k>=1, sparse or k>1)
#    Defaults: positivity="auto" ("conical" for k=1, radial "convex" for k>1),
#    energy_type="auto" ("nc" for k=1, "acc" for k>1),
#    mixing_algorithm="auto" ("pca"), alpha_mode="ric".
fit_k1 = pysimlr.fit_uniform_shrunk(prep["views"], k=1, random_state=0)
fit_k2 = pysimlr.fit_uniform_shrunk(prep["views"], k=2, sparseness=0.3, random_state=0)

# 3. Project new rows with support-restricted simplex blending (exact zeros preserved)
proj = pysimlr.project_uniform_shrunk(fit_k2, test_views, train_views=prep["views"])
u_scores, view_scores = proj["u"], proj["view_scores"]
```

## Idea

If every feature of view $m$ is oriented to a shared direction, the plain view mean $p_m^{-1}\mathbf{1}^\top z$ has zero estimation variance but no feature specificity. An unregularized non-negative weight vector $\mathbf{v}_m$ has estimation variance of order $p_m/n_m$. When $n_m \lesssim p_m$ the unregularized fit can trail the plain mean out of sample.

Combining global shrinkage with subtractive sparsity or multi-component orthogonality introduces three structural conflicts:

1. **Order-statistic soft-thresholding cancels global shrinkage.** If $\tilde{v} = (1-\alpha)v + \alpha\bar{v}\mathbf{1}$ is soft-thresholded at $\tau = \tilde{v}_{(m+1)} = (1-\alpha)v_{(m+1)} + \alpha\bar{v}$, the uniform floor cancels identically: $(\tilde{v} - \tau)_+ / \|(\tilde{v} - \tau)_+\|_2 = (v - v_{(m+1)})_+ / \|(v - v_{(m+1)})_+\|_2$ for all $\alpha \in [0, 1)$.
2. **Non-subtractive support restriction and radial simplex normalization.** The exact Euclidean and cosine projection of $y \in \mathbb{R}^p$ onto $\mathcal{C}_m = \{u \ge 0 : \|u\|_0 \le m, \|u\|_2 = 1\}$ is non-subtractive top-$m$ restriction $(y_+)_{S^*} / \|(y_+)_{S^*}\|_2$. Likewise, `positivity="convex"` applies radial simplex normalization $v_+ / \mathbf{1}^\top v_+$, preserving every positive coordinate rather than subtracting a Euclidean simplex threshold $\theta > 0$.
3. **Global shrinkage breaks $k \ge 2$ orthogonality.** For $V \in \mathbb{R}_+^{p \times k}$, $V^\top V = I_k$ holds if and only if the columns have unit $\ell_2$ norm and pairwise-disjoint supports $S_c \cap S_{c'} = \emptyset$. Pulling all columns toward a single global vector $\mathbf{1}_p$ forces them to collapse as $\alpha \to 1$.

## Two-Stage Support-Restricted RIC Estimator (`fit_uniform_shrunk`)

**Column-wise support-restricted smoother (`ColumnSupportShrinkageOperator`).** Given active supports $S_{m,1}, \dots, S_{m,k} \subseteq \{0, \dots, p_m - 1\}$ and shrinkage weights $\alpha_{m,1}, \dots, \alpha_{m,k} \in [0, 1]$, each call zeroes column $c$ outside $S_{m,c}$, takes the positive part on $S_{m,c}$, and applies

$$(S_{S_{m,c}} v_{m,c})_{S_{m,c}} = (1 - \alpha_{m,c})\,(v_{m,c})_{S_{m,c}} + \alpha_{m,c}\,\bar{v}_{S_{m,c}}\mathbf{1}_{S_{m,c}}.$$

Whenever $S_{m,1}, \dots, S_{m,k}$ are pairwise disjoint, $(S_{S_{m,c}} v_{m,c})^\top (S_{S_{m,c'}} v_{m,c'}) = 0$ identically for all $c \ne c'$ and every $\alpha_{m,c} \in [0, 1]$.

**Post-selection RIC and Jackknife James–Stein shrinkage (`alpha_mode`).** For support size $m_c = |S_{m,c}| \in [1, p_m]$:

* `alpha_mode="ric"` (default, Donoho–Johnstone / Birgé–Massart $\ell_0$ selection complexity): $d_{\text{eff}}(m_c, p_m) = m_c\bigl(1 + 2\log(p_m / m_c)\bigr)$, which equals $p_m$ identically when $m_c = p_m$.
* `alpha_mode="plugin"`: $d_{\text{eff}}(m_c, p_m) = m_c$.
* `alpha_mode="auto"` (or `"jk_js"`, parameter-free Delete-1 Jackknife Positive-Part James–Stein via `jackknife_js_alpha`): computes $\alpha_{m,c} = \min\bigl(1, \hat{\sigma}_{\text{JK}}^2 / \max(D^2, 10^{-14})\bigr) \in [0, 1]$ from leave-one-row-out non-negative marginal vectors without heuristic clip constants, and sets `post_shrink=0.0` automatically in `project_uniform_shrunk(..., post_shrink="auto")`. `sparseness="auto"` is also accepted on `fit_uniform_shrunk`.

The shrinkage schedule sets

$$\alpha_{m,c} = \operatorname{clip}\!\left(\frac{c\,d_{\text{eff}}(m_c, p_m)}{n_{\text{obs},m} + c\,d_{\text{eff}}(m_c, p_m)},\ 0.05,\ 0.85\right),$$

where $n_{\text{obs},m}$ is the count of rows in view $m$ with at least one finite feature.

**Stages:**

* **Single-stage dense ($k = 1$, `sparseness=None`):** fits `simlr` directly with `UniformShrinkageOperator` on all $p_m$ features and `nsa_w=0.0`.
* **Two-stage support-restricted ($k > 1$ or `sparseness > 0`):**
  1. *Stage 1 (basin & lobe discovery):* dense shrunk fit for $k = 1$; NSA-Flow fit (`nsa_w=stage1_nsa_w`, `initialization_type="nndsvd"`, falling back with a logged warning to `"pca"` on rank-deficient inputs) for $k > 1$.
  2. *Support extraction (`extract_disjoint_supports`):* for $k = 1$, keeps top $\max(1, \operatorname{round}((1-q)p_m))$; for $k \ge 2$, assigns each feature to $\arg\max_c (V^{(1)}_{m,i,c})_+$ (preserving $\ge 2$ features per column) and keeps top $\max(2, \operatorname{round}((1-q)|B_{m,c}|))$ per lobe (`sparseness=None` or `0.0` gives a full disjoint $k$-lobe partition).
  3. *Stage 2 (support-restricted polish):* initializes $V^{(2,0)}$ on $S_{m,c}$ and runs `simlr` with `ColumnSupportShrinkageOperator`, `nsa_w=0.0`, and `sparseness=None`.

**Automatic directional defaults (`resolve_directional_defaults`):**

* `positivity="auto"` resolves to `"conical"` (unit-$\ell_2$ cone) for $k = 1$ under degree-0 `nc` energy and `"convex"` (radial simplex $v_+ / \mathbf{1}^\top v_+$) for $k > 1$ under degree-1 `acc` energy.
* `energy_type="auto"` resolves to `"nc"` (signed normalized correlation) for $k = 1$ and `"acc"` for $k > 1$.
* `mixing_algorithm="auto"` resolves to `"pca"`.

**Support-restricted projection (`project_uniform_shrunk`).** For each view $m$ and column $c$, let $S_{m,c} = \{i : \hat{v}_{m,i,c} > 0\}$. The fitted column is normalized strictly on $S_{m,c}$ and blended with $|S_{m,c}|^{-1}\mathbf{1}_{S_{m,c}}$ (preserving exact zeros outside $S_{m,c}$):

$$w_{m,c}(\gamma_{m,c}) = (1 - \gamma_{m,c})\,\hat{w}_{m,c} + \gamma_{m,c}\,\frac{1}{|S_{m,c}|}\mathbf{1}_{S_{m,c}}, \qquad \gamma_{m,c} = \operatorname{clip}\!\left(1 - \frac{n_{\text{tr},m}}{1.5\,d_{\text{eff}}(|S_{m,c}|, p_m)},\ 0.0,\ 0.60\right).$$

View scores are $Z_m w_{m,c}(\gamma_{m,c})$ (`view_scaling="natural"`); consensus is $u_c = \operatorname{nanmean}_m (Z_m w_{m,c})$.

### Relation to the rest of pysimlr

* `UniformShrinkageOperator` is the closed-form $O(p_m k)$ resolvent $(\mathbf{I} + \lambda \mathbf{L})^{-1}$ of the complete-graph Laplacian $\mathbf{L} = \mathbf{I} - \mathbf{\Pi}$, $\lambda = \alpha / (1 - \alpha)$. It is stateless and not idempotent; `pysimlr.sparsification._apply_smoothing` caches each step's projected image so each iterate is smoothed once.
* **Sign orientation is a user contract.** Shrinking toward uniform positive weights assumes $+1$ has consistent polarity within each view; flipping a view negates its score.
* **`uniform_shrinkage_params(k > 1)`.** Accepts arbitrary $\alpha \in [0, 1]$ whenever pairwise-disjoint `supports=` are supplied; without `supports=` (or `groups=` with $\alpha \le 0.15$), `uniform_shrinkage_params(k > 1)` raises `ValueError` and directs callers to `fit_uniform_shrunk(..., k=k)`.

## Synthetic benchmark

Run `python scripts/uniform_shrinkage_benchmark.py --n-seeds 20 --n-boot 2000` (or `--smoke` with counts of 2 in `< 10 s`) to evaluate:

* `n_lt_p` ($k = 1$, $n = 40$, widths `[100, 120, 100]`, `sparseness=None`): dense uniform shrinkage vs unshrunk fit and plain mean.
* `n_lt_p_sparse0.3` ($k = 1$, $n = 40$, widths `[100, 120, 100]`, `sparseness=0.3`): two-stage support-restricted RIC shrinkage vs unshrunk fit.
* `n_gg_p` ($k = 1$, $n = 600$, widths `[20, 25, 15]`, `sparseness=None`): large-$n$ regime showing no loss when $\alpha \to 0.05$.
* `k2_disjoint` ($k = 2$, $n = 60$, widths `[80, 100, 60]`, two disjoint non-negative modes): two-stage disjoint support-restricted RIC shrinkage vs unshrunk NSA-Flow $k = 2$ on held-out mean $R^2$ and exact basis orthogonality ($\|V^\top V - I_2\|_F$).

## Should it be default ON in `pysimlr.simlr`?

No. It stays opt-in via `fit_uniform_shrunk` / `uniform_shrinkage_params`. It requires shared coordinate polarity within each view and adds little when $n \gg p$. On unoriented mixed-sign features, uniform shrinkage cancels the signal (`tests/test_uniform_shrinkage.py::test_mixed_sign_planted_signal_shows_why_default_is_off`).

| Regime | Feature polarity | $k$ | $n/p_{\max}$ | Recommendation |
| :-- | :-- | :-: | :-: | :-- |
| Oriented views (dense) | aligned | 1 | $n \lesssim 2.5\,p_{\max}$ | `fit_uniform_shrunk(views, k=1)` (defaults: `conical`, `nc`, `pca`) |
| Oriented views (sparse) | aligned | 1 | $n \lesssim 2.5\,p_{\max}$ | `fit_uniform_shrunk(views, k=1, sparseness=0.3)` (two-stage RIC; defaults: `conical`, `nc`, `pca`) |
| Multi-component oriented | aligned | $\ge 2$ | any | `fit_uniform_shrunk(views, k=k)` (two-stage disjoint RIC; defaults: radial `convex`, `acc`, `pca`) |
| Unknown or mixed signs | mixed | any | any | keep off (`positivity="either"`, or `SparseGraphResolvent` over a graph) |
