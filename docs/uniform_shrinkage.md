# Uniform Shrinkage for Directional Multi-View SiMLR (`pysimlr.uniform_shrinkage`)

```python
import pysimlr

# 1. Orient views to a shared directional convention (e.g. higher = worse pathology)
prep = pysimlr.prepare_views(raw_views, fit_rows=train_mask, signs=pathology_signs, nuisance=nuis)

# 2. Build dimension-adaptive uniform-shrinkage parameters (alpha_m = p_m / (n_obs + p_m))
params = pysimlr.uniform_shrinkage_params(prep["views"], alpha="auto", sparseness=0.3)

# 3. Fit exposure-free SiMLR and project on the natural simplex scale
fit_res = pysimlr.fit("simlr", prep["views"], k=1, random_state=0, **params)
proj = pysimlr.project_uniform_shrunk(fit_res, test_views)
u_scores = proj["u"]               # (n, 1) natural-amplitude consensus score
view_scores = proj["view_scores"]  # (n, M) per-view scores on the simplex scale
```

## Why This Strategy Exists (Bias–Variance Geometry at $n \lesssim p_m$)

Once every feature within view $m \in \{1, \dots, M\}$ is oriented to share a common directional meaning (for instance, cortical thinning, parenchymal atrophy, tract fixel loss, within-network hypoconnectivity, and relative hypoperfusion via `pysimlr.prepare_views(..., signs=...)`), two extreme estimators bound the bias–variance spectrum:

1. **Unweighted View Plain Mean ($\mathbf{w}_m^{(0)} = p_m^{-1}\mathbf{1}_{p_m}$):** Zero free parameters and **zero estimation variance**, capturing any diffuse global or high-coherence shift across a view, but assigning identical weight $1/p_m$ to uncoupled noise features and providing zero anatomical or network localization.
2. **Unregularized Sparse `SiMLR` ($\hat{\mathbf{v}}_m \in \mathcal{S}_{\ge 0}^{p_m - 1}$):** Isolates cross-modally coupled focal features, but estimates $p_m$ coordinates from $n_m$ observations (`O(p_m / n_m)` estimation variance). When $n_{\text{train}} \lesssim p_m$ (e.g. $n_{\text{train}} = 73$ against $p_m = 100$ regions), non-convex optimization admits distinct local optima (`multistart_fit`), single-bag out-of-bag reliability `ICC(1)` drops, and out-of-fold $\Delta R^2$ can trail the naive plain mean.

`pysimlr.uniform_shrinkage` connects these two endpoints continuously inside `simlr`'s native `smoothing_matrices` framework.

## Mathematical Formulation

### 1. Closed-Form Complete-Graph Green's Resolvent (`UniformShrinkageOperator`)

Let $\mathbf{L}_m = \mathbf{I}_{p_m} - \mathbf{\Pi}_m$ denote the mean-degree-normalized combinatorial Laplacian of the complete graph $K_{p_m}$ (where $\mathbf{\Pi}_m = p_m^{-1}\mathbf{1}_{p_m}\mathbf{1}_{p_m}^\top$ is the orthogonal projector onto the uniform direction, or block-mean projector when `groups=` partitions features into anatomical/functional systems). Because $\mathbf{\Pi}_m^2 = \mathbf{\Pi}_m$, its Green's resolvent has an exact Sherman–Morrison closed form requiring **$O(p_m k)$ time and $O(1)$ memory** (no $p_m \times p_m$ matrix is ever allocated):

$$\mathbf{S}_m(\alpha_m) = \bigl(\mathbf{I}_{p_m} + \lambda_m \mathbf{L}_m\bigr)^{-1} = (1 - \alpha_m)\mathbf{I}_{p_m} + \alpha_m \mathbf{\Pi}_m, \qquad \alpha_m = \frac{\lambda_m}{1 + \lambda_m} \in [0, 1].$$

* At $\alpha_m = 0$ ($\lambda_m = 0$): $\mathbf{S}_m(0) = \mathbf{I}_{p_m}$ (standard unshrunk `SiMLR`).
* At $\alpha_m \in (0, 1)$: Empirical-Bayes shrinkage pulling noisy coordinates toward the uniform background while allowing strongly cross-coupled focal features to rise multiple times above $1/p_m$.
* At $\alpha_m = 1$ ($\lambda_m \to \infty$): $\mathbf{S}_m(1) = \mathbf{\Pi}_m$ (exact uniform projector).

Like `SparseGraphResolvent`, `UniformShrinkageOperator` is idempotent on its own image so repeated calls inside `to_feasible` and post-step retraction never double-smooth a step, and `positive_part=True` protects signed iterates from cancelling before orthant projection.

### 2. Dimension-Adaptive Schedule (`auto_uniform_alpha`)

When `alpha="auto"` (default in `uniform_shrinkage_params`), each view receives shrinkage matched to its effective aspect ratio:

$$\alpha_m = \operatorname{clip}\!\left(\frac{p_m}{n_{\text{obs}, m} + p_m},\; 0.05,\; 0.85\right).$$

Narrower views (e.g. $p_m = 36$ white-matter bundles at $n = 93 \implies \alpha_m = 0.28$) retain sharper focal contrast automatically, wider views ($p_m = 100$ at $n = 73 \implies \alpha_m = 0.58$) receive stronger variance reduction, and $\alpha_m \to 0.05$ automatically in large cohorts ($n \gg p_m$).

### 3. Natural Simplex Projection (`project_uniform_shrunk`)

Standard `predict_simlr` unit-standardizes every view projection ($\operatorname{SD}(\mathbf{Z}_m\hat{\mathbf{v}}_m) = 1$), which artificially inflates views whose global mean was regressed out or cancelled by construction. Mapping each encoder column onto the probability simplex $\hat{\mathbf{w}}_m = \hat{\mathbf{v}}_m / (\mathbf{1}^\top \hat{\mathbf{v}}_m)$ and evaluating $\mathbf{Z}_m \hat{\mathbf{w}}_m$ directly (`view_scaling="natural"`) preserves each view's intrinsic within-modality coherence amplitude and guarantees **exact numerical identity** with the unweighted plain mean at $\alpha_m = 1$ (or `post_shrink=1.0`).

---

## Should Uniform Shrinkage Be Default ON in `pysimlr.simlr`?

**No. It must remain opt-in (`default OFF` in `pysimlr.simlr` and `pysimlr.fit`), and enabled explicitly via `uniform_shrinkage_params` when its structural assumptions hold.**

### Why Not Default ON Globally

1. **Requires Shared Coordinate Polarity:** Shrinking toward $+p_m^{-1}\mathbf{1}_{p_m}$ assumes $+1$ carries a consistent directional meaning across all features of view $m$. On unoriented raw features (where half of true loadings are $+0.6$ and half are $-0.6$; see `tests/test_uniform_shrinkage.py::test_mixed_sign_planted_signal_shows_why_default_is_off`), unshrunk `simlr(..., positivity="either")` recovers the latent at $|r| > 0.90$, whereas uniform nonnegative shrinkage drops recovery by $> 0.20$.
2. **Multi-Component Orthogonality ($k \ge 2$):** Higher-order components ($\mathbf{v}_{m,2}, \dots, \mathbf{v}_{m,k}$) must be mutually orthogonal or support-disjoint (`nsa_flow`). Pulling multiple columns simultaneously toward the same rank-1 vector $\mathbf{1}_{p_m}$ opposes column separation and increases dimensional collapse.
3. **Unnecessary Dilution When $n \gg p_m$:** In large biobank panels ($n \ge 5\, p_{\max}$, out-of-bag `ICC(1) > 0.90`), focal weights are already well-determined without leaking mass into uncoupled background features.

### Decision Matrix: When to Use `uniform_shrinkage_params`

| Study Regime | Feature Polarity | Rank $k$ | Sample-to-Width ($n / p_{\max}$) or Raw OOB `ICC(1)` | Recommendation |
| :--- | :--- | :---: | :---: | :--- |
| Directional disease / injury burden (e.g. `prepare_views` with pathology `signs`) | **Aligned** (`higher = worse`) | $k = 1$ | $n \lesssim 2.5\, p_{\max}$ or `ICC(1) < 0.85` | **Turn ON** (`uniform_shrinkage_params(..., alpha="auto")` + `project_uniform_shrunk`) |
| Directional disease / injury burden, large cohort | **Aligned** (`higher = worse`) | $k = 1$ | $n > 5\, p_{\max}$ and `ICC(1) >= 0.90` | Optional (`alpha="auto"` gently applies $\alpha_m \le 0.15$, or leave OFF for maximum sparsity) |
| Arbitrary-sign tabular, omics, or raw unoriented MRI | Mixed / unknown | Any $k$ | Any $n / p$ | **Keep OFF** (use `positivity="either"`, screen wide views or use `SparseGraphResolvent` over a spatial/network graph) |
| Multi-axis subspace discovery | Either | $k \ge 2$ | Any $n / p$ | **Keep OFF** (or restrict shrinkage to `groups=` anatomical modules with small `alpha <= 0.15`) |

## Measured Effect on Five-View Neuroimaging (`brain_systems@3`, $p = 63, 100, 36, 98, 99$)

Measured on 5-view pathology-oriented MRI (`Cortical thickness`, `Regional volume`, `White-matter bundles`, `Resting-state connectivity`, `Perfusion`; age-stratified Freedman–Lane partial $R^2$ and `5×5` fold-refitted out-of-fold $\Delta R^2$, `2,000` permutations):

| Estimator & Fit Regime | Single-Bag OOB `ICC(1)` (Gain) | Heavy-Weapon Rate $R^2$ ($p$) | Artillery `BEC 3` Rate $R^2$ ($p$) | `BEC 1` NegCtrl $p$ | `5×5` Fold-Refitted OOF $\Delta R^2$ ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Unweighted `plain_mean_5view` ($n=93$, 0 focal contrast) | — | `0.096` (`0.0050`) | `0.038` (`0.0812`) | `0.901` | `+0.071` (`0.0025`) |
| Unshrunk `conical k=1, sp=0.5` ($n=93$ single-cohort) | `0.226` (`-0.262`) | `0.040` (`0.0665`) | `0.014` (`0.2794`) | `0.985` | `+0.001` (`0.1359`) |
| **Uniform-shrunk (`n=93` single-cohort, `post_shrink="auto"`)** | — | **`0.049–0.072`** (**`0.0155–0.0440`**) | `0.029–0.040` (`0.076–0.120`) | `0.701–0.856` | **`+0.025 to +0.050`** (**`0.0100–0.0330`**) |
| Unshrunk `conical k=1, sp=0.5` ($n=186$ same-scanner basis) | `0.945` (`+0.537`) | `0.061` (`0.0222`) | `0.054` (`0.0334`) | `0.246` | — |
| **Uniform-shrunk (`n=186` basis, `alpha="auto", sp=0.3`, `post_shrink=0`)** | **`0.959` (`+0.533`)** | **`0.103` (`0.0030`)** | **`0.061` (`0.0210`)** | **`0.306`** | — |

On the $n = 186$ same-scanner basis (`post_shrink="auto"` resolves to `0.0` automatically since $n > 1.5\, p_{\max}$), `uniform_shrinkage_params(alpha="auto", sparseness=0.3)` + `project_uniform_shrunk` **surpasses `plain_mean_5view` on heavy-weapon rate (`10.3%, p = 0.0030` vs. `9.6%, p = 0.0050`) while beating `plain_mean_5view` by `+61%` on focal artillery `BEC 3` (`6.1%, p = 0.0210` vs. `3.8%, p = 0.0812`)** and maintaining `3.85×` focal tract contrast (`SLF`/`ILF` fixel density + salience connectivity). On $n = 93$ alone ($n_{\text{train}} = 73 < 100$), `post_shrink="auto"` activates sub-critical damping automatically and lifts `5×5` fold-refitted out-of-fold $\Delta R^2$ from `+0.001` ($p = 0.136$) to `+0.025–0.050` ($p = 0.010–0.033$).
