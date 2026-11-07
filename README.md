# PySiMLR: Similarity-Driven Multi-View Representation Learning in PyTorch

`pysimlr` is a PyTorch and Scikit-Learn implementation of the **Similarity-driven Multi-view Representation (SiMR)** framework for auditable multi-modal data integration, biomarker discovery, and clinical translation.

Every architecture in `pysimlr` enforces the **Linear Entrypoint Principle**: each input modality $\mathbf{X}_m \in \mathbb{R}^{N \times p_m}$ ($m \in \{1, \dots, M\}$) enters the model exclusively through an explicit, inspectable linear projection matrix $\mathbf{V}_m \in \mathbb{R}^{p_m \times k}$ regularized by **Non-Negative Stiefel Approximating Flow (`nsa_flow`)**:

$$
\mathbf{Z}_{m,\text{lin}} = \mathbf{X}_m \mathbf{V}_m
$$

Holding this linear input boundary fixed, `pysimlr` provides three interchangeable cross-modal alignment topologies:

1. **Linear Consensus (`simlr`)**: Alternating optimization of the feature dictionaries $\{\mathbf{V}_m\}_{m=1}^M$ via `nsa_flow` (optimizer `hybrid_adam` by default; `adam`, `nadam`, `lookahead`, `lars` and others are available) and a shared orthonormal subject embedding $\mathbf{U} \in \operatorname{St}(N, k)$ via Sylvester-differentiable polar decomposition and within-view similarity graphs $\mathbf{S}_m$.
2. **Linear Entrypoint with Non-Linear Decoders (`lend_simr` / `LEND`, `ned_simr` / `NED`, `ned_simr_shared_private` / `NEDPP`)**: Couples linear `nsa_flow`-regularized encoders $\mathbf{Z}_{m,\text{lin}} = \mathbf{X}_m \mathbf{V}_m$ with modality-specific multilayer perceptron decoders $g_{\phi_m}(\mathbf{U}) \to \mathbf{X}_m$, capturing curved cross-modal manifolds during training while keeping test-time projection strictly linear.
3. **Detached-Prior Normalizing Flows (`flow_simr_v` / `Flow-SiMR-V`, `flow_simr`)**: Routes the linear projections $\mathbf{Z}_{m,\text{lin}} = \mathbf{X}_m \mathbf{V}_m$ through invertible RealNVP affine coupling flows $f_{\psi_m} : \mathbb{R}^k \to \mathbb{R}^k$ with detached Gaussian prior gradients ($\operatorname{sg}(\mathbf{X}_m \mathbf{V}_m)$) to prevent rectangular subspace collapse and support exact conditional cross-view imputation (`FlowConditionalInference`).

---

## Installation

```bash
git clone https://github.com/stnava/pysimlr.git
cd pysimlr
pip install -e ".[dev]"
```

`pysimlr` requires Python $\ge 3.12$, `torch >= 2.0.0`, `numpy`, `scipy`, `pandas`, `scikit-learn`, and `nsa_flow >= 3.5.1`.

---

## Quick Start

The blocks below run in order and are executed by `tests/test_readme_examples.py`.

### 1. One interface to every method

`pysimlr.fit(method, views, k, **params)` fits any registered method with one parameter vocabulary; `pysimlr.predict(result, views)` projects new samples. Every result carries the same keys (`u`, `v`, `latents`, `convergence`, rank diagnostics, `random_state`) and a `parameters` record of what the method used and what it ignored.

```python
import numpy as np
import pysimlr

rng = np.random.default_rng(42)
u_true = rng.standard_normal((120, 3))
X1 = u_true @ rng.standard_normal((3, 60)) + 0.5 * rng.standard_normal((120, 60))
X2 = u_true @ rng.standard_normal((3, 40)) + 0.5 * rng.standard_normal((120, 40))
views_train = [X1[:90], X2[:90]]
views_test = [X1[90:], X2[90:]]

print(pysimlr.method_names())
# ['simlr', 'lend', 'ned', 'ned_shared_private',
#  'flow_v', 'pca', 'nsa_pipeline']

# Benchmark loop: one call shape, no per-method branch.
for name in ["simlr", "lend", "flow_v", "pca"]:
    res = pysimlr.fit(name, views_train, k=3, iterations=20, random_state=0,
                      sparseness=0.5, nsa_w=0.5)
    pred = pysimlr.predict(res, views_test)
    print(name, pred["u"].shape, "ignored:", sorted(res["parameters"]["ignored"]))
```

`iterations` is the budget for every method; `pysimlr.describe_method(name)` reports its unit (sweeps for `simlr`, epochs for the deep and flow models, none for the closed-form `pca`) and every parameter with its default. A parameter that another method uses but this one cannot is recorded in `res["parameters"]["ignored"]` and warned about once (`strict=True` raises); a name no method has raises `TypeError`.

### 2. Native entry points

The registry calls these functions; they take the same names directly.

```python
from pysimlr import simlr, lend_simr, flow_simr_v

# Linear SiMLR: non-negative, 50%-sparse dictionaries, hybrid Adam optimizer.
res_lin = simlr(views_train, k=3, iterations=30, optimizer_type="hybrid_adam",
                positivity="positive", sparseness=0.5)
V1, V2 = res_lin["v"]            # (60, 3) and (40, 3)

# LEND: linear NSA-Flow encoders, nonlinear decoders. `random_state` seeds the
# whole fit on a private RNG stream; the global RNG is left untouched.
res_lend = lend_simr(views_train, k=3, iterations=20, nsa_w=0.5, sparseness=0.5,
                     random_state=42)

# Flow-SiMR-V: linear encoders, RealNVP flows.
res_flow = flow_simr_v(views_train, k=3, iterations=10, nsa_w=0.5, random_state=42)
```

### 3. Scikit-learn

`MultiViewSiMR` is one estimator for every method. scikit-learn's cross-validation slices rows of a single array, so concatenate the views and give their column counts in `view_sizes`:

```python
from sklearn.model_selection import GridSearchCV
from pysimlr import MultiViewSiMR

X = np.hstack(views_train)
est = MultiViewSiMR(n_components=3, iterations=10, random_state=0, view_sizes=[60, 40])
search = GridSearchCV(est, {"method": ["simlr", "lend", "pca"]}, cv=3).fit(X)
print(search.best_params_)

Z_test = search.best_estimator_.transform(np.hstack(views_test))   # (30, 3)
```

`score` is held-out reconstruction R² averaged over views, computed in the same standardized space for every method; pass `scoring=` for another criterion. A list of views also works for `fit`/`transform` outside cross-validation.

### 4. Patient-level attribution and basis diagnostics

Because every view enters linearly via $\mathbf{Z}_{m,\text{lin}} = \mathbf{X}_m \mathbf{V}_m$, the exact additive contribution of feature $j$ in modality $m$ to latent coordinate $l$ for patient $i$ is $c_{m,i,j,l} = x_{m,i,j} v_{m,j,l}$:

```python
from pysimlr import summarize_basis_matrix

# Per-column sparsity, effective rank, and Stiefel Gram defect D(V_m)
summary_v1 = summarize_basis_matrix(res_lin["v"][0])
```

### 5. Edges between views: score-level path models

`pysimlr.path_model` evaluates directed or undirected dependency graphs over view scores and clinical outcomes using Freedman–Lane permutation sweeps with joint max-$T$ multiplicity control, omnibus tests, bootstrap effect-size intervals, and refit calibration. `pysimlr.compare_path_models` selects among competing graph topologies (chains, stars, complete graphs) using held-out Gaussian log-likelihood, paired bootstrap differences against the top model, and a strict parsimony rule. Missing views are reconstructed across significant path edges with `pysimlr.predict_path`. See `docs/usage_notes.md` for practical guidelines and what permutation tests cannot say.

---

## Architecture & Operating-Point Reference

### `nsa_flow` Trade-off Weight ($w \in (0, 1)$, default $w = 0.5$ for every method, `pysimlr.defaults`)
Each basis update minimizes the dimensionless `nsa_flow` energy:

$$
E_w(\mathbf{V}_m; \mathbf{V}_{m,0}) = (1 - w)\,\frac{\|\mathbf{V}_m - \mathbf{V}_{m,0}\|_F^2}{2\|\mathbf{V}_{m,0}\|_F^2} + w\,\frac{\|\mathbf{G}(\mathbf{V}_m) - \frac{1}{k}\mathbf{I}_k\|_F^2}{1 - 1/k}, \qquad \mathbf{G}(\mathbf{V}_m) = \frac{\mathbf{V}_m^\top \mathbf{V}_m}{\operatorname{tr}(\mathbf{V}_m^\top \mathbf{V}_m)}
$$

* **`w = 0.5` (default)**: Sits in the middle of the stable conditioning plateau ($w \in [0.20, 0.85]$), maintaining full effective rank ($\text{EffRank} \approx k$) and low Gram defect ($D \le 10^{-3}$).
* **`w < 0.2`**: Risks column rank collapse on correlated views.
* **`w > 0.85`**: Increases boundary stiffness and runtime ($2.5\times$–$3.8\times$) as columns are forced toward strictly disjoint supports.

### Quantile `sparseness` Operating Points ($q \in \{\text{None}, 0.5, 0.8\}$)
Following gradient polish, `sparseness=q` zeros out weights below the $q$-th quantile of each column's active entries and re-optimizes $E_w(\mathbf{V}_m)$ restricted to the surviving support mask:

* **`sparseness=0.5` (Default — Discovery Operating Point)**: Retains the top $50\%$ of active weights per column. Maximizes out-of-sample linear subspace fidelity ($0.679$ mean across 12 benchmarks) and single-axis Random Forest interpretability ($0.651$), and achieves $F_1 = 0.847$ pathway recovery on $50\%$-sparse ground-truth benchmarks.
* **`sparseness=0.8`–`0.85` (Focal Biomarker Panel Operating Point)**: Retains only the top $15\%$–$20\%$ of weights per column while preserving $>99\%$ of peak linear predictive performance ($0.673$ vs. $0.679$) and isolating compact anatomical or molecular panels (e.g., $9/62$ DKT cortical regions per network on `ADNI-MCI`).
* **`sparseness=None` (Unthresholded Soft Polish)**: Applies `nsa_flow` without post-optimization quantile thresholding.

### Basis constraint `positivity` (`"linear"` | `"conical"` | `"convex"`)
| value | aliases | each basis column `v` |
|---|---|---|
| `"linear"` | `"either"`, `False` (default) | any real weights |
| `"conical"` | `"positive"`, `True` | `v ≥ 0` |
| `"convex"` | — | `v ≥ 0`, `sum(v) = 1`: each latent is a weighted average of features |

`"convex"` is enforced inside the optimisation of every family with a basis (linear, LEND, NED, NEDPP, flows, anchored extension). The probability simplex replaces the unit-length column scaling, so it is not a rescaled `"conical"` fit. Unknown values raise. Details and measured behaviour: `docs/usage_notes.md` ("Constraining the basis") and `docs/tracking.md` D36.

---

## Technical Design Notes (v0.2.17)

1. **Orthogonal Varimax Subspace Rotation & Cross-View Column Alignment (`varimax`, `varimax_rotation`)**:
   * **Problem ("First Component Is the Mean")**: When input features share a positive correlation floor (e.g., regional cortical volumes in `ADNI-MCI` or clinical vitals in `Heart`/`Diabetes`), the Perron–Frobenius leading PCA eigenvector $\mathbf{v}_{m,1}$ is dense and single-signed, whereas $\mathbf{v}_{m,2}, \dots, \mathbf{v}_{m,k}$ are sign-alternating contrast vectors. Clamping raw PCA loadings directly ($\max(0, \mathbf{E}_m)$) traps $\mathbf{v}_{m,1}$ in a dense global-mean mode and forces the remaining columns into residual supports.
   * **Solution in `v0.2.17`**:
     1. **Exact $O(k)$ Subspace Rotation**: Both `nsa_flow_data` (used automatically under `positivity="positive"`) and `initialization_type="varimax"` (for signed or non-negative bases) rotate the leading $k$-dimensional SVD subspace $\mathbf{E}_m \leftarrow \operatorname{varimax}(\mathbf{E}_m) = \mathbf{E}_m \mathbf{R}_m$ ($\mathbf{R}_m \in O(k)$) before non-negative clamping or signed splitting. Because $(\mathbf{E}_m \mathbf{R}_m)(\mathbf{E}_m \mathbf{R}_m)^\top = \mathbf{E}_m \mathbf{E}_m^\top$, the $k$-dimensional subspace projector is preserved to machine precision while the dense Perron–Frobenius mean direction is redistributed across localized, simple-structure anatomical/molecular subnetworks.
     2. **Cross-View Hungarian Column Alignment**: Because per-view Varimax and `nsa_flow_data` are $O(k)$ permutation-invariant, `initialize_simlr` automatically aligns the column permutations and signs of views $m \ge 2$ to view $1$ in sample space ($\mathbf{P}_1^\top \mathbf{P}_m$, where $\mathbf{P}_m = \mathbf{X}_m \mathbf{V}_m$) via linear sum assignment before iteration 0.
     3. **One `varimax` across entry points (`v0.2.18`)**: `simlr`, `flow_simr_v`, `lend_simr`, `ned_simr`, `ned_simr_shared_private`, `nnh_embed` and `extend_simlr_embedding_with_new_modalities` all take `varimax: Optional[bool] = None` and pass it to `initialize_simlr` (the deep and flow entry points previously accepted it and dropped it). `None` rotates only a non-negative basis; `True` also rotates a signed basis; `False` never rotates. Under `initialization_type="gcca"`, a single rotation $\mathbf{R} = \operatorname{varimax}(\mathbf{U}_{\text{shared}})$ is applied to the joint MAXVAR subspace before each view is regressed onto it, so the views are column-aligned by construction. Varimax acts only at initialization: trained deep/flow bases are never rotated, because the decoders and RealNVP couplings are not $O(k)$-equivariant. Signed rotation is opt-in because measured fits after a rotated signed start were mixed (NED collapse basin 6/11 seeds against 4/11 unrotated); see `pysimlr.simlr.resolve_varimax`.
     4. **Additional Controls**: Combine with column-wise quantile thresholding (`sparseness=0.5` or `sparseness=0.85`) for exact per-column cardinality control, or in-fold subject row-mean centering when global per-subject amplitude is a nuisance factor.
2. **Subspace Capacity vs. Coordinate-Axis Disentanglement**:
   * Multi-variable linear probes (`Ridge(X_m V_m)`) are invariant to invertible right-mixing $\mathbf{V}_m \leftarrow \mathbf{V}_m \mathbf{M}$ ($\operatorname{span}(\mathbf{X}_m \mathbf{V}_m \mathbf{M}) = \operatorname{span}(\mathbf{X}_m \mathbf{V}_m)$). Always pair linear subspace probes with axis-sensitive tree probes (`RandomForest(X_m V_m)`), Gram defect $D(\mathbf{V}_m)$, and support recovery ($F_1$) when evaluating coordinate interpretability.
3. **Rectangular Normalizing Flow Prior Detachment**:
   * In `Flow-SiMR-V`, backpropagating the isotropic Gaussian prior $\|f_{\psi_m}(\mathbf{X}_m \mathbf{V}_m)\|_2^2$ into a rectangular matrix $\mathbf{V}_m \in \mathbb{R}^{p_m \times k}$ ($k \ll p_m$) rotates $\mathbf{V}_m$ onto the smallest-variance trailing eigenvectors of $\mathbf{X}_m^\top \mathbf{X}_m$. `flow_simr_v` prevents this by detaching $\mathbf{Z}_{m,\text{lin}}$ at the input to the prior term while training $\mathbf{V}_m$ via cross-view alignment and reconstruction.

---

## Development, Testing, and Building Documentation

```bash
# Run the fast unit and property test suite
make test

# pytest.ini runs 10 xdist workers and excludes @pytest.mark.slow by default:
pytest              # fast subset (~20 s)
pytest -m ""        # full suite, including the slow calibration/benchmark tests
pytest -n 0 -x ...  # single process, for debugging and pdb

# Render Quarto tutorials in docs/
make docs

# Render the full Quarto manuscript (HTML book + PDF in paper/_book/ and docs/manuscript/)
make paper

# Remove temporary build and cache artifacts
make clean
```

### Repository Structure

* `src/pysimlr/`: Core package (`api.py` -- the method registry, `fit` and `predict`; `simlr.py`, `deep.py`, `flows.py`, `sparsification.py`, `optimizers.py`, `sklearn.py`, `interpretability.py`, `consensus.py`, `benchmarks/`). The interface contract is in `docs/api_contract.md`; practical fitting and interpretation notes are in `docs/usage_notes.md`.
* `tests/`: Unit, statistical calibration, optimizer correctness, and end-to-end tests.
* `paper/`: Quarto book manuscript (`index.qmd`, `01_intro_background.qmd`, `02_methods.qmd`, `03_experiments.qmd`, `04_discussion.qmd`, `appendix_*.qmd`, `references.bib`, and cached empirical tables in `paper/results_cache/`).
* `docs/`: Quarto tutorials (`docs/*.qmd`) and rendered manuscript (`docs/manuscript/`).
* `reports/`: Standalone visual HTML benchmark reports (`deep_method_ranking_benchmark.html`, `sparseness_benchmark_report.html`, `simr_ablation_and_feature_report.html`).
* `scripts/`: Benchmark runners and data loaders (`run_deep_benchmark.py`, `adni_loader.py`, `tcga_loader.py`).

## License

Apache-2.0
