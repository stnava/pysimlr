# PySiMLR: Similarity-Driven Multi-View Representation Learning in PyTorch

`pysimlr` is a PyTorch and Scikit-Learn implementation of the **Similarity-driven Multi-view Representation (SiMR)** framework for auditable multi-modal data integration, biomarker discovery, and clinical translation.

Every architecture in `pysimlr` enforces the **Linear Entrypoint Principle**: each input modality $\mathbf{X}_m \in \mathbb{R}^{N \times p_m}$ ($m \in \{1, \dots, M\}$) enters the model exclusively through an explicit, inspectable linear projection matrix $\mathbf{V}_m \in \mathbb{R}^{p_m \times k}$ regularized by **Non-Negative Stiefel Approximating Flow (`nsa_flow`)**:

$$
\mathbf{Z}_{m,\text{lin}} = \mathbf{X}_m \mathbf{V}_m
$$

Holding this linear input boundary fixed, `pysimlr` provides three interchangeable cross-modal alignment topologies:

1. **Linear Consensus (`simlr` / `SiMLR`)**: Alternating optimization of the feature dictionaries $\{\mathbf{V}_m\}_{m=1}^M$ via `nsa_flow` (`L-BFGS-B`, `L-BFGS`, `Lookahead`, or `LARS`) and a shared orthonormal subject embedding $\mathbf{U} \in \operatorname{St}(N, k)$ via Sylvester-differentiable polar decomposition and within-view similarity graphs $\mathbf{S}_m$.
2. **Linear Entrypoint with Non-Linear Decoders (`lend_simr` / `LEND`, `ned_simr` / `NED`, `ned_simr_shared_private` / `NEDPP`)**: Couples linear `nsa_flow`-regularized encoders $\mathbf{Z}_{m,\text{lin}} = \mathbf{X}_m \mathbf{V}_m$ with modality-specific multilayer perceptron decoders $g_{\phi_m}(\mathbf{U}) \to \mathbf{X}_m$, capturing curved cross-modal manifolds during training while keeping test-time projection strictly linear.
3. **Detached-Prior Normalizing Flows (`flow_simr_v` / `Flow-SiMR-V`, `flow_simr`)**: Routes the linear projections $\mathbf{Z}_{m,\text{lin}} = \mathbf{X}_m \mathbf{V}_m$ through invertible RealNVP affine coupling flows $f_{\psi_m} : \mathbb{R}^k \to \mathbb{R}^k$ with detached Gaussian prior gradients ($\operatorname{sg}(\mathbf{X}_m \mathbf{V}_m)$) to prevent rectangular subspace collapse and support exact conditional cross-view imputation (`FlowConditionalInference`).

---

## Installation

```bash
git clone https://github.com/stnava/pysimlr.git
cd pysimlr
pip install -e ".[dev]"
```

`pysimlr` requires Python $\ge 3.12$, `torch >= 2.0.0`, `numpy`, `scipy`, `pandas`, `scikit-learn`, and `nsa_flow >= 3.0.0`.

---

## Quick Start

### 1. Functional API (`simlr`, `lend_simr`, `flow_simr_v`)

```python
import numpy as np
from pysimlr import (
    simlr,
    lend_simr,
    flow_simr_v,
    predict_simlr,
    predict_deep,
    attribute_prediction_to_features,
)

rng = np.random.default_rng(42)
X1 = rng.standard_normal((120, 60))
X2 = rng.standard_normal((120, 40))
views_train = [X1[:90], X2[:90]]
views_test = [X1[90:], X2[90:]]

# 1. Linear SiMLR with L-BFGS-B + NSA-Flow polish and 50% quantile sparsity
res_lin = simlr(
    views_train,
    k=3,
    optimizer="lbfgsb",
    constraint="orthox0.5x1",
    positivity="positive",
    sparseness=0.5,
    random_state=42,
)
U_train = res_lin["u"]          # (90, 3) shared consensus embedding
V1, V2 = res_lin["v"]           # (60, 3) and (40, 3) sparse non-negative dictionaries

# Out-of-sample linear entrypoint projection: Z_m = X_{m,test} @ V_m
Z1_test = views_test[0] @ np.asarray(V1)

# 2. LEND: Linear NSA-Flow encoder + non-linear MLP decoders
res_lend = lend_simr(
    views_train,
    k=3,
    nsa_w=0.5,
    positivity="positive",
    sparseness=0.5,
    epochs=150,
    random_state=42,
)

# 3. Flow-SiMR-V: Linear NSA-Flow bottleneck + invertible RealNVP flows
res_flow = flow_simr_v(
    views_train,
    k=3,
    nsa_w=0.5,
    positivity="positive",
    sparseness=0.5,
    epochs=150,
    random_state=42,
)
```

### 2. Scikit-Learn Estimator & Pipeline API

All multi-view architectures expose standard `fit`, `transform`, and `fit_transform` interfaces compatible with `scikit-learn` cross-validation and pipelines:

```python
from pysimlr import (
    SiMLR,
    LENDTransformer,
    NEDTransformer,
    FlowSiMLRTransformer,
    MultiViewPCA,
)

model = SiMLR(
    n_components=3,
    optimizer="lbfgsb",
    positivity="positive",
    sparseness=0.5,
    random_state=42,
)
Z_train = model.fit_transform(views_train)
Z_test = model.transform(views_test)
```

### 3. Patient-Level Attribution & Basis Diagnostics

Because every view enters linearly via $\mathbf{Z}_{m,\text{lin}} = \mathbf{X}_m \mathbf{V}_m$, the exact additive contribution of feature $j$ in modality $m$ to latent coordinate $l$ for patient $i$ is $c_{m,i,j,l} = x_{m,i,j} v_{m,j,l}$:

```python
from pysimlr import summarize_basis_matrix, build_interpretability_report

# Inspect per-column sparsity, effective rank, and Stiefel Gram defect D(V_m)
summary_v1 = summarize_basis_matrix(res_lin["v"][0])
```

---

## Architecture & Operating-Point Reference

### `nsa_flow` Trade-off Weight ($w \in (0, 1)$, default $w = 0.5$)
Each basis update minimizes the dimensionless `nsa_flow` energy:

$$
E_w(\mathbf{V}_m; \mathbf{V}_{m,0}) = (1 - w)\,\frac{\|\mathbf{V}_m - \mathbf{V}_{m,0}\|_F^2}{2\|\mathbf{V}_{m,0}\|_F^2} + w\,\frac{\|\mathbf{G}(\mathbf{V}_m) - \frac{1}{k}\mathbf{I}_k\|_F^2}{1 - 1/k}, \qquad \mathbf{G}(\mathbf{V}_m) = \frac{\mathbf{V}_m^\top \mathbf{V}_m}{\operatorname{tr}(\mathbf{V}_m^\top \mathbf{V}_m)}
$$

* **`w = 0.5` (Default)**: Sits in the middle of the stable conditioning plateau ($w \in [0.20, 0.85]$), maintaining full effective rank ($\text{EffRank} \approx k$) and low Gram defect ($D \le 10^{-3}$).
* **`w < 0.2`**: Risks column rank collapse on correlated views.
* **`w > 0.85`**: Increases boundary stiffness and runtime ($2.5\times$–$3.8\times$) as columns are forced toward strictly disjoint supports.

### Quantile `sparseness` Operating Points ($q \in \{\text{None}, 0.5, 0.8\}$)
Following gradient polish, `sparseness=q` zeros out weights below the $q$-th quantile of each column's active entries and re-optimizes $E_w(\mathbf{V}_m)$ restricted to the surviving support mask:

* **`sparseness=0.5` (Default — Discovery Operating Point)**: Retains the top $50\%$ of active weights per column. Maximizes out-of-sample linear subspace fidelity ($0.679$ mean across 12 benchmarks) and single-axis Random Forest interpretability ($0.651$), and achieves $F_1 = 0.847$ pathway recovery on $50\%$-sparse ground-truth benchmarks.
* **`sparseness=0.8`–`0.85` (Focal Biomarker Panel Operating Point)**: Retains only the top $15\%$–$20\%$ of weights per column while preserving $>99\%$ of peak linear predictive performance ($0.673$ vs. $0.679$) and isolating compact anatomical or molecular panels (e.g., $9/62$ DKT cortical regions per network on `ADNI-MCI`).
* **`sparseness=None` (Unthresholded Soft Polish)**: Applies `nsa_flow` without post-optimization quantile thresholding.

---

## Technical Design Notes (v0.2.17)

1. **Orthogonal Varimax Subspace Rotation & Cross-View Column Alignment (`varimax`, `varimax_rotation`)**:
   * **Problem ("First Component Is the Mean")**: When input features share a positive correlation floor (e.g., regional cortical volumes in `ADNI-MCI` or clinical vitals in `Heart`/`Diabetes`), the Perron–Frobenius leading PCA eigenvector $\mathbf{v}_{m,1}$ is dense and single-signed, whereas $\mathbf{v}_{m,2}, \dots, \mathbf{v}_{m,k}$ are sign-alternating contrast vectors. Clamping raw PCA loadings directly ($\max(0, \mathbf{E}_m)$) traps $\mathbf{v}_{m,1}$ in a dense global-mean mode and forces the remaining columns into residual supports.
   * **Solution in `v0.2.17`**:
     1. **Exact $O(k)$ Subspace Rotation**: Both `nsa_flow_data` (used automatically under `positivity="positive"`) and `initialization_type="varimax"` (for signed or non-negative bases) rotate the leading $k$-dimensional SVD subspace $\mathbf{E}_m \leftarrow \operatorname{varimax}(\mathbf{E}_m) = \mathbf{E}_m \mathbf{R}_m$ ($\mathbf{R}_m \in O(k)$) before non-negative clamping or signed splitting. Because $(\mathbf{E}_m \mathbf{R}_m)(\mathbf{E}_m \mathbf{R}_m)^\top = \mathbf{E}_m \mathbf{E}_m^\top$, the $k$-dimensional subspace projector is preserved to machine precision while the dense Perron–Frobenius mean direction is redistributed across localized, simple-structure anatomical/molecular subnetworks.
     2. **Cross-View Hungarian Column Alignment**: Because per-view Varimax and `nsa_flow_data` are $O(k)$ permutation-invariant, `initialize_simlr` automatically aligns the column permutations and signs of views $m \ge 2$ to view $1$ in sample space ($\mathbf{P}_1^\top \mathbf{P}_m$, where $\mathbf{P}_m = \mathbf{X}_m \mathbf{V}_m$) via linear sum assignment before iteration 0.
     3. **Additional Controls**: Combine with column-wise quantile thresholding (`sparseness=0.5` or `sparseness=0.85`) for exact per-column cardinality control, or in-fold subject row-mean centering when global per-subject amplitude is a nuisance factor.
2. **Subspace Capacity vs. Coordinate-Axis Disentanglement**:
   * Multi-variable linear probes (`Ridge(X_m V_m)`) are invariant to invertible right-mixing $\mathbf{V}_m \leftarrow \mathbf{V}_m \mathbf{M}$ ($\operatorname{span}(\mathbf{X}_m \mathbf{V}_m \mathbf{M}) = \operatorname{span}(\mathbf{X}_m \mathbf{V}_m)$). Always pair linear subspace probes with axis-sensitive tree probes (`RandomForest(X_m V_m)`), Gram defect $D(\mathbf{V}_m)$, and support recovery ($F_1$) when evaluating coordinate interpretability.
3. **Rectangular Normalizing Flow Prior Detachment**:
   * In `Flow-SiMR-V`, backpropagating the isotropic Gaussian prior $\|f_{\psi_m}(\mathbf{X}_m \mathbf{V}_m)\|_2^2$ into a rectangular matrix $\mathbf{V}_m \in \mathbb{R}^{p_m \times k}$ ($k \ll p_m$) rotates $\mathbf{V}_m$ onto the smallest-variance trailing eigenvectors of $\mathbf{X}_m^\top \mathbf{X}_m$. `flow_simr_v` prevents this by detaching $\mathbf{Z}_{m,\text{lin}}$ at the input to the prior term while training $\mathbf{V}_m$ via cross-view alignment and reconstruction.

---

## Development, Testing, and Building Documentation

```bash
# Run the fast unit and property test suite
make test

# Render Quarto tutorials in docs/
make docs

# Render the full Quarto manuscript (HTML book + PDF in paper/_book/ and docs/manuscript/)
make paper

# Remove temporary build and cache artifacts
make clean
```

### Repository Structure

* `src/pysimlr/`: Core package (`simlr.py`, `deep.py`, `flows.py`, `sparsification.py`, `optimizers.py`, `sklearn.py`, `interpretability.py`, `consensus.py`, `benchmarks/`).
* `tests/`: Unit, statistical calibration, optimizer correctness, and end-to-end tests.
* `paper/`: Quarto book manuscript (`index.qmd`, `01_intro_background.qmd`, `02_methods.qmd`, `03_experiments.qmd`, `04_discussion.qmd`, `appendix_*.qmd`, `references.bib`, and cached empirical tables in `paper/results_cache/`).
* `docs/`: Quarto tutorials (`docs/*.qmd`) and rendered manuscript (`docs/manuscript/`).
* `reports/`: Standalone visual HTML benchmark reports (`deep_method_ranking_benchmark.html`, `sparseness_benchmark_report.html`, `simr_ablation_and_feature_report.html`).
* `scripts/`: Benchmark runners and data loaders (`run_deep_benchmark.py`, `adni_loader.py`, `tcga_loader.py`).

## License

Apache-2.0
