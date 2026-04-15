import os

def write_methods():
    content = r"""
# Methods: The SiMLR Framework

In this section, we present a unified mathematical formalism for the SiMLR (Similarity-driven Multi-view Linear Reconstruction) framework. We derive our deep architectural extensions—LEND, NED, and NEDPP—as specific cases of a constrained energy minimization objective that balances cross-modal alignment, generative reconstruction, and latent space stabilization.

## 1. The Unified SiMLR Master Objective

Let $\{X_m\}_{m=1}^M$ denote $M$ distinct data modalities (views) for $N$ samples, where $X_m \in \mathbb{R}^{N \times P_m}$. The goal of the framework is to find a set of view-specific encoding functions $f_m: \mathbb{R}^{P_m} \to \mathbb{R}^K$ and a shared consensus latent space $U \in \mathbb{R}^{N \times K}$ that minimize the following "Master Objective" $\mathcal{J}$:

$$ \mathcal{J}(\{f_m, g_m\}, U) = \underbrace{\sum_m \alpha_m \mathcal{L}_{\text{recon}}(g_m(f_m(X_m)), X_m)}_{\text{Reconstruction}} + \underbrace{\sum_m \beta_m \mathcal{L}_{\text{sim}}(f_m(X_m), U)}_{\text{Alignment}} + \underbrace{\mathcal{R}_{\text{stab}}(\{f_m\}, U)}_{\text{Stabilization}} $$

where:
- $g_m$ is an optional decoding function mapping the latent space back to the modality space.
- $\mathcal{L}_{\text{recon}}$ is the reconstruction loss (typically Mean Squared Error).
- $\mathcal{L}_{\text{sim}}$ is the cross-modal alignment loss, which can take several functional forms (Regression, ACC, or LogCosh).
- $\mathcal{R}_{\text{stab}}$ is a collection of VICReg-style regularization terms that prevent latent collapse and maintain statistical independence.

### 1.1 Similarity and Alignment Energies ($\mathcal{L}_{\text{sim}}$)

The alignment energy $\mathcal{L}_{\text{sim}}(Z, U)$ measures the fidelity of the view-specific latent $Z$ to the global consensus $U$. Our framework supports three primary operational definitions:

1.  **Regression (MSE):** Optimizes for optimal linear reconstruction of the shared consensus:
    $$ \mathcal{L}_{\text{sim\_reg}}(Z, U) = \| \text{Norm}(Z) - \text{Norm}(U) \|_F^2 $$
2.  **Absolute Canonical Covariance (ACC):** Maximizes the absolute covariance between views, providing robustness to sign-flipping and outlier samples:
    $$ \mathcal{L}_{\text{sim\_acc}}(Z, U) = - \sum_{k=1}^K \left| \text{Cov}(z_{\cdot, k}, u_{\cdot, k}) \right| $$
3.  **LogCosh (Robust ICA):** A robust, negentropy-inspired metric for uncovering statistically independent sources:
    $$ \mathcal{L}_{\text{sim\_logcosh}}(Z, U) = \sum_{i=1}^N \sum_{k=1}^K \log(\cosh(u_{ik} z_{ik})) $$

### 1.2 Latent Stabilization and VICReg ($\mathcal{R}_{\text{stab}}$)

To prevent the neural network from converging to trivial solutions (latent collapse), we incorporate stabilization terms based on the VICReg (Variance-Invariance-Covariance Regularization) principles:

- **Variance Preservation:** Forces each latent dimension to maintain a target standard deviation $\gamma$:
  $$ \mathcal{L}_{\text{var}}(Z) = \frac{1}{K} \sum_{j=1}^K \max(0, \gamma - \sqrt{\text{Var}(z_{\cdot, j}) + \epsilon}) $$
- **Covariance Regularization (Anti-Collapse):** Penalizes redundancy by decorrelating latent components:
  $$ \mathcal{L}_{\text{cov}}(Z) = \frac{1}{K} \sum_{i \ne j} [C(Z)]_{i, j}^2 $$

## 2. The Interpretability Contract: Functional Variants

The "Interpretability Contract" is a structural constraint on the encoder $f_m$. In all SiMLR models, $f_m$ is decomposed as $f_m = h_m \circ \text{Proj}_{V_m}$, where $\text{Proj}_{V_m}(X_m) = X_m V_m$ is a linear projection onto the Stiefel manifold ($V_m^\top V_m = I_K$).

1.  **Classical SiMLR (Linear):** $h_m = \text{Id}$ (Identity). The encoder is purely linear, and $g_m$ is also linear ($g_m(Z) = Z V_m^\top$).
2.  **LEND (Linear Encoder, Nonlinear Decoder):** $h_m = \text{Id}$, but $g_m$ is a deep neural network. This preserves the linear interpretability of $V_m$ while allowing the model to "filter out" nonlinear measurement noise.
3.  **NED (Nonlinear Encoder Decoder):** $h_m$ is a nonlinear "head" (MLP). This provides maximum representational power while still allowing for a first-layer audit via $V_m$.

### 2.1 Formalizing the Audit Trail

The contract is quantified by the triplet $(\mathcal{V}, \mathcal{B}_{align}, \mathcal{B}_{shared})$:
- **$\mathcal{V} = \{V_m\}$**: The primary linear weights.
- **$\mathcal{B}_{align}$**: The alignment $R^2$ between the linear scores ($X_m V_m$) and the deep latents ($f_m(X_m)$).
- **$\mathcal{B}_{shared}$**: The attribution of the global consensus $U$ back to the modality-specific linear components.

## 3. Manifold Optimization: The NSA-Flow

Maintaining the Stiefel manifold constraint ($V_m^\top V_m = I_K$) during stochastic gradient descent is non-trivial. We utilize the **Non-negative Stiefel Approximating Flow (NSA-Flow)**, which implements a gradient-projection-retraction cycle.

### 3.1 Scheduled Projection Alpha

To ensure training stability, we employ a **Scheduled Projection Alpha** ($\alpha_t$) that implements a straight-through estimator (STE) homotopy. During training at epoch $t$, the active basis $V_{active}$ is:

$$ V_{active} = V_{raw} + \alpha_t \cdot (V_{projected} - V_{raw}).\text{detach}() $$

where $V_{projected} = \text{Retract}(V_{raw})$ is the SVD-based orthogonalization. As $\alpha_t$ ramps from 0 to 1, the model smoothly transitions from unconstrained Euclidean exploration to strict manifold adherence.

### 3.2 NSA-Flow Mechanism

The following diagram illustrates the internal logic of the NSA-Flow as implemented in `pysimlr`.

```{mermaid}
graph LR
    subgraph NSA_Mechanism [NSA Flow Logic]
        direction LR
        V_raw[Raw Weights V_raw] --> Flow[NSA Flow: nsa_flow_orth / NSAFlowLayer]
        Flow -->|Retraction & Orthogonality| Pos[Positivity Constraint]
        Pos -->|Non-negativity| Sparse[Quantile Sparsification]
        Sparse -->|Sparsity| V_proj[Projected Basis V]
        
        V_raw -.-> Mix{Scheduled Mixing}
        V_proj -.-> Mix
        Mix -->|Training Basis| V_active[V_active]
    end
    X[Input Data X] -->|X @ V_active| Z[Latent Scores Z]
```

## 4. Shared/Private Partitioning (NEDPP)

The NEDPP architecture decomposes the latent space into a shared consensus component $Z^{(S)}$ and a modality-specific private component $Z^{(P)}$.

### 4.1 Cross-Covariance Decorrelation

To ensure the private components do not capture shared information, we enforce statistical independence via a cross-covariance penalty:
$$ \mathcal{L}_{\text{cross}} = \| C(Z^{(S)}, Z^{(P)}) \|_F^2 $$

### 4.2 Shared-First Optimization Schedule

To prevent "modality starvation" (where one modality dominates the consensus), we employ a hierarchical training schedule. For the first $N_{\text{shared}}$ epochs, the private encoders are frozen ($Z^{(P)} \equiv 0$). This forces the shared encoders to capture the maximum cross-modal variance before the private encoders are allowed to "soak up" idiosyncratic noise.
"""
    with open("paper/02_methods.qmd", "w") as f:
        f.write(content)

if __name__ == "__main__":
    write_methods()
