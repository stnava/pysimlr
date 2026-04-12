import os

def write_methods():
    content = r"""
# Methods

In this section, we present the mathematical foundations and algorithmic details of our multi-modal integration framework. We begin with the classical linear formulation of Similarity-driven Multi-view Linear Reconstruction (SiMLR), proceed to the deep extensions (LEND and NED), and detail the critical constraint mechanisms, notably the NSA (Norm-preserving Spherical Addition) Flow, which enables stable optimization on the Stiefel manifold within deep learning contexts.

## 1. Mathematical Notation and Problem Formulation

Let $X_1, X_2, \ldots, X_M$ denote $M$ distinct data modalities (views) collected from the same $n$ samples. For the $m$-th modality, $X_m \in \mathbb{R}^{n \times d_m}$, where $d_m$ is the number of features. We assume, without loss of generality, that each $X_m$ is column-centered.

Our goal is to discover a low-dimensional consensus space $U \in \mathbb{R}^{n \times k}$ (where $k \ll \min_m d_m$) that captures the shared variance across all modalities. To achieve this, we seek view-specific encoding functions $f_m: \mathbb{R}^{d_m} \to \mathbb{R}^k$ that map the original features into latent representations $Z_m = f_m(X_m)$. 

The core objective is to maximize the alignment between these latent representations $Z_m$ and the consensus $U$, subject to constraints that ensure identifiability and prevent trivial scaling.

## 2. Classical SiMLR: The Linear Foundation

In the classical SiMLR framework, the encoding functions are strictly linear: $f_m(X_m) = X_m V_m$, where $V_m \in \mathbb{R}^{d_m \times k}$ is the projection matrix (also referred to as the feature weights or loadings).

### 2.1 The Objective Function

The SiMLR objective combines a data fidelity (similarity) term with regularization terms designed to induce sparsity and structural priors. The total energy function is minimized:

$$ \min_{V_m, U} E_{total} = \sum_{m=1}^M \alpha_m E_{sim}(X_m V_m, U) + \lambda \sum_{m=1}^M E_{reg}(V_m) $$

where:
- $E_{sim}$ measures the distance (or negative similarity) between the view-specific latent space $Z_m = X_m V_m$ and the consensus $U$.
- $\alpha_m$ are modality-specific weights, allowing the model to prioritize views with higher signal-to-noise ratios.
- $E_{reg}$ imposes structural constraints on the projection matrices $V_m$, primarily $L_1$ regularization to induce sparsity.
- $\lambda$ controls the overall strength of the regularization.

### 2.2 Similarity Metrics

The choice of $E_{sim}$ dictates how "alignment" is measured. SiMLR supports several variations:

1.  **Mean Squared Error (MSE):** $E_{sim}(Z_m, U) = \| Z_m - U \|_F^2$. This is the standard Euclidean distance, heavily penalizing large deviations.
2.  **Cosine Similarity:** $E_{sim}(Z_m, U) = - \text{tr}(Z_m^\top U)$. This focuses on angular alignment, often more robust to varying scales across modalities.
3.  **Centered Cosine (Correlation):** Equivalent to cosine similarity but enforced on column-centered matrices.

### 2.3 Manifold Constraints

To prevent dimensional collapse (e.g., all samples mapping to a single point) and to ensure that the $k$ components capture distinct, non-redundant information, we impose strict orthogonality constraints on the projection matrices:

$$ V_m^\top V_m = I_k \quad \forall m \in \{1, \ldots, M\} $$

This constraint defines the **Stiefel Manifold**, $\mathcal{S}_{d_m, k}$. Optimizing the energy function subject to this constraint requires specialized Riemannian optimization techniques. Classical SiMLR employs a projected gradient descent algorithm where the Euclidean gradient $\nabla E$ is computed, projected onto the tangent space of the Stiefel manifold, and then a retraction (e.g., via QR decomposition or SVD) is applied to step along the manifold.

## 3. Deep SiMLR Architectures

While the linear framework is robust and highly interpretable, it fails when the generative process linking the true consensus $U$ to the observed modalities $X_m$ is highly nonlinear. To address this, we extend SiMLR using deep neural networks, relaxing the linear encoding assumption in structured ways.

### 3.1 Intuitive Figure of the Methodology

The following figure illustrates the spectrum of architectures within the SiMLR framework, highlighting the trade-off between interpretability (enforced by linear encoders) and expressivity (enabled by deep decoders).

```{python}
#| label: fig-methodology
#| fig-cap: "Overview of the SiMLR architectural variants. (Left) Classical SiMLR relies on strictly linear encoders and consensus similarity. (Center) LEND introduces nonlinear decoders to model complex noise while retaining linear encoders for interpretability. (Right) NED fully embraces nonlinear encoders and decoders for maximum capacity."
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

def draw_model(ax, title, enc_type, dec_type=None):
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')
    
    # Inputs
    ax.add_patch(patches.Rectangle((0.1, 0.7), 0.2, 0.1, fill=True, color='lightblue'))
    ax.text(0.2, 0.75, '$X_1$', ha='center', va='center', fontsize=12)
    ax.add_patch(patches.Rectangle((0.7, 0.7), 0.2, 0.1, fill=True, color='lightgreen'))
    ax.text(0.8, 0.75, '$X_2$', ha='center', va='center', fontsize=12)
    
    # Encoders
    enc_color = 'orange' if enc_type == 'Linear' else 'red'
    ax.add_patch(patches.Rectangle((0.1, 0.4), 0.2, 0.1, fill=True, color=enc_color, alpha=0.6))
    ax.text(0.2, 0.45, enc_type + ' Enc', ha='center', va='center', fontsize=10)
    ax.annotate('', xy=(0.2, 0.5), xytext=(0.2, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    ax.add_patch(patches.Rectangle((0.7, 0.4), 0.2, 0.1, fill=True, color=enc_color, alpha=0.6))
    ax.text(0.8, 0.45, enc_type + ' Enc', ha='center', va='center', fontsize=10)
    ax.annotate('', xy=(0.8, 0.5), xytext=(0.8, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Latents to Consensus
    ax.add_patch(patches.Circle((0.5, 0.25), 0.1, fill=True, color='purple', alpha=0.5))
    ax.text(0.5, 0.25, 'Consensus\n$U$', ha='center', va='center', fontsize=12)
    ax.annotate('', xy=(0.4, 0.25), xytext=(0.3, 0.4), arrowprops=dict(arrowstyle="->", lw=1.5, ls='--'))
    ax.annotate('', xy=(0.6, 0.25), xytext=(0.7, 0.4), arrowprops=dict(arrowstyle="->", lw=1.5, ls='--'))
    
    # Decoders (if any)
    if dec_type:
        dec_color = 'red' if dec_type == 'Nonlinear' else 'orange'
        ax.add_patch(patches.Rectangle((0.1, -0.1), 0.2, 0.1, fill=True, color=dec_color, alpha=0.6))
        ax.text(0.2, -0.05, dec_type + ' Dec', ha='center', va='center', fontsize=10)
        ax.annotate('', xy=(0.2, -0.0), xytext=(0.4, 0.2), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate('', xy=(0.2, -0.2), xytext=(0.2, -0.1), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.text(0.2, -0.25, "$\hat{X}_1$", ha='center', va='center')
        
        ax.add_patch(patches.Rectangle((0.7, -0.1), 0.2, 0.1, fill=True, color=dec_color, alpha=0.6))
        ax.text(0.8, -0.05, dec_type + ' Dec', ha='center', va='center', fontsize=10)
        ax.annotate('', xy=(0.8, -0.0), xytext=(0.6, 0.2), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate('', xy=(0.8, -0.2), xytext=(0.8, -0.1), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.text(0.8, -0.25, "$\hat{X}_2$", ha='center', va='center')
        
    ax.set_ylim(-0.4, 1.0)

draw_model(axes[0], "Classical SiMLR", "Linear")
draw_model(axes[1], "LEND Model", "Linear", "Nonlinear")
draw_model(axes[2], "NED Model", "Nonlinear", "Nonlinear")

plt.tight_layout()
plt.show()
```

### 3.2 LEND: Linear Encoder, Nonlinear Decoder

The LEND architecture is designed for scenarios where feature-level interpretability is paramount (i.e., we must know exactly which original variables constitute the latent space), but the data is corrupted by complex, nonlinear noise.

In LEND, the encoder remains strictly linear and constrained to the Stiefel manifold: $Z_m = X_m V_m$, subject to $V_m^\top V_m = I_k$. 

However, unlike classical SiMLR which relies solely on a similarity loss, LEND incorporates a deep reconstruction pathway. The consensus $U$ (or the individual $Z_m$) is passed through a nonlinear decoder network $g_m$: $\hat{X}_m = g_m(U; \theta_m)$.

The loss function for LEND is a weighted combination of similarity, reconstruction, and regularization:

$$ \mathcal{L}_{LEND} = \sum_{m=1}^M \left[ \beta_m \| X_m - g_m(Z_m; \theta_m) \|_F^2 + \alpha_m E_{sim}(Z_m, U) \right] + \lambda \sum_{m=1}^M E_{reg}(V_m) $$

This formulation forces the linear encoder to extract a signal robust enough that the nonlinear decoder can accurately reconstruct the original modality. The nonlinear decoder acts as a flexible buffer, absorbing complex noise and preventing it from distorting the linear projection matrix $V_m$.

### 3.3 NED: Nonlinear Encoder Decoder

When the generative mapping from the true latent space to the observed data is fundamentally nonlinear, a linear encoder will fail to capture the shared variance. The NED architecture replaces the linear matrix $V_m$ with a deep neural network encoder $f_m(X_m; \phi_m)$.

$$ Z_m = f_m(X_m; \phi_m), \quad \hat{X}_m = g_m(Z_m; \theta_m) $$

The loss function is analogous to LEND, but the optimization occurs over network parameters $\phi_m$ rather than projection matrices:

$$ \mathcal{L}_{NED} = \sum_{m=1}^M \left[ \beta_m \| X_m - \hat{X}_m \|_F^2 + \alpha_m E_{sim}(Z_m, U) \right] $$

While NED maximizes representational capacity and latent alignment, it sacrifices the direct interpretability of $V_m$. Feature attribution in NED requires post-hoc explanation techniques (e.g., Integrated Gradients or SHAP), which are inherently more complex and less exact than reading weights from a linear projection.

## 4. Optimization Details: The NSA Flow

A critical technical challenge in training LEND (and linear SiMLR via PyTorch) is optimizing the projection matrices $V_m$ subject to the Stiefel manifold constraint $V_m^\top V_m = I_k$ using standard deep learning optimizers (e.g., Adam or SGD).

Standard optimizers update weights using Euclidean gradients. If $V_m \in \mathcal{S}_{d,k}$, a simple gradient step $V_m - \eta \nabla V_m$ will immediately pull $V_m$ off the manifold. Repeated unconstrained updates rapidly lead to poorly conditioned matrices and dimensional collapse.

To resolve this, we employ the **NSA (Norm-preserving Spherical Addition) Flow**. The core idea is to decouple the optimization variable from the constrained manifold. We maintain an unconstrained weight matrix $W_m \in \mathbb{R}^{d \times k}$. During the forward pass, we dynamically project $W_m$ onto the Stiefel manifold to produce $V_m$.

The projection utilizes a smooth, differentiable retraction, typically based on the Singular Value Decomposition (SVD):
1. Compute the SVD of the unconstrained weights: $W_m = U_w \Sigma_w V_w^\top$.
2. The orthogonalized projection is $V_m = U_w V_w^\top$.

This ensures that the forward pass always uses perfectly orthogonal columns. During backpropagation, the gradients flow through this SVD operation. However, SVD gradients can become unstable if singular values collapse or become degenerate. The NSA Flow mitigates this by applying a gentle regularization to the singular values or by utilizing a computationally cheaper but stable iterative orthogonalization (e.g., Newton-Schulz iterations) as a surrogate during the backward pass.

In our implementation, the strictness of this manifold enforcement can be modulated by a parameter $w$ (the NSA flow weight), allowing a smooth interpolation between unconstrained optimization and strict Stiefel manifold traversal.

## 5. Handling Structured Noise: Shared/Private Architectures

In many real-world datasets, a modality contains structured variance that is entirely orthogonal to the other modalities. If this "private" variance is large, it can overwhelm the cross-modality similarity loss, causing the consensus space $U$ to prioritize one view's noise over the shared signal.

To combat this, we introduce the **Shared/Private (NEDPP)** architecture. The latent space for each modality is explicitly partitioned into a shared component $Z_{m, shared}$ and a private component $Z_{m, private}$.

1.  **Shared Encoders:** Extract $Z_{m, shared}$ from $X_m$. These are pushed to align with the global consensus $U$.
2.  **Private Encoders:** Extract $Z_{m, private}$ from $X_m$. These are *penalized* if they correlate with $U$ or with the private spaces of other modalities.
3.  **Decoders:** The reconstruction of $X_m$ is generated from the concatenation of the shared consensus and the modality's private space: $\hat{X}_m = g_m([U; Z_{m, private}])$.

By providing an explicit "escape hatch" for modality-specific noise, the Shared/Private architecture ensures that the consensus space $U$ remains pure and exclusively represents cross-modality phenomena.
"""
    with open("paper/02_methods.qmd", "w") as f:
        f.write(content)

if __name__ == "__main__":
    write_methods()
"""