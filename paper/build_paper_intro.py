import os

def write_intro_background():
    content = """
# Introduction and Background

## Introduction

In the era of high-throughput data collection, scientific disciplines increasingly rely on multi-modal datasets to capture comprehensive snapshots of complex systems. For instance, modern biomedical research frequently pairs transcriptomics, proteomics, and epigenomics to understand cellular states, while neuroscience integrates functional magnetic resonance imaging (fMRI) with electroencephalography (EEG) and behavioral metrics. These disparate data types, termed *views* or *modalities*, offer complementary perspectives on an underlying shared reality. However, integrating these modalities remains a formidable challenge due to their heterogeneous noise profiles, varying dimensionalities, and fundamentally different generative processes.

The primary objective of multi-view learning is to extract a shared, low-dimensional representation—a consensus latent space—that captures the overlapping information across modalities while simultaneously filtering out modality-specific noise. Uncovering this consensus is not merely a data compression exercise; it serves as a critical step in hypothesis generation, biomarker discovery, and predictive modeling. When the shared representation correlates with external variables (e.g., disease status, clinical outcomes, or behavioral phenotypes), it provides actionable insights into the underlying mechanisms driving those variables.

Historically, the foundation of multi-view integration was laid by Canonical Correlation Analysis (CCA) [@hotelling1936relations], which seeks linear projections of two modalities that maximize their cross-correlation. While elegant and analytically tractable, classical CCA struggles with high-dimensional datasets where the number of features exceeds the number of samples, leading to overfitting and unstable representations. Subsequent extensions, such as Sparse CCA, introduced $L_1$ penalties to enforce feature selection, yielding more interpretable, sparse projection matrices.

Despite these advancements, linear models inherently assume that the underlying latent factors are linearly related to the observed data. In reality, biological and physical systems frequently exhibit complex, nonlinear behaviors. Deep Canonical Correlation Analysis (DCCA) [@andrew2013deep; @wang2015deep] and deep multi-view autoencoders emerged to address this limitation by replacing linear projections with deep neural networks. These architectures leverage universal function approximators to model highly nonlinear generative processes, often achieving superior predictive performance and latent space alignment. However, this increased flexibility comes at a severe cost: the loss of interpretability. In deep models, the mapping from the original features to the latent space is entangled across multiple hidden layers, obscuring which specific variables (e.g., which genes or brain regions) are driving the shared representation.

This tension between interpretability and expressivity is the central problem addressed in this work. We present **SiMLR** (Similarity-driven Multi-view Linear Reconstruction) and its deep extensions: **LEND** (Linear Encoder, Nonlinear Decoder) and **NED** (Nonlinear Encoder Decoder). Our framework bridges the gap between classical linear methods and modern deep learning by introducing a modular, manifold-constrained architecture that strictly enforces interpretability where it matters—at the feature-to-latent projection—while allowing flexible, nonlinear modeling of the generative process.

## Background

### Multi-View Representation Learning

Multi-view learning encompasses a broad class of algorithms designed to exploit the consensus among multiple representations of the same underlying entities. Formally, given $M$ data matrices $X_1 \in \mathbb{R}^{n \times d_1}, \ldots, X_M \in \mathbb{R}^{n \times d_M}$, where $n$ is the number of samples and $d_m$ is the feature dimension of the $m$-th view, the goal is to find a set of transformations $f_m: \mathbb{R}^{d_m} \to \mathbb{R}^k$ such that the latent representations $Z_m = f_m(X_m)$ are maximally aligned according to some similarity metric, typically subject to constraints that prevent trivial solutions (e.g., all $Z_m = 0$).

#### Linear Frameworks: CCA and Beyond

Classical CCA finds linear transformations $f_m(x) = W_m^\top x$ by maximizing the correlation between $Z_1$ and $Z_2$. This can be generalized to $M > 2$ views using generalized CCA (gCCA). These linear methods are deeply connected to factor analysis models, such as Joint and Individual Variation Explained (JIVE) [@lock2013joint] and Multi-Omics Factor Analysis (MOFA) [@argelaguet2018multi]. These models explicitly decompose the data into shared factors (affecting all views) and private factors (specific to individual views).

The interpretability of these linear models stems from the projection matrices $W_m$. The magnitude of the weights in $W_m$ directly quantifies the contribution of each original feature to the latent dimensions. By imposing sparsity constraints (e.g., via Elastic Net penalties), these matrices act as feature selectors, highlighting the most informative variables.

#### The Nonlinear Frontier: Deep Multi-View Learning

To model nonlinear relationships, researchers extended CCA using deep neural networks (DCCA). In DCCA, $f_m$ is parametrized by a multilayer perceptron (MLP). The network parameters are optimized to maximize the canonical correlation of the top-layer representations. Similarly, Deep CCA Autoencoders (DCCAE) combine the CCA objective with a reconstruction loss to ensure that the latent representations not only align across views but also preserve the information necessary to reconstruct the original inputs.

While highly effective at capturing complex structures, these deep models suffer from the "black box" problem. The nonlinear composition of layers makes it nearly impossible to attribute latent variations back to specific input features in a meaningful, quantitative manner. For scientific applications where the ultimate goal is feature discovery (e.g., identifying a gene regulatory network), this lack of transparency is often a fatal flaw.

### Manifold Optimization and the Stiefel Manifold

A recurrent theme in representation learning—both linear and deep—is the need to constrain the latent space to prevent dimensional collapse and trivial scaling. In classical methods like PCA and CCA, this is achieved by constraining the projection matrices or the latent representations to be orthogonal.

The set of all $k$-dimensional orthonormal frames in $\mathbb{R}^d$ forms a smooth Riemannian manifold known as the **Stiefel Manifold**, denoted as $\mathcal{S}_{d,k} = \{ V \in \mathbb{R}^{d \times k} : V^\top V = I_k \}$.

Optimization on the Stiefel manifold is non-trivial. Standard gradient descent algorithms operate in Euclidean space and will quickly step off the manifold, violating the orthogonality constraints. To maintain orthogonality, optimization must utilize the Riemannian geometry of $\mathcal{S}_{d,k}$ [@edelman1998geometry; @absil2009optimization]. This typically involves:
1. Computing the Euclidean gradient of the objective function.
2. Projecting the gradient onto the tangent space of the Stiefel manifold to obtain the Riemannian gradient.
3. Taking a step along a geodesic (or an approximation thereof, known as a retraction) on the manifold.

In deep learning contexts, strictly enforcing Stiefel manifold constraints on weight matrices is challenging and computationally expensive due to the need for singular value decompositions (SVD) or matrix exponentials during backpropagation. The NSA (Norm-preserving Spherical Addition) Flow, utilized in our framework, offers a stable and efficient mechanism to maintain these constraints dynamically during training.

### Architecture Visualizations and Contextualization

To understand the spectrum of models explored in this work, we present high-level architectural diagrams representing the core paradigms.

#### 1. Classical SiMLR (Linear)

The classical SiMLR architecture is purely linear. Each modality $X_m$ is projected via a sparse, orthogonal matrix $V_m$ onto a latent space. The objective is to maximize the similarity (e.g., inner product or correlation) between the latent projections $Z_m = X_m V_m$, often constrained by structural priors.

```{mermaid}
graph TD
    X1[Modality 1: X1] -->|Linear Projection V1| Z1[Latent Z1]
    X2[Modality 2: X2] -->|Linear Projection V2| Z2[Latent Z2]
    Z1 <-->|Maximize Similarity| Z2
    Z1 --> U[Consensus U]
    Z2 --> U
```

*Context:* SiMLR is heavily inspired by Sparse gCCA and MOFA. It is ideal when the true data generating process is close to linear. Its primary advantage is absolute interpretability: $V_m$ explicitly tells us which features are active.

#### 2. LEND (Linear Encoder, Nonlinear Decoder)

LEND introduces a deliberate architectural asymmetry. It retains the strictly linear encoder of classical SiMLR (maintaining the sparse, orthogonal $V_m$ matrices) to guarantee feature-level interpretability. However, it replaces the linear reconstruction or simple similarity matching with a deep, nonlinear decoder. The decoder maps the linear latent representations back to the original modalities, allowing the model to "explain away" complex, nonlinear noise that would otherwise corrupt the linear encoder.

```{mermaid}
graph TD
    X1[Modality 1: X1] -->|Strict Linear Encoder V1| Z1[Latent Z1]
    X2[Modality 2: X2] -->|Strict Linear Encoder V2| Z2[Latent Z2]
    
    Z1 --> U[Consensus U]
    Z2 --> U
    
    U -->|Nonlinear Decoder f1| X1_hat[Reconstruction X1']
    U -->|Nonlinear Decoder f2| X2_hat[Reconstruction X2']
    
    X1 -.->|Reconstruction Loss| X1_hat
    X2 -.->|Reconstruction Loss| X2_hat
```

*Context:* LEND is a hybrid model. It sits between linear factor analysis and deep autoencoders. It is particularly powerful in biological regimes where the driving biological signal (e.g., a genetic pathway) is linearly additive in its expression, but the measurement technology introduces nonlinear artifacts.

#### 3. NED (Nonlinear Encoder Decoder)

NED fully relaxes the linear constraints on the encoder, employing deep neural networks for both encoding and decoding. While this sacrifices the direct interpretability of $V_m$, it maximizes representational power, making it capable of aligning modalities that have highly complex, nonlinear relationships with the underlying consensus.

```{mermaid}
graph TD
    X1[Modality 1: X1] -->|Nonlinear Encoder g1| Z1[Latent Z1]
    X2[Modality 2: X2] -->|Nonlinear Encoder g2| Z2[Latent Z2]
    
    Z1 <-->|Similarity Loss| Z2
    Z1 --> U[Consensus U]
    Z2 --> U
    
    U -->|Nonlinear Decoder f1| X1_hat[Reconstruction X1']
    U -->|Nonlinear Decoder f2| X2_hat[Reconstruction X2']
```

*Context:* NED is closely related to DCCAE. It serves as our upper-bound baseline for representational capacity. We utilize NED primarily to assess how much performance is sacrificed when we impose the linear encoder constraints of LEND.

#### 4. Shared/Private Partitioning (NEDPP)

A significant challenge in multi-modal integration is the presence of strong structured noise that is specific to a single modality (private factors). If not explicitly modeled, these private factors can leak into the shared consensus space, reducing its relevance to cross-modality phenomena. The Shared/Private architecture (NEDPP) explicitly partitions the latent space into a shared consensus component $U_{shared}$ and modality-specific private components $P_m$.

```{mermaid}
graph TD
    X1[Modality 1] -->|Shared Enc| Z_shared1[Z_shared 1]
    X1 -->|Private Enc| Z_priv1[Z_private 1]
    
    X2[Modality 2] -->|Shared Enc| Z_shared2[Z_shared 2]
    X2 -->|Private Enc| Z_priv2[Z_private 2]
    
    Z_shared1 <-->|Maximize Similarity| Z_shared2
    Z_priv1 <-->|Orthogonality / Independence Penalty| Z_shared1
    Z_priv2 <-->|Orthogonality / Independence Penalty| Z_shared2
    
    Z_shared1 --> U[U_shared]
    Z_shared2 --> U
    
    U --> Comb1((Concat))
    Z_priv1 --> Comb1
    Comb1 -->|Nonlinear Decoder 1| X1_hat[Reconstruction X1']
    
    U --> Comb2((Concat))
    Z_priv2 --> Comb2
    Comb2 -->|Nonlinear Decoder 2| X2_hat[Reconstruction X2']
```

*Context:* Building on concepts from JIVE, the Shared/Private architecture ensures that the consensus space $U$ only captures variance that is truly mutual across views, pushing uncorrelated but highly structured variance into the private spaces $P_m$.

These architectures form a cohesive progression—from strict linearity, to guided nonlinearity, to full deep capacity, and finally to partitioned latent modeling—providing a comprehensive toolkit for multi-modal scientific inquiry.
"""
    with open("paper/01_intro_background.qmd", "w") as f:
        f.write(content)

if __name__ == "__main__":
    write_intro_background()
