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

While highly effective at capturing complex structures, these deep models suffer from the \"black box\" problem. The nonlinear composition of layers makes it nearly impossible to attribute latent variations back to specific input features in a meaningful, quantitative manner. For scientific applications where the ultimate goal is feature discovery (e.g., identifying a gene regulatory network), this lack of transparency is often a flaw.

### The First-Layer Contract: A Mechanical Bridge

To resolve the conflict between the depth required for complex pattern recognition and the transparency required for scientific discovery, we propose the **First-Layer Contract**. This "contract" is a structural constraint imposed on the architecture:

1.  **Linear Entry:** The very first operation performed on the input features must be a linear projection $Z = X V$.
2.  **Manifold Constraint:** The projection matrix $V$ must reside on the Stiefel manifold ($V^\top V = I$), ensuring that the discovered components are orthonormal and non-redundant.
3.  **Sparsity:** To facilitate feature selection, $V$ should be sparse, focusing the model on informative variables.

By enforcing this contract, we create a **mechanical bridge** between linear and deep learning. No matter how deep or nonlinear the subsequent "heads" or decoders become, the path from the original measurement space to the first internal representation is governed by a strictly interpretable linear map.

### Mechanical vs. Post-hoc Interpretability

Modern machine learning often relies on **Post-hoc Interpretability** methods like SHAP or LIME. While powerful, these methods operate by treating the model as a black box and observing how perturbations to the input affect the output. In contrast, SiMLR's **Mechanical Interpretability** ensures that interpretability is a structural property of the model, not an afterthought.

#### Visualizing the Interpretability Gap

The following figure illustrates the fundamental difference in how these two paradigms derive meaning from a model.

```{mermaid}
graph LR
    subgraph PostHoc [Post-hoc: Guessing Importance]
        direction TB
        InputP[Input X] --> BlackBox[Black Box Model] --> OutputP[Output Y]
        BlackBox -.->|Perturbations| Explainer[SHAP/LIME Explainer]
        Explainer -->|Local Surrogate| Guess[Feature Importance Estimate]
    end

    subgraph Mechanical [Mechanical: Enforcing Importance]
        direction TB
        InputM[Input X] -->|Linear Entry| SparseWeights["Sparse Weights V"]
        SparseWeights --> Latent[Latent Z] --> DeepHeads[Nonlinear Heads/Decoders]
        Latent -.->|Direct Readout| Meaning["Importance = Weights"]
    end
```

### Architecture Visualizations

To understand the spectrum of models explored in this work, we present high-level architectural diagrams representing the core paradigms.

#### 1. Classical SiMLR (Linear)

The classical SiMLR architecture is purely linear. The objective is to maximize the similarity between the latent projections $Z_m = X_m V_m$.

```{mermaid}
graph TD
    subgraph View1 [Modality 1]
        X1[Input X1] -->|"X1 * V1"| Z1[Latent Z1]
    end
    subgraph View2 [Modality 2]
        X2[Input X2] -->|"X2 * V2"| Z2[Latent Z2]
    end
    Z1 <-->|Maximize Similarity| Z2
    Z1 --> U[Consensus U]
    Z2 --> U
```

#### 2. LEND (Linear Encoder, Nonlinear Decoder)

LEND retains the strictly linear encoder of classical SiMLR but replaces the linear reconstruction with a deep, nonlinear decoder.

```{mermaid}
graph TD
    X1[Modality 1: X1] -->|Linear Encoder| Z1[Latent Z1]
    X2[Modality 2: X2] -->|Linear Encoder| Z2[Latent Z2]
    
    Z1 --> U[Consensus U]
    Z2 --> U
    
    U -->|Nonlinear Decoder f1| X1_hat[Reconstruction X1']
    U -->|Nonlinear Decoder f2| X2_hat[Reconstruction X2']
```

#### 3. NED (Nonlinear Encoder Decoder)

NED adds nonlinear \"heads\" after the interpretable linear encoder, maximizing representational power.

```{mermaid}
graph TD
    X1[Modality 1: X1] -->|Linear Entry| H1[Nonlinear Head h1]
    H1 --> Z1[Latent Z1]
    
    X2[Modality 2: X2] -->|Linear Entry| H2[Nonlinear Head h2]
    H2 --> Z2[Latent Z2]
    
    Z1 <-->|Similarity Loss| Z2
    Z1 --> U[Consensus U]
    Z2 --> U
    
    U -->|Nonlinear Decoder f1| X1_hat[Reconstruction X1']
    U -->|Nonlinear Decoder f2| X2_hat[Reconstruction X2']
```

#### 4. Shared/Private Partitioning (NEDPP)

The Shared/Private architecture (NEDPP) explicitly partitions the latent space into a shared consensus component $U$ and modality-specific private components $P_m$.

```{mermaid}
graph TD
    subgraph View1 [Modality 1]
        X1[Input X1] -->|Linear Entry| SH1[Shared Head]
        SH1 --> Z_shared1[Z_shared 1]
        X1 -->|Private Encoder| Z_priv1[Z_private 1]
    end
    
    subgraph View2 [Modality 2]
        X2[Input X2] -->|Linear Entry| SH2[Shared Head]
        SH2 --> Z_shared2[Z_shared 2]
        X2 -->|Private Encoder| Z_priv2[Z_private 2]
    end
    
    Z_shared1 <-->|Maximize Similarity| Z_shared2
    Z_shared1 <-->|Independence Penalty| Z_priv1
    Z_shared2 <-->|Independence Penalty| Z_priv2
    
    Z_shared1 --> U[Shared Consensus U]
    Z_shared2 --> U
    
    U --> Comb1((Concat))
    Z_priv1 --> Comb1
    Comb1 -->|Nonlinear Decoder 1| X1_hat[Reconstruction X1']
    
    U --> Comb2((Concat))
    Z_priv2 --> Comb2
    Comb2 -->|Nonlinear Decoder 2| X2_hat[Reconstruction X2']
```
"""
    with open("paper/01_intro_background.qmd", "w") as f:
        f.write(content)

if __name__ == "__main__":
    write_intro_background()
