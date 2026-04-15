# pysimlr

A professional, high-performance PyTorch implementation of SIMLR (Structured Identification of Multimodal Low-rank Relationships), based on the reference R implementation.

## Features

- **Multi-modal Low-rank Analysis**: Integrate multiple data modalities into a shared latent space.
- **PyTorch Backend**: High-performance numerical operations with GPU support (where available).
- **Professional Engineering**: Clean, modular structure with comprehensive testing.
- **R Parity**: Functionality aligned with the reference R implementation.
- **Robust SVD**: Automated fallback to randomized SVD for numerical stability.
- **Flexible Optimization**: Multiple optimizers including Hybrid Adam with backtracking line search.

## Installation

```bash
# From source
git clone https://github.com/stnava/pysimlr.git
cd pysimlr
pip install .
```

## Quick Start

```python
import torch
from pysimlr import simlr

# Generate some dummy multimodal data
n, p1, p2 = 100, 50, 40
x1 = torch.randn(n, p1)
x2 = torch.randn(n, p2)

# Run SIMLR
result = simlr([x1, x2], k=5, iterations=20, verbose=True)

# Access shared latent space
u = result['u'] # n x 5
# Access modality-specific basis matrices
v1, v2 = result['v'] # p1 x 5 and p2 x 5
```

## Development and Testing

The project uses `pytest` for unit and parity testing.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pytest tests/
```

## Parity Note

This implementation aims for functional parity with the reference R code. Any identified bugs in the original logic are noted in the source with a `BUG` comment and addressed in the Python version.

## License

Apache-2.0

## What Could Go Wrong? (Audit Findings)
During our code-level interpretability audit, we identified several edge cases and failure modes that users should be aware of:

1. **Mixing Alpha Starvation**: If the `mixing_alpha` (scheduled projection) is not properly annealed to 1.0, the model may achieve high performance but lose its "Mechanical Interpretability" as the latent scores will diverge from the interpretable Stiefel manifold. Always verify the `invariant_orthogonality_defect` in your results.
2. **Latent Collapse**: In deep models (NED/NEDPP), insufficient VICReg regularization (variance/covariance penalty) can lead to latent collapse where multiple latents become highly correlated. This renders the "First-Layer Contract" moot as the deep consensus will be redundant.
3. **Sparsity vs. Signal**: Over-aggressive quantile sparsification in `simlr_sparseness` can zero out critical predictive features before they reach the deep layers. We recommend monitoring the `Feature Importance Map` to ensure key biological/clinical signals are preserved.
4. **Shared-Private Starvation**: In NEDPP models, if the shared latent optimization schedule is too short, the private latents may "swallow" the shared variation to minimize loss quickly, leading to poor consensus recovery.
5. **Orthogonality Defect Thresholds**: While the NSA Flow maintains defects below $10^{-4}$, extreme learning rates can cause retraction instability. If your defect exceeds $10^{-2}$, reduce the step size or increase the NSA Flow iterations.
