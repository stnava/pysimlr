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
pytest                 # fast suite (~2.5 min); this is the default
pytest -m slow         # statistical calibration + multi-seed benchmarks only
pytest -m ""           # everything
```

`pytest.ini` sets `pythonpath` and excludes the `slow` marker by default, so no
`PYTHONPATH` export is needed.

## Parity Note

This implementation aims for functional parity with the reference R code. Any identified bugs in the original logic are noted in the source with a `BUG` comment and addressed in the Python version.

## License

Apache-2.0

## What Could Go Wrong? (Audit Findings)

Failure modes to watch for, with the diagnostics that detect them.

1. **Mixing Alpha Starvation**: If `mixing_alpha` (scheduled projection) is not
   annealed to 1.0, the model may perform well while losing interpretability,
   because the latent scores drift off the Stiefel manifold. Verify
   `invariant_orthogonality_defect` in your results.
2. **Latent Collapse**: In deep models (NED/NEDPP), insufficient VICReg
   regularization can leave several latents highly correlated, making the deep
   consensus redundant. Check `u_off_diag_cov` in the training diagnostics.
3. **Sparsity vs. Signal**: Aggressive quantile sparsification in
   `simlr_sparseness` can zero out predictive features before they reach the
   deep layers. Monitor the feature importance map.
4. **Shared-Private Starvation**: In NEDPP models, too short a shared-latent
   schedule lets the private latents absorb the shared variation, giving poor
   consensus recovery.
5. **Orthogonality defect**: measured invariant defect after 40 iterations on a
   synthetic 3-factor problem is roughly `1e-4` for `constraint="NewtonSchulz"`
   and `3e-4` for `constraint="Stiefel"`. The soft `ortho` family is looser
   (~2e-3) by construction — it is a partial projection, not a retraction. If
   your defect exceeds `1e-2`, reduce the step size or raise the constraint
   iteration count.
6. **The orthogonality weight is a hyperparameter, not a free improvement.**
   On a synthetic 3-factor benchmark, latent recovery was *better* without the
   soft orthogonality term (`orthox0`) than with it, and degraded further as the
   weight rose. Tune it against your own criterion.
7. **The energy is only comparable within a run.** `simlr` reports the
   lowest-energy iterate (`best_energy`, `best_iteration`), because alternating
   minimization is not monotone: `u` is recomputed after each sweep over the
   views, so the total typically falls steeply for a few iterations and then
   drifts slowly upward. Convergence usually happens within 5–10 iterations;
   the default `iterations=100` mostly spends compute after the optimum.
8. **Interpretability R² is reported both in-sample and cross-validated.** Use
   the `*_cv` fields. The in-sample values come from an effectively
   unregularized fit and approach 1 whenever the predictor is wide relative to
   the sample count — on pure noise with n=40 and p=35, in-sample R² is 0.92
   while cross-validated R² is −367.
9. **Results depend on whether the optional NSA-Flow backend is installed.**
   `pysimlr.nsa_backend.backend_report()` tells you which backend resolved;
   record it alongside your results.

10. **Check `simlr` against its own initialization on your data.** On
    well-conditioned multi-view data the SVD initialization is already at the
    optimum of the objective, and 40 iterations change latent recovery by
    ~0.0002. The iteration earns its keep only where the shared signal is not
    each view's leading singular direction — in a regime where it is buried
    under view-private structure, 60 iterations improved recovery from 0.076 to
    0.095. `result["best_iteration"]` tells you which iterate was actually
    returned; if it is 0, the optimizer found nothing better than where it
    started. See `CORRECTNESS_AUDIT.md` for the measurements.

## Statistical notes

- `simlr_perm` and `paths.permutation_test` report add-one permutation
  p-values, `(1 + #{null ≥ observed}) / (1 + n_perms)`. The smallest attainable
  value is `1 / (1 + n_perms)`, so choose `n_perms` for the resolution you need.
- `energy_type="acc"` and `"logcosh"` sum over the **full** K×K cross-covariance,
  including off-diagonal terms, so they reward cross-component coupling as well
  as per-component alignment. `"regression"` and `"nc"` are diagonally aligned.
  On a synthetic benchmark the choice made little difference to latent recovery
  (0.981 vs 0.980) but the diagonal-only variant had lower cross-component
  leakage.
