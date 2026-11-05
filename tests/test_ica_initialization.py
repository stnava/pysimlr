"""`initialization_type="ica"`: deflation FastICA from a seeded orthogonal start.

The identity start with the parallel algorithm is a stationary point of FastICA's update for an
orthonormal SVD basis. It stalled or diverged ('slice 0 has a NaN entry') on ADNI-MCI regional
volumes and on the heavy-tailed synthetic views below."""
import warnings

import numpy as np
import pytest
import torch
from sklearn.decomposition import FastICA

import pysimlr
from pysimlr.simlr import _ica_basis_for_view


def _view(seed, n=500, p=160):
    g = np.random.default_rng(seed)
    x = g.standard_normal((n, 3)) @ g.standard_normal((3, p)) + g.lognormal(0, 1, (n, p))
    return (x - x.mean(0)) / x.std(0)


def _old_setting_fails(x, k):
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = FastICA(n_components=k, algorithm="parallel", whiten="unit-variance", fun="logcosh",
                          w_init=np.eye(k), max_iter=500, tol=1e-5, random_state=0).fit_transform(vt[:k].T.copy())
        return not np.isfinite(out).all()
    except ValueError:
        return True


@pytest.mark.parametrize("seed,k", [(3, 5), (5, 6), (5, 8), (6, 5), (7, 6)])
def test_cases_that_broke_the_identity_start_now_initialize(seed, k):
    x = _view(seed)
    assert _old_setting_fails(x, k)                   # the regression these cases guard
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=Warning)   # no FastICA non-convergence either
        v = _ica_basis_for_view(torch.tensor(x).float(), k, sparseness=None)
    assert v.shape == (160, k) and torch.isfinite(v).all()


def test_ica_initialization_is_deterministic_and_fits():
    x = _view(5)
    a = _ica_basis_for_view(torch.tensor(x).float(), 6, sparseness=None)
    b = _ica_basis_for_view(torch.tensor(x).float(), 6, sparseness=None)
    assert torch.equal(a, b)
    g = np.random.default_rng(0)
    xs = [torch.tensor(_view(5)).float(), torch.tensor(_view(6)[:, :40]).float()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit("simlr", xs, 6, initialization_type="ica", iterations=20, random_state=0)
    assert all(torch.isfinite(v).all() for v in res["v"])


def test_broad_sweep_converges():
    """Every view x k of a heavy-tailed sweep initializes without error or non-convergence."""
    bad = []
    for seed in range(20):
        x = torch.tensor(_view(seed)).float()
        for k in (2, 4, 6, 8):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                v = _ica_basis_for_view(x, k, sparseness=None)
            if not torch.isfinite(v).all() or any("converge" in str(m.message) for m in w):
                bad.append((seed, k))
    assert bad == []
