"""
The non-negative initializer: `simlr` must not start from ``abs(V_pca)``.

`simlr_sparseness` enforces a non-negative constraint by reflection, so a
signed PCA initialization reached the first retraction rectified. That matrix
is not merely a worse starting point, it is not orthogonal at all -- and the
retraction is anchored, so it stays near the damaged target instead of
repairing it. `initialize_simlr` therefore fits the basis from the data when
positivity requires non-negativity.
"""

import warnings

import numpy as np
import pytest
import torch

from pysimlr import simlr
from pysimlr.nsa_backend import load_nsa_flow_data
from pysimlr.simlr import initialize_simlr

pytestmark = pytest.mark.skipif(load_nsa_flow_data() is None,
                                reason="no NSA-Flow backend installed")


def orthogonality_defect(v: torch.Tensor) -> float:
    """``||V'V/mean(diag) - I||``; scale-free, zero iff orthogonal and equinormed."""
    gram = v.double().T @ v.double()
    mean_diag = torch.diag(gram).mean()
    if mean_diag <= 0:
        return float("inf")
    return float(torch.norm(gram / mean_diag
                            - torch.eye(gram.shape[0], dtype=gram.dtype)))


def make_case(seed=0, n=100, ps=(36, 27), k=3):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, k, generator=g)
    mats, bases = [], []
    for p in ps:
        v = torch.rand(p, k, generator=g)
        mask = torch.zeros(p, k)
        block = p // k
        for j in range(k):
            mask[j * block:(j + 1) * block, j] = 1.0
        v = v * mask
        v = v / v.norm(dim=0, keepdim=True)
        mats.append(u @ v.T + 0.3 * torch.randn(n, p, generator=g))
        bases.append(v)
    return mats, bases


@pytest.mark.parametrize("positivity", ["positive", "hard", "nonneg"])
def test_nonnegative_positivity_yields_a_nonnegative_initialization(positivity):
    mats, _ = make_case()
    for i, v in enumerate(initialize_simlr(mats, k=3, positivity=positivity)):
        assert (v >= 0).all(), f"view {i} initialized with negative entries"
        assert v.shape == (mats[i].shape[1], 3)
        assert torch.isfinite(v).all()
        assert (v.abs().sum(dim=0) > 0).all(), f"view {i} has a zero column"


def test_signed_positivity_still_uses_the_pca_loadings():
    """The change is scoped to the non-negative case; the signed path is the
    PCA initializer it always was, and the backend's own experiments found
    non-negative starts measurably worse for a signed problem."""
    mats, _ = make_case()
    v_mats = initialize_simlr(mats, k=3, positivity="either")
    assert any(float(v.min()) < 0 for v in v_mats), (
        "a signed problem should keep the signed PCA loadings")


def test_the_initialization_is_far_more_orthogonal_than_rectified_pca():
    """
    The design-independent part of the argument: ``abs(PCA)`` is not an
    orthogonal basis, whatever the data looks like.
    """
    mats, _ = make_case()
    signed = initialize_simlr(mats, k=3, positivity="either")
    fitted = initialize_simlr(mats, k=3, positivity="positive")

    for i in range(len(mats)):
        rectified_defect = orthogonality_defect(signed[i].abs())
        fitted_defect = orthogonality_defect(fitted[i])
        assert fitted_defect < rectified_defect, (
            f"view {i}: fitted defect {fitted_defect:.3f} is no better than "
            f"rectified PCA's {rectified_defect:.3f}")
        # The signed loadings themselves are orthogonal; rectifying is what
        # breaks them, which is the whole reason this path exists.
        assert orthogonality_defect(signed[i]) < 1e-4
        assert rectified_defect > 0.5


def test_a_failing_backend_falls_back_to_the_pca_initializer():
    """The backend is optional; a non-negative request must still return a
    usable basis without it, rather than raising or returning nothing."""
    from unittest.mock import patch

    mats, _ = make_case()
    for return_value in [None]:
        with patch("pysimlr.nsa_backend.load_nsa_flow_data",
                   return_value=return_value):
            v_mats = initialize_simlr(mats, k=3, positivity="positive")
        for i, v in enumerate(v_mats):
            assert v.shape == (mats[i].shape[1], 3)
            assert torch.isfinite(v).all()

    def exploding(*args, **kwargs):
        raise RuntimeError("backend failure")

    with patch("pysimlr.nsa_backend.load_nsa_flow_data",
               return_value=exploding):
        v_mats = initialize_simlr(mats, k=3, positivity="positive")
    for i, v in enumerate(v_mats):
        assert v.shape == (mats[i].shape[1], 3)
        assert torch.isfinite(v).all()


@pytest.mark.parametrize("bad", ["zeros", "wrong_shape", "nan"])
def test_an_unusable_fitted_basis_is_rejected_for_the_pca_fallback(bad):
    from unittest.mock import patch

    mats, _ = make_case()

    def payload(x, k=None, w=None, **kwargs):
        p = x.shape[1]
        if bad == "zeros":
            return {"Y": np.zeros((p, k))}
        if bad == "wrong_shape":
            return {"Y": np.random.rand(p + 5, k)}
        return {"Y": np.full((p, k), np.nan)}

    with patch("pysimlr.nsa_backend.load_nsa_flow_data", return_value=payload):
        v_mats = initialize_simlr(mats, k=3, positivity="positive")

    for i, v in enumerate(v_mats):
        assert v.shape == (mats[i].shape[1], 3), f"view {i} has shape {v.shape}"
        assert torch.isfinite(v).all(), f"view {i} is not finite"
        assert v.abs().sum() > 0, f"view {i} collapsed to zero"


def test_a_wide_view_falls_back_to_pca():
    """``p <= k`` admits no orthogonal basis, so there is nothing to fit."""
    g = torch.Generator().manual_seed(1)
    mats = [torch.randn(50, 3, generator=g)]
    v_mats = initialize_simlr(mats, k=3, positivity="positive")
    assert v_mats[0].shape == (3, 3)
    assert torch.isfinite(v_mats[0]).all()


def test_simlr_threads_positivity_into_the_initializer():
    """End to end: a non-negative `simlr` run must never see a rectified start."""
    mats, _ = make_case()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = simlr(mats, k=3, iterations=6, constraint="orthox0.5",
                    positivity="positive", verbose=False)
    for v in res["v"]:
        assert (v >= -1e-8).all()
        assert torch.isfinite(v).all()
        assert v.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__])
