"""`safe_svd`'s LAPACK-driver fallback.

`torch.linalg.svd` uses `gesdd` (divide-and-conquer) on CPU, with no driver
switch exposed for CPU (the `driver=` kwarg is CUDA-only). Real ADNI+PPMI
data (`scripts/adni_loader.py`) reliably breaks it: after
`_standardize_deep`'s centering/scaling, `torch.linalg.svd` on the T1Hier
view raised `LinAlgError: ... failed to converge` in every `gcca`/`nndsvd`
initialization attempt (100% of runs, both the DX and CDRSB tasks) before
this fallback existed. `gesvd` (via SciPy, which does expose it on CPU)
decomposes the identical matrix fine.

These tests can't easily reproduce that exact failure without the external
data file, so they verify the fallback mechanism directly: monkeypatch
`torch.linalg.svd` to fail once, and check `safe_svd` recovers with a
numerically correct decomposition rather than propagating the error.
"""
import numpy as np
import pytest
import torch

from pysimlr.utils import safe_svd


def _make_matrix(seed=0, n=40, p=15):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((n, p))).double()


def test_safe_svd_matches_torch_on_well_conditioned_input():
    x = _make_matrix()
    u, s, vh = safe_svd(x, full_matrices=False)
    recon = u @ torch.diag(s) @ vh
    assert torch.allclose(recon, x, atol=1e-8)


def test_safe_svd_falls_back_when_torch_raises_linalgerror(monkeypatch):
    x = _make_matrix(seed=1)
    real_svd = torch.linalg.svd
    calls = {"n": 0}

    def flaky_svd(a, full_matrices=False):
        calls["n"] += 1
        raise torch.linalg.LinAlgError(
            "linalg.svd: The algorithm failed to converge because the input "
            "matrix is ill-conditioned or has too many repeated singular "
            "values (error code: 1)."
        )

    monkeypatch.setattr(torch.linalg, "svd", flaky_svd)
    u, s, vh = safe_svd(x, full_matrices=False)
    assert calls["n"] >= 1, "the patched torch.linalg.svd was never called -- test isn't exercising the fallback"
    recon = u @ torch.diag(s) @ vh
    assert torch.allclose(recon, x, atol=1e-6), "scipy gesvd fallback did not reconstruct the input"


def test_safe_svd_still_raises_a_clear_error_if_scipy_also_fails(monkeypatch):
    """Not silently swallowed into a garbage decomposition -- if the
    fallback itself fails, that failure should propagate, not be masked."""
    x = _make_matrix(seed=2)

    def flaky_svd(a, full_matrices=False):
        raise torch.linalg.LinAlgError("does not converge")

    def flaky_scipy_svd(a, full_matrices=True, lapack_driver="gesdd"):
        raise np.linalg.LinAlgError("scipy also failed")

    monkeypatch.setattr(torch.linalg, "svd", flaky_svd)
    import scipy.linalg
    monkeypatch.setattr(scipy.linalg, "svd", flaky_scipy_svd)
    with pytest.raises(Exception):
        safe_svd(x, full_matrices=False)


def test_gcca_and_nndsvd_still_work_under_a_flaky_svd(monkeypatch):
    """Reproduces the actual failure this fallback was added for, without the
    external ADNI file: force the first `torch.linalg.svd` call in
    `_gcca_basis` / `_nndsvd_basis_for_view` to raise, and confirm the
    initializer still returns a usable basis instead of propagating the
    LinAlgError.
    """
    from pysimlr.simlr import initialize_simlr

    torch.manual_seed(0)
    views = [torch.randn(50, 12), torch.randn(50, 9)]

    real_svd = torch.linalg.svd
    state = {"failed_once": False}

    def fail_first_call(a, full_matrices=False):
        if not state["failed_once"]:
            state["failed_once"] = True
            raise torch.linalg.LinAlgError("does not converge")
        return real_svd(a, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "svd", fail_first_call)
    v_mats = initialize_simlr(views, 3, initialization_type="gcca")
    assert state["failed_once"]
    for v in v_mats:
        assert torch.isfinite(v).all()

    state["failed_once"] = False
    v_mats = initialize_simlr(views, 3, initialization_type="nndsvd")
    for v in v_mats:
        assert torch.isfinite(v).all()
