"""svd.top_right_singular_vectors: ARPACK top-k for large views, ba_svd below the size gate."""
import torch

from pysimlr.svd import TRUNCATED_SVD_MIN_DIM, ba_svd, top_right_singular_vectors


def _x(n, p, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) @ torch.randn(k, p, generator=g)
            + 0.5 * torch.randn(n, p, generator=g))


def test_large_view_matches_the_full_svd_columnwise():
    x = _x(1200, 1500)
    assert min(x.shape) >= TRUNCATED_SVD_MIN_DIM
    v, ref = top_right_singular_vectors(x, 4), ba_svd(x, nu=0, nv=4)[2]
    cos = (v * ref).sum(0).abs()
    assert torch.all(cos > 1 - 1e-6), cos
    # deterministic sign: each column's largest-magnitude entry is positive
    assert torch.all(v[v.abs().argmax(0), torch.arange(4)] > 0)


def test_small_view_is_the_full_svd_unchanged():
    x = _x(300, 200)
    assert torch.equal(top_right_singular_vectors(x, 3), ba_svd(x, nu=0, nv=3)[2])
