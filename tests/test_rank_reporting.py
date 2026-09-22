"""`simlr` must say how many components it actually returned.

A basis can come back with fewer usable columns than requested while every
other field in the result looks healthy. The gradient step drives columns
together and the projection holds them apart, so a constraint that is too weak
lets them merge: measured on BRCA4 at k=3, the column overlap entering the
projection climbs 0.54, 0.79, 0.97, 1.00 over successive sweeps at nsa_w=0.1
and stays at 1.00.

The reason this needs reporting rather than testing-for-once is that nothing
downstream reveals it. Test accuracy over that same w sweep was flat -- 0.759
to 0.799 from w=0.05 to w=0.99 -- so a rank-2 answer to a rank-3 question
predicts just as well and is silently wrong about the structure, which is the
part SiMLR is for.
"""
import numpy as np
import pytest
import torch

from pysimlr.simlr import basis_rank_report, simlr

K = 3


def _views(n=80, dims=(30, 24), seed=0):
    rng = np.random.default_rng(seed)
    u = np.abs(rng.standard_normal((n, K)))
    out = []
    for p in dims:
        v = np.zeros((p, K))
        blk = p // K
        for j in range(K):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        out.append(torch.tensor(u @ v.T + 0.05 * rng.standard_normal((n, p))).float())
    return out


# --------------------------------------------------------------------------
# the measure itself, on bases whose rank is known by construction
# --------------------------------------------------------------------------
def test_effective_rank_is_exact_for_orthonormal_columns():
    q, _ = torch.linalg.qr(torch.randn(20, K, generator=torch.Generator().manual_seed(0)))
    r = basis_rank_report([q], K, warn=False)
    assert r["effective_rank"][0] == pytest.approx(float(K), abs=1e-6)
    assert r["numerical_rank"][0] == K
    assert r["condition_number"][0] == pytest.approx(1.0, abs=1e-6)
    assert r["max_column_overlap"][0] == pytest.approx(0.0, abs=1e-6)


def test_effective_rank_falls_as_columns_merge():
    """Continuous, which is the point: partial collapse must be visible before
    the integer rank drops."""
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(20, K, generator=g))
    got = []
    for t in (0.0, 0.5, 0.9, 0.999):          # slide column 1 onto column 0
        v = q.clone()
        v[:, 1] = (1 - t) * q[:, 1] + t * q[:, 0]
        got.append(basis_rank_report([v], K, warn=False)["effective_rank"][0])
    assert got == sorted(got, reverse=True), f"not monotone: {got}"
    assert got[0] == pytest.approx(3.0, abs=1e-6)
    assert got[-1] < 2.2, f"a near-duplicated column still scored {got[-1]:.3f}"


def test_a_duplicated_column_is_caught_although_the_integer_rank_is_not():
    """Why the warning keys off the effective rank.

    Measured on BRCA4 at nsa_w=0.1: effective rank 1.53 and 1.90 for two views
    while `numerical_rank` still returned 3, because near-collinear columns
    clear `matrix_rank`'s tolerance. A check on the integer rank alone stays
    silent through exactly the collapse it exists to catch.
    """
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(20, K, generator=g))
    v = q.clone()
    v[:, 1] = q[:, 0] + 1e-4 * q[:, 1]
    r = basis_rank_report([v], K, warn=False)
    assert r["numerical_rank"][0] == K, "premise changed: matrix_rank now catches this"
    assert r["effective_rank"][0] < K - 0.5
    with pytest.warns(RuntimeWarning, match="degenerate basis"):
        basis_rank_report([v], K, warn=True)


def test_a_healthy_basis_does_not_warn():
    q, _ = torch.linalg.qr(torch.randn(20, K, generator=torch.Generator().manual_seed(1)))
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        basis_rank_report([q], K, warn=True)
    assert not [m for m in caught if "degenerate" in str(m.message)]


# --------------------------------------------------------------------------
# reported by `simlr` itself
# --------------------------------------------------------------------------
def test_simlr_reports_the_rank_of_every_view():
    views = _views()
    res = simlr(views, k=K, iterations=6, positivity="positive",
                energy_type="recon_r2", verbose=False)
    for field in ("effective_rank", "numerical_rank",
                  "condition_number", "max_column_overlap"):
        assert field in res, f"{field} missing from the result"
        assert len(res[field]) == len(views)
    for e, v in zip(res["effective_rank"], views):
        assert 0.0 < e <= K + 1e-6


def test_the_reported_rank_describes_the_basis_actually_returned():
    """Not an intermediate iterate: `simlr` returns its best-energy iterate and
    may then consolidate, so the report must be computed after both."""
    views = _views()
    res = simlr(views, k=K, iterations=6, positivity="positive",
                energy_type="recon_r2", consolidate=True, verbose=False)
    recomputed = basis_rank_report([v.detach() for v in res["v"]], K, warn=False)
    assert res["effective_rank"] == pytest.approx(recomputed["effective_rank"])
    assert res["numerical_rank"] == recomputed["numerical_rank"]
