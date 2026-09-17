"""A retraction backend's output must be validated, not trusted.

`nsa_flow_orth` returns a result dict whose ``Y`` is an **all-zero matrix** for
some combinations of shape and retraction weight -- measured for a 5x2
non-negative input at w=0.1 and w=0.3 with ``apply_nonneg="hard"``. The default
constraint string is "orthox0.1x1", i.e. exactly w=0.1, and clinical views with
five features at k=2 hit it.

Every call site checked only ``if res['Y'] is not None``. A zero matrix is not
None, so it was accepted: V became identically zero on the first iteration,
every projection became zero, the latent carried no signal, and the first-layer
R-squared was exactly 0.000 -- with nothing raised anywhere. On the Diabetes
benchmark this drove the SiMLR baseline from R^2 0.393 to 0.000, identically
across all 160 configurations.
"""
import pytest
import torch

from pysimlr import simlr
from pysimlr.sparsification import (
    _svd_polar, _usable_retraction, simlr_sparseness,
)


# ----------------------------------------------------- the validator itself

def test_rejects_all_zero_candidate():
    ref = torch.linalg.qr(torch.randn(5, 2))[0]
    assert not _usable_retraction(torch.zeros(5, 2), ref)


def test_rejects_none():
    assert not _usable_retraction(None, torch.randn(5, 2))


def test_rejects_non_finite():
    ref = torch.randn(5, 2)
    for bad in (float("nan"), float("inf")):
        cand = torch.ones(5, 2)
        cand[0, 0] = bad
        assert not _usable_retraction(cand, ref)


def test_rejects_zero_column():
    ref = torch.linalg.qr(torch.randn(5, 3))[0]
    cand = ref.clone()
    cand[:, 1] = 0.0          # a component with no loading at all
    assert not _usable_retraction(cand, ref)


def test_rejects_rank_loss():
    ref = torch.linalg.qr(torch.randn(5, 3))[0]
    cand = ref.clone()
    cand[:, 2] = cand[:, 1]   # collinear -> rank 2 where the input had 3
    assert not _usable_retraction(cand, ref)


def test_accepts_a_healthy_retraction():
    ref = torch.randn(6, 3)
    assert _usable_retraction(_svd_polar(ref), ref)


def test_accepts_wrong_shape_never():
    assert not _usable_retraction(torch.randn(4, 2), torch.randn(5, 2))


# ------------------------------------------- the failure it has to prevent

@pytest.mark.parametrize("p_dim,k", [(5, 2), (5, 3), (5, 5), (10, 2), (25, 3)])
@pytest.mark.parametrize("constraint_weight", [0.1, 0.3, 0.5, 1.0])
def test_simlr_sparseness_never_returns_a_zero_basis(p_dim, k, constraint_weight):
    """The shape/weight grid that triggers the backend's zero output."""
    torch.manual_seed(0)
    v = torch.linalg.qr(torch.randn(p_dim, k))[0]
    out = simlr_sparseness(
        v, constraint_type="ortho", positivity="positive",
        sparseness_quantile=0.5, constraint_weight=constraint_weight,
        constraint_iterations=1,
    )
    assert torch.isfinite(out).all()
    assert float(out.norm()) > 0.0, (
        f"({p_dim}x{k}, w={constraint_weight}) produced an all-zero basis"
    )
    col_norms = torch.linalg.vector_norm(out, dim=0)
    assert (col_norms > 0).all(), f"zero column(s): {col_norms.tolist()}"


@pytest.mark.parametrize("k", [2, 3, 5])
def test_simlr_basis_is_non_degenerate_on_narrow_views(k):
    """Five-feature views at the default constraint -- the clinical shape."""
    torch.manual_seed(0)
    z = torch.randn(200, k)
    x = [z @ torch.randn(k, 5) + 0.3 * torch.randn(200, 5) for _ in range(2)]
    res = simlr(x, k=k, iterations=30, positivity="positive",
                sparseness_quantile=0.5)
    for i, v in enumerate(res["v"]):
        assert float(v.norm()) > 0.0, f"view {i}: V is identically zero"
        assert int(torch.linalg.matrix_rank(v)) == k, (
            f"view {i}: rank {int(torch.linalg.matrix_rank(v))} < k={k}"
        )


def test_narrow_view_basis_still_predicts():
    """The end-to-end symptom: a zero basis gives exactly 0.0 in-sample R^2."""
    torch.manual_seed(0)
    n, k = 300, 2
    z = torch.randn(n, k)
    w = torch.randn(k, 1)
    x = [z @ torch.randn(k, 5) + 0.3 * torch.randn(n, 5) for _ in range(2)]
    y = (z @ w).squeeze(1)

    res = simlr(x, k=k, iterations=30, positivity="positive",
                sparseness_quantile=0.5)
    scores = torch.cat([m @ v for m, v in zip(x, res["v"])], dim=1)
    scores = (scores - scores.mean(0)) / (scores.std(0) + 1e-8)
    design = torch.cat([scores, torch.ones(n, 1)], dim=1)
    beta = torch.linalg.lstsq(design, y.unsqueeze(1)).solution
    pred = (design @ beta).squeeze(1)
    r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    assert r2 > 0.5, f"first-layer R^2 collapsed to {r2:.4f}"


def test_nsa_flow_optimizer_does_not_take_a_zero_step():
    from pysimlr.optimizers import create_optimizer

    torch.manual_seed(0)
    v = torch.linalg.qr(torch.randn(5, 2))[0]
    opt = create_optimizer("nsa_flow", [v], learning_rate=0.05, nsa_w=0.1)
    out = opt.step(0, v, torch.randn(5, 2))
    assert torch.isfinite(out).all()
    assert float(out.norm()) > 0.0, "optimizer returned an all-zero iterate"
