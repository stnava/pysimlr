"""The averaging mixings ("avg", "newton") in `compute_shared_consensus`.

They pair column l of every view with column l of every other view. They used
to ignore `k` (returning the projections' full width), fail inside
`torch.stack` on unequal widths, and return no anchor in training, so a
prediction without a frozen map re-orthogonalised the prediction batch.
"""
import pytest
import torch

from pysimlr.consensus import compute_shared_consensus
from pysimlr.defaults import DEFAULT_MIXING_ALGORITHM

MIXINGS = ["svd", "pca", "avg", "newton"]


def _proj(n=40, k=3, m=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g)
    return [z @ torch.randn(k, k, generator=g) + 0.3 * torch.randn(n, k, generator=g)
            for _ in range(m)]


def test_default_mixing_is_the_package_default():
    import inspect
    sig = inspect.signature(compute_shared_consensus)
    assert sig.parameters["mixing_algorithm"].default == DEFAULT_MIXING_ALGORITHM


@pytest.mark.parametrize("mix", MIXINGS)
def test_anchor_replays_the_training_consensus(mix):
    p = _proj()
    u, anchor = compute_shared_consensus(p, mixing_algorithm=mix, k=3, training=True, topology="star")
    assert anchor is not None and anchor.shape == (9, 3)
    replay = compute_shared_consensus(p, mixing_algorithm=mix, k=3, training=False,
                                      anchor=anchor, topology="star")
    assert torch.allclose(u, replay, atol=1e-5)


@pytest.mark.parametrize("mix", ["avg", "newton"])
def test_anchored_prediction_does_not_depend_on_the_batch(mix):
    """Without the anchor the consensus of rows 0-9 alone was re-derived from
    those rows; with it each row's consensus is a fixed linear map (up to the
    per-batch standardization, which the frozen maps in the predict paths own)."""
    p = _proj(n=200)
    _, anchor = compute_shared_consensus(p, mixing_algorithm=mix, k=3, training=True, topology="star")
    big = torch.cat(p, 1)
    raw_full = big @ anchor
    raw_half = torch.cat([q[:50] for q in p], 1) @ anchor
    assert torch.allclose(raw_full[:50], raw_half)


@pytest.mark.parametrize("mix", ["avg", "newton"])
def test_averaging_mixings_reject_a_width_they_cannot_reduce(mix):
    p = _proj(k=4)
    with pytest.raises(ValueError, match="cannot reduce"):
        compute_shared_consensus(p, mixing_algorithm=mix, k=2, topology="star")


@pytest.mark.parametrize("mix", ["avg", "newton"])
def test_averaging_mixings_reject_unequal_widths(mix):
    g = torch.Generator().manual_seed(0)
    p = [torch.randn(30, 3, generator=g), torch.randn(30, 4, generator=g)]
    with pytest.raises(ValueError, match="same width"):
        compute_shared_consensus(p, mixing_algorithm=mix, topology="star")


@pytest.mark.parametrize("mix", ["avg", "newton"])
def test_k_none_means_the_common_width(mix):
    u = compute_shared_consensus(_proj(k=3), mixing_algorithm=mix, k=None, topology="star")
    assert u.shape == (40, 3)


def test_newton_consensus_is_nearly_orthogonal():
    """Ten Newton-Schulz iterations, then centring: measured max off-diagonal
    correlation 0.008 on this case, against 0.625 for "avg"."""
    p = _proj(n=200)
    u = compute_shared_consensus(p, mixing_algorithm="newton", k=3, topology="star")
    c = torch.corrcoef(u.T)
    assert float((c - torch.eye(3)).abs().max()) < 0.02


def test_bare_call_uses_the_package_default_topology():
    """`compute_shared_consensus` defaults to `DEFAULT_TOPOLOGY` ("loo"), the
    topology every fitting entry point uses; star is opt-in."""
    from pysimlr.defaults import DEFAULT_TOPOLOGY
    assert DEFAULT_TOPOLOGY == "loo"
    projs = [torch.randn(30, 2) for _ in range(3)]
    u = compute_shared_consensus(projs, k=2)
    assert isinstance(u, list) and len(u) == 3
    assert isinstance(compute_shared_consensus(projs, k=2, topology="star"), torch.Tensor)
