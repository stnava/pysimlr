"""Averaging latents across views requires a common sign convention.

A latent factorisation is defined only up to a sign per component: if ``u_j``
explains a view then so does ``-u_j`` with the loading negated. Two views can
therefore recover *the same* factors with opposite signs, and a consensus that
averages them cancels the signal instead of pooling it.

This was live in five places -- `avg` and `newton` mixing, the prune probe in
`compute_shared_consensus`, and the leave-one-out average inside every copy of
`update_mai`. Measured on a planted two-factor signal, the two views' latents
were correlated at -0.927 and -0.959 and the plain mean's per-column standard
deviation collapsed from ~1.0 to 0.175 and 0.121, taking the consensus R^2
from 0.795 to 0.023 while the per-view latents scored 0.784 and 0.785.

What made it hard to see is that it is silent and seed-dependent: the same
model scored 0.034, 0.347 and 0.183 over three seeds and a sibling scored
0.701, 0.034 and 0.787. It reads as an unstable method, not as arithmetic.
Concatenating estimators (`svd`, `pca`, `ica`) never showed it, because a sign
flip is absorbed by the basis they fit.
"""
import numpy as np
import pytest
import torch

from pysimlr.consensus import compute_shared_consensus
from pysimlr.utils import align_column_signs

K = 2


# --------------------------------------------------------------------------
# the utility, on cases whose answer is known by construction
# --------------------------------------------------------------------------
def test_a_negated_copy_is_flipped_back():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(50, K, generator=g)
    out = align_column_signs([a, -a])
    assert torch.allclose(out[0], a)
    assert torch.allclose(out[1], a), "the negated copy was not flipped back"
    mean = torch.stack(out).mean(0)
    assert torch.allclose(mean, a, atol=1e-6), "aligned inputs must not cancel"


def test_per_column_signs_are_independent():
    """A view may agree on one component and disagree on another -- measured
    in the wild as a cross-correlation diagonal of [+0.59, -0.76]."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(60, 3, generator=g)
    b = a * torch.tensor([[1.0, -1.0, 1.0]])
    out = align_column_signs([a, b])
    assert torch.allclose(out[1], a, atol=1e-6)


def test_alignment_leaves_agreeing_inputs_untouched():
    g = torch.Generator().manual_seed(2)
    mats = [torch.randn(40, K, generator=g) for _ in range(3)]
    mats = [mats[0], mats[0] * 0.9, mats[0] * 1.1]        # same sign throughout
    out = align_column_signs(mats)
    for a, b in zip(mats, out):
        assert torch.allclose(a, b)


def test_alignment_is_a_relabelling_not_a_rescaling():
    g = torch.Generator().manual_seed(3)
    mats = [torch.randn(40, K, generator=g) for _ in range(4)]
    for a, b in zip(mats, align_column_signs(mats)):
        assert torch.allclose(a.abs(), b.abs()), "magnitudes were changed"


def test_more_than_two_views_are_all_brought_into_agreement():
    g = torch.Generator().manual_seed(4)
    base = torch.randn(50, K, generator=g)
    mats = [base, -base, base, -base, -base]
    out = align_column_signs(mats)
    mean = torch.stack(out).mean(0)
    assert torch.allclose(mean.abs(), base.abs(), atol=1e-6), (
        "a majority-negated set still cancelled")


def test_a_single_view_is_returned_unchanged():
    a = torch.randn(10, K, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(align_column_signs([a])[0], a)
    assert align_column_signs([]) == []


# --------------------------------------------------------------------------
# the consensus itself: the failure this was found through
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mixing", ["avg", "newton", "svd"])
def test_the_consensus_of_anti_correlated_views_does_not_cancel(mixing):
    """The end-to-end property. Two views carrying the same factors with
    opposite signs must produce a consensus that carries them too."""
    g = torch.Generator().manual_seed(0)
    n = 80
    latent = torch.randn(n, K, generator=g)
    a = latent + 0.1 * torch.randn(n, K, generator=g)
    b = -(latent + 0.1 * torch.randn(n, K, generator=g))     # opposite sign

    u = compute_shared_consensus([a, b], mixing_algorithm=mixing, k=K,
                                 topology="star")
    u = u[0] if isinstance(u, list) else u
    u = u[0] if isinstance(u, tuple) else u

    from sklearn.linear_model import LinearRegression
    y = latent[:, 0].numpy() + 0.5 * latent[:, 1].numpy()
    got = LinearRegression().fit(u.detach().numpy(), y).score(u.detach().numpy(), y)
    solo = LinearRegression().fit(a.numpy(), y).score(a.numpy(), y)
    assert got > 0.8 * solo, (
        f"the {mixing} consensus scored {got:.3f} where one view alone scores "
        f"{solo:.3f}: pooling two views made the answer worse, which is the "
        f"signature of sign cancellation")


def test_the_consensus_is_not_seed_dependent_in_the_way_cancellation_was():
    """Cancellation showed up as wild seed-to-seed variance rather than as an
    error, so this pins the variance rather than any single value."""
    from sklearn.linear_model import LinearRegression
    scores = []
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        n = 80
        latent = torch.randn(n, K, generator=g)
        flip = -1.0 if seed % 2 else 1.0        # alternate which view is negated
        a = latent + 0.1 * torch.randn(n, K, generator=g)
        b = flip * (latent + 0.1 * torch.randn(n, K, generator=g))
        u = compute_shared_consensus([a, b], mixing_algorithm="newton", k=K,
                                     topology="star")
        u = u[0] if isinstance(u, list) else u
        u = u[0] if isinstance(u, tuple) else u
        y = latent[:, 0].numpy() + 0.5 * latent[:, 1].numpy()
        scores.append(LinearRegression().fit(u.detach().numpy(), y)
                      .score(u.detach().numpy(), y))
    assert min(scores) > 0.8, f"a seed collapsed: {np.round(scores, 3).tolist()}"
    assert float(np.std(scores)) < 0.1, (
        f"scores vary with the seed far more than the noise warrants: "
        f"{np.round(scores, 3).tolist()}")
