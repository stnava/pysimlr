"""How SiMLR starts, and why that is a measurement question rather than a detail.

`initialize_simlr` has always been PCA: a truncated SVD of each view, and so a
function of the data alone. Two consequences went unnoticed for a long time.

* Multi-start experiments were vacuous. Five different seeds produced bases
  identical to ``0.00e+00``, so "do independent starts agree?" -- the empirical
  content of any identifiability claim -- returned 1.0 by construction.
* SiMLR *begins* at PCA. With the default ``optimizer_type="lars"`` and
  ``learning_rate=0.001``, LARS moves ``V`` by exactly ``lr`` of its norm per
  sweep (``step = lr * (||V||/||g||) * g``), so a 15-sweep fit ends within
  ~0.4% of where it started. Measured on a planted basis, `recon_r2` separates
  the initialisation (support 0.708) from the exact truth (support 1.000) by
  0.13%, so there is also nothing to descend toward. A method that starts at
  PCA and moves 0.4% cannot be expected to beat PCA.

These tests pin the distinction so neither fact can quietly return.
"""
import numpy as np
import pytest
import torch

from pysimlr.simlr import initial_basis_for_view, initialize_simlr, simlr


def _views(n=120, dims=(30, 24), k=3, seed=0):
    rng = np.random.default_rng(seed)
    u = np.abs(rng.standard_normal((n, k)))
    out = []
    for p in dims:
        v = np.zeros((p, k))
        blk = p // k
        for j in range(k):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        out.append(torch.tensor(u @ v.T + 0.05 * rng.standard_normal((n, p))).float())
    return out


def test_pca_initialisation_is_deterministic_and_says_so():
    """The default ignores the seed entirely -- that is the documented contract,
    and the reason `"random"` had to be added."""
    views = _views()
    first = None
    for seed in range(4):
        torch.manual_seed(seed)
        np.random.seed(seed)
        got = initialize_simlr(views, 3, positivity="positive")
        if first is None:
            first = got
        for a, b in zip(first, got):
            assert torch.equal(a, b), (
                "PCA initialisation became seed-dependent; multi-start results "
                "recorded against it are no longer comparable")


def test_random_initialisation_actually_differs_across_seeds():
    views = _views()
    bases = []
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        bases.append(initialize_simlr(views, 3, positivity="positive",
                                      initialization_type="random", generator=g))
    for j in range(1, len(bases)):
        assert float((bases[0][0] - bases[j][0]).norm()) > 1e-6, (
            "two random starts coincided; the generator is not being used")


def test_random_initialisation_is_reproducible_from_its_seed():
    views = _views()
    def once(seed):
        g = torch.Generator().manual_seed(seed)
        return initialize_simlr(views, 3, positivity="positive",
                                initialization_type="random", generator=g)
    for a, b in zip(once(11), once(11)):
        assert torch.equal(a, b)


def test_random_initialisation_starts_feasible():
    """Drawn feasible rather than drawn and then repaired.

    A raw draw is neither non-negative-orthogonal nor sparse, so the first
    retraction would move it further than every later sweep combined, and the
    "starting point" a multi-start experiment records would not be the point
    the optimiser began from.
    """
    views = _views()
    g = torch.Generator().manual_seed(3)
    for v in initialize_simlr(views, 3, positivity="positive",
                              initialization_type="random", generator=g):
        assert (v >= 0).all(), "non-negativity was requested and not delivered"
        assert torch.isfinite(v).all()
        assert (v.abs().sum(dim=0) > 0).all(), "a column was dead at init"


def test_random_without_a_generator_is_refused_not_silently_global():
    """Seeding the global stream inside a library call makes a caller's own
    ``seed=`` inert and couples repeated benchmark runs to each other."""
    views = _views()
    with pytest.raises(ValueError, match="needs an explicit `generator`"):
        initialize_simlr(views, 3, initialization_type="random")
    with pytest.raises(ValueError, match="needs a `generator`"):
        initial_basis_for_view(views[0], 3, initialization_type="random")


def test_unknown_initialisation_is_refused():
    with pytest.raises(ValueError, match="not implemented"):
        initialize_simlr(_views(), 3, initialization_type="nmf")


def test_simlr_init_seed_selects_random_and_leaves_the_global_rng_alone():
    """`init_seed` on its own selects `"random"`: handing a seed to the
    deterministic initialiser and getting the same basis back is the trap."""
    views = _views()
    kw = dict(k=3, iterations=3, energy_type="recon_r2",
              positivity="positive", verbose=False)
    a = simlr(views, init_seed=1, **kw)
    b = simlr(views, init_seed=2, **kw)
    c = simlr(views, init_seed=1, **kw)
    assert float((a["v"][0] - b["v"][0]).norm()) > 1e-6, (
        "two init seeds gave the same fit; init_seed did not reach the initialiser")
    assert float((a["v"][0] - c["v"][0]).norm()) == pytest.approx(0.0, abs=1e-6)

    torch.manual_seed(99)
    expected = torch.randn(4)
    torch.manual_seed(99)
    simlr(views, init_seed=5, **kw)
    assert torch.equal(expected, torch.randn(4)), (
        "the fit consumed draws from the global RNG stream")
