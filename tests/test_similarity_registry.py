"""The similarity registry is the single definition of each energy.

These tests exist because the package previously carried two implementations
of the same names. `simlr.calculate_simlr_energy` and `deep.calculate_sim_loss`
both accepted ``energy_type="regression"`` and ``"nc"`` and computed different
functions, so holding ``energy_type`` fixed while varying the model silently
varied the objective. Nothing caught it: each path had tests, and each path was
self-consistent.

The load-bearing test here is `test_identity_equivalence_linear_and_deep`.
"""
import numpy as np
import pytest
import torch

from pysimlr.similarity import (
    SIMILARITY, SimilarityContext, describe_similarity, resolve_energy_name,
    similarity_energy, similarity_gradient, similarity_names,
)


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _case(n=60, p=11, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, generator=g)
    x = x - x.mean(0)
    v = torch.randn(p, k, generator=g)
    u = torch.randn(n, k, generator=g)
    u = u - u.mean(0)
    return x, v, u, x @ v, SimilarityContext(x=x, v=v)


# --------------------------------------------------------------------------
# every gradient is the gradient of its own energy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(SIMILARITY))
def test_gradient_matches_finite_differences(name):
    """The defect this guards: the shipped `regression` gradient dropped the
    `V u'u` term and was *orthogonal* to descent (cosine -0.0045), while six of
    twelve declared objectives returned an identically zero gradient."""
    x, v, u, s, ctx = _case()
    wrt = "v" if name == "recon" else "s"
    base = v if wrt == "v" else s
    g = similarity_gradient(name, s, u, ctx, wrt=wrt)
    assert g.shape == base.shape

    fd = torch.zeros_like(base)
    eps = 1e-6
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            pr = torch.zeros_like(base)
            pr[i, j] = eps
            if wrt == "v":
                ep = similarity_energy(name, x @ (v + pr), u, SimilarityContext(x=x, v=v + pr))
                em = similarity_energy(name, x @ (v - pr), u, SimilarityContext(x=x, v=v - pr))
            else:
                ep = similarity_energy(name, s + pr, u, ctx)
                em = similarity_energy(name, s - pr, u, ctx)
            fd[i, j] = (float(ep) - float(em)) / (2 * eps)

    cos = float((fd * g).sum() / (fd.norm() * g.norm() + 1e-30))
    assert cos > 0.9999, f"{name}: cos(fd, autograd) = {cos}"


@pytest.mark.parametrize("name", sorted(SIMILARITY))
def test_no_term_returns_a_zero_gradient(name):
    """A silently zero gradient is how six objectives optimised nothing."""
    x, v, u, s, ctx = _case()
    wrt = "v" if name == "recon" else "s"
    g = similarity_gradient(name, s, u, ctx, wrt=wrt)
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) > 0.0


# --------------------------------------------------------------------------
# the equivalence that makes cross-model comparison meaningful
# --------------------------------------------------------------------------
def test_identity_equivalence_linear_and_deep():
    """One definition per name, whatever path evaluates it.

    With the warp set to identity a deep model's representation *is* the
    linear model's ``s = XV``, so the two paths must score identically. Before
    the registry they did not, and that is precisely why 'which energy wins'
    was unanswerable from the benchmark.
    """
    from pysimlr.deep import calculate_sim_loss
    x, v, u, s, ctx = _case()
    weights = {"sim": 1.0, "var": 0.0, "collapse": 0.0, "u_var": 0.0}
    # Note "procrustes" maps to the deep path's *own* "procrustes", not to its
    # historical "nc": that name has been unified to the registry's normalised
    # correlation and now warns on the deep path (see `_DEEP_RENAMED`).
    legacy_name = {"align": "regression", "acc": "acc",
                   "nc": "nc", "procrustes": "procrustes",
                   "logcosh": "logcosh"}
    for new, old in legacy_name.items():
        got = float(similarity_energy(new, s, u, ctx))
        want = float(calculate_sim_loss([s], u, old, weights=weights)[0])
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), (
            f"{new} diverges from the deep path's {old}: {got} vs {want}")


def test_recon_matches_the_linear_definition():
    from pysimlr.simlr import calculate_simlr_energy
    x, v, u, s, ctx = _case()
    got = float(similarity_energy("recon", s, u, ctx))
    want = float(calculate_simlr_energy(v, x, u, "regression"))
    assert got == pytest.approx(want, rel=1e-9)


# --------------------------------------------------------------------------
# properties the rest of the library relies on
# --------------------------------------------------------------------------
def test_homogeneity_degrees_are_what_the_gauge_rule_assumes():
    """`sparsification.GAUGE_FREE_ENERGIES` keys off these degrees.

    Only `acc` is degree-1 (unbounded below under rescaling, so the column
    gauge must be fixed). Standardising the ICA contrasts dropped them from
    degree-1 to degree-0, which is a deliberate change from the linear path.
    """
    x, v, u, s, ctx = _case()
    expected_scale_free = {"align", "nc", "procrustes",
                           "logcosh", "exp", "gauss", "kurtosis"}
    for name in sorted(SIMILARITY):
        if name == "recon":
            continue
        e1 = float(similarity_energy(name, s, u, ctx))
        e8 = float(similarity_energy(name, 8.0 * s, u, ctx))
        # Tolerance 1e-6, not machine epsilon: the standardising `eps` inside
        # sqrt(var + eps) makes the scale-free terms invariant only to about
        # 1e-7, which is the price of a denominator that stays differentiable
        # as a column goes constant. Degree-1 terms miss by a factor of 8.
        scale_free = abs(e8 - e1) < 1e-6 * max(1.0, abs(e1))
        assert scale_free == (name in expected_scale_free), (
            f"{name}: scale_free={scale_free}, expected "
            f"{name in expected_scale_free} (E(s)={e1}, E(8s)={e8})")


def test_terms_are_centring_invariant():
    """Every cross-moment term centres internally, so a caller that forgot to
    centre gets a covariance rather than a second moment."""
    x, v, u, s, ctx = _case()
    for name in sorted(SIMILARITY):
        if name == "recon":       # depends on X, not only on s
            continue
        a = float(similarity_energy(name, s, u, ctx))
        b = float(similarity_energy(name, s + 7.0, u + 3.0, ctx))
        assert a == pytest.approx(b, rel=1e-8, abs=1e-10), f"{name} is not centring invariant"


# --------------------------------------------------------------------------
# the ambiguity is refused rather than guessed
# --------------------------------------------------------------------------
def test_regression_is_ambiguous_and_says_so():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_energy_name("regression")
    assert resolve_energy_name("regression", path="linear") == "recon"
    assert resolve_energy_name("regression", path="deep") == "align"


def test_unknown_energy_is_refused():
    with pytest.raises(ValueError, match="unknown similarity"):
        resolve_energy_name("cca")


def test_recon_without_data_is_refused_not_silently_zero():
    _x, _v, u, s, _ctx = _case()
    with pytest.raises(ValueError, match="needs a SimilarityContext"):
        similarity_energy("recon", s, u, None)


def test_every_registry_entry_is_documented():
    for name in similarity_names():
        assert describe_similarity(name).strip()


# --------------------------------------------------------------------------
# degenerate sample sizes are refused, not silently no-ops
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2])
def test_tiny_n_is_refused_for_moment_terms(n):
    """Centring kills these terms below n=3, and that must not be silent.

    Measured before the guard: at n=1 every cross-moment similarity returned
    exactly 0 with an exactly zero gradient; at n=2 the standardised terms had
    gradient magnitude ~1e-6. Either way the loop runs, reports convergence,
    and has moved nothing -- the precise failure this registry replaced.
    """
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 8, generator=g)
    v = torch.randn(8, 3, generator=g)
    u = torch.randn(n, 3, generator=g)
    ctx = SimilarityContext(x=x, v=v)
    for name in similarity_names():
        if name == "recon":       # a reconstruction, not a moment
            similarity_energy(name, x @ v, u, ctx)
            continue
        with pytest.raises(ValueError, match="at least 3 rows"):
            similarity_energy(name, x @ v, u, ctx)


def test_n3_is_enough():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 8, generator=g)
    v = torch.randn(8, 3, generator=g)
    u = torch.randn(3, 3, generator=g)
    ctx = SimilarityContext(x=x, v=v)
    for name in similarity_names():
        assert torch.isfinite(similarity_energy(name, x @ v, u, ctx))
