"""Auxiliary registry terms are refused as a fit's energy_type, on every path,
and still work where they belong (negentropy_weight)."""
import pytest
import torch

from pysimlr import api
from pysimlr.similarity import AUXILIARY_ENERGIES, check_fit_energy, similarity_energy


def _views(seed=0, n=40):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, 6, generator=g), torch.randn(n, 5, generator=g)]


def test_negentropy_has_no_dependence_on_the_consensus():
    """Why it is auxiliary: its value has no path to u at all."""
    g = torch.Generator().manual_seed(1)
    s = torch.randn(64, 2, generator=g, dtype=torch.float64, requires_grad=True)
    u = torch.randn(64, 2, generator=g, dtype=torch.float64, requires_grad=True)
    (gu,) = torch.autograd.grad(similarity_energy("negentropy", s, u), u, allow_unused=True)
    assert gu is None
    (gu_nc,) = torch.autograd.grad(similarity_energy("nc", s, u), u)
    assert float(gu_nc.abs().max()) > 1e-8


@pytest.mark.parametrize("method", ["simlr", "ned", "flow_v"])
def test_auxiliary_energies_are_refused_as_fit_objectives(method):
    assert set(AUXILIARY_ENERGIES) == {"negentropy"}
    with pytest.raises(ValueError, match="negentropy_weight"):
        api.fit(method, _views(), 2, energy_type="negentropy", iterations=1, random_state=0)
    with pytest.raises(ValueError, match="not a fit objective"):
        check_fit_energy("negentropy")


def test_negentropy_still_works_as_an_auxiliary_term():
    r = api.fit("simlr", _views(), 2, negentropy_weight=5.0, iterations=3, random_state=0)
    assert r["convergence"]["n_steps"] >= 1


def test_an_unconverged_fit_warns_and_a_converged_one_does_not():
    import warnings
    import pysimlr
    with pytest.warns(pysimlr.ConvergenceWarning, match="max_iter"):
        api.fit("simlr", _views(), 2, iterations=1, learning_rate=0.01, random_state=0)
    with pytest.warns(pysimlr.ConvergenceWarning, match="raise iterations"):
        api.fit("ned", _views(), 2, iterations=2, random_state=0)
    g = torch.Generator().manual_seed(3)
    z = torch.randn(80, 2, generator=g)
    views = [z @ torch.randn(2, p, generator=g) + 0.1 * torch.randn(80, p, generator=g) for p in (8, 6)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", pysimlr.ConvergenceWarning)
        r = api.fit("simlr", views, 2, random_state=0)  # auto learning rate: the probes must not warn
    assert r["convergence"]["converged"]
