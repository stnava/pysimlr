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
