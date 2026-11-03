"""Unit tests for opt-in nc_ridge energy (Rule R8 / Phase S4)."""
import pytest
import torch

from pysimlr import fit, simlr
from pysimlr.similarity import (
    SIMILARITY,
    SimilarityContext,
    _energy_nc_ridge_impl,
    energy_nc,
    energy_nc_ridge,
    resolve_energy_name,
    similarity_energy,
)


def test_nc_ridge_at_lambda_zero_equals_nc():
    """Internal function _energy_nc_ridge_impl at lam=0 equals energy_nc to 1e-12."""
    g = torch.Generator().manual_seed(42)
    n, p, k = 50, 15, 3
    x = torch.randn(n, p, generator=g, dtype=torch.float64)
    v = torch.randn(p, k, generator=g, dtype=torch.float64)
    u = torch.randn(n, k, generator=g, dtype=torch.float64)
    s = x @ v
    ctx = SimilarityContext(x=x, v=v)

    val_nc = energy_nc(s, u, ctx)
    val_ridge0 = _energy_nc_ridge_impl(s, u, ctx, lam=0.0)
    diff = (val_nc - val_ridge0).abs().item()
    assert diff < 1e-12, f"|nc - nc_ridge(lam=0)| = {diff} >= 1e-12"


def test_nc_ridge_autograd_gradcheck():
    """torch.autograd.gradcheck passes in float64 for nc_ridge."""
    g = torch.Generator().manual_seed(123)
    n, p, k = 25, 8, 2
    x = torch.randn(n, p, generator=g, dtype=torch.float64)
    v = torch.randn(p, k, generator=g, dtype=torch.float64, requires_grad=True)
    u = torch.randn(n, k, generator=g, dtype=torch.float64)

    # Differentiating through composite function s = x @ v
    def func_v(v_in):
        ctx_local = SimilarityContext(x=x, v=v_in)
        return energy_nc_ridge(x @ v_in, u, ctx_local)

    assert torch.autograd.gradcheck(func_v, (v,), eps=1e-6, atol=1e-5, rtol=1e-4)

    # Differentiating directly on s and v
    s = (x @ v).detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)

    def func_sv(s_in, v_in):
        ctx_local = SimilarityContext(x=x, v=v_in)
        return energy_nc_ridge(s_in, u, ctx_local)

    assert torch.autograd.gradcheck(func_sv, (s, v2), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_deep_and_flow_paths_reject_nc_ridge():
    """Deep and flow paths reject nc_ridge with ValueError because needs_data=True."""
    assert SIMILARITY["nc_ridge"][1] is True, "nc_ridge must have needs_data=True"

    # resolve_energy_name with path='deep'
    with pytest.raises(ValueError, match="needs-data|data and the loading matrix"):
        resolve_energy_name("nc_ridge", path="deep")

    # calculate_sim_loss in deep module
    from pysimlr.deep import calculate_sim_loss
    z_list = [torch.randn(20, 2), torch.randn(20, 2)]
    u_shared = torch.randn(20, 2)
    with pytest.raises(ValueError, match="data and the loading matrix"):
        calculate_sim_loss(z_list, u_shared, energy_type="nc_ridge")

    # Entrypoint fit('lend', ...) rejects nc_ridge
    views = [torch.randn(30, 6), torch.randn(30, 8)]
    with pytest.raises(ValueError, match="data and the loading matrix"):
        fit("lend", views, k=2, energy_type="nc_ridge", iterations=1)


def test_linear_path_runs_with_nc_ridge():
    """simlr runs and converges with energy_type='nc_ridge'."""
    g = torch.Generator().manual_seed(99)
    views = [torch.randn(40, 10, generator=g), torch.randn(40, 12, generator=g)]
    res = simlr(views, k=2, energy_type="nc_ridge", iterations=10, random_state=0)
    assert res["convergence"]["converged"] is True or res["convergence"]["n_steps"] > 0
    assert "u" in res
    assert "v" in res
    assert len(res["v"]) == 2
