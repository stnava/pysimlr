import torch
import pytest
import numpy as np
from pysimlr.optimizers import (
    create_optimizer,
    backtracking_linesearch,
    bidirectional_linesearch,
    SGD,
    Adam,
    Nadam,
    RMSProp,
    HybridAdam,
    ArmijoGradient,
    BidirectionalArmijoGradient,
    Lookahead,
    BidirectionalLookahead,
    LARS,
    NSAFlowOptimizer,
    TorchNativeOptimizer
)

def test_linesearch():
    def energy_func(v):
        return torch.sum(v**2)
    
    v_current = torch.tensor([1.0, 1.0])
    ascent_gradient = torch.tensor([2.0, 2.0])
    descent_direction = -ascent_gradient
    
    step = backtracking_linesearch(v_current, descent_direction, ascent_gradient, energy_func)
    assert step > 0
    assert energy_func(v_current + step * descent_direction) < energy_func(v_current)

    step_bi, dir_bi = bidirectional_linesearch(v_current, descent_direction, ascent_gradient, energy_func)
    assert step_bi > 0
    assert energy_func(v_current + step_bi * dir_bi) < energy_func(v_current)

def test_all_optimizers():
    v_mats = [torch.randn(10, 2), torch.randn(10, 2)]
    grad = torch.randn(10, 2)
    
    opt_types = [
        "gd", "adam", "nadam", "rmsprop", "hybrid_adam", 
        "armijo_gradient", "bidirectional_armijo_gradient",
        "lookahead", "bidirectional_lookahead", "lars", "nsa_flow",
        "torch_adamw", "torch_adagrad", "torch_nadam"
    ]
    
    def energy_func(v):
        return torch.sum(v**2)

    for opt_type in opt_types:
        opt = create_optimizer(opt_type, v_mats, learning_rate=0.1)
        # Test basic step
        v_next = opt.step(0, v_mats[0], grad)
        assert v_next.shape == v_mats[0].shape
        
        # Test step with energy function (for those that use it)
        v_next_e = opt.step(0, v_mats[0], grad, full_energy_function=energy_func)
        assert v_next_e.shape == v_mats[0].shape

def test_torch_lbfgs_is_removed_and_names_its_replacement():
    """Removed in 0.4.0: it rarely moved from its start and was once certified
    stationary there (docs/tracking.md D6)."""
    import pytest
    with pytest.raises(ValueError, match="nsa_lbfgsb"):
        create_optimizer("torch_lbfgs", [torch.randn(10, 2)])

def test_linesearch_error_handling():
    def broken_energy(v):
        raise ValueError("Broken")
    
    v_current = torch.tensor([1.0, 1.0])
    ascent_gradient = torch.tensor([2.0, 2.0])
    descent_direction = -ascent_gradient
    
    # The current iterate must be evaluable: an error there is raised, not
    # turned into a zero step (which hid the failure).
    with pytest.raises(ValueError, match="Broken"):
        backtracking_linesearch(v_current, descent_direction, ascent_gradient, broken_energy)


def test_linesearch_rejects_a_candidate_it_cannot_evaluate():
    calls = {"n": 0}

    def energy(v):
        calls["n"] += 1
        if calls["n"] > 1 and float(v.sum()) < 1.5:
            raise ValueError("degenerate candidate")
        return float((v ** 2).sum())

    v_current = torch.tensor([1.0, 1.0])
    g = torch.tensor([2.0, 2.0])
    step = backtracking_linesearch(v_current, -g, g, energy)
    assert step >= 0.0

def test_nsa_flow_fallback():
    # Force nsa_flow to None to test fallback to SVD
    v_mats = [torch.randn(10, 2)]
    opt = NSAFlowOptimizer("nsa_flow", v_mats)
    opt.nsa_flow = None
    v_next = opt.step(0, v_mats[0], torch.randn(10, 2))
    # Should be orthogonal
    u, s, vh = torch.linalg.svd(v_next, full_matrices=False)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
