import torch
import pytest
from pysimlr.benchmarks.synthetic_cases import build_case

def test_linear_footprint_shapes():
    n_samples = 200
    shared_k = 3
    p_list = [50, 40]
    case = build_case("linear_footprint", n_samples=n_samples, shared_k=shared_k, p_list=p_list)
    
    assert len(case["data"]) == 2
    assert case["data"][0].shape == (n_samples, p_list[0])
    assert case["true_u"].shape == (n_samples, shared_k)
    assert len(case["true_v"]) == 2
    assert case["true_v"][0].shape == (p_list[0], shared_k)

def test_nonlinear_shared_shapes():
    n_samples = 200
    case = build_case("nonlinear_shared", n_samples=n_samples)
    assert len(case["data"]) == 3
    assert case["data"][0].shape[0] == n_samples

def test_shared_plus_private_shapes():
    n_samples = 200
    shared_k = 2
    private_k = 2
    case = build_case("shared_plus_private", n_samples=n_samples, shared_k=shared_k, private_k=private_k)
    
    assert case["true_u"].shape == (n_samples, shared_k)
    assert len(case["true_u_private"]) == 2
    assert case["true_u_private"][0].shape == (n_samples, private_k)

def test_reproducibility():
    case1 = build_case("nonlinear_shared", seed=42)
    case2 = build_case("nonlinear_shared", seed=42)
    case3 = build_case("nonlinear_shared", seed=43)
    
    assert torch.allclose(case1["data"][0], case2["data"][0])
    assert not torch.allclose(case1["data"][0], case3["data"][0])

if __name__ == "__main__":
    pytest.main([__file__])
