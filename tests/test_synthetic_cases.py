import numpy as np
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


# --- The regime where the structural hypothesis is actually true -----------
# Every other generator draws V with torch.randn, so the truth is signed and
# dense and a non-negative method is scored on a basis it cannot represent.

def test_nonnegative_parts_truth_is_nonnegative_sparse_and_disjoint():
    from pysimlr.benchmarks.synthetic_cases import build_case
    case = build_case(kind="nonneg_parts", n_samples=200, seed=42)
    for v in case["true_v"]:
        arr = v.numpy()
        assert (arr >= 0).all(), "loadings must be non-negative"
        support = np.abs(arr) > 1e-9
        # Exactly one component per feature: the supports partition the view.
        assert (support.sum(axis=1) == 1).all(), "column supports must be disjoint"
        assert support.sum() < arr.size, "basis must be sparse"


def test_nonnegative_parts_weak_puts_signal_in_the_weakest_component():
    """Guards against a benchmark where top-variance is the answer for free."""
    from pysimlr.benchmarks.synthetic_cases import build_case
    case = build_case(kind="nonneg_parts_weak", n_samples=400, seed=42)
    u = case["true_u"].numpy()
    y = case["outcome"].numpy()
    variances = u.var(axis=0)
    corrs = np.abs([np.corrcoef(u[:, j], y)[0, 1] for j in range(u.shape[1])])
    assert int(np.argmax(corrs)) == int(np.argmin(variances)), (
        "the outcome must be carried by the lowest-variance latent"
    )
    assert int(np.argmax(corrs)) != int(np.argmax(variances))


def test_nonnegative_parts_is_seed_sensitive():
    from pysimlr.benchmarks.synthetic_cases import build_case
    a = build_case(kind="nonneg_parts", n_samples=200, seed=1)
    b = build_case(kind="nonneg_parts", n_samples=200, seed=2)
    assert not np.allclose(a["data"][0].numpy(), b["data"][0].numpy())
