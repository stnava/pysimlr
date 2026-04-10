import torch
import pytest
from pysimlr.utils import (
    mean_orthogonality_defect, 
    gradient_mean_orthogonality_defect,
    orthogonality_summary,
    invariant_orthogonality_defect
)

def test_orthogonality_metrics():
    # 1. Identity matrix should have zero defect
    eye = torch.eye(5)
    assert mean_orthogonality_defect(eye) == 0.0
    assert invariant_orthogonality_defect(eye) == 0.0
    
    # 2. Correlated matrix should have positive defect
    a = torch.ones(10, 3)
    defect = mean_orthogonality_defect(a)
    assert defect > 0.9 # Very high defect for identical columns
    
    # 3. Summary check
    summary = orthogonality_summary(eye)
    assert summary["invariant_defect"] == 0.0
    assert summary["mean_defect"] == 0.0
    assert summary["condition_number"] == 1.0

def test_orthogonality_gradient():
    a = torch.randn(10, 4, requires_grad=True)
    
    # Check if gradient is computable and has correct shape
    defect = mean_orthogonality_defect(a)
    defect.backward()
    
    manual_grad = gradient_mean_orthogonality_defect(a.detach())
    
    assert manual_grad.shape == a.shape
    # Check alignment between autograd and manual implementation
    cos_sim = torch.nn.functional.cosine_similarity(a.grad.flatten(), manual_grad.flatten(), dim=0)
    assert cos_sim > 0.99

if __name__ == "__main__":
    pytest.main([__file__])
