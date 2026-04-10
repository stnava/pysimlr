import torch
import pytest
from pysimlr import simlr

def test_simlr_ortho_summary():
    x1, x2 = torch.randn(50, 10), torch.randn(50, 8)
    res = simlr([x1, x2], k=2, iterations=2)
    
    assert "v_orthogonality" in res
    assert len(res["v_orthogonality"]) == 2
    for summary in res["v_orthogonality"]:
        assert "invariant_defect" in summary
        assert "mean_defect" in summary
        assert "condition_number" in summary
        assert "effective_rank_proxy" in summary

if __name__ == "__main__":
    pytest.main([__file__])
