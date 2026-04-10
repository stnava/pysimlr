import torch
import pytest
from pysimlr.utils import invariant_orthogonality_defect

def test_invariant_orthogonality_defect():
    # Orthonormal matrix
    v_ortho = torch.eye(5, 3)
    defect_ortho = invariant_orthogonality_defect(v_ortho)
    assert defect_ortho.item() < 1e-5

    # Non-orthonormal matrix
    v_non_ortho = torch.ones(5, 3)
    defect_non_ortho = invariant_orthogonality_defect(v_non_ortho)
    assert defect_non_ortho.item() > 1e-2

if __name__ == "__main__":
    pytest.main([__file__])
