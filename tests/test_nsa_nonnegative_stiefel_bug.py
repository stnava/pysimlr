import torch
import numpy as np
from pysimlr.deep import LENDNSAEncoder, lend_simr
from pysimlr.utils import invariant_orthogonality_defect
from scipy.stats import pearsonr

def test_nsa_nonnegative_stiefel_collapse():
    """
    This test reproduces the 'Non-Negative Stiefel Manifold' bug.
    When input_dim == latent_dim (e.g., 2x2), enforcing both strict 
    orthogonality and strict positivity via post-hoc clamping causes 
    the columns of the basis matrix V to lose orthogonality entirely.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n = 100
    k = 2
    
    shared_signal = torch.randn(n, k)
    x1 = shared_signal @ torch.randn(k, 2) + 0.1 * torch.randn(n, 2)
    x2 = shared_signal @ torch.randn(k, 10) + 0.1 * torch.randn(n, 10)
    
    data = [x1.numpy(), x2.numpy()]

    res = lend_simr(
        data, 
        k=k, 
        epochs=50, 
        positivity="positive", 
        use_nsa=True, 
        nsa_iterations=1,
        verbose=False
    )

    v_mat = res["v"][0].detach().cpu()
    
    print("\n--- Extracted V Matrix (Modality 1) ---")
    print(v_mat.numpy())
    
    ortho_defect = invariant_orthogonality_defect(v_mat).item()
    print(f"\nOrthogonality Defect: {ortho_defect:.4f}")
    
    col1 = v_mat[:, 0]
    col2 = v_mat[:, 1]
    
    # The true measure of non-overlapping columns is their dot product
    overlap = torch.dot(col1, col2).item()
    print(f"Column Overlap (Dot Product): {overlap:.4f}")
    
    # We expect the orthogonality defect to be very small (< 0.1) AND 
    # the overlap between the orthogonal columns to be near 0 (< 0.1)
    # If the bug is present, defect is > 0.2 and overlap is high.
    
    bug_fixed = ortho_defect < 0.1 and overlap < 0.1
    
    if not bug_fixed:
        print("\n[FAILURE] Bug is still present. The matrix collapsed.")
        assert False, f"Bug not fixed! Ortho Defect: {ortho_defect:.4f}, Overlap: {overlap:.4f}"
    
    print("\n[SUCCESS] Bug successfully fixed! The matrix correctly solved the Non-Negative Stiefel problem via disjoint supports.")

if __name__ == "__main__":
    test_nsa_nonnegative_stiefel_collapse()
