import torch
from typing import Tuple, Optional, List, Dict, Any

def ba_svd(x: torch.Tensor, 
           nu: Optional[int] = None, 
           nv: Optional[int] = None, 
           divide_by_max: bool = False, 
           na_to_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust SVD with Automatic Fallback.
    
    Args:
        x: Input matrix as torch.Tensor.
        nu: Number of left singular vectors to compute.
        nv: Number of right singular vectors to compute.
        divide_by_max: Scale x by its max absolute value.
        na_to_noise: Replaces NA values with small noise values.
        
    Returns:
        u: Left singular vectors.
        s: Singular values.
        v: Right singular vectors (not transposed).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()

    if na_to_noise:
        # Check for NaNs or Infs
        bad_mask = ~torch.isfinite(x)
        if torch.any(bad_mask):
            # Replace with small random noise
            noise = torch.randn(bad_mask.sum()) * 1e-6
            x = x.clone()
            x[bad_mask] = noise

    if divide_by_max:
        mx = torch.max(torch.abs(x))
        if mx > 0:
            x = x / mx

    m, n = x.shape
    if nu is None:
        nu = min(m, n)
    if nv is None:
        nv = min(m, n)

    try:
        # torch.linalg.svd returns (U, S, Vh) where Vh is V^H (V transposed for real matrices)
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        u = u[:, :nu]
        v = vh[:nv, :].t()
        s = s[:min(nu, nv)]
        return u, s, v
    except RuntimeError:
        # Randomized SVD fallback using torch
        # A simple implementation of randomized SVD
        k = max(nu, nv)
        p = 5 # oversampling
        q = 2 # power iterations
        
        m, n = x.shape
        # Sketch matrix Omega
        omega = torch.randn(n, k + p, device=x.device, dtype=x.dtype)
        # Form a matrix Y = (X * X^T)^q * X * Omega
        y = x @ omega
        for _ in range(q):
            y = x @ (x.t() @ y)
            
        # QR decomposition of Y
        q_mat, _ = torch.linalg.qr(y)
        # Form B = Q^T * X
        b = q_mat.t() @ x
        # SVD of B
        u_hat, s, vh = torch.linalg.svd(b, full_matrices=False)
        u = q_mat @ u_hat
        
        u = u[:, :nu]
        v = vh[:nv, :].t()
        s = s[:min(nu, nv)]
        return u, s, v

def safe_pca(x: torch.Tensor, 
             nc: Optional[int] = None, 
             center: bool = True, 
             scale: bool = True) -> Dict[str, torch.Tensor]:
    """
    Safe PCA implementation using torch.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
        
    if nc is None:
        nc = min(x.shape)
        
    x_centered = x
    if center:
        x_centered = x - torch.mean(x, dim=0)
    
    if scale:
        std = torch.std(x, dim=0)
        std[std == 0] = 1.0
        x_centered = x_centered / std
        
    u, s, v = ba_svd(x_centered, nu=nc, nv=nc)
    
    return {
        "u": u,
        "s": s,
        "v": v,
        "x": x_centered @ v # Scores
    }

def whiten_matrix(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Matrix whitening using SVD in torch.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
        
    x_mean = torch.mean(x, dim=0)
    x_centered = x - x_mean
    
    u, s, v = ba_svd(x_centered)
    
    # Compute the whitening matrix
    # whitening_matrix = V %*% D_inv_sqrt %*% t(V)
    s_inv_sqrt = torch.diag(1.0 / torch.sqrt(s))
    whitening_matrix = v @ s_inv_sqrt @ v.t()
    
    x_whitened = x_centered @ whitening_matrix
    
    return {
        "whitened_matrix": x_whitened,
        "whitening_matrix": whitening_matrix
    }

def multiscale_svd(x: torch.Tensor, 
                   r: torch.Tensor, 
                   locn: int, 
                   nev: int, 
                   knn: int = 0, 
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Multi-scale SVD algorithm using torch.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    if not isinstance(r, torch.Tensor):
        r = torch.from_numpy(r).float()

    n = x.shape[0]
    m_response = torch.full((len(r), nev), float('nan'))
    
    for scl_idx, my_r in enumerate(r):
        # Sample locn indices
        loc_sam = torch.randperm(n)[:locn]
        my_evs = torch.full((locn, nev), float('nan'))
        
        for i in range(locn):
            # Euclidean distance to all points
            diff = x - x[loc_sam[i]]
            row_dist = torch.sqrt(torch.sum(diff**2, dim=1))
            sel = row_dist < my_r
            
            if torch.sum(sel) > 2:
                if knn > 0 and torch.sum(sel) > knn:
                    sel_inds = torch.where(sel)[0]
                    subset_inds = sel_inds[torch.randperm(len(sel_inds))[:knn]]
                    sel = torch.zeros(n, dtype=torch.bool)
                    sel[subset_inds] = True
                
                l_mat = x[sel, :]
                # Compute covariance
                l_mat_centered = l_mat - torch.mean(l_mat, dim=0)
                # BUG note: R's cov() normalization is by N-1. 
                # torch.cov expects variables in rows, so we transpose.
                # Actually, let's just do (1/(N-1)) * X^T * X
                n_sel = l_mat.shape[0]
                l_cov = (l_mat_centered.t() @ l_mat_centered) / (n_sel - 1)
                
                _, temp_s, _ = ba_svd(l_cov, nu=nev, nv=0)
                
                num_evs = min(nev, len(temp_s))
                my_evs[i, :num_evs] = temp_s[:num_evs]
            
        m_response[scl_idx, :] = torch.nanmean(my_evs, dim=0)
        if verbose:
            print(f"Scale {my_r}: {m_response[scl_idx, :]}")
            
    return {"evals_vs_scale": m_response}
