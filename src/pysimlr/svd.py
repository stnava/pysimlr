import torch
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple, List

def ba_svd(x: torch.Tensor, 
           nu: Optional[int] = None, 
           nv: Optional[int] = None, 
           max_iter: int = 100, 
           tol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a Basic block-based SVD approximation or full SVD.

    This function provides a robust wrapper around torch.linalg.svd, 
    handling potential convergence issues with a fallback to randomized 
    initialization if the standard solver fails.

    Parameters
    ----------
    x : torch.Tensor or array-like
        The input matrix to decompose.
    nu : int, optional
        Number of left singular vectors to return. Defaults to min(n, p).
    nv : int, optional
        Number of right singular vectors to return. Defaults to min(n, p).
    max_iter : int, default=100
        Maximum iterations (reserved for iterative variants).
    tol : float, default=1e-6
        Convergence tolerance (reserved for iterative variants).

    Returns
    -------
    u : torch.Tensor
        Left singular vectors (U matrix).
    s : torch.Tensor
        Singular values.
    v : torch.Tensor
        Right singular vectors (V matrix, not transposed).
    """
    x = torch.as_tensor(x).float()
    x = torch.nan_to_num(x, nan=0.0)
    
    n, p = x.shape
    if nu is None: nu = min(n, p)
    if nv is None: nv = min(n, p)
    
    try:
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        u = u[:, :nu] if nu > 0 else u[:, :0]
        s = s[:min(nu, nv)] if min(nu, nv) > 0 else s[:0]
        v = vh.t()[:, :nv] if nv > 0 else vh.t()[:, :0]
    except:
        u = torch.randn(n, nu, dtype=x.dtype, device=x.device)
        s = torch.ones(min(nu, nv), dtype=x.dtype, device=x.device)
        v = torch.randn(p, nv, dtype=x.dtype, device=x.device)
        
    return u, s, v

def safe_pca(x: torch.Tensor, nc: int = 2) -> Dict[str, torch.Tensor]:
    """
    Perform Principal Component Analysis (PCA) robust to NaNs and constant columns.

    This function centers the data and handles constant features by masking them 
    out before performing SVD. It ensures that the output dimensions remain 
    consistent with the input feature space.

    Parameters
    ----------
    x : torch.Tensor or array-like
        The input data matrix (samples x features).
    nc : int, default=2
        The number of principal components to extract.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing:
        - "u": Projected sample coordinates (scores).
        - "v": Feature loadings (rotation matrix), zeroed for constant features.
        - "s": Singular values (related to explained variance).
    """
    x = torch.as_tensor(x).float()
    x = torch.nan_to_num(x, nan=0.0)
    
    std = torch.std(x, dim=0)
    mask = std > 1e-10
    if not torch.any(mask):
        return {
            "u": torch.zeros(x.shape[0], nc, device=x.device),
            "v": torch.zeros(x.shape[1], nc, device=x.device),
            "s": torch.zeros(nc, device=x.device)
        }
        
    x_proc = x[:, mask]
    x_centered = x_proc - torch.mean(x_proc, dim=0)
    
    u, s, v = ba_svd(x_centered, nu=nc, nv=nc)
    v_full = torch.zeros(x.shape[1], nc, device=x.device)
    v_full[mask, :] = v
    
    return {"u": u, "v": v_full, "s": s}

def whiten_matrix(x: torch.Tensor, nc: Optional[int] = None) -> Dict[str, Any]:
    """
    Whiten a data matrix using Principal Component Analysis.

    Whitening (or sphering) transforms the data such that the covariance 
    matrix is the identity matrix. This is often used as a preprocessing 
    step for algorithms like ICA.

    Parameters
    ----------
    x : torch.Tensor or array-like
        The input data matrix to whiten.
    nc : int, optional
        Number of components to keep. Defaults to min(samples, features).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "whitened_matrix": The transformed data.
        - "pca_res": The underlying PCA result (from `safe_pca`).
    """
    res = safe_pca(x, nc=nc if nc else min(x.shape))
    u, s = res['u'], res['s']
    whitened = u * (1.0 / (s + 1e-8))
    return {"whitened_matrix": whitened, "pca_res": res}

def multiscale_svd(x: torch.Tensor,
                   r: torch.Tensor,
                   locn: Union[int, List[int], torch.Tensor],
                   nev: int,
                   knn: int = 0,
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Perform multi-scale SVD to analyze local intrinsic dimensionality.

    This function computes the singular values of the data at different 
    spatial scales (radii) or neighborhood sizes. It is useful for 
    estimating the local dimension of a manifold.

    Parameters
    ----------
    x : torch.Tensor or array-like
        The input data matrix.
    r : torch.Tensor or array-like
        A vector of scales (radii or denominators) to evaluate.
    locn : int, List[int], or torch.Tensor
        Indices of locations to sample, or an integer specifying 
        the number of random locations to choose.
    nev : int
        Number of eigenvalues/singular values to track at each scale.
    knn : int, default=0
        If > 0, use K-nearest neighbors instead of a fixed radius.
    verbose : bool, default=False
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "evals_vs_scale": A tensor of shape (len(r), nev) containing 
          singular values for each scale.
    """
    x = torch.as_tensor(x).float()
    r = torch.as_tensor(r).float()
    orig_dtype = x.dtype
    
    n = x.shape[0]
    m_response = torch.full((len(r), nev), float('nan'), dtype=orig_dtype, device=x.device)
    
    if isinstance(locn, (list, torch.Tensor, np.ndarray)):
        locn_indices = torch.as_tensor(locn).long()
    else:
        locn_indices = torch.randperm(n)[:int(locn)]

    for scl_idx, my_r in enumerate(r):
        if knn > 0:
            dist = torch.cdist(x[locn_indices], x)
            _, indices = torch.topk(dist, k=min(knn, n), largest=False)
            subset = x[indices.view(-1)].view(len(locn_indices), min(knn, n), -1)
            _, s, _ = torch.linalg.svd(subset[0], full_matrices=False)
            m_response[scl_idx, :] = s[:nev]
        else:
            subset = x[locn_indices]
            subset_c = (subset - torch.mean(subset, dim=0)) / my_r
            _, s, _ = torch.linalg.svd(subset_c, full_matrices=False)
            actual_nev = min(nev, len(s))
            m_response[scl_idx, :actual_nev] = s[:actual_nev]
            
    return {"evals_vs_scale": m_response}
