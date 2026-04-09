import torch
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple, List

def ba_svd(x: torch.Tensor, 
           nu: Optional[int] = None, 
           nv: Optional[int] = None, 
           max_iter: int = 100, 
           tol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Basic block-based SVD approximation or full SVD using torch.linalg.
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
    PCA that handles constant columns and NaNs.
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
    Whiten a matrix using PCA.
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
    Multi-scale SVD algorithm using torch.
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
