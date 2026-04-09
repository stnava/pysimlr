import torch
from typing import Tuple, Optional, List, Dict, Any

def ba_svd(x: torch.Tensor, 
           nu: Optional[int] = None, 
           nv: Optional[int] = None, 
           divide_by_max: bool = False, 
           na_to_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust SVD with Automatic Fallback.
    """
    x = torch.as_tensor(x).float()
    orig_dtype = x.dtype

    if na_to_noise:
        bad_mask = ~torch.isfinite(x)
        if torch.any(bad_mask):
            x = x.clone()
            noise = torch.randn(bad_mask.sum(), device=x.device, dtype=x.dtype) * 1e-6
            x[bad_mask] = noise

    if divide_by_max:
        mx = torch.max(torch.abs(x))
        if mx > 0:
            x = x / mx

    m, n = x.shape
    if nu is None:
        nu = min(m, n)
    else:
        nu = min(nu, m, n)
        
    if nv is None:
        nv = min(m, n)
    else:
        nv = min(nv, m, n)

    try:
        # torch.linalg.svd is generally differentiable
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        u_out = u[:, :nu].to(orig_dtype)
        v_out = vh[:nv, :].t().to(orig_dtype)
        s_out = s[:min(nu, nv)].to(orig_dtype)
        return u_out, s_out, v_out
    except RuntimeError:
        k = max(nu, nv)
        p = 5 
        q = 2 
        
        m, n = x.shape
        omega = torch.randn(n, k + p, device=x.device, dtype=orig_dtype)
        y = x @ omega
        for _ in range(q):
            y = x @ (x.t() @ y)
            
        q_mat, _ = torch.linalg.qr(y)
        b = q_mat.t() @ x
        u_hat, s, vh = torch.linalg.svd(b, full_matrices=False)
        u = q_mat @ u_hat
        
        u_out = u[:, :nu].to(orig_dtype)
        v_out = vh[:nv, :].t().to(orig_dtype)
        s_out = s[:min(nu, nv)].to(orig_dtype)
        return u_out, s_out, v_out

def safe_pca(x: torch.Tensor, 
             nc: Optional[int] = None, 
             center: bool = True, 
             scale: bool = True) -> Dict[str, torch.Tensor]:
    """
    Safe PCA implementation using torch.
    """
    x = torch.as_tensor(x).float()
    orig_dtype = x.dtype
        
    if nc is None:
        nc = min(x.shape)
    else:
        nc = min(nc, x.shape[0], x.shape[1])
        
    x_proc = x
    if center:
        x_proc = x_proc - torch.mean(x_proc, dim=0)
    
    if scale:
        std = torch.std(x_proc, dim=0)
        # Avoid in-place modification for backprop
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        x_proc = x_proc / std_safe
        
    u, s, v = ba_svd(x_proc, nu=nc, nv=nc)
    
    return {
        "u": u,
        "s": s,
        "v": v,
        "x": (x_proc @ v).to(orig_dtype) # Scores
    }

def whiten_matrix(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Matrix whitening using SVD in torch.
    """
    x = torch.as_tensor(x).float()
    orig_dtype = x.dtype
        
    x_mean = torch.mean(x, dim=0)
    x_centered = x - x_mean
    u, s, v = ba_svd(x_centered)
    s_inv_sqrt = torch.diag(1.0 / (torch.sqrt(s) + 1e-10))
    whitening_matrix = (v @ s_inv_sqrt @ v.t()).to(orig_dtype)
    x_whitened = (x_centered @ whitening_matrix).to(orig_dtype)
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
    x = torch.as_tensor(x).float()
    r = torch.as_tensor(r).float()
    orig_dtype = x.dtype

    n = x.shape[0]
    m_response = torch.full((len(r), nev), float('nan'), dtype=orig_dtype, device=x.device)
    
    for scl_idx, my_r in enumerate(r):
        loc_sam = torch.randperm(n)[:locn]
        my_evs = torch.full((locn, nev), float('nan'), dtype=orig_dtype, device=x.device)
        for i in range(locn):
            diff = x - x[loc_sam[i]]
            row_dist = torch.sqrt(torch.sum(diff**2, dim=1))
            sel = row_dist < my_r
            if torch.sum(sel) > 2:
                if knn > 0 and torch.sum(sel) > knn:
                    sel_inds = torch.where(sel)[0]
                    subset_inds = sel_inds[torch.randperm(len(sel_inds))[:knn]]
                    sel = torch.zeros(n, dtype=torch.bool, device=x.device)
                    sel[subset_inds] = True
                l_mat = x[sel, :]
                l_mat_centered = l_mat - torch.mean(l_mat, dim=0)
                n_sel = l_mat.shape[0]
                l_cov = (l_mat_centered.t() @ l_mat_centered) / (n_sel - 1)
                _, temp_s, _ = ba_svd(l_cov, nu=nev, nv=0)
                num_evs = min(nev, len(temp_s))
                my_evs[i, :num_evs] = temp_s[:num_evs].to(orig_dtype)
        m_response[scl_idx, :] = torch.nanmean(my_evs, dim=0)
        if verbose:
            print(f"Scale {my_r}: {m_response[scl_idx, :]}")
    return {"evals_vs_scale": m_response}
