import torch
from typing import Optional, Union, Tuple
from .utils import l1_normalize_features

try:
    from nsa_flow import nsa_flow_orth
except ImportError:
    nsa_flow_orth = None

def optimize_indicator_matrix(m: torch.Tensor, 
                              max_iter: int = 1000, 
                              tol: float = 1e-6, 
                              preprocess: bool = True, 
                              verbose: bool = False) -> torch.Tensor:
    """
    Optimize Binary Indicator Matrix with Row Uniformity using torch.
    """
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m).float()
        
    m_opt = m.clone()
    if preprocess:
        for i in range(m_opt.shape[0]):
            if torch.sum(m_opt[i, :] < 0) > torch.sum(m_opt[i, :] > 0):
                m_opt[i, :] = -m_opt[i, :]
                
    prev_sum = float('-inf')
    
    for iteration in range(max_iter):
        I = torch.zeros_like(m_opt)
        row_used = torch.zeros(m_opt.shape[0], dtype=torch.bool)
        
        for j in range(m_opt.shape[1]):
            available_vals = m_opt[:, j].clone()
            available_vals[row_used] = float('-inf')
            max_val, selected_row = torch.max(available_vals, dim=0)
            
            if max_val > float('-inf'):
                I[selected_row, j] = 1.0
                row_used[selected_row] = True
                
        current_sum = torch.sum(m_opt * I)
        if torch.abs(current_sum - prev_sum) < tol:
            if verbose:
                print(f"Converged in {iteration+1} iterations with objective: {current_sum.item()}")
            break
        prev_sum = current_sum
        
    return m_opt * I

def indicator_opt_both_ways(m: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """
    Helper to Optimize Indicator Matrix with Best Sum.
    """
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m).float()
        
    I_m = optimize_indicator_matrix(m, preprocess=False, verbose=verbose)
    sum_m = torch.sum(m * I_m)
    
    I_neg_m = optimize_indicator_matrix(-m, preprocess=False, verbose=verbose)
    sum_neg_m = torch.sum(m * I_neg_m) 
    
    if sum_m >= sum_neg_m:
        return m * I_m
    else:
        return (-m) * I_neg_m

def rank_based_matrix_segmentation(v: torch.Tensor, 
                                   sparseness_quantile: float, 
                                   basic: bool = False, 
                                   positivity: str = "positive", 
                                   transpose: bool = False) -> torch.Tensor:
    """
    Rank-based segmentation of a matrix using torch.
    """
    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(v).float()
        
    if not basic:
        return indicator_opt_both_ways(v)
        
    if transpose:
        v = v.t()
        
    outmat = torch.zeros_like(v)
    n_to_keep = int(round(v.shape[1] * (1.0 - sparseness_quantile)))
    
    for k in range(v.shape[0]):
        row_values = v[k, :].clone()
        if torch.all(row_values == 0):
            continue
            
        if positivity == "either":
            _, loc_ord = torch.topk(torch.abs(row_values), k=min(n_to_keep, len(row_values)))
        elif positivity in ["positive", "negative"]:
            pos_mask = row_values > 0
            neg_mask = row_values < 0
            
            pos_sum = torch.sum(torch.abs(row_values[pos_mask]))
            neg_sum = torch.sum(torch.abs(row_values[neg_mask]))
            
            if pos_sum >= neg_sum:
                row_values[row_values < 0] = 0
                _, loc_ord = torch.topk(row_values, k=min(n_to_keep, len(row_values)))
            else:
                row_values[row_values > 0] = 0
                _, loc_ord = torch.topk(-row_values, k=min(n_to_keep, len(row_values)))
                
        outmat[k, loc_ord] = row_values[loc_ord]
        
    if transpose:
        return outmat.t()
    return outmat

def orthogonalize_and_q_sparsify(v: torch.Tensor, 
                                 sparseness_quantile: float = 0.0, 
                                 positivity: str = "either",
                                 orthogonalize: bool = True,
                                 unit_norm: bool = True,
                                 soft_thresholding: bool = False,
                                 sparseness_alg: Optional[str] = None) -> torch.Tensor:
    """
    Orthogonalize and/or sparsify a matrix.
    """
    if sparseness_alg == "orthorank":
        return rank_based_matrix_segmentation(v, sparseness_quantile, basic=False, positivity=positivity, transpose=True)
    elif sparseness_alg == "basic":
        return rank_based_matrix_segmentation(v, sparseness_quantile, basic=True, positivity=positivity, transpose=True)

    if torch.all(v == 0):
        return v.clone()

    v_out = v.clone()
    orig_dtype = v_out.dtype
    n, k = v_out.shape
    
    if orthogonalize and k > 1:
        try:
            if nsa_flow_orth is not None:
                # Use NSA-Flow for robust retraction if available
                precision = "float32" if orig_dtype == torch.float32 else "float64"
                apply_nonneg = 'hard' if positivity in ['positive', 'hard'] else 'none'
                res = nsa_flow_orth(v_out, w=0.5, retraction="soft_polar", precision=precision, max_iter=10, apply_nonneg=apply_nonneg)
                if res['Y'] is not None:
                    v_out = res['Y'].to(orig_dtype)
                else:
                    u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
                    v_out = u @ v_h
            else:
                u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
                v_out = u @ v_h
        except: pass
        
    if sparseness_quantile > 0:
        for vv in range(k):
            local_v = v_out[:, vv]
            if positivity == "positive":
                local_v = torch.clamp(local_v, min=0.0)
            elif positivity == "negative":
                local_v = torch.clamp(local_v, max=0.0)
                
            if soft_thresholding:
                thresh = torch.quantile(torch.abs(local_v), sparseness_quantile)
                local_v = torch.sign(local_v) * torch.clamp(torch.abs(local_v) - thresh, min=0.0)
            else:
                thresh = torch.quantile(torch.abs(local_v), sparseness_quantile)
                local_v[torch.abs(local_v) < thresh] = 0.0
                
            if unit_norm:
                norm = torch.norm(local_v)
                if norm > 0:
                    local_v = local_v / norm
            v_out[:, vv] = local_v
            
    return v_out.to(orig_dtype)

def project_to_orthonormal_nonnegative(x: torch.Tensor, 
                                       max_iter: int = 100, 
                                       tol: float = 1e-4, 
                                       constraint: str = 'positive') -> torch.Tensor:
    """
    Project a matrix to be orthonormal and nonnegative using Dykstra-like alternations.
    """
    v = x.clone()
    orig_dtype = v.dtype
    for i in range(max_iter):
        v_old = v.clone()
        if constraint == 'positive': v = torch.clamp(v, min=0.0)
        elif constraint == 'negative': v = torch.clamp(v, max=0.0)
        try:
            u, s, v_h = torch.linalg.svd(v, full_matrices=False)
            v = u @ v_h
        except: pass
        if torch.norm(v - v_old) < tol: break
    if constraint == 'positive': v = torch.clamp(v, min=0.0)
    elif constraint == 'negative': v = torch.clamp(v, max=0.0)
    return v.to(orig_dtype)

def project_to_partially_orthonormal_nonnegative(x: torch.Tensor, 
                                                 max_iter: int = 1, 
                                                 constraint: str = 'positive',
                                                 ortho_strength: float = 0.1) -> torch.Tensor:
    """
    Partially project to orthonormal and nonnegative.
    """
    v = x.clone()
    orig_dtype = v.dtype
    for _ in range(max_iter):
        if constraint == 'positive': v = torch.clamp(v, min=0.0)
        elif constraint == 'negative': v = torch.clamp(v, max=0.0)
        try:
            u, s, v_h = torch.linalg.svd(v, full_matrices=False)
            v_ortho = u @ v_h
            v = (1 - ortho_strength) * v + ortho_strength * v_ortho
        except: pass
    if constraint == 'positive': v = torch.clamp(v, min=0.0)
    elif constraint == 'negative': v = torch.clamp(v, max=0.0)
    return v.to(orig_dtype)

def simlr_sparseness(v: torch.Tensor, 
                     constraint_type: str = "none",
                     smoothing_matrix: Optional[torch.Tensor] = None,
                     positivity: str = 'either',
                     sparseness_quantile: float = 0.0,
                     constraint_weight: float = 0.0,
                     constraint_iterations: int = 1,
                     sparseness_alg: str = 'soft',
                     energy_type: Optional[str] = None) -> torch.Tensor:
    """
    Main sparsification and constraint enforcement function for SIMLR.
    """
    v_out = v.clone()
    orig_dtype = v_out.dtype
    
    if positivity == 'positive': v_out = torch.abs(v_out)
    elif positivity == 'negative': v_out = -torch.abs(v_out)
    if smoothing_matrix is not None: v_out = smoothing_matrix @ v_out
    
    apply_nonneg = 'hard' if positivity in ['positive', 'hard'] else 'none'

    # --- New Prioritized Flow ---
    # 1. Handle Hard Constraints (Stiefel/Grassmann) or prioritized nsa_flow_orth
    if constraint_type in ["Stiefel", "Grassmann"] or (constraint_type == "none" and nsa_flow_orth is not None and constraint_weight > 0):
        if sparseness_alg == 'nnorth':
            v_out = project_to_orthonormal_nonnegative(v_out, constraint=positivity)
        elif nsa_flow_orth is not None and (constraint_type in ["Stiefel", "Grassmann"] or constraint_weight > 0):
            precision = "float32" if orig_dtype == torch.float32 else "float64"
            # If Stiefel but weight is 0, use 1.0 as default for projection
            w = constraint_weight if constraint_weight > 0 else 1.0
            try:
                res = nsa_flow_orth(v_out, w=w, retraction="soft_polar", max_iter=max(5, constraint_iterations), precision=precision, apply_nonneg=apply_nonneg)
                if res['Y'] is not None:
                    v_out = res['Y'].to(orig_dtype)
            except:
                if constraint_type in ["Stiefel", "Grassmann"]:
                    u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
                    v_out = u @ v_h
        elif constraint_type in ["Stiefel", "Grassmann"]:
            # Fallback to SVD if nsa_flow not available
            u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
            v_out = u @ v_h

        # 2. Apply Sparsity on top of constrained matrix
        if sparseness_quantile != 0 and sparseness_alg == 'soft':
            v_out = orthogonalize_and_q_sparsify(v_out, sparseness_quantile=sparseness_quantile, positivity=positivity, orthogonalize=False, unit_norm=False, soft_thresholding=True)

    elif constraint_type == "ortho":
        if nsa_flow_orth is not None:
            precision = "float32" if orig_dtype == torch.float32 else "float64"
            try:
                res = nsa_flow_orth(v_out, w=constraint_weight, retraction="soft_polar", max_iter=max(5, constraint_iterations), precision=precision, apply_nonneg=apply_nonneg)
                if res['Y'] is not None:
                    v_out = res['Y'].to(orig_dtype)
            except:
                u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
                v_ortho = u @ v_h
                v_out = (1 - constraint_weight) * v_out + constraint_weight * v_ortho
        else:
            v_out = project_to_partially_orthonormal_nonnegative(v_out, max_iter=constraint_iterations, constraint=positivity, ortho_strength=constraint_weight)
            
        if sparseness_quantile != 0 and sparseness_alg == 'soft':
            v_out = orthogonalize_and_q_sparsify(v_out, sparseness_quantile=sparseness_quantile, positivity=positivity, orthogonalize=False, unit_norm=False, soft_thresholding=True)
            
    elif constraint_type == "none" and sparseness_quantile != 0 and sparseness_alg == 'soft':
        # Legacy behavior for purely unconstrained soft-sparsity
        v_out = orthogonalize_and_q_sparsify(v_out, sparseness_quantile=sparseness_quantile, positivity=positivity, orthogonalize=False, unit_norm=False, soft_thresholding=True)

    normalize_energy_types = ["acc", "cca", "nc", "normalized_correlation", "lowRankRegression", "lrr"]
    if energy_type is not None and energy_type in normalize_energy_types: v_out = l1_normalize_features(v_out)
    return v_out.to(orig_dtype)
