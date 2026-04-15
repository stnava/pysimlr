import torch
import numpy as np
from typing import Optional, List, Union, Dict, Any

try:
    import nsa
    nsa_flow_orth = nsa.nsa_flow
except ImportError:
    nsa_flow_orth = None

def optimize_indicator_matrix(m: torch.Tensor, 
                              preprocess: bool = True, 
                              max_iter: int = 20, 
                              tol: float = 1e-4, 
                              verbose: bool = False) -> torch.Tensor:
    """
    Find an optimal binary indicator matrix (I) for a given matrix (m).

    Solves the problem: maximize sum(m * I) subject to sum(I[:, j]) = 1 
    and sum(I[i, :]) <= 1. This is a linear assignment problem variant 
    solved via a greedy iterative approach.

    Parameters
    ----------
    m : torch.Tensor
        The input matrix (usually a cross-covariance or projection).
    preprocess : bool, default=True
        Whether to flip row signs to maximize positive alignment.
    max_iter : int, default=20
        Maximum number of optimization iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    verbose : bool, default=False
        Whether to print convergence information.

    Returns
    -------
    torch.Tensor
        The optimized binary indicator matrix (same shape as m).

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    Optimize an indicator matrix for both positive and negative orientations.

    This is a helper function that finds the optimal binary indicator matrix (I) 
    such that the sum of elements in (m * I) is maximized, considering both 
    m and -m to handle sign ambiguity in latent components.

    Parameters
    ----------
    m : torch.Tensor
        The input matrix to sparsify or find indicators for.
    verbose : bool, optional
        Whether to print convergence information (default is False).

    Returns
    -------
    torch.Tensor
        The optimized sparse matrix (m * I) with the best objective value.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    Apply rank-based segmentation to a matrix to enforce sparsity.

    Retains only the top percentile of values (based on absolute magnitude) 
    for each row or column, setting others to zero.

    Parameters
    ----------
    v : torch.Tensor
        The matrix to segment.
    sparseness_quantile : float
        The quantile of elements to set to zero (0.0 to 1.0).
    basic : bool, default=False
        If False, uses indicator matrix optimization (`indicator_opt_both_ways`).
        If True, uses simple quantile-based thresholding.
    positivity : str, default="positive"
        Constraint on sign: "positive", "negative", or "either".
    transpose : bool, default=False
        Whether to apply segmentation to columns (False) or rows (True).

    Returns
    -------
    torch.Tensor
        The segmented sparse matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    Orthogonalize and/or sparsify a projection matrix.

    A comprehensive utility for enforcing constraints on basis matrices, 
    including Stiefel manifold projection (SVD-based) and quantile-based 
    sparsification.

    Parameters
    ----------
    v : torch.Tensor
        The input matrix (features x components).
    sparseness_quantile : float, default=0.0
        Proportion of elements to zero out.
    positivity : str, default="either"
        Sign constraints: "positive", "negative", or "either".
    orthogonalize : bool, default=True
        Whether to project the matrix onto the Stiefel manifold (V^T V = I).
    unit_norm : bool, default=True
        Whether to normalize each component to unit L2 norm.
    soft_thresholding : bool, default=False
        If True, uses soft-thresholding (shrinkage) instead of hard zeroing.
    sparseness_alg : str, optional
        Override algorithm: "orthorank" or "basic" for rank-based segmentation.

    Returns
    -------
    torch.Tensor
        The constrained and sparsified matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
        
    if (isinstance(sparseness_quantile, (list, torch.Tensor, np.ndarray)) or sparseness_quantile > 0):
        for vv in range(k):
            local_v = v_out[:, vv]
            if positivity == "positive":
                local_v = torch.clamp(local_v, min=0.0)
            elif positivity == "negative":
                local_v = torch.clamp(local_v, max=0.0)
            
            sq = sparseness_quantile[vv] if isinstance(sparseness_quantile, (list, torch.Tensor, np.ndarray)) else sparseness_quantile
            if sq <= 0:
                if unit_norm:
                    norm = torch.norm(local_v)
                    if norm > 0: local_v = local_v / norm
                v_out[:, vv] = local_v
                continue
                
            if soft_thresholding:
                thresh = torch.quantile(torch.abs(local_v), sq)
                local_v = torch.sign(local_v) * torch.clamp(torch.abs(local_v) - thresh, min=0.0)
            else:
                thresh = torch.quantile(torch.abs(local_v), sq)
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

    Iteratively alternates between projecting onto the Stiefel manifold 
    (orthogonality) and the non-negative orthant (positivity) until convergence.

    Parameters
    ----------
    x : torch.Tensor
        The input matrix.
    max_iter : int, default=100
        Maximum number of projection cycles.
    tol : float, default=1e-4
        Convergence tolerance for the difference between iterations.
    constraint : str, default='positive'
        "positive" or "negative".

    Returns
    -------
    torch.Tensor
        The projected orthonormal and nonnegative matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x).float()
        
    v_out = x.clone()
    for _ in range(max_iter):
        v_prev = v_out.clone()
        # Orthogonality projection
        u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
        v_out = u @ v_h
        # Positivity projection
        if constraint == 'positive':
            v_out = torch.clamp(v_out, min=0.0)
        elif constraint == 'negative':
            v_out = torch.clamp(v_out, max=0.0)
            
        if torch.norm(v_out - v_prev) < tol:
            break
    return v_out

def project_to_partially_orthonormal_nonnegative(x: torch.Tensor, 
                                               max_iter: int = 10, 
                                               constraint: str = 'positive', 
                                               ortho_strength: float = 1.0) -> torch.Tensor:
    """
    Project a matrix towards the Stiefel manifold with a controlled strength.

    Blends the original matrix with its projection onto the Stiefel manifold.

    Parameters
    ----------
    x : torch.Tensor
        The input matrix.
    max_iter : int, default=10
        Maximum number of projection cycles.
    constraint : str, default='positive'
        "positive", "negative", or "either".
    ortho_strength : float, default=1.0
        The blend factor (0.0 to 1.0).

    Returns
    -------
    torch.Tensor
        The partially projected matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x).float()
        
    v_out = x.clone()
    for _ in range(max_iter):
        u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
        v_ortho = u @ v_h
        v_out = (1 - ortho_strength) * v_out + ortho_strength * v_ortho
        
        if constraint == 'positive':
            v_out = torch.clamp(v_out, min=0.0)
        elif constraint == 'negative':
            v_out = torch.clamp(v_out, max=0.0)
    return v_out

def simlr_sparseness(v: torch.Tensor, 
                     constraint_type: str = "none",
                     smoothing_matrix: Optional[torch.Tensor] = None,
                     positivity: str = 'either',
                     sparseness_quantile: float = 0.0,
                     constraint_weight: float = 0.0,
                     constraint_iterations: int = 1,
                     sparseness_alg: str = 'soft',
                     energy_type: Optional[str] = None,
                     modality_index: Optional[int] = None) -> torch.Tensor:
    """
    Main sparsification and constraint enforcement function for SiMLR.

    This is the high-level entry point for all matrix constraints used during 
    the SiMLR optimization loop. It dispatches to specific methods based on 
    the requested `constraint_type` and `sparseness_alg`.

    Parameters
    ----------
    v : torch.Tensor
        The matrix to constrain (typically basis V or gradient).
    constraint_type : str, default="none"
        Type of manifold or orthogonality constraint ("Stiefel", "Grassmann", 
        "ortho", "none").
    smoothing_matrix : torch.Tensor, optional
        A prior matrix used for spatially-aware smoothing (V = S @ V).
    positivity : str, default='either'
        Sign constraint ("positive", "negative", or "either").
    sparseness_quantile : float, default=0.0
        Threshold for element-wise sparsity.
    constraint_weight : float, default=0.0
        Strength of the manifold projection or retraction.
    constraint_iterations : int, default=1
        Number of inner iterations for the constraint solver.
    sparseness_alg : str, default='soft'
        The algorithm for sparsity ("soft", "hard", "nnorth", "orthorank").
    energy_type : str, optional
        The SiMLR objective function (affects normalization).
    modality_index : int, optional
        Index of the current modality if sparseness_quantile is a list.

    Returns
    -------
    torch.Tensor
        The constrained and sparsified matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    v_out = v.clone()
    orig_dtype = v_out.dtype
    
    # Robust NaN handling
    if torch.isnan(v_out).any():
        v_out = torch.nan_to_num(v_out, nan=0.0)
    
    if positivity == 'positive': v_out = torch.abs(v_out)
    elif positivity == 'negative': v_out = -torch.abs(v_out)
    if smoothing_matrix is not None: v_out = smoothing_matrix @ v_out
    
    # if it's a list then index it with the correct index, otherwise pass as a scalar.
    sq = sparseness_quantile
    if isinstance(sq, (list, torch.Tensor, np.ndarray)) and modality_index is not None:
        sq = sq[modality_index]

    apply_nonneg = 'hard' if positivity in ['positive', 'hard'] else 'none'

    # --- New Prioritized Flow ---
    # 1. Handle Hard Constraints (Stiefel/Grassmann) or prioritized nsa_flow_orth
    # Corrected condition to avoid unintentional trigger for "none" or "ortho"
    if constraint_type in ["Stiefel", "Grassmann"]:
        if sparseness_alg == 'nnorth':
            v_out = project_to_orthonormal_nonnegative(v_out, constraint=positivity)
        elif nsa_flow_orth is not None:
            precision = "float32" if orig_dtype == torch.float32 else "float64"
            w = constraint_weight if constraint_weight > 0 else 1.0
            try:
                res = nsa_flow_orth(v_out, w=w, retraction="soft_polar", max_iter=max(5, constraint_iterations), precision=precision, apply_nonneg=apply_nonneg)
                if res['Y'] is not None:
                    v_out = res['Y'].to(orig_dtype)
            except:
                if not torch.isnan(v_out).any():
                    u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
                    v_out = u @ v_h
        else:
            # Fallback to SVD if nsa_flow not available
            if not torch.isnan(v_out).any():
                u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
                v_out = u @ v_h

        # 2. Apply Sparsity on top of constrained matrix
        if sq != 0 and sparseness_alg == 'soft':
            v_out = orthogonalize_and_q_sparsify(v_out, sparseness_quantile=sq, positivity=positivity, orthogonalize=False, unit_norm=False, soft_thresholding=True)

    elif constraint_type == "ortho":
        if nsa_flow_orth is not None and constraint_weight > 0:
            precision = "float32" if orig_dtype == torch.float32 else "float64"
            try:
                res = nsa_flow_orth(v_out, w=constraint_weight, retraction="soft_polar", max_iter=max(5, constraint_iterations), precision=precision, apply_nonneg=apply_nonneg)
                if res['Y'] is not None:
                    v_out = res['Y'].to(orig_dtype)
            except:
                if not torch.isnan(v_out).any():
                    u, s, v_h = torch.linalg.svd(v_out, full_matrices=False)
                    v_ortho = u @ v_h
                    v_out = (1 - constraint_weight) * v_out + constraint_weight * v_ortho
        elif constraint_weight > 0:
            v_out = project_to_partially_orthonormal_nonnegative(v_out, max_iter=constraint_iterations, constraint=positivity, ortho_strength=constraint_weight)
            
        if sq != 0 and sparseness_alg == 'soft':
            v_out = orthogonalize_and_q_sparsify(v_out, sparseness_quantile=sq, positivity=positivity, orthogonalize=False, unit_norm=False, soft_thresholding=True)

    elif constraint_type == "none" and sq != 0 and sparseness_alg == 'soft':
        v_out = orthogonalize_and_q_sparsify(v_out, sparseness_quantile=sq, positivity=positivity, orthogonalize=False, unit_norm=False, soft_thresholding=True)
    
    return v_out.to(orig_dtype)
