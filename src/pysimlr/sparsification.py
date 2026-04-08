import torch
from typing import Optional, Union

def optimize_indicator_matrix(m: torch.Tensor, 
                              max_iter: int = 1000, 
                              tol: float = 1e-6, 
                              preprocess: bool = True, 
                              verbose: bool = False) -> torch.Tensor:
    """
    Optimize Binary Indicator Matrix with Row Uniformity using torch.
    """
    if not isinstance(m, torch.Tensor):
        m = torch.from_numpy(m).float()
        
    m_opt = m.clone()
    if preprocess:
        for i in range(m_opt.shape[0]):
            if torch.sum(m_opt[i, :] < 0) > torch.sum(m_opt[i, :] > 0):
                m_opt[i, :] = -m_opt[i, :]
                
    prev_sum = float('-inf')
    
    for iteration in range(max_iter):
        I = torch.zeros_like(m_opt)
        row_used = torch.zeros(m_opt.shape[0], dtype=torch.bool)
        
        # Greedy column-wise assignment
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
        m = torch.from_numpy(m).float()
        
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
        v = torch.from_numpy(v).float()
        
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
                # Keep positive
                row_values[row_values < 0] = 0
                _, loc_ord = torch.topk(row_values, k=min(n_to_keep, len(row_values)))
            else:
                # Keep negative
                row_values[row_values > 0] = 0
                _, loc_ord = torch.topk(-row_values, k=min(n_to_keep, len(row_values)))
                
        outmat[k, loc_ord] = row_values[loc_ord]
        
    if transpose:
        return outmat.t()
    return outmat

def orthogonalize_and_q_sparsify(v: torch.Tensor,
                                 sparseness_quantile: float = 0.5,
                                 positivity: str = "either",
                                 orthogonalize: bool = True,
                                 unit_norm: bool = False,
                                 sparseness_alg: Optional[str] = None) -> torch.Tensor:
    """
    Sparsify a matrix using torch.
    """
    if not isinstance(v, torch.Tensor):
        v = torch.from_numpy(v).float()
        
    if sparseness_alg == "orthorank":
        return rank_based_matrix_segmentation(v, sparseness_quantile, basic=False, positivity=positivity, transpose=True)
    elif sparseness_alg == "basic":
        return rank_based_matrix_segmentation(v, sparseness_quantile, basic=True, positivity=positivity, transpose=True)

    if sparseness_quantile == 0:
        return v
        
    v_out = v.clone()
    p = v.shape[1]
    
    for vv in range(p):
        if torch.var(v_out[:, vv]) > 0:
            if vv > 0 and orthogonalize:
                # Gram-Schmidt
                for vk in range(vv):
                    prev_v = v_out[:, vk]
                    denom = torch.sum(prev_v * prev_v)
                    if denom > 0:
                        ip = torch.sum(prev_v * v_out[:, vv]) / denom
                        v_out[:, vv] = v_out[:, vv] - prev_v * ip
            
            local_v = v_out[:, vv]
            
            # BUG fix: Move flip logic to the beginning or handle properly with positivity constraint
            # R code logic: if more negatives, flip to make them positive. 
            # But then we MUST re-enforce the positivity choice.
            if torch.sum(local_v > 0) < torch.sum(local_v < 0):
                local_v = -local_v
                
            # Sparsify and enforce positivity constraint
            if positivity == "either":
                abs_v = torch.abs(local_v)
                q_val = torch.quantile(abs_v, sparseness_quantile)
                local_v[abs_v < q_val] = 0
            else:
                if positivity == "positive":
                    local_v[local_v < 0] = 0 
                elif positivity == "negative":
                    local_v[local_v > 0] = 0
                
                q_val = torch.quantile(local_v, sparseness_quantile)
                if positivity == "positive":
                    if q_val > 0:
                        local_v[local_v <= q_val] = 0
                    else:
                        local_v[local_v >= q_val] = 0
                elif positivity == "negative":
                    local_v[local_v > q_val] = 0
            
            if unit_norm:
                norm = torch.norm(local_v)
                if norm > 0:
                    local_v = local_v / norm
                    
            v_out[:, vv] = local_v
            
    return v_out
