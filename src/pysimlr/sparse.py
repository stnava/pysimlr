import torch
from typing import Optional, Union, Tuple

def sparse_distance_matrix(x: torch.Tensor, 
                           k: int = 3, 
                           r: float = float('inf'), 
                           sigma: Optional[float] = None,
                           kmetric: str = "euclidean",
                           sinkhorn: bool = False,
                           verbose: bool = False) -> torch.Tensor:
    """
    Create sparse distance, covariance or correlation matrix using torch.
    Note: Returns a dense tensor with zeros for sparsity, or a torch.sparse tensor.
    For simplicity in porting, we'll use dense tensors if memory allows, 
    but for 'professional' use we should consider torch.sparse.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
        
    n, p = x.shape
    if k >= p:
        k = p - 1
        
    # We want a p x p similarity matrix between features
    # features are columns of x
    xt = x.t() # p x n
    
    # Compute all-to-all distances between features
    # (p x n) and (p x n) -> p x p distance matrix
    dist_mat = torch.cdist(xt, xt, p=2) # Euclidean distance
    
    if kmetric in ["correlation", "covariance"]:
        # Center and optionally scale features
        xt_centered = xt - torch.mean(xt, dim=1, keepdim=True)
        if kmetric == "correlation":
            xt_std = torch.std(xt, dim=1, keepdim=True)
            xt_std[xt_std == 0] = 1.0
            xt_centered = xt_centered / xt_std
            
        # Similarity is dot product / (n-1)
        sim_mat = (xt_centered @ xt_centered.t()) / (n - 1)
        
        # Convert distance to correlation-like if needed for k-NN
        # Actually we can just use the sim_mat to find top k
        vals, indices = torch.topk(sim_mat, k=k+1, dim=1, largest=True)
    else:
        # For euclidean/gaussian, find smallest distances
        vals, indices = torch.topk(dist_mat, k=k+1, dim=1, largest=False)
        sim_mat = dist_mat

    # Create sparse-like mask
    mask = torch.zeros_like(sim_mat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    
    # Apply metrics
    if kmetric == "gaussian" and sigma is not None:
        res_mat = torch.exp(-1.0 * (sim_mat**2) / (2.0 * sigma**2))
    else:
        res_mat = sim_mat
        
    # Apply radius and k-mask
    res_mat = res_mat * mask.float()
    
    if kmetric in ["correlation", "covariance"]:
        if r != float('inf'):
            res_mat[res_mat < r] = 0
    else:
        if r != float('inf'):
            res_mat[res_mat > r] = 0
            
    if sinkhorn:
        for _ in range(4):
            # Row sums
            r_sums = torch.sum(res_mat, dim=1, keepdim=True)
            r_sums[r_sums == 0] = 1.0
            res_mat = res_mat / r_sums
            # Col sums
            c_sums = torch.sum(res_mat, dim=0, keepdim=True)
            c_sums[c_sums == 0] = 1.0
            res_mat = res_mat / c_sums
            
    return res_mat

def sparse_distance_matrix_xy(x: torch.Tensor, 
                              y: torch.Tensor, 
                              k: int = 3, 
                              r: float = float('inf'), 
                              sigma: Optional[float] = None,
                              kmetric: str = "euclidean",
                              verbose: bool = False) -> torch.Tensor:
    """
    Create sparse distance, covariance or correlation matrix between x and y.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()
        
    n, p = x.shape
    nq, q = y.shape
    if n != nq:
        raise ValueError("x and y must have same number of rows")
        
    xt = x.t() # p x n
    yt = y.t() # q x n
    
    dist_mat = torch.cdist(xt, yt, p=2) # p x q
    
    if kmetric in ["correlation", "covariance"]:
        xt_centered = xt - torch.mean(xt, dim=1, keepdim=True)
        yt_centered = yt - torch.mean(yt, dim=1, keepdim=True)
        if kmetric == "correlation":
            xt_std = torch.std(xt, dim=1, keepdim=True)
            yt_std = torch.std(yt, dim=1, keepdim=True)
            xt_std[xt_std == 0] = 1.0
            yt_std[yt_std == 0] = 1.0
            xt_centered = xt_centered / xt_std
            yt_centered = yt_centered / yt_std
            
        sim_mat = (xt_centered @ yt_centered.t()) / (n - 1)
        vals, indices = torch.topk(sim_mat, k=k, dim=0, largest=True) # top k per column (y)
    else:
        vals, indices = torch.topk(dist_mat, k=k, dim=0, largest=False)
        sim_mat = dist_mat
        
    mask = torch.zeros_like(sim_mat, dtype=torch.bool)
    mask.scatter_(0, indices, True)
    
    if kmetric == "gaussian" and sigma is not None:
        res_mat = torch.exp(-1.0 * (sim_mat**2) / (2.0 * sigma**2))
    else:
        res_mat = sim_mat
        
    res_mat = res_mat * mask.float()
    
    if kmetric in ["correlation", "covariance"]:
        if r != float('inf'):
            res_mat[res_mat < r] = 0
    else:
        if r != float('inf'):
            res_mat[res_mat > r] = 0
            
    return res_mat
