import torch
from typing import Optional, Union

def sparse_distance_matrix(x: torch.Tensor, 
                           k: int, 
                           sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute a sparse distance matrix using k-nearest neighbors in torch.
    Returns a n x n distance matrix.
    """
    x = torch.as_tensor(x).float()
    n = x.shape[0]
    
    # Compute full distance matrix
    dist = torch.cdist(x, x)
    
    # Get k+1 nearest neighbors (including self)
    values, indices = torch.topk(dist, k=min(k+1, n), largest=False)
    
    # Create sparse representation
    mask = torch.zeros_like(dist, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    
    sparse_dist = torch.zeros_like(dist)
    sparse_dist[mask] = dist[mask]
    
    if sigma is not None:
        sparse_dist = torch.exp(-sparse_dist**2 / (2 * sigma**2))
        sparse_dist[~mask] = 0.0
        
    return sparse_dist

def sparse_distance_matrix_xy(x: torch.Tensor, 
                              y: torch.Tensor, 
                              k: int, 
                              sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute a sparse distance matrix between x and y in torch.
    Returns a nx x ny distance matrix.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    nx = x.shape[0]
    ny = y.shape[0]
    
    dist = torch.cdist(x, y)
    
    values, indices = torch.topk(dist, k=min(k, ny), largest=False)
    
    mask = torch.zeros_like(dist, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    
    sparse_dist = torch.zeros_like(dist)
    sparse_dist[mask] = dist[mask]
    
    if sigma is not None:
        sparse_dist = torch.exp(-sparse_dist**2 / (2 * sigma**2))
        sparse_dist[~mask] = 0.0
        
    return sparse_dist
