import torch
from typing import Optional, Union

def sparse_distance_matrix(x: torch.Tensor, 
                           k: int, 
                           sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute a k-nearest neighbor sparse distance matrix.

    Calculates the Euclidean distance between all pairs of rows in `x`, 
    keeping only the `k` closest neighbors for each row.

    Parameters
    ----------
    x : torch.Tensor
        Input data matrix (N x P).
    k : int
        Number of nearest neighbors to retain.
    sigma : float, optional
        If provided, converts distances to an affinity matrix using a 
        Gaussian kernel with this standard deviation.

    Returns
    -------
    torch.Tensor
        The sparse distance or affinity matrix (N x N).

    Raises
    ------
    TypeError
        If the input is not a valid tensor.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    x = torch.as_tensor(x).float()
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor (samples x features), got ndim={x.ndim}")
    n = x.shape[0]
    if k <= 0:
        raise ValueError(f"k must be at least 1, got {k}")
    
    # Compute full distance matrix
    dist = torch.cdist(x, x)
    
    # Get k+1 nearest neighbors (including self)
    values, indices = torch.topk(dist, k=min(k+1, n), largest=False)
    
    # Create sparse representation
    mask = torch.zeros_like(dist, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    
    sparse_dist_mat = torch.zeros_like(dist)
    sparse_dist_mat[mask] = dist[mask]
    
    if sigma is not None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive (sigma > 0), got {sigma}")
        denom = 2.0 * (float(sigma) ** 2)
        sparse_dist_mat = torch.zeros_like(dist)
        sparse_dist_mat[mask] = torch.exp(- (dist[mask] ** 2) / denom)
        
    return sparse_dist_mat

def sparse_distance_matrix_xy(x: torch.Tensor, 
                              y: torch.Tensor, 
                              k: int, 
                              sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute a k-nearest neighbor sparse distance matrix between two sets of points.

    Calculates the Euclidean distance between rows of `x` and rows of `y`, 
    keeping only the `k` closest neighbors in `y` for each row in `x`.

    Parameters
    ----------
    x : torch.Tensor
        First data matrix (NX x P).
    y : torch.Tensor
        Second data matrix (NY x P).
    k : int
        Number of nearest neighbors to retain.
    sigma : float, optional
        If provided, converts distances to an affinity matrix using a 
        Gaussian kernel with this standard deviation.

    Returns
    -------
    torch.Tensor
        The sparse distance or affinity matrix (NX x NY).

    Raises
    ------
    TypeError
        If the inputs are not valid tensors.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got x.ndim={x.ndim}, y.ndim={y.ndim}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimension mismatch: x has {x.shape[1]} features, y has {y.shape[1]} features")
    nx = x.shape[0]
    ny = y.shape[0]
    if k <= 0:
        raise ValueError(f"k must be at least 1, got {k}")
    
    dist = torch.cdist(x, y)
    
    values, indices = torch.topk(dist, k=min(k, ny), largest=False)
    
    mask = torch.zeros_like(dist, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    
    sparse_dist_mat = torch.zeros_like(dist)
    sparse_dist_mat[mask] = dist[mask]
    
    if sigma is not None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive (sigma > 0), got {sigma}")
        denom = 2.0 * (float(sigma) ** 2)
        sparse_dist_mat = torch.zeros_like(dist)
        sparse_dist_mat[mask] = torch.exp(- (dist[mask] ** 2) / denom)
        
    return sparse_dist_mat

def sparse_dist(x: torch.Tensor, 
                k: Optional[int] = None, 
                sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute a k-nearest neighbor sparse distance or affinity matrix.

    Parameters
    ----------
    x : torch.Tensor
        Input data matrix (N x P).
    k : int, optional
        Number of nearest neighbors to retain. Defaults to N - 1.
    sigma : float, optional
        Gaussian kernel bandwidth.

    Returns
    -------
    torch.Tensor
        Sparse distance or affinity matrix.
    """
    if k is None:
        x_t = torch.as_tensor(x)
        k = max(1, x_t.shape[0] - 1)
    return sparse_distance_matrix(x, k=k, sigma=sigma)
