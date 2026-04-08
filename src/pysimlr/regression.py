import torch
import pandas as pd
from typing import List, Optional, Union, Dict, Any
from .sparsification import orthogonalize_and_q_sparsify
from .svd import ba_svd

def smooth_matrix_prediction(x: Union[torch.Tensor, Any],
                             basis_df: pd.DataFrame,
                             iterations: int = 10,
                             gamma: float = 1.e-6,
                             sparseness_quantile: float = 0.0,
                             positivity: str = "either",
                             smoothing_matrix: Optional[torch.Tensor] = None,
                             verbose: bool = False) -> Dict[str, Any]:
    """
    Reconstruct a n by p matrix given basis predictors using torch.
    """
    x = torch.as_tensor(x).float()
        
    n, p = x.shape
    if n != basis_df.shape[0]:
        raise ValueError("x and basis_df must have same number of rows")
        
    # Extract basis from dataframe
    u_np = basis_df.select_dtypes(include=['number']).values
    u = torch.as_tensor(u_np).float()
    # Standardize u
    u = (u - torch.mean(u, dim=0)) / (torch.std(u, dim=0) + 1e-8)
    k = u.shape[1]
    
    # Initialize v
    v = torch.randn(p, k) * 0.01
    
    if smoothing_matrix is None:
        smoothing_matrix = torch.eye(p)
    else:
        smoothing_matrix = torch.as_tensor(smoothing_matrix).float()
        
    errs = []
    for i in range(iterations):
        # Smoothing
        v = smoothing_matrix @ v
        
        # Gradient descent
        # grad = (U^T U V^T - U^T X)^T = V (U^T U) - X^T U
        tu = u.t()
        tuu = tu @ u
        grad = v @ tuu - x.t() @ u
        v = v - grad * gamma
        
        # Orthogonalize and sparsify
        v = orthogonalize_and_q_sparsify(v, sparseness_quantile, positivity, unit_norm=True)
        
        # Intercept and error
        proj = u @ v.t()
        intercept = torch.mean(x - proj, dim=1, keepdim=True)
        err = torch.mean(torch.abs(x - (proj + intercept)))
        errs.append(err.item())
        
        if verbose:
            print(f"Iteration {i}: error {err.item()}")
            
    return {
        "u": u,
        "v": v,
        "intercept": intercept,
        "errors": errs
    }

def smooth_regression(x: Union[torch.Tensor, Any],
                      y: Union[torch.Tensor, Any],
                      iterations: int = 10,
                      sparseness_quantile: float = 0.0,
                      positivity: str = "either",
                      smoothing_matrix: Optional[torch.Tensor] = None,
                      nv: int = 2,
                      verbose: bool = False) -> Dict[str, Any]:
    """
    Reconstruct a n by 1 vector given n by p matrix of predictors using torch.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
        
    n, p = x.shape
    x_centered = x - torch.mean(x, dim=0)
    y_scaled = (y - torch.mean(y)) / (torch.std(y) + 1e-8)
    if y_scaled.dim() == 1:
        y_scaled = y_scaled.unsqueeze(1)
    
    if smoothing_matrix is None:
        smoothing_matrix = torch.eye(p)
    else:
        smoothing_matrix = torch.as_tensor(smoothing_matrix).float()
        
    # Initialize v
    # xgy <- scaledY %*% x
    xgy = (y_scaled.t() @ x_centered).squeeze(0) # length p
    v = xgy.repeat(nv, 1) + torch.randn(nv, p) * 1e-3
    
    gamma = 1e-8
    errs = []
    
    for i in range(iterations):
        # Gradient descent
        xv_t = x_centered @ v.t()
        temp = xv_t.t() @ x_centered
        grad = temp - xgy.unsqueeze(0)
        
        v = v - grad * gamma
        
        # Apply smoothing
        v = v @ smoothing_matrix
        
        # Sparsify and orthogonalize (v is nv x p, we need to transpose for our helper)
        v = orthogonalize_and_q_sparsify(v.t(), sparseness_quantile, positivity, unit_norm=True).t()
        
        # Adaptive gamma
        if i < 2:
            nonzero_v = v[v != 0]
            if len(nonzero_v) > 0:
                gamma = torch.quantile(torch.abs(nonzero_v), 0.5).item() * 1e-2
            else:
                gamma = 1e-8
            
        proj = x_centered @ v.t()
        intercept = torch.mean(y_scaled - proj, dim=0)
        err = torch.mean(torch.abs(y_scaled - (proj + intercept)))
        errs.append(err.item())
        
        if verbose:
            print(f"Iteration {i}: error {err.item()}")
            
    return {
        "v": v.t(),
        "errors": errs
    }
