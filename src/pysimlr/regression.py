import torch
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any

def smooth_matrix_prediction(x: Union[torch.Tensor, np.ndarray],
                             basis_df: Union[pd.DataFrame, np.ndarray, torch.Tensor],
                             iterations: int = 10,
                             gamma: float = 1.e-6,
                             sparseness_quantile: float = 0.0,
                             positivity: str = "either",
                             smoothing_matrix: Optional[torch.Tensor] = None,
                             verbose: bool = False) -> torch.Tensor:
    """
    Reconstruct a n by p matrix given basis predictors using torch.
    """
    x = torch.as_tensor(x).float()
    
    n, p = x.shape
    if isinstance(basis_df, pd.DataFrame):
        u = torch.from_numpy(basis_df.select_dtypes(include=['number']).values).float()
    else:
        u = torch.as_tensor(basis_df).float()
        
    if n != u.shape[0]:
        raise ValueError("x and basis_df must have same number of rows")
        
    # Standard regression: V = (U^T U)^-1 U^T X
    try:
        v = torch.linalg.lstsq(u, x).solution
    except:
        v = torch.linalg.pinv(u) @ x
        
    return u @ v

def smooth_regression(x: Union[torch.Tensor, np.ndarray],
                      y: Union[torch.Tensor, np.ndarray],
                      iterations: int = 10,
                      nv: Optional[int] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
    """
    Simple smooth regression wrapper.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    
    # Simple SVD-based implementation
    u, s, v = torch.linalg.svd(x, full_matrices=False)
    
    if nv is not None:
        u = u[:, :nv]
        v = v[:nv, :]
    
    return {
        "u": u,
        "v": v.t()
    }
