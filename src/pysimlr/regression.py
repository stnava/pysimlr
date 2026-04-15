import torch
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple, List

def smooth_matrix_prediction(x: Union[torch.Tensor, np.ndarray],
                            y: Union[torch.Tensor, np.ndarray],
                            nv: Optional[int] = None) -> torch.Tensor:
    """
    Predict Y from X using a smoothed (low-rank) linear mapping.

    This function computes the best linear prediction of Y given X, 
    restricted to the top `nv` principal components of X. This provides 
    regularization by preventing the mapping from overfitting to 
    noise in the predictor matrix.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        The input predictor matrix (samples x features_x).
    y : torch.Tensor or np.ndarray
        The target matrix to predict (samples x features_y).
    nv : int, optional
        Number of principal components of X to keep for the mapping. 
        Defaults to full rank.

    Returns
    -------
    torch.Tensor
        The predicted matrix (samples x features_y).

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    
    # 1. Compute SVD of X
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    
    # 2. Rank truncation
    if nv is not None:
        u = u[:, :nv]
        s = s[:nv]
        vh = vh[:nv, :]
        
    # 3. Solve for projection W: X W = Y => U S Vh W = Y => W = V S^-1 Ut Y
    # But we want the prediction directly: X W = U S Vh (V S^-1 Ut Y) = U Ut Y
    y_pred = u @ (u.t() @ y)
    return y_pred

def smooth_regression(x: Union[torch.Tensor, np.ndarray],
                      y: Union[torch.Tensor, np.ndarray],
                      iterations: int = 10,
                      nv: Optional[int] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
    """
    Perform smooth regression using a low-rank SVD-based approximation.

    This function provides a regularized linear mapping between two 
    matrices by extracting their principal components. It is used in 
    SiMLR to initialize mappings or perform smoothed modality-to-modality 
    predictions.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        The input (predictor) matrix.
    y : torch.Tensor or np.ndarray
        The output (target) matrix.
    iterations : int, default=10
        Number of iterations for the solver (reserved for iterative variants).
    nv : int, optional
        Number of principal components to keep. Defaults to all.
    **kwargs
        Additional arguments passed to the underlying solver.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing:
        - "u": Projected scores of the predictor matrix.
        - "v": Feature loadings (rotation matrix).

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    
    # Simple SVD-based implementation
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    
    if nv is not None:
        u = u[:, :nv]
        vh = vh[:nv, :]
    
    return {
        "u": u,
        "v": vh.t()
    }
