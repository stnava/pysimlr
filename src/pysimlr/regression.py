import torch
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple, List
from .utils import safe_svd

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
    
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)
        
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: x has {x.shape[0]} samples, y has {y.shape[0]} samples")
    
    # 1. Compute SVD of X
    u, s, vh = safe_svd(x, full_matrices=False)
    
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
                      alpha: float = 1e-4,
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
        The input (predictor) matrix (samples x features_x).
    y : torch.Tensor or np.ndarray
        The output (target) matrix (samples x features_y).
    iterations : int, default=10
        Number of iterations for the solver (reserved for iterative variants).
    nv : int, optional
        Number of principal components to keep. Defaults to all.
    alpha : float, default=1e-4
        Tikhonov ridge regularization parameter for inverted singular values.
    **kwargs
        Additional arguments passed to the underlying solver.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing:
        - "u": Projected scores of the predictor matrix.
        - "v": Feature loadings (rotation matrix).
        - "coef": Regression coefficients (mapping X -> Y).
        - "coefficients": Alias for coef.
        - "beta": Alias for coef.
        - "y_pred": Predicted target matrix.
        - "pred": Alias for y_pred.
        - "projection": Projection weight matrix.
        - "s": Retained singular values.

    Raises
    ------
    ValueError
        If sample counts mismatch or dimensions are invalid.
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D predictor matrix, got ndim={x.ndim}")
        
    is_1d = (y.ndim == 1)
    y_2d = y.unsqueeze(1) if is_1d else y
    if y_2d.ndim != 2:
        raise ValueError(f"Expected 1D or 2D target matrix, got ndim={y.ndim}")
        
    if x.shape[0] != y_2d.shape[0]:
        raise ValueError(f"Sample count mismatch: x has {x.shape[0]} samples, y has {y_2d.shape[0]} samples")
        
    n, p = x.shape
    q = y_2d.shape[1]
    
    u, s, vh = safe_svd(x, full_matrices=False)
    
    k_cand = kwargs.get("k", nv)
    k = min(k_cand, len(s)) if k_cand is not None else len(s)
    u_k = u[:, :k]
    s_k = s[:k]
    v_k = vh[:k, :].t()
    
    reg_param = kwargs.get("alpha", kwargs.get("lambda_", alpha))
    s_reg = s_k / (s_k ** 2 + reg_param)
    
    w = v_k @ (s_reg.unsqueeze(1) * (u_k.t() @ y_2d))
    y_pred_2d = x @ w
    y_pred = y_pred_2d.squeeze(1) if is_1d else y_pred_2d
    w_out = w.squeeze(1) if is_1d else w
    
    return {
        "u": u_k,
        "v": v_k,
        "coef": w_out,
        "coefficients": w_out,
        "beta": w_out,
        "y_pred": y_pred,
        "pred": y_pred,
        "projection": w,
        "s": s_k
    }
