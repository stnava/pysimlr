import torch
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple, List
from .utils import safe_svd
from .nsa_backend import load_nsa_estimator

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
        Unused. The ridge solution is computed in closed form from a single SVD,
        so there is nothing to iterate; the argument is retained only for
        backwards compatibility.
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


def build_nsa_pipeline(
    n_components: int = 6,
    w: float = 0.5,
    task: str = "classification",
    model: Optional[Any] = None,
    consolidate: bool = True,
    scale: bool = True,
    **nsa_kwargs: Any,
) -> Any:
    """
    Build a turnkey scikit-learn Pipeline with NSA-Flow dimensionality reduction.

    Adapts automatically to data sign and distribution:
    - If X >= 0, NSA-Flow extracts non-negative constituent components (mode="data").
    - If X has negative values, NSA-Flow uses signed contrast lifting with consolidate=True
      to produce strictly disjoint lobes with zero crosstalk and zero frame defect.

    Parameters
    ----------
    n_components : int, default=6
        Number of components to extract.
    w : float, default=0.5
        Trade-off weight between data fidelity (w=0.0) and frame defect/orthogonality (w=1.0).
        Default 0.5 provides the recommended balanced trade-off.
    task : {"classification", "regression"}, default="classification"
        Task type used to pick the default downstream predictor if `model` is None.
    model : estimator, optional
        Downstream scikit-learn estimator. If None, defaults to LogisticRegression(C=1.0, max_iter=500)
        for classification, or Ridge(alpha=1.0) for regression.
    consolidate : bool, default=True
        Whether to enforce strictly disjoint supports in signed mode (eliminating crosstalk).
    scale : bool, default=True
        Whether to prepend StandardScaler() to the pipeline.
    **nsa_kwargs : Any
        Additional keyword arguments forwarded to NSAFlow (e.g. mode="auto", max_iter=1000).

    Returns
    -------
    sklearn.pipeline.Pipeline
        The constructed turnkey pipeline.
    """
    nsa_cls = load_nsa_estimator()
    if nsa_cls is None:
        raise ImportError(
            "NSAFlow estimator is not available. Install nsa_flow to use build_nsa_pipeline."
        )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))

    kwargs = {"mode": "auto", "consolidate": consolidate}
    kwargs.update(nsa_kwargs)
    steps.append(("dim_reduction", nsa_cls(n_components=n_components, w=w, **kwargs)))

    if model is None:
        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=1.0, max_iter=500)
            steps.append(("classifier", model))
        elif task == "regression":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            steps.append(("regressor", model))
        else:
            raise ValueError(f"Unknown task: {task}. Expected 'classification' or 'regression'.")
    else:
        step_name = "classifier" if task == "classification" else "regressor"
        steps.append((step_name, model))

    return Pipeline(steps)

