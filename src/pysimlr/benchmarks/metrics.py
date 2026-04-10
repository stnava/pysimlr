import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, List, Optional, Union
from pysimlr import adjusted_rvcoef
from pysimlr.utils import invariant_orthogonality_defect

def latent_recovery_score(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """Calculate Adjusted RV coefficient between predicted and true latents."""
    return adjusted_rvcoef(u_pred, u_true)

def in_sample_latent_linear_fit_r2(u_pred: torch.Tensor, y_true: np.ndarray) -> float:
    """
    Deprecated: Use heldout_outcome_r2_score for true benchmarks.
    Calculate R2 score for outcome prediction using predicted latents, fitting on the same data.
    """
    u_np = u_pred.detach().cpu().numpy()
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    reg = LinearRegression().fit(u_np, y_true)
    return r2_score(y_true, reg.predict(u_np))

def outcome_r2_score(u_pred: torch.Tensor, y_true: np.ndarray) -> float:
    """Alias for backward compatibility."""
    return in_sample_latent_linear_fit_r2(u_pred, y_true)

def heldout_outcome_r2_score(u_train: torch.Tensor, y_train: np.ndarray, 
                             u_test: torch.Tensor, y_test: np.ndarray) -> float:
    """
    Calculate R2 score for outcome prediction on held-out data.
    Fits a downstream model on train latents and scores on test latents.
    """
    u_train_np = u_train.detach().cpu().numpy()
    u_test_np = u_test.detach().cpu().numpy()
    
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
        
    reg = LinearRegression().fit(u_train_np, y_train)
    y_pred = reg.predict(u_test_np)
    return r2_score(y_test, y_pred)

def heldout_outcome_mse(u_train: torch.Tensor, y_train: np.ndarray, 
                        u_test: torch.Tensor, y_test: np.ndarray) -> float:
    """
    Calculate MSE for outcome prediction on held-out data.
    Fits a downstream model on train latents and scores on test latents.
    """
    u_train_np = u_train.detach().cpu().numpy()
    u_test_np = u_test.detach().cpu().numpy()
    
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
        
    reg = LinearRegression().fit(u_train_np, y_train)
    y_pred = reg.predict(u_test_np)
    return mean_squared_error(y_test, y_pred)

def reconstruction_mse(data: List[torch.Tensor], recons: List[torch.Tensor]) -> float:
    """Calculate average normalized reconstruction error across modalities."""
    errors = []
    for x, r in zip(data, recons):
        err = torch.norm(x - r, p='fro').item() / (torch.norm(x, p='fro').item() + 1e-10)
        errors.append(err)
    return float(np.mean(errors))

def latent_variance_diagnostics(u: torch.Tensor) -> Dict[str, float]:
    """Check for latent collapse and scaling issues."""
    u_std = torch.std(u, dim=0)
    u_norm = torch.norm(u, dim=1)
    
    return {
        "u_std_mean": float(torch.mean(u_std).item()),
        "u_std_min": float(torch.min(u_std).item()),
        "u_norm_mean": float(torch.mean(u_norm).item()),
        "u_norm_sd": float(torch.std(u_norm).item()),
        "collapsed_dims": int(torch.sum(u_std < 1e-3).item())
    }

def shared_private_diagnostics(shared_latents: List[torch.Tensor], 
                               private_latents: List[torch.Tensor]) -> Dict[str, float]:
    """Calculate diagnostics specifically for models with shared/private branches."""
    diagnostics = {}
    
    for i, (s, p) in enumerate(zip(shared_latents, private_latents)):
        s_var = torch.mean(torch.var(s, dim=0)).item()
        p_var = torch.mean(torch.var(p, dim=0)).item()
        
        # Cross-covariance (absolute mean of cov matrix)
        s_c = s - s.mean(dim=0, keepdim=True)
        p_c = p - p.mean(dim=0, keepdim=True)
        cross_cov = torch.abs((s_c.T @ p_c) / (s.shape[0] - 1)).mean().item()
        
        diagnostics[f"mod{i}_shared_var"] = s_var
        diagnostics[f"mod{i}_private_var"] = p_var
        diagnostics[f"mod{i}_cross_cov"] = cross_cov
        
    return diagnostics

def calculate_v_orthogonality(v_mats: List[torch.Tensor]) -> float:
    """Calculate mean invariant orthogonality defect across all modality V matrices."""
    defects = []
    for v in v_mats:
        defects.append(invariant_orthogonality_defect(v).item())
    return float(np.mean(defects))

def calculate_all_metrics(u_pred: torch.Tensor, 
                          u_true: torch.Tensor, 
                          y_true: np.ndarray,
                          data: List[torch.Tensor],
                          recons: List[torch.Tensor],
                          shared_latents: Optional[List[torch.Tensor]] = None,
                          private_latents: Optional[List[torch.Tensor]] = None,
                          v_mats: Optional[List[torch.Tensor]] = None,
                          u_train: Optional[torch.Tensor] = None,
                          y_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Utility to calculate a standard suite of metrics.
    If u_train and y_train are provided, true held-out metrics are calculated.
    """
    res = {
        "recovery": latent_recovery_score(u_pred, u_true),
        "recon_error": reconstruction_mse(data, recons)
    }
    
    if u_train is not None and y_train is not None:
        res["test_r2"] = heldout_outcome_r2_score(u_train, y_train, u_pred, y_true)
        res["test_mse"] = heldout_outcome_mse(u_train, y_train, u_pred, y_true)
    else:
        res["in_sample_r2"] = in_sample_latent_linear_fit_r2(u_pred, y_true)
        # Keep test_r2 for backward compatibility but use in-sample value
        res["test_r2"] = res["in_sample_r2"]
        
    res.update(latent_variance_diagnostics(u_pred))
    
    if shared_latents is not None and private_latents is not None:
        res.update(shared_private_diagnostics(shared_latents, private_latents))
        
    if v_mats is not None:
        res["orthogonality_defect"] = calculate_v_orthogonality(v_mats)
        
    return res
