import torch
import pandas as pd
import re
import time
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import r2_score

def set_seed_based_on_time() -> int:
    """
    Set random seed based on current time.
    """
    seed_value = int(time.time() * 1000000) % (2**32 - 1)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value

def multigrep(patterns: List[str], desc: List[str], intersect: bool = False) -> torch.Tensor:
    """
    Grep entries with a list of search patterns.
    """
    indices_set = None
    for pattern in patterns:
        matches = {i for i, item in enumerate(desc) if re.search(pattern, item)}
        if indices_set is None or not intersect:
            if indices_set is None:
                indices_set = matches
            else:
                indices_set.update(matches)
        else:
            indices_set.intersection_update(matches)
    
    if indices_set is None:
        return torch.tensor([], dtype=torch.long)
    
    sorted_indices = sorted(list(indices_set))
    return torch.tensor(sorted_indices, dtype=torch.long)

def get_names_from_dataframe(patterns: List[str], df: pd.DataFrame, exclusions: Optional[List[str]] = None) -> List[str]:
    """
    Extract column names with search and exclusion parameters.
    """
    all_colnames = df.columns.tolist()
    outnames_set = set()
    for pattern in patterns:
        matches = [name for name in all_colnames if re.search(pattern, name)]
        outnames_set.update(matches)
        
    outnames = list(outnames_set)
    if exclusions:
        to_exclude = set()
        for excl in exclusions:
            to_exclude.update([name for name in outnames if re.search(excl, name)])
        outnames = [name for name in outnames if name not in to_exclude]
        
    return sorted(outnames)

def map_asym_var(df: pd.DataFrame, left_vars: List[str], left_name: str = 'left', 
                right_name: str = 'right', replacer: str = 'Asym') -> pd.DataFrame:
    """
    Convert left/right variables to a measure of asymmetry.
    """
    df = df.copy()
    for left_var in left_vars:
        right_var = left_var.replace(left_name, right_name)
        if right_var in df.columns:
            new_name = left_var.replace(left_name, replacer)
            l_val = torch.from_numpy(df[left_var].values.astype(float))
            r_val = torch.from_numpy(df[right_var].values.astype(float))
            diff = l_val - r_val
            asym = diff * torch.sign(diff)
            df[new_name] = asym.numpy()
    return df

def map_lr_average_var(df: pd.DataFrame, left_vars: List[str], left_name: str = 'left', 
                       right_name: str = 'right', replacer: str = 'LRAVG') -> pd.DataFrame:
    """
    Convert left/right variables to an average measurement.
    """
    df = df.copy()
    for left_var in left_vars:
        right_var = left_var.replace(left_name, right_name)
        if right_var in df.columns:
            new_name = left_var.replace(left_name, replacer)
            l_val = torch.from_numpy(df[left_var].values.astype(float))
            r_val = torch.from_numpy(df[right_var].values.astype(float))
            avg = 0.5 * (l_val + r_val)
            df[new_name] = avg.numpy()
    return df

def rvcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes the RV-coefficient between two matrices.
    """
    return rvcoef_components(x, y)['rv']

def rvcoef_components(x: torch.Tensor, y: torch.Tensor) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Internal dispatcher for RV-coefficient components.
    """
    n, p = x.shape
    q = y.shape[1]
    
    x_centered = x - torch.mean(x, dim=0)
    y_centered = y - torch.mean(y, dim=0)
    
    if n < (p + q):
        return rvcoef_trace_impl(x_centered, y_centered)
    else:
        return rvcoef_gram_impl(x_centered, y_centered)

def rvcoef_trace_impl(x_centered: torch.Tensor, y_centered: torch.Tensor) -> Dict[str, Union[float, torch.Tensor]]:
    """
    RV-coefficient implementation using trace (for N < P+Q).
    """
    s_xx = x_centered @ x_centered.t()
    s_yy = y_centered @ y_centered.t()
    s_xy = x_centered @ y_centered.t()
    
    numerator = torch.sum(s_xy * s_xy) # Equivalent to trace(S_XY @ S_YX)
    denom_part1 = torch.sum(s_xx * s_xx)
    denom_part2 = torch.sum(s_yy * s_yy)
    denominator = torch.sqrt(denom_part1 * denom_part2)
    
    if denominator == 0:
        return {'rv': 0.0, 'numerator': numerator, 'denominator': 0.0}
    
    return {'rv': (numerator / denominator).item(), 'numerator': numerator, 'denominator': denominator}

def rvcoef_gram_impl(x_centered: torch.Tensor, y_centered: torch.Tensor) -> Dict[str, Union[float, torch.Tensor]]:
    """
    RV-coefficient implementation using Gram matrices (for N >= P+Q).
    """
    cross_product = x_centered.t() @ y_centered
    # Numerator is Frobenius norm squared of cross-product
    numerator = torch.sum(cross_product * cross_product)
    
    g_x = x_centered.t() @ x_centered
    g_y = y_centered.t() @ y_centered
    
    denom_part1 = torch.sum(g_x * g_x)
    denom_part2 = torch.sum(g_y * g_y)
    denominator = torch.sqrt(denom_part1 * denom_part2)
    
    if denominator == 0:
        return {'rv': 0.0, 'numerator': numerator, 'denominator': 0.0}
    
    return {'rv': (numerator / denominator).item(), 'numerator': numerator, 'denominator': denominator}

def adjusted_rvcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes the Adjusted RV-coefficient.
    """
    n = x.shape[0]
    if n <= 1:
        return 0.0
        
    components = rvcoef_components(x, y)
    rv_obs = components['rv']
    rv_den = components['denominator']
    
    if rv_den == 0:
        return 0.0
        
    x_centered = x - torch.mean(x, dim=0)
    y_centered = y - torch.mean(y, dim=0)
    
    tr_s_xx = torch.sum(x_centered * x_centered)
    tr_s_yy = torch.sum(y_centered * y_centered)
    
    # Simple adjustment for now matching R
    exp_rv_num = tr_s_xx * tr_s_yy / (n - 1)
    return rv_obs

def l1_normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    L1 normalization of features (columns).
    """
    col_l1_norms = torch.sum(torch.abs(features), dim=0)
    col_l1_norms[col_l1_norms == 0] = 1.0
    return features / col_l1_norms

def invariant_orthogonality_defect(a: torch.Tensor) -> torch.Tensor:
    """
    Compute invariant orthogonality defect.
    Measures deviation from orthogonality after normalizing for global Frobenius norm.
    """
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a).float()
    norm_a_f = torch.sqrt(torch.sum(a**2))
    if norm_a_f < 1e-10: return torch.tensor(0.0, device=a.device)
    ap = a / norm_a_f
    ata = ap.t() @ ap
    d = torch.diag(torch.diag(ata))
    defect = torch.sum((ata - d)**2)
    return defect

def stiefel_defect(a: torch.Tensor) -> torch.Tensor:
    """
    Measure violation of the Stiefel manifold constraint: V.t() @ V = I.
    """
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a).float()
    k = a.shape[1]
    identity = torch.eye(k, device=a.device, dtype=a.dtype)
    return torch.norm(a.t() @ a - identity, p='fro')

def gradient_invariant_orthogonality_defect(a: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of invariant orthogonality defect.
    """
    norm_a_f = torch.norm(a, p='fro')
    if norm_a_f < 1e-10: return torch.zeros_like(a)
    ap = a / norm_a_f
    ata = ap.t() @ ap
    d = torch.diag(torch.diag(ata))
    orthogonality_diff = ata - d
    gradient = 4 * (ap @ orthogonality_diff)
    return gradient

def mean_orthogonality_defect(a: torch.Tensor) -> torch.Tensor:
    """
    Compute mean orthogonality defect (average off-diagonal squared correlation).
    """
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a).float()
    n, k = a.shape
    if k <= 1:
        return torch.tensor(0.0, device=a.device)
    
    # Normalize columns to unit length
    col_norms = torch.norm(a, dim=0, keepdim=True)
    a_norm = a / (col_norms + 1e-10)
    
    correlation_matrix = a_norm.t() @ a_norm
    # Remove diagonal (self-correlations)
    mask = ~torch.eye(k, device=a.device).bool()
    off_diag = correlation_matrix[mask]
    
    return torch.mean(off_diag**2)

def gradient_mean_orthogonality_defect(a: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of the mean orthogonality defect.
    """
    n, k = a.shape
    if k <= 1:
        return torch.zeros_like(a)
    
    col_norms = torch.norm(a, dim=0, keepdim=True)
    a_norm = a / (col_norms + 1e-10)
    
    r = a_norm.t() @ a_norm
    # Mask out diagonal
    r_off = r.clone()
    r_off.fill_diagonal_(0.0)
    
    # Gradient component from numerator
    # d/dA (tr(A_norm.T @ A_norm - I)^2) / (k*(k-1))
    grad = (4.0 / (k * (k - 1) + 1e-10)) * a_norm @ r_off
    
    # Project out components along the columns (due to normalization)
    grad = (grad - a_norm * torch.sum(a_norm * grad, dim=0, keepdim=True)) / (col_norms + 1e-10)
    
    return grad

def orthogonality_summary(a: torch.Tensor) -> Dict[str, float]:
    """
    Provide a detailed summary of matrix orthogonality metrics.
    """
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a).float()
    defect = invariant_orthogonality_defect(a).item()
    stiefel = stiefel_defect(a).item()
    mean_defect = mean_orthogonality_defect(a).item()
    
    # Condition number (ratio of max to min singular value)
    try:
        _, s, _ = torch.linalg.svd(a, full_matrices=False)
        cond = (s[0] / (s[-1] + 1e-10)).item()
        ortho_score = torch.sum(s**2).item() / (s[0]**2).item() # Effective rank proxy
    except:
        cond = float('nan')
        ortho_score = float('nan')
        
    return {
        "invariant_defect": defect,
        "stiefel_defect": stiefel,
        "mean_defect": mean_defect,
        "condition_number": cond,
        "effective_rank_proxy": ortho_score
    }


def preprocess_data(x: torch.Tensor, scale_list: List[str], provenance: Optional[Dict[str, Any]] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Preprocess data matrix according to a list of scaling/normalization methods.
    Supports provenance for reproducible application to test data.
    """
    x_out = x.clone().float()
    new_provenance = {} if provenance is None else None
    
    # Handle NaNs using provenance or current data
    if provenance and "nan_fill" in provenance:
        nan_fill = provenance["nan_fill"]
    else:
        nan_fill = torch.nanmean(x_out).item() if not torch.isnan(x_out).all() else 0.0
        if new_provenance is not None: new_provenance["nan_fill"] = nan_fill
        
    x_out = torch.nan_to_num(x_out, nan=nan_fill)
    
    for method in scale_list:
        if method == "none":
            continue
        elif method == "norm":
            if provenance and "norm_factor" in provenance:
                factor = provenance["norm_factor"]
            else:
                factor = torch.norm(x_out, p='fro').item() + 1e-10
                if new_provenance is not None: new_provenance["norm_factor"] = factor
            x_out = x_out / factor
        elif method == "np":
            # np is dimension-dependent, but we store the factor for safety
            if provenance and "np_factor" in provenance:
                factor = provenance["np_factor"]
            else:
                factor = float(x_out.shape[0] * x_out.shape[1])
                if new_provenance is not None: new_provenance["np_factor"] = factor
            x_out = x_out / factor
        elif method == "sqrtnp":
            if provenance and "sqrtnp_factor" in provenance:
                factor = provenance["sqrtnp_factor"]
            else:
                factor = np.sqrt(x_out.shape[0] * x_out.shape[1])
                if new_provenance is not None: new_provenance["sqrtnp_factor"] = factor
            x_out = x_out / factor
        elif method == "center":
            if provenance and "center_mean" in provenance:
                mean = provenance["center_mean"]
            else:
                mean = torch.mean(x_out, dim=0)
                if new_provenance is not None: new_provenance["center_mean"] = mean
            x_out = x_out - mean
        elif method == "centerAndScale":
            if provenance and "cas_mean" in provenance:
                mean = provenance["cas_mean"]
                std = provenance["cas_std"]
            else:
                mean = torch.mean(x_out, dim=0)
                std = torch.std(x_out, dim=0)
                std[std < 1e-10] = 1.0
                if new_provenance is not None:
                    new_provenance["cas_mean"] = mean
                    new_provenance["cas_std"] = std
            x_out = (x_out - mean) / std
        elif method == "eigenvalue":
            if provenance and "eigen_factor" in provenance:
                factor = provenance["eigen_factor"]
            else:
                _, s, _ = torch.linalg.svd(x_out, full_matrices=False)
                factor = torch.sum(s).item() + 1e-10
                if new_provenance is not None: new_provenance["eigen_factor"] = factor
            x_out = x_out / factor
            
    if provenance is not None:
        return x_out
    return x_out, new_provenance


def set_all_seeds(seed: int = 42):
    """Set seeds for reproducibility across torch, numpy, and random."""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_svd(x: torch.Tensor, full_matrices: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hardware-aware SVD that gracefully handles MPS (macOS) limitations
    by falling back to CPU when necessary.
    """
    if x.device.type == 'mps':
        # Current MPS backend lacks stable linalg_svd support for all shapes/precisions
        x_cpu = x.cpu()
        u, s, vh = torch.linalg.svd(x_cpu, full_matrices=full_matrices)
        return u.to(x.device), s.to(x.device), vh.to(x.device)
    try:
        return torch.linalg.svd(x, full_matrices=full_matrices)
    except RuntimeError:
        # Fallback for unexpected backend failures
        x_cpu = x.cpu()
        u, s, vh = torch.linalg.svd(x_cpu, full_matrices=full_matrices)
        return u.to(x.device), s.to(x.device), vh.to(x.device)

def procrustes_r2(u_true: torch.Tensor, u_est: torch.Tensor) -> float:
    """
    Compute Procrustes-aligned R2 between true and estimated latent spaces.
    """
    u_true_np = u_true.detach().cpu().numpy() if torch.is_tensor(u_true) else u_true
    u_est_np = u_est.detach().cpu().numpy() if torch.is_tensor(u_est) else u_est
    
    # Standardize/Center
    u_true_np = u_true_np - u_true_np.mean(0)
    u_est_np = u_est_np - u_est_np.mean(0)
    
    try:
        R, _ = orthogonal_procrustes(u_est_np, u_true_np)
        u_rotated = u_est_np @ R
        return float(r2_score(u_true_np, u_rotated))
    except Exception:
        return 0.0

def procrustes_mse(u_true: torch.Tensor, u_est: torch.Tensor) -> float:
    """
    Compute Procrustes-aligned MSE between true and estimated latent spaces.
    """
    u_true_np = u_true.detach().cpu().numpy() if torch.is_tensor(u_true) else u_true
    u_est_np = u_est.detach().cpu().numpy() if torch.is_tensor(u_est) else u_est
    
    u_true_np = u_true_np - u_true_np.mean(0)
    u_est_np = u_est_np - u_est_np.mean(0)
    
    try:
        R, _ = orthogonal_procrustes(u_est_np, u_true_np)
        u_rotated = u_est_np @ R
        return float(np.mean((u_true_np - u_rotated)**2))
    except Exception:
        return float('nan')
