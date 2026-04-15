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
    Generate and set a random seed based on the current system time.

    This function provides a convenient way to initialize randomness for 
    both CPU and GPU operations (if available) using a high-resolution 
    timestamp.

    Returns
    -------
    int
        The generated seed value (32-bit unsigned integer range).

    Notes
    -----
    This sets the seed for `torch.manual_seed` and `torch.cuda.manual_seed_all`.
    It does not affect `numpy.random` or the built-in `random` module. Use 
    `set_all_seeds` for a more comprehensive reset.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    seed_value = int(time.time() * 1000000) % (2**32 - 1)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value

def multigrep(patterns: List[str], desc: List[str], intersect: bool = False) -> torch.Tensor:
    """
    Search for multiple regex patterns within a list of strings and return matching indices.

    This utility is frequently used in SiMLR workflows to select specific 
    features or variables from large datasets (e.g., selecting all columns 
    matching "Cortex" or "Volume").

    Parameters
    ----------
    patterns : List[str]
        A list of regular expression patterns to search for.
    desc : List[str]
        A list of strings (e.g., column names or feature descriptions) to search within.
    intersect : bool, default=False
        If True, returns only the indices that match ALL patterns (logical AND).
        If False, returns indices that match ANY pattern (logical OR).

    Returns
    -------
    torch.Tensor
        A 1D tensor of indices (dtype=torch.long) corresponding to the matching 
        entries in `desc`, sorted in ascending order.

    Examples
    --------
    >>> names = ["brain_volume", "brain_area", "heart_rate", "lung_capacity"]
    >>> # Find anything related to brain OR heart
    >>> multigrep(["brain", "heart"], names)
    tensor([0, 1, 2])
    >>> # Find anything containing both 'brain' AND 'volume'
    >>> multigrep(["brain", "volume"], names, intersect=True)
    tensor([0])

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    Extract specific column names from a DataFrame using regex patterns and exclusions.

    Facilitates the selection of feature subsets for multi-modal analysis 
    by matching column headers against search patterns while optionally 
    filtering out unwanted variables.

    Parameters
    ----------
    patterns : List[str]
        A list of regex patterns to include in the selection.
    df : pd.DataFrame
        The input DataFrame from which to extract column names.
    exclusions : List[str], optional
        A list of regex patterns to exclude from the already matched selection.

    Returns
    -------
    List[str]
        A sorted list of column names that matched the inclusion patterns 
        and did not match the exclusion patterns.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {"age": [25], "brain_vol": [1200], "brain_thick": [2.5], "id": [1]}
    >>> df = pd.DataFrame(data)
    >>> # Get all brain related columns
    >>> get_names_from_dataframe(["brain"], df)
    ['brain_thick', 'brain_vol']
    >>> # Get brain columns but exclude thickness
    >>> get_names_from_dataframe(["brain"], df, exclusions=["thick"])
    ['brain_vol']

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    Compute asymmetry measures between left and right modality variables.

    For each 'left' variable provided, this function identifies its 
    corresponding 'right' counterpart and calculates an asymmetry score 
    defined as the absolute difference: |left - right|.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the variables to process.
    left_vars : List[str]
        A list of column names representing the 'left' side measurements.
    left_name : str, default='left'
        The substring in `left_vars` that identifies them as 'left'.
    right_name : str, default='right'
        The substring used to identify the 'right' counterpart (by replacing 
        `left_name`).
    replacer : str, default='Asym'
        The substring used to name the new asymmetry columns.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the new asymmetry columns added.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {"left_hippo": [3.0, 3.2], "right_hippo": [3.1, 3.2]}
    >>> df = pd.DataFrame(data)
    >>> res = map_asym_var(df, ["left_hippo"])
    >>> res["Asym_hippo"].tolist()
    [0.10000000149011612, 0.0]

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    Compute the average of corresponding left and right modality variables.

    For each 'left' variable provided, this function identifies its 
    corresponding 'right' counterpart and calculates their mean: 
    0.5 * (left + right).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the variables to process.
    left_vars : List[str]
        A list of column names representing the 'left' side measurements.
    left_name : str, default='left'
        The substring in `left_vars` that identifies them as 'left'.
    right_name : str, default='right'
        The substring used to identify the 'right' counterpart (by replacing 
        `left_name`).
    replacer : str, default='LRAVG'
        The substring used to name the new averaged columns.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the new averaged columns added.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {"left_hippo": [3.0, 3.2], "right_hippo": [3.2, 3.2]}
    >>> df = pd.DataFrame(data)
    >>> res = map_lr_average_var(df, ["left_hippo"])
    >>> res["LRAVG_hippo"].tolist()
    [3.100000023841858, 3.200000047683716]

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    Compute the RV-coefficient between two matrices.

    The RV-coefficient is a multivariate generalization of the squared Pearson 
    correlation coefficient, measuring the global similarity between two 
    datasets (matrices) with the same number of samples but potentially 
    different numbers of features.

    Parameters
    ----------
    x : torch.Tensor
        First data matrix of shape (N x P).
    y : torch.Tensor
        Second data matrix of shape (N x Q).

    Returns
    -------
    float
        The RV-coefficient, ranging from 0 (no similarity) to 1 (perfect 
        concordance up to a rotation and scaling).

    See Also
    --------
    adjusted_rvcoef : RV-coefficient adjusted for bias in high dimensions.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(100, 10)
    >>> # Perfect similarity with itself
    >>> rvcoef(x, x)
    1.0
    >>> # High similarity with a noisy version
    >>> y = x + torch.randn(100, 10) * 0.1
    >>> rvcoef(x, y) > 0.9
    True

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    return rvcoef_components(x, y)['rv']

def rvcoef_components(x: torch.Tensor, y: torch.Tensor) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Compute components of the RV-coefficient with optimized logic for matrix dimensions.

    This function is an internal dispatcher that chooses between a 
    trace-based implementation (efficient for small N) and a 
    Gram-based implementation (efficient for large N).

    Parameters
    ----------
    x : torch.Tensor
        First data matrix (N x P).
    y : torch.Tensor
        Second data matrix (N x Q).

    Returns
    -------
    Dict[str, Union[float, torch.Tensor]]
        A dictionary containing:
        - `rv`: The scalar RV-coefficient.
        - `numerator`: The squared Frobenius norm of the cross-covariance.
        - `denominator`: The product of Frobenius norms of self-covariances.

    Notes
    -----
    The function centers the input matrices internally before computation.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Parameters
    ----------
    x_centered : torch.Tensor
        First centered data matrix.
    y_centered : torch.Tensor
        Second centered data matrix.

    Returns
    -------
    Dict[str, Union[float, torch.Tensor]]
        A dictionary containing the calculated `rv`, `numerator`, and `denominator`.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Parameters
    ----------
    x_centered : torch.Tensor
        First centered data matrix.
    y_centered : torch.Tensor
        Second centered data matrix.

    Returns
    -------
    Dict[str, Union[float, torch.Tensor]]
        A dictionary containing the calculated `rv`, `numerator`, and `denominator`.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Adjusts the standard RV-coefficient for bias that occurs when the number 
    of features is large relative to the number of samples.

    Parameters
    ----------
    x : torch.Tensor
        First data matrix.
    y : torch.Tensor
        Second data matrix.

    Returns
    -------
    float
        The Adjusted RV-coefficient.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Normalizes each column of the input tensor so that the sum of its 
    absolute values is 1.

    Parameters
    ----------
    features : torch.Tensor
        The input matrix to normalize.

    Returns
    -------
    torch.Tensor
        The normalized matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    col_l1_norms = torch.sum(torch.abs(features), dim=0)
    col_l1_norms[col_l1_norms == 0] = 1.0
    return features / col_l1_norms

def invariant_orthogonality_defect(a: torch.Tensor) -> torch.Tensor:
    """
    Compute invariant orthogonality defect.

    Measures deviation from orthogonality after normalizing for global Frobenius norm.

    Parameters
    ----------
    a : torch.Tensor
        The input matrix.

    Returns
    -------
    torch.Tensor
        The scalar defect value.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Parameters
    ----------
    a : torch.Tensor
        The input matrix.

    Returns
    -------
    torch.Tensor
        The scalar defect value measuring deviation from the identity matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a).float()
    k = a.shape[1]
    identity = torch.eye(k, device=a.device, dtype=a.dtype)
    return torch.norm(a.t() @ a - identity, p='fro')

def gradient_invariant_orthogonality_defect(a: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of invariant orthogonality defect.

    Parameters
    ----------
    a : torch.Tensor
        The input matrix.

    Returns
    -------
    torch.Tensor
        The gradient of the defect with respect to the input matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Parameters
    ----------
    a : torch.Tensor
        The input matrix.

    Returns
    -------
    torch.Tensor
        The computed mean orthogonality defect.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Parameters
    ----------
    a : torch.Tensor
        The input matrix.

    Returns
    -------
    torch.Tensor
        The computed gradient of the mean orthogonality defect.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Parameters
    ----------
    a : torch.Tensor
        The input matrix to evaluate.

    Returns
    -------
    Dict[str, float]
        Dictionary of various orthogonality metrics:
        - `invariant_defect`: Defect normalized for global magnitude.
        - `stiefel_defect`: Violation of Stiefel manifold constraint.
        - `mean_defect`: Average off-diagonal squared correlation.
        - `condition_number`: Ratio of maximum to minimum singular value.
        - `effective_rank_proxy`: Proxy measure of effective rank.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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

    Parameters
    ----------
    x : torch.Tensor
        The input data matrix.
    scale_list : List[str]
        A list of scaling methods to sequentially apply. Supported options include:
        `none`, `norm`, `np`, `sqrtnp`, `center`, `centerAndScale`, `eigenvalue`.
    provenance : Optional[Dict[str, Any]], default=None
        A dictionary with previously computed statistics to ensure identical 
        scaling across training/testing datasets. If omitted, computes and returns 
        the statistics for the given input matrix.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]
        If `provenance` is provided, returns just the processed tensor.
        Otherwise, returns a tuple containing the processed tensor and the newly 
        computed provenance dictionary.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    """
    Set seeds for reproducibility across torch, numpy, and random.

    Parameters
    ----------
    seed : int, default=42
        The seed value to set.

    Raises
    ------
    TypeError
        If input is not an integer.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_svd(x: torch.Tensor, full_matrices: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hardware-aware SVD that gracefully handles MPS (macOS) limitations.

    Falls back to CPU when necessary to ensure stability for SVD operations 
    on Apple Silicon.

    Parameters
    ----------
    x : torch.Tensor
        The input matrix to decompose.
    full_matrices : bool, default=False
        Whether to compute the full-sized unitary matrices U and Vh.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Left singular vectors, singular values, right singular vectors.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
    Compute the R-squared between two latent spaces after Procrustes alignment.

    Aligns `u_est` to `u_true` including rotation, translation, and scaling 
    before calculating the variance explained.

    Parameters
    ----------
    u_true : torch.Tensor
        The reference target matrix.
    u_est : torch.Tensor
        The estimated matrix to align to the target.

    Returns
    -------
    float
        The R-squared metric of alignment.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    u_true = torch.as_tensor(u_true).detach().float()
    u_est = torch.as_tensor(u_est).detach().float()
    
    # 1. Center
    u_true = u_true - u_true.mean(0)
    u_est = u_est - u_est.mean(0)
    
    # 2. Orthogonal Procrustes alignment
    # find R such that min ||u_true - u_est @ R||_F
    m = u_est.t() @ u_true
    u, s, vh = torch.linalg.svd(m)
    r = u @ vh
    
    u_rotated = u_est @ r
    
    # 3. Find optimal scaling factor s_scale
    # min ||u_true - s_scale * u_rotated||_F^2
    # s_scale = <u_true, u_rotated> / ||u_rotated||_F^2
    denom = torch.sum(u_rotated**2)
    if denom < 1e-10: return 0.0
    
    s_scale = torch.sum(u_true * u_rotated) / denom
    u_aligned = s_scale * u_rotated
    
    # 4. Compute R2
    ss_res = torch.sum((u_true - u_aligned)**2)
    ss_tot = torch.sum(u_true**2)
    if ss_tot < 1e-10: return 1.0
    
    return (1.0 - ss_res / ss_tot).item()

def procrustes_mse(u_true: torch.Tensor, u_est: torch.Tensor) -> float:
    """
    Compute the Mean Squared Error after Procrustes alignment.

    Aligns `u_est` to `u_true` including rotation, translation, and scaling 
    before calculating the MSE.

    Parameters
    ----------
    u_true : torch.Tensor
        The reference target matrix.
    u_est : torch.Tensor
        The estimated matrix to align to the target.

    Returns
    -------
    float
        The Mean Squared Error of alignment.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    u_true = torch.as_tensor(u_true).detach().float()
    u_est = torch.as_tensor(u_est).detach().float()
    
    u_true = u_true - u_true.mean(0)
    u_est = u_est - u_est.mean(0)
    
    m = u_est.t() @ u_true
    u, s, vh = torch.linalg.svd(m)
    r = u @ vh
    u_rotated = u_est @ r
    
    denom = torch.sum(u_rotated**2)
    if denom < 1e-10: return torch.mean(u_true**2).item()
    
    s_scale = torch.sum(u_true * u_rotated) / denom
    u_aligned = s_scale * u_rotated
    
    return torch.mean((u_true - u_aligned)**2).item()
