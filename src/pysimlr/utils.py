from __future__ import annotations

import warnings

import torch


class _LazyPandas:
    """`pandas` on first use: only the save/load and data-frame helpers here
    need it, and importing it eagerly was 0.2 s of every ``import pysimlr``."""

    def __getattr__(self, name):
        import pandas
        return getattr(pandas, name)


pd = _LazyPandas()
import re
import time
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from .defaults import DEFAULT_COLLAPSE_THRESHOLD, DEFAULT_TOL, check_collapse_threshold

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
    >>> [round(v, 6) for v in res["Asym_hippo"].tolist()]
    [0.1, 0.0]

    Raises
    ------
    TypeError
        If inputs are of invalid types.
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
    >>> [round(v, 6) for v in res["LRAVG_hippo"].tolist()]
    [3.1, 3.2]

    Raises
    ------
    TypeError
        If inputs are of invalid types.
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
    >>> _ = torch.manual_seed(0)
    >>> x = torch.randn(100, 10)
    >>> # Perfect similarity with itself (float32, so compare with tolerance)
    >>> abs(rvcoef(x, x) - 1.0) < 1e-6
    True
    >>> # High similarity with a noisy version
    >>> y = x + torch.randn(100, 10) * 0.1
    >>> rvcoef(x, y) > 0.9
    True

    Raises
    ------
    TypeError
        If inputs are of invalid types.
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
    """
    s_xx = x_centered @ x_centered.t()
    s_yy = y_centered @ y_centered.t()
    
    
    numerator = torch.sum(s_xx * s_yy) # Equivalent to trace(S_XY @ S_YX)
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
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)
        
    n = x.shape[0]
    if n <= 1 or y.shape[0] <= 1 or x.shape[0] != y.shape[0]:
        return 0.0
        
    x_centered = x - torch.mean(x, dim=0)
    y_centered = y - torch.mean(y, dim=0)
    if n < (x.shape[1] + y.shape[1]):
        components = rvcoef_trace_impl(x_centered, y_centered)
    else:
        components = rvcoef_gram_impl(x_centered, y_centered)
    rv_obs = float(components['rv'])
    rv_den = components['denominator']
    if isinstance(rv_den, torch.Tensor):
        rv_den = float(rv_den.item())
    else:
        rv_den = float(rv_den)
    
    if rv_den == 0.0:
        return 0.0
    
    tr_s_xx = float(torch.sum(x_centered * x_centered).item())
    tr_s_yy = float(torch.sum(y_centered * y_centered).item())
    
    exp_rv_num = tr_s_xx * tr_s_yy / (n - 1)
    exp_rv = exp_rv_num / rv_den
    
    if exp_rv >= 1.0:
        return float('nan')
        
    adj_rv = (rv_obs - exp_rv) / (1.0 - exp_rv)
    return float(adj_rv)

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
    """
    col_l1_norms = torch.sum(torch.abs(features), dim=0)
    col_l1_norms[col_l1_norms == 0] = 1.0
    return features / col_l1_norms

class ConvergenceWarning(UserWarning):
    """A fit stopped without reaching its plateau (see `warn_if_unconverged`)."""


#: Stop reasons that mean the returned fit has not converged.
UNCONVERGED_STOP_REASONS = frozenset({"max_iter", "no_descent", "non_finite"})


def warn_if_unconverged(convergence: dict, method: str) -> None:
    """Warn when a fit ended without converging.

    The result still carries ``convergence`` / ``stop_reason``; this makes the
    case visible to a caller who does not look. Called once per user-facing fit
    (not for the learning-rate probes, which are short by design).
    """
    reason = (convergence or {}).get("stop_reason")
    if reason not in UNCONVERGED_STOP_REASONS:
        return
    import warnings
    steps, cap = convergence.get("n_steps"), convergence.get("max_steps")
    hint = ("pass a smaller fixed learning_rate (e.g. 0.01) and keep the converged fit with the "
            "lowest best_energy" if reason == "no_descent" else "raise iterations")
    warnings.warn(f"{method}: fit did not converge (stop_reason={reason!r}, {steps}/{cap} steps); "
                  f"results may depend on the budget or step size. To fix: {hint}. "
                  f"See result['convergence'].", ConvergenceWarning, stacklevel=3)


class WideViewWarning(UserWarning):
    """A view has at least as many features as samples, so its training coupling can be fitted to noise
    (see `coupling_capacity_report`)."""


def coupling_capacity_report(latents: List[torch.Tensor], u: torch.Tensor, widths: List[int],
                             n: int) -> Dict[str, Any]:
    r"""Training coupling and whether it can be trusted.

    ``coupling_train[m][j]`` is :math:`|\mathrm{corr}(X_m v_{mj}, u_j)|` on the training rows.
    A view with :math:`p_m \ge n` features can make its scores match the other views' noise:
    with an unpenalised basis any target is reachable exactly, and selecting a sparse support
    from :math:`p_m` candidates keeps most of that freedom. Measured on IBD stool metabolomics
    (four 500-feature platforms plus 156 species, n = 165 training subjects, k = 5), the training
    coupling was 1.00 / 0.99 / 0.92 / 0.83 at sparseness 0.5 / 0.8 / 0.9 / 0.95 while held-out
    coupling stayed at 0.56-0.61. So for such views the training coupling says nothing about
    shared signal; held-out coupling (`pysimlr.predict` on new rows) does.

    Returns ``coupling_train`` (per view, per component), ``view_width_ratio`` (:math:`p_m / n`)
    and ``wide_views`` (indices with :math:`p_m \ge n`).
    """
    uu = u[0] if isinstance(u, list) else u
    uu = uu.detach().double()
    coup = []
    n_per_view = []
    for z in latents:
        zd = z.detach().double()
        fin = torch.isfinite(zd).all(dim=1) & torch.isfinite(uu).all(dim=1)
        n_m = int(fin.sum().item())
        n_per_view.append(n_m if n_m > 0 else int(n))
        if n_m >= 2:
            zf = zd[fin]
            uf = uu[fin]
            zc = zf - zf.mean(0)
            uc = uf - uf.mean(0)
            num = (zc * uc).sum(0)
            den = zc.norm(dim=0) * uc.norm(dim=0)
            coup.append((num.abs() / den.clamp_min(1e-300)).tolist())
        else:
            coup.append([0.0] * int(zd.shape[1]))
    ratio = [float(p) / float(max(nm, 1)) for p, nm in zip(widths, n_per_view)]
    return {"coupling_train": coup, "view_width_ratio": ratio,
            "wide_views": [m for m, (p, nm) in enumerate(zip(widths, n_per_view)) if p >= nm]}


def warn_if_wide_views(report: Dict[str, Any], method: str, widths: List[int], n: int) -> None:
    """Warn when a view is at least as wide as the sample (see `coupling_capacity_report`)."""
    wide = report.get("wide_views") or []
    if not wide:
        return
    import warnings
    desc = ", ".join(f"view {m} (p={widths[m]}, training coupling "
                     f"{np.mean(report['coupling_train'][m]):.2f})" for m in wide)
    warnings.warn(f"{method}: {desc} has at least as many features as samples (n={n}); its training "
                  "coupling can be fitted to noise and is not evidence of shared signal. Check coupling "
                  "on held-out rows with pysimlr.heldout_coupling(result, held_out_views), or reduce the "
                  "features. See result['wide_views'].", WideViewWarning, stacklevel=3)


def _abs_corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|corr(a[:, j], b[:, j])| per column (0 for a constant column)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return np.where(den > 0, np.abs((a * b).sum(0)) / np.where(den > 0, den, 1.0), 0.0)


def _coupling_stats(latents: Dict[int, np.ndarray], u: np.ndarray) -> Tuple[Dict[int, List[float]], List[float]]:
    to_u = {m: _abs_corr_columns(z, u).tolist() for m, z in latents.items()}
    keys = sorted(latents)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    k = u.shape[1]
    cross = (np.mean([_abs_corr_columns(latents[a], latents[b]) for a, b in pairs], axis=0).tolist()
             if pairs else [float("nan")] * k)
    return to_u, cross


def heldout_coupling(result: Dict[str, Any], views: List[Any], *, train_views: Optional[List[Any]] = None) -> Dict[str, Any]:
    r"""Cross-view agreement on held-out rows, beside the same statistics on the training rows.

    Training coupling is not evidence of shared signal when a view is wide (``p_m >= n``; see
    `coupling_capacity_report` and `WideViewWarning`). Held-out coupling is. ``views`` are new
    subjects' views in the fit's order, ``None`` for a view that is missing; at least two must be
    present for the cross-view statistic. Statistics are computed over the available views only, in
    training and held-out alike, so the two are comparable. Scores come from `sem.view_scores` in
    the method's native frame (``rotate=False``); every statistic is an absolute correlation, so
    the sign alignment does not change it.

    Joint-basis fits (``pca``, ``nsa_pipeline``, ``gcca``) score each view through its block of
    ``v`` (``scores_source="joint_block"``). Their training per-view scores need ``train_views``
    (the fit's training views); without them the training statistics are ``None`` and
    ``train_reason`` says why.

    Returns
    -------
    dict with, per available view ``m`` and component ``j``:
      ``coupling_train`` / ``coupling_heldout``: :math:`|\mathrm{corr}(z_{mj}, u_j)|`;
    per component:
      ``cross_view_train`` / ``cross_view_heldout``: mean over view pairs of
      :math:`|\mathrm{corr}(z_{aj}, z_{bj})|`, and ``cross_view_gap`` (training minus held-out);
    and ``views`` (indices used), ``n_train``, ``n_heldout``, ``scores_source``, ``train_reason``.

    On IBD stool metabolomics (four 500-feature platforms plus 156 species, n = 165 training
    subjects) the mean cross-view coupling was 1.00 in training and 0.58 held-out; for independent
    noise views with p >= n it is about 0.88 against 0.03.
    """
    from .sem import view_scores
    n_views = len(result["v"])
    if len(views) != n_views:
        raise ValueError(f"views has {len(views)} entries; the fit has {n_views} (use None for a missing view).")
    avail = [m for m, x in enumerate(views) if x is not None]
    if not avail:
        raise ValueError("no held-out view is present.")
    sc = view_scores(result, list(views), train_views=train_views, standardize=False, rotate=False)
    c_te, x_te = _coupling_stats({m: sc["scores"][m] for m in avail}, sc["consensus"])
    out = {"views": avail, "n_train": int(sc["train_consensus"].shape[0]), "n_heldout": int(sc["n"]),
           "coupling_heldout": [c_te[m] for m in avail], "cross_view_heldout": x_te,
           "scores_source": sc["source"], "train_reason": None}
    if sc["train_scores"] is None:
        out.update(coupling_train=None, cross_view_train=None, cross_view_gap=None,
                   train_reason="joint-basis fit without train_views: its training per-view scores "
                                "cannot be formed from the result alone.")
        return out
    c_tr, x_tr = _coupling_stats({m: sc["train_scores"][m] for m in avail}, sc["train_consensus"])
    out.update(coupling_train=[c_tr[m] for m in avail], cross_view_train=x_tr,
               cross_view_gap=[a - b for a, b in zip(x_tr, x_te)])
    return out


_TRIU_MASK_CACHE: Dict[tuple, torch.Tensor] = {}


def correlation_penalty(z: torch.Tensor, threshold: Optional[float] = None,
                        eps: float = 1e-8) -> torch.Tensor:
    """Dimensional-collapse penalty on the columns of ``z`` (each view's scores).

    Sum over distinct column pairs of ``(max(|r| - t, 0) / (1 - t))**2``, with
    ``r`` their correlation and ``t = threshold``
    (`defaults.DEFAULT_COLLAPSE_THRESHOLD`): 0 for correlations up to ``t``, 1
    for each fully collinear pair, scale-free. Used by the linear and deep
    objectives because a consensus that mixing orthogonalises (`newton`,
    `svd`) cannot show collapse and "nc" scores a collapsed latent as well as
    a full-rank one.

    Summed so one collapsed pair costs the same at any ``k`` (the mean charged
    it 2/(k(k-1)); at k = 5 "nc" still merged two columns). Thresholded so
    moderate correlations the true factors have are not penalised: the plain
    squared-correlation form cut `nonneg_parts` support recovery from 1.000 to
    0.917 (hybrid_adam). t = 0.3 kept every check at its best: support 1.000,
    full rank at k = 5 on 5-feature views, narrow-view outcome R^2 0.947, no NED
    collapse (t = 0.5 left a rank-4 basis at k = 5; powers 4 and 8 failed the
    k = 5 case or the narrow view).
    """
    if threshold is None:
        t = DEFAULT_COLLAPSE_THRESHOLD
    elif (isinstance(threshold, (int, float))
          and not isinstance(threshold, bool)
          and 0.0 <= threshold < 1.0):
        t = float(threshold)
    else:
        t = check_collapse_threshold(threshold)
    n, d = z.shape
    if n < 2 or d < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    zc = z - z.mean(dim=0)
    zs = zc / (zc.norm(dim=0, keepdim=True) + eps)
    mkey = (d, z.device, z.dtype)
    mask = _TRIU_MASK_CACHE.get(mkey)
    if mask is None:
        mask = torch.triu(torch.ones((d, d), device=z.device, dtype=z.dtype), diagonal=1)
        _TRIU_MASK_CACHE[mkey] = mask
    r = (zs.t() @ zs).abs() * mask
    diff = torch.relu(r - t) * (1.0 / (1.0 - t))
    return torch.sum(diff * diff)


def stationarity_residual(v: torch.Tensor, grad: torch.Tensor,
                          nonneg: bool = False) -> Tuple[float, float]:
    r"""First-order stationarity of a sparse basis on its current face.

    Returns ``(on_face, off_support)``:

    ``on_face``
        The scale-invariant gradient mapping
        :math:`\lVert P_S(V - t\nabla E) - V\rVert_F / \lVert V\rVert_F`,
        :math:`t = \lVert V\rVert_F^2`, with :math:`P_S` the *convex*
        projection onto the face the iterate occupies: entries outside its
        support zeroed, and clamped at 0 under non-negativity. It is zero
        exactly where the support-restricted problem is stationary.
    ``off_support``
        Per column, the largest admissible gradient step on a coordinate
        outside the support, :math:`t\,|(\nabla E)_j|` (only increasing moves
        under non-negativity), over the smallest kept :math:`|v|`; the maximum
        over columns. Above 1, one gradient step would admit a coordinate larger
        than the smallest one kept, so the support is not settled.

    Why the face: the residual `simlr` used to report took P to be the full
    projection (quantile sparsity and the NSA-Flow retraction), which is not
    convex, and it did not vanish at stationary points: over 300 sweeps it fell
    only to 0.24-1.0 of its first value. This one fell by >= 2 decades in 7 of
    8 calibration runs whose energy had settled (relative range < 1e-4 over the
    last 50 sweeps; 1.74 decades in the eighth), and stayed high where the
    energy kept moving (scripts/stationarity_residual_calibration.py,
    docs/audit/evidence/stationarity_residual_calibration.csv).
    """
    v = v.detach().double()
    g = grad.detach().double()
    vn = v.norm()
    if float(vn) == 0.0:
        return float("inf"), float("inf")
    t = float(vn) ** 2
    mask = (v.abs() > 1e-12).to(v.dtype)
    z = (v - t * g) * mask
    if nonneg:
        z = z.clamp_min(0.0)
    on_face = float((z - v).norm() / vn)
    step = -t * g
    admissible = step.clamp_min(0.0) if nonneg else step.abs()
    off = 0.0
    for j in range(v.shape[1]):
        keep = mask[:, j] > 0
        if bool(keep.all()) or not bool(keep.any()):
            continue
        off = max(off, float(admissible[~keep, j].max() / v[keep, j].abs().min()))
    return on_face, off


_EYE_OVER_K_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def orthogonality_defect(a: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    r"""
    Trace-normalised orthogonality defect: ``||G - I/k||_F^2``, ``G = A'A/tr(A'A)``.

    Zero exactly on the scaled Stiefel manifold -- mutually orthogonal columns
    of equal norm -- and otherwise positive. Scale invariant, since ``G`` is.

    Parameters
    ----------
    a : torch.Tensor
        Basis matrix of shape (features, components).
    normalized : bool, default=True
        Divide by ``1 - 1/k`` so the result lies in [0, 1] and equals 1 at rank
        one, making it comparable across different numbers of components.

    Returns
    -------
    torch.Tensor
        Scalar defect. Zero for ``k == 1``, where orthogonality is vacuous.

    Notes
    -----
    This replaced ``invariant_orthogonality_defect`` (removed in 0.4.0) as the penalty used in
    SiMLR's energy and the deep models' losses. That function normalises by the
    *global* Frobenius norm and sums only the off-diagonal Gram entries, which
    has two consequences that make it unfit to optimise against:

    - **Rank collapse attains zero.** A matrix with one non-zero column and
      ``k - 1`` zero columns scores exactly 0.0, because a zero column is
      orthogonal to everything. A penalty meant to enforce orthogonality was
      therefore minimised by throwing components away. This measure scores that
      matrix 1.0, and in general obeys ``D >= 1/r - 1/k`` at rank ``r``, so
      collapse is penalised rather than rewarded.
    - **It is not a function of the angles.** Dividing by the global
      ``||A||_F^4`` rather than by ``||a_i||^2 ||a_j||^2`` leaves it invariant to
      a global rescaling but not to rescaling one column, so it can be reduced
      by inflating a single column while every angle stays fixed: on an
      orthonormal basis, scaling one column by 100 took it from 2.9e-15 down to
      6.0e-19.

    `D` admits the spectral reading ``D = 1/EffRank(A) - 1/k`` with
    ``EffRank = (tr S)^2 / ||S||_F^2``, so it is a participation-ratio deficit,
    and it decomposes into a decorrelation term plus a norm-balance term --
    ``D = 0`` requires orthogonality *and* equal column norms.

    Examples
    --------
    >>> import torch
    >>> q, _ = torch.linalg.qr(torch.randn(10, 3, generator=torch.Generator().manual_seed(0)))
    >>> bool(orthogonality_defect(q) < 1e-6)
    True
    >>> collapsed = torch.zeros(10, 3); collapsed[:, 0] = 1.0
    >>> bool(abs(float(orthogonality_defect(collapsed)) - 1.0) < 1e-6)
    True
    """
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a).float()
    k = a.shape[1]
    if k < 2:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    s_mat = a.t() @ a
    t = torch.diagonal(s_mat).sum()
    if t <= 0:
        # An all-zero basis has no defined Gram direction; report the rank-one
        # ceiling rather than dividing by zero.
        ceiling = torch.ones((), device=a.device, dtype=a.dtype)
        return ceiling if normalized else ceiling * (1.0 - 1.0 / k)
    gram = s_mat / t
    eye_key = (k, a.device, a.dtype)
    eye = _EYE_OVER_K_CACHE.get(eye_key)
    if eye is None:
        eye = torch.eye(k, device=a.device, dtype=a.dtype) / k
        _EYE_OVER_K_CACHE[eye_key] = eye
    # Formed as ||G - I/k||^2 directly. The algebraically identical
    # ||G||^2 - 1/k suffers catastrophic cancellation near the optimum and
    # returns small negative values.
    defect = torch.sum((gram - eye) ** 2)
    if normalized:
        defect = defect / (1.0 - 1.0 / k)
    return defect


def gradient_orthogonality_defect(a: torch.Tensor,
                                  normalized: bool = True) -> torch.Tensor:
    r"""
    Closed-form gradient of :func:`orthogonality_defect`.

    ``grad D = (4 / t^2) [A S - (||S||_F^2 / t) A]`` with ``S = A'A`` and
    ``t = tr S``, requiring no factorisation.

    Satisfies the Euler identity ``<grad D, A> = 0`` -- ``D`` is homogeneous of
    degree zero -- so the defect term cannot alter ``||A||_F``.

    Parameters
    ----------
    a : torch.Tensor
        Basis matrix of shape (features, components).
    normalized : bool, default=True
        Must match the flag used for the value.

    Returns
    -------
    torch.Tensor
        Gradient with the same shape as `a`.

    Examples
    --------
    >>> import torch
    >>> a = torch.randn(8, 3, generator=torch.Generator().manual_seed(1), requires_grad=True)
    >>> orthogonality_defect(a).backward()
    >>> manual = gradient_orthogonality_defect(a.detach())
    >>> bool(torch.allclose(a.grad, manual, atol=1e-6))
    True
    """
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a).float()
    k = a.shape[1]
    if k < 2:
        return torch.zeros_like(a)
    s_mat = a.t() @ a
    t = torch.diagonal(s_mat).sum()
    if t <= 0:
        return torch.zeros_like(a)
    grad = (4.0 / t.pow(2)) * (a @ s_mat - (torch.sum(s_mat ** 2) / t) * a)
    if normalized:
        grad = grad / (1.0 - 1.0 / k)
    return grad


def _angle_gram(a: torch.Tensor, k: int, eps: float, diagonal: bool):
    """Column norms, unit columns, their Gram (cosine) matrix and the
    reference it is compared with: I, or the Gram's own diagonal."""
    norms = a.norm(dim=0, keepdim=True).clamp_min(eps)
    unit = a / norms
    gram = unit.t() @ unit
    ref = (torch.eye(k, device=a.device, dtype=a.dtype) if diagonal
           else torch.diag(torch.diagonal(gram)))
    return norms, unit, gram, ref


def angle_defect(a: torch.Tensor, eps: float = 1e-12,
                 diagonal: bool = True) -> torch.Tensor:
    r"""
    Squared-cosine orthogonality defect ``C``, in [0, 1].

    With ``M`` the matrix of pairwise cosines between the columns,

    ``C = ||M - I||_F^2 / (k(k-1))``      (the default, ``diagonal=True``)

    Zero exactly when the columns are mutually orthogonal, *at any column
    norms*, and invariant to rescaling each column independently -- which is
    the gauge freedom of a basis.

    Note the reference is the identity, not the observed diagonal. The
    pairwise-only form

    ``C = 1/(k(k-1)) * sum_{i != j} cos^2(a_i, a_j)``   (``diagonal=False``)

    looks equivalent -- for a matrix with no dead column it is -- but scores a
    rank-collapsed matrix 0.0, because a zero column has zero cosine against
    everything. That is the same too-large zero set that makes
    the removed ``invariant_orthogonality_defect`` unusable as a penalty, so writing
    the sum form as the definition and then implementing the identity form is
    a documentation bug worth avoiding: the two disagree exactly where it
    matters and agree on random inputs.

    Parameters
    ----------
    a : torch.Tensor
        Basis matrix of shape (features, components).
    eps : float, default=1e-12
        Floor on a column norm. A zero column has no direction, so its cosines
        are undefined; flooring the norm avoids a NaN.
    diagonal : bool, default=True
        Subtract the identity rather than the observed diagonal. A column whose
        norm has been floored then contributes ``cos_ii = 0`` against a target
        of 1 and is charged ``1/(k(k-1))``, which doubles as a dead-column
        penalty. Without it a rank-collapsed matrix scores 0, since a zero
        column has zero cosine against everything -- the same failure that
        made the removed ``invariant_orthogonality_defect`` unusable as a penalty.
        Pass False only where a dead column is a legitimate outcome and
        something else keeps the basis non-degenerate.

    Returns
    -------
    torch.Tensor
        Scalar defect. Zero for ``k == 1``, where orthogonality is vacuous.

    Notes
    -----
    The companion to :func:`orthogonality_defect` (``D``), and the choice
    between them is about whether the components should carry equal weight.

    ``D`` decomposes into a decorrelation term plus a norm-balance term, so its
    zero set is orthonormality. Use it where equal column norms are part of the
    contract -- a Stiefel constraint, say.

    ``C`` charges only the angles. Use it where the column scale is absorbed
    downstream and so is not meaningful: a deep encoder whose decoder can
    reweight components freely is charged by ``D`` for an imbalance that costs
    the model nothing, which dilutes the decorrelation pressure the penalty is
    there to apply.

    ``C`` still penalises collapse, though less sharply than ``D``: on a matrix
    with one non-zero column and ``k - 1`` zero columns, ``C`` scores 1/3 at
    ``k = 3`` where ``D`` scores 1. Both are far from the 0.0 that
    the removed ``invariant_orthogonality_defect`` assigned it -- but only because
    `diagonal` is set; see that parameter.

    Examples
    --------
    >>> import torch
    >>> q, _ = torch.linalg.qr(torch.randn(10, 3, generator=torch.Generator().manual_seed(0)))
    >>> bool(angle_defect(q) < 1e-6)
    True
    >>> rescaled = q.clone(); rescaled[:, 0] *= 100.0
    >>> bool(abs(float(angle_defect(rescaled)) - float(angle_defect(q))) < 1e-6)
    True
    """
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a).float()
    k = a.shape[1]
    if k < 2:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    norms, unit, gram, ref = _angle_gram(a, k, eps, diagonal)
    return torch.sum((gram - ref) ** 2) / (k * (k - 1))


def gradient_angle_defect(a: torch.Tensor, eps: float = 1e-12,
                          diagonal: bool = True) -> torch.Tensor:
    r"""
    Closed-form gradient of :func:`angle_defect`.

    With ``U`` the column-normalised ``a``, ``H`` its Gram matrix with the
    diagonal removed and ``n`` the column norms,

    ``grad C = 4/(k(k-1)) * [U H - U diag(rowsum(H^2))] / n``.

    Tangential to each column's sphere, so ``C`` cannot alter any individual
    column norm -- the counterpart of the Euler identity that ``D`` satisfies
    for the global norm.

    Examples
    --------
    >>> import torch
    >>> a = torch.randn(9, 3, generator=torch.Generator().manual_seed(2), requires_grad=True)
    >>> angle_defect(a).backward()
    >>> bool(torch.allclose(a.grad, gradient_angle_defect(a.detach()), atol=1e-6))
    True
    """
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a).float()
    k = a.shape[1]
    if k < 2:
        return torch.zeros_like(a)
    norms, unit, gram, ref = _angle_gram(a, k, eps, diagonal)
    # For a column whose norm exceeds `eps` the diagonal residual is exactly
    # zero, so this agrees with the off-diagonal-only gradient there; the
    # difference is confined to floored columns, where the unit vector and
    # hence the gradient are zero anyway.
    off = gram - ref
    row_energy = torch.sum(off ** 2, dim=1)
    grad = (unit @ off - unit * row_energy.unsqueeze(0)) / norms
    return grad * (4.0 / (k * (k - 1)))


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
    """
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a).float()
    k = a.shape[1]
    identity = torch.eye(k, device=a.device, dtype=a.dtype)
    return torch.norm(a.t() @ a - identity, p='fro')

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
        - `invariant_defect`: the trace-normalised defect ``D`` from
          :func:`orthogonality_defect`, zero only on orthonormality. The key
          keeps its historical name; the quantity behind it changed when the
          old globally-normalised defect was found to be minimised by rank
          collapse.
        - `angle_defect`: the mean squared cosine ``C`` from
          :func:`angle_defect`, zero on orthogonality at any column norms.
          Reported beside ``D`` because ``D`` conflates two things: read
          together, a low ``C`` with a high ``D`` says the columns are
          decorrelated but unequal in norm, which for a basis feeding a layer
          that can rescale them is a gauge rather than a defect.
        - `stiefel_defect`: Violation of Stiefel manifold constraint.
        - `mean_defect`: Average off-diagonal squared correlation.
        - `condition_number`: Ratio of maximum to minimum singular value.
        - `effective_rank_proxy`: Proxy measure of effective rank.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a).float()
    k = a.shape[1]
    eye_k = torch.eye(k, device=a.device, dtype=a.dtype)
    s_mat = a.t() @ a
    stiefel = torch.norm(s_mat - eye_k, p='fro').item()
    if k < 2:
        defect = 0.0
        angle = 0.0
        mean_defect = 0.0
    else:
        t = torch.diagonal(s_mat).sum()
        if t <= 0:
            defect = 1.0
        else:
            gram_d = s_mat / t
            defect = (torch.sum((gram_d - eye_k * (1.0 / k)) ** 2) / (1.0 - 1.0 / k)).item()
        col_norms = torch.norm(a, dim=0, keepdim=True)
        unit = a / col_norms.clamp_min(1e-12)
        gram_u = unit.t() @ unit
        angle = (torch.sum((gram_u - eye_k) ** 2) / (k * (k - 1))).item()
        if bool((col_norms > 1e-12).all()):
            scale = col_norms / (col_norms + 1e-10)
            corr = gram_u * (scale.t() @ scale)
            off_sq = torch.sum((corr * (1.0 - eye_k)) ** 2)
            mean_defect = (off_sq / (k * (k - 1))).item()
        else:
            mean_defect = mean_orthogonality_defect(a).item()

    # Condition number (ratio of max to min singular value)
    try:
        s = torch.linalg.svdvals(a)
        cond = (s[0] / (s[-1] + 1e-10)).item()
        ortho_score = torch.sum(s * s).item() / (s[0] * s[0]).item() # Effective rank proxy
    except torch.linalg.LinAlgError:
        # A diagnostic: an unconverged SVD (non-finite input) is reported as
        # NaN rather than failing the fit that asked for the summary.
        cond = float('nan')
        ortho_score = float('nan')
        
    return {
        "invariant_defect": defect,
        "angle_defect": angle,
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
    """
    x_out = (
        x.clone(memory_format=torch.contiguous_format)
        if x.dtype == torch.float32
        else x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
    )
    replay = provenance is not None
    new_provenance = None if replay else {}

    def _need(key):
        """Fetch a replay statistic, refusing to silently recompute it.

        A missing key used to fall through to the `else` branch and recompute
        the statistic from whatever matrix was passed in -- i.e. silently refit
        the transform on the held-out data, with no error and no warning. An
        empty dict slipped through every guard because ``if provenance`` is
        False for ``{}``.
        """
        if key not in provenance:
            raise KeyError(
                f"preprocess_data was given a provenance dict that does not "
                f"contain '{key}', which is required to replay scale_list="
                f"{list(scale_list)}. Refusing to recompute it from the "
                f"supplied matrix, because that would refit the transform on "
                f"this data instead of reusing the training statistics. "
                f"Provenance keys present: {sorted(provenance)}."
            )
        return provenance[key]

    # A confound sink fitted by `pysimlr.fit(..., confounds=...)` is part of
    # the view's preprocessing: it is replayed first, on the raw features,
    # exactly as it was applied to the training rows (`pysimlr.confounds`).
    if replay and "confound_sink" in provenance:
        from .confounds import replay_sink
        x_out = replay_sink(x_out, provenance)

    # Handle NaNs using provenance or current data
    if replay:
        nan_fill = _need("nan_fill")
    else:
        nan_fill = torch.nanmean(x_out).item() if not torch.isnan(x_out).all() else 0.0
        new_provenance["nan_fill"] = nan_fill
        
    x_out = torch.nan_to_num(x_out, nan=nan_fill)
    
    # First position at which each method appears. Applying the same method
    # twice (e.g. ["center", "center"]) previously wrote both steps' statistics
    # to a single key, so the second overwrote the first and replay then used
    # the wrong statistics for both steps. Only repeat occurrences get a
    # position suffix, so provenance dicts written by earlier versions -- which
    # used bare keys -- still replay unchanged.
    first_step = {}
    for _idx, _m in enumerate(scale_list):
        first_step.setdefault(_m, _idx)

    def _prov_key(base, step, method):
        return base if step == first_step[method] else f"{base}@{step}"

    for step, method in enumerate(scale_list):
        if method == "none":
            continue
        elif method == "norm":
            key = _prov_key("norm_factor", step, method)
            if replay:
                factor = _need(key)
            else:
                factor = torch.norm(x_out, p='fro').item() + 1e-10
                new_provenance[key] = factor
            x_out = x_out / factor
        elif method == "np":
            # np is dimension-dependent, but we store the factor for safety
            key = _prov_key("np_factor", step, method)
            if replay:
                factor = _need(key)
            else:
                factor = float(x_out.shape[0] * x_out.shape[1])
                new_provenance[key] = factor
            x_out = x_out / factor
        elif method == "sqrtnp":
            key = _prov_key("sqrtnp_factor", step, method)
            if replay:
                factor = _need(key)
            else:
                factor = np.sqrt(x_out.shape[0] * x_out.shape[1])
                new_provenance[key] = factor
            x_out = x_out / factor
        elif method == "center":
            key = _prov_key("center_mean", step, method)
            if replay:
                mean = _need(key)
            else:
                mean = torch.mean(x_out, dim=0)
                new_provenance[key] = mean
            x_out = x_out - mean
        elif method == "centerAndScale":
            mkey, skey = _prov_key("cas_mean", step, method), _prov_key("cas_std", step, method)
            if replay:
                mean = _need(mkey)
                std = _need(skey)
            else:
                mean = torch.mean(x_out, dim=0)
                if x_out.shape[0] > 1:
                    std = torch.std(x_out, dim=0)
                    std = torch.where(torch.isnan(std) | (std < 1e-10), torch.ones_like(std), std)
                else:
                    std = torch.ones_like(mean)
                new_provenance[mkey] = mean
                new_provenance[skey] = std
            x_out = (x_out - mean) / std
        elif method == "eigenvalue":
            key = _prov_key("eigen_factor", step, method)
            if replay:
                factor = _need(key)
            else:
                _, s, _ = torch.linalg.svd(x_out, full_matrices=False)
                factor = torch.sum(s).item() + 1e-10
                new_provenance[key] = factor
            x_out = x_out / factor
            

    if replay:
        return x_out
    return x_out, new_provenance

def invert_preprocessing(x: torch.Tensor, scale_list: List[str],
                         provenance: Dict[str, Any]) -> torch.Tensor:
    """Map data from the space `preprocess_data` produced back to input units.

    Applies the inverse of each step in ``scale_list``, last step first, with
    the statistics recorded in ``provenance``. NaN filling is not inverted
    (the original positions of missing values are not recorded).

    Raises
    ------
    KeyError
        If ``provenance`` lacks a statistic a step needs.
    ValueError
        For a step with no stored inverse.
    """
    first_step = {}
    for idx, m in enumerate(scale_list):
        first_step.setdefault(m, idx)

    def key(base, step, method):
        return base if step == first_step[method] else f"{base}@{step}"

    out = x
    for step in range(len(scale_list) - 1, -1, -1):
        method = scale_list[step]
        if method == "none":
            continue
        if method in ("norm", "np", "sqrtnp", "eigenvalue"):
            base = {"norm": "norm_factor", "np": "np_factor", "sqrtnp": "sqrtnp_factor",
                    "eigenvalue": "eigen_factor"}[method]
            out = out * float(provenance[key(base, step, method)])
        elif method == "center":
            out = out + torch.as_tensor(provenance[key("center_mean", step, method)], dtype=out.dtype)
        elif method == "centerAndScale":
            mean = torch.as_tensor(provenance[key("cas_mean", step, method)], dtype=out.dtype)
            std = torch.as_tensor(provenance[key("cas_std", step, method)], dtype=out.dtype)
            out = out * std + mean
        else:
            raise ValueError(f"no inverse for preprocessing step {method!r}")
    return out


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
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _scipy_gesvd_fallback(x: torch.Tensor, full_matrices: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute an SVD with LAPACK's ``gesvd`` (QR-based) driver via SciPy.

    ``torch.linalg.svd`` uses ``gesdd`` (divide-and-conquer) on CPU, which is
    faster but can fail to converge on real, ill-conditioned data -- measured
    on the ADNI+PPMI cohort (`scripts/adni_loader.py`): after
    `_standardize_deep`'s centering/scaling, `torch.linalg.svd` on the
    T1Hier view raises ``LinAlgError: ... failed to converge`` reliably (100%
    of runs, both DX and CDRSB tasks), while the identical matrix decomposes
    fine under `gesvd`. `gesdd` has no CPU driver switch in PyTorch itself
    (the ``driver=`` kwarg to `torch.linalg.svd` is CUDA-only), so the
    fallback goes through SciPy, which exposes `gesvd` on CPU. Slower, but
    only reached when the fast path has already failed.
    """
    import scipy.linalg
    arr = x.detach().cpu().numpy()
    u, s, vh = scipy.linalg.svd(arr, full_matrices=full_matrices, lapack_driver='gesvd')
    to = dict(dtype=x.dtype, device=x.device)
    return (torch.from_numpy(u).to(**to), torch.from_numpy(s).to(**to), torch.from_numpy(vh).to(**to))


def _svd_on_cpu(x: torch.Tensor, full_matrices: bool):
    """SVD of `x` computed on the CPU (LAPACK gesdd, then scipy gesvd if that
    does not converge), returned on `x`'s device."""
    x_cpu = x.cpu().contiguous()
    try:
        u, s, vh = torch.linalg.svd(x_cpu, full_matrices=full_matrices)
    except torch.linalg.LinAlgError:
        u, s, vh = _scipy_gesvd_fallback(x_cpu, full_matrices)
    return u.to(x.device), s.to(x.device), vh.to(x.device)


def safe_svd(x: torch.Tensor, full_matrices: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hardware-aware, convergence-robust SVD.

    Falls back to CPU when necessary for MPS (Apple Silicon) stability, and
    falls back to LAPACK's `gesvd` driver (via SciPy, see
    `_scipy_gesvd_fallback`) when PyTorch's default `gesdd` driver raises
    `LinAlgError` on ill-conditioned real data -- measured on ADNI+PPMI,
    where this is not rare (every `gcca`/`nndsvd` initialization attempt
    failed before this fallback existed).

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
    """
    if x.device.type == 'mps':
        # Current MPS backend lacks stable linalg_svd support for all shapes/precisions
        return _svd_on_cpu(x, full_matrices)
    try:
        return torch.linalg.svd(x, full_matrices=full_matrices)
    except torch.linalg.LinAlgError:
        return _scipy_gesvd_fallback(x, full_matrices)
    except RuntimeError:
        # Fallback for unexpected backend failures
        return _svd_on_cpu(x, full_matrices)

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

_THREE_EYE_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def newton_schulz_orthogonalize(x: torch.Tensor, iterations: int = 5, tol: float = 1e-7) -> torch.Tensor:
    """
    Project a matrix onto the Stiefel manifold using Newton-Schulz iteration.

    This is an iterative method to compute the polar retraction without explicit
    SVD, defined by the update: Y_{k+1} = 0.5 * Y_k * (3I - Y_k^T Y_k).

    Warnings
    --------
    Convergence is only quadratic near singular values of 1, and the input is
    pre-scaled by its **Frobenius** norm rather than its spectral norm -- a
    bound that is loose by up to sqrt(k) and so pushes every singular value
    well below 1. With the default 5 iterations the result can remain
    percent-level away from orthonormal (a 100x10 Gaussian comes back about 2%
    off), and the `tol` early exit will not trigger. Raise `iterations` to
    roughly 20-30, or use an explicit SVD polar projection, when you need
    orthonormality to tight tolerance.

    Parameters
    ----------
    x : torch.Tensor
        The input matrix to orthogonalize.
    iterations : int, default=5
        Maximum number of iterations.
    tol : float, default=1e-7
        Convergence tolerance based on the change between iterations.

    Returns
    -------
    torch.Tensor
        The orthogonalized matrix.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x).float()
    
    y = x
    # Ensure convergence by scaling x so that spectral norm is within stable range.
    # We use Frobenius norm as a safe, upper-bound proxy for the spectral norm.
    norm_f = torch.norm(y, p='fro')
    norm_f_val = float(norm_f.detach())
    if norm_f_val == 0.0:
        return y.clone()
    scaled_unit = norm_f_val > 1.0
    if scaled_unit:
        y = y / (norm_f + 1e-10)
    else:
        y = y.clone()
    
    k = y.shape[1]
    eye_key = (k, y.device, y.dtype)
    three_eye = _THREE_EYE_CACHE.get(eye_key)
    if three_eye is None:
        three_eye = 3.0 * torch.eye(k, device=y.device, dtype=y.dtype)
        _THREE_EYE_CACHE[eye_key] = three_eye
    half_three_eye = 0.5 * three_eye
    min_check_step = 4 if (scaled_unit and k >= 2 and tol <= 1e-4) else 0
    needs_grad = y.requires_grad
    
    for step_idx in range(iterations):
        y_prev = y
        # Newton-Schulz update: Y = 0.5 * Y * (3I - Y^T Y)
        if needs_grad:
            y = y @ (0.5 * (three_eye - (y.t() @ y)))
        else:
            y = y @ torch.addmm(half_three_eye, y.t(), y, beta=1.0, alpha=-0.5)
        
        if tol > 0.0 and step_idx >= min_check_step and step_idx + 1 < iterations:
            with torch.no_grad():
                if torch.norm(y.detach() - y_prev.detach()) < tol:
                    break
            
    return y


def _recon_metrics(
    xs: Sequence[torch.Tensor],
    recons: Sequence[torch.Tensor],
    *,
    include_r2_mse: bool = False,
):
    """Compute Frobenius relative errors (and optionally ``(r2, mse)``) in a single pass."""
    import math
    errors: List[float] = []
    r2_list: List[float] = []
    mse_list: List[float] = []
    for x, r in zip(xs, recons):
        xc = x.detach().cpu() if x.device.type != "cpu" else x.detach()
        rc = (
            (r.detach().cpu() if r.device.type != "cpu" else r.detach())
            if isinstance(r, torch.Tensor)
            else torch.as_tensor(r)
        )
        if xc.ndim == 2 and rc.ndim == 2 and not bool(torch.isfinite(xc).all() and torch.isfinite(rc).all()):
            fin = torch.isfinite(xc).all(dim=1) & torch.isfinite(rc).all(dim=1)
            xc = xc[fin]
            rc = rc[fin]
        if xc.numel() == 0:
            errors.append(float("nan"))
            if include_r2_mse:
                r2_list.append(float("nan"))
                mse_list.append(float("nan"))
            continue
        diff = xc - rc
        if include_r2_mse:
            df64 = diff.reshape(-1).double()
            xf64 = xc.reshape(-1).double()
            ss_res = float(torch.dot(df64, df64).item())
            ss_tot = float(torch.dot(xf64, xf64).item())
            errors.append(math.sqrt(ss_res) / (math.sqrt(ss_tot) + 1e-10))
            r2_list.append(1.0 - ss_res / (ss_tot + 1e-12))
            mse_list.append(ss_res / max(1, xc.numel()))
        else:
            df = diff.reshape(-1)
            xf = xc.reshape(-1)
            errors.append(
                math.sqrt(float(torch.dot(df, df).item()))
                / (math.sqrt(float(torch.dot(xf, xf).item())) + 1e-10)
            )
    if include_r2_mse:
        return errors, (r2_list, mse_list)
    return errors

def write_simlr(simlr_object: Dict[str, Any], file_prefix: str, clear_dir: bool = False, use_r: bool = True):
    """
    Python implementation of ANTsR::write_simlr.
    Writes a SiMLR results dictionary to a directory containing CSV and JSON/RDS files.
    
    Parameters
    ----------
    simlr_object : Dict[str, Any]
        The SiMLR results dictionary/object to save.
    file_prefix : str
        The prefix/base name of the directory where results are saved.
    clear_dir : bool, default=False
        Whether to clear the destination directory if it already exists.
    use_r : bool, default=True
        Whether to run Rscript to generate R-compatible .rds files (manifest.rds, etc.)
        so that ANTsR::read_simlr can load them directly.
    """
    import os
    import shutil
    import json
    import subprocess
    import pandas as pd
    import numpy as np
    import torch
    
    if not isinstance(simlr_object, dict):
        raise TypeError("simlr_object must be a dictionary")
        
    outdir = f"{file_prefix}_simlr"
    if clear_dir and os.path.exists(outdir):
        shutil.rmtree(outdir)
        
    os.makedirs(outdir, exist_ok=True)
    
    def _json_default(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        return str(x)

    manifest = {}
    
    # Iterate and write contents
    for n, obj in simlr_object.items():
        if obj is None:
            manifest[n] = {"type": "null", "file": None}
            continue
            
        # Treat energyPath as dataframe if possible
        if n == "energyPath" and not isinstance(obj, pd.DataFrame):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                if torch.is_tensor(obj):
                    obj = obj.detach().cpu().numpy()
                obj = pd.DataFrame(obj)
                
        if isinstance(obj, pd.DataFrame):
            fname = f"{n}.csv"
            abs_fname = os.path.join(outdir, fname)
            obj.to_csv(abs_fname, index=True)
            manifest[n] = {
                "type": "data_frame",
                "file": fname,
                "has_rownames": True
            }
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            if torch.is_tensor(obj):
                obj = obj.detach().cpu().numpy()
            fname = f"{n}.csv"
            abs_fname = os.path.join(outdir, fname)
            pd.DataFrame(obj).to_csv(abs_fname, index=True)
            manifest[n] = {
                "type": "matrix",
                "file": fname,
                "has_rownames": True,
                "storage.mode": "double"
            }
        elif isinstance(obj, dict):
            subdir = os.path.join(outdir, n)
            os.makedirs(subdir, exist_ok=True)
            submanifest = {}
            for subk, subobj in obj.items():
                if subobj is None:
                    submanifest[subk] = {"file": None, "type": "null", "use_name": True}
                    continue
                    
                subfile_name = f"{subk}.csv"
                if isinstance(subobj, pd.DataFrame):
                    subfile = os.path.join(subdir, subfile_name)
                    subobj.to_csv(subfile, index=True)
                    submanifest[subk] = {
                        "file": f"{n}/{subfile_name}",
                        "type": "data_frame",
                        "use_name": True,
                        "has_rownames": True
                    }
                elif isinstance(subobj, (np.ndarray, torch.Tensor)):
                    if torch.is_tensor(subobj):
                        subobj = subobj.detach().cpu().numpy()
                    subfile = os.path.join(subdir, subfile_name)
                    pd.DataFrame(subobj).to_csv(subfile, index=True)
                    submanifest[subk] = {
                        "file": f"{n}/{subfile_name}",
                        "type": "matrix",
                        "use_name": True,
                        "has_rownames": True,
                        "storage.mode": "double"
                    }
                else:
                    json_file_name = f"{subk}.json"
                    subfile = os.path.join(subdir, json_file_name)
                    with open(subfile, "w") as f:
                        json.dump(subobj, f, default=_json_default)
                    submanifest[subk] = {
                        "file": f"{n}/{json_file_name}",
                        "type": "rds",
                        "use_name": True
                    }
            manifest[n] = {"type": "list", "contents": submanifest}
        elif isinstance(obj, list):
            subdir = os.path.join(outdir, n)
            os.makedirs(subdir, exist_ok=True)
            submanifest = {}
            for idx, subobj in enumerate(obj):
                subname = str(idx + 1) # 1-based index for R compatibility
                if subobj is None:
                    submanifest[subname] = {"file": None, "type": "null", "use_name": False}
                    continue
                    
                subfile_name = f"{subname}.csv"
                if isinstance(subobj, pd.DataFrame):
                    subfile = os.path.join(subdir, subfile_name)
                    subobj.to_csv(subfile, index=True)
                    submanifest[subname] = {
                        "file": f"{n}/{subfile_name}",
                        "type": "data_frame",
                        "use_name": False,
                        "has_rownames": True
                    }
                elif isinstance(subobj, (np.ndarray, torch.Tensor)):
                    if torch.is_tensor(subobj):
                        subobj = subobj.detach().cpu().numpy()
                    subfile = os.path.join(subdir, subfile_name)
                    pd.DataFrame(subobj).to_csv(subfile, index=True)
                    submanifest[subname] = {
                        "file": f"{n}/{subfile_name}",
                        "type": "matrix",
                        "use_name": False,
                        "has_rownames": True,
                        "storage.mode": "double"
                    }
                else:
                    json_file_name = f"{subname}.json"
                    subfile = os.path.join(subdir, json_file_name)
                    with open(subfile, "w") as f:
                        json.dump(subobj, f, default=_json_default)
                    submanifest[subname] = {
                        "file": f"{n}/{json_file_name}",
                        "type": "rds",
                        "use_name": False
                    }
            manifest[n] = {"type": "list", "contents": submanifest}
        else:
            fname = f"{n}.json"
            abs_fname = os.path.join(outdir, fname)
            with open(abs_fname, "w") as f:
                json.dump(obj, f, default=_json_default)
            manifest[n] = {
                "type": "rds",
                "file": fname
            }
            
    # Write Python manifest
    manifest_json_path = os.path.join(outdir, "manifest.json")
    with open(manifest_json_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
        
    # If use_r is True, try to compile to RDS using R
    if use_r:
        try:
            # Check if Rscript is available
            subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
            
            outdir_clean = os.path.abspath(outdir).replace('\\', '/')
            # Construct R compilation script using raw f-string
            r_script = fr"""
library(jsonlite)
outdir <- "{outdir_clean}"
# Load manifest.json
manifest <- fromJSON(file.path(outdir, "manifest.json"), simplifyVector=FALSE)

process_entry <- function(entry, name, parent_dir) {{
  if (entry$type == "rds") {{
    json_file <- file.path(parent_dir, entry$file)
    # Change file extension to .rds
    entry$file <- gsub("\\\\.json$", ".rds", entry$file)
    rds_file <- file.path(parent_dir, entry$file)
    if (file.exists(json_file)) {{
      val <- fromJSON(json_file, simplifyVector=TRUE)
      saveRDS(val, rds_file)
      file.remove(json_file)
    }}
  }} else if (entry$type == "list") {{
    for (subname in names(entry$contents)) {{
      sub_entry <- entry$contents[[subname]]
      if (sub_entry$type == "rds") {{
        json_file <- file.path(parent_dir, sub_entry$file)
        sub_entry$file <- gsub("\\\\.json$", ".rds", sub_entry$file)
        rds_file <- file.path(parent_dir, sub_entry$file)
        if (file.exists(json_file)) {{
          val <- fromJSON(json_file, simplifyVector=TRUE)
          saveRDS(val, rds_file)
          file.remove(json_file)
        }}
      }}
      entry$contents[[subname]] <- sub_entry
    }}
  }}
  return(entry)
}}

for (n in names(manifest)) {{
  manifest[[n]] <- process_entry(manifest[[n]], n, outdir)
}}

# Save manifest.rds
saveRDS(manifest, file.path(outdir, "manifest.rds"))
"""
            # run the script
            subprocess.run(["Rscript", "--vanilla", "-e", r_script], capture_output=True, text=True, check=True)
        except FileNotFoundError:
            warnings.warn("write_simlr(use_r=True): Rscript not found; the result was written "
                          "as JSON only (no .rds files).", RuntimeWarning, stacklevel=2)
        except subprocess.CalledProcessError as e:
            warnings.warn(f"write_simlr(use_r=True): the R conversion failed; the result was "
                          f"written as JSON only. Rscript stderr: {(e.stderr or '').strip()[:500]}",
                          RuntimeWarning, stacklevel=2)

def read_simlr(dir_path: str, use_r: bool = True) -> Dict[str, Any]:
    """
    Python implementation of ANTsR::read_simlr.
    Reads a saved SiMLR directory back into a Python dictionary.
    
    Parameters
    ----------
    dir_path : str
        The path to the SiMLR directory (e.g. "prefix_simlr").
    use_r : bool, default=True
        Whether to attempt loading RDS files via R if only manifest.rds or .rds files exist.
        
    Returns
    -------
    Dict[str, Any]
        The restored SiMLR results dictionary.
    """
    import os
    import json
    import subprocess
    import pandas as pd
    
    manifest_json = os.path.join(dir_path, "manifest.json")
    manifest_rds = os.path.join(dir_path, "manifest.rds")
    
    manifest = None
    has_r = False
    
    if use_r:
        try:
            subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
            has_r = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            has_r = False
            
    if os.path.exists(manifest_rds) and use_r and has_r:
        # Load manifest.rds using R and jsonlite
        dir_path_clean = os.path.abspath(dir_path).replace('\\', '/')
        cmd = ["Rscript", "--vanilla", "-e", f"cat(jsonlite::toJSON(readRDS(file.path('{dir_path_clean}', 'manifest.rds')), auto_unbox=TRUE))"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout_str = res.stdout
        json_start = stdout_str.find("{")
        if json_start == -1:
            raise RuntimeError(f"Could not parse manifest.rds JSON output: {stdout_str}")
        manifest = json.loads(stdout_str[json_start:])
    elif os.path.exists(manifest_json):
        with open(manifest_json, "r") as f:
            manifest = json.load(f)
    elif os.path.exists(manifest_rds):
        # manifest.rds exists but R is not found or use_r is False
        raise RuntimeError("Rscript is required to read manifest.rds, but R was not found or use_r=False.")
    else:
        raise FileNotFoundError(f"Neither manifest.json nor manifest.rds found in {dir_path}")
        
    result = {}
    
    # Helper to load RDS file via R
    def load_rds(file_path):
        if not has_r:
            raise RuntimeError(f"Rscript is required to load RDS file: {file_path}")
        file_path_clean = os.path.abspath(file_path).replace('\\', '/')
        cmd = ["Rscript", "--vanilla", "-e", f"cat(jsonlite::toJSON(readRDS('{file_path_clean}'), auto_unbox=TRUE))"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout_str = res.stdout
        json_start = stdout_str.find("[")
        if json_start == -1:
            json_start = stdout_str.find("{")
        if json_start == -1:
            json_start = stdout_str.find('"')
        if json_start == -1:
            return stdout_str.strip()
        return json.loads(stdout_str[json_start:])
        
    for n, entry in manifest.items():
        if entry["type"] == "null":
            result[n] = None
            continue
            
        file_name = entry.get("file")
        abs_file = os.path.join(dir_path, file_name) if file_name else None
        
        if entry["type"] == "data_frame":
            if entry.get("has_rownames", False):
                result[n] = pd.read_csv(abs_file, index_col=0)
            else:
                result[n] = pd.read_csv(abs_file)
        elif entry["type"] == "matrix":
            if entry.get("has_rownames", False):
                df = pd.read_csv(abs_file, index_col=0)
            else:
                df = pd.read_csv(abs_file)
            result[n] = df.values
        elif entry["type"] == "rds":
            if file_name.endswith(".rds"):
                result[n] = load_rds(abs_file)
            else:
                with open(abs_file, "r") as f:
                    result[n] = json.load(f)
        elif entry["type"] == "list":
            subcontents = entry.get("contents", {})
            is_seq_list = True
            for k in subcontents.keys():
                if not k.isdigit():
                    is_seq_list = False
                    break
                    
            if is_seq_list:
                sublist = []
                sorted_keys = sorted(subcontents.keys(), key=int)
                for k in sorted_keys:
                    subentry = subcontents[k]
                    if subentry["type"] == "null":
                        sublist.append(None)
                        continue
                    sub_file = os.path.join(dir_path, subentry["file"])
                    if subentry["type"] == "data_frame":
                        if subentry.get("has_rownames", False):
                            sublist.append(pd.read_csv(sub_file, index_col=0))
                        else:
                            sublist.append(pd.read_csv(sub_file))
                    elif subentry["type"] == "matrix":
                        if subentry.get("has_rownames", False):
                            df = pd.read_csv(sub_file, index_col=0)
                        else:
                            df = pd.read_csv(sub_file)
                        sublist.append(df.values)
                    elif subentry["type"] == "rds":
                        if subentry["file"].endswith(".rds"):
                            sublist.append(load_rds(sub_file))
                        else:
                            with open(sub_file, "r") as f:
                                sublist.append(json.load(f))
                result[n] = sublist
            else:
                subdict = {}
                for k, subentry in subcontents.items():
                    if subentry["type"] == "null":
                        subdict[k] = None
                        continue
                    sub_file = os.path.join(dir_path, subentry["file"])
                    if subentry["type"] == "data_frame":
                        if subentry.get("has_rownames", False):
                            subdict[k] = pd.read_csv(sub_file, index_col=0)
                        else:
                            subdict[k] = pd.read_csv(sub_file)
                    elif subentry["type"] == "matrix":
                        if subentry.get("has_rownames", False):
                            df = pd.read_csv(sub_file, index_col=0)
                        else:
                            df = pd.read_csv(sub_file)
                        subdict[k] = df.values
                    elif subentry["type"] == "rds":
                        if subentry["file"].endswith(".rds"):
                            subdict[k] = load_rds(sub_file)
                        else:
                            with open(sub_file, "r") as f:
                                subdict[k] = json.load(f)
                result[n] = subdict
                
    return result


def basis_rank_report(v_mats: List[torch.Tensor], k: int,
                      warn: bool = True) -> Dict[str, Any]:
    r"""How many components the returned basis actually carries.

    A basis can come back with fewer usable columns than requested and nothing
    else in the result says so. The gradient step drives columns together and
    the projection holds them apart, so when the constraint is too weak the
    columns become collinear: measured on BRCA4 at ``k=3``, the incoming
    column overlap climbs 0.54, 0.79, 0.97, 1.00 over successive sweeps at
    ``nsa_w=0.1`` and stays at 1.00, leaving an effective rank of 2.27 with a
    condition number near 2.8e2. Test accuracy over the same sweep was flat
    (0.759 to 0.799), so prediction does not reveal the collapse -- only the
    rank does.

    Returns, per view:

    ``effective_rank``
        The participation ratio :math:`(\sum_i \sigma_i)^2 / \sum_i \sigma_i^2`
        -- a continuous count of contributing directions, equal to ``k`` for
        equinormed orthogonal columns and falling smoothly as they merge.
        Reported because a hard rank hides partial collapse.
    ``numerical_rank``
        `torch.linalg.matrix_rank`, the integer count.
    ``condition_number``
        :math:`\sigma_{\max}/\sigma_{\min}`.
    ``max_column_overlap``
        Largest off-diagonal cosine between columns; 1.0 means two columns
        coincide.
    """
    eff, num, cond, ovl = [], [], [], []
    for v in v_mats:
        vd = v.detach().double()
        sv = torch.linalg.svdvals(vd)
        tot = float((sv ** 2).sum())
        eff.append(float((sv.sum() ** 2) / tot) if tot > 0 else 0.0)
        num.append(int(torch.linalg.matrix_rank(vd)))
        smin = float(sv.min())
        cond.append(float(sv.max()) / smin if smin > 0 else float("inf"))
        n = vd / vd.norm(dim=0, keepdim=True).clamp_min(1e-12)
        g = n.t() @ n
        g.fill_diagonal_(0)
        ovl.append(float(g.abs().max()))
    report = {"effective_rank": eff, "numerical_rank": num,
              "condition_number": cond, "max_column_overlap": ovl}
    if warn:
        # Gated on the *effective* rank, not the numerical one. Near-collinear
        # columns clear `matrix_rank`'s tolerance: measured on BRCA4 at
        # nsa_w=0.1 the effective rank was 1.53 and 1.90 for two views while
        # `numerical_rank` still said 3, so a check on the integer rank would
        # have stayed silent through the collapse it exists to catch.
        # Half a component is the threshold: `effective_rank` equals k exactly
        # for equinormed orthogonal columns and falls continuously from there.
        bad = [i for i in range(len(eff)) if eff[i] < k - 0.5 or num[i] < k]
        if bad:
            warnings.warn(
                f"simlr returned a degenerate basis for view(s) {bad} at k={k}: "
                f"effective rank {[round(eff[i], 2) for i in bad]}, numerical "
                f"rank {[num[i] for i in bad]}, condition number "
                f"{['%.1e' % cond[i] for i in bad]}, max column overlap "
                f"{[round(ovl[i], 3) for i in bad]}. Columns have merged, so "
                f"fewer than k distinct components were recovered -- note that "
                f"predictive accuracy does not reveal this. Raising nsa_w is "
                f"the usual fix; see the `nsa_w` docstring.",
                RuntimeWarning, stacklevel=3,
            )
    return report


#: Default relative tolerance for `ConvergenceMonitor`. Relative, not absolute:
#: the deep trainer used an absolute `tol=1e-6` against a loss of order 10, so
#: any decrease at all reset its patience counter and it never stopped early --
#: measured, it ran its full budget at 8, 30 and 80 epochs alike while the loss
#: was still falling at -0.37 per epoch at the 8-epoch cap.
CONVERGENCE_TOL = DEFAULT_TOL
CONVERGENCE_PATIENCE = 10


class ConvergenceMonitor:
    """One convergence rule for every method in the package.

    `simlr` stopped on a stationarity certificate (``grad_map <= tol``) or on a
    joint energy/certificate plateau; the deep trainer stopped on loss patience
    with an absolute tolerance. Two different questions, two different report
    shapes (``stop_reason``/``certificate``/``grad_map`` against
    ``converged_iter``), and no way to set "run until converged" once and have
    it mean the same thing everywhere.

    The rule here is deliberately the weaker, universal one: **relative
    improvement**. A step counts as progress when it improves the monitored
    quantity by more than ``tol`` *relative to the best value so far*; after
    ``patience`` consecutive non-improving steps the run has converged in the
    only sense that applies to every method here.

    Every family reports the same ``stop_reason`` vocabulary
    (`api.STOP_REASONS`): ``converged`` (this rule fired; certificate
    ``"energy_plateau"``), ``max_iter`` (budget exhausted), ``non_finite``.
    `simlr` adds ``no_descent``; closed-form baselines report ``closed_form``.

    Parameters
    ----------
    tol : float
        Relative improvement below which a step does not count as progress.
    patience : int
        Consecutive non-improving steps before declaring convergence.
    max_steps : int
        Hard cap. Reaching it is reported as ``stop_reason="max_iter"`` and
        ``converged=False`` -- a budget exhausted is not a fit that finished.
    """

    def __init__(self, tol: float = CONVERGENCE_TOL,
                 patience: int = CONVERGENCE_PATIENCE,
                 max_steps: int = 1000, smooth: int = 5):
        self.tol = float(tol)
        self.patience = int(patience)
        self.max_steps = int(max_steps)
        self.smooth = max(1, int(smooth))
        self.history: List[float] = []
        self.best = float("inf")
        self._stale = 0
        self.stop_reason = None
        self.certificate = None

    def update(self, value: float) -> bool:
        """Record one step. Returns True when the run should stop."""
        v = float(value)
        self.history.append(v)
        if not np.isfinite(v):
            # A non-finite objective is not convergence and not progress; it
            # is a failed step, and continuing to burn budget on it hides the
            # failure behind a "max_iter" report.
            self.stop_reason = "non_finite"
            return True
        # Compared on a trailing mean, not on the raw value. A stochastic
        # objective moves by more than `tol` from epoch to epoch on noise
        # alone, so the raw test makes the stopping point itself random:
        # measured on one fixture, budgets of 120 and 300 epochs both reported
        # "converged" but at losses of 4.45 and 3.71, because whether ten
        # consecutive non-improvements happened early was luck. Smoothing
        # makes the decision a statement about the trend.
        win = [x for x in self.history[-self.smooth:] if np.isfinite(x)]
        v = float(np.mean(win)) if win else v
        rel = (self.best - v) / max(abs(self.best), 1e-12) if np.isfinite(self.best) else np.inf
        if rel > self.tol:
            self.best = v
            self._stale = 0
        else:
            self.best = min(self.best, v)
            self._stale += 1
        if self._stale >= self.patience:
            self.stop_reason = "converged"
            self.certificate = "energy_plateau"
            return True
        if len(self.history) >= self.max_steps:
            self.stop_reason = "max_iter"
            return True
        return False

    def report(self) -> Dict[str, Any]:
        """The same keys for every method, so one checker reads them all."""
        h = self.history
        recent = h[-(self.patience + 1):] if len(h) > 1 else h
        rel = ((max(recent) - min(recent)) / max(abs(max(recent)), 1e-12)
               if len(recent) > 1 else float("nan"))
        return {
            "converged": self.stop_reason == "converged",
            "stop_reason": self.stop_reason or "max_iter",
            "certificate": self.certificate,
            "n_steps": len(h),
            "max_steps": self.max_steps,
            "tol": self.tol,
            "patience": self.patience,
            "best": self.best if np.isfinite(self.best) else float("nan"),
            "final": h[-1] if h else float("nan"),
            "recent_relative_range": rel,
        }


def align_column_signs(mats: List[torch.Tensor],
                       reference: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    r"""Put a set of latent matrices on a common sign convention, column by column.

    A latent factorisation is defined only up to a sign per component: if
    :math:`u_j` explains a view then so does :math:`-u_j`, with the loading
    negated. Two views can therefore recover *the same* structure with
    opposite signs, and averaging them then cancels it.

    That is not hypothetical. Measured on a planted two-factor signal, two
    views' latents were correlated at **-0.927 and -0.959** -- the same
    factors, opposite signs -- and the plain mean collapsed from a standard
    deviation of ~1.0 per column to 0.175 and 0.121, taking the predictive
    R^2 of the consensus from 0.795 to **0.023**. The per-view latents scored
    0.784 and 0.785; the average of them scored nothing. Aligning the signs
    first restored 0.795.

    The failure is silent and seed-dependent, which is what makes it
    dangerous: across three seeds on one fixture the same model scored 0.034,
    0.347 and 0.183, and a sibling model scored 0.701, 0.034 and 0.787. It
    reads as an unstable method rather than as arithmetic.

    Concatenating estimators (`svd`, `pca`, `ica`) are immune, because a sign
    flip is absorbed by the basis they fit. Only estimators that *average*
    across views need this.

    The reference is built greedily -- the first matrix, then each subsequent
    one flipped to agree with the running sum -- which is exact for two views
    and stable for more. Signs are detached, so this is a relabelling and
    does not change the gradient's magnitude.

    Parameters
    ----------
    mats : list of torch.Tensor
        Latents of matching shape ``(n, k)``.
    reference : torch.Tensor, optional
        Align to this instead of to the first matrix.

    Returns
    -------
    list of torch.Tensor
        The inputs, each with columns sign-flipped to agree.
    """
    if not mats:
        return mats
    if len(mats) == 1 and reference is None:
        return list(mats)
    running = (reference if reference is not None else mats[0]).detach().clone()
    out, start = [], 0
    if reference is None:
        out.append(mats[0])
        start = 1
    for m in mats[start:]:
        # sign of the per-column inner product with the running reference
        dots = (m.detach() * running).sum(dim=0, keepdim=True)
        sgn = torch.sign(dots)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        aligned = m * sgn
        out.append(aligned)
        running = running + aligned.detach()
    return out
