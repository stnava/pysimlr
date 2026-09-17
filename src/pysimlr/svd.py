import torch
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple, List
from .utils import safe_svd

def ba_svd(x: torch.Tensor, 
           nu: Optional[int] = None, 
           nv: Optional[int] = None, 
           max_iter: int = 100, 
           tol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a Basic block-based SVD approximation or full SVD.

    A wrapper around torch.linalg.svd that is NaN-safe on input and
    hardware-aware (see :func:`pysimlr.utils.safe_svd`). Requesting more
    vectors than the matrix rank supports returns fewer; use
    :func:`safe_pca` if you need the result zero-padded to a fixed width.

    Parameters
    ----------
    x : torch.Tensor or array-like
        The input matrix to decompose.
    nu : int, optional
        Number of left singular vectors to return. Defaults to min(n, p).
    nv : int, optional
        Number of right singular vectors to return. Defaults to min(n, p).
    max_iter : int, default=100
        Maximum iterations (reserved for iterative variants).
    tol : float, default=1e-6
        Convergence tolerance (reserved for iterative variants).

    Returns
    -------
    u : torch.Tensor
        Left singular vectors (U matrix).
    s : torch.Tensor
        Singular values.
    v : torch.Tensor
        Right singular vectors (V matrix, not transposed).

    Raises
    ------
    RuntimeError
        If the underlying SVD does not converge.
    TypeError
        If the input is not a tensor or array-like structure.
    """
    x = torch.as_tensor(x).float()
    x = torch.nan_to_num(x, nan=0.0)
    
    n, p = x.shape
    if nu is None: nu = min(n, p)
    if nv is None: nv = min(n, p)
    
    try:
        u, s, vh = safe_svd(x, full_matrices=False)
    except RuntimeError as exc:
        # Returning torch.randn() here made a numerical failure indistinguishable
        # from a successful decomposition: downstream code silently consumed
        # random vectors as singular vectors. A failed SVD is an error.
        raise RuntimeError(
            f"SVD failed for a {n}x{p} matrix. The input may contain infinities "
            f"or be otherwise ill-conditioned (finite: "
            f"{bool(torch.isfinite(x).all())})."
        ) from exc

    u = u[:, :nu] if nu > 0 else u[:, :0]
    s = s[:min(nu, nv)] if min(nu, nv) > 0 else s[:0]
    v = vh.t()[:, :nv] if nv > 0 else vh.t()[:, :0]

    return u, s, v

def safe_pca(x: torch.Tensor, nc: int = 2) -> Dict[str, torch.Tensor]:
    """
    Perform Principal Component Analysis (PCA) robust to NaNs and constant columns.

    This function centers the data and handles constant features by masking them 
    out before performing SVD. It ensures that the output dimensions remain 
    consistent with the input feature space.

    Parameters
    ----------
    x : torch.Tensor or array-like
        The input data matrix (samples x features).
    nc : int, default=2
        The number of principal components to extract.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing:
        - "u": Projected sample coordinates (scores).
        - "v": Feature loadings (rotation matrix), zeroed for constant features.
        - "s": Singular values (related to explained variance).

    Raises
    ------
    TypeError
        If the input is not a tensor or array-like structure.
    """
    x = torch.as_tensor(x).float()
    x = torch.nan_to_num(x, nan=0.0)
    
    if x.shape[0] > 1:
        std = torch.std(x, dim=0)
    else:
        std = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
    mask = std > 1e-10
    if not torch.any(mask):
        return {
            "u": torch.zeros(x.shape[0], nc, device=x.device),
            "v": torch.zeros(x.shape[1], nc, device=x.device),
            "s": torch.zeros(nc, device=x.device)
        }
        
    x_proc = x[:, mask]
    x_centered = x_proc - torch.mean(x_proc, dim=0)
    
    u, s, v = ba_svd(x_centered, nu=nc, nv=nc)

    # The decomposition can yield fewer than `nc` components when nc exceeds
    # min(n_samples, n_retained_features). Zero-pad so the returned shapes are
    # always (n, nc) / (p, nc) / (nc,), matching both the docstring and the
    # all-constant early-return branch above. Assigning the short `v` straight
    # into a width-nc buffer previously raised a shape-mismatch RuntimeError.
    n_found = min(u.shape[1], v.shape[1], s.shape[0])

    u_full = torch.zeros(x.shape[0], nc, device=x.device, dtype=u.dtype)
    u_full[:, :n_found] = u[:, :n_found]

    v_full = torch.zeros(x.shape[1], nc, device=x.device, dtype=v.dtype)
    v_full[mask, :n_found] = v[:, :n_found]

    s_full = torch.zeros(nc, device=x.device, dtype=s.dtype)
    s_full[:n_found] = s[:n_found]

    return {"u": u_full, "v": v_full, "s": s_full}

def whiten_matrix(x: torch.Tensor, nc: Optional[int] = None) -> Dict[str, Any]:
    """
    Whiten a data matrix using Principal Component Analysis.

    Whitening (or sphering) transforms the data such that the covariance 
    matrix is the identity matrix. This is often used as a preprocessing 
    step for algorithms like ICA.

    Parameters
    ----------
    x : torch.Tensor or array-like
        The input data matrix to whiten.
    nc : int, optional
        Number of components to keep. Defaults to min(samples, features).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "whitened_matrix": The transformed data, shape (n, nc). Each retained
          column has zero mean and unit variance, and distinct columns are
          uncorrelated, so ``W.T @ W / (n - 1)`` is the identity on the
          retained block.
        - "pca_res": The underlying PCA result (from `safe_pca`).
        - "rank": Number of components with non-negligible singular value.
          Columns beyond this are returned as exact zeros, because a direction
          with no variance cannot be scaled to unit variance.

    Notes
    -----
    `safe_pca` already returns the orthonormal left singular vectors `U`, which
    are the scores ``U @ S`` already divided by `S` -- that is, the whitened
    coordinates up to a constant. Earlier revisions divided by `S` a *second*
    time (``u * (1 / s)``), producing a matrix whose column variances fell off
    as ``1 / s**2`` rather than being constant. The only scaling needed is
    ``sqrt(n - 1)``, which turns the unit-norm columns into unit-variance ones.

    Examples
    --------
    >>> import torch
    >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
    <torch._C.Generator object at ...>
    >>> x = torch.randn(200, 4) @ torch.diag(torch.tensor([10.0, 5.0, 1.0, 0.5]))
    >>> w = whiten_matrix(x)["whitened_matrix"]
    >>> cov = (w - w.mean(0)).t() @ (w - w.mean(0)) / (w.shape[0] - 1)
    >>> bool(torch.allclose(cov, torch.eye(4), atol=1e-4))
    True

    Raises
    ------
    TypeError
        If the input is not a tensor or array-like structure.
    """
    x = torch.as_tensor(x).float()
    res = safe_pca(x, nc=nc if nc else min(x.shape))
    u, s = res['u'], res['s']

    n_samples = u.shape[0]
    # U has orthonormal columns with (exactly) zero mean, because the data was
    # centered before the SVD. Scaling by sqrt(n - 1) therefore yields unit
    # variance, i.e. an identity covariance, without touching S again.
    scale = float(np.sqrt(max(n_samples - 1, 1)))

    # Directions with no variance stay zero rather than being scaled up from
    # numerical noise. Uses the conventional numerical-rank cutoff
    # (max(shape) * eps * s_max), matching numpy.linalg.matrix_rank; a tighter
    # relative cutoff leaves float32 null-space directions in, and those are
    # not mean-zero, so they come back with variance != 1.
    if s.numel():
        eps = torch.finfo(s.dtype).eps
        tol = float(s.max()) * max(x.shape) * eps
        keep = s > max(tol, torch.finfo(s.dtype).tiny)
    else:
        keep = s > 0
    whitened = u * scale
    if keep.numel():
        whitened = whitened * keep.to(whitened.dtype)

    return {
        "whitened_matrix": whitened,
        "pca_res": res,
        "rank": int(keep.sum()) if keep.numel() else 0,
    }

def multiscale_svd(x: torch.Tensor,
                   r: torch.Tensor,
                   locn: Union[int, List[int], torch.Tensor],
                   nev: int,
                   knn: int = 0,
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Perform multi-scale SVD to analyze local intrinsic dimensionality.

    This function computes the singular values of the data at different
    neighborhood sizes. It is useful for estimating the local dimension of a
    manifold.

    Warnings
    --------
    With the default ``knn=0`` no neighbourhood selection happens at all: the
    same sampled subset is used at every scale and `r` acts purely as a
    divisor, so every row of the returned table is the same spectrum scaled by
    ``1 / r``. Pass ``knn > 0`` to get genuine multi-scale behaviour, where
    each scale averages the spectra of k-nearest-neighbour patches. The
    radius-based neighbour selection implied by calling `r` a radius is not
    implemented.

    Parameters
    ----------
    x : torch.Tensor or array-like
        The input data matrix.
    r : torch.Tensor or array-like
        A vector of scale denominators. Each returned spectrum is divided by
        the corresponding entry; see the warning about ``knn=0``.
    locn : int, List[int], or torch.Tensor
        Indices of locations to sample, or an integer specifying 
        the number of random locations to choose.
    nev : int
        Number of eigenvalues/singular values to track at each scale.
    knn : int, default=0
        If > 0, use K-nearest neighbors instead of a fixed radius.
    verbose : bool, default=False
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "evals_vs_scale": A tensor of shape (len(r), nev) containing 
          singular values for each scale.

    Raises
    ------
    TypeError
        If the inputs are not tensors or array-like structures.
    """
    x = torch.as_tensor(x).float()
    r = torch.as_tensor(r).float()
    orig_dtype = x.dtype
    
    if x.ndim == 1:
        x = x.unsqueeze(1)
    n, p = x.shape
    
    if isinstance(locn, (list, torch.Tensor, np.ndarray)):
        locn_indices = torch.as_tensor(locn).long()
    else:
        locn_indices = torch.randperm(n)[:int(locn)]

    k_eff = min(knn, n) if knn > 0 else len(locn_indices)
    max_rank = min(k_eff, p)
    actual_nev = min(nev, max_rank)
    
    m_response = torch.full((len(r), actual_nev), float('nan'), dtype=orig_dtype, device=x.device)
    if actual_nev == 0 or len(locn_indices) == 0:
        return {"evals_vs_scale": m_response}

    for scl_idx, my_r in enumerate(r):
        denom = my_r if torch.abs(my_r) > 1e-12 else 1.0
        if knn > 0:
            dist = torch.cdist(x[locn_indices], x)
            _, indices = torch.topk(dist, k=k_eff, largest=False)
            subset = x[indices]
            subset_c = (subset - subset.mean(dim=-2, keepdim=True)) / denom
            _, s_all, _ = safe_svd(subset_c, full_matrices=False)
            s = s_all.mean(dim=0)
            m_response[scl_idx, :] = s[:actual_nev]
        else:
            subset = x[locn_indices]
            subset_c = (subset - torch.mean(subset, dim=0)) / denom
            _, s, _ = safe_svd(subset_c, full_matrices=False)
            m_response[scl_idx, :actual_nev] = s[:actual_nev]
            
    return {"evals_vs_scale": m_response}
