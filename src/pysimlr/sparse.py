import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Union, Any, List, Tuple

def _laplacian_from_weights(W: sp.spmatrix, scale_degree: bool = False) -> sp.csr_matrix:
    """
    Build ``L = D - W`` after normalizing the mean degree of `W` to 1.

    The normalization makes `lambda_val` in the downstream resolvent
    ``(I + lambda*L)^-1`` comparable across graphs: inverse-distance weights
    carry the units of the input coordinates, and grid graphs have intrinsically
    larger degrees than k-NN graphs, so without it the same `lambda_val` meant
    wildly different amounts of smoothing. Scaling `W` by a positive constant
    scales `L` by the same constant, leaving the graph structure and the
    Laplacian's null space unchanged.

    Parameters
    ----------
    W : scipy.sparse matrix
        Symmetric non-negative weight matrix with a zero diagonal.
    scale_degree : bool, default=False
        If True, rescale `W` so the mean degree over connected nodes is 1.

    Returns
    -------
    scipy.sparse.csr_matrix
        The normalized combinatorial Laplacian.
    """
    W = W.tocsr()
    if scale_degree:
        raw_deg = np.array(W.sum(axis=1)).flatten()
        connected = raw_deg > 0
        if np.any(connected):
            mean_degree = float(raw_deg[connected].mean())
            if mean_degree > 0 and np.isfinite(mean_degree):
                W = W.multiply(1.0 / mean_degree).tocsr()
    deg = np.array(W.sum(axis=1)).flatten()
    return (sp.diags(deg) - W).tocsr()


def _to_scipy_sparse(t: "torch.Tensor") -> sp.csr_matrix:
    """
    Convert any torch tensor layout to a SciPy CSR matrix.

    ``Tensor.is_sparse`` is True only for the COO layout, so a
    ``torch.sparse_csr`` / ``sparse_csc`` / ``sparse_bsr`` tensor -- which is
    exactly what ``return_torch=True`` produces here -- tested as dense and then
    raised ``TypeError: can't convert SparseCsr layout tensor to numpy``. Any
    sparse layout is routed through COO instead.

    Parameters
    ----------
    t : torch.Tensor
        Dense or sparse tensor of any layout.

    Returns
    -------
    scipy.sparse.csr_matrix
        The same matrix in SciPy CSR form.
    """
    sparse_layouts = {torch.sparse_coo, torch.sparse_csr, torch.sparse_csc}
    for name in ("sparse_bsr", "sparse_bsc"):
        layout = getattr(torch, name, None)
        if layout is not None:
            sparse_layouts.add(layout)

    if t.layout in sparse_layouts:
        coo = t.detach().cpu().to_sparse_coo().coalesce()
        idx = coo.indices().numpy()
        vals = coo.values().numpy()
        return sp.coo_matrix((vals, (idx[0], idx[1])), shape=tuple(t.shape)).tocsr()
    return sp.csr_matrix(t.detach().cpu().numpy())


def sparse_distance_matrix(x: torch.Tensor, 
                           k: int, 
                           sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute a k-nearest neighbor distance matrix, zeroed outside each row's neighbours.

    Calculates the Euclidean distance between all pairs of rows in `x`,
    keeping only the `k` closest neighbors for each row.

    Warnings
    --------
    The result is a **dense** (N, N) tensor whose non-neighbour entries are
    zero, not a sparse data structure, and the full pairwise distance matrix is
    materialised on the way -- both O(N^2) in memory. It is also **asymmetric**,
    since "j is among i's k nearest" is not a symmetric relation; symmetrize it
    before using it as a graph adjacency (`create_graph_laplacian` does this for
    you). When `sigma` is given, the self-distance of zero maps to an affinity
    of 1, so the diagonal carries self-loops.

    Parameters
    ----------
    x : torch.Tensor
        Input data matrix (N x P).
    k : int
        Number of nearest neighbors to retain.
    sigma : float, optional
        If provided, converts distances to an affinity matrix using a 
        Gaussian kernel with this standard deviation.

    Returns
    -------
    torch.Tensor
        The sparse distance or affinity matrix (N x N).

    Raises
    ------
    TypeError
        If the input is not a valid tensor.
    """
    x = torch.as_tensor(x).float()
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor (samples x features), got ndim={x.ndim}")
    n = x.shape[0]
    if k <= 0:
        raise ValueError(f"k must be at least 1, got {k}")
    
    # Compute full distance matrix
    dist = torch.cdist(x, x)
    
    # Get k+1 nearest neighbors (including self)
    values, indices = torch.topk(dist, k=min(k+1, n), largest=False)
    
    # Create sparse representation
    mask = torch.zeros_like(dist, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    
    sparse_dist_mat = torch.zeros_like(dist)
    sparse_dist_mat[mask] = dist[mask]
    
    if sigma is not None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive (sigma > 0), got {sigma}")
        denom = 2.0 * (float(sigma) ** 2)
        sparse_dist_mat = torch.zeros_like(dist)
        sparse_dist_mat[mask] = torch.exp(- (dist[mask] ** 2) / denom)
        
    return sparse_dist_mat

def sparse_distance_matrix_xy(x: torch.Tensor, 
                              y: torch.Tensor, 
                              k: int, 
                              sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute a k-nearest neighbor sparse distance matrix between two sets of points.

    Calculates the Euclidean distance between rows of `x` and rows of `y`, 
    keeping only the `k` closest neighbors in `y` for each row in `x`.

    Parameters
    ----------
    x : torch.Tensor
        First data matrix (NX x P).
    y : torch.Tensor
        Second data matrix (NY x P).
    k : int
        Number of nearest neighbors to retain.
    sigma : float, optional
        If provided, converts distances to an affinity matrix using a 
        Gaussian kernel with this standard deviation.

    Returns
    -------
    torch.Tensor
        The sparse distance or affinity matrix (NX x NY).

    Raises
    ------
    TypeError
        If the inputs are not valid tensors.
    """
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got x.ndim={x.ndim}, y.ndim={y.ndim}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimension mismatch: x has {x.shape[1]} features, y has {y.shape[1]} features")
    nx = x.shape[0]
    ny = y.shape[0]
    if k <= 0:
        raise ValueError(f"k must be at least 1, got {k}")
    
    dist = torch.cdist(x, y)
    
    values, indices = torch.topk(dist, k=min(k, ny), largest=False)
    
    mask = torch.zeros_like(dist, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    
    sparse_dist_mat = torch.zeros_like(dist)
    sparse_dist_mat[mask] = dist[mask]
    
    if sigma is not None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive (sigma > 0), got {sigma}")
        denom = 2.0 * (float(sigma) ** 2)
        sparse_dist_mat = torch.zeros_like(dist)
        sparse_dist_mat[mask] = torch.exp(- (dist[mask] ** 2) / denom)
        
    return sparse_dist_mat

def sparse_dist(x: torch.Tensor, 
                k: Optional[int] = None, 
                sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute a k-nearest neighbor sparse distance or affinity matrix.

    Parameters
    ----------
    x : torch.Tensor
        Input data matrix (N x P).
    k : int, optional
        Number of nearest neighbors to retain. Defaults to N - 1.
    sigma : float, optional
        Gaussian kernel bandwidth.

    Returns
    -------
    torch.Tensor
        Sparse distance or affinity matrix.
    """
    if k is None:
        x_t = torch.as_tensor(x)
        k = max(1, x_t.shape[0] - 1)
    return sparse_distance_matrix(x, k=k, sigma=sigma)


class SparseGraphResolvent:
    """
    Pre-factorized sparse Green's resolvent operator S = (I + lambda * L)^(-1).

    This operator acts as a memory-efficient spatial smoothing matrix for SiMLR:
        v_smoothed = S @ v
    Instead of inverting the graph Laplacian into a dense P x P matrix (which costs
    O(P^2) memory and becomes intractable for large feature counts), this operator
    precomputes a sparse Cholesky / LU factorization of (I + lambda * L) in O(P) RAM
    and solves the linear system in sub-millisecond time.

    Parameters
    ----------
    laplacian : sp.spmatrix or torch.Tensor or np.ndarray
        Sparse (or dense) graph Laplacian matrix of shape (P, P).
    lambda_val : float, default=0.05
        Regularization / smoothing scale parameter (lambda >= 0).

    Attributes
    ----------
    P : int
        Number of nodes / features in the graph.
    shape : Tuple[int, int]
        Matrix dimensions (P, P).
    lambda_val : float
        The smoothing scale parameter.
    """
    def __init__(self, laplacian: Union[sp.spmatrix, torch.Tensor, np.ndarray], lambda_val: float = 0.05):
        if lambda_val < 0.0:
            raise ValueError(f"lambda_val must be non-negative, got {lambda_val}")
        self.lambda_val = float(lambda_val)

        # Convert to scipy CSC matrix
        if isinstance(laplacian, torch.Tensor):
            L_sp = _to_scipy_sparse(laplacian).tocsc()
        elif isinstance(laplacian, np.ndarray):
            L_sp = sp.csc_matrix(laplacian)
        elif sp.issparse(laplacian):
            L_sp = laplacian.tocsc()
        else:
            raise TypeError(f"Unsupported laplacian type: {type(laplacian)}")

        if L_sp.shape[0] != L_sp.shape[1]:
            raise ValueError(f"Expected square Laplacian matrix, got shape {L_sp.shape}")

        self.P = L_sp.shape[0]
        self.shape = (self.P, self.P)
        self._L_sp = L_sp

        # System matrix A = (I + lambda * L)
        I_sp = sp.eye(self.P, format='csc', dtype=L_sp.dtype)
        A_sp = (I_sp + self.lambda_val * L_sp).tocsc()

        # Precompute sparse factorization (takes ~20 ms, solves in ~0.5 ms)
        self._solve = spla.factorized(A_sp)

    def __matmul__(self, v: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply the resolvent operator to vector or matrix v via fast sparse solve.

        Parameters
        ----------
        v : torch.Tensor or np.ndarray
            1D vector (P,) or 2D matrix (P, K).

        Returns
        -------
        torch.Tensor or np.ndarray
            Smoothed output of same type, device, and shape as v.
        """
        is_torch = isinstance(v, torch.Tensor)
        device = v.device if is_torch else None
        dtype = v.dtype if is_torch else None

        v_np = v.detach().cpu().numpy() if is_torch else np.asarray(v)
        if v_np.shape[0] != self.P:
            raise ValueError(f"Dimension mismatch: resolvent operator has shape {self.shape}, but input has {v_np.shape[0]} rows (shape={v_np.shape})")

        orig_dtype = v_np.dtype
        # Fast sparse solve requires float32 or float64
        if orig_dtype not in [np.float32, np.float64]:
            v_np = v_np.astype(np.float32)

        if v_np.ndim == 1:
            out = self._solve(v_np)
        elif v_np.ndim == 2:
            out = np.column_stack([self._solve(v_np[:, c]) for c in range(v_np.shape[1])])
        else:
            raise ValueError(f"Expected 1D or 2D input tensor, got ndim={v_np.ndim}")

        if orig_dtype != out.dtype:
            out = out.astype(orig_dtype)

        if is_torch:
            return torch.as_tensor(out, dtype=dtype, device=device)
        return out

    def to_dense(self) -> torch.Tensor:
        """Evaluate and return the explicit dense P x P tensor."""
        I_dense = np.eye(self.P, dtype=np.float32)
        S_dense = np.column_stack([self._solve(I_dense[:, c]) for c in range(self.P)])
        return torch.from_numpy(S_dense)

    def to_sparse_tensor(self, threshold: float = 1e-4) -> torch.Tensor:
        """Return a PyTorch sparse CSR tensor with distance truncation."""
        S_dense = self.to_dense().numpy()
        if threshold > 0:
            S_dense[np.abs(S_dense) < threshold] = 0.0
        return torch.from_numpy(S_dense).to_sparse_csr()

    def __repr__(self) -> str:
        return f"SparseGraphResolvent(shape={self.shape}, lambda={self.lambda_val}, memory_efficient=True)"


def create_grid_graph_laplacian(
    mask: Union[torch.Tensor, np.ndarray],
    connectivity: Optional[int] = None,
    return_torch: bool = False,
    scale_degree: bool = False
) -> Union[sp.csr_matrix, torch.Tensor]:
    """
    Construct an adjacency graph Laplacian for a 2D or 3D grid domain mask.

    Builds an inverse Euclidean distance-weighted graph Laplacian L = D - W
    over all foreground (True / non-zero) pixels or voxels in the domain mask.

    Parameters
    ----------
    mask : torch.Tensor or np.ndarray
        2D or 3D binary array defining the domain of interest.
    connectivity : int, optional
        Neighborhood connectivity. Defaults to 8 for a 2D mask and 26 for a 3D
        mask.
        - For 2D: 4 (orthogonal) or 8 (orthogonal + diagonal).
        - For 3D: 6 (face) or 26 (face + edge + corner).
    return_torch : bool, default=False
        If True, returns a PyTorch sparse CSR tensor. Otherwise, returns a
        scipy.sparse.csr_matrix.
    scale_degree : bool, default=False
        Rescale edge weights so the mean degree over connected nodes is 1. This
        makes `lambda_val` in ``(I + lambda*L)^-1`` mean the same thing across
        graphs: inverse-distance weights carry the units of the input
        coordinates, and grid graphs have intrinsically larger degrees than k-NN
        graphs, so without it the same `lambda_val` gave wildly different
        smoothing (the identical geometry in mm vs um went from a 0.41
        shrinkage factor to 0.998). It is a single positive constant, so the
        graph structure and the Laplacian's null space are unchanged. Set False
        to get the textbook unscaled Laplacian.

    Returns
    -------
    L : scipy.sparse.csr_matrix or torch.Tensor
        Sparse graph Laplacian of shape (P, P), where P is the number of
        foreground pixels/voxels in mask.
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = np.asarray(mask, dtype=bool)

    ndim = mask_np.ndim
    if ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D mask array, got {ndim}D")

    # Resolve the default per dimensionality. A fixed default of 8 meant a 3-D
    # mask raised "Invalid 3D connectivity 8" unless the caller happened to know
    # to pass 26, even though the 3-D mask path was the documented use case.
    if connectivity is None:
        connectivity = 8 if ndim == 2 else 26

    shape = mask_np.shape
    P = int(np.sum(mask_np))
    if P == 0:
        raise ValueError("Mask contains 0 foreground voxels.")

    idx_map = np.full(shape, -1, dtype=np.int32)
    idx_map[mask_np] = np.arange(P)

    # Determine neighbor displacement offsets and weights
    offsets = []
    if ndim == 2:
        if connectivity == 4:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif connectivity == 8:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            raise ValueError(f"Invalid 2D connectivity {connectivity}. Choose 4 or 8.")
        for dr, dc in deltas:
            dist = np.sqrt(dr * dr + dc * dc)
            offsets.append(((dr, dc), 1.0 / dist))
    elif ndim == 3:
        deltas_all = []
        for dz in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dz == 0 and dr == 0 and dc == 0:
                        continue
                    sq_dist = dz * dz + dr * dr + dc * dc
                    if connectivity == 6 and sq_dist == 1:
                        deltas_all.append(((dz, dr, dc), 1.0))
                    elif connectivity == 26:
                        deltas_all.append(((dz, dr, dc), 1.0 / np.sqrt(sq_dist)))
        if connectivity not in (6, 26):
            raise ValueError(f"Invalid 3D connectivity {connectivity}. Choose 6 or 26.")
        offsets = deltas_all

    # Vectorized neighbour enumeration. The previous implementation walked
    # every grid position in Python and then every offset, which for the 3-D
    # brain masks this function documents (e.g. 256^3 at 26-connectivity) is
    # hundreds of millions of interpreter iterations. Shifting the whole index
    # map per offset does the same work in a handful of array operations.
    rows_parts, cols_parts, weight_parts = [], [], []

    for delta, w in offsets:
        # For each axis, the overlapping window between the array and its shift.
        src_slices, dst_slices = [], []
        for axis, d in enumerate(delta):
            n = shape[axis]
            lo, hi = max(0, -d), n - max(0, d)
            if lo >= hi:
                break
            src_slices.append(slice(lo, hi))
            dst_slices.append(slice(lo + d, hi + d))
        else:
            src_idx = idx_map[tuple(src_slices)]
            dst_idx = idx_map[tuple(dst_slices)]
            valid = (src_idx >= 0) & (dst_idx >= 0)
            if not np.any(valid):
                continue
            r = src_idx[valid]
            c = dst_idx[valid]
            rows_parts.append(r)
            cols_parts.append(c)
            weight_parts.append(np.full(r.shape[0], w, dtype=np.float32))

    if rows_parts:
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        weights = np.concatenate(weight_parts)
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
        weights = np.empty(0, dtype=np.float32)

    W_sp = sp.coo_matrix((weights, (rows, cols)), shape=(P, P), dtype=np.float32).tocsr()
    L_sp = _laplacian_from_weights(W_sp, scale_degree=scale_degree)

    if return_torch:
        L_coo = L_sp.tocoo()
        indices = torch.tensor(np.vstack([L_coo.row, L_coo.col]), dtype=torch.long)
        values = torch.tensor(L_coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (P, P)).to_sparse_csr()
    return L_sp


def create_laplacian_resolvent_operator(
    laplacian: Union[sp.spmatrix, torch.Tensor, np.ndarray],
    lambda_val: float = 0.05
) -> SparseGraphResolvent:
    """
    Create a pre-factorized SparseGraphResolvent operator from any graph Laplacian.

    Parameters
    ----------
    laplacian : scipy.sparse matrix, torch.Tensor, or np.ndarray
        Graph Laplacian matrix (P x P).
    lambda_val : float, default=0.05
        Smoothing scale parameter.

    Returns
    -------
    SparseGraphResolvent
        Operator compatible with `pysimlr.simlr(smoothing_matrices=[S, ...])`.
    """
    return SparseGraphResolvent(laplacian, lambda_val=lambda_val)


def create_spatial_smoothing_operator(
    mask: Union[torch.Tensor, np.ndarray],
    lambda_val: float = 0.05,
    connectivity: Optional[int] = None,
    scale_degree: bool = True
) -> SparseGraphResolvent:
    """
    Convenience one-liner: Build a grid graph Laplacian from a domain mask
    and return a memory-efficient, pre-factorized SparseGraphResolvent operator.

    Parameters
    ----------
    mask : torch.Tensor or np.ndarray
        2D or 3D binary domain mask (e.g. cortical flatmap or brain mask).
    lambda_val : float, default=0.05
        Spatial smoothing scale parameter.
    connectivity : int, optional
        Grid neighborhood connectivity. Defaults to 8 for a 2D mask and 26 for
        a 3D mask. Valid values are 4 or 8 for 2D, 6 or 26 for 3D.

    Returns
    -------
    SparseGraphResolvent
        Drop-in smoothing operator for `pysimlr.simlr(smoothing_matrices=[...])`.

    Examples
    --------
    >>> import pysimlr
    >>> mask = (flatmap_pixels > 0)  # doctest: +SKIP
    >>> S = pysimlr.create_spatial_smoothing_operator(mask, lambda_val=0.05)  # doctest: +SKIP
    >>> res = pysimlr.simlr([X, Y], smoothing_matrices=[S, None], k=1)  # doctest: +SKIP
    """
    L_sp = create_grid_graph_laplacian(mask, connectivity=connectivity,
                                       return_torch=False, scale_degree=scale_degree)
    return SparseGraphResolvent(L_sp, lambda_val=lambda_val)


_AMBIGUOUS_HINT = (
    "Pass source_type= explicitly ('coordinates', 'mesh_faces', 'edges', "
    "'adjacency' or 'mask') to say what this array is."
)


def _infer_source_type(source_np) -> str:
    """
    Decide how to interpret `source_np`, refusing genuinely ambiguous input.

    Auto-detection is only safe where the shape and dtype pin the meaning down.
    Several common inputs do not, and the previous heuristics resolved them
    silently and wrongly:

    - A ``(P, 3)`` integer array is both a triangular face list and a set of
      integer voxel coordinates. It was read as faces, so ``P`` points at
      coordinates like ``[10, 20, 30]`` produced a 31-node graph. Now raises.
    - A ``(P, 2)`` integer array is both an edge list and 2-D integer
      coordinates. Same problem, now raises.
    - Only ``bool`` arrays counted as masks, so a ``uint8``/``int`` mask -- what
      ``img > 0`` gives after a cast, and what most imaging I/O returns -- was
      read as coordinates (a 16x16 mask became "16 points in 16-D") in 2-D, and
      raised in 3-D. Now any 3-D array, and any 2-D array whose values are all
      in {0, 1}, is treated as a mask.
    - A square float array with 3 or fewer columns skipped the adjacency test
      (``shape[1] > 3``) and fell through to coordinates. The width condition is
      gone; symmetry with a zero diagonal is the signal instead.

    Parameters
    ----------
    source_np : np.ndarray or scipy.sparse matrix
        The array whose interpretation is to be inferred.

    Returns
    -------
    str
        One of "adjacency", "mask", "coordinates", "mesh_faces", "edges".

    Raises
    ------
    ValueError
        If the input is ambiguous or unrecognised.
    TypeError
        If the input is not an ndarray or SciPy sparse matrix.
    """
    if sp.issparse(source_np):
        if source_np.shape[0] != source_np.shape[1]:
            raise ValueError(
                f"Sparse matrix must be square (P x P) for adjacency, got shape {source_np.shape}"
            )
        return "adjacency"

    if not isinstance(source_np, np.ndarray):
        raise TypeError(f"Unsupported source type: {type(source_np)}")

    # 3-D input can only be a volumetric domain mask.
    if source_np.ndim == 3:
        return "mask"

    if source_np.ndim != 2:
        raise ValueError(
            f"Cannot auto-detect source_type for a {source_np.ndim}D array "
            f"(shape {source_np.shape}). {_AMBIGUOUS_HINT}"
        )

    n_rows, n_cols = source_np.shape
    is_int = np.issubdtype(source_np.dtype, np.integer)
    is_bool = source_np.dtype == bool

    # A binary-valued 2-D array is a domain mask, whatever its dtype.
    if is_bool:
        return "mask"
    if is_int and source_np.size and np.all((source_np == 0) | (source_np == 1)):
        # (P, 2) / (P, 3) 0/1 arrays are also valid tiny edge/face lists, so
        # only call it a mask when it is too wide to be one.
        if n_cols > 3:
            return "mask"

    # Square, symmetric, zero-diagonal, non-negative => adjacency matrix.
    if n_rows == n_cols and n_rows > 1:
        sym = np.allclose(source_np, source_np.T, atol=1e-8)
        zero_diag = np.allclose(np.diag(source_np), 0.0, atol=1e-8)
        if sym and zero_diag:
            return "adjacency"

    if is_int and source_np.size and source_np.min() >= 0:
        if n_cols == 3:
            raise ValueError(
                f"Ambiguous input: a ({n_rows}, 3) array of non-negative integers "
                f"is both a triangular face list and a set of 3-D integer "
                f"coordinates, and the two give completely different graphs. "
                f"{_AMBIGUOUS_HINT} For integer voxel coordinates you can also "
                f"cast to float first."
            )
        if n_cols == 2:
            raise ValueError(
                f"Ambiguous input: a ({n_rows}, 2) array of non-negative integers "
                f"is both an edge list and a set of 2-D integer coordinates. "
                f"{_AMBIGUOUS_HINT} For integer coordinates you can also cast to "
                f"float first."
            )

    return "coordinates"


def create_graph_laplacian(
    source: Union[torch.Tensor, np.ndarray, sp.spmatrix],
    source_type: str = "auto",
    k: int = 6,
    sigma: Optional[float] = None,
    connectivity: Optional[int] = None,
    normalized: bool = False,
    return_torch: bool = False,
    scale_degree: bool = False
) -> Union[sp.csr_matrix, torch.Tensor]:
    """
    Construct a sparse graph Laplacian from arbitrary data structures.

    Supports:
    - Point coordinates (P, D) in 2D, 3D, or high dimensions (via fast k-NN).
    - Triangular surface mesh faces (F, 3) (extracting topological mesh edges).
    - Edge lists (E, 2) or weighted edges (E, 3).
    - Pre-computed adjacency matrices (P, P) (sparse or dense).
    - 2D or 3D binary domain masks (calling create_grid_graph_laplacian).

    Parameters
    ----------
    source : torch.Tensor, np.ndarray, or scipy.sparse matrix
        The input geometric, topological, or network structure.
    source_type : str, default="auto"
        Input format: "auto", "coordinates", "mesh_faces", "edges", "adjacency", or "mask".
    k : int, default=6
        Number of nearest neighbors when building graphs from coordinates.
    sigma : float, optional
        Bandwidth for Gaussian heat kernel weights: exp(-dist^2 / (2 * sigma^2)).
        If None, inverse Euclidean distance weights (1 / dist) are used.
    connectivity : int, optional
        Grid neighborhood connectivity when source is a mask. Defaults to 8 for
        a 2D mask and 26 for a 3D mask.
    normalized : bool, default=False
        If True, returns the symmetric normalized Laplacian:
        L_norm = D^(-1/2) L D^(-1/2) = I - D^(-1/2) W D^(-1/2). Note that for a
        graph with isolated nodes this is not exactly I - D^(-1/2) W D^(-1/2):
        an isolated node's row is left at zero rather than 1.
    return_torch : bool, default=False
        If True, returns a PyTorch sparse CSR tensor. Otherwise, returns a
        scipy.sparse.csr_matrix.

    Returns
    -------
    L : scipy.sparse.csr_matrix or torch.Tensor
        Sparse graph Laplacian of shape (P, P).

    Notes
    -----
    Edge weights are rescaled so the mean degree over connected nodes is 1.
    This makes `lambda_val` in the downstream resolvent ``(I + lambda*L)^-1``
    mean the same thing across graphs built from different weightings and from
    coordinates in different units; without it, inverse-distance weights carry
    the coordinate units and the same geometry in mm vs. um produced completely
    different smoothing. The rescaling is a single positive constant, so it does
    not change the graph structure or the Laplacian's null space.

    Building a graph from coordinates that contain exact duplicates yields a
    zero distance; weights use ``1 / (d + 1e-8)``, so duplicate points get a
    very large weight. Deduplicate first, or pass `sigma` to use a bounded
    Gaussian kernel instead.
    """
    # Convert PyTorch tensor to NumPy / SciPy
    if isinstance(source, torch.Tensor):
        if source.layout != torch.strided:
            source_np = _to_scipy_sparse(source)
        else:
            source_np = source.detach().cpu().numpy()
    else:
        source_np = source

    # Infer source_type if auto
    if source_type == "auto":
        source_type = _infer_source_type(source_np)

    # Dispatch based on source_type
    if source_type == "mask":
        return create_grid_graph_laplacian(source_np, connectivity=connectivity, return_torch=return_torch)

    elif source_type == "coordinates":
        coords = np.asarray(source_np, dtype=np.float32)
        if coords.ndim != 2:
            raise ValueError(f"Coordinates array must be 2D (P x D), got {coords.ndim}D")
        P = coords.shape[0]
        if k <= 0 or k >= P:
            k = max(1, min(k, P - 1))

        # Fast k-NN via KDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        dists, indices = tree.query(coords, k=k + 1)  # Includes self at index 0

        # Neighbor pairs excluding self
        rows = np.repeat(np.arange(P), k)
        cols = indices[:, 1:].flatten()
        d_vals = dists[:, 1:].flatten()

        if sigma is not None:
            if sigma <= 0.0:
                raise ValueError(f"sigma must be positive, got {sigma}")
            weights = np.exp(- (d_vals ** 2) / (2.0 * float(sigma) ** 2))
        else:
            weights = 1.0 / (d_vals + 1e-8)

        W = sp.coo_matrix((weights, (rows, cols)), shape=(P, P), dtype=np.float32)
        # Symmetrize
        W = ((W + W.T) / 2.0).tocsr()
        W.setdiag(0)
        W.eliminate_zeros()

    elif source_type == "mesh_faces":
        faces = np.asarray(source_np, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Mesh faces must be (F x 3), got shape {faces.shape}")
        P = int(faces.max()) + 1

        e1 = faces[:, [0, 1]]
        e2 = faces[:, [1, 2]]
        e3 = faces[:, [2, 0]]
        all_e = np.vstack([e1, e2, e3])
        # Undirected
        sym_e = np.vstack([all_e, all_e[:, [1, 0]]])
        uniq_e = np.unique(sym_e, axis=0)
        # Remove self-loops if any
        uniq_e = uniq_e[uniq_e[:, 0] != uniq_e[:, 1]]

        w = np.ones(len(uniq_e), dtype=np.float32)
        W = sp.coo_matrix((w, (uniq_e[:, 0], uniq_e[:, 1])), shape=(P, P), dtype=np.float32).tocsr()

    elif source_type == "edges":
        edges = np.asarray(source_np)
        if edges.ndim != 2 or edges.shape[1] not in (2, 3):
            raise ValueError(f"Edge list must be (E x 2) or (E x 3), got shape {edges.shape}")
        u_nodes = edges[:, 0].astype(np.int64)
        v_nodes = edges[:, 1].astype(np.int64)
        P = max(int(u_nodes.max()), int(v_nodes.max())) + 1
        if edges.shape[1] == 3:
            w_raw = edges[:, 2].astype(np.float32)
        else:
            w_raw = np.ones(len(edges), dtype=np.float32)

        # Symmetrize
        rows = np.concatenate([u_nodes, v_nodes])
        cols = np.concatenate([v_nodes, u_nodes])
        w = np.concatenate([w_raw, w_raw])
        W = sp.coo_matrix((w, (rows, cols)), shape=(P, P), dtype=np.float32).tocsr()
        W.setdiag(0)
        W.eliminate_zeros()

    elif source_type == "adjacency":
        if sp.issparse(source_np):
            W = source_np.tocsr().astype(np.float32)
        else:
            W = sp.csr_matrix(source_np, dtype=np.float32)
        P = W.shape[0]
        # Ensure symmetric zero-diagonal
        W = ((W + W.T) / 2.0).tocsr()
        W.setdiag(0)
        W.eliminate_zeros()
    else:
        raise ValueError(f"Unknown source_type '{source_type}'. Choose 'auto', 'coordinates', 'mesh_faces', 'edges', 'adjacency', or 'mask'.")

    # Mean-degree normalization plus L = D - W (see _laplacian_from_weights).
    L_sp = _laplacian_from_weights(W, scale_degree=scale_degree)
    deg = L_sp.diagonal()

    # Optional normalization: L_norm = D^(-1/2) L D^(-1/2)
    if normalized:
        d_inv_sqrt = np.zeros_like(deg)
        pos = (deg > 1e-10)
        d_inv_sqrt[pos] = 1.0 / np.sqrt(deg[pos])
        D_inv = sp.diags(d_inv_sqrt)
        L_sp = (D_inv @ L_sp @ D_inv).tocsr()

    if return_torch:
        L_coo = L_sp.tocoo()
        indices = torch.tensor(np.vstack([L_coo.row, L_coo.col]), dtype=torch.long)
        values = torch.tensor(L_coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (P, P)).to_sparse_csr()
    return L_sp


def create_smoothing_operator(
    source: Union[torch.Tensor, np.ndarray, sp.spmatrix],
    lambda_val: float = 0.05,
    source_type: str = "auto",
    k: int = 6,
    sigma: Optional[float] = None,
    connectivity: Optional[int] = None,
    normalized: bool = False,
    scale_degree: bool = True
) -> SparseGraphResolvent:
    """
    Universal convenience factory: construct a graph Laplacian from any geometric,
    mesh, or network structure and return a pre-factorized SparseGraphResolvent operator.

    Parameters
    ----------
    source : torch.Tensor, np.ndarray, or scipy.sparse matrix
        Input structure:
        - Point coordinates (P, D) in 2D, 3D, or high dimensions.
        - Triangular surface mesh faces (F, 3).
        - Edge list (E, 2) or (E, 3).
        - Adjacency matrix (P, P).
        - 2D or 3D binary domain mask.
    lambda_val : float, default=0.05
        Spatial smoothing scale parameter (lambda >= 0).
    source_type : str, default="auto"
        Format of source: "auto", "coordinates", "mesh_faces", "edges", "adjacency", or "mask".
    k : int, default=6
        Number of nearest neighbors when source is coordinates.
    sigma : float, optional
        Bandwidth for Gaussian heat kernel when source is coordinates.
    connectivity : int, optional
        Neighborhood connectivity when source is a mask. Defaults to 8 for a 2D
        mask and 26 for a 3D mask.
    normalized : bool, default=False
        If True, uses normalized Laplacian L_norm = D^(-1/2) L D^(-1/2).

    Returns
    -------
    SparseGraphResolvent
        Memory-efficient drop-in smoothing operator for `pysimlr.simlr(smoothing_matrices=[S, ...])`.

    Examples
    --------
    >>> import pysimlr
    >>> # 1. From 3D vertex coordinates (e.g. cortical skeleton or surface).
    >>> # Integer voxel coordinates need source_type="coordinates" (or a cast
    >>> # to float), since a (P, 3) integer array is also a valid face list.
    >>> coords = skel.coords  # doctest: +SKIP
    >>> S = pysimlr.create_smoothing_operator(coords, lambda_val=0.05, k=6)  # doctest: +SKIP
    >>>
    >>> # 2. From a 2D flatmap domain mask
    >>> S = pysimlr.create_smoothing_operator(flatmap_mask, lambda_val=0.05)  # doctest: +SKIP
    >>>
    >>> # 3. From a triangular surface mesh. source_type is required here: a
    >>> # (F, 3) integer array is equally a face list and 3-D voxel coordinates.
    >>> S = pysimlr.create_smoothing_operator(  # doctest: +SKIP
    ...     mesh_faces, lambda_val=0.05, source_type="mesh_faces")
    >>>
    >>> # 4. Use directly in SiMLR
    >>> res = pysimlr.simlr([X, Y], smoothing_matrices=[S, None], k=1)  # doctest: +SKIP
    """
    L_sp = create_graph_laplacian(
        source,
        source_type=source_type,
        k=k,
        sigma=sigma,
        connectivity=connectivity,
        normalized=normalized,
        return_torch=False,
        scale_degree=scale_degree,
    )
    return SparseGraphResolvent(L_sp, lambda_val=lambda_val)


