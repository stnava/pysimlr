import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from .svd import ba_svd, safe_pca
from .optimizers import create_optimizer
from .sparsification import orthogonalize_and_q_sparsify, simlr_sparseness, NSA_DEFAULT_W
from .utils import (set_seed_based_on_time, adjusted_rvcoef, safe_svd,
                    invariant_orthogonality_defect, l1_normalize_features, orthogonality_defect,
                    gradient_orthogonality_defect,
                    orthogonality_summary, preprocess_data)
from .consensus import compute_shared_consensus

def parse_constraint(constraint_str: str) -> Dict[str, Any]:
    """
    Parse a constraint string into type, weight, and iterations.

    The constraint string follows the format "Type[xWeight[xIterations]]",
    e.g., "Stiefel", "Stiefel x 0.5", or "Stiefel x 0.5 x 5".

    Parameters
    ----------
    constraint_str : str
        The constraint string to parse.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys:
        - "type": The name of the constraint (e.g., "Stiefel", "Grassmann").
        - "weight": The constraint weight (float).
        - "iterations": The number of projection iterations (int).

    Raises
    ------
    TypeError
        If the input is not a string.
    """
    parts = constraint_str.split('x')
    constraint_type = parts[0].strip()

    # Default weight depends on type. Every soft-orthogonality branch in
    # simlr_sparseness is gated on `weight > 0`, so defaulting the "ortho" and
    # "nsaflow" families to 0.0 made a bare constraint="ortho" apply no
    # constraint whatsoever -- and zero the orthogonality penalty in simlr's
    # energy as well. An explicit "orthox0" still selects no constraint.
    if constraint_type in ("Stiefel", "Grassmann", "Stiefel_ns", "Grassmann_ns",
                           "Stiefel_polar", "Grassmann_polar", "NewtonSchulz"):
        weight = 1.0
    elif constraint_type in ("ortho", "nsaflow", "ortho_ns", "nsaflow_ns",
                             "ortho_polar", "nsaflow_polar"):
        weight = 0.1
    elif constraint_type == "none":
        weight = 0.0
    else:
        import warnings
        warnings.warn(
            f"Unrecognised constraint type {constraint_type!r} in "
            f"{constraint_str!r}; no manifold constraint will be applied. "
            f"Known types: Stiefel, Grassmann, NewtonSchulz, ortho, nsaflow "
            f"(optionally suffixed _ns or _polar), none.",
            UserWarning, stacklevel=2,
        )
        weight = 0.0

    iterations = 1
    if len(parts) > 1:
        try: weight = float(parts[1])
        except ValueError: pass
    if len(parts) > 2:
        try: iterations = int(parts[2])
        except ValueError: pass
    return {"type": constraint_type, "weight": weight, "iterations": iterations}

def project_gradient(v_grad: torch.Tensor, v_current: torch.Tensor, constraint_type: str) -> torch.Tensor:
    """
    Project the gradient onto the tangent space of a specific manifold.

    This ensures that optimization updates remain valid with respect to 
    manifold constraints (e.g., Stiefel or Grassmann manifolds), following 
    the theory in Edelman et al. (1998).

    Parameters
    ----------
    v_grad : torch.Tensor
        The raw gradient of the energy function.
    v_current : torch.Tensor
        The current value of the basis matrix (point on the manifold).
    constraint_type : str
        The type of manifold/constraint ("Stiefel", "Grassmann", etc.).

    Returns
    -------
    torch.Tensor
        The projected gradient.

    Raises
    ------
    TypeError
        If inputs are not tensors.
    """
    if constraint_type == "Grassmann":
        # Project onto Grassmann tangent space: G = G - V(V^T G)
        return v_grad - v_current @ (v_current.t() @ v_grad)
    elif constraint_type == "Stiefel":
        # Project onto Stiefel tangent space: G = G - V sym(V^T G)
        vtg = v_current.t() @ v_grad
        sym_vtg = 0.5 * (vtg + vtg.t())
        return v_grad - v_current @ sym_vtg
    return v_grad

def calculate_u(projections: List[torch.Tensor], 
                mixing_algorithm: str = "svd", 
                k: Optional[int] = None,
                orthogonalize: bool = False) -> torch.Tensor:
    """
    Deprecated: Use compute_shared_consensus from .consensus instead.

    Parameters
    ----------
    projections : List[torch.Tensor]
        List of projected data matrices (N x K).
    mixing_algorithm : str, default="svd"
        The algorithm used to mix projections ("svd", "pca", "ica", "avg", "newton").
        The algorithm to use for consensus computation.
    k : int, optional
        Target rank for the shared latent space.
    orthogonalize : bool, default=False
        Whether to orthogonalize the resulting consensus matrix.

    Returns
    -------
    torch.Tensor
        The shared consensus latent matrix (N x K).

    Raises
    ------
    TypeError
        If projections is not a list of tensors.
    """
    return compute_shared_consensus(projections, mixing_algorithm, k, orthogonalize)

def initialize_simlr(data_matrices: List[torch.Tensor],
                     k: int,
                     initialization_type: str = "pca",
                     joint_reduction: bool = True,
                     positivity: str = "either") -> List[torch.Tensor]:
    """
    Initialize the basis matrices (V) for SiMLR.

    Provides various initialization strategies, primarily based on SVD/PCA,
    to ensure the optimization starts from a reasonable point.

    Parameters
    ----------
    data_matrices : List[torch.Tensor]
        List of input data matrices (N x P_i).
    k : int
        Target rank for the latent space.
    initialization_type : str, default="pca"
        Strategy for initialization. Only "pca" (truncated SVD of each view) is
        implemented; any other value raises.
    joint_reduction : bool, default=True
        Reserved for a future joint-reduction initializer and currently
        ignored. Previously both of these arguments were accepted and silently
        discarded, so callers could believe they had selected a strategy that
        did not exist.
    positivity : str, default="either"
        When this requests a non-negative basis and the NSA-Flow backend is
        installed, the basis is fitted directly from the data instead of being
        taken from the signed PCA loadings.

    Returns
    -------
    List[torch.Tensor]
        Initial basis matrices (P_i x K) for each view.

    Raises
    ------
    TypeError
        If input matrices are not tensors.

    Notes
    -----
    Under a non-negative constraint the signed PCA loadings are not a usable
    starting point, and the fix is not to rectify them. `simlr_sparseness`
    applies the positivity constraint by reflection, so a signed
    initialization reached the first retraction as ``abs(V_pca)``: a matrix
    whose orthogonality has been destroyed outright (normalized Stiefel defect
    1.60, against 0.00 for the signed loadings it came from), and the anchored
    retraction then stays near that damaged target rather than repairing it.

    Fitting the basis from ``X`` with `nsa_flow_data` avoids the rectification
    entirely. Measured over 10 seeds against a known non-negative basis, in
    recovery and in defect:

    ==============================  ==========  ==========  ==========  ======
    candidate                       disjoint    defect      overlapping defect
    ==============================  ==========  ==========  ==========  ======
    ``abs(PCA)`` then retract       0.9162      0.62        0.8112      0.72
    signed PCA then retract         0.7451      0.43        0.7691      0.46
    ``nsa_flow_data(X)``            **0.9945**  **0.05**    **0.9032**  0.13
    ==============================  ==========  ==========  ==========  ======

    Note that feeding the *signed* loadings to the non-negative solver is worse
    than rectifying them, not better: the solve is anchored, so a signed target
    pushes entries to zero wherever the anchor is negative rather than where
    the data says the basis is small. Solving from the data is what removes the
    problem. The synthetic truth used above is itself non-negative, which
    flatters a non-negative fit; the defect column does not depend on that,
    since ``abs(PCA)`` simply is not orthogonal.
    """
    if initialization_type != "pca":
        raise ValueError(
            f"initialization_type={initialization_type!r} is not implemented; "
            f"only 'pca' is supported."
        )
    return [initial_basis_for_view(x, k, positivity=positivity)
            for x in data_matrices]


def initial_basis_for_view(x: torch.Tensor, k: int,
                           positivity: str = "either") -> torch.Tensor:
    """
    Build the initial ``(features, k)`` basis for one view.

    Shared by `initialize_simlr` and the deep models' ``initialize_v`` so that
    every entry point starts from the same basis for a given constraint.

    Under a non-negative constraint the basis is fitted from ``x`` with
    `nsa_flow_data` when the backend is available. Otherwise it is the signed
    PCA loadings, with each column's sign resolved so that its sum is positive
    -- a sign flip preserves orthogonality exactly, unlike the rectification
    that a non-negative constraint would otherwise apply downstream.

    The deep encoders need this as much as the linear model does: they rectify
    at use time (`LENDNSAEncoder.v` clamps negatives) rather than at init, so a
    signed start reaches the first forward pass as ``clamp(V_pca)``, which is
    the same loss of orthogonality as ``abs(V_pca)`` by another route.

    Parameters
    ----------
    x : torch.Tensor
        The view, shape (samples, features).
    k : int
        Target rank.
    positivity : str, default="either"
        When non-negative, fit from the data rather than rectifying PCA.

    Returns
    -------
    torch.Tensor
        Basis of shape (features, k) in `x`'s dtype.
    """
    from .nsa_backend import load_nsa_flow_data
    from .sparsification import NSA_DEFAULT_W, _usable_retraction

    want_nonneg = positivity in ('positive', 'hard', 'nonnegative', 'nonneg',
                                 'softplus')
    if want_nonneg and x.shape[1] > k:
        fit_from_data = load_nsa_flow_data()
        if fit_from_data is not None:
            fitted = _nonnegative_basis_from_data(
                fit_from_data, x, k, NSA_DEFAULT_W, _usable_retraction)
            if fitted is not None:
                return fitted.to(x.dtype)

    u, s, v = ba_svd(x, nu=0, nv=k)
    if v.shape[1] < k:
        padding = torch.randn(v.shape[0], k - v.shape[1],
                              dtype=v.dtype, device=v.device) * 1e-4
        v = torch.cat([v, padding], dim=1)
    if want_nonneg:
        # Resolve each column's sign so downstream rectification removes as
        # little as possible. This is a sign flip, not a rectification, so the
        # basis stays exactly as orthogonal as PCA made it.
        for j in range(v.shape[1]):
            if v[:, j].sum() < 0:
                v[:, j] *= -1
    return v.to(x.dtype)


def _nonnegative_basis_from_data(fit_from_data, x: torch.Tensor, k: int,
                                 w: float, usable) -> Optional[torch.Tensor]:
    """
    Fit a non-negative, near-orthogonal basis for `x` directly from the data.

    Returns None if the backend raises or returns something unusable, so the
    caller falls back to the PCA initializer rather than starting from a
    degenerate basis. A wide view (``p <= k``) is left to PCA: no basis with
    more columns than rows can be orthogonal, so there is nothing to fit.
    """
    reference = torch.ones(x.shape[1], k, dtype=torch.float64)
    try:
        result = fit_from_data(x.detach().double(), k=k, w=float(w))
    except Exception:
        return None
    candidate = None
    if hasattr(result, 'get'):
        candidate = result.get('V') or result.get('Y')
    if candidate is None:
        candidate = getattr(result, 'V', None) or getattr(result, 'Y', None)
    if candidate is None:
        return None
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    candidate = torch.clamp_min(candidate, 0.0)
    if not usable(candidate, reference):
        return None
    return candidate


def nsa_contrast_transform(x: Union[torch.Tensor, np.ndarray],
                           k: int = 6,
                           w: float = 0.5,
                           consolidate: bool = True,
                           optimizer: str = "torch_lbfgs",
                           max_iter: Optional[int] = None,
                           tol: Optional[float] = None) -> Dict[str, Any]:
    """
    Fit an NSA-Flow Signed Contrast representation (Recipe B / Guide v2.11.0+).

    Lifts standardized feature profiles into non-overlapping positive and negative
    lobes (V = V^+ - V^-) with strictly disjoint supports when `consolidate=True`.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input data matrix of shape (n_samples, n_features).
    k : int, default=6
        Number of components to extract.
    w : float, default=0.5
        Trade-off weight in [0, 1]. w=0 maximizes reconstruction fidelity,
        w=1 maximizes orthogonality (disjoint supports).
    consolidate : bool, default=True
        Guarantees strictly disjoint supports (zero lobe overlap).
    optimizer : str, default="torch_lbfgs"
        Optimization algorithm ("torch_lbfgs", "spg", or "lbfgs").
    max_iter : int, optional
        Maximum iterations cap.
    tol : float, optional
        Stationarity tolerance.

    Returns
    -------
    dict
        Dictionary containing:
        - 'v': Component loading matrix [features, k]
        - 'scores': Projection scores [samples, k]
        - 'result': Underlying NSAResult object
    """
    from .nsa_backend import load_nsa_flow
    fn = load_nsa_flow()
    if fn is None:
        raise RuntimeError("NSA-Flow is not installed. Install with `pip install nsa-flow`.")
    x_t = torch.as_tensor(x, dtype=torch.float32)
    res = fn(x_t, k=k, w=w, mode="signed", consolidate=consolidate,
             optimizer=optimizer, max_iter=max_iter, tol=tol)
    v = getattr(res, 'V', None)
    if v is None:
        v = getattr(res, 'Y', None)
    if v is None and hasattr(res, 'get'):
        v = res.get('V') if res.get('V') is not None else res.get('Y')
    scores = x_t @ v
    return {
        "v": v,
        "scores": scores,
        "result": res,
    }


def nsa_nonnegative_transform(x: Union[torch.Tensor, np.ndarray],
                              k: int = 6,
                              w: float = 0.5,
                              optimizer: str = "torch_lbfgs",
                              max_iter: Optional[int] = None,
                              tol: Optional[float] = None) -> Dict[str, Any]:
    """
    Fit an NSA-Flow Non-Negative representation on physical quantities (Recipe A).

    Extracts physically realizable non-negative constituent spectra without negative
    loadings, maintaining frame defect D ≈ 0. Do not mean-center input features.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input data matrix of shape (n_samples, n_features), where x >= 0.
    k : int, default=6
        Number of components to extract.
    w : float, default=0.5
        Trade-off weight in [0, 1].
    optimizer : str, default="torch_lbfgs"
        Optimization algorithm ("torch_lbfgs", "spg", or "lbfgs").
    max_iter : int, optional
        Maximum iterations cap.
    tol : float, optional
        Stationarity tolerance.

    Returns
    -------
    dict
        Dictionary containing:
        - 'v': Non-negative constituent loading matrix [features, k]
        - 'scores': Projection scores [samples, k]
        - 'result': Underlying NSAResult object
    """
    from .nsa_backend import load_nsa_flow
    fn = load_nsa_flow()
    if fn is None:
        raise RuntimeError("NSA-Flow is not installed. Install with `pip install nsa-flow`.")
    x_t = torch.as_tensor(x, dtype=torch.float32)
    res = fn(x_t, k=k, w=w, mode="data", nonneg=True,
             optimizer=optimizer, max_iter=max_iter, tol=tol)
    v = getattr(res, 'V', None)
    if v is None:
        v = getattr(res, 'Y', None)
    if v is None and hasattr(res, 'get'):
        v = res.get('V') if res.get('V') is not None else res.get('Y')
    if v is not None:
        v = torch.clamp_min(v, 0.0)
    scores = x_t @ v
    return {
        "v": v,
        "scores": scores,
        "result": res,
    }

def calculate_ica_energy(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nonlinearity: str = "logcosh", a: float = 1.0) -> torch.Tensor:
    """
    Compute the ICA-based energy (negentropy) for a given projection.

    Measures how non-Gaussian the projected data is, which is a common 
    objective in Independent Component Analysis.

    Parameters
    ----------
    x : torch.Tensor
        Input data matrix (N x P).
    u : torch.Tensor
        Shared latent space (N x K).
    v : torch.Tensor
        Basis matrix (P x K).
    nonlinearity : str, default="logcosh"
        The contrast function to use. Options: 'logcosh', 'exp', 'gauss', 'kurtosis'.
    a : float, default=1.0
        Hyperparameter for the contrast function (used by 'gauss').

    Returns
    -------
    torch.Tensor
        The computed energy value (scalar).

    Raises
    ------
    TypeError
        If inputs are not valid tensors.
    """
    s = (u.t() @ x) @ v
    n = x.shape[0]
    if nonlinearity == "logcosh":
        abs_s = torch.abs(s)
        return -torch.sum(abs_s - np.log(2.0) + torch.log1p(torch.exp(-2.0 * abs_s))) / n
    elif nonlinearity == "exp": return -torch.sum(-torch.exp(-s**2 / 2.0)) / n
    elif nonlinearity == "gauss": return -torch.sum(-0.5 * torch.exp(-a * s**2)) / n
    elif nonlinearity == "kurtosis": return -torch.sum((s**4.0) / 4.0) / n
    return torch.tensor(0.0, dtype=x.dtype, device=x.device)

def calculate_ica_gradient(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nonlinearity: str = "logcosh", a: float = 1.0) -> torch.Tensor:
    """
    Compute the gradient of the ICA-based energy with respect to the basis matrix V.

    Used by gradient-based optimizers to update the projection weights towards 
    maximum non-Gaussianity.

    Parameters
    ----------
    x : torch.Tensor
        Input data matrix (N x P).
    u : torch.Tensor
        Shared latent space (N x K).
    v : torch.Tensor
        Basis matrix (P x K).
    nonlinearity : str, default="logcosh"
        The contrast function used.
    a : float, default=1.0
        Hyperparameter for the contrast function.

    Returns
    -------
    torch.Tensor
        The gradient matrix (P x K).

    Raises
    ------
    TypeError
        If inputs are not valid tensors.
    """
    s = (u.t() @ x) @ v
    # Must match the normalizer used by calculate_ica_energy, which divides by
    # the sample count. This previously divided by s.shape[0] == k, so the
    # gradient was off by a factor of n/k relative to the energy it belongs to
    # -- enough to invalidate any Armijo sufficient-decrease test.
    n = x.shape[0]
    if nonlinearity == "logcosh": return (1.0 / n) * (x.t() @ u @ torch.tanh(s))
    elif nonlinearity == "exp": return (1.0 / n) * (x.t() @ u @ (s * torch.exp(-s**2 / 2.0)))
    elif nonlinearity == "gauss": return (1.0 / n) * (x.t() @ u @ (a * s * torch.exp(-a * s**2)))
    elif nonlinearity == "kurtosis": return (1.0 / n) * (x.t() @ u @ (s**3))
    return torch.zeros_like(v)

#: Objectives with both an energy and a matching gradient. Anything else is
#: refused by both, rather than silently evaluating to zero.
SUPPORTED_ENERGY_TYPES = frozenset({
    "regression", "acc", "logcosh", "exp", "gauss", "kurtosis",
    "nc", "normalized_correlation", "dat",
})


def calculate_simlr_energy(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, energy_type: str = "regression", lambda_val: float = 0.0, prior_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the energy (loss) for a single modality in SiMLR.

    Supports various objective functions including reconstruction error, 
    canonical correlation (ACC), ICA-based negentropy, and domain-knowledge 
    alignment.

    Parameters
    ----------
    v : torch.Tensor
        Basis matrix (P x K) for the current modality.
    x : torch.Tensor
        Input data matrix (N x P) for the current modality.
    u : torch.Tensor
        Shared latent space (N x K).
    energy_type : str, default="regression"
        The objective function to use. Options:
        - 'regression': Reconstruction MSE (||X - UV^T||^2).
        - 'acc': Maximum absolute covariance (canonical correlation).
        - 'logcosh', 'exp', 'gauss', 'kurtosis': ICA-based negentropy.
        - 'nc' or 'normalized_correlation': Cosine similarity.
        - 'dat': Alignment with domain knowledge `prior_matrix`.
    lambda_val : float, default=0.0
        Scaling factor for 'dat' energy.
    prior_matrix : Optional[torch.Tensor], default=None
        Domain knowledge matrix (K x P) for 'dat' energy.

    Returns
    -------
    torch.Tensor
        The computed energy value (scalar).

    Raises
    ------
    TypeError
        If inputs are not valid tensors.
    """
    from .similarity import (SimilarityContext, resolve_energy_name,
                             similarity_energy)
    u = u.to(x.dtype); v = v.to(x.dtype)
    if energy_type == "dat":
        # Domain regulariser, not a similarity: it scores V against a prior
        # matrix rather than against the shared latent, so it stays here.
        if prior_matrix is None:
            raise ValueError(
                "energy_type='dat' is a domain regulariser and requires "
                "prior_matrix. It used to return a constant 0.0, so a caller "
                "who passed it as a primary objective optimised nothing."
            )
        alignment = prior_matrix.to(x.dtype) @ v
        return -lambda_val * torch.sum(alignment ** 2)
    name = resolve_energy_name(energy_type, path="linear")
    return similarity_energy(name, x @ v, u, SimilarityContext(x=x, v=v))

def calculate_simlr_gradient(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, 
                             energy_type: str = "regression", lambda_val: float = 0.0, 
                             prior_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate the gradient of the SiMLR energy function for a single view.

    Parameters
    ----------
    v : torch.Tensor
        The view-specific basis matrix (p_i x k).
    x : torch.Tensor
        The view-specific data matrix (n x p_i).
    u : torch.Tensor
        The shared consensus latent matrix (n x k).
    energy_type : str, default="regression"
        The type of energy function used for gradient calculation.
        Options: "regression", "acc", "logcosh", "exp", "gauss", "kurtosis", "dat".
    lambda_val : float, default=0.0
        Regularization parameter for "dat" (Directed Alignment Transfer) energy.
    prior_matrix : torch.Tensor, optional
        Prior alignment matrix for "dat" energy.

    Returns
    -------
    torch.Tensor
        The computed gradient (p_i x k).

    Raises
    ------
    TypeError
        If inputs are not valid tensors.
    """
    from .similarity import (SimilarityContext, resolve_energy_name,
                             similarity_gradient)
    u = u.to(x.dtype); v = v.to(x.dtype)
    if energy_type == "dat":
        if prior_matrix is None:
            raise ValueError(
                "energy_type='dat' is a domain regulariser and requires "
                "prior_matrix."
            )
        prior_matrix = prior_matrix.to(x.dtype)
        return 2 * lambda_val * (prior_matrix.t() @ prior_matrix @ v)

    name = resolve_energy_name(energy_type, path="linear")
    ctx = SimilarityContext(x=x, v=v)
    s_rep = x @ v
    if name == "recon":
        # `recon` depends on V directly, not only through s = XV.
        g_v = similarity_gradient(name, s_rep, u, ctx, wrt="v")
    else:
        # Chain rule through the linear encoder: dE/dV = X' dE/ds.
        g_v = x.t() @ similarity_gradient(name, s_rep, u, ctx, wrt="s")
    # Historical convention, preserved: this function returns a *descent*
    # direction (-dE/dV), not the gradient. Callers add it to V.
    return -g_v

def simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
          k: int,
          iterations: int = 100,
          optimizer_type: str = "lars",
          energy_type: str = "acc",
          constraint: str = "orthox0.1x1",
          mixing_algorithm: str = "svd",
          sparseness_quantile: float = 0.0,
          positivity: str = "either",
          smoothing_matrices: Optional[List[torch.Tensor]] = None,
          domain_matrices: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
          domain_lambdas: Optional[Union[float, List[float]]] = None,
          orthogonalize_u: bool = False,
          topology: str = "star",
          path_graph: Optional[Dict[int, List[int]]] = None,
          scale_list: List[str] = ["centerAndScale", "np"],
          consolidate: bool = False,
          use_nsa: bool = True,
          nsa_w: Optional[float] = None,
          tol: float = 1e-6,
          verbose: bool = False,
          **opt_params) -> Dict[str, Any]:
    """
    Perform Similarity-driven Multi-view Linear Representation (SiMLR).

    SiMLR identifies a shared latent subspace across multiple data modalities
    by optimizing an energy function subject to manifold constraints and sparsity.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    k : int
        The dimensionality of the shared latent space.
    iterations : int, default=100
        Maximum number of optimization iterations.
    optimizer_type : str, default="lars"
        The optimizer to use.
    energy_type : str, default="acc"
        The similarity/reconstruction objective to minimize.
    constraint : str, default="Stiefel"
        The manifold constraint on basis matrices ("Stiefel", "Grassmann", "NewtonSchulz").

    mixing_algorithm : str, default="svd"
        The algorithm used to mix projections ("svd", "pca", "ica", "avg", "newton").

    sparseness_quantile : float, default=0.5
        Proportion of weights to shrink towards zero.
    positivity : str, default="either"
        Sign constraint on the basis ("either", "positive", "negative").
    smoothing_matrices : List[torch.Tensor], optional
        Spatial/prior smoothing operators for each modality.
    domain_matrices : List[Union[torch.Tensor, np.ndarray]], optional
        Matrices to align with for directed domain knowledge.
    domain_lambdas : Union[float, List[float]], optional
        Weight(s) for the domain knowledge alignment objective.
    orthogonalize_u : bool, default=False
        Whether to enforce orthogonality on the consensus matrix U.
    topology : str, default="star"
        The consensus topology ("star", "loo", "graph").
    path_graph : Dict[int, List[int]], optional
        Adjacency list for "graph" topology.
    scale_list : List[str], default=["centerAndScale", "np"]
        Preprocessing methods to apply to input data.
    tol : float, default=1e-6
        Convergence tolerance for the optimization.
    verbose : bool, default=False
        Whether to print convergence details.
    **opt_params : dict
        Additional parameters to pass to the optimizer or constraint function.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the fitted model parts:
        - "u": Shared latent consensus (N x K).
        - "v": List of view-specific basis matrices (P_i x K).
        - "w": Reconstruction mapping matrices.
        - "energy": The total energy after each iteration.
        - "converged_iter": The iteration at which convergence occurred.
        - "best_energy": The lowest total energy reached.
        - "best_iteration": Index into "energy" of that lowest value. The
          returned "v" and "u" come from this iterate, because alternating
          minimization is not monotone in the joint objective -- `u` is
          recomputed after each sweep over the views, so the energy typically
          drops steeply and then drifts slowly upward.

    Raises
    ------
    TypeError
        If the inputs are not valid data structures.
    """
    if 'sparsity' in opt_params:
        sparseness_quantile = opt_params.pop('sparsity')
    if 'sparseness' in opt_params:
        sparseness_quantile = opt_params.pop('sparseness')

    # Contract Validation: data_matrices structure
    if not isinstance(data_matrices, (list, tuple)) or len(data_matrices) == 0:
        raise ValueError("data_matrices must be a non-empty list or tuple of matrices/arrays.")

    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")

    torch_mats = []
    for idx, m in enumerate(data_matrices):
        if not isinstance(m, (torch.Tensor, np.ndarray)):
            raise TypeError(f"View {idx} must be a torch.Tensor or np.ndarray, got {type(m).__name__}.")
        m_t = torch.as_tensor(m).float()
        if m_t.ndim != 2:
            raise ValueError(
                f"Each view in data_matrices must be 2D (samples x features). "
                f"View {idx} has {m_t.ndim} dimension(s) with shape {tuple(m_t.shape)}."
            )
        if m_t.shape[0] == 0:
            raise ValueError(f"View {idx} has 0 samples.")
        if m_t.shape[1] == 0:
            raise ValueError(f"View {idx} has 0 features.")
        torch_mats.append(m_t)

    # Contract Validation: sample count consistency across all views
    n_samples_list = [m.shape[0] for m in torch_mats]
    if len(set(n_samples_list)) > 1:
        details = ", ".join(f"view {i}: {n}" for i, n in enumerate(n_samples_list))
        raise ValueError(
            f"All data matrices must have identical number of samples (rows), "
            f"but found mismatched sample counts: {details}."
        )
    
    provenance_list = []
    if scale_list is not None and len(scale_list) > 0 and scale_list[0] != "none":
        scaled_mats = []
        for m in torch_mats:
            m_scaled, prov = preprocess_data(m, scale_list)
            scaled_mats.append(m_scaled)
            provenance_list.append(prov)
        torch_mats = scaled_mats
        
    n_modalities = len(torch_mats)
    orig_dtype = torch_mats[0].dtype
    v_mats = initialize_simlr(torch_mats, k, positivity=positivity)
    opt_params.setdefault("use_nsa", use_nsa)
    # One weight.  `constraint` sets the prox weight; the "nsa_flow" optimizer's
    # intermediate soft retraction used to default to a DIFFERENT weight (0.5
    # against 0.1), so one call applied two different operators.  Unless the
    # caller separates them explicitly, they agree.  (parse_constraint is pure;
    # the full parse below is unchanged.)
    if nsa_w is None:
        _w0 = parse_constraint(constraint)["weight"]
        nsa_w = _w0 if _w0 > 0 else NSA_DEFAULT_W
    opt_params.setdefault("nsa_w", nsa_w)
    optimizer = create_optimizer(optimizer_type, v_mats, **opt_params)
    
    constraint_info = parse_constraint(constraint)
    constraint_type = constraint_info["type"]
    constraint_weight = constraint_info["weight"]
    constraint_iterations = constraint_info["iterations"]
    
    torch_domains = [torch.as_tensor(dm).float() if dm is not None else None for dm in domain_matrices] if domain_matrices else None
    if domain_lambdas is not None and not isinstance(domain_lambdas, (list, tuple)):
        # Accept any scalar, not just float; `domain_lambdas=1` used to reach
        # `domain_lambdas[i]` and raise "'int' object is not subscriptable".
        domain_lambdas = [float(domain_lambdas)] * n_modalities
    elif isinstance(domain_lambdas, (list, tuple)):
        domain_lambdas = list(domain_lambdas)
        if len(domain_lambdas) != n_modalities:
            raise ValueError(
                f"domain_lambdas has {len(domain_lambdas)} entries but there are "
                f"{n_modalities} views."
            )
    if torch_domains is not None and domain_lambdas is None:
        domain_lambdas = [1.0] * n_modalities
    
    from .nsa_backend import load_gradient_mapping
    _grad_mapping = load_gradient_mapping()
    # The feasible set the certificate projects onto must match the one the
    # iterate actually lives in, or the measure is of a different problem.
    _want_nonneg = positivity in ('positive', 'hard', 'nonnegative', 'nonneg',
                                  'softplus')
    _cert_proj = (lambda z: torch.clamp(z, min=0.0)) if _want_nonneg else None

    energy_history = []
    grad_map_history = []
    n_degenerate_iterates = 0
    stop_reason = "max_iter"
    certificate = None
    prev_total_energy = float('inf')
    prev_grad_map = float('inf')
    converged_iter = iterations
    # Alternating minimization is not monotone in the joint objective: each V_i
    # is optimized against a fixed u_i, and then u is recomputed from the new
    # projections, which can raise the total. In practice the energy drops
    # sharply over the first few iterations and then drifts slowly upward, so
    # the final iterate is not the best one found. Keep the best.
    best_total_energy = float('inf')
    best_v_mats = None
    # The solver's self-report for each modality's final projection. Kept
    # alongside the best iterate so the reported diagnostics describe the basis
    # actually returned, not whichever iteration the loop happened to stop on.
    retraction_diags = [{} for _ in range(n_modalities)]
    best_retraction_diags = None
    
    normalizing_weights = [1.0] * n_modalities
    orth_weights = [1.0] * n_modalities
    domain_weights = [1.0] * n_modalities
    
    prev_resid = [None] * n_modalities   # last prox-gradient fixed-point residual per view
    for it in range(iterations):
        cert_terms = []
        projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
        u = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u, topology=topology, path_graph=path_graph)
        # Capture the linear map this consensus used, so that
        # `predict_shared_latent` can apply it to new data instead of
        # deriving a fresh one. See `_capture_consensus_anchor`.
        consensus_anchor = _capture_consensus_anchor(
            projections, mixing_algorithm, k, orthogonalize_u, topology, path_graph)
        for i in range(n_modalities):
            u_i = u[i] if isinstance(u, list) else u
            # Local energy function that incorporates sparsification/retraction
            # Single definition of "the feasible point corresponding to v", so
            # the energy and the gradient are evaluated at the *same* place.
            # Previously the energy was measured after sparsification/retraction
            # while the gradient was taken at the raw iterate, which left every
            # Armijo/backtracking optimizer testing a sufficient-decrease
            # condition against the slope of a different function.
            def to_feasible(v_cand):
                return simlr_sparseness(
                    v_cand.to(orig_dtype), constraint_type=constraint_type,
                    smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None,
                    positivity=positivity, sparseness_quantile=sparseness_quantile,
                    constraint_weight=constraint_weight, constraint_iterations=constraint_iterations,
                    energy_type=energy_type, modality_index=i)

            def smooth_energy_fn(v_cand):
                v_sp = to_feasible(v_cand)
                sim_e = calculate_simlr_energy(v_sp, torch_mats[i], u_i, energy_type) * normalizing_weights[i]
                dom_e = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_e = calculate_simlr_energy(v_sp, torch_mats[i], u_i, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]) * domain_weights[i]
                orth_e = 0.0
                if constraint_type == "ortho": orth_e = orthogonality_defect(v_sp) * constraint_weight * orth_weights[i]
                return (sim_e + dom_e + orth_e).item()

            # Local gradient function that also incorporates manifold projection
            raw_grad_holder = [None]

            def smooth_gradient_fn(v_curr):
                # Evaluate at the feasible point the energy also uses.
                v_feas = to_feasible(v_curr)

                sim_grad = calculate_simlr_gradient(v_feas, torch_mats[i], u_i, energy_type) * normalizing_weights[i]
                dom_grad = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_grad = calculate_simlr_gradient(v_feas, torch_mats[i], u_i, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]) * domain_weights[i]
                
                # The energy `smooth_energy_fn` also carries an orthogonality
                # term, so omitting it here made energy and gradient describe
                # two different functions. Any line search or curvature
                # estimate built from the pair was then inconsistent, and the
                # stationarity certificate certified the wrong objective. The
                # mismatch was survivable only while the fidelity gradient was
                # itself ~(n-1) times too small; correcting that (see the
                # `regression` branch of `calculate_simlr_gradient`) made the
                # inconsistency dominate and drove `energy_reduction` to zero.
                orth_grad = 0.0
                if constraint_type == "ortho" and constraint_weight != 0.0:
                    orth_grad = (gradient_orthogonality_defect(v_feas)
                                 * constraint_weight * orth_weights[i])

                total_grad = sim_grad + dom_grad - orth_grad
                # Keep the analytic gradient before it is projected onto the
                # tangent space and retracted. The retraction orthonormalises
                # the direction and so throws its magnitude away, which is
                # correct for a search direction and useless for a certificate:
                # scoring the retracted direction returns a constant (3.0000 on
                # the 3-view case, independent of how far from stationary the
                # iterate is). The certificate has to see the real gradient.
                raw_grad_holder[0] = total_grad.detach().clone()
                # Project gradient onto tangent space
                total_grad = project_gradient(total_grad, v_feas, constraint_type)
                
                # The search direction is NOT retracted.  Proximal gradient is
                # `prox(v + eta * g)`: one operator, applied to the point, after
                # the step.  The direction used to be passed through
                # simlr_sparseness too (orthonormalising it and discarding its
                # magnitude); CORRECTNESS_AUDIT.md measured that as changing
                # latent recovery by < 0.0004, i.e. an extra solve per view per
                # iteration that bought nothing, and it is not part of any
                # proximal-gradient method.
                return total_grad.contiguous()

            total_grad = smooth_gradient_fn(v_mats[i])
            # Stationarity certificate at the iterate entering this sweep,
            # using the gradient that was going to be computed anyway. See
            # `nsa_backend.load_gradient_mapping`: energy plateau is not
            # stationarity, and this loop previously claimed convergence with
            # nothing measuring the latter.
            # Stationarity certificate for THE PROBLEM BEING SOLVED.  This loop
            # is proximal gradient on E + w*Dtilde over the feasible set, whose
            # fixed points satisfy v = prox(v + eta*d(v)).  The right residual is
            # therefore the relative fixed-point residual ||v+ - v|| / ||v||,
            # computed after the step and prox below.  The gradient mapping of
            # E alone under the sign projection -- what used to be certified --
            # is NOT zero at that fixed point (it settled at 6.2 on a problem
            # the iteration had converged on) and is kept only as a diagnostic.
            gm_E = None
            if _grad_mapping is not None and raw_grad_holder[0] is not None:
                try:
                    gm_E = float(_grad_mapping(v_mats[i], raw_grad_holder[0],
                                               _cert_proj))
                except Exception:
                    gm_E = None
            gm_i = prev_resid[i]
            # A view certified stationary takes no step and is not re-projected.
            # Proximal gradient has nothing to do at a stationary point, and
            # doing it anyway is not free: the prox is solved to `tol`, so
            # re-applying it to its own output moves the basis by O(tol) per
            # sweep -- measured as a 5e-5 drift in the overall energy over 12
            # sweeps from an already-optimal initialisation, which the
            # first-vs-last monotonicity tests caught.  The first sweep always
            # projects, so a raw initialisation never bypasses the constraint.
            if gm_i is not None and gm_i <= tol and it > 0:
                cert_terms.append(gm_i)
                retraction_diags[i] = {"skipped": "stationary", "fixed_point_residual": gm_i,
                                       "grad_map_E": gm_E}
                continue
            v_before = v_mats[i].detach().clone()
            # Optimizers that re-evaluate along their own line search (LBFGS)
            # need to recompute the analytic gradient at each trial point.
            if hasattr(optimizer, "gradient_function"):
                optimizer.gradient_function = smooth_gradient_fn
            v_updated = optimizer.step(i, v_mats[i], total_grad, smooth_energy_fn)
            
            # Apply final projection
            retraction_diags[i] = {}
            v_mats[i] = simlr_sparseness(v_updated, constraint_type=constraint_type, smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None, positivity=positivity, sparseness_quantile=sparseness_quantile, constraint_weight=constraint_weight, constraint_iterations=constraint_iterations, energy_type=energy_type, modality_index=i, retraction_diagnostics=retraction_diags[i])
            resid = float(torch.linalg.norm(v_mats[i] - v_before)
                          / torch.linalg.norm(v_before).clamp_min(1e-300))
            prev_resid[i] = resid
            cert_terms.append(resid)
            retraction_diags[i]["fixed_point_residual"] = resid
            retraction_diags[i]["grad_map_E"] = gm_E
            
        if it == 0:
            for i in range(n_modalities):
                u_i = u[i] if isinstance(u, list) else u
                sim_e = calculate_simlr_energy(v_mats[i], torch_mats[i], u_i, energy_type).item()
                normalizing_weights[i] = 1.0 / (abs(sim_e) * n_modalities + 1e-10)
                orth_e = orthogonality_defect(v_mats[i]).item()
                if orth_e > 1e-10: orth_weights[i] = abs(sim_e) * normalizing_weights[i] / orth_e
                else: orth_weights[i] = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_e_raw = calculate_simlr_energy(v_mats[i], torch_mats[i], u_i, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]).item()
                    if abs(dom_e_raw) > 1e-10: domain_weights[i] = abs(sim_e * normalizing_weights[i]) / abs(dom_e_raw)
                    else: domain_weights[i] = 1.0
        
        total_energy = 0.0
        for i in range(n_modalities):
            u_i = u[i] if isinstance(u, list) else u
            total_energy += calculate_simlr_energy(v_mats[i], torch_mats[i], u_i, energy_type).item() * normalizing_weights[i]
            
        energy_history.append(total_energy)

        # A modality whose basis is identically zero contributes zero energy,
        # so the cheapest way to reduce a *sum* of per-view energies is to kill
        # a view. Selecting the lowest-energy iterate therefore selects the
        # degenerate one: on the 3-view case the loop returned V with view 0
        # entirely zero at total energy 2/3 (exactly one of three views
        # surviving), and reported it as `best_iteration=1`.
        #
        # A rank-deficient basis is not a solution to a rank-k problem, so such
        # an iterate is not eligible to be "best" however low its energy.
        iterate_is_degenerate = any(
            not bool((v.detach().abs() > 1e-12).any(dim=0).all()) for v in v_mats
        )
        if total_energy < best_total_energy and not iterate_is_degenerate:
            best_total_energy = total_energy
            best_v_mats = [v.clone() for v in v_mats]
            best_retraction_diags = [dict(d) for d in retraction_diags]
        if iterate_is_degenerate:
            n_degenerate_iterates += 1
        
        # Stopping. An energy plateau is not stationarity: the total energy is
        # normalised to 1.0 at iteration 0, so a loop that is barely descending
        # produces relative changes below `tol` within two or three sweeps and
        # the old test then declared convergence. Measured on the 3-view case
        # it fired at iteration 3 for every problem and every `iterations`,
        # so the argument never bound; with the test disabled one seed carried
        # on improving from R^2 0.300 to 0.366.
        #
        # `grad_map` is the certificate: dimensionless, scale invariant, and
        # zero exactly at a stationary point of the constrained problem. Only
        # it may claim convergence. A plateau still stops the loop -- there is
        # no point burning sweeps that do nothing -- but it is certified only
        # when the certificate is also within a wide band of stationarity,
        # following nsa_flow's `CERTIFICATES` contract.
        this_grad_map = max(cert_terms) if cert_terms else float("nan")
        grad_map_history.append(this_grad_map)

        energy_stalled = abs(prev_total_energy - total_energy) < tol * (
            abs(prev_total_energy) + 1e-10)

        if cert_terms and this_grad_map <= tol:
            stop_reason, certificate = "grad_map", "stationary"
            converged_iter = it + 1
            if verbose: print(f"Stationary at iteration {it}: grad_map {this_grad_map:.3e}")
            break

        cert_stalled = (
            not cert_terms
            or abs(prev_grad_map - this_grad_map) < tol * (abs(prev_grad_map) + 1e-10)
        )
        if energy_stalled and cert_stalled:
            stop_reason = "plateau"
            # Weaker than stationary, and only claimed near it.
            certificate = ("numerical_floor"
                           if cert_terms and this_grad_map <= 1e3 * tol else None)
            converged_iter = it + 1
            if verbose:
                print(f"Plateau at iteration {it}: energy {total_energy}, "
                      f"grad_map {this_grad_map:.3e}, certificate {certificate}")
            break

        prev_total_energy = total_energy
        prev_grad_map = this_grad_map
        
        if verbose and it % 10 == 0: print(f"Iteration {it}: Total Energy {total_energy}")
        
    # Return the lowest-energy iterate rather than whichever one the loop
    # happened to stop on.
    if best_v_mats is not None:
        v_mats = best_v_mats
        if best_retraction_diags is not None:
            retraction_diags = best_retraction_diags

    if consolidate:
        from .nsa_backend import load_consolidate_supports
        cons_fn = load_consolidate_supports()
        if cons_fn is not None:
            consolidated_v = []
            for v_mat in v_mats:
                if positivity == "positive":
                    v_c = cons_fn(v_mat.double()).to(orig_dtype)
                    v_c = torch.nn.functional.normalize(v_c, p=2, dim=0, eps=1e-8)
                    consolidated_v.append(v_c)
                else:
                    k_v = v_mat.shape[1]
                    v_p = torch.clamp_min(v_mat, 0.0)
                    v_n = torch.clamp_min(-v_mat, 0.0)
                    W = torch.cat([v_p, v_n], dim=1)
                    W_c = cons_fn(W.double()).to(orig_dtype)
                    v_c = W_c[:, :k_v] - W_c[:, k_v:]
                    v_c = torch.nn.functional.normalize(v_c, p=2, dim=0, eps=1e-8)
                    consolidated_v.append(v_c)
            v_mats = consolidated_v

    # Re-calculate final shared consensus after the last V update
    projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
    u = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u, topology=topology, path_graph=path_graph)
    # Capture the linear map this consensus used, so that
    # `predict_shared_latent` can apply it to new data instead of
    # deriving a fresh one. See `_capture_consensus_anchor`.
    consensus_anchor = _capture_consensus_anchor(
        projections, mixing_algorithm, k, orthogonalize_u, topology, path_graph)
    
    v_summaries = [orthogonality_summary(v) for v in v_mats]
    
    # Compute reconstruction weights W_i such that X_i approx U @ W_i
    # W_i = pinv(U) @ X_i
    w_mats = []
    for i, x in enumerate(torch_mats):
        u_i = u[i] if isinstance(u, list) else u
        try:
            u_pinv = torch.linalg.pinv(u_i)
            w_mats.append(u_pinv @ x)
        except Exception:
            # Fallback if pinv fails
            w_mats.append(torch.zeros(k, x.shape[1], dtype=u_i.dtype, device=u_i.device))

    return {
        "u": u, "v": v_mats, "w": w_mats, "energy": energy_history, 
        "normalizing_weights": normalizing_weights, 
        "orth_weights": orth_weights, "domain_weights": domain_weights, 
        "converged_iter": converged_iter,
        # The stopping claim, and what backs it. `converged` is True only with
        # a certificate; `stop_reason` says why iteration ceased either way,
        # and `energy_reduction` answers "did this optimise anything at all",
        # which a bare `converged_iter` never could.
        "stop_reason": stop_reason,
        "certificate": certificate,
        "converged": certificate is not None,
        "grad_map": (grad_map_history[-1] if grad_map_history else None),
        "grad_map_history": grad_map_history,
        # How many sweeps produced a rank-deficient basis. Nonzero means the
        # objective is rewarding collapse and the result should be read with
        # that in mind, even though such iterates can no longer be returned.
        "n_degenerate_iterates": n_degenerate_iterates,
        "energy_start": (energy_history[0] if energy_history else None),
        "energy_reduction": ((energy_history[0] - min(energy_history))
                             if energy_history else None),
        "v_orthogonality": v_summaries,
        "best_energy": best_total_energy,
        "best_iteration": (int(np.argmin(energy_history)) if energy_history else None),
        # What the retraction solver reported for each modality: the stopping
        # rule, the stationarity certificate, the effective rank, the scale
        # drift and which fidelity it chose. Empty when no backend ran.
        "v_retraction": retraction_diags,
        # The consensus map fitted on the training projections. Required for
        # out-of-sample prediction: without it `predict_shared_latent`
        # re-derives the basis on the new data, and for svd/pca/ica that basis
        # is defined only up to rotation, sign and permutation.
        "consensus_anchor": consensus_anchor,
        "mixing_algorithm": mixing_algorithm,
        "orthogonalize_u": orthogonalize_u,
        "topology": topology,
        "path_graph": path_graph,
        "energy_type": energy_type,
        "scale_list": scale_list,
        "provenance_list": provenance_list,
        "consolidated": consolidate
    }


def pairwise_matrix_similarity(mat_list: List[torch.Tensor], v_list: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute pairwise similarity (Adjusted RV Coefficient) between all pairs of modalities.

    Parameters
    ----------
    mat_list : List[torch.Tensor]
        List of data matrices (N x P_i).
    v_list : List[torch.Tensor]
        List of basis matrices (P_i x K).

    Returns
    -------
    Dict[str, float]
        Dictionary where keys are "sim_i_j" and values are the similarity scores.

    Raises
    ------
    TypeError
        If inputs are not valid lists of tensors.
    """
    n_modalities = len(mat_list); similarities = {}
    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            l_i = mat_list[i] @ v_list[i]; l_j = mat_list[j] @ v_list[j]
            similarities[f"sim_{i}_{j}"] = adjusted_rvcoef(l_i, l_j)
    return similarities

def simlr_perm(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, n_perms: int = 50, verbose: bool = False, **simlr_params) -> Dict[str, Any]:
    """
    Perform permutation testing for SiMLR to assess the significance of shared latent structures.

    This function runs the SiMLR algorithm on the original data and then repeatedly on 
    permuted versions of the data (where rows of each modality are independently shuffled)
    to build a null distribution of the cross-modality similarity (ACC).

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        A list of data matrices (one for each modality).
    k : int
        The number of shared latent components to extract.
    n_perms : int, optional
        The number of permutations to perform (default is 50).
    verbose : bool, optional
        Whether to print progress during the SiMLR optimization (default is False).
    **simlr_params : dict
        Additional parameters passed to the `simlr` function (e.g., energy_type, constraint, optimizer_type).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "simlr_result": The result of SiMLR on the original data.
        - "stats": Per-modality-pair permutation statistics, keyed "sim_i_j",
          each a dict with "observed", "p_value", "n_permutations",
          "n_exceeding", "null_mean", "null_sd", "z_score" and
          "null_distribution".
        - "n_permutations": The requested number of permutations.

    Notes
    -----
    The p-value is the add-one permutation estimate
    ``(1 + #{null >= observed}) / (1 + n_perms)`` (Phipson & Smyth, 2010).

    Earlier revisions reported ``scipy.stats.ttest_1samp(null, observed,
    alternative='less')``, which tests whether the *mean* of the null lies below
    the observed value rather than how far into the null's upper tail the
    observation falls. That statistic assumes a normally distributed null,
    ignores the tail entirely, and shrinks without bound as `n_perms` grows, so
    it reported arbitrarily small p-values for data with no shared structure.
    The `t_stat` key it produced has been replaced by `z_score`, a plain
    standardized effect size that makes no distributional claim.

    Raises
    ------
    TypeError
        If the inputs are of an invalid type.
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]

    def _fit_and_score(mats):
        """Fit SiMLR and score pairwise similarity on the same matrices the fit saw."""
        fit = simlr(mats, k=k, verbose=False, **simlr_params)
        # The basis V is learned on preprocessed data, so the similarity must be
        # evaluated on preprocessed data too. Scoring raw matrices against a V
        # fitted on centered/scaled ones mixes two different feature scalings.
        scale_list = fit.get('scale_list') or []
        prov_list = fit.get('provenance_list') or []
        if scale_list and scale_list[0] != "none" and len(prov_list) == len(mats):
            scored = [preprocess_data(m, scale_list, provenance=pv)
                      for m, pv in zip(mats, prov_list)]
        else:
            scored = mats
        v_norm = [l1_normalize_features(v) for v in fit['v']]
        return fit, pairwise_matrix_similarity(scored, v_norm)

    res, obs_sims = _fit_and_score(torch_mats)
    if verbose:
        print(f"Observed pairwise similarities: {obs_sims}")

    null_results = {key: [] for key in obs_sims}
    for p in range(n_perms):
        if verbose and p % 10 == 0:
            print(f"Permutation {p}/{n_perms}")
        # Independently shuffling rows of every view destroys cross-view
        # correspondence while preserving each view's marginal distribution.
        mats_perm = [m[torch.randperm(m.shape[0])] for m in torch_mats]
        _, sims_p = _fit_and_score(mats_perm)
        for k_sim, v_sim in sims_p.items():
            null_results[k_sim].append(v_sim)

    stats = {}
    for k_sim, obs in obs_sims.items():
        null_dist = np.asarray(null_results[k_sim], dtype=float)
        null_dist = null_dist[np.isfinite(null_dist)]
        n_eff = int(null_dist.size)
        if n_eff == 0:
            stats[k_sim] = {"observed": obs, "p_value": float('nan'),
                            "n_permutations": 0, "null_mean": float('nan'),
                            "null_sd": float('nan'), "z_score": float('nan'),
                            "null_distribution": []}
            continue
        # Add-one (Phipson & Smyth 2010) permutation p-value: the observed
        # statistic is itself one realisation under the null, so the estimate is
        # (1 + #{null >= obs}) / (1 + n_perm). This is bounded below by
        # 1/(1 + n_perm) and never returns an impossible p of exactly 0.
        n_exceed = int(np.sum(null_dist >= obs))
        p_val = (1.0 + n_exceed) / (1.0 + n_eff)
        null_mean = float(null_dist.mean())
        null_sd = float(null_dist.std(ddof=1)) if n_eff > 1 else 0.0
        if null_sd > 0.0:
            z_score = (float(obs) - null_mean) / null_sd
        else:
            z_score = float('inf') if float(obs) > null_mean else 0.0
        stats[k_sim] = {
            "observed": float(obs),
            "p_value": float(p_val),
            "n_permutations": n_eff,
            "n_exceeding": n_exceed,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "z_score": float(z_score),
            "null_distribution": null_dist.tolist(),
        }
    return {"simlr_result": res, "stats": stats, "n_permutations": n_perms}

def _capture_consensus_anchor(projections: List[torch.Tensor],
                              mixing_algorithm: str,
                              k: int,
                              orthogonalize_u: bool,
                              topology: str,
                              path_graph: Optional[Dict[int, List[int]]]) -> Optional[torch.Tensor]:
    """
    Return the linear map the consensus used, or None if it has none.

    `compute_shared_consensus` returns ``(u, anchor)`` when asked in training
    mode. The anchor takes the concatenated projections to the shared latent,
    and it is what makes the consensus reproducible on data it was not fitted
    on.

    Only ``svd``, ``pca`` and ``ica`` have such a map, and only those need one:
    they derive a basis whose rotation, column signs and ordering are
    arbitrary, so re-deriving it on a new sample yields axes that need not
    correspond to the ones a downstream model was fitted against. ``avg`` and
    ``newton`` are fixed functions of their input and return None here.

    Returns None rather than raising if no anchor is available, since the fit
    itself does not depend on one.
    """
    try:
        result = compute_shared_consensus(
            projections, mixing_algorithm=mixing_algorithm, k=k,
            orthogonalize=orthogonalize_u, training=True,
            topology=topology, path_graph=path_graph)
    except Exception:
        return None
    if not isinstance(result, tuple) or len(result) != 2:
        return None
    anchor = result[1]
    return anchor.detach().clone() if isinstance(anchor, torch.Tensor) else None


def predict_shared_latent(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                          simlr_result: Dict[str, Any]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Compute the shared latent basis U for new data using the trained SIMLR model.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of new data matrices (one for each modality).
    simlr_result : Dict[str, Any]
        The result dictionary from a previous `simlr` call.

    Returns
    -------
    Union[torch.Tensor, List[torch.Tensor]]
        The shared consensus latent matrix (N x K) or list of matrices for the new data.

    Raises
    ------
    TypeError
        If inputs are of invalid type.
    """
    # 1. Contract Validation
    if not isinstance(data_matrices, (list, tuple)) or len(data_matrices) == 0:
        raise ValueError("data_matrices must be a non-empty list or tuple of matrices/arrays.")

    v_mats = simlr_result.get('v')
    if v_mats is not None and len(data_matrices) != len(v_mats):
        raise ValueError(
            f"Expected {len(v_mats)} data matrices matching model views, "
            f"but received {len(data_matrices)}."
        )

    for idx, m in enumerate(data_matrices):
        m_t = torch.as_tensor(m)
        if m_t.ndim != 2:
            raise ValueError(
                f"Each view in data_matrices must be 2D (samples x features). "
                f"View {idx} has {m_t.ndim} dimension(s) with shape {tuple(m_t.shape)}."
            )

    n_samples_list = [torch.as_tensor(m).shape[0] for m in data_matrices]
    if len(set(n_samples_list)) > 1:
        details = ", ".join(f"view {i}: {n}" for i, n in enumerate(n_samples_list))
        raise ValueError(
            f"All data matrices must have identical number of samples (rows), "
            f"but found mismatched sample counts: {details}."
        )

    # 2. Preprocess data matrices
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    scale_list = simlr_result.get('scale_list', [])
    provenance_list = simlr_result.get('provenance_list', [])
    if scale_list and scale_list[0] != "none":
        scaled_mats = []
        for i, m in enumerate(torch_mats):
            prov = provenance_list[i] if i < len(provenance_list) else None
            m_scaled = preprocess_data(m, scale_list, provenance=prov)
            scaled_mats.append(m_scaled)
        torch_mats = scaled_mats

    v_mats = simlr_result['v']
    mixing_alg = simlr_result.get('mixing_algorithm', 'svd')
    orthogonalize_u = simlr_result.get('orthogonalize_u', False)
    topology = simlr_result.get('topology', 'star')
    path_graph = simlr_result.get('path_graph', None)
    k = v_mats[0].shape[1]
    
    # 2. Project to shared space
    projections = [x @ v.to(x.dtype) for x, v in zip(torch_mats, v_mats)]
    
    # 3. Compute consensus U using the original mixing settings
    # Apply the consensus map fitted at training time rather than deriving a
    # fresh one. For svd/pca/ica the derived basis is arbitrary up to rotation,
    # sign and permutation, so re-deriving it here placed held-out samples in
    # different axes than the training latent: on Diabetes a regression fitted
    # on the training U scored -0.67 out of sample, against +0.40 through the
    # fixed map X @ V. `compute_shared_consensus` reuses the anchor only when
    # `training=False` and the algorithm is one of the volatile ones.
    anchor = simlr_result.get('consensus_anchor')
    u_new = compute_shared_consensus(projections, mixing_algorithm=mixing_alg, k=k, orthogonalize=orthogonalize_u, training=False, anchor=anchor, topology=topology, path_graph=path_graph)
    return u_new

def reconstruct_from_learned_maps(u: Union[torch.Tensor, List[torch.Tensor]], 
                                  simlr_result: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Reconstruct all data matrices (modalities) from the shared latent basis U.

    Parameters
    ----------
    u : Union[torch.Tensor, List[torch.Tensor]]
        The shared latent matrix (N x K) or list of matrices.
    simlr_result : Dict[str, Any]
        The result dictionary from a previous `simlr` call, which must 
        contain the learned reconstruction weights "w".

    Returns
    -------
    List[torch.Tensor]
        List of reconstructed data matrices (one for each modality).

    Raises
    ------
    TypeError
        If inputs are of invalid type.
    """
    if 'w' not in simlr_result:
        # For backward compatibility, but this should be avoided
        return []
    
    w_mats = simlr_result['w']
    reconstructions = []
    for i, w in enumerate(w_mats):
        u_i = u[i] if isinstance(u, list) else u
        x_pred = u_i @ w.to(u_i.dtype)
        reconstructions.append(x_pred)
    return reconstructions

def predict_simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                  simlr_result: Dict[str, Any],
                  allow_legacy_refit: bool = False) -> Dict[str, Any]:
    """
    Predict using a trained SiMLR model on new data matrices.

    Generates the shared latent representation `U` and, optionally, the 
    reconstructed data inputs from the new modalities based on the learned 
    model mappings.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of new data matrices to predict on.
    simlr_result : Dict[str, Any]
        The output dictionary from a previous `simlr` fit.
    allow_legacy_refit : bool, default=False
        Whether to allow least-squares estimation of reconstructions 
        if learned weights 'w' are missing.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "u": Predicted shared latent representation.
        - "reconstructions": Reconstructed input matrices.
        - "errors": Normalized Frobenius reconstruction error per modality.

    Raises
    ------
    ValueError
        If learned weights are missing and legacy refit is not allowed.
    TypeError
        If the inputs are not valid formats.
    """
    # 1. Contract Validation
    if not isinstance(data_matrices, (list, tuple)) or len(data_matrices) == 0:
        raise ValueError("data_matrices must be a non-empty list or tuple of matrices/arrays.")

    v_mats = simlr_result.get('v')
    if v_mats is not None and len(data_matrices) != len(v_mats):
        raise ValueError(
            f"Expected {len(v_mats)} data matrices matching model views, "
            f"but received {len(data_matrices)}."
        )

    for idx, m in enumerate(data_matrices):
        m_t = torch.as_tensor(m)
        if m_t.ndim != 2:
            raise ValueError(
                f"Each view in data_matrices must be 2D (samples x features). "
                f"View {idx} has {m_t.ndim} dimension(s) with shape {tuple(m_t.shape)}."
            )

    n_samples_list = [torch.as_tensor(m).shape[0] for m in data_matrices]
    if len(set(n_samples_list)) > 1:
        details = ", ".join(f"view {i}: {n}" for i, n in enumerate(n_samples_list))
        raise ValueError(
            f"All data matrices must have identical number of samples (rows), "
            f"but found mismatched sample counts: {details}."
        )

    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    
    scale_list = simlr_result.get('scale_list', [])
    provenance_list = simlr_result.get('provenance_list', [])
    if scale_list and scale_list[0] != "none":
        scaled_mats = []
        for i, m in enumerate(torch_mats):
            prov = provenance_list[i] if i < len(provenance_list) else None
            m_scaled = preprocess_data(m, scale_list, provenance=prov)
            scaled_mats.append(m_scaled)
        torch_mats = scaled_mats

    if 'model' in simlr_result:
        # Fallback to predict_deep logic if a deep model is detected
        from .deep import predict_deep
        return predict_deep(data_matrices, simlr_result)
    
    # For standard SIMLR:
    u_new = predict_shared_latent(data_matrices, simlr_result)
    
    if 'w' in simlr_result:
        reconstructions = reconstruct_from_learned_maps(u_new, simlr_result)
    elif allow_legacy_refit:
        # Fallback to old suspicious least-squares behavior for backward compatibility
        reconstructions = []
        v_mats = simlr_result['v']
        mixing_alg = simlr_result.get('mixing_algorithm', 'svd')
        orthogonalize_u = simlr_result.get('orthogonalize_u', False)
        for i, x in enumerate(torch_mats):
            u_i = u_new[i] if isinstance(u_new, list) else u_new
            u_ortho = orthogonalize_u or (mixing_alg in ["svd", "pca"])
            if u_ortho:
                weights = u_i.t() @ x
                x_pred = u_i @ weights
            else:
                u_pinv = torch.linalg.pinv(u_i)
                weights = u_pinv @ x
                x_pred = u_i @ weights
            reconstructions.append(x_pred)
    else:
        raise ValueError("Learned reconstruction weights 'w' missing from simlr_result. "
                         "Legacy refit is disabled by default. Set allow_legacy_refit=True to enable.")
    
    errors = []
    for i, x in enumerate(torch_mats):
        x_pred = reconstructions[i]
        err = torch.norm(x - x_pred, p='fro').item() / (torch.norm(x, p='fro').item() + 1e-10)
        errors.append(err)
        
    latents = [x @ v.to(x.dtype) for x, v in zip(torch_mats, simlr_result.get("v", []))]
    return {"u": u_new, "latents": latents, "reconstructions": reconstructions, "errors": errors}


def estimate_rank(data_matrices: List[Union[torch.Tensor, np.ndarray]], n_permutations: int = 20, var_threshold: float = 0.99) -> int:
    """
    Estimate the optimal shared rank `k` across multiple data modalities.

    Uses a heuristic approach relying on the singular value spectrum and 
    cross-modality alignment (RV coefficient) to suggest the number of 
    shared latent components, potentially augmented by permutation testing.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    n_permutations : int, default=20
        Number of random permutations for building a null distribution. 
        If 0, skips permutation testing and uses a fast heuristic.
    var_threshold : float, default=0.99
        Cumulative variance threshold to bound the maximum searched rank.

    Returns
    -------
    int
        The estimated optimal rank `k`.

    Raises
    ------
    TypeError
        If input types are invalid.
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    n_modalities = len(torch_mats); k_max_list = []
    for x in torch_mats:
        x_centered = x - torch.mean(x, dim=0); _, s, _ = safe_svd(x_centered, full_matrices=False)
        eigenvalues = s**2; prop_var = torch.cumsum(eigenvalues, dim=0) / (torch.sum(eigenvalues) + 1e-10)
        reached = torch.where(prop_var >= var_threshold)[0]
        # A degenerate view (near-zero total variance) never crosses the
        # threshold because of the 1e-10 floor in the denominator, and indexing
        # the empty result raised IndexError. Fall back to the full spectrum.
        k_max_list.append(int(reached[0].item()) + 1 if reached.numel() else int(s.numel()))
    k_max = min(k_max_list) if k_max_list else 1
    if k_max < 1: k_max = 1
    def calculate_rv_curve(mats, km):
        u_list = [safe_svd(m, full_matrices=False)[0][:, :km] for m in mats]
        scores = []
        for curr_k in range(1, km + 1):
            mod_scores = []
            for i in range(n_modalities):
                y_target = u_list[i][:, :curr_k]; other_inds = [j for j in range(n_modalities) if j != i]
                u_other = torch.cat([u_list[j][:, :curr_k] for j in other_inds], dim=1)
                consensus, _, _ = safe_svd(u_other, full_matrices=False)
                consensus = consensus[:, :curr_k]; mod_scores.append(adjusted_rvcoef(y_target, consensus))
            scores.append(np.mean(mod_scores))
        return scores
    proc_mats = [(m - torch.mean(m, dim=0)) / (torch.norm(m - torch.mean(m, dim=0), p='fro') + 1e-10) for m in torch_mats]
    real_curve = calculate_rv_curve(proc_mats, k_max)
    if n_permutations > 0 and n_modalities >= 2:
        null_curves = []
        for _ in range(n_permutations):
            perm_mats = [proc_mats[0]] + [m[torch.randperm(m.shape[0])] for m in proc_mats[1:]]
            null_curves.append(calculate_rv_curve(perm_mats, k_max))
        null_curve_mean = np.mean(null_curves, axis=0); signal = np.array(real_curve) - null_curve_mean; optimal_k = np.argmax(signal) + 1
    else:
        if len(real_curve) < 3: optimal_k = 1
        else:
            y = np.array(real_curve); x_vals = np.linspace(0, 1, len(y)); y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
            optimal_k = np.argmax(y_norm - x_vals) + 1
    return int(optimal_k)

def decompose_energy(data_matrices: List[Union[torch.Tensor, np.ndarray]], simlr_result: Dict[str, Any], energy_type: str = "acc") -> Dict[str, Any]:
    """
    Decompose the SiMLR objective energy across modalities and features.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    simlr_result : Dict[str, Any]
        The result dictionary from a fitted SiMLR model.
    energy_type : str, default="acc"
        The energy function type to evaluate.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "modality_energies": List of energy values for each modality.
        - "feature_importances": List of gradient-based importance arrays 
          per modality feature.

    Raises
    ------
    TypeError
        If input types are invalid.
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]

    # V and U were fitted on preprocessed data, so the energy has to be
    # evaluated on preprocessed data too. Skipping this made the returned
    # decomposition inconsistent with the model's own energies.
    scale_list = simlr_result.get('scale_list') or []
    prov_list = simlr_result.get('provenance_list') or []
    if scale_list and scale_list[0] != "none" and len(prov_list) == len(torch_mats):
        torch_mats = [preprocess_data(m, scale_list, provenance=pv)
                      for m, pv in zip(torch_mats, prov_list)]

    u_all = simlr_result['u']; v_mats = simlr_result['v']
    modality_energies = []; feature_importances = []
    for i, (x, v) in enumerate(zip(torch_mats, v_mats)):
        u = u_all[i] if isinstance(u_all, list) else u_all
        mod_energy = calculate_simlr_energy(v, x, u, energy_type).item()
        grad = calculate_simlr_gradient(v, x, u, energy_type)
        # .detach().cpu() is required: a bare .numpy() raises on CUDA/MPS.
        feat_imp = torch.sum(torch.abs(grad), dim=1).detach().cpu().numpy()
        modality_energies.append(mod_energy); feature_importances.append(feat_imp)
    return {"modality_energies": modality_energies, "feature_importances": feature_importances}
