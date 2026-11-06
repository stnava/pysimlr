import torch
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

def _planted_latent(n_samples: int, shared_k: int, private_scale: float,
                    scales, weights=None):
    """Shared latent and outcome for the planted generators (global RNG).

    With view-private background (`private_scale > 0`) the latent columns have
    standard deviations `scales` and the outcome is their `weights`-weighted
    sum (all ones when None) plus N(0, 0.15^2) noise. Without it the outcome
    is N(0, 1) and is added, doubled, to the first latent column.
    """
    if private_scale > 0.0:
        true_u = torch.randn(n_samples, shared_k) * torch.tensor(scales[:shared_k])
        w = torch.ones(shared_k) if weights is None else torch.tensor(weights[:shared_k])
        return true_u, (true_u @ w) + torch.randn(n_samples) * 0.15
    outcome_signal = torch.randn(n_samples)
    true_u = torch.randn(n_samples, shared_k)
    true_u[:, 0] += outcome_signal * 2.0
    return true_u, outcome_signal


def _sparse_signed_basis(shared_k: int, p: int, true_sparsity: float,
                         generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Construct a (shared_k, p) signed basis with exact per-component sparsity.

    ``generator`` (a ``torch.Generator``) draws from a private stream; ``None`` uses the
    global stream, which the older builders seed themselves."""
    if true_sparsity <= 0.0:
        return torch.randn(shared_k, p, generator=generator)
    active_frac = max(1.0 / p, min(1.0, 1.0 - float(true_sparsity)))
    m = max(1, min(p, int(round(active_frac * p))))
    v = torch.zeros(shared_k, p)
    order = torch.randperm(p, generator=generator)
    stride = max(1, p // shared_k)
    for j in range(shared_k):
        idx = torch.arange(j * stride, j * stride + m) % p
        cols = order[idx]
        mag = 0.5 + torch.rand(len(cols), generator=generator)
        signs = torch.where(torch.rand(len(cols), generator=generator) > 0.5, 1.0, -1.0)
        v[j, cols] = mag * signs
    return v


def build_linear_footprint_case(n_samples: int = 1000, 
                                shared_k: int = 3, 
                                p_list: Optional[List[int]] = None,
                                noise_scale: float = 0.5,
                                seed: int = 42,
                                **kwargs) -> Dict[str, Any]:
    """
    Generate a purely linear synthetic dataset for benchmarking SiMLR.

    Creates data where modalities are linear projections of a common latent 
    space (U), with an embedded predictive signal.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate.
    shared_k : int, default=3
        The rank of the shared latent space.
    p_list : List[int], optional
        Features per view; None gives [100, 80, 60].
        The dimensionality of each generated modality.
    noise_scale : float, default=0.5
        Standard deviation of Gaussian noise added to each modality.
    seed : int, default=42
        Random seed for reproducibility.
    **kwargs : Dict[str, Any]

    Returns
    -------
    Dict[str, Any]
        Standardized case dictionary containing data matrices and ground truth.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    private_scale = float(kwargs.get("private_scale", 0.0))
    true_sparsity = float(kwargs.get("true_sparsity", 0.5 if private_scale > 0.0 else 0.0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Clean linear multi-view regime: shared components are stronger than
    # view-private background (private_scale=0.5), with equal outcome weights.
    p_list = [100, 80, 60] if p_list is None else p_list
    true_u, outcome_signal = _planted_latent(n_samples, shared_k, private_scale, [1.2, 1.0, 0.85], None)
    
    data_matrices = []
    v_true = []
    
    for idx_v, p in enumerate(p_list):
        v = _sparse_signed_basis(shared_k, p, true_sparsity)
        x = true_u @ v + torch.randn(n_samples, p) * noise_scale
        if private_scale > 0.0:
            u_priv = torch.randn(n_samples, 2) * private_scale
            v_priv = torch.randn(2, p)
            x = x + u_priv @ v_priv
        data_matrices.append(x)
        v_true.append(v.t())
        
    return {
        "kind": "linear",
        "data": data_matrices,
        "true_u": true_u,
        "true_v": v_true,
        "outcome": outcome_signal,
        "shared_k": shared_k
    }

def build_nonlinear_shared_case(n_samples: int = 1000, 
                                shared_k: int = 3, 
                                p_list: Optional[List[int]] = None,
                                noise_scale: float = 0.5,
                                seed: int = 42,
                                regime: str = "mixed",
                                **kwargs) -> Dict[str, Any]:
    """
    Generate a nonlinear synthetic dataset for benchmarking deep SiMR.

    Creates data where modalities are nonlinear transformations of a common 
    latent space (U). This is used to test the capacity of LEND and NED to 
    recover latents that linear SiMLR cannot.
    """
    private_scale = float(kwargs.get("private_scale", 0.0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if p_list is None:
        p_list = [120, 80, 50] if private_scale > 0.0 else [100, 80, 60]
    true_u, outcome_signal = _planted_latent(n_samples, shared_k, private_scale, [1.15, 0.95, 0.80], [0.9, 1.0, 1.1])
    
    data_matrices = []
    v_true = [] # Approximate linear basis
    
    for i, p in enumerate(p_list):
        # Base linear mapping
        v = torch.randn(shared_k, p)
        v_true.append(v.t())
        
        if private_scale > 0.0:
            # Non-cancelling latent-space manifold warping z_m = psi_m(true_u) projected via v
            z_m = true_u.clone()
            u_roll = torch.roll(true_u, 1, dims=1)
            if regime == "polynomial":
                if i == 0:
                    z_m = true_u + 0.30 * (true_u ** 2 - 1.0) + 0.15 * (true_u * u_roll)
                elif i == 1:
                    z_m = true_u + 0.25 * (u_roll ** 2 - 1.0) + 0.20 * (true_u ** 3) / 3.0
                else:
                    z_m = 0.90 * true_u + 0.25 * (true_u ** 2 - 1.0) + 0.15 * (true_u ** 3) / 3.0
            elif regime == "sinusoidal":
                if i == 0:
                    z_m = 0.70 * true_u + 0.45 * torch.sin(1.6 * true_u) + 0.20 * torch.cos(1.3 * u_roll)
                elif i == 1:
                    z_m = 0.70 * true_u + 0.40 * torch.sin(1.4 * true_u + 0.35 * u_roll)
                else:
                    z_m = 0.70 * true_u + 0.35 * torch.sin(1.5 * true_u) + 0.25 * (torch.cos(1.4 * true_u) - 0.5)
            else:
                z_m = true_u
            x = z_m @ v
        else:
            x_lin = true_u @ v
            if regime == "mixed":
                if i == 0:
                    x = x_lin
                elif i == 1:
                    x = 0.5 * x_lin + 0.5 * ((true_u**2) @ torch.randn(shared_k, p))
                else:
                    x = 0.5 * x_lin + 0.5 * (torch.sin(true_u * 2.0) @ torch.randn(shared_k, p))
            elif regime == "polynomial":
                x = x_lin + 0.4 * (x_lin**2)
            elif regime == "sinusoidal":
                x = 0.65 * x_lin + 0.65 * torch.sin(2.0 * x_lin)
            else:
                x = x_lin
            
        x += torch.randn(n_samples, p) * noise_scale
        if private_scale > 0.0:
            u_priv = torch.randn(n_samples, 2) * private_scale
            v_priv = torch.randn(2, p)
            x = x + u_priv @ v_priv
        data_matrices.append(x)
    
    return {
        "kind": f"nonlinear_{regime}",
        "data": data_matrices,
        "true_u": true_u,
        "true_v": v_true,
        "outcome": outcome_signal,
        "shared_k": shared_k,
        "regime": regime
    }

def build_shared_plus_private_case(n_samples: int = 1000, 
                                   shared_k: Optional[int] = None, 
                                   private_k: int = 2,
                                   p_list: Optional[List[int]] = None,
                                   noise_scale: float = 0.5,
                                   seed: int = 42,
                                   **kwargs) -> Dict[str, Any]:
    """
    Generate synthetic data with both shared and private latent components.
    """
    has_explicit_priv = "private_scale" in kwargs
    # Regime defaults apply only to parameters the caller left unset; these
    # used to overwrite explicit `shared_k` and `p_list` whenever
    # `private_scale` was passed.
    if shared_k is None:
        shared_k = 3 if has_explicit_priv else 2
    if p_list is None:
        p_list = [160, 60, 30] if has_explicit_priv else [100, 100]
    private_scale = float(kwargs.get("private_scale", 1.0))
    true_sparsity = float(kwargs.get("true_sparsity", 0.8 if has_explicit_priv else 0.0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if has_explicit_priv:
        scales = torch.tensor([1.0 / (1.0 + 0.6 * i) for i in range(shared_k)])
        true_u_shared = torch.randn(n_samples, shared_k) * scales
        weights = torch.linspace(0.6, 1.4, shared_k)
        outcome_signal = (true_u_shared @ weights) + torch.randn(n_samples) * 0.15
    else:
        outcome_signal = torch.randn(n_samples)
        true_u_shared = torch.randn(n_samples, shared_k)
        true_u_shared[:, 0] += outcome_signal * 2.0
    
    data_matrices = []
    private_latents = []
    v_true = []
    
    for idx_v, p in enumerate(p_list):
        priv_mult = 1.6 if (has_explicit_priv and idx_v == 0) else 1.0
        true_u_private = torch.randn(n_samples, private_k) * (private_scale * priv_mult)
        private_latents.append(true_u_private)
        
        v_sh = _sparse_signed_basis(shared_k, p, true_sparsity)
        v_pr = torch.randn(private_k, p)
        v_true.append(v_sh.t())
        
        x = (true_u_shared @ v_sh) + (true_u_private @ v_pr) + torch.randn(n_samples, p) * noise_scale
        data_matrices.append(x)
        
    return {
        "kind": "shared_plus_private",
        "data": data_matrices,
        "true_u": true_u_shared,
        "true_u_private": private_latents,
        "true_v": v_true,
        "outcome": outcome_signal,
        "shared_k": shared_k,
        "private_k": private_k
    }

def build_nonnegative_parts_case(n_samples: int = 400,
                                 shared_k: int = 3,
                                 p_list: Optional[List[int]] = None,
                                 noise_scale: Optional[float] = None,
                                 seed: int = 42,
                                 signal_component: int = 0,
                                 **kwargs) -> Dict[str, Any]:
    r"""
    Generate data whose ground-truth basis is non-negative, sparse and disjoint.
    """
    private_scale = float(kwargs.get("private_scale", 0.0))
    active_frac = kwargs.get("active_frac", None)
    # Regime defaults apply only to parameters the caller left unset; the
    # private-background branch used to overwrite explicit values.
    if p_list is None:
        p_list = [120, 45, 30] if private_scale > 0.0 else [60, 40, 30]
    if noise_scale is None:
        noise_scale = 0.18 if private_scale > 0.0 else 0.3
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not 0 <= signal_component < shared_k:
        raise ValueError(
            f"signal_component must be in [0, {shared_k}), got {signal_component}."
        )

    if private_scale > 0.0:
        scales = torch.tensor([1.0, 0.65, 0.48][:shared_k])
    else:
        scales = torch.tensor([1.0 / (1.0 + 1.5 * i) for i in range(shared_k)])
    true_u = torch.rand(n_samples, shared_k) * scales

    data_matrices = []
    v_true = []
    for idx_v, p in enumerate(p_list):
        if p < shared_k:
            raise ValueError(f"Each view needs at least shared_k={shared_k} features, got p={p}.")
        v = torch.zeros(p, shared_k)
        order = torch.randperm(p)
        if active_frac is not None:
            m = max(1, min(p // shared_k, int(round(float(active_frac) * p))))
            for j in range(shared_k):
                block = order[j * m:(j + 1) * m]
                v[block, j] = 0.5 + torch.rand(len(block))
        else:
            bounds = np.linspace(0, p, shared_k + 1).astype(int)
            for j in range(shared_k):
                block = order[bounds[j]:bounds[j + 1]]
                v[block, j] = 0.5 + torch.rand(len(block))
        x = true_u @ v.t() + torch.randn(n_samples, p) * noise_scale
        if private_scale > 0.0:
            priv_mult = 1.5 if idx_v == 0 else 0.9
            u_priv = torch.rand(n_samples, 2) * (private_scale * priv_mult)
            v_priv = torch.rand(p, 2)
            x = x + u_priv @ v_priv.t()
        data_matrices.append(x)
        v_true.append(v)

    outcome_signal = (true_u[:, signal_component] * 3.0
                      + torch.randn(n_samples) * 0.15)

    return {
        "kind": "nonnegative_parts",
        "data": data_matrices,
        "true_u": true_u,
        "true_v": v_true,
        "outcome": outcome_signal,
        "shared_k": shared_k,
        "signal_component": signal_component,
    }


def _spanning_forest(edges: List[Tuple[int, int]], m: int) -> List[Tuple[int, int]]:
    """Breadth-first spanning forest of an undirected graph on ``range(m)``, as (parent, child)
    pairs; each connected component is rooted at its smallest node."""
    adj: Dict[int, List[int]] = {i: [] for i in range(m)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, tree = set(), []
    for root in range(m):
        if root in seen:
            continue
        seen.add(root)
        queue = [root]
        while queue:
            a = queue.pop(0)
            for b in sorted(adj[a]):
                if b not in seen:
                    seen.add(b)
                    tree.append((a, b))
                    queue.append(b)
    return tree


def build_path_graph_case(graph: Any, n_samples: int = 500, k: int = 3,
                          p_per_view: Sequence[int] = (20, 25, 18), noise: float = 0.3,
                          covariate_effect: float = 0.0, group_effect: float = 0.0,
                          random_state: int = 0, *, alpha: float = 0.7,
                          moderated_edge: Optional[Tuple[int, int]] = None,
                          n_covariates: int = 2) -> Dict[str, Any]:
    r"""Views whose latents follow an undirected graph, for the score-level path model.

    The latent of each view is ``(n, k)``. On a breadth-first spanning forest of ``graph`` (an
    edge list or adjacency dict over view indices ``0 .. M-1``, ``M = len(p_per_view)``) each
    root draws :math:`L_r \sim N(0, I_k)` and each child
    :math:`L_b = \alpha L_a + \sqrt{1 - \alpha^2} E_b`, so every latent has unit variance and a
    chain a - b - c has :math:`L_a \perp L_c \mid L_b` exactly. Graph edges that close a cycle
    are not planted (the forest is returned as ``tree``). Views are
    :math:`X_m = L_m B_m + \mathrm{noise}\, E_m` with ``B_m`` drawn by `_sparse_signed_basis`
    (dense Gaussian).

    ``covariate_effect`` adds :math:`C \beta_m` (``C`` is ``(n, n_covariates)`` standard normal,
    ``beta_m`` Gaussian) to every view, so unadjusted edges are inflated and a Freedman-Lane
    analysis that partials out ``C`` must remove it. ``group_effect`` plants a moderation: rows of
    group 0 (``G`` is a 0/1 draw) use :math:`\alpha (1 - \mathrm{group\_effect})` on
    ``moderated_edge`` (default: the first forest edge), group 1 uses :math:`\alpha`; the latent
    still has unit variance in both groups. The outcome is ``L_last[:, 0] + noise * e`` with
    ``L_last`` the latent of view ``M - 1``.

    All draws come from one private stream seeded by ``random_state``.

    Returns a dict: ``views`` (list of float32 tensors), ``latents`` (list of ``(n, k)`` arrays),
    ``graph`` (edge list), ``tree`` (planted (parent, child) edges), ``C``, ``G``, ``outcome``,
    ``alpha``, ``moderated_edge``.
    """
    m_views = len(p_per_view)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if not 0.0 <= group_effect <= 1.0:
        raise ValueError(f"group_effect must be in [0, 1]; got {group_effect}.")
    if graph is None:
        edges = [(a, b) for a in range(m_views) for b in range(a + 1, m_views)]
    elif isinstance(graph, dict):
        edges = [(int(a), int(b)) for a, bs in graph.items() for b in bs]
    else:
        edges = [(int(a), int(b)) for a, b in graph]
    for a, b in edges:
        if a == b or not (0 <= a < m_views and 0 <= b < m_views):
            raise ValueError(f"edge ({a}, {b}) is not between two of the {m_views} views.")
    tree = _spanning_forest(edges, m_views)
    if moderated_edge is None and group_effect > 0.0:
        if not tree:
            raise ValueError("group_effect needs at least one edge to moderate.")
        moderated_edge = tree[0]
    if moderated_edge is not None:
        me = tuple(int(x) for x in moderated_edge)
        hit = [t for t in tree if set(t) == set(me)]
        if not hit:
            raise ValueError(f"moderated_edge {me} is not a planted edge {tree}.")
        moderated_edge = hit[0]
    rng = np.random.default_rng(random_state)
    gen = torch.Generator().manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
    n = int(n_samples)
    C = rng.standard_normal((n, n_covariates))
    G = rng.integers(0, 2, n)
    lat: List[Optional[np.ndarray]] = [None] * m_views
    children = {b for _, b in tree}
    for r in range(m_views):
        if r not in children:
            lat[r] = rng.standard_normal((n, k))
    for a, b in tree:
        al = np.full(n, alpha)
        if moderated_edge is not None and (a, b) == moderated_edge:
            al = np.where(G == 0, alpha * (1.0 - group_effect), alpha)
        lat[b] = al[:, None] * lat[a] + np.sqrt(1.0 - al ** 2)[:, None] * rng.standard_normal((n, k))
    views = []
    for m, p in enumerate(p_per_view):
        B = _sparse_signed_basis(k, int(p), 0.0, generator=gen).double().numpy()
        x = lat[m] @ B + noise * rng.standard_normal((n, int(p)))
        if covariate_effect:
            x = x + covariate_effect * (C @ rng.standard_normal((n_covariates, int(p))))
        views.append(torch.tensor(x).float())
    outcome = lat[-1][:, 0] + noise * rng.standard_normal(n)
    return {"views": views, "latents": lat, "graph": edges, "tree": tree, "C": C, "G": G,
            "outcome": outcome, "alpha": alpha, "moderated_edge": moderated_edge}


def build_case(kind: str = "nonlinear", **kwargs) -> Dict[str, Any]:
    """
    Factory function to generate synthetic benchmark cases for SiMLR.

    Provides a unified interface for creating different data scenarios, 
    including linear, nonlinear, and shared/private structures.

    Parameters
    ----------
    kind : str, default="nonlinear"
        The type of case to build. Options:
        - 'linear' or 'linear_footprint': See `build_linear_footprint_case`.
        - 'nonlinear': See `build_nonlinear_shared_case`.
        - 'shared_plus_private': See `build_shared_plus_private_case`.
    **kwargs : Dict[str, Any]
        Keyword arguments passed to the specific builder function.

    Returns
    -------
    Dict[str, Any]
        A standardized case dictionary containing:
        - `data`: List[torch.Tensor] of modalities.
        - `true_u`: Ground truth shared latent space.
        - `outcome`: Ground truth outcome variable.
        - `shared_k`: Number of shared components.
        - `gen_func`: (Optional) The generator function used.

    Raises
    ------
    ValueError
        If an unknown `kind` is provided.
    TypeError
        If inputs are of invalid types.
    """
    builders = {
        "linear": build_linear_footprint_case,
        "nonlinear": build_nonlinear_shared_case,
        "shared_plus_private": build_shared_plus_private_case,
        # Non-negative, sparse, disjoint ground truth -- the regime in which
        # the structural constraints are true rather than violated.
        "nonneg_parts": build_nonnegative_parts_case,
        # Same basis family, with focal 20%-active (80%-sparse) columns and the
        # outcome carried by the weakest latent, so top-variance is no longer
        # the right answer by construction and focal sparsity is tested.
        "nonneg_parts_weak": lambda **kw: build_nonnegative_parts_case(
            **{
                "active_frac": 0.20 if float(kw.get("private_scale", 0.0)) > 0.0 else None,
                **kw,
                "signal_component": kw.get("shared_k", 3) - 1,
            }),
    }

    if kind not in builders:
        raise ValueError(f"Unknown case kind: {kind}")
    return builders[kind](**kwargs)
