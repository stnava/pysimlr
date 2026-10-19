import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

def _sparse_signed_basis(shared_k: int, p: int, true_sparsity: float) -> torch.Tensor:
    """Construct a (shared_k, p) signed basis with exact per-component sparsity."""
    if true_sparsity <= 0.0:
        return torch.randn(shared_k, p)
    active_frac = max(1.0 / p, min(1.0, 1.0 - float(true_sparsity)))
    m = max(1, min(p, int(round(active_frac * p))))
    v = torch.zeros(shared_k, p)
    order = torch.randperm(p)
    stride = max(1, p // shared_k)
    for j in range(shared_k):
        idx = torch.arange(j * stride, j * stride + m) % p
        cols = order[idx]
        mag = 0.5 + torch.rand(len(cols))
        signs = torch.where(torch.rand(len(cols)) > 0.5, 1.0, -1.0)
        v[j, cols] = mag * signs
    return v


def build_linear_footprint_case(n_samples: int = 1000, 
                                shared_k: int = 3, 
                                p_list: List[int] = [100, 80, 60],
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
    p_list : List[int], default=[100, 80, 60]
        The dimensionality of each generated modality.
    noise_scale : float, default=0.5
        Standard deviation of Gaussian noise added to each modality.
    seed : int, default=42
        Random seed for reproducibility.
    **kwargs : Dict[str, Any]
        Supports `noise_level` as an alias for `noise_scale`.

    Returns
    -------
    Dict[str, Any]
        Standardized case dictionary containing data matrices and ground truth.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    if 'noise_level' in kwargs: noise_scale = kwargs['noise_level']
    private_scale = float(kwargs.get("private_scale", 0.0))
    true_sparsity = float(kwargs.get("true_sparsity", 0.5 if private_scale > 0.0 else 0.0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if private_scale > 0.0:
        # Clean linear multi-view regime: shared components are stronger than
        # view-private background (private_scale=0.5), with equal outcome weights.
        p_list = kwargs.get("p_list", [100, 80, 60])
        scales = torch.tensor([1.2, 1.0, 0.85][:shared_k])
        true_u = torch.randn(n_samples, shared_k) * scales
        weights = torch.ones(shared_k)
        outcome_signal = (true_u @ weights) + torch.randn(n_samples) * 0.15
    else:
        outcome_signal = torch.randn(n_samples)
        true_u = torch.randn(n_samples, shared_k)
        true_u[:, 0] += outcome_signal * 2.0
    
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
        "kind": "linear_footprint",
        "data": data_matrices,
        "true_u": true_u,
        "true_v": v_true,
        "outcome": outcome_signal,
        "shared_k": shared_k
    }

def build_nonlinear_shared_case(n_samples: int = 1000, 
                                shared_k: int = 3, 
                                p_list: List[int] = [100, 80, 60],
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
    if 'noise_level' in kwargs: noise_scale = kwargs['noise_level']
    private_scale = float(kwargs.get("private_scale", 0.0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if private_scale > 0.0:
        p_list = kwargs.get("p_list", [120, 80, 50])
        scales = torch.tensor([1.15, 0.95, 0.80][:shared_k])
        true_u = torch.randn(n_samples, shared_k) * scales
        weights = torch.tensor([0.9, 1.0, 1.1][:shared_k])
        outcome_signal = (true_u @ weights) + torch.randn(n_samples) * 0.15
    else:
        outcome_signal = torch.randn(n_samples)
        true_u = torch.randn(n_samples, shared_k)
        true_u[:, 0] += outcome_signal * 2.0 
    
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
                                   shared_k: int = 2, 
                                   private_k: int = 2,
                                   p_list: List[int] = [100, 100],
                                   noise_scale: float = 0.5,
                                   seed: int = 42,
                                   **kwargs) -> Dict[str, Any]:
    """
    Generate synthetic data with both shared and private latent components.
    """
    if 'noise_level' in kwargs: noise_scale = kwargs['noise_level']
    has_explicit_priv = "private_scale" in kwargs
    private_scale = float(kwargs.get("private_scale", 1.0))
    true_sparsity = float(kwargs.get("true_sparsity", 0.8 if has_explicit_priv else 0.0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if has_explicit_priv:
        shared_k = kwargs.get("shared_k", 3)
        private_k = kwargs.get("private_k", 2)
        p_list = kwargs.get("p_list", [160, 60, 30])
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
                                 p_list: List[int] = [60, 40, 30],
                                 noise_scale: float = 0.3,
                                 seed: int = 42,
                                 signal_component: int = 0,
                                 **kwargs) -> Dict[str, Any]:
    r"""
    Generate data whose ground-truth basis is non-negative, sparse and disjoint.
    """
    if 'noise_level' in kwargs:
        noise_scale = kwargs['noise_level']
    private_scale = float(kwargs.get("private_scale", 0.0))
    active_frac = kwargs.get("active_frac", None)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not 0 <= signal_component < shared_k:
        raise ValueError(
            f"signal_component must be in [0, {shared_k}), got {signal_component}."
        )

    if private_scale > 0.0:
        p_list = kwargs.get("p_list", [120, 45, 30])
        scales = torch.tensor([1.0, 0.65, 0.48][:shared_k])
        noise_scale = 0.18
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


def build_case(kind: str = "nonlinear_shared", **kwargs) -> Dict[str, Any]:
    """
    Factory function to generate synthetic benchmark cases for SiMLR.

    Provides a unified interface for creating different data scenarios, 
    including linear, nonlinear, and shared/private structures.

    Parameters
    ----------
    kind : str, default="nonlinear_shared"
        The type of case to build. Options:
        - 'linear' or 'linear_footprint': See `build_linear_footprint_case`.
        - 'nonlinear' or 'nonlinear_shared': See `build_nonlinear_shared_case`.
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
    # Backward compat
    if kind == "nonlinear_shared": kind = "nonlinear"
    if kind == "linear_footprint": kind = "linear"

    if kind not in builders:
        raise ValueError(f"Unknown case kind: {kind}")
    return builders[kind](**kwargs)

def plot_case_generative_shape(case_res: Dict[str, Any], feature_idx: int = 0) -> plt.Figure:
    """
    Visualize the generative 'shape' of a synthetic case.

    Plots the ground truth latent space (U) and its relationship with the 
    observed features (X) across modalities. This helps confirm the 
    linearity or nonlinearity of the simulated data.

    Parameters
    ----------
    case_res : Dict[str, Any]
        Standardized case dictionary from `build_case`.
    feature_idx : int, default=0
        The index of the feature to plot against the latent signal.

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the latent space and modality-specific feature shapes.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    u = case_res["true_u"].numpy()
    data = case_res["data"]
    kind = case_res["kind"]
    
    n_views = len(data)
    fig, axes = plt.subplots(1, n_views + 1, figsize=(4 * (n_views + 1), 4))
    
    # 1. Latent Space (U1 vs U2)
    if u.shape[1] >= 2:
        axes[0].scatter(u[:, 0], u[:, 1], alpha=0.5, c=u[:, 0], cmap='viridis', s=10)
        axes[0].set_title("Ground Truth Consensus\n($U_1$ vs $U_2$)", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("$U_1$")
        axes[0].set_ylabel("$U_2$")
    else:
        sns.histplot(u[:, 0], bins=30, color='indigo', alpha=0.6, ax=axes[0], kde=True)
        axes[0].set_title("Shared Latent Signal ($U_1$)", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("$U_1$")
        
    # 2. View Transformations
    for i in range(n_views):
        x = data[i].numpy()
        axes[i+1].scatter(u[:, 0], x[:, feature_idx], alpha=0.3, c='teal', s=10)
        axes[i+1].set_title(f"View {i+1} Mapping\n($U_1 \\to X_{{{feature_idx}}}$)", fontsize=11, fontweight='bold')
        axes[i+1].set_xlabel("$U_1$ (Consensus)")
        axes[i+1].set_ylabel(f"Feature {feature_idx}")
        
    plt.suptitle(f"Generative Shape Analysis: {kind.replace('_', ' ').title()}", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
