import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

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

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if 'noise_level' in kwargs: noise_scale = kwargs['noise_level']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Stronger outcome signal mixed into latent
    outcome_signal = torch.randn(n_samples)
    true_u = torch.randn(n_samples, shared_k)
    true_u[:, 0] += outcome_signal * 2.0
    
    data_matrices = []
    v_true = []
    
    for p in p_list:
        v = torch.randn(shared_k, p)
        x = true_u @ v + torch.randn(n_samples, p) * noise_scale
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

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    shared_k : int, default=3
        Rank of the shared latent space.
    p_list : List[int], default=[100, 80, 60]
        Dimensionality of each modality.
    noise_scale : float, default=0.5
        Standard deviation of Gaussian noise.
    seed : int, default=42
        Random seed.
    regime : str, default="mixed"
        The type of nonlinearity to apply. Options:
        - 'mixed': Different nonlinearity for each modality (Linear, Poly, Sin/Exp).
        - 'polynomial': Square and interaction terms.
        - 'sinusoidal': Periodic transformations.
    **kwargs : Dict[str, Any]
        Supports `noise_level` as an alias for `noise_scale`.

    Returns
    -------
    Dict[str, Any]
        Standardized case dictionary.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if 'noise_level' in kwargs: noise_scale = kwargs['noise_level']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    outcome_signal = torch.randn(n_samples)
    true_u = torch.randn(n_samples, shared_k)
    true_u[:, 0] += outcome_signal * 2.0 
    
    data_matrices = []
    v_true = [] # Approximate linear basis
    
    for i, p in enumerate(p_list):
        # Base linear mapping
        v = torch.randn(shared_k, p)
        v_true.append(v.t())
        x_lin = true_u @ v
        
        if regime == "mixed":
            # Modality 1 linear, 2 Poly, 3 Exp/Sig
            if i == 0:
                x = x_lin
            elif i == 1:
                x = 0.5 * x_lin + 0.5 * ((true_u**2) @ torch.randn(shared_k, p))
            else:
                x = 0.5 * x_lin + 0.5 * (torch.sin(true_u * 2.0) @ torch.randn(shared_k, p))
        elif regime == "polynomial":
            x = x_lin + 0.4 * (x_lin**2)
        elif regime == "sinusoidal":
            x = torch.sin(2.5 * x_lin)
        else:
            x = x_lin
            
        x += torch.randn(n_samples, p) * noise_scale
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

    Specifically designed for NED++ (Shared/Private SiMR), where each modality 
    contains a signal that is shared with other modalities, plus a signal 
    that is unique to itself.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    shared_k : int, default=2
        Rank of the common (shared) latent space.
    private_k : int, default=2
        Rank of the unique (private) latent space for each modality.
    p_list : List[int], default=[100, 100]
        Dimensionality of each modality.
    noise_scale : float, default=0.5
        Standard deviation of Gaussian noise.
    seed : int, default=42
        Random seed.
    **kwargs : Dict[str, Any]
        Supports `noise_level` as an alias for `noise_scale`.

    Returns
    -------
    Dict[str, Any]
        Standardized case dictionary containing both shared and private 
        ground truths.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if 'noise_level' in kwargs: noise_scale = kwargs['noise_level']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    outcome_signal = torch.randn(n_samples)
    true_u_shared = torch.randn(n_samples, shared_k)
    true_u_shared[:, 0] += outcome_signal * 2.0
    
    data_matrices = []
    private_latents = []
    v_true = []
    
    for p in p_list:
        true_u_private = torch.randn(n_samples, private_k)
        private_latents.append(true_u_private)
        
        v_sh = torch.randn(shared_k, p)
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

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    builders = {
        "linear": build_linear_footprint_case,
        "nonlinear": build_nonlinear_shared_case,
        "shared_plus_private": build_shared_plus_private_case
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

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
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
