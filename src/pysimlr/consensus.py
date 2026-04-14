import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
try:
    from sklearn.decomposition import FastICA
except ImportError:
    FastICA = None

from .utils import safe_svd

def compute_shared_consensus(projections: List[torch.Tensor], 
                             mixing_algorithm: str = "svd", 
                             k: Optional[int] = None,
                             orthogonalize: bool = False,
                             training: bool = False) -> torch.Tensor:
    """
    Aggregate view-specific latent representations into a shared consensus space U.

    This function implements the 'Mixing' phase of the SiMLR framework. It identifies
     the central manifold that best represents the agreement between all modalities.

    Parameters
    ----------
    projections : list of torch.Tensor
        The view-specific latent matrices Z_m = f_m(X_m).
    mixing_algorithm : str, default="svd"
        The aggregation strategy:
        - "avg": Arithmetic mean. Fast but sensitive to modality-specific noise.
        - "pca" or "svd": Global variance maximization. Robust denoiser.
        - "ica": Statistical independence. Prevents signal leakage between dimensions.
        - "newton": Newton-Schulz iterative orthogonalization. Maximum stability.
    k : int, optional
        Target latent dimension. Defaults to the dimension of the projections.
    orthogonalize : bool, default=False
        If True, ensures the returned U is a strict orthonormal frame (U.T @ U = I).
    training : bool, default=False
        Whether the call is occurring during a training loop (affects gradient flow).

    Returns
    -------
    torch.Tensor
        The shared consensus matrix U.
    """
    n_views = len(projections)
    if n_views == 0:
        raise ValueError("projections list is empty")
        
    device = projections[0].device
    dtype = projections[0].dtype
    n_samples = projections[0].shape[0]
    latent_dim = projections[0].shape[1] if k is None else k
    
    # Pre-normalize projections to ensure equal voting power
    norm_projections = []
    for z in projections:
        z_c = z - z.mean(0, keepdim=True)
        z_std = torch.std(z_c, dim=0, keepdim=True) + 1e-6
        norm_projections.append(z_c / z_std)
        
    if mixing_algorithm == "avg":
        u = torch.stack(norm_projections).mean(dim=0)
        
    elif mixing_algorithm in ["svd", "pca"]:
        # Standard approach: extract top components of the concatenated space
        z_stack = torch.cat(norm_projections, dim=1)
        # Use hardware-aware safe_svd
        u_svd, s_svd, _ = safe_svd(z_stack, full_matrices=False)
        u = u_svd[:, :latent_dim]
        
    elif mixing_algorithm == "ica":
        if FastICA is None:
            # Fallback to PCA if sklearn not available
            z_stack = torch.cat(norm_projections, dim=1)
            u_svd, _, _ = safe_svd(z_stack, full_matrices=False)
            u = u_svd[:, :latent_dim]
        else:
            z_stack = torch.cat(norm_projections, dim=1).detach().cpu().numpy()
            ica = FastICA(n_components=latent_dim, random_state=42, max_iter=1000)
            u_np = ica.fit_transform(z_stack)
            u = torch.from_numpy(u_np).to(device=device, dtype=dtype)
            
    elif mixing_algorithm == "newton":
        # Iterative Orthogonal Procrustes update
        z_sum = torch.stack(norm_projections).sum(dim=0)
        u_svd, _, vh_svd = safe_svd(z_sum, full_matrices=False)
        u = u_svd @ vh_svd
        u = u[:, :latent_dim]
        
    else:
        # Default fallback
        u = torch.stack(norm_projections).mean(dim=0)
        
    if orthogonalize:
        u_svd, _, vh_svd = safe_svd(u, full_matrices=False)
        u = u_svd @ vh_svd
        
    # Scale to ensure downstream models see consistent variance
    u = u - u.mean(0, keepdim=True)
    u_std = torch.std(u, dim=0, keepdim=True) + 1e-6
    u = u / u_std
        
    return u
