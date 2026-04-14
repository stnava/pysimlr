import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
try:
    from sklearn.decomposition import FastICA
except ImportError:
    FastICA = None

def compute_shared_consensus(projections: List[torch.Tensor], 
                            mixing_algorithm: str = "svd", 
                            k: Optional[int] = None,
                            orthogonalize: bool = False,
                            training: bool = False,
                            batch_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """
    Combine modality-specific projections into a shared latent consensus (U).

    This function aggregates the projected latent representations from multiple 
    data views into a single shared basis. It supports various mixing strategies 
    and ensures the final consensus is standardized.

    Parameters
    ----------
    projections : List[torch.Tensor]
        A list of projected latent matrices (Z_m = X_m V_m) from each modality.
    mixing_algorithm : str, default="svd"
        The algorithm used to estimate the shared consensus:
        - "avg": Simple mean of all view-specific latents.
        - "newton": Newton-Schulz iteration for strict Stiefel manifold alignment.
        - "pca": First K principal components of the stacked latents.
        - "svd": SVD-based consensus (equivalent to PCA on uncentered latents).
        - "ica": Independent Component Analysis (requires scikit-learn).
        - "stochastic": Random projection mixing.
    k : int, optional
        The dimensionality of the shared latent space. Defaults to the number 
        of columns in the first projection.
    orthogonalize : bool, default=False
        If True, enforces orthogonality on the final consensus U.
    training : bool, default=False
        Whether the consensus is being computed during training (reserved 
        for future stabilization logic).
    batch_context : Dict[str, Any], optional
        Contextual information about the current training batch.

    Returns
    -------
    torch.Tensor
        The shared latent consensus U, standardized to unit variance per dimension.
    """
    if not projections:
        return torch.empty(0)
        
    n_modalities = len(projections)
    if k is None:
        k = projections[0].shape[1]
        
    norm_projs = []
    for p in projections:
        p_safe = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        # Use RMS normalization (Frobenius norm / sqrt(N)) to ensure the scale
        # is independent of the number of samples in the batch.
        p_rms = torch.norm(p_safe, p='fro') / np.sqrt(p_safe.shape[0])
        if p_rms > 1e-6:
            norm_projs.append(p_safe / (p_rms + 1e-8))
        else:
            norm_projs.append(torch.randn_like(p_safe) * 1e-4)
            
    if mixing_algorithm == "avg":
        u = torch.mean(torch.stack(norm_projs), dim=0)
    elif mixing_algorithm == "newton":
        u = torch.mean(torch.stack(norm_projs), dim=0)
        # Always project to Stiefel for alignment between train/eval.
        try:
            u_u, _, u_vh = torch.linalg.svd(u, full_matrices=False)
            u = u_u @ u_vh
        except:
            u = torch.nn.functional.normalize(u, p=2, dim=0)
    elif mixing_algorithm == "stochastic":
        big_p = torch.cat(norm_projs, dim=1)
        g = torch.randn(big_p.shape[1], k, device=big_p.device, dtype=big_p.dtype)
        u = big_p @ g
    elif mixing_algorithm == "ica":
        if FastICA is None:
            raise ImportError("scikit-learn is required for mixing_algorithm='ica'")
        avg_p = torch.cat(norm_projs, dim=1).detach().cpu().numpy()
        avg_p = np.nan_to_num(avg_p, nan=0.0, posinf=0.0, neginf=0.0)
        ica = FastICA(n_components=k, random_state=42, max_iter=1000)
        try:
            u_np = ica.fit_transform(avg_p)
        except:
            u_np = avg_p[:, :k]
        u = torch.from_numpy(u_np).to(projections[0].device).to(projections[0].dtype)
    else:
        big_p = torch.cat(norm_projs, dim=1)
        if mixing_algorithm == "pca":
            big_p_centered = big_p - torch.mean(big_p, dim=0)
            u, _, _ = torch.linalg.svd(big_p_centered, full_matrices=False)
            u = u[:, :k]
        else: # Default to svd
            u, _, _ = torch.linalg.svd(big_p, full_matrices=False)
            u = u[:, :k]
            
    if orthogonalize and mixing_algorithm != "newton":
        try:
            u_u, _, u_vh = torch.linalg.svd(u, full_matrices=False)
            u = u_u @ u_vh
            u = u[:, :k]
        except:
            pass
            
    # Standardize final U to unit variance per dimension.
    # This ensures that downstream models (like LinearRegression) see a consistent
    # scale regardless of the consensus algorithm or sample size.
    u = u - u.mean(0, keepdim=True)
    u_std = torch.std(u, dim=0, keepdim=True) + 1e-6
    u = u / u_std
        
    return u
