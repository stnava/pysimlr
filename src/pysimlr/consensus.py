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
    Combine projections from different modalities into a shared latent basis U.
    Shared by linear SIMLR and deep models.
    """
    if not projections:
        return torch.empty(0)
        
    n_modalities = len(projections)
    if k is None:
        k = projections[0].shape[1]
        
    norm_projs = []
    for p in projections:
        p_safe = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p_norm = torch.norm(p_safe, p='fro')
        if p_norm > 1e-6:
            # Frobenius normalization makes each modality contribute equally in total energy
            norm_projs.append(p_safe / (p_norm + 1e-8))
        else:
            # Avoid divide by zero for zero projections
            norm_projs.append(torch.randn_like(p_safe) * 1e-4)
            
    if mixing_algorithm == "avg":
        u = torch.mean(torch.stack(norm_projs), dim=0)
    elif mixing_algorithm == "newton":
        u = torch.mean(torch.stack(norm_projs), dim=0)
        # Always project to Stiefel for alignment between train/eval.
        # We use a try-except to handle numerical issues during training.
        try:
            u_u, _, u_vh = torch.linalg.svd(u, full_matrices=False)
            u = u_u @ u_vh
        except:
            # Fallback to column normalization if SVD fails (rare)
            u = torch.nn.functional.normalize(u, p=2, dim=0)
    elif mixing_algorithm == "stochastic":
        big_p = torch.cat(norm_projs, dim=1)
        # Use a random projection to reduce dimensionality
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
        # Default to SVD or PCA
        big_p = torch.cat(norm_projs, dim=1)
        if mixing_algorithm == "pca":
            # PCA logic: center then SVD
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
            
    # For regression stability and benchmark score consistency, 
    # we rescale the final U to have unit variance columns.
    # This prevents the scale from being purely determined by N and K.
    u_std = torch.std(u, dim=0, keepdim=True) + 1e-6
    u = u / u_std
        
    return u
