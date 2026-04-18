import torch
import numpy as np
import warnings
from typing import List, Optional, Union, Dict, Any, Tuple
from .utils import newton_schulz_orthogonalize, safe_svd
try:
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    FastICA = None
    ConvergenceWarning = None

def compute_shared_consensus(projections: List[torch.Tensor], 
                            mixing_algorithm: str = "svd", 
                            k: Optional[int] = None,
                            orthogonalize: bool = False,
                            training: bool = False,
                            anchor: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Combine modality-specific projections into a shared latent consensus (U).
    
    Anchor-based Prediction Fix:
    To prevent coordinate drift/rotation in SVD/PCA/ICA during prediction,
    we utilize a learned 'anchor' projection matrix.
    """
    if not projections:
        return torch.empty(0)
        
    n_modalities = len(projections)
    if k is None:
        k = projections[0].shape[1]
        
    norm_projs = []
    for p in projections:
        p_safe = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p_rms = torch.norm(p_safe, p='fro') / np.sqrt(p_safe.shape[0])
        if p_rms > 1e-6:
            norm_projs.append(p_safe / (p_rms + 1e-8))
        else:
            norm_projs.append(torch.randn_like(p_safe) * 1e-4)

    big_p = torch.cat(norm_projs, dim=1)

    # PREDICTION MODE: Use anchor for volatile algorithms (SVD/PCA/ICA)
    # Deterministic algorithms (Newton/Avg) can be recomputed directly.
    volatile = mixing_algorithm in ["svd", "pca", "ica"]
    if not training and volatile and anchor is not None:
        u = big_p @ anchor
    else:
        # TRAINING OR DETERMINISTIC MODE
        if mixing_algorithm == "avg":
            u = torch.mean(torch.stack(norm_projs), dim=0)
            new_anchor = None # Not needed for deterministic avg
        elif mixing_algorithm == "newton":
            u = torch.mean(torch.stack(norm_projs), dim=0)
            u = newton_schulz_orthogonalize(u, iterations=10)
            new_anchor = None
        elif mixing_algorithm == "ica":
            avg_p = big_p.detach().cpu().numpy()
            # Restore previously fixed ConvergenceWarning suppression
            with warnings.catch_warnings():
                if ConvergenceWarning is not None:
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                ica = FastICA(n_components=k, random_state=42, max_iter=2000, tol=1e-2)
                try:
                    u_np = ica.fit_transform(avg_p)
                    new_anchor = torch.from_numpy(ica.components_.T).to(big_p.device).to(big_p.dtype)
                except:
                    u_np = avg_p[:, :k]
                    new_anchor = torch.eye(big_p.shape[1], k, device=big_p.device).to(big_p.dtype)
            u = torch.from_numpy(u_np).to(big_p.device).to(big_p.dtype)
        else:
            # SVD / PCA
            if mixing_algorithm == "pca":
                big_p_c = big_p - torch.mean(big_p, dim=0)
                _, _, vh = safe_svd(big_p_c, full_matrices=False)
            else:
                _, _, vh = safe_svd(big_p, full_matrices=False)
            
            new_anchor = vh.T[:, :k]
            u = big_p @ new_anchor

        if training:
            return u, new_anchor
            
    # Standardize final U
    u = u - u.mean(0, keepdim=True)
    u_std = torch.std(u, dim=0, keepdim=True) + 1e-6
    u = u / u_std
        
    return u
