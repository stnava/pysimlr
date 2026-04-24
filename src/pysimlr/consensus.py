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
                            anchor: Optional[torch.Tensor] = None,
                            topology: str = "star") -> Union[torch.Tensor, List[torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[torch.Tensor]]]:
    """
    Combine modality-specific projections into a shared latent consensus (U).
    
    Topology:
    - 'star': All modalities align to a single shared consensus.
    - 'loo': (Leave-One-Out) Modality i aligns to the consensus of all OTHER modalities.
    
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

    # Core consensus logic separated into a helper to reuse for LOO
    def _get_u(proj_list, return_anchor=False):
        local_big_p = torch.cat(proj_list, dim=1)
        volatile = mixing_algorithm in ["svd", "pca", "ica"]
        if not training and volatile and anchor is not None and len(proj_list) == len(projections):
            local_u = local_big_p @ anchor
            if return_anchor: return local_u, anchor
            return local_u

        if mixing_algorithm == "avg":
            local_u = torch.mean(torch.stack(proj_list), dim=0)
            local_anchor = None
        elif mixing_algorithm == "newton":
            local_u = torch.mean(torch.stack(proj_list), dim=0)
            local_u = newton_schulz_orthogonalize(local_u, iterations=10)
            local_anchor = None
        elif mixing_algorithm == "ica":
            avg_p = local_big_p.detach().cpu().numpy()
            with warnings.catch_warnings():
                if ConvergenceWarning is not None:
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                ica = FastICA(n_components=k, random_state=42, max_iter=2000, tol=1e-2)
                try:
                    u_np = ica.fit_transform(avg_p)
                    local_anchor = torch.from_numpy(ica.components_.T).to(local_big_p.device).to(local_big_p.dtype)
                except:
                    u_np = avg_p[:, :k]
                    local_anchor = torch.eye(local_big_p.shape[1], k, device=local_big_p.device).to(local_big_p.dtype)
            local_u = torch.from_numpy(u_np).to(local_big_p.device).to(local_big_p.dtype)
        else:
            if mixing_algorithm == "pca":
                big_p_c = local_big_p - torch.mean(local_big_p, dim=0)
                _, _, vh = safe_svd(big_p_c, full_matrices=False)
            else:
                _, _, vh = safe_svd(local_big_p, full_matrices=False)
            
            local_anchor = vh.T[:, :k]
            local_u = local_big_p @ local_anchor
            
        local_u = local_u - local_u.mean(0, keepdim=True)
        u_std = torch.std(local_u, dim=0, keepdim=True) + 1e-6
        local_u = local_u / u_std
            
        if return_anchor:
            return local_u, local_anchor
        return local_u

    if topology == "star":
        if training:
            u, new_anchor = _get_u(norm_projs, return_anchor=True)
            return u, new_anchor
        else:
            return _get_u(norm_projs, return_anchor=False)
            
    elif topology == "loo":
        # For leave-one-out, we return a list of U matrices, one for each modality.
        # U_i is the consensus of all modalities EXCEPT i.
        # Note: Anchor EMA fix is theoretically complex for LOO since there are M anchors. 
        # For simplicity, we fallback to star-anchor for inference, or recompute if deterministic.
        
        if not training and anchor is not None:
            # If inference and we have an anchor, just use star topology for stability
            return _get_u(norm_projs, return_anchor=False)
            
        u_list = []
        for i in range(len(norm_projs)):
            loo_projs = norm_projs[:i] + norm_projs[i+1:]
            if len(loo_projs) == 0:
                # Edge case: only 1 modality. Fallback to its own projection.
                u_list.append(norm_projs[0])
            else:
                u_list.append(_get_u(loo_projs, return_anchor=False))
                
        if training:
            # We still compute the star anchor to save for inference stability
            _, new_anchor = _get_u(norm_projs, return_anchor=True)
            return u_list, new_anchor
        return u_list
    u = u - u.mean(0, keepdim=True)
    u_std = torch.std(u, dim=0, keepdim=True) + 1e-6
    u = u / u_std
        
    return u
