import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

def build_linear_footprint_case(n_samples: int = 1000, 
                                shared_k: int = 3, 
                                p_list: List[int] = [100, 80, 60],
                                noise_scale: float = 0.5,
                                seed: int = 42,
                                **kwargs) -> Dict[str, Any]:
    """Purely linear regime: Simple shared signal with linear mappings."""
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
                                **kwargs) -> Dict[str, Any]:
    """Heterogeneous regime: Modality 1 linear, Modality 2 Polynomial, Modality 3 Exponential."""
    if 'noise_level' in kwargs: noise_scale = kwargs['noise_level']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Restore the original stronger signal path
    outcome_signal = torch.randn(n_samples)
    true_u = torch.randn(n_samples, shared_k)
    true_u[:, 0] += outcome_signal * 2.0 
    
    data_matrices = []
    
    # Modality 1: Purely Linear
    x1 = true_u @ torch.randn(shared_k, p_list[0]) + torch.randn(n_samples, p_list[0]) * noise_scale
    data_matrices.append(x1)

    # Modality 2: Polynomial (Quadratic + Sine)
    x2 = 0.5 * (true_u @ torch.randn(shared_k, p_list[1])) + \
         0.5 * ((true_u**2) @ torch.randn(shared_k, p_list[1]) + torch.sin(true_u) @ torch.randn(shared_k, p_list[1]))
    x2 += torch.randn(n_samples, p_list[1]) * noise_scale
    data_matrices.append(x2)

    # Modality 3: Exponential / Sigmoidal
    x3 = 0.5 * (true_u @ torch.randn(shared_k, p_list[2])) + \
         0.5 * (torch.exp(true_u * 0.2) @ torch.randn(shared_k, p_list[2]) + torch.sigmoid(true_u) @ torch.randn(shared_k, p_list[2]))
    x3 += torch.randn(n_samples, p_list[2]) * noise_scale
    data_matrices.append(x3)
    
    return {
        "kind": "nonlinear_shared",
        "data": data_matrices,
        "true_u": true_u,
        "outcome": outcome_signal,
        "shared_k": shared_k
    }

def build_shared_plus_private_case(n_samples: int = 1000, 
                                   shared_k: int = 2, 
                                   private_k: int = 2,
                                   p_list: List[int] = [100, 100],
                                   noise_scale: float = 0.5,
                                   seed: int = 42,
                                   **kwargs) -> Dict[str, Any]:
    """Regime where each modality has its own private signal in addition to shared signal."""
    if 'noise_level' in kwargs: noise_scale = kwargs['noise_level']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    outcome_signal = torch.randn(n_samples)
    true_u_shared = torch.randn(n_samples, shared_k)
    true_u_shared[:, 0] += outcome_signal * 2.0
    
    data_matrices = []
    private_latents = []
    
    for p in p_list:
        true_u_private = torch.randn(n_samples, private_k)
        private_latents.append(true_u_private)
        
        v_sh = torch.randn(shared_k, p)
        v_pr = torch.randn(private_k, p)
        
        x = (true_u_shared @ v_sh) + (true_u_private @ v_pr) + torch.randn(n_samples, p) * noise_scale
        data_matrices.append(x)
        
    return {
        "kind": "shared_plus_private",
        "data": data_matrices,
        "true_u": true_u_shared,
        "true_u_private": private_latents,
        "outcome": outcome_signal,
        "shared_k": shared_k,
        "private_k": private_k
    }

def build_case(kind: str = "nonlinear_shared", **kwargs) -> Dict[str, Any]:
    """Factory for synthetic benchmark cases."""
    builders = {
        "linear_footprint": build_linear_footprint_case,
        "nonlinear_shared": build_nonlinear_shared_case,
        "shared_plus_private": build_shared_plus_private_case
    }
    if kind not in builders:
        raise ValueError(f"Unknown case kind: {kind}")
    return builders[kind](**kwargs)
