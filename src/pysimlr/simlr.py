import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Callable
from .svd import ba_svd, safe_pca
from .optimizers import create_optimizer
from .sparsification import orthogonalize_and_q_sparsify
from .utils import set_seed_based_on_time

def initialize_simlr(data_matrices: List[torch.Tensor], 
                     k: int, 
                     initialization_type: str = "pca") -> List[torch.Tensor]:
    """
    Initialize V matrices for SIMLR using torch.
    """
    v_mats = []
    for x in data_matrices:
        # Simple PCA initialization
        _, _, v = ba_svd(x, nu=0, nv=k)
        # v is p x k
        v_mats.append(v)
    return v_mats

def calculate_simlr_energy(v_mats: List[torch.Tensor], 
                           x_mats: List[torch.Tensor], 
                           u: torch.Tensor, 
                           energy_type: str = "regression") -> torch.Tensor:
    """
    Calculate SIMLR energy in torch.
    """
    total_energy = torch.tensor(0.0)
    for v, x in zip(v_mats, x_mats):
        if energy_type == "regression":
            # energy = || X - U V^T ||^2
            pred = u @ v.t()
            total_energy += torch.sum((x - pred)**2)
    return total_energy

def simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
          k: int,
          iterations: int = 10,
          optimizer_type: str = "hybrid_adam",
          energy_type: str = "regression",
          sparseness_quantile: float = 0.5,
          positivity: str = "either",
          verbose: bool = False,
          **opt_params) -> Dict[str, Any]:
    """
    SIMLR: Structured Identification of Multimodal Low-rank Relationships using torch.
    """
    # Ensure inputs are torch tensors
    torch_mats = []
    for m in data_matrices:
        if not isinstance(m, torch.Tensor):
            torch_mats.append(torch.from_numpy(m).float())
        else:
            torch_mats.append(m.float())
    
    # 1. Initialize
    v_mats = initialize_simlr(torch_mats, k)
    
    # 2. Setup Optimizer
    optimizer = create_optimizer(optimizer_type, v_mats, **opt_params)
    
    # 3. Optimization loop
    energy_history = []
    
    for it in range(iterations):
        # Update U (shared latent space)
        projections = []
        for v, x in zip(v_mats, torch_mats):
            # x is n x p, v is p x k. x @ v is n x k
            projections.append(x @ v)
        
        u = torch.mean(torch.stack(projections), dim=0)
        # Orthogonalize U using SVD
        u_orth, _, _ = torch.linalg.svd(u, full_matrices=False)
        u = u_orth
        
        # Update each V_i
        for i in range(len(v_mats)):
            def energy_fn(v_cand):
                pred = u @ v_cand.t()
                return torch.sum((torch_mats[i] - pred)**2).item()
            
            # Descent gradient for regression: 2 * (X^T U - V)
            # (Assuming U^T U = I)
            descent_grad = 2 * (torch_mats[i].t() @ u - v_mats[i])
            
            v_mats[i] = optimizer.step(i, v_mats[i], descent_grad, energy_fn)
            
            # Regularize/Sparsify V
            v_mats[i] = orthogonalize_and_q_sparsify(v_mats[i], 
                                                     sparseness_quantile, 
                                                     positivity)
            
        energy = calculate_simlr_energy(v_mats, torch_mats, u, energy_type)
        energy_history.append(energy.item())
        
        if verbose:
            print(f"Iteration {it}: Energy {energy.item()}")
            
    return {
        "u": u,
        "v": v_mats,
        "energy": energy_history
    }
