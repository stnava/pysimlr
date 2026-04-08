import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from .simlr import initialize_simlr
from .optimizers import create_optimizer
from .sparsification import orthogonalize_and_q_sparsify
from .utils import adjusted_rvcoef

def simlr_path(data_matrices: List[Union[torch.Tensor, np.ndarray]],
               k: int,
               path_model: List[List[int]],
               iterations: int = 20,
               optimizer_type: str = "hybrid_adam",
               energy_type: str = "regression",
               sparseness_quantile: float = 0.5,
               positivity: str = "either",
               verbose: bool = False,
               **opt_params) -> Dict[str, Any]:
    """
    SiMLR with Path Modeling: Define structured relationships between modalities.
    """
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    n_modalities = len(torch_mats)
    
    v_mats = initialize_simlr(torch_mats, k)
    u_mats = [x @ v for x, v in zip(torch_mats, v_mats)]
    for i in range(n_modalities):
        u_mats[i], _, _ = torch.linalg.svd(u_mats[i], full_matrices=False)
        u_mats[i] = u_mats[i][:, :k]

    optimizer = create_optimizer(optimizer_type, v_mats, **opt_params)
    energy_history = []
    
    for it in range(iterations):
        new_u_mats = []
        for i in range(n_modalities):
            self_proj = torch_mats[i] @ v_mats[i]
            connected_indices = path_model[i]
            if connected_indices:
                others = [u_mats[j] for j in connected_indices if j < n_modalities]
                if others:
                    target = self_proj + torch.sum(torch.stack(others), dim=0)
                else:
                    target = self_proj
            else:
                target = self_proj
            u_i_new, _, _ = torch.linalg.svd(target, full_matrices=False)
            new_u_mats.append(u_i_new[:, :k])
        u_mats = new_u_mats
        
        total_energy = 0.0
        for i in range(n_modalities):
            u_i = u_mats[i]
            def energy_fn(v_cand):
                pred = u_i @ v_cand.t()
                return torch.sum((torch_mats[i] - pred)**2).item()
            descent_grad = 2 * (torch_mats[i].t() @ u_i - v_mats[i])
            v_mats[i] = optimizer.step(i, v_mats[i], descent_grad, energy_fn)
            v_mats[i] = orthogonalize_and_q_sparsify(v_mats[i], sparseness_quantile, positivity)
            total_energy += energy_fn(v_mats[i])
            
        energy_history.append(total_energy)
        if verbose and it % 5 == 0:
            print(f"Iteration {it}: Path Energy {total_energy}")
            
    return {
        "u": u_mats,
        "v": v_mats,
        "energy": energy_history
    }

def permutation_test(data_matrices: List[Union[torch.Tensor, np.ndarray]],
                     k: int,
                     n_permutations: int = 50,
                     simlr_fn: Optional[Callable] = None,
                     **simlr_kwargs) -> Dict[str, Any]:
    """
    Statistically robust permutation testing for multi-modal relationships.
    """
    if simlr_fn is None:
        from .simlr import simlr as simlr_fn
        
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    n_modalities = len(torch_mats)
    
    obs_res = simlr_fn(torch_mats, k=k, **simlr_kwargs)
    
    def get_mean_rv(u_list):
        if not isinstance(u_list, list):
            v_mats = obs_res['v']
            u_projs = [x @ v for x, v in zip(torch_mats, v_mats)]
            u_list = u_projs
        rvs = []
        for i in range(len(u_list)):
            for j in range(i+1, len(u_list)):
                rvs.append(adjusted_rvcoef(u_list[i], u_list[j]))
        return np.mean(rvs) if rvs else 0.0

    obs_metric = get_mean_rv(obs_res['u'])
    
    null_metrics = []
    for p in range(n_permutations):
        perm_mats = [torch_mats[0]]
        for j in range(1, n_modalities):
            perm_mats.append(torch_mats[j][torch.randperm(torch_mats[j].shape[0])])
        perm_res = simlr_fn(perm_mats, k=k, **simlr_kwargs)
        v_perm = perm_res['v']
        u_perm_projs = [x @ v for x, v in zip(perm_mats, v_perm)]
        null_metrics.append(get_mean_rv(u_perm_projs))
        
    null_metrics = np.array(null_metrics)
    p_value = (np.sum(null_metrics >= obs_metric) + 1) / (n_permutations + 1)
    z_score = (obs_metric - np.mean(null_metrics)) / (np.std(null_metrics) + 1e-10)
    
    return {
        "p_value": p_value,
        "z_score": z_score,
        "observed_metric": obs_metric,
        "null_metrics": null_metrics
    }
