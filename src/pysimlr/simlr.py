import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Callable
from .svd import ba_svd, safe_pca
from .optimizers import create_optimizer
from .sparsification import orthogonalize_and_q_sparsify
from .utils import set_seed_based_on_time, adjusted_rvcoef

def initialize_simlr(data_matrices: List[torch.Tensor], 
                     k: int, 
                     initialization_type: str = "pca") -> List[torch.Tensor]:
    """
    Initialize V matrices for SIMLR using torch.
    """
    v_mats = []
    for x in data_matrices:
        u, s, v = ba_svd(x, nu=0, nv=k)
        if v.shape[1] < k:
            padding = torch.randn(v.shape[0], k - v.shape[1]) * 1e-4
            v = torch.cat([v, padding], dim=1)
        v_mats.append(v)
    return v_mats

def calculate_ica_energy(x: torch.Tensor, 
                         u: torch.Tensor, 
                         v: torch.Tensor, 
                         nonlinearity: str = "logcosh", 
                         a: float = 1.0) -> torch.Tensor:
    """
    Computes ICA energy based on nonlinearity.
    S = U^T X V
    """
    s = (u.t() @ x) @ v
    n = x.shape[0]
    
    if nonlinearity == "logcosh":
        return -torch.sum(torch.log(torch.cosh(s))) / n
    elif nonlinearity == "exp":
        return -torch.sum(-torch.exp(-s**2 / 2.0)) / n
    elif nonlinearity == "gauss":
        return -torch.sum(-0.5 * torch.exp(-a * s**2)) / n
    elif nonlinearity == "kurtosis":
        return -torch.sum((s**4.0) / 4.0) / n
    return torch.tensor(0.0)

def calculate_ica_gradient(x: torch.Tensor, 
                           u: torch.Tensor, 
                           v: torch.Tensor, 
                           nonlinearity: str = "logcosh", 
                           a: float = 1.0) -> torch.Tensor:
    """
    Computes analytical gradient for ICA energy wrt V.
    """
    s = (u.t() @ x) @ v
    nk = s.shape[0] # k
    
    if nonlinearity == "logcosh":
        return (1.0 / nk) * (x.t() @ u @ torch.tanh(s))
    elif nonlinearity == "exp":
        return (1.0 / nk) * (x.t() @ u @ (s * torch.exp(-s**2 / 2.0)))
    elif nonlinearity == "gauss":
        return (1.0 / nk) * (x.t() @ u @ (a * s * torch.exp(-a * s**2)))
    elif nonlinearity == "kurtosis":
        return (1.0 / nk) * (x.t() @ u @ (s**3))
    return torch.zeros_like(v)

def calculate_simlr_energy(v_mats: List[torch.Tensor], 
                           x_mats: List[torch.Tensor], 
                           u: torch.Tensor, 
                           energy_type: str = "regression",
                           domain_matrices: Optional[List[torch.Tensor]] = None,
                           domain_lambdas: Optional[List[float]] = None) -> torch.Tensor:
    """
    Calculate SIMLR energy in torch, including domain knowledge and ICA types.
    """
    total_energy = torch.tensor(0.0)
    ica_types = ["logcosh", "exp", "gauss", "kurtosis"]
    
    for i in range(len(v_mats)):
        v = v_mats[i]
        x = x_mats[i]
        
        if energy_type == "regression":
            pred = u @ v.t()
            total_energy += torch.sum((x - pred)**2)
        elif energy_type == "acc":
            cov = (u.t() @ x @ v) / (x.shape[0] - 1)
            total_energy -= torch.sum(torch.abs(cov))
        elif energy_type in ica_types:
            total_energy += calculate_ica_energy(x, u, v, nonlinearity=energy_type)
            
        if domain_matrices is not None and domain_lambdas is not None:
            if i < len(domain_matrices) and domain_matrices[i] is not None:
                alignment = domain_matrices[i] @ v
                total_energy -= domain_lambdas[i] * torch.sum(alignment**2)
                
    return total_energy

def simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
          k: int,
          iterations: int = 10,
          optimizer_type: str = "hybrid_adam",
          energy_type: str = "regression",
          sparseness_quantile: float = 0.5,
          positivity: str = "either",
          domain_matrices: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
          domain_lambdas: Optional[Union[float, List[float]]] = None,
          verbose: bool = False,
          **opt_params) -> Dict[str, Any]:
    """
    SIMLR: Structured Identification of Multimodal Low-rank Relationships.
    """
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    n_modalities = len(torch_mats)
    
    if domain_matrices is not None:
        torch_domains = []
        for dm in domain_matrices:
            if dm is not None:
                torch_domains.append(dm if isinstance(dm, torch.Tensor) else torch.from_numpy(dm).float())
            else:
                torch_domains.append(None)
    else:
        torch_domains = None
        
    if domain_lambdas is None:
        domain_lambdas = [0.0] * n_modalities
    elif isinstance(domain_lambdas, (float, int)):
        domain_lambdas = [float(domain_lambdas)] * n_modalities
        
    v_mats = initialize_simlr(torch_mats, k)
    optimizer = create_optimizer(optimizer_type, v_mats, **opt_params)
    energy_history = []
    ica_types = ["logcosh", "exp", "gauss", "kurtosis"]
    
    for it in range(iterations):
        projections = [x @ v for v, x in zip(v_mats, torch_mats)]
        u = torch.mean(torch.stack(projections), dim=0)
        u, _, _ = torch.linalg.svd(u, full_matrices=False)
        
        for i in range(n_modalities):
            def energy_fn(v_cand):
                mod_energy = torch.tensor(0.0)
                if energy_type == "regression":
                    mod_energy += torch.sum((torch_mats[i] - u @ v_cand.t())**2)
                elif energy_type == "acc":
                    cov = (u.t() @ torch_mats[i] @ v_cand) / (torch_mats[i].shape[0] - 1)
                    mod_energy -= torch.sum(torch.abs(cov))
                elif energy_type in ica_types:
                    mod_energy += calculate_ica_energy(torch_mats[i], u, v_cand, nonlinearity=energy_type)
                
                if torch_domains is not None and i < len(torch_domains) and torch_domains[i] is not None:
                    alignment = torch_domains[i] @ v_cand
                    mod_energy -= domain_lambdas[i] * torch.sum(alignment**2)
                return mod_energy.item()
            
            if energy_type == "regression":
                descent_grad = 2 * (torch_mats[i].t() @ u - v_mats[i])
            elif energy_type == "acc":
                cov = (u.t() @ torch_mats[i] @ v_mats[i]) / (torch_mats[i].shape[0] - 1)
                descent_grad = (torch_mats[i].t() @ u @ torch.sign(cov)) / (torch_mats[i].shape[0] - 1)
            elif energy_type in ica_types:
                descent_grad = calculate_ica_gradient(torch_mats[i], u, v_mats[i], nonlinearity=energy_type)
            else:
                descent_grad = torch.zeros_like(v_mats[i])
                
            if torch_domains is not None and i < len(torch_domains) and torch_domains[i] is not None:
                dom_grad = 2 * domain_lambdas[i] * (torch_domains[i].t() @ torch_domains[i] @ v_mats[i])
                descent_grad += dom_grad
            
            v_mats[i] = optimizer.step(i, v_mats[i], descent_grad, energy_fn)
            v_mats[i] = orthogonalize_and_q_sparsify(v_mats[i], sparseness_quantile, positivity)
            
        energy = calculate_simlr_energy(v_mats, torch_mats, u, energy_type, torch_domains, domain_lambdas)
        energy_history.append(energy.item())
        if verbose and it % 5 == 0:
            print(f"Iteration {it}: Energy {energy.item()}")
            
    return {
        "u": u,
        "v": v_mats,
        "energy": energy_history
    }

def predict_simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                   simlr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a trained SIMLR model (linear or deep) to new data.
    """
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    
    if 'model' in simlr_result:
        model = simlr_result['model']
        model.eval()
        device = next(model.parameters()).device
        torch_mats_device = [m.to(device) for m in torch_mats]
        
        with torch.no_grad():
            latents, reconstructions = model(torch_mats_device)
            u_new = torch.mean(torch.stack(latents), dim=0)
            
        errors = []
        for x, x_pred in zip(torch_mats_device, reconstructions):
            err = torch.norm(x - x_pred, p='fro') / torch.norm(x, p='fro')
            errors.append(err.item())
            
        return {
            "u": torch.nan_to_num(u_new.cpu()),
            "latents": [torch.nan_to_num(l.cpu()) for l in latents],
            "reconstructions": [torch.nan_to_num(r.cpu()) for r in reconstructions],
            "errors": errors
        }
    
    v_mats = simlr_result['v']
    projections = [x @ v for x, v in zip(torch_mats, v_mats)]
    u_new = torch.mean(torch.stack(projections), dim=0)
    u_new, _, _ = torch.linalg.svd(u_new, full_matrices=False)
    reconstructions = [u_new @ v.t() for v in v_mats]
    errors = []
    for x, x_pred in zip(torch_mats, reconstructions):
        err = torch.norm(x - x_pred, p='fro') / torch.norm(x, p='fro')
        errors.append(err.item())
    return {"u": torch.nan_to_num(u_new), "reconstructions": reconstructions, "errors": errors}

def estimate_rank(data_matrices: List[Union[torch.Tensor, np.ndarray]],
                  n_permutations: int = 20,
                  var_threshold: float = 0.99) -> int:
    """
    Estimate the optimal number of components (k) for SIMLR.
    """
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    n_modalities = len(torch_mats)
    k_max_list = []
    for x in torch_mats:
        x_centered = x - torch.mean(x, dim=0)
        _, s, _ = torch.linalg.svd(x_centered, full_matrices=False)
        eigenvalues = s**2
        prop_var = torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues)
        k_max_list.append(torch.where(prop_var >= var_threshold)[0][0].item() + 1)
    k_max = min(k_max_list)
    if k_max < 1: k_max = 1
    def calculate_rv_curve(mats, km):
        u_list = [torch.linalg.svd(m, full_matrices=False)[0][:, :km] for m in mats]
        scores = []
        for k in range(1, km + 1):
            mod_scores = []
            for i in range(n_modalities):
                y_target = u_list[i][:, :k]
                other_inds = [j for j in range(n_modalities) if j != i]
                u_other = torch.cat([u_list[j][:, :k] for j in other_inds], dim=1)
                consensus, _, _ = torch.linalg.svd(u_other, full_matrices=False)
                consensus = consensus[:, :k]
                mod_scores.append(adjusted_rvcoef(y_target, consensus))
            scores.append(np.mean(mod_scores))
        return scores
    proc_mats = []
    for x in torch_mats:
        xc = x - torch.mean(x, dim=0)
        norm = torch.norm(xc, p='fro')
        proc_mats.append(xc / norm if norm > 1e-10 else xc)
    real_curve = calculate_rv_curve(proc_mats, k_max)
    if n_permutations > 0 and n_modalities >= 2:
        null_curves = []
        for _ in range(n_permutations):
            perm_mats = [proc_mats[0]]
            for j in range(1, n_modalities):
                perm_mats.append(proc_mats[j][torch.randperm(proc_mats[j].shape[0])])
            null_curves.append(calculate_rv_curve(perm_mats, k_max))
        null_curve_mean = np.mean(null_curves, axis=0)
        signal = np.array(real_curve) - null_curve_mean
        optimal_k = np.argmax(signal) + 1
    else:
        if len(real_curve) < 3:
            optimal_k = 1
        else:
            y = np.array(real_curve)
            x_vals = np.linspace(0, 1, len(y))
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
            distances = y_norm - x_vals
            optimal_k = np.argmax(distances) + 1
    return int(optimal_k)

def decompose_energy(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                     simlr_result: Dict[str, Any],
                     energy_type: str = "regression") -> Dict[str, Any]:
    """
    Quantifies the energy contribution of each modality and each feature.
    """
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    u = simlr_result['u']
    v_mats = simlr_result['v']
    
    modality_energies = []
    feature_importances = []
    
    for i, (x, v) in enumerate(zip(torch_mats, v_mats)):
        if energy_type == "regression":
            # Reconstruct and find per-feature MSE
            pred = u @ v.t()
            sq_err = (x - pred)**2
            mod_energy = torch.sum(sq_err).item()
            feat_imp = torch.sum(sq_err, dim=0) # Total error contribution per feature
        elif energy_type == "acc":
            cov = (u.t() @ x @ v) / (x.shape[0] - 1)
            mod_energy = -torch.sum(torch.abs(cov)).item()
            # Feature importance for ACC: contribution to covariance
            feat_imp = torch.sum(torch.abs(u.t() @ x), dim=0) # Approximate
        else:
            mod_energy = 0.0
            feat_imp = torch.zeros(x.shape[1])
            
        modality_energies.append(mod_energy)
        feature_importances.append(feat_imp)
        
    return {
        "modality_energies": modality_energies,
        "feature_importances": feature_importances
    }
