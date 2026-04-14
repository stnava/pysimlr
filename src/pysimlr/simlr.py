import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from .svd import ba_svd, safe_pca
from .optimizers import create_optimizer
from .sparsification import orthogonalize_and_q_sparsify, simlr_sparseness
from .utils import (set_seed_based_on_time, adjusted_rvcoef, safe_svd,
                    invariant_orthogonality_defect, l1_normalize_features, orthogonality_summary, preprocess_data)
from .consensus import compute_shared_consensus
from scipy.stats import ttest_1samp

try:
    from sklearn.decomposition import FastICA
except ImportError:
    FastICA = None

def parse_constraint(constraint_str: str) -> Dict[str, Any]:
    parts = constraint_str.split('x')
    constraint_type = parts[0]
    weight = 1.0
    iterations = 1
    if len(parts) > 1:
        try: weight = float(parts[1])
        except ValueError: pass
    if len(parts) > 2:
        try: iterations = int(parts[2])
        except ValueError: pass
    return {"type": constraint_type, "weight": weight, "iterations": iterations}

def project_gradient(v_grad: torch.Tensor, v_current: torch.Tensor, constraint_type: str) -> torch.Tensor:
    """
    Project gradient onto the tangent space of the manifold.
    Following Edelman et al. (1998).
    """
    if constraint_type == "Grassmann":
        # Project onto Grassmann tangent space: G = G - V(V^T G)
        return v_grad - v_current @ (v_current.t() @ v_grad)
    elif constraint_type == "Stiefel":
        # Project onto Stiefel tangent space: G = G - V sym(V^T G)
        vtg = v_current.t() @ v_grad
        sym_vtg = 0.5 * (vtg + vtg.t())
        return v_grad - v_current @ sym_vtg
    return v_grad

def calculate_u(projections: List[torch.Tensor], 
                mixing_algorithm: str = "svd", 
                k: Optional[int] = None,
                orthogonalize: bool = False) -> torch.Tensor:
    """
    Deprecated: Use compute_shared_consensus from .consensus instead.
    """
    return compute_shared_consensus(projections, mixing_algorithm, k, orthogonalize)

def initialize_simlr(data_matrices: List[torch.Tensor], 
                     k: int, 
                     initialization_type: str = "pca",
                     joint_reduction: bool = True) -> List[torch.Tensor]:
    v_mats = []
    for x in data_matrices:
        u, s, v = ba_svd(x, nu=0, nv=k)
        if v.shape[1] < k:
            padding = torch.randn(v.shape[0], k - v.shape[1], dtype=v.dtype, device=v.device) * 1e-4
            v = torch.cat([v, padding], dim=1)
        v_mats.append(v.to(x.dtype))
    return v_mats

def calculate_ica_energy(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nonlinearity: str = "logcosh", a: float = 1.0) -> torch.Tensor:
    s = (u.t() @ x) @ v
    n = x.shape[0]
    if nonlinearity == "logcosh": return -torch.sum(torch.log(torch.cosh(s))) / n
    elif nonlinearity == "exp": return -torch.sum(-torch.exp(-s**2 / 2.0)) / n
    elif nonlinearity == "gauss": return -torch.sum(-0.5 * torch.exp(-a * s**2)) / n
    elif nonlinearity == "kurtosis": return -torch.sum((s**4.0) / 4.0) / n
    return torch.tensor(0.0, dtype=x.dtype, device=x.device)

def calculate_ica_gradient(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nonlinearity: str = "logcosh", a: float = 1.0) -> torch.Tensor:
    s = (u.t() @ x) @ v
    nk = s.shape[0] # k
    if nonlinearity == "logcosh": return (1.0 / nk) * (x.t() @ u @ torch.tanh(s))
    elif nonlinearity == "exp": return (1.0 / nk) * (x.t() @ u @ (s * torch.exp(-s**2 / 2.0)))
    elif nonlinearity == "gauss": return (1.0 / nk) * (x.t() @ u @ (a * s * torch.exp(-a * s**2)))
    elif nonlinearity == "kurtosis": return (1.0 / nk) * (x.t() @ u @ (s**3))
    return torch.zeros_like(v)

def calculate_simlr_energy(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, energy_type: str = "regression", lambda_val: float = 0.0, prior_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    ica_types = ["logcosh", "exp", "gauss", "kurtosis"]
    u = u.to(x.dtype); v = v.to(x.dtype)
    if energy_type == "regression":
        pred = u @ v.t()
        return torch.sum((x - pred)**2)
    elif energy_type == "acc":
        cov = (u.t() @ x @ v) / (x.shape[0] - 1)
        return -torch.sum(torch.abs(cov))
    elif energy_type in ica_types:
        return calculate_ica_energy(x, u, v, nonlinearity=energy_type)
    elif energy_type in ["normalized_correlation", "nc"]:
        proj = x @ v
        corr = torch.sum(u * proj) / (torch.norm(u) * torch.norm(proj) + 1e-10)
        return -corr
    elif energy_type == "dat" and prior_matrix is not None:
        alignment = prior_matrix.to(x.dtype) @ v
        return -lambda_val * torch.sum(alignment**2)
    return torch.tensor(0.0, dtype=u.dtype, device=u.device)

def calculate_simlr_gradient(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, energy_type: str = "regression", lambda_val: float = 0.0, prior_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    ica_types = ["logcosh", "exp", "gauss", "kurtosis"]
    u = u.to(x.dtype); v = v.to(x.dtype)
    if energy_type == "regression": return 2 * (x.t() @ u - v)
    elif energy_type == "acc":
        cov = (u.t() @ x @ v) / (x.shape[0] - 1)
        return (x.t() @ u @ torch.sign(cov)) / (x.shape[0] - 1)
    elif energy_type in ica_types: return calculate_ica_gradient(x, u, v, nonlinearity=energy_type)
    elif energy_type == "dat" and prior_matrix is not None: 
        prior_matrix = prior_matrix.to(x.dtype)
        return 2 * lambda_val * (prior_matrix.t() @ prior_matrix @ v)
    return torch.zeros_like(v)

def simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
          k: int,
          iterations: int = 100,
          optimizer_type: str = "lars",
          energy_type: str = "acc",
          constraint: str = "Stiefel",
          mixing_algorithm: str = "svd",
          sparseness_quantile: float = 0.5,
          positivity: str = "either",
          smoothing_matrices: Optional[List[torch.Tensor]] = None,
          domain_matrices: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
          domain_lambdas: Optional[Union[float, List[float]]] = None,
          orthogonalize_u: bool = False,
          scale_list: List[str] = ["centerAndScale", "np"],
          tol: float = 1e-6,
          verbose: bool = False,
          **opt_params) -> Dict[str, Any]:
    
    # Handle 'sparsity' alias
    if 'sparsity' in opt_params:
        sparseness_quantile = opt_params.pop('sparsity')

    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    
    provenance_list = []
    if scale_list is not None and len(scale_list) > 0 and scale_list[0] != "none":
        scaled_mats = []
        for m in torch_mats:
            m_scaled, prov = preprocess_data(m, scale_list)
            scaled_mats.append(m_scaled)
            provenance_list.append(prov)
        torch_mats = scaled_mats
        
    n_modalities = len(torch_mats)
    orig_dtype = torch_mats[0].dtype
    v_mats = initialize_simlr(torch_mats, k)
    optimizer = create_optimizer(optimizer_type, v_mats, **opt_params)
    
    constraint_info = parse_constraint(constraint)
    constraint_type = constraint_info["type"]
    constraint_weight = constraint_info["weight"]
    constraint_iterations = constraint_info["iterations"]
    
    torch_domains = [torch.as_tensor(dm).float() if dm is not None else None for dm in domain_matrices] if domain_matrices else None
    if isinstance(domain_lambdas, float): domain_lambdas = [domain_lambdas] * n_modalities
    
    energy_history = []
    prev_total_energy = float('inf')
    converged_iter = iterations
    
    normalizing_weights = [1.0] * n_modalities
    orth_weights = [1.0] * n_modalities
    domain_weights = [1.0] * n_modalities
    
    for it in range(iterations):
        projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
        u = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u)
        for i in range(n_modalities):
            # Local energy function that incorporates sparsification/retraction
            def smooth_energy_fn(v_cand):
                v_cand = v_cand.to(orig_dtype)
                if positivity == 'positive': v_cand = torch.abs(v_cand)
                elif positivity == 'negative': v_cand = -torch.abs(v_cand)
                
                v_sp = simlr_sparseness(v_cand, constraint_type=constraint_type, 
                                        smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None, 
                                        positivity=positivity, sparseness_quantile=sparseness_quantile, 
                                        constraint_weight=constraint_weight, constraint_iterations=constraint_iterations, 
                                        energy_type=energy_type)
                
                sim_e = calculate_simlr_energy(v_sp, torch_mats[i], u, energy_type) * normalizing_weights[i]
                dom_e = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_e = calculate_simlr_energy(v_sp, torch_mats[i], u, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]) * domain_weights[i]
                orth_e = 0.0
                if constraint_type == "ortho": orth_e = invariant_orthogonality_defect(v_sp) * constraint_weight * orth_weights[i]
                return (sim_e + dom_e + orth_e).item()

            # Local gradient function that also incorporates manifold projection
            def smooth_gradient_fn(v_curr):
                if positivity == 'positive': v_curr = torch.abs(v_curr)
                
                sim_grad = calculate_simlr_gradient(v_curr, torch_mats[i], u, energy_type) * normalizing_weights[i]
                dom_grad = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_grad = calculate_simlr_gradient(v_curr, torch_mats[i], u, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]) * domain_weights[i]
                
                total_grad = sim_grad + dom_grad
                # Project gradient onto tangent space
                total_grad = project_gradient(total_grad, v_curr, constraint_type)
                
                # Further sparsify/constrain the gradient direction
                total_grad = simlr_sparseness(total_grad, constraint_type=constraint_type, 
                                              smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None, 
                                              positivity=positivity, sparseness_quantile=sparseness_quantile, 
                                              constraint_weight=constraint_weight, constraint_iterations=constraint_iterations, 
                                              energy_type=energy_type)
                return total_grad

            total_grad = smooth_gradient_fn(v_mats[i])
            v_updated = optimizer.step(i, v_mats[i], total_grad, smooth_energy_fn)
            
            # Apply final projection
            v_mats[i] = simlr_sparseness(v_updated, constraint_type=constraint_type, smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None, positivity=positivity, sparseness_quantile=sparseness_quantile, constraint_weight=constraint_weight, constraint_iterations=constraint_iterations, energy_type=energy_type)
            
        if it == 0:
            for i in range(n_modalities):
                sim_e = calculate_simlr_energy(v_mats[i], torch_mats[i], u, energy_type).item()
                normalizing_weights[i] = 1.0 / (abs(sim_e) * n_modalities + 1e-10)
                orth_e = invariant_orthogonality_defect(v_mats[i]).item()
                if orth_e > 1e-10: orth_weights[i] = abs(sim_e) * normalizing_weights[i] / orth_e
                else: orth_weights[i] = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_e_raw = calculate_simlr_energy(v_mats[i], torch_mats[i], u, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]).item()
                    if abs(dom_e_raw) > 1e-10: domain_weights[i] = abs(sim_e * normalizing_weights[i]) / abs(dom_e_raw)
                    else: domain_weights[i] = 1.0
        total_energy = sum(calculate_simlr_energy(v_mats[i], torch_mats[i], u, energy_type).item() * normalizing_weights[i] for i in range(n_modalities))
        energy_history.append(total_energy)
        
        # Check for convergence
        if abs(prev_total_energy - total_energy) < tol * (abs(prev_total_energy) + 1e-10):
            if verbose: print(f"Converged at iteration {it}: Total Energy {total_energy}")
            converged_iter = it + 1
            break
        prev_total_energy = total_energy
        
        if verbose and it % 10 == 0: print(f"Iteration {it}: Total Energy {total_energy}")
        
    # Re-calculate final shared consensus after the last V update
    projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
    u = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u)
    
    v_summaries = [orthogonality_summary(v) for v in v_mats]
    
    # Compute reconstruction weights W_i such that X_i approx U @ W_i
    # W_i = pinv(U) @ X_i
    w_mats = []
    try:
        u_pinv = torch.linalg.pinv(u)
        for x in torch_mats:
            w_mats.append(u_pinv @ x)
    except:
        # Fallback if pinv fails
        for x in torch_mats:
            w_mats.append(torch.zeros(k, x.shape[1], dtype=u.dtype, device=u.device))

    return {
        "u": u, "v": v_mats, "w": w_mats, "energy": energy_history, 
        "normalizing_weights": normalizing_weights, 
        "orth_weights": orth_weights, "domain_weights": domain_weights, 
        "converged_iter": converged_iter, "v_orthogonality": v_summaries,
        "mixing_algorithm": mixing_algorithm,
        "orthogonalize_u": orthogonalize_u,
        "energy_type": energy_type,
        "scale_list": scale_list,
        "provenance_list": provenance_list
    }


def pairwise_matrix_similarity(mat_list: List[torch.Tensor], v_list: List[torch.Tensor]) -> Dict[str, float]:
    n_modalities = len(mat_list); similarities = {}
    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            l_i = mat_list[i] @ v_list[i]; l_j = mat_list[j] @ v_list[j]
            similarities[f"sim_{i}_{j}"] = adjusted_rvcoef(l_i, l_j)
    return similarities

def simlr_perm(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, n_perms: int = 50, verbose: bool = False, **simlr_params) -> Dict[str, Any]:
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    res = simlr(torch_mats, k=k, verbose=verbose, **simlr_params)
    v_norm = [l1_normalize_features(v) for v in res['v']]
    obs_sims = pairwise_matrix_similarity(torch_mats, v_norm)
    perm_results = {k: [v] for k, v in obs_sims.items()}
    for p in range(n_perms):
        mats_perm = [m[torch.randperm(m.shape[0])] for m in torch_mats]
        res_p = simlr(mats_perm, k=k, verbose=False, **simlr_params)
        v_p_norm = [l1_normalize_features(v) for v in res_p['v']]
        sims_p = pairwise_matrix_similarity(mats_perm, v_p_norm)
        for k_sim, v_sim in sims_p.items(): perm_results[k_sim].append(v_sim)
    stats = {}
    for k_sim, vals in perm_results.items():
        obs = vals[0]; null_dist = np.array(vals[1:])
        t_stat, p_val = ttest_1samp(null_dist, obs, alternative='less')
        stats[k_sim] = {"observed": obs, "p_value": p_val, "t_stat": t_stat}
    return {"simlr_result": res, "stats": stats}

def predict_shared_latent(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                          simlr_result: Dict[str, Any]) -> torch.Tensor:
    """
    Compute the shared latent basis U for new data using the trained SIMLR model.
    """
    # 1. Preprocess data matrices
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    scale_list = simlr_result.get('scale_list', [])
    provenance_list = simlr_result.get('provenance_list', [])
    if scale_list and scale_list[0] != "none":
        scaled_mats = []
        for i, m in enumerate(torch_mats):
            prov = provenance_list[i] if i < len(provenance_list) else None
            m_scaled = preprocess_data(m, scale_list, provenance=prov)
            scaled_mats.append(m_scaled)
        torch_mats = scaled_mats

    v_mats = simlr_result['v']
    mixing_alg = simlr_result.get('mixing_algorithm', 'svd')
    orthogonalize_u = simlr_result.get('orthogonalize_u', False)
    k = v_mats[0].shape[1]
    
    # 2. Project to shared space
    projections = [x @ v.to(x.dtype) for x, v in zip(torch_mats, v_mats)]
    
    # 3. Compute consensus U using the original mixing settings
    u_new = compute_shared_consensus(projections, mixing_algorithm=mixing_alg, k=k, orthogonalize=orthogonalize_u)
    return u_new

def reconstruct_from_learned_maps(u: torch.Tensor, 
                                  simlr_result: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Reconstruct each modality from the shared latent basis U using learned weights.
    """
    if 'w' not in simlr_result:
        # For backward compatibility, but this should be avoided
        return []
    
    w_mats = simlr_result['w']
    reconstructions = []
    for w in w_mats:
        x_pred = u @ w.to(u.dtype)
        reconstructions.append(x_pred)
    return reconstructions

def predict_simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                  simlr_result: Dict[str, Any],
                  allow_legacy_refit: bool = False) -> Dict[str, Any]:
    """
    Predict using trained SIMLR model.
    By default, requires learned reconstruction weights 'w' to be present.
    If 'w' is missing, it will raise ValueError unless allow_legacy_refit=True.
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    
    scale_list = simlr_result.get('scale_list', [])
    provenance_list = simlr_result.get('provenance_list', [])
    if scale_list and scale_list[0] != "none":
        scaled_mats = []
        for i, m in enumerate(torch_mats):
            prov = provenance_list[i] if i < len(provenance_list) else None
            m_scaled = preprocess_data(m, scale_list, provenance=prov)
            scaled_mats.append(m_scaled)
        torch_mats = scaled_mats

    if 'model' in simlr_result:
        model = simlr_result['model']; model.eval(); device = next(model.parameters()).device
        torch_mats_device = [m.to(device) for m in torch_mats]
        with torch.no_grad():
            output = model(torch_mats_device)
            latents = output[0]
            reconstructions = output[1]
            u_new = output[2]
        errors = [torch.norm(x - x_pred, p='fro').item() / (torch.norm(x, p='fro').item() + 1e-10) for x, x_pred in zip(torch_mats_device, reconstructions)]
        return {"u": torch.nan_to_num(u_new.cpu()), "latents": [torch.nan_to_num(l.cpu()) for l in latents], "reconstructions": [torch.nan_to_num(r.cpu()) for r in reconstructions], "errors": errors}
    
    # For standard SIMLR:
    u_new = predict_shared_latent(data_matrices, simlr_result)
    
    if 'w' in simlr_result:
        reconstructions = reconstruct_from_learned_maps(u_new, simlr_result)
    elif allow_legacy_refit:
        # Fallback to old suspicious least-squares behavior for backward compatibility
        reconstructions = []
        v_mats = simlr_result['v']
        mixing_alg = simlr_result.get('mixing_algorithm', 'svd')
        orthogonalize_u = simlr_result.get('orthogonalize_u', False)
        for i, x in enumerate(torch_mats):
            u_ortho = orthogonalize_u or (mixing_alg in ["svd", "pca"])
            if u_ortho:
                weights = u_new.t() @ x
                x_pred = u_new @ weights
            else:
                u_pinv = torch.linalg.pinv(u_new)
                weights = u_pinv @ x
                x_pred = u_new @ weights
            reconstructions.append(x_pred)
    else:
        raise ValueError("Learned reconstruction weights 'w' missing from simlr_result. "
                         "Legacy refit is disabled by default. Set allow_legacy_refit=True to enable.")
    
    errors = []
    for i, x in enumerate(torch_mats):
        x_pred = reconstructions[i]
        err = torch.norm(x - x_pred, p='fro').item() / (torch.norm(x, p='fro').item() + 1e-10)
        errors.append(err)
        
    return {"u": torch.nan_to_num(u_new), "reconstructions": reconstructions, "errors": errors}


def estimate_rank(data_matrices: List[Union[torch.Tensor, np.ndarray]], n_permutations: int = 20, var_threshold: float = 0.99) -> int:
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    n_modalities = len(torch_mats); k_max_list = []
    for x in torch_mats:
        x_centered = x - torch.mean(x, dim=0); _, s, _ = safe_svd(x_centered, full_matrices=False)
        eigenvalues = s**2; prop_var = torch.cumsum(eigenvalues, dim=0) / (torch.sum(eigenvalues) + 1e-10)
        k_max_list.append(torch.where(prop_var >= var_threshold)[0][0].item() + 1)
    k_max = min(k_max_list) if k_max_list else 1
    if k_max < 1: k_max = 1
    def calculate_rv_curve(mats, km):
        u_list = [safe_svd(m, full_matrices=False)[0][:, :km] for m in mats]
        scores = []
        for curr_k in range(1, km + 1):
            mod_scores = []
            for i in range(n_modalities):
                y_target = u_list[i][:, :curr_k]; other_inds = [j for j in range(n_modalities) if j != i]
                u_other = torch.cat([u_list[j][:, :curr_k] for j in other_inds], dim=1)
                consensus, _, _ = safe_svd(u_other, full_matrices=False)
                consensus = consensus[:, :curr_k]; mod_scores.append(adjusted_rvcoef(y_target, consensus))
            scores.append(np.mean(mod_scores))
        return scores
    proc_mats = [(m - torch.mean(m, dim=0)) / (torch.norm(m - torch.mean(m, dim=0), p='fro') + 1e-10) for m in torch_mats]
    real_curve = calculate_rv_curve(proc_mats, k_max)
    if n_permutations > 0 and n_modalities >= 2:
        null_curves = []
        for _ in range(n_permutations):
            perm_mats = [proc_mats[0]] + [m[torch.randperm(m.shape[0])] for m in proc_mats[1:]]
            null_curves.append(calculate_rv_curve(perm_mats, k_max))
        null_curve_mean = np.mean(null_curves, axis=0); signal = np.array(real_curve) - null_curve_mean; optimal_k = np.argmax(signal) + 1
    else:
        if len(real_curve) < 3: optimal_k = 1
        else:
            y = np.array(real_curve); x_vals = np.linspace(0, 1, len(y)); y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
            optimal_k = np.argmax(y_norm - x_vals) + 1
    return int(optimal_k)

def decompose_energy(data_matrices: List[Union[torch.Tensor, np.ndarray]], simlr_result: Dict[str, Any], energy_type: str = "acc") -> Dict[str, Any]:
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    u = simlr_result['u']; v_mats = simlr_result['v']
    modality_energies = []; feature_importances = []
    for i, (x, v) in enumerate(zip(torch_mats, v_mats)):
        mod_energy = calculate_simlr_energy(v, x, u, energy_type).item()
        grad = calculate_simlr_gradient(v, x, u, energy_type)
        feat_imp = torch.sum(torch.abs(grad), dim=1).numpy()
        modality_energies.append(mod_energy); feature_importances.append(feat_imp)
    return {"modality_energies": modality_energies, "feature_importances": feature_importances}
