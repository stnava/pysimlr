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
    """
    Parse a constraint string into type, weight, and iterations.

    Parameters
    ----------
    constraint_str : str
        String in format "typexweightxiterations" (e.g., "Stiefelx0.1x5").

    Returns
    -------
    dict
        Dictionary containing 'type' (str), 'weight' (float), and 'iterations' (int).
    """
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
    Project Euclidean gradient onto the tangent space of the manifold.

    Parameters
    ----------
    v_grad : torch.Tensor
        The Euclidean gradient of the energy function.
    v_current : torch.Tensor
        The current weight matrix V.
    constraint_type : str
        The manifold type ("Stiefel" or "Grassmann").

    Returns
    -------
    torch.Tensor
        The projected Riemannian gradient.
    """
    if constraint_type == "Grassmann":
        return v_grad - v_current @ (v_current.t() @ v_grad)
    elif constraint_type == "Stiefel":
        vtg = v_current.t() @ v_grad
        sym_vtg = 0.5 * (vtg + vtg.t())
        return v_grad - v_current @ sym_vtg
    return v_grad

def calculate_u(projections: List[torch.Tensor], 
                mixing_algorithm: str = "svd", 
                k: Optional[int] = None,
                orthogonalize: bool = False) -> torch.Tensor:
    """
    Aggregate view-specific projections into a shared consensus space U.

    Parameters
    ----------
    projections : list of torch.Tensor
        Latent projections Z_m from each modality.
    mixing_algorithm : str, default="svd"
        Algorithm to use ("avg", "pca", "ica", "newton").
    k : int, optional
        Target latent dimension.
    orthogonalize : bool, default=False
        Whether to force the final U to be orthonormal.

    Returns
    -------
    torch.Tensor
        The shared consensus matrix U.
    """
    return compute_shared_consensus(projections, mixing_algorithm, k, orthogonalize)

def initialize_simlr(data_matrices: List[torch.Tensor], 
                     k: int, 
                     initialization_type: str = "pca") -> List[torch.Tensor]:
    """
    Initialize the SiMLR weight matrices V using SVD/PCA.

    Parameters
    ----------
    data_matrices : list of torch.Tensor
        The input data matrices X_m.
    k : int
        The target latent dimension.
    initialization_type : str, default="pca"
        Currently only supports "pca" (SVD-based).

    Returns
    -------
    list of torch.Tensor
        Initialized projection matrices V_m.
    """
    v_mats = []
    for x in data_matrices:
        u, s, v = ba_svd(x, nu=0, nv=k)
        if v.shape[1] < k:
            padding = torch.randn(v.shape[0], k - v.shape[1], dtype=v.dtype, device=v.device) * 1e-4
            v = torch.cat([v, padding], dim=1)
        v_mats.append(v.to(x.dtype))
    return v_mats

def calculate_ica_energy(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nonlinearity: str = "logcosh", a: float = 1.0) -> torch.Tensor:
    """
    Compute the ICA-based energy using non-Gaussianity measures.

    Parameters
    ----------
    x, u, v : torch.Tensor
        Input modality, consensus, and weight matrix.
    nonlinearity : str, default="logcosh"
        Contrast function ("logcosh", "exp", "gauss", "kurtosis").
    a : float, default=1.0
        Nonlinearity scaling parameter.

    Returns
    -------
    torch.Tensor
        Scalar energy value.
    """
    s = (u.t() @ x) @ v
    n = x.shape[0]
    if nonlinearity == "logcosh": return -torch.sum(torch.log(torch.cosh(s))) / n
    elif nonlinearity == "exp": return -torch.sum(-torch.exp(-s**2 / 2.0)) / n
    elif nonlinearity == "gauss": return -torch.sum(-0.5 * torch.exp(-a * s**2)) / n
    elif nonlinearity == "kurtosis": return -torch.sum((s**4.0) / 4.0) / n
    return torch.tensor(0.0, dtype=x.dtype, device=x.device)

def calculate_ica_gradient(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, nonlinearity: str = "logcosh", a: float = 1.0) -> torch.Tensor:
    """Compute the gradient of the ICA energy."""
    s = (u.t() @ x) @ v
    nk = s.shape[0] # k
    if nonlinearity == "logcosh": return (1.0 / nk) * (x.t() @ u @ torch.tanh(s))
    elif nonlinearity == "exp": return (1.0 / nk) * (x.t() @ u @ (s * torch.exp(-s**2 / 2.0)))
    elif nonlinearity == "gauss": return (1.0 / nk) * (x.t() @ u @ (a * s * torch.exp(-a * s**2)))
    elif nonlinearity == "kurtosis": return (1.0 / nk) * (x.t() @ u @ (s**3))
    return torch.zeros_like(v)

def calculate_simlr_energy(v: torch.Tensor, x: torch.Tensor, u: torch.Tensor, energy_type: str = "regression", lambda_val: float = 0.0, prior_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the specific SiMLR energy for a single modality.

    Parameters
    ----------
    v : torch.Tensor
        The weight matrix for the current modality.
    x : torch.Tensor
        The input data for the current modality.
    u : torch.Tensor
        The shared consensus matrix.
    energy_type : str, default="regression"
        The loss function type ("regression", "acc", "nc", "logcosh", etc.).
    lambda_val : float, default=0.0
        Regularization strength for prior alignment.
    prior_matrix : torch.Tensor, optional
        Target matrix for domain alignment.

    Returns
    -------
    torch.Tensor
        Scalar energy.
    """
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
    """Compute the gradient of the SiMLR energy."""
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
    """
    Similarity-driven Multi-view Linear Reconstruction (SiMLR).

    This is the core linear multi-view learning entry point. It discovers a shared latent 
    consensus U and view-specific linear projectors V_m that maximize cross-modality similarity.

    Parameters
    ----------
    data_matrices : list of (Tensor or ndarray)
        List of data matrices X_m, one per modality.
    k : int
        Target dimension of the shared latent space.
    iterations : int, default=100
        Maximum number of optimization iterations.
    optimizer_type : str, default="lars"
        Optimizer to use ("lars", "adam", "sgd").
    energy_type : str, default="acc"
        The alignment objective ("acc", "regression", "nc", "logcosh").
    constraint : str, default="Stiefel"
        Manifold constraint string (e.g., "Stiefelx0.1x5").
    mixing_algorithm : str, default="svd"
        Consensus method ("avg", "pca", "ica", "newton").
    sparseness_quantile : float, default=0.5
        Proportion of weights to zero-out in V (0.0 to 1.0).
    positivity : str, default="either"
        Force non-negative weights ("positive", "hard") or allow signed ("either").
    smoothing_matrices : list of Tensor, optional
        Prior graph Laplacian or smoothing kernels for V.
    domain_matrices : list of Tensor, optional
        Feature-level prior information.
    domain_lambdas : float or list, optional
        Strength of domain alignment.
    orthogonalize_u : bool, default=False
        Force the shared consensus U to be orthonormal.
    scale_list : list of str, default=["centerAndScale", "np"]
        Preprocessing steps applied to each modality.
    tol : float, default=1e-6
        Convergence tolerance for total energy change.
    verbose : bool, default=False
        Print optimization progress.
    **opt_params
        Additional parameters passed to the optimizer.

    Returns
    -------
    dict
        A dictionary containing:
        - 'u': Shared latent consensus matrix.
        - 'v': List of projection matrices (loadings).
        - 'w': List of reconstruction weights.
        - 'energy': History of the total energy function.
        - 'v_orthogonality': Diagnostics for manifold adherence.
        - 'scale_list', 'provenance_list': Metadata for prediction.
    """
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

    n_modalities = len(torch_mats); orig_dtype = torch_mats[0].dtype

    c_info = parse_constraint(constraint)
    constraint_type, constraint_weight, constraint_iterations = c_info['type'], c_info['weight'], c_info['iterations']
    if domain_matrices is not None:
        torch_domains = [torch.as_tensor(dm).to(orig_dtype) if dm is not None else None for dm in domain_matrices]
    else: torch_domains = None
    if domain_lambdas is None: domain_lambdas = [0.0] * n_modalities
    elif isinstance(domain_lambdas, (float, int)): domain_lambdas = [float(domain_lambdas)] * n_modalities
    v_mats = initialize_simlr(torch_mats, k)
    optimizer = create_optimizer(optimizer_type, v_mats, **opt_params)
    normalizing_weights = torch.ones(n_modalities, dtype=orig_dtype)
    orth_weights = torch.zeros(n_modalities, dtype=orig_dtype)
    domain_weights = torch.ones(n_modalities, dtype=orig_dtype)
    energy_history = []
    
    prev_total_energy = float('inf')
    converged_iter = iterations

    for it in range(iterations):
        projections = [x @ v.to(orig_dtype) for v, x in zip(v_mats, torch_mats)]
        u = compute_shared_consensus(projections, mixing_algorithm=mixing_algorithm, k=k, orthogonalize=orthogonalize_u)
        for i in range(n_modalities):
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

            def smooth_gradient_fn(v_curr):
                if positivity == 'positive': v_curr = torch.abs(v_curr)
                sim_grad = calculate_simlr_gradient(v_curr, torch_mats[i], u, energy_type) * normalizing_weights[i]
                dom_grad = 0.0
                if torch_domains is not None and torch_domains[i] is not None:
                    dom_grad = calculate_simlr_gradient(v_curr, torch_mats[i], u, "dat", lambda_val=domain_lambdas[i], prior_matrix=torch_domains[i]) * domain_weights[i]
                total_grad = sim_grad + dom_grad
                total_grad = project_gradient(total_grad, v_curr, constraint_type)
                total_grad = simlr_sparseness(total_grad, constraint_type=constraint_type, 
                                              smoothing_matrix=smoothing_matrices[i] if smoothing_matrices else None, 
                                              positivity=positivity, sparseness_quantile=sparseness_quantile, 
                                              constraint_weight=constraint_weight, constraint_iterations=constraint_iterations, 
                                              energy_type=energy_type)
                return total_grad

            total_grad = smooth_gradient_fn(v_mats[i])
            v_updated = optimizer.step(i, v_mats[i], total_grad, smooth_energy_fn)
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
        if abs(prev_total_energy - total_energy) < tol * (abs(prev_total_energy) + 1e-10):
            if verbose: print(f"Converged at iteration {it}: Total Energy {total_energy}")
            converged_iter = it + 1; break
        prev_total_energy = total_energy
        if verbose and it % 10 == 0: print(f"Iteration {it}: Total Energy {total_energy}")
        
    v_summaries = [orthogonality_summary(v) for v in v_mats]
    w_mats = []
    try:
        u_pinv = torch.linalg.pinv(u)
        for x in torch_mats: w_mats.append(u_pinv @ x)
    except:
        for x in torch_mats: w_mats.append(torch.zeros(k, x.shape[1], dtype=u.dtype, device=u.device))

    return {
        "u": u, "v": v_mats, "w": w_mats, "energy": energy_history, 
        "normalizing_weights": normalizing_weights, 
        "orth_weights": orth_weights, "domain_weights": domain_weights, 
        "converged_iter": converged_iter, "v_orthogonality": v_summaries,
        "mixing_algorithm": mixing_algorithm, "orthogonalize_u": orthogonalize_u,
        "energy_type": energy_type, "scale_list": scale_list, "provenance_list": provenance_list
    }

def simlr_perm(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, n_perms: int = 50, verbose: bool = False, **simlr_params) -> Dict[str, Any]:
    """
    Perform a permutation significance test for the shared latent structure.

    Parameters
    ----------
    data_matrices : list of Tensor
        Input modalities.
    k : int
        Latent dimension.
    n_perms : int, default=50
        Number of permutations.
    verbose : bool, default=False
        Progress printing.
    **simlr_params
        Parameters passed to the underlying simlr call.

    Returns
    -------
    dict
        Dictionary containing 'simlr_result' and 'stats' (p-values, t-stats).
    """
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
    Map new data into the learned shared latent space U.

    Parameters
    ----------
    data_matrices : list of Tensor
        New data matrices (must match feature dimension of training data).
    simlr_result : dict
        The dictionary returned by a previous simlr() call.

    Returns
    -------
    torch.Tensor
        Estimated shared latent consensus U_new.
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
    v_mats = simlr_result['v']; mixing_alg = simlr_result.get('mixing_algorithm', 'svd')
    orthogonalize_u = simlr_result.get('orthogonalize_u', False); k = v_mats[0].shape[1]
    projections = [x @ v.to(x.dtype) for x, v in zip(torch_mats, v_mats)]
    u_new = compute_shared_consensus(projections, mixing_algorithm=mixing_alg, k=k, orthogonalize=orthogonalize_u)
    return u_new

def reconstruct_from_learned_maps(u: torch.Tensor, 
                                  simlr_result: Dict[str, Any]) -> List[torch.Tensor]:
    """Reconstruct original modalities from latent consensus U using learned weights."""
    if 'w' not in simlr_result: return []
    w_mats = simlr_result['w']; reconstructions = []
    for w in w_mats: reconstructions.append(u @ w.to(u.dtype))
    return reconstructions

def predict_simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                  simlr_result: Dict[str, Any],
                  allow_legacy_refit: bool = False) -> Dict[str, Any]:
    """
    Comprehensive prediction and reconstruction for new data.

    Parameters
    ----------
    data_matrices : list of Tensor
        New multi-modal data.
    simlr_result : dict
        Trained model result.
    allow_legacy_refit : bool, default=False
        Whether to re-estimate reconstruction weights if missing.

    Returns
    -------
    dict
        Contains 'u', 'reconstructions', and 'errors' (Frobenius norm error).
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    scale_list = simlr_result.get('scale_list', []); provenance_list = simlr_result.get('provenance_list', [])
    if scale_list and scale_list[0] != "none":
        scaled_mats = []
        for i, m in enumerate(torch_mats):
            prov = provenance_list[i] if i < len(provenance_list) else None
            m_scaled = preprocess_data(m, scale_list, provenance=prov)
            scaled_mats.append(m_scaled)
        torch_mats = scaled_mats
    if 'model' in simlr_result:
        from .deep import predict_deep
        return predict_deep(data_matrices, simlr_result)
    u_new = predict_shared_latent(data_matrices, simlr_result)
    if 'w' in simlr_result: reconstructions = reconstruct_from_learned_maps(u_new, simlr_result)
    elif allow_legacy_refit:
        reconstructions = []
        for i, x in enumerate(torch_mats):
            u_pinv = torch.linalg.pinv(u_new)
            reconstructions.append(u_new @ (u_pinv @ x))
    else: raise ValueError("Weights 'w' missing. Set allow_legacy_refit=True.")
    errors = [torch.norm(x - x_pred, p='fro').item() / (torch.norm(x, p='fro').item() + 1e-10) for x, x_pred in zip(torch_mats, reconstructions)]
    return {"u": torch.nan_to_num(u_new), "reconstructions": reconstructions, "errors": errors}

def estimate_rank(data_matrices: List[Union[torch.Tensor, np.ndarray]], n_permutations: int = 20, var_threshold: float = 0.99) -> int:
    """
    Estimate the optimal latent rank k using an RV-coefficient permutation test.

    Parameters
    ----------
    data_matrices : list of Tensor
        Input modalities.
    n_permutations : int, default=20
        Number of permutations for null distribution.
    var_threshold : float, default=0.99
        Cumulative variance threshold for max possible k.

    Returns
    -------
    int
        Optimal latent dimension k.
    """
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
    """Decompose the total SiMLR energy into modality-specific and feature-specific components."""
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    u = simlr_result['u']; v_mats = simlr_result['v']; modality_energies = []; feature_importances = []
    for i, (x, v) in enumerate(zip(torch_mats, v_mats)):
        mod_energy = calculate_simlr_energy(v, x, u, energy_type).item()
        feat_imp = torch.sum(torch.abs(calculate_simlr_gradient(v, x, u, energy_type)), dim=1).numpy()
        modality_energies.append(mod_energy); feature_importances.append(feat_imp)
    return {"modality_energies": modality_energies, "feature_importances": feature_importances}

def pairwise_matrix_similarity(mat_list: List[torch.Tensor], v_list: List[torch.Tensor]) -> Dict[str, float]:
    """Compute the adjusted RV coefficient similarity between all pairs of modalities."""
    n_modalities = len(mat_list); similarities = {}
    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            similarities[f"sim_{i}_{j}"] = adjusted_rvcoef(mat_list[i] @ v_list[i], mat_list[j] @ v_list[j])
    return similarities
