import torch
import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
from .simlr import initialize_simlr
from .optimizers import create_optimizer
from .sparsification import orthogonalize_and_q_sparsify
from .utils import adjusted_rvcoef
def _to_consensus_tensor(u: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """
    Ensure consensus representation is returned as a single 2D tensor (N x K).
    If u is a list of tensors (from LOO or graph topology), computes the mean
    consensus across all modalities.
    """
    if isinstance(u, list):
        if len(u) == 0:
            raise ValueError("Consensus list cannot be empty.")
        return torch.mean(torch.stack([torch.as_tensor(x).float() for x in u]), dim=0)
    return torch.as_tensor(u).float()

def simlr_path(data_matrices: List[Union[torch.Tensor, np.ndarray]],
               k: int,
               path_model: List[List[int]],
               iterations: int = 20,
               optimizer_type: str = "hybrid_adam",
               energy_type: str = "regression",
               verbose: bool = False,
               **kwargs) -> Dict[str, Any]:
    """
    Fit a sequential path of SiMLR models by adding modalities incrementally.

    This function explores how the shared latent consensus (U) evolves as 
    new data views are incorporated. It is useful for understanding the 
    contribution of each modality to the shared signal and for identifying 
    robust latent structures across different combinations of data.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        A list of all available data matrices.
    k : int
        The dimensionality of the shared latent space.
    path_model : List[List[int]]
        A list of lists, where each sub-list contains the indices of 
        modalities to include at that step of the path.
    iterations : int, default=20
        Number of optimization iterations per path step.
    optimizer_type : str, default="hybrid_adam"
        The optimizer to use (e.g., "hybrid_adam", "nsa_flow").
    energy_type : str, default="regression"
        The similarity/reconstruction energy objective.
    verbose : bool, default=False
        Whether to print progress.
    **kwargs
        Additional parameters passed to the SiMLR optimizer.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "path_results": A list of SiMLR results for each step in the path.
        - "consensus_correlations": Similarity between the final consensus 
          and the consensus at each path step.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    path_results = []
    
    for i, modality_indices in enumerate(path_model):
        if verbose: print(f"Path step {i+1}: modalities {modality_indices}")
        sub_mats = [torch_mats[idx] for idx in modality_indices]
        
        # Warm start from previous step if possible
        init_v = None
        if i > 0:
             # Logic for warm starting could be added here
             pass
             
        from .simlr import simlr
        res = simlr(sub_mats, k=k, iterations=iterations, 
                    optimizer_type=optimizer_type, 
                    energy_type=energy_type, verbose=verbose, **kwargs)
        path_results.append(res)
        
    # Calculate stability of U across the path
    final_u = _to_consensus_tensor(path_results[-1]['u'])
    correlations = []
    for res in path_results:
        u_i = _to_consensus_tensor(res['u'])
        # Compute similarity between consensus at this step and final consensus
        correlations.append(float(adjusted_rvcoef(u_i, final_u)))
        
    return {
        "path_results": path_results,
        "consensus_correlations": correlations
    }

def permutation_test(data_matrices: List[Union[torch.Tensor, np.ndarray]],
                     k: int,
                     n_permutations: int = 100,
                     verbose: bool = False,
                     **kwargs) -> Dict[str, Any]:
    """
    Assess the statistical significance of the SiMLR consensus via permutation.

    This function builds a null distribution of the cross-modality similarity 
    (ACC) by repeatedly shuffling the rows of each data modality independently 
    and re-fitting the SiMLR model. The observed similarity is then compared 
    against this distribution to calculate a p-value.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        The list of data matrices to analyze.
    k : int
        The dimensionality of the shared latent space.
    n_permutations : int, default=100
        Number of permutations to perform for the null distribution.
    verbose : bool, default=False
        Whether to print progress.
    **kwargs
        Additional parameters passed to the SiMLR algorithm.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - "observed_similarity": The cross-modality similarity of the original data.
        - "null_similarities": A list of similarities from the permuted data.
        - "p_value": The add-one permutation estimate
          ``(1 + #{null >= observed}) / (1 + n_permutations)`` of the
          probability of a similarity at least as extreme as the one measured,
          under the null hypothesis of no shared structure. NaN when
          `n_permutations` is 0.
        - "n_permutations": Number of null replicates actually drawn.
        - "n_exceeding": How many null replicates met or exceeded the observation.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    from .simlr import simlr
    from .utils import preprocess_data
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]

    def _score_against_u(mats, fit):
        scale_list = fit.get("scale_list") or []
        prov_list = fit.get("provenance_list") or []
        if scale_list and scale_list[0] != "none" and len(prov_list) == len(mats):
            scored = [preprocess_data(m, scale_list, provenance=pv) for m, pv in zip(mats, prov_list)]
        else:
            scored = mats
        u_res = fit["u"]
        sims = []
        for idx, (m, v) in enumerate(zip(scored, fit["v"])):
            u_target = u_res[idx] if isinstance(u_res, list) else u_res
            sims.append(float(adjusted_rvcoef(m @ v, u_target)))
        return float(np.mean(sims))
    
    # 1. Observed similarity
    obs_res = simlr(torch_mats, k=k, verbose=verbose, **kwargs)
    
    # 2. Null distribution
    null_sims = []
    for i in range(n_permutations):
        if verbose and i % 10 == 0: print(f"Permutation {i}/{n_permutations}")
        # Shuffle rows of each matrix independently
        perm_mats = [m[torch.randperm(m.shape[0])] for m in torch_mats]
        perm_res = simlr(perm_mats, k=k, verbose=False, **kwargs)
        null_sims.append(_score_against_u(perm_mats, perm_res))
        
    obs_sim = _score_against_u(torch_mats, obs_res)

    if len(null_sims) == 0:
        p_value = float('nan')
    else:
        # Add-one (Phipson & Smyth 2010) permutation p-value. A plain
        # mean(null >= obs) can return exactly 0, which is not an attainable
        # p-value from a finite permutation set and understates uncertainty:
        # the floor is 1 / (1 + n_permutations).
        n_exceed = int(np.sum(np.asarray(null_sims) >= obs_sim))
        p_value = (1.0 + n_exceed) / (1.0 + len(null_sims))

    return {
        "observed_similarity": obs_sim,
        "null_similarities": null_sims,
        "p_value": float(p_value),
        "n_permutations": len(null_sims),
        "n_exceeding": int(np.sum(np.asarray(null_sims) >= obs_sim)) if null_sims else 0,
    }
