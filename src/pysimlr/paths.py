import torch
import numpy as np
from typing import List, Union, Dict, Any
from .utils import adjusted_rvcoef
from .defaults import DEFAULT_ENERGY_TYPE
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
               energy_type: str = DEFAULT_ENERGY_TYPE,
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
        The optimizer to use (e.g., "hybrid_adam", "lars").
    energy_type : str, default=DEFAULT_ENERGY_TYPE ("nc")
        Similarity term (`pysimlr.defaults`, `similarity.SIMILARITY`).
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
    torch_mats = [torch.as_tensor(m).float().contiguous() for m in data_matrices]
    path_results = []
    
    for i, modality_indices in enumerate(path_model):
        if verbose: print(f"Path step {i+1}: modalities {modality_indices}")
        sub_mats = [torch_mats[idx] for idx in modality_indices]
        
        # Warm start from previous step if possible
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
                     method: str = "simlr",
                     verbose: bool = False,
                     random_state: int = 0,
                     n_jobs: int = 1,
                     **kwargs) -> Dict[str, Any]:
    """
    Whole-pipeline permutation test of shared structure, for any registered method.

    Fits ``method`` with `pysimlr.fit` (``kwargs`` are its parameters), then runs
    `pysimlr.sem.path_model` on the training rows over the complete graph with the refit
    null (``refit_scope=("omnibus",)``): ``n_permutations`` refits with every view but the
    first permuted (each view with its own confound rows), the global null of no shared
    structure. The statistic is the mean over view pairs of ``pillai``, the mean squared
    canonical correlation of the two views' scores.

    Parameters
    ----------
    data_matrices : list of matrices with matched rows.
    k : int
        Latent dimension.
    n_permutations : int, default=100
        Null refits.
    method : str, default="simlr"
        Any registered method (`pysimlr.method_names`).
    verbose : bool, default=False
        Passed to the fit when True.
    random_state : int, default=0
        Seed of the fit and of the permutations.
    n_jobs : int, default=1
        Worker processes for the refits.
    **kwargs
        Parameters of the method (`pysimlr.describe_method`).

    Returns
    -------
    Dict[str, Any]
        - "observed_similarity": mean ``pillai`` over view pairs on the training rows.
        - "null_similarities": the same statistic for each null refit.
        - "p_value": add-one ``(1 + #{null >= observed}) / (1 + n_permutations)``; NaN when
          ``n_permutations`` is 0.
        - "n_permutations", "n_exceeding".
        - "edges": the `path_model` edge table (pillai per pair; no per-edge refit p-values).
        - "omnibus", "result": the `path_model` omnibus and the observed fit.
    """
    from .api import fit
    from .sem import path_model
    views = [torch.as_tensor(np.asarray(m) if not isinstance(m, torch.Tensor) else m).float().contiguous() for m in data_matrices]
    if len(views) < 2:
        raise ValueError("permutation_test needs at least two views.")
    fit_kw = dict(kwargs, verbose=True) if verbose else dict(kwargs)
    res = fit(method, views, k, random_state=random_state, **fit_kw)
    nperm = int(n_permutations)
    pm = path_model(res, views, rows="train", null="refit", n_refits=nperm, n_permutations=0, n_bootstrap=0,
                    refit_scope=("omnibus",), random_state=random_state, n_jobs=n_jobs)
    om = pm["omnibus"]
    null = list(om["null_mean_pillai_refit"] or [])
    obs = float(om["mean_pillai"])
    n_exceed = int(np.sum(np.asarray(null) >= obs)) if null else 0
    return {
        "observed_similarity": obs,
        "null_similarities": null,
        "p_value": float(om["p_mean_refit"]) if nperm > 0 else float("nan"),
        "n_permutations": len(null),
        "n_exceeding": n_exceed,
        "edges": pm["edges"],
        "omnibus": om,
        "result": res,
    }
