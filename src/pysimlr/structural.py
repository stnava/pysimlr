import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from .deep import deep_simr, lend_simr, ned_simr, predict_deep

def fit_structural_models(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                          k: int, 
                          models: Dict[str, Dict[int, List[int]]],
                          model_type: str = "lend",
                          **kwargs) -> Dict[str, Any]:
    """
    Fit and compare multiple structural path models.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices for each modality.
    k : int
        Latent dimension.
    models : Dict[str, Dict[int, List[int]]]
        A dictionary where keys are model names and values are path graphs 
        (adjacency lists).
    model_type : str, default="lend"
        A registered deep or flow method (`pysimlr.method_names(("deep",
        "flow"))`): the comparison reads their training-loss histories. Any
        other name raises; it used to select `ned_simr` for every name except
        "lend", typos included.
    **kwargs
        Further parameters for `pysimlr.fit`.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results for each model and a comparison summary.
    """
    results = {}
    comparison = []
    
    from .api import fit as _fit, resolve_method
    spec = resolve_method(model_type)
    if spec.family not in ("deep", "flow"):
        raise ValueError(f"fit_structural_models needs a deep or flow method (it compares "
                         f"training-loss histories); {model_type!r} is {spec.family!r}.")

    for name, graph in models.items():
        print(f"Fitting model: {name}...")
        res = _fit(model_type, data_matrices, k, topology="graph", path_graph=graph, **kwargs)
        results[name] = res
        
        avg_recon_loss = float(np.mean(res["recon_history"][-5:]))
        avg_sim_loss = float(np.mean(res["sim_history"][-5:]))
        avg_total_loss = float(np.mean(res["loss_history"][-5:]))
        
        comparison.append({
            "model": name,
            "total_loss": avg_total_loss,
            "recon_loss": avg_recon_loss,
            "sim_loss": avg_sim_loss,
            "converged_iter": res["converged_iter"]
        })
        
    return {
        "models": results,
        "comparison": comparison
    }

def create_path_graph(edges: List[Tuple[int, int]], n_modalities: int) -> Dict[int, List[int]]:
    """
    Helper to create an adjacency list for compute_shared_consensus.
    
    In SiMLR, if A -> B, they should align. This helper treats edges as 
    undirected by default to ensure mutual alignment, unless otherwise specified.
    """
    graph = {i: [] for i in range(n_modalities)}
    for u, v in edges:
        if v not in graph[u]: graph[u].append(v)
        if u not in graph[v]: graph[v].append(u)
    return graph
