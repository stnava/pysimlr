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
        The deep SiMLR model type to use ("lend" or "ned").
    **kwargs
        Additional parameters passed to the training function.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results for each model and a comparison summary.
    """
    results = {}
    comparison = []
    
    fit_fn = lend_simr if model_type == "lend" else ned_simr
    
    for name, graph in models.items():
        print(f"Fitting model: {name}...")
        res = fit_fn(data_matrices, k, topology="graph", path_graph=graph, **kwargs)
        results[name] = res
        
        # Calculate some metrics for comparison
        # 1. Reconstruction Error
        avg_recon_loss = np.mean(res["recon_history"][-5:])
        # 2. Similarity (Alignment) Loss
        avg_sim_loss = np.mean(res["sim_history"][-5:])
        # 3. Total Loss
        avg_total_loss = np.mean(res["loss_history"][-5:])
        
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
