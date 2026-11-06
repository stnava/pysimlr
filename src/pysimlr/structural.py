"""Path graphs for ``topology="graph"`` fits.

`create_path_graph` builds the adjacency list that ``pysimlr.fit(..., topology="graph",
path_graph=...)`` aligns along. Comparing graphs is `pysimlr.compare_path_models`
(`pysimlr.sem`): held-out structural log-likelihood with the measurement model fixed, or
refits under each training topology over repeated splits. The former
``fit_structural_models`` (mean of the last five in-sample training losses, deep and flow
methods only) was removed: denser graphs have more loss terms, so the losses were not
comparable across graphs.
"""
from typing import Dict, List, Tuple

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
