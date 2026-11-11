"""Path graphs for ``topology="graph"`` fits.

`create_path_graph` builds the adjacency list that ``pysimlr.fit(..., topology="graph",
path_graph=...)`` aligns along. Comparing graphs is `pysimlr.compare_path_models`
(`pysimlr.sem`): held-out structural log-likelihood with the measurement model fixed, or
refits under each training topology over repeated splits. The former
``fit_structural_models`` (mean of the last five in-sample training losses, deep and flow
methods only) was removed: denser graphs have more loss terms, so the losses were not
comparable across graphs.
"""
from typing import Dict, List, Sequence, Tuple, Union

NodeKey = Union[int, str]


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


def fork_graph(
    root: NodeKey,
    children: Sequence[NodeKey],
    *,
    exogenous_root: bool = True,
    for_sem: bool = False,
) -> Dict[NodeKey, List[NodeKey]]:
    """Build a causal Fork adjacency dict rooted at ``root`` over ``children``.

    Parameters
    ----------
    root : int or str
        Index or view name of the root node (e.g., ``"Blast_Root"`` or ``0``).
    children : sequence of int or str
        Distinct child/prong node keys branching directly off ``root``.
    exogenous_root : bool, default=True
        Used when ``for_sem=False``. When ``True``, ``root`` has no training
        consensus neighbors (``graph[root] = []``), keeping ``V_root`` strictly
        exogenous while each child aligns solely to ``root``
        (``graph[child] = [root]``). When ``False``, ``graph[root] = list(children)``
        so ``root`` and its children align bidirectionally with zero inter-child edges.
    for_sem : bool, default=False
        When ``False`` (default), returns the alignment neighborhood dict for
        ``pysimlr.fit(..., topology="graph", path_graph=...)`` where each child
        lists ``[root]`` as its target. When ``True``, returns the outgoing
        parent-to-child adjacency dict ``{root: list(children), child: []}`` for
        ``pysimlr.path_model(..., graph=..., directed=True)``.
    """
    child_list = list(children)
    if not child_list:
        raise ValueError("fork_graph requires at least one child view.")
    if root in child_list or len(set(child_list)) != len(child_list):
        raise ValueError(f"root ({root!r}) and children ({child_list!r}) must all be distinct.")
    if for_sem:
        return {root: list(child_list), **{c: [] for c in child_list}}
    return {root: ([] if exogenous_root else list(child_list)), **{c: [root] for c in child_list}}
