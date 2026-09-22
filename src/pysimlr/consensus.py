import torch
import numpy as np
import warnings
from typing import List, Optional, Union, Dict, Any, Tuple
from .utils import align_column_signs
from .utils import newton_schulz_orthogonalize, safe_svd


def align_anchor_columns(new_anchor: torch.Tensor,
                         reference: Optional[torch.Tensor],
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Return per-column signs (shape ``(1, k)``) that orient `new_anchor` to `reference`.

    SVD/ICA basis vectors are only defined up to sign, so the basis recovered on
    one batch can come back with arbitrarily flipped columns relative to the
    previous one. Averaging such bases (as the deep models' EMA does) makes
    opposite-signed columns cancel toward zero, which silently shrinks the
    anchor and corrupts anchored prediction. Aligning signs first removes that.

    Columns whose reference is degenerate (norm below `eps`, e.g. a freshly
    zero-initialised anchor buffer) are left unflipped.

    Parameters
    ----------
    new_anchor : torch.Tensor
        Newly computed basis of shape (P, k).
    reference : torch.Tensor or None
        Basis to orient against, same shape. ``None`` yields all-positive signs.
    eps : float, default=1e-8
        Norm below which a reference column is treated as degenerate.

    Returns
    -------
    torch.Tensor
        Signs of shape (1, k), each entry +1.0 or -1.0.
    """
    k = new_anchor.shape[1]
    ones = torch.ones(1, k, device=new_anchor.device, dtype=new_anchor.dtype)
    if reference is None or reference.shape != new_anchor.shape:
        return ones
    ref_norm = torch.linalg.vector_norm(reference, dim=0, keepdim=True)
    dots = torch.sum(new_anchor * reference, dim=0, keepdim=True)
    signs = torch.where(dots < 0.0, -ones, ones)
    return torch.where(ref_norm < eps, ones, signs)

def compute_shared_consensus(projections: List[torch.Tensor], 
                            mixing_algorithm: str = "svd", 
                            k: Optional[int] = None,
                            orthogonalize: bool = False,
                            training: bool = False,
                            anchor: Optional[torch.Tensor] = None,
                            topology: str = "star",
                            path_graph: Optional[Dict[int, List[int]]] = None,
                            prune_threshold: Optional[float] = None,
                            modality_weights: Optional[torch.Tensor] = None) -> Union[torch.Tensor, List[torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[torch.Tensor]]]:
    """
    Combine modality-specific projections into a shared latent consensus (U).
    
    Topology (the return type depends on this, identically in training and
    prediction: 'star' yields a single tensor, 'loo' and 'graph' yield one
    consensus tensor per modality):
    - 'star': All modalities align to a single shared consensus.
    - 'loo': (Leave-One-Out) Modality i aligns to the consensus of all OTHER modalities.
    - 'graph': Modality i aligns to the consensus of its NEIGHBORS in path_graph.
    
    Anchor-based Prediction Fix:
    To prevent coordinate drift/rotation in SVD/PCA/ICA during prediction,
    we utilize a learned 'anchor' projection matrix.

    Parameters
    ----------
    orthogonalize : bool, default=False
        If True, decorrelate the consensus columns (polar projection) so that
        ``U.T @ U`` is diagonal, before the per-column standardization. Requires
        more samples than components; silently skipped otherwise. Note that the
        "svd" and "pca" mixing algorithms already yield orthogonal columns, so
        this only changes "avg", "newton" and "ica".
    """
    if not projections:
        return torch.empty(0)
        
    n_modalities = len(projections)
    if k is None:
        k = projections[0].shape[1]
        
    norm_projs = []
    for p in projections:
        p_safe = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p_rms = torch.norm(p_safe, p='fro') / np.sqrt(p_safe.shape[0])
        if p_rms > 1e-6:
            norm_projs.append(p_safe / (p_rms + 1e-8))
        else:
            norm_projs.append(torch.randn_like(p_safe) * 1e-4)
            
    if modality_weights is not None:
        scaled_projs = []
        for i, p in enumerate(norm_projs):
            scaled_projs.append(p * modality_weights[i].item())
        norm_projs = scaled_projs

    # Core consensus logic separated into a helper to reuse for LOO
    def _get_u(proj_list, return_anchor=False):
        local_big_p = torch.cat(proj_list, dim=1)
        volatile = mixing_algorithm in ["svd", "pca", "ica"]
        if not training and volatile and anchor is not None and len(proj_list) == len(projections):
            # Reuse the learned anchor so prediction does not re-derive a fresh
            # (arbitrarily rotated/sign-flipped) SVD basis. This branch must fall
            # through to the shared standardization block below: returning early
            # here produced a U on a different scale than the training-time U.
            local_u = local_big_p @ anchor
            local_anchor = anchor
        elif mixing_algorithm == "avg":
            # Signs aligned before averaging; see `utils.align_column_signs`.
            # A latent is defined only up to a sign per component, so two views
            # can recover the same factors with opposite signs and the mean
            # then cancels them. Measured on a planted signal: per-view latents
            # at R^2 0.784 and 0.785, their plain average at 0.023.
            local_u = torch.mean(torch.stack(align_column_signs(proj_list)), dim=0)
            local_anchor = None
        elif mixing_algorithm == "newton":
            local_u = torch.mean(torch.stack(align_column_signs(proj_list)), dim=0)
            local_u = newton_schulz_orthogonalize(local_u, iterations=10)
            local_anchor = None
        elif mixing_algorithm == "ica":
            avg_p = local_big_p.detach().cpu().numpy()
            with warnings.catch_warnings():
                try:
                    from sklearn.decomposition import FastICA
                    from sklearn.exceptions import ConvergenceWarning
                except ImportError:
                    FastICA = None
                    ConvergenceWarning = None
                
                if ConvergenceWarning is not None:
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                
                if FastICA is not None:
                    ica = FastICA(n_components=k, random_state=42, max_iter=2000, tol=1e-2)
                    try:
                        u_np = ica.fit_transform(avg_p)
                        local_anchor = torch.from_numpy(ica.components_.T).to(local_big_p.device).to(local_big_p.dtype)
                    except Exception:
                        u_np = avg_p[:, :k]
                        local_anchor = torch.eye(local_big_p.shape[1], k, device=local_big_p.device).to(local_big_p.dtype)
                else:
                    u_np = avg_p[:, :k]
                    local_anchor = torch.eye(local_big_p.shape[1], k, device=local_big_p.device).to(local_big_p.dtype)
            local_u = torch.from_numpy(u_np).to(local_big_p.device).to(local_big_p.dtype)
        else:
            if mixing_algorithm == "pca":
                big_p_c = local_big_p - torch.mean(local_big_p, dim=0)
                _, _, vh = safe_svd(big_p_c, full_matrices=False)
            else:
                _, _, vh = safe_svd(local_big_p, full_matrices=False)
            
            local_anchor = vh.T[:, :k]
            if local_anchor.shape[1] < k:
                padding = torch.zeros(local_anchor.shape[0], k - local_anchor.shape[1], device=local_anchor.device, dtype=local_anchor.dtype)
                local_anchor = torch.cat([local_anchor, padding], dim=1)
            local_u = local_big_p @ local_anchor
            
        # Decorrelate the consensus columns when asked. Applied before the
        # per-column standardization below, which rescales every column by the
        # same factor once the columns are mean-zero and hence preserves
        # orthogonality. This parameter was previously accepted and ignored, so
        # `simlr(..., orthogonalize_u=True)` did nothing at all.
        if orthogonalize and local_u.shape[0] > local_u.shape[1] and local_u.shape[1] > 1:
            try:
                u_o, _, vh_o = safe_svd(local_u - local_u.mean(0, keepdim=True),
                                        full_matrices=False)
                if u_o.shape[1] == local_u.shape[1]:
                    local_u = u_o @ vh_o
            except RuntimeError:
                pass

        if local_u.shape[0] > 1:
            local_u = local_u - local_u.mean(0, keepdim=True)
            u_std = torch.std(local_u, dim=0, keepdim=True)
            u_std = torch.where(torch.isnan(u_std) | (u_std < 1e-6), torch.ones_like(u_std), u_std)
            local_u = local_u / u_std
            
        # Orient the freshly derived basis to the stored anchor before it is
        # handed back for EMA accumulation, so sign-flipped columns do not
        # cancel. Flipping a column of the anchor flips the matching column of
        # U, and U is already standardized, so a +/-1 scale preserves that.
        if training and local_anchor is not None and anchor is not None:
            signs = align_anchor_columns(local_anchor, anchor)
            local_anchor = local_anchor * signs
            local_u = local_u * signs

        if return_anchor:
            if len(proj_list) != len(projections):
                return local_u, None
            return local_u, local_anchor
        return local_u

    # Variance-Weighted Pruning Option
    orig_norm_projs = norm_projs
    valid_indices = set(range(len(norm_projs)))
    
    if prune_threshold is not None and len(norm_projs) > 1:
        # Aligned too: this probe decides which views are pruned, and a
        # cancelled average makes every view look uninformative at once.
        u_pre = torch.mean(torch.stack(align_column_signs(norm_projs)), dim=0)
        u_pre = u_pre - u_pre.mean(dim=0, keepdim=True)
        u_pre_norm = torch.norm(u_pre, p='fro')
        
        valid_projs = []
        temp_valid_indices = set()
        for i, p in enumerate(norm_projs):
            p_c = p - p.mean(dim=0, keepdim=True)
            p_norm = torch.norm(p_c, p='fro')
            
            if u_pre_norm > 1e-8 and p_norm > 1e-8:
                sim = torch.trace(p_c.t() @ u_pre) / (p_norm * u_pre_norm)
                if sim >= prune_threshold:
                    valid_projs.append(p)
                    temp_valid_indices.add(i)
            else:
                valid_projs.append(p)
                temp_valid_indices.add(i)
                
        if len(valid_projs) > 0:
            norm_projs = valid_projs
            valid_indices = temp_valid_indices

    # Reject an unrecognised topology rather than quietly running a different
    # one. Every unknown string -- a typo, a wrong case like "LOO", or a name
    # that sounds plausible like "path" -- used to fall through to the `star`
    # consensus at the end of this function and return a perfectly reasonable
    # answer for a model the caller did not ask for. `path_graph` is only read
    # by "graph", so `topology="path", path_graph=...` silently discarded the
    # graph and every graph gave an identical result.
    _VALID_TOPOLOGIES = ("star", "loo", "graph")
    if topology not in _VALID_TOPOLOGIES:
        raise ValueError(
            f"unknown topology {topology!r}; choose one of "
            f"{list(_VALID_TOPOLOGIES)}. The graph-structured option is "
            f"'graph' (it reads `path_graph`); there is no 'path'."
        )
    if path_graph is not None and topology != "graph":
        warnings.warn(
            f"path_graph was supplied with topology={topology!r}, which does "
            f"not read it; it is ignored. Pass topology='graph' to use it.",
            RuntimeWarning, stacklevel=2,
        )

    if topology == "star":
        if training:
            u, new_anchor = _get_u(norm_projs, return_anchor=True)
            return u, new_anchor
        else:
            return _get_u(norm_projs, return_anchor=False)
            
    elif topology == "loo":
        # NOTE: a model trained with leave-one-out must also *predict* with
        # leave-one-out. This branch previously short-circuited to the star
        # consensus whenever an anchor was present, so prediction returned a
        # single tensor where training returned one consensus per modality --
        # a different estimator, and a different return type.
        u_list = []
        for i in range(len(orig_norm_projs)):
            loo_projs = [p for j, p in enumerate(orig_norm_projs) if j != i and j in valid_indices]
            if len(loo_projs) == 0:
                u_list.append(orig_norm_projs[i])
            else:
                u_list.append(_get_u(loo_projs, return_anchor=False))
                
        if training:
            _, new_anchor = _get_u(norm_projs, return_anchor=True)
            return u_list, new_anchor
        return u_list

    elif topology == "graph":
        if path_graph is None:
            raise ValueError("path_graph must be provided for topology='graph'")
            
        u_list = []
        for i in range(len(orig_norm_projs)):
            neighbors = path_graph.get(i, [])
            neighbor_projs = [orig_norm_projs[j] for j in neighbors if j in valid_indices]
            if len(neighbor_projs) == 0:
                u_list.append(orig_norm_projs[i])
            else:
                u_list.append(_get_u(neighbor_projs, return_anchor=False))
                
        if training:
            _, new_anchor = _get_u(norm_projs, return_anchor=True)
            return u_list, new_anchor
        return u_list

    # Unreachable: the topology is validated above. Kept so a future branch
    # that forgets to return has a defined result rather than None.
    return _get_u(norm_projs, return_anchor=training)
