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
                            modality_weights: Optional[torch.Tensor] = None,
                            frozen_map: Optional[Dict[str, Any]] = None) -> Union[torch.Tensor, List[torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[torch.Tensor]]]:
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
    frozen_map : dict, optional
        From `capture_consensus_map`. Outside training, the consensus is that
        frozen map applied row by row, so it does not depend on the batch.
    orthogonalize : bool, default=False
        If True, decorrelate the consensus columns (polar projection) so that
        ``U.T @ U`` is diagonal, before the per-column standardization. Requires
        more samples than components; silently skipped otherwise. Note that the
        "svd" and "pca" mixing algorithms already yield orthogonal columns, so
        this only changes "avg", "newton" and "ica".
    """
    if not projections:
        return torch.empty(0)
    if frozen_map is not None and not training:
        # Prediction with the training batch's frozen affine consensus; see
        # `capture_consensus_map`.
        return apply_consensus_map(projections, frozen_map)
        
    len(projections)
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
                    # A FastICA failure propagates: it used to be replaced by
                    # the first k averaged projection columns and an identity
                    # anchor, a different consensus with no warning.
                    u_np = ica.fit_transform(avg_p)
                    local_anchor = torch.from_numpy(ica.components_.T).to(local_big_p.device).to(local_big_p.dtype)
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


#: MAI metrics `mai_score` implements. ``trace`` was removed: tr(Z'u) depends
#: on the arbitrary rotation of the latent coordinates (measured 0.0348, 0.1319
#: and 0.0000 for one pair under three rotations), and on `flow_simr_v` it
#: ranked a planted pure-noise view above a signal view (0.276 vs 0.246).
MAI_METRICS = ("procrustes_r2", "procrustes_r2_sharp", "cca", "rvcoef")


def mai_score(z: torch.Tensor, u_loo: torch.Tensor, metric: str) -> float:
    """Agreement of one view's normalized latent ``z`` with the leave-one-out
    consensus ``u_loo`` (both centred, unit Frobenius norm).

    ``procrustes_r2`` is the scale-profiled Procrustes R^2, ``sum(s)^2`` for
    the singular values ``s`` of ``z' u_loo``: bounded in [0, 1] and never
    clamped (the un-profiled form reduced to ``2 sum(s) - 1`` and clamped to
    0 whenever ``sum(s) < 0.5``, so at k=3 every view scored 0 and MAI was
    inert; profiled, a signal view scores 0.989 against 0.020 for noise).
    ``procrustes_r2_sharp`` multiplies it by the spectral sharpness
    ``s_1 / sum(s)`` of ``z`` itself (isotropic noise has a flat spectrum).
    ``cca`` is the mean singular value; ``rvcoef`` the RV coefficient.

    One implementation for the linear, deep and flow models, which carried
    six copies of this code.
    """
    if metric not in MAI_METRICS:
        raise ValueError(f"unknown mai_metric {metric!r}. Choose one of {list(MAI_METRICS)}. "
                         f"'trace' has been removed: tr(Z'u) depends on the arbitrary rotation "
                         f"of the latent coordinates, so it does not measure agreement between views.")
    cross = z.t() @ u_loo
    s = torch.linalg.svdvals(cross)
    if metric in ("procrustes_r2", "procrustes_r2_sharp"):
        r2 = float(s.sum().item()) ** 2
        if metric == "procrustes_r2_sharp":
            s_lat = torch.linalg.svdvals(z)
            r2 *= float((s_lat[0] / (s_lat.sum() + 1e-8)).item())
        return r2
    if metric == "cca":
        return float(s.mean().item())
    num = torch.norm(cross, p="fro") ** 2
    den = torch.norm(z.t() @ z, p="fro") * torch.norm(u_loo.t() @ u_loo, p="fro")
    return float((num / (den + 1e-8)).item())


def mai_score_or_zero(z: torch.Tensor, u_loo: torch.Tensor, metric: str, view: int) -> float:
    """`mai_score`, scoring a view whose SVD does not converge as 0 with a
    `RuntimeWarning` naming the view (a non-finite latent is the usual cause).
    Scoring it 0 down-weights that view in the consensus; the warning makes
    the substitution visible."""
    try:
        return mai_score(z, u_loo, metric)
    except torch.linalg.LinAlgError as err:
        warnings.warn(f"MAI for view {view} could not be computed ({err}); scored 0, which "
                      f"down-weights the view.", RuntimeWarning, stacklevel=3)
        return 0.0


#: Same set as `MAI_METRICS`, as a frozenset for membership checks.
VALID_MAI_METRICS = frozenset(MAI_METRICS)


def check_mai_metric(metric: str) -> str:
    """Return `metric` or raise naming the valid choices. Checked before any
    scoring so a misspelled metric cannot become a uniform weighting."""
    if metric not in VALID_MAI_METRICS:
        raise ValueError(
            f"unknown mai_metric {metric!r}; choose one of {sorted(VALID_MAI_METRICS)}. "
            f"'trace' was removed because tr(Z'u) is not invariant to the arbitrary "
            f"rotation of the latent coordinates.")
    return metric


def loo_mai_scores(latents: List[torch.Tensor], metric: str,
                   use_rank: bool = False) -> List[float]:
    """Leave-one-out agreement of every view with the others, one float per view.

    Each latent is centred and scaled to unit Frobenius norm (optionally after a
    column-wise rank transform); view i is scored by `mai_score` against the
    sign-aligned mean of the other views (`utils.align_column_signs`), and a
    view scores 1.0 when it is the only one. Shared by `simlr` and every deep
    and flow model; only the weighting schedule built on these scores differs.
    """
    check_mai_metric(metric)
    if use_rank:
        latents = [torch.argsort(torch.argsort(p, dim=0), dim=0).to(p.dtype) for p in latents]
    norm_projs = []
    for p in latents:
        p_c = p - p.mean(dim=0, keepdim=True)
        p_norm = torch.norm(p_c, p="fro")
        norm_projs.append(p_c / p_norm if p_norm > 1e-8 else torch.zeros_like(p_c))
    mais = []
    for i in range(len(norm_projs)):
        others = [norm_projs[j] for j in range(len(norm_projs)) if j != i]
        if not others:
            mais.append(1.0)
            continue
        # Sign-aligned: unaligned views cancel in this average exactly as they
        # do in the consensus.
        u_loo = torch.mean(torch.stack(align_column_signs(others)), dim=0)
        u_norm = torch.norm(u_loo, p="fro")
        mais.append(mai_score_or_zero(norm_projs[i], u_loo / u_norm, metric, i)
                    if u_norm > 1e-8 else 0.0)
    return mais


def mai_gated_target(mai: torch.Tensor, steepness: float) -> torch.Tensor:
    """Target modality weights from smoothed MAI scores: each score times a
    sigmoid gate centred on the mean score, renormalised to sum to one."""
    gate = torch.sigmoid(steepness * (mai - mai.mean()))
    raw = mai * gate
    return raw / (raw.sum() + 1e-8)


def capture_consensus_map(projections: List[torch.Tensor],
                          u: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, Any]:
    """Freeze the training batch's consensus as a row-wise affine map.

    For a fixed batch, every mixing rule `compute_shared_consensus` applies at
    prediction time -- per-view RMS scaling, per-column sign alignment, the
    learned SVD/PCA/ICA anchor, Newton-Schulz orthogonalization (a right
    multiplication by a k x k matrix), and the final per-column centring and
    scaling -- is an affine function of the concatenated per-view projections.
    Re-deriving those statistics from each prediction batch made out-of-sample
    latents depend on which other rows were in the batch (measured: up to 2.2
    for `lend`, 0.58 for `simlr` under topology="loo"). Fitting the affine map
    on the training batch by least squares and replaying it per row makes
    prediction batch-size invariant, including N=1.

    Returns
    -------
    dict
        ``proj_rms`` (per view), ``mu`` (mean of the normalized concatenation),
        ``maps`` (one ``(sum k_m, k)`` matrix per consensus output), ``offsets``,
        ``is_list`` and ``max_residual`` (largest absolute training residual;
        near machine precision when the consensus is affine, as it is for every
        rule except an ICA refit without an anchor).
    """
    proj_rms, normed = [], []
    for p in projections:
        p = torch.nan_to_num(p.detach().double(), nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(torch.norm(p, p="fro") / np.sqrt(max(1, p.shape[0])))
        proj_rms.append(rms)
        normed.append(p / (rms + 1e-8))
    z = torch.cat(normed, dim=1)
    mu = z.mean(dim=0, keepdim=True)
    zc = z - mu
    targets = list(u) if isinstance(u, (list, tuple)) else [u]
    maps, offsets, resid = [], [], 0.0
    for t in targets:
        t = t.detach().double()
        b = t.mean(dim=0, keepdim=True)
        r = torch.linalg.lstsq(zc, t - b, driver="gelsd").solution
        maps.append(r)
        offsets.append(b)
        resid = max(resid, float((zc @ r + b - t).abs().max()))
    return {"proj_rms": proj_rms, "mu": mu, "maps": maps, "offsets": offsets,
            "is_list": isinstance(u, (list, tuple)), "max_residual": resid}


def apply_consensus_map(projections: List[torch.Tensor],
                        frozen: Dict[str, Any]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Replay `capture_consensus_map` row-wise on new projections."""
    dtype, device = projections[0].dtype, projections[0].device
    normed = [torch.nan_to_num(p.double(), nan=0.0, posinf=0.0, neginf=0.0) / (rms + 1e-8)
              for p, rms in zip(projections, frozen["proj_rms"])]
    zc = torch.cat(normed, dim=1) - frozen["mu"].to(device)
    outs = [(zc @ r.to(device) + b.to(device)).to(dtype)
            for r, b in zip(frozen["maps"], frozen["offsets"])]
    return outs if frozen["is_list"] else outs[0]
