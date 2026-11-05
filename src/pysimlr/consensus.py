import torch
import numpy as np
import warnings
from typing import List, Optional, Union, Dict, Any, Sequence, Tuple
from .utils import align_column_signs
from .utils import newton_schulz_orthogonalize, safe_svd
from .defaults import DEFAULT_MIXING_ALGORITHM, DEFAULT_TOPOLOGY


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

#: Mixings that average corresponding columns across views.
_AVERAGING_MIXINGS = ("avg", "newton")


def compute_shared_consensus(projections: List[torch.Tensor], 
                            mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM, 
                            k: Optional[int] = None,
                            orthogonalize: bool = False,
                            training: bool = False,
                            anchor: Optional[torch.Tensor] = None,
                            topology: str = DEFAULT_TOPOLOGY,
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
        
    if mixing_algorithm in _AVERAGING_MIXINGS:
        # Averaging mixings pair column l of every view with column l of every
        # other view, so every projection must have the same width and that
        # width is the rank. They cannot reduce a wider projection to `k`
        # (only the concatenating mixings, svd/pca/ica, fit a basis that can);
        # they used to return the full width and ignore `k`, and unequal widths
        # failed inside `torch.stack` with a shape error.
        widths = sorted({int(p.shape[1]) for p in projections})
        if len(widths) != 1:
            raise ValueError(
                f"mixing_algorithm={mixing_algorithm!r} averages corresponding "
                f"columns, so all projections need the same width; got widths "
                f"{[int(p.shape[1]) for p in projections]}. Use 'svd', 'pca' or "
                f"'ica' to combine projections of different widths.")
        if k is not None and int(k) != widths[0]:
            raise ValueError(
                f"mixing_algorithm={mixing_algorithm!r} cannot reduce projections "
                f"of width {widths[0]} to k={k}: it averages corresponding columns. "
                f"Pass projections of width k, or use 'svd', 'pca' or 'ica'.")
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
            
    unweighted_projs = norm_projs
    weight_vals = None
    if modality_weights is not None:
        # One host transfer per call (was one .item() per view): the zero-weight
        # guard below needs the values on the host.
        mw = modality_weights.detach() if torch.is_tensor(modality_weights) else torch.as_tensor(modality_weights)
        weight_vals = [float(w) for w in mw.cpu().tolist()][:len(norm_projs)]
        norm_projs = [p * w for p, w in zip(norm_projs, weight_vals)]

    def _neighbourhood(indices):
        """Weighted projections of ``indices``; unweighted if their weights sum
        to 0. All-zero weights carry no information about the mix, and a zero
        neighbourhood gave a zero consensus whose SVD/ICA returned NaN (on
        ADNI's bipartite graph, where the gate zeroed every MRI view)."""
        if weight_vals is not None and sum(weight_vals[j] for j in indices) <= 1e-12:
            warnings.warn(f"views {sorted(indices)} all have modality weight 0; their consensus "
                          f"uses them unweighted.", RuntimeWarning, stacklevel=3)
            return [unweighted_projs[j] for j in indices]
        return [orig_norm_projs[j] for j in indices]

    # Core consensus logic separated into a helper to reuse for LOO
    def _get_u(proj_list, return_anchor=False):
        need_anchor = bool(return_anchor) or (
            training and anchor is not None and anchor.shape == (len(proj_list) * k, k)
        )
        local_big_p = torch.cat(proj_list, dim=1)
        # Every mixing has a linear anchor (U = big_p @ anchor for the batch it
        # was derived on), so prediction can replay it instead of re-deriving a
        # batch-dependent consensus. "avg"/"newton" had none and re-orthogonalised
        # the prediction batch; the package's predict paths use `frozen_map`
        # and were not affected (measured: |u_full - u_half| <= 9e-8).
        volatile = mixing_algorithm in ["svd", "pca", "ica", "avg", "newton"]
        if not training and volatile and anchor is not None and len(proj_list) == len(projections):
            # Reuse the learned anchor so prediction does not re-derive a fresh
            # (arbitrarily rotated/sign-flipped) SVD basis. This branch must fall
            # through to the shared standardization block below: returning early
            # here produced a U on a different scale than the training-time U.
            local_u = local_big_p @ anchor
            local_anchor = anchor
        elif mixing_algorithm in _AVERAGING_MIXINGS:
            # Signs aligned before averaging; see `utils.align_column_signs`.
            # A latent is defined only up to a sign per component, so two views
            # can recover the same factors with opposite signs and the mean
            # then cancels them. Measured on a planted signal: per-view latents
            # at R^2 0.784 and 0.785, their plain average at 0.023.
            if len(proj_list) == 1:
                local_u = proj_list[0]
                signs = [torch.ones(k, device=local_u.device, dtype=local_u.dtype)] if need_anchor else None
            else:
                aligned = align_column_signs(proj_list)
                local_u = torch.mean(torch.stack(aligned), dim=0)
                # The mean is big_p @ A with A = [diag(s_1); ...; diag(s_M)] / M.
                signs = ([torch.where((a * p).sum(dim=0) < 0, -1.0, 1.0).to(p.dtype)
                          for a, p in zip(aligned, proj_list)]
                         if need_anchor else None)
            mean_u = local_u
            if mixing_algorithm == "newton":
                local_u = newton_schulz_orthogonalize(mean_u, iterations=10)
            if need_anchor:
                with torch.no_grad():
                    local_anchor = torch.cat([torch.diag(sg) for sg in signs], dim=0) / len(proj_list)
                    if mixing_algorithm == "newton":
                        # Newton-Schulz iterates X <- X p(X^T X), so its result is
                        # mean_u @ W for a k x k W; recover W to keep the anchor exact.
                        local_anchor = local_anchor @ (torch.linalg.pinv(mean_u.detach()) @ local_u.detach())
            else:
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
                    # Fit on the non-constant columns. A view weighted 0 by
                    # MAI contributes all-zero columns, and FastICA's whitening
                    # returned NaN on them. Those columns get zero anchor rows,
                    # so big_p @ anchor is unchanged.
                    keep = np.ptp(avg_p, axis=0) > 0
                    u_np = ica.fit_transform(avg_p[:, keep])
                    comp = np.zeros((avg_p.shape[1], ica.components_.shape[0]), dtype=ica.components_.dtype)
                    comp[keep] = ica.components_.T
                    local_anchor = torch.from_numpy(comp).to(local_big_p.device).to(local_big_p.dtype)
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
        if weight_vals is not None and sum(weight_vals) <= 1e-12:
            norm_projs = _neighbourhood(list(range(len(norm_projs))))
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
            loo_idx = [j for j in range(len(orig_norm_projs)) if j != i and j in valid_indices]
            loo_projs = _neighbourhood(loo_idx) if loo_idx else []
            if len(loo_projs) == 0:
                u_list.append(orig_norm_projs[i])
            else:
                u_list.append(_get_u(loo_projs, return_anchor=False))
                
        if training:
            with torch.no_grad():
                _, new_anchor = _get_u([p.detach() for p in norm_projs], return_anchor=True)
            return u_list, new_anchor
        return u_list

    elif topology == "graph":
        if path_graph is None:
            raise ValueError("path_graph must be provided for topology='graph'")
            
        u_list = []
        for i in range(len(orig_norm_projs)):
            neighbors = path_graph.get(i, [])
            nb_idx = [j for j in neighbors if j in valid_indices]
            neighbor_projs = _neighbourhood(nb_idx) if nb_idx else []
            if len(neighbor_projs) == 0:
                u_list.append(orig_norm_projs[i])
            else:
                u_list.append(_get_u(neighbor_projs, return_anchor=False))
                
        if training:
            with torch.no_grad():
                _, new_anchor = _get_u([p.detach() for p in norm_projs], return_anchor=True)
            return u_list, new_anchor
        return u_list

    # Unreachable: the topology is validated above. Kept so a future branch
    # that forgets to return has a defined result rather than None.
    return _get_u(norm_projs, return_anchor=training)


#: MAI metrics `mai_score` implements. ``trace`` was removed: tr(Z'u) depends
#: on the arbitrary rotation of the latent coordinates (measured 0.0348, 0.1319
#: and 0.0000 for one pair under three rotations), and on `flow_simr_v` it
#: ranked a planted pure-noise view above a signal view (0.276 vs 0.246).
MAI_METRICS = ("procrustes_r2", "procrustes_r2_sharp", "cca", "rvcoef", "max_cc")

#: Metrics on a correlation scale (r); the others are on an R^2 scale. The
#: degrees-of-freedom correction squares an r-scale score, adjusts it as an
#: R^2, and takes the root again.
_R_SCALE_MAI = frozenset({"cca", "max_cc"})


def _orthobasis(x: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis of the centred column space of ``x``, rank-adaptive:
    directions with singular value below ``rtol * s_max`` are dropped, so a
    view whose latent has rank r < k is scored on its r real directions."""
    xc = x - x.mean(dim=0, keepdim=True)
    u, sv, _ = torch.linalg.svd(xc, full_matrices=False)
    keep = sv > rtol * (sv.max() if sv.numel() else 0.0)
    return u[:, keep]


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
    ``max_cc`` is the first canonical correlation, sigma_1(Q_z' Q_u) with Q
    orthonormal bases of the (rank-adaptive) column spaces: the strongest
    single direction the view shares with the consensus. Every other metric
    aggregates over all k directions, so a view that shares one component
    strongly and nothing else (a sub-coupling) is down-weighted by them, and
    their value depends on how the view's variance is spread over columns,
    which differs when views have very different rank or width. Whitening
    (rather than standardising each column) keeps it invariant to the
    latents' arbitrary rotation, the property `trace` failed.

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
    if metric == "max_cc":
        qz, qu = _orthobasis(z), _orthobasis(u_loo)
        if qz.shape[1] == 0 or qu.shape[1] == 0:
            return 0.0
        return float(torch.linalg.svdvals(qz.t() @ qu)[0].clamp(max=1.0).item())
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


def mai_dof_adjust(score: float, metric: str, n: int, dof: Optional[float], k: int = 1) -> float:
    """Remove the agreement a view's latent can reach by fitting alone.

    A latent column is a projection with ``dof`` free coefficients, fitted on
    the same n rows the agreement is scored on, so even unrelated data reach
    R^2 ~ dof / (n - 1). Adjusted as an R^2:
    1 - (1 - R^2)(n - 1)/(n - dof - 1), clipped to [0, 1]; r-scale metrics are
    squared first and rooted after. With dof >= n - 1 the view can fit
    anything, so its agreement says nothing either way: the score is NaN (not
    assessable), and `mai_gated_target` gives the view its uniform share
    rather than 0. ``dof=None`` returns the score unchanged.
    ``cca`` is the mean of k singular values of unit-norm latents, so it lies
    in [0, 1/k]; it is adjusted as the correlation k * cca and scaled back.
    """
    if dof is None:
        return float(score)
    if n - dof - 1 <= 0:
        return float("nan")
    scale = float(k) if metric == "cca" else 1.0
    r2 = (scale * float(score)) ** 2 if metric in _R_SCALE_MAI else float(score)
    adj = min(1.0, max(0.0, 1.0 - (1.0 - r2) * (n - 1) / (n - dof - 1)))
    return float(np.sqrt(adj)) / scale if metric in _R_SCALE_MAI else adj


def loo_mai_calibrated(latents: List[torch.Tensor], metric: str, dofs: Optional[Sequence[float]] = None,
                       use_rank: bool = False, n_perm: int = 20, quantile: float = 0.95,
                       seed: int = 0, n_fit: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """Leave-one-out MAI, degrees-of-freedom corrected, with a per-view null floor.

    Returns ``(scores, floors)``; ``floors`` is ``None`` when ``n_perm=0``
    (the caller reuses earlier floors; see `MAI_FLOOR_EVERY`). ``scores`` are `loo_mai_scores` adjusted by
    `mai_dof_adjust` with each view's ``dofs`` (effective coefficients per
    latent column; ``None`` skips the correction). ``floors`` are the
    ``quantile`` of the same adjusted metric with the rows of the consensus
    permuted ``n_perm`` times: what the metric gives with no shared signal, at
    this n, k, rank and metric (for example sigma_1 of random matrices is near
    sqrt(k / n), not 0). Gate on the floor (`mai_gated_target`), not on the
    mean: a view is kept because it beats chance, not because it beats the
    other views.
    """
    check_mai_metric(metric)
    if use_rank:
        latents = [torch.argsort(torch.argsort(p, dim=0), dim=0).to(p.dtype) for p in latents]
    norm = []
    for p in latents:
        pc = p - p.mean(dim=0, keepdim=True); pn = torch.norm(pc, p="fro")
        norm.append(pc / pn if pn > 1e-8 else torch.zeros_like(pc))
    n = int(norm[0].shape[0]); M = len(norm); k = int(norm[0].shape[1])
    # The dof correction removes the agreement a basis gains by being fitted,
    # which scales with the rows it was fitted on. The deep and flow models
    # score MAI on a mini-batch (64 rows by default) but fit on the whole
    # training set: correcting with the batch n scored any view with ~64+
    # non-zero loadings per column as not assessable and most others as 0
    # (ADNI-MCI: dof 80/80/31, n_fit 758). Scores and floors stay on the batch.
    n_adj = int(n_fit) if n_fit is not None else n
    want_floor = int(n_perm) > 0
    dofs = [None] * M if dofs is None else list(dofs)
    g = torch.Generator().manual_seed(int(seed))
    scores, floors = [], []
    for i in range(M):
        others = [norm[j] for j in range(M) if j != i]
        if not others:
            scores.append(1.0); floors.append(0.0); continue
        u = torch.mean(torch.stack(align_column_signs(others)), dim=0)
        un = torch.norm(u, p="fro")
        if un <= 1e-8:
            scores.append(0.0); floors.append(0.0); continue
        u = u / un
        scores.append(mai_dof_adjust(mai_score_or_zero(norm[i], u, metric, i), metric, n_adj, dofs[i], k))
        if not want_floor:
            continue
        if metric == "max_cc":
            # Permuting u's rows permutes its basis rows: build both bases once,
            # and each null draw is one k x k singular value.
            qz, qu = _orthobasis(norm[i]), _orthobasis(u)
            raw = ([] if qz.shape[1] == 0 or qu.shape[1] == 0 else
                   [float(torch.linalg.svdvals(qz.t() @ qu[torch.randperm(n, generator=g)])[0])
                    for _ in range(int(n_perm))])
        else:
            raw = [mai_score_or_zero(norm[i], u[torch.randperm(n, generator=g)], metric, i)
                   for _ in range(int(n_perm))]
        null = [mai_dof_adjust(r, metric, n_adj, dofs[i], k) for r in raw]
        null = [x for x in null if np.isfinite(x)]
        floors.append(float(np.quantile(null, quantile)) if null else
                      (float("nan") if not np.isfinite(scores[-1]) else 0.0))
    return scores, (floors if want_floor else None)


#: MAI weights move from uniform to their target over this many updates (sweeps
#: for simlr, epochs for the deep and flow fits, after a 10-epoch warmup). It
#: was half the budget: at the 300 default a simlr fit converges in ~50 sweeps
#: and its weights went only a third of the way.
MAI_RAMP_STEPS = 20

#: Null floors change slowly with the latents; recompute them every this many
#: MAI updates and reuse them in between (the permutations dominate the cost).
MAI_FLOOR_EVERY = 10


def view_dofs(v_mats: Sequence[torch.Tensor], n: int, metric: str = "procrustes_r2") -> List[float]:
    """Effective coefficients behind each view's score, from its first-layer basis.

    For the aggregate metrics, the mean number of non-zero entries per column
    (each latent column is one fitted projection). For ``max_cc``, the number of
    features any column uses: the first canonical direction combines all k
    columns, so it is a projection on their union (per-column counts let a
    p > n noise view keep r ~ 1). Not capped: a view with n - 1 or more can fit
    any target, and `mai_dof_adjust` scores it 0.
    """
    out = []
    for v in v_mats:
        nz = v.detach().abs() > 1e-12
        out.append(float(nz.any(dim=1).sum().item()) if metric == "max_cc"
                   else float(nz.sum(dim=0).float().mean().item()))
    return out


def mai_ema(prev: torch.Tensor, new: torch.Tensor, decay: float, first: bool) -> torch.Tensor:
    """Exponential moving average of MAI scores or floors, NaN-aware: a view
    not assessable now (NaN) is NaN, and one that becomes assessable restarts
    from its new value instead of averaging with NaN forever."""
    if first:
        return new
    blend = decay * prev + (1.0 - decay) * new
    return torch.where(torch.isnan(new), new, torch.where(torch.isnan(prev), new, blend))


#: Bounds on every view's modality weight. The upper bound keeps any single
#: view from supplying more than half the consensus (on ADNI-MCI a 0.006
#: excess over a null floor of 0 took all the weight); the lower bound lets a
#: view with no evidence of coupling fall to 1% of it without ever reaching
#: exactly 0. Weights are the Euclidean projection of the gated target onto
#: {w : sum w = 1, lo <= w <= hi} (`capped_simplex`). With two views the
#: bounds force 0.5 / 0.5: MAI cannot reweight two views.
MAI_WEIGHT_BOUNDS = (0.01, 0.5)


def capped_simplex(target: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Euclidean projection of ``target`` onto ``{w : sum w = 1, lo <= w <= hi}``.

    ``w_i = clip(target_i - tau, lo, hi)`` with ``tau`` found by bisection
    (the map tau -> sum w is monotone). Requires ``M lo <= 1 <= M hi``; with
    one view the only weight is 1.
    """
    M = len(target)
    if M == 1:
        return torch.ones_like(target)
    if M * lo > 1.0 + 1e-12 or M * hi < 1.0 - 1e-12:
        raise ValueError(f"weight bounds ({lo}, {hi}) are infeasible for {M} views.")
    t = target.double()
    a, b = float(t.min()) - hi - 1.0, float(t.max()) - lo + 1.0
    for _ in range(100):
        tau = 0.5 * (a + b)
        if float((t - tau).clamp(lo, hi).sum()) > 1.0:
            a = tau
        else:
            b = tau
    return (t - 0.5 * (a + b)).clamp(lo, hi).to(target.dtype)


def mai_gated_target(mai: torch.Tensor, steepness: float, floor: torch.Tensor,
                     bounds: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """Target modality weights: each view's score above its null floor, times a
    sigmoid gate centred on that floor, normalised, then projected onto the
    weight bounds (`MAI_WEIGHT_BOUNDS`, default ``(0.01, 0.5)``; see
    `capped_simplex`).

    Gating on the floor (from `loo_mai_calibrated`) favours every view that
    beats chance over those that do not. If no view exceeds its floor the
    weights are uniform, with a warning. A NaN score or floor (a view with
    dof >= n - 1, not assessable) gets the uniform share 1/M before the
    projection. If fewer than two views exceed their floors, the consensus
    rests on one view: the bound still caps it at ``hi``, and a warning says
    the rest of the weight goes to views with no evidence of coupling.
    """
    lo, hi = MAI_WEIGHT_BOUNDS if bounds is None else bounds
    M = len(mai)
    uniform = torch.ones_like(mai) / M
    if M == 1:
        return uniform
    ok = torch.isfinite(mai) & torch.isfinite(floor)
    if not bool(ok.any()):
        return uniform
    excess = (mai[ok] - floor[ok]).clamp(min=0.0)
    raw = excess * torch.sigmoid(steepness * (mai[ok] - floor[ok]))
    if float(raw.sum()) <= 1e-8:
        warnings.warn("no view's MAI exceeds its null floor; modality weights left uniform.",
                      RuntimeWarning, stacklevel=2)
        return uniform
    if int((raw > 1e-8).sum()) < 2 and M > 2:
        warnings.warn("only one view's MAI exceeds its null floor: the consensus rests on that view, "
                      f"capped at {hi:g}; the remaining weight goes to views with no evidence of "
                      "coupling.", RuntimeWarning, stacklevel=2)
    target = uniform.clone()
    # Not-assessable views keep 1/M; the assessable ones share the rest.
    target[ok] = raw / raw.sum() * (float(ok.sum()) / M)
    return capped_simplex(target, lo, hi)


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
        # float64 on the CPU: one solve per fit, and MPS has no float64.
        p = torch.nan_to_num(p.detach().cpu().double(), nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(torch.norm(p, p="fro") / np.sqrt(max(1, p.shape[0])))
        proj_rms.append(rms)
        normed.append(p / (rms + 1e-8))
    z = torch.cat(normed, dim=1)
    mu = z.mean(dim=0, keepdim=True)
    zc = z - mu
    targets = list(u) if isinstance(u, (list, tuple)) else [u]
    maps, offsets, resid = [], [], 0.0
    for t in targets:
        t = t.detach().cpu().double()
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
    work = torch.device("cpu") if device.type == "mps" else device     # MPS has no float64
    normed = [torch.nan_to_num(p.to(work).double(), nan=0.0, posinf=0.0, neginf=0.0) / (rms + 1e-8)
              for p, rms in zip(projections, frozen["proj_rms"])]
    zc = torch.cat(normed, dim=1) - frozen["mu"].to(work)
    outs = [(zc @ r.to(work) + b.to(work)).to(device=device, dtype=dtype)
            for r, b in zip(frozen["maps"], frozen["offsets"])]
    return outs if frozen["is_list"] else outs[0]
