"""First-class confound sink for multi-view representation learning.

`simlr(..., confounds=...)` isolates known nuisance variation (scanner drift,
head motion FD, tSNR, site, or physiological covariates) into a dedicated
per-view signed sink subspace while keeping the primary biological encoders
``V_m`` (`result["v"]`) strictly sparse and sign-constrained (`positivity`) and
the shared consensus ``U`` (`result["u"]`) orthogonal to (or softly damped
against) the nuisance basis ``Q_C``.

At prediction time (`predict_simlr` / `pysimlr.predict`), each view's frozen
signed sink encoder ``V_{m,sink}`` estimates the artifact score directly from
the held-out features ``X_{m,test}`` and partials it out via training-frozen
weights ``B_m`` -- requiring zero held-out confound measurements and preserving
exact ``N = 1`` single-subject batch invariance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .utils import safe_svd

_RIDGE_GRID = (1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0)


def validate_confound_params(confound_strength: float,
                             confound_rank: Optional[int],
                             confound_partial_out: bool) -> Tuple[float, Optional[int], bool]:
    """Validate `confound_strength`, `confound_rank` and `confound_partial_out`."""
    if isinstance(confound_strength, bool) or not isinstance(confound_strength, (int, float, np.floating, np.integer)):
        raise ValueError(f"confound_strength must be a float in [0.0, 1.0]; got {confound_strength!r}.")
    gamma = float(confound_strength)
    if not np.isfinite(gamma) or gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"confound_strength must lie in [0.0, 1.0]; got {confound_strength!r}.")
    rank: Optional[int] = None
    if confound_rank is not None:
        if isinstance(confound_rank, bool) or not isinstance(confound_rank, (int, np.integer)) or int(confound_rank) < 1:
            raise ValueError(f"confound_rank must be a positive integer or None; got {confound_rank!r}.")
        rank = int(confound_rank)
    if not isinstance(confound_partial_out, bool):
        raise ValueError(f"confound_partial_out must be a bool; got {confound_partial_out!r}.")
    return gamma, rank, bool(confound_partial_out)


def _is_single_matrix(obj: Any) -> bool:
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return True
    return hasattr(obj, "to_numpy")


def _coerce_confound_matrix(mat: Any, n_samples: int, label: str) -> torch.Tensor:
    """Convert a 1-D or 2-D array/tensor/DataFrame/Series to a float64 ``(n, q)`` CPU tensor."""
    if hasattr(mat, "to_numpy"):
        arr = np.asarray(mat.to_numpy(), dtype=np.float64)
    elif isinstance(mat, torch.Tensor):
        arr = mat.detach().cpu().to(torch.float64).numpy()
    else:
        arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{label} must be 1-D or 2-D (samples x confounds); got shape {tuple(arr.shape)}.")
    if arr.shape[0] != n_samples:
        raise ValueError(f"{label} has {arr.shape[0]} rows; views have {n_samples} samples.")
    if arr.shape[1] == 0:
        raise ValueError(f"{label} has 0 columns.")
    return torch.from_numpy(np.array(arr, dtype=np.float64, copy=True, order="C"))


def _resolve_view_key(key: Any, n_views: int) -> int:
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        idx = int(key)
        if 0 <= idx < n_views:
            return idx
    if isinstance(key, str) and key.startswith("view") and key[4:].isdigit():
        idx = int(key[4:])
        if 0 <= idx < n_views:
            return idx
    raise ValueError(f"unknown view key {key!r} in confounds; expected an index in 0..{n_views - 1} "
                     f"or 'view0'..'view{n_views - 1}'.")


def _slice_one(mat: Any, idx_np: np.ndarray) -> Any:
    if mat is None:
        return None
    if hasattr(mat, "iloc"):
        return mat.iloc[idx_np]
    if isinstance(mat, torch.Tensor):
        return mat[torch.as_tensor(idx_np, dtype=torch.long, device=mat.device)]
    arr = np.asarray(mat)
    return arr[idx_np]


def slice_confounds(confounds: Any, indices: Any) -> Any:
    """Slice row-indexed `confounds` (shared matrix, per-view sequence, or dict) by `indices`."""
    if confounds is None:
        return None
    idx_np = (indices.detach().cpu().numpy() if isinstance(indices, torch.Tensor)
              else np.asarray(indices, dtype=np.int64))
    if _is_single_matrix(confounds):
        return _slice_one(confounds, idx_np)
    if isinstance(confounds, Mapping):
        return {k: _slice_one(v, idx_np) for k, v in confounds.items()}
    if isinstance(confounds, (list, tuple)):
        sliced = [_slice_one(v, idx_np) for v in confounds]
        return tuple(sliced) if isinstance(confounds, tuple) else sliced
    return _slice_one(confounds, idx_np)


def permute_multiview_confounds(confounds: Any, perm_list: Sequence[np.ndarray], n_views: int) -> Any:
    """Permute confounds for views ``1..n_views-1`` matching `loading_inference` null shuffles."""
    if confounds is None:
        return None
    all_perms = [None] + [np.asarray(p, dtype=np.int64) for p in perm_list]
    if _is_single_matrix(confounds):
        return [confounds if p is None else _slice_one(confounds, p) for p in all_perms]
    if isinstance(confounds, Mapping):
        out = {}
        for k, v in confounds.items():
            idx = _resolve_view_key(k, n_views)
            p = all_perms[idx]
            out[k] = v if (v is None or p is None) else _slice_one(v, p)
        return out
    if isinstance(confounds, (list, tuple)):
        items = list(confounds)
        if len(items) != n_views:
            return confounds
        return [v if (v is None or all_perms[m] is None) else _slice_one(v, all_perms[m])
                for m, v in enumerate(items)]
    return confounds


def permute_single_view_confounds(confounds: Any, view_idx: int, perm: Any, n_views: int) -> Any:
    """Permute view `view_idx`'s confound rows matching `feature_relevance` null refits."""
    if confounds is None:
        return None
    p_np = perm.detach().cpu().numpy() if isinstance(perm, torch.Tensor) else np.asarray(perm, dtype=np.int64)
    if _is_single_matrix(confounds):
        return [_slice_one(confounds, p_np) if m == view_idx else confounds for m in range(n_views)]
    if isinstance(confounds, Mapping):
        out = {}
        for k, v in confounds.items():
            idx = _resolve_view_key(k, n_views)
            out[k] = _slice_one(v, p_np) if (idx == view_idx and v is not None) else v
        return out
    if isinstance(confounds, (list, tuple)):
        items = list(confounds)
        if len(items) != n_views:
            return confounds
        return [_slice_one(v, p_np) if (m == view_idx and v is not None) else v
                for m, v in enumerate(items)]
    return confounds


def _standardize_confound_columns(c_raw: torch.Tensor) -> torch.Tensor:
    """Standardize each column over finite entries and impute non-finite entries to ``0.0``.

    Subtracting the finite mean and setting missing entries to ``0.0`` leaves
    every column with exact zero mean across all ``n`` rows; columns with
    near-zero variance (fewer than 2 finite values or SD <= 1e-12) are dropped.
    """
    cols: List[torch.Tensor] = []
    for j in range(c_raw.shape[1]):
        col = c_raw[:, j]
        fin = torch.isfinite(col)
        n_fin = int(fin.sum().item())
        if n_fin < 2:
            continue
        vals = col[fin]
        mu = vals.mean()
        centered = torch.where(fin, col - mu, torch.zeros_like(col))
        sd = float(vals.std(unbiased=True).item())
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        cols.append(centered / sd)
    if not cols:
        return torch.zeros((c_raw.shape[0], 0), dtype=torch.float64)
    return torch.stack(cols, dim=1)


def _orthonormal_nuisance_basis(c_std: torch.Tensor,
                                confound_rank: Optional[int],
                                n_samples: int) -> torch.Tensor:
    """Extract top ``k_c`` zero-mean orthonormal left singular vectors ``Q_C``."""
    q_tot = int(c_std.shape[1])
    if q_tot == 0 or n_samples < 2:
        return torch.zeros((n_samples, 0), dtype=torch.float64)
    c_centered = c_std - c_std.mean(dim=0, keepdim=True)
    u_svd, s_svd, _ = safe_svd(c_centered, full_matrices=False)
    if s_svd.numel() == 0 or float(s_svd[0].item()) <= 1e-12:
        return torch.zeros((n_samples, 0), dtype=torch.float64)
    if confound_rank is None:
        cap = max(1, min(q_tot, max(1, n_samples // 4), n_samples - 1))
    else:
        cap = max(1, min(int(confound_rank), q_tot, max(1, n_samples - 1)))
    keep = int((s_svd > 1e-6 * float(s_svd[0].item())).sum().item())
    k_eff = min(cap, max(1, keep)) if keep > 0 else 0
    if k_eff == 0:
        return torch.zeros((n_samples, 0), dtype=torch.float64)
    q = u_svd[:, :k_eff].double()
    q = q - q.mean(dim=0, keepdim=True)
    q_qr, r_qr = torch.linalg.qr(q, mode="reduced")
    diag_signs = torch.where(torch.diagonal(r_qr) < 0.0, -1.0, 1.0)
    return (q_qr * diag_signs.unsqueeze(0)).contiguous()


def prepare_confound_inputs(confounds: Any,
                            n_samples: int,
                            n_views: int,
                            confound_strength: float = 1.0,
                            confound_rank: Optional[int] = None,
                            confound_partial_out: bool = True) -> Optional[Dict[str, Any]]:
    """Parse `confounds` into global ``Q_C`` and per-view ``Q_{C,m}`` bases."""
    gamma, rank, partial_out = validate_confound_params(
        confound_strength, confound_rank, confound_partial_out
    )
    if confounds is None:
        return None

    per_view_raw: List[Optional[torch.Tensor]]
    is_shared = False
    if _is_single_matrix(confounds):
        c_mat = _coerce_confound_matrix(confounds, n_samples, "confounds")
        per_view_raw = [c_mat] * n_views
        is_shared = True
    elif isinstance(confounds, Mapping):
        if not confounds:
            raise ValueError("confounds mapping is empty.")
        per_view_raw = [None] * n_views
        for k, v in confounds.items():
            idx = _resolve_view_key(k, n_views)
            if v is not None:
                per_view_raw[idx] = _coerce_confound_matrix(v, n_samples, f"confounds[{k!r}]")
        if all(v is None for v in per_view_raw):
            raise ValueError("confounds mapping contains only None entries.")
    elif isinstance(confounds, (list, tuple)):
        if len(confounds) != n_views:
            raise ValueError(f"per-view confounds has {len(confounds)} entries; there are {n_views} views.")
        per_view_raw = [
            None if v is None else _coerce_confound_matrix(v, n_samples, f"confounds[{m}]")
            for m, v in enumerate(confounds)
        ]
        if all(v is None for v in per_view_raw):
            raise ValueError("per-view confounds contains only None entries.")
    else:
        c_mat = _coerce_confound_matrix(confounds, n_samples, "confounds")
        per_view_raw = [c_mat] * n_views
        is_shared = True

    if is_shared:
        c_std = _standardize_confound_columns(per_view_raw[0])
        q_global = _orthonormal_nuisance_basis(c_std, rank, n_samples)
        q_per_view = [q_global for _ in range(n_views)]
        c_std_all = c_std
    else:
        std_blocks: List[torch.Tensor] = []
        q_per_view = []
        for m in range(n_views):
            if per_view_raw[m] is None:
                q_per_view.append(torch.zeros((n_samples, 0), dtype=torch.float64))
            else:
                cm_std = _standardize_confound_columns(per_view_raw[m])
                if cm_std.shape[1] > 0:
                    std_blocks.append(cm_std)
                q_per_view.append(_orthonormal_nuisance_basis(cm_std, rank, n_samples))
        c_std_all = (torch.cat(std_blocks, dim=1) if std_blocks
                     else torch.zeros((n_samples, 0), dtype=torch.float64))
        q_global = _orthonormal_nuisance_basis(c_std_all, rank, n_samples)

    return {
        "q_global": q_global,
        "q_per_view": q_per_view,
        "c_std_all": c_std_all,
        "gamma": gamma,
        "confound_rank": int(q_global.shape[1]),
        "confound_partial_out": partial_out,
    }


def deflate_views_sample_space(torch_mats: Sequence[torch.Tensor],
                               q_global: Optional[torch.Tensor],
                               gamma: float) -> List[torch.Tensor]:
    """Return ``[X_m - gamma * Q_C (Q_C^T X_m)]`` without mutating `torch_mats`."""
    if q_global is None or q_global.shape[1] == 0 or gamma <= 0.0:
        return list(torch_mats)
    out: List[torch.Tensor] = []
    for x in torch_mats:
        q = q_global.to(device=x.device, dtype=torch.float64)
        x64 = x.double()
        xd = x64 - float(gamma) * (q @ (q.t() @ x64))
        out.append(xd.to(x.dtype))
    return out


def deflate_views_per_view(torch_mats: Sequence[torch.Tensor],
                           q_per_view: Sequence[torch.Tensor],
                           gamma: float) -> List[torch.Tensor]:
    """Return ``[X_m - gamma * Q_{C,m} (Q_{C,m}^T X_m)]`` without mutating `torch_mats`."""
    if gamma <= 0.0:
        return list(torch_mats)
    out: List[torch.Tensor] = []
    for x, qm in zip(torch_mats, q_per_view):
        if qm is None or int(qm.shape[1]) == 0:
            out.append(x)
            continue
        q = qm.to(device=x.device, dtype=torch.float64)
        x64 = x.double()
        xd = x64 - float(gamma) * (q @ (q.t() @ x64))
        out.append(xd.to(x.dtype))
    return out


def _centered_gcv_ridge(xc: torch.Tensor, yc: torch.Tensor,
                        rel_grid: Sequence[float] = _RIDGE_GRID,
                        max_df_frac: float = 0.65) -> torch.Tensor:
    """Solve ``xc W ~ yc`` on zero-mean ``(n, d)`` inputs with bounded-DOF GCV ridge.

    Centred ``(n, d)`` matrices live in the ``(n - 1)``-dimensional zero-mean
    subspace, so when ``d >= n - 1`` an unregularized denominator ``1 - df / n``
    stays positive (``1 / n``) at ``df = n - 1`` and interpolates noise. Using
    ``n_eff = n - 1`` and capping ``df_eff <= max_df_frac * n_eff`` keeps out-of-
    sample sink predictions regularized on wide views (``p_m >= n``) as well as
    narrow ones.
    """
    n, d = int(xc.shape[0]), int(xc.shape[1])
    k = int(yc.shape[1])
    if n < 3 or d == 0 or k == 0:
        return torch.zeros((d, k), dtype=torch.float64, device=xc.device)
    u_svd, s_svd, vh_svd = safe_svd(xc, full_matrices=False)
    if s_svd.numel() == 0 or float(s_svd[0].item()) <= 1e-15:
        return torch.zeros((d, k), dtype=torch.float64, device=xc.device)
    s2 = s_svd ** 2
    live = s2 > 1e-24 * float(s2[0].item())
    scale = float(s2[live].mean().item()) if bool(live.any()) else float(s2.mean().item())
    ut_y = u_svd.t() @ yc
    y_norm2 = float((yc ** 2).sum().item())
    proj_norm2 = float((ut_y ** 2).sum().item())
    orth_res2 = max(0.0, y_norm2 - proj_norm2)
    n_eff = float(max(n - 1, 1))
    best_lam = float(rel_grid[-1]) * scale
    best_gcv = float("inf")
    for rel in rel_grid:
        lam = float(rel) * scale
        filt = s2 / (s2 + lam)
        df_eff = float(filt.sum().item())
        if df_eff > max_df_frac * n_eff:
            continue
        denom = (1.0 - df_eff / n_eff) ** 2
        if denom <= 1e-6:
            continue
        in_span_res2 = float((((1.0 - filt).unsqueeze(1) * ut_y) ** 2).sum().item())
        gcv = (orth_res2 + in_span_res2) / (n_eff * denom)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam
    coeff = (s_svd / (s2 + best_lam)).unsqueeze(1) * ut_y
    return vh_svd.t() @ coeff


def fit_sink_encoders(torch_mats: Sequence[torch.Tensor],
                      prepped_conf: Dict[str, Any]) -> Dict[str, Any]:
    """Fit per-view signed sink encoders ``V_{m,sink}`` and feature-space bases ``H_m``."""
    q_global: torch.Tensor = prepped_conf["q_global"]
    q_per_view: List[torch.Tensor] = prepped_conf["q_per_view"]
    k_c = int(q_global.shape[1])
    v_sink: List[torch.Tensor] = []
    latents_sink: List[torch.Tensor] = []
    sink_means: List[torch.Tensor] = []
    sink_to_q: List[torch.Tensor] = []
    sink_to_u: List[torch.Tensor] = []
    h_per_view: List[torch.Tensor] = []

    for m, x in enumerate(torch_mats):
        orig_dtype, dev = x.dtype, x.device
        p_m = int(x.shape[1])
        qm = q_per_view[m].to(device=dev, dtype=torch.float64)
        qg = q_global.to(device=dev, dtype=torch.float64)
        k_cm = int(qm.shape[1])
        if k_cm == 0:
            v_sink.append(torch.zeros((p_m, 0), dtype=orig_dtype, device=dev))
            latents_sink.append(torch.zeros((int(x.shape[0]), 0), dtype=orig_dtype, device=dev))
            sink_means.append(torch.zeros((1, 0), dtype=torch.float64, device="cpu"))
            sink_to_q.append(torch.zeros((0, 0), dtype=torch.float64, device="cpu"))
            sink_to_u.append(torch.zeros((0, k_c), dtype=torch.float64, device="cpu"))
            h_per_view.append(torch.zeros((p_m, 0), dtype=torch.float64, device=dev))
            continue

        x64 = x.double()
        xc = x64 - x64.mean(dim=0, keepdim=True)
        raw_v = _centered_gcv_ridge(xc, qm)
        fallback = xc.t() @ qm
        norms = torch.linalg.vector_norm(raw_v, dim=0, keepdim=True)
        fb_norms = torch.linalg.vector_norm(fallback, dim=0, keepdim=True)
        use_v = torch.where(
            norms > 1e-15,
            raw_v / norms.clamp_min(1e-30),
            torch.where(fb_norms > 1e-15, fallback / fb_norms.clamp_min(1e-30), torch.zeros_like(raw_v)),
        )
        vm_out = use_v.to(orig_dtype)
        z_sink = x64 @ vm_out.double()
        mu_sink = z_sink.mean(dim=0, keepdim=True)
        z_sink_c = z_sink - mu_sink

        # Narrow ridge maps from view m's centered sink scores to Q_{C,m} and global Q_C.
        m_q = _centered_gcv_ridge(z_sink_c, qm, rel_grid=(1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 10.0),
                                  max_df_frac=0.85)
        m_u = (_centered_gcv_ridge(z_sink_c, qg, rel_grid=(1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 10.0),
                                   max_df_frac=0.85)
               if k_c > 0 else torch.zeros((k_cm, 0), dtype=torch.float64, device=dev))

        cov_pat = xc.t() @ qm
        uh, sh, _ = safe_svd(cov_pat, full_matrices=False)
        keep_h = int((sh > 1e-8 * float(sh[0].item())).sum().item()) if sh.numel() > 0 and float(sh[0].item()) > 0 else 0
        hm = uh[:, :keep_h].contiguous() if keep_h > 0 else torch.zeros((p_m, 0), dtype=torch.float64, device=dev)

        v_sink.append(vm_out.detach())
        latents_sink.append(z_sink.to(orig_dtype).detach())
        sink_means.append(mu_sink.detach().cpu())
        sink_to_q.append(m_q.detach().cpu())
        sink_to_u.append(m_u.detach().cpu())
        h_per_view.append(hm.detach())

    return {
        **prepped_conf,
        "v_sink": v_sink,
        "latents_sink": latents_sink,
        "sink_means": sink_means,
        "sink_to_q": sink_to_q,
        "sink_to_u": sink_to_u,
        "h_per_view": h_per_view,
    }


def max_abs_confound_corr(u: torch.Tensor, c_std_all: torch.Tensor) -> float:
    """Maximum absolute Pearson correlation across columns of `u` and `c_std_all`."""
    if c_std_all.numel() == 0 or u.numel() == 0 or int(u.shape[0]) < 2:
        return 0.0
    u64 = u.detach().cpu().double()
    c64 = c_std_all.detach().cpu().double()
    uc = u64 - u64.mean(dim=0, keepdim=True)
    cc = c64 - c64.mean(dim=0, keepdim=True)
    un = torch.linalg.vector_norm(uc, dim=0, keepdim=True)
    cn = torch.linalg.vector_norm(cc, dim=0, keepdim=True)
    us = torch.where(un > 1e-15, uc / un.clamp_min(1e-30), torch.zeros_like(uc))
    cs = torch.where(cn > 1e-15, cc / cn.clamp_min(1e-30), torch.zeros_like(cc))
    corr = (us.t() @ cs).abs()
    return float(corr.max().item()) if corr.numel() > 0 else 0.0


def build_confound_sink_result(torch_mats: Sequence[torch.Tensor],
                               v_mats: Sequence[torch.Tensor],
                               sink_fit: Dict[str, Any]) -> Dict[str, Any]:
    """Compute frozen partial-out weights ``B_m`` from fitted ``V_{m,bio}`` and ``V_{m,sink}``."""
    orig_dtype = torch_mats[0].dtype
    dev = torch_mats[0].device
    q_global: torch.Tensor = sink_fit["q_global"]
    q_per_view: List[torch.Tensor] = sink_fit["q_per_view"]
    k = int(v_mats[0].shape[1])
    partial_weights: List[torch.Tensor] = []

    for m, (x, v) in enumerate(zip(torch_mats, v_mats)):
        qm = q_per_view[m].to(device=x.device, dtype=torch.float64)
        k_cm = int(qm.shape[1])
        if k_cm == 0:
            partial_weights.append(torch.zeros((0, k), dtype=orig_dtype, device=x.device))
            continue
        s_raw = x.double() @ v.double()
        s_raw_c = s_raw - s_raw.mean(dim=0, keepdim=True)
        # Nuisance loading of X_m V_{m,bio} on Q_{C,m}, mapped from image-derived Z_{m,sink}:
        # Z_{m,sink,c} @ B_m = (Z_{m,sink,c} @ M_{q,m}) @ (Q_{C,m}^T S_{m,raw}).
        a_m = qm.t() @ s_raw_c
        m_q = sink_fit["sink_to_q"][m].to(device=x.device, dtype=torch.float64)
        b_m = m_q @ a_m
        partial_weights.append(b_m.to(orig_dtype).detach())

    return {
        "u_sink": q_global.to(device=dev, dtype=orig_dtype).detach(),
        "v_sink": sink_fit["v_sink"],
        "latents_sink": sink_fit["latents_sink"],
        "partial_weights": partial_weights,
        "confound_rank": int(sink_fit["confound_rank"]),
        "confound_strength": float(sink_fit["gamma"]),
        "confound_partial_out": bool(sink_fit["confound_partial_out"]),
        "max_abs_corr_before": 0.0,
        "max_abs_corr_after": 0.0,
        "_sink_means": sink_fit["sink_means"],
        "_sink_to_u": sink_fit["sink_to_u"],
    }


def apply_bio_projections(torch_mats: Sequence[torch.Tensor],
                          v_mats: Sequence[torch.Tensor],
                          confound_sink: Optional[Dict[str, Any]]) -> List[torch.Tensor]:
    """Compute per-view biological projections, partialing out frozen sink scores when enabled."""
    raw = [x @ v.to(x.dtype) for x, v in zip(torch_mats, v_mats)]
    if confound_sink is None:
        return raw
    if not confound_sink.get("confound_partial_out", True):
        return raw
    gamma = float(confound_sink.get("confound_strength", 1.0))
    if gamma <= 0.0:
        return raw
    v_sink = confound_sink["v_sink"]
    b_list = confound_sink["partial_weights"]
    means = confound_sink.get("_sink_means")
    out: List[torch.Tensor] = []
    for m, (x, r) in enumerate(zip(torch_mats, raw)):
        vs = v_sink[m]
        bm = b_list[m]
        if int(vs.shape[1]) == 0:
            out.append(r)
            continue
        x64 = x.double()
        z_sink = x64 @ vs.to(device=x.device, dtype=torch.float64)
        if means is not None and m < len(means) and means[m].numel() > 0:
            z_sink = z_sink - means[m].to(device=x.device, dtype=torch.float64)
        clean = r.double() - gamma * (z_sink @ bm.to(device=x.device, dtype=torch.float64))
        out.append(clean.to(r.dtype))
    return out


def align_training_projections_to_sink(raw_projections: Sequence[torch.Tensor],
                                       bio_projections: Sequence[torch.Tensor],
                                       q_global: torch.Tensor,
                                       gamma: float) -> List[torch.Tensor]:
    """Ensure training consensus inputs carry exact ``(1 - gamma)`` of ``raw_projections`` along ``Q_C``."""
    if q_global.shape[1] == 0:
        return list(bio_projections)
    out: List[torch.Tensor] = []
    for r, p in zip(raw_projections, bio_projections):
        q = q_global.to(device=p.device, dtype=torch.float64)
        r64, p64 = r.double(), p.double()
        target_q = (1.0 - float(gamma)) * (q.t() @ r64)
        curr_q = q.t() @ p64
        adj = p64 + q @ (target_q - curr_q)
        out.append(adj.to(p.dtype))
    return out


def predict_confound_sink(torch_mats: Sequence[torch.Tensor],
                          confound_sink: Dict[str, Any],
                          available_indices: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute ``(u_sink, latents_sink)`` on new preprocessed views (batch-size ``N = 1`` invariant)."""
    indices = list(range(len(torch_mats))) if available_indices is None else list(available_indices)
    n_rows = int(torch_mats[0].shape[0])
    dtype, dev = torch_mats[0].dtype, torch_mats[0].device
    k_c = int(confound_sink["confound_rank"])
    v_sink = confound_sink["v_sink"]
    means = confound_sink.get("_sink_means")
    sink_to_u = confound_sink.get("_sink_to_u")

    latents_sink: List[torch.Tensor] = []
    u_parts: List[torch.Tensor] = []
    for pos, m in enumerate(indices):
        x = torch_mats[pos]
        vs = v_sink[m].to(device=dev, dtype=torch.float64)
        if int(vs.shape[1]) == 0:
            latents_sink.append(torch.zeros((n_rows, 0), dtype=dtype, device=dev))
            continue
        z_s = x.double() @ vs
        latents_sink.append(z_s.to(dtype))
        if k_c > 0 and sink_to_u is not None and m < len(sink_to_u) and sink_to_u[m].numel() > 0:
            zc = z_s
            if means is not None and m < len(means) and means[m].numel() > 0:
                zc = zc - means[m].to(device=dev, dtype=torch.float64)
            u_parts.append(zc @ sink_to_u[m].to(device=dev, dtype=torch.float64))

    if u_parts:
        u_sink = torch.mean(torch.stack(u_parts, dim=0), dim=0).to(dtype)
    else:
        u_sink = torch.zeros((n_rows, k_c), dtype=dtype, device=dev)
    return u_sink, latents_sink
