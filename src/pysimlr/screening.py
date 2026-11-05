r"""Outcome-blind cross-view predictor screening for wide views (``p_m > n``).

As shown in ``D25`` (`WideViewWarning`), feeding an unscreened wide view
(``p_m >= n``) directly into an iterative multi-view solver allows the basis
``V_m`` to align training noise with the other views even under strong in-solver
``sparseness``, inflating ``coupling_train`` far above held-out coupling.

`screen_wide_views` gates wide views **strictly from the training views
themselves** (no outcome argument exists, and test rows are never seen):

1. **Regularized other-view target subspaces.** Each view :math:`\ell` is
   represented by the top-:math:`r_\ell` left singular vectors :math:`U_\ell` of
   its standardized training columns (:math:`r_\ell \ll n`, so another wide view
   cannot overfit the target). Stacking :math:`H_{-m} = [U_\ell]_{\ell \ne m} /
   \sqrt{M - 1}` automatically weights directions shared by multiple modalities
   by their MAXVAR canonical singular value squared.
2. **Cross-view subspace communality & permutation null.** Each standardized
   feature :math:`z_{m,f}` in view :math:`m` receives
   :math:`h_{m,f} = \| H_{-m}^\top z_{m,f} \|_2^2 \in [0, 1]`, the average
   multiple :math:`R^2` explained by the other views' leading subspaces. Row
   permutations of :math:`H_{-m}` yield per-feature ``p_perm`` as well as
   omnibus view-coupling p-values ``p_view_mean`` and ``p_view_max`` (testing
   whether the view shares any signal with the battery above chance).
3. **Subsample stability & redundancy-aware capacity control.** Half-sample
   training refits of :math:`U_\ell^{(b)}` record stability selection
   frequencies, and candidate features passing the cross-view gate are admitted
   greedily in descending score order while pruning near-duplicate collinear
   columns (``|r| > max_collinearity``) and optionally bounding the admitted
   width to ``floor(max_width_ratio * n)`` so ``p_out < n`` adapts smoothly with
   sample size rather than a fixed hand-chosen feature count.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


def _std_unit_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute NaNs, center columns, and L2-normalize non-constant columns."""
    arr = np.asarray(x, dtype=np.float64).copy()
    if np.isnan(arr).any():
        col_means = np.nanmean(arr, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        nan_r, nan_c = np.where(np.isnan(arr))
        arr[nan_r, nan_c] = col_means[nan_c]
    arr -= arr.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(arr, axis=0, keepdims=True)
    live = nrm[0] > 1e-12
    z = np.where(nrm > 1e-12, arr / np.where(nrm > 1e-12, nrm, 1.0), 0.0)
    return z, live


def _leading_u(z: np.ndarray, r_target: int) -> np.ndarray:
    """Top ``r`` orthonormal left singular vectors of ``z``."""
    n = z.shape[0]
    if z.shape[1] == 0:
        return np.zeros((n, 0), dtype=np.float64)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    if len(s) == 0 or s[0] <= 0.0:
        return np.zeros((n, 0), dtype=np.float64)
    rk = int((s > 1e-10 * s[0]).sum())
    rk = min(rk, int(r_target), max(1, (n - 2) // 2))
    return u[:, :rk]


def _compact_other_basis(us: Sequence[np.ndarray], exclude: int) -> np.ndarray:
    """Compact ``(n, q)`` matrix ``B`` such that ``||B.T @ z_f||_2^2 == h_{m,f}``."""
    others = [u for i, u in enumerate(us) if i != exclude and u.shape[1] > 0]
    if not others:
        return np.zeros((us[exclude].shape[0], 0), dtype=np.float64)
    h = np.hstack(others) / np.sqrt(float(len(others)))
    p_u, d_s, _ = np.linalg.svd(h, full_matrices=False)
    keep = d_s > 1e-12 * max(1.0, float(d_s[0]) if len(d_s) else 1.0)
    return p_u[:, keep] * d_s[None, keep]


def _greedy_nonredundant(z: np.ndarray, order: np.ndarray, max_keep: int,
                         max_collinearity: float | None) -> np.ndarray:
    """Select up to ``max_keep`` indices from ``order`` with pairwise ``|r| <= max_collinearity``."""
    if len(order) == 0 or max_keep <= 0:
        return np.empty(0, dtype=np.int64)
    if max_collinearity is None or max_collinearity >= 1.0:
        return np.sort(order[:max_keep].astype(np.int64))
    chosen: list[int] = []
    chosen_cols: list[np.ndarray] = []
    for idx in order:
        col = z[:, int(idx)]
        if chosen_cols:
            dots = np.abs(np.column_stack(chosen_cols).T @ col)
            if float(dots.max()) > max_collinearity:
                continue
        chosen.append(int(idx))
        chosen_cols.append(col)
        if len(chosen) >= max_keep:
            break
    return np.sort(np.asarray(chosen, dtype=np.int64))


def screen_wide_views(
    views: Sequence[Any],
    *,
    k: int = 4,
    max_width_ratio: float = 0.75,
    subspace_rank: int | None = None,
    alpha: float = 0.05,
    alpha_view: float = 0.05,
    n_permutations: int = 1000,
    n_subsamples: int = 30,
    subsample_fraction: float = 0.5,
    stability_threshold: float = 0.5,
    expected_false: float = 2.0,
    max_collinearity: float | None = 0.92,
    cap_to_sample_capacity: bool = True,
    min_features: int | None = None,
    feature_names: Sequence[Sequence[str]] | None = None,
    view_names: Sequence[str] | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    """Outcome-blind cross-view communality & stability gate for wide views.

    Parameters
    ----------
    views
        Sequence of ``(n, p_m)`` training matrices (at least 2 views). No outcome
        is accepted; all statistics are computed strictly from ``views``.
    k
        Target latent rank of the downstream multi-view fit. Sets the default
        per-view subspace rank ``min(max(2 * k, 4), max(2, n // 5))`` and the
        minimum retained feature floor ``max(k + 2, 4)``.
    max_width_ratio
        Views with ``p_m / n > max_width_ratio`` are treated as wide and
        screened; narrower views retain all columns unchanged unless constant.
        When ``cap_to_sample_capacity=True``, wide views are also bounded to at
        most ``floor(max_width_ratio * n)`` non-redundant features so the gated
        view cannot trigger wide-view noise coupling (`WideViewWarning`).
    subspace_rank
        Leading PCs per other view used to form the cross-view target subspace.
    alpha, alpha_view
        Feature-level permutation cutoff ``p_perm < alpha`` and view-level
        omnibus coupling cutoff on ``min(p_view_mean, p_view_max)``.
    n_permutations
        Row permutations of the other views' subspace basis for ``p_perm``,
        ``p_view_mean`` and ``p_view_max``.
    n_subsamples, subsample_fraction, stability_threshold, expected_false
        Half-sample stability selection settings (Meinshausen & Buhlmann 2010),
        recomputing both the other views' subspaces and ``h_{m,f}`` on each
        subsample of the training rows.
    max_collinearity
        Maximum allowed ``|corr|`` among admitted features within a screened wide
        view (``None`` disables redundancy pruning).
    cap_to_sample_capacity
        If ``True`` (default), cap admitted features in wide views at
        ``max(floor_k, floor(max_width_ratio * n))``. Set ``False`` to retain all
        non-redundant features passing the cross-view statistical gate regardless
        of ``p_out / n``.
    min_features
        Minimum features kept in any view (default ``min(p_m, max(k + 2, 4))``).
    feature_names, view_names
        Optional per-view feature names and modality names.
    random_state
        Seed for row permutations and half-sample draws.

    Returns
    -------
    dict
        ``views`` (gated training views preserving input tensor/array type),
        ``indices`` (per-view 1-D int64 index arrays), ``feature_names`` (gated
        names or ``None``), ``view_names``, ``view_coupled`` (per-view bool),
        ``p_view_mean``, ``p_view_max``, ``communality``, ``p_perm``,
        ``selection_frequency`` and ``summary`` (pandas DataFrame).
    """
    import pandas as pd

    from .api import as_views

    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    if max_width_ratio <= 0.0:
        raise ValueError(f"max_width_ratio must be > 0; got {max_width_ratio}.")
    if not 0.0 < subsample_fraction < 1.0:
        raise ValueError(f"subsample_fraction must be in (0, 1); got {subsample_fraction}.")
    if not 0.5 <= stability_threshold <= 1.0:
        raise ValueError(f"stability_threshold must be in [0.5, 1]; got {stability_threshold}.")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}.")

    raw_items = list(views)
    if len(raw_items) < 2:
        raise ValueError("screen_wide_views requires at least two views.")
    tensors = as_views(raw_items)
    n = int(tensors[0].shape[0])
    m_views = len(tensors)
    vnames = list(view_names) if view_names is not None else [f"view_{i}" for i in range(m_views)]
    if len(vnames) != m_views:
        raise ValueError(f"view_names has {len(vnames)} entries; there are {m_views} views.")
    if feature_names is not None and len(feature_names) != m_views:
        raise ValueError(f"feature_names has {len(feature_names)} entries; there are {m_views} views.")

    r_target = int(subspace_rank) if subspace_rank is not None else min(max(2 * k, 4), max(2, n // 5))
    floor_default = max(k + 2, 4)
    cap_n = max(floor_default, int(np.floor(max_width_ratio * n)))

    zs: list[np.ndarray] = []
    lives: list[np.ndarray] = []
    us: list[np.ndarray] = []
    for t in tensors:
        z, live = _std_unit_norm(t.detach().double().numpy())
        zs.append(z)
        lives.append(live)
        us.append(_leading_u(z[:, live], r_target))

    rng = np.random.default_rng(random_state)
    nsub = max(floor_default, round(subsample_fraction * n))
    sub_rows = [np.sort(rng.choice(n, nsub, replace=False)) for _ in range(max(0, int(n_subsamples)))]

    # Precompute subsample leading subspaces once across all views.
    sub_us: list[list[np.ndarray]] = []
    for rows in sub_rows:
        r_sub = min(r_target, max(2, nsub // 5))
        u_b = []
        for t in tensors:
            zb, lb = _std_unit_norm(t.detach().double().numpy()[rows])
            u_b.append(_leading_u(zb[:, lb], r_sub))
        sub_us.append(u_b)

    indices_out: list[np.ndarray] = []
    gated_views: list[Any] = []
    gated_fnames: list[list[str]] | None = [] if feature_names is not None else None
    comm_list: list[np.ndarray] = []
    pperm_list: list[np.ndarray] = []
    freq_list: list[np.ndarray] = []
    p_view_list: list[float] = []
    p_view_mean_list: list[float] = []
    p_view_max_list: list[float] = []
    coupled_list: list[bool] = []
    summary_rows: list[dict[str, Any]] = []

    # Pass 1: Per-view permutation communality & Westfall-Young omnibus p-values.
    raw_pv_omni: list[float] = []
    for m in range(m_views):
        z = zs[m]
        live = lives[m]
        p = int(z.shape[1])
        b_mat = _compact_other_basis(us, m)
        obs = ((b_mat.T @ z) ** 2).sum(axis=0) if b_mat.shape[1] > 0 else np.zeros(p, dtype=np.float64)
        obs = np.where(live, obs, 0.0)

        ge = np.zeros(p, dtype=np.int64)
        null_mean = np.zeros(n_permutations, dtype=np.float64)
        null_tail = np.zeros(n_permutations, dtype=np.float64)
        null_max = np.zeros(n_permutations, dtype=np.float64)
        n_live = int(live.sum())
        k_tail = max(1, min(25, n_live // 20))
        if b_mat.shape[1] > 0 and n_live > 0:
            z_live = z[:, live]
            for b in range(n_permutations):
                bp = b_mat[rng.permutation(n)]
                nh_live = ((bp.T @ z_live) ** 2).sum(axis=0)
                ge[live] += nh_live >= obs[live]
                null_mean[b] = float(nh_live.mean())
                null_tail[b] = float(np.partition(nh_live, -k_tail)[-k_tail:].mean())
                null_max[b] = float(nh_live.max())
        p_perm = np.where(live, (1.0 + ge) / (n_permutations + 1.0), 1.0)
        obs_live_mean = float(obs[live].mean()) if n_live > 0 else 0.0
        obs_live_tail = float(np.partition(obs[live], -k_tail)[-k_tail:].mean()) if n_live > 0 else 0.0
        obs_live_max = float(obs[live].max()) if n_live > 0 else 0.0
        if n_live > 0 and b_mat.shape[1] > 0:
            pv_mean = float((1.0 + (null_mean >= obs_live_mean).sum()) / (n_permutations + 1.0))
            pv_max = float((1.0 + (null_max >= obs_live_max).sum()) / (n_permutations + 1.0))
            all_mean = np.append(null_mean, obs_live_mean)
            all_tail = np.append(null_tail, obs_live_tail)
            all_max = np.append(null_max, obs_live_max)
            pm_all = (len(all_mean) - np.searchsorted(np.sort(all_mean), all_mean, side="left")) / float(len(all_mean))
            pt_all = (len(all_tail) - np.searchsorted(np.sort(all_tail), all_tail, side="left")) / float(len(all_tail))
            px_all = (len(all_max) - np.searchsorted(np.sort(all_max), all_max, side="left")) / float(len(all_max))
            minp_all = np.minimum(np.minimum(pm_all, pt_all), px_all)
            pv_omni = float((minp_all <= minp_all[-1]).sum() / float(len(minp_all)))
        else:
            pv_mean = pv_max = pv_omni = 1.0

        comm_list.append(obs)
        pperm_list.append(p_perm)
        raw_pv_omni.append(pv_omni)
        p_view_mean_list.append(pv_mean)
        p_view_max_list.append(pv_max)

    # Holm-Bonferroni step-down across views controls multi-view FWER under pure noise with zero penalty on the last view when M-1 views couple.
    order_v = np.argsort(raw_pv_omni)
    holm_pv = np.empty(m_views, dtype=np.float64)
    running_max = 0.0
    for rank_i, vm in enumerate(order_v):
        adj = min(1.0, raw_pv_omni[int(vm)] * float(m_views - rank_i))
        running_max = max(running_max, adj)
        holm_pv[int(vm)] = running_max

    # Pass 2: Subsample stability selection & greedy collinearity-aware gating.
    for m in range(m_views):
        z = zs[m]
        live = lives[m]
        n_live = int(live.sum())
        p = int(z.shape[1])
        was_wide = (p / float(n)) > max_width_ratio
        min_keep = min(p, int(min_features) if min_features is not None else floor_default)
        obs = comm_list[m]
        p_perm = pperm_list[m]
        obs_live_max = float(obs[live].max()) if n_live > 0 else 0.0
        pv_holm = float(holm_pv[m])
        is_coupled = bool(pv_holm < alpha_view)

        freq = np.zeros(p, dtype=np.float64)
        if sub_rows and n_live > 0:
            denom_tau = max(0.2, 2.0 * stability_threshold - 1.0)
            qm_mb = int(np.ceil(np.sqrt(expected_false * denom_tau * p)))
            n_sig = int((p_perm < alpha).sum()) if is_coupled else 0
            qm = min(n_live, max(min_keep, qm_mb, n_sig))
            raw_np = tensors[m].detach().double().numpy()
            for rows, u_b in zip(sub_rows, sub_us):
                zm_b, lm_b = _std_unit_norm(raw_np[rows])
                bm_b = _compact_other_basis(u_b, m)
                if bm_b.shape[1] == 0 or not lm_b.any():
                    continue
                hs = np.where(lm_b, ((bm_b.T @ zm_b) ** 2).sum(axis=0), -1.0)
                top_q = np.argsort(-hs)[:qm]
                freq[top_q] += 1.0 / len(sub_rows)

        if not was_wide:
            idx = np.flatnonzero(live).astype(np.int64)
            if len(idx) == 0:
                idx = np.arange(min(p, max(1, min_keep)), dtype=np.int64)
        else:
            norm_obs = obs / max(obs_live_max, 1e-12)
            score = 0.65 * norm_obs + 0.35 * freq
            live_idx = np.flatnonzero(live)
            ranked_all = live_idx[np.argsort(-score[live_idx])]
            pass_mask = live & ((p_perm < alpha) | (freq >= stability_threshold))
            cand_idx = np.flatnonzero(pass_mask)
            ranked_cand = cand_idx[np.argsort(-score[cand_idx])]
            max_keep = cap_n if cap_to_sample_capacity else n_live
            if not is_coupled:
                idx = _greedy_nonredundant(z, ranked_all, min_keep, max_collinearity)
            else:
                idx = _greedy_nonredundant(z, ranked_cand, max_keep, max_collinearity)
                if len(idx) < min_keep:
                    rem = np.asarray([i for i in ranked_all if i not in set(idx.tolist())], dtype=np.int64)
                    order_aug = np.concatenate([idx, rem])
                    idx = _greedy_nonredundant(z, order_aug, min(max_keep, max(min_keep, len(idx))), max_collinearity)
            if len(idx) == 0 and len(ranked_all) > 0:
                idx = np.sort(ranked_all[:min_keep].astype(np.int64))

        indices_out.append(idx)
        orig = raw_items[m]
        if isinstance(orig, torch.Tensor):
            gated_views.append(orig[:, torch.as_tensor(idx, dtype=torch.long)])
        elif hasattr(orig, "iloc"):
            gated_views.append(orig.iloc[:, idx])
        else:
            gated_views.append(np.asarray(orig)[:, idx])

        if gated_fnames is not None and feature_names is not None:
            fn_m = list(feature_names[m])
            gated_fnames.append([fn_m[int(i)] for i in idx])

        freq_list.append(freq)
        p_view_list.append(pv_holm)
        coupled_list.append(is_coupled)
        summary_rows.append({
            "view": m,
            "view_name": vnames[m],
            "p_in": p,
            "p_out": len(idx),
            "width_ratio_in": float(p / float(n)),
            "width_ratio_out": float(len(idx) / float(n)),
            "was_wide": bool(was_wide),
            "view_coupled": bool(is_coupled),
            "p_view": pv_holm,
            "p_view_mean": p_view_mean_list[m],
            "p_view_max": p_view_max_list[m],
            "n_p_alpha": int((live & (p_perm < alpha)).sum()),
            "n_stable": int((live & (freq >= stability_threshold)).sum()),
        })

    return {
        "views": gated_views,
        "indices": indices_out,
        "feature_names": gated_fnames,
        "view_names": vnames,
        "view_coupled": coupled_list,
        "p_view": p_view_list,
        "p_view_mean": p_view_mean_list,
        "p_view_max": p_view_max_list,
        "communality": comm_list,
        "p_perm": pperm_list,
        "selection_frequency": freq_list,
        "summary": pd.DataFrame(summary_rows),
        "n_samples": n,
        "max_width_ratio": float(max_width_ratio),
        "alpha": float(alpha),
    }


def apply_view_screen(screen: Mapping[str, Any], views: Any) -> Any:
    """Apply a fitted `screen_wide_views` column gate to held-out ``views``.

    Accepts either a sequence aligned with the training views (supporting ``None``
    for missing test-time modalities) or a ``{view_name: matrix}`` mapping (for
    subset prediction via ``pysimlr.predict``).
    """
    indices = screen["indices"]
    vnames = screen["view_names"]

    def _slice_one(mat: Any, idx: np.ndarray) -> Any:
        if mat is None:
            return None
        if isinstance(mat, torch.Tensor):
            return mat[:, torch.as_tensor(idx, dtype=torch.long)]
        if hasattr(mat, "iloc"):
            return mat.iloc[:, idx]
        return np.asarray(mat)[:, idx]

    if isinstance(views, Mapping):
        name_to_idx = {nm: idx for nm, idx in zip(vnames, indices)}
        out_map = {}
        for nm, mat in views.items():
            if nm not in name_to_idx:
                raise KeyError(f"unknown view {nm!r}; known views: {list(vnames)}.")
            out_map[nm] = _slice_one(mat, name_to_idx[nm])
        return out_map

    items = list(views)
    if len(items) != len(indices):
        raise ValueError(f"views has {len(items)} entries; screen has {len(indices)} views.")
    return [_slice_one(mat, idx) for mat, idx in zip(items, indices)]
