r"""Uniform and block-uniform shrinkage priors for directional multi-view `SiMLR`.

When all features within each view share a common directional polarity (for
example after :func:`pysimlr.prepare_views` with ``signs=...`` so higher values
denote greater pathology), the unweighted view mean
:math:`\bar{z}_{im} = p_m^{-1} \sum_{j=1}^{p_m} Z_{imj}` has **zero estimation
variance**, whereas an unregularized multi-view weight vector
:math:`\mathbf{v}_m \in \mathbb{R}_{\ge 0}^{p_m}` has estimation variance scaling
as :math:`O(p_m / n_m)`. Whenever :math:`n_m \lesssim p_m` (or calibrated
out-of-bag ``ICC(1)`` in :func:`pysimlr.latent_stability` is low), unregularized
`SiMLR` overfits within-view noise and can trail naive unweighted averaging out
of fold, whereas naive averaging sacrifices all regional, tract, and network
specificity.

This module bridges both regimes smoothly through two matched primitives:

1. :class:`UniformShrinkageOperator` (and factory :func:`create_uniform_smoother`):
   an :math:`O(p_m k)`-time, :math:`O(1)`-memory matrix-free Green's resolvent
   of the mean-degree-normalized complete-graph (or block-complete) Laplacian
   :math:`\mathbf{L}_m = \mathbf{I}_{p_m} - \mathbf{\Pi}_m`,

   .. math::

      \mathbf{S}_m(\alpha_m)
      = (\mathbf{I}_{p_m} + \lambda_m \mathbf{L}_m)^{-1}
      = (1 - \alpha_m)\mathbf{I}_{p_m} + \alpha_m \mathbf{\Pi}_m,
      \qquad \alpha_m = \frac{\lambda_m}{1 + \lambda_m} \in [0, 1],

   where :math:`\mathbf{\Pi}_m = p_m^{-1}\mathbf{1}_{p_m}\mathbf{1}_{p_m}^\top`
   (or block-wise group means when ``groups`` is supplied). Passed in
   ``smoothing_matrices``, it shrinks each solver iterate toward the uniform
   direction while allowing data-supported cross-view focal drivers to rise
   above the background.

2. :func:`uniform_shrinkage_params` and :func:`project_uniform_shrunk`:
   a parameter builder (mirroring :func:`pysimlr.target_root_fit_params`) and
   simplex-normalized natural-amplitude projector. On the simplex
   (:math:`\mathbf{w}_m \ge 0, \mathbf{1}^\top \mathbf{w}_m = 1`),
   :math:`\mathbf{Z}_m \mathbf{w}_m` preserves each view's natural within-view
   coherence amplitude rather than inflating noisy or mean-cancelled views to
   unit variance, and converges **identically** to the naive unweighted average
   as shrinkage approaches ``1``.

Should this be default ON in ``pysimlr.simlr``?
-----------------------------------------------
**No — it must remain opt-in (`default OFF` in generic `simlr`), and turned ON
specifically for sign-oriented directional single-axis (`k = 1`) studies when
`n` is comparable to or smaller than `p_m`.**

* *Why not default ON globally:* Shrinking toward :math:`+p_m^{-1}\mathbf{1}`
  assumes that ``+1`` means the same thing on every coordinate of a view. On
  raw arbitrary-sign tabular, genomic, or multi-component (:math:`k \ge 2`,
  where higher components must carry mixed signs or disjoint supports to remain
  orthogonal) inputs, uniform shrinkage biases weights toward an arbitrary
  coordinate-sum axis.
* *When to turn it ON:* Whenever views have been sign-oriented to a shared
  polarity (e.g. pathology-oriented neuroimaging views from ``prepare_views``)
  and :math:`n_{\text{train}} \lesssim 2 p_{\max}` (or single-bag out-of-bag
  ``ICC(1) < 0.85``). With ``alpha="auto"``, :func:`auto_uniform_alpha` sets
  :math:`\alpha_m = \operatorname{clip}(p_m / (n_{\text{obs}, m} + p_m), 0.05, 0.85)`,
  shrinking wide small-sample views strongly while automatically vanishing
  (:math:`\alpha_m \to 0`) as :math:`n_{\text{obs}, m} / p_m \to \infty`.

See ``docs/uniform_shrinkage.md`` for empirical benchmarks and equations.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

import numpy as np
import torch


def _fingerprint(arr: np.ndarray) -> tuple[Any, ...]:
    contig = np.ascontiguousarray(arr)
    return (
        contig.shape,
        str(contig.dtype),
        hashlib.blake2b(contig.tobytes(), digest_size=16).digest(),
    )


def _extract_view_list(views: Any) -> list[np.ndarray]:
    if isinstance(views, Mapping):
        raw_items = list(views.values())
    elif isinstance(views, (list, tuple)):
        raw_items = list(views)
    elif hasattr(views, "shape"):
        raw_items = [views]
    else:
        raise TypeError(f"views must be a sequence or mapping of 2-D matrices; got {type(views).__name__}.")
    if not raw_items:
        raise ValueError("views must contain at least one matrix.")

    out: list[np.ndarray] = []
    n_rows: int | None = None
    for idx, item in enumerate(raw_items):
        if torch.is_tensor(item):
            arr = item.detach().cpu().double().numpy()
        elif hasattr(item, "to_numpy"):
            arr = np.asarray(item.to_numpy(dtype=np.float64), dtype=np.float64)
        else:
            arr = np.asarray(item, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"view {idx} must be 2-D (n, p); got shape {arr.shape}.")
        if n_rows is None:
            n_rows = int(arr.shape[0])
        elif int(arr.shape[0]) != n_rows:
            raise ValueError(f"view {idx} has {arr.shape[0]} rows; view 0 has {n_rows}.")
        out.append(arr)
    return out


def auto_uniform_alpha(
    n_obs: int,
    n_features: int,
    *,
    prior_scale: float = 1.0,
    clip: tuple[float, float] = (0.05, 0.85),
) -> float:
    r"""Dimension-adaptive Empirical-Bayes shrinkage weight :math:`\alpha_m \in [0, 1]`.

    Computes

    .. math::

       \alpha_m = \operatorname{clip}\!\left(
           \frac{c \, p_m}{n_{\text{obs}} + c \, p_m},\;
           \alpha_{\min},\; \alpha_{\max}
       \right),

    where :math:`c = \text{prior\_scale} > 0`. At :math:`n_{\text{obs}} = p_m`
    and :math:`c = 1`, :math:`\alpha_m = 0.5`; as :math:`n_{\text{obs}} / p_m \to \infty`,
    :math:`\alpha_m \to \alpha_{\min}`.
    """
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1; got {n_features}.")
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1; got {n_obs}.")
    if prior_scale <= 0.0 or not np.isfinite(prior_scale):
        raise ValueError(f"prior_scale must be finite and > 0; got {prior_scale}.")
    lo, hi = float(clip[0]), float(clip[1])
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"clip bounds must satisfy 0 <= lo <= hi <= 1; got {clip}.")
    num = float(prior_scale) * float(n_features)
    raw = num / (float(n_obs) + num)
    return float(np.clip(raw, lo, hi))


class UniformShrinkageOperator:
    r"""Matrix-free :math:`O(PK)` uniform or block-uniform Green's resolvent smoother.

    Implements the closed-form resolvent of the mean-degree-normalized
    complete-graph Laplacian :math:`\mathbf{L} = \mathbf{I}_P - \mathbf{\Pi}`:

    .. math::

       \mathbf{S}(\alpha)\,\mathbf{V}
       = (\mathbf{I}_P + \lambda \mathbf{L})^{-1}\mathbf{V}
       = (1 - \alpha)\,\tilde{\mathbf{V}} + \alpha\,\mathbf{\Pi}\tilde{\mathbf{V}},
       \qquad \alpha = \frac{\lambda}{1 + \lambda} \in [0, 1],

    where :math:`\mathbf{\Pi}\tilde{\mathbf{V}}` replaces each entry by its
    column mean (or within-group column mean when ``groups`` is provided).

    Like :class:`pysimlr.sparse.SparseGraphResolvent`, ``self @ v`` is
    idempotent on its own image so repeated calls inside ``simlr``'s feasibility
    retraction and post-step projection never apply shrinkage twice in one sweep.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float | None = 0.5,
        *,
        lambda_val: float | None = None,
        groups: Sequence[Any] | np.ndarray | None = None,
        positive_part: bool = True,
    ) -> None:
        p = int(n_features)
        if p < 1:
            raise ValueError(f"n_features must be >= 1; got {n_features}.")
        if lambda_val is not None:
            lam = float(lambda_val)
            if not np.isfinite(lam) or lam < 0.0:
                raise ValueError(f"lambda_val must be finite and >= 0; got {lambda_val}.")
            eff_alpha = lam / (1.0 + lam)
        else:
            if alpha is None:
                eff_alpha = 0.5
            else:
                eff_alpha = float(alpha)
            if not np.isfinite(eff_alpha) or not (0.0 <= eff_alpha <= 1.0):
                raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
            lam = float("inf") if eff_alpha >= 1.0 else eff_alpha / max(1.0 - eff_alpha, 1e-15)

        self.P = p
        self.shape = (p, p)
        self.alpha = float(eff_alpha)
        self.lambda_val = float(lam)
        self.positive_part = bool(positive_part)

        self._group_indices: list[np.ndarray] | None = None
        if groups is not None:
            g_arr = np.asarray(list(groups) if not isinstance(groups, np.ndarray) else groups)
            if g_arr.shape != (p,):
                raise ValueError(f"groups must have length {p}; got shape {g_arr.shape}.")
            buckets: dict[Any, list[int]] = {}
            for idx, g_val in enumerate(g_arr.tolist()):
                if g_val is None or (isinstance(g_val, float) and np.isnan(g_val)) or g_val == -1:
                    continue
                buckets.setdefault(g_val, []).append(idx)
            self._group_indices = [np.asarray(ix, dtype=np.int64) for ix in buckets.values() if len(ix) > 0]

        self._idempotent_outputs: OrderedDict[tuple[Any, ...], bool] = OrderedDict()
        self._cache_maxsize = 256

    def __repr__(self) -> str:
        n_grp = 1 if self._group_indices is None else len(self._group_indices)
        return (
            f"UniformShrinkageOperator(P={self.P}, alpha={self.alpha:.4f}, "
            f"n_groups={n_grp}, positive_part={self.positive_part})"
        )

    def _smooth_2d(self, mat: np.ndarray) -> np.ndarray:
        if self.alpha <= 0.0:
            return mat.copy()
        work = mat.copy()
        if self.positive_part:
            for col in range(work.shape[1]):
                c = work[:, col]
                if np.any(c > 0.0):
                    work[:, col] = np.maximum(c, 0.0)
                elif np.any(c < 0.0):
                    # All non-positive column: flip to positive orthant before uniform averaging
                    work[:, col] = np.maximum(-c, 0.0)

        if self._group_indices is None:
            means = np.mean(work, axis=0, keepdims=True)
            return (1.0 - self.alpha) * work + self.alpha * means

        out = work.copy()
        for idx in self._group_indices:
            g_mean = np.mean(work[idx, :], axis=0, keepdims=True)
            out[idx, :] = (1.0 - self.alpha) * work[idx, :] + self.alpha * g_mean
        return out

    def __matmul__(self, v: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        is_torch = isinstance(v, torch.Tensor)
        v_np = v.detach().cpu().double().numpy() if is_torch else np.asarray(v, dtype=np.float64)
        if v_np.ndim not in (1, 2) or v_np.shape[0] != self.P:
            raise ValueError(
                f"Dimension mismatch: UniformShrinkageOperator expects ({self.P},) or ({self.P}, K), "
                f"got shape {v_np.shape}."
            )

        orig_1d = v_np.ndim == 1
        in_raw = v.detach().cpu().numpy() if is_torch else np.asarray(v)
        if _fingerprint(in_raw) in self._idempotent_outputs:
            return v

        mat_2d = v_np[:, None] if orig_1d else v_np
        smoothed = self._smooth_2d(mat_2d)
        out_np = smoothed[:, 0] if orig_1d else smoothed

        if is_torch:
            res_t = torch.as_tensor(out_np, dtype=v.dtype, device=v.device)
            fp_out = _fingerprint(res_t.detach().cpu().numpy())
            self._idempotent_outputs[fp_out] = True
            if len(self._idempotent_outputs) > self._cache_maxsize:
                self._idempotent_outputs.popitem(last=False)
            return res_t

        cast_out = out_np.astype(in_raw.dtype if np.issubdtype(in_raw.dtype, np.floating) else np.float64, copy=False)
        self._idempotent_outputs[_fingerprint(cast_out)] = True
        if len(self._idempotent_outputs) > self._cache_maxsize:
            self._idempotent_outputs.popitem(last=False)
        return cast_out


def create_uniform_smoother(
    source: int | np.ndarray | torch.Tensor,
    alpha: float | str = "auto",
    *,
    lambda_val: float | None = None,
    prior_scale: float = 1.0,
    clip: tuple[float, float] = (0.05, 0.85),
    groups: Sequence[Any] | np.ndarray | None = None,
    positive_part: bool = True,
) -> UniformShrinkageOperator:
    """Construct a :class:`UniformShrinkageOperator` from a feature count ``P`` or view matrix ``(n, P)``."""
    if isinstance(source, (int, np.integer)) and not isinstance(source, bool):
        p = int(source)
        if lambda_val is not None:
            return UniformShrinkageOperator(p, lambda_val=lambda_val, groups=groups, positive_part=positive_part)
        if isinstance(alpha, str):
            raise ValueError(
                "alpha='auto' requires a 2-D view matrix (n, P) so observed sample size n_obs is known; "
                "pass a numeric alpha in [0, 1] when source is an integer feature count."
            )
        return UniformShrinkageOperator(p, alpha=float(alpha), groups=groups, positive_part=positive_part)

    arr = source.detach().cpu().double().numpy() if torch.is_tensor(source) else np.asarray(source, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected integer feature count P or 2-D view matrix (n, P); got shape {arr.shape}.")
    n_obs = max(1, int(np.isfinite(arr).all(axis=1).sum()))
    p = int(arr.shape[1])
    if lambda_val is not None:
        return UniformShrinkageOperator(p, lambda_val=lambda_val, groups=groups, positive_part=positive_part)
    if isinstance(alpha, str):
        if alpha != "auto":
            raise ValueError(f"alpha must be 'auto' or a float in [0, 1]; got {alpha!r}.")
        eff_a = auto_uniform_alpha(n_obs, p, prior_scale=prior_scale, clip=clip)
    else:
        eff_a = float(alpha)
    return UniformShrinkageOperator(p, alpha=eff_a, groups=groups, positive_part=positive_part)


def uniform_shrinkage_params(
    views: Any,
    *,
    k: int = 1,
    alpha: float | str | Sequence[float | str | None] = "auto",
    prior_scale: float = 1.0,
    clip: tuple[float, float] = (0.05, 0.85),
    positivity: str = "conical",
    sparseness: float | Sequence[float | None] | None = 0.3,
    sparseness_type: str = "soft",
    nsa_w: float | None = None,
    energy_type: str = "recon_r2",
    mixing_algorithm: str = "svd",
    scale_list: Sequence[str] = ("none",),
    missing_rows: str = "mask",
    groups: Sequence[Sequence[Any] | np.ndarray | None] | None = None,
    positive_part: bool = True,
) -> dict[str, Any]:
    """Build keyword arguments for uniform-shrunk directional `pysimlr.fit("simlr", ...)`.

    Designed for sign-oriented multi-view datasets (e.g. views prepared with
    :func:`pysimlr.prepare_views` so higher values share a consistent directional
    meaning) when :math:`k = 1` and sample size :math:`n` is comparable to or
    smaller than view width :math:`p_m`.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    if positivity not in ("conical", "convex"):
        raise ValueError(
            f"uniform_shrinkage_params requires non-negative positivity ('conical' or 'convex'); got {positivity!r}."
        )
    mats = _extract_view_list(views)
    m_cnt = len(mats)

    if isinstance(alpha, (list, tuple)):
        if len(alpha) != m_cnt:
            raise ValueError(f"alpha sequence has {len(alpha)} entries for {m_cnt} views.")
        alpha_list = list(alpha)
    else:
        alpha_list = [alpha] * m_cnt

    if groups is not None:
        if len(groups) != m_cnt:
            raise ValueError(f"groups sequence has {len(groups)} entries for {m_cnt} views.")
        grp_list = list(groups)
    else:
        grp_list = [None] * m_cnt

    smoothers: list[UniformShrinkageOperator | None] = []
    init_v: list[np.ndarray] = []
    for idx, mat in enumerate(mats):
        p_m = int(mat.shape[1])
        a_spec = alpha_list[idx]
        if a_spec is None:
            smoothers.append(None)
        else:
            smoothers.append(
                create_uniform_smoother(
                    mat,
                    alpha=a_spec,
                    prior_scale=prior_scale,
                    clip=clip,
                    groups=grp_list[idx],
                    positive_part=positive_part,
                )
            )
        scale = (1.0 / float(p_m)) if positivity == "convex" else (1.0 / np.sqrt(float(p_m)))
        if int(k) == 1:
            v0 = np.full((p_m, 1), scale, dtype=np.float64)
        else:
            # Column 0 uniform; columns 1..k-1 small deterministic sinusoidal perturbations kept non-negative
            v0 = np.full((p_m, int(k)), scale, dtype=np.float64)
            grid = np.linspace(0.0, np.pi, p_m, dtype=np.float64)
            for c_idx in range(1, int(k)):
                wave = 0.5 * (1.0 + np.cos(float(c_idx) * grid))
                col = np.maximum(wave, 1e-3)
                v0[:, c_idx] = col / (np.sum(col) if positivity == "convex" else np.linalg.norm(col))
        init_v.append(v0)

    has_sparse = sparseness is not None and (
        isinstance(sparseness, (list, tuple)) or float(sparseness) > 0.0
    )
    eff_nsa_w = (0.20 if has_sparse else 0.0) if nsa_w is None else float(nsa_w)
    eff_sp_type = sparseness_type if has_sparse else "hard"

    return {
        "scale_list": list(scale_list),
        "energy_type": energy_type,
        "mixing_algorithm": mixing_algorithm,
        "positivity": positivity,
        "nsa_w": eff_nsa_w,
        "sparseness": sparseness,
        "sparseness_type": eff_sp_type,
        "smoothing_matrices": smoothers,
        "initial_v": init_v,
        "missing_rows": missing_rows,
    }


def project_uniform_shrunk(
    result: Mapping[str, Any],
    views: Any,
    *,
    post_shrink: float | str = "auto",
    view_scaling: str = "natural",
    train_views: Any = None,
) -> dict[str, Any]:
    r"""Project ``views`` onto a fitted ``simlr`` result on the simplex / natural-amplitude scale.

    For each view :math:`m` and component :math:`c`, the fitted encoder column
    :math:`\hat{\mathbf{v}}_{m,c}` is mapped onto the probability simplex
    :math:`\hat{\mathbf{w}}_{m,c} = \max(\hat{\mathbf{v}}_{m,c}, 0) / \mathbf{1}^\top \max(\hat{\mathbf{v}}_{m,c}, 0)`
    and optionally blended with the exact uniform weight :math:`p_m^{-1}\mathbf{1}_{p_m}`:

    .. math::

       \mathbf{w}_{m,c}(\gamma) = (1 - \gamma)\,\hat{\mathbf{w}}_{m,c} + \gamma\,\frac{1}{p_m}\mathbf{1}_{p_m},
       \qquad \gamma \in [0, 1].

    When ``post_shrink="auto"`` (default), sub-critical wide-view damping sets
    :math:`\gamma = \operatorname{clip}(1 - n_{\text{tr}} / (1.5\, p_{\max}), 0.0, 0.60)`:
    **identically zero** whenever :math:`n_{\text{tr}} \ge 1.5\, p_{\max}` (leaving
    well-determined fits untouched), and increasing smoothly when
    :math:`n_{\text{tr}} < 1.5\, p_{\max}` to protect out-of-fold predictions
    from :math:`p > n` rank deficiency.

    When ``view_scaling="natural"`` (default), each view's projection is
    :math:`\mathbf{Z}_m \mathbf{w}_{m,c}(\gamma)` without post-hoc per-view unit-SD
    inflation, so views with stronger within-view coherence keep their natural
    amplitude and :math:`\gamma = 1` (or uniform :math:`\hat{\mathbf{v}}_m`)
    reproduces the exact unweighted multi-view plain mean.
    """
    res_core = result["result"] if ("result" in result and "v" not in result) else result
    if "v" not in res_core:
        raise KeyError("result must contain encoder basis list 'v'.")
    if view_scaling not in ("natural", "unit_variance"):
        raise ValueError(f"view_scaling must be 'natural' or 'unit_variance'; got {view_scaling!r}.")

    eval_mats = _extract_view_list(views)
    tr_mats = _extract_view_list(train_views) if train_views is not None else eval_mats
    v_list = res_core["v"]
    if len(v_list) != len(eval_mats) or len(tr_mats) != len(eval_mats):
        raise ValueError(
            f"View count mismatch: result has {len(v_list)} views, views has {len(eval_mats)}, "
            f"train_views has {len(tr_mats)}."
        )

    if isinstance(post_shrink, str):
        if post_shrink != "auto":
            raise ValueError(f"post_shrink must be 'auto' or a float in [0, 1]; got {post_shrink!r}.")
        n_tr_eff = max(int(np.isfinite(m).all(axis=1).sum()) for m in tr_mats)
        p_max = max(int(m.shape[1]) for m in tr_mats)
        gamma = float(np.clip(1.0 - float(n_tr_eff) / (1.5 * float(p_max)), 0.0, 0.60))
    else:
        gamma = float(post_shrink)
        if not np.isfinite(gamma) or not (0.0 <= gamma <= 1.0):
            raise ValueError(f"post_shrink must be in [0, 1]; got {post_shrink}.")

    n_eval = eval_mats[0].shape[0]
    m_cnt = len(eval_mats)
    first_v = v_list[0].detach().cpu().double().numpy() if torch.is_tensor(v_list[0]) else np.asarray(v_list[0], dtype=np.float64)
    k = int(first_v.shape[1])

    simplex_weights: list[np.ndarray] = []
    view_scores = np.full((n_eval, m_cnt, k), np.nan, dtype=np.float64)

    for m_i, (z_ev, z_tr, raw_v) in enumerate(zip(eval_mats, tr_mats, v_list)):
        v_arr = raw_v.detach().cpu().double().numpy() if torch.is_tensor(raw_v) else np.asarray(raw_v, dtype=np.float64)
        p_m = int(z_ev.shape[1])
        if v_arr.shape != (p_m, k):
            raise ValueError(f"view {m_i} basis has shape {v_arr.shape}, expected ({p_m}, {k}).")

        w_mat = np.zeros((p_m, k), dtype=np.float64)
        u_unif = np.full(p_m, 1.0 / float(p_m), dtype=np.float64)
        for c_i in range(k):
            col = v_arr[:, c_i]
            pos = np.maximum(col, 0.0) if float(np.sum(col)) >= 0.0 else np.maximum(-col, 0.0)
            s = float(np.sum(pos))
            w_fit = (pos / s) if s > 1e-12 else u_unif
            w_mat[:, c_i] = (1.0 - gamma) * w_fit + gamma * u_unif
        simplex_weights.append(w_mat)

        ok_ev = np.isfinite(z_ev).all(axis=1)
        ok_tr = np.isfinite(z_tr).all(axis=1)
        s_ev = np.full((n_eval, k), np.nan, dtype=np.float64)
        if np.any(ok_ev):
            s_ev[ok_ev] = z_ev[ok_ev] @ w_mat
        if view_scaling == "unit_variance" and np.any(ok_tr):
            s_tr = z_tr[ok_tr] @ w_mat
            mu = np.mean(s_tr, axis=0, keepdims=True)
            sd = np.std(s_tr, axis=0, keepdims=True)
            sd = np.where(sd > 1e-12, sd, 1.0)
            s_ev = (s_ev - mu) / sd
        view_scores[:, m_i, :] = s_ev

    u_cons = np.nanmean(view_scores, axis=1)
    alphas = []
    sm_list = res_core.get("parameters", {}).get("smoothing_matrices") if isinstance(res_core, Mapping) else None
    if isinstance(sm_list, (list, tuple)):
        for op in sm_list:
            alphas.append(float(getattr(op, "alpha", np.nan)) if op is not None else 0.0)

    return {
        "u": u_cons,
        "view_scores": view_scores[:, :, 0] if k == 1 else view_scores,
        "weights": simplex_weights,
        "post_shrink": gamma,
        "view_scaling": view_scaling,
        "alphas": alphas,
    }


__all__ = [
    "UniformShrinkageOperator",
    "auto_uniform_alpha",
    "create_uniform_smoother",
    "project_uniform_shrunk",
    "uniform_shrinkage_params",
]
