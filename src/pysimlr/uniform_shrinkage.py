r"""Uniform and support-restricted shrinkage priors for directional multi-view `SiMLR`.

When all features within each view share a common directional polarity (the
caller orients the signs, for example with :func:`pysimlr.prepare_views`), the
unweighted view mean has zero estimation variance, whereas an unregularized
non-negative weight vector :math:`\mathbf{v}_m` has estimation variance of order
:math:`p_m / n_m`. When :math:`n_m \lesssim p_m` the unregularized fit can
trail the plain mean out of sample, while the plain mean gives up feature-level
specificity.

Primitives
----------
1. :class:`UniformShrinkageOperator` (factory :func:`create_uniform_smoother`):
   stateless, matrix-free :math:`O(p_m k)` operator

   .. math::

      S_m(\alpha) = (1 - \alpha) I + \alpha \Pi_m, \qquad
      \alpha = \lambda / (1 + \lambda),

   with :math:`\Pi_m` the (group-wise) mean projector.
2. :class:`ColumnSupportShrinkageOperator` + :func:`extract_disjoint_supports` +
   :func:`effective_support_dof`: column-wise support-restricted shrinkage
   applying :math:`(1 - \alpha_c)(v_c)_{S_c} + \alpha_c \bar{v}_{S_c}\mathbf{1}_{S_c}`
   strictly on :math:`S_c` and zeroing outside :math:`S_c`. When
   :math:`S_1, \dots, S_k` are pairwise disjoint, this commutes identically with
   conical orthogonality (:math:`V^\top V = I_k`) and disjoint radial simplex
   faces for every :math:`\alpha_c \in [0, 1]`.
3. :func:`resolve_directional_defaults`, :func:`uniform_shrinkage_params`,
   :func:`fit_uniform_shrunk` (single-stage dense for :math:`k=1, q=0`;
   two-stage support-restricted RIC shrinkage for :math:`k > 1` or :math:`q > 0`),
   and :func:`project_uniform_shrunk` (support-restricted simplex projection
   preserving exact zeros outside :math:`S_{m,c}`).

Scope: sign-oriented directional views. Orientation is a user contract; nothing
checks it (a view with flipped signs yields a negated score). It stays opt-in.
See ``docs/uniform_shrinkage.md``.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


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


def _n_observed_rows(arr: np.ndarray) -> int:
    """Rows of a view with at least one finite feature (the rows ``missing_rows='mask'`` keeps)."""
    return max(1, int(np.isfinite(arr).any(axis=1).sum()))


def effective_support_dof(n_active: int, n_features: int, *, mode: str = "ric") -> float:
    r"""Effective degrees of freedom for a support of size ``n_active`` selected from ``n_features``.

    - ``"ric"`` (Donoho–Johnstone / Birgé–Massart :math:`\ell_0` selection complexity):
      :math:`d_{\text{eff}}(m, p) = m\bigl(1 + 2\log(p / m)\bigr)` (equals :math:`p` identically
      when :math:`m = p`).
    - ``"plugin"``: :math:`d_{\text{eff}}(m, p) = m`.
    """
    m = int(n_active)
    p = int(n_features)
    if p < 1:
        raise ValueError(f"n_features must be >= 1; got {n_features}.")
    if not (1 <= m <= p):
        raise ValueError(f"n_active must satisfy 1 <= n_active <= n_features ({p}); got {n_active}.")
    m_key = str(mode).lower().strip()
    if m_key == "ric":
        return float(m) * (1.0 + 2.0 * math.log(float(p) / float(m)))
    if m_key == "plugin":
        return float(m)
    raise ValueError(f"mode must be 'ric' or 'plugin'; got {mode!r}.")


def resolve_directional_defaults(
    k: int,
    *,
    positivity: str = "auto",
    energy_type: str = "auto",
    mixing_algorithm: str = "auto",
) -> tuple[str, str, str]:
    """Resolve automatic directional shrinkage defaults for rank ``k``.

    - ``positivity="auto"`` -> ``"conical"`` for ``k == 1``, ``"convex"`` for ``k > 1``.
    - ``energy_type="auto"`` -> ``"nc"`` for ``k == 1``, ``"acc"`` for ``k > 1``.
    - ``mixing_algorithm="auto"`` -> ``"pca"``.
    """
    k_i = int(k)
    if k_i < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    resolved_pos = ("conical" if k_i == 1 else "convex") if str(positivity).lower() == "auto" else str(positivity)
    resolved_energy = ("nc" if k_i == 1 else "acc") if str(energy_type).lower() == "auto" else str(energy_type)
    resolved_mix = "pca" if str(mixing_algorithm).lower() == "auto" else str(mixing_algorithm)
    return resolved_pos, resolved_energy, resolved_mix


def auto_uniform_alpha(
    n_obs: int,
    n_features: int,
    *,
    prior_scale: float = 1.0,
    clip: tuple[float, float] = (0.05, 0.85),
    n_active: int | None = None,
    alpha_mode: str = "ric",
) -> float:
    r"""Dimension-adaptive Empirical-Bayes shrinkage weight :math:`\alpha_m \in [0, 1]`.

    Computes

    .. math::

       \alpha_m = \operatorname{clip}\!\left(
           \frac{c \, d_{\text{eff}}}{n_{\text{obs}} + c \, d_{\text{eff}}},\;
           \alpha_{\min},\; \alpha_{\max}
       \right),

    where :math:`d_{\text{eff}} = p_m` when ``n_active is None``, or
    :func:`effective_support_dof(n_active, n_features, mode=alpha_mode)` when
    ``n_active`` is supplied.
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
    d_eff = (
        float(n_features)
        if n_active is None
        else effective_support_dof(int(n_active), int(n_features), mode=alpha_mode)
    )
    num = float(prior_scale) * d_eff
    raw = num / (float(n_obs) + num)
    return float(np.clip(raw, lo, hi))


class UniformShrinkageOperator:
    r"""Stateless matrix-free :math:`O(PK)` uniform or block-uniform shrinkage step.

    For each column ``c`` of the input, with ``c+ = max(c, 0)`` (or ``max(-c, 0)``
    when the column has no positive entry; only if ``positive_part=True``),

    .. math::

       S(\alpha)\,c = (1 - \alpha)\,c_+ + \alpha\,\operatorname{mean}(c_+),

    with the mean taken within each group when ``groups`` is given (entries with
    group label ``-1``/``None``/NaN are left unchanged). With
    ``positive_part=False`` this is exactly the dense resolvent
    :math:`(I + \lambda L)^{-1}`, :math:`L = I - \Pi`, :math:`\lambda =
    \alpha / (1 - \alpha)`.

    The operator is applied once per call; it keeps no state, and
    :math:`S(Sv) = (1-\alpha)^2 v + (1 - (1-\alpha)^2)\,\Pi v` (``positive_part=False``).
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
        mat_2d = v_np[:, None] if orig_1d else v_np
        smoothed = self._smooth_2d(mat_2d)
        out_np = smoothed[:, 0] if orig_1d else smoothed
        if is_torch:
            return torch.as_tensor(out_np, dtype=v.dtype, device=v.device)
        in_dtype = np.asarray(v).dtype
        return out_np.astype(in_dtype if np.issubdtype(in_dtype, np.floating) else np.float64, copy=False)


class ColumnSupportShrinkageOperator:
    r"""Stateless matrix-free :math:`O(PK)` column-wise support-restricted uniform shrinkage operator.

    For each column ``c`` with active index set :math:`S_c \subseteq \{0, \dots, P-1\}`
    and shrinkage weight :math:`\alpha_c \in [0, 1]`, ``op @ V``:

    1. Zeroes column ``c`` strictly outside :math:`S_c`.
    2. When ``positive_part=True``, clamps column ``c`` on :math:`S_c` to
       :math:`\mathbb{R}_+^{S_c}` (reflecting all-non-positive slices or falling
       back to uniform on :math:`S_c` when identically zero).
    3. Applies :math:`(1 - \alpha_c)(v_c)_{S_c} + \alpha_c \bar{v}_{S_c}\mathbf{1}_{S_c}`.

    Whenever :math:`S_1, \dots, S_k` are pairwise disjoint,
    :math:`(S_{S_c} v_c)^\top (S_{S_{c'}} v_{c'}) = 0` identically for all
    :math:`c \ne c'` and every :math:`\alpha_c \in [0, 1]`.
    """

    def __init__(
        self,
        n_features: int,
        supports: Sequence[Sequence[int] | np.ndarray],
        alphas: Sequence[float] | float,
        *,
        positive_part: bool = True,
    ) -> None:
        p = int(n_features)
        if p < 1:
            raise ValueError(f"n_features must be >= 1; got {n_features}.")
        if not supports:
            raise ValueError("supports must contain at least one column support.")
        k = len(supports)

        clean_supports: list[np.ndarray] = []
        for c, raw_idx in enumerate(supports):
            raw_arr = np.asarray(raw_idx)
            if raw_arr.dtype == np.bool_:
                if raw_arr.shape != (p,):
                    raise ValueError(f"boolean support for column {c} must have shape ({p},); got {raw_arr.shape}.")
                idx_arr = np.flatnonzero(raw_arr).astype(np.int64)
            else:
                idx_arr = np.unique(raw_arr.astype(np.int64).reshape(-1))
            if idx_arr.size == 0:
                raise ValueError(f"support for column {c} is empty.")
            if int(idx_arr.min()) < 0 or int(idx_arr.max()) >= p:
                raise ValueError(f"support for column {c} has out-of-range indices for n_features={p}.")
            clean_supports.append(idx_arr)

        if isinstance(alphas, (int, float, np.floating, np.integer)) and not isinstance(alphas, bool):
            clean_alphas = [float(alphas)] * k
        else:
            clean_alphas = [float(a) for a in alphas]  # type: ignore[union-attr]
            if len(clean_alphas) != k:
                raise ValueError(f"alphas has {len(clean_alphas)} entries for k={k} columns.")
        for c, a in enumerate(clean_alphas):
            if not np.isfinite(a) or not (0.0 <= a <= 1.0):
                raise ValueError(f"alphas[{c}] must be in [0, 1]; got {a}.")

        self.P = p
        self.shape = (p, p)
        self.k = k
        self.supports = clean_supports
        self.alphas = clean_alphas
        self.alpha = float(np.mean(clean_alphas))
        self.positive_part = bool(positive_part)

    def is_disjoint(self) -> bool:
        """Return ``True`` iff all column supports are pairwise disjoint."""
        seen: set[int] = set()
        for s in self.supports:
            for idx in s.tolist():
                if idx in seen:
                    return False
                seen.add(idx)
        return True

    def __repr__(self) -> str:
        sizes = [int(s.size) for s in self.supports]
        return (
            f"ColumnSupportShrinkageOperator(P={self.P}, k={self.k}, "
            f"support_sizes={sizes}, alphas={[round(a, 4) for a in self.alphas]})"
        )

    def _apply_2d(self, mat: np.ndarray) -> np.ndarray:
        if mat.shape[1] != self.k:
            raise ValueError(
                f"Dimension mismatch: ColumnSupportShrinkageOperator expects k={self.k} columns, "
                f"got shape {mat.shape}."
            )
        out = np.zeros_like(mat, dtype=np.float64)
        for c in range(self.k):
            idx = self.supports[c]
            a = self.alphas[c]
            sub = mat[idx, c].astype(np.float64, copy=True)
            if self.positive_part:
                if np.any(sub > 0.0):
                    sub = np.maximum(sub, 0.0)
                elif np.any(sub < 0.0):
                    sub = np.maximum(-sub, 0.0)
                else:
                    sub = np.full(idx.size, 1.0 / float(idx.size), dtype=np.float64)
            mean_s = float(np.mean(sub))
            out[idx, c] = (1.0 - a) * sub + a * mean_s
        return out

    def __matmul__(self, v: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        is_torch = isinstance(v, torch.Tensor)
        v_np = v.detach().cpu().double().numpy() if is_torch else np.asarray(v, dtype=np.float64)
        if v_np.ndim not in (1, 2) or v_np.shape[0] != self.P:
            raise ValueError(
                f"Dimension mismatch: ColumnSupportShrinkageOperator expects ({self.P},) or "
                f"({self.P}, {self.k}), got shape {v_np.shape}."
            )
        orig_1d = v_np.ndim == 1
        mat_2d = v_np[:, None] if orig_1d else v_np
        smoothed = self._apply_2d(mat_2d)
        out_np = smoothed[:, 0] if orig_1d else smoothed
        if is_torch:
            return torch.as_tensor(out_np, dtype=v.dtype, device=v.device)
        in_dtype = np.asarray(v).dtype
        return out_np.astype(in_dtype if np.issubdtype(in_dtype, np.floating) else np.float64, copy=False)


def extract_disjoint_supports(
    v: np.ndarray | torch.Tensor,
    sparseness: float | None = None,
    *,
    min_per_col: int = 2,
) -> list[np.ndarray]:
    """Extract per-column active supports ``[S_1, ..., S_k]`` (pairwise disjoint for ``k >= 2``).

    - For ``k == 1``: retains top ``max(1, round((1 - q) * p))`` features of ``(v_1)_+``
      (all ``p`` features when ``sparseness is None`` or ``0.0``).
    - For ``k >= 2``: assigns each feature ``i`` to its dominant positive lobe
      ``argmax_c (v_{i,c})_+``, guarantees at least ``min_per_col`` features per column
      while maintaining strict pairwise disjointness, and retains the top
      ``max(min_per_col, round((1 - q) * |B_c|))`` features within each lobe ``B_c``
      (``sparseness=None`` or ``0.0`` yields a full disjoint ``k``-lobe partition).
    """
    arr = v.detach().cpu().double().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"v must be 1-D or 2-D; got shape {arr.shape}.")
    p, k = int(arr.shape[0]), int(arr.shape[1])
    if p < 1 or k < 1:
        raise ValueError(f"v must be non-empty; got shape {arr.shape}.")
    q = 0.0 if sparseness is None else float(sparseness)
    if not np.isfinite(q) or not (0.0 <= q < 1.0):
        raise ValueError(f"sparseness must be None or in [0, 1); got {sparseness}.")

    min_s = 1 if k == 1 else max(1, int(min_per_col))
    if k * min_s > p:
        if k > p:
            raise ValueError(f"Cannot build k={k} disjoint non-empty supports from p={p} features.")
        min_s = 1

    v_pos = np.zeros_like(arr, dtype=np.float64)
    for c in range(k):
        col = arr[:, c]
        if np.any(col > 0.0):
            v_pos[:, c] = np.maximum(col, 0.0)
        elif np.any(col < 0.0):
            v_pos[:, c] = np.maximum(-col, 0.0)
        else:
            v_pos[:, c] = 1.0 / float(p)

    if k == 1:
        m_q = max(1, min(p, round((1.0 - q) * p)))
        order = np.argsort(-v_pos[:, 0], kind="mergesort")
        return [np.sort(order[:m_q]).astype(np.int64)]

    owner = np.argmax(v_pos, axis=1)
    buckets: list[list[int]] = [np.flatnonzero(owner == c).tolist() for c in range(k)]

    for c in range(k):
        while len(buckets[c]) < min_s:
            donor = max(range(k), key=lambda d: len(buckets[d]))
            if len(buckets[donor]) <= min_s:
                raise RuntimeError(f"Unable to allocate disjoint min_per_col={min_s} for p={p}, k={k}.")
            best_feat = max(buckets[donor], key=lambda i: (v_pos[i, c], -v_pos[i, donor], -i))
            buckets[donor].remove(best_feat)
            buckets[c].append(best_feat)

    supports: list[np.ndarray] = []
    for c in range(k):
        b = buckets[c]
        m_c = max(min_s, min(len(b), round((1.0 - q) * len(b))))
        b_sorted = sorted(b, key=lambda i: (-v_pos[i, c], i))
        supports.append(np.sort(np.asarray(b_sorted[:m_c], dtype=np.int64)))
    return supports


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
    n_obs = _n_observed_rows(arr)
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


class UniformShrinkageParams(dict):
    """Keyword arguments for ``pysimlr.fit("simlr", ...)`` plus a ``provenance`` attribute.

    Only the dict items are passed on by ``**params``; ``provenance`` records the
    resolved values (per-view ``n_obs``, ``p``, ``alpha``, ``nsa_w``, ...).
    """

    provenance: dict[str, Any]


MAX_ALPHA_MULTI_COMPONENT = 0.15
"""Largest per-view global alpha allowed for ``k > 1`` without disjoint ``supports=`` (only with ``groups=``)."""


def uniform_shrinkage_params(
    views: Any,
    *,
    k: int = 1,
    alpha: float | str | Sequence[float | str | None] = "auto",
    alpha_mode: str = "ric",
    prior_scale: float = 1.0,
    clip: tuple[float, float] = (0.05, 0.85),
    positivity: str = "auto",
    sparseness: float | Sequence[float | None] | None = None,
    sparseness_type: str = "hard",
    nsa_w: float | str = "auto",
    energy_type: str = "auto",
    mixing_algorithm: str = "auto",
    scale_list: Sequence[str] = ("none",),
    missing_rows: str = "mask",
    groups: Sequence[Sequence[Any] | np.ndarray | None] | None = None,
    supports: Sequence[Sequence[Sequence[int] | np.ndarray] | None] | None = None,
    positive_part: bool = True,
) -> UniformShrinkageParams:
    """Build keyword arguments for uniform- or support-restricted directional `pysimlr.fit("simlr", ...)`.

    Defaults (echoed in ``result.provenance``):

    * ``positivity="auto"`` -> ``"conical"`` for ``k == 1`` (unit-L2 cone), ``"convex"`` for
      ``k > 1`` (radial simplex normalization ``v_+ / 1^T v_+``),
      ``sparseness=None``, ``sparseness_type="hard"``, ``alpha_mode="ric"``.
    * ``energy_type="auto"`` -> ``"nc"`` for ``k == 1``, ``"acc"`` for ``k > 1``;
      ``mixing_algorithm="auto"`` -> ``"pca"``.
    * ``nsa_w="auto"``: ``0.0`` when ``supports`` is given or ``sparseness is None``,
      else ``0.20``.

    When ``supports`` is passed (one length-``k`` list of index arrays per view),
    builds :class:`ColumnSupportShrinkageOperator` smoothers using
    :func:`effective_support_dof` (``alpha_mode="ric"`` or ``"plugin"``) and permits
    ``k > 1`` with any ``alpha in [0, 1]`` whenever each view's supports are pairwise
    disjoint. Without ``supports`` or ``groups``, ``k > 1`` raises ``ValueError`` and
    directs the caller to :func:`fit_uniform_shrunk`.
    """
    k_i = int(k)
    if k_i < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    resolved_pos, resolved_energy, resolved_mix = resolve_directional_defaults(
        k_i, positivity=positivity, energy_type=energy_type, mixing_algorithm=mixing_algorithm
    )
    if resolved_pos not in ("conical", "convex"):
        raise ValueError(
            f"uniform_shrinkage_params requires non-negative positivity ('conical', 'convex', or 'auto'); got {positivity!r}."
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

    if supports is not None:
        if len(supports) != m_cnt:
            raise ValueError(f"supports sequence has {len(supports)} entries for {m_cnt} views.")
        supp_list = list(supports)
    else:
        supp_list = [None] * m_cnt

    smoothers: list[UniformShrinkageOperator | ColumnSupportShrinkageOperator | None] = []
    init_v: list[np.ndarray] = []
    deff_prov: list[list[float]] = []

    for idx, mat in enumerate(mats):
        p_m = int(mat.shape[1])
        n_m = _n_observed_rows(mat)
        a_spec = alpha_list[idx]
        s_spec = supp_list[idx]

        if s_spec is not None:
            if len(s_spec) != k_i:
                raise ValueError(f"supports[{idx}] has {len(s_spec)} columns; expected k={k_i}.")
            clean_s = [np.unique(np.asarray(col_s, dtype=np.int64).reshape(-1)) for col_s in s_spec]
            deff_m = [effective_support_dof(int(cs.size), p_m, mode=alpha_mode) for cs in clean_s]
            deff_prov.append(deff_m)
            if a_spec is None:
                smoothers.append(None)
            else:
                if isinstance(a_spec, str):
                    if a_spec != "auto":
                        raise ValueError(f"alpha must be 'auto' or a float in [0, 1]; got {a_spec!r}.")
                    col_alphas = [
                        auto_uniform_alpha(
                            n_m, p_m, prior_scale=prior_scale, clip=clip,
                            n_active=int(cs.size), alpha_mode=alpha_mode,
                        )
                        for cs in clean_s
                    ]
                elif isinstance(a_spec, (list, tuple)):
                    col_alphas = [float(x) for x in a_spec]
                else:
                    col_alphas = [float(a_spec)] * k_i
                op_col = ColumnSupportShrinkageOperator(
                    p_m, supports=clean_s, alphas=col_alphas, positive_part=positive_part
                )
                if k_i > 1 and not op_col.is_disjoint():
                    raise ValueError(
                        f"supports[{idx}] for k={k_i} > 1 must be pairwise disjoint to prevent component collapse."
                    )
                smoothers.append(op_col)

            v0 = np.zeros((p_m, k_i), dtype=np.float64)
            for c_idx, cs in enumerate(clean_s):
                col_scale = (1.0 / float(cs.size)) if resolved_pos == "convex" else (1.0 / np.sqrt(float(cs.size)))
                v0[cs, c_idx] = col_scale
            init_v.append(v0)
        else:
            deff_prov.append([float(p_m)] * k_i)
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
            scale = (1.0 / float(p_m)) if resolved_pos == "convex" else (1.0 / np.sqrt(float(p_m)))
            if k_i == 1:
                v0 = np.full((p_m, 1), scale, dtype=np.float64)
            else:
                v0 = np.full((p_m, k_i), scale, dtype=np.float64)
                grid = np.linspace(0.0, np.pi, p_m, dtype=np.float64)
                for c_idx in range(1, k_i):
                    wave = 0.5 * (1.0 + np.cos(float(c_idx) * grid))
                    col = np.maximum(wave, 1e-3)
                    v0[:, c_idx] = col / (np.sum(col) if resolved_pos == "convex" else np.linalg.norm(col))
            init_v.append(v0)

    if k_i > 1:
        for idx, op in enumerate(smoothers):
            if op is None or isinstance(op, ColumnSupportShrinkageOperator):
                continue
            if grp_list[idx] is None or op.alpha > MAX_ALPHA_MULTI_COMPONENT:
                raise ValueError(
                    f"k={k_i} > 1 collapses components under global uniform shrinkage: view {idx} needs "
                    f"disjoint supports= (or use fit_uniform_shrunk(..., k={k_i})), or groups= with "
                    f"alpha <= {MAX_ALPHA_MULTI_COMPONENT} (got groups="
                    f"{'given' if grp_list[idx] is not None else None}, alpha={op.alpha:.3f})."
                )

    has_sparse = sparseness is not None and (
        isinstance(sparseness, (list, tuple)) or float(sparseness) > 0.0
    )
    has_supports = any(s is not None for s in supp_list)
    if isinstance(nsa_w, str):
        if nsa_w != "auto":
            raise ValueError(f"nsa_w must be 'auto' or a float; got {nsa_w!r}.")
        eff_nsa_w = 0.0 if (has_supports or not has_sparse) else 0.20
    else:
        eff_nsa_w = float(nsa_w)
    eff_sp_type = sparseness_type if has_sparse else "hard"

    out = UniformShrinkageParams(
        scale_list=list(scale_list),
        energy_type=resolved_energy,
        mixing_algorithm=resolved_mix,
        positivity=resolved_pos,
        nsa_w=eff_nsa_w,
        sparseness=sparseness,
        sparseness_type=eff_sp_type,
        smoothing_matrices=smoothers,
        initial_v=init_v,
        missing_rows=missing_rows,
    )
    out.provenance = {
        "k": k_i,
        "n_obs": [_n_observed_rows(m) for m in mats],
        "n_features": [int(m.shape[1]) for m in mats],
        "alpha": [
            None
            if op is None
            else (list(op.alphas) if isinstance(op, ColumnSupportShrinkageOperator) else op.alpha)
            for op in smoothers
        ],
        "alpha_mode": str(alpha_mode),
        "effective_dof": deff_prov,
        "grouped": [g is not None for g in grp_list],
        "supports_provided": [s is not None for s in supp_list],
        "positive_part": bool(positive_part),
        "prior_scale": float(prior_scale),
        "clip": tuple(clip),
        "scale_list": list(scale_list),
        "energy_type": resolved_energy,
        "energy_type_requested": energy_type,
        "mixing_algorithm": resolved_mix,
        "mixing_algorithm_requested": mixing_algorithm,
        "positivity": resolved_pos,
        "positivity_requested": positivity,
        "nsa_w": eff_nsa_w,
        "nsa_w_requested": nsa_w,
        "sparseness": sparseness,
        "sparseness_type": eff_sp_type,
        "missing_rows": missing_rows,
    }
    return out


def _stage2_initial_basis(
    v1: np.ndarray,
    supports: Sequence[np.ndarray],
    positivity: str,
) -> np.ndarray:
    """Build support-restricted Stage-2 initial basis normalized for ``convex`` (L1) or ``conical`` (L2)."""
    p, k = v1.shape
    init = np.zeros((p, k), dtype=np.float64)
    is_convex = positivity == "convex"
    for c, idx in enumerate(supports):
        col = v1[idx, c]
        if np.any(col > 0.0):
            sub = np.maximum(col, 0.0)
        elif np.any(col < 0.0):
            sub = np.maximum(-col, 0.0)
        else:
            sub = np.ones(idx.size, dtype=np.float64)
        if not np.any(sub > 0.0):
            sub = np.ones(idx.size, dtype=np.float64)
        sub = sub / (float(np.sum(sub)) if is_convex else float(np.linalg.norm(sub)))
        init[idx, c] = sub
    return init


def fit_uniform_shrunk(
    views: Any,
    *,
    k: int = 1,
    alpha: float | str | Sequence[float | str | None] = "auto",
    alpha_mode: str = "ric",
    prior_scale: float = 1.0,
    clip: tuple[float, float] = (0.05, 0.85),
    positivity: str = "auto",
    sparseness: float | Sequence[float | None] | None = None,
    energy_type: str = "auto",
    mixing_algorithm: str = "auto",
    stage1_nsa_w: float = 0.5,
    stage1_init: str = "nndsvd",
    scale_list: Sequence[str] = ("none",),
    missing_rows: str = "mask",
    random_state: int = 0,
    iterations: int | None = None,
    learning_rate: float | str | None = None,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    r"""Fit directional uniform-shrunk or two-stage support-restricted ``simlr``.

    - Defaults via :func:`resolve_directional_defaults`: ``positivity="auto"`` resolves
      to ``"conical"`` (unit-:math:`\ell_2` cone) at :math:`k=1` under degree-0 ``"nc"``
      and ``"convex"`` (radial simplex) at :math:`k>1` under degree-1 ``"acc"``.
    - When ``k == 1`` and ``sparseness`` is ``None`` (or ``0.0``): runs a single-stage
      dense uniform-shrunk fit via :func:`uniform_shrinkage_params`.
    - Whenever ``k > 1`` or any view has ``sparseness > 0``: runs **Two-Stage
      Support-Restricted Uniform Shrinkage**:

      1. **Stage 1** discovers the basin and non-negative lobes (dense shrunk fit for
         ``k == 1``; NSA-Flow fit with ``nsa_w=stage1_nsa_w`` for ``k > 1``).
      2. :func:`extract_disjoint_supports` extracts pairwise-disjoint active supports
         ``S_{m,c}`` (non-subtractive top-``m`` within each lobe).
      3. **Stage 2** polishes the weights with :class:`ColumnSupportShrinkageOperator`
         calibrated via ``alpha_mode`` (``"ric"`` or ``"plugin"``), ``nsa_w=0.0``, and
         ``sparseness=None``, guaranteeing exact zeros outside ``S_{m,c}`` and exact
         orthogonality (:math:`V_m^\top V_m = I_k` for ``conical``; disjoint radial
         simplex faces for ``convex``) for every :math:`\alpha_{m,c} \in [0, 1]`.

    Returns the standard ``pysimlr.fit("simlr", ...)`` result dict enriched with
    ``result["uniform_shrinkage_provenance"]``.
    """
    from .api import fit as _api_fit

    k_i = int(k)
    if k_i < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    resolved_pos, resolved_energy, resolved_mix = resolve_directional_defaults(
        k_i, positivity=positivity, energy_type=energy_type, mixing_algorithm=mixing_algorithm
    )
    if resolved_pos not in ("conical", "convex"):
        raise ValueError(
            f"fit_uniform_shrunk requires non-negative positivity ('conical', 'convex', or 'auto'); got {positivity!r}."
        )
    mats = _extract_view_list(views)
    m_cnt = len(mats)
    mode_key = str(alpha_mode).lower().strip()
    is_jk_mode = mode_key in ("auto", "jk_js")
    if not is_jk_mode and mode_key not in ("ric", "plugin"):
        raise ValueError(f"alpha_mode must be 'ric', 'plugin', 'auto', or 'jk_js'; got {alpha_mode!r}.")
    allow_flip = resolved_energy == "nc_abs" or k_i > 1

    if sparseness == "auto" or (
        isinstance(sparseness, (list, tuple)) and any(x == "auto" for x in sparseness)
    ):
        from .presets import resolve_auto_sparseness
        q_auto = resolve_auto_sparseness(
            views, k_i, positivity=resolved_pos, path_graph=fit_kwargs.get("path_graph")
        )
        if sparseness == "auto":
            q_list = list(q_auto)
        else:
            if len(sparseness) != m_cnt:  # type: ignore[arg-type]
                raise ValueError(f"sparseness sequence has {len(sparseness)} entries for {m_cnt} views.")  # type: ignore[arg-type]
            q_list = [
                q_auto[i] if sq == "auto" else (None if sq is None else float(sq))
                for i, sq in enumerate(sparseness)  # type: ignore[arg-type]
            ]
    elif isinstance(sparseness, (list, tuple)):
        if len(sparseness) != m_cnt:
            raise ValueError(f"sparseness sequence has {len(sparseness)} entries for {m_cnt} views.")
        q_list = [None if q is None else float(q) for q in sparseness]
    else:
        q_list = [None if sparseness is None else float(sparseness)] * m_cnt
    for idx_q, qv in enumerate(q_list):
        if qv is not None and (not np.isfinite(qv) or not (0.0 <= qv < 1.0)):
            raise ValueError(f"sparseness[{idx_q}] must be None or in [0, 1); got {qv}.")
    has_sparse = any(qv is not None and qv > 0.0 for qv in q_list)

    extra_kw: dict[str, Any] = dict(fit_kwargs)
    if iterations is not None:
        extra_kw["iterations"] = int(iterations)
    if learning_rate is not None:
        extra_kw["learning_rate"] = learning_rate

    # ----------------------------------------------------------------- Single-stage dense k == 1
    if k_i == 1 and not has_sparse:
        eff_alpha = alpha
        eff_clip = clip
        if is_jk_mode and alpha == "auto":
            from .presets import _thin_subspace, jackknife_js_alpha
            n_rows = mats[0].shape[0]
            sub_u: list[np.ndarray] = []
            for mat in mats:
                om = np.isfinite(mat).any(axis=1)
                u_f = np.full(n_rows, np.nan, dtype=np.float64)
                if int(om.sum()) >= 4 and mat.shape[1] >= 1:
                    sub = np.nan_to_num(mat[om], nan=0.0)
                    sd = np.std(sub, axis=0, keepdims=True)
                    sc = np.where(sd > 1e-12, (sub - np.mean(sub, axis=0, keepdims=True)) / np.maximum(sd, 1e-12), 0.0)
                    ur, _, vr = _thin_subspace(sc, max_r=1)
                    if ur.shape[1] > 0:
                        sgn_v = 1.0 if float(np.sum(vr[:, 0])) >= 0.0 else -1.0
                        u_f[om] = ur[:, 0] * sgn_v
                sub_u.append(u_f)
            eff_alpha = [
                jackknife_js_alpha(
                    mat,
                    (
                        np.nanmean(np.column_stack([sub_u[j] for j in range(m_cnt) if j != m_idx]), axis=1)
                        if m_cnt > 1
                        else sub_u[m_idx]
                    ),
                    allow_sign_flip=allow_flip,
                )
                for m_idx, mat in enumerate(mats)
            ]
            eff_clip = (0.0, 1.0)

        kw = uniform_shrinkage_params(
            views,
            k=1,
            alpha=eff_alpha,
            alpha_mode="ric" if is_jk_mode else alpha_mode,
            prior_scale=prior_scale,
            clip=eff_clip,
            positivity=resolved_pos,
            sparseness=None,
            sparseness_type="hard",
            nsa_w=0.0,
            energy_type=resolved_energy,
            mixing_algorithm=resolved_mix,
            scale_list=scale_list,
            missing_rows=missing_rows,
        )
        res = _api_fit("simlr", views, k=1, random_state=int(random_state), **kw, **extra_kw)
        supports_dense = [[np.arange(int(m.shape[1]), dtype=np.int64)] for m in mats]
        res["uniform_shrinkage_provenance"] = {
            "stage_mode": "single_stage_dense",
            "k": 1,
            "positivity": resolved_pos,
            "positivity_requested": positivity,
            "alpha_mode": str(alpha_mode),
            "sparseness": sparseness,
            "sparseness_resolved": q_list,
            "resolved_energy_type": resolved_energy,
            "resolved_mixing_algorithm": resolved_mix,
            "n_obs": kw.provenance["n_obs"],
            "n_features": kw.provenance["n_features"],
            "supports": [[s.tolist() for s in sm] for sm in supports_dense],
            "alphas": kw.provenance["alpha"],
            "effective_dof": kw.provenance["effective_dof"],
        }
        return res

    # ----------------------------------------------------------------- Two-Stage support-restricted
    stage1_init_used: str | None = None
    if k_i == 1:
        s1_kw = uniform_shrinkage_params(
            views,
            k=1,
            alpha=alpha,
            alpha_mode="ric" if is_jk_mode else alpha_mode,
            prior_scale=prior_scale,
            clip=clip,
            positivity=resolved_pos,
            sparseness=None,
            sparseness_type="hard",
            nsa_w=0.0,
            energy_type=resolved_energy,
            mixing_algorithm=resolved_mix,
            scale_list=scale_list,
            missing_rows=missing_rows,
        )
        stage1_res = _api_fit("simlr", views, k=1, random_state=int(random_state), **s1_kw, **extra_kw)
        stage1_init_used = "uniform"
    else:
        stage1_init_used = str(stage1_init)
        s1_common: dict[str, Any] = {
            "scale_list": list(scale_list),
            "positivity": resolved_pos,
            "sparseness": None,
            "sparseness_type": "hard",
            "nsa_w": float(stage1_nsa_w),
            "energy_type": resolved_energy,
            "mixing_algorithm": resolved_mix,
            "missing_rows": missing_rows,
            "random_state": int(random_state),
            **extra_kw,
        }
        try:
            stage1_res = _api_fit("simlr", views, k=k_i, initialization_type=stage1_init_used, **s1_common)
        except (ValueError, RuntimeError) as exc:
            if stage1_init_used == "pca":
                raise
            logger.warning(
                "fit_uniform_shrunk Stage 1 initialization_type=%r failed (%s); falling back to 'pca'.",
                stage1_init_used,
                exc,
            )
            stage1_init_used = "pca"
            stage1_res = _api_fit("simlr", views, k=k_i, initialization_type="pca", **s1_common)

    v1_mats = [
        v.detach().cpu().double().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float64)
        for v in stage1_res["v"]
    ]
    supports_by_view: list[list[np.ndarray]] = []
    init_v2: list[np.ndarray] = []
    for m_idx, v1_m in enumerate(v1_mats):
        supps_m = extract_disjoint_supports(v1_m, sparseness=q_list[m_idx], min_per_col=2 if k_i > 1 else 1)
        supports_by_view.append(supps_m)
        init_v2.append(_stage2_initial_basis(v1_m, supps_m, resolved_pos))

    eff_s2_alpha: Any = alpha
    eff_s2_clip = clip
    if is_jk_mode and alpha == "auto":
        from .presets import jackknife_js_alpha
        u_s1 = (
            stage1_res["u"].detach().cpu().double().numpy()
            if torch.is_tensor(stage1_res["u"])
            else np.asarray(stage1_res["u"], dtype=np.float64)
        )
        eff_s2_alpha = [
            [
                jackknife_js_alpha(mat[:, s_c], u_s1[:, c_i], allow_sign_flip=allow_flip)
                for c_i, s_c in enumerate(supps_m)
            ]
            for mat, supps_m in zip(mats, supports_by_view)
        ]
        eff_s2_clip = (0.0, 1.0)

    s2_kw = uniform_shrinkage_params(
        views,
        k=k_i,
        alpha=eff_s2_alpha,
        alpha_mode="ric" if is_jk_mode else alpha_mode,
        prior_scale=prior_scale,
        clip=eff_s2_clip,
        positivity=resolved_pos,
        sparseness=None,
        sparseness_type="hard",
        nsa_w=0.0,
        energy_type=resolved_energy,
        mixing_algorithm=resolved_mix,
        scale_list=scale_list,
        missing_rows=missing_rows,
        supports=supports_by_view,
    )
    s2_kw["initial_v"] = init_v2

    stage2_res = _api_fit("simlr", views, k=k_i, random_state=int(random_state), **s2_kw, **extra_kw)
    stage2_res["uniform_shrinkage_provenance"] = {
        "stage_mode": "two_stage_support_restricted",
        "k": k_i,
        "positivity": resolved_pos,
        "positivity_requested": positivity,
        "alpha_mode": str(alpha_mode),
        "sparseness": sparseness,
        "sparseness_resolved": q_list,
        "resolved_energy_type": resolved_energy,
        "resolved_mixing_algorithm": resolved_mix,
        "stage1_nsa_w": float(stage1_nsa_w) if k_i > 1 else 0.0,
        "stage1_init_used": stage1_init_used,
        "n_obs": s2_kw.provenance["n_obs"],
        "n_features": s2_kw.provenance["n_features"],
        "supports": [[s.tolist() for s in sm] for sm in supports_by_view],
        "alphas": s2_kw.provenance["alpha"],
        "effective_dof": s2_kw.provenance["effective_dof"],
    }
    return stage2_res


def project_uniform_shrunk(
    result: Mapping[str, Any],
    views: Any,
    *,
    post_shrink: float | str = "auto",
    view_scaling: str = "natural",
    train_views: Any = None,
    alpha_mode: str | None = None,
) -> dict[str, Any]:
    r"""Project ``views`` onto a fitted ``simlr`` result using support-restricted simplex weights.

    For each view :math:`m` and component :math:`c`, let
    :math:`S_{m,c} = \{i : \hat{v}_{m,i,c} > 0\}` (flipped to :math:`\max(-\hat{v}_{m,c}, 0)`
    if :math:`\sum_i \hat{v}_{m,i,c} < 0`). The fitted column is normalized strictly on
    :math:`S_{m,c}` and blended toward the support-restricted uniform vector
    :math:`|S_{m,c}|^{-1}\mathbf{1}_{S_{m,c}}` (preserving exact zeros outside :math:`S_{m,c}`
    and keeping :math:`k \ge 2` disjoint components uncollapsed):

    .. math::

       \mathbf{w}_{m,c}(\gamma_{m,c}) = (1 - \gamma_{m,c})\,\hat{\mathbf{w}}_{m,c}
           + \gamma_{m,c}\,\frac{1}{|S_{m,c}|}\mathbf{1}_{S_{m,c}},
       \qquad \gamma_{m,c} \in [0, 1].

    When ``post_shrink="auto"`` (default), per-(view, component) damping sets
    :math:`\gamma_{m,c} = \operatorname{clip}\bigl(1 - n_{\text{tr},m} / (1.5\,d_{\text{eff}}(|S_{m,c}|, p_m)), 0.0, 0.60\bigr)`
    where :math:`d_{\text{eff}}` uses :func:`effective_support_dof` (default ``"ric"``).
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

    prov_mode = (
        res_core.get("uniform_shrinkage_provenance", {}).get("alpha_mode")
        if isinstance(res_core, Mapping)
        else None
    )
    eff_alpha_mode = str(alpha_mode or prov_mode or "ric")

    if isinstance(post_shrink, str):
        if post_shrink != "auto":
            raise ValueError(f"post_shrink must be 'auto' or a float in [0, 1]; got {post_shrink!r}.")
        auto_gamma = True
        fixed_gamma = 0.0
    else:
        auto_gamma = False
        fixed_gamma = float(post_shrink)
        if not np.isfinite(fixed_gamma) or not (0.0 <= fixed_gamma <= 1.0):
            raise ValueError(f"post_shrink must be in [0, 1]; got {post_shrink}.")

    n_eval = eval_mats[0].shape[0]
    m_cnt = len(eval_mats)
    first_v = (
        v_list[0].detach().cpu().double().numpy()
        if torch.is_tensor(v_list[0])
        else np.asarray(v_list[0], dtype=np.float64)
    )
    k = int(first_v.shape[1])

    simplex_weights: list[np.ndarray] = []
    supports_out: list[list[np.ndarray]] = []
    gammas_by_view: list[list[float]] = []
    view_scores = np.full((n_eval, m_cnt, k), np.nan, dtype=np.float64)

    for m_i, (z_ev, z_tr, raw_v) in enumerate(zip(eval_mats, tr_mats, v_list)):
        v_arr = (
            raw_v.detach().cpu().double().numpy()
            if torch.is_tensor(raw_v)
            else np.asarray(raw_v, dtype=np.float64)
        )
        p_m = int(z_ev.shape[1])
        if v_arr.shape != (p_m, k):
            raise ValueError(f"view {m_i} basis has shape {v_arr.shape}, expected ({p_m}, {k}).")

        n_tr_m = _n_observed_rows(z_tr)
        w_mat = np.zeros((p_m, k), dtype=np.float64)
        supp_m: list[np.ndarray] = []
        gam_m: list[float] = []
        for c_i in range(k):
            col = v_arr[:, c_i]
            pos = np.maximum(col, 0.0) if float(np.sum(col)) >= 0.0 else np.maximum(-col, 0.0)
            supp = np.flatnonzero(pos > 0.0).astype(np.int64)
            if supp.size == 0:
                supp = np.arange(p_m, dtype=np.int64)
                w_fit_sub = np.full(p_m, 1.0 / float(p_m), dtype=np.float64)
            else:
                w_fit_sub = pos[supp] / float(np.sum(pos[supp]))

            if auto_gamma:
                if eff_alpha_mode.lower().strip() in ("auto", "jk_js"):
                    gamma_mc = 0.0
                else:
                    d_eff = effective_support_dof(int(supp.size), p_m, mode=eff_alpha_mode)
                    gamma_mc = float(np.clip(1.0 - float(n_tr_m) / (1.5 * d_eff), 0.0, 0.60))
            else:
                gamma_mc = fixed_gamma

            u_supp = np.full(supp.size, 1.0 / float(supp.size), dtype=np.float64)
            w_mat[supp, c_i] = (1.0 - gamma_mc) * w_fit_sub + gamma_mc * u_supp
            supp_m.append(supp)
            gam_m.append(gamma_mc)

        simplex_weights.append(w_mat)
        supports_out.append(supp_m)
        gammas_by_view.append(gam_m)

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
    alphas: list[float] = []
    sm_list = res_core.get("parameters", {}).get("smoothing_matrices") if isinstance(res_core, Mapping) else None
    if isinstance(sm_list, (list, tuple)):
        for op in sm_list:
            alphas.append(float(getattr(op, "alpha", np.nan)) if op is not None else 0.0)

    flat_gammas = [g for row in gammas_by_view for g in row]
    return {
        "u": u_cons,
        "view_scores": view_scores[:, :, 0] if k == 1 else view_scores,
        "weights": simplex_weights,
        "supports": supports_out,
        "post_shrink": float(np.mean(flat_gammas)) if flat_gammas else 0.0,
        "post_shrink_by_view": gammas_by_view,
        "view_scaling": view_scaling,
        "alphas": alphas,
    }


__all__ = [
    "MAX_ALPHA_MULTI_COMPONENT",
    "ColumnSupportShrinkageOperator",
    "UniformShrinkageOperator",
    "UniformShrinkageParams",
    "auto_uniform_alpha",
    "create_uniform_smoother",
    "effective_support_dof",
    "extract_disjoint_supports",
    "fit_uniform_shrunk",
    "project_uniform_shrunk",
    "resolve_directional_defaults",
    "uniform_shrinkage_params",
]
