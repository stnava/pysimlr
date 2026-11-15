"""Cross-view system prior on first-layer loadings ``P_i V_i``.

`domain_matrices` and `smoothing_matrices` act within each view independently,
so permuting the rows of ``P_i`` changes nothing and a shared anatomical or
functional system index across modalities is never read. `system_matrices`
couples system-level loadings across views directly:

* ``system_matrices[i]`` has shape ``(S, p_i)`` (or ``None`` when view ``i``
  does not participate), with a common system row index ``0 .. S - 1`` across
  views.
* Non-zero rows are normalised to unit :math:`\\ell_2` norm so a uniform
  unit-:math:`\\ell_2` loading supported on system ``s`` has identical system
  projection ``1.0`` regardless of how many features represent system ``s`` in
  each view; all-zero rows mark systems absent from view ``i`` and are skipped.
* ``system_penalty="l2"`` adds
  :math:`\\mu \\sum_{i < j} \\sum_{s \\in A_i \\cap A_j} \\lVert (\\tilde{P}_i W_i)_{s,:} - (\\tilde{P}_j W_j)_{s,:} \\rVert_2^2`.
* ``system_penalty="group_l2"`` adds the smoothed group-:math:`\\ell_{2,1}`
  penalty
  :math:`\\mu \\sum_{s : |A_s| \\ge 1} \\sum_{c=1}^{k} (\\sqrt{\\sum_{i \\in A_s} (\\tilde{P}_i W_i)_{s,c}^2 + \\tau^2} - \\tau)`
  so a system is selected or dropped across participating views together.

Positivity and sparseness projections remain per view.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

SYSTEM_PENALTIES: tuple[str, str] = ("l2", "group_l2")
_ROW_NORM_EPS: float = 1e-12
_GROUP_L2_TAU: float = 1e-8

__all__ = [
    "SYSTEM_PENALTIES",
    "SystemPrior",
    "normalize_system_matrices_arg",
    "prepare_system_prior",
    "system_total_energy",
    "system_view_context",
    "system_view_energy",
]


@dataclass(frozen=True)
class SystemPrior:
    """Validated, row-normalised cross-view system prior."""

    matrices: tuple[torch.Tensor | None, ...]
    active: torch.Tensor
    weight: float
    penalty: str
    n_systems: int

    def to(self, *, dtype: torch.dtype, device: torch.device | None = None) -> SystemPrior:
        mats = tuple(
            None if m is None else m.to(dtype=dtype, device=device).contiguous()
            for m in self.matrices
        )
        act = self.active.to(device=device) if device is not None else self.active
        return SystemPrior(
            matrices=mats,
            active=act,
            weight=self.weight,
            penalty=self.penalty,
            n_systems=self.n_systems,
        )


def _resolve_idx(key: Any, n_views: int, view_names: Sequence[str] | None) -> int:
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool) and 0 <= int(key) < n_views:
        return int(key)
    if view_names is not None and isinstance(key, str) and key in view_names:
        return int(list(view_names).index(key))
    if isinstance(key, str) and key.startswith("view") and key[4:].isdigit() and int(key[4:]) < n_views:
        return int(key[4:])
    names_hint = f" or one of {list(view_names)}" if view_names is not None else ""
    raise ValueError(
        f"unknown view key {key!r} in system_matrices; expected 0..{n_views - 1}"
        f" or 'view0'..'view{n_views - 1}'{names_hint}."
    )


def normalize_system_matrices_arg(
    system_matrices: Any,
    n_views: int,
    view_names: Sequence[str] | None = None,
) -> list[Any] | None:
    """Convert a sequence or view-keyed mapping of system matrices into a per-view list."""
    if system_matrices is None:
        return None
    if isinstance(system_matrices, Mapping):
        if not system_matrices:
            raise ValueError("system_matrices mapping is empty.")
        out: list[Any] = [None] * n_views
        seen = [False] * n_views
        for raw_k, val in system_matrices.items():
            idx = _resolve_idx(raw_k, n_views, view_names)
            if seen[idx]:
                raise ValueError(f"system_matrices gives view {idx} twice.")
            seen[idx] = True
            out[idx] = val
        return out
    if isinstance(system_matrices, (list, tuple)):
        if len(system_matrices) != n_views:
            raise ValueError(
                f"system_matrices has {len(system_matrices)} entries, expected {n_views} (one per view)."
            )
        return list(system_matrices)
    raise ValueError(
        f"system_matrices must be a per-view list/tuple or dict of (S, p_i) matrices; "
        f"got {type(system_matrices).__name__}."
    )


def _validate_system_lambda(system_lambda: Any, has_matrices: bool) -> float:
    if system_lambda is None:
        return 1.0 if has_matrices else 0.0
    if isinstance(system_lambda, bool) or not isinstance(system_lambda, (int, float, np.floating, np.integer)):
        raise ValueError(f"system_lambda must be a non-negative float; got {system_lambda!r}.")  # noqa: TRY004
    val = float(system_lambda)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"system_lambda must be a finite non-negative float; got {system_lambda!r}.")
    if not has_matrices and val != 0.0:
        raise ValueError(
            "system_lambda was set without system_matrices; pass system_matrices to couple "
            "system-level loadings across views."
        )
    return val


def prepare_system_prior(
    system_matrices: Any,
    system_lambda: Any = None,
    system_penalty: str = "l2",
    *,
    view_dims: Sequence[int],
    view_names: Sequence[str] | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> SystemPrior | None:
    """Validate and row-normalise `system_matrices` for a multi-view fit."""
    if not isinstance(system_penalty, str) or system_penalty not in SYSTEM_PENALTIES:
        raise ValueError(
            f"system_penalty must be one of {SYSTEM_PENALTIES}; got {system_penalty!r}."
        )
    has_matrices = system_matrices is not None
    weight = _validate_system_lambda(system_lambda, has_matrices)
    if not has_matrices:
        return None
    n_views = len(view_dims)
    per_view = normalize_system_matrices_arg(system_matrices, n_views, view_names=view_names)
    assert per_view is not None
    if all(m is None for m in per_view):
        raise ValueError("system_matrices gives None for every view.")

    n_systems: int | None = None
    norm_mats: list[torch.Tensor | None] = [None] * n_views
    active_rows: list[torch.Tensor | None] = [None] * n_views
    for idx, (raw_m, p_dim) in enumerate(zip(per_view, view_dims)):
        if raw_m is None:
            continue
        if hasattr(raw_m, "to_numpy"):
            arr_t = torch.as_tensor(np.asarray(raw_m.to_numpy(), dtype=np.float64), dtype=torch.float64)
        else:
            try:
                arr_t = torch.as_tensor(raw_m, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"system_matrices[{idx}] must be a numeric 2-D matrix; {exc}") from exc
        if arr_t.ndim != 2:
            raise ValueError(
                f"system_matrices[{idx}] must be 2-D of shape (n_systems, {p_dim}); "
                f"got shape {tuple(arr_t.shape)}."
            )
        s_rows, p_cols = int(arr_t.shape[0]), int(arr_t.shape[1])
        if s_rows < 1:
            raise ValueError(f"system_matrices[{idx}] must have at least 1 system row; got 0.")
        if p_cols != int(p_dim):
            raise ValueError(
                f"system_matrices[{idx}] has {p_cols} feature columns, but view {idx} has {int(p_dim)} features."
            )
        if not bool(torch.isfinite(arr_t).all()):
            raise ValueError(f"system_matrices[{idx}] contains non-finite values.")
        if n_systems is None:
            n_systems = s_rows
        elif s_rows != n_systems:
            raise ValueError(
                f"system_matrices must share a common system row count across views; "
                f"got {n_systems} and {s_rows} (view {idx})."
            )
        row_norms = torch.linalg.vector_norm(arr_t, dim=1)
        act = row_norms > _ROW_NORM_EPS
        normed = torch.where(
            act[:, None],
            arr_t / row_norms.clamp_min(_ROW_NORM_EPS)[:, None],
            torch.zeros_like(arr_t),
        )
        norm_mats[idx] = normed.to(dtype=dtype, device=device).contiguous()
        active_rows[idx] = act

    assert n_systems is not None
    active = torch.zeros((n_views, n_systems), dtype=torch.bool, device=device)
    for idx, act in enumerate(active_rows):
        if act is not None:
            active[idx] = act.to(device=device) if device is not None else act

    if system_penalty == "l2":
        if not bool((active.sum(dim=0) >= 2).any()):
            raise ValueError(
                "system_matrices with system_penalty='l2' requires at least two views to share "
                "at least one non-zero system row."
            )
    elif not bool(active.any()):
        raise ValueError("system_matrices has no non-zero system rows in any view.")

    return SystemPrior(
        matrices=tuple(norm_mats),
        active=active,
        weight=weight,
        penalty=system_penalty,
        n_systems=n_systems,
    )


def system_total_energy(w_list: Sequence[torch.Tensor], prior: SystemPrior) -> torch.Tensor:
    """Evaluate the cross-view system penalty across all views."""
    ref = w_list[0]
    if prior.weight == 0.0:
        return ref.sum() * 0.0
    projs: list[torch.Tensor | None] = [
        None if p is None else (p.to(dtype=w.dtype, device=w.device) @ w)
        for p, w in zip(prior.matrices, w_list)
    ]
    if prior.penalty == "l2":
        acc = ref.sum() * 0.0
        n_views = len(projs)
        for i in range(n_views):
            gi = projs[i]
            if gi is None:
                continue
            for j in range(i + 1, n_views):
                gj = projs[j]
                if gj is None:
                    continue
                mask_ij = prior.active[i] & prior.active[j]
                if bool(mask_ij.any()):
                    diff = gi[mask_ij] - gj[mask_ij]
                    acc = acc + torch.sum(diff * diff)
        return prior.weight * acc

    k_cols = int(ref.shape[1])
    sq_sum = torch.zeros((prior.n_systems, k_cols), dtype=ref.dtype, device=ref.device)
    for i, gi in enumerate(projs):
        if gi is None:
            continue
        act_i = prior.active[i].to(device=gi.device)
        sq_sum = sq_sum + torch.where(act_i[:, None], gi * gi, torch.zeros_like(gi))
    sys_active = prior.active.any(dim=0).to(device=ref.device)
    tau = _GROUP_L2_TAU
    return prior.weight * torch.sum(torch.sqrt(sq_sum[sys_active] + tau * tau) - tau)


def system_view_context(
    i: int,
    w_list: Sequence[torch.Tensor],
    prior: SystemPrior | None,
) -> dict[str, Any] | None:
    """Precompute fixed other-view system projections for block coordinate updates of view `i`."""
    if prior is None or prior.weight == 0.0:
        return None
    p_i = prior.matrices[i]
    if p_i is None or not bool(prior.active[i].any()):
        return None
    act_i = prior.active[i]
    if prior.penalty == "l2":
        targets: list[tuple[torch.Tensor, torch.Tensor]] = []
        for j, (p_j, w_j) in enumerate(zip(prior.matrices, w_list)):
            if j == i or p_j is None:
                continue
            mask_ij = act_i & prior.active[j]
            if bool(mask_ij.any()):
                wj_d = w_j.detach().to(dtype=p_j.dtype, device=p_j.device)
                gj = p_j[mask_ij] @ wj_d
                targets.append((p_i[mask_ij], gj))
        if not targets:
            return None
        return {"penalty": "l2", "weight": prior.weight, "targets": targets}

    k_cols = int(w_list[0].shape[1])
    n_act = int(act_i.sum().item())
    other_sq = torch.zeros((n_act, k_cols), dtype=p_i.dtype, device=p_i.device)
    for j, (p_j, w_j) in enumerate(zip(prior.matrices, w_list)):
        if j == i or p_j is None:
            continue
        both = act_i & prior.active[j]
        if not bool(both.any()):
            continue
        wj_d = w_j.detach().to(dtype=p_j.dtype, device=p_j.device)
        gj_act = p_j[act_i] @ wj_d
        rel_mask = prior.active[j][act_i]
        other_sq = other_sq + torch.where(rel_mask[:, None], gj_act * gj_act, torch.zeros_like(gj_act))
    return {
        "penalty": "group_l2",
        "weight": prior.weight,
        "p_i_act": p_i[act_i],
        "other_sq": other_sq,
    }


def system_view_energy(w_i: torch.Tensor, view_ctx: dict[str, Any] | None) -> torch.Tensor:
    """Single-view smooth energy term matching `system_total_energy` up to other-view constants."""
    if view_ctx is None:
        return w_i.sum() * 0.0
    if view_ctx["penalty"] == "l2":
        acc = w_i.sum() * 0.0
        for p_sub, gj in view_ctx["targets"]:
            diff = (p_sub @ w_i) - gj
            acc = acc + torch.sum(diff * diff)
        return view_ctx["weight"] * acc
    gi = view_ctx["p_i_act"] @ w_i
    tau = _GROUP_L2_TAU
    return view_ctx["weight"] * torch.sum(
        torch.sqrt(gi * gi + view_ctx["other_sq"] + tau * tau) - tau
    )
