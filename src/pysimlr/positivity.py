"""The ``positivity`` vocabulary: one resolver, one non-negativity test.

The combination hierarchy for a basis column ``v`` (features x 1):

* ``"linear"``  -- any real weights. Alias of ``"either"``.
* ``"conical"`` -- ``v >= 0``. Alias of ``"positive"``.
* ``"convex"``  -- ``v >= 0`` and ``sum(v) = 1``: the latent ``X v`` is a
  weighted average of the (standardized) features. The simplex is then the
  column gauge: the energy sees ``V`` on the simplex rather than unit-L2
  columns, so the column sums enter the objective (see `simlr`).

Unknown values raise. They used to fall through every membership test and
behave as ``"either"`` -- ``positivity="conical"`` silently fitted a signed
basis.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

#: Values understood somewhere in the package, by their canonical name.
CANONICAL = frozenset({
    "either", "positive", "hard", "nonnegative", "nonneg", "softplus",
    "negative", "nonpositive", "project", "project_negative", "convex",
})

#: Names of the combination hierarchy that are spellings of existing values.
ALIASES = {"linear": "either", "conical": "positive"}

#: Values that constrain the basis to be non-negative.
NONNEG = frozenset({"positive", "hard", "nonnegative", "nonneg", "softplus", "convex"})


def resolve_positivity(positivity: Any) -> str:
    """Canonical name of a ``positivity`` value; raises `ValueError` on anything unknown."""
    if positivity is True:
        return "positive"
    if positivity is False or positivity is None:
        return "either"
    if not isinstance(positivity, str):
        raise TypeError(
            f"positivity must be a string or bool, got {positivity!r}; "
            "for a different constraint per view pass a list/dict to pysimlr.fit/simlr "
            "or use per_view_params={i: {'positivity': ...}}"
        )
    key = positivity.strip().lower()
    if key == "true":
        return "positive"
    if key == "false":
        return "either"
    key = ALIASES.get(key, key)
    if key not in CANONICAL:
        raise ValueError(
            f"unknown positivity={positivity!r}; use one of "
            f"{sorted(CANONICAL | set(ALIASES))} (or True/False)")
    return key


#: In the deep and flow encoders the non-negative spellings are one constraint: the conical one.
DEEP_ALIASES = {"nonnegative": "positive", "nonneg": "positive"}


def resolve_deep_positivity(positivity: Any) -> str:
    """`resolve_positivity` for the deep and flow encoders, where ``"nonnegative"`` and ``"nonneg"``
    mean conical (``"positive"``: clamp and the first-layer normalisation).

    The linear path keeps ``"nonnegative"`` as a separate projection (no degeneracy fallback, see
    `sparsification.apply_positivity`); the deep encoders have no such variant, and before this
    mapping those spellings fell through every membership test and applied no constraint at all.
    """
    key = resolve_positivity(positivity)
    return DEEP_ALIASES.get(key, key)


def is_nonneg(positivity: Any) -> bool:
    """Whether ``positivity`` constrains the basis to be non-negative."""
    return resolve_positivity(positivity) in NONNEG


def is_convex(positivity: Any) -> bool:
    """Whether ``positivity`` constrains each basis column to the probability simplex."""
    return resolve_positivity(positivity) == "convex"


def validate_nonneg_mask(
    mask: Any,
    n_features: Optional[int] = None,
    *,
    positivity: Optional[str] = None,
    view_label: str = "view",
) -> Optional[np.ndarray]:
    """Validate a per-view 1-D boolean ``nonneg_mask`` against feature count and ``positivity``."""
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        if mask.dtype != torch.bool or mask.ndim != 1:
            raise TypeError(
                f"nonneg_mask for {view_label} must be a 1-D boolean array, tensor or sequence; "
                f"got tensor of dtype={mask.dtype}, shape={tuple(mask.shape)}."
            )
        arr = mask.detach().cpu().numpy().astype(bool, copy=True)
    elif isinstance(mask, np.ndarray):
        if mask.dtype != bool or mask.ndim != 1:
            raise TypeError(
                f"nonneg_mask for {view_label} must be a 1-D boolean array, tensor or sequence; "
                f"got ndarray of dtype={mask.dtype}, shape={mask.shape}."
            )
        arr = mask.astype(bool, copy=True)
    elif isinstance(mask, (list, tuple)):
        if not all(isinstance(x, (bool, np.bool_)) for x in mask):
            raise TypeError(
                f"nonneg_mask for {view_label} must contain only booleans (True/False)."
            )
        arr = np.asarray(mask, dtype=bool)
    else:
        raise TypeError(
            f"nonneg_mask for {view_label} must be a 1-D boolean array, tensor or sequence; "
            f"got {type(mask).__name__}."
        )
    if n_features is not None and arr.shape[0] != int(n_features):
        raise ValueError(
            f"nonneg_mask for {view_label} has {arr.shape[0]} entries, expected {int(n_features)} features."
        )
    if positivity is not None:
        p = resolve_positivity(positivity)
        if p == "convex":
            if not bool(arr.all()):
                raise ValueError(
                    f"positivity='convex' cannot be combined with a partial nonneg_mask ({view_label}): "
                    "the unit-sum simplex gauge is undefined when unmasked features are signed; "
                    "use positivity='positive' ('conical') instead."
                )
        elif p not in NONNEG:
            raise ValueError(
                f"nonneg_mask requires a non-negative positivity ('positive'/'conical' or 'nonnegative') "
                f"for {view_label}, got positivity={p!r}."
            )
    return arr
