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

from typing import Any

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
        raise TypeError(f"positivity must be a string or bool, got {positivity!r}")
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
