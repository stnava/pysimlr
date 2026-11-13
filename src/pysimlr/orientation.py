r"""Orient latents across fits: signs, component order and (optionally) rotation.

A latent space is identified only up to a sign per component, an ordering of the components and, for
a rotationally invariant objective, a rotation. Two fits (two seeds, two subsamples, two cohorts) of
the same model therefore return latents, weights and maps that agree only after they are put in one
frame. `orient_latents` does that for a list of fits without privileging any of them: the reference
is a function of the *set* of fits, not of their order.

Reference rule (``reference=None``). Write :math:`Z_i` for fit ``i``'s latents, centered, with unit-norm
columns, on the rows finite in every fit.

1. Pairwise agreement :math:`S_{ii'}` is the total absolute cosine of the best match between the two
   fits' columns under the chosen ``mode`` (``"sign"``: diagonal only; ``"signed_permutation"``: the
   Hungarian match on :math:`|Z_i^\top Z_{i'}|`; ``"procrustes"``: the nuclear norm of
   :math:`Z_i^\top Z_{i'}`). It is symmetric in the two fits.
2. The medoid is the fit with the largest :math:`\sum_{i' \ne i} S_{ii'}`; its latents are the initial reference.
3. Every fit is oriented to the reference (signs, signed permutation or orthogonal rotation), the
   reference is replaced by the renormalized mean of the oriented latents, and this repeats until no
   transform changes (``max_iter``).
4. The final frame is fixed by two conventions that depend on the data only: components are ordered by
   decreasing mean column norm of the oriented raw latents (ties keep the medoid's order), and each
   component's sign is set so the sum of cubes of the mean latent is non-negative (the convention
   `pysimlr.loadings` uses for its varimax columns; an exact zero keeps ``+``).

Every step is a function of the unordered set of fits, so permuting the list permutes the outputs and
changes nothing else (barring exact ties among the :math:`S`, which only identical fits, or fits
differing by a sign or column permutation, produce; those ties are harmless because the final frame
does not depend on which tied fit was the medoid).
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

ORIENT_MODES = ("sign", "signed_permutation", "procrustes")


def _as_array(x: Any) -> np.ndarray:
    if isinstance(x, dict):
        x = x["u"]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x, dtype=np.float64)


def _standardize(a: np.ndarray) -> np.ndarray:
    ac = a - a.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(ac, axis=0, keepdims=True)
    return np.where(nrm > 0, ac / np.where(nrm > 0, nrm, 1.0), 0.0)


def _transform(z: np.ndarray, ref: np.ndarray, mode: str) -> np.ndarray:
    """``(k, k)`` matrix ``T`` with ``z @ T`` closest to ``ref`` under ``mode``."""
    from .loadings import _nearest_signed_permutation, _procrustes
    c = z.T @ ref
    if mode == "sign":
        s = np.sign(np.diag(c))
        s[s == 0] = 1.0
        return np.diag(s)
    if mode == "signed_permutation":
        return _nearest_signed_permutation(c)
    return _procrustes(z, ref)


def _agreement(zi: np.ndarray, zj: np.ndarray, mode: str) -> float:
    from scipy.optimize import linear_sum_assignment
    c = zi.T @ zj
    if mode == "sign":
        return float(np.abs(np.diag(c)).sum())
    if mode == "signed_permutation":
        r, cc = linear_sum_assignment(-np.abs(c))
        return float(np.abs(c[r, cc]).sum())
    return float(np.linalg.svd(c, compute_uv=False).sum())


def _apply(obj: Any, t: np.ndarray) -> Any:
    """``obj @ t`` for an array or a list/tuple of arrays (per-view weights or maps)."""
    if isinstance(obj, (list, tuple)):
        return [_apply(o, t) for o in obj]
    return _as_array(obj) @ t


def orient_latents(latents: Sequence[Any], *, weights: Optional[Sequence[Any]] = None,
                   maps: Optional[Sequence[Any]] = None, mode: str = "signed_permutation",
                   reference: Any = None, max_iter: int = 20) -> Dict[str, Any]:
    """Orient the latents (and optionally weights and maps) of several fits to one frame.

    Parameters
    ----------
    latents : sequence of ``(n, k)`` arrays or `pysimlr.fit` results
        One per fit, rows in a common order (a result contributes ``result["u"]``; pass
        ``u @ result["loading_rotation"]`` arrays to orient the varimax frame instead). Rows that are
        not finite in every fit are ignored when estimating transforms and stay NaN in the output.
    weights, maps : sequence (one entry per fit) of ``(p, k)`` arrays or of lists of per-view arrays, optional
        Carried through the same ``k x k`` transform as the latents of their fit (``w @ T``). For
        squared or share-valued maps (``relative_weight``) use ``abs(T)`` from the returned
        ``transforms`` (only meaningful for ``mode != "procrustes"``).
    mode : {"sign", "signed_permutation", "procrustes"}
        ``"sign"`` flips columns; ``"signed_permutation"`` (default) also matches and reorders them
        (``k > 1``); ``"procrustes"`` applies the best orthogonal ``k x k`` rotation (use for
        rotation-invariant objectives, on latents of comparable scale).
    reference : ``(n, k)`` array, optional
        Orient to this frame instead of the symmetric rule; the output frame is then the reference's
        (no ordering or sign convention is applied).
    max_iter : int
        Iterations of the mean-reference refinement.

    Returns
    -------
    dict
        ``latents`` (list of float64 arrays ``L_i @ T_i``), ``transforms`` (``(F, k, k)``;
        ``oriented = original @ T``, undo with ``oriented @ T.T`` for orthogonal ``T``), ``reference``
        (``(n, k)`` frame the fits were oriented to, NaN on ignored rows), ``reference_index`` (medoid
        position in the input, ``None`` with an explicit reference), ``agreement`` (``(F, F)`` matrix
        ``S``, ``None`` with an explicit reference), ``n_iter``, ``converged``, and ``weights`` / ``maps``
        when given. The rule is described in the module docstring and is invariant to the order of the fits.
    """
    if mode not in ORIENT_MODES:
        raise ValueError(f"mode must be one of {ORIENT_MODES}; got {mode!r}.")
    arrs = [_as_array(x) for x in latents]
    if not arrs:
        raise ValueError("latents is empty.")
    n, k = arrs[0].shape
    if any(a.ndim != 2 or a.shape != (n, k) for a in arrs):
        raise ValueError("every fit's latents must have the same shape (n, k).")
    for name, extra in (("weights", weights), ("maps", maps)):
        if extra is not None and len(extra) != len(arrs):
            raise ValueError(f"{name} has {len(extra)} entries for {len(arrs)} fits.")
    common = np.all([np.isfinite(a).all(axis=1) for a in arrs], axis=0)
    if int(common.sum()) < 3:
        raise ValueError(f"only {int(common.sum())} rows are finite in every fit; need at least 3.")
    z = [_standardize(a[common]) for a in arrs]
    f = len(arrs)
    agreement = medoid = None
    converged, n_iter = True, 0
    if reference is not None:
        ref_full = _as_array(reference)
        if ref_full.shape != (n, k):
            raise ValueError(f"reference has shape {ref_full.shape}; expected {(n, k)}.")
        ref = _standardize(ref_full[common])
        trans = [_transform(zi, ref, mode) for zi in z]
        out_ref = ref_full
    else:
        agreement = np.zeros((f, f))
        for i in range(f):
            for j in range(i + 1, f):
                agreement[i, j] = agreement[j, i] = _agreement(z[i], z[j], mode)
        totals = agreement.sum(axis=1)
        medoid = int(np.argmax(totals))
        ref = z[medoid]
        trans = [_transform(zi, ref, mode) for zi in z]
        converged = False
        for n_iter in range(1, int(max_iter) + 1):
            ref = _standardize(np.mean([zi @ t for zi, t in zip(z, trans)], axis=0))
            new = [_transform(zi, ref, mode) for zi in z]
            same = all(np.array_equal(a, b) for a, b in zip(new, trans)) if mode != "procrustes" else \
                all(np.allclose(a, b, atol=1e-10) for a, b in zip(new, trans))
            trans = new
            if same:
                converged = True
                break
        # Final frame: components by decreasing mean raw column norm, signs by the sum of cubes of the mean latent.
        raw = [(a[common] - a[common].mean(axis=0, keepdims=True)) @ t for a, t in zip(arrs, trans)]
        size = np.mean([np.linalg.norm(r_, axis=0) for r_ in raw], axis=0)
        order = np.argsort(-size, kind="stable")
        mean_lat = np.mean([zi @ t for zi, t in zip(z, trans)], axis=0)[:, order]
        sgn = np.where((mean_lat ** 3).sum(axis=0) < 0, -1.0, 1.0)
        q = np.zeros((k, k))
        q[order, np.arange(k)] = sgn
        trans = [t @ q for t in trans]
        ref_std = mean_lat * sgn[None, :]
        out_ref = np.full((n, k), np.nan)
        out_ref[common] = ref_std
    out: Dict[str, Any] = {
        "latents": [a @ t for a, t in zip(arrs, trans)],
        "transforms": np.asarray(trans),
        "reference": out_ref, "reference_index": medoid, "agreement": agreement,
        "n_iter": int(n_iter), "converged": bool(converged), "mode": mode,
    }
    if weights is not None:
        out["weights"] = [_apply(w, t) for w, t in zip(weights, trans)]
    if maps is not None:
        out["maps"] = [_apply(mp, t) for mp, t in zip(maps, trans)]
    return out
