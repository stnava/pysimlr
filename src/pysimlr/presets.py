"""Automated multi-view data-collection analysis, rank-ordered preset menu, and 3-tier execution.

Provides three operating tiers for multi-view modeling without manual threshold tuning:

1. **Human-selected preset** (:func:`recommend_presets` + ``fit_preset(..., preset="<name>")``):
   inspects the rank-ordered menu and transparent statistical rationales, then executes the
   selected preset with collection-adapted ``missing_rows``, ``topology``, and per-view support.
2. **Automated top-1 selection** (``fit_preset(..., preset="auto")``, the default when no human
   choice is supplied): runs :func:`recommend_presets` in milliseconds using strictly unsupervised
   delete-1 leave-one-out (LOO) and QR hat-matrix PRESS diagnostics, then executes Rank #1.
3. **Static natural setting** (``fit_preset(..., preset=None)``): skips view profiling and fits
   the general linear natural configuration (`simlr` with ``positivity="either"``,
   ``sparseness="auto"``, ``energy_type="recon_r2"``, and $O(k)$-invariant ``mixing_algorithm="pca"``).
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import time
import types
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

SD_FLOOR = 1e-12


@dataclasses.dataclass(frozen=True)
class PresetTemplate:
    """Immutable definition of a general multi-view modeling preset."""

    name: str
    title: str
    family: str
    complexity_tier: int
    estimator: str
    method: str
    static_params: Mapping[str, Any]
    summary: str
    when_to_prefer: str


def _freeze(d: dict[str, Any]) -> Mapping[str, Any]:
    return types.MappingProxyType(dict(d))


PRESET_CATALOG: tuple[PresetTemplate, ...] = (
    PresetTemplate(
        name="general_sparse",
        title="General Sparse Consensus (Natural Default)",
        family="linear",
        complexity_tier=1,
        estimator="fit",
        method="simlr",
        static_params=_freeze(
            {
                "positivity": "either",
                "sparseness": "auto",
                "sparseness_type": "hard",
                "energy_type": "recon_r2",
                "mixing_algorithm": "pca",
                "topology": "loo",
                "nsa_w": 0.5,
            }
        ),
        summary=(
            "Signed linear SiMLR with per-view paired-1SE delete-1 LOO support selection, "
            "variance-explaining recon_r2 energy, and O(k)-invariant PCA consensus."
        ),
        when_to_prefer=(
            "General-purpose starting point when features carry mixed +/- signs and "
            "per-view selectivity should adapt automatically to view width and signal-to-noise."
        ),
    ),
    PresetTemplate(
        name="dense_consensus",
        title="Dense Signed Consensus",
        family="linear",
        complexity_tier=1,
        estimator="fit",
        method="simlr",
        static_params=_freeze(
            {
                "positivity": "either",
                "sparseness": None,
                "sparseness_type": "hard",
                "energy_type": "recon_r2",
                "mixing_algorithm": "pca",
                "topology": "loo",
                "nsa_w": 0.5,
            }
        ),
        summary="Unthresholded signed linear SiMLR retaining all features under recon_r2 and PCA consensus.",
        when_to_prefer=(
            "Narrow tables (n >> p) or broadly distributed bidirectional signals where "
            "hard feature pruning discards coupled variance."
        ),
    ),
    PresetTemplate(
        name="sparse_focal",
        title="Focal Sparse Contrast",
        family="linear",
        complexity_tier=1,
        estimator="fit",
        method="simlr",
        static_params=_freeze(
            {
                "positivity": "either",
                "sparseness": 0.8,
                "sparseness_type": "hard",
                "energy_type": "recon_r2",
                "mixing_algorithm": "pca",
                "topology": "loo",
                "nsa_w": 0.5,
            }
        ),
        summary="High-selectivity signed linear SiMLR isolating compact driver subsets amid wide background noise.",
        when_to_prefer=(
            "Wide omics or parcellation tables where cross-view coupling concentrates in "
            "a small fraction of focal features."
        ),
    ),
    PresetTemplate(
        name="directional_shrunk",
        title="Directional Support-Restricted Uniform Shrinkage",
        family="linear",
        complexity_tier=1,
        estimator="fit_uniform_shrunk",
        method="uniform_shrinkage",
        static_params=_freeze(
            {
                "positivity": "auto",
                "sparseness": None,
                "alpha_mode": "auto",
                "energy_type": "auto",
                "mixing_algorithm": "pca",
            }
        ),
        summary=(
            "Non-negative cone (k=1 conical, k>1 disjoint radial simplex) stabilized by "
            "delete-1 jackknife positive-part James-Stein shrinkage toward the subnetwork mean."
        ),
        when_to_prefer=(
            "Coherent one-sided biological or clinical batteries and high-p/n collinear views "
            "where individual feature weights are noisy and external cohort transfer matters."
        ),
    ),
    PresetTemplate(
        name="directional_parts",
        title="Sparse Non-Negative Parts",
        family="linear",
        complexity_tier=1,
        estimator="fit",
        method="simlr",
        static_params=_freeze(
            {
                "positivity": "convex",
                "sparseness": "auto",
                "sparseness_type": "hard",
                "energy_type": "acc",
                "mixing_algorithm": "pca",
                "topology": "loo",
                "nsa_w": 0.5,
            }
        ),
        summary="Orthogonal non-negative radial-simplex parts decomposition without uniform weight flattening.",
        when_to_prefer=(
            "Additive part-based decompositions (k >= 2) with sharp within-part peaks "
            "and ample sample size."
        ),
    ),
    PresetTemplate(
        name="nonlinear_lend",
        title="Linear-Encoder Nonlinear-Decoder (LEND)",
        family="deep",
        complexity_tier=2,
        estimator="fit",
        method="lend",
        static_params=_freeze(
            {
                "positivity": "either",
                "sparseness": 0.5,
                "energy_type": "align",
                "mixing_algorithm": "newton",
                "topology": "loo",
                "nsa_w": 0.5,
                "first_layer_mode": "scheduled",
            }
        ),
        summary="Interpretable linear feature weights paired with nonlinear MLP view decoders.",
        when_to_prefer=(
            "Curved modality manifolds where linear projections capture the shared axis "
            "but reconstructing raw views is nonlinear."
        ),
    ),
    PresetTemplate(
        name="nonlinear_ned",
        title="Nonlinear Encoder-Decoder (NED)",
        family="deep",
        complexity_tier=2,
        estimator="fit",
        method="ned",
        static_params=_freeze(
            {
                "positivity": "either",
                "sparseness": 0.5,
                "energy_type": "align",
                "mixing_algorithm": "newton",
                "topology": "loo",
                "nsa_w": 0.5,
                "first_layer_mode": "scheduled",
            }
        ),
        summary="Sparse linear first layer followed by nonlinear encoder heads and decoders.",
        when_to_prefer=(
            "Cross-view alignment itself requires nonlinear feature interactions beyond "
            "any linear subspace."
        ),
    ),
    PresetTemplate(
        name="nonlinear_shared_private",
        title="Shared-plus-Private Nonlinear Decomposition (NED++)",
        family="deep",
        complexity_tier=2,
        estimator="fit",
        method="ned_shared_private",
        static_params=_freeze(
            {
                "positivity": "either",
                "sparseness": 0.5,
                "energy_type": "align",
                "mixing_algorithm": "newton",
                "topology": "loo",
                "nsa_w": 0.5,
                "first_layer_mode": "scheduled",
            }
        ),
        summary="Nonlinear shared latents explicitly separated from structured view-private factors.",
        when_to_prefer=(
            "Views dominated by strong structured within-view variance orthogonal to the "
            "shared cross-view signal alongside nonlinear coupling."
        ),
    ),
    PresetTemplate(
        name="nonlinear_flow",
        title="Invertible Normalizing-Flow Alignment (Flow-SiMR-V)",
        family="flow",
        complexity_tier=2,
        estimator="fit",
        method="flow_v",
        static_params=_freeze(
            {
                "positivity": "either",
                "sparseness": 0.5,
                "energy_type": "align",
                "mixing_algorithm": "newton",
                "topology": "loo",
                "nsa_w": 0.5,
                "first_layer_mode": "scheduled",
            }
        ),
        summary="Sparse linear first layer coupled to per-view invertible normalizing flows.",
        when_to_prefer=(
            "Strongly skewed or heavy-tailed view marginals and monotone nonlinear warps "
            "where exact likelihood and conditional inference are useful."
        ),
    ),
)

PRESET_BY_NAME: Mapping[str, PresetTemplate] = _freeze({p.name: p for p in PRESET_CATALOG})


def list_presets() -> list[dict[str, Any]]:
    """Return metadata dictionaries for all registered modeling presets."""
    return [
        {
            "name": p.name,
            "title": p.title,
            "family": p.family,
            "complexity_tier": p.complexity_tier,
            "estimator": p.estimator,
            "method": p.method,
            "static_params": dict(p.static_params),
            "summary": p.summary,
            "when_to_prefer": p.when_to_prefer,
        }
        for p in PRESET_CATALOG
    ]


def get_preset(name: str) -> dict[str, Any]:
    """Return static definition of a named preset, or raise ``ValueError``."""
    if not isinstance(name, str) or name not in PRESET_BY_NAME:
        raise ValueError(
            f"unknown preset {name!r}; choose 'auto' or one of {list(PRESET_BY_NAME)}."
        )
    p = PRESET_BY_NAME[name]
    return {
        "name": p.name,
        "title": p.title,
        "family": p.family,
        "complexity_tier": p.complexity_tier,
        "estimator": p.estimator,
        "method": p.method,
        "static_params": dict(p.static_params),
        "summary": p.summary,
        "when_to_prefer": p.when_to_prefer,
    }


def _extract_views_with_meta(views: Any) -> tuple[list[str], list[np.ndarray], dict[str, Any]]:
    """Extract ``(view_names, float64_arrays, metadata)`` from raw views or ``prepare_views`` dicts."""
    meta: dict[str, Any] = {
        "is_prepared": False,
        "domain_signs_supplied": False,
        "embedded_fit_rows": None,
    }
    if isinstance(views, Mapping) and "views" in views:
        if isinstance(views.get("params"), Mapping) and isinstance(views["params"].get("views"), list):
            meta["is_prepared"] = True
            meta["embedded_fit_rows"] = views.get("fit_rows")
            vps = views["params"]["views"]
            meta["domain_signs_supplied"] = any(
                vp.get("signs") is not None for vp in vps if isinstance(vp, Mapping)
            )
            views = views["views"]
        elif "prepared" in views and "signs" in views:
            meta["is_prepared"] = True
            meta["embedded_fit_rows"] = views.get("rows")
            sg = views.get("signs")
            meta["domain_signs_supplied"] = bool(sg) if isinstance(sg, Mapping) else (sg is not None)
            views = views["views"]

    if isinstance(views, Mapping):
        names = [str(k) for k in views]
        mats = list(views.values())
    elif isinstance(views, (list, tuple)):
        mats = list(views)
        names = [f"view{i}" for i in range(len(mats))]
    else:
        mats = [views]
        names = ["view0"]
    if not mats:
        raise ValueError("views must contain at least one 2-D matrix.")
    out: list[np.ndarray] = []
    for nm, m in zip(names, mats):
        if isinstance(m, torch.Tensor):
            arr = m.detach().cpu().double().numpy()
        elif hasattr(m, "to_numpy"):
            arr = np.asarray(m.to_numpy(), dtype=np.float64)
        else:
            arr = np.asarray(m, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"view {nm!r} must be 2-D; got shape {arr.shape}.")
        out.append(arr)
    rows = {a.shape[0] for a in out}
    if len(rows) != 1:
        raise ValueError(f"views have mismatched row counts: {[a.shape[0] for a in out]}.")
    return names, out, meta


def _resolve_adjacency(
    m_views: int,
    names: Sequence[str],
    path_graph: Mapping[Any, Sequence[Any]] | None,
) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {}
    if path_graph is not None and m_views > 1:
        for key, nbrs in path_graph.items():
            idx_a = names.index(key) if isinstance(key, str) else int(key)
            parsed: list[int] = []
            for nb in nbrs:
                idx_b = names.index(nb) if isinstance(nb, str) else int(nb)
                if idx_b != idx_a:
                    parsed.append(idx_b)
            adj[idx_a] = parsed
    for i in range(m_views):
        if i not in adj or not adj[i]:
            adj[i] = [j for j in range(m_views) if j != i]
    return adj


def _thin_subspace(x_obs: np.ndarray, max_r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(U_r, evals_all, V_r)`` for centered/scaled ``x_obs`` in ``O(min(n^2 p, n p^2))``."""
    n_obs, p = x_obs.shape
    if n_obs < 2 or p == 0:
        return np.zeros((n_obs, 0)), np.zeros(0), np.zeros((p, 0))
    if p <= n_obs:
        gram = (x_obs.T @ x_obs) / float(max(n_obs - 1, 1))
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order]
        pos = evals > 1e-12
        evals, evecs = evals[pos], evecs[:, pos]
        r = min(max_r, evecs.shape[1])
        vr = evecs[:, :r]
        ur = x_obs @ vr
        nrm = np.linalg.norm(ur, axis=0, keepdims=True)
        ur = np.where(nrm > 1e-12, ur / np.maximum(nrm, 1e-12), 0.0)
        return ur, evals, vr
    gram = (x_obs @ x_obs.T) / float(max(n_obs - 1, 1))
    evals, uvecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    uvecs = uvecs[:, order]
    pos = evals > 1e-12
    evals, uvecs = evals[pos], uvecs[:, pos]
    r = min(max_r, uvecs.shape[1])
    ur = uvecs[:, :r]
    vr = x_obs.T @ ur
    nrm = np.linalg.norm(vr, axis=0, keepdims=True)
    vr = np.where(nrm > 1e-12, vr / np.maximum(nrm, 1e-12), 0.0)
    return ur, evals, vr


def jackknife_js_alpha(
    x_lobe: np.ndarray,
    t_vec: np.ndarray,
    *,
    allow_sign_flip: bool = True,
) -> float:
    r"""Exact Delete-1 Jackknife Positive-Part James-Stein uniform shrinkage :math:`\alpha \in [0, 1]`.

    Measures the delete-1 jackknife dispersion :math:`\hat{\sigma}_{\text{JK}}^2` of unit-norm
    non-negative correlation vectors :math:`\hat{v}^{(-i)} \propto (\xi \, X_{-i}^\top t_{-i})_+`
    relative to squared chordal distance :math:`D^2 = \|\hat{v} - m^{-1/2}\mathbf{1}_m\|_2^2`,
    returning :math:`\min(1, \hat{\sigma}_{\text{JK}}^2 / \max(D^2, 10^{-14}))` with zero magic
    clip constants.
    """
    p_c = int(x_lobe.shape[1])
    ok = np.isfinite(x_lobe).all(axis=1) & np.isfinite(t_vec)
    n = int(ok.sum())
    if p_c < 3 or n < 6:
        return 0.0
    xo = np.asarray(x_lobe[ok], dtype=np.float64)
    mu = np.mean(xo, axis=0, keepdims=True)
    sd = np.std(xo, axis=0, keepdims=True)
    xo = np.where(sd > SD_FLOOR, (xo - mu) / np.maximum(sd, SD_FLOOR), 0.0)
    to = np.asarray(t_vec[ok], dtype=np.float64)
    tc = to - float(np.mean(to))
    tn = float(np.linalg.norm(tc))
    if tn <= SD_FLOOR:
        return 1.0
    tu = tc / tn
    z_mat = tu[:, None] * xo
    g_full = np.sum(z_mat, axis=0)
    g_loo = g_full[None, :] - z_mat
    if allow_sign_flip:
        xi_f = (
            -1.0
            if float(np.linalg.norm(np.maximum(-g_full, 0.0)))
            > float(np.linalg.norm(np.maximum(g_full, 0.0)))
            else 1.0
        )
    else:
        xi_f = 1.0
    gf_pos = np.maximum(xi_f * g_full, 0.0)
    gn = float(np.linalg.norm(gf_pos))
    if gn <= SD_FLOOR:
        return 1.0
    vh = gf_pos / gn
    d2 = float(2.0 * max(0.0, 1.0 - float(np.sum(vh)) / math.sqrt(float(p_c))))
    if d2 <= 1e-14:
        return 1.0
    gp_i = np.maximum(xi_f * g_loo, 0.0)
    ni = np.linalg.norm(gp_i, axis=1, keepdims=True)
    vi = np.where(ni > 1e-15, gp_i / np.maximum(ni, 1e-15), 1.0 / math.sqrt(float(p_c)))
    v_bar = np.mean(vi, axis=0, keepdims=True)
    sig2_jk = float(((n - 1.0) / float(n)) * np.sum((vi - v_bar) ** 2))
    return float(np.clip(sig2_jk / d2, 0.0, 1.0))


def resolve_auto_sparseness(
    views: Any,
    k: int,
    *,
    positivity: str = "either",
    path_graph: Mapping[Any, Sequence[Any]] | None = None,
) -> list[float | None]:
    r"""Resolve per-view quantile ``sparseness`` via Paired Delete-1 Jackknife 1-SE parsimony.

    For each view :math:`m` and leave-one-view-out neighbor axis :math:`c \in \{1, \dots, k\}`,
    evaluates exact delete-1 out-of-sample scores :math:`s_i^{(-i)}(m_{\text{card}})` across a
    geometric cardinality grid, applies the dense-preferring paired 1-SE rule, and retains
    :math:`m_{\text{keep}} = \max_c m_{\text{1SE}}(c)` so no shared component is truncated.
    """
    names, mats, _ = _extract_views_with_meta(views)
    m_views = len(mats)
    n_rows = mats[0].shape[0]
    k_i = max(1, int(k))
    adj = _resolve_adjacency(m_views, names, path_graph)

    std_list: list[np.ndarray] = []
    obs_list: list[np.ndarray] = []
    sub_list: list[np.ndarray] = []
    for a in mats:
        om = np.isfinite(a).any(axis=1)
        obs_list.append(om)
        x = np.zeros_like(a, dtype=np.float64)
        if int(om.sum()) >= 4 and a.shape[1] >= 1:
            sub = np.nan_to_num(a[om], nan=0.0)
            mu = np.mean(sub, axis=0, keepdims=True)
            sd = np.std(sub, axis=0, keepdims=True)
            x[om] = np.where(sd > SD_FLOOR, (sub - mu) / np.maximum(sd, SD_FLOOR), 0.0)
            ur, ev, vr = _thin_subspace(x[om], max_r=min(max(k_i, 2), 6))
            uw = np.zeros((n_rows, ur.shape[1]), dtype=np.float64)
            if ur.shape[1] > 0:
                w_sv = np.sqrt(ev[: ur.shape[1]] / max(float(ev[0]), SD_FLOOR))
                sgn_v = np.where(np.sum(vr, axis=0) >= 0.0, 1.0, -1.0)
                uw[om] = ur * (w_sv * sgn_v)[None, :]
            sub_list.append(uw)
        else:
            sub_list.append(np.zeros((n_rows, 0), dtype=np.float64))
        std_list.append(x)

    signed = str(positivity).lower() in ("either", "project", "linear")
    out_q: list[float | None] = []
    for m_idx, (xm_full, om) in enumerate(zip(std_list, obs_list)):
        p_m = xm_full.shape[1]
        n_ob = int(om.sum())
        if p_m <= max(10, 2 * k_i) or n_ob < 8 or m_views < 2:
            out_q.append(None)
            continue
        nbrs = [j for j in adj[m_idx] if sub_list[j].shape[1] > 0]
        if not nbrs:
            out_q.append(None)
            continue
        nbr_stack = np.hstack([sub_list[j][om] for j in nbrs])
        ut, st, _ = np.linalg.svd(nbr_stack, full_matrices=False)
        r_cols = min(k_i, int(np.sum(st > 1e-10)))
        if r_cols == 0:
            out_q.append(None)
            continue

        xo = xm_full[om]
        if n_ob > 512:
            s_ix = np.linspace(0, n_ob - 1, num=512, dtype=int)
            xo = xo[s_ix]
            ut = ut[s_ix]
            n_ob = 512

        m_grid = np.unique(
            np.clip(
                np.round(np.geomspace(float(k_i), float(p_m), num=min(p_m, 32))).astype(int),
                k_i,
                p_m,
            )
        )
        if p_m not in m_grid:
            m_grid = np.append(m_grid, p_m)

        m_1se_cols: list[int] = []
        for c_i in range(r_cols):
            t_vec = ut[:, c_i]
            tc = t_vec - float(np.mean(t_vec))
            tn = float(np.linalg.norm(tc))
            if tn <= SD_FLOOR:
                continue
            tu = tc / tn
            z_mat = tu[:, None] * xo
            g_full = np.sum(z_mat, axis=0)
            g_loo = g_full[None, :] - z_mat

            if signed:
                mag = np.abs(g_loo)
                ord_i = np.argsort(-mag, axis=1)
                gs = np.take_along_axis(g_loo, ord_i, axis=1)
                xs = np.take_along_axis(xo, ord_i, axis=1)
            else:
                pos_n = np.linalg.norm(np.maximum(g_loo, 0.0), axis=1, keepdims=True)
                neg_n = np.linalg.norm(np.maximum(-g_loo, 0.0), axis=1, keepdims=True)
                xi = np.where(neg_n > pos_n, -1.0, 1.0)
                gp = np.maximum(xi * g_loo, 0.0)
                ord_i = np.argsort(-gp, axis=1)
                gs = np.take_along_axis(gp, ord_i, axis=1)
                xs = np.take_along_axis(xo * xi, ord_i, axis=1)

            s_all = np.cumsum(gs * xs, axis=1)[:, m_grid - 1] / np.sqrt(
                np.maximum(np.cumsum(gs * gs, axis=1)[:, m_grid - 1], 1e-30)
            )
            sc = s_all - np.mean(s_all, axis=0, keepdims=True)
            sn = np.maximum(np.linalg.norm(sc, axis=0), 1e-15)
            dots = sc.T @ tu
            rhos = np.abs(dots) / sn
            sgns = np.where(dots >= 0.0, 1.0, -1.0)
            j_best = int(np.argmax(rhos))
            if rhos[j_best] <= 1.645 / math.sqrt(max(n_ob - 3, 4)):
                continue
            s_unit = (sc / sn[None, :]) * sgns[None, :]
            p_contrib = float(n_ob) * (s_unit * tu[:, None])
            diff = p_contrib[:, [j_best]] - p_contrib
            se = np.std(diff, axis=0, ddof=1) / math.sqrt(float(n_ob))
            elig = np.flatnonzero((rhos[j_best] - rhos) <= se + 1e-15)
            j_1se = int(elig[-1]) if elig.size > 0 else j_best
            m_1se_cols.append(int(m_grid[j_1se]))

        if not m_1se_cols:
            out_q.append(None)
            continue
        m_keep = max(m_1se_cols)
        q_m = float(1.0 - m_keep / float(p_m))
        out_q.append(None if q_m <= 0.02 else round(min(q_m, 0.95), 4))
    return out_q


def _standardize_observed(
    arr: np.ndarray, active_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, bool, bool]:
    n, p = arr.shape
    finite = np.isfinite(arr)
    row_has_any = finite.any(axis=1)
    whole_missing = bool((~row_has_any & active_rows).any())
    isolated_nan = bool((~finite[row_has_any]).any()) if row_has_any.any() else False
    obs = active_rows & row_has_any
    n_obs = int(obs.sum())
    x_full = np.zeros((n, p), dtype=np.float64)
    if n_obs < 2 or p == 0:
        return x_full, obs, n_obs, whole_missing, isolated_nan
    sub = arr[obs].copy()
    if isolated_nan:
        med = np.nanmedian(sub, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        bad_r, bad_c = np.where(~np.isfinite(sub))
        sub[bad_r, bad_c] = med[bad_c]
    mu = sub.mean(axis=0, keepdims=True)
    sd = sub.std(axis=0, ddof=0, keepdims=True)
    sub_std = np.where(sd > SD_FLOOR, (sub - mu) / np.where(sd > SD_FLOOR, sd, 1.0), 0.0)
    x_full[obs] = sub_std
    return x_full, obs, n_obs, whole_missing, isolated_nan


def _normal_scores_1d(v: np.ndarray) -> np.ndarray:
    n = v.size
    if n < 3:
        return np.zeros_like(v)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64) / float(n + 1)
    x = 2.0 * ranks - 1.0
    a = 0.147
    ln = np.log(np.maximum(1.0 - x * x, 1e-15))
    term = 2.0 / (math.pi * a) + 0.5 * ln
    erfinv_approx = np.sign(x) * np.sqrt(np.maximum(np.sqrt(term * term - ln / a) - term, 0.0))
    z = math.sqrt(2.0) * erfinv_approx
    sd = np.std(z)
    return (z - np.mean(z)) / sd if sd > SD_FLOOR else np.zeros_like(v)


def _press_r2_per_row(phi: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact QR hat-matrix Leave-One-Out PRESS R^2 and per-row skill contributions."""
    n = phi.shape[0]
    yc = y - np.mean(y, axis=0, keepdims=True)
    if yc.ndim == 1:
        yc = yc[:, None]
    tot_var = np.sum(yc ** 2, axis=1)
    denom = float(np.sum(tot_var))
    if denom <= 1e-15 or n <= phi.shape[1] + 4:
        return 0.0, np.zeros(n, dtype=np.float64)
    design = np.column_stack(
        [np.ones(n, dtype=np.float64), phi - np.mean(phi, axis=0, keepdims=True)]
    )
    u_q, s_q, _ = np.linalg.svd(design, full_matrices=False)
    keep = s_q > 1e-8 * max(float(s_q[0]), 1.0)
    u_q = u_q[:, keep]
    y_hat = u_q @ (u_q.T @ yc)
    h_diag = np.sum(u_q ** 2, axis=1)
    one_minus_h = np.maximum(1.0 - h_diag, 2.0 / float(n))
    resid_loo = (yc - y_hat) / one_minus_h[:, None]
    err_loo = np.sum(resid_loo ** 2, axis=1)
    mean_null = denom / float(n)
    skill_row = (tot_var - err_loo) / mean_null
    r2_press = float(1.0 - np.sum(err_loo) / denom)
    return r2_press, skill_row


def _honest_oos_corr_skill(s_loo: np.ndarray, target_u: np.ndarray) -> tuple[float, np.ndarray]:
    """Sign-honest out-of-sample LOO signed R^2: ``sign(r_oos) * r_oos^2`` with per-row contributions."""
    n = target_u.size
    sc = s_loo - np.mean(s_loo)
    sn = float(np.linalg.norm(sc))
    if sn <= SD_FLOOR:
        return 0.0, np.zeros(n, dtype=np.float64)
    su = sc / sn
    r_oos = float(np.dot(su, target_u))
    signed_r2 = float(np.sign(r_oos) * (r_oos ** 2))
    row_contrib = float(n) * su * target_u
    skill_row = signed_r2 + 2.0 * abs(r_oos) * (row_contrib - r_oos)
    return signed_r2, skill_row


def _varimax_orth(phi: np.ndarray, *, max_iter: int = 20, tol: float = 1e-6) -> np.ndarray:
    """Fast orthogonal Varimax rotation on ``(n, k)`` scores (``k <= 4``)."""
    n, k = phi.shape
    if k <= 1 or n < 3:
        return phi
    rot = np.eye(k, dtype=np.float64)
    d_prev = 0.0
    for _ in range(max_iter):
        lam = phi @ rot
        u, s, vt = np.linalg.svd(
            phi.T @ (lam ** 3 - (lam @ np.diag(np.sum(lam ** 2, axis=0))) / float(n)),
            full_matrices=False,
        )
        rot = u @ vt
        d_cur = float(np.sum(s))
        if d_prev > 0.0 and abs(d_cur - d_prev) / d_prev < tol:
            break
        d_prev = d_cur
    out = phi @ rot
    skew = np.sum(out ** 3, axis=0)
    sgn = np.where(skew >= 0.0, 1.0, -1.0)
    out = out * sgn[None, :]
    out -= np.mean(out, axis=0, keepdims=True)
    nrm = np.linalg.norm(out, axis=0, keepdims=True)
    return np.where(nrm > SD_FLOOR, out / np.maximum(nrm, SD_FLOOR), out)


def _select_1se_idx(s_mat: np.ndarray, tu_sub: np.ndarray) -> int:
    """Select Paired-1SE dense-preferring column index on subset rows."""
    n_s = tu_sub.size
    if n_s < 4:
        return s_mat.shape[1] - 1
    sc = s_mat - np.mean(s_mat, axis=0, keepdims=True)
    sn = np.maximum(np.linalg.norm(sc, axis=0), 1e-15)
    tu_c = tu_sub - float(np.mean(tu_sub))
    tu_n = float(np.linalg.norm(tu_c))
    if tu_n <= SD_FLOOR:
        return s_mat.shape[1] - 1
    tu_u = tu_c / tu_n
    corrs = (sc.T @ tu_u) / sn
    j_best = int(np.argmax(corrs))
    if corrs[j_best] <= 0.0:
        return s_mat.shape[1] - 1
    u_best = sc[:, j_best] / sn[j_best]
    prod_best = float(n_s) * u_best * tu_u
    for j_cand in range(s_mat.shape[1] - 1, j_best, -1):
        u_c = sc[:, j_cand] / sn[j_cand]
        diff = prod_best - float(n_s) * u_c * tu_u
        se = float(np.std(diff, ddof=1) / math.sqrt(n_s))
        if float(np.mean(diff)) <= se:
            return j_cand
    return j_best


def _delete1_linear_geometries(
    x_obs: np.ndarray,
    t_obs: np.ndarray,
    *,
    max_probe_rows: int = 256,
    max_probe_cols: int = 384,
) -> dict[str, Any]:
    """Exact delete-1 LOO cross-view signed R^2 across linear geometries on shared target ``t_obs``."""
    n_full, p_full = x_obs.shape
    if t_obs.ndim == 1:
        t_obs = t_obs[:, None]

    if n_full > max_probe_rows:
        stride_idx = np.linspace(0, n_full - 1, num=max_probe_rows, dtype=int)
        x_use = x_obs[stride_idx]
        ts_use = t_obs[stride_idx]
    else:
        x_use = x_obs
        ts_use = t_obs

    n, p = x_use.shape
    if n < 6 or p == 0:
        z_row = np.zeros(max(n, 1), dtype=np.float64)
        return {
            "signed_dense_r2": 0.0,
            "signed_dense_row": z_row,
            "signed_1se_r2": 0.0,
            "signed_1se_row": z_row,
            "q_signed_1se": 0.5,
            "signed_focal_r2": 0.0,
            "signed_focal_row": z_row,
            "q_signed_focal": 0.75,
            "dir_co_shrunk_r2": 0.0,
            "dir_co_shrunk_row": z_row,
            "dir_abs_shrunk_r2": 0.0,
            "dir_abs_shrunk_row": z_row,
            "dir_parts_r2": 0.0,
            "dir_parts_row": z_row,
            "alpha_jk": 0.0,
            "hoyer_localization": 0.0,
        }

    mc = ts_use - np.mean(ts_use, axis=0, keepdims=True)
    q_mat, r_mat = np.linalg.qr(mc)
    diag_r = np.abs(np.diag(r_mat)) if r_mat.ndim == 2 and min(r_mat.shape) > 0 else np.zeros(0)
    ok = diag_r > 1e-10
    if not np.any(ok):
        z_row = np.zeros(n, dtype=np.float64)
        return {
            "signed_dense_r2": 0.0,
            "signed_dense_row": z_row,
            "signed_1se_r2": 0.0,
            "signed_1se_row": z_row,
            "q_signed_1se": 0.5,
            "signed_focal_r2": 0.0,
            "signed_focal_row": z_row,
            "q_signed_focal": 0.75,
            "dir_co_shrunk_r2": 0.0,
            "dir_co_shrunk_row": z_row,
            "dir_abs_shrunk_r2": 0.0,
            "dir_abs_shrunk_row": z_row,
            "dir_parts_r2": 0.0,
            "dir_parts_row": z_row,
            "alpha_jk": 0.0,
            "hoyer_localization": 0.0,
        }

    tu_s_cols = q_mat[:, ok].copy()
    mc_ok = mc[:, ok]
    for j in range(tu_s_cols.shape[1]):
        if float(np.dot(tu_s_cols[:, j], mc_ok[:, j])) < 0.0:
            tu_s_cols[:, j] = -tu_s_cols[:, j]
        tu_s_cols[:, j] -= np.mean(tu_s_cols[:, j])
        nrm_j = float(np.linalg.norm(tu_s_cols[:, j]))
        if nrm_j > SD_FLOOR:
            tu_s_cols[:, j] /= nrm_j
    tu_s_cols = tu_s_cols[:, : min(tu_s_cols.shape[1], 4)]
    tu_c_cols = _varimax_orth(tu_s_cols) if tu_s_cols.shape[1] >= 2 else tu_s_cols

    if p_full > max_probe_cols:
        col_var = np.sum(x_use ** 2, axis=0)
        keep_cols = np.argsort(col_var)[::-1][:max_probe_cols]
        x_use = x_use[:, keep_cols]
        p = x_use.shape[1]

    m_grid = np.unique(
        np.clip(np.round(np.geomspace(1.0, float(p), num=min(p, 20))).astype(int), 1, p)
    )
    if p not in m_grid:
        m_grid = np.append(m_grid, p)

    acc_sd_row = np.zeros(n, dtype=np.float64)
    acc_s1se_row = np.zeros(n, dtype=np.float64)
    acc_sfoc_row = np.zeros(n, dtype=np.float64)
    acc_dco_row = np.zeros(n, dtype=np.float64)
    acc_dabs_row = np.zeros(n, dtype=np.float64)
    acc_dparts_row = np.zeros(n, dtype=np.float64)
    q_1se_list: list[float] = []
    q_foc_list: list[float] = []
    alpha_jk_list: list[float] = []
    hoyer_list: list[float] = []

    even_idx = np.arange(0, n, 2)
    odd_idx = np.arange(1, n, 2)
    focal_low = max(2, round(0.10 * p))
    focal_cap = max(focal_low, round(0.30 * p))
    focal_idxs = np.flatnonzero((m_grid >= focal_low) & (m_grid <= focal_cap))
    if focal_idxs.size == 0:
        focal_idxs = np.flatnonzero(m_grid <= focal_cap)
    if focal_idxs.size == 0:
        focal_idxs = np.array([0], dtype=int)

    k_denom = float(tu_s_cols.shape[1])
    ks_eff = tu_s_cols.shape[1]
    for c_idx in range(ks_eff):
        tu = tu_s_cols[:, c_idx]
        z_mat = tu[:, None] * x_use
        g_full = np.sum(z_mat, axis=0)
        gn_full = float(np.linalg.norm(g_full))
        if gn_full > SD_FLOOR and p > 1:
            l1_norm = float(np.sum(np.abs(g_full)) / gn_full)
            hoyer = float((math.sqrt(p) - l1_norm) / (math.sqrt(p) - 1.0))
        else:
            hoyer = 0.0
        hoyer_list.append(float(np.clip(hoyer, 0.0, 1.0)))

        g_loo = g_full[None, :] - z_mat
        order_s = np.argsort(-np.abs(g_loo), axis=1)
        gs = np.take_along_axis(g_loo, order_s, axis=1)
        xs = np.take_along_axis(x_use, order_s, axis=1)
        cum_gx = np.cumsum(gs * xs, axis=1)
        cum_g2 = np.sqrt(np.maximum(np.cumsum(gs * gs, axis=1), 1e-30))
        s_signed_all = cum_gx[:, m_grid - 1] / cum_g2[:, m_grid - 1]

        _, row_sd = _honest_oos_corr_skill(s_signed_all[:, -1], tu)

        j_1se_full = _select_1se_idx(s_signed_all, tu)
        m_1se = int(m_grid[j_1se_full])
        if c_idx == 0:
            q_1se_list.append(float(1.0 - m_1se / float(p)))

        if even_idx.size >= 4 and odd_idx.size >= 4:
            s_sc_all = s_signed_all - np.mean(s_signed_all, axis=0, keepdims=True)
            s_unit_all = s_sc_all / np.maximum(
                np.linalg.norm(s_sc_all, axis=0, keepdims=True), 1e-15
            )
            j_1se_ev = _select_1se_idx(s_unit_all[even_idx], tu[even_idx])
            j_1se_od = _select_1se_idx(s_unit_all[odd_idx], tu[odd_idx])
            s_1se_cf = np.empty(n, dtype=np.float64)
            s_1se_cf[odd_idx] = s_unit_all[odd_idx, j_1se_ev]
            s_1se_cf[even_idx] = s_unit_all[even_idx, j_1se_od]

            jf_ev = int(
                focal_idxs[int(np.argmax(s_unit_all[even_idx][:, focal_idxs].T @ tu[even_idx]))]
            )
            jf_od = int(
                focal_idxs[int(np.argmax(s_unit_all[odd_idx][:, focal_idxs].T @ tu[odd_idx]))]
            )
            s_foc_cf = np.empty(n, dtype=np.float64)
            s_foc_cf[odd_idx] = s_unit_all[odd_idx, jf_ev]
            s_foc_cf[even_idx] = s_unit_all[even_idx, jf_od]
            j_foc_rep = jf_ev
        else:
            s_1se_cf = s_signed_all[:, j_1se_full]
            j_foc_rep = int(focal_idxs[0])
            s_foc_cf = s_signed_all[:, j_foc_rep]

        if c_idx == 0:
            q_foc_list.append(float(np.clip(1.0 - int(m_grid[j_foc_rep]) / float(p), 0.65, 0.85)))
        _, row_s1se = _honest_oos_corr_skill(s_1se_cf, tu)
        _, row_sfoc = _honest_oos_corr_skill(s_foc_cf, tu)

        acc_sd_row += row_sd / k_denom
        acc_s1se_row += row_s1se / k_denom
        acc_sfoc_row += row_sfoc / k_denom

    kc_eff = tu_c_cols.shape[1]
    for c_idx in range(kc_eff):
        tuc = tu_c_cols[:, c_idx]
        zc = tuc[:, None] * x_use
        gc_full = np.sum(zc, axis=0)
        gc_loo = gc_full[None, :] - zc

        pos_m_i = np.sum(np.maximum(gc_loo, 0.0) ** 2, axis=1, keepdims=True)
        neg_m_i = np.sum(np.maximum(-gc_loo, 0.0) ** 2, axis=1, keepdims=True)
        sgn_i = np.where(pos_m_i >= neg_m_i, 1.0, -1.0)

        g_abs = np.maximum(sgn_i * gc_loo, 0.0)
        v1_abs = g_abs / np.maximum(np.linalg.norm(g_abs, axis=1, keepdims=True), 1e-30)

        dom_full = (
            1.0
            if float(np.sum(np.maximum(gc_full, 0.0) ** 2))
            >= float(np.sum(np.maximum(-gc_full, 0.0) ** 2))
            else -1.0
        )
        g_dom = np.maximum(dom_full * gc_full, 0.0)
        gn_dom = float(np.linalg.norm(g_dom))
        if gn_dom > SD_FLOOR and p >= 2:
            vh = g_dom / gn_dom
            d2 = float(2.0 * max(0.0, 1.0 - np.sum(vh) / math.sqrt(p)))
            v_dom_aligned = np.maximum(dom_full * gc_loo, 0.0)
            v_dom_aligned /= np.maximum(
                np.linalg.norm(v_dom_aligned, axis=1, keepdims=True), 1e-30
            )
            v_bar = np.mean(v_dom_aligned, axis=0, keepdims=True)
            sig2_jk = float(((n - 1.0) / float(n)) * np.sum((v_dom_aligned - v_bar) ** 2))
            alpha_jk = float(min(1.0, sig2_jk / max(d2, 1e-15)))
        else:
            alpha_jk = 0.0
        alpha_jk_list.append(alpha_jk)

        g_co = np.maximum(gc_loo, 0.0)
        v1_co = g_co / np.maximum(np.linalg.norm(g_co, axis=1, keepdims=True), 1e-30)
        v_co_shr = (1.0 - alpha_jk) * v1_co + alpha_jk * (1.0 / math.sqrt(p))
        v_co_shr /= np.maximum(np.linalg.norm(v_co_shr, axis=1, keepdims=True), 1e-30)
        s_co = np.sum(x_use * v_co_shr, axis=1)
        _, row_dco = _honest_oos_corr_skill(s_co, tuc)

        v_abs_shr = (1.0 - alpha_jk) * v1_abs + alpha_jk * (1.0 / math.sqrt(p))
        v_abs_shr /= np.maximum(np.linalg.norm(v_abs_shr, axis=1, keepdims=True), 1e-30)
        s_abs = np.sum(x_use * v_abs_shr, axis=1) * sgn_i.ravel()
        _, row_dabs = _honest_oos_corr_skill(s_abs, tuc)

        g_base_part = g_co if float(np.mean(row_dco)) >= float(np.mean(row_dabs)) - 0.01 else g_abs
        sgn_part = np.ones((n, 1), dtype=np.float64) if g_base_part is g_co else sgn_i
        m_part = max(1, p // 2)
        part_idx = np.argpartition(g_base_part, -m_part, axis=1)[:, -m_part:]
        pmask = np.zeros_like(g_base_part, dtype=bool)
        np.put_along_axis(pmask, part_idx, True, axis=1)
        gp = np.where(pmask, g_base_part, 0.0)
        s_parts = (
            np.sum(
                x_use * (gp / np.maximum(np.linalg.norm(gp, axis=1, keepdims=True), 1e-30)),
                axis=1,
            )
            * sgn_part.ravel()
        )
        _, row_dparts = _honest_oos_corr_skill(s_parts, tuc)

        acc_dco_row += row_dco / k_denom
        acc_dabs_row += row_dabs / k_denom
        acc_dparts_row += row_dparts / k_denom

    return {
        "signed_dense_r2": float(np.mean(acc_sd_row)),
        "signed_dense_row": acc_sd_row,
        "signed_1se_r2": float(np.mean(acc_s1se_row)),
        "signed_1se_row": acc_s1se_row,
        "q_signed_1se": float(np.median(q_1se_list)) if q_1se_list else 0.5,
        "signed_focal_r2": float(np.mean(acc_sfoc_row)),
        "signed_focal_row": acc_sfoc_row,
        "q_signed_focal": float(np.median(q_foc_list)) if q_foc_list else 0.75,
        "dir_co_shrunk_r2": float(np.mean(acc_dco_row)),
        "dir_co_shrunk_row": acc_dco_row,
        "dir_abs_shrunk_r2": float(np.mean(acc_dabs_row)),
        "dir_abs_shrunk_row": acc_dabs_row,
        "dir_parts_r2": float(np.mean(acc_dparts_row)),
        "dir_parts_row": acc_dparts_row,
        "alpha_jk": float(np.mean(alpha_jk_list)),
        "hoyer_localization": float(np.mean(hoyer_list)),
    }


def _wachter_gcca_null_edge(m_views: int, r_dims: Sequence[int], n_eff: int) -> float:
    """Wachter / Marchenko-Pastur upper spectral edge (+ Tracy-Widom margin) for M random projectors."""
    if m_views < 2 or n_eff < 4:
        return 1.0
    r_bar = float(np.mean([max(r, 1) for r in r_dims]))
    f = min(0.49, r_bar / float(n_eff))
    edge = 1.0 + (m_views - 2.0) * f + 2.0 * math.sqrt(max((m_views - 1.0) * f * (1.0 - f), 0.0))
    tw_margin = 2.5 * (float(n_eff) ** (-2.0 / 3.0)) * float(m_views)
    return float(edge + tw_margin)


def analyze_views(
    views: Any,
    *,
    k: int | str | None = None,
    fit_rows: Any = None,
    path_graph: Mapping[int | str, Sequence[int | str]] | None = None,
    directional: bool | None = None,
) -> dict[str, Any]:
    """Run an unsupervised statistical profile of a multi-view data collection."""
    t0 = time.perf_counter()
    names, raw_mats, meta = _extract_views_with_meta(views)
    n_rows = raw_mats[0].shape[0]
    m_views = len(raw_mats)

    eff_fit_rows = fit_rows if fit_rows is not None else meta["embedded_fit_rows"]
    if eff_fit_rows is None:
        active = np.ones(n_rows, dtype=bool)
    else:
        fr = np.asarray(eff_fit_rows)
        if fr.dtype == bool:
            active = fr.copy()
        else:
            active = np.zeros(n_rows, dtype=bool)
            active[fr] = True
    domain_signs = (
        bool(directional) if directional is not None else bool(meta["domain_signs_supplied"])
    )
    n_fit = int(active.sum())
    if n_fit < 6:
        raise ValueError(f"analyze_views requires at least 6 active rows; got {n_fit}.")

    std_mats: list[np.ndarray] = []
    obs_masks: list[np.ndarray] = []
    n_obs_list: list[int] = []
    whole_miss_views: list[str] = []
    iso_nan_views: list[str] = []
    widths = [int(a.shape[1]) for a in raw_mats]

    for nm, a in zip(names, raw_mats):
        xs, om, n_ob, wm, inan = _standardize_observed(a, active)
        std_mats.append(xs)
        obs_masks.append(om)
        n_obs_list.append(n_ob)
        if wm:
            whole_miss_views.append(nm)
        if inan:
            iso_nan_views.append(nm)

    complete_mask = active.copy()
    for om in obs_masks:
        complete_mask &= om
    n_complete = int(complete_mask.sum())

    adj = _resolve_adjacency(m_views, names, path_graph)

    eff_ranks: list[float] = []
    pos_manifold: list[float] = []
    non_gaussianity: list[float] = []
    subspaces_full: list[np.ndarray] = []
    weighted_subspaces_full: list[np.ndarray] = []

    max_sub = min(8, max(2, n_fit // 8))
    for xs, om, n_ob, p in zip(std_mats, obs_masks, n_obs_list, widths):
        ur, ev, vr = _thin_subspace(xs[om], max_r=max_sub)
        if ev.size > 0 and float(np.sum(ev ** 2)) > 1e-15:
            er = float((np.sum(ev) ** 2) / np.sum(ev ** 2))
        else:
            er = 1.0
        eff_ranks.append(er)
        if vr.shape[1] > 0:
            w1 = vr[:, 0]
            pos_e = float(np.sum(np.maximum(w1, 0.0) ** 2))
            neg_e = float(np.sum(np.maximum(-w1, 0.0) ** 2))
            pos_manifold.append(max(pos_e, neg_e))
        else:
            pos_manifold.append(0.5)
        sub = xs[om][:, : min(p, 64)]
        if sub.size > 0 and n_ob >= 8:
            skew2 = float(np.mean(np.mean(sub ** 3, axis=0) ** 2))
            kurt_ex = float(np.mean(np.maximum(np.abs(np.mean(sub ** 4, axis=0) - 3.0), 0.0)))
            non_gaussianity.append(float(math.sqrt(skew2) + 0.25 * kurt_ex))
        else:
            non_gaussianity.append(0.0)
        u_full = np.zeros((n_rows, ur.shape[1]), dtype=np.float64)
        u_full[om] = ur
        subspaces_full.append(u_full)

        uw_full = np.zeros((n_rows, ur.shape[1]), dtype=np.float64)
        if ur.shape[1] > 0:
            ev_r = ev[: ur.shape[1]]
            w_sv = np.sqrt(ev_r / max(float(ev_r[0]), SD_FLOOR))
            sgn_v = np.where(np.sum(vr, axis=0) >= 0.0, 1.0, -1.0)
            uw_full[om] = ur * (w_sv * sgn_v)[None, :]
        weighted_subspaces_full.append(uw_full)

    p_min = min(widths)
    r_dims = [u.shape[1] for u in subspaces_full if u.shape[1] > 0]
    n_eff_mp = max(n_complete if n_complete >= 12 else int(np.median(n_obs_list)), 12)
    if m_views >= 2 and r_dims:
        stacked = np.hstack([u[active] for u in subspaces_full if u.shape[1] > 0])
        _, s_gcca, _ = np.linalg.svd(stacked, full_matrices=False)
        lam_gcca = s_gcca ** 2
        mp_edge = _wachter_gcca_null_edge(m_views, r_dims, n_eff_mp)
        k_mp = int(np.sum(lam_gcca > mp_edge))
        k_rec = int(np.clip(max(1, k_mp), 1, min(p_min, 8)))
    else:
        lam_gcca = np.ones(1, dtype=np.float64)
        mp_edge = 1.0
        k_rec = 1

    if k is None or k == "auto":
        k_used = k_rec
    else:
        k_used = int(k)
        if k_used < 1:
            raise ValueError(f"k must be >= 1 or 'auto'; got {k!r}.")
    k_eval = min(k_used, max(p_min, 1))

    n_probe = min(n_fit, 256)
    view_geoms: list[dict[str, Any]] = []
    press_lend_vecs: list[np.ndarray] = []
    press_ned_vecs: list[np.ndarray] = []
    press_flow_vecs: list[np.ndarray] = []
    priv_fracs: list[float] = []

    if m_views == 1:
        x0 = std_mats[0][obs_masks[0]]
        p0 = x0.shape[1]
        half = max(1, p0 // 2)
        u_a, _, _ = _thin_subspace(x0[:, :half], max_r=min(k_eval, half))
        u_b, _, _ = _thin_subspace(
            x0[:, half:] if p0 > 1 else x0, max_r=min(k_eval, max(1, p0 - half))
        )
        t_single = u_b if u_b.shape[1] > 0 else u_a
        geom = _delete1_linear_geometries(x0[:, :half], t_single, max_probe_rows=n_probe)
        view_geoms.append(geom)
        press_lend_vecs.append(geom["signed_dense_row"] - 0.02)
        press_ned_vecs.append(geom["signed_dense_row"] - 0.02)
        press_flow_vecs.append(geom["signed_dense_row"] - 0.02)
        priv_fracs.append(0.0)
    else:
        for m_idx in range(m_views):
            om = obs_masks[m_idx]
            xm_obs = std_mats[m_idx][om]
            nbrs = [j for j in adj[m_idx] if weighted_subspaces_full[j].shape[1] > 0]
            if not nbrs:
                t_other = np.zeros((xm_obs.shape[0], 1), dtype=np.float64)
            else:
                nbr_stack = np.hstack([weighted_subspaces_full[j][om] for j in nbrs])
                ut, st, _ = np.linalg.svd(nbr_stack, full_matrices=False)
                r_keep = min(k_eval, int(np.sum(st > 1e-10)))
                t_other = ut[:, : max(1, r_keep)].copy()
                nbr_row_sum = np.sum(
                    np.column_stack([np.mean(std_mats[j][om], axis=1) for j in nbrs]),
                    axis=1,
                )
                for c_i in range(t_other.shape[1]):
                    if float(np.dot(t_other[:, c_i], nbr_row_sum)) < 0.0:
                        t_other[:, c_i] = -t_other[:, c_i]

            geom = _delete1_linear_geometries(xm_obs, t_other, max_probe_rows=n_probe)
            view_geoms.append(geom)
            n_g = geom["signed_dense_row"].size
            if xm_obs.shape[0] > n_g:
                s_idx = np.linspace(0, xm_obs.shape[0] - 1, num=n_g, dtype=int)
                lm = subspaces_full[m_idx][om][s_idx, : min(4, subspaces_full[m_idx].shape[1])]
                xm_sub = xm_obs[s_idx, : min(xm_obs.shape[1], 64)]
                t_sub = t_other[s_idx]
                nbr_lm = (
                    np.hstack(
                        [subspaces_full[j][om][s_idx, : min(3, subspaces_full[j].shape[1])] for j in nbrs]
                    )
                    if nbrs
                    else t_sub
                )
            else:
                lm = subspaces_full[m_idx][om][:, : min(4, subspaces_full[m_idx].shape[1])]
                xm_sub = xm_obs[:, : min(xm_obs.shape[1], 64)]
                t_sub = t_other
                nbr_lm = (
                    np.hstack(
                        [subspaces_full[j][om][:, : min(3, subspaces_full[j].shape[1])] for j in nbrs]
                    )
                    if nbrs
                    else t_sub
                )

            um_k = lm[:, : min(k_eval, lm.shape[1])]
            if um_k.shape[1] > 0 and t_sub.shape[1] > 0:
                tc = t_sub - np.mean(t_sub, axis=0, keepdims=True)
                qt, _ = np.linalg.qr(tc)
                shared_lin_e = float(np.sum((qt.T @ um_k) ** 2) / float(um_k.shape[1]))
            else:
                shared_lin_e = 0.5

            best_lin_candidates = [
                geom["signed_1se_row"],
                geom["signed_dense_row"],
                geom["dir_co_shrunk_row"],
                geom["dir_abs_shrunk_row"],
            ]
            best_lin_row = max(best_lin_candidates, key=lambda v: float(np.mean(v)))

            if lm.shape[1] > 0 and n_g >= 24:
                lm_s = lm - np.mean(lm, axis=0, keepdims=True)
                lm_sd = np.std(lm_s, axis=0, keepdims=True)
                lm_s = np.where(lm_sd > SD_FLOOR, lm_s / np.maximum(lm_sd, SD_FLOOR), 0.0)

                nbr_s = nbr_lm - np.mean(nbr_lm, axis=0, keepdims=True)
                nbr_sd = np.std(nbr_s, axis=0, keepdims=True)
                nbr_s = np.where(nbr_sd > SD_FLOOR, nbr_s / np.maximum(nbr_sd, SD_FLOOR), 0.0)

                ts_c = t_sub - np.mean(t_sub, axis=0, keepdims=True)
                ts_sd = np.std(ts_c, axis=0, keepdims=True)
                ts_c = np.where(ts_sd > SD_FLOOR, ts_c / np.maximum(ts_sd, SD_FLOOR), 0.0)

                nbr_c = nbr_s[:, : min(3, nbr_s.shape[1])]
                r2_lin_fwd, row_lin_sub = _press_r2_per_row(lm_s, nbr_c)
                base_fwd_row = row_lin_sub - min(r2_lin_fwd, 0.0)
                quad_a = [lm_s, (lm_s ** 2 - 1.0) / math.sqrt(2.0)]
                if lm_s.shape[1] >= 2:
                    quad_a.append((lm_s[:, 0] * lm_s[:, 1])[:, None])
                phi_ned_a = np.hstack(quad_a)
                r2_ned_fwd, row_ned_fwd = _press_r2_per_row(phi_ned_a, nbr_c)

                phi_ned_b = np.hstack([nbr_c, (nbr_c ** 2 - 1.0) / math.sqrt(2.0)])
                r2_lin_rev, row_lin_rev = _press_r2_per_row(nbr_c, lm_s)
                base_rev_row = row_lin_rev - min(r2_lin_rev, 0.0)
                r2_ned_rev, row_ned_rev = _press_r2_per_row(phi_ned_b, lm_s)

                gain_fwd = r2_ned_fwd - max(r2_lin_fwd, 0.0)
                gain_rev = r2_ned_rev - max(r2_lin_rev, 0.0)
                if gain_fwd >= gain_rev:
                    ned_delta = (
                        (row_ned_fwd - base_fwd_row)
                        if r2_ned_fwd > max(r2_lin_fwd, 0.0)
                        else np.full(n_g, -0.02)
                    )
                else:
                    ned_delta = (
                        (row_ned_rev - base_rev_row)
                        if r2_ned_rev > max(r2_lin_rev, 0.0)
                        else np.full(n_g, -0.02)
                    )
                press_ned_vecs.append(best_lin_row + ned_delta)

                r2_dec_lin, row_dec_lin = _press_r2_per_row(ts_c, lm_s)
                base_dec_row = row_dec_lin - min(r2_dec_lin, 0.0)
                phi_dec_nl = np.hstack([ts_c, (ts_c ** 2 - 1.0) / math.sqrt(2.0)])
                r2_dec_nl, row_dec_nl = _press_r2_per_row(phi_dec_nl, lm_s)
                lend_delta = (
                    (row_dec_nl - base_dec_row)
                    if r2_dec_nl > max(r2_dec_lin, 0.0)
                    else np.full(n_g, -0.02)
                )
                press_lend_vecs.append(best_lin_row + lend_delta)

                xm_g = np.column_stack(
                    [_normal_scores_1d(xm_sub[:, j]) for j in range(xm_sub.shape[1])]
                )
                ug_r, _, _ = _thin_subspace(xm_g, max_r=min(4, xm_g.shape[1]))
                nbr_g = np.column_stack(
                    [_normal_scores_1d(nbr_c[:, j]) for j in range(nbr_c.shape[1])]
                )
                lm_g = np.column_stack(
                    [_normal_scores_1d(lm_s[:, j]) for j in range(lm_s.shape[1])]
                )
                r2_flow_lm, row_flow_lm = _press_r2_per_row(lm_g, nbr_g)
                if ug_r.shape[1] > 0:
                    ug_s = (ug_r - np.mean(ug_r, axis=0, keepdims=True)) / np.maximum(
                        np.std(ug_r, axis=0, keepdims=True), SD_FLOOR
                    )
                    r2_flow_g, row_flow_g = _press_r2_per_row(ug_s, nbr_g)
                    if r2_flow_lm > r2_flow_g:
                        r2_flow_best, row_flow_best = r2_flow_lm, row_flow_lm
                    else:
                        r2_flow_best, row_flow_best = r2_flow_g, row_flow_g
                else:
                    r2_flow_best, row_flow_best = r2_flow_lm, row_flow_lm
                flow_delta = (
                    (row_flow_best - base_fwd_row)
                    if r2_flow_best > max(r2_lin_fwd, 0.0)
                    else np.full(n_g, -0.02)
                )
                press_flow_vecs.append(best_lin_row + flow_delta)

                um_top = lm_s[:, : min(2, lm_s.shape[1])]
                r2_top_nl, _ = _press_r2_per_row(phi_ned_b, um_top)
                total_coupled = max(shared_lin_e, r2_top_nl, 0.0)
                priv_fracs.append(float(np.clip(1.0 - total_coupled, 0.0, 1.0)))
            else:
                press_ned_vecs.append(best_lin_row - 0.02)
                press_lend_vecs.append(best_lin_row - 0.02)
                press_flow_vecs.append(best_lin_row - 0.02)
                priv_fracs.append(float(np.clip(1.0 - shared_lin_e, 0.0, 1.0)))

    len_common = min(n_fit, 256)

    def _align_len(vecs: list[np.ndarray]) -> np.ndarray:
        aligned = []
        for v in vecs:
            if v.size == len_common:
                aligned.append(v)
            elif v.size > 1:
                xp = np.linspace(0.0, 1.0, num=v.size)
                xq = np.linspace(0.0, 1.0, num=len_common)
                aligned.append(np.interp(xq, xp, v))
            else:
                aligned.append(np.zeros(len_common, dtype=np.float64))
        return np.mean(np.vstack(aligned), axis=0)

    row_dense = _align_len([g["signed_dense_row"] for g in view_geoms])
    row_gen_sparse = _align_len([g["signed_1se_row"] for g in view_geoms])
    row_focal = _align_len([g["signed_focal_row"] for g in view_geoms])
    row_dir_co = _align_len([g["dir_co_shrunk_row"] for g in view_geoms])
    row_dir_abs = _align_len([g["dir_abs_shrunk_row"] for g in view_geoms])
    row_dir_parts = _align_len([g["dir_parts_row"] for g in view_geoms])
    row_lend = _align_len(press_lend_vecs)
    row_ned = _align_len(press_ned_vecs)
    row_flow = _align_len(press_flow_vecs)

    co_r2 = float(np.mean(row_dir_co))
    abs_r2 = float(np.mean(row_dir_abs))
    diff_co_abs = row_dir_abs - row_dir_co
    se_co_abs = float(np.std(diff_co_abs, ddof=1) / math.sqrt(max(len_common, 2)))
    co_directional = bool(co_r2 > 0.0 and float(np.mean(diff_co_abs)) <= max(se_co_abs, 0.005))
    row_dir_best = row_dir_co if co_directional else row_dir_abs

    best_lin_r2 = float(
        max(
            np.mean(row_dense),
            np.mean(row_gen_sparse),
            np.mean(row_focal),
            np.mean(row_dir_best),
            np.mean(row_dir_parts),
        )
    )
    null_r2_bar = float((2.576 ** 2) / max(n_fit - 3, 4))
    signal_detected = bool(
        best_lin_r2 > null_r2_bar
        or float(np.mean(row_ned)) > 1.5 * null_r2_bar
        or float(np.mean(row_flow)) > 1.5 * null_r2_bar
    )

    priv_mean = float(np.mean(priv_fracs))
    base_lin_vec = max(
        (row_gen_sparse, row_dense, row_dir_best),
        key=lambda v: float(np.mean(v)),
    )
    nonlin_ned_gain = float(np.mean(row_ned - base_lin_vec))
    nonlin_ned_se = float(
        np.std(row_ned - base_lin_vec, ddof=1) / math.sqrt(max(len_common, 2))
    )
    nonlin_lend_gain = float(np.mean(row_lend - base_lin_vec))
    nonlin_flow_gain = float(np.mean(row_flow - base_lin_vec))

    sp_factor = (priv_mean - 0.60) if nonlin_ned_gain > 0.0 else -0.02
    row_ned_sp = row_ned + sp_factor * max(nonlin_ned_gain, 0.05)

    aspect_ratios = [round(p / float(max(n_ob, 1)), 4) for p, n_ob in zip(widths, n_obs_list)]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "n_rows": n_rows,
        "n_fit_rows": n_fit,
        "n_complete_rows": n_complete,
        "n_views": m_views,
        "view_names": names,
        "view_widths": widths,
        "n_observed_per_view": n_obs_list,
        "is_prepared": bool(meta["is_prepared"]),
        "domain_signs_supplied": domain_signs,
        "has_whole_view_missing_rows": bool(whole_miss_views),
        "whole_missing_views": whole_miss_views,
        "has_isolated_nans": bool(iso_nan_views),
        "width_imbalance": round(float(max(widths) / max(min(widths), 1)), 3),
        "aspect_ratios": aspect_ratios,
        "max_aspect_ratio": round(float(max(aspect_ratios)), 4),
        "effective_ranks": [round(er, 2) for er in eff_ranks],
        "samples_per_effective_dim": [
            round(n_ob / max(er, 1.0), 2) for n_ob, er in zip(n_obs_list, eff_ranks)
        ],
        "positive_manifold_coherence": [round(pm, 3) for pm in pos_manifold],
        "mean_positive_manifold": round(float(np.mean(pos_manifold)), 3),
        "non_gaussianity": [round(ng, 3) for ng in non_gaussianity],
        "mean_non_gaussianity": round(float(np.mean(non_gaussianity)), 3),
        "private_variance_fraction": round(priv_mean, 3),
        "recommended_k": k_rec,
        "k_used": k_used,
        "gcca_eigenvalues": [round(float(v), 3) for v in lam_gcca[: min(8, lam_gcca.size)]],
        "gcca_mp_null_edge": round(float(mp_edge), 3),
        "signal_detected": signal_detected,
        "null_r2_bar": round(null_r2_bar, 4),
        "co_directional": co_directional,
        "recommended_sparseness_1se": round(
            float(np.median([g["q_signed_1se"] for g in view_geoms])), 3
        ),
        "recommended_sparseness_focal": round(
            float(np.median([g["q_signed_focal"] for g in view_geoms])), 3
        ),
        "per_view_sparseness_1se": [round(g["q_signed_1se"], 3) for g in view_geoms],
        "mean_alpha_jk": round(float(np.mean([g["alpha_jk"] for g in view_geoms])), 3),
        "mean_hoyer_localization": round(
            float(np.mean([g["hoyer_localization"] for g in view_geoms])), 3
        ),
        "loo_r2": {
            "dense_consensus": round(float(np.mean(row_dense)), 4),
            "general_sparse": round(float(np.mean(row_gen_sparse)), 4),
            "sparse_focal": round(float(np.mean(row_focal)), 4),
            "directional_co_pos": round(co_r2, 4),
            "directional_abs_pos": round(abs_r2, 4),
            "directional_shrunk": round(float(np.mean(row_dir_best)), 4),
            "directional_parts": round(float(np.mean(row_dir_parts)), 4),
            "nonlinear_lend": round(float(np.mean(row_lend)), 4),
            "nonlinear_ned": round(float(np.mean(row_ned)), 4),
            "nonlinear_shared_private": round(float(np.mean(row_ned_sp)), 4),
            "nonlinear_flow": round(float(np.mean(row_flow)), 4),
        },
        "nonlinearity": {
            "lend_press_gain": round(nonlin_lend_gain, 4),
            "ned_press_gain": round(nonlin_ned_gain, 4),
            "ned_press_se": round(nonlin_ned_se, 4),
            "flow_press_gain": round(nonlin_flow_gain, 4),
        },
        "_row_skills": {
            "dense_consensus": row_dense,
            "general_sparse": row_gen_sparse,
            "sparse_focal": row_focal,
            "directional_shrunk": row_dir_best,
            "directional_parts": row_dir_parts,
            "nonlinear_lend": row_lend,
            "nonlinear_ned": row_ned,
            "nonlinear_shared_private": row_ned_sp,
            "nonlinear_flow": row_flow,
        },
        "elapsed_ms": round(elapsed_ms, 2),
    }


def recommend_presets(
    views: Any,
    *,
    k: int | str | None = None,
    fit_rows: Any = None,
    path_graph: Mapping[int | str, Sequence[int | str]] | None = None,
    directional: bool | None = None,
    allow_nonlinear: bool = True,
) -> dict[str, Any]:
    """Analyze a multi-view collection and return a rank-ordered menu of presets with rationales."""
    prof = analyze_views(
        views, k=k, fit_rows=fit_rows, path_graph=path_graph, directional=directional
    )
    row_skills: dict[str, np.ndarray] = prof.pop("_row_skills")
    len_common = next(iter(row_skills.values())).size
    n_fit = prof["n_fit_rows"]
    k_used = prof["k_used"]
    miss_mode = "mask" if prof["has_whole_view_missing_rows"] else "impute"
    topo = "graph" if path_graph is not None else "loo"

    candidates = [p for p in PRESET_CATALOG if allow_nonlinear or p.complexity_tier == 1]

    raw_means = {p.name: float(np.mean(row_skills[p.name])) for p in candidates}
    lin_names = [p.name for p in candidates if p.complexity_tier == 1]
    best_lin_name = max(lin_names, key=lambda nm: raw_means[nm])
    best_lin_vec = row_skills[best_lin_name]

    eff_scores: dict[str, float] = {}
    paired_stats: dict[str, dict[str, float]] = {}

    diff_foc_gen = row_skills["sparse_focal"] - row_skills["general_sparse"]
    se_foc_gen = float(np.std(diff_foc_gen, ddof=1) / math.sqrt(max(len_common, 2)))

    for p in candidates:
        vec = row_skills[p.name]
        mu = raw_means[p.name]
        diff_vs_lin = vec - best_lin_vec
        se_vs_lin = float(np.std(diff_vs_lin, ddof=1) / math.sqrt(max(len_common, 2)))
        z_vs_lin = (
            float(np.mean(diff_vs_lin) / max(se_vs_lin, 1e-9))
            if p.name != best_lin_name
            else 0.0
        )
        paired_stats[p.name] = {
            "loo_r2": round(mu, 4),
            "delta_vs_best_linear": round(float(np.mean(diff_vs_lin)), 4),
            "paired_se": round(se_vs_lin, 4),
            "z_vs_best_linear": round(z_vs_lin, 2),
        }

        if not prof["signal_detected"] and not prof["domain_signs_supplied"]:
            null_tie = {
                "general_sparse": 0.020,
                "dense_consensus": 0.015,
                "directional_shrunk": 0.010,
                "directional_parts": 0.004,
                "sparse_focal": 0.002,
            }.get(p.name, -0.02 * p.complexity_tier)
            eff_scores[p.name] = 0.1 * mu + null_tie
            continue

        if p.complexity_tier == 2:
            n_aic = max(prof["n_complete_rows"] if prof["n_complete_rows"] >= 20 else n_fit // 2, 20)
            d_extra = {
                "nonlinear_lend": 3 * k_used + 4,
                "nonlinear_ned": 4 * k_used + 6,
                "nonlinear_flow": (2 * k_used + 3 if k_used == 1 else 4 * k_used + 6),
                "nonlinear_shared_private": 4 * k_used + 6,
            }.get(p.name, 4 * k_used + 6)
            aic_cost = (2.0 * float(d_extra)) / float(n_aic)
            eff_scores[p.name] = mu - aic_cost - 1.5 * se_vs_lin
        elif p.name == "sparse_focal":
            eff_scores[p.name] = mu - se_foc_gen
        elif p.name == "directional_parts":
            diff_dp_ds = row_skills["directional_parts"] - row_skills["directional_shrunk"]
            se_dp_ds = float(np.std(diff_dp_ds, ddof=1) / math.sqrt(max(len_common, 2)))
            k_pen = 0.01 if k_used == 1 else 0.0
            eff_scores[p.name] = mu - se_dp_ds - k_pen
        else:
            bonus = 0.0
            if p.name == "directional_shrunk":
                cone_coh = max(0.0, 2.0 * prof["mean_positive_manifold"] - 1.0)
                bonus = min(0.75 * max(se_vs_lin, 0.01), 0.03 * prof["mean_alpha_jk"] * cone_coh)
            elif p.name == "dense_consensus":
                diff_dg = row_skills["general_sparse"] - row_skills["dense_consensus"]
                se_dg = float(np.std(diff_dg, ddof=1) / math.sqrt(max(len_common, 2)))
                if float(np.mean(diff_dg)) <= se_dg and prof["max_aspect_ratio"] < 0.20:
                    bonus = float(max(0.0, np.mean(diff_dg))) + 0.002
            eff_scores[p.name] = mu + bonus

    def _sort_key(tmpl: PresetTemplate) -> tuple[int, float]:
        if prof["domain_signs_supplied"]:
            cone_ok = 1 if tmpl.name in ("directional_shrunk", "directional_parts") else 0
            return (cone_ok, eff_scores[tmpl.name])
        return (1, eff_scores[tmpl.name])

    ranked_templates = sorted(candidates, key=_sort_key, reverse=True)

    recommendations: list[dict[str, Any]] = []
    miss_note = (
        f"Whole-view missing rows detected in {prof['whole_missing_views']} -> missing_rows='mask'."
        if prof["has_whole_view_missing_rows"]
        else f"All {n_fit} fit rows observed across views -> missing_rows='impute'."
    )

    for rank_idx, p in enumerate(ranked_templates, start=1):
        params = dict(p.static_params)
        params["topology"] = topo
        if path_graph is not None:
            params["path_graph"] = path_graph
        params["missing_rows"] = miss_mode
        if prof["is_prepared"] and p.estimator == "fit":
            params["scale_list"] = ["none"]

        q_1se = prof["recommended_sparseness_1se"]
        q_foc = prof["recommended_sparseness_focal"]
        if p.name == "general_sparse":
            params["sparseness"] = "auto"
        elif p.name == "sparse_focal":
            params["sparseness"] = round(max(0.65, q_foc), 2)
        elif p.name == "directional_shrunk":
            if k_used == 1:
                params["energy_type"] = (
                    "nc" if (prof["domain_signs_supplied"] or prof["co_directional"]) else "nc_abs"
                )
            else:
                params["energy_type"] = "acc"
            if (
                not prof["domain_signs_supplied"]
                and prof["max_aspect_ratio"] < 0.5
                and q_1se >= 0.5
                and prof["mean_alpha_jk"] < 0.35
            ):
                params["sparseness"] = "auto"
            else:
                params["sparseness"] = None
        elif p.name == "directional_parts":
            params["sparseness"] = "auto"
        elif p.complexity_tier == 2:
            params["sparseness"] = (
                None if max(prof["view_widths"]) <= 12 else "auto"
            )

        st = paired_stats[p.name]
        why_parts: list[str] = []
        if prof["domain_signs_supplied"]:
            if p.name in ("directional_shrunk", "directional_parts"):
                why_parts.append(
                    "Domain sign orientation supplied (`signs`) -> preserving +1 target cone."
                )
            else:
                why_parts.append(
                    "Unconstrained-sign alternative (ignores supplied +1 domain sign cone)."
                )
        if not prof["signal_detected"]:
            why_parts.append(
                f"Max cross-view LOO R^2 ({prof['loo_r2'][best_lin_name]:.4f}) is below the n={n_fit} "
                f"Fisher-z 1% bar ({prof['null_r2_bar']:.4f}); favoring parsimonious linear settings."
            )
        elif p.name == "directional_shrunk":
            pol_str = (
                "co-directional (+1 across views -> energy_type='nc')"
                if prof["co_directional"]
                else "mixed cross-view polarity (-> energy_type='nc_abs' at k=1)"
            )
            why_parts.append(
                f"Directional cone LOO R^2={st['loo_r2']:.4f} (vs signed {prof['loo_r2']['general_sparse']:.4f}), "
                f"positive-manifold coherence={prof['mean_positive_manifold']:.2f}, {pol_str}, "
                f"max(p/n)={prof['max_aspect_ratio']:.2f}, and jackknife shrinkage alpha={prof['mean_alpha_jk']:.2f}."
            )
        elif p.name == "dense_consensus":
            why_parts.append(
                f"Dense signed LOO R^2={st['loo_r2']:.4f} (vs sparse {prof['loo_r2']['general_sparse']:.4f}) with "
                f"aspect ratios p/n={prof['aspect_ratios']}; retaining all features avoids pruning loss."
            )
        elif p.name == "general_sparse":
            why_parts.append(
                f"Signed paired-1SE LOO R^2={st['loo_r2']:.4f} with per-view sparseness='auto' "
                f"(probe 1SE={prof['per_view_sparseness_1se']}, localization={prof['mean_hoyer_localization']:.2f}, "
                f"directional cone R^2={prof['loo_r2']['directional_shrunk']:.4f})."
            )
        elif p.name == "sparse_focal":
            why_parts.append(
                f"Focal sparse LOO R^2={st['loo_r2']:.4f} at sparseness={params['sparseness']} "
                f"(Hoyer localization={prof['mean_hoyer_localization']:.2f} across widths {prof['view_widths']})."
            )
        elif p.name == "directional_parts":
            why_parts.append(
                f"Non-negative parts LOO R^2={st['loo_r2']:.4f} at k={k_used} "
                f"(GCCA eigenvalues={prof['gcca_eigenvalues'][:3]} vs Wachter null {prof['gcca_mp_null_edge']:.2f})."
            )
        elif p.name == "nonlinear_lend":
            why_parts.append(
                f"Decoder QR-PRESS R^2={st['loo_r2']:.4f} (gain {prof['nonlinearity']['lend_press_gain']:+.4f} "
                f"over linear, z={st['z_vs_best_linear']:+.2f})."
            )
        elif p.name == "nonlinear_ned":
            why_parts.append(
                f"Nonlinear encoder QR-PRESS R^2={st['loo_r2']:.4f} (gain {prof['nonlinearity']['ned_press_gain']:+.4f} "
                f"over linear, z={st['z_vs_best_linear']:+.2f})."
            )
        elif p.name == "nonlinear_shared_private":
            why_parts.append(
                f"Nonlinear QR-PRESS R^2={st['loo_r2']:.4f} (gain {prof['nonlinearity']['ned_press_gain']:+.4f}, "
                f"z={st['z_vs_best_linear']:+.2f}) with structured private variance={prof['private_variance_fraction']:.2f}."
            )
        elif p.name == "nonlinear_flow":
            why_parts.append(
                f"Rank-Gaussianized flow QR-PRESS R^2={st['loo_r2']:.4f} (gain {prof['nonlinearity']['flow_press_gain']:+.4f}, "
                f"z={st['z_vs_best_linear']:+.2f}) with marginal non-Gaussianity={prof['mean_non_gaussianity']:.2f}."
            )
        why_parts.append(miss_note)

        recommendations.append(
            {
                "rank": rank_idx,
                "name": p.name,
                "title": p.title,
                "family": p.family,
                "complexity_tier": p.complexity_tier,
                "estimator": p.estimator,
                "method": p.method,
                "k": k_used,
                "params": params,
                "score": round(eff_scores[p.name], 4),
                "metrics": st,
                "why": " ".join(why_parts),
                "when_to_prefer": p.when_to_prefer,
            }
        )

    report = {
        "profile": prof,
        "recommended": recommendations[0],
        "recommendations": recommendations,
    }
    report["summary_text"] = format_preset_menu(report)
    return report


def format_preset_menu(report: Mapping[str, Any], *, max_rows: int | None = None) -> str:
    """Format a :func:`recommend_presets` report as a concise human-readable table."""
    prof = report["profile"]
    recs = list(report["recommendations"])
    if max_rows is not None:
        recs = recs[: int(max_rows)]
    lines = [
        (
            f"PySiMLR Data Profile ({prof['elapsed_ms']:.1f} ms): "
            f"n={prof['n_fit_rows']} rows ({prof['n_complete_rows']} complete), "
            f"M={prof['n_views']} views {prof['view_widths']}, "
            f"max(p/n)={prof['max_aspect_ratio']:.2f}, "
            f"recommended_k={prof['recommended_k']} (using k={prof['k_used']}), "
            f"signal_detected={prof['signal_detected']}"
        ),
        "-" * 98,
        f"{'Rank':<5} {'Preset':<26} {'Method':<20} {'Score':>8} {'LOO R2':>8}  Key Settings",
        "-" * 98,
    ]
    for r in recs:
        prm = r["params"]
        key_cfg = (
            f"pos={prm.get('positivity')}, sp={prm.get('sparseness')}, "
            f"energy={prm.get('energy_type')}, miss={prm.get('missing_rows')}"
        )
        lines.append(
            f"#{r['rank']:<4} {r['name']:<26} {r['method']:<20} "
            f"{r['score']:>8.4f} {r['metrics']['loo_r2']:>8.4f}  {key_cfg}"
        )
        lines.append(f"      Why: {r['why']}")
    return "\n".join(lines)


def _unwrap_views_for_fit(views: Any) -> Any:
    """Unwrap ``prepare_views`` bundle dicts so estimators receive ``views['views']``."""
    if (
        isinstance(views, Mapping)
        and "views" in views
        and (
            (isinstance(views.get("params"), Mapping) and "views" in views["params"])
            or ("prepared" in views and "signs" in views)
        )
    ):
        return views["views"]
    return views


def fit_preset(
    views: Any,
    k: int | str | None = None,
    *,
    preset: str | None = "auto",
    analyze: bool = True,
    analysis: Mapping[str, Any] | None = None,
    fit_rows: Any = None,
    path_graph: Mapping[int | str, Sequence[int | str]] | None = None,
    allow_nonlinear: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """Fit a multi-view model through the 3-tier preset interface.

    - ``preset="<name>"``: runs the requested preset from :data:`PRESET_CATALOG` (adapting
      ``missing_rows`` / ``topology`` via :func:`recommend_presets` unless ``analyze=False``).
    - ``preset="auto"`` (default when no human choice is given): runs :func:`recommend_presets`
      and executes the Rank #1 recommended preset automatically.
    - ``preset=None`` (static Tier-1 fallback when view profiling is skipped): executes the
      general natural setting (`general_sparse`) directly without running :func:`analyze_views`.
    """
    from .api import fit as _api_fit
    from .uniform_shrinkage import fit_uniform_shrunk as _fit_uniform_shrunk

    report: dict[str, Any] | None = None
    if preset is None:
        tmpl = PRESET_BY_NAME["general_sparse"]
        k_eff = 2 if (k is None or k == "auto") else int(k)
        _, raw_mats, meta = _extract_views_with_meta(views)
        has_whole_miss = any((~np.isfinite(m).any(axis=1)).any() for m in raw_mats)
        params_none = dict(tmpl.static_params)
        params_none["missing_rows"] = "mask" if has_whole_miss else "impute"
        if meta["is_prepared"]:
            params_none["scale_list"] = ["none"]
        if path_graph is not None:
            params_none["topology"] = "graph"
            params_none["path_graph"] = path_graph
        chosen = {
            "rank": None,
            "name": tmpl.name,
            "title": tmpl.title,
            "family": tmpl.family,
            "estimator": tmpl.estimator,
            "method": tmpl.method,
            "k": k_eff,
            "params": params_none,
            "why": "Static Tier-1 natural default (automated view analysis skipped).",
        }
    elif preset == "auto":
        report = (
            dict(analysis)
            if analysis is not None
            else recommend_presets(
                views,
                k=k,
                fit_rows=fit_rows,
                path_graph=path_graph,
                allow_nonlinear=allow_nonlinear,
            )
        )
        chosen = dict(report["recommended"])
        chosen["params"] = dict(chosen["params"])
    else:
        if preset not in PRESET_BY_NAME:
            raise ValueError(
                f"unknown preset {preset!r}; choose 'auto', None, or one of {list(PRESET_BY_NAME)}."
            )
        if analyze or analysis is not None:
            report = (
                dict(analysis)
                if analysis is not None
                else recommend_presets(
                    views, k=k, fit_rows=fit_rows, path_graph=path_graph, allow_nonlinear=True
                )
            )
            matches = [r for r in report["recommendations"] if r["name"] == preset]
            chosen = dict(matches[0])
            chosen["params"] = dict(chosen["params"])
        else:
            tmpl = PRESET_BY_NAME[preset]
            k_eff = 2 if (k is None or k == "auto") else int(k)
            chosen = {
                "rank": None,
                "name": tmpl.name,
                "title": tmpl.title,
                "family": tmpl.family,
                "estimator": tmpl.estimator,
                "method": tmpl.method,
                "k": k_eff,
                "params": dict(tmpl.static_params),
                "why": f"Explicit preset {preset!r} with static template settings (analyze=False).",
            }
            if path_graph is not None:
                chosen["params"]["topology"] = "graph"
                chosen["params"]["path_graph"] = path_graph

    call_params = {**chosen["params"], **overrides}
    k_call = int(chosen["k"]) if (k is None or k == "auto") else int(k)
    fit_views = _unwrap_views_for_fit(views)
    if fit_rows is not None:
        fr = np.asarray(fit_rows)
        if isinstance(fit_views, Mapping):
            fit_views = {nm: np.asarray(v)[fr] for nm, v in fit_views.items()}
        else:
            fit_views = [
                (v[fr] if hasattr(v, "__getitem__") else np.asarray(v)[fr])
                for v in (fit_views if isinstance(fit_views, (list, tuple)) else [fit_views])
            ]

    if chosen["estimator"] == "fit_uniform_shrunk":
        sig = inspect.signature(_fit_uniform_shrunk)
        has_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_kw:
            res = _fit_uniform_shrunk(fit_views, k=k_call, **call_params)
        else:
            valid_kw = {kk: vv for kk, vv in call_params.items() if kk in sig.parameters}
            res = _fit_uniform_shrunk(fit_views, k=k_call, **valid_kw)
    else:
        res = _api_fit(chosen["method"], fit_views, k=k_call, **call_params)

    res["preset_used"] = {
        "name": chosen["name"],
        "rank": chosen.get("rank"),
        "estimator": chosen["estimator"],
        "method": chosen["method"],
        "k": k_call,
        "params": call_params,
        "overrides": dict(overrides),
        "why": chosen["why"],
    }
    if report is not None:
        res["preset_report"] = report
    return res


__all__ = [
    "PRESET_BY_NAME",
    "PRESET_CATALOG",
    "PresetTemplate",
    "analyze_views",
    "fit_preset",
    "format_preset_menu",
    "get_preset",
    "jackknife_js_alpha",
    "list_presets",
    "recommend_presets",
    "resolve_auto_sparseness",
]
