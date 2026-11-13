r"""Feature loadings: the interpretable, identified summary of a fit.

Every method returns an encoder ``V`` (``u = X V``). ``V`` is the scoring rule
for new subjects, and it is *not* the right object to read feature by feature.
Under a correlation energy (``nc``) the scores fix ``V`` only in the directions
where ``X`` has variance; in its low-variance directions ``V`` is weakly
identified. Measured on the synthetic cells of the D11/amigo study: 98%
subsample refits move ``V`` by mean sin^2 0.28 while ``U`` moves 0.004 and the
structure coefficients below move 0.001 (``docs/tracking_amigo.md``). This is
the classical instability of canonical *weights* against canonical *structure
coefficients*.

So interpretation uses quantities computed from ``U``:

``structure``
    :math:`C_{fj} = \mathrm{corr}(x_f, u_j)` per view: how strongly feature
    ``f`` moves with component ``j``. Association, not unique contribution: a
    feature correlated with a driver gets a large value too. Invariant to any
    per-feature affine scaling, so it is the same object for every method.
``loadings``
    The OLEE summary of ``structure``, with orthogonal columns.
    Under ``positivity="positive"`` (contract: disjoint supports) each feature
    is assigned to the component it correlates with most positively, keeping
    that correlation; a feature with no positive correlation is dropped. This
    is the Frobenius-nearest non-negative matrix with orthogonal columns.
    Otherwise (contract: orthogonality) it is the polar factor of ``C`` with
    each column rescaled to ``C``'s column norm, the nearest matrix with
    orthogonal columns of the same column scale.
``loading_rotation``
    The k x k rotation R described below.
``loading_margin`` (positive only)
    Per feature, the largest minus the second-largest positive correlation:
    how firmly the hard assignment holds.

The latent space fixes components only up to rotation: refits on 98% of the
rows return the same span of ``U`` with its axes rotated and reordered (one
planted case: a component with structure column sum +11.1 in one refit came
back as another column with sum -11.07). Reading columns, and assigning
features to them, therefore needs a canonical basis. `loadings` reports
``structure`` in the varimax basis of all views stacked (Kaiser 1958; it is
rotation-equivariant, so every rotation of the same span gives the same axes
up to order and sign), each column's sign set so its dominant loadings are
positive (the sum of cubes is >= 0). ``loading_rotation`` is that k x k
orthogonal R: the columns of ``structure`` and ``loadings`` describe
``u @ R``.

These are point estimates. ``U`` was fitted on the same rows, which inflates
``|C|``: with the cross-view link destroyed by permutation, in-sample
:math:`\max |C|` still reached 0.4-0.5 where ~0.07 is expected. Thresholds
and claims therefore come from `loading_inference`, which refits the whole
pipeline.
"""
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .positivity import NONNEG as _NONNEG
from .positivity import resolve_positivity
from .utils import MissingRowsWarning, center_within_groups, group_codes, within_group_permutation


def _drop_small_groups(codes: np.ndarray, min_size: int = 3) -> np.ndarray:
    """Boolean mask of rows whose group has at least ``min_size`` rows among ``codes``."""
    counts = np.bincount(codes)
    return counts[codes] >= int(min_size)


def _structure_grouped(views, u, obs_masks, groups, scale_within_group) -> List[torch.Tensor]:
    """`structure_coefficients` with features and latent columns centered within group.

    Per view, on its observed rows with at least 3 rows in their group: the pooled within-group
    correlation (group means removed, one global scale; ``scale_within_group=True`` also scales each
    group). Groups with fewer than 3 observed rows for the view are dropped from that view with a
    `MissingRowsWarning`; fewer than 3 remaining rows is a ``ValueError`` as without groups.
    """
    u64 = torch.as_tensor(u).detach().double().cpu().numpy()
    n = u64.shape[0]
    codes = group_codes(groups, n)
    out = []
    for m_idx, x in enumerate(views):
        x64 = torch.as_tensor(x).detach().double().cpu().numpy()
        mask = None if obs_masks is None else obs_masks[m_idx]
        if mask is not None:
            rows = np.flatnonzero(torch.as_tensor(mask, dtype=torch.bool).cpu().numpy())
            if x64.shape[0] == n:
                x64 = x64[rows]
            elif x64.shape[0] != rows.shape[0]:
                raise ValueError(f"view has {x64.shape[0]} rows but mask selects {rows.shape[0]}.")
        else:
            if x64.shape[0] != n:
                raise ValueError(f"view has {x64.shape[0]} rows but u has {n}.")
            rows = np.arange(n)
        keep = _drop_small_groups(codes[rows])
        if not keep.all():
            warnings.warn(f"structure_coefficients: view {m_idx}: {int((~keep).sum())} row(s) in groups with fewer than "
                          "3 observed rows were left out.", MissingRowsWarning, stacklevel=3)
        rows, x64 = rows[keep], x64[keep]
        if rows.shape[0] < 3:
            raise ValueError(f"view {m_idx} has {int(rows.shape[0])} usable rows (groups need at least 3 observed rows "
                             "each); structure coefficients need at least 3.")
        g = codes[rows]
        xc = center_within_groups(x64, g, scale_within_group)
        uc = center_within_groups(u64[rows], g, scale_within_group)
        nx, nu = np.linalg.norm(xc, axis=0, keepdims=True), np.linalg.norm(uc, axis=0, keepdims=True)
        xs = np.where(nx > 0, xc / np.where(nx > 0, nx, 1.0), 0.0)
        us = np.where(nu > 0, uc / np.where(nu > 0, nu, 1.0), 0.0)
        out.append(torch.as_tensor(xs.T @ us).float())
    return out


def structure_coefficients(views: Sequence[torch.Tensor], u: torch.Tensor,
                           obs_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
                           groups: Any = None, scale_within_group: bool = False) -> List[torch.Tensor]:
    """Per-view ``(p_m, k)`` correlations between each feature and each latent column.

    A constant feature or latent column has no correlation; its entries are 0.
    When ``obs_masks`` gives a boolean row mask for a view, that view's
    correlations are evaluated strictly on its observed rows.

    ``groups`` (length-n labels, default ``None``) makes the correlations within-group: features and
    latent columns are centered within each group (site, cohort, scanner) before the correlation, so
    group mean differences cannot produce an association; the result is the pooled within-group
    correlation. ``scale_within_group=False`` (default) scales globally after centering;
    ``True`` also divides each group by its own SD. A group with fewer than 3 observed rows for a
    view is left out of that view (``MissingRowsWarning``). ``groups=None`` is the pooled computation, unchanged.
    """
    if groups is not None:
        return _structure_grouped(views, u, obs_masks, groups, scale_within_group)
    u64 = torch.as_tensor(u).detach().double()
    uc_all = u64 - u64.mean(0, keepdim=True)
    nu_all = uc_all.norm(dim=0, keepdim=True)
    us_all = torch.where(nu_all > 0, uc_all / nu_all.clamp_min(1e-300), torch.zeros_like(uc_all))
    out = []
    for m_idx, x in enumerate(views):
        x64 = torch.as_tensor(x).detach().double()
        mask = None if obs_masks is None else obs_masks[m_idx]
        if mask is not None:
            mask_b = torch.as_tensor(mask, dtype=torch.bool, device=x64.device)
            u_m = u64[mask_b.to(u64.device)]
            if x64.shape[0] == mask_b.shape[0]:
                x64 = x64[mask_b]
            elif x64.shape[0] != u_m.shape[0]:
                raise ValueError(f"view has {x64.shape[0]} rows but mask selects {int(mask_b.sum())}.")
            if int(u_m.shape[0]) < 3:
                raise ValueError(f"view {m_idx} has {int(u_m.shape[0])} observed rows; structure coefficients need at least 3.")
            uc = u_m - u_m.mean(0, keepdim=True)
            nu = uc.norm(dim=0, keepdim=True)
            us = torch.where(nu > 0, uc / nu.clamp_min(1e-300), torch.zeros_like(uc))
        else:
            if x64.shape[0] != u64.shape[0]:
                raise ValueError(f"view has {x64.shape[0]} rows but u has {u64.shape[0]}.")
            us = us_all
        xc = x64 - x64.mean(0, keepdim=True)
        nx = xc.norm(dim=0, keepdim=True)
        xs = torch.where(nx > 0, xc / nx.clamp_min(1e-300), torch.zeros_like(xc))
        out.append((xs.T @ us).float())
    return out


def olee_loadings(structure: Sequence[torch.Tensor], positivity: str = "either"):
    """The OLEE summary of `structure_coefficients` (see the module docstring).

    Returns ``(loadings, margins)``; ``margins`` is ``None`` unless the
    positivity is non-negative.
    """
    loads, margins = [], []
    nonneg = resolve_positivity(positivity) in _NONNEG
    for c in structure:
        c64 = torch.as_tensor(c).double()
        p, k = c64.shape
        if nonneg:
            cp = c64.clamp_min(0.0)
            top, arg = cp.max(dim=1)
            second = (cp.topk(2, dim=1).values[:, 1] if k > 1 else torch.zeros(p, dtype=cp.dtype))
            lm = torch.zeros_like(cp)
            keep = top > 0
            lm[keep.nonzero(as_tuple=True)[0], arg[keep]] = top[keep]
            loads.append(lm.float())
            margins.append((top - second).float())
        else:
            uu, _, vh = torch.linalg.svd(c64, full_matrices=False)
            loads.append(((uu @ vh) * c64.norm(dim=0, keepdim=True)).float())
            margins.append(None)
    return loads, (margins if nonneg else None)


def varimax_rotation(structure: Sequence[torch.Tensor], max_iter: int = 500,
                     tol: float = 1e-10) -> torch.Tensor:
    """Orthogonal ``R`` (k x k) maximising the varimax criterion of the stacked
    views, with each column's sign set so its sum of cubed loadings is >= 0."""
    a = np.ascontiguousarray(torch.cat([torch.as_tensor(c).double() for c in structure], dim=0).numpy())
    p, k = a.shape
    r = np.eye(k)
    if k > 1:
        a_t = np.ascontiguousarray(a.T)
        inv_p = 1.0 / p
        crit = 0.0
        for _ in range(int(max_iter)):
            lam = a @ r
            lam2 = lam * lam
            uu, sv, vt = np.linalg.svd(a_t @ (lam2 * lam - lam * (lam2.sum(axis=0, keepdims=True) * inv_p)))
            r = uu @ vt
            sv_sum = float(sv.sum())
            if sv_sum <= crit * (1.0 + tol):
                break
            crit = sv_sum
    lam_f = a @ r
    signs = np.sign((lam_f * lam_f * lam_f).sum(axis=0))
    signs[signs == 0] = 1.0
    return torch.as_tensor(r * signs[None, :])


def _result_positivity(result: Dict[str, Any]) -> str:
    from .extension import _used
    return resolve_positivity(_used(result, "positivity", "either"))


def _result_energy_type(result: Dict[str, Any]) -> str:
    from .extension import _used
    return str(_used(result, "energy_type", "nc"))


def view_coupling_signs(structure: Sequence[torch.Tensor],
                        v_mats: Optional[Sequence[torch.Tensor]] = None,
                        *, positivity: str = "either",
                        energy_type: str = "nc") -> List[torch.Tensor]:
    """Per-view ``(k,)`` coupling polarity (``+1`` or ``-1``) against ``u @ R``.

    Under ``positivity="positive"`` with a column-sign-invariant similarity
    (`similarity.SIGN_INVARIANT_ENERGIES`, e.g. ``"nc_abs"`` or ``"recon_r2"``),
    a non-negative basis ``V_m >= 0`` can couple negatively to a rotated
    consensus axis ``(u @ R)_j`` (for example regional hypoperfusion opposing
    hyperconnectivity). Orienting ``structure[m][:, j]`` by the active-feature
    third-moment sign before `olee_loadings` preserves disjoint non-negative
    parcel loadings while recording ``-1.0`` here so the opposing biological
    direction is explicit. Signed objectives (``"nc"``, etc.) and unconstrained
    fits always return ``+1.0``.
    """
    from .similarity import SIGN_INVARIANT_ENERGIES
    allow_flip = (resolve_positivity(positivity) in _NONNEG) and (energy_type in SIGN_INVARIANT_ENERGIES)
    signs: List[torch.Tensor] = []
    for m, c in enumerate(structure):
        c64 = torch.as_tensor(c).double()
        k = int(c64.shape[1])
        if not allow_flip:
            signs.append(torch.ones(k, dtype=torch.float32))
            continue
        if (v_mats is not None and m < len(v_mats) and v_mats[m] is not None
                and int(v_mats[m].shape[0]) == int(c64.shape[0])):
            w_row = torch.as_tensor(v_mats[m]).detach().double().abs().sum(dim=1, keepdim=True)
            if float(w_row.max()) <= 1e-12:
                w_row = torch.ones((c64.shape[0], 1), dtype=torch.float64)
        else:
            w_row = torch.ones((c64.shape[0], 1), dtype=torch.float64)
        skew = torch.sum(w_row * (c64 ** 3), dim=0)
        signs.append(torch.where(skew < 0.0, -1.0, 1.0).float())
    return signs


def _oriented_olee(c: List[torch.Tensor], v_list: Optional[Sequence[torch.Tensor]],
                   pos: str, etype: str):
    csigns = view_coupling_signs(c, v_list, positivity=pos, energy_type=etype)
    c_oriented = [ci * sg.to(ci.dtype).unsqueeze(0) for ci, sg in zip(c, csigns)]
    load, margin = olee_loadings(c_oriented, pos)
    return csigns, load, margin


def loadings(result: Dict[str, Any], views: Any, u: Optional[torch.Tensor] = None, *,
             groups: Any = None, scale_within_group: bool = False) -> Dict[str, Any]:
    """`structure`, `loadings`, `loading_margin`, `coupling_signs` and per-feature
    `communality` for a `pysimlr.fit` result on ``views``.

    ``views`` are replayed through the fit's own preprocessing
    (``scale_list``, ``provenance_list``), so this works for every method. With
    the training views and ``u=None`` it reproduces the keys `pysimlr.fit`
    stored; with held-out views pass ``u=pysimlr.predict(result, views)["u"]``.

    ``groups`` (length-n labels, default ``None``) computes everything within group (see
    `structure_coefficients`): ``structure``, the varimax ``loading_rotation`` derived from it,
    ``loadings`` and ``communality`` all describe within-group associations. The returned
    ``loading_rotation`` then differs from the pooled one; to feed it to `feature_maps`, merge it
    into the result (``{**result, **loadings(result, views, groups=g)}``). ``scale_within_group``
    is described in `structure_coefficients`. With ``groups=None`` the output is unchanged.
    """
    from .api import _preprocess_views, as_views
    view_list = as_views(views)
    if u is None:
        u = result["u"]
    obs_masks: Optional[List[Optional[torch.Tensor]]] = None
    if result.get("missing_rows") == "mask":
        from .utils import preprocess_data
        obs_masks = [~torch.isnan(v).all(dim=1) for v in view_list]
        scale_list = result["scale_list"]
        prov_list = result["provenance_list"]
        xs = []
        for m_idx, (v, obs_m) in enumerate(zip(view_list, obs_masks)):
            v_obs = v[obs_m]
            if scale_list and scale_list[0] != "none" and m_idx < len(prov_list):
                xs.append(preprocess_data(v_obs, scale_list, provenance=prov_list[m_idx]))
            else:
                xs.append(v_obs)
    else:
        xs, _ = _preprocess_views(view_list, result["scale_list"], result["provenance_list"])
    gkw = {} if groups is None else {"groups": groups, "scale_within_group": scale_within_group}
    c0 = structure_coefficients(xs, u, obs_masks=obs_masks, **gkw)
    rot = varimax_rotation(c0)
    # Correlations with the rotated scores themselves, not C0 @ R: the two agree
    # only when u's columns are uncorrelated with equal variance (orthogonal
    # mixings). Under "avg" they differed by up to 0.07, so "structure describes
    # u @ R" held only approximately.
    u_rot = torch.as_tensor(u).detach().double() @ rot
    c = structure_coefficients(xs, u_rot, obs_masks=obs_masks, **gkw)
    pos = _result_positivity(result)
    etype = _result_energy_type(result)
    v_list = None if result.get("basis_is_joint") else result.get("v")
    csigns, load, margin = _oriented_olee(c, v_list, pos, etype)
    return {"structure": c, "loadings": load, "loading_margin": margin,
            "loading_rotation": rot.float(), "coupling_signs": csigns,
            "communality": [(ci.double() ** 2).sum(1).float() for ci in c]}


FEATURE_MAP_KINDS = ("structure", "partial", "global_partial", "component_partial", "relative_weight")


def _zscore(a: np.ndarray) -> np.ndarray:
    """Column z-scores (population SD); a constant column becomes 0."""
    a = a - a.mean(0, keepdims=True)
    sd = a.std(0, keepdims=True)
    return np.where(sd > 0, a / np.where(sd > 0, sd, 1.0), 0.0)


def _corr_map(xz: np.ndarray, tz: np.ndarray) -> np.ndarray:
    return xz.T @ tz / xz.shape[0]


def _ledoit_wolf_centered(x: np.ndarray, d: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Exact ``(shrinkage, mu)`` matching ``sklearn.covariance.ledoit_wolf_shrinkage(x, assume_centered=True)``
    without forming a ``p x p`` precision matrix (``O(min(n, p)^2 max(n, p))``, or ``O(n p)`` given SVD ``d``)."""
    n, p = x.shape
    if p <= 1:
        return 0.0, (float(np.sum(x ** 2)) / (n * max(p, 1)) if p == 1 and n > 0 else 0.0)
    x2 = x ** 2
    trace_sum = float(np.sum(x2)) / n
    mu = trace_sum / p
    beta_ = float(np.sum(np.sum(x2, axis=1) ** 2))
    if d is not None:
        delta_ = float(np.sum(d ** 4)) / (n ** 2)
    elif p > n:
        g = x @ x.T
        delta_ = float(np.sum(g ** 2)) / (n ** 2)
    else:
        c = x.T @ x
        delta_ = float(np.sum(c ** 2)) / (n ** 2)
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * trace_sum + p * (mu ** 2)) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if (beta <= 0.0 or delta <= 0.0) else float(beta / delta)
    return shrinkage, mu


def _ledoit_wolf_joint_from_svd(row_sq_x: np.ndarray, sum_x2: float, sum_d4: float,
                                pp: np.ndarray, d: np.ndarray, p_x: int,
                                t_j: np.ndarray) -> Tuple[float, float]:
    """Exact Ledoit-Wolf ``(shrinkage, mu)`` for ``[xl, t_j]`` in ``O(n r)`` from thin SVD of ``xl``."""
    n = len(t_j)
    p = p_x + 1
    t2 = t_j ** 2
    t2_sum = float(np.sum(t2))
    if p <= 1:
        return 0.0, t2_sum / max(n, 1)
    trace_sum = (sum_x2 + t2_sum) / n
    mu = trace_sum / p
    beta_ = float(np.sum((row_sq_x + t2) ** 2))
    u = pp.T @ t_j
    xt_norm2 = float(np.sum((d * u) ** 2))
    delta_ = (sum_d4 + 2.0 * xt_norm2 + t2_sum ** 2) / (n ** 2)
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * trace_sum + p * (mu ** 2)) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if (beta <= 0.0 or delta <= 0.0) else float(beta / delta)
    return shrinkage, mu


def _partial_map(xz: np.ndarray, tz: np.ndarray, shrinkage) -> np.ndarray:
    """Partial correlation of each feature with each target column, given every
    other feature of the view: -P_ft / sqrt(P_ff P_tt), P the inverse of the
    (shrunk) correlation matrix of [X, t_j]."""
    n, p = xz.shape
    out = np.zeros((p, tz.shape[1]))
    t_live = [j for j in range(tz.shape[1]) if tz[:, j].std() > 0]
    if not t_live:
        return out
    if shrinkage != "ledoit_wolf":
        lam_fixed = float(shrinkage)
        if not 0.0 <= lam_fixed < 1.0:
            raise ValueError(f"shrinkage must be 'ledoit_wolf' or a float in [0, 1); got {shrinkage!r}.")
    else:
        lam_fixed = 0.0
    live = xz.std(0) > 0
    if not live.any():
        return out
    xl = xz[:, live]
    p_l = xl.shape[1]
    if shrinkage == "ledoit_wolf":
        pp, d, qt = np.linalg.svd(xl, full_matrices=False)
        s_x = d / np.sqrt(n)
        s_x2 = s_x ** 2
        qt2 = qt ** 2
        null_diag = np.maximum(1.0 - qt2.sum(axis=0), 0.0) if len(d) < p_l else None
        row_sq_x = np.sum(xl ** 2, axis=1)
        sum_x2 = float(row_sq_x.sum())
        sum_d4 = float(np.sum(d ** 4))
        for j in t_live:
            t_j = tz[:, j]
            lam_j, mu_j = _ledoit_wolf_joint_from_svd(row_sq_x, sum_x2, sum_d4, pp, d, p_l, t_j)
            alpha = lam_j * mu_j
            if alpha <= 0.0 and (p_l >= n or d[-1] <= d[0] * 1e-12):
                joint = np.column_stack([xl, t_j])
                s = (1.0 - lam_j) * (joint.T @ joint / n) + alpha * np.eye(p_l + 1)
                prec = np.linalg.inv(s)
                dg = np.diag(prec)
                out[live, j] = -prec[:-1, -1] / np.sqrt(dg[:-1] * dg[-1])
                continue
            eig = (1.0 - lam_j) * s_x2 + alpha
            pt = pp.T @ t_j
            u_t = pt / np.sqrt(n)
            w = qt.T @ (((1.0 - lam_j) * s_x * u_t) / eig)
            t_perp2 = max(float(np.sum((t_j - pp @ pt) ** 2)) / n, 0.0)
            s_schur = (1.0 - lam_j) * t_perp2 + alpha + float(np.sum((alpha * (1.0 - lam_j) * (u_t ** 2)) / eig))
            diag_ainv = qt2.T @ (1.0 / eig)
            if null_diag is not None:
                diag_ainv = diag_ainv + (1.0 / alpha) * null_diag
            out[live, j] = w / np.sqrt(diag_ainv * s_schur + w ** 2)
        return out

    rms_x = np.sqrt(np.mean(xl ** 2, axis=0))
    xl_u = xl / rms_x
    pp, d, qt = np.linalg.svd(xl_u, full_matrices=False)
    if lam_fixed == 0.0 and (p_l >= n or d[-1] <= d[0] * 1e-12):
        for j in t_live:
            joint = np.column_stack([xl, tz[:, j]])
            s = joint.T @ joint / n
            prec = np.linalg.inv(s)
            dg = np.diag(prec)
            out[live, j] = -prec[:-1, -1] / np.sqrt(dg[:-1] * dg[-1])
        return out
    s_x = d / np.sqrt(n)
    eig = (1.0 - lam_fixed) * (s_x ** 2) + lam_fixed
    qt2 = qt ** 2
    diag_ainv = qt2.T @ (1.0 / eig)
    if len(d) < p_l:
        diag_ainv = diag_ainv + (1.0 / lam_fixed) * np.maximum(1.0 - qt2.sum(axis=0), 0.0)
    for j in t_live:
        t_j = tz[:, j]
        t_u = t_j / np.sqrt(float(np.mean(t_j ** 2)))
        pt = pp.T @ t_u
        u_t = pt / np.sqrt(n)
        w = qt.T @ (((1.0 - lam_fixed) * s_x * u_t) / eig)
        t_perp2 = max(float(np.sum((t_u - pp @ pt) ** 2)) / n, 0.0)
        s_schur = (1.0 - lam_fixed) * t_perp2 + lam_fixed + float(
            np.sum((lam_fixed * (1.0 - lam_fixed) * (u_t ** 2)) / eig)
        )
        out[live, j] = w / np.sqrt(diag_ainv * s_schur + w ** 2)
    return out


def _global_factor(xz: np.ndarray, control) -> np.ndarray:
    """(n, c) design of the view's global factor(s), from the rows given."""
    if isinstance(control, str):
        if control == "pc1":
            return np.linalg.svd(xz, full_matrices=False)[0][:, :1]
        if control == "mean":
            return xz.mean(1, keepdims=True)
        raise ValueError(f"control must be 'pc1', 'mean' or an (n, c) array; got {control!r}.")
    g = np.asarray(control, dtype=np.float64)
    g = g[:, None] if g.ndim == 1 else g
    if g.shape[0] != xz.shape[0]:
        raise ValueError(f"control has {g.shape[0]} rows; the view has {xz.shape[0]}.")
    return g


def _global_partial_map(xz: np.ndarray, tz: np.ndarray, control) -> np.ndarray:
    """corr(x_f, t_j) after regressing both on the view's global factor."""
    g = _global_factor(xz, control)
    g = np.column_stack([np.ones(len(g)), g])
    q, _ = np.linalg.qr(g)
    def resid(a):
        return a - q @ (q.T @ a)
    return _corr_map(_zscore(resid(xz)), _zscore(resid(tz)))


def _component_partial_map(xz: np.ndarray, tz: np.ndarray) -> np.ndarray:
    """Haufe et al. (2014) forward pattern on standardized data: corr(X, T) corr(T)^{-1}."""
    s_tt = tz.T @ tz / tz.shape[0]
    return _corr_map(xz, tz) @ np.linalg.pinv(s_tt)


def _resolve_rw_shrinkage(xl: np.ndarray, d: np.ndarray, shrinkage: Any,
                          joint: Optional[np.ndarray] = None) -> float:
    """Resolve the shrinkage intensity ``lam`` in ``[0, 1]`` for Johnson relative weights.

    Two Ledoit-Wolf definitions, named apart because they answer different needs:
    ``"ledoit_wolf"`` estimates ``lam`` from the view ``X`` alone, so it does not depend on the
    target (required by a permutation test of the target, `feature_relevance`);
    ``"ledoit_wolf_joint"`` estimates it from ``[X, t_j]`` (pass ``joint``), which makes the
    weights sum exactly to the shrunk Schur R^2 of ``t_j`` (`feature_maps`). Here ``"auto"`` is
    ``0.0`` on a full-rank view with ``p < n`` and ``"ledoit_wolf"`` otherwise.
    """
    if shrinkage == "auto":
        if xl.shape[1] < xl.shape[0] and len(d) > 0 and d.min() > d.max() * 1e-10:
            return 0.0
        shrinkage = "ledoit_wolf"
    if shrinkage in ("ledoit_wolf", "ledoit_wolf_joint"):
        if shrinkage == "ledoit_wolf_joint":
            if joint is None:
                raise ValueError("'ledoit_wolf_joint' needs the target: it is defined per target in feature_maps.")
            return _ledoit_wolf_centered(joint)[0]
        return _ledoit_wolf_centered(xl, d=d)[0]
    if isinstance(shrinkage, (str, bool)):
        raise ValueError(f"rw_shrinkage must be 'auto', 'ledoit_wolf', 'ledoit_wolf_joint' or a float in [0, 1); "
                         f"got {shrinkage!r}.")
    lam = float(shrinkage)
    if not 0.0 <= lam < 1.0:
        raise ValueError(f"rw_shrinkage must be 'auto', 'ledoit_wolf', 'ledoit_wolf_joint' or a float in [0, 1); "
                         f"got {shrinkage!r}.")
    if lam == 0.0:
        if xl.shape[1] >= xl.shape[0]:
            raise ValueError(f"relative_weight needs fewer features than rows; the view has "
                             f"{xl.shape[1]} non-constant features and {xl.shape[0]} rows. "
                             f"Use 'partial' or rw_shrinkage='auto'.")
        if len(d) == 0 or d.min() <= d.max() * 1e-10:
            raise ValueError("relative_weight needs a full-rank view; it has collinear features. "
                             "Use 'partial' or rw_shrinkage='auto'.")
    return lam


def _basis_from_svd(pp: np.ndarray, d: np.ndarray, qt: np.ndarray, n: int, lam: float):
    """``(Z_lam, Lambda_lam**2)`` from thin SVD ``xl = pp diag(d) qt`` at shrinkage ``lam``."""
    if lam == 0.0:
        return pp @ qt, ((qt.T * d[None, :]) @ qt / np.sqrt(n)) ** 2
    s = d / np.sqrt(n)
    d_lam = (1.0 - lam) * (s ** 2) + lam
    zo = (pp * (s / np.sqrt(d_lam))[None, :]) @ qt
    lam_mat = (qt.T * (np.sqrt(d_lam) - np.sqrt(lam))[None, :]) @ qt
    lam_mat[np.diag_indices(qt.shape[1])] += np.sqrt(lam)
    return zo, lam_mat ** 2


def _rw_weights_from_svd(pp: np.ndarray, d: np.ndarray, qt: np.ndarray,
                         n: int, lam: float, t_cols: np.ndarray) -> np.ndarray:
    """Compute ``(Lambda_lam ** 2) @ (beta ** 2)`` from thin SVD without forming a ``p x p``
    matrix when ``p > 2 r``."""
    r, p_l = qt.shape
    if lam == 0.0 or p_l <= 2 * r:
        zo, lam2 = _basis_from_svd(pp, d, qt, n, lam)
        beta = (1.0 - lam) * (zo.T @ t_cols / np.sqrt(n)) if lam > 0.0 else (zo.T @ t_cols / np.sqrt(n))
        return lam2 @ (beta ** 2)
    s = d / np.sqrt(n)
    d_lam = (1.0 - lam) * (s ** 2) + lam
    v_r = qt.T
    pt = pp.T @ t_cols
    scale_s = s / np.sqrt(d_lam)
    scaled_pt = scale_s * pt if pt.ndim == 1 else scale_s[:, None] * pt
    beta = (1.0 - lam) * (v_r @ scaled_pt / np.sqrt(n))
    u = beta ** 2
    c = np.sqrt(d_lam) - np.sqrt(lam)
    gamma = np.sqrt(lam)
    diag_scale = lam + 2.0 * gamma * ((v_r ** 2) @ c)
    cc = c[:, None] * c[None, :]
    if u.ndim == 1:
        m_mat = (v_r * u[:, None]).T @ v_r
        return diag_scale * u + np.sum((v_r @ (cc * m_mat)) * v_r, axis=1)
    cols = []
    for j in range(u.shape[1]):
        m_mat = (v_r * u[:, j:j + 1]).T @ v_r
        cols.append(diag_scale * u[:, j] + np.sum((v_r @ (cc * m_mat)) * v_r, axis=1))
    return np.column_stack(cols)


def _relative_weight_map(xz: np.ndarray, tz: np.ndarray, shrinkage: Any = 0.0) -> np.ndarray:
    """Johnson's relative weights (see `_relative_weight_map_lam`, which also returns ``lam``)."""
    return _relative_weight_map_lam(xz, tz, shrinkage)[0]


def _relative_weight_map_lam(xz: np.ndarray, tz: np.ndarray, shrinkage: Any = 0.0):
    """``(weights, lam)``: Johnson's (2000) relative weights, each feature's share of R^2(t_j ~ X),
    and the shrinkage intensity used for each target (``(k,)``; 0.0 is the classical estimator).

    X = P D Q' (SVD of the standardized view); Z = P Q' is its closest orthonormal
    counterpart, and Lambda = Q D Q' / sqrt(n) maps Z back to X. The weight of
    feature f is sum_l Lambda_fl^2 beta_l^2, beta the regression of t_j on Z; the
    weights are non-negative and sum to R^2.

    ``shrinkage``: ``0.0`` (classical; needs a full-rank view with ``p < n``); a float in
    ``(0, 1)``; ``"ledoit_wolf"`` (``lam`` from ``X`` alone, one per view); ``"ledoit_wolf_joint"``
    (``lam_j`` from ``[X, t_j]``, using ``S_lam = (1 - lam) corr([X, t_j]) + lam I``, so the
    weights sum exactly to the Schur-complement ``R^2_lam(t_j ~ X) in [0, 1)``); or ``"auto"``
    (``0.0`` on a full-rank ``p < n`` view, else ``"ledoit_wolf_joint"``). Thin SVD,
    ``O(min(n, p)^2 p)``, so any width ``p >= n`` is defined.
    """
    n, p = xz.shape
    k = tz.shape[1]
    out = np.zeros((p, k))
    lams = np.full(k, np.nan)
    live = xz.std(0) > 0
    if not live.any():
        _resolve_rw_shrinkage(xz[:, :0], np.empty(0),
                              0.5 if shrinkage in ("auto", "ledoit_wolf", "ledoit_wolf_joint") else shrinkage)
        return out, lams
    xl = xz[:, live]
    pp, d, qt = np.linalg.svd(xl, full_matrices=False)
    is_full_narrow = xl.shape[1] < n and len(d) > 0 and d.min() > d.max() * 1e-10
    joint = shrinkage == "ledoit_wolf_joint" or (shrinkage == "auto" and not is_full_narrow)
    if not joint:
        lam = _resolve_rw_shrinkage(xl, d, 0.0 if shrinkage == "auto" else shrinkage)
        out[live] = _rw_weights_from_svd(pp, d, qt, n, lam, tz)
        return out, np.full(k, lam)
    row_sq_x = np.sum(xl ** 2, axis=1)
    sum_x2 = float(row_sq_x.sum())
    sum_d4 = float(np.sum(d ** 4))
    p_l = xl.shape[1]
    for j in range(k):
        if not tz[:, j].std() > 0:
            continue
        lam_j = _ledoit_wolf_joint_from_svd(row_sq_x, sum_x2, sum_d4, pp, d, p_l, tz[:, j])[0]
        lams[j] = lam_j
        if lam_j >= 1.0:
            continue
        out[live, j] = _rw_weights_from_svd(pp, d, qt, n, lam_j, tz[:, j])
    return out, lams


def _relative_weight_basis(xl: np.ndarray, shrinkage: Any = 0.0):
    """``(Z, Lambda**2)``; see `_relative_weight_basis_lam`, which also returns ``lam``."""
    return _relative_weight_basis_lam(xl, shrinkage)[:2]


def _relative_weight_basis_lam(xl: np.ndarray, shrinkage: Any = 0.0):
    """``(Z, Lambda**2, lam)`` of Johnson's relative weights for standardized ``xl`` (n x p).

    At ``shrinkage=0.0`` (default), requires full-rank ``p < n``: ``Z = P Q'``
    (SVD ``X = P D Q'``), ``Lambda = Q D Q' / sqrt(n)``. With ``shrinkage='auto'``,
    ``'ledoit_wolf'`` (``lam`` from ``X`` alone, independent of any target) or ``lam in (0, 1)``,
    uses ``R_lam = (1 - lam) R_XX + lam I_p`` so shares are defined for any ``p >= n`` or collinear
    view. ``'ledoit_wolf_joint'`` is per target and is not accepted here.
    """
    n, p = xl.shape
    if p == 0:
        return np.zeros((n, 0)), np.zeros((0, 0)), 0.0
    if shrinkage == "ledoit_wolf_joint":
        raise ValueError("'ledoit_wolf_joint' depends on the target; the relative-weight basis takes "
                         "'auto', 'ledoit_wolf' or a float.")
    pp, d, qt = np.linalg.svd(xl, full_matrices=False)
    lam = _resolve_rw_shrinkage(xl, d, shrinkage)
    return (*_basis_from_svd(pp, d, qt, n, lam), lam)


def _relative_weight_shares(zo: np.ndarray, lam2: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Relative-weight shares (columns sum to 1) for every column of ``targets`` (n x B) at
    once: the (p x B) fraction of each target's explained variance owed to each feature."""
    n = targets.shape[0]
    zt = _zscore(targets) * (1.0 / np.sqrt(n))
    u = np.ascontiguousarray(zo.T) @ zt
    np.multiply(u, u, out=u)
    w = np.ascontiguousarray(lam2) @ u
    tot = w.sum(0, keepdims=True)
    return w / tot if (tot > 0).all() else np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 0.0)


def _to_probe_tensor(m: Any, n_rows: int, view_idx: int) -> torch.Tensor:
    from .extension import _to_tensor
    t = _to_tensor(m, f"probe_views[{view_idx}]")
    if int(t.shape[0]) != n_rows:
        raise ValueError(f"probe_views[{view_idx}] has {int(t.shape[0])} rows; views have {n_rows}.")
    return t


def _per_view_arg(val: Any, n_views: int, name: str, *, noun: str = "there are") -> List[Any]:
    if isinstance(val, (list, tuple)):
        if len(val) != n_views:
            raise ValueError(f"{name} has {len(val)} entries; {noun} {n_views} views.")
        return list(val)
    return [val] * n_views


def _preprocess_observed(v: torch.Tensor, obs_m: np.ndarray, result: Dict[str, Any],
                         view_idx: Optional[int]) -> np.ndarray:
    """``(n, p)`` float64 array of ``v`` replayed through the fit's preprocessing on its observed rows
    (``obs_m``); unobserved rows are NaN. ``view_idx=None`` is a probe matrix (no stored provenance)."""
    from .utils import preprocess_data
    scale_list, prov_list = result["scale_list"], result["provenance_list"]
    v_obs = v.detach().cpu()[torch.as_tensor(obs_m)]
    if view_idx is None:
        x = preprocess_data(v_obs, scale_list)[0]
    elif scale_list and scale_list[0] != "none" and view_idx < len(prov_list):
        x = preprocess_data(v_obs, scale_list, provenance=prov_list[view_idx])
    else:
        x = v_obs
    out = np.full(tuple(v.shape), np.nan, dtype=np.float64)
    out[np.asarray(obs_m, dtype=bool)] = x.detach().double().numpy()
    return out


def _excluded_view_indices(result: Dict[str, Any], exclude_views: Optional[Sequence[Any]], n_views: int) -> frozenset:
    """Indices of ``exclude_views`` (ints, view names, or ``"viewN"``) among the fit's views."""
    if exclude_views is None:
        return frozenset()
    if isinstance(exclude_views, (str, bytes)):
        raise TypeError("exclude_views must be a sequence of view names or indices, not a bare string.")
    from .extension import view_names
    names = view_names(result)[:n_views]
    out = set()
    for ev in exclude_views:
        if isinstance(ev, (int, np.integer)) and not isinstance(ev, bool) and 0 <= int(ev) < n_views:
            out.add(int(ev))
        elif isinstance(ev, str) and ev in names:
            out.add(names.index(ev))
        elif isinstance(ev, str) and ev.startswith("view") and ev[4:].isdigit() and int(ev[4:]) < n_views:
            out.add(int(ev[4:]))
        else:
            raise ValueError(f"unknown view {ev!r} in exclude_views; the fit has views {names}.")
    return frozenset(out)


def feature_maps(result: Dict[str, Any], views: Any, kinds: Sequence[str] = FEATURE_MAP_KINDS,
                 u: Optional[torch.Tensor] = None, target: str = "other_views",
                 shrinkage: Any = "ledoit_wolf", control: Any = "pc1",
                 rw_shrinkage: Any = 0.0, return_shrinkage: bool = False,
                 probe_views: Optional[Sequence[Any]] = None,
                 min_rows: int = 10, return_info: bool = False, *,
                 groups: Any = None, scale_within_group: bool = False,
                 exclude_views: Optional[Sequence[Any]] = None):
    r"""Alternative feature maps beside `structure`, all in the fit's rotated frame.

    `structure` (corr(x_f, u_j)) measures association: a feature that only
    correlates with a driver scores as high as the driver, and on a view dominated
    by one global factor (cortical thickness) every feature loads on the component
    that tracks it. These maps answer other questions about the same component
    ``j`` of ``u @ loading_rotation``:

    ``structure``
        corr(x_f, t_j). Association.
    ``partial``
        Partial correlation of x_f with t_j given all other features of the view,
        from the inverse of the shrunk correlation matrix of [X_m, t_j]
        (``shrinkage``: ``"ledoit_wolf"`` or a float in [0, 1) towards the diagonal).
        Unique contribution; can rank suppressors highly, so check its stability
        with `loading_inference` (``maps=``).
    ``global_partial``
        corr(x_f, t_j) after regressing both on the view's global factor
        (``control``: ``"pc1"``, ``"mean"``, an (n, c) array, or a list with one of
        these per view, ``None`` giving NaN for that view). Choose it per view:
        removing PC1 helps only where PC1 is a global size factor (on ADNI it does
        for DiReCT thickness; on volume / surface area it promotes tissue totals).
    ``component_partial``
        corr(X, T) corr(T)^{-1}, the forward pattern of Haufe et al. (2014):
        association with t_j holding the other components fixed. Equals
        ``structure`` when the components are uncorrelated.
    ``relative_weight``
        Johnson's (2000) relative weights: each feature's non-negative share of
        R^2(t_j ~ X_m); they sum to that R^2. At default ``rw_shrinkage=0.0``,
        defined only for a full-rank view with fewer features than rows (otherwise
        that view's map is NaN, with a warning). Pass ``rw_shrinkage="auto"``
        (classical when full-rank ``p < n``, joint Ledoit-Wolf Schur R^2 when
        ``p >= n`` or collinear), ``"ledoit_wolf_joint"`` (lam per target from ``[X, t_j]``),
        ``"ledoit_wolf"`` (lam from ``X`` alone) or a float in ``[0, 1)`` (or a per-view list)
        to compute shrunk relative weights on wide views. With ``return_shrinkage=True`` the
        call returns ``(maps, {"relative_weight": [lam per target, per view]})`` so the
        estimator actually used is on record (0.0 = classical; NaN = map undefined).

    ``target="other_views"`` (default) sets t to the mean of the *other* views'
    standardized scores. The regression-based maps need this: a linear fit's own
    score is exactly X_m V_m, so regressing the consensus on X_m would only
    recover the encoder V_m, which is weakly identified. ``"consensus"`` uses
    ``u @ R`` itself; for ``partial`` and ``relative_weight`` it is circular.

    ``probe_views`` (optional sequence of length ``len(views)``, ``None`` for views
    kept at their fitted resolution) implements coarse-to-fine split-scale readout:
    the latent scores are computed from the well-conditioned ``views`` the model was
    fitted on, while any non-``None`` ``probe_views[m]`` (for example a ``p >> n``
    connectome-edge or fixel matrix) is preprocessed with ``result["scale_list"]`` and
    mapped against the *other* views' scores (requires ``target="other_views"``).

    **Unobserved rows** (``result["missing_rows"] == "mask"``, rows that are all-NaN in a view).
    For view ``m`` every map is computed on the rows where view ``m`` is observed *and* its
    target exists, pairwise-complete per feature-latent pair (all features of a view share its
    observed rows, so this is one row set per view). With ``target="other_views"`` the target at
    a row is the mean of the standardized scores of the other views observed at that row, and is
    undefined (the row is dropped) where no other view is observed; a row observed in only one
    view therefore enters no map (its own view has no target there, and the other views are
    unobserved there). With no missing rows the result is exactly the complete-data result.
    It equals the complete-case computation (the same call on the rows observed everywhere) when
    the views share their observed rows (including any two views); with three or more views whose
    observed rows differ, each view's score is standardized on its own observed rows before
    averaging, so the target is a slightly different weighting. Partially observed rows enter with a target built from fewer
    views (noisier, same expected signal); pass complete rows only if that matters. A view with
    fewer than ``min_rows`` (default 10, at least 3) usable rows, a constant feature, or a constant
    latent column gives NaN (never 0) for the affected map entries, with a ``MissingRowsWarning``.
    ``return_info=True`` returns ``(maps, info)`` where ``info`` holds ``n_eff`` (usable rows per
    view), ``n_eff_feature`` (per view, rows behind each feature; 0 where the feature is constant),
    ``n_constant_features``, ``constant_latent_columns``, ``reasons[kind][view]`` (``None`` or why the
    map is NaN), ``relative_weight`` (as ``return_shrinkage``), ``min_rows``, ``n_rows`` and ``partial``.

    **Within-group maps** (``groups``, length-n labels, default ``None``). Features, the target and
    any ``control`` array are centered within each group (site, cohort, scanner) before any
    correlation or map is computed, so a difference in group means cannot produce an association;
    every map is then a within-group quantity (pooled within-group correlation for ``structure``).
    ``scale_within_group=False`` (default) scales globally after centering; ``True`` also divides each
    group by its own SD (one cohort's larger variance then does not dominate). With
    ``target="other_views"`` the scores come from ``sem.view_scores(groups=...)``, so the target is
    centered within group too. A group with fewer than 3 usable rows for a view (rows where the view
    and its target exist) is left out of that view, with a ``MissingRowsWarning``; if fewer than
    ``min_rows`` rows remain the view's maps are NaN with a reason (the rules above). It composes with
    ``missing_rows="mask"``. ``info["groups"]`` (``return_info=True``) records groups used and left out
    per view. ``groups=None`` leaves every output unchanged. The rotation applied is the result's
    ``loading_rotation``; for the within-group rotation merge in ``loadings(result, views, groups=g)``.

    **Excluded views** (``exclude_views``: view indices or names, default ``None``; requires
    ``target="other_views"``). The listed views are left out of the other-views target of *every*
    view: view ``m``'s target is the mean of the standardized scores of the non-excluded views other
    than ``m``. Excluded views still receive their own maps (against the target built from the
    non-excluded views); ignore them if not wanted (no separate ``skip_maps_for`` is needed). A view
    with no non-excluded other view gets NaN maps with a reason and a ``MissingRowsWarning``. This
    changes the target only: the fitted consensus ``u`` and every view's encoder were estimated with
    the excluded view in the model, so the scores of the remaining views still carry information
    pulled in from it. For a strictly non-circular test (for example when the excluded view is the
    outcome "root" view), refit without it: ``fit(method, [v for v in views if v is not outcome], k)``,
    then compute maps on that result with ``exclude_views=None`` and relate the outcome to the
    refit's latents or maps.

    Views are replayed through the fit's own preprocessing and ``u`` defaults to
    ``pysimlr.predict(result, views)["u"]``, so this works for every method and
    for held-out views alike.

    Returns ``{kind: [ (p_m, k) tensor per view ]}``.
    """
    from .api import _preprocess_views, as_views, predict
    from .sem import view_scores
    from .utils import preprocess_data
    bad = [k_ for k_ in kinds if k_ not in FEATURE_MAP_KINDS]
    if bad:
        raise ValueError(f"unknown feature map(s) {bad}; choose from {FEATURE_MAP_KINDS}.")
    if target not in ("other_views", "consensus"):
        raise ValueError(f"target must be 'other_views' or 'consensus'; got {target!r}.")
    view_list = as_views(views)
    if target == "other_views" and len(view_list) < 2:
        raise ValueError("target='other_views' needs at least two views.")
    min_rows = int(min_rows)
    if min_rows < 3:
        raise ValueError(f"min_rows must be at least 3; got {min_rows}.")
    excl = _excluded_view_indices(result, exclude_views, len(view_list))
    if excl and target != "other_views":
        raise ValueError("exclude_views only changes the other-views target; use target='other_views'.")
    n_total = int(view_list[0].shape[0])
    codes = None if groups is None else group_codes(groups, n_total)
    grouped = codes is not None
    # Uneven missingness: with ``missing_rows="mask"`` a row that is all-NaN in a view is unobserved
    # there. Every quantity is then computed on the rows where its inputs exist (branch below); with
    # no unobserved rows the original complete-data code runs unchanged.
    obs: Optional[List[np.ndarray]] = None
    if result.get("missing_rows") == "mask":
        obs = [(~torch.isnan(v.detach().cpu()).all(dim=1)).numpy() for v in view_list]
    partial = obs is not None and any(not o.all() for o in obs)
    general = partial or grouped          # the row-subset code path; complete ungrouped data takes the original one
    if general and obs is None:
        obs = [np.ones(n_total, dtype=bool) for _ in view_list]
    if partial:
        xs = [_preprocess_observed(v, o, result, m) for m, (v, o) in enumerate(zip(view_list, obs))]
    else:
        xs_t, _ = _preprocess_views(view_list, result["scale_list"], result["provenance_list"])
        xs = [x.detach().double().numpy() for x in xs_t]
    rot = result["loading_rotation"].detach().double().numpy()
    pred = predict(result, view_list) if (u is None or target == "other_views") else None
    if u is None:
        u = pred["u"]
    u_rot = np.asarray(torch.as_tensor(u).detach().double()) @ rot
    # Signs learned on these rows against `u` (not on the training rows): the maps describe
    # the columns of `u @ loading_rotation` as given.
    gkw = {"groups": groups, "scale_within_group": scale_within_group} if grouped else {}
    scores = (view_scores(result, view_list, sign_against=u, _pred=pred, **gkw)["scores"]
              if target == "other_views" else None)
    if probe_views is not None:
        if target != "other_views":
            raise ValueError("probe_views requires target='other_views' so probe features are scored against the other views.")
        from .confounds import refuse_probe_views
        refuse_probe_views(result, "feature_maps")
        p_items = list(probe_views)
        if len(p_items) != len(xs):
            raise ValueError(f"probe_views has {len(p_items)} entries; there are {len(xs)} views.")
        n_rows = int(view_list[0].shape[0])
        for m, pv in enumerate(p_items):
            if pv is not None:
                pt = _to_probe_tensor(pv, n_rows, m)
                if partial:
                    xs[m] = _preprocess_observed(pt, obs[m], result, None)
                else:
                    xp, _ = preprocess_data(pt, result["scale_list"])
                    xs[m] = xp.detach().double().numpy()
    controls = _per_view_arg(control, len(xs), "control")
    rw_shrinkages = _per_view_arg(rw_shrinkage, len(xs), "rw_shrinkage")
    out: Dict[str, List[torch.Tensor]] = {k_: [] for k_ in kinds}
    lam_used: List[np.ndarray] = []
    info: Dict[str, Any] = {"min_rows": min_rows, "n_rows": n_total, "partial": bool(partial),
                            "n_eff": [], "n_eff_feature": [], "n_constant_features": [],
                            "constant_latent_columns": [], "reasons": {k_: [] for k_ in kinds}}
    if grouped:
        info["groups"] = {"scale_within_group": bool(scale_within_group), "n_groups": [], "left_out_groups": []}
        group_labels = np.unique(np.asarray(groups.to_numpy() if hasattr(groups, "to_numpy") else groups))
    if excl:
        info["exclude_views"] = sorted(excl)
    k_lat = int(u_rot.shape[1])

    def nan_view(why, p_m, m):
        warnings.warn(f"feature_maps: {why}; its maps are NaN.", MissingRowsWarning, stacklevel=3)
        for k_ in kinds:
            out[k_].append(torch.full((p_m, k_lat), float("nan")))
            info["reasons"][k_].append(why)
        if "relative_weight" in kinds:
            lam_used.append(np.full(k_lat, np.nan))

    def one_map(k_, xz, tz, ctrl, rw_shr, m):
        """(map, lam, reason) of one kind on the rows given; lam is None except for relative_weight."""
        if k_ == "structure":
            return _corr_map(xz, tz), None, None
        if k_ == "partial":
            return _partial_map(xz, tz, shrinkage), None, None
        if k_ == "global_partial":
            if ctrl is None:
                return np.full((xz.shape[1], tz.shape[1]), np.nan), None, "control=None for this view"
            return _global_partial_map(xz, tz, ctrl), None, None
        if k_ == "component_partial":
            return _component_partial_map(xz, tz), None, None
        try:
            mp, lam_m = _relative_weight_map_lam(xz, tz, rw_shr)
            return mp, lam_m, None
        except ValueError as err:
            if not ("fewer features than rows" in str(err) or "full-rank view" in str(err)):
                raise
            # Not defined on this view (p >= n or collinear at rw_shrinkage=0): NaN, named, not a substitute map.
            warnings.warn(f"relative_weight is NaN for view {m}: {err}", RuntimeWarning, stacklevel=4)
            return (np.full((xz.shape[1], tz.shape[1]), np.nan), np.full(tz.shape[1], np.nan), str(err))

    for m, x in enumerate(xs):
        p_m = int(x.shape[1])
        oth = [i for i in range(len(xs)) if i != m and i not in excl] if target == "other_views" else None
        if not general:
            if target == "other_views" and not oth:
                info["n_eff"].append(0)
                info["n_eff_feature"].append(np.zeros(p_m, dtype=int))
                info["n_constant_features"].append(0)
                info["constant_latent_columns"].append([])
                nan_view(f"view {m}: no non-excluded other view to build its other-views target", p_m, m)
                continue
            xz = _zscore(x)
            if target == "other_views":
                tz = _zscore(sum(z for i, z in enumerate(scores) if i != m and i not in excl))
            else:
                tz = _zscore(u_rot)
            info["n_eff"].append(n_total)
            info["n_eff_feature"].append(np.full(p_m, n_total, dtype=int))
            info["n_constant_features"].append(int((x.std(0) <= 0).sum()))
            info["constant_latent_columns"].append([])
            for k_ in kinds:
                mp, lam_m, why = one_map(k_, xz, tz, controls[m], rw_shrinkages[m], m)
                if k_ == "relative_weight":
                    lam_used.append(lam_m)
                info["reasons"][k_].append(why)
                out[k_].append(torch.as_tensor(mp).float())
            continue
        # Rows where view m is observed and its target exists. The other-views target at a row is the
        # mean of the standardized scores of the *other* views observed at that row (undefined where
        # none is); for complete rows it equals the all-views sum up to a constant, which the z-score removes.
        if target == "other_views" and not oth:
            t_raw = np.full((n_total, k_lat), np.nan)
            rows = np.zeros(n_total, dtype=bool)
        elif target == "other_views":
            stack = np.stack([scores[i] for i in oth])      # (n_other, n, k), NaN rows
            fin = np.isfinite(stack).all(axis=2)
            cnt = fin.sum(0)
            t_raw = np.where(fin[..., None], stack, 0.0).sum(0) / np.maximum(cnt, 1)[:, None]
            rows = obs[m] & (cnt > 0)
        else:
            t_raw = u_rot
            rows = obs[m] & np.isfinite(u_rot).all(axis=1)
        if grouped:
            small = ~_drop_small_groups(codes[rows])
            hit = set(np.unique(codes[rows][small]).tolist())
            if small.any():
                drop_full = np.zeros(n_total, dtype=bool)
                drop_full[np.flatnonzero(rows)[small]] = True
                rows = rows & ~drop_full
            # groups with rows observed in this view that lost every usable row (their scores were NaN in the
            # other views, see sem.view_scores) are reported too
            hit |= set(np.unique(codes[obs[m]]).tolist()) - set(np.unique(codes[rows]).tolist())
            lost = sorted(str(x_) for x_ in group_labels[sorted(hit)])
            if hit:
                warnings.warn(f"feature_maps: view {m}: group(s) {lost} have fewer than 3 usable rows (rows where the "
                              "view and its target exist); they were left out.", MissingRowsWarning, stacklevel=2)
            info["groups"]["left_out_groups"].append(lost)
            info["groups"]["n_groups"].append(int(np.unique(codes[rows]).shape[0]))
        n_d = int(rows.sum())
        xd, td = x[rows], t_raw[rows]
        if grouped and n_d:
            xd = center_within_groups(xd, codes[rows], scale_within_group)
            td = center_within_groups(td, codes[rows], scale_within_group)
        f_const = ~(xd.std(0) > 0) if n_d else np.ones(p_m, dtype=bool)
        t_const = ~(td.std(0) > 0) if n_d else np.ones(k_lat, dtype=bool)
        info["n_eff"].append(n_d)
        info["n_eff_feature"].append(np.where(f_const, 0, n_d).astype(int))
        info["n_constant_features"].append(int(f_const.sum()))
        info["constant_latent_columns"].append([int(j) for j in np.where(t_const)[0]])
        why_all = None
        if target == "other_views" and not oth:
            why_all = f"view {m}: no non-excluded other view to build its other-views target"
        elif n_d < min_rows:
            why_all = f"only {n_d} rows with view {m} and its {target} target observed (min_rows={min_rows})"
        elif f_const.all() or t_const.all():
            why_all = (f"view {m}: every feature is constant on the {n_d} usable rows" if f_const.all()
                       else f"view {m}: every latent column is constant on the {n_d} usable rows")
        if why_all is not None:
            warnings.warn(f"feature_maps: {why_all}; its maps are NaN.", MissingRowsWarning, stacklevel=2)
            for k_ in kinds:
                out[k_].append(torch.full((p_m, k_lat), float("nan")))
                info["reasons"][k_].append(why_all)
            if "relative_weight" in kinds:
                lam_used.append(np.full(k_lat, np.nan))
            continue
        if f_const.any() or t_const.any():
            warnings.warn(f"feature_maps: view {m} has {int(f_const.sum())} constant feature(s) and "
                          f"{int(t_const.sum())} constant latent column(s) on its {n_d} usable rows; "
                          "their map entries are NaN.", MissingRowsWarning, stacklevel=2)
        keep_f, keep_t = ~f_const, ~t_const
        xz, tz = _zscore(xd[:, keep_f]), _zscore(td[:, keep_t])
        ctrl = controls[m]
        if ctrl is not None and not isinstance(ctrl, str):
            ctrl = np.asarray(ctrl, dtype=np.float64)
            if ctrl.shape[0] == n_total:
                ctrl = ctrl[rows]
                if grouped:
                    ctrl = center_within_groups(ctrl, codes[rows], scale_within_group)
        for k_ in kinds:
            mp_k, lam_m, why = one_map(k_, xz, tz, ctrl, rw_shrinkages[m], m)
            mp = np.full((p_m, k_lat), np.nan)
            mp[np.ix_(keep_f, keep_t)] = mp_k
            if k_ == "relative_weight":
                lam_full = np.full(k_lat, np.nan)
                if lam_m is not None:
                    lam_full[keep_t] = lam_m
                lam_used.append(lam_full)
            info["reasons"][k_].append(why)
            out[k_].append(torch.as_tensor(mp).float())
    if return_shrinkage or return_info:
        extra: Dict[str, Any] = {"relative_weight": [torch.as_tensor(lm).float() for lm in lam_used]}
        if return_info:
            extra.update(info)
        return out, extra
    return out


def _procrustes(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||a R - ref||_F."""
    uu, _, vh = np.linalg.svd(a.T @ ref)
    return uu @ vh


#: Rank selection rules exposed by `select_rank`.
RANK_CRITERIA = ("component", "subspace", "oos_r2", "oos_r2_1se", "oos_coupling", "heldout_coupling")

_FISHER_Z_001 = 2.3263478740408408


def _principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Descending principal cosines (length ``k``) between col(a) and col(b)."""
    k = a.shape[1]
    ua, sa, _ = np.linalg.svd(a, full_matrices=False)
    ub, sb, _ = np.linalg.svd(b, full_matrices=False)
    ma = sa > 1e-12 * max(1.0, float(sa[0]) if len(sa) else 1.0)
    mb = sb > 1e-12 * max(1.0, float(sb[0]) if len(sb) else 1.0)
    out = np.zeros(k, dtype=float)
    if np.any(ma) and np.any(mb):
        sv = np.linalg.svd(ua[:, ma].T @ ub[:, mb], compute_uv=False)
        out[:len(sv)] = np.clip(sv, 0.0, 1.0)
    return out


def _masked_coupling(z_std: Sequence[np.ndarray], min_rows: int = 3) -> Tuple[List[float], List[int]]:
    """Leave-one-view-out coupling when views have unobserved (NaN) rows.

    For view ``m`` the other-views score at a row is the mean of the other views' scores observed at that
    row; the correlation with ``z_m`` is taken on the rows where both exist (centered on those rows). A
    view with fewer than ``min_rows`` such rows, or a constant score, gives NaN for that component and is
    left out of the mean over views; if no view is left the component is NaN. Returns the per-component
    coupling and the number of rows used per view.
    """
    k = z_std[0].shape[1]
    per_view, n_used = [], []
    for m, zm in enumerate(z_std):
        oth = np.stack([z for i, z in enumerate(z_std) if i != m])        # (M-1, n, k)
        fin = np.isfinite(oth).all(axis=2)
        cnt = fin.sum(0)
        other = np.where(fin[..., None], oth, 0.0).sum(0) / np.maximum(cnt, 1)[:, None]
        rows = np.isfinite(zm).all(axis=1) & (cnt > 0)
        n_d = int(rows.sum())
        n_used.append(n_d)
        if n_d < min_rows:
            per_view.append(np.full(k, np.nan))
            continue
        a, b = zm[rows], other[rows]
        a, b = a - a.mean(0), b - b.mean(0)
        den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        per_view.append(np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1.0), np.nan))
    pv = np.asarray(per_view)
    ok = np.isfinite(pv)
    cnt_ok = ok.sum(0)
    coup = np.where(cnt_ok > 0, np.where(ok, pv, 0.0).sum(0) / np.maximum(cnt_ok, 1), np.nan)
    return [float(x) for x in coup], n_used


def _oob_metrics(res: Dict[str, Any], sub_views: Sequence[torch.Tensor],
                 oob_views: Optional[Sequence[torch.Tensor]], return_scores: bool = False,
                 _info: Optional[Dict[str, Any]] = None, _groups: Any = None,
                 _scale_within_group: bool = False):
    """Held-out reconstruction R^2 (adjusted for random k-plane self-projection)
    and leave-one-view-out coupling on unused rows. Scores come from `sem.view_scores`
    (rotated, signed on the refit's training rows, standardised on the unused rows);
    ``return_scores=True`` also returns them (the hook `sem.path_model(inference=...)` reads)."""
    if oob_views is None or oob_views[0].shape[0] < 2:
        return (None, None, None) if return_scores else (None, None)
    from .api import predict
    from .sem import view_scores
    pred = predict(res, oob_views)
    k = res["structure"][0].shape[1]
    ps = [int(x.shape[1]) for x in sub_views]
    p_tot, m_views = sum(ps), len(ps)
    r2 = []
    for raw, pm in zip(pred["r2"], ps):
        d0 = min(0.999, (k / p_tot) if res.get("basis_is_joint") else (k / (m_views * pm)))
        r2.append(float((float(raw) - d0) / (1.0 - d0)))
    if m_views < 2 or oob_views[0].shape[0] < 3:
        return (r2, None, None) if return_scores else (r2, None)
    gkw = {} if _groups is None else {"groups": _groups, "scale_within_group": _scale_within_group}
    z_std = view_scores(res, list(oob_views), train_views=list(sub_views), _pred=pred, **gkw)["scores"]
    if all(np.isfinite(z).all() for z in z_std):
        tot = sum(z_std)
        per_view = []
        for zm in z_std:
            other = tot - zm
            den = np.sqrt((zm ** 2).sum(0) * (other ** 2).sum(0))
            per_view.append(np.where(den > 0, (zm * other).sum(0) / np.where(den > 0, den, 1.0), 0.0))
        coup = [float(x) for x in np.mean(per_view, axis=0)]
        n_used = [int(zm.shape[0]) for zm in z_std]
    else:
        coup, n_used = _masked_coupling(z_std)
    if _info is not None:
        _info["coupling_n"] = n_used
    if return_scores:
        return r2, coup, [z.astype(np.float32) for z in z_std]
    return r2, coup


def _stability_q(p_w: int, expected_false: float, threshold: float) -> int:
    return max(1, min(p_w, int(np.floor(np.sqrt(expected_false * (2 * threshold - 1) * p_w)))))


def _top_q_mask(h: np.ndarray, q: int) -> np.ndarray:
    sel = np.zeros(h.shape[0], bool)
    sel[np.argsort(-h)[:q]] = True
    return sel


def _col_corr_stability(al_slice: np.ndarray, ref_mat: np.ndarray, *, nan_safe: bool = False) -> List[float]:
    if al_slice.shape[0] > 0 and (
        not nan_safe or (np.isfinite(al_slice).all() and np.isfinite(ref_mat).all())
    ):
        ac = al_slice - al_slice.mean(axis=1, keepdims=True)
        rc = ref_mat - ref_mat.mean(axis=0, keepdims=True)
        num = (ac * rc[None, :, :]).sum(axis=1)
        den = np.sqrt((ac * ac).sum(axis=1) * (rc * rc).sum(axis=0)[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            rs_mat = np.where(den > 0, num / den, np.nan)
        return [float(np.nanmedian(rs_mat[:, j])) for j in range(ref_mat.shape[1])]
    stab = []
    for j in range(ref_mat.shape[1]):
        rs = []
        for t in range(al_slice.shape[0]):
            x_, y_ = al_slice[t, :, j], ref_mat[:, j]
            ok = np.isfinite(x_) & np.isfinite(y_)
            rs.append(np.corrcoef(x_[ok], y_[ok])[0, 1] if ok.sum() > 2 and x_[ok].std() > 0
                      and y_[ok].std() > 0 else np.nan)
        stab.append(float(np.nanmedian(rs)) if np.isfinite(rs).any() else float("nan"))
    return stab


def _nanmedian_cols(rows: Sequence[Sequence[float]]) -> List[float]:
    """Column medians ignoring NaN; a column that is NaN in every row is NaN (no warning)."""
    a = np.asarray(rows, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(a, axis=0)
    return [float(x) for x in med]


def _nearest_signed_permutation(rot: np.ndarray) -> np.ndarray:
    """Signed permutation matrix nearest an orthogonal ``rot`` (Hungarian match on ``|rot|``, signs of the matched entries)."""
    from scipy.optimize import linear_sum_assignment
    rows_m, cols_m = linear_sum_assignment(-np.abs(rot))
    perm = np.zeros_like(rot)
    perm[rows_m, cols_m] = np.sign(rot[rows_m, cols_m])
    return perm


def _proportional_allocation(sizes: np.ndarray, total: int) -> np.ndarray:
    """Integer draws per stratum summing to ``total``, proportional to ``sizes`` (largest remainder; ties to the
    earlier stratum), never above a stratum's size."""
    exact = sizes * (float(total) / float(sizes.sum()))
    base = np.minimum(np.floor(exact).astype(int), sizes)
    left = int(total - base.sum())
    if left > 0:
        order = np.argsort(-(exact - base), kind="stable")
        for g in order:
            if left == 0:
                break
            if base[g] < sizes[g]:
                base[g] += 1
                left -= 1
    return base


def _draw_subsample(rng: np.random.Generator, n: int, m: int, scheme: str,
                    strata_codes: Optional[np.ndarray]) -> np.ndarray:
    """Sorted row indices of one resample. Unstratified without replacement is the historical draw
    (``rng.choice(n, m, replace=False)``) and must stay bit-identical."""
    boot = scheme == "bootstrap"
    if strata_codes is None:
        return np.sort(rng.choice(n, n if boot else m, replace=boot))
    rows_by = [np.flatnonzero(strata_codes == g) for g in np.unique(strata_codes)]
    sizes = np.asarray([len(r) for r in rows_by])
    take = sizes if boot else _proportional_allocation(sizes, m)
    parts = [rng.choice(r, int(t), replace=boot) for r, t in zip(rows_by, take) if t > 0]
    return np.sort(np.concatenate(parts))


REFIT_OUTPUTS = ("weights", "maps", "latents", "rows")
SUBSAMPLE_SCHEMES = ("half", "fraction", "bootstrap")


def loading_inference(method: str, views: Any, k: int, *, n_subsamples: int = 100,
                      fraction: float = 0.5, threshold: float = 0.8,
                      expected_false: float = 1.0,
                      n_permutations: int = 0, exclude_unconverged: bool = False,
                      random_state: int = 0, n_jobs: int = 1,
                      maps: Sequence[str] = (), map_options: Optional[Dict[str, Any]] = None,
                      groups: Any = None, scale_within_group: bool = False,
                      subsample: str = "half", strata: Any = None,
                      return_refits: Sequence[str] = (), max_refit_bytes: int = 2_000_000_000,
                      _executor: Any = None, _warn_unconverged: bool = True,
                      **params: Any) -> Dict[str, Any]:
    r"""Uncertainty and selection for `structure`, by refitting the whole pipeline.

    Each of ``n_subsamples`` refits uses a random ``fraction`` of the rows
    (without replacement). Its structure coefficients are rotated onto the
    reference fit's (orthogonal Procrustes on all views stacked), since the
    latent space fixes components only up to rotation. Every subsample refit
    also evaluates `pysimlr.predict` on its unused (out-of-bag) complement at
    no extra fit cost. With ``exclude_unconverged=True``, subsample refits with
    ``converged=False`` are omitted from stability, interval, selection and
    out-of-bag aggregates, and ``n_effective_refits`` records how many remained.
    A ``RuntimeWarning`` is emitted when ``converged_fraction < 0.5``.

    Returns
    -------
    dict
        ``result`` (the reference `pysimlr.fit` on all rows),
        ``subspace_stability`` (length ``k``: median over refits of the
        principal cosines between the refit and reference stacked structure
        subspaces, rotation-invariant), ``oos_r2`` (per-view mean held-out
        reconstruction :math:`R^2` across out-of-bag splits, or ``None``),
        ``oos_r2_splits`` (length ``n_subsamples`` view-averaged held-out
        :math:`R^2` per split, or ``None``), ``oos_coupling`` (length ``k``
        median held-out leave-one-view-out score correlation per component, or
        ``None``), ``oob_scores`` (per subsample refit: ``rows``, the out-of-bag row
        indices, and ``scores``, its float32 `sem.view_scores` on them; read by
        `sem.path_model(inference=...)`), ``n_rows``, ``converged_fraction`` and ``effective_rank`` (median over
        refits), and per view: ``structure`` (reference), ``ci_low`` /
        ``ci_high`` (2.5/97.5% percentiles of the aligned refits; subsamples of
        half the rows make these conservative), ``component_stability`` (length
        ``k``: median over refits of the correlation between the aligned and
        reference column; per-component claims need this high, e.g. >= 0.9),
        ``selection_frequency`` and ``selected`` (stability selection,
        Meinshausen & Buhlmann 2010: each refit selects the ``q_m`` features of
        largest communality; ``selected`` is frequency >= ``threshold``),
        ``q`` and ``expected_false_selections`` (the bound
        :math:`q^2 / ((2\pi - 1) p)`, valid for uniform half-samples, ``fraction = 0.5``,
        and ``threshold > 0.5``; NaN under any other resampling design, see below), and, if ``n_permutations > 0``,
        ``null_max_abs`` (95th percentile of :math:`\max_{f,j} |C|` with the
        rows of every view but the first permuted independently, i.e. no
        shared signal).

    Unobserved rows (``missing_rows="mask"`` passed to the fit). Subsamples keep the NaN rows.
    ``oos_coupling`` is computed per held-out split on the rows where the needed views are observed
    (view ``m`` against the mean of the other views observed at that row; at least 3 such rows, else NaN
    for that view and left out of the mean); a component that cannot be computed in any split is NaN, never
    0. ``oos_coupling_n`` is the median over splits of the rows used per view, and every ``maps[kind]``
    entry carries ``n_eff`` / ``n_eff_feature`` / ``reasons`` from `feature_maps(return_info=True)` (pass
    ``min_rows`` through ``map_options``).

    Alternative feature maps. ``maps`` (kinds from `feature_maps`, with
    ``map_options`` passed to it: ``target``, ``shrinkage``, ``control``,
    ``rw_shrinkage``, ``probe_views``) are computed on the reference fit and on
    every subsample refit at no extra fits, and reported under
    ``out["maps"][kind]`` with per view ``map`` (reference), ``ci_low`` /
    ``ci_high``, ``component_stability`` and ``selection_frequency`` (the same
    ``q_m`` rule, ranking features by the sum over components of the squared
    map). With ``map_options={"probe_views": ...}``, each subsample refit fits
    on the well-conditioned ``views`` and evaluates ``maps`` on the matching
    subsample rows of ``probe_views`` against the other views' scores. Each
    refit's columns are matched to the reference by the signed permutation
    nearest its structure Procrustes rotation (the varimax frames are canonical
    up to order and sign; a full rotation would mix the squared
    ``relative_weight`` shares), without the sign for ``relative_weight``. A
    map's stability is its own: precision maps can be far less stable than
    ``structure``.

    Speed. ``n_jobs`` runs the refits (and permutation fits) in worker
    processes (``-1``: all cores). All subsample indices are drawn in the parent
    in a fixed order, so the result does not depend on ``n_jobs``. Each refit
    tunes its own learning rate; reusing the reference fit's instead was
    measured and rejected: 1.3x faster, but it moved an under-identified
    component's stability by 0.18 (``docs/audit/evidence/select_rank_speed.csv``).

    Within-group inference (``groups``, length-n labels, default ``None``; ``scale_within_group``
    as in `structure_coefficients`). The reference fit and every refit get within-group
    ``structure`` / ``loadings`` / ``loading_rotation`` (`loadings(..., groups=)`; ``out["result"]``
    carries those keys, not the pooled ones), the maps are computed within group
    (`feature_maps(groups=)`, with the subsample's labels), the out-of-bag ``oos_coupling`` is
    computed on within-group scores, and the permutation null (``n_permutations``) permutes rows
    only within group (one `utils.within_group_permutation` draw per view, so ``null_max_abs`` is a
    null for within-group association). ``oos_r2`` (reconstruction) is not grouped. Resampling is
    not stratified by ``groups`` unless you also pass ``strata=groups``; with small groups do, so
    every refit sees every group.

    Resampling scheme (``subsample``, ``fraction``, ``strata``). ``"half"`` (default) draws
    ``round(fraction * n)`` rows without replacement (at least ``k + 2``); at the default
    ``fraction=0.5`` these are half-samples and the draws are exactly those of earlier versions
    (same seed, same rows). ``"fraction"`` is the same draw, the explicit spelling for
    ``fraction != 0.5`` (``"half"`` with another fraction keeps working for callers that already
    pass one, such as `select_rank`). ``"bootstrap"`` draws ``n`` rows with replacement
    (``fraction`` is not used); the out-of-bag rows are the never-drawn rows (about 37%), the refit
    sees duplicated rows, and the intervals are percentile-bootstrap intervals, not the conservative
    half-sample ones. ``strata`` (length-n labels) makes every draw proportional within strata
    (largest-remainder allocation of the total, so stratum shares are kept to rounding; for the
    bootstrap, each stratum is resampled to its own size). Stratifying by the view-observation pattern
    under ``missing_rows="mask"`` is not a built-in option; build the labels and pass them:
    ``strata = [''.join('1' if o else '0' for o in obs) for obs in zip(*[~v.isnan().all(1) for v in views])]``.
    Guarantees: only uniform half-sampling (``subsample`` in {"half", "fraction"}, ``fraction == 0.5``,
    ``strata=None``) carries the stability-selection false-selection bound; for every other design
    ``expected_false_selections`` is NaN per view and ``expected_false_selections_reason`` says why
    (``None`` when the bound is reported), ``out["resampling"]`` records the design and
    ``bound_valid``. ``selected`` and ``selection_frequency`` are still computed (as a descriptive
    stability measure with no error-rate guarantee), and so are intervals and component stability
    (empirical resampling variability, with no coverage guarantee: a bootstrap or large-fraction
    subsample understates the variability of the estimator that half-samples would show).

    Per-refit outputs (``return_refits``, default ``()``: nothing is kept). Any of
    ``"weights"``, ``"maps"`` (needs ``maps=``), ``"latents"``, ``"rows"``. ``out["refits"]`` then
    holds, for the ``B`` refits kept (unconverged ones are dropped under ``exclude_unconverged``;
    ``index`` gives their position among the draws): ``structure`` (``(B, sum p, k)`` float32, the
    aligned structure the summaries use) and ``view_sizes``; ``alignment`` with ``rotation``
    (``(B, k, k)``: ``u_refit @ rotation[b]`` is the refit's latent in the reference frame; undo with
    ``@ rotation[b].T``, it is orthogonal), ``procrustes`` (the structure alignment), ``loading_rotation``
    (the refit's own varimax rotation; ``rotation = loading_rotation @ procrustes``) and
    ``signed_permutation`` (what the maps are aligned with; its absolute value for
    ``relative_weight``); ``weights`` (per view, ``(B, p_m, k)``, the encoders ``V_m @ rotation[b]``);
    ``maps[kind]`` (per view ``(B, p_m, k)``, aligned as the summary maps are); ``latents``
    (``(B, n, k)``: the consensus latent of every row of the full data in the reference frame; in-bag
    rows from the refit's own ``u`` (first draw of a duplicated row), out-of-bag rows from
    `predict`); ``in_bag`` / ``n_draws`` (``(B, n)``, with ``rows`` or ``latents``); ``rows`` and
    ``oob_rows`` (lists of index arrays, ``rows`` with duplicates under the bootstrap). The reference
    frame is the one used for the stability summaries (Procrustes onto the full-data fit's structure),
    chosen from the full-data fit alone and so independent of the order of the draws. Memory: about
    ``4 B (sum_m p_m k [structure] + sum_m p_m k [weights] + len(maps) sum p_m k [maps] + n k [latents])
    + 5 B n [rows]`` bytes; the call raises a ``ValueError`` before fitting if that exceeds
    ``max_refit_bytes`` (default 2e9).

    ``q_m``, the number of features each refit selects in view ``m``, is set
    from the tolerated number of false selections (``expected_false``, per
    view) by inverting the bound: ``q_m = floor(sqrt(expected_false *
    (2 threshold - 1) p_m))``. Tying ``q`` to the fit's sparsity instead (half
    of the features) made the bound vacuous: on ADNI-MCI it promised ~165
    expected false selections among ~170 selected.
    """
    from scipy.optimize import linear_sum_assignment
    from .api import as_views, fit
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction}.")
    if not 0.5 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0.5, 1]; got {threshold}.")
    view_list = as_views(views)
    n = view_list[0].shape[0]
    if subsample not in SUBSAMPLE_SCHEMES:
        raise ValueError(f"subsample must be one of {SUBSAMPLE_SCHEMES}; got {subsample!r}.")
    want = tuple(return_refits)
    bad_w = [w for w in want if w not in REFIT_OUTPUTS]
    if bad_w:
        raise ValueError(f"unknown return_refits entr{'y' if len(bad_w) == 1 else 'ies'} {bad_w}; choose from {REFIT_OUTPUTS}.")
    g_codes = None if groups is None else group_codes(groups, n)
    s_codes = None if strata is None else group_codes(strata, n, name="strata")
    if map_options and "groups" in map_options:
        raise ValueError("pass groups= to loading_inference, not through map_options.")
    rng = np.random.default_rng(random_state)
    maps = tuple(maps)
    bad = [k_ for k_ in maps if k_ not in FEATURE_MAP_KINDS]
    if bad:
        raise ValueError(f"unknown feature map(s) {bad}; choose from {FEATURE_MAP_KINDS}.")
    if "maps" in want and not maps:
        raise ValueError("return_refits includes 'maps' but no map kinds were requested (maps=...).")
    map_opts = dict(map_options or {}, kinds=maps) if maps else None
    probe_tensors = None
    if map_opts and map_opts.get("probe_views") is not None:
        p_items = list(map_opts["probe_views"])
        if len(p_items) != len(view_list):
            raise ValueError(f"probe_views has {len(p_items)} entries; there are {len(view_list)} views.")
        probe_tensors = [None if pv is None else _to_probe_tensor(pv, n, vi) for vi, pv in enumerate(p_items)]
        map_opts = dict(map_opts, probe_views=probe_tensors)
    ref = fit(method, view_list, k, random_state=random_state, **params)
    if g_codes is not None:
        ref = _with_grouped_loadings(ref, view_list, g_codes, scale_within_group)
        map_opts = None if map_opts is None else dict(map_opts, groups=g_codes, scale_within_group=scale_within_group)
    c_ref = [c.double().numpy() for c in ref["structure"]]
    stacked_ref = np.vstack(c_ref)
    if expected_false <= 0:
        raise ValueError(f"expected_false must be positive; got {expected_false}.")
    qs = [_stability_q(c.shape[0], expected_false, threshold) for c in c_ref]
    sizes = np.cumsum([0] + [c.shape[0] for c in c_ref])
    aligned, chosen = [], [[] for _ in c_ref]
    m = max(k + 2, int(round(fraction * n)))
    refit_params = dict(params)
    # Confounds follow their rows: a subsample refit gets the subsample's
    # confound rows, and a permutation refit permutes each view together with
    # its own confound rows, so every view of a null refit is cleaned exactly
    # as in the observed fit (`pysimlr.confounds.permute_view_confounds`).
    conf = refit_params.get("confounds")
    from .confounds import permute_view_confounds, slice_confounds
    # Draw every index in the parent, in a fixed order, before any refit runs:
    # the result is then the same for every n_jobs.
    # The default scheme is the historical draw; the bound below is only reported for it.
    sub_idx = [_draw_subsample(rng, n, m, subsample, s_codes) for _ in range(int(n_subsamples))]
    if g_codes is None:
        perm_idx = [[rng.permutation(n) for _ in view_list[1:]] for _ in range(int(n_permutations))]
    else:
        # Within-group permutation null: a row only moves to a position of its own group, so group
        # structure shared by the views stays in the null (the groups= option of this function).
        perm_idx = [[within_group_permutation(rng, g_codes) for _ in view_list[1:]]
                    for _ in range(int(n_permutations))]
    all_rows = np.arange(n)
    if want:
        p_widths = [int(x.shape[1]) for x in view_list] if probe_tensors is None else [
            int((view_list[i] if pt is None else pt).shape[1]) for i, pt in enumerate(probe_tensors)]
        n_beff = int(n_subsamples)
        est = 4 * n_beff * (sum(p_widths) * k                                   # aligned structure
                            + ("weights" in want) * sum(int(x.shape[1]) for x in view_list) * k
                            + ("maps" in want) * len(maps) * sum(p_widths) * k
                            + ("latents" in want) * n * k) + (n_beff * n * 5 if want else 0)
        if est > int(max_refit_bytes):
            raise ValueError(f"return_refits={want} would hold about {est / 1e6:.0f} MB for {n_beff} refits "
                             f"(n={n}, k={k}), above max_refit_bytes={int(max_refit_bytes)}; request fewer outputs or "
                             "refits, or raise max_refit_bytes.")
    use_extra = g_codes is not None or bool(want)
    tasks, oob_rows = [], []
    for idx in sub_idx:
        oob = np.setdiff1d(all_rows, np.unique(idx) if subsample == "bootstrap" else idx, assume_unique=True)
        oob_rows.append(oob)
        t_idx = torch.as_tensor(idx)
        t_oob = torch.as_tensor(oob) if len(oob) >= 2 else None
        sub_v = [x[t_idx].numpy() if x.device.type == "cpu" else x[t_idx] for x in view_list]
        oob_v = (
            [x[t_oob].numpy() if x.device.type == "cpu" else x[t_oob] for x in view_list]
            if t_oob is not None else None
        )
        sub_p = refit_params if conf is None else dict(refit_params, confounds=slice_confounds(conf, idx))
        task = (method, sub_v, oob_v, k, random_state, sub_p)
        if map_opts:
            t_opts = (dict(map_opts, probe_views=[None if pt is None else pt[t_idx]
                                                  for pt in probe_tensors])
                      if probe_tensors is not None else map_opts)
            if g_codes is not None:
                t_opts = dict(t_opts, groups=g_codes[idx])
            task = task + (t_opts,)
        if use_extra:
            extra = {"groups": None if g_codes is None else g_codes[idx],
                     "oob_groups": None if g_codes is None or len(oob) < 2 else g_codes[oob],
                     "scale_within_group": bool(scale_within_group), "want": want}
            task = task + ((None,) if len(task) == 6 else ()) + (extra,)
        tasks.append(task)
    tasks += [(method,
               ([view_list[0].numpy()] if view_list[0].device.type == "cpu" else [view_list[0]])
               + [x[torch.as_tensor(pi)].numpy() if x.device.type == "cpu" else x[torch.as_tensor(pi)]
                  for x, pi in zip(view_list[1:], pis)],
               None, k, random_state,
               refit_params if conf is None else
               dict(refit_params, confounds=permute_view_confounds(conf, [None] + list(pis))))
              + ((None, {"groups": g_codes, "oob_groups": None, "scale_within_group": bool(scale_within_group),
                         "want": ()}) if g_codes is not None else ())
              for pis in perm_idx]
    records = _run_fits(tasks, n_jobs, _executor)
    sub_records = records[:len(sub_idx)]
    conv_flags = [bool(rec["converged"]) for rec in sub_records]
    conv_steps = [int(rec["n_steps"]) for rec in sub_records
                  if rec["converged"] and rec.get("n_steps") is not None]
    median_conv_iter = float(np.median(conv_steps)) if conv_steps else None
    conv_frac = float(np.mean(conv_flags)) if conv_flags else 1.0
    from .extension import _used
    used_iter = params.get("iterations")
    if used_iter is None:
        used_iter = _used(ref, "iterations", (ref.get("convergence") or {}).get("max_steps"))
    if _warn_unconverged and conv_frac < 0.5:
        warnings.warn(
            f"loading_inference: only {sum(conv_flags)}/{len(conv_flags)} ({conv_frac:.0%}) subsample refits "
            f"converged; raise iterations; this run used {used_iter}.",
            RuntimeWarning, stacklevel=2,
        )
    cosines, r2_splits, coups, eff_ranks = [], [], [], []
    coup_splits_all: List[Optional[List[float]]] = []
    kept_oob_scores = []
    coup_n = [rec["oos_coupling_n"] for rec in sub_records if rec.get("oos_coupling_n") is not None]
    map_aligned = {kk: [] for kk in maps}
    kept_idx: List[int] = []
    rf: Dict[str, List[Any]] = {"rotation": [], "procrustes": [], "loading_rotation": [], "signed_permutation": [],
                                "weights": [], "latents": [], "in_bag": [], "n_draws": [], "rows": [], "oob_rows": [],
                                "maps": {kk: [] for kk in maps}}
    for b_i, (rows_, rec) in enumerate(zip(oob_rows, sub_records)):
        keep_rec = (not exclude_unconverged) or bool(rec["converged"])
        st_list = rec["structure"]
        st = np.vstack(st_list)
        if keep_rec and rec["oos_coupling"] is not None:
            _, col_ind = linear_sum_assignment(-np.abs(st.T @ stacked_ref))
            inv = np.empty_like(col_ind)
            inv[col_ind] = np.arange(k)
            c_aligned = np.asarray(rec["oos_coupling"], float)[inv]
            coups.append(c_aligned)
            coup_splits_all.append([float(x) for x in c_aligned])
        else:
            coup_splits_all.append(None)
        if not keep_rec:
            continue
        rp = _procrustes(st, stacked_ref)
        aligned.append(st @ rp)
        perm = _nearest_signed_permutation(rp) if (maps or want) else None
        if maps:
            for kk in maps:
                mk = np.vstack(rec["maps"][kk])
                map_aligned[kk].append(mk @ (np.abs(perm) if kk == "relative_weight" else perm))
        if want:
            kept_idx.append(b_i)
            a_b = rec["loading_rotation"] @ rp          # raw refit frame -> reference frame
            rf["rotation"].append(a_b)
            rf["procrustes"].append(rp)
            rf["loading_rotation"].append(rec["loading_rotation"])
            rf["signed_permutation"].append(perm)
            idx_b = sub_idx[b_i]
            if "rows" in want or "latents" in want:
                draws = np.bincount(idx_b, minlength=n)
                rf["n_draws"].append(draws.astype(np.int32))
                rf["in_bag"].append(draws > 0)
            if "rows" in want:
                rf["rows"].append(idx_b)
                rf["oob_rows"].append(rows_)
            if "weights" in want:
                rf["weights"].append([w @ a_b for w in rec["weights"]])
            if "latents" in want:
                lat = np.full((n, k), np.nan)
                first = np.unique(idx_b, return_index=True)       # first draw of each in-bag row
                lat[first[0]] = rec["u_train"][first[1]] @ a_b
                if rec["u_oob"] is not None:
                    lat[rows_] = rec["u_oob"] @ a_b
                rf["latents"].append(lat)
            if "maps" in want:
                for kk in maps:
                    rf["maps"][kk].append(map_aligned[kk][-1])
        cosines.append(_principal_cosines(st, stacked_ref))
        eff_ranks.append(rec["effective_rank"])
        if rec["oos_r2"] is not None:
            r2_splits.append(rec["oos_r2"])
        if rec.get("oob_scores") is not None:
            kept_oob_scores.append({"rows": rows_, "scores": rec["oob_scores"]})
        for vi, c in enumerate(st_list):
            chosen[vi].append(_top_q_mask((c ** 2).sum(1), qs[vi]))
    n_eff = len(aligned)
    er_ref_key = "joint_effective_rank" if ref.get("basis_is_joint") else "effective_rank"
    if n_eff > 0:
        al = np.stack(aligned)                      # (B_eff, p_total, k)
        lo, hi = np.percentile(al, (2.5, 97.5), axis=0)
        r2_arr = np.asarray(r2_splits, float) if r2_splits else None
        sub_stab = [float(x) for x in np.nanmedian(cosines, axis=0)]
        eff_rk = [float(x) for x in np.nanmedian(eff_ranks, axis=0)]
    else:
        al = None
        lo = hi = np.full(stacked_ref.shape, np.nan, dtype=float)
        r2_arr = None
        sub_stab = [float("nan")] * int(k)
        eff_rk = [float("nan")] * len(ref[er_ref_key])
    out = {"result": ref, "structure": ref["structure"], "ci_low": [], "ci_high": [],
           "component_stability": [], "subspace_stability": sub_stab,
           "oos_r2": ([float(x) for x in np.nanmean(r2_arr, axis=0)] if r2_arr is not None else None),
           "oos_r2_splits": ([float(x) for x in np.nanmean(r2_arr, axis=1)] if r2_arr is not None else None),
           "oos_coupling": (_nanmedian_cols(coups) if coups else None),
           "oos_coupling_n": ([float(x) for x in np.median(coup_n, axis=0)] if coup_n else None),
           "oos_coupling_splits": (coup_splits_all if coups else None),
           "oos_n": (int(n - m) if subsample != "bootstrap" and s_codes is None
                     else int(np.median([len(o) for o in oob_rows])) if oob_rows else 0),
           "n_rows": int(n),
           "oob_scores": kept_oob_scores,
           "converged_fraction": conv_frac,
           "median_converged_iter": median_conv_iter,
           "n_effective_refits": int(n_eff),
           "used_iterations": used_iter,
           "effective_rank": eff_rk,
           "selection_frequency": [], "selected": [],
           "q": qs, "expected_false_selections": []}
    # The Meinshausen-Buhlmann bound is derived for subsamples of size n/2 drawn uniformly without replacement.
    bound_reason = None
    if subsample == "bootstrap":
        bound_reason = "bootstrap resampling (with replacement): the half-sampling bound does not apply"
    elif s_codes is not None:
        bound_reason = "stratified draws: the half-sampling bound was derived for uniform draws"
    elif abs(fraction - 0.5) > 1e-12:
        bound_reason = f"fraction={fraction} (the bound holds for half-samples, fraction=0.5)"
    elif m != int(round(0.5 * n)):
        bound_reason = f"subsample size {m} is not n/2: it was raised to k + 2"
    out["expected_false_selections_reason"] = bound_reason
    out["resampling"] = {"subsample": subsample, "fraction": float(fraction), "size": int(m) if subsample != "bootstrap" else int(n),
                         "stratified": s_codes is not None, "bound_valid": bound_reason is None,
                         "within_group": g_codes is not None}
    for vi, c in enumerate(c_ref):
        a, b = sizes[vi], sizes[vi + 1]
        out["ci_low"].append(torch.as_tensor(lo[a:b]).float())
        out["ci_high"].append(torch.as_tensor(hi[a:b]).float())
        if al is not None:
            stab = _col_corr_stability(al[:, a:b], c)
            freq = np.mean(chosen[vi], axis=0)
        else:
            stab = [float("nan")] * int(k)
            freq = np.zeros(c.shape[0], dtype=float)
        out["component_stability"].append(stab)
        out["selection_frequency"].append(torch.as_tensor(freq).float())
        out["selected"].append(torch.as_tensor(freq >= threshold))
        out["expected_false_selections"].append(
            qs[vi] ** 2 / ((2 * threshold - 1) * c.shape[0]) if bound_reason is None else float("nan"))
    if maps:
        ref_maps, ref_lam = feature_maps(ref, view_list, return_info=True, **map_opts)
        out["maps"] = {}
        for kk in maps:
            m_widths = [int(rm.shape[0]) for rm in ref_maps[kk]]
            m_sizes = np.cumsum([0] + m_widths)
            m_qs = [_stability_q(pw, expected_false, threshold) for pw in m_widths]
            mal = np.stack(map_aligned[kk]) if map_aligned[kk] else None
            if mal is None:
                mlo = mhi = np.full((m_sizes[-1], k), np.nan)
            elif not np.isnan(mal).any():
                mlo, mhi = np.percentile(mal, (2.5, 97.5), axis=0)
            else:
                mlo = np.empty((m_sizes[-1], k), dtype=float)
                mhi = np.empty((m_sizes[-1], k), dtype=float)
                nan_cols = np.isnan(mal).any(axis=0)
                ok_cols = ~nan_cols
                if ok_cols.any():
                    mlo[ok_cols], mhi[ok_cols] = np.percentile(mal[:, ok_cols], (2.5, 97.5), axis=0)
                if nan_cols.any():
                    mlo[nan_cols], mhi[nan_cols] = np.nanpercentile(mal[:, nan_cols], (2.5, 97.5), axis=0)
            entry = {"map": ref_maps[kk], "ci_low": [], "ci_high": [], "component_stability": [],
                     "selection_frequency": []}
            for vi in range(len(c_ref)):
                a, b = m_sizes[vi], m_sizes[vi + 1]
                rm = ref_maps[kk][vi].double().numpy()
                entry["ci_low"].append(torch.as_tensor(mlo[a:b]).float())
                entry["ci_high"].append(torch.as_tensor(mhi[a:b]).float())
                if mal is not None:
                    mal_v = mal[:, a:b]
                    stab = _col_corr_stability(mal_v, rm, nan_safe=True)
                    mal_sq = mal_v * mal_v
                    sel = [(_top_q_mask(np.nansum(mal_sq[t], axis=1), m_qs[vi])
                            if np.isfinite(mal_v[t]).any() else np.zeros(b - a, bool))
                           for t in range(mal.shape[0])]
                    entry["component_stability"].append(stab)
                    entry["selection_frequency"].append(torch.as_tensor(np.mean(sel, axis=0)).float())
                else:
                    entry["component_stability"].append([float("nan")] * int(k))
                    entry["selection_frequency"].append(torch.zeros(b - a, dtype=torch.float32))
            # Rows behind each view's reference map and why any map is NaN (see `feature_maps(return_info=True)`).
            entry["n_eff"] = list(ref_lam["n_eff"])
            entry["n_eff_feature"] = [np.asarray(a) for a in ref_lam["n_eff_feature"]]
            entry["reasons"] = list(ref_lam["reasons"][kk])
            if kk == "relative_weight":
                entry["rw_shrinkage"] = ref_lam["relative_weight"]    # lam per target, per view (reference fit)
            out["maps"][kk] = entry
    if want:
        refits: Dict[str, Any] = {
            "index": np.asarray(kept_idx, dtype=int),
            "structure": (np.asarray(al, dtype=np.float32) if al is not None
                          else np.zeros((0, stacked_ref.shape[0], int(k)), dtype=np.float32)),
            "view_sizes": [int(x) for x in np.diff(sizes)],
            "alignment": {"rotation": np.asarray(rf["rotation"]), "procrustes": np.asarray(rf["procrustes"]),
                          "loading_rotation": np.asarray(rf["loading_rotation"]),
                          "signed_permutation": np.asarray(rf["signed_permutation"]),
                          "doc": "latents_ref_frame = u_refit @ rotation[b]; undo with aligned @ rotation[b].T. "
                                 "weights use the same rotation; maps use signed_permutation (abs for relative_weight)."},
        }
        if "rows" in want or "latents" in want:
            refits["in_bag"] = np.asarray(rf["in_bag"], dtype=bool).reshape(len(kept_idx), n)
            refits["n_draws"] = np.asarray(rf["n_draws"], dtype=np.int32).reshape(len(kept_idx), n)
        if "rows" in want:
            refits["rows"] = rf["rows"]
            refits["oob_rows"] = rf["oob_rows"]
        if "weights" in want:
            refits["weights"] = [np.asarray([rf["weights"][b][vi] for b in range(len(kept_idx))], dtype=np.float32)
                                 for vi in range(len(view_list))]
        if "latents" in want:
            refits["latents"] = np.asarray(rf["latents"], dtype=np.float32).reshape(len(kept_idx), n, int(k))
        if "maps" in want:
            refits["maps"] = {}
            for kk in maps:
                arr = np.asarray(rf["maps"][kk], dtype=np.float32)
                widths = out["maps"][kk]["map"]
                offs = np.cumsum([0] + [int(t.shape[0]) for t in widths])
                refits["maps"][kk] = [arr[:, offs[vi]:offs[vi + 1]] if arr.size else arr
                                      for vi in range(len(view_list))]
        out["refits"] = refits
    if n_permutations > 0:
        null = [max(float(np.abs(c).max()) for c in rec["structure"])
                for rec in records[len(sub_idx):]]
        out["null_max_abs"] = float(np.percentile(null, 95))
    return out


def _with_grouped_loadings(res: Dict[str, Any], views: Any, codes: np.ndarray, scale_within_group: bool) -> Dict[str, Any]:
    """Copy of fit ``res`` whose loadings keys (``structure``, ``loadings``, ``loading_margin``,
    ``loading_rotation``, ``coupling_signs``) are the within-group versions."""
    lo = loadings(res, views, groups=codes, scale_within_group=scale_within_group)
    out = dict(res)
    for key in ("structure", "loadings", "loading_margin", "loading_rotation", "coupling_signs"):
        out[key] = lo[key]
    return out


def _fit_structure(task):
    """One refit, returning structure coefficients and out-of-bag diagnostics as
    plain NumPy/Python objects. Module level so worker processes can import it."""
    from .api import fit
    from .extension import _used
    map_opts = None
    extra: Dict[str, Any] = {}
    if len(task) == 5:
        method, views, k, random_state, params = task
        oob_views = None
    elif len(task) == 6:
        method, views, oob_views, k, random_state, params = task
    elif len(task) == 7:
        method, views, oob_views, k, random_state, params, map_opts = task
    else:
        method, views, oob_views, k, random_state, params, map_opts, extra = task
    r = fit(method, views, k, random_state=random_state, **params)
    if extra.get("groups") is not None:
        r = _with_grouped_loadings(r, views, extra["groups"], extra["scale_within_group"])
    want = extra.get("want", ())
    er_key = "joint_effective_rank" if r.get("basis_is_joint") else "effective_rank"
    coup_info: Dict[str, Any] = {}
    oos_r2, oos_coup, oob_scores = _oob_metrics(r, views, oob_views, return_scores=True, _info=coup_info,
                                                _groups=extra.get("oob_groups"),
                                                _scale_within_group=extra.get("scale_within_group", False))
    kept: Dict[str, Any] = {}
    if want:
        kept["loading_rotation"] = r["loading_rotation"].detach().double().numpy()
        if "weights" in want:
            if "v" not in r:
                raise ValueError(f"{method} returns no per-view weights; drop 'weights' from return_refits.")
            kept["weights"] = [v.detach().double().numpy() for v in r["v"]]
        if "latents" in want:
            from .api import predict
            kept["u_train"] = r["u"].detach().double().numpy()
            kept["u_oob"] = (None if oob_views is None or oob_views[0].shape[0] < 1
                             else predict(r, oob_views)["u"].detach().double().numpy())
    conv = r.get("convergence") or {}
    used_iter = params.get("iterations")
    if used_iter is None:
        used_iter = _used(r, "iterations", conv.get("max_steps"))
    return {
        "oob_scores": oob_scores,
        "structure": [c.double().numpy() for c in r["structure"]],
        "converged": bool(conv.get("converged", True)),
        "stop_reason": str(conv.get("stop_reason", "converged")),
        "n_steps": (None if conv.get("n_steps") is None else int(conv["n_steps"])),
        "used_iterations": used_iter,
        "effective_rank": [float(x) for x in r[er_key]],
        "oos_r2": oos_r2,
        "oos_coupling": oos_coup,
        "oos_coupling_n": coup_info.get("coupling_n"),
        "maps": ({kk: [t.double().numpy() for t in vv] for kk, vv in feature_maps(r, views, **map_opts).items()}
                 if map_opts else None),
        **kept,
    }


def _worker_init():
    torch.set_num_threads(1)


def _n_workers(n_jobs: int) -> int:
    import os
    if n_jobs is None or n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be a positive integer or -1; got {n_jobs}.")
    return (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)


def _pool(workers: int):
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)


def _run_fits(tasks, n_jobs: int, executor=None):
    """Run `_fit_structure` over ``tasks`` serially or in worker processes,
    returning results in task order. A caller-owned ``executor`` is reused
    (`select_rank` shares one across candidates: worker start-up, which
    imports torch and pysimlr, was a large share of the time)."""
    workers = min(_n_workers(n_jobs), len(tasks))
    if executor is not None:
        return list(executor.map(_fit_structure, tasks))
    if workers <= 1:
        return [_fit_structure(t) for t in tasks]
    with _pool(workers) as ex:
        return list(ex.map(_fit_structure, tasks))


def _pa_eigenvalues(x: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of ``x.T @ x / (n - 1)`` (Gram trick when ``p > n``)."""
    n, p = x.shape
    g = x @ x.T if p > n else x.T @ x
    ev = np.linalg.eigvalsh(g / (n - 1.0))[::-1]
    return np.clip(ev, 0.0, None)


def _pa_standardise(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd > 0.0, sd, 1.0)
    return (x - x.mean(axis=0)) / sd


def _cross_singular_values(zm: np.ndarray, g_others: np.ndarray, n: int) -> np.ndarray:
    """Descending singular values of ``zm.T @ z_others / (n - 1)`` via the n x n Gram of the
    other views: ``sqrt(eig(zm.T @ G_o @ zm)) / (n - 1)`` costs ``p_m^2 n`` per call."""
    m = zm.T @ g_others @ zm
    ev = np.clip(np.linalg.eigvalsh(m)[::-1], 0.0, None)
    return np.sqrt(ev) / (n - 1.0)


def _padded_cross_sv(f: np.ndarray, g: np.ndarray, n: int, pad: Optional[np.ndarray]) -> np.ndarray:
    s = _cross_singular_values(f, g, n)
    return s if pad is None else np.concatenate([s, pad])


def _stacked_qr_eigs(z_mat: np.ndarray, offsets: np.ndarray, pad_z: np.ndarray) -> np.ndarray:
    _, r_m = np.linalg.qr(z_mat, mode="reduced")
    c_off = r_m.T @ r_m
    for a_o, b_o in zip(offsets[:-1], offsets[1:]):
        c_off[a_o:b_o, a_o:b_o] = 0.0
    return np.sort(np.concatenate([np.linalg.eigvalsh(r_m @ c_off @ r_m.T), pad_z]))[::-1]


def _permutation_rank(observed: np.ndarray, null: np.ndarray, quantile: float) -> Tuple[int, np.ndarray]:
    """Horn-style count: leading indices where observed > the null quantile, stopping at the first miss."""
    thr = np.quantile(null, quantile, axis=0)
    rank = 0
    for o, t in zip(observed, thr):
        if o > t:
            rank += 1
        else:
            break
    return rank, thr


def bottleneck_rank(views: Any, *, confounds: Any = None, mode: str = "cross",
                    bound: str = "min", n_permutations: int = 100,
                    quantile: float = 0.95, random_state: int = 0) -> Dict[str, Any]:
    """Bound on the shared rank ``k`` from cross-view permutation analysis.

    For each view m, ``mode="cross"`` (default) takes the singular values of the
    cross-covariance between the standardised view and the other views stacked, and counts the
    leading ones that exceed the ``quantile`` of the same singular values under
    ``n_permutations`` row permutations of view m (counting stops at the first miss). Private
    structure in a view does not enter a cross-covariance, and a shared latent counts however
    small its within-view footprint.

    ``bound`` selects which multi-view summary sets ``k_max`` (all three are always returned as
    ``k_max_min``, ``k_max_max`` and ``k_max_stacked``):

    - ``"min"`` (default): bounds rank shared by every view (``ranks.min()``,
      ``bottleneck_view = ranks.argmin()``).
    - ``"max"``: bounds rank a consensus can carry from any view (``ranks.max()``,
      ``bottleneck_view = ranks.argmax()``).
    - ``"stacked"``: bounds pooled cross-view rank via Horn-style permutation analysis of the
      top eigenvalues of the between-view operator ``M = g_all @ g_all - sum_i G_i @ G_i``
      (``G_i = z_i @ z_i.T``, ``g_all = sum_i G_i``) against independent row permutations of
      each view (``bottleneck_view = None``).

    ``mode="within"`` is Horn's parallel analysis per view (eigenvalues of the view's
    correlation matrix against column-permuted copies): the view's own reliable rank, which
    also counts private structure (overestimate) and misses a shared latent with a weak
    within-view footprint (underestimate; planted k = 3 with a 6-column view gave 2 on three
    of four seeds, where the cross-view rank gave 3 on all four). Both ranks are reported in
    ``per_view``; ``mode`` picks which sets ``k_max`` for ``"min"`` and ``"max"``. The bound is
    a guide for the ``ks`` grid of `select_rank`, whose refit-based criteria still decide;
    `select_rank` only warns.

    Parameters
    ----------
    views : sequence of array-like or torch.Tensor
        At least two 2-D matrices with matched rows (n >= 3), finite values.
    confounds : array-like, sequence or dict, optional
        As `pysimlr.fit` takes them: one shared ``(n, q)`` matrix, a per-view sequence
        (``None`` for a view without confounds) or a dict by view. Each view is residualised
        on ``[1, its confounds]`` by least squares first; otherwise a covariate leaking into
        every view is a shared dimension and counts.
    mode : {"cross", "within"}, default "cross"
        Which per-view rank sets ``k_max_min`` and ``k_max_max``.
    bound : {"min", "max", "stacked"}, default "min"
        Which summary sets ``k_max``.
    n_permutations : int, default 100
        Null draws per view (and for the stacked between-view operator).
    quantile : float, default 0.95
        Null quantile a singular value (or eigenvalue) must exceed.
    random_state : int, default 0
        Seed of a local generator, spawned per view (no global seeding).

    Returns
    -------
    dict
        ``k_max`` (int), ``k_max_min`` (int), ``k_max_max`` (int), ``k_max_stacked`` (int),
        ``bottleneck_view`` (int or None), ``per_view`` (DataFrame: view, n, p, cross_rank,
        within_rank, first_singular_value, threshold_first, sigma1_over_tau95, tau95_rel_se
        [bootstrap relative SE of the null quantile ``threshold_first`` only; sampling
        variance of ``sigma1`` is not included], borderline [``|sigma1_over_tau95 - 1| < 2 *
        tau95_rel_se``, i.e. ``sigma1`` within two null-threshold SEs of ``tau95``]),
        ``singular_values`` and
        ``thresholds`` (cross-view, lists of arrays per view), ``eigenvalues`` and
        ``eigen_thresholds`` (within-view), and ``parameters``.
    """
    import pandas as pd
    from .api import as_views
    if mode not in ("cross", "within"):
        raise ValueError(f"mode must be 'cross' or 'within'; got {mode!r}.")
    if bound not in ("min", "max", "stacked"):
        raise ValueError(f"bound must be 'min', 'max' or 'stacked'; got {bound!r}.")
    if not isinstance(views, (list, tuple)) or len(views) < 2:
        raise ValueError("bottleneck_rank needs at least 2 views.")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}.")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1); got {quantile}.")
    arrs = [v.double().numpy() for v in as_views(views)]
    n = arrs[0].shape[0]
    if n < 3:
        raise ValueError(f"bottleneck_rank needs n >= 3 rows; got {n}.")
    for i, a in enumerate(arrs):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"view {i} contains non-finite values.")
    if confounds is not None:
        from .confounds import per_view_confounds
        per_view, _ = per_view_confounds(confounds, n, len(arrs))   # shared matrix, per-view list or dict
        out = []
        for a, c in zip(arrs, per_view):
            if c is None:
                out.append(a)
                continue
            c = np.asarray(c, dtype=np.float64).reshape(n, -1)
            if not np.all(np.isfinite(c)):
                raise ValueError("confounds must be finite.")
            design = np.column_stack([np.ones(n), c])
            out.append(a - design @ np.linalg.lstsq(design, a, rcond=None)[0])
        arrs = out
    zs = [_pa_standardise(a) for a in arrs]
    grams = [z @ z.T for z in zs]
    g_all = sum(grams)                                    # n x n Gram of all views stacked
    children = np.random.default_rng(random_state).spawn(len(zs) + 1)
    svs, thrs, eigs, ethrs, rows = [], [], [], [], []
    for vi, (z, g_self, rng) in enumerate(zip(zs, grams, children[:len(zs)])):
        g_others = g_all - g_self
        p_m = z.shape[1]
        zp = np.empty_like(z)
        if p_m > n:
            others_z = [zs[j] for j in range(len(zs)) if j != vi]
            p_others = int(sum(z_o.shape[1] for z_o in others_z))
            if p_others <= n:
                factor = np.hstack(others_z)
            else:
                w_o, v_o = np.linalg.eigh(g_others)
                pos = w_o > 1e-12 * max(float(w_o[-1]), 0.0)
                factor = v_o[:, pos] * np.sqrt(w_o[pos])
            gram_target = g_self
            pad_sv: Optional[np.ndarray] = np.zeros(p_m - int(factor.shape[1]), dtype=np.float64)
            f_buf: Optional[np.ndarray] = np.empty_like(factor)
        else:
            factor = z
            gram_target = g_others
            pad_sv = None
            f_buf = None
        sv = _padded_cross_sv(factor, gram_target, n, pad_sv)
        null_sv = np.empty((n_permutations, sv.size))
        ev = _pa_eigenvalues(z)
        null_ev = np.empty((n_permutations, ev.size))
        for b in range(n_permutations):
            p_idx = rng.permutation(n)
            if f_buf is not None:
                f_buf[p_idx] = factor
                null_sv[b] = _padded_cross_sv(f_buf, gram_target, n, pad_sv)
            else:
                null_sv[b] = _padded_cross_sv(factor[p_idx], gram_target, n, None)
            rng.permuted(z, axis=0, out=zp)
            null_ev[b] = _pa_eigenvalues(zp)
        cross_rank, thr = _permutation_rank(sv, null_sv, quantile)
        within_rank, ethr = _permutation_rank(ev, null_ev, quantile)
        tau0 = float(thr[0])
        ratio = float(sv[0] / tau0) if tau0 > 0.0 else float("inf")
        boot_idx = rng.integers(0, n_permutations, size=(200, n_permutations))
        boot_q = np.quantile(null_sv[:, 0][boot_idx], quantile, axis=1)
        rel_se = float(np.std(boot_q, ddof=1) / tau0) if (tau0 > 0.0 and n_permutations > 1) else 0.0
        borderline = bool(abs(ratio - 1.0) < 2.0 * rel_se)
        svs.append(sv)
        thrs.append(thr)
        eigs.append(ev)
        ethrs.append(ethr)
        rows.append({"view": vi, "n": n, "p": z.shape[1], "cross_rank": cross_rank, "within_rank": within_rank,
                     "first_singular_value": float(sv[0]), "threshold_first": tau0,
                     "sigma1_over_tau95": ratio, "tau95_rel_se": rel_se, "borderline": borderline})

    rng_stk = children[-1]
    p_widths = [z.shape[1] for z in zs]
    p_tot = int(sum(p_widths))
    if p_tot < n:
        offsets = np.cumsum([0] + p_widths)
        pad_z = np.zeros(n - p_tot, dtype=np.float64)
        obs_stk = _stacked_qr_eigs(np.hstack(zs), offsets, pad_z)
        null_stk = np.empty((n_permutations, n), dtype=np.float64)
        z_buf = np.empty((n, p_tot), dtype=np.float64)
        for b in range(n_permutations):
            for z, a_o, b_o in zip(zs, offsets[:-1], offsets[1:]):
                z_buf[:, a_o:b_o] = z[rng_stk.permutation(n)]
            null_stk[b] = _stacked_qr_eigs(z_buf, offsets, pad_z)
    else:
        grams_sq = [g @ g for g in grams]
        obs_stk = np.linalg.eigvalsh(g_all @ g_all - sum(grams_sq))[::-1]
        null_stk = np.empty((n_permutations, n), dtype=np.float64)
        for b in range(n_permutations):
            g_perm_all = np.zeros((n, n), dtype=np.float64)
            g_sq_perm_all = np.zeros((n, n), dtype=np.float64)
            for g, g_sq in zip(grams, grams_sq):
                p_idx = rng_stk.permutation(n)
                ix = np.ix_(p_idx, p_idx)
                g_perm_all += g[ix]
                g_sq_perm_all += g_sq[ix]
            null_stk[b] = np.linalg.eigvalsh(g_perm_all @ g_perm_all - g_sq_perm_all)[::-1]
    k_max_stacked, _ = _permutation_rank(obs_stk, null_stk, quantile)

    per_view = pd.DataFrame(rows)
    col = "cross_rank" if mode == "cross" else "within_rank"
    ranks = per_view[col].to_numpy()
    k_max_min = int(ranks.min())
    k_max_max = int(ranks.max())
    if bound == "min":
        k_max = k_max_min
        bview: Optional[int] = int(ranks.argmin())
    elif bound == "max":
        k_max = k_max_max
        bview = int(ranks.argmax())
    else:
        k_max = int(k_max_stacked)
        bview = None
    return {"k_max": k_max, "k_max_min": k_max_min, "k_max_max": k_max_max,
            "k_max_stacked": int(k_max_stacked), "bottleneck_view": bview, "per_view": per_view,
            "singular_values": svs, "thresholds": thrs, "eigenvalues": eigs, "eigen_thresholds": ethrs,
            "parameters": {"mode": mode, "bound": bound, "n_permutations": n_permutations,
                           "quantile": quantile, "random_state": random_state,
                           "confounds": confounds is not None}}


def select_rank(method: str, views: Any, ks: Any, *,
                criterion: str = "component", bound: str = "max",
                n_subsamples: int = 20,
                fraction: float = 0.5, min_stability: float = 0.9,
                min_oos_coupling: Optional[float] = None,
                heldout_views: Any = None,
                exclude_unconverged: bool = False,
                random_state: int = 0, n_jobs: int = 1,
                _executor: Any = None, _bottleneck: Any = None,
                _warn_bottleneck: bool = True,
                **params: Any) -> Dict[str, Any]:
    """Choose ``k`` under whole-pipeline subsample refits and out-of-bag validation.

    For every ``k`` in ``ks``, `loading_inference` refits ``method`` on
    ``n_subsamples`` random ``fraction``-subsamples and evaluates each refit on
    its unused ``1 - fraction`` complement. With ``exclude_unconverged=True``,
    refits with ``converged=False`` are omitted from stability and out-of-bag
    summaries and ``n_effective_refits`` is recorded per row. ``bound`` (default
    ``"max"``; also ``"min"`` or ``"stacked"``) controls the `bottleneck_rank`
    summary used for ``ks="auto"`` and over-rank warnings. One pass produces
    six rank rules
    (all reported in ``criteria``; ``criterion`` picks which one sets ``k``):

    ``"component"`` (default)
        Largest ``k`` whose every individual varimax-rotated component has
        median structure correlation >= ``min_stability`` (Ben-Hur et al. 2002;
        Lange et al. 2004). Appropriate when naming individual components.
    ``"subspace"``
        Largest ``k`` whose smallest Grassmannian principal cosine between
        refit and reference stacked structure subspaces is >= ``min_stability``.
        Rotation-invariant: stays high when components share near-equal singular
        values and rotate within a stable ``k``-dimensional span.
    ``"oos_r2"``
        Candidate ``k`` maximising mean out-of-bag held-out reconstruction
        :math:`R^2` (provided the peak is > 0).
    ``"oos_r2_1se"``
        Smallest ``k`` within one split standard error of the peak out-of-bag
        held-out :math:`R^2` (and > 0): parsimonious predictive subspace rank.
    ``"oos_coupling"``
        Largest ``k`` whose every component has median out-of-bag
        leave-one-view-out score correlation >= ``min_oos_coupling`` (default:
        one-sided ``alpha = 0.01`` Fisher-z null threshold for the out-of-bag
        sample size).
    ``"heldout_coupling"``
        When ``heldout_views`` is given, largest ``k`` whose every component has
        mean pairwise held-out cross-view correlation (`heldout_coupling`'s
        ``cross_view_heldout``) >= ``min_oos_coupling`` (default: one-sided
        ``alpha = 0.01`` Fisher-z null threshold for ``n_heldout``); ``None``
        when ``heldout_views`` is omitted.

    Each row in ``table`` also reports ``converged_fraction``,
    ``min_effective_rank`` and ``basis_degenerate`` (``min_effective_rank < k -
    0.5``, which indicates merged basis columns — raise ``nsa_w`` when True).
    """
    import contextlib
    if criterion not in RANK_CRITERIA:
        raise ValueError(f"criterion must be one of {RANK_CRITERIA}; got {criterion!r}.")
    n_views = len(views) if isinstance(views, (list, tuple)) else 1
    bott = (_bottleneck if _bottleneck is not None
            else (bottleneck_rank(views, confounds=params.get("confounds"), bound=bound) if n_views >= 2 else None))
    if isinstance(ks, str):
        if ks != "auto":
            raise ValueError(f"ks must be a sequence of ints or 'auto'; got {ks!r}.")
        if bott is None:
            raise ValueError("ks='auto' needs at least 2 views (bottleneck_rank).")
        if bott["k_max"] < 1:
            pv = bott["per_view"]
            ratios = pv["first_singular_value"].to_numpy() / np.maximum(pv["threshold_first"].to_numpy(), 1e-30)
            worst_i = int(np.argmin(ratios))
            worst_r = float(ratios[worst_i])
            if bound == "max":
                raise ValueError(
                    f"ks='auto': cross-view bottleneck bound (bound={bound!r}) is 0: no view's first "
                    f"cross-view singular value exceeds its permutation threshold (worst view {worst_i}, "
                    f"sigma1/tau95 = {worst_r:.2f}). Pass ks explicitly."
                )
            raise ValueError(
                f"ks='auto': cross-view bottleneck bound (bound={bound!r}) is 0 "
                f"(worst view {worst_i}, sigma1/tau95 = {worst_r:.2f}). Pass ks explicitly."
            )
        ks = range(1, bott["k_max"] + 1)
    ks_sorted = _normalize_ks(ks)
    if _warn_bottleneck:
        _warn_if_above_bottleneck(ks_sorted, bott)
    rows = []
    workers = _n_workers(n_jobs)
    own_pool = (_executor is None) and (workers > 1)
    with (_pool(workers) if own_pool else contextlib.nullcontext()) as ex:
        pool_ex = ex if own_pool else _executor
        for k in ks_sorted:
            rows.append(_rank_row(method, views, k, n_subsamples, fraction, min_stability,
                                  min_oos_coupling, heldout_views, exclude_unconverged,
                                  random_state, n_jobs, pool_ex, params))
    criteria = _select_by_criteria(rows)
    return {"k": criteria[criterion], "criterion": criterion, "criteria": criteria, "table": rows,
            "bottleneck": bott}


def _normalize_ks(ks: Any) -> List[int]:
    ks_sorted = sorted(dict.fromkeys(int(x) for x in ks))
    if not ks_sorted or ks_sorted[0] < 1:
        raise ValueError("ks must be a non-empty sequence of positive integers.")
    return ks_sorted


def _warn_if_above_bottleneck(ks_sorted: Sequence[int], bott: Optional[Dict[str, Any]]) -> None:
    if bott is not None and ks_sorted[-1] > bott["k_max"]:
        detail = (f"view {bott['bottleneck_view']} bound"
                  if bott["bottleneck_view"] is not None else "stacked cross-view operator")
        warnings.warn(f"max(ks)={ks_sorted[-1]} exceeds the bottleneck bound k_max={bott['k_max']} "
                      f"({detail}).",
                      RuntimeWarning, stacklevel=3)


def _eval_coupling_gate(
    coup_vals: Any,
    n_eval: int,
    min_oos_coupling: Optional[float],
    require_all_finite: bool = False,
) -> Tuple[Optional[List[float]], Optional[float], Optional[float], bool]:
    if coup_vals is None:
        return None, None, None, False
    cp = np.asarray(coup_vals, float)
    if cp.size == 0 or (require_all_finite and not np.all(np.isfinite(cp))):
        return None, None, None, False
    min_c = float(np.min(cp))
    from .extension import _null_bar
    thr = float(min_oos_coupling) if min_oos_coupling is not None else _null_bar(int(n_eval))
    return [float(x) for x in cp], min_c, thr, bool(np.isfinite(min_c) and min_c >= thr)


def _select_by_criteria(rows: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    comp = [r["k"] for r in rows if r["all_stable"]]
    sub = [r["k"] for r in rows if r["subspace_stable"]]
    coup = [r["k"] for r in rows if r["all_coupled"]]
    hcoup = [r["k"] for r in rows if r["heldout_all_coupled"]]
    r2_rows = [r for r in rows if r["oos_r2"] is not None and np.isfinite(r["oos_r2"])]
    k_r2 = k_r2_1se = None
    if r2_rows:
        best = max(r2_rows, key=lambda r: r["oos_r2"])
        se = best["oos_r2_se"] or 0.0
        if best["oos_r2"] > max(0.0, 2.0 * se):
            k_r2 = best["k"]
            floor = best["oos_r2"] - se
            cand_1se = [r["k"] for r in r2_rows if r["oos_r2"] > 0.0 and r["oos_r2"] >= floor]
            k_r2_1se = min(cand_1se) if cand_1se else best["k"]
    return {
        "component": max(comp) if comp else None,
        "subspace": max(sub) if sub else None,
        "oos_r2": k_r2,
        "oos_r2_1se": k_r2_1se,
        "oos_coupling": max(coup) if coup else None,
        "heldout_coupling": max(hcoup) if hcoup else None,
    }


def _summarize_cell_records(
    c_ref: List[np.ndarray],
    n_er_blocks: int,
    used_iter: Any,
    sub_records: Sequence[Dict[str, Any]],
    k: int,
    n: int,
    m: int,
    exclude_unconverged: bool,
) -> Dict[str, Any]:
    """Reduce a reference structure list + subsample `_fit_structure` records into `_rank_row` summary keys."""
    from scipy.optimize import linear_sum_assignment

    conv_flags = [bool(rec["converged"]) for rec in sub_records]
    conv_steps = [
        int(rec["n_steps"])
        for rec in sub_records
        if rec["converged"] and rec.get("n_steps") is not None
    ]
    median_conv_iter = float(np.median(conv_steps)) if conv_steps else None
    conv_frac = float(np.mean(conv_flags)) if conv_flags else 1.0
    stacked_ref = np.vstack(c_ref)
    sizes = np.cumsum([0] + [c.shape[0] for c in c_ref])
    aligned, cosines, r2_splits, coups, eff_ranks = [], [], [], [], []
    coup_splits_all: List[Optional[List[float]]] = []
    for rec in sub_records:
        keep_rec = (not exclude_unconverged) or bool(rec["converged"])
        st_list = rec["structure"]
        st = np.vstack(st_list)
        if keep_rec and rec["oos_coupling"] is not None:
            _, col_ind = linear_sum_assignment(-np.abs(st.T @ stacked_ref))
            inv = np.empty_like(col_ind)
            inv[col_ind] = np.arange(k)
            c_aligned = np.asarray(rec["oos_coupling"], float)[inv]
            coups.append(c_aligned)
            coup_splits_all.append([float(x) for x in c_aligned])
        else:
            coup_splits_all.append(None)
        if not keep_rec:
            continue
        rp = _procrustes(st, stacked_ref)
        aligned.append(st @ rp)
        cosines.append(_principal_cosines(st, stacked_ref))
        eff_ranks.append(rec["effective_rank"])
        if rec["oos_r2"] is not None:
            r2_splits.append(rec["oos_r2"])

    n_eff = len(aligned)
    if n_eff > 0:
        al = np.stack(aligned)
        r2_arr = np.asarray(r2_splits, float) if r2_splits else None
        sub_stab = [float(x) for x in np.nanmedian(cosines, axis=0)]
        eff_rk = [float(x) for x in np.nanmedian(eff_ranks, axis=0)]
        comp_stab = [
            _col_corr_stability(al[:, sizes[vi]:sizes[vi + 1]], c)
            for vi, c in enumerate(c_ref)
        ]
    else:
        r2_arr = None
        sub_stab = [float("nan")] * int(k)
        eff_rk = [float("nan")] * int(n_er_blocks)
        comp_stab = [[float("nan")] * int(k) for _ in c_ref]

    return {
        "component_stability": comp_stab,
        "subspace_stability": sub_stab,
        "oos_r2": ([float(x) for x in np.nanmean(r2_arr, axis=0)] if r2_arr is not None else None),
        "oos_r2_splits": ([float(x) for x in np.nanmean(r2_arr, axis=1)] if r2_arr is not None else None),
        "oos_coupling": ([float(x) for x in np.nanmedian(coups, axis=0)] if coups else None),
        "oos_coupling_splits": (coup_splits_all if coups else None),
        "oos_n": int(n - m),
        "converged_fraction": conv_frac,
        "median_converged_iter": median_conv_iter,
        "n_effective_refits": int(n_eff),
        "used_iterations": used_iter,
        "effective_rank": eff_rk,
    }


def _rank_row_from_summary(
    inf: Dict[str, Any],
    k: int,
    min_stability: float,
    min_oos_coupling: Optional[float],
    heldout_views: Any = None,
    views: Any = None,
) -> Dict[str, Any]:
    """Build one `select_rank` / `select_model` diagnostic row from summary metrics."""
    if inf["converged_fraction"] < 0.5:
        warnings.warn(
            f"select_rank(k={k}): only {inf['converged_fraction']:.0%} of subsample refits converged; "
            f"raise iterations; this run used {inf['used_iterations']}.",
            RuntimeWarning, stacklevel=3,
        )
    cs_mat = np.asarray(inf["component_stability"], float)
    cs = np.nanmedian(cs_mat, axis=0) if np.isfinite(cs_mat).any() else np.full(k, np.nan)
    ss = np.asarray(inf["subspace_stability"], float)
    cs_ok = bool(cs.size > 0 and np.all(np.isfinite(cs)))
    ss_ok = bool(ss.size > 0 and np.all(np.isfinite(ss)))
    splits = inf["oos_r2_splits"]
    if splits is not None and len(splits) > 0:
        sp = np.asarray(splits, float)
        oos_r2 = float(np.nanmean(sp))
        oos_r2_se = float(np.nanstd(sp, ddof=1) / np.sqrt(len(sp))) if len(sp) > 1 else 0.0
    else:
        oos_r2 = oos_r2_se = None
    coup = inf["oos_coupling"]
    _, min_coup, thr, all_coupled = _eval_coupling_gate(coup, inf["oos_n"], min_oos_coupling)
    er_arr = np.asarray(inf["effective_rank"], float)
    er_ok = bool(er_arr.size > 0 and np.all(np.isfinite(er_arr)))
    min_er = float(np.min(er_arr)) if er_ok else None
    if heldout_views is not None:
        from .sem import _view_list
        from .utils import heldout_coupling as _heldout_coupling
        tr_vl = list(views) if isinstance(views, (list, tuple)) else [views]
        te_vl = _view_list(inf["result"], heldout_views)
        hc = _heldout_coupling(inf["result"], te_vl, train_views=tr_vl)
        h_coup, h_min, h_thr, h_all = _eval_coupling_gate(
            hc["cross_view_heldout"], hc["n_heldout"], min_oos_coupling, require_all_finite=True
        )
    else:
        h_coup = h_min = h_thr = None
        h_all = False
    return {
        "k": k,
        "component_stability": [float(x) for x in cs],
        "min_stability": (float(np.min(cs)) if cs_ok else None),
        "all_stable": (bool(np.all(cs >= min_stability)) if cs_ok else False),
        "subspace_stability": [float(x) for x in ss],
        "min_subspace_stability": (float(np.min(ss)) if ss_ok else None),
        "subspace_stable": (bool(np.min(ss) >= min_stability) if ss_ok else False),
        "oos_r2": oos_r2,
        "oos_r2_se": oos_r2_se,
        "oos_r2_per_view": inf["oos_r2"],
        "oos_coupling": coup,
        "oos_coupling_splits": inf["oos_coupling_splits"],
        "min_oos_coupling": min_coup,
        "oos_coupling_threshold": thr,
        "all_coupled": all_coupled,
        "heldout_coupling": h_coup,
        "heldout_min_coupling": h_min,
        "heldout_coupling_threshold": h_thr,
        "heldout_all_coupled": h_all,
        "converged_fraction": inf["converged_fraction"],
        "median_converged_iter": inf["median_converged_iter"],
        "n_effective_refits": inf["n_effective_refits"],
        "min_effective_rank": min_er,
        "basis_degenerate": (bool(min_er < k - 0.5) if min_er is not None else False),
    }


def _rank_row(method, views, k, n_subsamples, fraction, min_stability, min_oos_coupling,
              heldout_views, exclude_unconverged, random_state, n_jobs, executor, params):
    """One `select_rank` table row at rank ``k``."""
    inf = loading_inference(method, views, k, n_subsamples=n_subsamples, fraction=fraction,
                            exclude_unconverged=exclude_unconverged, random_state=random_state,
                            n_jobs=n_jobs, _executor=executor, _warn_unconverged=False,
                            **params)
    return _rank_row_from_summary(inf, k, min_stability, min_oos_coupling, heldout_views, views)


def select_model(views: Any, candidates: Sequence[Mapping[str, Any]], ks: Any, *,
                 confounds: Any = None, confound_strength: float = 1.0,
                 n_subsamples: int = 20, fraction: float = 0.7,
                 min_stability: float = 0.5, exclude_unconverged: bool = True,
                 min_converged_fraction: float = 0.5, random_state: int = 0,
                 n_jobs: int = 1, **shared_params: Any) -> Dict[str, Any]:
    """Choose ``(candidate, k)`` across pre-specified models on one paired out-of-bag design.

    Thin wrapper over `select_rank`: every ``(candidate, k)`` cell runs with the
    identical ``random_state``, ``n_subsamples``, ``fraction`` and ``confounds`` so
    out-of-bag splits are paired across cells, sharing one worker pool and one
    `bottleneck_rank` computation.

    1. **Health gate**: ``converged_fraction >= min_converged_fraction`` and
       ``not basis_degenerate``.
    2. **Signal gate**: every component's median out-of-bag leave-one-view-out
       coupling >= ``oos_coupling_threshold`` (the calibrated Fisher-z ``alpha = 0.01``
       bar for the out-of-bag ``n``, i.e. ``all_coupled == True``).
    3. **1-SE parsimony rule**: among signal-bearing, health-passing cells, compute
       ``S = sum_j max(median_coupling_j, 0)^2``; admit cells whose paired per-split
       difference to the highest-``S`` cell has ``mean >= -SE``; select by
       ``(complexity_tier, k, -sparseness, -min_subspace_stability)`` with tiers
       ``0`` (``pca``, ``nsa_pipeline``), ``1`` (``simlr``), ``2`` (other methods).

    If no cell is signal-bearing, ``signal_detected`` is ``False`` and ``selected``
    is the ``k = 1`` health-passing cell with the highest median coupling (with
    ``note = "no candidate cleared the out-of-bag cross-view coupling bar"``).
    """
    import pandas as pd
    from .api import METHODS, as_views, resolve_method
    from .confounds import SINK_PARAMS, slice_confounds

    if isinstance(ks, str):
        raise ValueError("select_model requires an explicit sequence of positive ints for ks ('auto' is not accepted).")
    ks_sorted = _normalize_ks(ks)
    if not candidates:
        raise ValueError("candidates must be a non-empty sequence of {'label', 'method', 'params'} mappings.")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction}.")

    labels_seen = set()
    cand_list = []
    for c in candidates:
        if not isinstance(c, Mapping) or "label" not in c or "method" not in c:
            raise ValueError("each candidate must be a mapping with keys 'label' and 'method'.")
        lbl = str(c["label"])
        if lbl in labels_seen:
            raise ValueError(f"duplicate candidate label {lbl!r}.")
        labels_seen.add(lbl)
        cand_list.append({"label": lbl, "method": str(c["method"]), "params": dict(c.get("params") or {})})

    _n_workers(n_jobs)
    n_views = len(views) if isinstance(views, (list, tuple)) else 1
    bott = bottleneck_rank(views, confounds=confounds, bound="max") if n_views >= 2 else None
    _warn_if_above_bottleneck(ks_sorted, bott)

    view_list = as_views(views)
    n = view_list[0].shape[0]
    views_np = [x.numpy() if x.device.type == "cpu" else x for x in view_list]
    all_rows = np.arange(n)
    n_sub = int(n_subsamples)

    split_cache: Dict[int, Tuple[List[np.ndarray], List[Any], List[Any]]] = {}

    def _get_splits(m_val: int) -> Tuple[List[np.ndarray], List[Any], List[Any]]:
        if m_val not in split_cache:
            rng_m = np.random.default_rng(random_state)
            sub_idx_m = [np.sort(rng_m.choice(n, m_val, replace=False)) for _ in range(n_sub)]
            sub_v_list, oob_v_list = [], []
            for idx in sub_idx_m:
                oob = np.setdiff1d(all_rows, idx, assume_unique=True)
                t_idx = torch.as_tensor(idx)
                t_oob = torch.as_tensor(oob) if len(oob) >= 2 else None
                sub_v_list.append([x[t_idx].numpy() if x.device.type == "cpu" else x[t_idx] for x in view_list])
                oob_v_list.append(
                    [x[t_oob].numpy() if x.device.type == "cpu" else x[t_oob] for x in view_list]
                    if t_oob is not None else None
                )
            split_cache[m_val] = (sub_idx_m, sub_v_list, oob_v_list)
        return split_cache[m_val]

    sink_keys = frozenset(SINK_PARAMS)
    all_known = frozenset().union(*(s.accepts for s in METHODS.values())) | sink_keys
    all_tasks: List[Any] = []
    cell_specs: List[Tuple[Dict[str, Any], str, int, float, int, int, Optional[float], int]] = []

    for cand in cand_list:
        method = cand["method"]
        spec = resolve_method(method)
        accepted = spec.accepts | sink_keys
        call_params: Dict[str, Any] = {}
        for sk, sv in shared_params.items():
            if sk in accepted or sk not in all_known:
                call_params[sk] = sv
        call_params.update(cand["params"])
        min_oos_c = call_params.pop("min_oos_coupling", None)
        if confounds is not None:
            call_params["confounds"] = confounds
            call_params["confound_strength"] = confound_strength
        tier = 0 if method in ("pca", "nsa_pipeline") else (1 if method == "simlr" else 2)
        if method in ("pca", "nsa_pipeline"):
            sp_scalar = 0.0
        else:
            sp_raw = call_params.get("sparseness")
            if sp_raw is None:
                sp_scalar = 0.5
            elif isinstance(sp_raw, (list, tuple, np.ndarray)):
                sp_scalar = float(np.mean(sp_raw))
            else:
                sp_scalar = float(sp_raw)
        conf_c = call_params.get("confounds")
        for k in ks_sorted:
            m = max(k + 2, int(round(fraction * n)))
            sub_idx_m, sub_v_list, oob_v_list = _get_splits(m)
            offset = len(all_tasks)
            all_tasks.append((method, views_np, k, random_state, call_params))
            for idx, sub_v, oob_v in zip(sub_idx_m, sub_v_list, oob_v_list):
                sub_p = call_params if conf_c is None else dict(call_params, confounds=slice_confounds(conf_c, idx))
                all_tasks.append((method, sub_v, oob_v, k, random_state, sub_p))
            cell_specs.append((cand, method, tier, sp_scalar, k, m, min_oos_c, offset))

    all_records = _run_fits(all_tasks, n_jobs)

    cells: List[Dict[str, Any]] = []
    for cand, method, tier, sp_scalar, k, m, min_oos_c, offset in cell_specs:
        ref_rec = all_records[offset]
        sub_records = all_records[offset + 1 : offset + 1 + n_sub]
        summary = _summarize_cell_records(
            ref_rec["structure"],
            len(ref_rec["effective_rank"]),
            ref_rec["used_iterations"],
            sub_records,
            k,
            n,
            m,
            exclude_unconverged,
        )
        r = _rank_row_from_summary(summary, k, min_stability, min_oos_c)
        health_ok = bool(r["converged_fraction"] >= min_converged_fraction and not r["basis_degenerate"])
        signal_bearing = bool(health_ok and r["all_coupled"])
        oc = r["oos_coupling"]
        score_s = float(sum(max(float(v), 0.0) ** 2 for v in oc)) if oc is not None else 0.0
        split_s = [
            (float(sum(max(float(v), 0.0) ** 2 for v in sp_c)) if sp_c is not None else None)
            for sp_c in (r["oos_coupling_splits"] or [])
        ]
        cell = {
            "label": cand["label"],
            "method": method,
            "params": dict(cand["params"]),
            "k": int(k),
            "complexity_tier": tier,
            "sparseness": sp_scalar,
            "health_ok": health_ok,
            "signal_bearing": signal_bearing,
            "score_S": score_s,
            "paired_diff_mean": None,
            "paired_diff_se": None,
            "within_1se": False,
            **r,
            "note": None,
            "_split_S": split_s,
        }
        cells.append(cell)

    sig_cells = [c for c in cells if c["signal_bearing"]]
    if sig_cells:
        signal_detected = True
        best_cell = max(sig_cells, key=lambda c: c["score_S"])
        for c in cells:
            if c is best_cell:
                c["paired_diff_mean"] = 0.0
                c["paired_diff_se"] = 0.0
                c["within_1se"] = True
                continue
            diffs = [
                sc - sb for sc, sb in zip(c["_split_S"], best_cell["_split_S"])
                if sc is not None and sb is not None
            ]
            if len(diffs) >= 2:
                md = float(np.mean(diffs))
                sd = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
                c["paired_diff_mean"] = md
                c["paired_diff_se"] = sd
                c["within_1se"] = bool(c["signal_bearing"] and md >= -sd - 1e-12)
            elif len(diffs) == 1:
                md = float(diffs[0])
                c["paired_diff_mean"] = md
                c["paired_diff_se"] = 0.0
                c["within_1se"] = bool(c["signal_bearing"] and md >= -1e-12)
            else:
                c["within_1se"] = bool(c["signal_bearing"] and c["score_S"] >= best_cell["score_S"] - 1e-12)
        admitted = [c for c in sig_cells if c["within_1se"]]
        winner = min(
            admitted,
            key=lambda c: (
                c["complexity_tier"],
                c["k"],
                -c["sparseness"],
                -(c["min_subspace_stability"] if c["min_subspace_stability"] is not None else -1.0),
                -c["score_S"],
            ),
        )
    else:
        signal_detected = False
        hp_k1 = [c for c in cells if c["health_ok"] and c["k"] == 1]
        pool_fb = hp_k1 or [c for c in cells if c["health_ok"]] or list(cells)
        min_k_fb = min(c["k"] for c in pool_fb)
        pool_fb = [c for c in pool_fb if c["k"] == min_k_fb]

        def _med_coup(c: Dict[str, Any]) -> float:
            oc_v = c["oos_coupling"]
            return float(np.mean(oc_v)) if oc_v is not None and len(oc_v) else float("-inf")

        winner = max(pool_fb, key=lambda c: (_med_coup(c), -c["complexity_tier"], c["sparseness"]))
        winner["note"] = "no candidate cleared the out-of-bag cross-view coupling bar"

    clean_rows = [{kk: vv for kk, vv in c.items() if kk != "_split_S"} for c in cells]
    selected = {kk: vv for kk, vv in winner.items() if kk != "_split_S"}
    return {
        "selected": selected,
        "signal_detected": signal_detected,
        "table": pd.DataFrame(clean_rows),
        "bottleneck": bott,
    }
