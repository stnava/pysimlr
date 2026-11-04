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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_NONNEG = ("positive", "hard", "nonnegative", "nonneg", "softplus")


def structure_coefficients(views: Sequence[torch.Tensor], u: torch.Tensor) -> List[torch.Tensor]:
    """Per-view ``(p_m, k)`` correlations between each feature and each latent column.

    A constant feature or latent column has no correlation; its entries are 0.
    """
    u64 = torch.as_tensor(u).detach().double()
    uc = u64 - u64.mean(0, keepdim=True)
    us = uc / uc.norm(dim=0, keepdim=True).clamp_min(1e-300)
    us = torch.where(uc.norm(dim=0, keepdim=True) > 0, us, torch.zeros_like(us))
    out = []
    for x in views:
        x64 = torch.as_tensor(x).detach().double()
        if x64.shape[0] != u64.shape[0]:
            raise ValueError(f"view has {x64.shape[0]} rows but u has {u64.shape[0]}.")
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
    nonneg = positivity in _NONNEG
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
    a = torch.cat([torch.as_tensor(c).double() for c in structure], dim=0).numpy()
    p, k = a.shape
    r = np.eye(k)
    if k > 1:
        crit = 0.0
        for _ in range(int(max_iter)):
            lam = a @ r
            uu, sv, vt = np.linalg.svd(a.T @ (lam ** 3 - lam @ np.diag((lam ** 2).sum(0)) / p))
            r = uu @ vt
            if sv.sum() <= crit * (1.0 + tol):
                break
            crit = sv.sum()
    signs = np.sign(((a @ r) ** 3).sum(0))
    signs[signs == 0] = 1.0
    return torch.as_tensor(r * signs[None, :])


def _result_positivity(result: Dict[str, Any]) -> str:
    prm = result.get("parameters", {})
    used, defaults = prm.get("used", {}), prm.get("defaults", {})
    return used.get("positivity", defaults.get("positivity", "either"))


def loadings(result: Dict[str, Any], views: Any, u: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """`structure`, `loadings`, `loading_margin` and per-feature `communality`
    for a `pysimlr.fit` result on ``views``.

    ``views`` are replayed through the fit's own preprocessing
    (``scale_list``, ``provenance_list``), so this works for every method. With
    the training views and ``u=None`` it reproduces the keys `pysimlr.fit`
    stored; with held-out views pass ``u=pysimlr.predict(result, views)["u"]``.
    """
    from .api import _preprocess_views, as_views
    view_list = as_views(views)
    if u is None:
        u = result["u"]
    xs, _ = _preprocess_views(view_list, result["scale_list"], result["provenance_list"])
    c0 = structure_coefficients(xs, u)
    rot = varimax_rotation(c0)
    # Correlations with the rotated scores themselves, not C0 @ R: the two agree
    # only when u's columns are uncorrelated with equal variance (orthogonal
    # mixings). Under "avg" they differed by up to 0.07, so "structure describes
    # u @ R" held only approximately.
    u_rot = torch.as_tensor(u).detach().double() @ rot
    c = structure_coefficients(xs, u_rot)
    load, margin = olee_loadings(c, _result_positivity(result))
    return {"structure": c, "loadings": load, "loading_margin": margin,
            "loading_rotation": rot.float(),
            "communality": [(ci.double() ** 2).sum(1).float() for ci in c]}


FEATURE_MAP_KINDS = ("structure", "partial", "global_partial", "component_partial", "relative_weight")


def _zscore(a: np.ndarray) -> np.ndarray:
    """Column z-scores (population SD); a constant column becomes 0."""
    a = a - a.mean(0, keepdims=True)
    sd = a.std(0, keepdims=True)
    return np.where(sd > 0, a / np.where(sd > 0, sd, 1.0), 0.0)


def _corr_map(xz: np.ndarray, tz: np.ndarray) -> np.ndarray:
    return xz.T @ tz / xz.shape[0]


def _partial_map(xz: np.ndarray, tz: np.ndarray, shrinkage) -> np.ndarray:
    """Partial correlation of each feature with each target column, given every
    other feature of the view: -P_ft / sqrt(P_ff P_tt), P the inverse of the
    (shrunk) correlation matrix of [X, t_j]."""
    n, p = xz.shape
    out = np.zeros((p, tz.shape[1]))
    live = xz.std(0) > 0
    for j in range(tz.shape[1]):
        if not tz[:, j].std() > 0:
            continue
        joint = np.column_stack([xz[:, live], tz[:, j]])
        if shrinkage == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf
            s = LedoitWolf(assume_centered=True).fit(joint).covariance_
        else:
            lam = float(shrinkage)
            if not 0.0 <= lam < 1.0:
                raise ValueError(f"shrinkage must be 'ledoit_wolf' or a float in [0, 1); got {shrinkage!r}.")
            s = joint.T @ joint / n
            s = (1.0 - lam) * s + lam * np.diag(np.diag(s))
        prec = np.linalg.inv(s)
        d = np.diag(prec)
        out[live, j] = -prec[:-1, -1] / np.sqrt(d[:-1] * d[-1])
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


def _relative_weight_map(xz: np.ndarray, tz: np.ndarray) -> np.ndarray:
    """Johnson's (2000) relative weights: each feature's share of R^2(t_j ~ X).

    X = P D Q' (SVD of the standardized view); Z = P Q' is its closest orthonormal
    counterpart, and Lambda = Q D Q' / sqrt(n) maps Z back to X. The weight of
    feature f is sum_l Lambda_fl^2 beta_l^2, beta the regression of t_j on Z; the
    weights are non-negative and sum to R^2.
    """
    n, p = xz.shape
    live = xz.std(0) > 0
    if live.sum() >= n:
        raise ValueError(f"relative_weight needs fewer features than rows; the view has "
                         f"{int(live.sum())} non-constant features and {n} rows. Use 'partial'.")
    xl = xz[:, live]
    pp, d, qt = np.linalg.svd(xl, full_matrices=False)
    if d.min() <= d.max() * 1e-10:
        raise ValueError("relative_weight needs a full-rank view; it has collinear features. Use 'partial'.")
    zo = pp @ qt
    lam = qt.T @ np.diag(d) @ qt / np.sqrt(n)
    beta = zo.T @ tz / np.sqrt(n)
    out = np.zeros((p, tz.shape[1]))
    out[live] = (lam ** 2) @ (beta ** 2)
    return out


def _view_scores(result: Dict[str, Any], xs: Sequence[np.ndarray], u_rot: np.ndarray,
                 rot: np.ndarray, views) -> List[np.ndarray]:
    """Per-view latent scores in the rotated frame, each column signed to agree with u."""
    from .api import predict
    if not result.get("basis_is_joint") and len(result.get("latents", [])) == len(xs):
        zs = [np.asarray(z.detach().double()) @ rot for z in predict(result, views)["latents"]]
    else:
        zs = [x @ (v.detach().double().numpy() @ rot) for x, v in zip(xs, result["v"])]
    uc = u_rot - u_rot.mean(0, keepdims=True)
    out = []
    for z in zs:
        sgn = np.where(((z - z.mean(0, keepdims=True)) * uc).sum(0) >= 0.0, 1.0, -1.0)
        out.append(_zscore(z * sgn[None, :]))
    return out


def feature_maps(result: Dict[str, Any], views: Any, kinds: Sequence[str] = FEATURE_MAP_KINDS,
                 u: Optional[torch.Tensor] = None, target: str = "other_views",
                 shrinkage: Any = "ledoit_wolf", control: Any = "pc1") -> Dict[str, List[torch.Tensor]]:
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
        R^2(t_j ~ X_m); they sum to that R^2. Defined only for a full-rank view with
        fewer features than rows; otherwise that view's map is NaN, with a warning.

    ``target="other_views"`` (default) sets t to the mean of the *other* views'
    standardized scores. The regression-based maps need this: a linear fit's own
    score is exactly X_m V_m, so regressing the consensus on X_m would only
    recover the encoder V_m, which is weakly identified. ``"consensus"`` uses
    ``u @ R`` itself; for ``partial`` and ``relative_weight`` it is circular.

    Views are replayed through the fit's own preprocessing and ``u`` defaults to
    ``pysimlr.predict(result, views)["u"]``, so this works for every method and
    for held-out views alike.

    Returns ``{kind: [ (p_m, k) tensor per view ]}``.
    """
    from .api import _preprocess_views, as_views, predict
    bad = [k_ for k_ in kinds if k_ not in FEATURE_MAP_KINDS]
    if bad:
        raise ValueError(f"unknown feature map(s) {bad}; choose from {FEATURE_MAP_KINDS}.")
    if target not in ("other_views", "consensus"):
        raise ValueError(f"target must be 'other_views' or 'consensus'; got {target!r}.")
    view_list = as_views(views)
    if target == "other_views" and len(view_list) < 2:
        raise ValueError("target='other_views' needs at least two views.")
    xs_t, _ = _preprocess_views(view_list, result["scale_list"], result["provenance_list"])
    xs = [x.detach().double().numpy() for x in xs_t]
    rot = result["loading_rotation"].detach().double().numpy()
    if u is None:
        u = predict(result, view_list)["u"]
    u_rot = np.asarray(torch.as_tensor(u).detach().double()) @ rot
    scores = _view_scores(result, xs, u_rot, rot, view_list) if target == "other_views" else None
    if isinstance(control, (list, tuple)):
        if len(control) != len(xs):
            raise ValueError(f"control has {len(control)} entries; there are {len(xs)} views.")
        controls = list(control)
    else:
        controls = [control] * len(xs)
    out: Dict[str, List[torch.Tensor]] = {k_: [] for k_ in kinds}
    for m, x in enumerate(xs):
        xz = _zscore(x)
        if target == "other_views":
            tz = _zscore(sum(z for i, z in enumerate(scores) if i != m))
        else:
            tz = _zscore(u_rot)
        for k_ in kinds:
            if k_ == "structure":
                mp = _corr_map(xz, tz)
            elif k_ == "partial":
                mp = _partial_map(xz, tz, shrinkage)
            elif k_ == "global_partial":
                if controls[m] is None:
                    mp = np.full((x.shape[1], tz.shape[1]), np.nan)
                else:
                    mp = _global_partial_map(xz, tz, controls[m])
            elif k_ == "component_partial":
                mp = _component_partial_map(xz, tz)
            else:
                try:
                    mp = _relative_weight_map(xz, tz)
                except ValueError as err:
                    # Not defined on this view (p >= n or collinear): NaN, named, not a substitute map.
                    warnings.warn(f"relative_weight is NaN for view {m}: {err}", RuntimeWarning, stacklevel=2)
                    mp = np.full((x.shape[1], tz.shape[1]), np.nan)
            out[k_].append(torch.as_tensor(mp).float())
    return out


def _procrustes(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||a R - ref||_F."""
    uu, _, vh = np.linalg.svd(a.T @ ref)
    return uu @ vh


#: Rank selection rules exposed by `select_rank`.
RANK_CRITERIA = ("component", "subspace", "oos_r2", "oos_r2_1se", "oos_coupling")

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


def _oob_metrics(res: Dict[str, Any], sub_views: Sequence[torch.Tensor],
                 oob_views: Optional[Sequence[torch.Tensor]]):
    """Held-out reconstruction R^2 (adjusted for random k-plane self-projection)
    and leave-one-view-out coupling on unused rows."""
    if oob_views is None or oob_views[0].shape[0] < 2:
        return None, None
    from .api import _preprocess_views, predict
    pred = predict(res, oob_views)
    k = res["structure"][0].shape[1]
    ps = [int(x.shape[1]) for x in sub_views]
    p_tot, m_views = sum(ps), len(ps)
    r2 = []
    for raw, pm in zip(pred["r2"], ps):
        d0 = min(0.999, (k / p_tot) if res.get("basis_is_joint") else (k / (m_views * pm)))
        r2.append(float((float(raw) - d0) / (1.0 - d0)))
    if m_views < 2 or oob_views[0].shape[0] < 3:
        return r2, None
    rot = res["loading_rotation"].detach().double().numpy()
    u_tr = res["u"].detach().double().numpy() @ rot
    if not res.get("basis_is_joint") and len(res["latents"]) == len(sub_views):
        z_tr = [z.detach().double().numpy() @ rot for z in res["latents"]]
        z_oob = [z.detach().double().numpy() @ rot for z in pred["latents"]]
    else:
        xs_tr, _ = _preprocess_views(list(sub_views), res["scale_list"], res["provenance_list"])
        xs_oob, _ = _preprocess_views(list(oob_views), res["scale_list"], res["provenance_list"])
        v_rot = [v.detach().double().numpy() @ rot for v in res["v"]]
        z_tr = [x.double().numpy() @ vr for x, vr in zip(xs_tr, v_rot)]
        z_oob = [x.double().numpy() @ vr for x, vr in zip(xs_oob, v_rot)]
    z_std = []
    u_tr_c = u_tr - u_tr.mean(0, keepdims=True)
    for zm_tr, zm_oob in zip(z_tr, z_oob):
        cov = ((zm_tr - zm_tr.mean(0, keepdims=True)) * u_tr_c).sum(0)
        sgn = np.where(cov >= 0.0, 1.0, -1.0)
        zc = zm_oob * sgn[None, :]
        zc = zc - zc.mean(0, keepdims=True)
        sd = zc.std(0, keepdims=True)
        z_std.append(np.where(sd > 0, zc / np.where(sd > 0, sd, 1.0), 0.0))
    tot = sum(z_std)
    per_view = []
    for zm in z_std:
        other = tot - zm
        den = np.sqrt((zm ** 2).sum(0) * (other ** 2).sum(0))
        per_view.append(np.where(den > 0, (zm * other).sum(0) / np.where(den > 0, den, 1.0), 0.0))
    return r2, [float(x) for x in np.mean(per_view, axis=0)]


def loading_inference(method: str, views: Any, k: int, *, n_subsamples: int = 100,
                      fraction: float = 0.5, threshold: float = 0.8,
                      expected_false: float = 1.0,
                      n_permutations: int = 0, random_state: int = 0, n_jobs: int = 1,
                      maps: Sequence[str] = (), map_options: Optional[Dict[str, Any]] = None,
                      _executor: Any = None, **params: Any) -> Dict[str, Any]:
    r"""Uncertainty and selection for `structure`, by refitting the whole pipeline.

    Each of ``n_subsamples`` refits uses a random ``fraction`` of the rows
    (without replacement). Its structure coefficients are rotated onto the
    reference fit's (orthogonal Procrustes on all views stacked), since the
    latent space fixes components only up to rotation. Every subsample refit
    also evaluates `pysimlr.predict` on its unused (out-of-bag) complement at
    no extra fit cost.

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
        ``None``), ``converged_fraction`` and ``effective_rank`` (median over
        refits), and per view: ``structure`` (reference), ``ci_low`` /
        ``ci_high`` (2.5/97.5% percentiles of the aligned refits; subsamples of
        half the rows make these conservative), ``component_stability`` (length
        ``k``: median over refits of the correlation between the aligned and
        reference column; per-component claims need this high, e.g. >= 0.9),
        ``selection_frequency`` and ``selected`` (stability selection,
        Meinshausen & Buhlmann 2010: each refit selects the ``q_m`` features of
        largest communality; ``selected`` is frequency >= ``threshold``),
        ``q`` and ``expected_false_selections`` (the bound
        :math:`q^2 / ((2\pi - 1) p)`, valid for ``fraction = 0.5`` and
        ``threshold > 0.5``), and, if ``n_permutations > 0``,
        ``null_max_abs`` (95th percentile of :math:`\max_{f,j} |C|` with the
        rows of every view but the first permuted independently, i.e. no
        shared signal).

    Alternative feature maps. ``maps`` (kinds from `feature_maps`, with
    ``map_options`` passed to it: ``target``, ``shrinkage``, ``control``) are
    computed on the reference fit and on every subsample refit at no extra fits,
    and reported under ``out["maps"][kind]`` with per view ``map`` (reference),
    ``ci_low`` / ``ci_high``, ``component_stability`` and ``selection_frequency``
    (the same ``q_m`` rule, ranking features by the sum over components of the
    squared map). Each refit's columns are matched to the reference by the signed
    permutation nearest its structure Procrustes rotation (the varimax frames are
    canonical up to order and sign; a full rotation would mix the squared
    ``relative_weight`` shares), without the sign for ``relative_weight``. A
    map's stability is its own: precision maps can be far less stable than
    ``structure``.

    Speed. ``n_jobs`` runs the refits (and permutation fits) in worker
    processes (``-1``: all cores). All subsample indices are drawn in the parent
    in a fixed order, so the result does not depend on ``n_jobs``. Each refit
    tunes its own learning rate; reusing the reference fit's instead was
    measured and rejected: 1.3x faster, but it moved an under-identified
    component's stability by 0.18 (``docs/audit/evidence/select_rank_speed.csv``).

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
    rng = np.random.default_rng(random_state)
    maps = tuple(maps)
    bad = [k_ for k_ in maps if k_ not in FEATURE_MAP_KINDS]
    if bad:
        raise ValueError(f"unknown feature map(s) {bad}; choose from {FEATURE_MAP_KINDS}.")
    map_opts = dict(map_options or {}, kinds=maps) if maps else None
    ref = fit(method, view_list, k, random_state=random_state, **params)
    c_ref = [c.double().numpy() for c in ref["structure"]]
    stacked_ref = np.vstack(c_ref)
    if expected_false <= 0:
        raise ValueError(f"expected_false must be positive; got {expected_false}.")
    qs = [max(1, min(c.shape[0], int(np.floor(np.sqrt(expected_false * (2 * threshold - 1) * c.shape[0])))))
          for c in c_ref]
    sizes = np.cumsum([0] + [c.shape[0] for c in c_ref])
    aligned, chosen = [], [[] for _ in c_ref]
    m = max(k + 2, int(round(fraction * n)))
    refit_params = dict(params)
    # Draw every index in the parent, in a fixed order, before any refit runs:
    # the result is then the same for every n_jobs.
    sub_idx = [np.sort(rng.choice(n, m, replace=False)) for _ in range(int(n_subsamples))]
    perm_idx = [[rng.permutation(n) for _ in view_list[1:]] for _ in range(int(n_permutations))]
    all_rows = np.arange(n)
    tasks = []
    for idx in sub_idx:
        oob = np.setdiff1d(all_rows, idx, assume_unique=True)
        oob_v = [x[torch.as_tensor(oob)] for x in view_list] if len(oob) >= 2 else None
        task = (method, [x[torch.as_tensor(idx)] for x in view_list], oob_v, k, random_state, refit_params)
        tasks.append(task + (map_opts,) if map_opts else task)
    tasks += [(method, [view_list[0]] + [x[torch.as_tensor(pi)] for x, pi in zip(view_list[1:], pis)],
               None, k, random_state, refit_params) for pis in perm_idx]
    records = _run_fits(tasks, n_jobs, _executor)
    cosines, r2_splits, coups, conv_flags, eff_ranks = [], [], [], [], []
    map_aligned = {kk: [] for kk in maps}
    for rec in records[:len(sub_idx)]:
        st_list = rec["structure"]
        st = np.vstack(st_list)
        rp = _procrustes(st, stacked_ref)
        aligned.append(st @ rp)
        if maps:
            rows_, cols_ = linear_sum_assignment(-np.abs(rp))
            perm = np.zeros_like(rp)
            perm[rows_, cols_] = np.sign(rp[rows_, cols_])
            for kk in maps:
                mk = np.vstack(rec["maps"][kk])
                map_aligned[kk].append(mk @ (np.abs(perm) if kk == "relative_weight" else perm))
        cosines.append(_principal_cosines(st, stacked_ref))
        conv_flags.append(rec["converged"])
        eff_ranks.append(rec["effective_rank"])
        if rec["oos_r2"] is not None:
            r2_splits.append(rec["oos_r2"])
        if rec["oos_coupling"] is not None:
            _, col_ind = linear_sum_assignment(-np.abs(st.T @ stacked_ref))
            inv = np.empty_like(col_ind)
            inv[col_ind] = np.arange(k)
            coups.append(np.asarray(rec["oos_coupling"], float)[inv])
        for vi, c in enumerate(st_list):
            h = (c ** 2).sum(1)
            sel = np.zeros(h.shape[0], bool)
            sel[np.argsort(-h)[:qs[vi]]] = True
            chosen[vi].append(sel)
    al = np.stack(aligned)                      # (B, p_total, k)
    lo, hi = np.percentile(al, 2.5, axis=0), np.percentile(al, 97.5, axis=0)
    r2_arr = np.asarray(r2_splits, float) if r2_splits else None
    out = {"result": ref, "structure": ref["structure"], "ci_low": [], "ci_high": [],
           "component_stability": [], "subspace_stability": [float(x) for x in np.nanmedian(cosines, axis=0)],
           "oos_r2": ([float(x) for x in np.nanmean(r2_arr, axis=0)] if r2_arr is not None else None),
           "oos_r2_splits": ([float(x) for x in np.nanmean(r2_arr, axis=1)] if r2_arr is not None else None),
           "oos_coupling": ([float(x) for x in np.nanmedian(coups, axis=0)] if coups else None),
           "oos_n": int(n - m),
           "converged_fraction": float(np.mean(conv_flags)),
           "effective_rank": [float(x) for x in np.nanmedian(eff_ranks, axis=0)],
           "selection_frequency": [], "selected": [],
           "q": qs, "expected_false_selections": []}
    for vi, c in enumerate(c_ref):
        a, b = sizes[vi], sizes[vi + 1]
        out["ci_low"].append(torch.as_tensor(lo[a:b]).float())
        out["ci_high"].append(torch.as_tensor(hi[a:b]).float())
        stab = []
        for j in range(k):
            rs = [np.corrcoef(al[t, a:b, j], c[:, j])[0, 1] for t in range(al.shape[0])]
            stab.append(float(np.nanmedian(rs)))
        out["component_stability"].append(stab)
        freq = np.mean(chosen[vi], axis=0)
        out["selection_frequency"].append(torch.as_tensor(freq).float())
        out["selected"].append(torch.as_tensor(freq >= threshold))
        out["expected_false_selections"].append(qs[vi] ** 2 / ((2 * threshold - 1) * c.shape[0]))
    if maps:
        ref_maps = feature_maps(ref, view_list, **map_opts)
        out["maps"] = {}
        for kk in maps:
            mal = np.stack(map_aligned[kk])                 # (B, p_total, k)
            mlo, mhi = np.nanpercentile(mal, 2.5, axis=0), np.nanpercentile(mal, 97.5, axis=0)
            entry = {"map": ref_maps[kk], "ci_low": [], "ci_high": [], "component_stability": [],
                     "selection_frequency": []}
            for vi in range(len(c_ref)):
                a, b = sizes[vi], sizes[vi + 1]
                rm = ref_maps[kk][vi].double().numpy()
                entry["ci_low"].append(torch.as_tensor(mlo[a:b]).float())
                entry["ci_high"].append(torch.as_tensor(mhi[a:b]).float())
                stab = []
                for j in range(k):
                    rs = []
                    for t in range(mal.shape[0]):
                        x_, y_ = mal[t, a:b, j], rm[:, j]
                        ok = np.isfinite(x_) & np.isfinite(y_)
                        rs.append(np.corrcoef(x_[ok], y_[ok])[0, 1] if ok.sum() > 2 and x_[ok].std() > 0
                                  and y_[ok].std() > 0 else np.nan)
                    stab.append(float(np.nanmedian(rs)) if np.isfinite(rs).any() else float("nan"))
                entry["component_stability"].append(stab)
                sel = []
                for t in range(mal.shape[0]):
                    h = np.nansum(mal[t, a:b] ** 2, axis=1)
                    pick = np.zeros(b - a, bool)
                    if np.isfinite(mal[t, a:b]).any():
                        pick[np.argsort(-h)[:qs[vi]]] = True
                    sel.append(pick)
                entry["selection_frequency"].append(torch.as_tensor(np.mean(sel, axis=0)).float())
            out["maps"][kk] = entry
    if n_permutations > 0:
        null = [max(float(np.abs(c).max()) for c in rec["structure"])
                for rec in records[len(sub_idx):]]
        out["null_max_abs"] = float(np.percentile(null, 95))
    return out


def _fit_structure(task):
    """One refit, returning structure coefficients and out-of-bag diagnostics as
    plain NumPy/Python objects. Module level so worker processes can import it."""
    from .api import fit
    map_opts = None
    if len(task) == 5:
        method, views, k, random_state, params = task
        oob_views = None
    elif len(task) == 6:
        method, views, oob_views, k, random_state, params = task
    else:
        method, views, oob_views, k, random_state, params, map_opts = task
    r = fit(method, views, k, random_state=random_state, **params)
    er_key = "joint_effective_rank" if r.get("basis_is_joint") else "effective_rank"
    oos_r2, oos_coup = _oob_metrics(r, views, oob_views)
    return {
        "structure": [c.double().numpy() for c in r["structure"]],
        "converged": bool((r.get("convergence") or {}).get("converged", True)),
        "effective_rank": [float(x) for x in r[er_key]],
        "oos_r2": oos_r2,
        "oos_coupling": oos_coup,
        "maps": ({kk: [t.double().numpy() for t in vv] for kk, vv in feature_maps(r, views, **map_opts).items()}
                 if map_opts else None),
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


def select_rank(method: str, views: Any, ks: Sequence[int], *,
                criterion: str = "component", n_subsamples: int = 20,
                fraction: float = 0.5, min_stability: float = 0.9,
                min_oos_coupling: Optional[float] = None,
                random_state: int = 0, n_jobs: int = 1, **params: Any) -> Dict[str, Any]:
    """Choose ``k`` under whole-pipeline subsample refits and out-of-bag validation.

    For every ``k`` in ``ks``, `loading_inference` refits ``method`` on
    ``n_subsamples`` random ``fraction``-subsamples and evaluates each refit on
    its unused ``1 - fraction`` complement. One pass produces five rank rules
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

    Each row in ``table`` also reports ``converged_fraction``,
    ``min_effective_rank`` and ``basis_degenerate`` (``min_effective_rank < k -
    0.5``, which indicates merged basis columns — raise ``nsa_w`` when True).
    """
    import contextlib
    if criterion not in RANK_CRITERIA:
        raise ValueError(f"criterion must be one of {RANK_CRITERIA}; got {criterion!r}.")
    ks_sorted = sorted(dict.fromkeys(int(x) for x in ks))
    if not ks_sorted or ks_sorted[0] < 1:
        raise ValueError("ks must be a non-empty sequence of positive integers.")
    rows = []
    workers = _n_workers(n_jobs)
    with (_pool(workers) if workers > 1 else contextlib.nullcontext()) as ex:
        for k in ks_sorted:
            rows.append(_rank_row(method, views, k, n_subsamples, fraction, min_stability,
                                  min_oos_coupling, random_state, n_jobs, ex, params))
    criteria = _select_by_criteria(rows)
    return {"k": criteria[criterion], "criterion": criterion, "criteria": criteria, "table": rows}


def _select_by_criteria(rows: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    comp = [r["k"] for r in rows if r["all_stable"]]
    sub = [r["k"] for r in rows if r["subspace_stable"]]
    coup = [r["k"] for r in rows if r["all_coupled"]]
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
    }


def _rank_row(method, views, k, n_subsamples, fraction, min_stability, min_oos_coupling,
              random_state, n_jobs, executor, params):
    """One `select_rank` table row at rank ``k``."""
    inf = loading_inference(method, views, k, n_subsamples=n_subsamples, fraction=fraction,
                            random_state=random_state, n_jobs=n_jobs, _executor=executor,
                            **params)
    cs = np.nanmedian(np.asarray(inf["component_stability"], float), axis=0)
    ss = np.asarray(inf["subspace_stability"], float)
    splits = inf["oos_r2_splits"]
    if splits is not None and len(splits) > 0:
        sp = np.asarray(splits, float)
        oos_r2 = float(np.nanmean(sp))
        oos_r2_se = float(np.nanstd(sp, ddof=1) / np.sqrt(len(sp))) if len(sp) > 1 else 0.0
    else:
        oos_r2 = oos_r2_se = None
    coup = inf["oos_coupling"]
    if coup is not None:
        cp = np.asarray(coup, float)
        min_coup = float(np.min(cp))
        thr = (float(min_oos_coupling) if min_oos_coupling is not None
               else float(np.tanh(_FISHER_Z_001 / np.sqrt(max(1, inf["oos_n"] - 3)))))
        all_coupled = bool(min_coup >= thr)
    else:
        min_coup = thr = None
        all_coupled = False
    min_er = float(np.min(inf["effective_rank"]))
    return {
        "k": k,
        "component_stability": [float(x) for x in cs],
        "min_stability": float(np.min(cs)),
        "all_stable": bool(np.all(cs >= min_stability)),
        "subspace_stability": [float(x) for x in ss],
        "min_subspace_stability": float(np.min(ss)),
        "subspace_stable": bool(np.min(ss) >= min_stability),
        "oos_r2": oos_r2,
        "oos_r2_se": oos_r2_se,
        "oos_r2_per_view": inf["oos_r2"],
        "oos_coupling": coup,
        "min_oos_coupling": min_coup,
        "oos_coupling_threshold": thr,
        "all_coupled": all_coupled,
        "converged_fraction": inf["converged_fraction"],
        "min_effective_rank": min_er,
        "basis_degenerate": bool(min_er < k - 0.5),
    }
