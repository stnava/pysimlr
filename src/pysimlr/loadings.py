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
    c = [(ci.double() @ rot).float() for ci in c0]
    load, margin = olee_loadings(c, _result_positivity(result))
    return {"structure": c, "loadings": load, "loading_margin": margin,
            "loading_rotation": rot.float(),
            "communality": [(ci.double() ** 2).sum(1).float() for ci in c0]}


def _procrustes(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||a R - ref||_F."""
    uu, _, vh = np.linalg.svd(a.T @ ref)
    return uu @ vh


def loading_inference(method: str, views: Any, k: int, *, n_subsamples: int = 100,
                      fraction: float = 0.5, threshold: float = 0.8,
                      n_permutations: int = 0, random_state: int = 0,
                      **params: Any) -> Dict[str, Any]:
    r"""Uncertainty and selection for `structure`, by refitting the whole pipeline.

    Each of ``n_subsamples`` refits uses a random ``fraction`` of the rows
    (without replacement). Its structure coefficients are rotated onto the
    reference fit's (orthogonal Procrustes on all views stacked), since the
    latent space fixes components only up to rotation.

    Returns
    -------
    dict
        ``result`` (the reference `pysimlr.fit` on all rows) and per view:
        ``structure`` (reference), ``ci_low`` / ``ci_high`` (2.5/97.5%
        percentiles of the aligned refits; subsamples of half the rows make
        these conservative), ``component_stability`` (length ``k``: median
        over refits of the correlation between the aligned and reference
        column; per-component claims need this high, e.g. >= 0.9),
        ``selection_frequency`` and ``selected`` (stability selection,
        Meinshausen & Buhlmann 2010: each refit selects the ``q_m`` features of
        largest communality; ``selected`` is frequency >= ``threshold``),
        ``q`` and ``expected_false_selections`` (the bound
        :math:`q^2 / ((2\pi - 1) p)`, valid for ``fraction = 0.5`` and
        ``threshold > 0.5``), and, if ``n_permutations > 0``,
        ``null_max_abs`` (95th percentile of :math:`\max_{f,j} |C|` with the
        rows of every view but the first permuted independently, i.e. no
        shared signal).

    ``q_m`` is the number of features the fit's own ``sparseness`` keeps,
    ``round((1 - sparseness) p_m)``, or ``ceil(sqrt(p_m))`` without sparsity.
    """
    from .api import as_views, fit
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction}.")
    if not 0.5 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0.5, 1]; got {threshold}.")
    view_list = as_views(views)
    n = view_list[0].shape[0]
    rng = np.random.default_rng(random_state)
    ref = fit(method, view_list, k, random_state=random_state, **params)
    c_ref = [c.double().numpy() for c in ref["structure"]]
    stacked_ref = np.vstack(c_ref)
    sp = ref.get("sparseness")
    sp = float(sp[0] if isinstance(sp, (list, tuple)) else sp) if sp is not None else None
    qs = [max(1, int(round((1 - sp) * c.shape[0]))) if sp else int(np.ceil(np.sqrt(c.shape[0])))
          for c in c_ref]
    sizes = np.cumsum([0] + [c.shape[0] for c in c_ref])
    aligned, chosen = [], [[] for _ in c_ref]
    m = max(k + 2, int(round(fraction * n)))
    for _ in range(int(n_subsamples)):
        idx = torch.as_tensor(np.sort(rng.choice(n, m, replace=False)))
        sub = [x[idx] for x in view_list]
        r = fit(method, sub, k, random_state=random_state, **params)
        st = np.vstack([c.double().numpy() for c in r["structure"]])
        aligned.append(st @ _procrustes(st, stacked_ref))
        for vi, c in enumerate(r["structure"]):
            h = (c.double() ** 2).sum(1).numpy()
            s = np.zeros(h.shape[0], bool)
            s[np.argsort(-h)[:qs[vi]]] = True
            chosen[vi].append(s)
    al = np.stack(aligned)                      # (B, p_total, k)
    lo, hi = np.percentile(al, 2.5, axis=0), np.percentile(al, 97.5, axis=0)
    out = {"result": ref, "structure": ref["structure"], "ci_low": [], "ci_high": [],
           "component_stability": [], "selection_frequency": [], "selected": [],
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
    if n_permutations > 0:
        null = []
        for _ in range(int(n_permutations)):
            perm = [view_list[0]] + [x[torch.as_tensor(rng.permutation(n))] for x in view_list[1:]]
            r = fit(method, perm, k, random_state=random_state, **params)
            null.append(max(float(c.abs().max()) for c in r["structure"]))
        out["null_max_abs"] = float(np.percentile(null, 95))
    return out
