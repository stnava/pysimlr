r"""Score-level path model: edges between views, tested, compared, and used for prediction.

This module provides `path_model` (edge-wise statistics, Freedman-Lane fixed-model and
refit nulls with joint sweeps, multiplicity correction, omnibus and graph fit tests,
bootstrap effect-size intervals, and out-of-bag refit intervals), `compare_path_models`
(structural comparison with the measurement model fixed, and repeated-split
training-topology comparison via refits), `predict_path` (missing-view and outcome
prediction across path graphs), and `view_scores` (the single score extractor for the
package, also used by `loadings._oob_metrics` and `utils.heldout_coupling`).

The model
---------
Any `pysimlr.fit` result is the measurement model. `view_scores` puts each view's scores into the
reference frame of ``result["structure"]`` (rotated by ``loading_rotation``, sign-aligned to
``u`` on the training rows) and standardises them on the evaluation rows. Linear and deep fits
score each view through their own latents (``source="latents"``); joint-basis fits (``pca``,
``nsa_pipeline``, ``gcca``) score each view through its block of ``v`` (``"joint_block"``).

The structural model is a graph over nodes: the views, optionally an outcome block
(``"outcome"``) and, for diagnostics only, the consensus. For an edge (a, b) with conditioning
set S (the nuisance design N: intercept, standardised covariates, group dummies; under
``conditioning="parents"`` also the scores of the other neighbours of a and b; the consensus
never enters S) both blocks are residualised on S (rank-revealing QR): ``Ra``, ``Rb``.

* ``pillai``: mean squared canonical correlation of (Ra, Rb). In [0, 1], invariant to rotation
  of either block. The primary edge statistic.
* ``r`` per component j: partial correlation of z_aj and z_bj given S, and ``beta`` its slope
  (z_bj on z_aj). Conditioned on the reference rotation, as every per-component claim in this
  package is. Into a one-column outcome, each component of the view is correlated with the
  outcome; between multi-column blocks of different width only ``pillai`` is reported.
* ``rv``: adjusted RV coefficient of (Ra, Rb); reported, not tested.
* ``moderation`` (with ``groups``): per component, the F statistic of adding ``Ra_j * G`` (group
  dummies) to the regression of ``Rb_j`` on [S, ``Ra_j``]; the edge statistic is the sum over
  components. It asks whether the edge differs across groups.

Inference
---------
Fixed-model null (Freedman and Lane 1983): per sweep, one permutation per node, shared by all of
its columns; an edge's statistic is recomputed with ``Z_b* = P_S Z_b + pi_b(Rb)`` (only the part
of Z_b that S does not explain is permuted; for moderation, the residual of the reduced model
[S, Ra_j]). All edges are recomputed in the same sweep, so each sweep's maximum over a family is a
draw from that family's joint null and ``p_maxt`` (single-step max-T) controls the family-wise
error under the global null; subset pivotality is assumed, not proved, so Holm (``p_holm``) and
Benjamini-Hochberg (``q_bh``) are reported beside it. Families: ``edges`` (pillai, every tested
edge), ``edge_components`` (|r|), ``moderation`` (sum F per edge), ``outcome_edges`` (pillai of
the edges into the outcome: the ANCOVA question). p-values are add-one,
``(1 + #{null >= observed}) / (1 + B)``, floor ``1 / (B + 1)``; never 0.

Effect-size intervals: ``n_bootstrap`` subject-bootstrap replicates of the evaluation rows with
the measurement model fixed (percentile, 95%) for ``pillai``, ``r`` and ``beta``. With
``inference=`` (a `loading_inference` result) each subsample refit's out-of-bag scores, already
computed by `loadings._oob_metrics`, give the edge's ``pillai`` under a refitted measurement
model; their 2.5/97.5% percentiles are ``ci_refit_low`` / ``ci_refit_high`` (no extra fits).

Graph fit (``conditioning="parents"``): ``present_pass`` is the fraction of hypothesised edges
with ``p_holm < alpha``; ``absent_pass`` the fraction of absent edges with ``p_perm >= alpha``
(unadjusted: the claim is independence, and correction would make it easier to pass);
``max_absent_pillai`` with its max-T p over the absent edges is the permutation analogue of the
SEM chi-square test (the graph is rejected when that p is below alpha).

What the test cannot say
------------------------
On training rows a wide view (p_m >= n_train) couples to anything, so fixed-model p-values are
refused there unless ``allow_train_fixed=True``; use held-out rows or the refit null. ``pillai``
is an effect size in the fitted model's frame, not evidence that the components are the right
ones. Conditioning on *estimated* scores attenuates an absent edge rather than removing it: the
scores of the middle node of a chain a - b - c measure its latent with error, so partialling them
out leaves residual a - c association, and the graph-fit test errs toward rejecting absent edges,
more so for small k and noisy views (planted chain, n = 500: the absent edge's ``p_perm`` given
its parent was 0.027 and 0.072 on two draws). Read ``fit["absent_p"]`` beside the pass rates.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .stats import bh_fdr, holm

_CHUNK = 500
_RIDGE = 1e-8
_RANK_TOL = 1e-10
_NULL_STORE = 2000


# ----------------------------------------------------------------------------- scores
def _np(t: Any) -> np.ndarray:
    return np.asarray(torch.as_tensor(t).detach().cpu().double())


def _u_train(result: Dict[str, Any]) -> Any:
    u = result["u"]
    return u[0] if isinstance(u, list) else u


def _is_joint(result: Dict[str, Any]) -> bool:
    return bool(result.get("basis_is_joint")) or len(result.get("latents") or []) != len(result["v"])


def _view_list(result: Dict[str, Any], views: Any) -> List[Any]:
    """Views as a list in the fit's order with ``None`` for a missing view."""
    m_views = len(result["v"])
    if isinstance(views, Mapping):
        from .extension import view_names as _names
        names = list(_names(result))[:m_views]
        extra = [k for k in views if k not in names]
        if extra:
            raise ValueError(f"unknown view name(s) {extra}; the fit has {names}.")
        return [views.get(nm) for nm in names]
    vl = list(views)
    if len(vl) != m_views:
        raise ValueError(f"views has {len(vl)} entries; the fit has {m_views} (use None for a missing view).")
    return vl


def _block_scores(result: Dict[str, Any], x: Any, m: int, rot: np.ndarray) -> np.ndarray:
    """One view's scores through its block of a joint basis: preprocess(x) @ (v_m @ rot)."""
    from .api import _preprocess_views
    from .extension import _to_tensor
    xs, _ = _preprocess_views([_to_tensor(x)], result["scale_list"], [result["provenance_list"][m]])
    return xs[0].double().numpy() @ (result["v"][m].detach().double().numpy() @ rot)


def view_scores(result: Dict[str, Any], views: Any, *, train_views: Any = None,
                standardize: bool = True, rotate: bool = True,
                signs: Optional[Sequence[Optional[np.ndarray]]] = None,
                sign_against: Any = None, _pred: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Per-view scores of ``views`` in the frame of ``result["structure"]``.

    ``views`` is a list in the fit's order (``None`` for a missing view) or a dict by view name.
    Each view's scores are rotated by ``loading_rotation`` (``rotate=False`` keeps the method's
    native frame), each column's sign is set so it covaries non-negatively with ``u`` on the
    training rows, and, with ``standardize=True``, columns are standardised on the evaluation
    rows. Joint-basis fits score each view through its block of ``v``; their training-row signs
    need ``train_views`` (the fit's training views); without them the sign is learned on the
    evaluation rows against the evaluation consensus (``sign_rows="evaluation"``). ``signs``
    overrides the learned signs (one ``(k,)`` array of +-1 per view). ``sign_against`` (an
    ``(n, k)`` consensus on the evaluation rows, in the native frame) learns every sign on the
    evaluation rows against it instead (``sign_rows="evaluation"``); `feature_maps` uses this with
    its own ``u``.

    Returns ``scores`` (list of ``(n, k)`` float64 arrays, ``None`` for a missing view),
    ``consensus`` (``(n, k)``), ``signs``, ``source`` (``"latents"`` or ``"joint_block"``),
    ``sign_rows`` (``"train"``, ``"evaluation"`` or ``"given"``), ``available`` (view indices),
    ``n``, and the training-row counterparts ``train_scores`` (``None`` for a joint-basis fit
    without ``train_views``) and ``train_consensus``.
    """
    from .api import predict
    from .extension import view_names as _names
    m_views = len(result["v"])
    vlist = _view_list(result, views)
    avail = [m for m, x in enumerate(vlist) if x is not None]
    if not avail:
        raise ValueError("no view is present.")
    joint = _is_joint(result)
    k = int(_u_train(result).shape[1])
    rot = result["loading_rotation"].detach().double().numpy() if rotate else np.eye(k)
    pred = predict(result, vlist) if _pred is None else _pred
    u_tr = _np(_u_train(result)) @ rot
    if not joint:
        lat = pred["latents"]
        names = list(_names(result))
        z_ev = {m: _np(lat[names[m]] if isinstance(lat, dict) else lat[m]) @ rot for m in avail}
        z_tr = [_np(z) @ rot for z in result["latents"]]
        source = "latents"
    else:
        z_ev = {m: _block_scores(result, vlist[m], m, rot) for m in avail}
        tv = None if train_views is None else _view_list(result, train_views)
        if tv is not None and any(t is None for t in tv):
            raise ValueError("train_views must contain every view.")
        z_tr = None if tv is None else [_block_scores(result, tv[m], m, rot) for m in range(m_views)]
        source = "joint_block"
    u_ev = _np(pred["u"]) @ rot
    u_tr_c = u_tr - u_tr.mean(0, keepdims=True)
    sgn: List[Optional[np.ndarray]] = [None] * m_views
    if signs is not None and sign_against is not None:
        raise ValueError("pass signs or sign_against, not both.")
    if signs is not None:
        sgn = [None if s is None else np.asarray(s, float) for s in signs]
        sign_rows = "given"
    elif sign_against is not None:
        ua = _np(sign_against) @ rot
        if ua.shape != (int(z_ev[avail[0]].shape[0]), k):
            raise ValueError(f"sign_against has shape {ua.shape}; expected ({z_ev[avail[0]].shape[0]}, {k}).")
        uac = ua - ua.mean(0, keepdims=True)
        for m in avail:
            cov = ((z_ev[m] - z_ev[m].mean(0, keepdims=True)) * uac).sum(0)
            sgn[m] = np.where(cov >= 0.0, 1.0, -1.0)
        sign_rows = "evaluation"
    elif z_tr is not None:
        for m in range(m_views):
            cov = ((z_tr[m] - z_tr[m].mean(0, keepdims=True)) * u_tr_c).sum(0)
            sgn[m] = np.where(cov >= 0.0, 1.0, -1.0)
        sign_rows = "train"
    else:
        u_ev_c = u_ev - u_ev.mean(0, keepdims=True)
        for m in avail:
            cov = ((z_ev[m] - z_ev[m].mean(0, keepdims=True)) * u_ev_c).sum(0)
            sgn[m] = np.where(cov >= 0.0, 1.0, -1.0)
        sign_rows = "evaluation"
    fin = _zscore if standardize else (lambda a: a)
    out: List[Optional[np.ndarray]] = [None] * m_views
    for m in avail:
        if sgn[m] is None:
            raise ValueError(f"no sign given for view {m}.")
        out[m] = fin(z_ev[m] * sgn[m][None, :])
    train_scores = None
    if z_tr is not None and all(s is not None for s in sgn):
        train_scores = [fin(z_tr[m] * sgn[m][None, :]) for m in range(m_views)]
    return {"scores": out, "consensus": fin(u_ev), "signs": sgn, "source": source,
            "sign_rows": sign_rows, "available": avail, "n": int(out[avail[0]].shape[0]),
            "train_scores": train_scores, "train_consensus": fin(u_tr)}


def _zscore(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    zc = z - z.mean(0, keepdims=True)
    sd = zc.std(0, keepdims=True)
    return np.where(sd > 0, zc / np.where(sd > 0, sd, 1.0), 0.0)


# ----------------------------------------------------------------------------- design
def _nuisance(n: int, covariates=None, groups=None) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """Intercept, standardised covariates, one-hot groups minus one level; and the dummies."""
    cols, names = [np.ones((n, 1))], ["intercept"]
    if covariates is not None:
        c = np.asarray(covariates, dtype=float)
        if c.shape[0] != n:
            raise ValueError(f"covariates has {c.shape[0]} rows; the views have {n}.")
        c = c.reshape(n, -1)
        if not np.isfinite(c).all():
            raise ValueError("covariates contain non-finite values.")
        cols.append(_zscore(c))
        names += [f"cov{i}" for i in range(c.shape[1])]
    dummies = None
    if groups is not None:
        g = np.asarray(groups).ravel()
        if len(g) != n:
            raise ValueError(f"groups has {len(g)} rows; the views have {n}.")
        levels = sorted(set(g.tolist()))
        if len(levels) < 2:
            raise ValueError("groups has a single level; it cannot define a group factor.")
        dummies = np.column_stack([(g == lev).astype(float) for lev in levels[1:]])
        cols.append(dummies)
        names += [f"group={lev}" for lev in levels[1:]]
    return np.column_stack(cols), names, dummies


def _orth(S: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column space of S (rank-revealing)."""
    if S.shape[1] == 0:
        return S
    q, r = np.linalg.qr(S)
    d = np.abs(np.diag(r))
    keep = d > _RANK_TOL * max(1.0, d.max())
    return q[:, keep]


def _residualize(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return Z - Q @ (Q.T @ Z)


# ----------------------------------------------------------------------------- multiplicity
def _addone(ge: Any, B: int) -> np.ndarray:
    return (1.0 + np.asarray(ge, float)) / (1.0 + B)


# ----------------------------------------------------------------------------- graphs
def _node_graph(graph, nodes: List[str], directed: bool) -> Dict[str, List[str]]:
    """Adjacency over node names. ``graph`` is None (complete), an edge list, or an adjacency
    dict keyed by index or name."""
    idx = {i: nm for i, nm in enumerate(nodes)}
    adj: Dict[str, List[str]] = {nm: [] for nm in nodes}

    def nm(x):
        return idx[x] if isinstance(x, (int, np.integer)) else str(x)

    if graph is None:
        for a in nodes:
            adj[a] = [b for b in nodes if b != a]
        return adj
    if isinstance(graph, Mapping):
        edges = [(nm(a), nm(b)) for a, bs in graph.items() for b in bs]
    else:
        edges = [(nm(a), nm(b)) for a, b in graph]
    for a, b in edges:
        if a == b or a not in adj or b not in adj:
            raise ValueError(f"edge ({a}, {b}) is not between two distinct nodes of {nodes}.")
        if b not in adj[a]:
            adj[a].append(b)
        if not directed and a not in adj[b]:
            adj[b].append(a)
    return adj


def _parents(adj: Dict[str, List[str]], node: str, directed: bool) -> List[str]:
    if not directed:
        return list(adj[node])
    return [a for a, bs in adj.items() if node in bs]


def _all_pairs(nodes: List[str], directed: bool):
    if directed:
        return [(a, b) for a in nodes for b in nodes if a != b]
    return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]]


def _conditioning_nodes(adj, a: str, b: str, directed: bool, conditioning: str) -> List[str]:
    if conditioning == "pairwise":
        return []
    others = (set(_parents(adj, a, directed)) | set(_parents(adj, b, directed))) - {a, b, "consensus"}
    return sorted(others)


# ----------------------------------------------------------------------------- edge statistics
def _comp_index(ka: int, kb: int) -> List[Tuple[int, int]]:
    """Column pairs (ia, ib) for per-component statistics (plan F5)."""
    if ka == kb:
        return [(j, j) for j in range(ka)]
    if ka == 1:
        return [(0, j) for j in range(kb)]
    if kb == 1:
        return [(j, 0) for j in range(ka)]
    return []


def _edge_design(Z: Dict[str, np.ndarray], N: np.ndarray, a: str, b: str, cond: List[str],
                 dummies: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Residuals and the fixed matrices the sweeps need for edge (a, b) given S = [N, Z_cond]."""
    S = np.column_stack([N] + [Z[c] for c in cond])
    Q = _orth(S)
    Ra, Rb = _residualize(Z[a], Q), _residualize(Z[b], Q)
    n = Ra.shape[0]
    ka, kb = Ra.shape[1], Rb.shape[1]
    if n - Q.shape[1] <= ka + kb:
        raise ValueError(f"edge ({a}, {b}): {n} rows cannot support {Q.shape[1]} conditioning columns "
                         f"and {ka} + {kb} score columns.")
    Qa, Ra_R = np.linalg.qr(Ra)
    for nm_, R_ in ((a, Ra_R), (b, np.linalg.qr(Rb)[1])):
        d = np.abs(np.diag(R_))
        if d.min() <= _RANK_TOL * max(1.0, d.max()):
            raise ValueError(f"edge ({a}, {b}): the scores of {nm_} are collinear with the conditioning set "
                             f"or with each other; the edge statistic is not defined.")
    comps = _comp_index(ka, kb)
    des = {"a": a, "b": b, "cond": cond, "S": S, "Q": Q, "Ra": Ra, "Rb": Rb, "ka": ka, "kb": kb,
           "q": Q.shape[1], "A": torch.as_tensor(np.hstack([Q, Qa])), "RaR": torch.as_tensor(Ra_R),
           "RbtRb": torch.as_tensor(Rb.T @ Rb), "Rb_t": torch.as_tensor(Rb),
           "ra_norm": torch.as_tensor(np.linalg.norm(Ra, axis=0)), "comps": comps, "mod": None}
    if dummies is not None and comps:
        g1 = dummies.shape[1]
        bases, e0s, df1, df2 = [], [], [], []
        for ia, ib in comps:
            x = Ra[:, ia] / np.linalg.norm(Ra[:, ia])
            X0 = np.column_stack([Q, x])
            inter = _residualize(Ra[:, [ia]] * dummies, X0)
            W = _orth(inter)
            Wp = np.zeros((n, g1))
            Wp[:, :W.shape[1]] = W
            bases.append(np.hstack([X0, Wp]))
            e0s.append(Rb[:, ib] - x * (x @ Rb[:, ib]))
            df1.append(W.shape[1])
            df2.append(n - X0.shape[1] - W.shape[1])
        des["mod"] = {"B": torch.as_tensor(np.stack(bases)), "E0": torch.as_tensor(np.column_stack(e0s)),
                      "e0sq": torch.as_tensor(np.array([e @ e for e in e0s])), "x0": Q.shape[1] + 1,
                      "df1": np.array(df1, float), "df2": np.array(df2, float)}
    return des


def _stats_from_perm(des: Dict[str, Any], P: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Statistics of edge (a, b) with b's residual permuted by each row of ``P`` (c, n).

    With ``A = [Q, Qa]`` (Qa an orthonormal basis of Ra, orthogonal to Q) and
    ``T = A^T pi(Rb)``: ``Rb*^T Rb* = Rb^T Rb - T_Q^T T_Q`` and pillai is
    ``trace((Rb*^T Rb*)^{-1} T_a^T T_a) / min(ka, kb)``; ``Ra^T Rb* = RaR^T T_a``.
    The identity permutation gives the observed statistics."""
    RbP = des["Rb_t"][P]                                            # (c, n, kb)
    T = des["A"].transpose(0, 1) @ RbP                              # (c, q + ka, kb)
    q = des["q"]
    Hq, Ha = T[:, :q], T[:, q:]
    G = des["RbtRb"] - Hq.transpose(-1, -2) @ Hq                    # (c, kb, kb)
    M = Ha.transpose(-1, -2) @ Ha
    pil = torch.linalg.solve(G, M).diagonal(dim1=-2, dim2=-1).sum(-1) / min(des["ka"], des["kb"])
    out = {"pillai": pil}
    if des["comps"]:
        C = des["RaR"].transpose(0, 1) @ Ha                          # (c, ka, kb) = Ra^T Rb*
        ia = torch.as_tensor([c[0] for c in des["comps"]])
        ib = torch.as_tensor([c[1] for c in des["comps"]])
        num = C[:, ia, ib]
        gd = G.diagonal(dim1=-2, dim2=-1)[:, ib]
        out["r"] = num / (des["ra_norm"][ia] * torch.sqrt(gd))
    if des["mod"] is not None:
        md = des["mod"]
        E0P = md["E0"][P]                                            # (c, n, kc)
        Tm = torch.einsum("jnd,cnj->cjd", md["B"], E0P)              # (c, kc, d)
        x0 = md["x0"]
        ss0 = md["e0sq"] - (Tm[:, :, :x0] ** 2).sum(-1)
        diff = (Tm[:, :, x0:] ** 2).sum(-1)
        df1 = torch.as_tensor(md["df1"])
        df2 = torch.as_tensor(md["df2"])
        F = torch.where(df1 > 0, (diff / df1.clamp_min(1.0)) / ((ss0 - diff) / df2), torch.zeros_like(diff))
        out["F"] = F
    return out


def _observed(des: Dict[str, Any]) -> Dict[str, Any]:
    n = des["Ra"].shape[0]
    st = _stats_from_perm(des, torch.arange(n)[None, :])
    obs = {"pillai": float(st["pillai"][0])}
    Ra, Rb = des["Ra"], des["Rb"]
    if des["comps"]:
        obs["r"] = st["r"][0].numpy().copy()
        obs["beta"] = np.array([(Ra[:, ia] @ Rb[:, ib]) / (Ra[:, ia] @ Ra[:, ia]) for ia, ib in des["comps"]])
    if des["mod"] is not None:
        obs["F"] = st["F"][0].numpy().copy()
    return obs


def _boot_edge(des: Dict[str, Any], idx: torch.Tensor, Za: torch.Tensor, Zb: torch.Tensor) -> Dict[str, torch.Tensor]:
    """pillai, r and beta on bootstrap rows ``idx`` (c, n): S, Z_a, Z_b resampled, residualised
    by ridge least squares (lambda = 1e-8 trace / cols) in each replicate."""
    S = torch.as_tensor(des["S"])[idx]                              # (c, n, s)
    StS = S.transpose(-1, -2) @ S
    lam = _RIDGE * StS.diagonal(dim1=-2, dim2=-1).sum(-1) / S.shape[-1]
    StS = StS + lam[:, None, None] * torch.eye(S.shape[-1], dtype=S.dtype)
    Xa, Xb = Za[idx], Zb[idx]
    Ra = Xa - S @ torch.linalg.solve(StS, S.transpose(-1, -2) @ Xa)
    Rb = Xb - S @ torch.linalg.solve(StS, S.transpose(-1, -2) @ Xb)
    Saa = Ra.transpose(-1, -2) @ Ra
    Sbb = Rb.transpose(-1, -2) @ Rb
    Sab = Ra.transpose(-1, -2) @ Rb
    pil = (torch.linalg.solve(Saa, Sab) @ torch.linalg.solve(Sbb, Sab.transpose(-1, -2))).diagonal(
        dim1=-2, dim2=-1).sum(-1) / min(des["ka"], des["kb"])
    out = {"pillai": pil}
    if des["comps"]:
        ia = torch.as_tensor([c[0] for c in des["comps"]])
        ib = torch.as_tensor([c[1] for c in des["comps"]])
        num = Sab[:, ia, ib]
        da = Saa.diagonal(dim1=-2, dim2=-1)[:, ia]
        db = Sbb.diagonal(dim1=-2, dim2=-1)[:, ib]
        out["r"] = num / torch.sqrt(da * db)
        out["beta"] = num / da
    return out


def _adjusted_rv(Ra: np.ndarray, Rb: np.ndarray) -> float:
    from .utils import adjusted_rvcoef
    return float(adjusted_rvcoef(torch.as_tensor(Ra).float(), torch.as_tensor(Rb).float()))


def _pct(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    return np.percentile(x, 2.5, axis=axis), np.percentile(x, 97.5, axis=axis)


# ----------------------------------------------------------------------------- path_model
def _wide_views(result: Dict[str, Any], nodes: List[str]) -> List[str]:
    n_train = int(_u_train(result).shape[0])
    return [nodes[m] for m, v in enumerate(result["v"]) if int(v.shape[0]) >= n_train]


def path_model(result: Dict[str, Any], views: Any, *, rows: str = "heldout", graph=None,
               directed: bool = False, view_names: Optional[Sequence[str]] = None,
               covariates=None, groups=None, outcome=None, conditioning: str = "pairwise",
               include_consensus: bool = False, null: Optional[str] = None,
               n_permutations: int = 10_000, n_refits: int = 200, n_bootstrap: int = 2000,
               train_views: Any = None, train_covariates=None, train_groups=None, train_outcome=None,
               inference: Optional[Dict[str, Any]] = None, alpha: float = 0.05, random_state: int = 0,
               n_jobs: int = 1, allow_train_fixed: bool = False,
               refit_scope: Tuple[str, ...] = ("edges", "omnibus")) -> Dict[str, Any]:
    """Edge-wise, omnibus and graph-fit permutation inference on the scores of a fitted model.

    See the module docstring for the statistics, nulls and what the test cannot say. ``views``
    are the rows to test, every view present: ``rows="heldout"`` for subjects not used in fitting
    (default null ``"fixed"``), ``"train"`` for the training rows (default null ``"refit"``).
    ``covariates`` (``(n, q)``), ``groups`` (``(n,)`` labels) and ``outcome`` (``(n,)`` or
    ``(n, r)``) are aligned with ``views``. ``graph`` is None (complete), an edge list or an
    adjacency dict over view indices or node names (``"outcome"`` for the outcome).

    ``null``: ``"fixed"`` (Freedman-Lane sweeps, ``n_permutations``), ``"refit"`` (plan 3.2) or
    ``"both"``. The refit null needs ``rows="train"`` (``views`` are the training views): for each
    view node b ending a tested edge, ``n_refits`` refits of `pysimlr.fit` with the result's
    ``parameters["used"]``, b's rows permuted together with its confound rows
    (`confounds.permute_view_confounds`), give ``p_refit`` for every edge into b (``p_refit_holm``
    across edges; no max-T, since each null permutes one node); edges into the outcome need no
    refit and use Freedman-Lane permutations of the outcome. ``"omnibus"`` in ``refit_scope`` adds
    ``n_refits`` refits with every view but the first permuted: ``omnibus["p_mean_refit"]`` and
    ``["p_max_refit"]``. Refits run through ``n_jobs`` worker processes; all permutations are drawn
    in the parent, so the result does not depend on ``n_jobs``.

    ``train_views`` gives a joint-basis fit its training-row signs. ``inference`` (a
    `loading_inference` result on the training views) adds ``ci_refit_low`` / ``ci_refit_high``
    for ``pillai``; with covariates, groups or an outcome it needs ``train_covariates``,
    ``train_groups``, ``train_outcome`` aligned with the views `loading_inference` was given.

    Returns a dict: ``nodes``, ``graph``, ``directed``, ``conditioning``, ``rows``, ``n``, ``k``,
    ``edges`` (DataFrame: a, b, present, diagnostic, pillai, rv, p_perm, p_maxt, p_holm, q_bh,
    p_refit, ci_low, ci_high, ci_refit_low, ci_refit_high), ``edge_components`` (DataFrame: a, b,
    component, r, beta, p_perm, p_maxt, p_holm, q_bh, ci_low, ci_high, beta_ci_low,
    beta_ci_high), ``moderation`` (DataFrame or None), ``outcome_edges`` (DataFrame or None),
    ``omnibus``, ``fit``, ``families``, ``edge_models`` (for `predict_path`),
    ``feature_contributions`` (DataFrame), ``null`` (per edge, up to 2000 null ``pillai`` draws),
    ``scores_source``, ``signs``, ``nuisance_names``, ``parameters`` and ``warnings``.
    """
    if conditioning not in ("pairwise", "parents"):
        raise ValueError("conditioning must be 'pairwise' or 'parents'.")
    if rows not in ("heldout", "train"):
        raise ValueError("rows must be 'heldout' or 'train'.")
    null = ("refit" if rows == "train" else "fixed") if null is None else null
    if null not in ("fixed", "refit", "both"):
        raise ValueError("null must be 'fixed', 'refit' or 'both'.")
    use_fixed, use_refit = null in ("fixed", "both"), null in ("refit", "both")
    if use_refit and rows != "train":
        raise ValueError("the refit null permutes and refits the training rows; it needs rows='train' and the "
                         "training views. Held-out rows use the fixed-model null.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    B, n_boot, n_ref = int(n_permutations), int(n_bootstrap), int(n_refits)
    if B < 0 or n_boot < 0 or n_ref < 0:
        raise ValueError("n_permutations, n_refits and n_bootstrap must be non-negative.")
    if not use_fixed:
        B = 0
    vlist = _view_list(result, views)
    if any(v is None for v in vlist):
        raise ValueError("path_model needs every view; use predict_path for missing views.")
    sc = view_scores(result, vlist, train_views=train_views)
    m_views = len(sc["scores"])
    n = sc["n"]
    from .extension import view_names as _names
    nodes = list(view_names) if view_names is not None else list(_names(result))[:m_views]
    if len(nodes) != m_views:
        raise ValueError(f"{len(nodes)} view names for {m_views} views.")
    if len(set(nodes)) != len(nodes) or {"outcome", "consensus"} & set(nodes):
        raise ValueError(f"view names must be distinct and not 'outcome' or 'consensus'; got {nodes}.")
    view_nodes = list(nodes)
    Z: Dict[str, np.ndarray] = {nm: z for nm, z in zip(nodes, sc["scores"])}
    if include_consensus:
        nodes.append("consensus")
        Z["consensus"] = sc["consensus"]
    if outcome is not None:
        y = np.asarray(outcome, dtype=float)
        if y.shape[0] != n:
            raise ValueError(f"outcome has {y.shape[0]} rows; the views have {n}.")
        y = y.reshape(n, -1)
        if not np.isfinite(y).all():
            raise ValueError("outcome contains non-finite values.")
        nodes.append("outcome")
        Z["outcome"] = _zscore(y)
    k = int(Z[nodes[0]].shape[1])

    warn: List[str] = []

    def _warn(msg: str) -> None:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        warn.append(msg)

    wide = _wide_views(result, view_nodes)
    report_p = use_fixed
    if rows == "train" and use_fixed:
        if wide and not allow_train_fixed:
            report_p = False
            _warn(f"rows='train' with wide views {wide}: training coupling is not evidence; fixed-model "
                  "p-values are not reported (use held-out rows, the refit null, or allow_train_fixed=True).")
        elif wide:
            _warn(f"rows='train' with wide views {wide} and allow_train_fixed=True: the fixed-model null "
                  "holds the fitted model fixed and ignores that it was fitted to these rows.")
        else:
            _warn("rows='train' with the fixed-model null: it holds the fitted model fixed and ignores that "
                  "it was fitted to these rows; held-out rows or the refit null are the safer evidence.")
    if B == 0:
        report_p = False

    N, nuis_names, dummies = _nuisance(n, covariates, groups)
    adj = _node_graph(graph, nodes, directed)
    pairs = _all_pairs(nodes, directed)
    present = {(a, b): (b in adj[a]) for a, b in pairs}
    diagnostic = {e: ("consensus" in e) for e in pairs}
    tested = [e for e in pairs if not diagnostic[e]]
    if not tested:
        raise ValueError("no edge to test: the model needs at least two nodes besides the consensus.")
    absent = [e for e in tested if not present[e]]
    out_edges = [e for e in tested if "outcome" in e]

    designs = {e: _edge_design(Z, N, e[0], e[1], _conditioning_nodes(adj, e[0], e[1], directed, conditioning),
                               dummies) for e in pairs}
    obs = {e: _observed(designs[e]) for e in pairs}

    # ---- fixed-model null: joint sweeps, one permutation per node per sweep
    ss = np.random.SeedSequence(random_state)
    rng_perm, rng_boot, rng_refit = (np.random.default_rng(s) for s in ss.spawn(3))
    ge_p = {e: 0 for e in pairs}
    ge_r = {e: np.zeros(len(designs[e]["comps"])) for e in pairs}
    ge_f = {e: np.zeros(len(designs[e]["comps"])) for e in pairs if designs[e]["mod"] is not None}
    ge_fs = {e: 0 for e in ge_f}
    mods = [e for e in tested if e in ge_f]
    mx = {nm: np.full(B, -np.inf) for nm in ("edges", "absent", "outcome", "components", "moderation")}
    mean_null = np.zeros(B)
    stored: Dict[Tuple[str, str], List[float]] = {e: [] for e in pairs}
    base = np.tile(np.arange(n), (min(_CHUNK, max(B, 1)), 1))
    for s0 in range(0, B, _CHUNK):
        c = min(_CHUNK, B - s0)
        perms = {nm: torch.as_tensor(rng_perm.permuted(base[:c], axis=1)) for nm in nodes}
        sl = slice(s0, s0 + c)
        for e in pairs:
            st = _stats_from_perm(designs[e], perms[e[1]])
            pil = st["pillai"].numpy()
            ge_p[e] += int((pil >= obs[e]["pillai"]).sum())
            if len(stored[e]) < _NULL_STORE:
                stored[e].extend(pil[:_NULL_STORE - len(stored[e])].tolist())
            if "r" in st:
                ar = np.abs(st["r"].numpy())
                ge_r[e] += (ar >= np.abs(obs[e]["r"])[None, :]).sum(0)
            if e in ge_f:
                F = st["F"].numpy()
                ge_f[e] += (F >= obs[e]["F"][None, :]).sum(0)
                ge_fs[e] += int((F.sum(1) >= obs[e]["F"].sum()).sum())
            if diagnostic[e]:
                continue
            mx["edges"][sl] = np.maximum(mx["edges"][sl], pil)
            mean_null[sl] += pil / len(tested)
            if e in absent:
                mx["absent"][sl] = np.maximum(mx["absent"][sl], pil)
            if e in out_edges:
                mx["outcome"][sl] = np.maximum(mx["outcome"][sl], pil)
            if "r" in st:
                mx["components"][sl] = np.maximum(mx["components"][sl], ar.max(1))
            if e in ge_f:
                mx["moderation"][sl] = np.maximum(mx["moderation"][sl], F.sum(1))

    def p_of(ge):
        return float(_addone(ge, B)) if report_p else None

    def p_max(family: str, value: float):
        return float(_addone((mx[family] >= value).sum(), B)) if report_p else None

    # ---- bootstrap intervals (measurement fixed, subjects resampled)
    boot: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {e: {"pillai": [], "r": [], "beta": []} for e in pairs}
    Zt = {nm: torch.as_tensor(z) for nm, z in Z.items()}
    for s0 in range(0, n_boot, _CHUNK):
        c = min(_CHUNK, n_boot - s0)
        idx = torch.as_tensor(rng_boot.integers(0, n, (c, n)))
        for e in pairs:
            st = _boot_edge(designs[e], idx, Zt[e[0]], Zt[e[1]])
            for key, val in st.items():
                boot[e][key].append(val.numpy())

    # ---- refit null (plan 3.2): per permuted node, plus a global null for the omnibus
    refit = None
    if use_refit:
        if n != int(_u_train(result).shape[0]):
            raise ValueError(f"the refit null needs the training views; views has {n} rows, the fit "
                             f"{int(_u_train(result).shape[0])}.")
        bad = set(refit_scope) - {"edges", "omnibus"}
        if bad or not refit_scope:
            raise ValueError(f"refit_scope must be a non-empty subset of ('edges', 'omnibus'); got {refit_scope}.")
        refit = _refit_null(result, vlist, view_nodes, Z, N, adj, tested, directed, conditioning, n_ref,
                            rng_refit, n_jobs, obs, designs, scope=tuple(refit_scope))

    # ---- refit intervals from loading_inference's out-of-bag scores
    refit_pillai = _refit_pillai(inference, view_nodes, nodes, pairs, adj, directed, conditioning,
                                 covariates is not None, groups is not None, "outcome" in nodes,
                                 train_covariates, train_groups, train_outcome) if inference is not None else None

    # ---- tables
    edge_rows = []
    for e in pairs:
        a, b = e
        des = designs[e]
        row = {"a": a, "b": b, "present": bool(present[e]), "diagnostic": bool(diagnostic[e]),
               "pillai": obs[e]["pillai"], "rv": _adjusted_rv(des["Ra"], des["Rb"]),
               "p_perm": p_of(ge_p[e]), "p_maxt": None if diagnostic[e] else p_max("edges", obs[e]["pillai"]),
               "p_holm": None, "q_bh": None,
               "p_refit": None if (refit is None or diagnostic[e] or n_ref == 0) else refit["p"].get(e),
               "p_refit_holm": None, "ci_low": None, "ci_high": None,
               "ci_refit_low": None, "ci_refit_high": None}
        if n_boot:
            lo, hi = _pct(np.concatenate(boot[e]["pillai"]))
            row["ci_low"], row["ci_high"] = float(lo), float(hi)
        if refit_pillai is not None:
            lo, hi = _pct(np.asarray(refit_pillai[e]))
            row["ci_refit_low"], row["ci_refit_high"] = float(lo), float(hi)
        edge_rows.append(row)
    edges = pd.DataFrame(edge_rows)
    tmask = ~edges["diagnostic"].to_numpy()
    if report_p:
        pp = edges.loc[tmask, "p_perm"].to_numpy(float)
        edges.loc[tmask, "p_holm"] = holm(pp)
        edges.loc[tmask, "q_bh"] = bh_fdr(pp)
    if refit is not None and n_ref > 0 and "edges" in refit_scope:
        edges.loc[tmask, "p_refit_holm"] = holm(edges.loc[tmask, "p_refit"].to_numpy(float))

    comp_rows = []
    for e in pairs:
        if not designs[e]["comps"]:
            continue
        bt = {key: np.concatenate(boot[e][key]) for key in ("r", "beta")} if n_boot else None
        for j, (ia, ib) in enumerate(designs[e]["comps"]):
            r_obs = float(obs[e]["r"][j])
            row = {"a": e[0], "b": e[1], "component": j, "r": r_obs, "beta": float(obs[e]["beta"][j]),
                   "diagnostic": bool(diagnostic[e]), "p_perm": p_of(ge_r[e][j]),
                   "p_maxt": None if diagnostic[e] else p_max("components", abs(r_obs)),
                   "p_holm": None, "q_bh": None, "ci_low": None, "ci_high": None,
                   "beta_ci_low": None, "beta_ci_high": None}
            if bt is not None:
                lo, hi = _pct(bt["r"][:, j])
                row["ci_low"], row["ci_high"] = float(lo), float(hi)
                lo, hi = _pct(bt["beta"][:, j])
                row["beta_ci_low"], row["beta_ci_high"] = float(lo), float(hi)
            comp_rows.append(row)
    edge_components = pd.DataFrame(comp_rows, columns=["a", "b", "component", "r", "beta", "diagnostic", "p_perm",
                                                       "p_maxt", "p_holm", "q_bh", "ci_low", "ci_high",
                                                       "beta_ci_low", "beta_ci_high"])
    if report_p and len(edge_components):
        cm = ~edge_components["diagnostic"].to_numpy(bool)
        if cm.any():
            pc = edge_components.loc[cm, "p_perm"].to_numpy(float)
            edge_components.loc[cm, "p_holm"] = holm(pc)
            edge_components.loc[cm, "q_bh"] = bh_fdr(pc)

    moderation = None
    if dummies is not None:
        mrows = []
        for e in mods:
            md = designs[e]["mod"]
            for j in range(len(designs[e]["comps"])):
                mrows.append({"a": e[0], "b": e[1], "component": str(j), "F": float(obs[e]["F"][j]),
                              "df1": int(md["df1"][j]), "df2": int(md["df2"][j]), "p_perm": p_of(ge_f[e][j]),
                              "p_maxt": None, "p_holm": None, "q_bh": None})
            fs = float(obs[e]["F"].sum())
            mrows.append({"a": e[0], "b": e[1], "component": "sum", "F": fs, "df1": int(md["df1"].sum()),
                          "df2": None, "p_perm": p_of(ge_fs[e]), "p_maxt": p_max("moderation", fs),
                          "p_holm": None, "q_bh": None})
        moderation = pd.DataFrame(mrows, columns=["a", "b", "component", "F", "df1", "df2", "p_perm", "p_maxt",
                                                  "p_holm", "q_bh"])
        smask = (moderation["component"] == "sum").to_numpy()
        if report_p and smask.any():
            ps = moderation.loc[smask, "p_perm"].to_numpy(float)
            moderation.loc[smask, "p_holm"] = holm(ps)
            moderation.loc[smask, "q_bh"] = bh_fdr(ps)

    outcome_edges = None
    if out_edges:
        orows = [{"a": e[0], "b": e[1], "pillai": obs[e]["pillai"], "p_perm": p_of(ge_p[e]),
                  "p_maxt": p_max("outcome", obs[e]["pillai"]), "p_holm": None, "q_bh": None} for e in out_edges]
        outcome_edges = pd.DataFrame(orows)
        if report_p:
            po = outcome_edges["p_perm"].to_numpy(float)
            outcome_edges["p_holm"] = holm(po)
            outcome_edges["q_bh"] = bh_fdr(po)

    # ---- omnibus and graph fit
    obs_t = np.array([obs[e]["pillai"] for e in tested])
    obs_max, obs_mean = float(obs_t.max()), float(obs_t.mean())
    omnibus = {"max_pillai": obs_max, "mean_pillai": obs_mean, "p_max": p_max("edges", obs_max),
               "p_mean": float(_addone((mean_null >= obs_mean).sum(), B)) if report_p else None,
               "null_mean_pillai": mean_null.tolist(), "n_edges": len(tested),
               "p_mean_refit": None, "p_max_refit": None, "null_mean_pillai_refit": None}
    if refit is not None and n_ref > 0 and "omnibus" in refit_scope:
        omnibus["p_mean_refit"] = float(_addone((refit["null_mean"] >= obs_mean).sum(), n_ref))
        omnibus["p_max_refit"] = float(_addone((refit["null_max"] >= obs_max).sum(), n_ref))
        omnibus["null_mean_pillai_refit"] = refit["null_mean"].tolist()
    fit = _graph_fit(edges, tested, absent, present, obs, conditioning, report_p,
                     refit is not None and n_ref > 0 and "edges" in refit_scope, alpha, p_max)

    families = {"edges": {"n_tests": len(tested), "statistic": "pillai"},
                "edge_components": {"n_tests": int((~edge_components["diagnostic"].to_numpy(bool)).sum())
                                    if len(edge_components) else 0, "statistic": "|r|"},
                "moderation": {"n_tests": len(mods), "statistic": "sum over components of F"},
                "outcome_edges": {"n_tests": len(out_edges), "statistic": "pillai"}}

    edge_models = _fit_node_models(Z, nodes, adj, N, directed)
    feature_contributions = _feature_contributions(result, view_nodes, pairs, present, diagnostic, designs, obs)
    if not use_fixed:
        reason = f"null={null!r}: no fixed-model p-values (p_perm); see p_refit"
    elif not report_p and B > 0:
        reason = "p-values not reported: " + warn[-1] if warn else "p-values not reported"
    else:
        reason = None if B > 0 else "n_permutations = 0: no permutation p-values"
    return {"nodes": nodes, "graph": adj, "directed": directed, "conditioning": conditioning, "rows": rows,
            "n": n, "k": k, "edges": edges, "edge_components": edge_components, "moderation": moderation,
            "outcome_edges": outcome_edges, "omnibus": omnibus, "fit": fit, "families": families,
            "edge_models": edge_models, "feature_contributions": feature_contributions, "result": result,
            "refit_null": None if refit is None else {"per_edge": {e: v.tolist() for e, v in refit["null"].items()},
                                                      "outcome_edges": refit["outcome_note"]},
            "nuisance_names": nuis_names, "null": stored, "scores_source": sc["source"],
            "sign_rows": sc["sign_rows"], "signs": sc["signs"], "wide_views": wide, "p_reason": reason,
            "parameters": {"rows": rows, "null": null, "n_permutations": B, "n_refits": n_refits,
                           "refit_scope": tuple(refit_scope) if use_refit else None,
                           "n_bootstrap": n_boot, "alpha": alpha, "random_state": random_state, "n_jobs": n_jobs,
                           "allow_train_fixed": allow_train_fixed, "conditioning": conditioning,
                           "include_consensus": include_consensus, "directed": directed,
                           "inference": inference is not None,
                           "n_refit_pillai": None if refit_pillai is None else len(next(iter(refit_pillai.values()))),
                           "residualization": f"rank-revealing QR (tolerance {_RANK_TOL}); bootstrap: ridge "
                                              f"lambda = {_RIDGE} * trace / cols",
                           "chunk": _CHUNK},
            "warnings": warn}


def _graph_fit(edges, tested, absent, present, obs, conditioning, report_p, have_refit, alpha,
               p_max) -> Dict[str, Any]:
    """Plan 3.5 from the fixed-model null when it was reported, else from the refit null (no
    max-T: each refit null permutes one node, so ``max_absent_p`` is None)."""
    fit: Dict[str, Any] = {"conditioning": conditioning, "null": None, "present_pass": None, "absent_pass": None,
                           "absent_p": None, "max_absent_pillai": None, "max_absent_p": None,
                           "rejected": None, "reason": None}
    if conditioning != "parents":
        fit["reason"] = "the graph-fit test is defined under conditioning='parents'."
        return fit
    if not (report_p or have_refit):
        fit["reason"] = "p-values were not reported."
        return fit
    p_col, h_col = ("p_perm", "p_holm") if report_p else ("p_refit", "p_refit_holm")
    fit["null"] = "fixed" if report_p else "refit"
    lookup = {(r["a"], r["b"]): r for r in edges.to_dict("records")}
    pres = [e for e in tested if present[e]]
    if pres:
        fit["present_pass"] = float(np.mean([lookup[e][h_col] < alpha for e in pres]))
    if absent:
        fit["absent_pass"] = float(np.mean([lookup[e][p_col] >= alpha for e in absent]))
        fit["absent_p"] = {f"{a} - {b}": float(lookup[(a, b)][p_col]) for a, b in absent}
        fit["max_absent_pillai"] = float(max(obs[e]["pillai"] for e in absent))
        if report_p:
            fit["max_absent_p"] = p_max("absent", fit["max_absent_pillai"])
            fit["rejected"] = bool(fit["max_absent_p"] < alpha)
        else:
            fit["reason"] = "refit null: no joint sweep, so no max-T test over absent edges."
    return fit


# ----------------------------------------------------------------------------- refit null
def _refit_scores_task(args):
    """Refit with the given per-view row permutations (each view's confound rows follow it) and
    return the refit's standardised `view_scores` on its own (permuted) training rows. Module level
    so worker processes can import it."""
    method, k, params, random_state, views, perms = args
    torch.set_num_threads(1)
    from .api import fit
    sub = [v if p is None else v[torch.as_tensor(p)] for v, p in zip(views, perms)]
    prm = dict(params)
    if prm.get("confounds") is not None:
        from .confounds import permute_view_confounds
        prm["confounds"] = permute_view_confounds(prm["confounds"], list(perms))
    # A null refit repeats the observed fit's own warnings (wide views, learning rate) once per
    # refit; the observed fit already raised them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit(method, sub, k, random_state=random_state, **prm)
    return [z for z in view_scores(res, sub, train_views=sub)["scores"]]


def _refit_null(result, vlist, view_nodes, Z, N, adj, tested, directed, conditioning, n_refits, rng, n_jobs,
                obs, designs, scope: Tuple[str, ...] = ("edges", "omnibus")) -> Dict[str, Any]:
    """Refit null (plan 3.2). For each view node b that ends a tested edge, ``n_refits`` refits with
    b's training rows (and its confound rows) permuted serve every tested edge into b. Edges into
    the outcome need no refit (the measurement model does not see the outcome): their null permutes
    the outcome's residual given S (Freedman-Lane), ``n_refits`` times. ``"omnibus"`` adds
    ``n_refits`` refits with every view but the first (and the outcome) permuted, the global null of
    the mean and max ``pillai``."""
    from .extension import _to_tensor
    from .relevance import _run
    views = [_to_tensor(v) for v in vlist]
    n = int(views[0].shape[0])
    method, k = result["method"], int(_u_train(result).shape[1])
    params = dict((result.get("parameters") or {}).get("used") or {})
    rs = params.pop("random_state", (result.get("parameters") or {}).get("random_state"))
    ends = [b for b in view_nodes if any(e[1] == b for e in tested)] if "edges" in scope else []
    tasks, keys = [], []
    for b in ends:
        for _ in range(n_refits):
            tasks.append((method, k, params, rs, views,
                          [rng.permutation(n) if nm == b else None for nm in view_nodes]))
            keys.append(("edge", b))
    out_perm = []
    if "omnibus" in scope:
        for _ in range(n_refits):
            tasks.append((method, k, params, rs, views,
                          [None] + [rng.permutation(n) for _ in view_nodes[1:]]))
            keys.append(("global", None))
            out_perm.append(rng.permutation(n))
    scores = _run(_refit_scores_task, tasks, n_jobs) if tasks else []
    null: Dict[Tuple[str, str], List[float]] = {e: [] for e in tested}
    g_mean, g_max = [], []
    gi = 0
    for (kind, b), zs in zip(keys, scores):
        Zr = dict(zip(view_nodes, zs))
        if "outcome" in Z:
            Zr["outcome"] = Z["outcome"] if kind == "edge" else Z["outcome"][out_perm[gi]]
        es = [e for e in tested if e[1] == b] if kind == "edge" else tested
        vals = []
        for e in es:
            des = _edge_design(Zr, N, e[0], e[1], _conditioning_nodes(adj, e[0], e[1], directed, conditioning))
            v = _observed(des)["pillai"]
            vals.append(v)
            if kind == "edge":
                null[e].append(v)
        if kind == "global":
            g_mean.append(float(np.mean(vals)))
            g_max.append(float(np.max(vals)))
            gi += 1
    note = None
    oe = [e for e in tested if e[1] == "outcome"] if "edges" in scope else []
    if oe:
        P = torch.as_tensor(np.stack([rng.permutation(n) for _ in range(n_refits)])) if n_refits else None
        for e in oe:
            null[e] = [] if P is None else _stats_from_perm(designs[e], P)["pillai"].numpy().tolist()
        note = "edges into the outcome: Freedman-Lane permutations of the outcome (no refit needed)"
    arr = {e: np.asarray(v, float) for e, v in null.items()}
    p = {e: float(_addone((a >= obs[e]["pillai"]).sum(), n_refits)) for e, a in arr.items() if len(a)}
    return {"null": arr, "p": p, "null_mean": np.asarray(g_mean), "null_max": np.asarray(g_max),
            "outcome_note": note}


def _refit_pillai(inference, view_nodes, nodes, pairs, adj, directed, conditioning, has_cov, has_groups,
                  has_outcome, train_covariates, train_groups, train_outcome) -> Dict[Tuple[str, str], List[float]]:
    """Edge pillai on each `loading_inference` subsample refit's out-of-bag scores."""
    recs = inference.get("oob_scores")
    if not recs:
        raise ValueError("inference carries no out-of-bag scores ('oob_scores'); rerun loading_inference.")
    n_rows = int(inference["n_rows"])
    for flag, val, nm in ((has_cov, train_covariates, "train_covariates"), (has_groups, train_groups, "train_groups"),
                          (has_outcome, train_outcome, "train_outcome")):
        if flag and val is None:
            raise ValueError(f"inference= with this model needs {nm} (aligned with the views given to "
                             "loading_inference), so each refit's out-of-bag rows get the same nuisance and nodes.")
        if flag and len(np.asarray(val)) != n_rows:
            raise ValueError(f"{nm} has {len(np.asarray(val))} rows; loading_inference saw {n_rows}.")
    tc = None if train_covariates is None else np.asarray(train_covariates, float).reshape(n_rows, -1)
    tg = None if train_groups is None else np.asarray(train_groups).ravel()
    ty = None if train_outcome is None else np.asarray(train_outcome, float).reshape(n_rows, -1)
    out: Dict[Tuple[str, str], List[float]] = {e: [] for e in pairs if "consensus" not in e}
    for rec in recs:
        rows_ = np.asarray(rec["rows"])
        if len(rec["scores"]) != len(view_nodes):
            raise ValueError(f"inference has {len(rec['scores'])} views; the model has {len(view_nodes)}.")
        Zr = {nm: np.asarray(z, float) for nm, z in zip(view_nodes, rec["scores"])}
        if has_outcome:
            Zr["outcome"] = _zscore(ty[rows_])
        Nr, _, _ = _nuisance(len(rows_), None if tc is None else tc[rows_], None if tg is None else tg[rows_])
        for e in out:
            des = _edge_design(Zr, Nr, e[0], e[1], _conditioning_nodes(adj, e[0], e[1], directed, conditioning))
            out[e].append(_observed(des)["pillai"])
    return out


# ----------------------------------------------------------------------------- feature read-out
def _feature_contributions(result, view_nodes, pairs, present, diagnostic, designs, obs) -> pd.DataFrame:
    """structure_a[:, j] * beta_abj for every present edge whose node a is a view (plan 5)."""
    rows = []
    for e in pairs:
        a, b = e
        if not present[e] or diagnostic[e] or a not in view_nodes or not designs[e]["comps"]:
            continue
        st = result["structure"][view_nodes.index(a)].detach().double().numpy()
        for j, (ia, _) in enumerate(designs[e]["comps"]):
            contrib = st[:, ia] * obs[e]["beta"][j]
            rows.extend({"edge": f"{a} - {b}", "a": a, "b": b, "component": j, "feature": f,
                         "contribution": float(cv)} for f, cv in enumerate(contrib))
    return pd.DataFrame(rows, columns=["edge", "a", "b", "component", "feature", "contribution"])


# ----------------------------------------------------------------------------- node models
def _ridge_fit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    lam = _RIDGE * np.trace(X.T @ X) / max(1, X.shape[1])
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def _fit_node_models(Z, nodes, adj, N, directed):
    models = {}
    for b in nodes:
        pa = [a for a in _parents(adj, b, directed) if a != "consensus"]
        X = np.column_stack([N] + [Z[a] for a in pa])
        W = _ridge_fit(X, Z[b])
        resid = Z[b] - X @ W
        models[b] = {"parents": pa, "W": W, "resid_var": resid.var(0, ddof=X.shape[1]).clip(1e-12),
                     "n_params": int(W.size), "n_nuisance": int(N.shape[1]),
                     "parent_cols": {a: int(Z[a].shape[1]) for a in pa}}
    return models


def _node_loglik(model, N, Zpa: List[np.ndarray], Zb: np.ndarray) -> float:
    X = np.column_stack([N] + Zpa)
    r = Zb - X @ model["W"]
    v = model["resid_var"]
    return float(-0.5 * ((r ** 2) / v).sum() - 0.5 * r.shape[0] * np.log(2 * np.pi * v).sum())


def _n_params(models: Dict[str, Any], Z: Dict[str, np.ndarray], adj, directed: bool) -> int:
    """Parameters of a set of node regressions. Directed: every coefficient plus a residual
    variance per column. Undirected (plan F4): each edge's k_a x k_b block counted once, plus each
    node's nuisance block and residual variances."""
    if directed:
        return int(sum(m["W"].size + len(m["resid_var"]) for m in models.values()))
    tot = sum(m["n_nuisance"] * Z[b].shape[1] + len(m["resid_var"]) for b, m in models.items())
    seen = set()
    for a, bs in adj.items():
        for b in bs:
            if a in models and b in models and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                tot += Z[a].shape[1] * Z[b].shape[1]
    return int(tot)


# ----------------------------------------------------------------------------- comparison
def _graph_loglik(Ztr, Zte, Ntr, Nte, nodes, adj, directed) -> Tuple[np.ndarray, float, int]:
    """Node regressions on parents fitted on the training scores: per-subject held-out Gaussian
    log-likelihood, the training log-likelihood and the parameter count (plan F4)."""
    models = _fit_node_models(Ztr, nodes, adj, Ntr, directed)
    ll_rows = np.zeros(next(iter(Zte.values())).shape[0])
    ll_tr = 0.0
    for b in nodes:
        pa = models[b]["parents"]
        X = np.column_stack([Nte] + [Zte[a] for a in pa])
        r = Zte[b] - X @ models[b]["W"]
        v = models[b]["resid_var"]
        ll_rows += -0.5 * ((r ** 2) / v).sum(1) - 0.5 * np.log(2 * np.pi * v).sum()
        ll_tr += _node_loglik(models[b], Ntr, [Ztr[a] for a in pa], Ztr[b])
    return ll_rows, ll_tr, _n_params(models, Ztr, adj, directed)


def _n_edges(adj, directed: bool) -> int:
    return int(sum(len(v) for v in adj.values()) // (1 if directed else 2))


def compare_path_models(result: Optional[Dict[str, Any]] = None, views_train: Any = None, views_test: Any = None,
                        graphs: Optional[Mapping[str, Any]] = None, *, method: Optional[str] = None,
                        k: Optional[int] = None, views: Any = None, n_splits: int = 5,
                        train_fraction: float = 0.7, fit_params: Optional[Dict[str, Any]] = None,
                        covariates=None, groups=None, outcome=None,
                        directed: bool = False, view_names: Optional[Sequence[str]] = None,
                        covariates_train=None, covariates_test=None, groups_train=None, groups_test=None,
                        outcome_train=None, outcome_test=None, n_bootstrap: int = 2000,
                        n_permutations: int = 2000, alpha: float = 0.05, random_state: int = 0) -> Dict[str, Any]:
    """Compare graphs over views (and an optional outcome node).

    **Structural comparison, measurement model fixed** (plan 4.1): pass ``result`` (a `pysimlr.fit`
    result), ``views_train`` (its training views), ``views_test`` (held-out rows) and ``graphs``
    (``{name: graph}``, graphs as in `path_model`; ``"outcome"`` names the outcome node). Each graph
    is a set of node regressions on parents fitted on the training scores and scored by Gaussian
    log-likelihood on the held-out scores (``heldout_loglik``, higher is better), plus in-sample
    ``bic`` (undirected edges counted once, plan F4) and the graph-fit test of `path_model` on the
    held-out rows. A paired subject bootstrap over held-out rows gives each graph's
    ``delta_vs_top`` (its log-likelihood minus the top graph's) with a 95% CI and
    ``best_fraction`` (share of replicates in which the graph has the top log-likelihood).
    ``best`` (plan F1): the graph with the fewest edges whose delta CI against the top includes 0
    (ties: higher log-likelihood). Held-out likelihood cannot penalise a weakly supported extra
    edge, so the raw top is reported beside it as ``best_by_loglik``. Without bootstrap replicates
    ``best`` is ``best_by_loglik`` and ``best_rule`` says so.

    **Training-topology comparison, refits** (plan 4.2): pass ``method``, ``k``, ``views`` (all
    rows), ``graphs`` and optionally ``fit_params`` (``confounds`` are sliced per split),
    ``covariates``, ``groups``, ``outcome`` (aligned with ``views``). For each of ``n_splits``
    random ``train_fraction`` splits (seed ``random_state + s``) every graph is refitted with
    ``topology="graph", path_graph=<its view edges>`` (``"star"`` for a graph without view edges)
    on the same training rows and evaluated on the same held-out rows: ``heldout_loglik_own`` and
    ``heldout_loglik_complete`` (the 4.1 structural log-likelihood of that fit's scores under its
    own graph and under the complete graph), the graph-fit pass rates under its own graph,
    ``oos_coupling`` (mean held-out leave-one-view-out coupling, `loadings._oob_metrics`),
    ``oos_r2`` (mean adjusted held-out reconstruction R^2) and, with an outcome, ``outcome_r2``
    (held-out R^2 of a least-squares fit of the outcome on the training consensus). The training
    loss is not reported: it is in-sample and graph-conditional. ``splits`` holds one row per
    (split, graph); ``summary`` the mean per graph and the mean paired difference against the
    graph with the highest mean of each criterion; ``best`` the top graph per criterion.
    """
    if not graphs:
        raise ValueError("graphs is required: a non-empty {name: graph}.")
    if result is not None and method is not None:
        raise ValueError("pass result (structural comparison) or method and k (training topologies), not both.")
    if result is None:
        if method is None or k is None or views is None:
            raise ValueError("the training-topology comparison needs method, k and views.")
        return _compare_training_topologies(method, int(k), views, graphs, n_splits=n_splits,
                                            train_fraction=train_fraction, fit_params=fit_params,
                                            covariates=covariates, groups=groups, outcome=outcome,
                                            directed=directed, view_names=view_names,
                                            n_permutations=n_permutations, alpha=alpha, random_state=random_state)
    if views_train is None or views_test is None:
        raise ValueError("the structural comparison needs views_train and views_test.")
    rng = np.random.default_rng(random_state)
    sc_tr = view_scores(result, views_train, train_views=views_train)
    sc_te = view_scores(result, views_test, train_views=views_train)
    from .extension import view_names as _names
    nodes = list(view_names) if view_names is not None else list(_names(result))[:len(sc_tr["scores"])]
    Ztr = {nm: z for nm, z in zip(nodes, sc_tr["scores"])}
    Zte = {nm: z for nm, z in zip(nodes, sc_te["scores"])}
    if outcome_train is not None:
        nodes = nodes + ["outcome"]
        Ztr["outcome"] = _zscore(np.asarray(outcome_train, float).reshape(sc_tr["n"], -1))
        Zte["outcome"] = _zscore(np.asarray(outcome_test, float).reshape(sc_te["n"], -1))
    Ntr, _, _ = _nuisance(sc_tr["n"], covariates_train, groups_train)
    Nte, _, _ = _nuisance(sc_te["n"], covariates_test, groups_test)
    n_te = sc_te["n"]
    boot_idx = [rng.integers(0, n_te, n_te) for _ in range(n_bootstrap)]
    rows, per_subject = [], {}
    for name, g in graphs.items():
        adj = _node_graph(g, nodes, directed)
        ll_rows, ll_tr, n_par = _graph_loglik(Ztr, Zte, Ntr, Nte, nodes, adj, directed)
        per_subject[name] = ll_rows
        pm = path_model(result, views_test, rows="heldout", graph=g, directed=directed, view_names=view_names,
                        covariates=covariates_test, groups=groups_test, outcome=outcome_test,
                        conditioning="parents", n_permutations=n_permutations, n_bootstrap=0, alpha=alpha,
                        random_state=random_state, train_views=views_train)
        rows.append({"graph": name, "n_edges": _n_edges(adj, directed),
                     "heldout_loglik": float(ll_rows.sum()), "n_params": n_par,
                     "bic": float(n_par * np.log(sc_tr["n"]) - 2.0 * ll_tr),
                     "present_pass": pm["fit"]["present_pass"], "absent_pass": pm["fit"]["absent_pass"],
                     "max_absent_p": pm["fit"]["max_absent_p"], "edges": pm["edges"]})
    top = max(rows, key=lambda r: r["heldout_loglik"])["graph"]
    wins = {r["graph"]: 0 for r in rows}
    deltas: Dict[str, List[float]] = {r["graph"]: [] for r in rows}
    for idx in boot_idx:
        tot = {nm: float(per_subject[nm][idx].sum()) for nm in per_subject}
        wins[max(tot, key=tot.get)] += 1
        for nm in tot:
            deltas[nm].append(tot[nm] - tot[top])
    top_ll = next(x["heldout_loglik"] for x in rows if x["graph"] == top)
    for r in rows:
        r["delta_vs_top"] = float(r["heldout_loglik"] - top_ll)
        if n_bootstrap:
            lo, hi = _pct(np.asarray(deltas[r["graph"]]))
            r["ci_low"], r["ci_high"] = float(lo), float(hi)
            r["best_fraction"] = wins[r["graph"]] / n_bootstrap
        else:
            r["ci_low"] = r["ci_high"] = r["best_fraction"] = None
    if n_bootstrap:
        ok = [r for r in rows if r["ci_low"] <= 0.0 <= r["ci_high"]]
        best = min(ok, key=lambda r: (r["n_edges"], -r["heldout_loglik"]))["graph"]
        rule = "fewest edges with delta CI including 0 (plan F1)"
    else:
        best, rule = top, "top held-out log-likelihood (no bootstrap replicates)"
    return {"comparison": rows, "best": best, "best_by_loglik": top, "best_rule": rule, "nodes": nodes,
            "mode": "structural",
            "parameters": {"n_bootstrap": n_bootstrap, "n_permutations": n_permutations,
                           "random_state": random_state, "alpha": alpha}}


_TOPOLOGY_CRITERIA = ("heldout_loglik_own", "heldout_loglik_complete", "oos_coupling", "oos_r2", "outcome_r2")


def _compare_training_topologies(method, k, views, graphs, *, n_splits, train_fraction, fit_params, covariates,
                                 groups, outcome, directed, view_names, n_permutations, alpha,
                                 random_state) -> Dict[str, Any]:
    """Plan 4.2 (see `compare_path_models`)."""
    from .api import as_views, fit, predict
    from .confounds import slice_confounds
    from .extension import _default_names
    from .loadings import _oob_metrics
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}.")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")
    vl = as_views(views)
    n, m_views = int(vl[0].shape[0]), len(vl)
    names = list(view_names) if view_names is not None else list(_default_names(m_views))
    has_y = outcome is not None
    nodes = names + (["outcome"] if has_y else [])
    y_all = None if not has_y else np.asarray(outcome, float).reshape(n, -1)
    c_all = None if covariates is None else np.asarray(covariates, float).reshape(n, -1)
    g_all = None if groups is None else np.asarray(groups).ravel()
    params = dict(fit_params or {})
    for bad in ("topology", "path_graph"):
        if bad in params:
            raise ValueError(f"fit_params must not set {bad!r}: each graph sets it.")
    adjs = {nm: _node_graph(g, nodes, directed) for nm, g in graphs.items()}
    complete = _node_graph(None, nodes, directed)
    rows, split_rows = [], []
    n_tr = int(round(train_fraction * n))
    for s in range(n_splits):
        perm = np.random.default_rng(random_state + s).permutation(n)
        tr, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
        split_rows.append(te)
        v_tr = [x[torch.as_tensor(tr)] for x in vl]
        v_te = [x[torch.as_tensor(te)] for x in vl]
        prm = dict(params)
        if prm.get("confounds") is not None:
            prm["confounds"] = slice_confounds(prm["confounds"], tr)
        Ntr, _, _ = _nuisance(len(tr), None if c_all is None else c_all[tr], None if g_all is None else g_all[tr])
        Nte, _, _ = _nuisance(len(te), None if c_all is None else c_all[te], None if g_all is None else g_all[te])
        for gname, adj in adjs.items():
            view_adj = {i: sorted(names.index(b) for b in adj[a] if b in names)
                        for i, a in enumerate(names)}
            topo = "graph" if any(view_adj.values()) else "star"
            extra = {"topology": topo} if topo == "star" else {"topology": "graph", "path_graph": view_adj}
            res = fit(method, v_tr, k, random_state=random_state, **prm, **extra)
            sc_tr = view_scores(res, v_tr, train_views=v_tr)
            sc_te = view_scores(res, v_te, train_views=v_tr)
            Ztr = dict(zip(names, sc_tr["scores"]))
            Zte = dict(zip(names, sc_te["scores"]))
            if has_y:
                Ztr["outcome"], Zte["outcome"] = _zscore(y_all[tr]), _zscore(y_all[te])
            ll_own, _, _ = _graph_loglik(Ztr, Zte, Ntr, Nte, nodes, adj, directed)
            ll_cmp, _, _ = _graph_loglik(Ztr, Zte, Ntr, Nte, nodes, complete, directed)
            pm = path_model(res, v_te, graph=graphs[gname], directed=directed, view_names=names,
                            covariates=None if c_all is None else c_all[te], groups=None if g_all is None else g_all[te],
                            outcome=None if not has_y else y_all[te], conditioning="parents",
                            n_permutations=n_permutations, n_bootstrap=0, alpha=alpha,
                            random_state=random_state + s, train_views=v_tr)
            r2, coup = _oob_metrics(res, v_tr, v_te)
            row = {"split": s, "graph": gname, "topology": topo, "n_edges": _n_edges(adj, directed),
                   "n_test": len(te), "heldout_loglik_own": float(ll_own.sum()),
                   "heldout_loglik_complete": float(ll_cmp.sum()),
                   "present_pass": pm["fit"]["present_pass"], "absent_pass": pm["fit"]["absent_pass"],
                   "oos_coupling": None if coup is None else float(np.mean(coup)),
                   "oos_r2": None if r2 is None else float(np.mean(r2)), "outcome_r2": None}
            if has_y:
                u_tr = _np(_u_train(res))
                X = np.column_stack([np.ones(len(tr)), u_tr])
                W = _ridge_fit(X, y_all[tr])
                yhat = np.column_stack([np.ones(len(te)), _np(predict(res, v_te)["u"])]) @ W
                yt = y_all[te]
                row["outcome_r2"] = float(1.0 - ((yt - yhat) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
            rows.append(row)
    splits = pd.DataFrame(rows)
    crit = [c for c in _TOPOLOGY_CRITERIA if splits[c].notna().all()]
    summ = splits.groupby("graph", sort=False).agg(
        {"topology": "first", "n_edges": "first", **{c: "mean" for c in crit}}).reset_index()
    best = {}
    for c in crit:
        top = summ.loc[summ[c].astype(float).idxmax(), "graph"]
        best[c] = top
        wide = splits.pivot(index="split", columns="graph", values=c).astype(float)
        d = wide.sub(wide[top], axis=0)
        summ[f"delta_{c}"] = summ["graph"].map(d.mean())
        summ[f"delta_{c}_min"] = summ["graph"].map(d.min())
        summ[f"delta_{c}_max"] = summ["graph"].map(d.max())
        summ[f"wins_{c}"] = summ["graph"].map(wide.idxmax(axis=1).value_counts()).fillna(0).astype(int)
    return {"mode": "training_topology", "splits": splits, "summary": summ, "best": best, "nodes": nodes,
            "test_rows": split_rows,
            "parameters": {"method": method, "k": k, "n_splits": n_splits, "train_fraction": train_fraction,
                           "n_permutations": n_permutations, "random_state": random_state, "alpha": alpha}}


# ----------------------------------------------------------------------------- prediction
def predict_path(pm: Dict[str, Any], views: Any, *, covariates=None, groups=None, views_true: Any = None,
                 outcome: bool = True, max_iter: int = 10) -> Dict[str, Any]:
    """Scores of every node from the views that are present, through the node regressions.

    ``pm`` is a `path_model` result (it carries its measurement model, ``pm["result"]``, and the
    node regressions ``pm["edge_models"]``, fitted on ``pm``'s rows). ``views`` is a list in the
    fit's order with ``None`` for a missing view, or a dict by view name; ``covariates`` and
    ``groups`` are aligned with its rows. A missing node is predicted by its ridge regression on
    [N, Z_pa] from its neighbours that are present or already predicted (iterated to a fixed point;
    a missing parent contributes its mean, zero); a node with no available neighbour gets the
    nuisance-only prediction (``predicted_from[node] == []``). ``outcome=True`` predicts the outcome
    node when the model has one.

    Returns ``scores`` (node -> ``(n, k)``), ``predicted_from`` (predicted node -> neighbours used),
    ``outcome_pred`` (or None) and, with ``views_true`` (every view, same rows), ``r2_per_node``:
    for each predicted view, 1 - SSE / SST of its predicted scores against `view_scores` of the
    true view. With every view present, ``scores`` equal `view_scores` exactly.
    """
    result = pm["result"]
    vlist = _view_list(result, views)
    sc = view_scores(result, vlist, signs=pm["signs"])
    n = sc["n"]
    N, _, _ = _nuisance(n, covariates, groups)
    if N.shape[1] != pm["edge_models"][pm["nodes"][0]]["n_nuisance"]:
        raise ValueError("covariates / groups do not match the nuisance design the path model was fitted with "
                         f"({pm['nuisance_names']}).")
    nodes, models = pm["nodes"], pm["edge_models"]
    Z: Dict[str, np.ndarray] = {nodes[m]: sc["scores"][m] for m in sc["available"]}
    predicted_from: Dict[str, List[str]] = {}
    skip = {"consensus"} | (set() if outcome else {"outcome"})
    pending = [nm for nm in nodes if nm not in Z and nm not in skip]
    for _ in range(max_iter):
        if not pending:
            break
        progress = False
        for b in list(pending):
            pa = [a for a in models[b]["parents"] if a in Z]
            if pa or not any(a in pending for a in models[b]["parents"]):
                Z[b] = _node_predict(models[b], N, Z)
                predicted_from[b] = pa
                pending.remove(b)
                progress = True
        if not progress:
            break
    for b in pending:
        Z[b] = N @ models[b]["W"][:N.shape[1]]
        predicted_from[b] = []
    out: Dict[str, Any] = {"scores": Z, "predicted_from": predicted_from,
                           "outcome_pred": Z.get("outcome") if outcome else None, "r2_per_node": None}
    if views_true is not None:
        st = view_scores(result, _view_list(result, views_true), signs=pm["signs"])
        r2 = {}
        for m, nm in enumerate(nodes[:len(st["scores"])]):
            if nm in predicted_from:
                t = st["scores"][m]
                r2[nm] = float(1.0 - ((Z[nm] - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())
        out["r2_per_node"] = r2
    return out


def _node_predict(model: Dict[str, Any], N: np.ndarray, Z: Dict[str, np.ndarray]) -> np.ndarray:
    W = model["W"]
    pred = N @ W[:N.shape[1]]
    off = N.shape[1]
    for a in model["parents"]:
        ka = model["parent_cols"][a]
        if a in Z:
            pred = pred + Z[a] @ W[off:off + ka]
        off += ka
    return pred
