r"""Score-level path model: edges between views, tested, compared, and used for prediction.

This module provides `path_model` (edge-wise statistics, Freedman-Lane fixed-model and
refit nulls with joint sweeps, multiplicity correction, omnibus and graph fit tests,
bootstrap effect-size intervals, and out-of-bag refit intervals), `path_model_from_scores`
(the same inference on ``(n, k)`` score blocks from any source), `compare_path_models`
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
On training rows the fixed-model null holds the fitted model fixed and ignores that it was fitted
to those rows, and a wide view (p_m >= n_train) couples to anything there, so fixed-model p-values
are not reported on ``rows="train"`` for any view unless ``allow_train_fixed=True``; the refit null
is the default there, and held-out rows are the evidence. ``pillai``
is an effect size in the fitted model's frame, not evidence that the components are the right
ones. Conditioning on *estimated* scores attenuates an absent edge rather than removing it: the
scores of the middle node of a chain a - b - c measure its latent with error, so partialling them
out leaves residual a - c association, and the graph-fit test errs toward rejecting absent edges,
more so for small k and noisy views (planted chain, n = 500: the absent edge's ``p_perm`` given
its parent was 0.027 and 0.072 on two draws). Read ``fit["absent_p"]`` beside the pass rates.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .loadings import _zscore
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
            sgn[m] = _col_signs(z_ev[m], uac)
        sign_rows = "evaluation"
    elif z_tr is not None:
        for m in range(m_views):
            sgn[m] = _col_signs(z_tr[m], u_tr_c)
        sign_rows = "train"
    else:
        u_ev_c = u_ev - u_ev.mean(0, keepdims=True)
        for m in avail:
            sgn[m] = _col_signs(z_ev[m], u_ev_c)
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


def _col_signs(z: np.ndarray, ref_c: np.ndarray) -> np.ndarray:
    cov = ((z - z.mean(0, keepdims=True)) * ref_c).sum(0)
    return np.where(cov >= 0.0, 1.0, -1.0)


def _zscore_pair(
    a_tr: np.ndarray,
    a_te: Optional[np.ndarray],
    std_mode: str = "per_split",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if std_mode == "per_split":
        return _zscore(a_tr), (None if a_te is None else _zscore(a_te))
    if std_mode == "train":
        mu = a_tr.mean(0, keepdims=True)
        sd = a_tr.std(0, keepdims=True)
        sd_safe = np.where(sd > 0, sd, 1.0)
        z_tr = np.where(sd > 0, (a_tr - mu) / sd_safe, 0.0)
        z_te = None if a_te is None else np.where(sd > 0, (a_te - mu) / sd_safe, 0.0)
        return z_tr, z_te
    return a_tr, a_te


def _check_node_names(
    names: Sequence[str],
    m_views: Optional[int] = None,
    label: str = "view",
) -> List[str]:
    out = [str(nm) for nm in names]
    if m_views is not None and len(out) != m_views:
        raise ValueError(f"{len(out)} {label} names for {m_views} views.")
    if len(set(out)) != len(out) or {"outcome", "consensus"} & set(out):
        raise ValueError(f"{label} names must be distinct and not 'outcome' or 'consensus'; got {out}.")
    return out


def _check_outcome(outcome: Any, n: int, label: str = "outcome") -> np.ndarray:
    y = np.asarray(outcome, dtype=float)
    if y.shape[0] != n:
        raise ValueError(f"{label} has {y.shape[0]} rows; the views have {n}.")
    y = y.reshape(n, -1)
    if not np.isfinite(y).all():
        raise ValueError(f"{label} contains non-finite values.")
    return y


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(1.0 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean(0)) ** 2).sum())


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
    a_stack = np.hstack([Q, Qa])
    ra_norm_t = torch.as_tensor(np.linalg.norm(Ra, axis=0))
    ia_t = torch.as_tensor([c[0] for c in comps], dtype=torch.long) if comps else None
    ib_t = torch.as_tensor([c[1] for c in comps], dtype=torch.long) if comps else None
    s_contig = np.ascontiguousarray(S)
    des = {"a": a, "b": b, "cond": cond, "S": S, "Q": Q, "Ra": Ra, "Rb": Rb, "ka": ka, "kb": kb,
           "q": Q.shape[1], "A": torch.as_tensor(a_stack),
           "A_T": torch.as_tensor(np.ascontiguousarray(a_stack.T)),
           "RaR": torch.as_tensor(Ra_R),
           "RaR_T": torch.as_tensor(np.ascontiguousarray(Ra_R.T)),
           "RbtRb": torch.as_tensor(Rb.T @ Rb),
           "Rb_t": torch.as_tensor(np.ascontiguousarray(Rb)),
           "S_t": torch.as_tensor(s_contig),
           "eye_s": torch.eye(S.shape[1], dtype=torch.float64),
           "ra_norm": ra_norm_t,
           "ia_t": ia_t, "ib_t": ib_t,
           "ra_norm_ia": (ra_norm_t[ia_t] if ia_t is not None else None),
           "comps": comps, "mod": None}
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
        df1_arr = np.array(df1, float)
        df2_arr = np.array(df2, float)
        df1_t = torch.as_tensor(df1_arr)
        des["mod"] = {"B": torch.as_tensor(np.ascontiguousarray(np.stack(bases))),
                      "E0": torch.as_tensor(np.ascontiguousarray(np.column_stack(e0s))),
                      "e0sq": torch.as_tensor(np.array([e @ e for e in e0s])), "x0": Q.shape[1] + 1,
                      "df1": df1_arr, "df2": df2_arr,
                      "df1_pos": (df1_t > 0), "df1_clamp": df1_t.clamp_min(1.0),
                      "df2_t": torch.as_tensor(df2_arr)}
    return des


def _stats_from_perm(des: Dict[str, Any], P: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Statistics of edge (a, b) with b's residual permuted by each row of ``P`` (c, n).

    With ``A = [Q, Qa]`` (Qa an orthonormal basis of Ra, orthogonal to Q) and
    ``T = A^T pi(Rb)``: ``Rb*^T Rb* = Rb^T Rb - T_Q^T T_Q`` and pillai is
    ``trace((Rb*^T Rb*)^{-1} T_a^T T_a) / min(ka, kb)``; ``Ra^T Rb* = RaR^T T_a``.
    The identity permutation gives the observed statistics."""
    c, n = P.shape
    kb = des["kb"]
    RbP = des["Rb_t"][P]                                            # (c, n, kb)
    T = (des["A_T"] @ RbP.permute(1, 0, 2).reshape(n, c * kb)).reshape(-1, c, kb).permute(1, 0, 2)
    q = des["q"]
    Hq, Ha = T[:, :q], T[:, q:]
    G = des["RbtRb"] - Hq.transpose(-1, -2) @ Hq                    # (c, kb, kb)
    M = Ha.transpose(-1, -2) @ Ha
    if kb == 1:
        pil = M[:, 0, 0] / G[:, 0, 0]
    else:
        pil = torch.linalg.solve(G, M).diagonal(dim1=-2, dim2=-1).sum(-1) / min(des["ka"], kb)
    out = {"pillai": pil}
    if des["comps"]:
        C = des["RaR_T"] @ Ha                                       # (c, ka, kb) = Ra^T Rb*
        ia = des["ia_t"]
        ib = des["ib_t"]
        num = C[:, ia, ib]
        gd = G.diagonal(dim1=-2, dim2=-1)[:, ib]
        out["r"] = num / (des["ra_norm_ia"] * torch.sqrt(gd))
    if des["mod"] is not None:
        md = des["mod"]
        E0P = md["E0"][P]                                           # (c, n, kc)
        Tm = torch.bmm(E0P.permute(2, 0, 1), md["B"]).permute(1, 0, 2)  # (c, kc, d)
        x0 = md["x0"]
        Tm_sq = Tm * Tm
        ss0 = md["e0sq"] - Tm_sq[:, :, :x0].sum(-1)
        diff = Tm_sq[:, :, x0:].sum(-1)
        F = torch.where(md["df1_pos"], (diff / md["df1_clamp"]) / ((ss0 - diff) / md["df2_t"]), torch.zeros_like(diff))
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


def _boot_edge_from_residuals(
    des: Dict[str, Any],
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    Saa: torch.Tensor,
    Sbb: torch.Tensor,
    Sab: Optional[torch.Tensor] = None,
    pil: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if Sab is None:
        Sab = Ra.transpose(-1, -2) @ Rb
    if pil is None:
        ka, kb = des["ka"], des["kb"]
        if ka == 1 and kb == 1:
            pil = (Sab[:, 0, 0] ** 2) / (Saa[:, 0, 0] * Sbb[:, 0, 0])
        elif kb == 1:
            pil = (Sab[:, :, 0] * torch.linalg.solve(Saa, Sab)[:, :, 0]).sum(-1) / Sbb[:, 0, 0]
        elif ka == 1:
            pil = (Sab[:, 0, :] * torch.linalg.solve(Sbb, Sab.transpose(-1, -2))[:, :, 0]).sum(-1) / Saa[:, 0, 0]
        else:
            pil = (torch.linalg.solve(Saa, Sab) @ torch.linalg.solve(Sbb, Sab.transpose(-1, -2))).diagonal(
                dim1=-2, dim2=-1).sum(-1) / min(ka, kb)
    out = {"pillai": pil}
    if des["comps"]:
        ia = des["ia_t"] if des.get("ia_t") is not None else torch.as_tensor([c[0] for c in des["comps"]])
        ib = des["ib_t"] if des.get("ib_t") is not None else torch.as_tensor([c[1] for c in des["comps"]])
        num = Sab[:, ia, ib]
        da = Saa.diagonal(dim1=-2, dim2=-1)[:, ia]
        db = Sbb.diagonal(dim1=-2, dim2=-1)[:, ib]
        out["r"] = num / torch.sqrt(da * db)
        out["beta"] = num / da
    return out, Sab, pil


def _ridge_residualize_batch(
    S: torch.Tensor, X: torch.Tensor, eye_s: Optional[torch.Tensor] = None
) -> torch.Tensor:
    St = S.transpose(-1, -2)
    StS = St @ S
    lam = _RIDGE * StS.diagonal(dim1=-2, dim2=-1).sum(-1) / S.shape[-1]
    if eye_s is None:
        eye_s = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    StS = StS + lam[:, None, None] * eye_s
    StX = St @ X
    L, info = torch.linalg.cholesky_ex(StS)
    if not bool(info.any()):
        coef = torch.cholesky_solve(StX, L)
    else:
        coef = torch.linalg.solve(StS, StX)
    return X - S @ coef


def _boot_edge(des: Dict[str, Any], idx: torch.Tensor, Za: torch.Tensor, Zb: torch.Tensor) -> Dict[str, torch.Tensor]:
    """pillai, r and beta on bootstrap rows ``idx`` (c, n): S, Z_a, Z_b resampled, residualised
    by ridge least squares (lambda = 1e-8 trace / cols) in each replicate."""
    ka = Za.shape[-1]
    S_t = des["S_t"] if "S_t" in des else torch.as_tensor(des["S"])
    R = _ridge_residualize_batch(S_t[idx], torch.cat([Za[idx], Zb[idx]], dim=-1), eye_s=des.get("eye_s"))
    Ra, Rb = R[..., :ka], R[..., ka:]
    Saa = Ra.transpose(-1, -2) @ Ra
    Sbb = Rb.transpose(-1, -2) @ Rb
    return _boot_edge_from_residuals(des, Ra, Rb, Saa, Sbb)[0]


def _adjusted_rv(Ra: np.ndarray, Rb: np.ndarray) -> float:
    from .utils import adjusted_rvcoef
    return float(adjusted_rvcoef(torch.as_tensor(Ra).float(), torch.as_tensor(Rb).float()))


def _pct(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(x, [2.5, 97.5], axis=axis)
    return lo, hi


# ----------------------------------------------------------------------------- path_model
def _wide_views(result: Dict[str, Any], nodes: List[str],
                orig_view_indices: Optional[Sequence[int]] = None) -> List[str]:
    n_train = int(_u_train(result).shape[0])
    idxs = list(range(len(nodes))) if orig_view_indices is None else list(orig_view_indices)
    return [nodes[pos] for pos, orig in enumerate(idxs) if int(result["v"][orig].shape[0]) >= n_train]


def _remap_graph_for_available_views(graph, all_nodes: List[str], absent_names: Sequence[str]):
    """Validate ``graph`` against ``all_nodes``, raise if any node in ``absent_names`` is
    referenced, and remap integer node indices to node names."""
    if graph is None:
        return None
    idx = {i: nm for i, nm in enumerate(all_nodes)}
    absent_set = set(absent_names)

    def _nm(x):
        if isinstance(x, (int, np.integer)):
            if int(x) not in idx:
                raise ValueError(f"node index {x} out of range for nodes {all_nodes}.")
            return idx[int(x)]
        return str(x)

    if isinstance(graph, Mapping):
        remapped_dict: Dict[str, List[str]] = {}
        referenced = set()
        for a, bs in graph.items():
            na = _nm(a)
            nbs = [_nm(b) for b in bs]
            referenced.add(na)
            referenced.update(nbs)
            remapped_dict[na] = nbs
        bad = sorted(referenced & absent_set)
        if bad:
            raise ValueError(f"graph references view(s) absent on evaluation rows: {bad}.")
        return remapped_dict

    remapped_edges = [(_nm(a), _nm(b)) for a, b in graph]
    referenced = {u for e in remapped_edges for u in e}
    bad = sorted(referenced & absent_set)
    if bad:
        raise ValueError(f"graph references view(s) absent on evaluation rows: {bad}.")
    return remapped_edges


def _parse_allow_train_fixed(allow_train_fixed: Union[bool, str]) -> Tuple[bool, Optional[str]]:
    if isinstance(allow_train_fixed, bool):
        return allow_train_fixed, None
    if isinstance(allow_train_fixed, str):
        if not allow_train_fixed.strip():
            raise ValueError("allow_train_fixed reason string must not be empty or whitespace.")
        return True, allow_train_fixed
    raise ValueError(f"allow_train_fixed must be a bool or non-empty reason string; got {allow_train_fixed!r}.")


def _check_path_params(rows, null, conditioning, alpha, n_permutations, n_refits, n_bootstrap,
                       allow_train_fixed: Union[bool, str] = False):
    if conditioning not in ("pairwise", "parents"):
        raise ValueError("conditioning must be 'pairwise' or 'parents'.")
    if rows not in ("heldout", "train"):
        raise ValueError("rows must be 'heldout' or 'train'.")
    null = ("refit" if rows == "train" else "fixed") if null is None else null
    if null not in ("fixed", "refit", "both"):
        raise ValueError("null must be 'fixed', 'refit' or 'both'.")
    if null in ("refit", "both") and rows != "train":
        raise ValueError("the refit null permutes and refits the training rows; it needs rows='train' and the "
                         "training views. Held-out rows use the fixed-model null.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if int(n_permutations) < 0 or int(n_bootstrap) < 0 or int(n_refits) < 0:
        raise ValueError("n_permutations, n_refits and n_bootstrap must be non-negative.")
    _parse_allow_train_fixed(allow_train_fixed)
    return null


def path_model(result: Dict[str, Any], views: Any, *, rows: str = "heldout", graph=None,
               directed: bool = False, view_names: Optional[Sequence[str]] = None,
               covariates=None, groups=None, outcome=None, conditioning: str = "pairwise",
               include_consensus: bool = False, null: Optional[str] = None,
               n_permutations: int = 10_000, n_refits: int = 200, n_bootstrap: int = 2000,
               train_views: Any = None, train_covariates=None, train_groups=None, train_outcome=None,
               inference: Optional[Dict[str, Any]] = None, alpha: float = 0.05, random_state: int = 0,
               n_jobs: int = 1, allow_train_fixed: Union[bool, str] = False,
               refit_scope: Tuple[str, ...] = ("edges", "omnibus"),
               _emit_drop_warning: bool = True,
               _precomputed_sc: Optional[Dict[str, Any]] = None,
               _edge_cache: Optional[Dict[Any, Any]] = None) -> Dict[str, Any]:
    """Edge-wise, omnibus and graph-fit permutation inference on the scores of a fitted model.

    See the module docstring for the statistics, nulls and what the test cannot say. ``views``
    are the rows to test: ``rows="heldout"`` for subjects not used in fitting (default null
    ``"fixed"``; missing views may be ``None``, dropping absent views from the graph with a
    `RuntimeWarning`), ``"train"`` for the training rows (default null ``"refit"``, every view
    required).
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
    in the parent, so the result does not depend on ``n_jobs``. Worker processes are spawned, so
    ``n_jobs > 1`` must be called from a script under ``if __name__ == "__main__":`` (or from an
    importable module), not from the interactive prompt or a notebook cell, where the workers cannot
    import the caller and the pool breaks (``BrokenProcessPool``).

    Fixed-model p-values on ``rows="train"`` are reported only with ``allow_train_fixed=True``
    (the null ignores that the model was fitted to these rows); the refit null is the default
    there. `path_model_from_scores` runs the same inference on score blocks from any source.

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
    null = _check_path_params(rows, null, conditioning, alpha, n_permutations, n_refits, n_bootstrap,
                              allow_train_fixed=allow_train_fixed)
    vlist = _view_list(result, views)
    m_views = len(vlist)
    from .extension import view_names as _names
    all_names = _check_node_names(
        list(view_names) if view_names is not None else list(_names(result))[:m_views],
        m_views=m_views,
    )
    has_missing = any(v is None for v in vlist)
    dropped_views: Optional[List[str]] = None
    if has_missing:
        if rows == "train":
            raise ValueError("path_model with rows='train' needs every view.")
        avail = [m for m, v in enumerate(vlist) if v is not None]
        if len(avail) == 0 or (len(avail) < 2 and outcome is None and not include_consensus):
            raise ValueError("path_model needs at least two present views, or one view with an outcome.")
        absent_names = [all_names[m] for m in range(m_views) if m not in avail]
        all_nodes = (list(all_names)
                     + (["consensus"] if include_consensus else [])
                     + (["outcome"] if outcome is not None else []))
        graph = _remap_graph_for_available_views(graph, all_nodes, absent_names)
        sc_full = view_scores(result, vlist, train_views=train_views)
        sc = dict(sc_full)
        sc["scores"] = [sc_full["scores"][m] for m in avail]
        sc["signs"] = [sc_full["signs"][m] for m in avail]
        names = [all_names[m] for m in avail]
        orig_view_indices: Optional[List[int]] = avail
        dropped_views = absent_names
    else:
        sc = _precomputed_sc if _precomputed_sc is not None else view_scores(result, vlist, train_views=train_views)
        names = all_names
        orig_view_indices = list(range(m_views))
    return _path_model_core(result, vlist, sc, names, rows=rows, graph=graph, directed=directed,
                            covariates=covariates, groups=groups, outcome=outcome, conditioning=conditioning,
                            include_consensus=include_consensus, null=null, n_permutations=n_permutations,
                            n_refits=n_refits, n_bootstrap=n_bootstrap, train_covariates=train_covariates,
                            train_groups=train_groups, train_outcome=train_outcome, inference=inference,
                            alpha=alpha, random_state=random_state, n_jobs=n_jobs,
                            allow_train_fixed=allow_train_fixed, refit_scope=refit_scope, standardize=True,
                            orig_view_indices=orig_view_indices, dropped_views=dropped_views,
                            _emit_drop_warning=_emit_drop_warning, _edge_cache=_edge_cache)


def _given_scores(scores: Any, standardize: bool, allow_missing: bool = False):
    """Validate a mapping node name -> ``(n, k)`` block; returns names and float64 arrays."""
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("scores must be a non-empty mapping of node name -> (n, k) array.")
    names = _check_node_names(list(scores), label="score")
    arrays: List[Optional[np.ndarray]] = []
    for nm in scores:
        z = scores[nm]
        if z is None and allow_missing:
            arrays.append(None)
            continue
        z = np.asarray(torch.as_tensor(z).detach().cpu().double()) if isinstance(z, torch.Tensor) \
            else np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[0] < 3 or z.shape[1] < 1:
            raise ValueError(f"scores[{nm!r}] must be a 2-d (n, k) array with n >= 3; got shape {z.shape}.")
        if not np.isfinite(z).all():
            raise ValueError(f"scores[{nm!r}] contains non-finite values.")
        arrays.append(_zscore(z) if standardize else z.copy())
    rows = {z.shape[0] for z in arrays if z is not None}
    if len(rows) > 1:
        raise ValueError(f"every score block needs the same number of rows; got {sorted(rows)}.")
    return names, arrays


def path_model_from_scores(scores: Mapping[str, Any], *, rows: str = "heldout", graph=None,
                           directed: bool = False, covariates=None, groups=None, outcome=None,
                           conditioning: str = "pairwise", null: str = "fixed", n_permutations: int = 10_000,
                           n_bootstrap: int = 2000, alpha: float = 0.05, random_state: int = 0,
                           allow_train_fixed: Union[bool, str] = False, standardize: bool = True,
                           _edge_cache: Optional[Dict[Any, Any]] = None) -> Dict[str, Any]:
    """`path_model` on score blocks from any source: embeddings, factor scores, another package's
    latents. ``scores`` maps node name -> ``(n, k_m)`` array (every block the same rows; blocks of
    different width are tested through ``pillai`` only). Each block is standardised column-wise
    (``standardize=True``, as `view_scores` does) and then enters the identical inference
    (`_path_model_core`): the fixed-model null, max-T / Holm / BH families, omnibus, graph fit,
    bootstrap intervals and node regressions. On `pysimlr` scores
    (``dict(zip(names, view_scores(result, views)["scores"]))``) the result equals `path_model`'s.

    There is no measurement model, so there is no refit null (``null`` must be ``"fixed"``), no
    ``inference=`` intervals, no consensus node and no ``feature_contributions`` (empty). ``rows``
    is a declaration: ``"train"`` means the scores were fitted on these rows, and fixed-model
    p-values are then reported only with ``allow_train_fixed=True`` (or a non-empty reason string).
    `predict_path` on the result takes a mapping of the same names (``None`` for a missing block).
    The result carries ``result=None`` and ``scores_source="given"``.
    """
    if null != "fixed":
        raise ValueError("path_model_from_scores has no measurement model to refit; null must be 'fixed' "
                         "(use path_model with a pysimlr fit for the refit null).")
    null = _check_path_params(rows, null, conditioning, alpha, n_permutations, 0, n_bootstrap,
                              allow_train_fixed=allow_train_fixed)
    names, arrays = _given_scores(scores, standardize)
    if len(names) < 2 and outcome is None:
        raise ValueError("path_model_from_scores needs at least two score blocks, or one block and an outcome.")
    sc = {"scores": arrays, "consensus": None, "signs": None, "source": "given", "sign_rows": "given",
          "available": list(range(len(arrays))), "n": int(arrays[0].shape[0]),
          "train_scores": None, "train_consensus": None}
    return _path_model_core(None, None, sc, names, rows=rows, graph=graph, directed=directed,
                            covariates=covariates, groups=groups, outcome=outcome, conditioning=conditioning,
                            include_consensus=False, null=null, n_permutations=n_permutations, n_refits=0,
                            n_bootstrap=n_bootstrap, train_covariates=None, train_groups=None,
                            train_outcome=None, inference=None, alpha=alpha, random_state=random_state,
                            n_jobs=1, allow_train_fixed=allow_train_fixed, refit_scope=("edges", "omnibus"),
                            standardize=standardize, _edge_cache=_edge_cache)


def _update_family_max(
    mx: Dict[str, np.ndarray],
    mean_null: np.ndarray,
    sl: slice,
    e: Tuple[str, str],
    pil: np.ndarray,
    ar_max: Optional[np.ndarray],
    f_sum: Optional[np.ndarray],
    n_tested: int,
    absent_set: set,
    out_set: set,
) -> None:
    mx["edges"][sl] = np.maximum(mx["edges"][sl], pil)
    mean_null[sl] += pil / n_tested
    if e in absent_set:
        mx["absent"][sl] = np.maximum(mx["absent"][sl], pil)
    if e in out_set:
        mx["outcome"][sl] = np.maximum(mx["outcome"][sl], pil)
    if ar_max is not None:
        mx["components"][sl] = np.maximum(mx["components"][sl], ar_max)
    if f_sum is not None:
        mx["moderation"][sl] = np.maximum(mx["moderation"][sl], f_sum)


def _assign_holm_bh(df: pd.DataFrame, mask: Optional[np.ndarray] = None) -> None:
    if mask is None:
        p = df["p_perm"].to_numpy(float)
        df["p_holm"] = holm(p)
        df["q_bh"] = bh_fdr(p)
    elif mask.any():
        p = df.loc[mask, "p_perm"].to_numpy(float)
        df.loc[mask, "p_holm"] = holm(p)
        df.loc[mask, "q_bh"] = bh_fdr(p)


def _path_model_core(result, vlist, sc, names, *, rows, graph, directed, covariates, groups, outcome,
                     conditioning, include_consensus, null, n_permutations, n_refits, n_bootstrap,
                     train_covariates, train_groups, train_outcome, inference, alpha, random_state, n_jobs,
                     allow_train_fixed, refit_scope, standardize,
                     orig_view_indices: Optional[Sequence[int]] = None,
                     dropped_views: Optional[Sequence[str]] = None,
                     _emit_drop_warning: bool = True,
                     _edge_cache: Optional[Dict[Any, Any]] = None) -> Dict[str, Any]:
    """The inference shared by `path_model` (``result``, ``vlist`` from a fit) and
    `path_model_from_scores` (``result is None``); ``sc`` has `view_scores`' keys."""
    use_fixed, use_refit = null in ("fixed", "both"), null in ("refit", "both")
    B, n_boot, n_ref = int(n_permutations), int(n_bootstrap), int(n_refits)
    if not use_fixed:
        B = 0
    if (use_refit or inference is not None) and result is None:
        raise ValueError("the refit null and inference= need a measurement model (path_model).")
    m_views = len(sc["scores"])
    n = sc["n"]
    nodes = _check_node_names(names, m_views=m_views)
    view_nodes = list(nodes)
    Z: Dict[str, np.ndarray] = {nm: z for nm, z in zip(nodes, sc["scores"])}
    if include_consensus:
        if sc["consensus"] is None:
            raise ValueError("include_consensus needs a measurement model (path_model).")
        nodes.append("consensus")
        Z["consensus"] = sc["consensus"]
    if outcome is not None:
        y = _check_outcome(outcome, n)
        nodes.append("outcome")
        Z["outcome"] = _zscore(y)
    k = int(Z[nodes[0]].shape[1])

    warn: List[str] = []

    def _warn(msg: str) -> None:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        warn.append(msg)

    if dropped_views:
        drop_msg = f"views absent on evaluation rows and dropped from the graph: {list(dropped_views)}"
        if _emit_drop_warning:
            _warn(drop_msg)
        else:
            warn.append(drop_msg)

    allow_tf_bool, allow_tf_reason = _parse_allow_train_fixed(allow_train_fixed)
    wide = _wide_views(result, view_nodes, orig_view_indices=orig_view_indices) if result is not None else []
    report_p = use_fixed
    if rows == "train" and use_fixed:
        why = (f"wide views {wide} couple to anything on their training rows, and " if wide else "")
        if not allow_tf_bool:
            report_p = False
            _warn(f"rows='train' with the fixed-model null: {why}the null holds the fitted model fixed and "
                  "ignores that it was fitted to these rows; fixed-model p-values are not reported (use "
                  "held-out rows, the refit null, or allow_train_fixed=True).")
        elif allow_tf_reason is not None:
            _warn(f"rows='train' with allow_train_fixed={allow_tf_reason!r}: {why}the fixed-model null holds "
                  "the fitted model fixed and ignores that it was fitted to these rows; held-out rows or the "
                  "refit null are the evidence.")
        else:
            _warn(f"rows='train' with allow_train_fixed=True: {why}the fixed-model null holds the fitted "
                  "model fixed and ignores that it was fitted to these rows; held-out rows or the refit "
                  "null are the evidence.")
    if B == 0:
        report_p = False

    if rows == "heldout":
        evidence_label = "heldout-fixed"
    elif null == "refit":
        evidence_label = "train-refit"
    elif null == "both":
        evidence_label = "train-both"
    elif allow_tf_bool:
        evidence_label = "train-fixed"
    else:
        evidence_label = "train-fixed-unreported"

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
    absent_set, out_set = set(absent), set(out_edges)

    cond_by_edge = {e: tuple(_conditioning_nodes(adj, e[0], e[1], directed, conditioning)) for e in pairs}
    designs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    obs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in pairs:
        ckey = (e[0], e[1], cond_by_edge[e])
        if _edge_cache is not None and ckey in _edge_cache:
            designs[e] = _edge_cache[ckey]["des"]
            obs[e] = _edge_cache[ckey]["obs"]
        else:
            designs[e] = _edge_design(Z, N, e[0], e[1], list(cond_by_edge[e]), dummies)
            obs[e] = _observed(designs[e])

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

    def _eval_chunk(e: Tuple[str, str], perm_b: torch.Tensor):
        st = _stats_from_perm(designs[e], perm_b)
        pil = st["pillai"].numpy()
        dp = int((pil >= obs[e]["pillai"]).sum())
        dr = ar_max = df = f_sum = None
        dfs = 0
        if "r" in st:
            ar = np.abs(st["r"].numpy())
            dr = (ar >= np.abs(obs[e]["r"])[None, :]).sum(0)
            ar_max = ar.max(1)
        if e in ge_f:
            F = st["F"].numpy()
            df = (F >= obs[e]["F"][None, :]).sum(0)
            f_sum = F.sum(1)
            dfs = int((f_sum >= obs[e]["F"].sum()).sum())
        return pil, dp, dr, ar_max, df, f_sum, dfs

    if _edge_cache is not None and B > 0:
        if "_perms" not in _edge_cache:
            base = np.tile(np.arange(n), (min(_CHUNK, max(B, 1)), 1))
            _edge_cache["_perms"] = [
                (s0, min(_CHUNK, B - s0),
                 {nm: torch.as_tensor(rng_perm.permuted(base[:min(_CHUNK, B - s0)], axis=1)) for nm in nodes})
                for s0 in range(0, B, _CHUNK)
            ]
        perms_chunks = _edge_cache["_perms"]
        for e in pairs:
            ckey = (e[0], e[1], cond_by_edge[e])
            if ckey not in _edge_cache:
                pil_all = np.empty(B, dtype=np.float64)
                has_r = len(designs[e]["comps"]) > 0
                ar_max_all = np.empty(B, dtype=np.float64) if has_r else None
                f_sum_all = np.empty(B, dtype=np.float64) if e in ge_f else None
                ge_p_e, ge_fs_e = 0, 0
                ge_r_e = np.zeros(len(designs[e]["comps"]))
                ge_f_e = np.zeros(len(designs[e]["comps"])) if e in ge_f else None
                stored_e: List[float] = []
                for s0, c, perms in perms_chunks:
                    sl = slice(s0, s0 + c)
                    pil, dp, dr, ar_max, df, f_sum, dfs = _eval_chunk(e, perms[e[1]])
                    pil_all[sl] = pil
                    ge_p_e += dp
                    if len(stored_e) < _NULL_STORE:
                        stored_e.extend(pil[:_NULL_STORE - len(stored_e)].tolist())
                    if dr is not None:
                        ge_r_e += dr
                        if ar_max_all is not None and ar_max is not None:
                            ar_max_all[sl] = ar_max
                    if df is not None and ge_f_e is not None and f_sum_all is not None and f_sum is not None:
                        ge_f_e += df
                        ge_fs_e += dfs
                        f_sum_all[sl] = f_sum
                _edge_cache[ckey] = {
                    "des": designs[e], "obs": obs[e],
                    "ge_p": ge_p_e, "ge_r": ge_r_e, "ge_f": ge_f_e, "ge_fs": ge_fs_e,
                    "stored": stored_e, "pil_all": pil_all,
                    "ar_max_all": ar_max_all, "f_sum_all": f_sum_all,
                }
            ce = _edge_cache[ckey]
            ge_p[e] = ce["ge_p"]
            ge_r[e] = ce["ge_r"].copy()
            stored[e] = list(ce["stored"])
            if e in ge_f and ce["ge_f"] is not None:
                ge_f[e] = ce["ge_f"].copy()
                ge_fs[e] = ce["ge_fs"]
            if not diagnostic[e]:
                _update_family_max(
                    mx, mean_null, slice(None), e, ce["pil_all"],
                    ce["ar_max_all"], ce["f_sum_all"] if e in ge_f else None,
                    len(tested), absent_set, out_set,
                )
    else:
        base = np.tile(np.arange(n), (min(_CHUNK, max(B, 1)), 1))
        for s0 in range(0, B, _CHUNK):
            c = min(_CHUNK, B - s0)
            perms = {nm: torch.as_tensor(rng_perm.permuted(base[:c], axis=1)) for nm in nodes}
            sl = slice(s0, s0 + c)
            for e in pairs:
                pil, dp, dr, ar_max, df, f_sum, dfs = _eval_chunk(e, perms[e[1]])
                ge_p[e] += dp
                if len(stored[e]) < _NULL_STORE:
                    stored[e].extend(pil[:_NULL_STORE - len(stored[e])].tolist())
                if dr is not None:
                    ge_r[e] += dr
                if df is not None:
                    ge_f[e] += df
                    ge_fs[e] += dfs
                if not diagnostic[e]:
                    _update_family_max(
                        mx, mean_null, sl, e, pil, ar_max, f_sum,
                        len(tested), absent_set, out_set,
                    )

    def p_of(ge):
        return float(_addone(ge, B)) if report_p else None

    def p_max(family: str, value: float):
        return float(_addone((mx[family] >= value).sum(), B)) if report_p else None

    # ---- bootstrap intervals (measurement fixed, subjects resampled)
    boot: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {e: {"pillai": [], "r": [], "beta": []} for e in pairs}
    if n_boot > 0:
        Zt = {nm: torch.as_tensor(z) for nm, z in Z.items()}
        cond_groups: Dict[Tuple[str, ...], List[Tuple[str, str]]] = {}
        for e in pairs:
            cond_groups.setdefault(cond_by_edge[e], []).append(e)
        cond_prep = {}
        for c_tuple, grp_edges in cond_groups.items():
            u_nodes = list(dict.fromkeys([u for e in grp_edges for u in e]))
            u_widths = [int(Zt[u].shape[1]) for u in u_nodes]
            u_offs = np.cumsum([0] + u_widths)
            u_slices = {u: slice(int(u_offs[i]), int(u_offs[i + 1])) for i, u in enumerate(u_nodes)}
            S_t = torch.as_tensor(designs[grp_edges[0]]["S"])
            X_cat = torch.cat([Zt[u] for u in u_nodes], dim=1)
            eye_s = torch.eye(S_t.shape[-1], dtype=S_t.dtype)
            cond_prep[c_tuple] = (grp_edges, u_nodes, u_slices, S_t, X_cat, eye_s)
        for s0 in range(0, n_boot, _CHUNK):
            c = min(_CHUNK, n_boot - s0)
            idx = torch.as_tensor(rng_boot.integers(0, n, (c, n)))
            for c_tuple, (grp_edges, u_nodes, u_slices, S_t, X_cat, eye_s) in cond_prep.items():
                R_all = _ridge_residualize_batch(S_t[idx], X_cat[idx], eye_s)
                R_map: Dict[str, torch.Tensor] = {}
                Suu_map: Dict[str, torch.Tensor] = {}
                for u in u_nodes:
                    ru = R_all[:, :, u_slices[u]]
                    R_map[u] = ru
                    Suu_map[u] = ru.transpose(-1, -2) @ ru
                pair_done: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
                for e in grp_edges:
                    a, b = e
                    rev = (b, a)
                    if rev in pair_done:
                        sab_rev, pil_rev = pair_done[rev]
                        st, _, _ = _boot_edge_from_residuals(
                            designs[e], R_map[a], R_map[b], Suu_map[a], Suu_map[b],
                            Sab=sab_rev.transpose(-1, -2), pil=pil_rev,
                        )
                    else:
                        st, sab_e, pil_e = _boot_edge_from_residuals(
                            designs[e], R_map[a], R_map[b], Suu_map[a], Suu_map[b]
                        )
                        pair_done[e] = (sab_e, pil_e)
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
                                 train_covariates, train_groups, train_outcome,
                                 orig_view_indices=orig_view_indices) if inference is not None else None

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
               "ci_refit_low": None, "ci_refit_high": None, "null_rows": evidence_label}
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
        _assign_holm_bh(edges, tmask)
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
                   "beta_ci_low": None, "beta_ci_high": None, "null_rows": evidence_label}
            if bt is not None:
                lo, hi = _pct(bt["r"][:, j])
                row["ci_low"], row["ci_high"] = float(lo), float(hi)
                lo, hi = _pct(bt["beta"][:, j])
                row["beta_ci_low"], row["beta_ci_high"] = float(lo), float(hi)
            comp_rows.append(row)
    edge_components = pd.DataFrame(comp_rows, columns=["a", "b", "component", "r", "beta", "diagnostic", "p_perm",
                                                       "p_maxt", "p_holm", "q_bh", "ci_low", "ci_high",
                                                       "beta_ci_low", "beta_ci_high", "null_rows"])
    if report_p and len(edge_components):
        _assign_holm_bh(edge_components, ~edge_components["diagnostic"].to_numpy(bool))

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
        if report_p:
            _assign_holm_bh(moderation, (moderation["component"] == "sum").to_numpy())

    outcome_edges = None
    if out_edges:
        orows = [{"a": e[0], "b": e[1], "pillai": obs[e]["pillai"], "p_perm": p_of(ge_p[e]),
                  "p_maxt": p_max("outcome", obs[e]["pillai"]), "p_holm": None, "q_bh": None,
                  "null_rows": evidence_label} for e in out_edges]
        outcome_edges = pd.DataFrame(orows)
        if report_p:
            _assign_holm_bh(outcome_edges)

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
                     refit is not None and n_ref > 0 and "edges" in refit_scope, alpha, p_max,
                     evidence=evidence_label)

    families = {"edges": {"n_tests": len(tested), "statistic": "pillai"},
                "edge_components": {"n_tests": int((~edge_components["diagnostic"].to_numpy(bool)).sum())
                                    if len(edge_components) else 0, "statistic": "|r|"},
                "moderation": {"n_tests": len(mods), "statistic": "sum over components of F"},
                "outcome_edges": {"n_tests": len(out_edges), "statistic": "pillai"}}

    edge_dofs = _result_node_dofs(result, view_nodes, n, orig_view_indices) if (result is not None and rows == "train") else None
    edge_models = {} if _edge_cache is not None else _fit_node_models(Z, nodes, adj, N, directed, node_dofs=edge_dofs)
    feature_contributions = (
        _feature_contributions(result, view_nodes, pairs, present, diagnostic, designs, obs,
                               orig_view_indices=orig_view_indices)
        if (result is not None and _edge_cache is None)
        else pd.DataFrame(columns=["edge", "a", "b", "component", "feature", "contribution"])
    )
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
            "orig_view_indices": (list(orig_view_indices) if orig_view_indices is not None else None),
            "refit_null": None if refit is None else {"per_edge": {e: v.tolist() for e, v in refit["null"].items()},
                                                      "outcome_edges": refit["outcome_note"]},
            "nuisance_names": nuis_names, "null": stored, "scores_source": sc["source"],
            "sign_rows": sc["sign_rows"], "signs": sc["signs"], "wide_views": wide, "p_reason": reason,
            "parameters": {"rows": rows, "null": null, "n_permutations": B, "n_refits": n_refits,
                           "refit_scope": tuple(refit_scope) if use_refit else None,
                           "n_bootstrap": n_boot, "alpha": alpha, "random_state": random_state, "n_jobs": n_jobs,
                           "allow_train_fixed": allow_tf_bool,
                           "allow_train_fixed_reason": allow_tf_reason,
                           "conditioning": conditioning,
                           "include_consensus": include_consensus, "directed": directed,
                           "inference": inference is not None, "standardize": bool(standardize),
                           "scores_given": result is None,
                           "n_refit_pillai": None if refit_pillai is None else len(next(iter(refit_pillai.values()))),
                           "residualization": f"rank-revealing QR (tolerance {_RANK_TOL}); bootstrap: ridge "
                                              f"lambda = {_RIDGE} * trace / cols",
                           "chunk": _CHUNK},
            "warnings": warn}


def _graph_fit(edges, tested, absent, present, obs, conditioning, report_p, have_refit, alpha,
               p_max, evidence: Optional[str] = None) -> Dict[str, Any]:
    """Plan 3.5 from the fixed-model null when it was reported, else from the refit null (no
    max-T: each refit null permutes one node, so ``max_absent_p`` is None)."""
    fit: Dict[str, Any] = {"conditioning": conditioning, "null": None, "evidence": evidence,
                           "present_pass": None, "absent_pass": None,
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
                  has_outcome, train_covariates, train_groups, train_outcome,
                  orig_view_indices: Optional[Sequence[int]] = None) -> Dict[Tuple[str, str], List[float]]:
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
    idxs = list(range(len(view_nodes))) if orig_view_indices is None else list(orig_view_indices)
    out: Dict[Tuple[str, str], List[float]] = {e: [] for e in pairs if "consensus" not in e}
    for rec in recs:
        rows_ = np.asarray(rec["rows"])
        if max(idxs, default=-1) >= len(rec["scores"]) or (orig_view_indices is None and len(rec["scores"]) != len(view_nodes)):
            raise ValueError(f"inference has {len(rec['scores'])} views; the model needs indices {idxs}.")
        Zr = {nm: np.asarray(rec["scores"][orig], float) for nm, orig in zip(view_nodes, idxs)}
        if has_outcome:
            Zr["outcome"] = _zscore(ty[rows_])
        Nr, _, _ = _nuisance(len(rows_), None if tc is None else tc[rows_], None if tg is None else tg[rows_])
        for e in out:
            des = _edge_design(Zr, Nr, e[0], e[1], _conditioning_nodes(adj, e[0], e[1], directed, conditioning))
            out[e].append(_observed(des)["pillai"])
    return out


# ----------------------------------------------------------------------------- feature read-out
def _feature_contributions(result, view_nodes, pairs, present, diagnostic, designs, obs,
                           orig_view_indices: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """structure_a[:, j] * beta_abj for every present edge whose node a is a view (plan 5)."""
    idxs = list(range(len(view_nodes))) if orig_view_indices is None else list(orig_view_indices)
    orig_of = {nm: orig for nm, orig in zip(view_nodes, idxs)}
    view_set = set(view_nodes)
    edge_cols, a_cols, b_cols, comp_cols, feat_cols, val_cols = [], [], [], [], [], []
    st_cache: Dict[Any, np.ndarray] = {}
    for e in pairs:
        a, b = e
        if not present[e] or diagnostic[e] or a not in view_set or not designs[e]["comps"]:
            continue
        st = st_cache.get(a)
        if st is None:
            st = result["structure"][orig_of[a]].detach().double().numpy()
            st_cache[a] = st
        p_feat = st.shape[0]
        comps_e = designs[e]["comps"]
        n_comp = len(comps_e)
        ia_idx = [ia for ia, _ in comps_e]
        betas = np.asarray(obs[e]["beta"], dtype=np.float64)
        contrib_mat = (st[:, ia_idx] * betas[None, :]).T.reshape(-1)
        total = n_comp * p_feat
        edge_cols.append(np.full(total, f"{a} - {b}", dtype=object))
        a_cols.append(np.full(total, a, dtype=object))
        b_cols.append(np.full(total, b, dtype=object))
        comp_cols.append(np.repeat(np.arange(n_comp, dtype=np.int64), p_feat))
        feat_cols.append(np.tile(np.arange(p_feat, dtype=np.int64), n_comp))
        val_cols.append(contrib_mat)
    if not val_cols:
        return pd.DataFrame(columns=["edge", "a", "b", "component", "feature", "contribution"])
    return pd.DataFrame({
        "edge": np.concatenate(edge_cols),
        "a": np.concatenate(a_cols),
        "b": np.concatenate(b_cols),
        "component": np.concatenate(comp_cols),
        "feature": np.concatenate(feat_cols),
        "contribution": np.concatenate(val_cols),
    })


# ----------------------------------------------------------------------------- node models
def _ridge_fit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    lam = _RIDGE * np.trace(X.T @ X) / max(1, X.shape[1])
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def _result_node_dofs(
    result: Optional[Dict[str, Any]],
    view_nodes: Sequence[str],
    n: int,
    orig_view_indices: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, float]]:
    """Per-view effective basis degrees of freedom (`consensus.view_dofs`) for nodes backed by ``result``."""
    if result is None or not result.get("v"):
        return None
    from .consensus import view_dofs
    dofs_all = view_dofs(result["v"], int(n))
    idxs = list(range(len(view_nodes))) if orig_view_indices is None else list(orig_view_indices)
    out: Dict[str, float] = {}
    for nm, orig_m in zip(view_nodes, idxs):
        if 0 <= int(orig_m) < len(dofs_all):
            out[str(nm)] = float(dofs_all[int(orig_m)])
    for nm, ext in (result.get("extensions") or {}).items():
        if nm in view_nodes and "v" in ext:
            out[str(nm)] = float(view_dofs([ext["v"]], int(n))[0])
    return out


def _coerce_node_dofs(
    dofs: Any,
    view_names_list: Sequence[str],
) -> Optional[Dict[str, float]]:
    if dofs is None:
        return None
    if isinstance(dofs, Mapping):
        allowed = set(view_names_list) | {"outcome"}
        extra = [str(k) for k in dofs if str(k) not in allowed]
        if extra:
            raise ValueError(f"unknown node name(s) in dofs {extra}; expected a subset of {list(view_names_list)}.")
        items = [(str(k), v, f"dofs[{k!r}]") for k, v in dofs.items()]
    else:
        seq = list(dofs)
        if len(seq) != len(view_names_list):
            raise ValueError(f"dofs has {len(seq)} entries; expected {len(view_names_list)} for {list(view_names_list)}.")
        items = [(str(nm), v, f"dofs for {nm!r}") for nm, v in zip(view_names_list, seq)]
    out: Dict[str, float] = {}
    for nm, v, label in items:
        if v is not None:
            val = float(v)
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"{label} must be a non-negative finite number; got {v!r}.")
            out[nm] = val
    return out


def _gcv_press_node_fit(
    N: np.ndarray,
    X_pa: Optional[np.ndarray],
    Y: np.ndarray,
    dof_eff: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit ``Y ~ N W_N + X_pa W_pa`` with GCV ridge (Golub, Heath & Wahba 1979, Technometrics 21:215-223),
    PRESS leave-one-out hat-matrix residuals ``r_i / (1 - h_ii)`` (Allen 1974, Technometrics 16:125-127),
    and measurement-basis degrees-of-freedom shrinkage ``gamma = (n - 1) / (n - 1 + dof_eff)``
    (Ezekiel 1930; Wherry 1931, Ann. Math. Stat. 2:440-457; matching `extension._affine_map`).
    """
    from .extension import _gcv_ridge_svd
    n = int(Y.shape[0])
    q_n = _orth(N)
    r_n = int(q_n.shape[1])
    y_perp = Y - q_n @ (q_n.T @ Y) if r_n else Y.copy()
    h_n = (q_n ** 2).sum(axis=1) if r_n else np.zeros(n, dtype=float)
    v0_loo = np.mean((y_perp / np.maximum(1.0 - h_n, 1e-3)[:, None]) ** 2, axis=0).clip(1e-12)
    if X_pa is None or X_pa.shape[1] == 0:
        return _ridge_fit(N, Y), v0_loo, 1.0

    x_perp = X_pa - q_n @ (q_n.T @ X_pa) if r_n else X_pa.copy()
    w_gcv, h_pa = _gcv_ridge_svd(x_perp, y_perp, n_eff=float(max(n - r_n, 1)))
    if not np.any(h_pa):
        w_pa = np.zeros((X_pa.shape[1], Y.shape[1]), dtype=float)
        return np.vstack([_ridge_fit(N, Y), w_pa]), v0_loo, 1.0

    res_gcv = y_perp - x_perp @ w_gcv
    denom_tot = np.maximum(1.0 - (h_n + h_pa), 1e-3)[:, None]
    v_loo = np.mean((res_gcv / denom_tot) ** 2, axis=0).clip(1e-12)

    n_m1 = float(max(n - 1, 1))
    gamma = n_m1 / (n_m1 + max(float(dof_eff), 0.0))
    w_pa = gamma * w_gcv
    v_cal = np.maximum(v_loo, (1.0 - gamma) * v0_loo + gamma * v_loo).clip(1e-12)
    w_n = _ridge_fit(N, Y - X_pa @ w_pa)
    return np.vstack([w_n, w_pa]), v_cal, float(gamma)


def _fit_node_models(
    Z, nodes, adj, N, directed,
    node_dofs: Optional[Mapping[str, float]] = None,
    node_model_cache: Optional[Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]]] = None,
):
    models = {}
    for b in nodes:
        pa = [a for a in _parents(adj, b, directed) if a != "consensus"]
        ckey = (b, tuple(pa))
        if node_model_cache is not None and ckey in node_model_cache:
            models[b] = node_model_cache[ckey]
            continue
        x_pa = np.column_stack([Z[a] for a in pa]) if pa else None
        dof_eff = (
            max([float(node_dofs.get(b, 0.0))] + [float(node_dofs.get(a, 0.0)) for a in pa])
            if (pa and node_dofs is not None) else 0.0
        )
        W, resid_var, gamma = _gcv_press_node_fit(N, x_pa, Z[b], dof_eff=dof_eff)
        m_b = {"parents": pa, "W": W, "resid_var": resid_var,
               "n_params": int(W.size), "n_nuisance": int(N.shape[1]),
               "parent_cols": {a: int(Z[a].shape[1]) for a in pa},
               "dof_eff": float(dof_eff), "gamma": float(gamma)}
        if node_model_cache is not None:
            node_model_cache[ckey] = m_b
        models[b] = m_b
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
def _graph_loglik(
    Ztr, Zte, Ntr, Nte, nodes, adj, directed,
    node_dofs: Optional[Mapping[str, float]] = None,
    node_model_cache: Optional[Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, float, int]:
    """Node regressions on parents fitted on the training scores: per-subject held-out Gaussian
    log-likelihood, the training log-likelihood and the parameter count (plan F4)."""
    models = _fit_node_models(
        Ztr, nodes, adj, Ntr, directed, node_dofs=node_dofs, node_model_cache=node_model_cache
    )
    ll_rows = np.zeros(next(iter(Zte.values())).shape[0])
    ll_tr = 0.0
    for b in nodes:
        mb = models[b]
        if "_ll_rows" not in mb:
            pa = mb["parents"]
            X = np.column_stack([Nte] + [Zte[a] for a in pa])
            r = Zte[b] - X @ mb["W"]
            v = mb["resid_var"]
            mb["_ll_rows"] = -0.5 * ((r ** 2) / v).sum(1) - 0.5 * np.log(2 * np.pi * v).sum()
            mb["_ll_tr"] = _node_loglik(mb, Ntr, [Ztr[a] for a in pa], Ztr[b])
        ll_rows += mb["_ll_rows"]
        ll_tr += mb["_ll_tr"]
    return ll_rows, ll_tr, _n_params(models, Ztr, adj, directed)


def _n_edges(adj, directed: bool) -> int:
    return int(sum(len(v) for v in adj.values()) // (1 if directed else 2))


def _compare_structural_core(
    all_names: List[str],
    arrays_tr: Sequence[np.ndarray],
    arrays_te: Sequence[Optional[np.ndarray]],
    avail: List[int],
    graphs: Mapping[str, Any],
    *,
    directed: bool,
    covariates_train,
    covariates_test,
    groups_train,
    groups_test,
    outcome_train,
    outcome_test,
    std_mode: str,
    node_dofs: Optional[Mapping[str, float]],
    n_bootstrap: int,
    n_permutations: int,
    alpha: float,
    random_state: int,
    scores_source: Optional[str] = None,
    standardize_param: Any = None,
) -> Dict[str, Any]:
    m_views = len(all_names)
    absent_names = [all_names[m] for m in range(m_views) if m not in avail]
    all_nodes = list(all_names) + (["outcome"] if outcome_train is not None else [])
    remapped_graphs = {
        name: _remap_graph_for_available_views(g, all_nodes, absent_names)
        for name, g in graphs.items()
    }
    warn_list: List[str] = []
    if absent_names:
        drop_msg = f"views absent on evaluation rows and dropped from the graph: {absent_names}"
        warnings.warn(drop_msg, RuntimeWarning, stacklevel=3)
        warn_list.append(drop_msg)

    rng = np.random.default_rng(random_state)
    n_tr = int(arrays_tr[0].shape[0])
    n_te = int(arrays_te[avail[0]].shape[0])
    nodes = [all_names[m] for m in avail]
    Ztr = {all_names[m]: arrays_tr[m] for m in avail}
    Zte = {all_names[m]: arrays_te[m] for m in avail}
    if outcome_train is not None:
        y_tr_raw = np.asarray(outcome_train, float)
        y_te_raw = np.asarray(outcome_test, float)
        if y_tr_raw.shape[0] != n_tr or y_te_raw.shape[0] != n_te:
            raise ValueError(
                f"outcome_train / outcome_test have ({y_tr_raw.shape[0]}, {y_te_raw.shape[0]}) rows; "
                f"expected ({n_tr}, {n_te})."
            )
        y_tr_arr = y_tr_raw.reshape(n_tr, -1)
        y_te_arr = y_te_raw.reshape(n_te, -1)
        if not (np.isfinite(y_tr_arr).all() and np.isfinite(y_te_arr).all()):
            raise ValueError("outcome_train / outcome_test contain non-finite values.")
        nodes = nodes + ["outcome"]
        Ztr["outcome"], Zte["outcome"] = _zscore_pair(y_tr_arr, y_te_arr, std_mode)
    Ntr, _, _ = _nuisance(n_tr, covariates_train, groups_train)
    Nte, _, _ = _nuisance(n_te, covariates_test, groups_test)
    boot_idx = [rng.integers(0, n_te, n_te) for _ in range(n_bootstrap)]
    te_present_std = {all_names[m]: arrays_te[m] for m in avail}
    edge_cache: Dict[Any, Any] = {}
    node_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
    rows, per_subject = [], {}
    for name, g_remapped in remapped_graphs.items():
        adj = _node_graph(g_remapped, nodes, directed)
        ll_rows, ll_tr, n_par = _graph_loglik(
            Ztr, Zte, Ntr, Nte, nodes, adj, directed, node_dofs=node_dofs, node_model_cache=node_cache
        )
        per_subject[name] = ll_rows
        pm = path_model_from_scores(
            te_present_std, rows="heldout", graph=g_remapped, directed=directed,
            covariates=covariates_test, groups=groups_test, outcome=outcome_test,
            conditioning="parents", n_permutations=n_permutations, n_bootstrap=0,
            alpha=alpha, random_state=random_state, standardize=False,
            _edge_cache=edge_cache,
        )
        rows.append({
            "graph": name, "n_edges": _n_edges(adj, directed),
            "heldout_loglik": float(ll_rows.sum()), "n_params": n_par,
            "bic": float(n_par * np.log(n_tr) - 2.0 * ll_tr),
            "present_pass": pm["fit"]["present_pass"], "absent_pass": pm["fit"]["absent_pass"],
            "max_absent_p": pm["fit"]["max_absent_p"], "edges": pm["edges"],
        })
    top = max(rows, key=lambda r: r["heldout_loglik"])["graph"]
    g_names = [r["graph"] for r in rows]
    wins = {nm: 0 for nm in g_names}
    deltas: Dict[str, np.ndarray] = {}
    if n_bootstrap:
        ll_mat = np.vstack([per_subject[nm] for nm in g_names])
        boot_idx_arr = np.asarray(boot_idx, dtype=np.intp)
        tot_mat = ll_mat[:, boot_idx_arr].sum(axis=2)
        win_counts = np.bincount(np.argmax(tot_mat, axis=0), minlength=len(g_names))
        for gi, nm in enumerate(g_names):
            wins[nm] = int(win_counts[gi])
        top_idx = g_names.index(top)
        delta_mat = tot_mat - tot_mat[top_idx : top_idx + 1, :]
        for gi, nm in enumerate(g_names):
            deltas[nm] = delta_mat[gi]
    top_ll = next(x["heldout_loglik"] for x in rows if x["graph"] == top)
    for r in rows:
        r["delta_vs_top"] = float(r["heldout_loglik"] - top_ll)
        if n_bootstrap:
            lo, hi = _pct(deltas[r["graph"]])
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
    params = {
        "n_bootstrap": n_bootstrap, "n_permutations": n_permutations,
        "random_state": random_state, "alpha": alpha,
    }
    out: Dict[str, Any] = {
        "comparison": rows, "best": best, "best_by_loglik": top, "best_rule": rule,
        "nodes": nodes, "mode": "structural",
    }
    if scores_source is not None:
        out["scores_source"] = scores_source
        params["standardize"] = standardize_param
    out["warnings"] = warn_list
    out["parameters"] = params
    return out


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
    if (outcome_train is None) != (outcome_test is None):
        raise ValueError("pass both outcome_train and outcome_test, or neither.")
    sc_tr = view_scores(result, views_train, train_views=views_train)
    sc_te = view_scores(result, views_test, train_views=views_train)
    from .extension import view_names as _names
    m_views = len(sc_tr["scores"])
    all_names = _check_node_names(
        list(view_names) if view_names is not None else list(_names(result))[:m_views],
        m_views=m_views,
    )
    avail = list(sc_te["available"])
    if len(avail) == 0 or (len(avail) < 2 and outcome_train is None):
        raise ValueError("compare_path_models needs at least two present test views, or one view with an outcome.")
    node_dofs = _result_node_dofs(result, [all_names[m] for m in avail], sc_tr["n"], avail)
    return _compare_structural_core(
        all_names, sc_tr["scores"], sc_te["scores"], avail, graphs,
        directed=directed, covariates_train=covariates_train, covariates_test=covariates_test,
        groups_train=groups_train, groups_test=groups_test, outcome_train=outcome_train,
        outcome_test=outcome_test, std_mode="per_split", node_dofs=node_dofs,
        n_bootstrap=n_bootstrap, n_permutations=n_permutations, alpha=alpha, random_state=random_state,
    )


def compare_path_models_from_scores(
    scores_train: Mapping[str, Any],
    scores_test: Mapping[str, Any],
    graphs: Mapping[str, Any],
    *,
    directed: bool = False,
    covariates_train=None,
    covariates_test=None,
    groups_train=None,
    groups_test=None,
    outcome_train=None,
    outcome_test=None,
    standardize: Union[bool, str] = True,
    dofs: Optional[Union[Mapping[str, float], Sequence[float]]] = None,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Compare candidate graphs over score blocks from any source (measurement-free structural mode).

    ``scores_train`` maps node name -> ``(n_train, k_m)`` array (all present). ``scores_test``
    maps the same node names -> ``(n_test, k_m)`` array or ``None`` (or omits a missing block,
    which is dropped with the same `RuntimeWarning` and graph validation as `compare_path_models`).
    Node regressions are fitted on ``scores_train`` with GCV ridge (Golub, Heath & Wahba 1979) and
    PRESS leave-one-out residual variance (Allen 1974), calibrated by ``dofs`` when given
    (`consensus.view_dofs`; Ezekiel 1930 / Wherry 1931), and evaluated by Gaussian log-likelihood on
    ``scores_test`` (`_graph_loglik`), paired with `path_model_from_scores` graph-fit checks and
    the paired-bootstrap F1 parsimony rule.

    ``standardize`` controls block scaling: ``True`` (or ``"per_split"``) z-scores ``scores_train``
    on training rows and ``scores_test`` on test rows (matching `compare_path_models` / `view_scores`);
    ``"train"`` standardizes both ``scores_train`` and ``scores_test`` (and ``outcome_test``) with
    the training-row mean and SD; ``False`` uses the given arrays without rescaling.
    """
    if not graphs:
        raise ValueError("graphs is required: a non-empty {name: graph}.")
    if (outcome_train is None) != (outcome_test is None):
        raise ValueError("pass both outcome_train and outcome_test, or neither.")
    if isinstance(standardize, bool):
        std_mode = "per_split" if standardize else "none"
    elif standardize in ("per_split", "train", "none"):
        std_mode = standardize
    else:
        raise ValueError(f"standardize must be True, False, 'per_split' or 'train'; got {standardize!r}.")
    names_tr, raw_tr = _given_scores(scores_train, False, allow_missing=False)
    if not isinstance(scores_test, Mapping):
        raise ValueError("scores_test must be a mapping of node name -> (n_test, k_m) array or None.")
    extra = [str(k) for k in scores_test if str(k) not in names_tr]
    if extra:
        raise ValueError(f"unknown score name(s) {extra}; scores_train has {names_tr}.")
    full_te = {nm: scores_test.get(nm) for nm in names_tr}
    _, raw_te = _given_scores(full_te, False, allow_missing=True)
    pairs_std = [_zscore_pair(a_tr, a_te, std_mode) for a_tr, a_te in zip(raw_tr, raw_te)]
    arrays_tr = [p[0] for p in pairs_std]
    arrays_te = [p[1] for p in pairs_std]
    avail = [m for m, z in enumerate(arrays_te) if z is not None]
    if len(avail) == 0 or (len(avail) < 2 and outcome_train is None):
        raise ValueError(
            "compare_path_models_from_scores needs at least two present test score blocks, "
            "or one block with an outcome."
        )
    for m in avail:
        if arrays_te[m].shape[1] != arrays_tr[m].shape[1]:
            raise ValueError(
                f"scores_test[{names_tr[m]!r}] has {arrays_te[m].shape[1]} columns; "
                f"scores_train[{names_tr[m]!r}] has {arrays_tr[m].shape[1]}."
            )
    node_dofs = _coerce_node_dofs(dofs, names_tr)
    return _compare_structural_core(
        names_tr, arrays_tr, arrays_te, avail, graphs,
        directed=directed, covariates_train=covariates_train, covariates_test=covariates_test,
        groups_train=groups_train, groups_test=groups_test, outcome_train=outcome_train,
        outcome_test=outcome_test, std_mode=std_mode, node_dofs=node_dofs,
        n_bootstrap=n_bootstrap, n_permutations=n_permutations, alpha=alpha, random_state=random_state,
        scores_source="given", standardize_param=(standardize if isinstance(standardize, bool) else std_mode),
    )


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
        topo_cache: Dict[Any, Any] = {}
        for gname, adj in adjs.items():
            view_adj = {i: sorted(names.index(b) for b in adj[a] if b in names)
                        for i, a in enumerate(names)}
            topo = "graph" if any(view_adj.values()) else "star"
            topo_key = (topo, tuple((i, tuple(v)) for i, v in sorted(view_adj.items())))
            if topo_key not in topo_cache:
                extra = {"topology": topo} if topo == "star" else {"topology": "graph", "path_graph": view_adj}
                res = fit(method, v_tr, k, random_state=random_state, **prm, **extra)
                sc_tr = view_scores(res, v_tr, train_views=v_tr)
                sc_te = view_scores(res, v_te, train_views=v_tr)
                Ztr = dict(zip(names, sc_tr["scores"]))
                Zte = dict(zip(names, sc_te["scores"]))
                if has_y:
                    Ztr["outcome"], Zte["outcome"] = _zscore(y_all[tr]), _zscore(y_all[te])
                split_dofs = _result_node_dofs(res, names, len(tr))
                node_cache_t: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
                ll_cmp, _, _ = _graph_loglik(
                    Ztr, Zte, Ntr, Nte, nodes, complete, directed,
                    node_dofs=split_dofs, node_model_cache=node_cache_t,
                )
                r2, coup = _oob_metrics(res, v_tr, v_te)
                out_r2 = None
                if has_y:
                    u_tr = _np(_u_train(res))
                    X = np.column_stack([np.ones(len(tr)), u_tr])
                    W = _ridge_fit(X, y_all[tr])
                    yhat = np.column_stack([np.ones(len(te)), _np(predict(res, v_te)["u"])]) @ W
                    out_r2 = _r2_score(y_all[te], yhat)
                topo_cache[topo_key] = (
                    res, sc_te, Ztr, Zte, split_dofs, node_cache_t, ll_cmp, r2, coup, out_r2, {}
                )
            (res, sc_te, Ztr, Zte, split_dofs, node_cache_t, ll_cmp, r2, coup, out_r2, edge_cache_t) = topo_cache[topo_key]
            ll_own, _, _ = _graph_loglik(
                Ztr, Zte, Ntr, Nte, nodes, adj, directed,
                node_dofs=split_dofs, node_model_cache=node_cache_t,
            )
            pm = path_model(
                res, v_te, rows="heldout", train_views=v_tr, graph=graphs[gname], directed=directed,
                view_names=names, covariates=None if c_all is None else c_all[te],
                groups=None if g_all is None else g_all[te],
                outcome=None if not has_y else y_all[te],
                conditioning="parents", n_permutations=n_permutations, n_bootstrap=0,
                alpha=alpha, random_state=random_state + s,
                _precomputed_sc=sc_te, _edge_cache=edge_cache_t,
            )
            row = {"split": s, "graph": gname, "topology": topo, "n_edges": _n_edges(adj, directed),
                   "n_test": len(te), "heldout_loglik_own": float(ll_own.sum()),
                   "heldout_loglik_complete": float(ll_cmp.sum()),
                   "present_pass": pm["fit"]["present_pass"], "absent_pass": pm["fit"]["absent_pass"],
                   "oos_coupling": None if coup is None else float(np.mean(coup)),
                   "oos_r2": None if r2 is None else float(np.mean(r2)), "outcome_r2": out_r2}
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
    nodes, models = pm["nodes"], pm["edge_models"]
    view_nodes = [nm for nm in nodes if nm not in ("outcome", "consensus")]
    if result is None:
        orig_map = {pos: pos for pos in range(len(view_nodes))}
        signs_full: Optional[List[Optional[np.ndarray]]] = None
    else:
        orig_idx = pm.get("orig_view_indices")
        if orig_idx is None:
            orig_idx = list(range(len(view_nodes)))
        orig_map = {int(orig_m): pos for pos, orig_m in enumerate(orig_idx)}
        signs_full = [None] * len(result["v"])
        for pos, orig_m in enumerate(orig_idx):
            signs_full[int(orig_m)] = pm["signs"][pos]

    def _eval_scores(v_in: Any) -> Dict[str, Any]:
        if result is None:
            return _given_scores_for(pm, v_in)
        v_raw = _view_list(result, v_in)
        v_sub = [v_raw[m] if m in orig_map else None for m in range(len(v_raw))]
        return view_scores(result, v_sub, signs=signs_full)

    sc = _eval_scores(views)
    n = sc["n"]
    N, _, _ = _nuisance(n, covariates, groups)
    if N.shape[1] != models[nodes[0]]["n_nuisance"]:
        raise ValueError("covariates / groups do not match the nuisance design the path model was fitted with "
                         f"({pm['nuisance_names']}).")
    Z: Dict[str, np.ndarray] = {view_nodes[orig_map[m]]: sc["scores"][m] for m in sc["available"]}
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
        st = _eval_scores(views_true)
        r2 = {}
        for m in st["available"]:
            nm = view_nodes[orig_map[m]]
            if nm in predicted_from:
                r2[nm] = _r2_score(st["scores"][m], Z[nm])
        out["r2_per_node"] = r2
    return out


def _given_scores_for(pm: Dict[str, Any], scores: Any) -> Dict[str, Any]:
    """Score blocks for `predict_path` on a `path_model_from_scores` result: a mapping of the model's
    view names (``None`` or absent for a missing block), standardised as the model was."""
    view_nodes = [nm for nm in pm["nodes"] if nm not in ("outcome", "consensus")]
    if not isinstance(scores, Mapping):
        raise ValueError(f"a scores-only path model takes a mapping of its view names {view_nodes}.")
    extra = [k for k in scores if str(k) not in view_nodes]
    if extra:
        raise ValueError(f"unknown score name(s) {extra}; the model has {view_nodes}.")
    full = {nm: scores.get(nm) for nm in view_nodes}
    _, arrays = _given_scores(full, pm["parameters"]["standardize"], allow_missing=True)
    avail = [m for m, z in enumerate(arrays) if z is not None]
    if not avail:
        raise ValueError("no score block is present.")
    return {"scores": arrays, "available": avail, "n": int(arrays[avail[0]].shape[0])}


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
