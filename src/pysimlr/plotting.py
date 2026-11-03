"""Standard plotting for pysimlr results.

Conventions (every function here follows them, and new plots should too):

* **Takes a result object**, the dict returned by a fit/inference function,
  never loose arrays the caller had to assemble. A raw-array helper is named for
  what it plots (`plot_predictions`).
* **``ax=None``**: draw into the given Matplotlib axes, or create a figure.
  **Returns the Figure.** No ``plt.show()``, no saving; the caller decides.
* **Matplotlib only**, imported lazily, so ``import pysimlr`` never needs it.
* **Plot what is interpretable**: coefficients, structure and loadings, with
  their uncertainty where it exists. Signed quantities use one diverging
  colormap centred at 0 with a symmetric range.
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np

__all__ = [
    # data for interpretation
    "feature_table",
    # optimisation health, basis and rank
    "plot_convergence", "plot_basis_diagnostics", "plot_rank_selection",
    # cross-view coupling, weights, flows
    "plot_latent_coupling", "plot_modality_weights", "plot_flow_warping", "plot_view_correlations",
    # interpretable features (never the encoder v)
    "plot_top_features", "plot_feature_groups", "plot_loadings", "plot_variance_explained",
    # scores
    "plot_scores", "plot_latent_recovery",
    # one-call sheet
    "plot_fit_summary",
    # benchmarks and generic
    "plot_benchmark", "plot_benchmark_uncertainty", "plot_predictions",
    # fused ridge
    "plot_fused_coefficients", "plot_coefficient_paths", "plot_fused_ridge_cv",
]

_DIVERGING = "RdBu_r"
#: Okabe-Ito colours (colour-blind safe), used for views in every multi-view plot.
OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
#: Stability reference bars: conservative and low-power (see docs/usage_notes.md).
STABILITY_BARS = (0.9, 0.8)
_UNCONVERGED = "#D55E00"


def _figure(ax=None, figsize=(6.0, 4.0)):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _feature_names(result: Dict[str, Any], p: int):
    names = result.get("feature_names")
    return list(names) if names is not None else [f"x{i}" for i in range(p)]


def _task_labels(result: Dict[str, Any]):
    pre = "y >= " if result.get("model") == "thresholds" else ""
    return [f"{pre}{t:.3g}" if isinstance(t, float) else f"{pre}{t}" for t in result["tasks"]]


def plot_fused_coefficients(result: Dict[str, Any], ax=None, max_features: Optional[int] = 40):
    """Heatmap of a fused ridge's standardised coefficients: tasks (rows) x features.

    Tasks are groups (strata) or outcome thresholds. With more than
    ``max_features`` features, shows those with the largest maximum
    |coefficient|; with ``top_k``, only the selected features.
    """
    w = np.asarray(result["coef_std"]); names = _feature_names(result, w.shape[1])
    cols = np.asarray(result.get("selected_features", range(w.shape[1])))
    if max_features is not None and len(cols) > max_features:
        cols = np.sort(cols[np.argsort(-np.abs(w[:, cols]).max(0))[:max_features]])
    fig, ax = _figure(ax, figsize=(max(6.0, 0.25 * len(cols) + 2), max(3.0, 0.3 * w.shape[0] + 1.5)))
    vmax = float(np.abs(w[:, cols]).max()) or 1.0
    im = ax.imshow(w[:, cols], aspect="auto", cmap=_DIVERGING, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([names[c] for c in cols], rotation=90, fontsize=8)
    ax.set_yticks(range(w.shape[0])); ax.set_yticklabels(_task_labels(result), fontsize=8)
    ax.set_xlabel("feature"); ax.set_ylabel("threshold" if result.get("model") == "thresholds" else "group")
    ax.set_title(f"Fused ridge ({result['family']}, lambda1={result['lambda1']:g}, "
                 f"lambda2={result['lambda2']:g}" + (f", top_k={result['top_k']}" if result.get("top_k") else "") + ")")
    fig.colorbar(im, ax=ax, label="standardised coefficient")
    return fig


def plot_coefficient_paths(result: Dict[str, Any], features: Optional[Sequence] = None,
                           top: int = 8, ax=None):
    """Each feature's standardised coefficient across tasks (groups or thresholds, in order).

    ``features`` are names or column indices; by default the ``top`` features
    by maximum |coefficient|. Fusion shows here: with larger lambda2 the paths
    flatten toward one shared value.
    """
    w = np.asarray(result["coef_std"]); names = _feature_names(result, w.shape[1])
    if features is None:
        cand = np.asarray(result.get("selected_features", range(w.shape[1])))  # top_k: selected only
        cols = list(cand[np.argsort(-np.abs(w[:, cand]).max(0))[:top]])
    else:
        cols = [names.index(f) if isinstance(f, str) else int(f) for f in features]
    fig, ax = _figure(ax)
    x = np.arange(w.shape[0])
    for c in cols:
        ax.plot(x, w[:, c], marker="o", lw=1.5, label=names[c])
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(_task_labels(result), rotation=45, fontsize=8)
    ax.set_xlabel("threshold" if result.get("model") == "thresholds" else "group")
    ax.set_ylabel("standardised coefficient")
    ax.set_title(f"Coefficient paths (lambda2={result['lambda2']:g})")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    return fig


def plot_fused_ridge_cv(cv: Dict[str, Any], ax=None):
    """Cross-validated mean loss over (lambda1, lambda2) at the chosen top_k; best cell marked."""
    k = list(cv["top_ks"]).index(cv["best"][2])
    loss = np.asarray(cv["loss"])[:, :, k]
    fig, ax = _figure(ax)
    im = ax.imshow(loss, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(cv["lambda2s"]))); ax.set_xticklabels([f"{v:g}" for v in cv["lambda2s"]])
    ax.set_yticks(range(len(cv["lambda1s"]))); ax.set_yticklabels([f"{v:g}" for v in cv["lambda1s"]])
    i = list(cv["lambda1s"]).index(cv["best"][0]); j = list(cv["lambda2s"]).index(cv["best"][1])
    ax.scatter([j], [i], marker="*", s=200, color="white", edgecolor="black")
    ax.set_xlabel("lambda2 (fusion)"); ax.set_ylabel("lambda1 (ridge)")
    ax.set_title(f"Cross-validated {cv['family']} loss (top_k={cv['best'][2]})")
    fig.colorbar(im, ax=ax, label="mean held-out loss")
    return fig


def plot_predictions(y_true, y_pred, ax=None, title: str = "Predicted vs observed"):
    """Scatter of predictions against observations, with the identity line and r."""
    y_true = np.asarray(y_true, float).ravel(); y_pred = np.asarray(y_pred, float).ravel()
    fig, ax = _figure(ax, figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.7)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1, ls="--")
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    ax.set_xlabel("observed"); ax.set_ylabel("predicted"); ax.set_title(f"{title} (r = {r:.3f})")
    return fig


# ----------------------------------------------------------------------------- helpers
def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _rotated_scores(res: Dict[str, Any], u=None):
    """``u @ R``: scores in the frame of ``res["loadings"]`` / ``res["structure"]``."""
    u = _np(res["u"] if u is None else u).astype(float)
    return u @ _np(res["loading_rotation"]).astype(float)


def _aligned_frame(res: Dict[str, Any], pred: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The one place the rotation rule lives: scores and latents are rotated by
    ``R = res["loading_rotation"]``; ``structure`` and ``loadings`` already are,
    so they are never rotated again. ``joint`` marks baselines (pca,
    nsa_pipeline) whose single latent is shared by all views."""
    R = _np(res["loading_rotation"]).astype(float)
    src = res if pred is None else pred
    lat = [_np(z).astype(float) @ R for z in src["latents"]]
    return {"R": R, "u": _np(src["u"]).astype(float) @ R, "latents": lat,
            "structure": [_np(c).astype(float) for c in res["structure"]],
            "loadings": [_np(c).astype(float) for c in res["loadings"]],
            "joint": len(res["latents"]) == 1 and len(res["structure"]) > 1}


def _view_feature_names(names, view, p):
    if names is None:
        return [f"v{view}.{i}" for i in range(p)]
    return list(names[view]) if isinstance(names[0], (list, tuple)) else list(names)


def _view_names(view_names, m):
    return list(view_names) if view_names is not None else [f"view {i}" for i in range(m)]


def _stability_color(s, bars=STABILITY_BARS):
    if s is None or not np.isfinite(s):
        return "black"
    return "black" if s >= bars[0] else ("#E69F00" if s >= bars[1] else _UNCONVERGED)


def _status(res: Dict[str, Any]) -> str:
    c = res.get("convergence", {}) or {}
    if c.get("converged", True):
        return ""
    return f"UNCONVERGED: stop_reason={c.get('stop_reason', '?')!r}"


def _fig(figsize):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=figsize)


def _annotate(ax, M, fmt="{:.2f}", extra=None, threshold=0.6):
    """Write each cell's value, white on strong colours and black otherwise."""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            t = fmt.format(M[i, j]) + ("" if extra is None else extra[i][j])
            ax.text(j, i, t, ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(M[i, j]) and abs(M[i, j]) > threshold else "black")


def _correlate_columns(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1.0), np.nan)


# ----------------------------------------------------------------------------- feature data
def feature_table(res: Dict[str, Any], inf: Optional[Dict[str, Any]] = None,
                  view_names=None, feature_names=None):
    """One row per (view, feature, component): the interpretable quantities of a fit.

    Columns: ``view``, ``view_name``, ``feature``, ``feature_name``,
    ``component`` (1-based), ``structure`` (corr(x, u R)), ``loading``, and,
    with a `pysimlr.loading_inference` result ``inf``: ``ci_low``, ``ci_high``,
    ``excludes_zero``, ``selected`` (stability selection), ``selection_frequency``
    and ``component_stability``. Sort, filter or group it however the study
    needs; every feature plot here draws from it.
    """
    import pandas as pd
    F = _aligned_frame(res)
    vn = _view_names(view_names, len(F["structure"]))
    rows = []
    for m, C in enumerate(F["structure"]):
        p, k = C.shape
        names = _view_feature_names(feature_names, m, p)
        L = F["loadings"][m]
        for j in range(k):
            d = {"view": m, "view_name": vn[m], "feature": np.arange(p), "feature_name": names,
                 "component": j + 1, "structure": C[:, j], "loading": L[:, j]}
            if inf is not None:
                lo = _np(inf["ci_low"][m])[:, j].astype(float); hi = _np(inf["ci_high"][m])[:, j].astype(float)
                d.update(ci_low=lo, ci_high=hi, excludes_zero=(lo > 0) | (hi < 0),
                         selected=_np(inf["selected"][m]).astype(bool),
                         selection_frequency=_np(inf["selection_frequency"][m]).astype(float),
                         component_stability=float(inf["component_stability"][m][j]))
            rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- optimisation health
def _draw_convergence(res, ax):
    conv = res.get("convergence", {}) or {}
    status = _status(res)
    if res.get("energy") is not None:
        h = np.asarray([float(_np(v)) for v in res["energy"]])
        ax.plot(np.arange(1, len(h) + 1), h, lw=1.5, color=OKABE_ITO[0], label="objective")
        bi = res.get("best_iteration")
        if bi is not None and 0 <= bi < len(h):
            ax.scatter([bi + 1], [h[bi]], marker="*", s=120, color=OKABE_ITO[1], zorder=3, label="best (returned)")
        unit = "sweep"
    elif res.get("loss_history") is not None:
        h = np.asarray(res["loss_history"], float)
        x = np.arange(1, len(h) + 1)
        ax.plot(x, h, lw=1.8, color=OKABE_ITO[0], label="total loss")
        tw = ax.twinx()
        for key, col in (("sim_history", OKABE_ITO[2]), ("recon_history", OKABE_ITO[3])):
            if res.get(key) is not None and len(res[key]):
                tw.plot(np.arange(1, len(res[key]) + 1), np.asarray(res[key], float), lw=1, ls="--",
                        color=col, label=key.replace("_history", ""))
        tw.set_ylabel("components (dashed)")
        be = conv.get("best_epoch")
        if be is not None and 0 <= be < len(h):
            ax.scatter([be + 1], [h[be]], marker="*", s=120, color=OKABE_ITO[1], zorder=3, label="best (returned)")
        lines = ax.get_legend_handles_labels(); lines2 = tw.get_legend_handles_labels()
        ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=7, frameon=False)
        unit = "epoch"
    else:
        ax.axis("off")
        txt = "\n".join(f"{k}: {conv.get(k)}" for k in ("stop_reason", "converged", "n_steps", "certificate", "grad_map")
                         if k in conv)
        ax.text(0.02, 0.95, f"{res.get('method', '')}: no iteration history\n{txt}", va="top", family="monospace", fontsize=8)
        ax.set_title("Convergence" + (f"  [{status}]" if status else ""), color=_UNCONVERGED if status else "black")
        return
    if res.get("energy") is not None:
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel(unit); ax.set_ylabel("objective")
    gm = conv.get("grad_map")
    sub = f"{conv.get('stop_reason', '?')} after {conv.get('n_steps', len(h))} of {conv.get('max_steps', '?')} {unit}s"
    if gm is not None and np.isfinite(float(gm)):
        sub += f"; grad_map {float(gm):.2g}"
    ax.set_title(("NOT converged -- " if status else "Converged: ") + sub,
                 color=_UNCONVERGED if status else "black", fontsize=9)


def plot_convergence(res: Dict[str, Any], ax=None):
    """Objective history of any fit, titled with stop reason and budget.

    Linear: energy per sweep with the returned best iterate starred. Deep/flow:
    total loss with the similarity and reconstruction terms (dashed, right
    axis) and the returned best epoch. Closed-form baselines: a summary card.
    An unconverged fit is titled "NOT converged" in orange.
    """
    fig, ax = _figure(ax)
    _draw_convergence(res, ax)
    return fig


def _draw_effective_rank(res, ax, view_names=None):
    er = np.asarray(res.get("effective_rank", []), float); nr = np.asarray(res.get("numerical_rank", []), float)
    k = res["structure"][0].shape[1]
    vn = _view_names(view_names, len(er)); y = np.arange(len(er))
    ax.barh(y - 0.18, er, height=0.35, color=[_UNCONVERGED if e < k - 0.5 else OKABE_ITO[0] for e in er],
            label="effective rank")
    if len(nr):
        ax.barh(y + 0.18, nr, height=0.35, color=OKABE_ITO[4], label="numerical rank")
    ax.axvline(k, color="0.3", ls="--", lw=1, label=f"k = {k}")
    ax.set_yticks(y); ax.set_yticklabels(vn, fontsize=8); ax.set_xlabel("rank")
    ax.set_title("Basis rank (orange: columns merging)", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="lower left")


def plot_basis_diagnostics(res: Dict[str, Any], view_names=None, figsize=None):
    """Basis health per view: rank against k, column overlap and conditioning,
    sparsity against the requested level (with the loading margin).

    Predictive R^2 does not reveal merged basis columns; an effective rank
    below k - 0.5 is drawn in orange. These panels read the encoder V because
    they diagnose the optimisation, not because V is interpretable.
    """
    fig = _fig(figsize or (14.0, 0.55 * len(res["v"]) + 2.8)); axes = fig.subplots(1, 3)
    _draw_effective_rank(res, axes[0], view_names)
    vn = _view_names(view_names, len(res["v"])); y = np.arange(len(vn))
    ov = np.asarray(res.get("max_column_overlap", [np.nan] * len(vn)), float)
    cn = np.log10(np.asarray(res.get("condition_number", [np.nan] * len(vn)), float))
    axes[1].scatter(ov, y, color=OKABE_ITO[0], label="max |cos| between columns")
    ax2 = axes[1].twiny(); ax2.scatter(cn, y, color=OKABE_ITO[1], marker="s", label="log10 condition")
    axes[1].set_yticks(y); axes[1].set_yticklabels(vn, fontsize=8); axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("max |cos|"); ax2.set_xlabel("log10 cond(V)", color=OKABE_ITO[1])
    axes[1].set_title("Overlap and conditioning", fontsize=9)
    sp = [float((np.abs(_np(v)) < 1e-4).mean()) for v in res["v"]]
    axes[2].barh(y, sp, color=OKABE_ITO[2])
    req = res.get("sparseness")
    if isinstance(req, (int, float)):
        axes[2].axvline(float(req), color="0.3", ls="--", lw=1, label=f"requested {req:g}")
        axes[2].legend(fontsize=7, frameon=False)
    mg = res.get("loading_margin")
    for m in range(len(vn)):
        if mg is not None and mg[m] is not None:
            axes[2].text(min(sp[m] + 0.02, 0.95), m, f"margin {float(np.median(_np(mg[m]))):.2f}", va="center", fontsize=7)
    axes[2].set_xlim(0, 1); axes[2].set_yticks(y); axes[2].set_yticklabels(vn, fontsize=8)
    axes[2].set_xlabel("fraction of zero entries"); axes[2].set_title("Sparsity", fontsize=9)
    st = _status(res)
    fig.suptitle(f"Basis diagnostics ({res.get('method', '')})" + (f"  [{st}]" if st else ""),
                 color=_UNCONVERGED if st else "black")
    fig.tight_layout()
    return fig


def plot_rank_selection(rank_result: Dict[str, Any], ax=None, bars: Sequence[float] = STABILITY_BARS,
                        criterion: Optional[str] = None):
    """Rank-selection diagnostic for any `pysimlr.select_rank` criterion.

    ``criterion`` defaults to ``rank_result["criterion"]`` (``"component"`` if
    absent): component stability, Grassmannian subspace cosines
    (``"subspace"``), out-of-bag held-out :math:`R^2` +- SE (``"oos_r2"``,
    ``"oos_r2_1se"``), or out-of-bag cross-view coupling (``"oos_coupling"``).
    Candidates with ``basis_degenerate=True`` have their tick coloured orange.
    """
    from .loadings import RANK_CRITERIA
    crit = criterion or rank_result.get("criterion", "component")
    if crit not in RANK_CRITERIA:
        raise ValueError(f"criterion must be one of {RANK_CRITERIA}; got {crit!r}.")
    table = rank_result["table"]
    chosen = (rank_result.get("criteria") or {}).get(crit, rank_result.get("k") if criterion is None else None)
    fig, ax = _figure(ax, figsize=(5.8, 4.0))
    ks = [r["k"] for r in table]
    if crit in ("oos_r2", "oos_r2_1se"):
        ys = np.asarray([np.nan if r.get("oos_r2") is None else r["oos_r2"] for r in table], float)
        se = np.asarray([0.0 if r.get("oos_r2_se") is None else r["oos_r2_se"] for r in table], float)
        ax.errorbar(ks, ys, yerr=se, color=OKABE_ITO[0], lw=1.8, marker="o", capsize=3, label="out-of-bag R^2 +- SE")
        ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
        if np.any(np.isfinite(ys)):
            bi = int(np.nanargmax(ys))
            if ys[bi] > 0:
                ax.axhline(ys[bi] - se[bi], color="0.35", lw=1, ls="--", label="peak - 1 SE")
        if chosen is not None and chosen in ks:
            ax.scatter([chosen], [ys[ks.index(chosen)]], marker="*", s=300, color=OKABE_ITO[1],
                       edgecolor="black", zorder=3, label=f"selected k = {chosen}")
        ax.set_ylabel("held-out reconstruction R^2")
        ax.set_title(f"Rank selection by {crit}")
    elif crit == "oos_coupling":
        mins, meds, thr = [], [], None
        for r in table:
            cp = r.get("oos_coupling")
            if cp is None:
                mins.append(np.nan); meds.append(np.nan); continue
            s = np.asarray(cp, float); k = r["k"]
            mins.append(s.min()); meds.append(np.median(s)); thr = r.get("oos_coupling_threshold", thr)
            ax.vlines(k, s.min(), s.max(), color="0.7", lw=4, zorder=1)
            ax.scatter(np.full(len(s), k), s, color=OKABE_ITO[0], s=28, zorder=2)
        ax.plot(ks, mins, color="black", lw=2, marker="o", label="weakest component")
        ax.plot(ks, meds, color="0.4", lw=1, ls="--", label="median")
        ax.axhline(0.0, color="0.6", lw=0.8)
        if thr is not None:
            ax.axhline(thr, color="0.3", lw=1, ls="-.", label=f"null bar ({thr:.2f})")
        if chosen is not None and chosen in ks:
            ax.scatter([chosen], [mins[ks.index(chosen)]], marker="*", s=300, color=OKABE_ITO[1],
                       edgecolor="black", zorder=3, label=f"selected k = {chosen}")
        ax.set_ylabel("out-of-bag view coupling (r)")
        ax.set_title("Rank selection by held-out coupling")
    else:
        key = "subspace_stability" if crit == "subspace" else "component_stability"
        mins, meds = [], []
        for r in table:
            s = np.asarray(r[key], float); k = r["k"]
            mins.append(s.min()); meds.append(np.median(s))
            ax.vlines(k, s.min(), s.max(), color="0.7", lw=4, zorder=1)
            ax.scatter(np.full(len(s), k), s, color=[_stability_color(v, bars) for v in s], s=28, zorder=2)
        ax.plot(ks, mins, color="black", lw=2, marker="o",
                label="min principal cosine" if crit == "subspace" else "weakest component")
        ax.plot(ks, meds, color="0.4", lw=1, ls="--", label="median")
        if crit == "component" and all("min_subspace_stability" in r for r in table):
            ax.plot(ks, [r["min_subspace_stability"] for r in table], color=OKABE_ITO[0], lw=1.2, ls=":",
                    marker="s", ms=4, label="subspace min cosine")
        for i, b in enumerate(bars):
            ax.axhline(b, color="0.3", lw=1, ls=":" if i else "-.", label=f"{b:g}")
        if chosen is not None and chosen in ks:
            ax.scatter([chosen], [mins[ks.index(chosen)]], marker="*", s=300, color=OKABE_ITO[1],
                       edgecolor="black", zorder=3, label=f"selected k = {chosen}")
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("subspace principal cosine" if crit == "subspace" else "component stability")
        ax.set_title("Rank selection by " + ("subspace stability" if crit == "subspace" else "component stability"))
    ax.set_xticks(ks); ax.set_xlabel("k")
    for lbl, r in zip(ax.get_xticklabels(), table):
        if r.get("basis_degenerate"):
            lbl.set_color(_UNCONVERGED)
    ax.legend(fontsize=7, frameon=False, loc="best")
    return fig


# ----------------------------------------------------------------------------- coupling, weights, flows
def plot_latent_coupling(res: Dict[str, Any], pred: Optional[Dict[str, Any]] = None, view_names=None,
                         component: Optional[int] = None, figsize=None):
    """How strongly each view's scores follow the consensus, per rotated component.

    Panel A: corr(view m score_j, consensus_j) for every view m and component
    j on the training data, and, with ``pred = pysimlr.predict(res, held_out)``,
    the same on held-out subjects with the change annotated. In p >> N studies
    training coupling is optimistic; held-out coupling separates shared signal
    from memorisation. Panel B: view-by-view correlation of one component
    (default: the most strongly coupled). Joint-basis methods (pca,
    nsa_pipeline) have no per-view scores; the figure says so.
    """
    F = _aligned_frame(res)
    vn = _view_names(view_names, len(F["structure"]))
    if F["joint"]:
        fig = _fig(figsize or (5.0, 2.0)); ax = fig.subplots()
        ax.axis("off"); ax.text(0.02, 0.5, f"{res.get('method')}: one joint latent for all views;\n"
                                "per-view coupling is not defined.", va="center")
        return fig
    tr = np.stack([_correlate_columns(z, F["u"]) for z in F["latents"]])
    panels = [("training", tr)]
    if pred is not None:
        P = _aligned_frame(res, pred)
        panels.append(("held-out", np.stack([_correlate_columns(z, P["u"]) for z in P["latents"]])))
    M, k = tr.shape
    fig = _fig(figsize or (3.2 * (len(panels) + 1) + 1.0, 0.5 * M + 2.6))
    axes = fig.subplots(1, len(panels) + 1)
    for ax, (name, Cm) in zip(axes, panels):
        im = ax.imshow(Cm, cmap=_DIVERGING, vmin=-1, vmax=1, aspect="auto")
        extra = ([[f"\n({Cm[i, j] - tr[i, j]:+.2f})" for j in range(k)] for i in range(M)]
                 if name == "held-out" else None)
        _annotate(ax, Cm, extra=extra)
        ax.set_xticks(range(k)); ax.set_xticklabels([f"C{j + 1}" for j in range(k)])
        ax.set_yticks(range(M)); ax.set_yticklabels(vn, fontsize=8); ax.set_title(f"view vs consensus, {name}", fontsize=9)
    fig.colorbar(im, ax=axes[:len(panels)].tolist(), shrink=0.8, label="r")
    j = int(np.nanargmax(np.nanmean(tr, 0))) if component is None else int(component)
    Z = np.stack([z[:, j] for z in F["latents"]], 1)
    pv = np.corrcoef(Z.T)
    ax = axes[-1]
    ax.imshow(pv, cmap=_DIVERGING, vmin=-1, vmax=1)
    _annotate(ax, pv)
    ax.set_xticks(range(M)); ax.set_xticklabels(vn, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(M)); ax.set_yticklabels(vn, fontsize=8); ax.set_title(f"view vs view, C{j + 1}", fontsize=9)
    st = _status(res)
    fig.suptitle("Latent coupling" + (f"  [{st}]" if st else ""), color=_UNCONVERGED if st else "black")
    return fig


def plot_modality_weights(res: Dict[str, Any], view_names=None, figsize=(10.0, 3.6)):
    """Final view weights against uniform 1/M, the modality alignment index, and
    (``flow_v``) their trajectories over training.

    A small view with low weight was down-weighted by the alignment index; check
    it is not a view that matters for the study.
    """
    w = res.get("modality_weights"); mai = res.get("mai")
    if w is None:
        raise ValueError(f"{res.get('method')} result has no modality_weights.")
    w = np.asarray(w, float); M = len(w); vn = _view_names(view_names, M); y = np.arange(M)
    hist = res.get("weight_history") is not None and len(res["weight_history"]) > 1
    fig = _fig(figsize if hist else (5.5, 0.5 * M + 2.2)); axes = np.atleast_1d(fig.subplots(1, 2 if hist else 1))
    ax = axes[0]
    ax.barh(y - 0.18, w, height=0.35, color=[OKABE_ITO[m % 8] for m in range(M)], label="weight")
    if mai is not None:
        ax.barh(y + 0.18, np.asarray(mai, float), height=0.35, color="0.75", label="alignment index (MAI)")
    ax.axvline(1.0 / M, color="0.3", ls="--", lw=1, label="uniform 1/M")
    ax.set_yticks(y); ax.set_yticklabels(vn, fontsize=8); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Modality weights", fontsize=9)
    if hist:
        W = np.asarray(res["weight_history"], float); A = np.asarray(res.get("mai_history"), float)
        for m in range(M):
            axes[1].plot(W[:, m], color=OKABE_ITO[m % 8], lw=1.5, label=f"{vn[m]} weight")
            if A.ndim == 2:
                axes[1].plot(A[:, m], color=OKABE_ITO[m % 8], lw=1, ls=":")
        axes[1].set_xlabel("epoch"); axes[1].set_title("Trajectory (dotted: MAI)", fontsize=9)
        axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def plot_flow_warping(res: Dict[str, Any], views, components=(0, 1), view_names=None, figsize=None):
    """What the nonlinear model does after the linear first layer, per view and component.

    x: the rotated linear projection (X_m V_m R)_j; y: the rotated latent
    (Z_m R)_j the model actually uses. A straight line means the model is
    linear there (the identity for simlr); a monotone curve preserves the sign
    interpretation of V; a non-monotone fold does not. Spearman rho is reported.
    """
    from scipy.stats import spearmanr
    from .api import _preprocess_views
    F = _aligned_frame(res)
    if F["joint"]:
        raise ValueError(f"{res.get('method')} has one joint latent; there is no per-view warp.")
    xs, _ = _preprocess_views(views, res["scale_list"], res["provenance_list"])
    vn = _view_names(view_names, len(xs)); comps = list(components)
    fig = _fig(figsize or (3.0 * len(comps), 2.7 * len(xs)))
    axes = np.atleast_2d(fig.subplots(len(xs), len(comps), squeeze=False))
    for m, x in enumerate(xs):
        lin = (_np(x).astype(float) @ _np(res["v"][m]).astype(float)) @ F["R"]
        for c_i, j in enumerate(comps):
            ax = axes[m, c_i]
            ax.scatter(lin[:, j], F["latents"][m][:, j], s=8, alpha=0.6, color=OKABE_ITO[m % 8])
            rho = spearmanr(lin[:, j], F["latents"][m][:, j]).correlation
            ax.set_title(f"{vn[m]}, C{j + 1}: Spearman {rho:.2f}", fontsize=8)
            ax.set_xlabel("linear X V R", fontsize=7); ax.set_ylabel("latent Z R", fontsize=7)
    fig.suptitle(f"First layer to latent ({res.get('method')})"); fig.tight_layout()
    return fig


def plot_view_correlations(views, names: Optional[Sequence[str]] = None, ax=None):
    """Adjusted RV coefficient between every pair of views (shared multivariate signal)."""
    import torch
    from .utils import adjusted_rvcoef
    mats = [torch.as_tensor(np.asarray(_np(x), float)).float() for x in views]
    n = len(mats); names = list(names) if names is not None else [f"view {i}" for i in range(n)]
    rv = np.array([[float(adjusted_rvcoef(mats[i], mats[j])) for j in range(n)] for i in range(n)])
    fig, ax = _figure(ax, figsize=(1.1 * n + 3, 1.0 * n + 2.5))
    im = ax.imshow(rv, cmap="Blues", vmin=0, vmax=max(1.0, float(rv.max())))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rv[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_title("Inter-view adjusted RV"); fig.colorbar(im, ax=ax)
    return fig


# ----------------------------------------------------------------------------- interpretable features
def _draw_top_features(ax, tab, m, component, kind, top_n, rank_by, null, bars):
    d = tab[(tab.view == m) & (tab.component == component + 1)].copy()
    val = d[kind].to_numpy()
    if rank_by == "selection" and "selection_frequency" in d:
        order = np.lexsort((-np.abs(val), -d["selection_frequency"].to_numpy()))[:top_n]
    else:
        order = np.argsort(-np.abs(val))[:top_n]
    d = d.iloc[order[::-1]]; y = np.arange(len(d))
    has_ci = "ci_low" in d and kind == "structure"
    if has_ci:
        ax.hlines(y, d.ci_low, d.ci_high, color="0.55", lw=1.4)
        sel = d.selected.to_numpy(); exc = d.excludes_zero.to_numpy()
        ax.scatter(d[kind][sel | exc], y[sel | exc], color=OKABE_ITO[5], zorder=3, s=26, label="selected or CI excludes 0")
        ax.scatter(d[kind][~(sel | exc)], y[~(sel | exc)], facecolor="white", edgecolor="0.35", zorder=3, s=26,
                   label="CI includes 0")
    else:
        ax.scatter(d[kind], y, color=OKABE_ITO[0], s=26, zorder=3)
    if null is not None and kind == "structure":
        for sgn in (-1, 1):
            ax.axvline(sgn * null, color=OKABE_ITO[3], ls="--", lw=1)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(d.feature_name, fontsize=7)
    if "selected" in d:
        for lbl, s_ in zip(ax.get_yticklabels(), d.selected):
            if s_:
                lbl.set_fontweight("bold")
    title = f"{d.view_name.iloc[0]}, C{component + 1}"
    color = "black"
    if "component_stability" in d:
        st = float(d.component_stability.iloc[0]); title += f"  (stability {st:.2f})"; color = _stability_color(st, bars)
    ax.set_title(title, fontsize=9, color=color)
    ax.set_xlabel("structure corr(x, u R)" if kind == "structure" else "loading")


def plot_top_features(res: Dict[str, Any], inf: Optional[Dict[str, Any]] = None, component: int = 0,
                      kind: str = "structure", views: Optional[Sequence[int]] = None, view_names=None,
                      feature_names=None, top_n: int = 15, rank_by: str = "magnitude",
                      bars: Sequence[float] = STABILITY_BARS, figsize=None):
    """The features that define one component, one panel per view.

    ``kind="structure"`` (default; defined for every method) or ``"loadings"``.
    With ``inf = pysimlr.loading_inference(...)``: 95% subsample intervals,
    filled markers for stability-selected features or intervals excluding 0
    (selected names in bold), the permutation null (+-``null_max_abs``) when
    computed, and the component's stability in each title, coloured black /
    amber / red against ``bars``. ``rank_by="selection"`` ranks by selection
    frequency instead of magnitude.
    """
    if kind not in ("structure", "loading", "loadings"):
        raise ValueError("kind must be 'structure' or 'loadings'.")
    kind = "loading" if kind.startswith("loading") else "structure"
    tab = feature_table(res, inf, view_names, feature_names)
    vs = list(range(len(res["structure"]))) if views is None else list(views)
    fig = _fig(figsize or (4.2 * len(vs), 0.28 * top_n + 1.6)); axes = np.atleast_1d(fig.subplots(1, len(vs)))
    null = inf.get("null_max_abs") if inf is not None else None
    for ax, m in zip(axes, vs):
        _draw_top_features(ax, tab, m, component, kind, top_n, rank_by, null, bars)
    if inf is not None:
        axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    st = _status(res)
    fig.suptitle(f"Top features, component {component + 1}" + (f"  [{st}]" if st else ""),
                 color=_UNCONVERGED if st else "black")
    fig.tight_layout()
    return fig


def plot_feature_groups(res: Dict[str, Any], groups, inf: Optional[Dict[str, Any]] = None, view: int = 0,
                        feature_names=None, statistic: str = "mean_abs", ax=None):
    """Interpretable features summarised by feature group (lobe, network, pathway...).

    ``groups``: one label per feature of ``view`` (or a dict feature name ->
    label). Heatmap of groups x components: ``statistic="mean_abs"`` (mean
    |structure|), ``"max_abs"``, or, with ``inf``, ``"n_significant"`` (count
    of features whose 95% interval excludes 0) or ``"n_selected"`` (count of
    stability-selected features; strict by design, so often small). Groups are
    ordered by their strongest component.
    """
    tab = feature_table(res, inf, feature_names=feature_names)
    d = tab[tab.view == view].copy()
    if isinstance(groups, dict):
        d["group"] = d.feature_name.map(groups).fillna("other")
    else:
        g = np.asarray(groups)
        if len(g) != d.feature.max() + 1:
            raise ValueError(f"groups has {len(g)} labels; view {view} has {d.feature.max() + 1} features.")
        d["group"] = g[d.feature.to_numpy()]
    if statistic in ("n_selected", "n_significant"):
        if inf is None:
            raise ValueError(f"statistic={statistic!r} needs inf (a loading_inference result).")
        d["v"] = (d.selected if statistic == "n_selected" else d.excludes_zero).astype(float)
        agg, label, cmap = "sum", statistic.replace("_", " "), "viridis"
    elif statistic in ("mean_abs", "max_abs"):
        d["v"] = d.structure.abs(); agg, label, cmap = statistic.split("_")[0], f"{statistic.replace('_', ' ')} structure", "viridis"
    else:
        raise ValueError("statistic must be 'mean_abs', 'max_abs', 'n_significant' or 'n_selected'.")
    tbl = d.pivot_table(index="group", columns="component", values="v", aggfunc=agg)
    tbl = tbl.loc[tbl.max(1).sort_values(ascending=False).index]
    fig, ax = _figure(ax, figsize=(1.0 * tbl.shape[1] + 3.0, 0.32 * tbl.shape[0] + 1.6))
    im = ax.imshow(tbl.to_numpy(float), cmap=cmap, aspect="auto")
    for i in range(tbl.shape[0]):
        for j in range(tbl.shape[1]):
            v = tbl.iat[i, j]
            ax.text(j, i, f"{v:.0f}" if statistic.startswith("n_") else f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > np.nanmax(tbl.to_numpy(float)) * 0.6 else "black")
    ax.set_xticks(range(tbl.shape[1])); ax.set_xticklabels([f"C{c}" for c in tbl.columns])
    ax.set_yticks(range(tbl.shape[0])); ax.set_yticklabels(tbl.index, fontsize=8)
    ax.set_title(f"Feature groups, view {view}"); fig.colorbar(im, ax=ax, label=label)
    return fig


def plot_loadings(res: Dict[str, Any], view: int = 0, feature_names=None,
                  max_features: Optional[int] = 60, ax=None):
    """Loadings of one view: features x components, in the rotated frame.

    Positive fits (hard assignment) show each feature's loading on its
    component; the title reports the median ``loading_margin``. Signed fits
    use the diverging scale. With more than ``max_features`` features, shows
    those with the largest |loading|.
    """
    L = _np(res["loadings"][view]).astype(float); p, k = L.shape
    names = _view_feature_names(feature_names, view, p)
    rows = np.arange(p)
    if max_features is not None and p > max_features:
        rows = np.sort(np.argsort(-np.abs(L).max(1))[:max_features])
    fig, ax = _figure(ax, figsize=(max(3.5, 0.6 * k + 2.5), max(4.0, 0.16 * len(rows) + 1.5)))
    vmax = float(np.abs(L[rows]).max()) or 1.0
    im = ax.imshow(L[rows], aspect="auto", cmap=_DIVERGING, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([names[r] for r in rows], fontsize=7)
    ax.set_xticks(range(k)); ax.set_xticklabels([f"C{j + 1}" for j in range(k)])
    title = f"Loadings, view {view}"
    margin = res.get("loading_margin")
    if margin is not None and margin[view] is not None:
        title += f" (median margin {float(np.median(_np(margin[view]))):.2f})"
    ax.set_title(title); fig.colorbar(im, ax=ax, label="loading")
    return fig


def plot_variance_explained(res: Dict[str, Any], views, pred: Optional[Dict[str, Any]] = None,
                            view_names=None, figsize=(10.0, 3.8)):
    """Reconstruction R^2 per view (training, and held-out with ``pred``) and the
    share of each view's correlation structure carried by each component.

    Panel A: training R^2 from ``predict(res, views)`` against held-out
    ``pred["r2"]``; values below -0.25 are clipped and labelled. Panel B: mean
    squared structure coefficient per component, stacked: how much of each
    view's correlation structure the components capture.
    """
    from .api import predict
    tr = np.asarray([float(_np(v)) for v in predict(res, views)["r2"]])
    M = len(tr); vn = _view_names(view_names, M); y = np.arange(M)
    fig = _fig(figsize); axes = fig.subplots(1, 2)
    ax = axes[0]
    ax.scatter(np.clip(tr, -0.25, 1), y, color=OKABE_ITO[0], s=40, zorder=3, label="training")
    if pred is not None:
        te = np.asarray([float(_np(v)) for v in pred["r2"]])
        ax.scatter(np.clip(te, -0.25, 1), y, color=OKABE_ITO[1], s=40, zorder=3, label="held-out")
        ax.hlines(y, np.clip(np.minimum(tr, te), -0.25, 1), np.clip(np.maximum(tr, te), -0.25, 1), color="0.6")
        for m in range(M):
            if te[m] < -0.25:
                ax.text(-0.24, m + 0.2, f"{te[m]:.2f}", fontsize=7, color=_UNCONVERGED)
    ax.axvline(0, color="0.6", lw=0.8); ax.set_xlim(-0.27, 1.0)
    ax.set_yticks(y); ax.set_yticklabels(vn, fontsize=8); ax.set_xlabel("reconstruction R^2")
    ax.legend(fontsize=7, frameon=False); ax.set_title("Variance explained per view", fontsize=9)
    F = _aligned_frame(res); left = np.zeros(M)
    for j in range(F["structure"][0].shape[1]):
        share = np.array([float((C[:, j] ** 2).mean()) for C in F["structure"]])
        axes[1].barh(y, share, left=left, color=OKABE_ITO[j % 8], label=f"C{j + 1}"); left += share
    axes[1].set_yticks(y); axes[1].set_yticklabels(vn, fontsize=8)
    axes[1].set_xlabel("mean squared structure coefficient"); axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title("Structure captured, by component", fontsize=9)
    fig.tight_layout()
    return fig


def plot_scores(res: Dict[str, Any], components=(0, 1), color_by=None, new_views=None,
                ax=None):
    """Subject scores in the rotated frame (``u @ R``), with optional held-out subjects.

    ``new_views`` are projected with `pysimlr.predict` and drawn as crosses.
    ``color_by`` is one value per training subject (numeric or labels).
    """
    a, b = components
    U = _rotated_scores(res)
    fig, ax = _figure(ax, figsize=(5.0, 4.5))
    if color_by is None:
        ax.scatter(U[:, a], U[:, b], s=14, alpha=0.75, label="training")
    else:
        cb = np.asarray(color_by)
        if cb.dtype.kind in "fiu":
            sc = ax.scatter(U[:, a], U[:, b], s=14, c=cb, cmap="viridis", alpha=0.8)
            fig.colorbar(sc, ax=ax)
        else:
            for lvl in dict.fromkeys(cb.tolist()):
                m = cb == lvl
                ax.scatter(U[m, a], U[m, b], s=14, alpha=0.8, label=str(lvl))
    if new_views is not None:
        from .api import predict
        V = _rotated_scores(res, predict(res, new_views)["u"])
        ax.scatter(V[:, a], V[:, b], s=30, marker="x", color="k", label="held-out")
    ax.set_xlabel(f"component {a + 1} (u R)"); ax.set_ylabel(f"component {b + 1} (u R)")
    ax.set_title("Subject scores")
    if color_by is None or np.asarray(color_by).dtype.kind not in "fiu" or new_views is not None:
        ax.legend(fontsize=7, frameon=False)
    return fig


def plot_latent_recovery(res: Dict[str, Any], z_true, ax=None):
    """|corr| between rotated components and planted latents (synthetic studies)."""
    U = _rotated_scores(res); Z = np.asarray(_np(z_true), float)
    k, q = U.shape[1], Z.shape[1]
    C = np.abs(np.corrcoef(U.T, Z.T)[:k, k:])
    fig, ax = _figure(ax, figsize=(1.0 * q + 3, 1.0 * k + 2))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(q):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(q)); ax.set_xticklabels([f"z{j + 1}" for j in range(q)])
    ax.set_yticks(range(k)); ax.set_yticklabels([f"C{i + 1}" for i in range(k)])
    ax.set_title("Latent recovery |corr|"); fig.colorbar(im, ax=ax)
    return fig


# ----------------------------------------------------------------------------- benchmarks


# ----------------------------------------------------------------------------- one-call sheet
def plot_fit_summary(res: Dict[str, Any], views=None, pred: Optional[Dict[str, Any]] = None,
                     inf: Optional[Dict[str, Any]] = None, view_names=None, feature_names=None,
                     top_n: int = 8, save_path=None, figsize=(17.0, 10.0)):
    """One diagnostic sheet for a fit: convergence, basis rank, coupling
    (training and held-out), variance explained, and the top features of the
    most strongly coupled component in the first two views (with intervals when
    ``inf`` is given). ``views`` (the training views) enables the variance
    panel. Saved to ``save_path`` if given.
    """
    import pandas as pd  # noqa: F401  (feature_table)
    fig = _fig(figsize); gs = fig.add_gridspec(2, 3)
    _draw_convergence(res, fig.add_subplot(gs[0, 0]))
    if res.get("effective_rank") is not None:
        _draw_effective_rank(res, fig.add_subplot(gs[0, 1]), view_names)
    F = _aligned_frame(res); vn = _view_names(view_names, len(F["structure"]))
    ax = fig.add_subplot(gs[0, 2])
    if views is not None:
        from .api import predict
        tr = [float(_np(v)) for v in predict(res, views)["r2"]]
        y = np.arange(len(tr))
        ax.scatter(np.clip(tr, -0.25, 1), y, color=OKABE_ITO[0], label="training")
        if pred is not None:
            ax.scatter(np.clip([float(_np(v)) for v in pred["r2"]], -0.25, 1), y, color=OKABE_ITO[1], label="held-out")
        ax.set_yticks(y); ax.set_yticklabels(vn, fontsize=8); ax.set_xlim(-0.27, 1); ax.axvline(0, color="0.6", lw=0.8)
        ax.legend(fontsize=7, frameon=False); ax.set_title("Reconstruction R^2", fontsize=9)
    else:
        ax.axis("off"); ax.text(0.02, 0.5, "pass views= for variance explained", fontsize=8)
    ax = fig.add_subplot(gs[1, 0])
    if F["joint"]:
        ax.axis("off"); ax.text(0.02, 0.5, "joint latent: no per-view coupling", fontsize=8); j_top = 0
    else:
        tr = np.stack([_correlate_columns(z, F["u"]) for z in F["latents"]])
        Cm, title = tr, "view vs consensus (training)"
        if pred is not None:
            P = _aligned_frame(res, pred)
            Cm, title = np.stack([_correlate_columns(z, P["u"]) for z in P["latents"]]), "view vs consensus (held-out)"
        ax.imshow(Cm, cmap=_DIVERGING, vmin=-1, vmax=1, aspect="auto")
        _annotate(ax, Cm)
        ax.set_xticks(range(Cm.shape[1])); ax.set_xticklabels([f"C{j + 1}" for j in range(Cm.shape[1])])
        ax.set_yticks(range(Cm.shape[0])); ax.set_yticklabels(vn, fontsize=8); ax.set_title(title, fontsize=9)
        j_top = int(np.nanargmax(np.nanmean(tr, 0)))
    tab = feature_table(res, inf, view_names, feature_names)
    null = inf.get("null_max_abs") if inf is not None else None
    for c_i, m in enumerate(range(min(2, len(F["structure"])))):
        _draw_top_features(fig.add_subplot(gs[1, 1 + c_i]), tab, m, j_top, "structure", top_n, "magnitude",
                           null, STABILITY_BARS)
    st = _status(res)
    fig.suptitle(f"{res.get('method', 'fit')} summary" + (f"  [{st}]" if st else ""),
                 color=_UNCONVERGED if st else "black", fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------- benchmarks
def plot_benchmark(df, x: str, y: str, by: Optional[str] = "model", kind: str = "line", ax=None):
    """One benchmark metric against a setting, per method, from a results DataFrame.

    ``kind="line"``: mean +- standard error of ``y`` at each ``x``, one line per
    ``by``. ``kind="scatter"``: every row. ``kind="bar"``: grouped means with
    standard-error bars. This replaces the column-specific benchmark plots
    (sparsity sensitivity, stability diagnostics, Pareto and trade-off plots):
    each is ``plot_benchmark(df, x=..., y=..., kind=...)``.
    """
    if kind not in ("line", "scatter", "bar"):
        raise ValueError(f"kind must be 'line', 'scatter' or 'bar'; got {kind!r}.")
    missing = [c for c in (x, y, by) if c is not None and c not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing}; available: {list(df.columns)}")
    groups = [(None, df)] if by is None else list(df.groupby(by, sort=False))
    fig, ax = _figure(ax, figsize=(6.5, 4.5))
    xs = list(dict.fromkeys(df[x].tolist()))
    for gi, (name, d) in enumerate(groups):
        lbl = None if name is None else str(name)
        if kind == "scatter":
            ax.scatter(d[x], d[y], s=18, alpha=0.7, label=lbl)
            continue
        st = d.groupby(x, sort=False)[y].agg(["mean", "std", "count"]).reindex(xs)
        se = (st["std"] / np.sqrt(st["count"])).fillna(0.0)
        if kind == "line":
            ax.errorbar(range(len(xs)) if df[x].dtype == object else xs, st["mean"], yerr=se,
                        marker="o", capsize=3, label=lbl)
        else:
            w = 0.8 / max(1, len(groups))
            ax.bar(np.arange(len(xs)) + gi * w - 0.4 + w / 2, st["mean"], width=w, yerr=se,
                   capsize=2, label=lbl)
    if kind == "bar" or (kind == "line" and df[x].dtype == object):
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(v) for v in xs])
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} by {x}")
    if by is not None:
        ax.legend(fontsize=7, frameon=True, framealpha=0.85, title=by, loc="best")
    return fig


def plot_benchmark_uncertainty(df, method: str = "method", split: str = "split", score: str = "score",
                               baseline: Optional[str] = None, n_boot: int = 2000, random_state: int = 0,
                               metric_label: Optional[str] = None, figsize=(11.0, None)):
    """Out-of-sample comparison with its uncertainty: per method, the mean score
    with a 95% bootstrap interval over splits and every split shown; and, with
    ``baseline``, the paired difference (method - baseline) per split with its
    bootstrap interval. Intervals that exclude 0 are drawn solid.

    ``df`` is long: one row per (method, split) with a score (e.g. repeated
    K-fold held-out r). Intervals are computed here from the split scores, not
    taken from the caller.
    """
    rng = np.random.default_rng(random_state)
    methods = list(dict.fromkeys(df[method].tolist()))
    piv = df.pivot_table(index=split, columns=method, values=score)

    def boot(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if len(x) < 2:
            return np.nan, np.nan
        b = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)
    h = figsize[1] or 0.45 * len(methods) + 1.8
    fig = _fig((figsize[0], h)); axes = np.atleast_1d(fig.subplots(1, 2 if baseline else 1))
    y = np.arange(len(methods))
    ax = axes[0]
    for i, mth in enumerate(methods):
        x = piv[mth].dropna().to_numpy(); lo, hi = boot(x)
        ax.scatter(x, np.full(len(x), i) + rng.uniform(-0.15, 0.15, len(x)), s=10, color="0.65", zorder=1)
        ax.hlines(i, lo, hi, color=OKABE_ITO[0], lw=3, zorder=2); ax.scatter([x.mean()], [i], color="black", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=8); ax.set_xlabel(metric_label or score)
    ax.set_title("Score per method (95% bootstrap CI; dots = splits)", fontsize=9)
    if baseline:
        if baseline not in piv:
            raise ValueError(f"baseline {baseline!r} not among methods {methods}.")
        ax = axes[1]
        for i, mth in enumerate(methods):
            d = (piv[mth] - piv[baseline]).dropna().to_numpy()
            if mth == baseline or len(d) == 0:
                continue
            lo, hi = boot(d); excl = lo > 0 or hi < 0
            ax.hlines(i, lo, hi, color=OKABE_ITO[5] if excl else "0.6", lw=3 if excl else 2,
                      linestyles="solid" if excl else "dashed")
            ax.scatter([d.mean()], [i], color="black", zorder=3)
        ax.axvline(0, color="0.3", lw=1); ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=8)
        ax.set_xlabel(f"paired difference vs {baseline}")
        ax.set_title("Paired difference (solid: CI excludes 0)", fontsize=9)
    fig.tight_layout()
    return fig


