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
    # interpretation of a fit (never the encoder v)
    "plot_loadings", "plot_structure_intervals", "plot_component_stability", "plot_scores",
    "plot_view_agreement", "plot_convergence", "plot_view_correlations", "plot_latent_recovery",
    # fused ridge
    "plot_fused_coefficients", "plot_coefficient_paths", "plot_fused_ridge_cv",
    # generic
    "plot_predictions", "plot_benchmark",
]

_DIVERGING = "RdBu_r"


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


def _view_feature_names(names, view, p):
    if names is None:
        return [f"v{view}.{i}" for i in range(p)]
    return list(names[view]) if isinstance(names[0], (list, tuple)) else list(names)


# ----------------------------------------------------------------------------- interpretation
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


def plot_structure_intervals(inference: Dict[str, Any], view: int = 0, component: int = 0,
                             top: int = 20, feature_names=None, ax=None):
    """Forest plot of one component's structure coefficients with their 95% intervals.

    Takes a `pysimlr.loading_inference` result. Shows the ``top`` features by
    |structure|. Filled markers: the interval excludes 0. Bold names: the
    feature is stability-selected.
    """
    c = _np(inference["structure"][view])[:, component].astype(float)
    lo = _np(inference["ci_low"][view])[:, component].astype(float)
    hi = _np(inference["ci_high"][view])[:, component].astype(float)
    sel = _np(inference["selected"][view]).astype(bool)
    names = _view_feature_names(feature_names, view, len(c))
    order = np.argsort(-np.abs(c))[:top][::-1]
    fig, ax = _figure(ax, figsize=(6.0, max(3.0, 0.25 * len(order) + 1.2)))
    y = np.arange(len(order))
    excl = (lo[order] > 0) | (hi[order] < 0)
    ax.hlines(y, lo[order], hi[order], color="0.4", lw=1.5)
    ax.scatter(c[order][excl], y[excl], color="C3", zorder=3, label="95% interval excludes 0")
    ax.scatter(c[order][~excl], y[~excl], facecolor="white", edgecolor="0.3", zorder=3,
               label="interval includes 0")
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=8)
    for lbl, i in zip(ax.get_yticklabels(), order):
        if sel[i]:
            lbl.set_fontweight("bold")
    stab = inference.get("component_stability")
    st = f", stability {float(stab[view][component]):.2f}" if stab is not None else ""
    ax.set_xlabel("structure coefficient corr(x, u R)")
    ax.set_title(f"View {view}, component {component + 1}{st}")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    return fig


def plot_component_stability(rank_result: Dict[str, Any], ax=None,
                             reference_lines: Sequence[float] = (0.9, 0.8)):
    """Per-component stability for each candidate k (a `pysimlr.select_rank` result).

    Stability is graded: the reference lines mark the conservative (0.9) and
    low-power (0.8) bars, and the plot states rather than judges them.
    """
    table = rank_result["table"]
    fig, ax = _figure(ax, figsize=(1.2 * sum(r["k"] for r in table) + 2.5, 4.0))
    x0, ticks, labels = 0, [], []
    for r in table:
        s = np.asarray(r["component_stability"], float); xs = x0 + np.arange(len(s))
        ax.bar(xs, s, color=["C0" if v >= reference_lines[0] else "C1" for v in s], width=0.8)
        ticks += list(xs); labels += [f"k={r['k']}\nC{j + 1}" for j in range(len(s))]
        x0 += len(s) + 1
    for i, v in enumerate(reference_lines):
        ax.axhline(v, color="0.3", lw=1, ls="--" if i == 0 else ":", label=f"{v:g}")
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.02); ax.set_ylabel("component stability")
    chosen = rank_result.get("k")
    ax.set_title(f"Component stability by rank (selected k = {chosen})")
    ax.legend(title="reference", fontsize=7, frameon=False)
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


def plot_view_agreement(res: Dict[str, Any], component: int = 0, ax=None):
    """Each view's scores against the consensus for one rotated component, with r per view."""
    R = _np(res["loading_rotation"]).astype(float)
    u = _rotated_scores(res)[:, component]
    fig, ax = _figure(ax, figsize=(5.0, 4.5))
    for m, z in enumerate(res["latents"]):
        zm = (_np(z).astype(float) @ R)[:, component]
        zm = (zm - zm.mean()) / (zm.std() or 1.0)
        r = float(np.corrcoef(u, zm)[0, 1])
        ax.scatter(u, zm, s=10, alpha=0.6, label=f"view {m} (r = {r:.2f})")
    ax.set_xlabel(f"consensus, component {component + 1}"); ax.set_ylabel("view score (standardised)")
    ax.set_title("Agreement of each view with the consensus")
    ax.legend(fontsize=7, frameon=False)
    return fig


def plot_convergence(res: Dict[str, Any], ax=None):
    """The fit's objective history, with the stop reason and budget stated."""
    hist = res.get("energy") if res.get("energy") is not None else res.get("loss_history")
    if hist is None:
        raise ValueError("result carries no 'energy' or 'loss_history'.")
    h = np.asarray([float(_np(v)) for v in hist])
    conv = res.get("convergence", {})
    fig, ax = _figure(ax)
    ax.plot(np.arange(1, len(h) + 1), h, lw=1.5)
    unit = "sweep" if res.get("energy") is not None else "epoch"
    ax.set_xlabel(unit); ax.set_ylabel("objective")
    ok = conv.get("converged")
    ax.set_title(f"Convergence: {conv.get('stop_reason', '?')} after {conv.get('n_steps', len(h))} "
                 f"of {conv.get('max_steps', '?')} {unit}s" + ("" if ok else "  (NOT converged)"),
                 color="black" if ok else "C3")
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
        ax.legend(fontsize=7, frameon=False, title=by)
    return fig
