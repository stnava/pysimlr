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

__all__ = ["plot_fused_coefficients", "plot_coefficient_paths", "plot_fused_ridge_cv",
           "plot_predictions"]

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
    return [f"{pre}{t:g}" if isinstance(t, float) else f"{pre}{t}" for t in result["tasks"]]


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
        cols = list(np.argsort(-np.abs(w).max(0))[:top])
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
