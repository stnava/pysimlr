"""Standard plotting for pysimlr results.

Conventions (every function here follows them, and new plots should too):

* **Takes a result object**, the dict returned by a fit/inference function,
  never loose arrays the caller had to assemble. A raw-array helper is named for
  what it plots (`plot_predictions`).
* **``ax=None``**: draw into the given Matplotlib axes, or create a figure.
  **Returns the Figure.** No ``plt.show()``, no saving; the caller decides.
* **Matplotlib only**, imported lazily, so ``import pysimlr`` never needs it.
* **Plot what is interpretable**: coefficients, structure and loadings, with
  their uncertainty where it exists, never the encoder ``v``.
* **One visual system** (`theme`): a sans-serif type hierarchy (a left-aligned
  title, a one-line reading of the figure beneath it), hairline recessive axes,
  thin marks with surface rings, and colour by role. Views take the
  categorical slots `VIEW_COLORS` in fixed order (validated for colour-vision
  deficiency); magnitude uses one blue ramp; signed quantities use a blue-red
  diverging scale with a neutral midpoint, symmetric about 0. Every plot is
  drawn inside ``theme()``; wrap your own figures in it to match.
"""
import functools
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence

import numpy as np

__all__ = [
    # style and data for interpretation
    "theme", "feature_table",
    # optimisation health, basis and rank
    "plot_convergence", "plot_basis_diagnostics", "plot_rank_selection",
    # cross-view coupling, weights, flows
    "plot_latent_coupling", "plot_view_chord", "plot_component_radar", "plot_view_radar",
    "plot_modality_weights", "plot_flow_warping", "plot_view_correlations",
    # interpretable features (never the encoder v)
    "plot_top_features", "plot_feature_chord", "plot_feature_groups", "plot_loadings",
    "plot_variance_explained", "plot_feature_relevance", "plot_component_attribution",
    # scores
    "plot_scores", "plot_latent_recovery",
    # one-call sheet
    "plot_fit_summary",
    # benchmarks and generic
    "plot_benchmark", "plot_benchmark_uncertainty", "plot_predictions",
    # fused ridge
    "plot_fused_coefficients", "plot_coefficient_paths", "plot_fused_ridge_cv",
    # score-level path model
    "plot_path_model", "plot_edge_matrix", "plot_path_comparison", "plot_edge_null", "plot_edge_features",
    "plot_edge_brain_map",
]

# ----------------------------------------------------------------------------- the visual system
#: Categorical colours, assigned to views (and to methods in benchmarks) in this
#: fixed order, never cycled or re-ranked. Validated with the dataviz palette
#: checker on the light surface: every adjacent pair clears the colour-vision-
#: deficiency floor (worst dE 9.1) and the normal-vision floor (19.6); the first
#: three clear it for all pairs. Aqua, yellow and magenta sit below 3:1 contrast,
#: so views are always direct-labelled as well as coloured.
VIEW_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
#: Stability reference bars: conservative and low-power (see docs/usage_notes.md).
STABILITY_BARS = (0.9, 0.8)

_INK = "#0b0b0b"          # primary text
_INK2 = "#52514e"         # secondary text
_MUTED = "#898781"        # axis labels, ticks
_GRID = "#e1e0d9"         # gridline hairline
_AXIS = "#c3c2b7"         # baseline / axis
_SURFACE = "#fcfcfb"      # chart surface
_NEUTRAL = "#f0efec"      # diverging midpoint
_GOOD, _WARN, _SERIOUS, _CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"   # status, never series
_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
_RED = ("#fbd5cf", "#f6aea4", "#ef8478", "#e34948", "#c93432", "#a42523", "#7d1a19")

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10.0,
    "text.color": _INK,
    "axes.titlesize": 11.0, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 9.0, "axes.titlecolor": _INK,
    "axes.labelsize": 9.5, "axes.labelcolor": _INK2, "axes.labelpad": 6.0,
    "axes.edgecolor": _AXIS, "axes.linewidth": 0.8, "axes.facecolor": _SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.axisbelow": True,
    "grid.color": _GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": _AXIS, "ytick.color": _AXIS, "xtick.labelcolor": _INK2, "ytick.labelcolor": _INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.major.size": 3.0, "ytick.major.size": 0.0, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.pad": 4.0, "ytick.major.pad": 5.0,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2, "legend.labelcolor": _INK2,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "lines.markersize": 6.0, "patch.linewidth": 0.0,
    "figure.facecolor": _SURFACE, "figure.dpi": 110,
    "savefig.facecolor": _SURFACE, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    "image.interpolation": "nearest",
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica Neue", "mathtext.it": "Helvetica Neue:italic",
}


@contextmanager
def theme():
    """The pysimlr figure style as a context manager: ``with theme(): ...``.

    Every plotting function here already draws inside it; use it to make your
    own figures match.
    """
    import matplotlib as mpl
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    with mpl.rc_context(_RC):
        yield


def _themed(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with theme():
            return fn(*args, **kwargs)
    return wrapped


def _cmap(kind: str):
    """``"seq"``: one blue ramp from near-surface to dark; ``"div"``: blue - neutral - red."""
    from matplotlib.colors import LinearSegmentedColormap
    if kind == "seq":
        return LinearSegmentedColormap.from_list("pysimlr_seq", ("#f3f7fd",) + _BLUE[1::2] + (_BLUE[-1],))
    return LinearSegmentedColormap.from_list(
        "pysimlr_div", (_BLUE[-2], _BLUE[5], _BLUE[2], _NEUTRAL, _RED[1], _RED[3], _RED[5]))


def _view_color(m: int) -> str:
    return VIEW_COLORS[m % len(VIEW_COLORS)]


def _rgba(c, a=1.0):
    from matplotlib.colors import to_rgba
    r = to_rgba(c)
    return (r[0], r[1], r[2], a)


def _new(figsize, title=None, subtitle=None, status=None, **subplot_kw):
    """A constrained-layout figure with a header band (title, one-line reading,
    and a status line for an unconverged fit) reserved above the panels."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize, layout="constrained")
    _header(fig, title, subtitle, status)
    return fig


def _header(fig, title=None, subtitle=None, status=None):
    if not (title or subtitle or status):
        return
    w, h = fig.get_size_inches()
    top = 0.16 + (0.30 if title else 0) + (0.24 if subtitle else 0) + (0.24 if status else 0)
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(rect=(0, 0, 1, max(0.2, 1 - (top + 0.02) / h)))
    x, y = 0.10 / w, 1 - 0.12 / h
    if title:
        fig.text(x, y, title, ha="left", va="top", fontsize=13.5, weight="semibold", color=_INK)
        y -= 0.30 / h
    if subtitle:
        fig.text(x, y, subtitle, ha="left", va="top", fontsize=9.5, color=_INK2)
        y -= 0.24 / h
    if status:
        fig.text(x, y, "●", ha="left", va="top", fontsize=9.5, color=_CRITICAL)
        fig.text(x + 0.16 / w, y, status, ha="left", va="top", fontsize=9.5, color=_INK, weight="semibold")


def _figure(ax=None, figsize=(6.0, 4.0), title=None, subtitle=None, status=None):
    """``(fig, ax)``: the caller's axes, or a new single-panel figure with a header."""
    if ax is not None:
        return ax.figure, ax
    fig = _new(figsize, title, subtitle, status)
    return fig, fig.add_subplot()


def _clean(ax, grid: Optional[str] = "x", baseline: bool = True):
    """Recessive frame: no top/right spines, hairline grid on the value axis only."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(grid != "y" and baseline)
    ax.spines["bottom"].set_visible(baseline)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_AXIS); ax.spines[s].set_linewidth(0.8)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=_GRID, lw=0.8)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=_GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor(_SURFACE)


def _dots(ax, x, y, color, s=46, hollow=False, zorder=3, **kw):
    """Markers with a 2px surface ring, so they stay legible where they overlap."""
    return ax.scatter(x, y, s=s, facecolor=_SURFACE if hollow else color, edgecolor=color if hollow else _SURFACE,
                      linewidth=1.6 if hollow else 1.4, zorder=zorder, **kw)


def _status(res: Dict[str, Any]) -> str:
    c = res.get("convergence", {}) or {}
    if c.get("converged", True):
        return ""
    return f"UNCONVERGED: stop_reason={c.get('stop_reason', '?')!r}"


def _status_line(res: Dict[str, Any]) -> Optional[str]:
    c = res.get("convergence", {}) or {}
    if c.get("converged", True):
        return None
    return (f"Not converged: stopped by {c.get('stop_reason', '?')!r} after {c.get('n_steps', '?')} of "
            f"{c.get('max_steps', '?')} steps. Read the rest with care.")


def _cell_text_color(rgba) -> str:
    r, g, b = rgba[:3]
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.52 else _INK


def _matrix(ax, M, cmap, vmin, vmax, fmt="{:.2f}", rows=None, cols=None, annotate=True, extra=None):
    """A heatmap drawn as separate cells with 2px surface gaps (no colour block),
    annotated in white or near-black by the cell's luminance."""
    import matplotlib as mpl
    M = np.asarray(M, float)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(np.arange(M.shape[1] + 1) - 0.5, np.arange(M.shape[0] + 1) - 0.5, np.ma.masked_invalid(M),
                         cmap=cmap, norm=norm, edgecolors=_SURFACE, linewidth=2.0)
    ax.set_xlim(-0.5, M.shape[1] - 0.5); ax.set_ylim(M.shape[0] - 0.5, -0.5)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isfinite(M[i, j]):
                    continue
                val_ij = M[i, j]
                if abs(val_ij) < 0.005:
                    val_ij = 0.0
                t = fmt.format(val_ij) + ("" if extra is None else extra[i][j])
                ax.text(j, i, t, ha="center", va="center", fontsize=8.5,
                        color=_cell_text_color(mesh.cmap(norm(M[i, j]))))
    if cols is not None:
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(cols)
    else:
        ax.set_xticks([])
    if rows is not None:
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(rows)
    else:
        ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return mesh


def _colorbar(fig, mappable, ax, label, ticks=None):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.75, aspect=22, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8, labelcolor=_INK2)
    cb.set_label(label, color=_INK2, fontsize=8.5)
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


def _dodge(vals, gap):
    """Positions for end labels: the values, pushed apart to at least ``gap``
    (in data units) in their original order, so labels never overlap."""
    vals = np.asarray(vals, float); order = np.argsort(vals); out = vals.copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = (out[order].mean() - vals[order].mean())
    return out - shift


def _legend_handles(items):
    """Legend entries as ringed dots (identity is never colour-alone: the label sits beside)."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="", marker="o", ms=7, mfc=c, mec=_SURFACE, mew=1.2, label=l) for l, c in items]


# ----------------------------------------------------------------------------- result helpers
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
    raw = src["latents"]
    if isinstance(raw, dict):                     # predict on a subset / extended views: by name
        from .extension import view_names
        raw = [raw[nm] for nm in view_names(res) if nm in raw]
    else:
        raw = list(raw)
        if pred is None:                          # views added by pysimlr.extend, same subjects
            n = _np(res["u"]).shape[0]
            raw += [e["latent"] for e in (res.get("extensions") or {}).values()
                    if e.get("latent") is not None and e["latent"].shape[0] == n]
    lat = [_np(z).astype(float) @ R for z in raw]
    return {"R": R, "u": _np(src["u"]).astype(float) @ R, "latents": lat,
            "structure": [_np(c).astype(float) for c in res["structure"]],
            "loadings": [_np(c).astype(float) for c in res["loadings"]],
            "joint": len(res["latents"]) == 1 and len(res["structure"]) > 1}


def _view_feature_names(names, view, p):
    if names is None:
        return [f"v{view}.{i}" for i in range(p)]
    return list(names[view]) if isinstance(names[0], (list, tuple)) else list(names)


def _view_names(view_names, m):
    return list(view_names) if view_names is not None else [f"view {i + 1}" for i in range(m)]


def _stability_color(s, bars=STABILITY_BARS):
    if s is None or not np.isfinite(s):
        return _INK
    return _INK if s >= bars[0] else (_SERIOUS if s >= bars[1] else _CRITICAL)


def _correlate_columns(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1.0), np.nan)


def _coupling(res, pred=None):
    """``(M, k)`` corr(view score_j, consensus_j), training or (with ``pred``) held-out."""
    F = _aligned_frame(res, pred)
    return np.stack([_correlate_columns(z, F["u"]) for z in F["latents"]])


def _pairwise(res, pred=None):
    """``(k, M, M)`` correlation between the views' scores, per rotated component."""
    F = _aligned_frame(res, pred)
    k = F["latents"][0].shape[1]
    return np.stack([np.corrcoef(np.stack([z[:, j] for z in F["latents"]], 0)) for j in range(k)])


def _joint_card(fig, ax, res, what):
    ax.axis("off")
    ax.text(0.0, 0.5, f"{res.get('method')}: one joint latent for all views;\n{what} is not defined.",
            va="center", fontsize=10, color=_INK2)
    return fig


# ----------------------------------------------------------------------------- feature data
_MAP_LABELS = {"structure": "structure  corr(x, u R)", "partial": "partial correlation (inverse corr.)",
               "global_partial": "structure beyond the global factor", "component_partial":
               "component-partial pattern", "relative_weight": "relative weight (share of R²)"}


def _maps_of(inf, maps):
    """{kind: (map per view, inference entry or None)} from an explicit `feature_maps`
    output and/or ``inf["maps"]``; an explicit map wins."""
    out = {}
    for kk, e in ((inf or {}).get("maps") or {}).items():
        out[kk] = (e["map"], e)
    for kk, mv in (maps or {}).items():
        out[kk] = (mv, None)
    return out


def feature_table(res: Dict[str, Any], inf: Optional[Dict[str, Any]] = None,
                  view_names=None, feature_names=None, maps: Optional[Dict[str, Any]] = None):
    """One row per (view, feature, component): the interpretable quantities of a fit.

    Columns: ``view``, ``view_name``, ``feature``, ``feature_name``,
    ``component`` (1-based), ``structure`` (corr(x, u R)), ``loading``, and,
    with a `pysimlr.loading_inference` result ``inf``: ``ci_low``, ``ci_high``,
    ``excludes_zero``, ``selected`` (stability selection), ``selection_frequency``
    and ``component_stability``. Alternative feature maps (``maps``, from
    `pysimlr.feature_maps`, and any in ``inf["maps"]`` from
    ``loading_inference(maps=...)``) add a column named by kind and, with
    inference, ``<kind>_ci_low``, ``<kind>_ci_high``,
    ``<kind>_selection_frequency`` and ``<kind>_stability``. Sort, filter or
    group it however the study needs; every feature plot here draws from it.
    """
    import pandas as pd
    F = _aligned_frame(res)
    vn = _view_names(view_names, len(F["structure"]))
    extra = _maps_of(inf, maps)
    rows = []
    for m, C in enumerate(F["structure"]):
        p, k = C.shape
        probe_sizes = {int(_np(mv[m]).shape[0]) for mv, _ in extra.values() if int(_np(mv[m]).shape[0]) != p}
        if len(probe_sizes) > 1:
            raise ValueError(f"view {m} has maps of inconsistent feature counts {sorted(probe_sizes)}.")
        probe_p = next(iter(probe_sizes), None)
        if probe_p is None:
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
                for kk, (mv, e) in extra.items():
                    if kk == "structure" and "structure" in d:
                        kk = "structure_other_views"
                    d[kk] = _np(mv[m])[:, j].astype(float)
                    if e is not None:
                        d[f"{kk}_ci_low"] = _np(e["ci_low"][m])[:, j].astype(float)
                        d[f"{kk}_ci_high"] = _np(e["ci_high"][m])[:, j].astype(float)
                        d[f"{kk}_selection_frequency"] = _np(e["selection_frequency"][m]).astype(float)
                        d[f"{kk}_stability"] = float(e["component_stability"][m][j])
                rows.append(pd.DataFrame(d))
        else:
            names = _view_feature_names(feature_names, m, probe_p)
            for j in range(k):
                d = {"view": m, "view_name": vn[m], "feature": np.arange(probe_p), "feature_name": names,
                     "component": j + 1, "loading": np.full(probe_p, np.nan)}
                for kk, (mv, e) in extra.items():
                    if int(_np(mv[m]).shape[0]) != probe_p:
                        continue
                    d[kk] = _np(mv[m])[:, j].astype(float)
                    if e is not None:
                        lo = _np(e["ci_low"][m])[:, j].astype(float)
                        hi = _np(e["ci_high"][m])[:, j].astype(float)
                        freq = _np(e["selection_frequency"][m]).astype(float)
                        stab = float(e["component_stability"][m][j])
                        if kk == "structure":
                            d.update(ci_low=lo, ci_high=hi, excludes_zero=(lo > 0) | (hi < 0),
                                     selected=freq >= 0.8, selection_frequency=freq, component_stability=stab)
                        else:
                            d[f"{kk}_ci_low"] = lo
                            d[f"{kk}_ci_high"] = hi
                            d[f"{kk}_selection_frequency"] = freq
                            d[f"{kk}_stability"] = stab
                if "structure" not in d:
                    d["structure"] = np.full(probe_p, np.nan)
                rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- optimisation health
def _convergence_title(res):
    conv = res.get("convergence", {}) or {}
    unit = "epoch" if res.get("energy") is None else "sweep"
    n = conv.get("n_steps"); reason = conv.get("stop_reason", "?")
    sub = f"{n}/{conv.get('max_steps', '?')} {unit}s"
    if reason != "converged":
        sub = f"{reason}, " + sub
    gm = conv.get("grad_map")
    if gm is not None and np.isfinite(float(gm)):
        sub += f" · grad map {float(gm):.2g}"
    return ("NOT converged · " if _status(res) else "Converged · ") + sub


def _draw_curve(ax, h, color, label=None, best=None, unit="sweep", ylabel="objective"):
    x = np.arange(1, len(h) + 1)
    ax.plot(x, h, color=color, lw=2.0, label=label, zorder=2)
    lo = np.nanmin(h); hi = np.nanmax(h); pad = 0.06 * (hi - lo or 1.0)
    ax.fill_between(x, h, lo - pad, color=color, alpha=0.09, lw=0, zorder=1)
    ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(0.5, len(h) + 0.5)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    if best is not None and 0 <= best < len(h):
        _dots(ax, [best + 1], [h[best]], _INK, s=60, zorder=4)
        ax.annotate("returned", (best + 1, h[best]), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=_INK2)
    _clean(ax, grid="y"); ax.set_xlabel(unit); ax.set_ylabel(ylabel)


def _draw_convergence(res, ax):
    conv = res.get("convergence", {}) or {}
    bad = bool(_status(res))
    if res.get("energy") is not None:
        h = np.asarray([float(_np(v)) for v in res["energy"]])
        _draw_curve(ax, h, VIEW_COLORS[0], best=res.get("best_iteration"))
    elif res.get("loss_history") is not None:
        h = np.asarray(res["loss_history"], float)
        _draw_curve(ax, h, VIEW_COLORS[0], best=conv.get("best_epoch"), unit="epoch", ylabel="total loss")
    else:
        ax.axis("off")
        rows = [(k, conv.get(k)) for k in ("stop_reason", "converged", "n_steps", "certificate", "grad_map") if k in conv]
        ax.text(0.0, 0.92, f"{res.get('method', '')}: no iteration history", fontsize=11, weight="semibold",
                va="top", transform=ax.transAxes)
        for i, (k, v) in enumerate(rows):
            ax.text(0.0, 0.74 - 0.13 * i, k, fontsize=9, color=_MUTED, va="top", transform=ax.transAxes)
            ax.text(0.42, 0.74 - 0.13 * i, str(v), fontsize=9, color=_INK, va="top", transform=ax.transAxes)
        ax.set_title("Closed form", fontsize=10.5)
        return
    ax.set_title(_convergence_title(res), fontsize=10, color=_INK2)


@_themed
def plot_convergence(res: Dict[str, Any], ax=None):
    """Objective history of any fit, titled with stop reason and budget.

    Linear: energy per sweep, the returned best iterate ringed. Deep/flow: total
    loss, with the similarity and reconstruction terms as two small panels below
    (separate scales, never a second axis). Closed-form baselines: a summary
    card. An unconverged fit is titled "NOT converged" in the critical colour.
    """
    has_terms = (res.get("energy") is None and res.get("loss_history") is not None
                 and any(res.get(k) is not None and len(res[k]) for k in ("sim_history", "recon_history")))
    if ax is not None or not has_terms:
        fig, ax = _figure(ax, (6.4, 3.8), status=_status_line(res))
        _draw_convergence(res, ax)
        return fig
    fig = _new((6.4, 5.6), status=_status_line(res))
    gs = fig.add_gridspec(3, 1, height_ratios=(2.2, 1, 1))
    top = fig.add_subplot(gs[0]); _draw_convergence(res, top); top.set_xlabel("")
    for i, (key, name, c) in enumerate((("sim_history", "similarity term", VIEW_COLORS[1]),
                                         ("recon_history", "reconstruction term", VIEW_COLORS[2]))):
        ax = fig.add_subplot(gs[1 + i], sharex=top)
        if res.get(key) is not None and len(res[key]):
            _draw_curve(ax, np.asarray(res[key], float), c, unit="epoch" if i else "", ylabel=name)
        if not i:
            ax.set_xlabel("")
    return fig


def _draw_effective_rank(res, ax, view_names=None):
    er = np.asarray(res.get("effective_rank", []), float); nr = np.asarray(res.get("numerical_rank", []), float)
    k = res["structure"][0].shape[1]
    vn = _view_names(view_names, len(er)); y = np.arange(len(er))
    if len(nr):
        ax.hlines(y, 0, nr, color=_GRID, lw=6, zorder=1)
    for m in range(len(er)):
        bad = er[m] < k - 0.5
        ax.hlines(m, 0, er[m], color=_CRITICAL if bad else _view_color(m), lw=2.5, zorder=2)
        _dots(ax, [er[m]], [m], _CRITICAL if bad else _view_color(m), s=60)
        ax.text(er[m] + 0.06 * k, m, f"{er[m]:.2f}", va="center", fontsize=8.5, color=_INK2)
    ax.axvline(k, color=_INK2, lw=1)
    ax.text(k, -0.6, f" k = {k}", ha="left", va="bottom", fontsize=8, color=_INK2)
    ax.set_yticks(y); ax.set_yticklabels(vn); ax.set_ylim(len(er) - 0.4, -0.9)
    ax.set_xlim(0, k * 1.25); ax.set_xlabel("effective rank (bar: numerical rank)")
    _clean(ax, grid="x")
    ax.set_title("Basis rank", fontsize=10.5)


@_themed
def plot_basis_diagnostics(res: Dict[str, Any], view_names=None, figsize=None):
    """Basis health per view: rank against k, column overlap, conditioning, and
    sparsity against the requested level (with the loading margin).

    Predictive R^2 does not reveal merged basis columns; an effective rank
    below k - 0.5 is drawn in the critical colour. These panels read the
    encoder V because they diagnose the optimisation, not because V is
    interpretable.
    """
    M = len(res["v"]); vn = _view_names(view_names, M); y = np.arange(M)
    fig = _new(figsize or (13.0, 0.42 * M + 2.6), "Basis diagnostics",
               "Is each view's basis full rank, well separated and as sparse as asked?", _status_line(res))
    axes = fig.subplots(1, 4, gridspec_kw={"width_ratios": (1.25, 1, 1, 1.15)})
    _draw_effective_rank(res, axes[0], view_names)
    ov = np.asarray(res.get("max_column_overlap", [np.nan] * M), float)
    cn = np.log10(np.asarray(res.get("condition_number", [np.nan] * M), float))
    sp = [float((np.abs(_np(v)) < 1e-4).mean()) for v in res["v"]]
    for ax, vals, lim, label, title, fmt in ((axes[1], ov, (0, 1), "max |cos| between columns", "Column overlap", "{:.2f}"),
                                             (axes[2], cn, (0, max(2.0, np.nanmax(cn) * 1.2 if np.isfinite(cn).any() else 2)),
                                              "log10 condition number", "Conditioning", "{:.1f}"),
                                             (axes[3], np.asarray(sp), (0, 1), "fraction of zero entries", "Sparsity", "{:.2f}")):
        for m in range(M):
            ax.hlines(m, lim[0], vals[m], color=_GRID, lw=2.5, zorder=1)
            _dots(ax, [vals[m]], [m], _view_color(m), s=60)
            ax.text(vals[m], m - 0.32, fmt.format(vals[m]), ha="center", fontsize=8, color=_INK2)
        ax.set_xlim(*lim); ax.set_ylim(M - 0.4, -0.9); ax.set_yticks(y); ax.set_yticklabels([])
        ax.set_xlabel(label); ax.set_title(title, fontsize=10.5); _clean(ax, grid="x")
    req = res.get("sparseness")
    if isinstance(req, (int, float)):
        axes[3].axvline(float(req), color=_INK2, lw=1)
        axes[3].text(float(req), -0.6, f"requested {req:g}", ha="center", va="bottom", fontsize=8, color=_INK2)
    mg = res.get("loading_margin")
    if mg is not None:
        for m in range(M):
            if mg[m] is not None:
                axes[3].text(1.0, m + 0.3, f"margin {float(np.median(_np(mg[m]))):.2f}", ha="right",
                             fontsize=7.5, color=_MUTED)
    return fig


@_themed
def plot_rank_selection(rank_result: Dict[str, Any], ax=None, bars: Sequence[float] = STABILITY_BARS,
                        criterion: Optional[str] = None):
    """Rank-selection diagnostic for any `pysimlr.select_rank` criterion.

    ``criterion`` defaults to ``rank_result["criterion"]`` (``"component"`` if
    absent): component stability, Grassmannian subspace cosines
    (``"subspace"``), out-of-bag held-out R^2 +- SE (``"oos_r2"``,
    ``"oos_r2_1se"``), or out-of-bag cross-view coupling (``"oos_coupling"``).
    Candidates with ``basis_degenerate=True`` have their tick in the critical colour.
    """
    from .loadings import RANK_CRITERIA
    crit = criterion or rank_result.get("criterion", "component")
    if crit not in RANK_CRITERIA:
        raise ValueError(f"criterion must be one of {RANK_CRITERIA}; got {crit!r}.")
    table = rank_result["table"]
    chosen = (rank_result.get("criteria") or {}).get(crit, rank_result.get("k") if criterion is None else None)
    names = {"oos_r2": "held-out reconstruction R²", "oos_r2_1se": "held-out R², one-SE rule",
             "oos_coupling": "held-out view coupling", "subspace": "subspace stability",
             "component": "component stability"}
    fig, ax = _figure(ax, (6.4, 4.6), f"Rank selection by {names.get(crit, crit)}",
                      f"Selected k = {chosen}." if chosen is not None else
                      "No candidate k passes this criterion; read the spread before choosing.")
    ks = [r["k"] for r in table]

    def spread(vals_per_k, colors=None):
        mins, meds = [], []
        for i, (k, s) in enumerate(zip(ks, vals_per_k)):
            if s is None:
                mins.append(np.nan); meds.append(np.nan); continue
            s = np.asarray(s, float)
            mins.append(s.min()); meds.append(np.median(s))
            ax.vlines(k, s.min(), s.max(), color=_GRID, lw=7, zorder=1)
            cols = colors[i] if colors is not None else [VIEW_COLORS[0]] * len(s)
            ax.scatter(np.full(len(s), k), s, s=30, c=cols, edgecolor=_SURFACE, linewidth=1.0, zorder=3)
        return np.asarray(mins), np.asarray(meds)

    def mark(yv):
        if chosen is not None and chosen in ks:
            yy = yv[ks.index(chosen)]
            ax.scatter([chosen], [yy], s=260, facecolor="none", edgecolor=_INK, linewidth=1.6, zorder=5)
            ax.annotate(f"k = {chosen}", (chosen, yy), xytext=(12, 10), textcoords="offset points",
                        fontsize=9, weight="semibold", color=_INK)

    if crit in ("oos_r2", "oos_r2_1se"):
        ys = np.asarray([np.nan if r.get("oos_r2") is None else r["oos_r2"] for r in table], float)
        se = np.asarray([0.0 if r.get("oos_r2_se") is None else r["oos_r2_se"] for r in table], float)
        ax.fill_between(ks, ys - se, ys + se, color=VIEW_COLORS[0], alpha=0.12, lw=0)
        ax.plot(ks, ys, color=VIEW_COLORS[0], lw=2); _dots(ax, ks, ys, VIEW_COLORS[0], s=50)
        ax.axhline(0.0, color=_AXIS, lw=0.8)
        if np.any(np.isfinite(ys)):
            bi = int(np.nanargmax(ys))
            if ys[bi] > 0:
                ax.axhline(ys[bi] - se[bi], color=_INK2, lw=1)
                ax.text(ks[0], ys[bi] - se[bi], " peak − 1 SE", va="bottom", fontsize=8, color=_INK2)
        mark(ys); ax.set_ylabel("held-out reconstruction R²")
    elif crit == "oos_coupling":
        mins, meds = spread([r.get("oos_coupling") for r in table])
        ax.plot(ks, mins, color=_INK, lw=1.8, zorder=2, label="weakest component")
        ax.plot(ks, meds, color=_MUTED, lw=1.2, zorder=2, label="median")
        thr = next((r.get("oos_coupling_threshold") for r in table if r.get("oos_coupling_threshold") is not None), None)
        ax.axhline(0.0, color=_AXIS, lw=0.8)
        if thr is not None:
            ax.axhspan(-1, thr, color=_GRID, alpha=0.5, lw=0, zorder=0)
            ax.text(ks[-1], thr, f"null bar {thr:.2f} ", ha="right", va="bottom", fontsize=8, color=_INK2)
        mark(mins); ax.set_ylabel("held-out view coupling (r)")
        ax.set_ylim(min(0, np.nanmin(mins) - 0.05) if np.isfinite(mins).any() else 0, 1.02)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    else:
        key = "subspace_stability" if crit == "subspace" else "component_stability"
        vals = [r[key] for r in table]
        mins, meds = spread(vals, [[_stability_color(v, bars) if v < bars[0] else VIEW_COLORS[0] for v in s]
                                   for s in vals])
        ax.plot(ks, mins, color=_INK, lw=1.8, zorder=2,
                label="min principal cosine" if crit == "subspace" else "weakest component")
        ax.plot(ks, meds, color=_MUTED, lw=1.2, zorder=2, label="median")
        if crit == "component" and all("min_subspace_stability" in r for r in table):
            ax.plot(ks, [r["min_subspace_stability"] for r in table], color=VIEW_COLORS[6], lw=1.2,
                    marker="s", ms=4, label="subspace min cosine")
        for b in bars:
            ax.axhline(b, color=_INK2 if b == bars[0] else _MUTED, lw=0.9, zorder=0)
            ax.text(ks[-1] + 0.35, b, f"{b:g}", va="center", fontsize=8, color=_INK2)
        mark(mins); ax.set_ylim(0, 1.04)
        ax.set_ylabel("subspace principal cosine" if crit == "subspace" else "component stability")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    _clean(ax, grid="y")
    ax.set_xticks(ks); ax.set_xlabel("k (number of components)"); ax.set_xlim(min(ks) - 0.5, max(ks) + 0.6)
    for lbl, r in zip(ax.get_xticklabels(), table):
        if r.get("basis_degenerate"):
            lbl.set_color(_CRITICAL)
    return fig


# ----------------------------------------------------------------------------- coupling, weights, flows
@_themed
def plot_latent_coupling(res: Dict[str, Any], pred: Optional[Dict[str, Any]] = None, view_names=None,
                         component: Optional[int] = None, figsize=None):
    """How strongly each view's scores follow the consensus, per rotated component.

    Left: one row per view and one column per component; each cell is
    corr(view m score_j, consensus_j) on the training data (hollow ring) and,
    with ``pred = pysimlr.predict(res, held_out)``, on held-out subjects (filled
    dot, value printed). In p >> N studies training coupling is optimistic;
    held-out coupling separates shared signal from memorisation. Right: the
    view-by-view correlation of one component (default: the most strongly
    coupled). Joint-basis methods (pca, nsa_pipeline) have no per-view scores;
    the figure says so.
    """
    F = _aligned_frame(res)
    vn = _view_names(view_names, len(F["structure"]))
    if F["joint"]:
        fig = _new(figsize or (6.0, 1.8), "Latent coupling")
        return _joint_card(fig, fig.add_subplot(), res, "per-view coupling")
    tr = _coupling(res); te = _coupling(res, pred) if pred is not None else None
    M, k = tr.shape
    j = int(np.nanargmax(np.nanmean(tr, 0))) if component is None else int(component)
    fig = _new(figsize or (2.0 * k + 0.7 * M + 5.4, 0.55 * M + 2.5), "Latent coupling",
               "Each view's scores against the consensus, per component"
               + (": ring = training, dot = held-out." if te is not None else "."), _status_line(res))
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": (2.0 * k, 0.75 * M + 1.4)})
    ax = axes[0]
    show = te if te is not None else tr
    lo = float(np.floor(min(0.0, np.nanmin(np.r_[tr.ravel(), show.ravel()])) * 10) / 10)
    for c in range(k):
        x0 = c * 1.0
        ax.axvspan(x0 + 0.04, x0 + 0.96, color=_GRID, alpha=0.35, lw=0, zorder=0)
        for m in range(M):
            sx = lambda r: x0 + 0.08 + 0.84 * (r - lo) / (1 - lo)
            if te is not None:
                ax.plot([sx(tr[m, c]), sx(te[m, c])], [m, m], color=_view_color(m), lw=1.5, alpha=0.6, zorder=2)
                _dots(ax, [sx(tr[m, c])], [m], _view_color(m), s=44, hollow=True)
            _dots(ax, [sx(show[m, c])], [m], _view_color(m), s=54, zorder=4)
            ax.text(sx(show[m, c]), m - 0.3, f"{show[m, c]:.2f}", ha="center", va="bottom", fontsize=8,
                    color=_INK2)
        ax.text(x0 + 0.5, M - 0.35, f"C{c + 1}", ha="center", va="top", fontsize=9.5, weight="semibold")
        ax.text(x0 + 0.08, M - 0.62, f"{lo:g}", ha="center", va="top", fontsize=7, color=_MUTED)
        ax.text(x0 + 0.92, M - 0.62, "1", ha="center", va="top", fontsize=7, color=_MUTED)
    ax.set_xlim(0, k); ax.set_ylim(M - 0.2, -0.7)
    ax.set_yticks(range(M)); ax.set_yticklabels(vn); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("view vs consensus (r)", fontsize=10.5)
    P = _pairwise(res, pred)[j]
    mesh = _matrix(axes[1], P, _cmap("seq"), 0, 1, rows=vn, cols=vn)
    axes[1].set_xticklabels(vn, rotation=40, ha="right")
    axes[1].set_title(f"view vs view, C{j + 1}" + (" (held-out)" if pred is not None else ""), fontsize=10.5)
    fig._pysimlr_values = {"training": tr, "held_out": te, "pairwise_component": j, "pairwise": P}
    return fig


def _arc_points(a0, a1, r, n=48):
    t = np.linspace(a0, a1, max(2, int(n * abs(a1 - a0) / np.pi) + 2))
    return np.c_[r * np.cos(t), r * np.sin(t)]


def _ribbon(ax, a, b, r, color_a, color_b, alpha=0.62, z=1):
    """A chord ribbon between arc spans ``a=(a0, a1)`` and ``b=(b0, b1)`` at
    radius ``r``, filled with a gradient from ``color_a`` to ``color_b``."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    A = _arc_points(a[0], a[1], r); B = _arc_points(b[0], b[1], r)
    verts = [tuple(A[0])] + [tuple(p) for p in A[1:]]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(A) - 1)
    verts += [(0.0, 0.0), tuple(B[0])]; codes += [Path.CURVE3, Path.CURVE3]
    verts += [tuple(p) for p in B[1:]]; codes += [Path.LINETO] * (len(B) - 1)
    verts += [(0.0, 0.0), tuple(A[0])]; codes += [Path.CURVE3, Path.CURVE3]
    patch = PathPatch(Path(verts, codes), facecolor="none", edgecolor="none", zorder=z)
    ax.add_patch(patch)
    ma, mb = np.mean(a), np.mean(b)
    pa, pb = np.array([np.cos(ma), np.sin(ma)]), np.array([np.cos(mb), np.sin(mb)])
    g = np.linspace(0, 1, 200)
    yy, xx = np.mgrid[1.05:-1.05:200j, -1.05:1.05:200j]
    d = pb - pa; t = np.clip(((xx - pa[0]) * d[0] + (yy - pa[1]) * d[1]) / (d @ d + 1e-12), 0, 1)
    ca, cb = np.array(_rgba(color_a)[:3]), np.array(_rgba(color_b)[:3])
    img = (1 - t)[..., None] * ca + t[..., None] * cb
    img = np.concatenate([img, np.full(t.shape + (1,), alpha)], -1)
    im = ax.imshow(img, extent=(-1.05, 1.05, -1.05, 1.05), interpolation="bilinear", zorder=z)
    im.set_clip_path(patch)
    del g
    return patch


def _chord_axes(ax):
    ax.set_xlim(-1.42, 1.42); ax.set_ylim(-1.32, 1.32); ax.set_aspect("equal"); ax.axis("off")


def _draw_view_chord(ax, W, names, colors, center=None, value_fmt="{:.2f}", label_values=True, gap=0.06,
                     label_min=0.25):
    """Views as arcs; every pair of views has a fixed slot at each end, and its
    ribbon fills that slot in proportion to r (clipped to [0, 1]). Weak
    coupling therefore shows as empty arc, never as a stretched ribbon."""
    from matplotlib.patches import Wedge
    M = W.shape[0]
    W = np.where(np.eye(M, dtype=bool), 0.0, np.clip(np.nan_to_num(W), 0, 1))
    span = 2 * np.pi / M - gap
    starts = [np.pi / 2 - gap / 2 - i * (span + gap) for i in range(M)]   # clockwise from the top
    for i in range(M):
        a1, a0 = starts[i], starts[i] - span
        ax.add_patch(Wedge((0, 0), 1.06, np.degrees(a0), np.degrees(a1), width=0.07, facecolor=colors[i],
                           edgecolor=_SURFACE, lw=1.5, zorder=3))
        mid = (a0 + a1) / 2; c, s_ = np.cos(mid), np.sin(mid)
        ax.text(1.17 * c, 1.17 * s_, names[i], ha="left" if c > 0.15 else ("right" if c < -0.15 else "center"),
                va="bottom" if s_ > 0.15 else ("top" if s_ < -0.15 else "center"), fontsize=10, weight="semibold",
                color=_INK)
    slot = span / max(M - 1, 1)
    seg = {}
    for i in range(M):
        order = sorted((j for j in range(M) if j != i), key=lambda j: (j - i) % M)
        for q, j in enumerate(order):
            hi_ = starts[i] - q * slot; mid = hi_ - slot / 2; half = 0.46 * slot * W[i, j]
            seg[(i, j)] = (mid - half, mid + half)
    pairs = sorted(((W[i, j], i, j) for i in range(M) for j in range(i + 1, M)))
    for rank, (w, i, j) in enumerate(pairs):
        if w <= 0.02:
            continue
        _ribbon(ax, seg[(i, j)], seg[(j, i)], 0.985, colors[i], colors[j], alpha=0.25 + 0.5 * w, z=1 + rank * 0.01)
    if label_values:
        shown = [p for p in pairs[::-1] if p[0] >= label_min][: (None if M <= 4 else 5)]
        for w, i, j in shown:
            ma, mb = np.mean(seg[(i, j)]), np.mean(seg[(j, i)])
            A = 0.985 * np.array([np.cos(ma), np.sin(ma)]); B = 0.985 * np.array([np.cos(mb), np.sin(mb)])
            t = 0.22; p = (1 - t) ** 2 * A + t ** 2 * B
            ax.text(p[0], p[1], value_fmt.format(w), ha="center", va="center", fontsize=8.5, color=_INK,
                    bbox=dict(boxstyle="round,pad=0.22", fc=_SURFACE, ec="none", alpha=0.9), zorder=6)
    if center:
        ax.text(0, -1.27, center, ha="center", va="center", fontsize=10.5, weight="semibold", color=_INK)
    _chord_axes(ax)


@_themed
def plot_view_chord(res: Dict[str, Any], pred: Optional[Dict[str, Any]] = None, view_names=None,
                    components: Optional[Sequence[int]] = None, figsize=None):
    """Which views share which component: one chord diagram per component.

    Each view is an arc with one fixed slot per partner view. A ribbon joins
    every pair; it fills its slots in proportion to r = corr(view a score_j,
    view b score_j) (held-out with ``pred``) and is more opaque as r grows, so
    a component carried by only some views shows as a few full ribbons and
    empty arc elsewhere. Negative r draws nothing. Pairs with r >= 0.25 are
    labelled (the five strongest beyond four views).
    """
    F = _aligned_frame(res)
    if F["joint"]:
        fig = _new(figsize or (6.0, 1.8), "View chords")
        return _joint_card(fig, fig.add_subplot(), res, "a per-view chord")
    P = _pairwise(res, pred); k, M = P.shape[0], P.shape[1]
    comps = list(range(k)) if components is None else list(components)
    vn = _view_names(view_names, M); cols = [_view_color(m) for m in range(M)]
    fig = _new(figsize or (3.9 * len(comps), 4.6), "Which views share each component",
               "Each pair of views has a slot on both arcs; the ribbon fills it in proportion to r between their scores"
               + (" on held-out subjects." if pred is not None else "."), _status_line(res))
    axes = np.atleast_1d(fig.subplots(1, len(comps)))
    for ax, j in zip(axes, comps):
        _draw_view_chord(ax, P[j], vn, cols, center=f"C{j + 1}")
    fig._pysimlr_values = {"pairwise": P}
    return fig


def _radar_axes(fig, spec, labels, rmax=1.0, rings=(0.25, 0.5, 0.75, 1.0)):
    ax = fig.add_subplot(spec, projection="polar")
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax); ax.set_yticks([]); ax.set_xticks([])
    ax.spines["polar"].set_visible(False); ax.grid(False); ax.set_facecolor(_SURFACE)
    t = np.linspace(0, 2 * np.pi, 200)
    for r in rings:
        ax.plot(t, np.full_like(t, r * rmax), color=_GRID, lw=0.8, zorder=0)
        ax.text(np.pi / n, r * rmax, f"{r * rmax:g}", fontsize=7, color=_MUTED, ha="center", va="center", zorder=1)
    for a, lab in zip(ang, labels):
        ax.plot([a, a], [0, rmax], color=_GRID, lw=0.8, zorder=0)
        ax.text(a, rmax * 1.16, lab, ha="center", va="center", fontsize=8.5, color=_INK2, linespacing=1.0)
    return ax, ang


def _radar_poly(ax, ang, vals, color, label=None, fill=0.12, lw=2.0, dots=True):
    v = np.clip(np.nan_to_num(np.asarray(vals, float)), 0, None)
    a = np.r_[ang, ang[:1]]; vv = np.r_[v, v[:1]]
    ax.fill(a, vv, color=color, alpha=fill, lw=0, zorder=2)
    ax.plot(a, vv, color=color, lw=lw, zorder=3, label=label)
    if dots:
        _dots(ax, ang, v, color, s=34, zorder=4)


@_themed
def plot_component_radar(res: Dict[str, Any], pred: Optional[Dict[str, Any]] = None, view_names=None,
                         figsize=None):
    """Each view's coupling profile across components, one radar per view.

    Spokes are components; the radius is corr(view score_j, consensus_j)
    (held-out with ``pred``; negative r is drawn at 0). A view carried by every
    component fills the radar; a sub-coupled view is a spike. The faint grey
    polygon in every panel is the mean over views, for comparison.
    """
    F = _aligned_frame(res)
    if F["joint"]:
        fig = _new(figsize or (6.0, 1.8), "Component radar")
        return _joint_card(fig, fig.add_subplot(), res, "a per-view coupling profile")
    C = _coupling(res, pred); M, k = C.shape
    vn = _view_names(view_names, M)
    ncol = 2 if M == 4 else min(M, 3); nrow = int(np.ceil(M / ncol))
    fig = _new(figsize or (3.9 * ncol + 0.4, 3.9 * nrow + 0.8), "Coupling profile of each view",
               "Radius: r between the view's scores and the consensus on each component"
               + (", held-out." if pred is not None else "."), _status_line(res))
    gs = fig.add_gridspec(nrow, ncol)
    mean = np.nanmean(np.clip(C, 0, None), 0)
    for m in range(M):
        ax, ang = _radar_axes(fig, gs[m // ncol, m % ncol], [f"C{j + 1}" for j in range(k)] if k > 2 else
                              [f"C{j + 1}" for j in range(k)] + [""] * (3 - k))
        n = len(ang)
        pad = lambda v: np.r_[v, np.zeros(n - len(v))]
        _radar_poly(ax, ang, pad(mean), _AXIS, fill=0.0, lw=1.0, dots=False)
        _radar_poly(ax, ang, pad(C[m]), _view_color(m))
        ax.set_title(vn[m], fontsize=11, pad=30, loc="center", weight="semibold")
    fig._pysimlr_values = {"coupling": C}
    return fig


@_themed
def plot_view_radar(res: Dict[str, Any], views=None, pred: Optional[Dict[str, Any]] = None, view_names=None,
                    figsize=None):
    """A fingerprint of every view on the quantities that decide whether to trust it.

    Spokes (each on 0-1): mean coupling with the consensus (training, and
    held-out with ``pred``); reconstruction R^2 (training with ``views``,
    held-out with ``pred``); effective rank / k; sparsity; and, when the fit
    has them, the view weight relative to uniform (M w, capped at 1). Up to
    three views are overlaid; more are drawn as small multiples.
    """
    F = _aligned_frame(res)
    M = len(F["structure"]); k = F["structure"][0].shape[1]; vn = _view_names(view_names, M)
    spokes, vals = [], []
    if not F["joint"]:
        spokes.append("coupling\ntraining"); vals.append(np.nanmean(np.clip(_coupling(res), 0, None), 1))
        if pred is not None:
            spokes.append("coupling\nheld-out"); vals.append(np.nanmean(np.clip(_coupling(res, pred), 0, None), 1))
    if views is not None:
        from .api import predict
        spokes.append("R²\ntraining"); vals.append(np.clip([float(_np(v)) for v in predict(res, views)["r2"]], 0, 1))
    if pred is not None and pred.get("r2") is not None:
        spokes.append("R²\nheld-out"); vals.append(np.clip([float(_np(v)) for v in pred["r2"]], 0, 1))
    if res.get("effective_rank") is not None:
        spokes.append("rank / k"); vals.append(np.clip(np.asarray(res["effective_rank"], float) / k, 0, 1))
    if res.get("v") is not None:
        spokes.append("sparsity"); vals.append(np.array([float((np.abs(_np(v)) < 1e-4).mean()) for v in res["v"]]))
    mw = res.get("modality_weights")
    if mw is not None and np.ptp(np.asarray(mw, float)) > 1e-6:       # uniform weights carry nothing
        spokes.append("weight × M"); vals.append(np.clip(np.asarray(res["modality_weights"], float) * M, 0, 1))
    if len(spokes) < 3:
        raise ValueError("plot_view_radar needs at least three quantities; pass views= and/or pred=.")
    V = np.stack(vals, 1)
    overlay = M <= 3
    ncol = 1 if overlay else min(M, 2); nrow = 1 if overlay else int(np.ceil(M / ncol))
    fig = _new(figsize or ((5.8, 5.4) if overlay else (4.4 * ncol + 0.4, 4.3 * nrow + 0.8)), "View fingerprints",
               "Every spoke runs 0 to 1, larger is better except sparsity, which is read against the request.",
               _status_line(res))
    gs = fig.add_gridspec(nrow, ncol)
    if overlay:
        ax, ang = _radar_axes(fig, gs[0, 0], spokes)
        for m in range(M):
            _radar_poly(ax, ang, V[m], _view_color(m), label=vn[m])
        ax.legend(handles=_legend_handles([(vn[m], _view_color(m)) for m in range(M)]),
                  loc="upper left", bbox_to_anchor=(-0.18, 1.12), ncol=1)
    else:
        for m in range(M):
            ax, ang = _radar_axes(fig, gs[m // ncol, m % ncol], spokes)
            _radar_poly(ax, ang, V.mean(0), _AXIS, fill=0.0, lw=1.0, dots=False)
            _radar_poly(ax, ang, V[m], _view_color(m))
            ax.set_title(vn[m], fontsize=11, pad=34, loc="center", weight="semibold")
    fig._pysimlr_values = {"spokes": spokes, "values": V}
    return fig


@_themed
def plot_modality_weights(res: Dict[str, Any], view_names=None, figsize=(10.0, 3.6)):
    """Final view weights against uniform 1/M with the modality alignment index,
    and (``flow_v``) their trajectories over training, each directly labelled.

    A small view with low weight was down-weighted by the alignment index; check
    it is not a view that matters for the study.
    """
    w = res.get("modality_weights"); mai = res.get("mai")
    if w is None:
        raise ValueError(f"{res.get('method')} result has no modality_weights.")
    w = np.asarray(w, float); M = len(w); vn = _view_names(view_names, M); y = np.arange(M)
    hist = res.get("weight_history") is not None and len(res["weight_history"]) > 1
    fig = _new(figsize if hist else (6.0, 0.5 * M + 2.4), "Modality weights",
               "Final weight of each view against the uniform 1/M; the alignment index (MAI) that set it.",
               _status_line(res))
    axes = np.atleast_1d(fig.subplots(1, 2 if hist else 1, gridspec_kw={"width_ratios": (1, 1.3)} if hist else None))
    ax = axes[0]
    for m in range(M):
        ax.hlines(m, 1.0 / M, w[m], color=_view_color(m), lw=2.5, zorder=2)
        _dots(ax, [w[m]], [m], _view_color(m), s=64)
        ax.text(w[m], m - 0.28, f"{w[m]:.3f}", ha="center", va="bottom", fontsize=8, color=_INK2)
        if mai is not None and np.isfinite(mai[m]):
            ax.text(1.0, m, f"MAI {float(mai[m]):.2f}", ha="right", va="center", fontsize=8, color=_MUTED,
                    transform=ax.get_yaxis_transform())
    ax.axvline(1.0 / M, color=_INK2, lw=1)
    ax.text(1.0 / M, -0.75, " 1/M", ha="left", va="bottom", fontsize=8, color=_INK2)
    ax.set_yticks(y); ax.set_yticklabels(vn); ax.set_ylim(M - 0.4, -0.9)
    ax.set_xlim(0, max(1.0 / M * 2.2, w.max() * 1.25)); ax.set_xlabel("weight")
    _clean(ax, grid="x"); ax.set_title("final", fontsize=10.5)
    if hist:
        W = np.asarray(res["weight_history"], float); A = np.asarray(res.get("mai_history"), float)
        a2 = axes[1]
        ly = _dodge(W[-1], 0.07 * (np.nanmax(W) - np.nanmin(W) + 1e-9) + 0.004)
        for m in range(M):
            a2.plot(W[:, m], color=_view_color(m), lw=2)
            if A.ndim == 2:
                a2.plot(A[:, m], color=_view_color(m), lw=1, alpha=0.45)
            a2.text(len(W) + 0.5, ly[m], f" {vn[m]}", va="center", fontsize=8.5, color=_INK2)
        a2.axhline(1.0 / M, color=_AXIS, lw=0.8)
        a2.set_xlabel("epoch"); a2.set_ylabel("weight (thin: MAI)"); _clean(a2, grid="y")
        a2.set_title("trajectory", fontsize=10.5); a2.set_xlim(0, len(W) * 1.18)
    return fig


@_themed
def plot_flow_warping(res: Dict[str, Any], views, components=(0, 1), view_names=None, figsize=None):
    """What the nonlinear model does after the linear first layer, per view and component.

    x: the rotated linear projection (X_m V_m R)_j; y: the rotated latent
    (Z_m R)_j the model actually uses, with a binned median trace. A straight
    line means the model is linear there (the identity for simlr); a monotone
    curve preserves the sign interpretation of V; a fold does not. Spearman rho
    is in each panel title.
    """
    from scipy.stats import spearmanr
    from .api import _preprocess_views
    F = _aligned_frame(res)
    if F["joint"]:
        raise ValueError(f"{res.get('method')} has one joint latent; there is no per-view warp.")
    xs, _ = _preprocess_views(views, res["scale_list"], res["provenance_list"])
    vn = _view_names(view_names, len(xs)); comps = list(components)
    fig = _new(figsize or (3.1 * len(comps) + 0.4, 2.7 * len(xs) + 0.9), f"First layer to latent ({res.get('method')})",
               "Straight: linear. Monotone curve: the sign of V still reads. A fold: it does not.",
               _status_line(res))
    axes = np.atleast_2d(fig.subplots(len(xs), len(comps), squeeze=False))
    for m, x in enumerate(xs):
        lin = (_np(x).astype(float) @ _np(res["v"][m]).astype(float)) @ F["R"]
        for c_i, j in enumerate(comps):
            ax = axes[m, c_i]; a, b = lin[:, j], F["latents"][m][:, j]
            ax.scatter(a, b, s=11, color=_view_color(m), alpha=0.35, linewidth=0, zorder=2)
            order = np.argsort(a); nb = max(5, min(12, len(a) // 25))
            bins = np.array_split(order, nb)
            bx = np.array([np.median(a[i]) for i in bins]); by = np.array([np.median(b[i]) for i in bins])
            ax.plot(bx, np.convolve(np.r_[by[0], by, by[-1]], np.ones(3) / 3, "valid"), color=_INK, lw=1.8, zorder=3)
            qa, qb = np.percentile(a, [1, 99]), np.percentile(b, [1, 99])
            ax.set_xlim(qa[0] - 0.08 * np.ptp(qa), qa[1] + 0.08 * np.ptp(qa))
            ax.set_ylim(qb[0] - 0.08 * np.ptp(qb), qb[1] + 0.08 * np.ptp(qb))
            rho = spearmanr(a, b).correlation
            ax.set_title(f"{vn[m]}, C{j + 1}: Spearman {rho:.2f}", fontsize=9.5)
            _clean(ax, grid="both")
            if m == len(xs) - 1:
                ax.set_xlabel("linear  X V R")
            if c_i == 0:
                ax.set_ylabel("latent  Z R")
    return fig


@_themed
def plot_view_correlations(views, names: Optional[Sequence[str]] = None, ax=None):
    """Adjusted RV coefficient between every pair of views (shared multivariate
    signal before any fit), as a chord diagram, or a gap-separated matrix when
    drawn into a given axes."""
    import torch
    from .utils import adjusted_rvcoef
    mats = [torch.as_tensor(np.asarray(_np(x), float)).float() for x in views]
    n = len(mats); names = list(names) if names is not None else [f"view {i + 1}" for i in range(n)]
    rv = np.array([[float(adjusted_rvcoef(mats[i], mats[j])) for j in range(n)] for i in range(n)])
    if ax is not None:
        fig = ax.figure
        mesh = _matrix(ax, rv, _cmap("seq"), 0, max(1.0, float(np.nanmax(rv))), rows=names, cols=names)
        ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_title("Inter-view adjusted RV")
        fig._pysimlr_values = {"rv": rv}
        return fig
    fig = _new((5.6, 5.6), "Shared signal between views",
               "Adjusted RV coefficient of each pair of views, before any fit.")
    _draw_view_chord(fig.add_subplot(), rv, names, [_view_color(m) for m in range(n)])
    fig._pysimlr_values = {"rv": rv}
    return fig


# ----------------------------------------------------------------------------- interpretable features
def _draw_top_features(ax, tab, m, component, kind, top_n, rank_by, null, bars, color):
    d = tab[(tab.view == m) & (tab.component == component + 1)].copy()
    is_struct = kind == "structure"
    lo_c, hi_c = ("ci_low", "ci_high") if is_struct else (f"{kind}_ci_low", f"{kind}_ci_high")
    r2_total = None
    if kind == "relative_weight":
        # Shares of R^2 read naturally as percent of the variance the view explains.
        r2_total = float(np.nansum(d[kind]))
        scale = 100.0 / r2_total if r2_total > 0 else np.nan
        for c_ in (kind, lo_c, hi_c):
            if c_ in d:
                d[c_] = d[c_] * scale
    val = d[kind].to_numpy()
    freq_c = "selection_frequency" if is_struct else f"{kind}_selection_frequency"
    stab_c = "component_stability" if is_struct else f"{kind}_stability"
    if rank_by == "selection" and freq_c in d:
        order = np.lexsort((-np.abs(val), -d[freq_c].to_numpy()))[:top_n]
    else:
        order = np.argsort(-np.nan_to_num(np.abs(val), nan=-1.0))[:top_n]
    d = d.iloc[order[::-1]]; y = np.arange(len(d))
    vmax = float(np.nanmax(np.abs(d[kind]))) if len(d) and np.isfinite(d[kind]).any() else 1.0
    lim = max(1.0 if is_struct else 0.0, vmax * 1.08)
    has_ci = lo_c in d and kind != "loading"
    if has_ci:
        lim = max(lim, float(np.nanmax(np.abs(np.r_[d[lo_c], d[hi_c]]))) * 1.05)
    if null is not None and is_struct:
        ax.axvspan(-null, null, color=_GRID, alpha=0.55, lw=0, zorder=0)
        for sgn in (-1, 1):
            ax.axvline(sgn * null, color=_MUTED, ls="--", lw=0.8, zorder=0)
        ax.text(null, len(d) - 0.35, " permutation null", fontsize=7.5, color=_MUTED, va="top")
    ax.axvline(0, color=_AXIS, lw=0.9, zorder=1)
    if has_ci:
        if kind == "relative_weight":
            # Non-negative shares: an interval "excluding 0" says nothing, so mark
            # stability selection (selected in >= 80% of refits) instead.
            strong = (d[freq_c] >= 0.8).to_numpy() if freq_c in d else np.zeros(len(d), bool)
        else:
            exc = ((d[lo_c] > 0) | (d[hi_c] < 0)).to_numpy()
            strong = (d.selected.to_numpy() | exc) if is_struct else exc
        ax.hlines(y, d[lo_c], d[hi_c], color=np.where(strong, color, _AXIS).tolist(), lw=2.2, zorder=2,
                  alpha=0.55)
        _dots(ax, d[kind][strong], y[strong], color, s=50)
        _dots(ax, d[kind][~strong], y[~strong], _MUTED, s=40, hollow=True)
    else:
        ax.hlines(y, 0, d[kind], color=color, lw=2.0, alpha=0.5, zorder=2)
        _dots(ax, d[kind], y, color, s=50)
    ax.set_yticks(y); ax.set_yticklabels(d.feature_name)
    if is_struct and "selected" in d:
        for lbl, s_ in zip(ax.get_yticklabels(), d.selected):
            if s_:
                lbl.set_fontweight("bold"); lbl.set_color(_INK)
    nonneg = kind == "relative_weight"
    ax.set_xlim(0 if nonneg else -lim, lim); ax.set_ylim(-0.7, len(d) - 0.3)
    title = f"{d.view_name.iloc[0]}"
    tcolor = _INK
    if stab_c in d and np.isfinite(d[stab_c].iloc[0]):
        st = float(d[stab_c].iloc[0]); title += f"  ·  stability {st:.2f}"; tcolor = _stability_color(st, bars)
    ax.set_title(title, fontsize=10.5, color=tcolor)
    if r2_total is not None:
        ax.set_xlabel(f"% of the variance this view explains (R² = {r2_total:.2f})")
    else:
        ax.set_xlabel("loading" if kind == "loading" else _MAP_LABELS.get(kind, kind))
    _clean(ax, grid="x")


@_themed
def plot_top_features(res: Dict[str, Any], inf: Optional[Dict[str, Any]] = None, component: int = 0,
                      kind: str = "structure", views: Optional[Sequence[int]] = None, view_names=None,
                      feature_names=None, top_n: int = 15, rank_by: str = "magnitude",
                      bars: Sequence[float] = STABILITY_BARS, figsize=None,
                      maps: Optional[Dict[str, Any]] = None):
    """The features that define one component, one panel per view.

    ``kind="structure"`` (default; defined for every method), ``"loadings"``, or
    an alternative map from `pysimlr.feature_maps` (``"partial"``,
    ``"global_partial"``, ``"component_partial"``, ``"relative_weight"``),
    taken from ``maps=`` or from ``inf["maps"]`` (then with its own intervals and
    stability; filled dots mark intervals that exclude 0).
    With ``inf = pysimlr.loading_inference(...)``: 95% subsample intervals,
    filled dots for stability-selected features or intervals excluding 0
    (selected names in bold), the permutation null as a grey band
    (+-``null_max_abs``), and the component's stability in each title, in
    primary ink / serious / critical against ``bars``. ``rank_by="selection"``
    ranks by selection frequency instead of magnitude.
    """
    tab = feature_table(res, inf, view_names, feature_names, maps=maps)
    kind = "loading" if kind in ("loading", "loadings") else kind
    if kind not in tab.columns or kind in ("view", "feature", "component"):
        avail = ["structure", "loadings"] + sorted(_maps_of(inf, maps))
        raise ValueError(f"kind {kind!r} is not available; choose from {avail} "
                         "(alternative maps come from maps= or loading_inference(maps=...)).")
    vs = list(range(len(res["structure"]))) if views is None else list(views)
    if kind == "structure":
        sub = ("Filled: selected or 95% interval excludes 0; hollow: interval includes 0; grey band: permutation null."
               if inf is not None else "Structure coefficients: correlation of each feature with the component's scores.")
    elif kind == "loading":
        sub = "Hard-assigned loadings."
    else:
        lbl = _MAP_LABELS.get(kind, kind)
        sub = lbl[:1].upper() + lbl[1:] + " against the other views' scores."
        if f"{kind}_ci_low" in tab.columns:
            sub += (" Filled: selected in ≥ 80% of refits; bars: 95% half-sample interval."
                    if kind == "relative_weight" else " Filled: 95% interval excludes 0.")
    fig = _new(figsize or (3.9 * len(vs) + 0.6, 0.27 * top_n + 2.2), f"What defines component {component + 1}",
               sub, _status_line(res))
    axes = np.atleast_1d(fig.subplots(1, len(vs)))
    null = inf.get("null_max_abs") if inf is not None else None
    for ax, m in zip(axes, vs):
        _draw_top_features(ax, tab, m, component, kind, top_n, rank_by, null, bars, _view_color(m))
    return fig


def _sel_freq(d, kind):
    c = "selection_frequency" if kind == "structure" else f"{kind}_selection_frequency"
    return d[c].to_numpy(float) if c in d else None


def _map_stability(d, kind):
    c = "component_stability" if kind == "structure" else f"{kind}_stability"
    return float(d[c].iloc[0]) if c in d and np.isfinite(d[c].iloc[0]) else None


@_themed
def plot_feature_map_shift(res: Dict[str, Any], maps: Optional[Dict[str, Any]] = None,
                           inf: Optional[Dict[str, Any]] = None, component: int = 0, view: int = 0,
                           from_kind: str = "structure", to_kind: str = "relative_weight", top_n: int = 12,
                           min_shift: int = 3, view_names=None, feature_names=None,
                           bars: Sequence[float] = STABILITY_BARS, figsize=None):
    """How the feature ranking of one component changes between two maps.

    Each line joins a feature's rank by ``|from_kind|`` (left) to its rank by
    ``|to_kind|`` (right); features outside the top ``top_n`` on a side sit in a
    "lower" row. Lines that climb at least ``min_shift`` places are drawn in the
    view's colour, lines that fall in grey. With ``inf =
    loading_inference(..., maps=...)``, a filled end dot means the feature is
    selected in >= 80% of refits under that map, and each header carries the
    map's component stability. The default question it answers: which features
    gain once each feature's unique share of the prediction (relative weight)
    replaces plain association with the component (structure)?
    """
    tab = feature_table(res, inf, view_names, feature_names, maps=maps)
    for k_ in (from_kind, to_kind):
        if k_ not in tab.columns:
            raise ValueError(f"map {k_!r} is not available; pass it in maps= or loading_inference(maps=...).")
    d = tab[(tab.view == view) & (tab.component == component + 1)].reset_index(drop=True)
    a = np.nan_to_num(np.abs(d[from_kind].to_numpy(float)), nan=-1.0)
    b = np.nan_to_num(np.abs(d[to_kind].to_numpy(float)), nan=-1.0)
    ra = np.argsort(np.argsort(-a)) + 1
    rb = np.argsort(np.argsort(-b)) + 1
    feats = sorted(set(np.where(ra <= top_n)[0]) | set(np.where(rb <= top_n)[0]))
    low = top_n + 1.8
    fa, fb = _sel_freq(d, from_kind), _sel_freq(d, to_kind)
    color = _view_color(view)
    label = {"structure": "structure", "relative_weight": "relative weight", "partial": "partial (inverse corr.)",
             "global_partial": "global-partial", "component_partial": "component-partial"}
    fig = _new(figsize or (8.6, 0.36 * top_n + 2.9),
               f"Component {component + 1} of {d.view_name.iloc[0]}: {label.get(from_kind, from_kind)} vs "
               f"{label.get(to_kind, to_kind)}",
               f"Rank by |map|. Coloured lines climb ≥ {min_shift} places, grey lines fall ≥ {min_shift}."
               + (" Filled dots: selected in ≥ 80% of refits." if fb is not None else ""),
               _status_line(res))
    ax = fig.subplots(1, 1)
    for i in feats:
        ya = ra[i] if ra[i] <= top_n else low
        yb = rb[i] if rb[i] <= top_n else low
        shift = ra[i] - rb[i]
        if shift >= min_shift:
            c_, lw, al, z = color, 2.6, 0.95, 3
        elif shift <= -min_shift:
            c_, lw, al, z = _MUTED, 1.5, 0.55, 2
        else:
            c_, lw, al, z = _AXIS, 1.3, 0.8, 2
        ax.plot([0, 1], [ya, yb], color=c_, lw=lw, alpha=al, zorder=z, solid_capstyle="round")
        for x, y, f in ((0, ya, fa), (1, yb, fb)):
            sel = f is not None and f[i] >= 0.8
            dc = color if shift >= min_shift else _MUTED
            _dots(ax, [x], [y], dc, s=70 if sel else 40, hollow=not sel, zorder=z + 1)
    names = d.feature_name.to_numpy()
    for i in feats:
        if ra[i] <= top_n:
            ax.text(-0.05, ra[i], f"{names[i]}  {ra[i]}", ha="right", va="center", fontsize=8.6, color=_INK2)
        if rb[i] <= top_n:
            ax.text(1.05, rb[i], f"{rb[i]}  {names[i]}", ha="left", va="center", fontsize=8.6,
                    color=_INK if ra[i] - rb[i] >= min_shift else _INK2,
                    weight="semibold" if ra[i] - rb[i] >= min_shift else "normal")
    n_low_a = sum(ra[i] > top_n for i in feats); n_low_b = sum(rb[i] > top_n for i in feats)
    if n_low_a:
        ax.text(-0.05, low, f"lower ({n_low_a})", ha="right", va="center", fontsize=8, color=_MUTED, style="italic")
    if n_low_b:
        ax.text(1.05, low, f"lower ({n_low_b})", ha="left", va="center", fontsize=8, color=_MUTED, style="italic")
    for x, k_ in ((0, from_kind), (1, to_kind)):
        st = _map_stability(d, k_)
        head = label.get(k_, k_) + (f"\nstability {st:.2f}" if st is not None else "")
        ax.text(x, 0.0, head, ha="center", va="bottom", fontsize=9.4, weight="semibold",
                color=_stability_color(st, bars) if st is not None else _INK)
    ax.set_xlim(-0.95, 1.95); ax.set_ylim(low + 0.7, -0.9)
    ax.axis("off")
    return fig


_MAP_SHORT = {"structure": "structure", "relative_weight": "relative weight", "partial": "partial (inverse corr.)",
              "global_partial": "global-partial", "component_partial": "component-partial"}


def map_colormap(view: int = 0):
    """The sequential colour scale every feature-map display uses (grid, brain):
    near-surface to the view's colour to that colour darkened."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    c = _view_color(view)
    return LinearSegmentedColormap.from_list(f"pysimlr_map_{view}", ["#f3f2ec", c, tuple(0.45 * np.array(to_rgb(c)))])


def feature_map_values(res: Dict[str, Any], maps: Optional[Dict[str, Any]] = None,
                       inf: Optional[Dict[str, Any]] = None, component: int = 0, view: int = 0,
                       kinds: Sequence[str] = ("structure", "relative_weight"), feature_names=None,
                       view_names=None) -> Dict[str, Dict[str, Any]]:
    """The one encoding behind `plot_feature_map_grid` and brain renderings
    (`plot_brain_map`, drawn by antsxviz), so the two always show the same thing.

    Per kind: ``values`` (per feature, display units: |map|, or for
    ``relative_weight`` percent of the variance the view explains), ``vmax``
    (scale 0..vmax), ``selected`` (selection frequency >= 0.8, or None without
    inference), ``stability`` (or None), ``units``, ``label`` and ``names``.
    """
    tab = feature_table(res, inf, view_names, feature_names, maps=maps)
    d = tab[(tab.view == view) & (tab.component == component + 1)].reset_index(drop=True)
    out = {}
    for k_ in kinds:
        if k_ not in d.columns:
            raise ValueError(f"map {k_!r} is not available; pass it in maps= or loading_inference(maps=...).")
        v = np.abs(d[k_].to_numpy(float))
        units = "|corr(x, u)|" if k_ == "structure" else "|map|"
        if k_ == "relative_weight":
            tot = np.nansum(v)
            v = 100.0 * v / tot if tot > 0 else v * np.nan
            units = "% of explained variance"
        f = _sel_freq(d, k_)
        out[k_] = {"values": v, "vmax": float(np.nanmax(v)) if np.isfinite(v).any() else 1.0,
                   "selected": (f >= 0.8) if f is not None else None, "stability": _map_stability(d, k_),
                   "units": units, "label": _MAP_SHORT.get(k_, k_), "names": d.feature_name.tolist(),
                   "view_name": d.view_name.iloc[0]}
    return out


def plot_feature_map_grid(res: Dict[str, Any], maps: Optional[Dict[str, Any]] = None,
                          inf: Optional[Dict[str, Any]] = None, component: int = 0, view: int = 0,
                          kinds: Sequence[str] = ("structure", "relative_weight"), split=None, groups=None,
                          feature_names=None, view_names=None, value_fmt: str = "{:.2f}", figsize=None,
                          scale: str = "minmax", clim=None):
    """Feature maps of one component as coloured tables, one block per map.

    The tabular twin of `plot_brain_map`: both read `feature_map_values`, so values,
    the 0..max scale per map (`map_colormap`), selection rings (>= 80% of refits) and
    stability are identical. Drawn by antsxviz (``region_table_figure``), the ANTsX
    house renderer, which needs the ``viz`` extra.

    ``split``: optional callable feature name -> (row label, column label), e.g.
    ``antsxviz.split_hemisphere`` so hemispheres share a row. ``groups``: optional
    dict row label -> group (lobe, network); rows are ordered by group, then by the
    last map's largest value. Works for any features, not only brain regions.
    """
    av = _require_antsxviz()
    enc = feature_map_values(res, maps, inf, component, view, kinds, feature_names, view_names)
    rows = av.rows_from_encoding(enc, cmap=map_colormap(view))
    _apply_scale(rows, scale, clim, view, av)   # the same limits and colours as plot_brain_map
    sub = ("Colour: each map on its own scale from 0 to its maximum"
           + ("; dark ring: selected in ≥ 80% of refits" if inf is not None else "") + ".")
    vname = next(iter(enc.values()))["view_name"]
    return av.region_table_figure(rows, split=split, groups=groups,
                                  title=f"Component {component + 1} of {vname}: feature maps", subtitle=sub,
                                  value_fmt=value_fmt, figsize=figsize)


# ----------------------------------------------------------------------------- brain maps (antsxviz)
def _require_antsxviz():
    """antsxviz, or an ImportError that says how to get it (never a fallback renderer)."""
    try:
        import antsxviz
    except ImportError as err:
        raise ImportError("pysimlr brain maps are drawn by antsxviz, the ANTsX house renderer, which is not "
                          "installed: pip install 'pysimlr[viz]' (or antsxviz>=0.1.0).") from err
    return antsxviz


def cortical_region_name(name: str) -> Optional[str]:
    """A feature name as a DKT cortical region name for the brain renderer, or None.

    ``"left_entorhinal"`` stays as it is; ANTsPyT1w hierarchical names
    ``"left_entorhinaldktcortex"`` (and the ``...dktregions`` parcellation variant)
    become ``"left_entorhinal"``. Anything else (tissue totals, lobes, MTL or CIT168
    labels) is not a cortical region and returns None.
    """
    import re
    m = re.match(r"^(left|right)_(.+?)(dktcortex|dktregions)?$", str(name))
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def _region_index(names: Sequence[str], region_names, av):
    """``(keep, dropped)``: ``keep`` lists ``(feature index, DKT region name)`` for features that
    map to a cortical region (the ``dktcortex`` variant wins when two map to one region);
    ``dropped`` lists the features that are not on the cortex."""
    to_region = region_names if callable(region_names) else (
        (lambda n: region_names.get(n)) if isinstance(region_names, dict) else cortical_region_name)
    chosen, dropped = {}, []                                  # region -> feature index
    for i, n in enumerate(names):
        r = to_region(n)
        if r is not None:
            try:
                av.dkt_id(r)
            except KeyError:
                r = None
        if r is None:
            dropped.append(n)
            continue
        if r not in chosen:
            chosen[r] = i
        elif str(n).endswith("dktcortex") and not str(names[chosen[r]]).endswith("dktcortex"):
            dropped.append(names[chosen[r]]); chosen[r] = i  # the DKT cortex label wins
        else:
            dropped.append(n)
    return sorted((i, r) for r, i in chosen.items()), dropped


_TIER_GREY = "#e4e3dd"
_TIER_UNITS = "relevance tier"


def _label_tier_colorbar(fig):
    """Categorical ticks on the tier colorbar (antsxviz draws a continuous 0..vmax bar)."""
    for ax in fig.axes:
        if ax.get_ylabel() == _TIER_UNITS:
            ax.set_yticks([1 / 3, 1.0, 5 / 3])
            ax.set_yticklabels(["neither", "training\nonly", "replicated"])
            ax.set_ylabel("")
    return fig
_TEXT_BG = dict(fc=_SURFACE, ec="none", pad=0.6)   # keeps a label legible over reference lines


def _tier_colors(view: int):
    """(neither, training only, replicated) colours for relevance tiers in a view's hue."""
    from matplotlib.colors import to_rgb
    dark = _view_color(view)
    return _TIER_GREY, tuple(0.45 * np.array(to_rgb(dark)) + 0.55), dark


def _tier_rows(rel: Dict[str, Any], view: int, region_names, av):
    """antsxviz ``rows`` painting a view's relevance tiers (0 / 1 / 2) on the cortex."""
    from matplotlib.colors import ListedColormap
    f = rel["features"][rel["features"].view == view].reset_index(drop=True)
    if f.empty:
        raise ValueError(f"view {view} is not in this feature_relevance result.")
    keep, dropped = _region_index(list(f.feature), region_names, av)
    vals = {r: float(f.tier.iloc[i]) for i, r in keep}
    rows = {"relevance tier": {"values": vals, "vmax": 2.0, "selected": [], "stability": None,
                               "units": _TIER_UNITS, "note": None,
                               "cmap": ListedColormap(list(_tier_colors(view)))}}
    return rows, dropped


def brain_rows(enc: Dict[str, Dict[str, Any]], region_names=None, notes: Optional[Dict[str, str]] = None,
               view: int = 0):
    """`feature_map_values` output as antsxviz ``rows``, restricted to cortical regions.

    ``region_names``: a callable or dict mapping a feature name to a DKT region name
    (or None for a feature that is not on the cortex); default `cortical_region_name`.
    When two features map to one region (``dktcortex`` and ``dktregions`` variants),
    the ``dktcortex`` one is kept. Returns ``(rows, dropped)``, where ``dropped`` lists
    the features that cannot be drawn on cortex, so the caller can state them.
    """
    av = _require_antsxviz()
    first = next(iter(enc.values()))
    keep, dropped = _region_index(list(first["names"]), region_names, av)
    sub = {}
    for kind, e in enc.items():
        idx = [i for i, _ in keep]
        sub[kind] = dict(e, names=[r for _, r in keep], values=np.asarray(e["values"], float)[idx],
                         selected=None if e.get("selected") is None else np.asarray(e["selected"])[idx])
    rows = av.rows_from_encoding(sub, cmap=map_colormap(view), notes=notes)
    return rows, dropped


def plot_brain_map(res: Dict[str, Any], inf: Optional[Dict[str, Any]] = None, maps: Optional[Dict[str, Any]] = None,
                   component: int = 0, view: int = 0, kinds: Sequence[str] = ("structure", "relative_weight"),
                   feature_names=None, view_names=None, region_names=None, title: Optional[str] = None,
                   subtitle: Optional[str] = None, notes: Optional[Dict[str, str]] = None, meshes=None,
                   density: str = "41k", surface: str = "inflated", tiers: Optional[Dict[str, Any]] = None,
                   atlas: str = "dkt", scale: str = "minmax", clim=None, **scene_kwargs):
    """Feature maps of one component painted on the brain, via antsxviz.

    ``atlas`` chooses the anatomy: ``"dkt"`` (default) the fsaverage cortical surface described
    below; ``"hoa"`` (Harvard-Oxford subcortical) or ``"jhu"`` (JHU white-matter regions) as lit 3D
    structures inside a glass brain (``antsxviz.region_scene_figure``); ``"tracts"`` or
    ``"tract_sheets"`` the 20 population tracts as tubes or medial sheets
    (``antsxviz.tract_scene_figure``); ``"yeo"`` Yeo-17 homotopic parcel values (``rsf_yeo_alff_ROI<n>_*``,
    ``rsf_yeo_falff_*``, ``rsf_yeo_homotopic_*``) painted on the fsaverage cortex
    (``antsxviz.yeo_surface_figure``). One row per map. Volumetric atlases read antsxmm column names
    as they are (e.g. ``FA_HOA_Thalamus_Left``, ``FA_JHU_labels_edited_Genu_of_corpus_callosum``,
    ``troi_fa_mean_Uncinate_fasciculus_L``); ``region_names`` ({feature: atlas feature name})
    renames features that do not follow that grammar. Features not drawn are listed in the subtitle.
    ``scene_kwargs`` go to the antsxviz scene function -- e.g. ``labels=`` and ``brain_mask=`` (a
    subject's own label image and brain mask), ``tracts=``/``sheets=`` (a subject's own antsxdwi
    bundles), ``views=``, ``exclude=``. Selection outlines and ``tiers`` are drawn on the DKT surface
    only.

    Colour limits (every atlas): ``clim=(lo, hi)`` (or ``{map: (lo, hi)}``) when given, otherwise
    ``scale``: ``"minmax"`` (default; min..max of each map), ``"zero_max"`` (0..max, the earlier
    convention) or ``"symmetric"``.

    With ``tiers`` (a `pysimlr.feature_relevance` result) the brain shows that view's relevance
    tiers instead of continuous maps: grey = neither, pale = p < alpha in training only, dark =
    replicated (training and held-out).

    The brain twin of `plot_feature_map_grid`: both read `feature_map_values`, so
    values, the 0..max scale per map (`map_colormap`), selection outlines (>= 80% of
    refits) and stability are identical. Rows are maps, columns are L lateral,
    L medial, ventral (R | L), R medial, R lateral. Features that are not DKT
    cortical regions (see `cortical_region_name`) are not drawn; they are listed in
    the subtitle. ``meshes`` (from ``antsxviz.load_fsaverage``) can be passed to reuse
    them across figures. Requires the ``viz`` extra (antsxviz).
    """
    av = _require_antsxviz()
    if atlas != "dkt":
        if tiers is not None:
            raise ValueError("tiers= is drawn on the DKT surface only; use atlas='dkt'.")
        if atlas not in ("hoa", "jhu", "tracts", "tract_sheets", "yeo"):
            raise ValueError(f"atlas must be 'dkt', 'hoa', 'jhu', 'tracts', 'tract_sheets' or 'yeo'; got {atlas!r}")
        enc = feature_map_values(res, maps, inf, component, view, kinds, feature_names, view_names)
        rename = dict(region_names or {})
        maps_ = {e["label"]: {rename.get(n, n): v for n, v in zip(e["names"], e["values"])} for e in enc.values()}
        vname = next(iter(enc.values()))["view_name"]
        common = dict(units={e["label"]: e["units"] for e in enc.values()}, scale=scale, clim=clim,
                      view_color=view, subtitle=subtitle,
                      title=title or f"Component {component + 1} of {vname}: feature maps", **scene_kwargs)
        if atlas == "yeo":
            common.pop("view_color")
            return av.yeo_surface_figure(maps_, **common)
        if atlas in ("hoa", "jhu"):
            return av.region_scene_figure(maps_, atlas, **common)
        return av.tract_scene_figure(maps_, style="sheets" if atlas == "tract_sheets" else "tubes", **common)
    if tiers is not None:
        rows, dropped = _tier_rows(tiers, view, region_names, av)
        if meshes is None:
            meshes = av.load_fsaverage(density=density, surface=surface)
        a = tiers["alpha"]
        sub = subtitle if subtitle is not None else (
            f"Relevance tiers (feature_relevance): dark = permutation p < {a:g} in training and held-out; "
            f"pale = training only; grey = neither.")
        if dropped:
            sub += f" {len(dropped)} features are not cortical regions and are not drawn."
        vname = tiers["view_names"][view] if view < len(tiers["view_names"]) else f"view {view}"
        return _label_tier_colorbar(av.surface_figure(
            rows, meshes, title=title or f"Component {tiers['component'] + 1} of {vname}: relevance tiers",
            subtitle=sub))
    enc = feature_map_values(res, maps, inf, component, view, kinds, feature_names, view_names)
    rows, dropped = brain_rows(enc, region_names, notes, view)
    if not rows or not next(iter(rows.values()))["values"]:
        raise ValueError("no feature of this view is a DKT cortical region; pass region_names= to map them.")
    if meshes is None:
        meshes = av.load_fsaverage(density=density, surface=surface)
    vname = next(iter(enc.values()))["view_name"]
    _apply_scale(rows, scale, clim, view, av)
    sub = subtitle if subtitle is not None else (
        _scale_phrase(scale, clim)
        + ("; dark outline: selected in ≥ 80% of refits" if inf is not None else "") + ".")
    if dropped:
        sub += (f" Not on cortex ({len(dropped)}): " + ", ".join(map(str, dropped[:6]))
                + (" ..." if len(dropped) > 6 else "") + ".")
    return av.surface_figure(rows, meshes, title=title or f"Component {component + 1} of {vname}: feature maps",
                             subtitle=sub)


def plot_connectome(res: Dict[str, Any], inf: Optional[Dict[str, Any]] = None, maps: Optional[Dict[str, Any]] = None,
                    component: int = 0, view: int = 0, kind: str = "structure", feature_names=None, view_names=None,
                    top_k: int = 30, scale: str = "minmax", clim=None, title: Optional[str] = None,
                    subtitle: Optional[str] = None, **kwargs):
    """A component's map over network-pair features (antsxmm ``rsf_yeo_netconn_{a}_2_{b}``) as a connectome.

    Draws `feature_map_values` of one map (``kind``) through ``antsxviz.connectome_figure``: Yeo-17 +
    CIT168 network nodes per hemisphere in a glass brain (MNI), the ``top_k`` strongest pairs as
    arcs, with a chord diagram and a block-ordered matrix on the same colour scale. Features that are
    not network pairs (e.g. ``rsf_yeo_netconn_nparcels_*``) are listed in the subtitle. ``kwargs`` go to
    ``antsxviz.connectome_figure`` (e.g. ``nodes=``, ``groups=``, ``views=``, ``width_by=``).
    """
    av = _require_antsxviz()
    enc = feature_map_values(res, maps, inf, component, view, (kind,), feature_names, view_names)
    e = enc[kind]
    vals = {n: float(v) for n, v in zip(e["names"], e["values"]) if np.isfinite(v)}
    return av.connectome_figure(vals, top_k=top_k, scale=scale, clim=clim, units=e["units"],
                                title=title or f"Component {component + 1} of {e['view_name']}: network map",
                                subtitle=subtitle, **kwargs)


def _scale_phrase(scale: str, clim) -> str:
    if clim is not None:
        return "Colour: fixed limits"
    return {"minmax": "Colour: each map on its own scale from its minimum to its maximum",
            "zero_max": "Colour: each map on its own scale from 0 to its maximum",
            "symmetric": "Colour: each map on a symmetric scale about 0"}[scale]


def _apply_scale(rows, scale: str, clim, view: int, av) -> None:
    """Set each antsxviz row's colour limits (``vmin``/``vmax``) and colour scale from ``scale``/``clim``
    (`antsxviz.resolve_clim`), so the surface and its table twin share them."""
    from antsxviz.scene import value_cmap

    for label, row in rows.items():
        rc = clim.get(label) if isinstance(clim, dict) else clim
        vals = [float(v) for v in row["values"].values() if v is not None and np.isfinite(float(v))]
        lo, hi, div = av.resolve_clim(vals, rc, scale)
        row["vmin"], row["vmax"] = lo, hi
        row["cmap"] = value_cmap(div, view)


def _fp(p) -> str:
    return "n/a" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def _crop_rgba(img, pad=6):
    """Trim the transparent border of a rendered RGBA image."""
    ys, xs = np.where(img[..., 3] > 0)
    return img if not len(ys) else img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


_BRAIN_VIEWS = (("left", "lateral", "L lateral"), ("left", "medial", "L medial"), ("both", "ventral", "ventral (R | L)"),
                ("right", "medial", "R medial"), ("right", "lateral", "R lateral"))


def _tier_brain_images(rel, view, region_names, meshes):
    """Rendered, cropped surface views of a view's relevance tiers (one RGBA image per column)."""
    from matplotlib.colors import ListedColormap
    av = _require_antsxviz()
    rows, dropped = _tier_rows(rel, view, region_names, av)
    row = rows["relevance tier"]
    if not row["values"]:
        raise ValueError("no feature of this view is a DKT cortical region; pass region_names= to map them.")
    if meshes is None:
        meshes = av.load_fsaverage()
    cmap = ListedColormap(list(_tier_colors(view)))

    def render(hemi, v):
        return _crop_rgba(av.render_surface(row["values"], meshes[hemi], hemi, v, cmap, (0.0, 2.0)))
    imgs = []
    for hemi, v, _ in _BRAIN_VIEWS:
        if hemi == "both":
            a, b = render("left", "ventral"), render("right", "ventral")
            h = max(a.shape[0], b.shape[0])
            imgs.append(np.concatenate([np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0))) for im in (b, a)], axis=1))
        else:
            imgs.append(render(hemi, v))
    return imgs, dropped


def _draw_relevance_forest(ax, show, fair, cols, label, alpha, title, legend, r_label, room):
    """Forest of shares: filled = training, ring = held-out; p values, refit frequency and held-out r."""
    from matplotlib.patches import Patch
    y = np.arange(len(show))
    has_r = show["r_outcome_test"].notna().any()
    top = 100 * np.nanmax(show[["share_train", "share_test"]].to_numpy(float)) if len(show) else 100 * fair
    xmax = max(top, 100 * fair) / (1.0 - room - (0.08 if has_r else 0.0))   # room: axis fraction for the text
    ax.axvline(100 * fair, color=_INK2, lw=1, ls="--", zorder=1)
    ax.text(100 * fair, -1.0, " fair share", fontsize=8, color=_INK2, va="bottom")
    if has_r:
        ax.text(xmax, -1.0, r_label, fontsize=7.6, color=_INK2, ha="right", va="bottom")
    for yi, (_, r) in zip(y, show.iterrows()):
        c, st, ste = cols[int(r.tier)], 100 * r.share_train, 100 * r.share_test
        if np.isfinite(ste):
            ax.plot([st, ste], [yi, yi], color=_AXIS, lw=1.2, zorder=1)
            ax.scatter(ste, yi, s=46, facecolor=_SURFACE, edgecolors=c if r.tier else _MUTED, linewidths=1.6,
                       zorder=3)
        ax.scatter(st, yi, s=58, color=c, edgecolors=_SURFACE, linewidths=1.2, zorder=3)
        ref = r.refit_p05_freq
        ax.annotate(f"p {_fp(r.p_train)} / {_fp(r.p_test)}"
                    + (f"   refits {ref:.0%}" if r.tier == 2 and ref is not None and np.isfinite(ref) else ""),
                    (np.nanmax([st, ste]), yi), xytext=(8, 0), textcoords="offset points", va="center", bbox=_TEXT_BG,
                    fontsize=7.6, color=_INK if r.tier == 2 else _INK2)
        if has_r and r.r_outcome_test is not None and np.isfinite(r.r_outcome_test):
            strong = abs(r.r_outcome_test) >= 0.3
            ax.text(xmax, yi, f"{r.r_outcome_test:+.2f}", va="center", ha="right", fontsize=8,
                    color=_INK if strong else _INK2, weight="semibold" if strong else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([label(n) for n in show.feature], fontsize=8.3)
    for lbl, t in zip(ax.get_yticklabels(), show.tier):
        lbl.set_color(_INK if t == 2 else _INK2)
        lbl.set_fontweight("semibold" if t == 2 else "normal")
    ax.set_ylim(len(show) - 0.4, -1.6)
    ax.set_xlim(0, xmax)
    _clean(ax, grid="x")
    ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(handles=[Patch(color=cols[2], label=f"replicated (training and held-out p < {alpha:g})"),
                           Patch(color=cols[1], label="training only")], loc="lower right", frameon=False, fontsize=8,
                  bbox_to_anchor=(0.88, 0.0))


def _draw_relevance_groups(ax, g, fair, cols, alpha, title):
    """Group shares as a ratio to their fair share: bar = training, ring = held-out."""
    g = g.sort_values("ratio_fair", ascending=True)
    yy = np.arange(len(g))
    full = "p_null_refit" in g and np.isfinite(g["p_null_refit"].astype(float)).any()
    ax.axvline(1.0, color=_INK2, lw=1, ls="--")
    hi = 1.0
    for yi, (_, r) in zip(yy, g.iterrows()):
        c = cols[int(r.tier)] if r.tier else "#c9c8c1"
        test_ratio = r.share_test / (r.n * fair) if np.isfinite(r.share_test) else np.nan
        ax.hlines(yi, 1.0, r.ratio_fair, color=c, lw=6, alpha=0.9)
        if np.isfinite(test_ratio):
            ax.scatter(test_ratio, yi, s=46, facecolor=_SURFACE, edgecolors=c if r.tier == 2 else _MUTED,
                       linewidths=1.6, zorder=3)
        end = np.nanmax([1.0, r.ratio_fair, test_ratio])   # past the bar, which starts at 1
        hi = max(hi, end)
        ax.annotate(f"{r.ratio_fair:.2f}x   p {_fp(r.p_train)} / {_fp(r.p_test)}"
                    + (f" / {_fp(r.p_null_refit)}" if full else ""), (end, yi), xytext=(8, 0),
                    textcoords="offset points", va="center", bbox=_TEXT_BG, fontsize=7.8, color=_INK if r.tier == 2 else _INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{n} ({k})" for n, k in zip(g.name, g.n)], fontsize=9)
    lo = min(0.4, float(np.nanmin(g.ratio_fair)) - 0.1) if len(g) else 0.4
    ax.set_xlim(max(lo, 0.0), hi / 0.55)
    ax.set_xlabel("share ÷ fair share (bar: training; ring: held-out)")
    ax.set_title(title + ("  (p: training / held-out" + (" / refit null)" if full else ")")), fontsize=10)
    _clean(ax, grid="x")


@_themed
def plot_feature_relevance(rel: Dict[str, Any], view: int = 0, *, label=None, separate=None,
                           separate_title: str = "Measured separately (not regions)", grouping: Optional[str] = None,
                           grouping_title: Optional[str] = None,
                           brain: bool = False, region_names=None, meshes=None, show_all: bool = False,
                           r_label: str = "r with outcome\n(held-out)", title: Optional[str] = None,
                           subtitle: Optional[str] = None, figsize=None):
    """Tiered relevance of one view's features, from `pysimlr.feature_relevance`.

    Left: a forest of each feature's share of the component variance this view explains
    (filled: training; ring: held-out), against the fair share 1/p, with permutation p in
    training / held-out, the fraction of half-sample refits at p < alpha (tier 2 only) and,
    when the result has it, each feature's held-out correlation with the outcome. Only tiers 1
    and 2 are listed unless ``show_all``. Right: the groups of ``grouping`` (default: the first
    grouping in the result) as a ratio to their fair share, and the features picked by
    ``separate`` (a predicate on the feature name, e.g. whole-brain totals) in their own panel,
    since a single global measure is on a different footing from a region.

    Colour is the tier: dark = p < alpha in training and held-out, pale = training only,
    grey = neither. ``brain=True`` adds the tiers painted on the fsaverage cortex
    (antsxviz; ``region_names`` as in `plot_brain_map`). ``label`` maps a feature name to its
    display label; ``grouping_title`` names the group panel (default: the grouping).
    """
    f = rel["features"]
    f = f[f.view == view].copy()
    if f.empty:
        raise ValueError(f"view {view} is not in this feature_relevance result (views: "
                         f"{sorted(rel['features'].view.unique().tolist())}).")
    alpha, fair = float(rel["alpha"]), 1.0 / len(f)
    vname = rel["view_names"][view] if view < len(rel["view_names"]) else f"view {view}"
    label = label or str
    cols = _tier_colors(view)
    g = rel["groups"]
    g = g[g.view == view] if len(g) else g
    if len(g):
        grouping = grouping or g.grouping.iloc[0]
        if grouping not in set(g.grouping):
            raise ValueError(f"grouping {grouping!r} not in the result (have {sorted(set(g.grouping))}).")
        g = g[g.grouping == grouping].copy()
        g["name"] = [s[len(grouping) + 1:] if s.startswith(grouping + " ") else s for s in g.group]
    elif grouping is not None:
        raise ValueError("this feature_relevance result has no groups; pass groups= to feature_relevance.")
    is_sep = f.feature.map(separate).astype(bool) if separate is not None else np.zeros(len(f), bool)
    main = f[~is_sep]
    shown = main if show_all else main[main.tier > 0]
    shown = shown.sort_values("share_train", ascending=False)
    seps = f[is_sep].sort_values("share_train", ascending=False)

    imgs, dropped = (_tier_brain_images(rel, view, region_names, meshes) if brain else (None, []))
    right_panels = int(len(g) > 0) + int(len(seps) > 0)
    n_rows = max(len(shown), 4)
    if figsize is None:
        figsize = (15.5 if right_panels else 9.5, 1.2 + 0.24 * n_rows + (3.4 if brain else 0.0) + 1.0)
    s = rel["summary"]
    srow = s[s.view == view].iloc[0] if "view" in s else None
    n_main = len(main)
    n1, n2 = int((main.tier > 0).sum()), int((main.tier == 2).sum())
    if subtitle is None:
        subtitle = (f"Share of the variance this view explains in component {rel['component'] + 1} (relative weights); "
                    f"fair share = 1/{len(f)}. Rule: dark = permutation p < {alpha:g} in training AND held-out; "
                    f"pale = training only; grey = neither. {rel['n_permutations']:,} training / "
                    f"{rel['n_permutations_test']:,} held-out permutations, {rel['n_refits']} half-sample refits; "
                    "p uncorrected.")
        if srow is not None:
            subtitle += f" n = {int(srow.n_train)} training, {int(srow.n_test)} held-out."
        if dropped:
            subtitle += f"\nBrain: {len(dropped)} features are not cortical regions and are not drawn."
    fig = _new(figsize, title or f"{vname}, component {rel['component'] + 1}: relevance tiers", subtitle)
    if brain:
        gs = fig.add_gridspec(2, len(imgs), height_ratios=[1, max(1.6, 0.11 * n_rows)],
                              width_ratios=[im.shape[1] / im.shape[0] for im in imgs])
        for c, (im, (_, _, head)) in enumerate(zip(imgs, _BRAIN_VIEWS)):
            ax = fig.add_subplot(gs[0, c])
            ax.imshow(im)
            ax.axis("off")
            ax.set_title(head, fontsize=9.5, color=_INK2)
        body = gs[1, :]
    else:
        body = fig.add_gridspec(1, 1)[0, 0]
    sub = body.subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.08) if right_panels else body.subgridspec(1, 1)
    ax = fig.add_subplot(sub[0, 0])
    head = (f"{'All features' if show_all else f'Features with training p < {alpha:g}'}: {n1} of {n_main} "
            f"(chance {alpha * n_main:.1f}); replicated held-out: {n2}")
    _draw_relevance_forest(ax, shown, fair, cols, label, alpha, head, True, r_label, room=0.36)
    ax.set_xlabel("% of explained variance (filled: training; ring: held-out)")
    if right_panels:
        heights = ([1.2 + 0.5 * len(g)] if len(g) else []) + ([1.4 + 0.45 * len(seps)] if len(seps) else [])
        right = sub[0, 1].subgridspec(len(heights), 1, height_ratios=heights, hspace=0.35)
        k = 0
        if len(g):
            _draw_relevance_groups(fig.add_subplot(right[k, 0]), g, fair, cols, alpha, grouping_title or grouping)
            k += 1
        if len(seps):
            ax = fig.add_subplot(right[k, 0])
            _draw_relevance_forest(ax, seps, fair, cols, label, alpha, separate_title, False, r_label, room=0.62)
            ax.set_xlabel("% of explained variance")
    return fig


@_themed
def plot_component_attribution(att: Dict[str, Any], *, spectrum_view: Optional[int] = None,
                               selection_views: Optional[Sequence[int]] = None,
                               compare: Optional[Dict[int, str]] = None, label=None,
                               view_labels: Optional[Dict[int, str]] = None, view_colors: Optional[Dict[int, Any]] = None,
                               titles: Optional[Dict[str, str]] = None, outcome_label: str = "outcome",
                               variant_labels: Optional[Dict[str, str]] = None, feature_word: str = "feature",
                               top_n: int = 14, figsize=(12.4, 10.2), title: Optional[str] = None,
                               subtitle: Optional[str] = None):
    """Draw `pysimlr.component_attribution`: which components carry the held-out prediction (A),
    a view's variance spectrum (B), and how often refits select each feature of the predictive
    component (one panel per view in ``selection_views``).

    ``spectrum_view`` defaults to the view whose first principal component explains the most
    variance; ``selection_views`` to the (up to two) views whose best component predicts best.
    ``compare={view: variant}`` draws that view's selection frequency under the default refits
    (hollow) beside a variant (filled), with the share of refits keeping two or fewer features.
    ``label`` maps a feature name to its display label; ``view_labels``, ``view_colors`` and
    ``titles`` (keys ``"A"``, ``"B"`` and the view index as a string) override the defaults;
    ``variant_labels`` renames refit variants in legends and notes (e.g. ``{"default":
    "default (collapse_weight = 1)"}``) and ``feature_word`` names the features ("region").
    Save the figure inside ``with pysimlr.plotting.theme():`` to keep the house font.
    """
    import matplotlib.pyplot as plt
    views = list(att["views"]); compare = dict(compare or {})
    vlab = {m: (view_labels or {}).get(m, att["view_names"][m]) for m in views}
    vcol = {m: (view_colors or {}).get(m, _view_color(m)) for m in views}
    label = label or str; titles = dict(titles or {})
    vl = lambda name: (variant_labels or {}).get(name, name)
    fw = feature_word
    comps = att["components"]
    if spectrum_view is None:
        spectrum_view = max(views, key=lambda m: att["spectra"][str(m)]["eigen_share"][0])
    if selection_views is None:
        best = {m: max(c["alone"] for c in comps if c["view"] == m) for m in views}
        selection_views = sorted(sorted(views, key=lambda m: -best[m])[:2])
    sel = list(selection_views)
    fig = _new(figsize, title, subtitle) if (title or subtitle) else plt.figure(figsize=figsize, layout="constrained")
    ncol = max(2, len(sel))
    gs = fig.add_gridspec(2, ncol, width_ratios=[1.1] + [1] * (ncol - 1), height_ratios=(1, 1.2))

    # A. each component's held-out R2 alone, and its unique contribution
    ax = fig.add_subplot(gs[0, 0]); y = 0; yt, yl = [], []
    for m in views:
        for c in [c for c in comps if c["view"] == m]:
            ax.hlines(y, 0, max(c["alone"], 0), color=vcol[m], lw=2.6, alpha=0.55)
            _dots(ax, [c["alone"]], [y], vcol[m], s=64)
            hi = c["unique"] > 0.01
            ax.text(max(c["alone"], 0) + 0.006, y, f"{c['alone']:.3f}" + (f"   unique {c['unique']:+.3f}" if hi else ""),
                    va="center", fontsize=8.2, color=_INK if hi else _INK2, weight="semibold" if hi else "normal")
            yt.append(y); yl.append(f"{vlab[m]} C{c['component']}  ({c['nnz']} feature{'s' if c['nnz'] != 1 else ''})")
            y += 1
        y += 0.6
    full = att["full_r2"]
    ax.axvline(full, color=_INK2, lw=1)
    ax.text(full, -0.9, f" all {len(comps)} columns {full:.3f}", fontsize=8, color=_INK2, va="bottom")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8.3); ax.set_ylim(y - 0.4, -1.2)
    hi_x = max([full] + [c["alone"] for c in comps])
    ax.set_xlim(-0.02, max(0.05, hi_x * 1.45)); ax.set_xlabel(f"held-out {outcome_label} R² of the component alone")
    _clean(ax, grid="x"); ax.set_title(titles.get("A", f"A. Held-out {outcome_label} R² of each component"), fontsize=10)

    # B. variance spectrum of one view
    ax = fig.add_subplot(gs[0, 1:]); sp = att["spectra"][str(spectrum_view)]
    ev = np.asarray(sp["eigen_share"])[:10]; nb = len(ev); nf = len(att["feature_names"][str(spectrum_view)])
    ax.bar(np.arange(1, nb + 1), ev, width=0.55, color=[vcol[spectrum_view]] + [_AXIS] * (nb - 1))
    ax.text(1.4, ev[0], f"  PC1 {ev[0]:.0%} of variance", va="center", fontsize=8.6, color=_INK)
    k = att["k"]; cg = sp["consensus_vs_mean"]; ks = " / ".join(f"C{j + 1}" for j in range(k))
    txt = (f"mean correlation between features: {sp['mean_corr']:.2f}\n"
           f"corr(consensus {ks}, view mean):\n      " + " / ".join(f"{c:.2f}" for c in cg))
    if sp.get("hard_assignment") is not None:
        txt += f"\nfeatures hard-assigned to {ks}:\n      " + " / ".join(str(h) for h in sp["hard_assignment"])
    txt = txt.replace("features", fw + "s")
    ax.text(0.98, 0.64, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=_INK2, linespacing=1.5)
    ax.set_xticks(range(1, nb + 1)); ax.set_xlabel(f"principal component of {vlab[spectrum_view]} ({nf} {fw}s)")
    ax.set_ylabel("share of variance"); _clean(ax, grid="y")
    ax.set_title(titles.get("B", f"B. {vlab[spectrum_view]}: variance spectrum and the consensus"), fontsize=10)

    # C, D, ... selection frequency of the predictive component, per view
    nh = att["n_refits"]; letters = "CDEFGH"
    for i, m in enumerate(sel):
        ax = fig.add_subplot(gs[1, i]); names = att["feature_names"][str(m)]; key = str(m)
        base = att["refits"]["default"]
        Fc = np.mean([h[key]["support"] for h in base], 0)
        if m in compare:
            alt = att["refits"][compare[m]]
            Fn = np.mean([h[key]["support"] for h in alt], 0)
            idx = [j for j in np.argsort(-(Fn + 1e-3 * Fc), kind="stable") if max(Fn[j], Fc[j]) > 0][:top_n]; yy = np.arange(len(idx))
            ax.hlines(yy, Fc[idx], Fn[idx], color=_AXIS, lw=1.4)
            ax.scatter(Fc[idx], yy, s=46, facecolors=_SURFACE, edgecolors=_INK2, linewidths=1.3, zorder=3, label=vl("default"))
            _dots(ax, Fn[idx], yy, vcol[m], s=54)
            ax.scatter([], [], s=54, c=[vcol[m]], label=vl(compare[m]))
            nc = [sum(h[key]["support"]) for h in base]; nn = [sum(h[key]["support"]) for h in alt]
            rc = np.median([h[key]["r2"] for h in base]); rn = np.median([h[key]["r2"] for h in alt])
            ax.set_xlabel(f"fraction of {nh} refits selecting the {fw}")
            ax.text(0.0, -0.17, f"default: ≤ 2 {fw}s in {sum(x <= 2 for x in nc)} of {nh} refits, median R² {rc:.3f}\n"
                    f"{vl(compare[m])}: ≤ 2 {fw}s in {sum(x <= 2 for x in nn)} of {nh}, median R² {rn:.3f}",
                    transform=ax.transAxes, fontsize=8.2, color=_INK2, va="top")
            ax.legend(loc="lower right", fontsize=8, frameon=False); ax.set_xlim(-0.02, 1.0)
        else:
            fullsup = np.asarray(att["full_support"][key]); r2 = np.median([h[key]["r2"] for h in base])
            idx = [j for j in np.argsort(-Fc, kind="stable") if Fc[j] > 0][:top_n]; yy = np.arange(len(idx))
            ax.barh(yy, Fc[idx], height=0.55, color=[vcol[m] if Fc[j] >= 0.5 else _AXIS for j in idx])
            for yv, j in zip(yy, idx):
                ax.text(Fc[j] + 0.015, yv, f"{Fc[j]:.2f}" + ("  ● full fit" if fullsup[j] else ""), va="center", fontsize=7.8, color=_INK2)
            ax.set_xlim(0, 1.2)
            ax.set_xlabel(f"fraction of {nh} refits selecting the {fw}   (median held-out R² {r2:.3f})")
        ax.set_yticks(yy); ax.set_yticklabels([label(names[j]) for j in idx], fontsize=8.2); ax.invert_yaxis()
        ax.axvline(0.5, color=_INK2, lw=0.9); _clean(ax, grid="x")
        ax.set_title(titles.get(str(m), f"{letters[i]}. {vlab[m]}: refit selection of the predictive component"), fontsize=10)
    return fig


@_themed
def plot_feature_map_comparison(res: Dict[str, Any], maps: Optional[Dict[str, Any]] = None,
                                inf: Optional[Dict[str, Any]] = None, component: int = 0, view: int = 0,
                                kinds: Optional[Sequence[str]] = None, top_n: int = 8, view_names=None,
                                feature_names=None, bars: Sequence[float] = STABILITY_BARS, figsize=None):
    """One component of one view read through several feature maps side by side.

    Rows are the union of each map's ``top_n`` features; columns are maps. Dot
    size and shade are |map| / max |map| within that map, and the numbers are each
    map's ranks (top 10). With ``inf = loading_inference(..., maps=...)`` each
    column header carries that map's component stability, coloured against
    ``bars``. ``structure`` is the fit's own column; the alternatives come from
    ``maps`` (`pysimlr.feature_maps`) or ``inf["maps"]``.
    """
    tab = feature_table(res, inf, view_names, feature_names, maps=maps)
    avail = _maps_of(inf, maps)
    dropped = []
    if kinds is None:
        kinds = ["structure"]
        dv = tab[(tab.view == view) & (tab.component == component + 1)]
        for k_ in avail:
            if k_ == "structure":
                continue
            if k_ == "component_partial" and np.nanmax(np.abs(dv[k_] - dv["structure"])) < 0.02:
                dropped.append(k_)          # equals structure here (uncorrelated components)
                continue
            kinds.append(k_)
    kinds = list(kinds)
    if len(kinds) < 2:
        raise ValueError("plot_feature_map_comparison needs at least two maps; pass maps= or "
                         "inf from loading_inference(maps=...).")
    for k_ in kinds:
        if k_ not in tab.columns:
            raise ValueError(f"map {k_!r} is not available; pass it in maps= or loading_inference(maps=...).")
    d = tab[(tab.view == view) & (tab.component == component + 1)].reset_index(drop=True)
    mags = {k_: np.nan_to_num(np.abs(d[k_].to_numpy()), nan=0.0) for k_ in kinds}
    ranks = {k_: np.argsort(np.argsort(-mags[k_])) for k_ in kinds}
    rows = []
    for k_ in kinds:
        for i in np.argsort(-mags[k_])[:top_n]:
            if i not in rows:
                rows.append(int(i))
    rows.sort(key=lambda i: np.mean([ranks[k_][i] for k_ in kinds]))
    color = _view_color(view)
    fig = _new(figsize or (1.45 * len(kinds) + 3.6, 0.36 * len(rows) + 2.4),
               f"Component {component + 1} of {d.view_name.iloc[0]}, read through {len(kinds)} feature maps",
               "Dot size: |map| / max |map| within that map; number: rank (top 10)"
               + ("; fill: selected in ≥ 80% of refits (pale: ≥ 50%, ring: below)" if inf is not None else "")
               + ". Header colour: stability ≥ 0.9 ink, ≥ 0.8 amber, below red."
               + (f" Not shown: {', '.join(dropped)} (equals structure here)." if dropped else ""),
               _status_line(res))
    ax = fig.subplots(1, 1)
    for x, k_ in enumerate(kinds):
        top = mags[k_].max() or 1.0
        freq = _sel_freq(d, k_)
        for y, i in enumerate(rows):
            v = mags[k_][i] / top
            size = 30 + 300 * v ** 1.5
            if freq is None:
                ax.scatter(x, y, s=size, color=color, alpha=0.22 + 0.78 * v, edgecolors="none", zorder=2)
                dark = v > 0.6
            elif freq[i] >= 0.8:
                ax.scatter(x, y, s=size, color=color, edgecolors=_SURFACE, linewidths=1.2, zorder=2); dark = True
            elif freq[i] >= 0.5:
                ax.scatter(x, y, s=size, color=color, alpha=0.4, edgecolors="none", zorder=2); dark = False
            else:
                ax.scatter(x, y, s=size, facecolor=_SURFACE, edgecolors=color, linewidths=1.3, zorder=2); dark = False
            if ranks[k_][i] < 10:
                ax.text(x, y, str(ranks[k_][i] + 1), ha="center", va="center", fontsize=7.2, zorder=3,
                        color=_SURFACE if dark else _INK2)
    heads = []
    for k_ in kinds:
        h = {"structure": "structure", "structure_other_views": "structure\n(other views)",
             "partial": "partial\n(inverse corr.)", "global_partial": "global-\npartial",
             "component_partial": "component-\npartial", "relative_weight": "relative\nweight"}.get(k_, k_)
        sc = "component_stability" if k_ == "structure" else f"{k_}_stability"
        if sc in d and np.isfinite(d[sc].iloc[0]):
            h += f"\nstab. {float(d[sc].iloc[0]):.2f}"
        heads.append(h)
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels(heads, fontsize=8.4); ax.xaxis.tick_top()
    for lbl, k_ in zip(ax.get_xticklabels(), kinds):
        sc = "component_stability" if k_ == "structure" else f"{k_}_stability"
        if sc in d and np.isfinite(d[sc].iloc[0]):
            lbl.set_color(_stability_color(float(d[sc].iloc[0]), bars))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(d.feature_name.iloc[rows], fontsize=8.4)
    ax.set_ylim(len(rows) - 0.4, -0.7); ax.set_xlim(-0.6, len(kinds) - 0.4)
    _clean(ax, grid=None)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    return fig


@_themed
def plot_feature_chord(res: Dict[str, Any], inf: Optional[Dict[str, Any]] = None, components=None,
                       top_n: int = 6, view_names=None, feature_names=None, figsize=(9.0, 8.4)):
    """Features to components across all views, as one chord diagram.

    Left half: each view's ``top_n`` features (largest |structure| over the
    shown components), grouped and coloured by view. Right half: the
    components. A ribbon joins feature i to component j with width
    proportional to |structure_ij|; opacity is higher for positive structure
    and lower for negative (the sign is also in the feature label's marker).
    With ``inf``, only features whose interval excludes 0 or that are
    stability-selected are drawn. Read it as: which views, and which of their
    features, build each component.
    """
    from matplotlib.patches import Wedge
    tab = feature_table(res, inf, view_names, feature_names)
    k = int(tab.component.max()); comps = list(range(1, k + 1)) if components is None else [c + 1 for c in components]
    M = int(tab.view.max()) + 1; vn = _view_names(view_names, M)
    feats = []
    for m in range(M):
        d = tab[(tab.view == m) & tab.component.isin(comps)]
        if inf is not None:
            d = d[d.selected | d.excludes_zero]
        if d.empty:
            continue
        # the strongest features of each component in turn, so every component is represented;
        # ordered by the component each feature loads on most, then by strength
        per = max(1, int(np.ceil(top_n / len(comps)))); chosen = []
        for c in comps:
            dc = d[d.component == c].assign(a=d[d.component == c].structure.abs()).sort_values("a", ascending=False)
            chosen += [f for f in dc.feature if f not in chosen][:per]
        dd = d[d.feature.isin(chosen)].assign(a=lambda t: t.structure.abs())
        best = dd.loc[dd.groupby("feature").a.idxmax()].set_index("feature")
        order = sorted(chosen, key=lambda f: (int(best.loc[f, "component"]), -float(best.loc[f, "a"])))
        for f in order[:top_n]:
            row = d[d.feature == f]
            feats.append((m, row.feature_name.iloc[0], {int(c): float(s) for c, s in zip(row.component, row.structure)}))
    fig = _new(figsize, "Features to components, across views",
               "Ribbon width: |structure| of the feature on the component. Faint ribbons: negative structure.",
               _status_line(res))
    ax = fig.add_subplot()
    nF, nC = len(feats), len(comps)
    gapF, gapC = 0.012, 0.05
    # features on the left half (from top, going counter-clockwise), with gaps between views
    left0, left1 = np.pi / 2 + 0.12, 3 * np.pi / 2 - 0.12
    n_breaks = len({m for m, _, _ in feats}) - 1
    fspan = (left1 - left0 - gapF * nF - 0.06 * max(n_breaks, 0)) / max(nF, 1)
    fpos, a, prev = [], left0, None
    for m, name, s in feats:
        if prev is not None and m != prev:
            a += 0.06
        fpos.append((a, a + fspan)); a += fspan + gapF; prev = m
    right0, right1 = -np.pi / 2 + 0.12, np.pi / 2 - 0.12
    tot = np.array([sum(abs(s.get(c, 0.0)) for _, _, s in feats) for c in comps])
    cspan_total = right1 - right0 - gapC * (nC - 1)
    cpos, a = [], right1
    for t in tot:
        w = cspan_total * t / max(tot.sum(), 1e-12); cpos.append((a - w, a)); a -= w + gapC
    # arcs
    for (a0, a1), (m, name, s) in zip(fpos, feats):
        ax.add_patch(Wedge((0, 0), 1.06, np.degrees(a0), np.degrees(a1), width=0.07, facecolor=_view_color(m),
                           edgecolor=_SURFACE, lw=0.8, zorder=3))
        mid = (a0 + a1) / 2; rot = np.degrees(mid) - 180
        ax.text(1.11 * np.cos(mid), 1.11 * np.sin(mid), name, rotation=rot, rotation_mode="anchor", ha="right",
                va="center", fontsize=8, color=_INK2)
    for (a0, a1), c in zip(cpos, comps):
        ax.add_patch(Wedge((0, 0), 1.06, np.degrees(a0), np.degrees(a1), width=0.07, facecolor=_MUTED,
                           edgecolor=_SURFACE, lw=1.2, zorder=3))
        mid = (a0 + a1) / 2
        ax.text(1.14 * np.cos(mid), 1.14 * np.sin(mid), f"C{c}", ha="left", va="center", fontsize=11,
                weight="semibold", color=_INK)
    # view labels around the left arcs
    for m in sorted({m for m, _, _ in feats}):
        idx = [i for i, (mm, _, _) in enumerate(feats) if mm == m]
        mid = (fpos[idx[0]][0] + fpos[idx[-1]][1]) / 2
        ax.add_patch(Wedge((0, 0), 1.0 + 0.005, np.degrees(fpos[idx[0]][0]), np.degrees(fpos[idx[-1]][1]),
                           width=0.012, facecolor=_view_color(m), alpha=0.5, zorder=2))
    # ribbons: each feature's span split across components, each component's span across features
    fcur = [p[1] for p in fpos]; ccur = [p[1] for p in cpos]
    ftot = [sum(abs(s.get(c, 0.0)) for c in comps) for _, _, s in feats]
    links = []
    for ci, c in enumerate(comps):
        for fi, (m, name, s) in enumerate(feats):
            v = s.get(c, 0.0)
            if abs(v) < 1e-9:
                continue
            wf = (fpos[fi][1] - fpos[fi][0]) * abs(v) / max(ftot[fi], 1e-12)
            wc = (cpos[ci][1] - cpos[ci][0]) * abs(v) / max(tot[ci], 1e-12)
            links.append((abs(v), (fcur[fi] - wf, fcur[fi]), (ccur[ci] - wc, ccur[ci]), m, v))
            fcur[fi] -= wf; ccur[ci] -= wc
    for w, fs, cs, m, v in sorted(links, key=lambda t: t[0]):
        _ribbon(ax, fs, cs, 0.985, _view_color(m), _view_color(m), alpha=(0.22 + 0.45 * w) if v > 0 else 0.10)
    ax.legend(handles=_legend_handles([(vn[m], _view_color(m)) for m in sorted({m for m, _, _ in feats})]),
              loc="lower left", bbox_to_anchor=(-0.02, -0.04), ncol=1)
    _chord_axes(ax); ax.set_xlim(-1.62, 1.42)
    fig._pysimlr_values = {"features": feats, "components": comps}
    return fig


@_themed
def plot_feature_groups(res: Dict[str, Any], groups, inf: Optional[Dict[str, Any]] = None, view: int = 0,
                        feature_names=None, statistic: str = "mean_abs", ax=None, view_names=None):
    """Interpretable features summarised by feature group (lobe, network, pathway...).

    ``groups``: one label per feature of ``view`` (or a dict feature name ->
    label). A dot matrix of groups x components: dot area and colour carry
    ``statistic="mean_abs"`` (mean |structure|), ``"max_abs"``, or, with
    ``inf``, ``"n_significant"`` (count of features whose 95% interval excludes
    0) or ``"n_selected"`` (count of stability-selected features; strict by
    design, so often small). Groups are ordered by their strongest component.
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
        agg, label = "sum", ("features selected" if statistic == "n_selected" else "intervals excluding 0")
    elif statistic in ("mean_abs", "max_abs"):
        d["v"] = d.structure.abs(); agg = statistic.split("_")[0]; label = f"{agg} |structure|"
    else:
        raise ValueError("statistic must be 'mean_abs', 'max_abs', 'n_significant' or 'n_selected'.")
    tbl = d.pivot_table(index="group", columns="component", values="v", aggfunc=agg)
    tbl = tbl.loc[tbl.max(1).sort_values(ascending=False).index]
    G, K = tbl.shape; V = tbl.to_numpy(float); vmax = float(np.nanmax(V)) if np.isfinite(V).any() else 1.0
    vname = _view_names(view_names, len(res["structure"]))[view]
    fig, ax = _figure(ax, (1.05 * K + 3.4, 0.42 * G + 2.0), f"Feature groups: {vname}",
                      f"Dot area and shade: {label} per group and component.")
    cm = _cmap("seq")
    for i in range(G):
        for j in range(K):
            v = V[i, j]
            if not np.isfinite(v):
                continue
            f = v / vmax if vmax > 0 else 0
            ax.scatter([j], [i], s=30 + 900 * f, color=cm(0.25 + 0.75 * f), edgecolor=_SURFACE, linewidth=1.2, zorder=2)
            ax.text(j + 0.3, i, f"{v:.0f}" if statistic.startswith("n_") else f"{v:.2f}", va="center", fontsize=8,
                    color=_INK2)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"C{c}" for c in tbl.columns])
    ax.set_yticks(range(G)); ax.set_yticklabels(tbl.index); ax.set_xlim(-0.6, K - 0.1); ax.set_ylim(G - 0.4, -0.6)
    _clean(ax, grid=None, baseline=False); ax.tick_params(axis="x", length=0)
    for i in range(G):
        ax.axhline(i, color=_GRID, lw=0.8, zorder=0)
    fig._pysimlr_values = {"table": tbl}
    return fig


@_themed
def plot_loadings(res: Dict[str, Any], view: int = 0, feature_names=None,
                  max_features: Optional[int] = 60, ax=None):
    """Loadings of one view: features x components, in the rotated frame.

    Positive fits (hard assignment) use the sequential scale and report the
    median ``loading_margin``; signed fits use the diverging scale, symmetric
    about 0. Cells are separated by surface gaps. With more than
    ``max_features`` features, shows those with the largest |loading|,
    ordered by component and then strength.
    """
    L = _np(res["loadings"][view]).astype(float); p, k = L.shape
    names = _view_feature_names(feature_names, view, p)
    rows = np.arange(p)
    if max_features is not None and p > max_features:
        rows = np.argsort(-np.abs(L).max(1))[:max_features]
    rows = sorted(rows, key=lambda r: (int(np.argmax(np.abs(L[r]))), -float(np.abs(L[r]).max())))
    signed = bool((L[rows] < -1e-9).any())
    margin = res.get("loading_margin")
    sub = ("Signed loadings, blue negative, red positive." if signed else "Hard-assigned loadings: each feature on one component.")
    if margin is not None and margin[view] is not None:
        sub += f" Median margin {float(np.median(_np(margin[view]))):.2f}."
    fig, ax = _figure(ax, (max(3.6, 0.75 * k + 3.0), max(4.0, 0.2 * len(rows) + 2.0)),
                      f"Loadings, view {view + 1}", sub)
    vmax = float(np.abs(L[rows]).max()) or 1.0
    mesh = _matrix(ax, L[rows], _cmap("div" if signed else "seq"), -vmax if signed else 0.0, vmax,
                   rows=[names[r] for r in rows], cols=[f"C{j + 1}" for j in range(k)], annotate=False)
    ax.xaxis.tick_top()
    _colorbar(fig, mesh, ax, "loading")
    return fig


@_themed
def plot_variance_explained(res: Dict[str, Any], views, pred: Optional[Dict[str, Any]] = None,
                            view_names=None, figsize=(10.5, 3.9)):
    """Reconstruction R^2 per view (training ring, held-out dot with ``pred``) and the
    share of each view's correlation structure carried by each component.

    Values below -0.25 are clipped and printed. Right: mean squared structure
    coefficient per component, as a segmented bar per view.
    """
    from .api import predict
    tr = np.asarray([float(_np(v)) for v in predict(res, views)["r2"]])
    M = len(tr); vn = _view_names(view_names, M); y = np.arange(M)
    fig = _new(figsize, "Variance explained",
               "Left: reconstruction R² (ring training, dot held-out). Right: structure carried by each component.",
               _status_line(res))
    axes = fig.subplots(1, 2)
    ax = axes[0]
    te = np.asarray([float(_np(v)) for v in pred["r2"]]) if pred is not None else None
    for m in range(M):
        if te is not None:
            ax.hlines(m, np.clip(min(tr[m], te[m]), -0.25, 1), np.clip(max(tr[m], te[m]), -0.25, 1),
                      color=_view_color(m), lw=2, alpha=0.5)
            _dots(ax, [np.clip(tr[m], -0.25, 1)], [m], _view_color(m), s=50, hollow=True)
            _dots(ax, [np.clip(te[m], -0.25, 1)], [m], _view_color(m), s=60)
            ax.text(np.clip(te[m], -0.25, 1), m - 0.3, f"{te[m]:.2f}", ha="center", va="bottom", fontsize=8,
                    color=_CRITICAL if te[m] < -0.25 else _INK2)
        else:
            _dots(ax, [np.clip(tr[m], -0.25, 1)], [m], _view_color(m), s=60)
            ax.text(np.clip(tr[m], -0.25, 1), m - 0.3, f"{tr[m]:.2f}", ha="center", va="bottom", fontsize=8, color=_INK2)
    ax.axvline(0, color=_AXIS, lw=0.9); ax.set_xlim(-0.28, 1.0)
    ax.set_yticks(y); ax.set_yticklabels(vn); ax.set_ylim(M - 0.5, -0.8); ax.set_xlabel("reconstruction R²")
    _clean(ax, grid="x"); ax.set_title("per view", fontsize=10.5)
    F = _aligned_frame(res); k = F["structure"][0].shape[1]
    share = np.array([[float((C[:, j] ** 2).mean()) for j in range(k)] for C in F["structure"]])
    ramp = [_BLUE[i] for i in np.linspace(9, 3, k).astype(int)]
    a2 = axes[1]; left = np.zeros(M)
    for j in range(k):
        a2.barh(y, share[:, j], left=left, height=0.42, color=ramp[j], edgecolor=_SURFACE, linewidth=2, label=f"C{j + 1}")
        for m in range(M):
            if share[m, j] > 0.06 * share.sum(1).max():
                a2.text(left[m] + share[m, j] / 2, m, f"C{j + 1}", ha="center", va="center", fontsize=7.5,
                        color=_cell_text_color(_rgba(ramp[j])))
        left += share[:, j]
    a2.set_yticks(y); a2.set_yticklabels([]); a2.set_ylim(M - 0.5, -0.8)
    a2.set_xlabel("mean squared structure coefficient"); _clean(a2, grid="x")
    a2.legend(ncol=k, loc="lower right", bbox_to_anchor=(1, 1.0)); a2.set_title("by component", fontsize=10.5)
    return fig


@_themed
def plot_scores(res: Dict[str, Any], components=(0, 1), color_by=None, new_views=None,
                ax=None, color_label: str = "colour value"):
    """Subject scores in the rotated frame (``u @ R``), with optional held-out subjects.

    ``new_views`` are projected with `pysimlr.predict` and drawn as ringed
    dots. ``color_by`` is one value per training subject: numeric values use
    the sequential scale (with a colour bar); labels use the categorical slots.
    Marginal densities sit on the two outer edges.
    """
    a, b = components
    U = _rotated_scores(res)
    V = None
    if new_views is not None:
        from .api import predict
        V = _rotated_scores(res, predict(res, new_views)["u"])
    own = ax is None
    sub = (f"Components {a + 1} and {b + 1} in the rotated frame u R"
           + ("; held-out subjects ringed." if V is not None else "."))
    if own:
        fig = _new((6.4, 6.0), "Subject scores", sub)
        gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5))
        ax = fig.add_subplot(gs[1, 0]); top = fig.add_subplot(gs[0, 0], sharex=ax)
        right = fig.add_subplot(gs[1, 1], sharey=ax); corner = fig.add_subplot(gs[0, 1]); corner.axis("off")
    else:
        fig = ax.figure
    cb = None if color_by is None else np.asarray(color_by)
    if cb is None:
        ax.scatter(U[:, a], U[:, b], s=22, color=VIEW_COLORS[0], alpha=0.55, linewidth=0, zorder=2)
    elif cb.dtype.kind in "fiu":
        sc = ax.scatter(U[:, a], U[:, b], s=24, c=cb, cmap=_cmap("seq"), edgecolor=_SURFACE, linewidth=0.5, zorder=2)
        if own:
            cax = corner.inset_axes([0.08, 0.42, 0.84, 0.16])
            cbar = fig.colorbar(sc, cax=cax, orientation="horizontal"); cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0, labelsize=7.5, labelcolor=_INK2)
            cax.set_title(color_label, fontsize=8, color=_INK2, loc="left", pad=3)
        else:
            _colorbar(fig, sc, ax, color_label)
    else:
        for i, lvl in enumerate(dict.fromkeys(cb.tolist())):
            mk = cb == lvl
            ax.scatter(U[mk, a], U[mk, b], s=24, color=VIEW_COLORS[i % 8], alpha=0.75, linewidth=0, label=str(lvl), zorder=2)
        ax.legend(loc="best")
    if V is not None:
        ax.scatter(V[:, a], V[:, b], s=46, facecolor="none", edgecolor=VIEW_COLORS[1], linewidth=1.5, zorder=3,
                   label="held-out")
    _clean(ax, grid="both")
    ax.set_xlabel(f"component {a + 1}"); ax.set_ylabel(f"component {b + 1}")
    if own:
        from scipy.stats import gaussian_kde
        for axm, col, horiz in ((top, a, False), (right, b, True)):
            vals = U[:, col]; g = np.linspace(vals.min() - 0.1 * np.ptp(vals), vals.max() + 0.1 * np.ptp(vals), 200)
            dens = gaussian_kde(vals)(g) if np.ptp(vals) > 0 else np.zeros_like(g)
            if horiz:
                axm.fill_betweenx(g, 0, dens, color=VIEW_COLORS[0], alpha=0.18, lw=0); axm.plot(dens, g, color=VIEW_COLORS[0], lw=1.2)
            else:
                axm.fill_between(g, 0, dens, color=VIEW_COLORS[0], alpha=0.18, lw=0); axm.plot(g, dens, color=VIEW_COLORS[0], lw=1.2)
            axm.axis("off")
    return fig


@_themed
def plot_latent_recovery(res: Dict[str, Any], z_true, ax=None):
    """|corr| between rotated components and planted latents (synthetic studies)."""
    U = _rotated_scores(res); Z = np.asarray(_np(z_true), float)
    k, q = U.shape[1], Z.shape[1]
    C = np.abs(np.corrcoef(U.T, Z.T)[:k, k:])
    fig, ax = _figure(ax, (0.95 * q + 3.2, 0.8 * k + 2.4), "Latent recovery",
                      "|corr| of each fitted component with each planted latent.")
    mesh = _matrix(ax, C, _cmap("seq"), 0, 1, rows=[f"C{i + 1}" for i in range(k)], cols=[f"z{j + 1}" for j in range(q)])
    ax.xaxis.tick_top()
    fig._pysimlr_values = {"abs_corr": C}
    return fig


# ----------------------------------------------------------------------------- one-call sheet
@_themed
def plot_fit_summary(res: Dict[str, Any], views=None, pred: Optional[Dict[str, Any]] = None,
                     inf: Optional[Dict[str, Any]] = None, view_names=None, feature_names=None,
                     top_n: int = 8, save_path=None, figsize=(13.5, 9.6)):
    """One diagnostic sheet for a fit: convergence, basis rank, per-view
    coupling (training and held-out), the view chord of the most coupled
    component, and that component's top features in the first two views (with
    intervals when ``inf`` is given). Saved to ``save_path`` if given.
    """
    F = _aligned_frame(res); vn = _view_names(view_names, len(F["structure"]))
    fig = _new(figsize, f"{res.get('method', 'fit')} summary",
               "Optimisation, basis, coupling and the features of the most strongly coupled component.",
               _status_line(res))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.15))
    _draw_convergence(res, fig.add_subplot(gs[0, 0]))
    if res.get("effective_rank") is not None:
        _draw_effective_rank(res, fig.add_subplot(gs[0, 1]), view_names)
    ax = fig.add_subplot(gs[0, 2])
    if F["joint"]:
        ax.axis("off"); ax.text(0, 0.5, "joint latent: no per-view coupling", color=_INK2); j_top = 0
    else:
        tr = _coupling(res); te = _coupling(res, pred) if pred is not None else None
        j_top = int(np.nanargmax(np.nanmean(tr, 0)))
        show = te if te is not None else tr; M, k = show.shape
        for m in range(M):
            for c in range(k):
                if te is not None:
                    ax.plot([c - 0.18, c + 0.18], [tr[m, c], te[m, c]], color=_view_color(m), lw=1.2, alpha=0.5)
                    _dots(ax, [c - 0.18], [tr[m, c]], _view_color(m), s=34, hollow=True)
                _dots(ax, [c + (0.18 if te is not None else 0)], [show[m, c]], _view_color(m), s=44)
        ax.set_xticks(range(k)); ax.set_xticklabels([f"C{c + 1}" for c in range(k)])
        both = np.r_[tr.ravel(), show.ravel()]
        ax.set_ylim(min(0, np.nanmin(both) - 0.06), 1.04); ax.set_ylabel("r with consensus")
        _clean(ax, grid="y"); ax.tick_params(axis="x", length=0)
        ax.set_title("Coupling " + ("(ring training, dot held-out)" if te is not None else "(training)"), fontsize=10.5)
        ax.legend(handles=_legend_handles([(vn[m], _view_color(m)) for m in range(M)]), loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), ncol=min(M, 4))
    bottom = gs[1, :].subgridspec(1, 3, width_ratios=(1.05, 1, 1))
    if not F["joint"]:
        _draw_view_chord(fig.add_subplot(bottom[0]), _pairwise(res, pred)[j_top], vn,
                         [_view_color(m) for m in range(len(vn))], center=f"C{j_top + 1}" + (" held-out" if pred is not None else ""))
    tab = feature_table(res, inf, view_names, feature_names)
    null = inf.get("null_max_abs") if inf is not None else None
    for c_i, m in enumerate(range(min(2, len(F["structure"])))):
        _draw_top_features(fig.add_subplot(bottom[1 + c_i]), tab, m, j_top, "structure", top_n, "magnitude",
                           null, STABILITY_BARS, _view_color(m))
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------- benchmarks
@_themed
def plot_benchmark(df, x: str, y: str, by: Optional[str] = "model", kind: str = "line", ax=None):
    """One benchmark metric against a setting, per method, from a results DataFrame.

    ``kind="line"``: mean with a +-1 SE band at each ``x``, one line per ``by``,
    labelled at its end. ``kind="scatter"``: every row. ``kind="bar"``: grouped
    thin bars with SE whiskers. Methods take the categorical slots in order of
    first appearance.
    """
    if kind not in ("line", "scatter", "bar"):
        raise ValueError(f"kind must be 'line', 'scatter' or 'bar'; got {kind!r}.")
    missing = [c for c in (x, y, by) if c is not None and c not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing}; available: {list(df.columns)}")
    groups = [(None, df)] if by is None else list(df.groupby(by, sort=False))
    fig, ax = _figure(ax, (6.6, 4.4), f"{y} by {x}",
                      "Mean ± 1 standard error over repeats." if kind != "scatter" else "Every run.")
    xs = list(dict.fromkeys(df[x].tolist()))
    cat = df[x].dtype == object
    for gi, (name, d) in enumerate(groups):
        lbl = None if name is None else str(name); col = VIEW_COLORS[gi % 8]
        if kind == "scatter":
            ax.scatter(d[x], d[y], s=26, color=col, alpha=0.7, edgecolor=_SURFACE, linewidth=0.6, label=lbl)
            continue
        st = d.groupby(x, sort=False)[y].agg(["mean", "std", "count"]).reindex(xs)
        se = (st["std"] / np.sqrt(st["count"])).fillna(0.0).to_numpy(); mu = st["mean"].to_numpy()
        if kind == "line":
            xx = np.arange(len(xs)) if cat else np.asarray(xs, float)
            ax.fill_between(xx, mu - se, mu + se, color=col, alpha=0.12, lw=0)
            ax.plot(xx, mu, color=col, lw=2, label=lbl); _dots(ax, xx, mu, col, s=40)
            if lbl is not None:
                ax.text(xx[-1], mu[-1], f"  {lbl}", va="center", fontsize=9, color=_INK2)
        else:
            w = min(0.6 / max(1, len(groups)), 0.16)
            pos = np.arange(len(xs)) + (gi - (len(groups) - 1) / 2) * (w + 0.03)
            ax.bar(pos, mu, width=w, color=col, label=lbl, edgecolor=_SURFACE, linewidth=0)
            ax.errorbar(pos, mu, yerr=se, fmt="none", ecolor=_INK2, elinewidth=1, capsize=0)
    if kind == "bar" or (kind == "line" and cat):
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(v) for v in xs])
    if kind == "line" and not cat:
        ax.set_xticks(xs); ax.set_xlim(min(xs) - 0.05 * np.ptp(xs), max(xs) + 0.22 * np.ptp(xs))
    ax.set_xlabel(x); ax.set_ylabel(y); _clean(ax, grid="y")
    if by is not None and kind != "line":
        ax.legend(title=by, loc="best")
    return fig


@_themed
def plot_benchmark_uncertainty(df, method: str = "method", split: str = "split", score: str = "score",
                               baseline: Optional[str] = None, n_boot: int = 2000, random_state: int = 0,
                               metric_label: Optional[str] = None, figsize=(11.0, None)):
    """Out-of-sample comparison with its uncertainty: per method, the mean score
    with a 95% bootstrap interval over splits and every split shown; and, with
    ``baseline``, the paired difference (method - baseline) per split with its
    bootstrap interval. Intervals that exclude 0 are drawn solid and in colour;
    those that include it are dashed grey.

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
    h = figsize[1] or 0.55 * len(methods) + 2.4
    fig = _new((figsize[0], h), "Out-of-sample comparison",
               f"Mean {metric_label or score} with a 95% bootstrap interval; dots are splits."
               + (f" Right: paired difference against {baseline}." if baseline else ""))
    axes = np.atleast_1d(fig.subplots(1, 2 if baseline else 1))
    y = np.arange(len(methods))
    ax = axes[0]
    for i, mth in enumerate(methods):
        x = piv[mth].dropna().to_numpy(); lo, hi = boot(x); col = VIEW_COLORS[i % 8]
        ax.scatter(x, np.full(len(x), i) + rng.uniform(-0.17, 0.17, len(x)), s=16, color=col, alpha=0.35,
                   linewidth=0, zorder=1)
        ax.hlines(i, lo, hi, color=col, lw=4, zorder=2, capstyle="round")
        _dots(ax, [x.mean()], [i], _INK, s=54, zorder=4)
        ax.text(hi, i - 0.3, f" {x.mean():.3f}", fontsize=8.5, color=_INK2, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(methods); ax.set_ylim(len(methods) - 0.5, -0.7)
    ax.set_xlabel(metric_label or score); _clean(ax, grid="x"); ax.set_title("score per method", fontsize=10.5)
    if baseline:
        if baseline not in piv:
            raise ValueError(f"baseline {baseline!r} not among methods {methods}.")
        ax = axes[1]
        for i, mth in enumerate(methods):
            d = (piv[mth] - piv[baseline]).dropna().to_numpy()
            if mth == baseline or len(d) == 0:
                ax.text(0, i, "baseline", ha="center", va="center", fontsize=8.5, color=_MUTED)
                continue
            lo, hi = boot(d); excl = lo > 0 or hi < 0
            ax.hlines(i, lo, hi, color=VIEW_COLORS[i % 8] if excl else _MUTED, lw=4 if excl else 2,
                      linestyles="solid" if excl else "dashed")
            _dots(ax, [d.mean()], [i], _INK if excl else _MUTED, s=50, zorder=4)
            ax.text(hi, i - 0.3, f" {d.mean():+.3f}", fontsize=8.5, color=_INK2 if excl else _MUTED, va="bottom")
        ax.axvline(0, color=_INK2, lw=1); ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(len(methods) - 0.5, -0.7)
        ax.set_xlabel(f"paired difference vs {baseline}"); _clean(ax, grid="x")
        ax.set_title("paired difference (solid: interval excludes 0)", fontsize=10.5)
    return fig


@_themed
def plot_predictions(y_true, y_pred, ax=None, title: str = "Predicted vs observed"):
    """Predictions against observations, with the identity line and r.

    Binary observations are drawn as two jittered strips with their means.
    """
    y_true = np.asarray(y_true, float).ravel(); y_pred = np.asarray(y_pred, float).ravel()
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    fig, ax = _figure(ax, (5.0, 4.8), title, f"r = {r:.3f}, n = {len(y_true)}")
    binary = set(np.unique(y_true)) <= {0.0, 1.0}
    if binary:
        jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(y_true))
        for v, col in ((0.0, VIEW_COLORS[0]), (1.0, VIEW_COLORS[1])):
            mk = y_true == v
            ax.scatter(v + jit[mk], y_pred[mk], s=18, color=col, alpha=0.45, linewidth=0)
            if mk.any():
                ax.hlines(y_pred[mk].mean(), v - 0.25, v + 0.25, color=_INK, lw=2)
        ax.set_xticks([0, 1]); ax.set_xlim(-0.5, 1.5)
    else:
        ax.scatter(y_true, y_pred, s=20, color=VIEW_COLORS[0], alpha=0.55, linewidth=0)
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], color=_INK2, lw=1)
    if ax.get_title() == "" and fig.texts == []:
        ax.set_title(f"{title} (r = {r:.3f})")
    ax.set_xlabel("observed"); ax.set_ylabel("predicted"); _clean(ax, grid="both")
    return fig


# ----------------------------------------------------------------------------- fused ridge
def _feature_names(result: Dict[str, Any], p: int):
    names = result.get("feature_names")
    return list(names) if names is not None else [f"x{i}" for i in range(p)]


def _task_labels(result: Dict[str, Any]):
    pre = "y ≥ " if result.get("model") == "thresholds" else ""
    return [f"{pre}{t:.3g}" if isinstance(t, float) else f"{pre}{t}" for t in result["tasks"]]


def _fused_subtitle(result):
    s = f"{result['family']}, λ₁ = {result['lambda1']:g}, λ₂ = {result['lambda2']:g}"
    return s + (f", top_k = {result['top_k']}" if result.get("top_k") else "")


@_themed
def plot_fused_coefficients(result: Dict[str, Any], ax=None, max_features: Optional[int] = 40):
    """A fused ridge's standardised coefficients, tasks (rows) x features, on the
    diverging scale with surface gaps between cells.

    Tasks are groups (strata) or outcome thresholds. With more than
    ``max_features`` features, shows those with the largest maximum
    |coefficient|; with ``top_k``, only the selected features.
    """
    w = np.asarray(result["coef_std"]); names = _feature_names(result, w.shape[1])
    cols = np.asarray(result.get("selected_features", range(w.shape[1])))
    if max_features is not None and len(cols) > max_features:
        cols = np.sort(cols[np.argsort(-np.abs(w[:, cols]).max(0))[:max_features]])
    fig, ax = _figure(ax, (max(5.5, 0.55 * len(cols) + 3), max(3.0, 0.45 * w.shape[0] + 2.0)),
                      "Fused coefficients", _fused_subtitle(result))
    vmax = float(np.abs(w[:, cols]).max()) or 1.0
    mesh = _matrix(ax, w[:, cols], _cmap("div"), -vmax, vmax, rows=_task_labels(result),
                   cols=[names[c] for c in cols], annotate=len(cols) * w.shape[0] <= 80)
    ax.set_xticklabels([names[c] for c in cols], rotation=45 if len(cols) > 6 else 0, ha="right" if len(cols) > 6 else "center")
    _colorbar(fig, mesh, ax, "standardised coefficient")
    return fig


@_themed
def plot_coefficient_paths(result: Dict[str, Any], features: Optional[Sequence] = None,
                           top: int = 8, ax=None):
    """Each feature's standardised coefficient across tasks (groups or thresholds, in order),
    labelled at its end. Fusion shows here: with larger lambda2 the paths flatten
    toward one shared value.

    ``features`` are names or column indices; by default the ``top`` features
    by maximum |coefficient|.
    """
    w = np.asarray(result["coef_std"]); names = _feature_names(result, w.shape[1])
    if features is None:
        cand = np.asarray(result.get("selected_features", range(w.shape[1])))  # top_k: selected only
        cols = list(cand[np.argsort(-np.abs(w[:, cand]).max(0))[:top]])
    else:
        cols = [names.index(f) if isinstance(f, str) else int(f) for f in features]
    fig, ax = _figure(ax, (6.8, 4.4), "Coefficient paths", _fused_subtitle(result))
    x = np.arange(w.shape[0])
    ly = _dodge(w[-1, cols], 0.055 * (np.ptp(w[:, cols]) + 1e-9))
    for i, c in enumerate(cols):
        col = VIEW_COLORS[i % 8]
        ax.plot(x, w[:, c], color=col, lw=2, label=names[c]); _dots(ax, x, w[:, c], col, s=36)
        ax.plot([x[-1] + 0.04, x[-1] + 0.14], [w[-1, c], ly[i]], color=_AXIS, lw=0.8)
        ax.text(x[-1] + 0.16, ly[i], names[c], va="center", fontsize=8.5, color=_INK2)
    ax.axhline(0.0, color=_AXIS, lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(_task_labels(result))
    ax.set_xlabel("threshold" if result.get("model") == "thresholds" else "group")
    ax.set_ylabel("standardised coefficient"); _clean(ax, grid="y"); ax.set_xlim(-0.3, w.shape[0] + 0.25)
    return fig


@_themed
def plot_fused_ridge_cv(cv: Dict[str, Any], ax=None):
    """Cross-validated mean loss over (lambda1, lambda2) at the chosen top_k; best cell ringed."""
    k = list(cv["top_ks"]).index(cv["best"][2])
    loss = np.asarray(cv["loss"])[:, :, k]
    fig, ax = _figure(ax, (5.8, 4.2), "Cross-validated loss",
                      f"{cv['family']} deviance, top_k = {cv['best'][2]}; lower is better.")
    from matplotlib.colors import LinearSegmentedColormap
    rev = LinearSegmentedColormap.from_list("pysimlr_seq_r", _cmap("seq")(np.linspace(1, 0, 256)))
    mesh = _matrix(ax, loss, rev, float(np.nanmin(loss)), float(np.nanmax(loss)), fmt="{:.3f}",
                   rows=[f"{v:g}" for v in cv["lambda1s"]], cols=[f"{v:g}" for v in cv["lambda2s"]])
    i = list(cv["lambda1s"]).index(cv["best"][0]); j = list(cv["lambda2s"]).index(cv["best"][1])
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec=_INK, lw=2.2, zorder=5))
    ax.set_xlabel("λ₂ (fusion)"); ax.set_ylabel("λ₁ (ridge)")
    _colorbar(fig, mesh, ax, "mean held-out loss")
    return fig


# ----------------------------------------------------------------------------- score-level path model
def _format_p(p: Optional[float], b_sweeps: Optional[int] = None) -> str:
    """Format permutation p-value cleanly avoiding raw strings like 0.000999."""
    if p is None or not np.isfinite(p):
        return "N/A"
    p = float(p)
    if b_sweeps and b_sweeps > 0 and p <= (1.0001 / (b_sweeps + 1)):
        floor_val = 1.0 / (b_sweeps + 1)
        if floor_val < 0.001:
            return f"p < {floor_val:.1g}"
        return "p < 0.001"
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3g}"


@_themed
def plot_path_model(pm: Dict[str, Any], component: Optional[int] = None, alpha: float = 0.05,
                    ax=None, node_labels: Optional[Dict[str, str]] = None) -> Any:
    """Directed/staged SEM path model layout with auto-sized rounded node cards.

    When ``component=None`` (default), edges show omnibus association across
    components (Pillai trace, width in 0.6..6 pt), solid if ``p_maxt < alpha``,
    dashed otherwise. Significant present edges follow the sequential ``_BLUE``
    ramp. Absent-but-significant edges are drawn in ``_WARN``.
    When ``component=j``, edge widths show per-component ``|r_j|`` with sign colour
    (``_BLUE[6]`` positive, ``_RED[4]`` negative). Moderated edges (Holm p < alpha)
    carry an indicator mark. A critical status line is displayed when the graph-fit
    test rejects (``pm["fit"]["rejected"] == True``).

    Parameters
    ----------
    pm : dict
        Result of `pysimlr.path_model`.
    component : int or float, optional
        Component index (0-based) to plot signed partial correlations for. If a
        float < 1.0 is passed positionally as the second argument, it is interpreted
        as `alpha`.
    alpha : float, default=0.05
        Significance threshold for edge testing and moderation.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If None, a new Figure is created.
    node_labels : dict, optional
        Mapping from raw node names to human-readable labels
        (e.g. ``{"T1Hier_vol": "Regional Volume", "outcome": "Cognition\\n(CogPC1)"}``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(component, (float, np.floating)) and 0.0 < component < 1.0:
        alpha = float(component)
        component = None
    elif component is not None:
        component = int(component)

    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D

    nodes = list(pm["nodes"])
    m = len(nodes)

    # Directed / staged SEM layout: imaging views on left arc, outcome on right
    if "outcome" in nodes:
        img_nodes = [nm for nm in nodes if nm not in ("outcome", "consensus")]
        pos = {}
        k_img = len(img_nodes)
        if k_img == 1:
            pos[img_nodes[0]] = (-0.85, 0.0)
        elif k_img == 2:
            pos[img_nodes[0]] = (-1.15, 0.65)
            pos[img_nodes[1]] = (-0.35, -0.65)
        elif k_img == 3:
            pos[img_nodes[0]] = (-1.15, 0.75)
            pos[img_nodes[1]] = (-0.35, 0.0)
            pos[img_nodes[2]] = (-1.15, -0.75)
        else:
            for i, nm in enumerate(img_nodes):
                yi = 0.85 - 1.70 * i / (k_img - 1)
                xi = -1.15 if (i == 0 or i == k_img - 1) else (-0.35 if i % 2 == 1 else -0.65)
                pos[nm] = (xi, yi)
        pos["outcome"] = (1.25, 0.0)
        if "consensus" in nodes:
            pos["consensus"] = (0.40, 0.0)
    else:
        ang = {nm: np.pi / 2 + 2 * np.pi * i / m for i, nm in enumerate(nodes)}
        pos = {nm: (np.cos(a), np.sin(a)) for nm, a in ang.items()}

    col = {nm: (_INK if nm == "outcome" else _view_color(i)) for i, nm in enumerate(nodes)}

    fit = pm.get("fit") or {}
    status = None
    if fit.get("rejected"):
        max_abs_p = fit.get("max_abs_p")
        p_str = _format_p(max_abs_p) if max_abs_p is not None else f"< {alpha}"
        status = f"Graph-fit test rejected: absent edges carry significant association (max absent p = {p_str})"

    params = pm.get("parameters") if isinstance(pm.get("parameters"), dict) else {}
    null_name = params.get("null", "fixed")
    b_sweeps = params.get("n_permutations", 0)
    ev_label = fit.get("evidence")
    if ev_label is None and isinstance(pm.get("edges"), pd.DataFrame) and "null_rows" in pm["edges"].columns and len(pm["edges"]):
        ev_label = str(pm["edges"]["null_rows"].iloc[0])
    ev_str = f", evidence = {ev_label}" if ev_label else ""
    omnibus_p = pm.get("omnibus", {}).get("p_max") if isinstance(pm.get("omnibus"), dict) else None
    omni_str = f"; omnibus {_format_p(omnibus_p, b_sweeps)}" if omnibus_p is not None else ""
    weight_desc = f"|r| (component {component + 1})" if component is not None else "Pillai's trace V"
    title = f"Fitted path model: component {component + 1}" if component is not None else "Fitted path model"
    subtitle = (f"{pm.get('rows', 'heldout')} rows, n = {pm.get('n', '?')}, conditioning = {pm.get('conditioning', '?')}, "
                f"null = {null_name}{ev_str}, B = {b_sweeps}; edge weights = {weight_desc}{omni_str}")

    fig, ax = _figure(ax, figsize=(7.0, 5.8), title=title, subtitle=subtitle, status=status)

    comps = pm.get("edge_components")
    # Moderation lookup
    mod_edges = set()
    if pm.get("moderation") is not None and len(pm["moderation"]):
        for _, mr in pm["moderation"].iterrows():
            if mr.get("component") == "sum" and mr.get("p_holm") is not None and mr["p_holm"] < alpha:
                mod_edges.add((mr["a"], mr["b"]))
                mod_edges.add((mr["b"], mr["a"]))

    edges_df = pm["edges"]
    had_warn = False
    had_hyp_nonsig = False
    had_sig = False
    had_pos = False
    had_neg = False
    had_mod = False

    callout_list = []

    for e in edges_df.to_dict("records"):
        a, b = e["a"], e["b"]
        (x1, y1), (x2, y2) = pos[a], pos[b]
        sig = e.get("p_maxt") is not None and e["p_maxt"] < alpha

        r_val = 0.0
        comp_p = None
        if component is not None and comps is not None and len(comps):
            crow = comps[(((comps.a == a) & (comps.b == b)) | ((comps.a == b) & (comps.b == a))) &
                         (comps.component == component)]
            if len(crow):
                r_val = float(crow.iloc[0]["r"])
                comp_p = crow.iloc[0].get("p_maxt")
                if comp_p is not None:
                    sig = comp_p < alpha

        if not e["present"] and not sig:
            continue

        if component is not None:
            width = 0.6 + 5.4 * min(1.0, abs(r_val))
            if sig and not e["present"]:
                color = _WARN
                ls = "-"
                al = 0.95
                had_warn = True
            elif sig:
                color = _BLUE[6] if r_val >= 0 else _RED[4]
                ls = "-"
                al = 0.95
                if r_val >= 0:
                    had_pos = True
                else:
                    had_neg = True
            else:
                color = _INK2
                ls = (0, (3, 2.5))
                al = 0.55
                had_hyp_nonsig = True
        else:
            pillai_val = float(e.get("pillai", 0.0))
            width = 0.6 + 5.4 * min(1.0, max(0.0, pillai_val))
            if sig and not e["present"]:
                color = _WARN
                ls = "-"
                al = 0.95
                had_warn = True
            elif sig:
                # Map significant present edges along _BLUE sequential palette by Pillai magnitude
                c_idx = min(9, max(3, int(round(3 + 6 * min(1.0, max(0.0, pillai_val))))))
                color = _BLUE[c_idx]
                ls = "-"
                al = 0.95
                had_sig = True
            else:
                color = _INK2
                ls = (0, (3, 2.5))
                al = 0.55
                had_hyp_nonsig = True

        dx, dy = x2 - x1, y2 - y1
        nrm = np.hypot(dx, dy) or 1.0
        nx, ny = -dy / nrm, dx / nrm
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Curved chords with quadratic Bezier
        rad = 0.0
        if "outcome" in nodes:
            if x1 < -0.8 and x2 < -0.8:
                # Top to bottom view chord curves leftward away from middle view
                rad = -0.22 if nx > 0 else 0.22
            elif (x1 > 0.8 or x2 > 0.8) and (y1 > 0.3 or y2 > 0.3):
                # Top view to outcome arcs upward
                rad = 0.16 if ny > 0 else -0.16
            elif (x1 > 0.8 or x2 > 0.8) and (y1 < -0.3 or y2 < -0.3):
                # Bottom view to outcome arcs downward
                rad = -0.16 if ny > 0 else 0.16

        cx = mx + rad * nrm * nx
        cy = my + rad * nrm * ny
        t_arr = np.linspace(0, 1, 51)
        bx = (1 - t_arr) ** 2 * x1 + 2 * (1 - t_arr) * t_arr * cx + t_arr ** 2 * x2
        by = (1 - t_arr) ** 2 * y1 + 2 * (1 - t_arr) * t_arr * cy + t_arr ** 2 * y2

        ax.plot(bx, by, color=color, lw=width, ls=ls, alpha=al,
                solid_capstyle="round", zorder=1)

        is_mod = (a, b) in mod_edges or (b, a) in mod_edges
        if is_mod:
            had_mod = True
            ax.plot([bx[25]], [by[25]], marker="^", markersize=0, color="none", zorder=0)

        if component is not None:
            p_display = comp_p if comp_p is not None else e.get("p_maxt")
            label_text = f"r = {r_val:+.2f}\n{_format_p(p_display, b_sweeps)}"
        else:
            p_display = e.get("p_maxt", e.get("p_perm"))
            pillai_val = float(e.get("pillai", 0.0))
            label_text = f"V = {pillai_val:.2f}\n{_format_p(p_display, b_sweeps)}"

        if is_mod:
            label_text += "\n▲ mod"

        init_cx = bx[25] + nx * 0.16
        init_cy = by[25] + ny * 0.16
        callout_list.append({"x": init_cx, "y": init_cy, "text": label_text})

    # Bounding-box repulsion to prevent edge callout collisions
    w_box, h_box = 0.44, 0.26
    for _ in range(40):
        for cr in callout_list:
            for n_pos in pos.values():
                dx_n = cr["x"] - n_pos[0]
                dy_n = cr["y"] - n_pos[1]
                dn = np.hypot(dx_n, dy_n)
                if 0 < dn < 0.45:
                    push = (0.45 - dn) / dn * 0.25
                    cr["x"] += dx_n * push
                    cr["y"] += dy_n * push
        for j in range(len(callout_list)):
            for k in range(j + 1, len(callout_list)):
                dx_c = callout_list[k]["x"] - callout_list[j]["x"]
                dy_c = callout_list[k]["y"] - callout_list[j]["y"]
                if abs(dx_c) < w_box and abs(dy_c) < h_box:
                    sx = np.sign(dx_c) if dx_c != 0 else 1.0
                    sy = np.sign(dy_c) if dy_c != 0 else 1.0
                    ox = (w_box - abs(dx_c)) * sx * 0.25
                    oy = (h_box - abs(dy_c)) * sy * 0.25
                    callout_list[k]["x"] += ox
                    callout_list[k]["y"] += oy
                    callout_list[j]["x"] -= ox
                    callout_list[j]["y"] -= oy

    for cr in callout_list:
        ax.text(cr["x"], cr["y"], cr["text"], ha="center", va="center",
                fontsize=8, color=_INK, weight="medium",
                bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.25", fc=_SURFACE, ec=_GRID, lw=0.7, alpha=0.96),
                zorder=3)

    # Auto-sized rounded node cards
    for i, nm in enumerate(nodes):
        x, y = pos[nm]
        disp_nm = node_labels.get(nm, nm) if node_labels else nm
        if nm == "outcome":
            ax.text(x, y, disp_nm, ha="center", va="center", fontsize=9.0, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.35", fc=_INK, ec=_INK, lw=2.0), zorder=5)
        elif nm == "consensus":
            ax.text(x, y, disp_nm, ha="center", va="center", fontsize=9.0, color=_INK, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.35", fc=_SURFACE, ec=_INK, lw=2.0), zorder=5)
        else:
            c = col[nm]
            rgb = np.array(to_rgb(c))
            tint = tuple(0.14 * rgb + 0.86 * np.ones(3))
            ax.text(x, y, disp_nm, ha="center", va="center", fontsize=9.0, color=_INK, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.35", fc=tint, ec=c, lw=2.0), zorder=5)

    handles = []
    if component is not None:
        if had_pos:
            handles.append(Line2D([], [], color=_BLUE[6], lw=2.5, label=f"positive r (component {component + 1})"))
        if had_neg:
            handles.append(Line2D([], [], color=_RED[4], lw=2.5, label=f"negative r (component {component + 1})"))
        if had_hyp_nonsig:
            handles.append(Line2D([], [], color=_INK2, lw=2.0, ls=(0, (3, 2.5)), alpha=0.6, label="hypothesised, not significant"))
        if had_warn:
            handles.append(Line2D([], [], color=_WARN, lw=2.5, label="not hypothesised, significant"))
    else:
        handles.append(Line2D([], [], color=_BLUE[6], lw=2.5, label=f"edge, max-T p < {alpha}"))
        handles.append(Line2D([], [], color=_INK2, lw=2.0, ls=(0, (3, 2.5)), alpha=0.6, label="hypothesised, not significant"))
        if had_warn:
            handles.append(Line2D([], [], color=_WARN, lw=2.5, label="not hypothesised, significant"))

    if had_mod:
        handles.append(Line2D([], [], marker="^", color=_INK, markeredgecolor=_SURFACE, markersize=8,
                              ls="none", label=f"moderated edge (Holm p < {alpha})"))

    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=8)
    ax.set_xlim(-1.65, 1.70); ax.set_ylim(-1.30, 1.30); ax.set_aspect("equal"); ax.axis("off")
    return fig


@_themed
def plot_edge_matrix(pm: Dict[str, Any], what: str = "r", alpha: float = 0.05,
                     node_labels: Optional[Dict[str, str]] = None) -> Any:
    """Per-component edge association matrix (r or beta) with Pillai omnibus column.

    Parameters
    ----------
    pm : dict
        Result of `pysimlr.path_model`.
    what : {"r", "beta"}, default="r"
        Quantity to display in the main heatmap: partial correlation "r" or
        standardized regression slope "beta".
    alpha : float, default=0.05
        False discovery rate threshold for hatching cells (BH q >= alpha).
    node_labels : dict, optional
        Mapping of raw node identifiers to human-readable display labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(what, (float, np.floating)):
        alpha = float(what)
        what = "r"

    import matplotlib.pyplot as plt

    def _disp(n: str) -> str:
        if node_labels and n in node_labels:
            return node_labels[n].replace("\n", " ")
        return n

    comps = pm["edge_components"]
    raw_labels = sorted({f"{a} - {b}" for a, b in zip(comps.a, comps.b)})
    k = int(pm.get("k", len(comps["component"].unique())))
    M = np.zeros((len(raw_labels), k))
    Q = np.ones((len(raw_labels), k))

    val_col = "beta" if what == "beta" else "r"
    for _, r in comps.iterrows():
        lab = f"{r.a} - {r.b}"
        if lab not in raw_labels:
            lab = f"{r.b} - {r.a}"
        if lab in raw_labels:
            i = raw_labels.index(lab)
            c_idx = int(r.component)
            if 0 <= c_idx < k:
                M[i, c_idx] = float(r.get(val_col, 0.0))
                Q[i, c_idx] = float(r.get("q_bh", 1.0))

    labels = []
    for raw in raw_labels:
        parts = [p.strip() for p in raw.split(" - ", 1)]
        labels.append(f"{_disp(parts[0])} - {_disp(parts[1])}")

    fig = _new((1.1 * k + 4.2, 0.55 * len(labels) + 2.0),
               title=f"Per-component partial {'correlations' if what == 'r' else 'slopes'}",
               subtitle=f"hatched: BH q >= {alpha}; right column: edge Pillai")
    gs = fig.add_gridspec(1, 2, width_ratios=[k, 1.1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    lim = max(0.3, float(np.abs(M).max())) if len(M) else 1.0
    im = _matrix(ax, M, _cmap("div"), -lim, lim, rows=labels, cols=[f"c{j + 1}" for j in range(k)])
    for i in range(len(labels)):
        for j in range(k):
            if Q[i, j] >= alpha:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec=_MUTED, lw=0))

    edges_records = pm["edges"].to_dict("records")
    pil_list = []
    for raw in raw_labels:
        val = 0.0
        for e in edges_records:
            if f"{e['a']} - {e['b']}" == raw or f"{e['b']} - {e['a']}" == raw:
                val = float(e.get("pillai", 0.0))
                break
        pil_list.append([val])
    pil = np.array(pil_list)

    _matrix(ax2, pil, _cmap("seq"), 0, 1, rows=None, cols=["Pillai"])
    ax2.set_yticks([])
    ax2.set_ylim(ax.get_ylim())
    _colorbar(fig, im, ax, f"partial {what}")
    return fig


@_themed
def plot_path_comparison(cmp: Dict[str, Any]) -> Any:
    """Forest plot comparing candidate path models against the top graph.

    Displays held-out log-likelihood deltas relative to the best graph with
    paired bootstrap confidence intervals, alongside the fraction of present
    and absent edges passing hypothesis tests. When present-pass and absent-pass
    fractions are equal, markers are dodged vertically to prevent occlusion.
    When a baseline model (e.g. ``empty``) is an extreme outlier (>5x CI span),
    competitive models are zoomed and the extreme baseline is pinned at a marked
    break zone with a callout pill.

    Parameters
    ----------
    cmp : dict
        Result of `pysimlr.compare_path_models`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = sorted(cmp["comparison"], key=lambda r: r["heldout_loglik"])
    fig = _new((8.2, 0.58 * len(rows) + 2.4), title="Which graph predicts held-out scores best",
               subtitle="held-out log-likelihood minus the top graph's; paired subject bootstrap 95% CI")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.3], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax)
    y = np.arange(len(rows))

    comp_rows = [r for r in rows if r["graph"] != "empty"] if any(r["graph"] == "empty" for r in rows) and len(rows) > 1 else rows
    comp_min = min(r["ci_low"] for r in comp_rows)
    comp_max = max(r["ci_high"] for r in comp_rows)
    comp_span = max(1.0, comp_max - comp_min)

    # Check for extreme baseline outlier (> 5x competitive span)
    has_outlier = len(rows) > 1 and any((r["ci_low"] < comp_min - 5.0 * comp_span) for r in rows)
    outlier_graph = rows[0]["graph"] if has_outlier else None

    if has_outlier:
        xlim_min = comp_min - 0.08 * comp_span
        xlim_max = comp_max + 0.12 * comp_span
    else:
        min_ci = min(r["ci_low"] for r in rows)
        max_ci = max(r["ci_high"] for r in rows)
        span = max(1.0, max_ci - min_ci)
        xlim_min = min_ci - 0.06 * span
        xlim_max = max_ci + 0.08 * span

    data_span = max(1.0, xlim_max - xlim_min)
    ax_right_limit = xlim_max + 0.48 * data_span
    ax.set_xlim(xlim_min, ax_right_limit)

    p_nums, y_press = [], []
    a_nums, y_abss = [], []
    for i, r in enumerate(rows):
        best = r["graph"] == cmp["best"]
        c = _BLUE[6] if best else _INK2

        if has_outlier and r["graph"] == outlier_graph:
            # Pinned extreme baseline with break indicator and callout pill
            x_break = xlim_min + 0.03 * data_span
            ax.text(x_break, i, "//", ha="center", va="center", fontsize=11, color=_MUTED, weight="bold")
            pill_txt = f"\u0394 = {r['delta_vs_top']:,.0f} [{r['ci_low']:,.0f}, {r['ci_high']:,.0f}]"
            ax.text(x_break + 0.05 * data_span, i, pill_txt, va="center", ha="left",
                    fontsize=8, color=_INK, weight="semibold",
                    bbox=dict(boxstyle="round,pad=0.30", fc=_SURFACE, ec=_GRID, lw=0.8, alpha=0.96))
        else:
            ax.hlines(i, r["ci_low"], r["ci_high"], color=c, lw=2.2, alpha=0.6)
            _dots(ax, [r["delta_vs_top"]], [i], c, s=52)

        # Right-aligned text & mini-bar column outside error bar data coordinates
        bar_x0 = xlim_max + 0.05 * data_span
        bar_w = 0.16 * data_span
        bf = float(r.get("best_fraction", 0.0))
        ax.barh(i, bar_w * bf, left=bar_x0, height=0.32, color=_BLUE[3] if best else _GRID, alpha=0.75)
        ax.barh(i, bar_w, left=bar_x0, height=0.32, fill=False, edgecolor=_GRID, lw=0.7)
        ax.text(bar_x0 + bar_w + 0.02 * data_span, i, f"best in {100 * bf:.0f}%",
                va="center", ha="left", fontsize=8, color=_INK2)

        pres_val = r.get("present_pass")
        abs_val = r.get("absent_pass")
        p_num = np.nan if pres_val is None else float(pres_val)
        a_num = np.nan if abs_val is None else float(abs_val)

        # Dodge vertically when equal to resolve overlapping markers
        if not np.isnan(p_num) and not np.isnan(a_num) and np.isclose(p_num, a_num, atol=1e-4):
            y_pres = i + 0.12
            y_abs = i - 0.12
        else:
            y_pres = float(i)
            y_abs = float(i)

        p_nums.append(p_num)
        y_press.append(y_pres)
        a_nums.append(a_num)
        y_abss.append(y_abs)

        if np.isnan(p_num):
            ax2.text(0.5, i, "—", ha="center", va="center", fontsize=11, color=_MUTED)

    ax2.scatter(p_nums, y_press, color=_BLUE[6], s=40, label="present edges pass")
    ax2.scatter(a_nums, y_abss, color=_RED[4], s=40, marker="D", label="absent edges pass")

    ax.axvline(0, color=_AXIS, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['graph']}  ({r['n_edges']} edges)" for r in rows])
    ax.set_xlabel("delta held-out log-likelihood")
    _clean(ax, grid="x")

    ax2.set_xlim(-0.08, 1.08)
    ax2.set_xlabel("fraction passing")
    ax2.tick_params(labelleft=False)
    _clean(ax2, grid="x")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=1, frameon=False)
    return fig


@_themed
def plot_edge_null(pm: Dict[str, Any], edges: Optional[Sequence[Any]] = None,
                   alpha: float = 0.05, ax=None,
                   node_labels: Optional[Dict[str, str]] = None) -> Any:
    """Permutation null distribution histograms for observed edge statistics.

    Parameters
    ----------
    pm : dict
        Result of `pysimlr.path_model`.
    edges : sequence of edge specifications, optional
        Edges to plot (e.g. ``[("view0", "view1")]`` or ``["view0 - view1"]``).
        If None, all present edges plus any significant absent edges are plotted.
    alpha : float, default=0.05
        Significance threshold for default edge filtering.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into when plotting a single edge.
    node_labels : dict, optional
        Mapping of raw node identifiers to human-readable display labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not pm.get("null"):
        raise ValueError("pm has no stored null distribution (null was empty or n_permutations was 0).")

    def _disp(n: str) -> str:
        if node_labels and n in node_labels:
            return node_labels[n].replace("\n", " ")
        return n

    all_edges = pm["edges"].to_dict("records")
    if edges is not None:
        target_edges = []
        for item in edges:
            if isinstance(item, tuple):
                a_name, b_name = str(item[0]), str(item[1])
            elif isinstance(item, str):
                parts = [s.strip() for s in item.split("-", 1)] if "-" in item else [item, ""]
                a_name, b_name = parts[0], parts[1]
            else:
                a_name, b_name = str(item), ""
            match = next((e for e in all_edges
                          if (e["a"] == a_name and e["b"] == b_name) or (e["a"] == b_name and e["b"] == a_name)
                          or f"{e['a']} - {e['b']}" == item or f"{e['b']} - {e['a']}" == item), None)
            if match is not None:
                target_edges.append(match)
        edge_list = target_edges
    else:
        edge_list = [e for e in all_edges if e["present"] or (e.get("p_maxt") is not None and e["p_maxt"] < alpha)]

    if not edge_list:
        edge_list = all_edges[:1]

    n = len(edge_list)
    params = pm.get("parameters") if isinstance(pm.get("parameters"), dict) else {}
    b_sweeps = params.get("n_permutations", len(next(iter(pm["null"].values()))))

    if ax is not None:
        fig = ax.figure
        axes = [ax]
    else:
        cols = min(3, n)
        rws = int(np.ceil(n / cols))
        fig = _new((3.6 * cols + 0.6, 2.6 * rws + 1.2),
                   title="Observed edge statistic against its permutation null",
                   subtitle=f"Freedman-Lane, {b_sweeps} sweeps; the p-value is add-one")
        axes = [fig.add_subplot(rws, cols, i + 1) for i in range(n)]

    for i, e in enumerate(edge_list):
        cur_ax = axes[i]
        key = (e["a"], e["b"])
        if key not in pm["null"]:
            key = (e["b"], e["a"])
        null_vals = np.asarray(pm["null"].get(key, []))

        x_obs = float(e.get("pillai", 0.0))
        p_val = e.get("p_perm", e.get("p_refit"))
        p_str = _format_p(p_val, b_sweeps)
        q95 = float(np.percentile(null_vals, 95)) if len(null_vals) else 0.0
        q995 = float(np.percentile(null_vals, 99.5)) if len(null_vals) else 0.0

        is_extreme = len(null_vals) > 0 and q995 > 0 and x_obs > 3.5 * q995

        if is_extreme:
            cur_ax.set_title(f"{_disp(e['a'])} - {_disp(e['b'])}", fontsize=9.5)
            cur_ax.set_ylabel("Permutation count", fontsize=8.5, color=_INK2)
            cur_ax.set_xticks([])
            cur_ax.set_yticks([])
            for sp in cur_ax.spines.values():
                sp.set_visible(False)

            ax_l = cur_ax.inset_axes([0.0, 0.0, 0.68, 1.0])
            ax_r = cur_ax.inset_axes([0.76, 0.0, 0.24, 1.0], sharey=ax_l)

            counts, _, _ = ax_l.hist(null_vals, bins=35, histtype="stepfilled", color=_BLUE[2], ec=_BLUE[5], alpha=0.85)
            ax_l.axvline(q95, color=_MUTED, ls="--", lw=1.5, label="95th %ile")
            max_c = int(counts.max()) if len(counts) else 1
            ax_l.set_xlim(0, max(q995 * 1.35, q95 * 1.5))
            ax_l.set_yticks([0, max_c])
            cur_ax.set_yticks([0, max_c])
            ax_l.set_xlabel("Pillai under null", fontsize=8.5, color=_INK2)
            _clean(ax_l, grid=None, baseline=True)

            ax_r.axvline(x_obs, color=_RED[4], lw=2.2)
            dx_win = max(0.04, 0.05 * x_obs)
            ax_r.set_xlim(x_obs - dx_win, x_obs + dx_win)
            ax_r.set_xticks([round(x_obs, 2)])
            ax_r.set_yticks([])
            ax_r.set_xlabel("observed", fontsize=8.5, color=_INK2)
            _clean(ax_r, grid=None, baseline=True)
            ax_r.spines["left"].set_visible(False)

            ax_r.text(x_obs, 0.90, f"V = {x_obs:.2f}\n{p_str}", transform=ax_r.get_xaxis_transform(),
                      ha="center", va="top", fontsize=8, color=_INK, weight="semibold",
                      bbox=dict(boxstyle="round,pad=0.28", fc=_SURFACE, ec=_GRID, lw=0.7, alpha=0.95))

            cur_ax.text(0.72, 0.0, "//", transform=cur_ax.transAxes, ha="center", va="center",
                        fontsize=11, color=_MUTED, weight="bold")
        else:
            counts, _, _ = cur_ax.hist(null_vals, bins=35, histtype="stepfilled", color=_BLUE[2], ec=_BLUE[5], alpha=0.85)
            cur_ax.axvline(q95, color=_MUTED, ls="--", lw=1.5, label="95th %ile")
            cur_ax.axvline(x_obs, color=_RED[4], lw=2.2)
            cur_ax.set_title(f"{_disp(e['a'])} - {_disp(e['b'])}", fontsize=9.5)

            max_c = int(counts.max()) if len(counts) else 1
            max_x = max(null_vals.max() if len(null_vals) else 1.0, x_obs)
            lim_x = max(1.0, x_obs * 1.1) if x_obs > 0.8 else max_x * 1.18
            cur_ax.set_xlim(0, lim_x)
            cur_ax.set_yticks([0, max_c])
            _clean(cur_ax, grid=None, baseline=True)
            cur_ax.set_xlabel("Pillai under the null", fontsize=8.5, color=_INK2)
            cur_ax.set_ylabel("Permutation count", fontsize=8.5, color=_INK2)

            frac = x_obs / lim_x if lim_x > 0 else 0.5
            label_txt = f"V = {x_obs:.2f}\n{p_str}"
            if frac > 0.38:
                cur_ax.text(x_obs - 0.04 * lim_x, 0.90, label_txt,
                            transform=cur_ax.get_xaxis_transform(), ha="right", va="top",
                            fontsize=8, color=_INK, weight="semibold",
                            bbox=dict(boxstyle="round,pad=0.25", fc=_SURFACE, ec=_GRID, lw=0.7, alpha=0.9))
            else:
                cur_ax.text(x_obs + 0.04 * lim_x, 0.90, label_txt,
                            transform=cur_ax.get_xaxis_transform(), ha="left", va="top",
                            fontsize=8, color=_INK, weight="semibold",
                            bbox=dict(boxstyle="round,pad=0.25", fc=_SURFACE, ec=_GRID, lw=0.7, alpha=0.9))

    return fig


def _clean_tissue_label(name: str) -> str:
    """Format tissue and subcortical abbreviations into clean neuroanatomical names."""
    s = str(name).strip()
    replacements = {
        "deepgraytissues": "Deep Gray Matter",
        "Deepgraytissues": "Deep Gray Matter",
        "deep_gray_matter": "Deep Gray Matter",
        "csftissues": "CSF",
        "Csftissues": "CSF",
        "brainstemtissues": "Brainstem",
        "Brainstemtissues": "Brainstem",
        "cortical_compartmenttissues": "Cortical Compartment",
        "Cortical Compartmenttissues": "Cortical Compartment",
        "cortical_gray_mattertissues": "Cortical Gray Matter",
        "Cortical Gray Mattertissues": "Cortical Gray Matter",
        "white_mattertissues": "White Matter",
        "White Mattertissues": "White Matter",
        "gmtissues": "Total Gray Matter",
        "Gmtissues": "Total Gray Matter",
        "wmtissues": "White Matter",
        "Wmtissues": "White Matter",
        "medullabrainstem": "Medulla",
        "Medullabrainstem": "Medulla",
        "midbrainbrainstem": "Midbrain",
        "Midbrainbrainstem": "Midbrain",
        "ponsbrainstem": "Pons",
        "Ponsbrainstem": "Pons",
    }
    for k, v in replacements.items():
        if k in s:
            s = s.replace(k, v)
    return s


@_themed
def plot_edge_features(pm: Dict[str, Any], edge: Any, top_n: int = 15,
                       component: Optional[int] = None, ax=None,
                       feature_names=None, alpha: float = 0.05,
                       node_labels: Optional[Dict[str, str]] = None) -> Any:
    """Feature contributions to a path model edge (structure_a * beta_ab).

    Plots horizontal lollipop bars for the top features contributing to the edge
    between node `a` and node `b`, using signed diverging colors (blue positive,
    red negative), in the style of `plot_top_features`. Significant component
    panels show correlation metrics (r, q); non-significant components are muted.

    Parameters
    ----------
    pm : dict
        Result of `pysimlr.path_model`.
    edge : str or tuple
        Edge to display (e.g. ``"view0 - view1"``, ``("view0", "view1")``, or ``(0, 1)``).
    top_n : int, default=15
        Number of top features to display per component panel.
    component : int, optional
        Component index (0-based) to plot. If None and ax is None, all components
        are plotted side-by-side. If ax is given and component is None, defaults to 0.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into for a single component panel.
    feature_names : list or dict, optional
        Feature names for view `a`.
    alpha : float, default=0.05
        FDR / significance threshold for component significance badging.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "result" in pm and pm["result"] is None:
        raise ValueError(
            "scores-only path_model (pm['result'] is None) has no measurement model or "
            "feature_contributions; pass a fit-backed path_model result."
        )
    fc = pm.get("feature_contributions")
    if fc is None or len(fc) == 0:
        raise ValueError("pm has no feature_contributions table.")

    nodes = list(pm.get("nodes", []))
    if isinstance(edge, tuple):
        a_raw, b_raw = edge
        a_name = nodes[a_raw] if isinstance(a_raw, int) and a_raw < len(nodes) else str(a_raw)
        b_name = nodes[b_raw] if isinstance(b_raw, int) and b_raw < len(nodes) else str(b_raw)
        edge_str = f"{a_name} - {b_name}"
    elif isinstance(edge, str):
        edge_str = edge
        if " - " in edge_str:
            a_name, b_name = [s.strip() for s in edge_str.split(" - ", 1)]
        elif "-" in edge_str:
            a_name, b_name = [s.strip() for s in edge_str.split("-", 1)]
        else:
            a_name, b_name = edge_str, ""
    else:
        a_name, b_name = str(edge), ""
        edge_str = str(edge)

    d_edge = fc[(fc.edge == edge_str) |
                ((fc.a == a_name) & (fc.b == b_name)) |
                ((fc.a == b_name) & (fc.b == a_name))]

    if len(d_edge) == 0:
        avail = sorted(fc["edge"].unique())
        raise ValueError(f"edge {edge!r} not found in feature_contributions; available: {avail}")

    node_a = d_edge.iloc[0]["a"]
    node_b = d_edge.iloc[0]["b"]
    disp_edge = f"{node_a} - {node_b}"
    view_idx = nodes.index(node_a) if node_a in nodes else 0

    comps_avail = sorted(d_edge["component"].unique())
    if component is not None:
        if component not in comps_avail:
            raise ValueError(f"component {component} not found for edge {disp_edge}; available: {comps_avail}")
        comps = [component]
    elif ax is not None:
        comps = [comps_avail[0]]
    else:
        comps = comps_avail

    if ax is not None:
        fig = ax.figure
        axes = [ax]
    else:
        n_panels = len(comps)
        figsize = (4.2 * n_panels + 0.6, 0.28 * top_n + 2.2)
        fig = _new(figsize,
                   title=f"Edge features: {disp_edge}",
                   subtitle=f"Feature contributions of {node_a} to {node_b} (structure × beta); top {top_n} by magnitude.")
        axes = np.atleast_1d(fig.subplots(1, n_panels))

    params = pm.get("parameters") if isinstance(pm.get("parameters"), dict) else {}
    b_sweeps = params.get("n_permutations", 0)
    comps_df = pm.get("edge_components")

    for comp_idx, cur_ax in zip(comps, axes):
        d_c = d_edge[d_edge.component == comp_idx].copy()
        vals = d_c["contribution"].to_numpy(float)
        order = np.argsort(-np.abs(vals))[:top_n]
        d_top = d_c.iloc[order[::-1]]
        y = np.arange(len(d_top))
        c_vals = d_top["contribution"].to_numpy(float)

        vmax = float(np.nanmax(np.abs(c_vals))) if len(c_vals) else 1.0
        lim = max(0.1, vmax * 1.15)

        r_val, q_val, p_val = 0.0, None, None
        is_sig = True
        if comps_df is not None and len(comps_df):
            crow = comps_df[(((comps_df.a == node_a) & (comps_df.b == node_b)) |
                             ((comps_df.a == node_b) & (comps_df.b == node_a))) &
                            (comps_df.component == comp_idx)]
            if len(crow):
                r_val = float(crow.iloc[0]["r"])
                q_val = crow.iloc[0].get("q_bh")
                p_val = crow.iloc[0].get("p_maxt", crow.iloc[0].get("p_perm"))
                ci_lo = crow.iloc[0].get("ci_low")
                ci_hi = crow.iloc[0].get("ci_high")
                ci_sig = (ci_lo is not None and ci_hi is not None and
                          ((ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)))

                if q_val is not None and not np.isnan(float(q_val)):
                    is_sig = (float(q_val) < alpha) or (b_sweeps < 100 and ci_sig)
                elif p_val is not None and not np.isnan(float(p_val)):
                    is_sig = float(p_val) < alpha
                else:
                    is_sig = ci_sig if (ci_lo is not None and ci_hi is not None) else True

        if is_sig:
            if q_val is not None and not np.isnan(float(q_val)):
                q_f = float(q_val)
                q_str = "q < 0.001" if q_f < 0.001 else f"q = {q_f:.3f}"
            elif p_val is not None and not np.isnan(float(p_val)):
                p_f = float(p_val)
                q_str = "p < 0.001" if p_f < 0.001 else f"p = {p_f:.3f}"
            else:
                q_str = ""
            q_part = f", {q_str}" if q_str else ""
            panel_title = f"Component {comp_idx + 1}  (r = {r_val:+.2f}{q_part})"
            feat_colors = [_BLUE[6] if val >= 0 else _RED[4] for val in c_vals]
            stem_alpha = 0.80
            dot_alpha = 1.0
        else:
            panel_title = f"Component {comp_idx + 1}  (r = {r_val:+.2f}, NS)"
            feat_colors = [_MUTED for _ in c_vals]
            stem_alpha = 0.45
            dot_alpha = 0.45

        cur_ax.axvline(0, color=_AXIS, lw=0.9, zorder=1)
        cur_ax.hlines(y, 0, c_vals, color=feat_colors, lw=2.0, alpha=stem_alpha, zorder=2)
        _dots(cur_ax, c_vals, y, feat_colors, s=50, alpha=dot_alpha)

        labels = []
        for f_idx in d_top["feature"].to_numpy(int):
            if feature_names is not None:
                if isinstance(feature_names, dict):
                    fn_list = feature_names.get(node_a, feature_names.get(view_idx))
                    raw_fn = fn_list[f_idx] if fn_list and f_idx < len(fn_list) else f"feat_{f_idx}"
                elif isinstance(feature_names, (list, tuple)) and f_idx < len(feature_names):
                    raw_fn = feature_names[f_idx]
                else:
                    raw_fn = f"feat_{f_idx}"
            else:
                raw_fn = f"feat_{f_idx}"
            labels.append(_clean_tissue_label(raw_fn))

        cur_ax.set_yticks(y)
        cur_ax.set_yticklabels(labels)
        cur_ax.set_xlim(-lim, lim)
        cur_ax.set_ylim(-0.7, len(d_top) - 0.3)
        cur_ax.set_title(panel_title, loc="center", fontsize=9.5, color=_INK)
        cur_ax.set_xlabel(f"contribution of {node_a} to {node_b} (structure × beta)", fontsize=9.0)
        _clean(cur_ax, grid="x")

    return fig


@_themed
def plot_edge_brain_map(pm: Dict[str, Any], edge: Any, component: int = 0,
                        feature_names=None, region_names=None, title: Optional[str] = None,
                        subtitle: Optional[str] = None, meshes=None, density: str = "41k",
                        surface: str = "inflated", atlas: str = "dkt",
                        scale: str = "symmetric", clim=None, **scene_kwargs) -> Any:
    """Paint path-model edge feature contributions (structure_a * beta_ab) on the brain.

    Connects score-level path model feature contributions to `antsxviz` (the ANTsX
    house renderer), plotting the cortical or volumetric distribution of the
    features of node `a` that carry the association to node `b`.

    Parameters
    ----------
    pm : dict
        Result of `pysimlr.path_model`.
    edge : str or tuple
        Edge to display (e.g. ``"view0 - view1"``, ``("view0", "view1")``, or ``(0, 1)``).
    component : int, default=0
        Component index (0-based) whose feature contributions to display.
    feature_names : list or dict, optional
        Feature names for view `a`.
    region_names : callable or dict, optional
        Map from feature name to DKT region name; defaults to `cortical_region_name`.
    title : str, optional
        Figure title.
    subtitle : str, optional
        Figure subtitle.
    meshes : dict, optional
        Pre-loaded meshes from `antsxviz.load_fsaverage`.
    density : str, default="41k"
        fsaverage surface mesh density ("41k" or "164k").
    surface : str, default="inflated"
        fsaverage surface type ("inflated", "white", "pial").
    atlas : str, default="dkt"
        Anatomical atlas: "dkt" (cortical surface), "hoa", "jhu", "tracts", "tract_sheets", or "yeo".
    scale : str, default="symmetric"
        Color scaling: "symmetric" (symmetric about 0 using diverging blue-red map),
        "minmax", or "zero_max".
    clim : tuple or float, optional
        Explicit color limits (lo, hi).
    **scene_kwargs :
        Additional kwargs passed to antsxviz scene functions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "result" in pm and pm["result"] is None:
        raise ValueError(
            "scores-only path_model (pm['result'] is None) has no measurement model or "
            "feature_contributions; pass a fit-backed path_model result."
        )
    av = _require_antsxviz()

    fc = pm.get("feature_contributions")
    if fc is None or len(fc) == 0:
        raise ValueError("pm has no feature_contributions table.")

    nodes = list(pm.get("nodes", []))
    if isinstance(edge, tuple):
        a_raw, b_raw = edge
        a_name = nodes[a_raw] if isinstance(a_raw, int) and a_raw < len(nodes) else str(a_raw)
        b_name = nodes[b_raw] if isinstance(b_raw, int) and b_raw < len(nodes) else str(b_raw)
        edge_str = f"{a_name} - {b_name}"
    elif isinstance(edge, str):
        edge_str = edge
        if " - " in edge_str:
            a_name, b_name = [s.strip() for s in edge_str.split(" - ", 1)]
        elif "-" in edge_str:
            a_name, b_name = [s.strip() for s in edge_str.split("-", 1)]
        else:
            a_name, b_name = edge_str, ""
    else:
        a_name, b_name = str(edge), ""
        edge_str = str(edge)

    d_edge = fc[(fc.edge == edge_str) |
                ((fc.a == a_name) & (fc.b == b_name)) |
                ((fc.a == b_name) & (fc.b == a_name))]

    if len(d_edge) == 0:
        avail = sorted(fc["edge"].unique())
        raise ValueError(f"edge {edge!r} not found in feature_contributions; available: {avail}")

    node_a = d_edge.iloc[0]["a"]
    node_b = d_edge.iloc[0]["b"]
    disp_edge = f"{node_a} - {node_b}"
    view_idx = nodes.index(node_a) if node_a in nodes else 0

    d_c = d_edge[d_edge.component == component].copy()
    if len(d_c) == 0:
        avail_c = sorted(d_edge["component"].unique())
        raise ValueError(f"component {component} not found for edge {disp_edge}; available: {avail_c}")

    features = d_c["feature"].to_numpy(int)
    c_vals = d_c["contribution"].to_numpy(float)

    names = []
    for f_idx in features:
        if feature_names is not None:
            if isinstance(feature_names, dict):
                fn_list = feature_names.get(node_a, feature_names.get(view_idx))
                names.append(fn_list[f_idx] if fn_list and f_idx < len(fn_list) else f"feat_{f_idx}")
            elif isinstance(feature_names, (list, tuple)) and f_idx < len(feature_names):
                names.append(feature_names[f_idx])
            else:
                names.append(f"feat_{f_idx}")
        else:
            names.append(f"feat_{f_idx}")

    vmax = float(np.nanmax(np.abs(c_vals))) if len(c_vals) else 1.0
    enc = {
        "contribution": {
            "label": "contribution",
            "units": "structure × beta",
            "names": names,
            "values": c_vals,
            "vmax": vmax,
            "selected": None,
            "view_name": node_a,
            "cmap": _cmap("div"),
        }
    }

    if atlas != "dkt":
        if atlas not in ("hoa", "jhu", "tracts", "tract_sheets", "yeo"):
            raise ValueError(f"atlas must be 'dkt', 'hoa', 'jhu', 'tracts', 'tract_sheets' or 'yeo'; got {atlas!r}")
        rename = dict(region_names or {})
        maps_ = {"contribution": {rename.get(n, n): float(v) for n, v in zip(names, c_vals)}}
        common = dict(units={"contribution": "structure × beta"}, scale=scale, clim=clim,
                      view_color=view_idx, subtitle=subtitle,
                      title=title or f"Edge {disp_edge} · Component {component + 1}: feature contributions",
                      **scene_kwargs)
        if atlas == "yeo":
            common.pop("view_color")
            return av.yeo_surface_figure(maps_, **common)
        if atlas in ("hoa", "jhu"):
            return av.region_scene_figure(maps_, atlas, **common)
        return av.tract_scene_figure(maps_, style="sheets" if atlas == "tract_sheets" else "tubes", **common)

    rows, dropped = brain_rows(enc, region_names, view=view_idx)
    if not rows or not next(iter(rows.values()))["values"]:
        raise ValueError(f"no feature of {node_a} is a DKT cortical region; pass region_names= to map them.")
    if meshes is None:
        meshes = av.load_fsaverage(density=density, surface=surface)
    _apply_scale(rows, scale, clim, view_idx, av)
    sub = subtitle if subtitle is not None else (
        f"Feature contributions of {node_a} to {node_b} (component {component + 1}). "
        + _scale_phrase(scale, clim) + ".")
    if dropped:
        sub += (f" Not on cortex ({len(dropped)}): " + ", ".join(map(str, dropped[:4]))
                + (" ..." if len(dropped) > 4 else "") + ".")
    return av.surface_figure(rows, meshes,
                             title=title or f"Edge {disp_edge} · Component {component + 1}: feature contributions",
                             subtitle=sub)


