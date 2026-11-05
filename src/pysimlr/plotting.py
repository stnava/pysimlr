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
    "plot_variance_explained", "plot_feature_relevance",
    # scores
    "plot_scores", "plot_latent_recovery",
    # one-call sheet
    "plot_fit_summary",
    # benchmarks and generic
    "plot_benchmark", "plot_benchmark_uncertainty", "plot_predictions",
    # fused ridge
    "plot_fused_coefficients", "plot_coefficient_paths", "plot_fused_ridge_cv",
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
                t = fmt.format(M[i, j]) + ("" if extra is None else extra[i][j])
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
                          feature_names=None, view_names=None, value_fmt: str = "{:.2f}", figsize=None):
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
                   density: str = "41k", surface: str = "inflated", tiers: Optional[Dict[str, Any]] = None):
    """Feature maps of one component painted on the fsaverage cortex, via antsxviz.

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
    sub = subtitle if subtitle is not None else (
        "Colour: each map on its own scale from 0 to its maximum"
        + ("; dark outline: selected in ≥ 80% of refits" if inf is not None else "") + ".")
    if dropped:
        sub += (f" Not on cortex ({len(dropped)}): " + ", ".join(map(str, dropped[:6]))
                + (" ..." if len(dropped) > 6 else "") + ".")
    return av.surface_figure(rows, meshes, title=title or f"Component {component + 1} of {vname}: feature maps",
                             subtitle=sub)


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
