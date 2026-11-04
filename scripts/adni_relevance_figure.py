"""Tiered relevance figure (run scripts/adni_relevance_test.py first) for one view: brain (DKT cortex), forest plot of shares, lobe panel."""
import os, re, sys, numpy as np, pandas as pd
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [W + "/src", W + "/scripts", W + "/paper"]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap
from matplotlib.patches import Patch
import antsxviz


def fp(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
from pysimlr.plotting import theme, _new, _clean, _dots, _INK, _INK2, _MUTED, _AXIS, _GRID, VIEW_COLORS
from paper_numbers import region_label

R = os.environ.get("RELEVANCE_DIR", os.path.join(W, "docs", "audit", "evidence", "relevance")) + "/"
fd, gd = pd.read_csv(R + "features.csv"), pd.read_csv(R + "groups.csv")
VIEW, COLOR = "vol_area", VIEW_COLORS[0]
f = fd[fd.view == VIEW].copy(); g = gd[gd.view == VIEW].copy()
f["tier"] = np.where((f.p_train < .05) & (f.p_test < .05), 2, np.where(f.p_train < .05, 1, 0))
dark = COLOR; light = tuple(0.45 * np.array(to_rgb(COLOR)) + 0.55)
TIER_COL = {2: dark, 1: light, 0: "#d9d8d2"}
fair = 1 / len(f)
import paper_adni_ablations as paa                                   # held-out r with CogPC1, per feature
_d = paa.protocol_b(42)
RCOG = {n: float(np.corrcoef(_d["x_te"][1][:, i], _d["pc1_te"])[0, 1]) for i, n in enumerate(_d["names"][1])}


def pretty(n):
    return region_label(n).replace(" (DKT cortex label)", "").replace(" (DKT parcellation)", " (parc.)")


# ---- brain: DKT cortex features only (one value per DKT region: the "dktcortex" variant)
vals = {}
for _, r in f.iterrows():
    m = re.match(r"^(left|right)_(.+)dktcortex$", r.feature)
    if m and r.tier > 0:
        vals[antsxviz.dkt_id(f"{m.group(1)}_{m.group(2)}")] = float(r.tier)
cmap = ListedColormap([light, dark])
meshes = antsxviz.load_fsaverage()


def crop(img, pad=6):
    """Trim the transparent border of a rendered RGBA image."""
    ys, xs = np.where(img[..., 3] > 0)
    return img if not len(ys) else img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def render(hemi, view):
    return crop(antsxviz.render_surface(vals, meshes[hemi], hemi, view, cmap, (1, 2)))
cols = [("left", "lateral", "L lateral"), ("left", "medial", "L medial"), ("both", "ventral", "ventral (R | L)"),
        ("right", "medial", "R medial"), ("right", "lateral", "R lateral")]
imgs = []
for hemi, view, _ in cols:
    if hemi == "both":
        a, b = render("left", "ventral"), render("right", "ventral")
        h = max(a.shape[0], b.shape[0]); pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)))
        imgs.append(np.concatenate([pad(b), pad(a)], axis=1))
    else:
        imgs.append(render(hemi, view))
aspect = [im.shape[1] / im.shape[0] for im in imgs]

with theme():
    fig = _new((15.5, 11.6), "ADNI-MCI volume / surface area, predictive component 2: where its variance concentrates",
               "Share of the variance this view explains in the component (relative weights). Fair share = 1/160. "
               "Rule (set before plotting): dark = permutation p < 0.05 in training AND in 326 held-out subjects; "
               "pale = training only; grey = neither.\nSame test on DiReCT cortical thickness: 0 of 62 regions at p < 0.05 "
               "in training, 1 held-out (chance 3.1), no lobe significant: its signal is the global thickness factor, not a "
               "region. L/R verified: per-subject ADNI DKT labels are correct (antsxviz label_hemisphere_report).\n"
               "Share and r answer different questions: a large share with a weak r (CSF) is a non-redundant slice of the "
               "component, not a strong marker; relative weights split variance among correlated features, so an isolated "
               "measure keeps its whole share.")
    gs = fig.add_gridspec(2, len(cols), height_ratios=[1, 2.3], width_ratios=aspect)
    for c, (im, (_, _, head)) in enumerate(zip(imgs, cols)):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(im); ax.axis("off"); ax.set_title(head, fontsize=9.5, color=_INK2)
    sub = gs[1, :].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.08)

    # ---- forest: regional features passing in training; whole-brain totals get their own panel
    def draw_forest(ax, show, title, legend, room=2.9):
        y = np.arange(len(show))
        top = 100 * show[["share_train", "share_test"]].max().max()
        xmax = top + room                                    # room for the p / refits text and the r column
        ax.axvline(100 * fair, color=_INK2, lw=1, ls="--", zorder=1)
        ax.text(100 * fair, -1.0, " fair share", fontsize=8, color=_INK2, va="bottom")
        ax.text(xmax - 0.05, -1.0, "r with CogPC1\n(held-out)", fontsize=7.6, color=_INK2, ha="right", va="bottom")
        for yi, (_, r) in zip(y, show.iterrows()):
            c = TIER_COL[r.tier]
            ax.plot([100 * r.share_train, 100 * r.share_test], [yi, yi], color=_AXIS, lw=1.2, zorder=1)
            ax.scatter(100 * r.share_train, yi, s=58, color=c, edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
            ax.scatter(100 * r.share_test, yi, s=46, facecolor="#fcfcfb", edgecolors=c if r.tier else _MUTED,
                       linewidths=1.6, zorder=3)
            ax.text(100 * max(r.share_train, r.share_test) + 0.12, yi,
                    f"p {fp(r.p_train)} / {fp(r.p_test)}"
                    + (f"   refits {r.refit_p05_freq:.0%}" if r.tier == 2 and np.isfinite(r.refit_p05_freq) else ""),
                    va="center", fontsize=7.6, color=_INK if r.tier == 2 else _INK2)
            ax.text(xmax - 0.05, yi, f"{RCOG[r.feature]:+.2f}", va="center", ha="right", fontsize=8,
                    color=_INK if abs(RCOG[r.feature]) >= 0.3 else _INK2,
                    weight="semibold" if abs(RCOG[r.feature]) >= 0.3 else "normal")
        ax.set_yticks(y); ax.set_yticklabels([pretty(n) for n in show.feature], fontsize=8.3)
        for lbl, t in zip(ax.get_yticklabels(), show.tier):
            lbl.set_color(_INK if t == 2 else _INK2); lbl.set_fontweight("semibold" if t == 2 else "normal")
        ax.set_ylim(len(show) - 0.4, -1.6); ax.set_xlim(0, xmax)
        _clean(ax, grid="x")
        ax.set_title(title, fontsize=10)
        if legend:
            ax.legend(handles=[Patch(color=dark, label="replicated (train and held-out p < 0.05)"),
                               Patch(color=light, label="training only")], loc="lower right", frameon=False,
                      fontsize=8, bbox_to_anchor=(0.86, 0.0))

    is_total = f.feature.str.endswith("tissues")
    regions = f[~is_total & (f.tier > 0)].sort_values("share_train", ascending=False)
    totals = f[is_total].sort_values("share_train", ascending=False)
    ax = fig.add_subplot(sub[0, 0])
    nreg = int((~is_total).sum())
    draw_forest(ax, regions, f"Regions with training p < 0.05: {int((f.tier[~is_total] > 0).sum())} of {nreg} "
                f"(chance: {0.05 * nreg:.0f}); replicated held-out: {int((f.tier[~is_total] == 2).sum())}", True)
    ax.set_xlabel("% of explained variance (filled: training fit; ring: held-out subjects)")
    right = sub[0, 1].subgridspec(2, 1, height_ratios=[6, 2.6 + 0.35 * len(totals)], hspace=0.35)

    # ---- lobes
    ax = fig.add_subplot(right[0, 0])
    lb = g[g.group.str.startswith("lobe")].copy(); lb["name"] = lb.group.str.replace("lobe ", "")
    lb = lb.sort_values("ratio_fair", ascending=True)
    yy = np.arange(len(lb))
    ax.axvline(1.0, color=_INK2, lw=1, ls="--")
    for yi, (_, r) in zip(yy, lb.iterrows()):
        rep = (r.p_train < .05) and (r.p_test < .05)
        c = dark if rep else (light if r.p_train < .05 else "#c9c8c1")
        ax.hlines(yi, 1.0, r.ratio_fair, color=c, lw=6, alpha=0.9)
        ax.scatter(r.share_test / (r.n * fair), yi, s=46, facecolor="#fcfcfb", edgecolors=c if rep else _MUTED,
                   linewidths=1.6, zorder=3)
        ax.text(max(r.ratio_fair, r.share_test / (r.n * fair)) + 0.05, yi,
                f"{r.ratio_fair:.2f}x   p {fp(r.p_train)} / {fp(r.p_test)}"
                + (f" / {fp(r.p_fullrefit_null)}" if np.isfinite(r.p_fullrefit_null) else ""),
                va="center", fontsize=7.8, color=_INK if rep else _INK2)
    ax.set_yticks(yy); ax.set_yticklabels([f"{n} ({k})" for n, k in zip(lb.name, lb.n)], fontsize=9)
    ax.set_xlim(0.4, 3.3); ax.set_xlabel("share ÷ fair share (bar: training; ring: held-out)")
    full = np.isfinite(lb.p_fullrefit_null).any()
    ax.set_title("DKT lobes  (p: training / held-out" + (" / full-refit null)" if full else ")"), fontsize=10)
    _clean(ax, grid="x")

    # ---- whole-brain totals: on a different footing from regions (one number per brain)
    ax = fig.add_subplot(right[1, 0])
    draw_forest(ax, totals, "Whole-brain tissue totals (not regions)", False, room=4.3)
    ax.set_xlabel("% of explained variance")
    out = os.path.join(W, "docs", "figures", "adni_volarea_relevance.png")
    fig.savefig(out, dpi=160); print("wrote", out)
